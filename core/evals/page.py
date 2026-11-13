"""The evaluation store's web page: every player's Elo and record, the latest games, and each
player's record against everyone it has played. It only reads the store.

Elo is fit to every game in the store (`core.league.ratings`), whatever its match's setup. Players
who are linked by games, directly or through others, make a pool; pools are fit apart, each with
its players' mean at 0, as nothing says how one pool's players compare with another's. Pool 1 is
the biggest.

    GET /                  the page
    GET /api/summary       {"games", "players": [...], "recent": [...]}
    GET /api/player?id=ID  {"player": {...}, "opponents": [...]}
"""

import json
import math
import sqlite3
import threading
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from core.league.ratings import Ratings, connected_components, fit_ratings
from core.league.store import PairRecord

PAGE = Path(__file__).with_name("page.html")
RECENT_GAMES = 100


def _score(wins: int, draws: int, losses: int) -> float | None:
    games = wins + draws + losses
    return (wins + draws / 2) / games if games else None


class Evals:
    """What the page shows, read from the store at `path`."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.version: tuple[Any, ...] | None = None
        self.players: list[tuple[str, str, str]] = []
        # by pair of player ids, in order: the first one's wins, draws and losses
        self.pairs: dict[tuple[str, str], list[int]] = {}
        # by the number of each player with games: its pool, and its pool's fit
        self.pool: dict[int, int] = {}
        self.fits: dict[int, Ratings] = {}

    def _connect(self) -> closing[sqlite3.Connection]:
        return closing(
            sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
        )

    def _refresh(self, db: sqlite3.Connection) -> None:
        """Refits the ratings if there are new games or players."""
        version = (
            db.execute("SELECT MAX(id), COUNT(*) FROM games").fetchone(),
            db.execute("SELECT COUNT(*) FROM players").fetchone(),
        )
        if version == self.version:
            return
        self.players = db.execute(
            "SELECT id, label, kind FROM players ORDER BY added"
        ).fetchall()
        self.pairs = {}
        rows = db.execute(
            "SELECT black_id, white_id, (result > 0) - (result < 0) AS sign, COUNT(*) "
            "FROM games GROUP BY black_id, white_id, sign"
        )
        for black, white, sign, games in rows:
            a, b = sorted((black, white))
            # the result for `a`: wins, draws, losses
            sign = sign if a == black else -sign
            self.pairs.setdefault((a, b), [0, 0, 0])[1 - sign] += games
        number = {id_: i for i, (id_, _, _) in enumerate(self.players)}
        records = [
            PairRecord(number[a], number[b], *record)
            for (a, b), record in self.pairs.items()
        ]
        n = len(self.players)
        parts = connected_components(n, [(r.a, r.b) for r in records])
        played = {r.a for r in records} | {r.b for r in records}
        members: dict[int, list[int]] = {}
        for i in sorted(played):
            members.setdefault(int(parts[i]), []).append(i)
        self.pool, self.fits = {}, {}
        by_size = sorted(members.values(), key=lambda m: (-len(m), m[0]))
        for pool, ids in enumerate(by_size, start=1):
            inside = set(ids)
            fit = fit_ratings(ids, [r for r in records if r.a in inside])
            for i in ids:
                self.pool[i], self.fits[i] = pool, fit
        self.version = version

    def _row(self, i: int, record: list[int]) -> dict[str, Any]:
        id_, label, kind = self.players[i]
        fit = self.fits.get(i)
        return {
            "id": id_,
            "name": label,
            "kind": kind,
            "pool": self.pool.get(i),
            "elo": None if fit is None else float(fit.elo[fit.index(i)]),
            "sd": None if fit is None else float(fit.sd[fit.index(i)]),
            "games": sum(record),
            "wins": record[0],
            "draws": record[1],
            "losses": record[2],
            "score": _score(*record),
        }

    def summary(self) -> dict[str, Any]:
        with self.lock, self._connect() as db:
            self._refresh(db)
            records = {id_: [0, 0, 0] for id_, _, _ in self.players}
            for (a, b), (wins, draws, losses) in self.pairs.items():
                for player, counts in (
                    (a, (wins, draws, losses)),
                    (b, (losses, draws, wins)),
                ):
                    for k in range(3):
                        records[player][k] += counts[k]
            rows = [
                self._row(i, records[id_]) for i, (id_, _, _) in enumerate(self.players)
            ]
            rows.sort(key=lambda row: (row["pool"] or math.inf, -(row["elo"] or 0)))
            return {
                "games": sum(row["games"] for row in rows) // 2,
                "players": rows,
                "recent": self._recent(db),
            }

    def _recent(self, db: sqlite3.Connection) -> list[dict[str, Any]]:
        names = {id_: label for id_, label, _ in self.players}
        rows = db.execute(
            "SELECT id, finished, black_id, white_id, result, margin, ending, solved_at, "
            "flagged FROM games ORDER BY finished DESC, id DESC LIMIT ?",
            (RECENT_GAMES,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "finished": r[1],
                "black": r[2],
                "black_name": names.get(r[2], r[2]),
                "white": r[3],
                "white_name": names.get(r[3], r[3]),
                "result": r[4],
                "margin": r[5],
                "ending": r[6],
                "solved_at": r[7],
                "flagged": r[8],
            }
            for r in rows
        ]

    def player(self, player: str) -> dict[str, Any] | None:
        """`player`'s rating and record, and its record against each opponent, or None if
        there's no such player."""
        with self.lock, self._connect() as db:
            self._refresh(db)
            index = {id_: i for i, (id_, _, _) in enumerate(self.players)}
            if player not in index:
                return None
            opponents = []
            total = [0, 0, 0]
            for (a, b), (wins, draws, losses) in self.pairs.items():
                if player == a:
                    other, record = b, [wins, draws, losses]
                elif player == b:
                    other, record = a, [losses, draws, wins]
                else:
                    continue
                total = [t + c for t, c in zip(total, record, strict=True)]
                row = self._row(index[other], record)
                row["diff_sd"] = self.fits[index[player]].pair_sd(
                    index[player], index[other]
                )
                opponents.append(row)
            opponents.sort(key=lambda row: -row["elo"])
            return {"player": self._row(index[player], total), "opponents": opponents}


def make_handler(evals: Evals) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, body: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: Any, status: int = 200) -> None:
            self._send(json.dumps(value).encode(), "application/json", status)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/summary":
                    self._json(evals.summary())
                elif url.path == "/api/player":
                    ids = parse_qs(url.query).get("id")
                    found = evals.player(ids[0]) if ids else None
                    if found is None:
                        self._json({"error": "no such player"}, 404)
                    else:
                        self._json(found)
                else:
                    self._json({"error": "not found"}, 404)
            except BrokenPipeError:
                pass

    return Handler


def serve(path: Path | str, host: str, port: int) -> None:
    """Serves the page for the store at `path` until it's stopped."""
    server = ThreadingHTTPServer((host, port), make_handler(Evals(path)))
    print(f"serving on {host}:{port}", flush=True)
    server.serve_forever()
