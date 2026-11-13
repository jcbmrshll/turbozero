"""The league's web page: every player's rating and record, the latest games, and each
player's record against everyone it has played. It only reads the store.

    GET /                  the page
    GET /api/summary       {"game", "games", "players": [...], "recent": [...]}
    GET /api/player?id=N   {"player": {...}, "opponents": [...]}
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from core.league.ratings import Ratings, fit_ratings
from core.league.store import PairRecord, Player, Store

PAGE = Path(__file__).with_name("dashboard.html")
RECENT_GAMES = 100


def _score(wins: int, draws: int, losses: int) -> float | None:
    games = wins + draws + losses
    return (wins + draws / 2) / games if games else None


class League:
    """What the page shows, read from `store` one thread at a time."""

    def __init__(self, store: Store, name: str):
        self.store = store
        self.name = name
        self.lock = threading.Lock()
        self.version: tuple[Any, ...] | None = None
        self.players: list[Player] = []
        self.pairs: list[PairRecord] = []
        self.ratings: Ratings | None = None

    def _refresh(self) -> Ratings:
        """Reads the players again, refitting the ratings if there are new games or players,
        or a new anchor."""
        self.players = self.store.players()
        version = (
            self.store.db.execute("SELECT MAX(id) FROM games").fetchone()[0],
            tuple(p.id for p in self.players),
            self.store.get_meta("anchor"),
        )
        if version != self.version or self.ratings is None:
            self.pairs = self.store.pair_records()
            self.ratings = fit_ratings(list(version[1]), self.pairs, anchor=version[2])
            self.version = version
        return self.ratings

    def _row(self, i: int, ratings: Ratings, record: list[int]) -> dict[str, Any]:
        p = self.players[i]
        return {
            "id": p.id,
            "name": p.name,
            "elo": float(ratings.elo[i]),
            "sd": float(ratings.sd[i]),
            "games": int(ratings.games[i]),
            "wins": record[0],
            "draws": record[1],
            "losses": record[2],
            "score": _score(*record),
            "run": p.run,
            "external": p.external,
            "retired": p.retired,
        }

    def summary(self) -> dict[str, Any]:
        with self.lock:
            ratings = self._refresh()
            records = {p.id: [0, 0, 0] for p in self.players}
            for r in self.pairs:
                for player, counts in (
                    (r.a, (r.wins, r.draws, r.losses)),
                    (r.b, (r.losses, r.draws, r.wins)),
                ):
                    for k in range(3):
                        records[player][k] += counts[k]
            rows = [
                self._row(i, ratings, records[p.id])
                for i, p in enumerate(self.players)
                if ratings.games[i]
            ]
            rows.sort(key=lambda row: -row["elo"])
            return {
                "game": self.name,
                "games": sum(row["games"] for row in rows) // 2,
                "players": rows,
                "recent": self._recent(),
            }

    def _recent(self) -> list[dict[str, Any]]:
        rows = self.store.db.execute(
            "SELECT g.id, b.finished, g.first_id, g.second_id, g.result, "
            "g.first_score, g.second_score FROM games g "
            "JOIN batches b ON b.id = g.batch_id ORDER BY g.id DESC LIMIT ?",
            (RECENT_GAMES,),
        ).fetchall()
        names = {p.id: p.name for p in self.players}
        return [
            {
                "id": row["id"],
                "finished": row["finished"],
                "first": row["first_id"],
                "first_name": names.get(row["first_id"], "?"),
                "second": row["second_id"],
                "second_name": names.get(row["second_id"], "?"),
                "result": row["result"],
                "first_score": row["first_score"],
                "second_score": row["second_score"],
            }
            for row in rows
        ]

    def player(self, player: int) -> dict[str, Any] | None:
        """`player`'s rating and record, and its record against each opponent, or None if
        there's no such player."""
        with self.lock:
            ratings = self._refresh()
            if player not in ratings.ids:
                return None
            index = {p.id: i for i, p in enumerate(self.players)}
            opponents = []
            total = [0, 0, 0]
            for r in self.pairs:
                if player not in (r.a, r.b):
                    continue
                if r.a == player:
                    other, record = r.b, [r.wins, r.draws, r.losses]
                else:
                    other, record = r.a, [r.losses, r.draws, r.wins]
                total = [t + c for t, c in zip(total, record)]
                row = self._row(index[other], ratings, record)
                row["games"] = sum(record)
                row["diff_sd"] = ratings.pair_sd(player, other)
                opponents.append(row)
            opponents.sort(key=lambda row: -row["elo"])
            return {
                "player": self._row(index[player], ratings, total),
                "opponents": opponents,
            }


def make_handler(league: League) -> type[BaseHTTPRequestHandler]:
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
                    self._json(league.summary())
                elif url.path == "/api/player":
                    found = league.player(int(parse_qs(url.query)["id"][0]))
                    if found is None:
                        self._json({"error": "no such player"}, 404)
                    else:
                        self._json(found)
                else:
                    self._json({"error": "not found"}, 404)
            except (KeyError, ValueError):
                self._json({"error": "needs a player id"}, 400)
            except BrokenPipeError:
                pass

    return Handler


def serve(store: Store, name: str, host: str, port: int) -> None:
    """Serves the page until it's stopped."""
    server = ThreadingHTTPServer((host, port), make_handler(League(store, name)))
    print(f"serving on {host}:{port}", flush=True)
    server.serve_forever()
