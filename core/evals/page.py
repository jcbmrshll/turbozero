"""The evaluation store's web page: every player's Elo and record, the latest games, each
player's record against everyone it has played and its latest games, each opening's record for
black and its games, and a game move by move: each move's time and what each player's search made
of its position. It only reads the store.

What it counts is either every game or competition games alone (the default): those whose players
both play on a clock of at least `COMPETITION_SECONDS` a game, as in a contest, rather than at so
many visits or an engine's level. Elo is fit to the games counted (`core.league.ratings`), whatever
their match's setup. Players who are linked by games, directly or through others, make a pool;
pools are fit apart, each with its players' mean at 0, as nothing says how one pool's players
compare with another's. Pool 1 is the biggest.

An opening's record is black's over its games; as each opening is played with both colors by the
same two players, their strengths mostly cancel. It favours a color when black's score is more
than two standard errors from 50%, the standard error being an even opening's without draws,
0.5 / sqrt(games).

Every question takes `games=all` to count every game.

    GET /                       the page
    GET /api/summary            {"games", "players": [...], "recent": [...]}
    GET /api/player?id=ID       {"player": {...}, "opponents": [...], "games": [...]}
    GET /api/openings           {"openings": [...]}: each start's record for black
    GET /api/opening?start=S    {"opening": {...}, "moves": [...], "players": [...], "games": [...]}
    GET /api/game?id=ID         {"game": {...}}: its moves, their seconds, `move_info` and its
                                opening's record
"""

import json
import math
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from core.league.ratings import Ratings, connected_components, fit_ratings
from core.league.store import PairRecord

PAGE = Path(__file__).with_name("page.html")
RECENT_GAMES = 100
COMPETITION_SECONDS = 30.0

Scope = Literal["competition", "all"]
# the games a scope counts, as a condition on `games` (none for every game)
_COMPETITOR = (
    "SELECT id FROM players "
    f"WHERE json_extract(config, '$.budget.clock.ini') >= {COMPETITION_SECONDS}"
)
SCOPES: dict[str, str | None] = {
    "competition": f"black_id IN ({_COMPETITOR}) AND white_id IN ({_COMPETITOR})",
    "all": None,
}
# a game's margin is exact unless it was solved for its result alone
_EXACT = "(ending != 'solved' OR solve_exact)"


def _score(wins: int, draws: int, losses: int) -> float | None:
    games = wins + draws + losses
    return (wins + draws / 2) / games if games else None


def _where(scope: Scope, *conditions: str) -> str:
    """A WHERE clause for `scope`'s games that also meet `conditions`."""
    found = [c for c in (SCOPES[scope], *conditions) if c]
    return f"WHERE {' AND '.join(found)} " if found else ""


def opening_row(
    start: str, black: int, draws: int, white: int, margin: int | None, exact: int
) -> dict[str, Any]:
    """An opening's record for black: `black` wins, `draws`, and `white` wins; `margin`, black's
    discs summed over the `exact` games whose margin is known."""
    games = black + draws + white
    score = (black + draws / 2) / games
    sd = 0.5 / math.sqrt(games)
    favours = None
    if abs(score - 0.5) > 2 * sd:
        favours = "black" if score > 0.5 else "white"
    return {
        "start": start,
        "games": games,
        "black": black,
        "draws": draws,
        "white": white,
        "score": score,
        "sd": sd,
        "favours": favours,
        "margin": margin / exact if exact and margin is not None else None,
    }


@dataclass
class _Counted:
    """What's fit to a scope's games, as `Evals` keeps it."""

    # by pair of player ids, in order: the first one's wins, draws and losses
    pairs: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    # by the number of each player with games: its pool, and its pool's fit
    pool: dict[int, int] = field(default_factory=dict)
    fits: dict[int, Ratings] = field(default_factory=dict)
    # by start, its record for black, once asked for
    openings: dict[str, dict[str, Any]] | None = None


class Evals:
    """What the page shows, read from the store at `path`."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.version: tuple[Any, ...] | None = None
        self.players: list[tuple[str, str, str]] = []
        # by scope, once asked for
        self.counted: dict[str, _Counted] = {}

    def _connect(self) -> closing[sqlite3.Connection]:
        return closing(
            sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
        )

    def _refresh(self, db: sqlite3.Connection, scope: Scope) -> _Counted:
        """What's fit to `scope`'s games, fit again if there are new games or players."""
        version = (
            db.execute("SELECT MAX(id), COUNT(*) FROM games").fetchone(),
            db.execute("SELECT COUNT(*) FROM players").fetchone(),
        )
        if version != self.version:
            self.players = db.execute(
                "SELECT id, label, kind FROM players ORDER BY added"
            ).fetchall()
            self.counted = {}
            self.version = version
        if scope not in self.counted:
            self.counted[scope] = self._fit(db, scope)
        return self.counted[scope]

    def _fit(self, db: sqlite3.Connection, scope: Scope) -> _Counted:
        counted = _Counted()
        pairs = counted.pairs
        rows = db.execute(
            "SELECT black_id, white_id, (result > 0) - (result < 0) AS sign, COUNT(*) "
            f"FROM games {_where(scope)}GROUP BY black_id, white_id, sign"
        )
        for black, white, sign, games in rows:
            a, b = sorted((black, white))
            # the result for `a`: wins, draws, losses
            sign = sign if a == black else -sign
            pairs.setdefault((a, b), [0, 0, 0])[1 - sign] += games
        number = {id_: i for i, (id_, _, _) in enumerate(self.players)}
        records = [
            PairRecord(number[a], number[b], *record)
            for (a, b), record in pairs.items()
        ]
        n = len(self.players)
        parts = connected_components(n, [(r.a, r.b) for r in records])
        played = {r.a for r in records} | {r.b for r in records}
        members: dict[int, list[int]] = {}
        for i in sorted(played):
            members.setdefault(int(parts[i]), []).append(i)
        by_size = sorted(members.values(), key=lambda m: (-len(m), m[0]))
        for pool, ids in enumerate(by_size, start=1):
            inside = set(ids)
            fit = fit_ratings(ids, [r for r in records if r.a in inside])
            for i in ids:
                counted.pool[i], counted.fits[i] = pool, fit
        return counted

    def _openings(
        self, db: sqlite3.Connection, scope: Scope
    ) -> dict[str, dict[str, Any]]:
        """Each start's record for black over `scope`'s games."""
        counted = self._refresh(db, scope)
        if counted.openings is None:
            rows = db.execute(
                "SELECT start, SUM(result > 0), SUM(result = 0), SUM(result < 0), "
                f"SUM(CASE WHEN {_EXACT} THEN margin END), SUM({_EXACT}) "
                f"FROM games {_where(scope)}GROUP BY start"
            )
            counted.openings = {r[0]: opening_row(*r) for r in rows}
        return counted.openings

    def _row(self, counted: _Counted, i: int, record: list[int]) -> dict[str, Any]:
        id_, label, kind = self.players[i]
        fit = counted.fits.get(i)
        return {
            "id": id_,
            "name": label,
            "kind": kind,
            "pool": counted.pool.get(i),
            "elo": None if fit is None else float(fit.elo[fit.index(i)]),
            "sd": None if fit is None else float(fit.sd[fit.index(i)]),
            "games": sum(record),
            "wins": record[0],
            "draws": record[1],
            "losses": record[2],
            "score": _score(*record),
        }

    def summary(self, scope: Scope = "competition") -> dict[str, Any]:
        with self.lock, self._connect() as db:
            counted = self._refresh(db, scope)
            records = {id_: [0, 0, 0] for id_, _, _ in self.players}
            for (a, b), (wins, draws, losses) in counted.pairs.items():
                for player, counts in (
                    (a, (wins, draws, losses)),
                    (b, (losses, draws, wins)),
                ):
                    for k in range(3):
                        records[player][k] += counts[k]
            rows = [
                self._row(counted, i, records[id_])
                for i, (id_, _, _) in enumerate(self.players)
            ]
            rows.sort(key=lambda row: (row["pool"] or math.inf, -(row["elo"] or 0)))
            return {
                "games": sum(row["games"] for row in rows) // 2,
                "players": rows,
                "recent": self._games(db, scope),
            }

    def _games(
        self,
        db: sqlite3.Connection,
        scope: Scope,
        player: str | None = None,
        start: str | None = None,
        limit: int | None = RECENT_GAMES,
    ) -> list[dict[str, Any]]:
        """`scope`'s latest games, or `player`'s, or those from `start`; `limit` of them, or
        all."""
        names = {id_: label for id_, label, _ in self.players}
        conditions, args = [], []
        if player is not None:
            conditions.append("(black_id = ? OR white_id = ?)")
            args += [player, player]
        if start is not None:
            conditions.append("start = ?")
            args.append(start)
        rows = db.execute(
            "SELECT id, finished, black_id, white_id, result, margin, ending, solved_at, "
            f"flagged, solve_exact, start FROM games {_where(scope, *conditions)}"
            f"ORDER BY finished DESC, id DESC{'' if limit is None else ' LIMIT ?'}",
            (*args, *(() if limit is None else (limit,))),
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
                # a solve for the result alone gives a margin whose sign alone is right
                "margin": None if r[6] == "solved" and not r[9] else r[5],
                "ending": r[6],
                "solved_at": r[7],
                "flagged": r[8],
                "start": r[10],
            }
            for r in rows
        ]

    def player(
        self, player: str, scope: Scope = "competition"
    ) -> dict[str, Any] | None:
        """`player`'s rating and record, and its record against each opponent, or None if
        there's no such player."""
        with self.lock, self._connect() as db:
            counted = self._refresh(db, scope)
            index = {id_: i for i, (id_, _, _) in enumerate(self.players)}
            if player not in index:
                return None
            opponents = []
            total = [0, 0, 0]
            for (a, b), (wins, draws, losses) in counted.pairs.items():
                if player == a:
                    other, record = b, [wins, draws, losses]
                elif player == b:
                    other, record = a, [losses, draws, wins]
                else:
                    continue
                total = [t + c for t, c in zip(total, record, strict=True)]
                row = self._row(counted, index[other], record)
                row["diff_sd"] = counted.fits[index[player]].pair_sd(
                    index[player], index[other]
                )
                opponents.append(row)
            opponents.sort(key=lambda row: -row["elo"])
            return {
                "player": self._row(counted, index[player], total),
                "opponents": opponents,
                "games": self._games(db, scope, player=player),
            }

    def openings(self, scope: Scope = "competition") -> dict[str, Any]:
        """Each start's record for black, the most played first."""
        with self.lock, self._connect() as db:
            rows = sorted(
                self._openings(db, scope).values(),
                key=lambda row: (-row["games"], row["start"]),
            )
            return {"openings": rows}

    def opening(
        self, start: str, scope: Scope = "competition"
    ) -> dict[str, Any] | None:
        """`start`'s record for black, its moves, each player's record from it with each color,
        and all its games; None if no game counted started there."""
        with self.lock, self._connect() as db:
            found = self._openings(db, scope).get(start)
            if found is None:
                return None
            moves, opening_moves = db.execute(
                "SELECT moves, opening_moves FROM games WHERE start = ? LIMIT 1",
                (start,),
            ).fetchone()
            games = self._games(db, scope, start=start, limit=None)
            # by player: wins, draws and losses as black, then as white
            records: dict[str, list[int]] = {}
            for g in games:
                sign = (g["result"] > 0) - (g["result"] < 0)
                records.setdefault(g["black"], [0] * 6)[1 - sign] += 1
                records.setdefault(g["white"], [0] * 6)[4 + sign] += 1
            names = {id_: label for id_, label, _ in self.players}
            players = [
                {
                    "id": id_,
                    "name": names.get(id_, id_),
                    "black": r[:3],
                    "white": r[3:],
                    "games": sum(r),
                    "score": _score(r[0] + r[3], r[1] + r[4], r[2] + r[5]),
                }
                for id_, r in records.items()
            ]
            players.sort(key=lambda row: (-row["games"], row["name"]))
            return {
                "opening": found,
                "moves": json.loads(moves)[:opening_moves],
                "players": players,
                "games": games,
            }

    def game(self, game: int, scope: Scope = "competition") -> dict[str, Any] | None:
        """Game `game` move by move, with its opening's record for black over `scope`'s games
        (None if it has none), or None if there's no such game."""
        with self.lock, self._connect() as db:
            self._refresh(db, scope)
            row = db.execute(
                "SELECT g.id, g.match_id, g.number, g.black_id, g.white_id, g.start, g.moves, "
                "g.opening_moves, g.move_seconds, g.ending, g.solved_at, g.solve_exact, "
                "g.margin, g.result, g.flagged, g.hard, g.black_seconds, g.white_seconds, "
                "h.info, g.load, g.finished, gm.info "
                "FROM games g LEFT JOIN hosts h ON h.id = g.host_id "
                "LEFT JOIN game_moves gm ON gm.game_id = g.id WHERE g.id = ?",
                (game,),
            ).fetchone()
            if row is None:
                return None
            names = {id_: label for id_, label, _ in self.players}
            return {
                "game": {
                    "id": row[0],
                    "match": row[1],
                    "number": row[2],
                    "black": row[3],
                    "black_name": names.get(row[3], row[3]),
                    "white": row[4],
                    "white_name": names.get(row[4], row[4]),
                    "start": row[5],
                    "moves": json.loads(row[6]),
                    "opening_moves": row[7],
                    "move_seconds": json.loads(row[8]),
                    "ending": row[9],
                    "solved_at": row[10],
                    "solve_exact": None if row[11] is None else bool(row[11]),
                    # a solve for the result alone gives a margin whose sign alone is right
                    "margin": None if row[9] == "solved" and not row[11] else row[12],
                    "result": row[13],
                    "flagged": row[14],
                    "hard": row[15],
                    "black_seconds": row[16],
                    "white_seconds": row[17],
                    "host": json.loads(row[18]) if row[18] else None,
                    "load": row[19],
                    "finished": row[20],
                    "move_info": json.loads(row[21]) if row[21] else [],
                    "opening": self._openings(db, scope).get(row[5]),
                }
            }


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
            query = parse_qs(url.query)
            scope: Scope = "all" if query.get("games") == ["all"] else "competition"
            try:
                if url.path == "/":
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/summary":
                    self._json(evals.summary(scope))
                elif url.path == "/api/player":
                    ids = query.get("id")
                    found = evals.player(ids[0], scope) if ids else None
                    if found is None:
                        self._json({"error": "no such player"}, 404)
                    else:
                        self._json(found)
                elif url.path == "/api/openings":
                    self._json(evals.openings(scope))
                elif url.path == "/api/opening":
                    starts = query.get("start")
                    found = evals.opening(starts[0], scope) if starts else None
                    if found is None:
                        self._json({"error": "no such opening"}, 404)
                    else:
                        self._json(found)
                elif url.path == "/api/game":
                    ids = query.get("id")
                    game = (
                        evals.game(int(ids[0]), scope)
                        if ids and ids[0].isdigit()
                        else None
                    )
                    if game is None:
                        self._json({"error": "no such game"}, 404)
                    else:
                        self._json(game)
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
