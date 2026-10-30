"""The league's web page: a server that only reads the store, and never touches the GPU.

Ratings are fit from the games whenever there are new ones.
"""

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from core.league.game import LeagueGame
from core.league.ratings import Ratings, expected_score, fit_ratings
from core.league.store import Player, Store, decode_moves

PAGE = Path(__file__).with_name("dashboard.html")
# how long the supervisor can go without a word before it counts as not running
SUPERVISOR_SILENCE = 300.0


class League:
    """What the page shows, read from `store`, which it uses one thread at a time."""

    def __init__(self, store: Store, game: LeagueGame):
        self.store = store
        self.game = game
        self.lock = threading.Lock()
        self.fitted: tuple[int, int] | None = None
        self.ratings: Ratings | None = None

    def _fit(self, players: list[Player]) -> Ratings:
        """The fit to all the games, redone if there's more."""
        version = (self.store.num_games(), len(players))
        if self.fitted != version or self.ratings is None:
            ids = [p.id for p in players]
            records = self.store.pair_records()
            anchor = self.store.get_meta("anchor")
            self.ratings = fit_ratings(ids, records, anchor=anchor)
            self.fitted = version
        return self.ratings

    def summary(self) -> dict[str, Any]:
        with self.lock:
            store = self.store
            players = store.players()
            ratings = self._fit(players)
            anchor = store.get_meta("anchor")
            rows = store.db.execute(
                "SELECT b.id, pa.name AS a, pb.name AS b, b.seconds, b.compiled, "
                "b.code_version, b.finished, COUNT(g.id) AS games FROM batches b "
                "JOIN players pa ON pa.id = b.a_id JOIN players pb ON pb.id = b.b_id "
                "JOIN games g ON g.batch_id = b.id GROUP BY b.id "
                "ORDER BY b.id DESC LIMIT 12"
            ).fetchall()
            per_move = store.seconds_per_move()
            total = store.db.execute(
                "SELECT COUNT(*), COALESCE(SUM(seconds), 0) FROM batches"
            ).fetchone()
            return {
                "game": self.game.name,
                "games": store.num_games(),
                "batches": total[0],
                "seconds": total[1],
                "anchor": anchor,
                "parts": len(set(ratings.components.tolist())),
                "players": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "elo": float(ratings.elo[i]),
                        "sd": float(ratings.sd[i]),
                        "games": int(ratings.games[i]),
                        "sims": p.sims,
                        "network": p.network,
                        "net": p.net,
                        "url": p.url,
                        "seconds_per_move": per_move.get(p.id),
                        "run": p.run,
                        "external": p.external,
                        "settings": p.settings,
                        "retired": p.retired,
                        "part": int(ratings.components[i]),
                    }
                    for i, p in enumerate(players)
                ],
                "pairs": [
                    [r.a, r.b, r.wins, r.draws, r.losses] for r in store.pair_records()
                ],
                "recent": [dict(row) for row in rows],
                "worker": self._worker(),
                "time": time.time(),
            }

    def _worker(self) -> dict[str, Any]:
        supervisor = self.store.get_meta("supervisor") or {}
        player = self.store.get_meta("player") or {}
        running = (
            supervisor.get("state") not in (None, "stopped")
            and time.time() - supervisor.get("time", 0) < SUPERVISOR_SILENCE
        )
        return {
            "state": supervisor.get("state", "stopped") if running else "stopped",
            "reason": supervisor.get("reason"),
            "job": supervisor.get("job"),
            "player": player,
        }

    def pair(self, a: int, b: int) -> dict[str, Any]:
        with self.lock:
            players = self.store.players()
            ratings = self._fit(players)
            i, j = ratings.index(a), ratings.index(b)
            difference = float(ratings.elo[i] - ratings.elo[j])
            return {
                "difference": difference,
                "sd": ratings.pair_sd(a, b),
                "expected": float(expected_score(difference)),
            }

    def games(
        self, player: int | None, opponent: int | None, offset: int, limit: int
    ) -> dict[str, Any]:
        where, values = [], []
        if player is not None:
            where.append("(g.first_id = ? OR g.second_id = ?)")
            values += [player, player]
        if opponent is not None:
            where.append("(g.first_id = ? OR g.second_id = ?)")
            values += [opponent, opponent]
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self.lock:
            rows = self.store.db.execute(
                "SELECT g.id, pf.name AS first, ps.name AS second, g.first_id, g.second_id, "
                "g.opening, g.result, g.first_score, g.second_score, b.finished, "
                "b.code_version, length(g.moves) / 2 AS plies FROM games g "
                "JOIN players pf ON pf.id = g.first_id "
                "JOIN players ps ON ps.id = g.second_id "
                f"JOIN batches b ON b.id = g.batch_id {clause} "
                "ORDER BY g.id DESC LIMIT ? OFFSET ?",
                (*values, limit, offset),
            ).fetchall()
            total = self.store.db.execute(
                f"SELECT COUNT(*) FROM games g {clause}", values
            ).fetchone()[0]
        return {"games": [dict(row) for row in rows], "total": total}

    def game_record(self, game: int) -> dict[str, Any] | None:
        with self.lock:
            row = self.store.db.execute(
                "SELECT g.id, pf.name AS first, ps.name AS second, g.opening, g.result, "
                "g.first_score, g.second_score, g.moves, b.finished, b.code_version, "
                "b.dtype FROM games g JOIN players pf ON pf.id = g.first_id "
                "JOIN players ps ON ps.id = g.second_id "
                "JOIN batches b ON b.id = g.batch_id WHERE g.id = ?",
                (game,),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["moves"] = decode_moves(row["moves"])
        record["opening_plies"] = self.game.openings(
            np.array([row["opening"]], dtype=int)
        ).shape[1]
        return record


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
            query = parse_qs(url.query)

            def number(name: str, default: int | None = None) -> int | None:
                values = query.get(name)
                return int(values[0]) if values and values[0] else default

            try:
                if url.path == "/":
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/game.js":
                    board = league.game.board_js
                    if board is None:
                        self._send(b"export default null;", "text/javascript")
                    else:
                        self._send(board.read_bytes(), "text/javascript")
                elif url.path == "/api/summary":
                    self._json(league.summary())
                elif url.path == "/api/pair":
                    a, b = number("a"), number("b")
                    if a is None or b is None:
                        self._json({"error": "needs a and b"}, 400)
                    else:
                        self._json(league.pair(a, b))
                elif url.path == "/api/games":
                    self._json(
                        league.games(
                            number("player"),
                            number("opponent"),
                            number("offset", 0) or 0,
                            min(number("limit", 50) or 50, 200),
                        )
                    )
                elif url.path == "/api/game":
                    record = league.game_record(number("id", -1) or -1)
                    if record is None:
                        self._json({"error": "no such game"}, 404)
                    else:
                        self._json(record)
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as e:
                self._json({"error": str(e)}, 400)

    return Handler


def serve(
    store: Store, make_game: Callable[[], LeagueGame], host: str, port: int
) -> None:
    """Serves the page until it's stopped."""
    server = ThreadingHTTPServer((host, port), make_handler(League(store, make_game())))
    print(f"serving on {host}:{port}", flush=True)
    server.serve_forever()
