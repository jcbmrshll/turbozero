"""The league's web pages, and the server behind them, which never touches the GPU.

Ratings are fit from the games whenever there are new ones. The server writes to the store only
for what's edited on the page: pauses, descriptions and tags, and the queue of series. The page
runs networks itself, in the browser, on their ONNX exports (see `core.league.runtime`).

It has no login. Anything that writes takes a JSON body, which another site's page can't send
to it from a browser on the tailnet without the server agreeing to it first, which it doesn't.

`timings.json`, beside the store, holds what players' moves cost on fixed hardware, for the
runtime against rating plot. Each of its `runs` is one host: see `pareto`.
"""

import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from core.league.game import LeagueGame, check_match, check_match_games
from core.league.ratings import Ratings, expected_score, fit_ratings
from core.league.runtime import VERSION, has_runtime, runtime_dir
from core.league.store import Player, Store, decode_moves

PAGE = Path(__file__).with_name("dashboard.html")
WEB = Path(__file__).with_name("web")
KINDS = {
    ".css": "text/css",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".wasm": "application/wasm",
    ".onnx": "application/octet-stream",
}
# how long the supervisor can go without a word before it counts as not running
SUPERVISOR_SILENCE = 300.0


class NoSuchPlayer(Exception):
    pass


def _config_of(player: Player) -> tuple[str | int | None, ...]:
    """What a timing is of, for `player`: every network of an architecture costs the same."""
    if player.external is not None:
        return ("engine", player.external, player.settings.get("level"))
    return ("net", player.net, player.sims)


def pareto(
    timings: dict[str, Any], players: list[Player], ratings: Ratings
) -> list[dict[str, Any]]:
    """Each run in `timings`, with its players that have a rating and a measured time: their
    seconds per game (one side's mean) against their rating. Those whose time assumes a batch
    size that hasn't been shown to lose no strength aren't `solid`. `frontier` says who no solid
    player beats on both, among the solid ones. Runs stay apart: each is its own hardware."""
    runs = []
    for run in timings.get("runs", []):
        configs = {}
        for config in run.get("configs", []):
            if config.get("source") != "measured" or not config.get("plot"):
                continue
            key = (
                ("net", config["net"], config.get("sims"))
                if "net" in config
                else ("engine", config.get("engine"), config.get("level"))
            )
            configs[key] = config
        points = []
        for i, player in enumerate(players):
            config = configs.get(_config_of(player))
            if config is None or not ratings.games[i]:
                continue
            points.append(
                {
                    "player": player.id,
                    "name": player.name,
                    "elo": float(ratings.elo[i]),
                    "sd": float(ratings.sd[i]),
                    "seconds": config["seconds_per_game"],
                    "games": config.get("games"),
                    "batch": config.get("batch"),
                    "threads": config.get("threads"),
                    "solid": config.get("batch_strength") != "unmeasured",
                    "frontier": False,
                }
            )
        best = -np.inf
        for point in sorted(points, key=lambda p: (p["seconds"], -p["elo"])):
            if point["solid"] and point["elo"] > best:
                point["frontier"] = True
                best = point["elo"]
        runs.append(
            {
                "name": run.get("name", ""),
                "hardware": run.get("hardware", {}),
                "software": run.get("software", {}),
                "points": points,
            }
        )
    return runs


class League:
    """What the page shows, read from `store`, which it uses one thread at a time."""

    def __init__(self, store: Store, game: LeagueGame):
        self.store = store
        self.game = game
        self.lock = threading.Lock()
        self.fitted: tuple[int, int] | None = None
        self.ratings: Ratings | None = None
        self.timings_read: tuple[float, dict[str, Any]] = (0.0, {})

    def _timings(self) -> dict[str, Any]:
        """`timings.json`, read again if it's changed."""
        path = self.store.root / "timings.json"
        try:
            changed = path.stat().st_mtime
            if changed != self.timings_read[0]:
                self.timings_read = (changed, json.loads(path.read_text()))
        except (OSError, ValueError):
            return {}
        return self.timings_read[1]

    def pareto(self) -> list[dict[str, Any]]:
        with self.lock:
            players = self.store.players()
            return pareto(self._timings(), players, self._fit(players))

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
                        "description": p.description,
                        "tags": list(p.tags),
                        "facts": self.game.facts(p),
                        "onnx": self._onnx(p),
                        "can_play": self.game.can_play(p),
                        "part": int(ratings.components[i]),
                    }
                    for i, p in enumerate(players)
                ],
                "pairs": [
                    [r.a, r.b, r.wins, r.draws, r.losses] for r in store.pair_records()
                ],
                "recent": [dict(row) for row in rows],
                "worker": self._worker(),
                "playing": self._playing(),
                "requests": self._queue(players),
                "runtime": VERSION if has_runtime(store.root) else None,
                "time": time.time(),
            }

    def _onnx(self, player: Player) -> str | None:
        """The name of `player`'s network's ONNX export, if it has one yet."""
        path = self.store.onnx_path(player)
        return path.name if path is not None and path.exists() else None

    def _queue(self, players: list[Player]) -> list[dict[str, Any]]:
        """The series still to finish, the next first, each with what's happening to it:
        "playing", "next" (the first the worker can play after what's playing), "paused" or
        "can't play" (skipped until that changes), or "" for its turn to come."""
        by_id = {p.id: p for p in players}
        playing = (self._playing() or {}).get("request")
        queue, waiting = [], True
        for request in self.store.pending_matches():
            sides = by_id[request.a], by_id[request.b]
            status = ""
            if request.id == playing:
                status = "playing"
            elif not all(self.game.can_play(p) for p in sides):
                status = "can't play"
            elif any(p.retired for p in sides):
                status = "paused"
            elif waiting:
                status = "next"
                waiting = False
            queue.append({**asdict(request), "status": status})
        return queue

    def add_match(self, a: int, b: int, games: int) -> int:
        """Asks for a series, after those already asked for.

        Raises:
            ValueError: if it can't be played
        """
        with self.lock:
            by_id = {p.id: p for p in self.store.players()}
            if a not in by_id or b not in by_id:
                raise ValueError("no such player")
            check_match(self.game, by_id[a], by_id[b], games)
            return self.store.request_match(a, b, games)

    def move_match(self, request: int, index: int) -> None:
        with self.lock:
            self.store.move_match(request, index)

    def set_match_games(self, request: int, games: int) -> None:
        with self.lock:
            check_match_games(games)
            self.store.set_match_games(request, games)

    def cancel_match(self, request: int) -> None:
        with self.lock:
            self.store.cancel_match(request)

    def set_notes(
        self, player: int, description: str | None, tags: list[str] | None
    ) -> None:
        with self.lock:
            self.store.set_notes(player, description, tags)

    def _playing(self) -> dict[str, Any] | None:
        """The batch being played, if its player is still alive to be playing it."""
        player = self.store.get_meta("player") or {}
        if player.get("state") != "playing":
            return None
        try:
            os.kill(player["pid"], 0)
        except OSError:
            return None
        return player

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

    def set_paused(self, player: int, paused: bool) -> bool:
        """Takes `player` out of the draw for new games, or puts it back; False if there's no
        such player or it can't play here."""
        with self.lock:
            for p in self.store.players():
                if p.id == player and self.game.can_play(p):
                    self.store.set_retired(p.name, paused)
                    return True
        return False

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
                "SELECT g.id, pf.name AS first, ps.name AS second, g.first_id, g.second_id, "
                "g.opening, g.result, g.first_score, g.second_score, g.moves, b.finished, b.code_version, "
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

        def _file(self, path: Path, cached: bool) -> None:
            """Sends a file, which `cached` says will never change."""
            try:
                body = path.read_bytes()
            except OSError:
                self._json({"error": "not found"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", KINDS[path.suffix])
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Cache-Control",
                "public, max-age=31536000, immutable" if cached else "no-store",
            )
            self.end_headers()
            self.wfile.write(body)

        def _post(self, path: str, body: dict[str, Any]) -> Any:
            """Does what's asked: its answer, or None if there's no such thing to ask."""
            if not isinstance(body, dict):
                raise TypeError("needs a JSON object")
            if path == "/api/paused":
                player, paused = int(body["player"]), body["paused"]
                if not isinstance(paused, bool):
                    raise TypeError("paused is true or false")
                if not league.set_paused(player, paused):
                    raise NoSuchPlayer("no such player can play")
                return {"player": player, "paused": paused}
            if path == "/api/notes":
                description, tags = body.get("description"), body.get("tags")
                if not isinstance(description, str | None):
                    raise TypeError("a description is text")
                if tags is not None and not (
                    isinstance(tags, list) and all(isinstance(t, str) for t in tags)
                ):
                    raise TypeError("tags are a list of text")
                league.set_notes(int(body["player"]), description, tags)
                return {}
            if path == "/api/match/add":
                request = league.add_match(
                    int(body["a"]), int(body["b"]), int(body["games"])
                )
                return {"id": request}
            if path == "/api/match/move":
                league.move_match(int(body["id"]), int(body["index"]))
                return {}
            if path == "/api/match/games":
                league.set_match_games(int(body["id"]), int(body["games"]))
                return {}
            if path == "/api/match/cancel":
                league.cancel_match(int(body["id"]))
                return {}
            return None

        def do_POST(self) -> None:
            if self.headers.get("Content-Type") != "application/json":
                self._json({"error": "needs a JSON body"}, 415)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                answer = self._post(
                    urlparse(self.path).path, json.loads(self.rfile.read(length))
                )
            except NoSuchPlayer as e:
                self._json({"error": str(e)}, 404)
            except (KeyError, TypeError, ValueError) as e:
                self._json(
                    {"error": repr(e) if isinstance(e, KeyError) else str(e)}, 400
                )
            else:
                if answer is None:
                    self._json({"error": "not found"}, 404)
                else:
                    self._json(answer)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)

            def number(name: str, default: int | None = None) -> int | None:
                values = query.get(name)
                return int(values[0]) if values and values[0] else default

            try:
                if url.path == "/":
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif match := re.fullmatch(r"/web/(\w+\.(?:js|css))", url.path):
                    self._file(WEB / match[1], cached=False)
                elif match := re.fullmatch(
                    rf"/ort/{re.escape(VERSION)}/([\w.-]+\.(?:mjs|wasm))", url.path
                ):
                    self._file(runtime_dir(league.store.root) / match[1], cached=True)
                elif match := re.fullmatch(r"/onnx/([0-9a-f]{64}\.onnx)", url.path):
                    self._file(league.store.root / "networks" / match[1], cached=True)
                elif url.path == "/game.js":
                    board = league.game.board_js
                    if board is None:
                        self._send(b"export default null;", "text/javascript")
                    else:
                        self._send(board.read_bytes(), "text/javascript")
                elif url.path == "/api/summary":
                    self._json(league.summary())
                elif url.path == "/api/pareto":
                    self._json(league.pareto())
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
