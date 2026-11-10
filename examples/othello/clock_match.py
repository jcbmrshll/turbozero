"""Games on a clock, one at a time on the same cores, as in a contest: each side has the same
seconds for the whole game, and only its own moves are charged. Run under `taskset`, with both
sides on the same cores and nothing else on those cores or their SMT siblings.

    taskset -c 0-5,12-13 uv run examples/othello/clock_match.py \
        --a edax:threads=8 --b egaroucid:threads=8 --seconds 10 --openings 100

A player is `net:net=NAME,batch=16,...` (a network of the league's store, see `cpu_match.py`), or
`name:key=value,...`: `edax` or `egaroucid`, with `threads`, `dir` (where the
engine is built), `level` (play at this level, ignoring the clock, its book off) and `args`
(more of the engine's own options, separated by spaces). Each opening is played with both
colors, each game by fresh processes, so nothing carries over from one game to the next.

Who runs out of time, by more than `GRACE`, loses. A game's `board` result ignores that, to
show what clock losses changed.
"""

import argparse
import itertools
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# a CPU script: JAX would otherwise take the GPU's memory, and the league would give way
if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from engines import CLOCKED, ENGINES, GTPEngine
from game import square
from xot import XOT_PATH, load_xot

# seconds past the clock that don't lose a game: an engine that plans its whole clock can pass
# it by milliseconds, the pipes and the operating system's doing as much as its own
GRACE = 0.05
DIRECTIONS = (1, -1, 8, -8, 9, -9, 7, -7)


def flips(mine: int, theirs: int, move: int) -> int:
    """The discs that `move` turns, as a bitboard (bit 0 is a1, bit 63 is h8)."""
    result = 0
    for step in DIRECTIONS:
        run, at = 0, move
        while True:
            column = at % 8
            at += step
            # off the board, or wrapped around from one edge to the other
            if not 0 <= at < 64 or abs(at % 8 - column) > 1:
                run = 0
                break
            if theirs >> at & 1:
                run |= 1 << at
            else:
                if not mine >> at & 1:
                    run = 0
                break
        result |= run
    return result


class Board:
    """The rules, enough to referee: the discs, who's to move, and passes."""

    def __init__(self) -> None:
        # bits for black's and white's discs (d5 and e4 are black, d4 and e5 white)
        self.discs = [1 << 28 | 1 << 35, 1 << 27 | 1 << 36]
        self.black_to_move = True

    @property
    def color(self) -> str:
        return "b" if self.black_to_move else "w"

    def legal(self) -> list[int]:
        """The moves of the player to move."""
        mine = self.discs[not self.black_to_move]
        theirs = self.discs[self.black_to_move]
        taken = mine | theirs
        return [m for m in range(64) if not taken >> m & 1 and flips(mine, theirs, m)]

    def play(self, move: int) -> None:
        """Plays a legal `move`, then passes for the next player if they have no move."""
        side = not self.black_to_move
        turned = flips(self.discs[side], self.discs[not side], move)
        if not turned:
            raise ValueError(f"illegal move {square(move)}")
        self.discs[side] |= turned | 1 << move
        self.discs[not side] &= ~turned
        self.black_to_move = not self.black_to_move
        if not self.legal():
            self.black_to_move = not self.black_to_move

    def margin(self) -> int:
        """Black's discs less white's."""
        return self.discs[0].bit_count() - self.discs[1].bit_count()


class Player(Protocol):
    """One side of a game: `ClockedEngine` is one, and so is a network's search."""

    def play(self, color: str, move: int) -> None: ...

    def move(self, color: str, seconds_left: float) -> int: ...

    def close(self) -> None: ...


class LevelEngine:
    """An engine at a fixed level, which ignores the clock."""

    def __init__(self, engine: GTPEngine):
        self.engine = engine

    def play(self, color: str, move: int) -> None:
        self.engine.play(color, move)

    def move(self, color: str, seconds_left: float) -> int:
        return self.engine.genmove(color)

    def close(self) -> None:
        self.engine.close()


@dataclass
class Game:
    """How a game went.

    Attributes:
        margin: black's discs less white's at the end
        seconds: each side's time used, black's then white's
        move_seconds: the time of each move searched, as (empties, black?, seconds)
        flagged: who ran out of time first, "b" or "w", if anyone did
    """

    margin: int = 0
    seconds: list[float] = field(default_factory=lambda: [0.0, 0.0])
    move_seconds: list[tuple[int, bool, float]] = field(default_factory=list)
    flagged: str | None = None


def play_game(
    black: Player, white: Player, opening: Sequence[int], seconds: float
) -> Game:
    """Plays a game to its end from `opening`, charging each side its own moves. It's played
    out after a clock runs out (with no time left, for that side) for the board result."""
    board = Board()
    game = Game()
    players = (black, white)
    for move in opening:
        color = board.color
        board.play(int(move))
        for player in players:
            player.play(color, int(move))
    while board.legal():
        side = not board.black_to_move
        color = board.color
        empties = 64 - (board.discs[0] | board.discs[1]).bit_count()
        start = time.perf_counter()
        move = players[side].move(color, seconds - game.seconds[side])
        spent = time.perf_counter() - start
        game.seconds[side] += spent
        game.move_seconds.append((empties, not side, spent))
        if game.flagged is None and game.seconds[side] > seconds + GRACE:
            game.flagged = color
        board.play(move)
        players[not side].play(color, move)
    game.margin = board.margin()
    return game


def points(margin: int, flagged: str | None, a_black: bool) -> float:
    """What a game is worth to player a: 1 for a win, 0.5 for a draw."""
    if flagged is not None:
        return float((flagged == "w") == a_black)
    return 0.5 + 0.5 * np.sign(margin if a_black else -margin)


def elo(score: float) -> float:
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def phase_seconds(games: list[Game], a_black: list[bool], a: bool) -> dict[str, float]:
    """A player's mean seconds a game in three parts of the game by empty squares."""
    parts = {"mid_34+": 0.0, "late_24-33": 0.0, "end_0-23": 0.0}
    for game, black in zip(games, a_black):
        for empties, by_black, spent in game.move_seconds:
            if by_black == (black == a):
                part = (
                    "mid_34+"
                    if empties >= 34
                    else "late_24-33"
                    if empties >= 24
                    else "end_0-23"
                )
                parts[part] += spent
    return {k: round(v / max(len(games), 1), 3) for k, v in parts.items()}


def summarize(games: list[Game], a_black: list[bool], seconds: float) -> dict:
    """What a match showed, for player a against b."""
    board = [points(g.margin, None, b) for g, b in zip(games, a_black)]
    final = [points(g.margin, g.flagged, b) for g, b in zip(games, a_black)]
    score = float(np.mean(final))
    used = {"a": [], "b": []}
    moves = {"a": [], "b": []}
    flags = {"a": 0, "b": 0}
    for game, black in zip(games, a_black):
        used["a"].append(game.seconds[not black])
        used["b"].append(game.seconds[black])
        for _, by_black, spent in game.move_seconds:
            moves["a" if by_black == black else "b"].append(spent)
        if game.flagged is not None:
            flags["a" if (game.flagged == "b") == black else "b"] += 1
    result = {
        "games": len(games),
        "seconds": seconds,
        "score": round(score, 4),
        "stderr": round(float(np.std(final) / math.sqrt(max(len(final), 1))), 4),
        "elo": round(elo(score), 1),
        "board_score": round(float(np.mean(board)), 4),
        "board_wins": int(sum(p == 1 for p in board)),
        "board_draws": int(sum(p == 0.5 for p in board)),
        "board_losses": int(sum(p == 0 for p in board)),
        # a's discs less b's, game by game
        "margins": [int(g.margin if b else -g.margin) for g, b in zip(games, a_black)],
        "flags": flags,
    }
    for side in ("a", "b"):
        result[side] = {
            "seconds_a_game": round(float(np.mean(used[side])), 3),
            "seconds_a_game_max": round(float(np.max(used[side])), 3),
            "seconds_a_move": round(float(np.mean(moves[side])), 4),
            "seconds_a_move_max": round(float(np.max(moves[side])), 3),
            "seconds_a_move_p99": round(float(np.percentile(moves[side], 99)), 3),
            "phases": phase_seconds(games, a_black, side == "a"),
        }
    return result


def net_player(options: dict[str, str]) -> Callable[[], Player]:
    """What builds a network's player, which all games share a network for. Its `options` are
    `cpu_match.Player.parse`'s, with `net`, `taper` (see `cpu_match.move_budget`), `engine=1`
    to search with a worker thread to a core (`cpu_search.Engine`), `batch` then the leaves each
    evaluates at a time, and `threads`, the cores used, by default all it may run on. With
    `visits`, it searches so many a move, ignoring the clock."""
    from cpu_match import NetPlayer, make_engine, make_eval
    from cpu_match import Player as NetSpec

    options = dict(options)
    native = bool(int(options.pop("engine", 0)))
    cores = int(options.pop("threads", len(os.sched_getaffinity(0))))
    fields = ",".join(f"{k}={v}" for k, v in options.items())
    spec = NetSpec.parse(f"visits=0,{fields}", options["net"], "bf16")
    evaluate = (
        make_engine(spec.net, spec.backend, cores, spec.settings.batch)
        if native
        else make_eval(spec.net, spec.backend, cores)
    )
    taper = int(options.get("taper", 0))
    # a seed to a game, for its draws of symmetries
    seeds = itertools.count()
    return lambda: NetPlayer(spec.settings, evaluate, taper, next(seeds), spec.visits)


def make_player(spec: str, seconds: int) -> Callable[[], Player]:
    """What builds a player from its `name:key=value,...`, for each game."""
    name, _, settings = spec.partition(":")
    options = dict(s.split("=", 1) for s in settings.split(",") if s)
    if name == "net":
        return net_player(options)
    engine_dir = (
        Path(options["dir"]) if "dir" in options else ENGINES[name].default_dir()
    )
    threads = int(options.get("threads", len(os.sched_getaffinity(0))))
    args = options.get("args", "").split()
    if "level" in options:
        level = int(options["level"])
        return lambda: LevelEngine(
            ENGINES[name](engine_dir, level, args=args, threads=threads)
        )
    return lambda: CLOCKED[name](engine_dir, seconds, threads, args)


def play_match(
    a: Callable[[], Player],
    b: Callable[[], Player],
    openings: np.ndarray,
    seconds: float,
    on_game: Callable[[list[Game], list[bool]], None] = lambda _games, _a_black: None,
) -> tuple[list[Game], list[bool]]:
    """Each opening played with both colors, one game at a time, by fresh players."""
    games, a_black = [], []
    for opening in openings:
        for black in (True, False):
            first, second = a(), b()
            try:
                black_player, white_player = (
                    (first, second) if black else (second, first)
                )
                games.append(play_game(black_player, white_player, opening, seconds))
                a_black.append(black)
            finally:
                first.close()
                second.close()
            on_game(games, a_black)
    return games, a_black


def main() -> None:
    parser = argparse.ArgumentParser(description="Games on a clock, one at a time.")
    parser.add_argument("--a", required=True, help="e.g. edax:threads=8")
    parser.add_argument("--b", required=True)
    parser.add_argument("--seconds", type=int, required=True, help="each, per game")
    parser.add_argument("--openings", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    pool = load_xot(XOT_PATH)
    rng = np.random.default_rng(args.seed)
    openings = pool[rng.choice(len(pool), size=args.openings, replace=False)]

    def report(games: list[Game], a_black: list[bool]) -> None:
        if len(games) % 20 == 0:
            print(json.dumps(summarize(games, a_black, args.seconds)), flush=True)

    games, a_black = play_match(
        make_player(args.a, args.seconds),
        make_player(args.b, args.seconds),
        openings,
        args.seconds,
        report,
    )
    record = {
        "player_a": args.a,
        "player_b": args.b,
        "seed": args.seed,
        "cores": sorted(os.sched_getaffinity(0)),
        "load": os.getloadavg()[0],
        **summarize(games, a_black, args.seconds),
    }
    print(json.dumps(record), flush=True)
    if args.results is not None:
        with open(args.results, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
