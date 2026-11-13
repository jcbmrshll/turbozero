"""The referee of evaluation games (`evaluate.py`), and its players: a game on each side's own
clock or at its fixed budget, one move at a time, each side charged only its own moves, from the
standard start, an opening's moves or a position given whole; with `solve_at`, stopped at so many
empties and solved for the result (`GameRun`). GGS's synchro pairs are two games in lockstep
(`play_synchro`).

A player (`player_maker`) is a network (`net_player`: `cpu_match.NetPlayer` on the CPU, or
`gpu_play.py`'s on the GPU), or an engine at a level (`LevelEngine`) or on its clock
(`engines.ClockedEngine`).

On a contest's clock, who runs out of time, by more than `GRACE`, loses, and the game is played out
with no time for them. On GGS's clock, GGS's rules (`ggs.PlayerClock`, `ggs.game_result`). A game's
`board` result ignores the clock, to show what timeouts changed. Pairs are scored as GGS scores a
synchro match, by the two games' disc results together (see `pair_summary`).
"""

import itertools
import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from engines import (
    CLOCKED,
    ENGINES,
    Board,
    ClockedEgaroucid,
    Egaroucid,
    GTPEngine,
    default_dir,
)
from ggs import (
    Clock,
    PlayerClock,
    Start,
    disc_result,
    game_result,
)

# seconds past the clock that don't lose a game: an engine that plans its whole clock can pass
# it by milliseconds, the pipes and the operating system's doing as much as its own
GRACE = 0.05
# a 95% interval's half width, in standard errors
Z95 = 1.96


class Player(Protocol):
    """One side of a game: `ClockedEngine` is one, and so is a network's search."""

    def set_position(self, black: int, white: int, color: str) -> None: ...

    def play(self, color: str, move: int) -> None: ...

    def move(self, color: str, seconds_left: float) -> int: ...

    def close(self) -> None: ...


class LevelEngine:
    """An engine at a fixed level, which ignores the clock."""

    def __init__(self, engine: GTPEngine):
        self.engine = engine

    def set_position(self, black: int, white: int, color: str) -> None:
        self.engine.set_position(black, white, color)

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
        margin: black's discs less white's at the end, whatever the clock
        result: black's result, as GGS counts it: the empty squares the winner's, and
            timeouts' losses (`ggs.game_result`)
        seconds: each side's time used, black's then white's
        move_seconds: the time of each move searched, as (empties, black?, seconds)
        flagged: who ran out of time first, "b" or "w", if anyone did (a soft timeout)
        hard: who ran out of a GGS clock's extension too, which ended the game
        start: the position it started from, if not the standard one (`ggs.Start.text`)
        moves: every move, the opening's first, as squares (bit 0 is a1); passes aren't moves
        solved_at: the empties of the position solved for the result, if the game was stopped
            there (`GameRun`'s `solve_at`)
    """

    margin: int = 0
    result: int = 0
    seconds: list[float] = field(default_factory=lambda: [0.0, 0.0])
    move_seconds: list[tuple[int, bool, float]] = field(default_factory=list)
    flagged: str | None = None
    hard: str | None = None
    start: str | None = None
    moves: list[int] = field(default_factory=list)
    solved_at: int | None = None


# the final margin for the player to move of a position, as (mover's discs, opponent's)
Solve = Callable[[int, int], int]

# a clock that never runs out, for a player at a fixed budget, who ignores it
UNTIMED = Clock.contest(1e9)


def endgame_solve(exact: bool) -> Solve:
    """Our solver (`endgame.py`) on all the cores we may run on, for a position's final margin,
    or with `exact` False, about ten times faster, a margin whose sign alone is right."""
    from endgame import Solver

    solver = Solver(threads=len(os.sched_getaffinity(0)))

    def solve(me: int, opp: int) -> int:
        solved = solver.solve(me, opp, exact=exact)
        assert solved is not None
        return solved.margin

    return solve


class GameRun:
    """A game being played, a move at a time (`step`), from `start` (else the standard start)
    after `opening`'s moves, each side charged its own moves on `clock` (a float: a contest's
    clock of so many seconds, `Clock.contest`), or each on its own, (black's, white's); `UNTIMED`
    for a side at a fixed budget. On a contest's clock it's played out after a timeout, with no
    time left for that side; on a GGS clock, a hard timeout ends it. With `solve_at`, it stops
    at the first position with so many empties or fewer and `solve` gives the result, as if both
    sides played perfectly from there.

    A pass is a step of its own, as on GGS, where a player with no move sends "pass": it takes
    no time, and the players aren't asked (both engines and the network pass by themselves)."""

    def __init__(
        self,
        black: Player,
        white: Player,
        opening: Sequence[int],
        clock: Clock | float | tuple[Clock, Clock],
        start: Start | None = None,
        solve_at: int | None = None,
        solve: Solve | None = None,
    ):
        if solve_at is not None and solve is None:
            raise ValueError("solve_at needs a solve")
        if isinstance(clock, tuple):
            clocks = clock
        else:
            clock = clock if isinstance(clock, Clock) else Clock.contest(clock)
            clocks = (clock, clock)
        self.solve_at = solve_at
        self.solve = solve
        self.board = start.board() if start is not None else Board()
        self.game = Game(start=start.text() if start is not None else None)
        self.players = (black, white)
        if start is not None:
            for player in self.players:
                player.set_position(start.black, start.white, start.color)
        for move in opening:
            color = self.board.color
            self.board.play(int(move))
            self.game.moves.append(int(move))
            for player in self.players:
                player.play(color, int(move))
        self.clocks = (PlayerClock(clocks[0], GRACE), PlayerClock(clocks[1], GRACE))
        # whether the next step is a pass, of the player who moved last's opponent
        self.passing = False
        # black's margin of the position solved, once the game stopped to solve it
        self.solved_margin = 0
        self.over = not self.board.legal()

    def step(self) -> None:
        """The next move of the game, not over."""
        if self.passing:
            self.passing = False
            return
        board, game = self.board, self.game
        side = not board.black_to_move
        color = board.color
        empties = 64 - (board.discs[0] | board.discs[1]).bit_count()
        if self.solve_at is not None and empties <= self.solve_at:
            assert self.solve is not None
            margin = self.solve(board.discs[side], board.discs[not side])
            game.solved_at = empties
            self.solved_margin = margin if board.black_to_move else -margin
            self.over = True
            return
        begin = time.perf_counter()
        move = self.players[side].move(color, self.clocks[side].seconds_left())
        spent = time.perf_counter() - begin
        clock = self.clocks[side]
        clock.charge(spent)
        game.seconds[side] += spent
        game.move_seconds.append((empties, not side, spent))
        if game.flagged is None and clock.soft:
            game.flagged = color
        board.play(move)
        game.moves.append(int(move))
        if clock.hard:
            game.hard = color
            self.over = True
            return
        self.players[not side].play(color, move)
        self.over = not board.legal()
        # the referee passed for the opponent, with no move
        self.passing = not self.over and board.color == color

    def finish(self) -> Game:
        """The game, once it's over."""
        if self.game.solved_at is not None:
            self.game.margin = self.solved_margin
            board_result = self.solved_margin
        else:
            self.game.margin = self.board.margin()
            board_result = disc_result(self.board)
        self.game.result = game_result(board_result, self.game.flagged, self.game.hard)
        return self.game


def play_game(
    black: Player,
    white: Player,
    opening: Sequence[int],
    clock: Clock | float,
    start: Start | None = None,
) -> Game:
    """Plays a game to its end (see `GameRun`)."""
    run = GameRun(black, white, opening, clock, start)
    while not run.over:
        run.step()
    return run.finish()


def step_catching(run: GameRun, errors: list[BaseException]) -> None:
    """`run.step()`, what it raises put in `errors`, for a thread to raise in its caller's."""
    try:
        run.step()
    except BaseException as error:  # noqa: BLE001 - raised in the caller's thread
        errors.append(error)


def play_synchro(runs: Sequence[GameRun]) -> list[Game]:
    """Plays games in lockstep, as GGS plays a synchro match's two: in each round, each game
    not over makes its next move (or pass), all at once, each in a thread of its own, and the
    next round starts once all have."""
    while active := [run for run in runs if not run.over]:
        errors: list[BaseException] = []
        threads = [
            threading.Thread(target=step_catching, args=(run, errors)) for run in active
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
    return [run.finish() for run in runs]


def points(result: int, a_black: bool) -> float:
    """What a game of black's `result` is worth to player a: 1 for a win, 0.5 for a draw."""
    return 0.5 + 0.5 * float(np.sign(result if a_black else -result))


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


def wilson(wins: int, n: int) -> tuple[float, float]:
    """A 95% interval of a proportion of `wins` in `n` (Wilson's)."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    center = (p + Z95**2 / (2 * n)) / (1 + Z95**2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / (1 + Z95**2 / n)
    return center - half, center + half


def pair_summary(games: list[Game], a_black: list[bool]) -> dict:
    """What a match's pairs showed, for player a: each pair two games from one start, the
    colors swapped, one after the other in `games`.

    - `split`, `won`, `lost`: pairs by game points, 1-1, more or less (a draw a half)
    - `ggs_won`, `ggs_drawn`, `ggs_lost`: by GGS's synchro result, the sign of the two disc
      results' sum
    - `disc_sum`: the mean of a's two disc results summed, a pair, with a 95% interval
    - `decisive_elo`: from the decisive pairs alone, the Elo of a's share of them, won to lost,
      with a 95% interval (Wilson's). Were a pair's games independent, it would be twice a
      game's Elo; pairs whose start decides both games make it more
    - `game_elo`: from every game's points, with an interval from the pairs' spread, as the
      two games of a pair are not independent
    """
    sums, scores = [], []
    for i in range(0, len(games) - 1, 2):
        pair = [(games[i], a_black[i]), (games[i + 1], a_black[i + 1])]
        sums.append(sum(g.result if b else -g.result for g, b in pair))
        scores.append(sum(points(g.result, b) for g, b in pair))
    n = len(sums)
    if n == 0:
        return {"pairs": 0}
    sums_, scores_ = np.array(sums, float), np.array(scores) / 2
    won, lost = int(np.sum(scores_ > 0.5)), int(np.sum(scores_ < 0.5))
    # intervals need two pairs, and the decisive pairs' one a decisive pair: None without
    spread = float(np.std(sums_, ddof=1)) / math.sqrt(n) if n > 1 else None
    score_spread = float(np.std(scores_, ddof=1)) / math.sqrt(n) if n > 1 else None
    low, high = wilson(won, won + lost)
    score, mean = float(scores_.mean()), float(sums_.mean())
    return {
        "pairs": n,
        "split": n - won - lost,
        "won": won,
        "lost": lost,
        "ggs_won": int(np.sum(sums_ > 0)),
        "ggs_drawn": int(np.sum(sums_ == 0)),
        "ggs_lost": int(np.sum(sums_ < 0)),
        "disc_sum": round(mean, 2),
        "disc_sum_ci95": None
        if spread is None
        else [round(mean - Z95 * spread, 2), round(mean + Z95 * spread, 2)],
        "decisive_elo": round(elo(won / (won + lost)), 1) if won + lost else None,
        "decisive_elo_ci95": [round(elo(low), 1), round(elo(high), 1)]
        if won + lost
        else None,
        "game_elo": round(elo(score), 1),
        "game_elo_ci95": None
        if score_spread is None
        else [
            round(elo(score - Z95 * score_spread), 1),
            round(elo(score + Z95 * score_spread), 1),
        ],
        "disc_sums": [int(s) for s in sums],
    }


def summarize(games: list[Game], a_black: list[bool], clock: Clock | float) -> dict:
    """What a match showed, for player a against b."""
    board = [points(g.margin, b) for g, b in zip(games, a_black)]
    final = [points(g.result, b) for g, b in zip(games, a_black)]
    score = float(np.mean(final))
    used = {"a": [], "b": []}
    moves = {"a": [], "b": []}
    flags = {"a": 0, "b": 0}
    hard = {"a": 0, "b": 0}
    for game, black in zip(games, a_black):
        used["a"].append(game.seconds[not black])
        used["b"].append(game.seconds[black])
        for _, by_black, spent in game.move_seconds:
            moves["a" if by_black == black else "b"].append(spent)
        if game.flagged is not None:
            flags["a" if (game.flagged == "b") == black else "b"] += 1
        if game.hard is not None:
            hard["a" if (game.hard == "b") == black else "b"] += 1
    result = {
        "games": len(games),
        "clock": str(clock),
        "score": round(score, 4),
        "stderr": round(float(np.std(final) / math.sqrt(max(len(final), 1))), 4),
        "elo": round(elo(score), 1),
        "board_score": round(float(np.mean(board)), 4),
        "board_wins": int(sum(p == 1 for p in board)),
        "board_draws": int(sum(p == 0.5 for p in board)),
        "board_losses": int(sum(p == 0 for p in board)),
        # a's discs less b's, game by game
        "margins": [int(g.margin if b else -g.margin) for g, b in zip(games, a_black)],
        # a's results as GGS counts them, timeouts and all
        "results": [int(g.result if b else -g.result) for g, b in zip(games, a_black)],
        # clock losses: soft timeouts (the first to run out of a GGS clock's ini, or of a
        # contest's), hard ones, and games whose winner the clock changed
        "flags": flags,
        "hard_timeouts": hard,
        "clock_changed": int(sum(f != b for f, b in zip(final, board))),
        "pairs": pair_summary(games, a_black),
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
    `visits`, it searches so many a move, ignoring the clock. With `solve=N`, positions with N or
    fewer empties are solved on the same cores (see `cpu_match.Solving`), `solve_exact=1` for the
    best margin, `solve_share` what of the clock left a solve may take.

    With `device=gpu`, a GGS-only player (engine comparisons stay on CPU vector units): its tree
    on the host, its leaves evaluated on the GPU many at a time (see `gpu_play.player_factory`
    for its options), the endgame solved on `threads` CPU cores; all its games share one GPU
    stream, as a synchro pair's two do."""
    from cpu_match import NetPlayer, Solving, make_engine, make_eval
    from cpu_match import Player as NetSpec

    options = dict(options)
    if options.get("device", "cpu") == "gpu":
        from gpu_play import player_factory

        cores = int(options.pop("threads", len(os.sched_getaffinity(0))))
        return player_factory(options, cores)
    options.pop("device", None)
    native = bool(int(options.pop("engine", 0)))
    cores = int(options.pop("threads", len(os.sched_getaffinity(0))))
    solving = None
    if "solve" in options:
        solving = Solving(
            empties=int(options.pop("solve")),
            exact=bool(int(options.pop("solve_exact", 0))),
            share=float(options.pop("solve_share", Solving.share)),
            threads=cores,
        )
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
    return lambda: NetPlayer(
        spec.settings, evaluate, taper, next(seeds), spec.visits, solving
    )


def player_maker(
    name: str, options: dict[str, str], clock: Clock
) -> Callable[[], Player]:
    """What builds the player `name` ("net" or an engine's) with `options`, for each game.
    Engines plan for `clock.ini`, told what's left of the clock before each move."""
    options = dict(options)
    if name == "net":
        return net_player(options)
    engine_dir = Path(options["dir"]) if "dir" in options else default_dir(name)
    threads = int(options.get("threads", len(os.sched_getaffinity(0))))
    args = options.get("args", "").split()
    if "level" in options:
        level = int(options["level"])
        if name == Egaroucid.name:
            # its console, which can take a board, as its GTP can't
            return lambda: ClockedEgaroucid(engine_dir, 0, threads, args, level=level)
        return lambda: LevelEngine(
            ENGINES[name](engine_dir, level, args=args, threads=threads)
        )
    return lambda: CLOCKED[name](engine_dir, clock.ini, threads, args)
