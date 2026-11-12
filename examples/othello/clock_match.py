"""Games on a clock, one at a time on the same cores, as in a contest: each side has the same
seconds for the whole game, and only its own moves are charged. Run under `taskset`, with both
sides on the same cores and nothing else on those cores or their SMT siblings.

    taskset -c 0-5,12-13 uv run examples/othello/clock_match.py \
        --a edax:threads=8 --b egaroucid:threads=8 --seconds 10 --openings 100

Or as GGS plays its synchro matches (see `ggs.py`): from random 14-disc starts, on its clock of
1 minute a game and 2 of extension:

    taskset -c 0-7 uv run examples/othello/clock_match.py \
        --a net:net=NAME,engine=1 --b edax --clock 01:00//02:00 --starts random:14 --openings 50

`--starts` is `xot` (XOT openings, played as moves from the standard start), `random:N` (GGS's
random starts of N discs) or `file:PATH` (starts drawn from a file: a GGS archive's of
`--match-type`, or `ggs.py extract`'s). Each start is played with both colors, a pair of games,
each by fresh processes, so nothing carries over from one game to the next. With `--parallel`,
a pair's two games are played at the same time on the same cores, as GGS plays them.

A player is `net:net=NAME,batch=16,...` (a network of the league's store, see `cpu_match.py`; with
`solve=20`, it solves positions with 20 or fewer empties, see `net_player`), or
`name:key=value,...`: `edax` or `egaroucid`, with `threads`, `dir` (where the
engine is built), `level` (play at this level, ignoring the clock, its book off) and `args`
(more of the engine's own options, separated by spaces).

On `--seconds`, who runs out of time, by more than `GRACE`, loses, and the game is played out
with no time for them. On `--clock`, GGS's rules (`ggs.PlayerClock`, `ggs.game_result`). A
game's `board` result ignores the clock, to show what timeouts changed. Pairs are scored as
GGS scores a synchro match, by the two games' disc results together (see `pair_summary`).
"""

import argparse
import itertools
import json
import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

# a CPU script: JAX would otherwise take the GPU's memory, and the league would give way
if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from engines import CLOCKED, ENGINES, Board, ClockedEgaroucid, Egaroucid, GTPEngine
from ggs import (
    Clock,
    PlayerClock,
    Start,
    disc_result,
    game_result,
    load_starts,
    random_start,
)
from xot import XOT_PATH, load_xot

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
    """

    margin: int = 0
    result: int = 0
    seconds: list[float] = field(default_factory=lambda: [0.0, 0.0])
    move_seconds: list[tuple[int, bool, float]] = field(default_factory=list)
    flagged: str | None = None
    hard: str | None = None
    start: str | None = None
    moves: list[int] = field(default_factory=list)


class GameRun:
    """A game being played, a move at a time (`step`), from `start` (else the standard start)
    after `opening`'s moves, each side charged its own moves on `clock` (a float: a contest's
    clock of so many seconds, `Clock.contest`). On a contest's clock it's played out after a
    timeout, with no time left for that side; on a GGS clock, a hard timeout ends it.

    A pass is a step of its own, as on GGS, where a player with no move sends "pass": it takes
    no time, and the players aren't asked (both engines and the network pass by themselves)."""

    def __init__(
        self,
        black: Player,
        white: Player,
        opening: Sequence[int],
        clock: Clock | float,
        start: Start | None = None,
    ):
        clock = clock if isinstance(clock, Clock) else Clock.contest(clock)
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
        self.clocks = (PlayerClock(clock, GRACE), PlayerClock(clock, GRACE))
        # whether the next step is a pass, of the player who moved last's opponent
        self.passing = False
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
        self.game.margin = self.board.margin()
        self.game.result = game_result(
            disc_result(self.board), self.game.flagged, self.game.hard
        )
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
    best margin, `solve_share` what of the clock left a solve may take."""
    from cpu_match import NetPlayer, Solving, make_engine, make_eval
    from cpu_match import Player as NetSpec

    options = dict(options)
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


def make_player(spec: str, clock: Clock) -> Callable[[], Player]:
    """What builds a player from its `name:key=value,...`, for each game. Engines plan for
    `clock.ini`, told what's left of the clock before each move."""
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
        if name == Egaroucid.name:
            # its console, which can take a board, as its GTP can't
            return lambda: ClockedEgaroucid(engine_dir, 0, threads, args, level=level)
        return lambda: LevelEngine(
            ENGINES[name](engine_dir, level, args=args, threads=threads)
        )
    return lambda: CLOCKED[name](engine_dir, clock.ini, threads, args)


# where a game starts: the moves of an opening, played from a position (None: the standard one)
Opening = tuple[Sequence[int], Start | None]


def play_one(
    a: Callable[[], Player],
    b: Callable[[], Player],
    opening: Opening,
    clock: Clock | float,
    a_black: bool,
) -> Game:
    """One game, by fresh players, closed after."""
    first, second = a(), b()
    try:
        black, white = (first, second) if a_black else (second, first)
        return play_game(black, white, opening[0], clock, opening[1])
    finally:
        first.close()
        second.close()


def play_pair_synchro(
    players: Sequence[tuple[Callable[[], Player], Callable[[], Player]]],
    opening: Opening,
    clock: Clock | float,
) -> list[Game]:
    """A pair's two games in lockstep (`play_synchro`), by fresh players, a black in the first:
    each of `players` (a's maker, b's) plays one game."""
    made: list[Player] = []
    try:
        runs = []
        for (a, b), a_black in zip(players, (True, False)):
            first, second = a(), b()
            made += [first, second]
            black, white = (first, second) if a_black else (second, first)
            runs.append(GameRun(black, white, opening[0], clock, opening[1]))
        return play_synchro(runs)
    finally:
        for player in made:
            player.close()


def play_match(
    a: Callable[[], Player],
    b: Callable[[], Player],
    openings: Sequence[Opening],
    clock: Clock | float,
    on_game: Callable[[list[Game], list[bool]], None] = lambda _games, _a_black: None,
    parallel: tuple[Callable[[], Player], Callable[[], Player]] | None = None,
) -> tuple[list[Game], list[bool]]:
    """Each opening played with both colors, a black first, by fresh players: one game at a
    time, or with `parallel` (players for the second game: a network's can't search two trees
    at once) the pair's two games in lockstep, as GGS plays them (`play_synchro`)."""
    games, a_black = [], []
    for opening in openings:
        if parallel is None:
            for black in (True, False):
                games.append(play_one(a, b, opening, clock, black))
                a_black.append(black)
                on_game(games, a_black)
            continue
        for game, black in zip(
            play_pair_synchro([(a, b), parallel], opening, clock), (True, False)
        ):
            games.append(game)
            a_black.append(black)
            on_game(games, a_black)
    return games, a_black


def choose_openings(
    starts: str, count: int, seed: int, match_type: str, since: str
) -> list[Opening]:
    """`count` openings from `--starts`'s source, drawn by `seed`."""
    rng = np.random.default_rng(seed)
    kind, _, value = starts.partition(":")
    if kind == "xot":
        pool = load_xot(XOT_PATH)
        return [
            (o, None) for o in pool[rng.choice(len(pool), size=count, replace=False)]
        ]
    if kind == "random":
        return [((), random_start(int(value), rng).playable()) for _ in range(count)]
    if kind == "file":
        found = load_starts(value, match_type, since)
        picked = rng.choice(len(found), size=min(count, len(found)), replace=False)
        return [((), found[i].playable()) for i in picked]
    raise ValueError(f"unknown starts {starts!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Games on a clock, one at a time.")
    parser.add_argument("--a", required=True, help="e.g. edax:threads=8")
    parser.add_argument("--b", required=True)
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument("--seconds", type=float, help="each, per game, as in a contest")
    timing.add_argument("--clock", type=Clock.parse, help="GGS's, e.g. 01:00//02:00")
    parser.add_argument("--openings", type=int, default=100, help="pairs of games")
    parser.add_argument("--starts", default="xot", help="xot, random:N or file:PATH")
    parser.add_argument("--match-type", default="s8r14", help="an archive's starts of")
    parser.add_argument(
        "--since", default="", help='an archive\'s starts from, e.g. "2024"'
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="a pair's two games at once, on the same cores",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument(
        "--games",
        type=Path,
        default=None,
        help="where each game is written, a line each",
    )
    args = parser.parse_args()

    clock = args.clock if args.clock is not None else Clock.contest(args.seconds)
    openings = choose_openings(
        args.starts, args.openings, args.seed, args.match_type, args.since
    )

    def report(games: list[Game], a_black: list[bool]) -> None:
        if args.games is not None:
            game = games[-1]
            with open(args.games, "a") as f:
                f.write(json.dumps({"a_black": a_black[-1], **asdict(game)}) + "\n")
        if len(games) % 20 == 0:
            print(json.dumps(summarize(games, a_black, clock)), flush=True)

    second = (
        (make_player(args.a, clock), make_player(args.b, clock))
        if args.parallel
        else None
    )
    games, a_black = play_match(
        make_player(args.a, clock),
        make_player(args.b, clock),
        openings,
        clock,
        report,
        second,
    )
    record = {
        "player_a": args.a,
        "player_b": args.b,
        "starts": args.starts,
        "parallel": args.parallel,
        "seed": args.seed,
        "cores": sorted(os.sched_getaffinity(0)),
        "load": os.getloadavg()[0],
        **summarize(games, a_black, clock),
    }
    print(json.dumps(record), flush=True)
    if args.results is not None:
        with open(args.results, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
