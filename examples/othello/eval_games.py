"""The games of an evaluation match (`evaluate.py`), wherever they're played: each written as a
`core.evals.store.GameRecord` as it ends.

A match is two players (`eval_players.PlayerSpec`) and its games' setup: `openings` starts drawn
by `seed` from `starts` (XOT openings, GGS's random starts, or a file's), each played with both
colors; with `solve_at`, each game stopped at its first position with that many empties or fewer
and solved for the result, the same for both players (`endgame.py`; for a win, a draw or a loss,
or with `solve_exact` the margin). Start `i`'s games are numbers 2i, a black, and 2i + 1.

Games are played one way or the other:

- one at a time (`play_sequential`), each side charged its own moves, on its own clock or at its
  fixed budget, by fresh players (a network's shared), each move timed; with `synchro`, a
  start's two games at once on the same cores, as GGS plays them
- in batches on the GPU (`play_batched`), when a network plays on `gpu-batched`: many games
  moving in step, as the Elo league plays them (`core.league.match`), against engines at a level
  (a process a game) or another such network. Moves aren't timed alone: a batch's timing is the
  match's. A game to be solved is played out and cut where it would have stopped, which at fixed
  budgets is the same game
"""

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from typing import Any

import numpy as np
from clock_match import (
    UNTIMED,
    Game,
    GameRun,
    endgame_solve,
    play_synchro,
)
from engines import Board
from eval_players import CACHE_DIR, PlayerSpec, maker
from ggs import Clock, Start, disc_result, load_starts, random_start
from xot import XOT_PATH, load_xot

from core.evals import hardware
from core.evals.store import GameRecord, digest

PASS = 64


@dataclass(frozen=True)
class Opening:
    """Where a pair of games starts: `label` for the record, the opening's `moves` from `start`
    (the standard start if None)."""

    label: str
    moves: tuple[int, ...]
    start: Start | None


@dataclass(frozen=True)
class MatchSpec:
    """A match: its players and its games' setup (see the module's docstring). `tag` sets apart
    matches otherwise the same, to play them again rather than continue them."""

    a: PlayerSpec
    b: PlayerSpec
    openings: int
    starts: str = "xot"
    seed: int = 0
    solve_at: int | None = None
    solve_exact: bool = False
    synchro: bool = False
    match_type: str = "s8r14"
    since: str = ""
    tag: str = ""

    def __post_init__(self) -> None:
        if not self.batched and "pgx" in (self.a.config["kind"], self.b.config["kind"]):
            raise ValueError("pgx plays only in batched games")
        if self.batched:
            for p in (self.a, self.b):
                if p.config["kind"] == "net" and p.device != "gpu-batched":
                    raise ValueError(
                        f"{p.label}: a gpu-batched network's opponent plays in its batches, "
                        "so a network must be gpu-batched too"
                    )
                if p.clock is not None:
                    raise ValueError(f"{p.label}: batched games are at fixed budgets")
                if p.config["kind"] == "engine" and p.config["threads"] != 1:
                    raise ValueError(f"{p.label}: batched games run an engine a thread")
            if self.starts != "xot":
                raise ValueError("batched games start from XOT openings")
            if self.synchro:
                raise ValueError("batched games can't be synchro")
            dtypes = {
                p.config["search"]["dtype"]
                for p in (self.a, self.b)
                if p.config["kind"] == "net"
            }
            if len(dtypes) > 1:
                raise ValueError("batched networks compute in one dtype")
        if self.solve_at is not None and not 0 < self.solve_at < 60:
            raise ValueError(f"solve_at={self.solve_at}: empties, 1 to 59")

    @property
    def batched(self) -> bool:
        return any(p.device == "gpu-batched" for p in (self.a, self.b))

    @property
    def uses_gpu(self) -> bool:
        return any(p.device in ("gpu", "gpu-batched") for p in (self.a, self.b))

    @property
    def games(self) -> int:
        return 2 * self.openings

    def config(self) -> dict[str, Any]:
        """The match's setup, its players by id: what it is, apart from where it's played."""
        return {
            "a": self.a.id,
            "b": self.b.id,
            "openings": self.openings,
            "starts": self.starts,
            "seed": self.seed,
            "solve_at": self.solve_at,
            "solve_exact": self.solve_exact if self.solve_at is not None else None,
            "synchro": self.synchro,
            "match_type": self.match_type if self.starts.startswith("file:") else None,
            "since": self.since if self.starts.startswith("file:") else None,
            "tag": self.tag,
        }

    def id(self, placement: str) -> str:
        """The match's id: the same setup played in the same place is the same match, which a
        run again continues."""
        return "m-" + digest({"config": self.config(), "placement": placement}, 12)

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k not in ("a", "b")},
            "a": asdict(self.a),
            "b": asdict(self.b),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchSpec":
        data = dict(data)
        a, b = (
            PlayerSpec(**{**p, "files": tuple(p["files"])})
            for p in (data.pop("a"), data.pop("b"))
        )
        return cls(a, b, **data)

    def choose_openings(self) -> list[Opening]:
        """The match's starts, drawn by its seed: the same every time."""
        rng = np.random.default_rng(self.seed)
        kind, _, value = self.starts.partition(":")
        if kind == "xot":
            pool = load_xot(XOT_PATH)
            picked = rng.choice(len(pool), size=self.openings, replace=False)
            return [
                Opening(f"xot:{i}", tuple(int(m) for m in pool[i]), None)
                for i in picked
            ]
        if kind == "random":
            starts = [
                random_start(int(value), rng).playable() for _ in range(self.openings)
            ]
        elif kind == "file":
            found = load_starts(value, self.match_type, self.since)
            if len(found) < self.openings:
                raise ValueError(
                    f"{value} has {len(found)} starts, not {self.openings}"
                )
            picked = rng.choice(len(found), size=self.openings, replace=False)
            starts = [found[i].playable() for i in picked]
        else:
            raise ValueError(f"unknown starts {self.starts!r}")
        return [Opening(f"board:{s.text()}", (), s) for s in starts]


def numbers_of(openings: int, slice_: tuple[int, int] = (0, 1)) -> list[int]:
    """The games a job plays: start `i`'s two where `i` is `k` of every `n`, for `slice_` (k, n)."""
    k, n = slice_
    return [2 * i + side for i in range(k, openings, n) for side in (0, 1)]


def record_of(
    match_id: str,
    number: int,
    spec: MatchSpec,
    opening: Opening,
    game: Game,
    host: dict[str, Any],
    timed: bool = True,
) -> GameRecord:
    """A finished game, number `number`, as the store keeps it."""
    a_black = number % 2 == 0
    if game.solved_at is not None:
        ending = "solved"
    elif game.hard is not None:
        ending = "timeout"
    else:
        ending = "end"
    return GameRecord(
        match=match_id,
        number=number,
        black=(spec.a if a_black else spec.b).id,
        white=(spec.b if a_black else spec.a).id,
        start=opening.label,
        moves=[int(m) for m in game.moves],
        opening_moves=len(opening.moves),
        move_seconds=[round(s, 6) if timed else None for _, _, s in game.move_seconds],
        ending=ending,
        margin=int(game.margin),
        result=int(game.result),
        solved_at=game.solved_at,
        solve_exact=spec.solve_exact if game.solved_at is not None else None,
        flagged=game.flagged,
        hard=game.hard,
        black_seconds=round(game.seconds[0], 6) if timed else None,
        white_seconds=round(game.seconds[1], 6) if timed else None,
        host=host,
        load=round(hardware.load(), 2),
        finished=time.time(),
    )


Emit = Callable[[GameRecord], None]


def play_sequential(
    spec: MatchSpec,
    match_id: str,
    numbers: Sequence[int],
    emit: Emit,
    solve: Callable[[int, int], int] | None = None,
    make: Callable[[PlayerSpec, Clock], Callable[[], Any]] | None = None,
) -> None:
    """Plays the games `numbers` one at a time (with `synchro`, a start's two at once), each
    by fresh players from `make` (`eval_players.maker` of their runtime), emitting each as it
    ends."""
    openings = spec.choose_openings()
    host = hardware.describe(with_gpu=spec.uses_gpu)
    if spec.solve_at is not None and solve is None:
        solve = endgame_solve(spec.solve_exact)
    build = make or (lambda p, clock: maker(p.runtime, clock))
    clocks = {p.id: p.clock or UNTIMED for p in (spec.a, spec.b)}
    a, b = (build(p, clocks[p.id]) for p in (spec.a, spec.b))
    wanted = set(numbers)
    second = None
    if spec.synchro:
        # the pair's second game needs its own players: a network can't search two trees at once
        second = (build(spec.a, clocks[spec.a.id]), build(spec.b, clocks[spec.b.id]))

    def run(make_a, make_b, opening: Opening, a_black: bool, made: list) -> GameRun:
        first, other = make_a(), make_b()
        made += [first, other]
        black, white = (first, other) if a_black else (other, first)
        black_spec, white_spec = (spec.a, spec.b) if a_black else (spec.b, spec.a)
        return GameRun(
            black,
            white,
            opening.moves,
            (clocks[black_spec.id], clocks[white_spec.id]),
            opening.start,
            spec.solve_at,
            solve,
        )

    for i, opening in enumerate(openings):
        pair = [n for n in (2 * i, 2 * i + 1) if n in wanted]
        if not pair:
            continue
        made: list = []
        try:
            if second is not None and len(pair) == 2:
                runs = [
                    run(a, b, opening, True, made),
                    run(second[0], second[1], opening, False, made),
                ]
                for number, game in zip(pair, play_synchro(runs)):
                    emit(record_of(match_id, number, spec, opening, game, host))
                continue
            for number in pair:
                game_run = run(a, b, opening, number % 2 == 0, made)
                while not game_run.over:
                    game_run.step()
                emit(
                    record_of(match_id, number, spec, opening, game_run.finish(), host)
                )
                for player in made:
                    player.close()
                made = []
        finally:
            for player in made:
                player.close()


def play_parallel(
    spec: MatchSpec,
    match_id: str,
    numbers: Sequence[int],
    emit: Emit,
    parallel: int,
    make: Callable[[PlayerSpec, Clock], Callable[[], Any]] | None = None,
    solver: Callable[[], Callable[[int, int], int]] | None = None,
) -> None:
    """Plays the games `numbers` `parallel` at a time, each in a thread by fresh players, on
    the same cores: for players a game's thread can have to itself, engines (a process a game)
    and networks on the GPU (whose games share one stream, its calls the bigger for it). At fixed
    budgets the moves are the same as one at a time; on a clock, games slow each other."""
    if any(p.device == "cpu" for p in (spec.a, spec.b)):
        raise ValueError(
            "networks on the CPU play one game at a time (--jobs: processes)"
        )
    if spec.synchro:
        raise ValueError("synchro pairs are already two games at once")
    openings = spec.choose_openings()
    host = hardware.describe(with_gpu=spec.uses_gpu)
    build = make or (lambda p, clock: maker(p.runtime, clock))
    clocks = {p.id: p.clock or UNTIMED for p in (spec.a, spec.b)}
    a, b = (build(p, clocks[p.id]) for p in (spec.a, spec.b))
    new_solver = solver or (lambda: endgame_solve(spec.solve_exact))

    queue = list(reversed(numbers))
    lock = threading.Lock()
    errors: list[BaseException] = []
    done: Queue[GameRecord] = Queue()

    def worker() -> None:
        # a solver to a thread: one solves a position at a time
        solve = new_solver() if spec.solve_at is not None else None
        while not errors:
            with lock:
                if not queue:
                    return
                number = queue.pop()
            opening = openings[number // 2]
            a_black = number % 2 == 0
            first, other = a(), b()
            try:
                black, white = (first, other) if a_black else (other, first)
                black_spec, white_spec = (
                    (spec.a, spec.b) if a_black else (spec.b, spec.a)
                )
                run = GameRun(
                    black,
                    white,
                    opening.moves,
                    (clocks[black_spec.id], clocks[white_spec.id]),
                    opening.start,
                    spec.solve_at,
                    solve,
                )
                while not run.over:
                    run.step()
                done.put(record_of(match_id, number, spec, opening, run.finish(), host))
            except BaseException as error:  # noqa: BLE001 - raised in the caller's thread
                errors.append(error)
            finally:
                first.close()
                other.close()

    threads = [threading.Thread(target=worker) for _ in range(parallel)]
    for thread in threads:
        thread.start()
    # games are emitted here, in the caller's thread: what keeps them (a SQLite store) may be
    # usable only in the thread that made it
    while any(thread.is_alive() for thread in threads) or not done.empty():
        try:
            emit(done.get(timeout=0.1))
        except Empty:
            pass
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def replay(
    moves: Sequence[int],
    opening_moves: int,
    solve_at: int | None,
    solve: Callable[[int, int], int] | None,
) -> Game:
    """A game from its moves (passes left out), stopped and solved at `solve_at` empties as
    `GameRun` would have."""
    board = Board()
    game = Game()
    for ply, move in enumerate(moves):
        empties = 64 - (board.discs[0] | board.discs[1]).bit_count()
        if solve_at is not None and ply >= opening_moves and empties <= solve_at:
            assert solve is not None
            side = not board.black_to_move
            margin = solve(board.discs[side], board.discs[not side])
            game.margin = game.result = margin if board.black_to_move else -margin
            game.solved_at = empties
            return game
        board.play(move)
        game.moves.append(move)
        if ply >= opening_moves:
            game.move_seconds.append((empties, True, 0.0))
    game.margin = board.margin()
    game.result = disc_result(board)
    return game


# a batched player's `net`, for pgx's pretrained model of that name (see `batched_game`)
PGX = "pgx:"


def batched_game() -> Any:
    """The league's Othello (`elo_league.Othello`), which also plays pgx's pretrained model, a
    `net` of `PGX` and its name, as our networks play: no root noise, temperature 0."""
    from elo_league import Othello
    from game import make_test_evaluator, state_to_nn_input

    from core.evaluators.evaluation_fns import make_nn_eval_fn_no_params_callable

    class Evaluations(Othello):
        def load_network(self, path: str, net: str, dtype: str) -> Any:
            if net.startswith(PGX):
                # the model is its own parameters: none to pass the search
                return net, None
            return super().load_network(path, net, dtype)

        def make_evaluator(self, network: Any, sims: int, tree_factor: int = 4) -> Any:
            if isinstance(network, str) and network.startswith(PGX):
                import pgx

                model = pgx.make_baseline_model(network[len(PGX) :])  # pyright: ignore[reportArgumentType]
                return make_test_evaluator(
                    make_nn_eval_fn_no_params_callable(model, state_to_nn_input),
                    sims,
                    noise=False,
                    tree_factor=tree_factor,
                )
            return super().make_evaluator(network, sims, tree_factor)

    return Evaluations()


def play_batched(
    spec: MatchSpec,
    match_id: str,
    numbers: Sequence[int],
    emit: Emit,
    record_run: Callable[[dict[str, Any], dict[str, Any]], None] = lambda _i, _h: None,
    games_at_once: int = 64,
    solve: Callable[[int, int], int] | None = None,
) -> None:
    """Plays the games `numbers` on the GPU in batches of `games_at_once` starts (each start's
    two games: one batch half with a black, the other b), emitting each batch's games as it
    ends. `record_run` gets each batch's timing and the host."""
    from core.league.match import Programs, play_batch
    from core.league.store import Player

    openings = spec.choose_openings()
    host = hardware.describe(with_gpu=True)
    if spec.solve_at is not None and solve is None:
        solve = endgame_solve(spec.solve_exact)
    game = batched_game()
    dtype = next(
        (
            p.config["search"]["dtype"]
            for p in (spec.a, spec.b)
            if p.config["kind"] == "net"
        ),
        "float32",
    )
    programs = Programs(game, dtype)

    def league_player(index: int, p: PlayerSpec) -> Player:
        budget = p.config["budget"]
        if p.config["kind"] == "pgx":
            return Player(
                id=index,
                name=p.label,
                sims=budget["visits"],
                network="pgx",
                path="pgx",
                net=PGX + p.config["model"],
                settings={"tree_factor": p.config["search"]["tree_factor"]},
            )
        if p.config["kind"] == "net":
            network = p.runtime["network"]
            path = network["checkpoint"]
            return Player(
                id=index,
                name=p.label,
                sims=budget["visits"],
                network=network["name"],
                path=path if path.startswith("/") else str(CACHE_DIR / path),
                net=network["net"],
                settings={"tree_factor": p.config["search"]["tree_factor"]},
            )
        return Player(
            id=index,
            name=p.label,
            external=p.config["engine"],
            settings={"level": budget["level"], "args": p.config["args"]},
        )

    a, b = league_player(0, spec.a), league_player(1, spec.b)
    wanted = set(numbers)
    # starts whose two games are both wanted, in batches; a start wanted only in part is played
    # whole, its other game dropped
    starts = [i for i in range(len(openings)) if {2 * i, 2 * i + 1} & wanted]
    xot_index = {i: int(openings[i].label.split(":")[1]) for i in starts}
    for at in range(0, len(starts), games_at_once):
        chunk = starts[at : at + games_at_once]
        indices = np.array([xot_index[i] for i in chunk])
        result = play_batch(game, programs, a, b, indices, seed=spec.seed + at)
        records = []
        for played in result.games:
            i = chunk[int(np.flatnonzero(indices == played.opening)[0])]
            number = 2 * i + (0 if played.first == a.id else 1)
            if number not in wanted:
                continue
            moves = [int(m) for m in played.moves if int(m) != PASS]
            replayed = replay(moves, len(openings[i].moves), spec.solve_at, solve)
            records.append(
                record_of(
                    match_id, number, spec, openings[i], replayed, host, timed=False
                )
            )
        for record in sorted(records, key=lambda r: r.number):
            emit(record)
        record_run(
            {
                "batch_starts": len(chunk),
                "seconds": round(result.seconds, 3),
                "compiled": result.compiled,
                "a_seconds": round(result.a_clock.seconds, 3),
                "a_moves": result.a_clock.moves,
                "b_seconds": round(result.b_clock.seconds, 3),
                "b_moves": result.b_clock.moves,
                "device": result.settings.device if result.settings else None,
            },
            host,
        )


def play(
    spec: MatchSpec,
    match_id: str,
    numbers: Sequence[int],
    emit: Emit,
    record_run: Callable[[dict[str, Any], dict[str, Any]], None] = lambda _i, _h: None,
    parallel: int = 1,
) -> None:
    """The games `numbers`, batched on the GPU if a network plays there, else one at a time,
    or `parallel` at a time (`play_parallel`)."""
    if not numbers:
        return
    if spec.batched:
        play_batched(spec, match_id, numbers, emit, record_run)
    elif parallel > 1:
        play_parallel(spec, match_id, numbers, emit, parallel)
    else:
        play_sequential(spec, match_id, numbers, emit)
