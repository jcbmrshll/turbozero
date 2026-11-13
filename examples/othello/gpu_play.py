"""Play on the GPU, for GGS (where hardware is free; engine comparisons stay on CPU vector units): a
game's tree on the host, `cpu_search.cpp`'s, its leaves evaluated on the GPU many at a time
(`gpu_net.py`).

One `Stream` owns the GPU. Every search under way, a game's move, the other game of a synchro
pair, or a ponder, is a `Job` in it. The stream fills each call of the network with a batch of
each job's leaves, keeps `calls` calls in flight so the host collects the next while the GPU
computes, and backs each up when it comes back. A job has `inflight` batches of its own: its
traversals in flight are about `inflight * batch`, steered apart by virtual loss as in the CPU
engine.

    # a player for clock_match.py
    net:net=NAME,device=gpu,batch=256,inflight=2,solve=20

    # one game's speed (or a synchro pair's, --games 2), against the CPU engine's
    uv run --extra cuda examples/othello/gpu_play.py speed --net NAME --batch 64 256 --visits 20000

    # strength at equal visits, two settings on the GPU, many games at once
    uv run --extra cuda examples/othello/gpu_play.py match --net NAME \\
        --a visits=4000,batch=8,inflight=2 --b visits=4000,batch=128,inflight=2 --openings 200

On GGS, `make_ggs_bridge` is the client's player (`ggs_client/ggs_player.py`'s `Bridge`): a tree a
game, and a ponder of each game where the opponent is to move.
"""

import argparse
import itertools
import json
import os
import resource
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from cpu_match import ENDED_VISITS, Solving, move_budget
from cpu_search import PASS, Batch, Search, SearchSettings

# the most leaves one call of the network evaluates, by default
MAX_CALL = 1024


class Network(Protocol):
    """What the stream evaluates with: `gpu_net.GpuNet`, or a fake."""

    sizes: tuple[int, ...]

    def launch(self, bits: np.ndarray, count: int | None = None) -> Any: ...

    def wait(self, pending: Any) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(eq=False)
class Job:
    """A search under way in a `Stream`: `visits` visits of `search`'s tree, or with none, until
    `deadline` (`time.perf_counter()`'s) or until cancelled; either way stopping once
    `stop_ended` traversals in a row end the game (0 for never). Each call carries up to `batch`
    of its leaves, with up to `inflight` calls of them out at once. Higher `priority` fills calls
    first."""

    search: Search
    batch: int
    inflight: int = 2
    visits: int = 0
    deadline: float | None = None
    stop_ended: int = 0
    priority: int = 0
    # filled in by the stream
    leaves: int = 0
    calls: int = 0
    seconds: float = 0.0
    done: threading.Event = field(default_factory=threading.Event)
    stopping: bool = False
    cancelled: bool = False
    free: list[Batch] = field(default_factory=list)
    started: float = 0.0

    def should_stop(self, now: float) -> bool:
        search = self.search
        return (
            self.cancelled
            or search.remaining <= 0
            or (self.deadline is not None and now >= self.deadline)
            or (self.stop_ended > 0 and search.ended_in_a_row() >= self.stop_ended)
        )

    def cancel(self) -> None:
        """Stops collecting; `done` is set once its leaves in flight are backed up."""
        self.cancelled = True


class Stream:
    """The network's calls, shared by every job (see the module's doc). Its thread is the only
    one touching a job's tree while the job runs."""

    def __init__(self, network: Network, max_call: int = MAX_CALL, calls: int = 2):
        self.network = network
        self.max_call = min(max_call, max(network.sizes))
        self.calls = calls
        self.jobs: list[Job] = []
        self.lock = threading.Condition()
        self.closed = False
        # what calls carried, for reports: calls, leaves
        self.total_calls = 0
        self.total_leaves = 0
        # the thread's time waiting for the network: near all of it when the GPU is the limit
        self.wait_seconds = 0.0
        # and its time in the tree, collecting and backing up, and launching calls
        self.collect_seconds = 0.0
        self.apply_seconds = 0.0
        self.launch_seconds = 0.0
        # where the next call starts in the jobs of a priority, for fairness among many
        self.turn = 0
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="gpu-stream")
        self.thread.start()

    def submit(self, job: Job) -> Job:
        """Starts `job` (its search begun by the caller: `begin`)."""
        job.free = [Batch() for _ in range(job.inflight)]
        job.started = time.perf_counter()
        with self.lock:
            if self.error is not None:
                raise RuntimeError("the GPU stream failed") from self.error
            self.jobs.append(job)
            self.lock.notify_all()
        return job

    def run(self, job: Job) -> Job:
        """`submit`, then waits for it."""
        self.submit(job).done.wait()
        if self.error is not None:
            raise RuntimeError("the GPU stream failed") from self.error
        return job

    def close(self) -> None:
        """Cancels the jobs under way, and ends the thread once their leaves are back."""
        with self.lock:
            self.closed = True
            for job in self.jobs:
                job.cancel()
            self.lock.notify_all()
        self.thread.join()

    def _run(self) -> None:
        try:
            self._loop()
        except BaseException as error:  # noqa: BLE001 - raised in the jobs' threads
            self.error = error
            with self.lock:
                for job in self.jobs:
                    job.done.set()

    def _fill(self, jobs: list[Job], now: float) -> tuple[np.ndarray, list, int]:
        """A call's leaves, a batch of each job with one free, the highest priority first:
        packed in a buffer of its own (the GPU may read it after the call is launched), and
        how many."""
        buffer = np.empty((self.max_call, 2), np.uint64)
        parts = []
        offset = 0
        self.turn += 1
        order = sorted(
            range(len(jobs)),
            key=lambda i: (-jobs[i].priority, (i - self.turn) % len(jobs)),
        )
        for i in order:
            job = jobs[i]
            if job.stopping or not job.free:
                continue
            if job.should_stop(now):
                job.stopping = True
                continue
            room = min(job.batch, self.max_call - offset)
            if room <= 0:
                break
            batch = job.free.pop()
            count = job.search.collect_bits(batch, room, buffer[offset:])
            if count == 0:
                # a game's ends, collisions: nothing to evaluate, but collisions to undo
                job.search.apply_from(batch, None, None)
                job.free.append(batch)
                continue
            parts.append((job, batch, offset, count))
            offset += count
        return buffer, parts, offset

    def _loop(self) -> None:
        in_flight: deque = deque()
        while True:
            with self.lock:
                while not self.jobs and not in_flight and not self.closed:
                    self.lock.wait()
                if self.closed and not self.jobs and not in_flight:
                    return
                jobs = list(self.jobs)
            now = time.perf_counter()
            launched = False
            if len(in_flight) < self.calls:
                bits, parts, count = self._fill(jobs, now)
                began = time.perf_counter()
                self.collect_seconds += began - now
                if parts:
                    in_flight.append((self.network.launch(bits, count), parts))
                    launched = True
                    self.launch_seconds += time.perf_counter() - began
            if in_flight and (not launched or len(in_flight) >= self.calls):
                pending, parts = in_flight.popleft()
                began = time.perf_counter()
                logits, values = self.network.wait(pending)
                applying = time.perf_counter()
                self.wait_seconds += applying - began
                self.total_calls += 1
                self.total_leaves += len(values)
                for job, batch, offset, count in parts:
                    job.search.apply_from(
                        batch,
                        logits[offset : offset + count],
                        values[offset : offset + count],
                    )
                    job.free.append(batch)
                    job.leaves += count
                    job.calls += 1
                self.apply_seconds += time.perf_counter() - applying
            # jobs whose leaves are all back and which collect no more
            now = time.perf_counter()
            finished = []
            for job in jobs:
                if not job.stopping and job.should_stop(now):
                    job.stopping = True
                if job.stopping and len(job.free) == job.inflight:
                    finished.append(job)
            if finished:
                with self.lock:
                    for job in finished:
                        self.jobs.remove(job)
                for job in finished:
                    job.search.begin(0)
                    job.seconds = now - job.started
                    job.done.set()


@dataclass(frozen=True)
class GpuSettings:
    """How a GPU player searches: `batch` leaves a call, `inflight` calls of them out at once,
    and the tree's settings (`cpu_search.SearchSettings`; its own `batch` is unused)."""

    batch: int = 256
    inflight: int = 2
    search: SearchSettings = field(default_factory=SearchSettings)

    @classmethod
    def parse(cls, options: dict[str, str]) -> "GpuSettings":
        """From `net:`'s options, which it removes: batch, inflight, virtual_loss, weight,
        c_init, max_collisions, random_symmetry, reuse."""
        batch = int(options.pop("batch", cls.batch))
        collisions = options.pop("max_collisions", None)
        search = SearchSettings(
            batch=batch,
            virtual_loss=options.pop("virtual_loss", "loss"),
            weight=float(options.pop("weight", 1)),
            c_init=float(options.pop("c_init", 1.25)),
            max_collisions=int(collisions) if collisions is not None else None,
            random_symmetry=bool(int(options.pop("random_symmetry", 0))),
            reuse=bool(int(options.pop("reuse", 1))),
        )
        return cls(batch, int(options.pop("inflight", cls.inflight)), search)


class GpuPlayer:
    """Our side of a game (`clock_match.Player`, and shaped as `cpu_match.NetPlayer` for
    `ggs_player.PlayerBridge`): a tree searched through `stream`; with `solving`, the endgame
    solved on the CPU instead."""

    def __init__(
        self,
        stream: Stream,
        settings: GpuSettings,
        taper: int = 0,
        seed: int = 0,
        visits: int = 0,
        solving: Solving | None = None,
        ponder: bool = False,
    ):
        """With `ponder`, it ponders after each of its moves until told the opponent's."""
        self.stream = stream
        self.ponders = ponder
        self.settings = settings
        self.search = Search(settings.search, seed)
        self.taper = taper
        self.visits = visits
        self.to_move = "b"
        self.solving = solving
        self.solver = None
        if solving is not None:
            from endgame import Solver

            self.solver = Solver(solving.threads)
        self.solved = 0
        self.gave_up = 0
        # as `NetPlayer.last_value`: ("q", root value), ("margin", discs), ("wdl", sign) or None
        self.last_value: tuple[str, float] | None = None
        # the last search's job, for its counts
        self.last_job: Job | None = None
        # a ponder under way (see `ponder`)
        self.pondering: Job | None = None

    def sync(self, color: str) -> None:
        """Passes for the other color if `color` moves out of turn (see `NetPlayer.sync`)."""
        if color != self.to_move:
            self.search.play(PASS)
        self.to_move = "w" if color == "b" else "b"

    def play(self, color: str, move: int) -> None:
        self.stop_pondering()
        self.sync(color)
        self.search.play(move)

    def set_position(self, black: int, white: int, color: str) -> None:
        self.stop_pondering()
        me, opp = (black, white) if color == "b" else (white, black)
        self.search.set_position(me, opp)
        self.to_move = color

    def solve(self, me: int, opp: int, seconds_left: float) -> int | None:
        """As `NetPlayer.solve`."""
        solving = self.solving
        assert solving is not None and self.solver is not None
        seconds = 0.0 if self.visits else max(solving.share * seconds_left, 1e-3)
        solved = self.solver.solve(me, opp, seconds, solving.exact)
        if solved is None:
            self.gave_up += 1
            return None
        self.last_value = ("margin" if solving.exact else "wdl", float(solved.margin))
        if not solving.exact and solved.margin < 0:
            return None
        self.solved += 1
        return solved.move

    def searched(self, seconds: float, priority: int = 1) -> Job:
        """A search of the position to move from, `visits` visits, or `seconds`."""
        self.stop_pondering()
        visits = self.visits
        self.search.begin(visits if visits else 1 << 30)
        job = Job(
            self.search,
            self.settings.batch,
            self.settings.inflight,
            visits=visits,
            deadline=None if visits else time.perf_counter() + seconds,
            stop_ended=0 if visits else ENDED_VISITS,
            priority=priority,
        )
        self.last_job = self.stream.run(job)
        return job

    def move(self, color: str, seconds_left: float) -> int:
        """Our move, which it also plays (see `NetPlayer.move`)."""
        self.sync(color)
        me, opp, _, _ = self.search.position()
        empties = 64 - (me | opp).bit_count()
        action = None
        self.last_value = None
        if self.search.num_legal() == 1:
            action = self.search.first_legal()
        elif self.solving is not None and empties <= self.solving.empties:
            start = time.perf_counter()
            self.stop_pondering()
            action = self.solve(me, opp, seconds_left)
            seconds_left -= time.perf_counter() - start
        if action is None:
            self.searched(move_budget(max(seconds_left, 0.0), empties, self.taper))
            action = self.search.best_action()
            self.last_value = ("q", self.search.root_q())
        self.search.play(action)
        if self.ponders:
            self.ponder()
        return action

    def ponder(self, batch: int | None = None, priority: int = 0) -> None:
        """Searches the position, the opponent to move, until anything else is asked of the
        player: what it finds under the move they play is kept. In the stream at a lower
        priority than moves, so it takes what room their calls leave."""
        if self.pondering is not None or self.search.terminated():
            return
        if self.solving is not None:
            me, opp, _, _ = self.search.position()
            if 64 - (me | opp).bit_count() <= self.solving.empties + 1:
                return
        self.search.begin(1 << 30)
        self.pondering = self.stream.submit(
            Job(
                self.search,
                batch or self.settings.batch,
                self.settings.inflight,
                stop_ended=ENDED_VISITS,
                priority=priority,
            )
        )

    def stop_pondering(self) -> None:
        job, self.pondering = self.pondering, None
        if job is not None:
            job.cancel()
            job.done.wait()

    def close(self) -> None:
        self.stop_pondering()


def parse_solving(options: dict[str, str], cores: int, exact: bool) -> Solving | None:
    """`solve`, `solve_exact`, `solve_share` of `net:`'s options, which it removes."""
    if "solve" not in options:
        return None
    return Solving(
        empties=int(options.pop("solve")),
        exact=bool(int(options.pop("solve_exact", int(exact)))),
        share=float(options.pop("solve_share", Solving.share)),
        threads=cores,
    )


# the stream of each network in this process, made on first use: players of one network share
# the GPU's calls, as a synchro pair's two games should
_STREAMS: dict[tuple, Stream] = {}
_STREAMS_LOCK = threading.Lock()


def shared_stream(net: str, dtype: str, max_call: int, calls: int) -> Stream:
    from gpu_net import GpuNet

    key = (net, dtype, max_call, calls)
    with _STREAMS_LOCK:
        if key not in _STREAMS:
            sizes = tuple(
                s
                for s in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
                if s <= max_call
            )
            network = GpuNet.load(net, dtype, sizes=sizes)
            network.warm()
            _STREAMS[key] = Stream(network, max_call, calls)
        return _STREAMS[key]


def player_factory(options: dict[str, str], cores: int) -> Callable[[], GpuPlayer]:
    """What builds `clock_match.py`'s `net:...,device=gpu` player for each game: all share the
    network's stream. Options: `net`, `batch` (leaves a call, of one search), `inflight` (its
    calls out at once), `max_call` (the most leaves a call, of all searches), `calls` (calls out
    at once), `dtype` (bfloat16 or float32), `visits`, `taper`, `solve`/`solve_exact`/`solve_share`
    (`cpu_match.Solving`, on `cores` CPU threads), `ponder=1` to search on while the opponent
    thinks, and the tree's (`GpuSettings.parse`)."""
    options = dict(options)
    options.pop("device", None)
    options.pop("engine", None)
    options.pop("backend", None)
    options.pop("threads", None)
    net = options.pop("net")
    dtype = options.pop("dtype", "bfloat16")
    max_call = int(options.pop("max_call", MAX_CALL))
    calls = int(options.pop("calls", 2))
    visits = int(options.pop("visits", 0))
    taper = int(options.pop("taper", 0))
    ponder = bool(int(options.pop("ponder", 0)))
    solving = parse_solving(options, cores, exact=False)
    settings = GpuSettings.parse(options)
    if options:
        raise ValueError(f"unknown GPU player options: {sorted(options)}")
    stream = shared_stream(net, dtype, max_call, calls)
    seeds = itertools.count()
    return lambda: GpuPlayer(
        stream, settings, taper, next(seeds), visits, solving, ponder
    )


# --- GGS ------------------------------------------------------------------------------------

# discs for a root value, as `ggs_player.value_discs` has it: DISC_SCALE * atanh(value)
DISC_SCALE = 6.5


def value_discs(value: float) -> float:
    """A search's root value, -1 to 1, as discs for the player to move (`ggs_player`'s)."""
    import math

    value = min(max(value, -0.999999), 0.999999)
    return max(-64.0, min(64.0, DISC_SCALE * math.atanh(value)))


def last_eval(value: tuple[str, float] | None, fallback: float) -> float:
    if value is None:
        return fallback
    kind, v = value
    if kind == "margin":
        return v
    if kind == "wdl":
        return 2.0 * (v > 0) - 2.0 * (v < 0)
    return value_discs(v)


def replay(
    start: tuple[int, int, str], moves: Sequence[int | None]
) -> list[tuple[str, int]]:
    """Each move since `start` that isn't a pass, with the colour that made it, and the board."""
    from engines import Board

    board = Board.at(*start)
    out = []
    for move in moves:
        if move is None:
            if not board.legal():
                board.black_to_move = not board.black_to_move
            continue
        out.append((board.color, move))
        board.play(move)
    return out


@dataclass(eq=False)
class _GgsGame:
    player: GpuPlayer
    start: tuple[int, int, str]
    moves: list = field(default_factory=list)
    told: int = 0
    last_eval: float = 0.0
    desyncs: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class GgsBridge:
    """`ggs_player.Bridge` on the GPU: a `GpuPlayer` a game, all sharing one stream; each game
    where the opponent is to move pondered meanwhile (in a synchro match, while we decide the
    other game), its tree kept for our turn. `decision` builds the client's `Decision`;
    `time_policy`, its `TimePolicy`, plans the clock."""

    def __init__(
        self,
        new_player: Callable[[], GpuPlayer],
        decision: Callable[..., Any],
        time_policy: Any,
        ponder: bool = True,
        ponder_batch: int | None = None,
    ):
        self.new_player = new_player
        self.decision = decision
        self.time = time_policy
        self.ponder = ponder
        self.ponder_batch = ponder_batch
        self.games: dict[str, _GgsGame] = {}
        self.lock = threading.Lock()

    def _get(self, view: Any) -> _GgsGame:
        with self.lock:
            game = self.games.get(view.game_id)
            if game is None:
                game = _GgsGame(self.new_player(), view.start)
                game.player.set_position(*view.start)
                self.games[view.game_id] = game
            return game

    def _sync(self, game: _GgsGame, view: Any) -> None:
        """The game's player told every move so far, under its lock; started afresh from the
        server's board if what it was told before isn't how the game went (see
        `PlayerBridge._game`)."""
        from engines import Board

        moves = list(view.moves)
        if game.start != view.start or moves[: len(game.moves)] != game.moves:
            game.player.set_position(*view.start)
            game.start, game.moves, game.told = view.start, [], 0
        colored = replay(game.start, moves)
        for color, move in colored[game.told :]:
            game.player.play(color, move)
            game.told += 1
        game.moves = moves
        board = Board.at(*game.start)
        for _, move in colored:
            board.play(move)
        if board.discs != [view.black, view.white]:
            game.desyncs += 1
            game.player.set_position(view.black, view.white, view.color)
            game.start = (view.black, view.white, view.color)
            game.moves, game.told = [], 0

    def observe(self, view: Any) -> None:
        """Ponders a game where the opponent is to move. Returns at once."""
        mine = getattr(view, "mine", None)
        if (
            not self.ponder
            or mine is None
            or view.color == mine
            or getattr(view, "komi", False)
        ):
            return
        game = self._get(view)
        # a decide under way has the game: no ponder then
        if game.lock.acquire(blocking=False):
            try:
                self._sync(game, view)
                game.player.ponder(self.ponder_batch)
            finally:
                game.lock.release()

    def decide(self, view: Any, cores: int) -> Any:
        from engines import Board

        start = time.perf_counter()
        if not Board.at(view.black, view.white, view.color).legal():
            return self.decision(None, 0.0, time.perf_counter() - start, {"pass": True})
        game = self._get(view)
        with game.lock:
            self._sync(game, view)
            player = game.player
            if view.extended:
                player.taper = 0
                per_move = self.time.extension_move(view)
                move = player.move(view.color, per_move * (view.empties + 1) / 2 / 0.95)
            elif player.solving is not None and hasattr(self.time, "planned_left"):
                # the clock spread over the moves searched, the solver's held back
                left = self.time.planned_left(view, player.solving.empties)
                move = player.move(view.color, left)
            else:
                move = player.move(view.color, self.time.seconds_left(view))
            game.told += 1
            game.moves = list(view.moves) + [move]
        game.last_eval = last_eval(player.last_value, game.last_eval)
        job = player.last_job
        info = {
            "value": player.last_value,
            "desyncs": game.desyncs,
            "leaves": job.leaves if job else 0,
            "seconds_searched": round(job.seconds, 3) if job else 0.0,
        }
        player.last_job = None
        return self.decision(move, game.last_eval, time.perf_counter() - start, info)

    def forget(self, game_id: str) -> None:
        with self.lock:
            game = self.games.pop(game_id, None)
        if game is not None:
            game.player.close()

    def close(self) -> None:
        with self.lock:
            games, self.games = list(self.games.values()), {}
        for game in games:
            game.player.close()


def make_ggs_bridge(
    options: dict[str, str],
    cores: int,
    *,
    decision: Callable[..., Any],
    time_policy: Any | None = None,
) -> GgsBridge:
    """The GGS client's player on the GPU (`ggs_player.make_bridge` with `device=gpu`). Options
    as `player_factory`'s, `solve_exact` on by default (GGS scores by discs), and `ponder=0` to
    not ponder, `ponder_batch` its leaves a call."""
    options = dict(options)
    ponder = bool(int(options.pop("ponder", 1)))
    ponder_batch = int(options["ponder_batch"]) if "ponder_batch" in options else None
    options.pop("ponder_batch", None)
    options.setdefault("solve_exact", "1")
    if time_policy is None:
        raise ValueError("pass the client's TimePolicy")
    return GgsBridge(
        player_factory(options, cores), decision, time_policy, ponder, ponder_batch
    )


# --- measurement ----------------------------------------------------------------------------


def speed_game(
    opening: np.ndarray,
    settings: SearchSettings,
    seed: int,
    solve: int,
    out: list,
    run_move: Callable[[Search], int],
) -> None:
    """A game against itself from `opening` to `solve` empties, each move's search by
    `run_move` (returning its leaves); appends (visits, leaves, collisions, reused) to `out`."""
    search = Search(settings, seed)
    for action in opening:
        search.play(int(action))
    visits = leaves = 0
    while not search.terminated():
        me, opp, _, _ = search.position()
        if 64 - (me | opp).bit_count() <= solve:
            # the endgame solver's from here, on the CPU
            break
        if search.num_legal() == 1:
            search.play(search.first_legal())
            continue
        before = search.total_visits()
        leaves += run_move(search)
        visits += search.total_visits() - before
        search.play(search.best_action())
    out.append((visits, leaves, search.collisions(), search.reused()))


def speed(args: argparse.Namespace) -> None:
    """Games against itself, one at a time (or `--games` at once, a synchro pair's two), a tree
    each searched `--visits` a move, to `--solve` empties (the solver's from there): visits and
    evaluations a second end to end. On the GPU through one stream;
    with `--device cpu`, by the native int8 engine, each game's a worker to each of its share of
    the cores (as `clock_match.py`'s `net:...,engine=1`)."""
    from cpu_search import SearchSettings as Settings
    from xot import XOT_PATH, load_xot

    if args.device == "cpu":
        # nothing of it on the GPU, so as not to take its memory from the league's player
        os.environ["JAX_PLATFORMS"] = "cpu"
    pool = load_xot(XOT_PATH)
    rng = np.random.default_rng(args.seed)
    openings = pool[rng.choice(len(pool), size=args.openings, replace=False)]
    network = None
    if args.device == "gpu":
        from gpu_net import GpuNet

        network = GpuNet.load(args.net, args.dtype, sizes=tuple(args.sizes))
        network.warm()
    for batch, inflight in itertools.product(args.batch, args.inflight):
        stream = None
        engines: list = []
        if network is not None:
            stream = Stream(network, args.max_call, args.calls)
        else:
            from cpu_match import make_engine

            cores = len(os.sched_getaffinity(0)) // args.games
            engines = [
                make_engine(args.net, "native", cores, batch) for _ in range(args.games)
            ]

        def run_move(
            search: Search,
            slot: int,
            batch=batch,
            inflight=inflight,
            stream=stream,
            engines=engines,
        ) -> int:
            """A move's search, returning its leaves."""
            if stream is not None:
                search.begin(args.visits)
                return stream.run(
                    Job(search, batch, inflight, visits=args.visits)
                ).leaves
            return engines[slot].search(search, visits=args.visits)

        begin = time.perf_counter()
        results: list = []
        for i in range(0, len(openings), args.games):
            threads = [
                threading.Thread(
                    target=speed_game,
                    args=(o, Settings(batch=batch), i + k, args.solve, results),
                    kwargs={"run_move": lambda search, k=k: run_move(search, k)},
                )
                for k, o in enumerate(openings[i : i + args.games])
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        seconds = time.perf_counter() - begin
        visits = sum(r[0] for r in results)
        leaves = sum(r[1] for r in results)
        record = {
            "net": args.net,
            "device": args.device,
            "dtype": args.dtype if stream else "int8",
            "cores": len(os.sched_getaffinity(0)),
            "games_at_once": args.games,
            "batch": batch,
            "inflight": inflight
            if stream
            else len(os.sched_getaffinity(0)) // args.games,
            "visits_per_move": args.visits,
            "visits_per_s": round(visits / seconds),
            "evals_per_s": round(leaves / seconds),
            "collisions_per_visit": round(sum(r[2] for r in results) / visits, 4),
            "reused_per_visit": round(sum(r[3] for r in results) / visits, 4),
            "seconds": round(seconds, 1),
            "load": os.getloadavg()[0],
            # the process's peak, trees included
            "max_rss_gb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2
            ),
        }
        if stream is not None:
            stream.close()
            record["calls"] = args.calls
            record["mean_call"] = round(
                stream.total_leaves / max(stream.total_calls, 1), 1
            )
            record["gpu_wait_share"] = round(stream.wait_seconds / seconds, 3)
            record["collect_share"] = round(stream.collect_seconds / seconds, 3)
            record["apply_share"] = round(stream.apply_seconds / seconds, 3)
            record["launch_share"] = round(stream.launch_seconds / seconds, 3)
        print(json.dumps(record), flush=True)
        if args.results:
            with open(args.results, "a") as f:
                f.write(json.dumps(record) + "\n")


def match(args: argparse.Namespace) -> None:
    """Two GPU settings at fixed visits a move, `--parallel` games at once through one stream
    (each game's moves its own jobs), from XOT openings with both colours."""
    from clock_match import play_game, summarize
    from gpu_net import GpuNet
    from xot import XOT_PATH, load_xot

    pool = load_xot(XOT_PATH)
    rng = np.random.default_rng(args.seed)
    openings = pool[rng.choice(len(pool), size=args.openings, replace=False)]
    network = GpuNet.load(args.net, args.dtype, sizes=tuple(args.sizes))
    network.warm()
    stream = Stream(network, args.max_call, args.calls)
    sides = []
    for spec in (args.a, args.b):
        options = dict(item.split("=", 1) for item in spec.split(","))
        visits = int(options.pop("visits"))
        settings = GpuSettings.parse(options)
        if options:
            raise ValueError(f"unknown options {sorted(options)}")
        sides.append((settings, visits))
    queue = [(o, black) for o in range(len(openings)) for black in (True, False)]
    games: list = [None] * len(queue)
    lock = threading.Lock()
    begin = time.perf_counter()

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                index = len(queue) - 1
                o, a_black = queue.pop()
            seed = (args.seed << 20) + 2 * o + (not a_black)
            a = GpuPlayer(stream, sides[0][0], seed=2 * seed, visits=sides[0][1])
            b = GpuPlayer(stream, sides[1][0], seed=2 * seed + 1, visits=sides[1][1])
            black, white = (a, b) if a_black else (b, a)
            games[index] = (play_game(black, white, openings[o], 1e9), a_black)
            with lock:
                done = sum(g is not None for g in games)
            if done % 50 == 0:
                print(
                    f"  {done} games, {time.perf_counter() - begin:.0f} s", flush=True
                )

    threads = [threading.Thread(target=worker) for _ in range(args.parallel)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stream.close()
    played = [g for g in games if g is not None]
    summary = summarize([g for g, _ in played], [b for _, b in played], 1e9)
    record = {
        "net": args.net,
        "a": args.a,
        "b": args.b,
        "seed": args.seed,
        "seconds": round(time.perf_counter() - begin),
        "score": summary.get("score"),
        "elo": summary.get("elo"),
        "games": len(played),
        "summary": summary,
    }
    print(json.dumps(record), flush=True)
    if args.results:
        with open(args.results, "a") as f:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(required=True)
    for name, fn in (("speed", speed), ("match", match)):
        p = commands.add_parser(name)
        p.add_argument("--net", required=True)
        p.add_argument("--dtype", default="bfloat16")
        p.add_argument(
            "--sizes",
            type=int,
            nargs="+",
            default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        )
        p.add_argument("--max-call", type=int, default=MAX_CALL)
        p.add_argument("--calls", type=int, default=2)
        p.add_argument("--openings", type=int, default=2 if name == "speed" else 100)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--results", default=None)
        p.set_defaults(fn=fn)
        if name == "speed":
            p.add_argument("--batch", type=int, nargs="+", default=[64, 256])
            p.add_argument("--inflight", type=int, nargs="+", default=[2])
            p.add_argument("--visits", type=int, default=20000)
            p.add_argument("--games", type=int, default=1)
            p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
            p.add_argument(
                "--solve",
                type=int,
                default=20,
                help="games stop at this many empties, where play solves instead",
            )
        else:
            p.add_argument(
                "--a", required=True, help="e.g. visits=4000,batch=8,inflight=2"
            )
            p.add_argument("--b", required=True)
            p.add_argument("--parallel", type=int, default=64)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
