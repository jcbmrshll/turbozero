"""gpu_play.py's stream, players and GGS bridge, with a fake network in place of the GPU's."""

import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")

START = (1 << 28 | 1 << 35, 1 << 27 | 1 << 36, "b")


def fake_eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed numbers, different for each position (as test_cpu_search's)."""
    weights = np.sin(np.arange(128) * 1.3).astype(np.float32)
    flat = x.reshape(len(x), 128)
    mix = (flat * weights).sum(1, keepdims=True)
    logits = np.sin(mix * np.arange(1, 66, dtype=np.float32)).astype(np.float32)
    values = (flat[:, :64].sum(1) - flat[:, 64:].sum(1)) / 64
    return logits, values.astype(np.float32)


class FakeNetwork:
    """`gpu_play.Network`, evaluating at `wait`, recording each call's size."""

    def __init__(self, sizes=(8, 64, 256), delay: float = 0.0):
        self.sizes = sizes
        self.delay = delay
        self.calls: list[int] = []

    def launch(self, x: np.ndarray):
        self.calls.append(len(x))
        return x.copy()

    def wait(self, pending):
        if self.delay:
            time.sleep(self.delay)
        return fake_eval(pending)


def test_one_leaf_a_call_is_the_sequential_search():
    """batch 1, one call out at a time: the stream's search is `Search.run`'s at batch 1."""
    from cpu_search import Search, SearchSettings
    from gpu_play import Job, Stream

    stream = Stream(FakeNetwork(), calls=1)
    try:
        a, b = Search(SearchSettings()), Search(SearchSettings())
        for _ in range(3):
            a.run(300, fake_eval)
            b.begin(300)
            stream.run(Job(b, batch=1, inflight=1, visits=300))
            assert np.array_equal(a.visits(), b.visits())
            assert a.best_action() == b.best_action()
            a.play(a.best_action())
            b.play(b.best_action())
    finally:
        stream.close()


def test_fixed_visits_with_batches_in_flight():
    from cpu_search import Search, SearchSettings
    from gpu_play import Job, Stream

    network = FakeNetwork()
    stream = Stream(network, max_call=256, calls=2)
    try:
        search = Search(SearchSettings(batch=32))
        search.begin(2000)
        job = stream.run(Job(search, batch=32, inflight=2, visits=2000))
        assert search.total_visits() == 2000
        assert search.visits().sum() == 1999  # the root's own first visit has no edge
        assert max(network.calls) <= 32
        assert job.leaves + search.reused() <= 2000
        assert job.done.is_set() and job.seconds > 0
        assert search.remaining == 0
    finally:
        stream.close()


def test_jobs_share_calls_highest_priority_first():
    from cpu_search import Search, SearchSettings
    from gpu_play import Job, Stream

    network = FakeNetwork(delay=0.002)
    stream = Stream(network, max_call=48, calls=2)
    try:
        searches = [Search(SearchSettings(batch=32), seed) for seed in range(2)]
        for s in searches:
            s.begin(1500)
        jobs = [
            Job(searches[0], batch=32, inflight=2, visits=1500, priority=1),
            Job(searches[1], batch=32, inflight=2, visits=1500, priority=0),
        ]
        for job in jobs:
            stream.submit(job)
        for job in jobs:
            job.done.wait()
        # calls hold both jobs' leaves, up to the call's limit, the first job's batch whole
        assert max(network.calls) > 32
        assert max(network.calls) <= 48
        assert all(s.total_visits() == 1500 for s in searches)
        # the high-priority job's calls are fuller: it finishes in fewer
        assert jobs[0].calls <= jobs[1].calls
    finally:
        stream.close()


def test_deadline_and_cancel():
    from cpu_search import Search, SearchSettings
    from gpu_play import Job, Stream

    stream = Stream(FakeNetwork(delay=0.001), calls=2)
    try:
        search = Search(SearchSettings(batch=8))
        search.begin(1 << 30)
        start = time.perf_counter()
        stream.run(Job(search, batch=8, deadline=start + 0.05))
        assert 0.05 <= time.perf_counter() - start < 0.5
        assert search.total_visits() > 0
        search.begin(1 << 30)
        job = stream.submit(Job(search, batch=8))
        time.sleep(0.02)
        job.cancel()
        assert job.done.wait(1.0)
        assert search.remaining == 0
    finally:
        stream.close()


def test_stream_errors_reach_the_caller():
    from cpu_search import Search, SearchSettings
    from gpu_play import Job, Stream

    class Broken(FakeNetwork):
        def wait(self, pending):
            raise ValueError("boom")

    stream = Stream(Broken(), calls=1)
    search = Search(SearchSettings(batch=8))
    search.begin(100)
    with pytest.raises(RuntimeError):
        stream.run(Job(search, batch=8, visits=100))


def test_player_plays_whole_games_and_ponder_keeps_the_tree():
    from clock_match import Board, play_game
    from gpu_play import GpuPlayer, GpuSettings, Stream

    stream = Stream(FakeNetwork(), calls=2)
    try:
        settings = GpuSettings(batch=16, inflight=2)
        a = GpuPlayer(stream, settings, seed=0, visits=200)
        b = GpuPlayer(stream, settings, seed=1, visits=200)
        game = play_game(a, b, [], 1e9)
        board = Board()
        for move in game.moves:
            board.play(move)
        assert not board.legal()

        # a ponder, the opponent to move: what it found under their move is kept
        p = GpuPlayer(stream, settings, seed=2)
        p.set_position(*START)
        p.ponder()
        time.sleep(0.05)
        p.play("b", 19)  # d3
        assert p.pondering is None
        me, opp, _, _ = p.search.position()
        assert (me | opp).bit_count() == 5
        assert p.search.visits().sum() > 0
        p.close()
    finally:
        stream.close()


def test_player_from_a_given_position_with_a_clock():
    from gpu_play import GpuPlayer, GpuSettings, Stream

    stream = Stream(FakeNetwork(), calls=2)
    try:
        p = GpuPlayer(stream, GpuSettings(batch=8), seed=0)
        p.set_position(*START)
        start = time.perf_counter()
        move = p.move("b", 2.0)
        assert move in (19, 26, 37, 44)
        assert time.perf_counter() - start < 1.0
        assert p.last_value is not None and p.last_value[0] == "q"
        assert p.last_job is not None and p.last_job.leaves > 0
    finally:
        stream.close()


@dataclass(frozen=True)
class View:
    """The fields of `ggs_player.GameView` the bridge reads."""

    game_id: str
    start: tuple[int, int, str]
    moves: tuple
    black: int
    white: int
    color: str
    seconds_left: float = 30.0
    extension: float = 0.0
    extended: bool = False
    increment: float = 0.0
    komi: bool = False
    mine: str | None = "b"

    @property
    def empties(self) -> int:
        return 64 - (self.black | self.white).bit_count()


@dataclass(frozen=True)
class Decision:
    square: int | None
    eval: float
    seconds: float
    info: dict = field(default_factory=dict)


class Policy:
    def seconds_left(self, view) -> float:
        return 0.3

    def extension_move(self, view) -> float:
        return 0.01


def view_after(game_id: str, moves: list, mine: str = "b") -> View:
    from clock_match import Board

    board = Board.at(*START)
    for m in moves:
        board.play(m)
    return View(
        game_id,
        START,
        tuple(moves),
        board.discs[0],
        board.discs[1],
        board.color,
        mine=mine,
    )


def first_legal(moves: list) -> int:
    from clock_match import Board

    board = Board.at(*START)
    for m in moves:
        board.play(m)
    return board.legal()[0]


def test_ggs_bridge_decides_ponders_and_resyncs():
    from gpu_play import GgsBridge, GpuPlayer, GpuSettings, Stream

    stream = Stream(FakeNetwork(), calls=2)
    try:
        seeds = iter(range(100))
        bridge = GgsBridge(
            lambda: GpuPlayer(stream, GpuSettings(batch=8), seed=next(seeds)),
            Decision,
            Policy(),
        )
        # synchro: we're black in .1.0, white in .1.1
        decided = bridge.decide(view_after(".1.0", []), cores=1)
        assert decided.square in (19, 26, 37, 44)
        bridge.observe(view_after(".1.1", [], mine="w"))
        assert bridge.games[".1.1"].player.pondering is not None
        bridge.observe(
            view_after(".1.0", [decided.square])
        )  # their turn in .1.0: ponder
        assert bridge.games[".1.0"].player.pondering is not None
        # their move in .1.1 came: our turn there, the ponder's tree kept
        reply = bridge.decide(view_after(".1.1", [19], mine="w"), cores=1)
        assert reply.square is not None
        assert bridge.games[".1.1"].player.pondering is None
        # a board the player wasn't told: started afresh from the server's
        moves = [decided.square, first_legal([decided.square])]
        view = view_after(".1.0", moves)
        # what it thinks it was told disagrees
        bridge.games[".1.0"].moves = [44 if decided.square != 44 else 37]
        again = bridge.decide(view, cores=1)
        assert again.square is not None
        bridge.forget(".1.0")
        assert ".1.0" not in bridge.games
        bridge.close()
        assert not bridge.games
    finally:
        stream.close()


def test_threads_deciding_at_once_share_the_stream():
    from gpu_play import GgsBridge, GpuPlayer, GpuSettings, Stream

    network = FakeNetwork(delay=0.001)
    stream = Stream(network, calls=2)
    try:
        seeds = iter(range(100))
        bridge = GgsBridge(
            lambda: GpuPlayer(stream, GpuSettings(batch=8), seed=next(seeds)),
            Decision,
            Policy(),
            ponder=False,
        )
        out = {}
        threads = [
            threading.Thread(
                target=lambda g=g: out.setdefault(
                    g, bridge.decide(view_after(g, []), 1)
                )
            )
            for g in (".1", ".2")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(out) == 2
        assert max(network.calls) > 8  # some calls carried both games' leaves
    finally:
        stream.close()


def test_net_player_on_the_gpu_parses_its_options(monkeypatch):
    import gpu_play
    from clock_match import net_player

    made = {}

    def fake_stream(net, dtype, max_call, calls):
        made.update(net=net, dtype=dtype, max_call=max_call, calls=calls)
        return gpu_play.Stream(FakeNetwork(), max_call, calls)

    monkeypatch.setattr(gpu_play, "shared_stream", fake_stream)
    new = net_player(
        {
            "net": "N",
            "device": "gpu",
            "batch": "64",
            "inflight": "3",
            "max_call": "512",
            "visits": "100",
            "solve": "12",
            "ponder": "1",
            "threads": "2",
        }
    )
    player = new()
    try:
        assert made == {"net": "N", "dtype": "bfloat16", "max_call": 512, "calls": 2}
        assert player.settings.batch == 64 and player.settings.inflight == 3
        assert player.visits == 100 and player.ponders
        assert player.solving is not None and player.solving.empties == 12
        assert player.solving.threads == 2
    finally:
        player.close()
        player.stream.close()
    with pytest.raises(ValueError):
        net_player({"net": "N", "device": "gpu", "bogus": "1"})


def test_ggs_bridge_factory_takes_the_clients_classes(monkeypatch):
    import gpu_play

    monkeypatch.setattr(
        gpu_play,
        "shared_stream",
        lambda *a: gpu_play.Stream(FakeNetwork(), 256, 2),
    )
    bridge = gpu_play.make_ggs_bridge(
        {"net": "N", "batch": "8", "ponder_batch": "4", "solve": "14"},
        cores=2,
        decision=Decision,
        time_policy=Policy(),
    )
    try:
        assert bridge.ponder and bridge.ponder_batch == 4
        decided = bridge.decide(view_after(".7", []), cores=2)
        assert isinstance(decided, Decision)
        player = bridge.games[".7"].player
        assert player.solving is not None and player.solving.exact
    finally:
        bridge.close()
        bridge.games.clear()
