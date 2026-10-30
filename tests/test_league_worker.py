"""The league's worker, with fakes for the GPU, the queue and the games."""

import signal

import pytest

from core.league import gate as gate_module
from core.league.game import LeagueGame
from core.league.gate import SlickGate, ThrottledGate
from core.league.match import BatchResult, Stopped
from core.league.store import GameRecord, Store
from core.league.worker import play, supervise


class FakeGame(LeagueGame):
    name = "fake"
    env = None

    def metadata(self, state):
        raise NotImplementedError

    @property
    def step_fn(self):
        raise NotImplementedError

    def load_network(self, path, net, dtype):
        raise NotImplementedError

    def make_evaluator(self, network, sims):
        raise NotImplementedError

    def openings(self, indices):
        raise NotImplementedError

    def external_seconds(self, player):
        return 0.1

    def external_parallel(self, player):
        return 8

    def num_openings(self):
        return 100_000


class FakeGate:
    def __init__(self, reasons=()):
        self.reasons = list(reasons)

    def busy(self):
        return self.reasons.pop(0) if self.reasons else None


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "league")
    store.add_external_player("edax-1", "edax")
    store.add_external_player("edax-2", "edax")
    store.add_network("n", _checkpoint(tmp_path), "resnet:2x8", "r")
    store.add_net_player("n", 8)
    store.add_net_player("n", 16)
    yield store
    store.close()


def _checkpoint(tmp_path):
    path = tmp_path / "c.eqx"
    path.write_bytes(b"x")
    return path


def fake_playing(game, programs, a, b, openings, seed, should_stop):
    return BatchResult(
        [GameRecord(a.id, b.id, int(o), 1, [0, 1]) for o in openings]
        + [GameRecord(b.id, a.id, int(o), 1, [0, 1]) for o in openings],
        1.0,
        False,
    )


def run_play(store, gate=None, **kwargs):
    return play(
        store,
        FakeGame(),
        gate or FakeGate(),
        version="abc",
        seed=0,
        playing=fake_playing,
        **kwargs,
    )


def test_plays_batches_and_records_them_with_their_code_version(store):
    assert run_play(store, max_batches=3) == "played its batches"
    assert store.db.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 3
    assert store.num_games() > 0
    versions = store.db.execute("SELECT DISTINCT code_version FROM batches").fetchall()
    assert [row[0] for row in versions] == ["abc"]


def test_a_pair_doesnt_repeat_its_openings(store):
    run_play(store, max_batches=20)
    rows = store.db.execute(
        "SELECT MIN(first_id, second_id), MAX(first_id, second_id), opening, COUNT(*) "
        "FROM games GROUP BY 1, 2, 3"
    ).fetchall()
    assert rows
    # each opening once with each moving first
    assert all(row[3] == 2 for row in rows)


def test_stops_before_a_batch_when_the_gpu_is_wanted(store):
    assert run_play(store, FakeGate(["queue waiting #4"])) == "queue waiting #4"
    assert store.num_games() == 0
    assert store.get_meta("player")["state"] == "idle"


def test_the_stop_flag_stops_it(store):
    store.set_meta("stop", True)
    assert run_play(store) == "stopped"


def test_an_abandoned_batch_writes_nothing_and_says_why(store):
    def abandoned(game, programs, a, b, openings, seed, should_stop):
        assert should_stop()
        raise Stopped

    reason = play(
        store,
        FakeGame(),
        FakeGate([None, "on the GPU: 123"]),
        version="abc",
        seed=0,
        playing=abandoned,
    )
    assert reason == "on the GPU: 123"
    assert store.num_games() == 0


def test_signals_are_put_back(store):
    before = signal.getsignal(signal.SIGHUP)
    run_play(store, max_batches=1)
    assert signal.getsignal(signal.SIGHUP) == before


def test_resumes_from_the_store_alone(tmp_path):
    root = tmp_path / "again"
    store = Store(root)
    store.add_external_player("edax", "edax")
    store.add_network("n", _checkpoint(tmp_path), "resnet:2x8", "r")
    store.add_net_player("n", 8)
    run_play(store, max_batches=2)
    games = store.num_games()
    store.close()
    store = Store(root)
    run_play(store, max_batches=2)
    assert store.num_games() > games
    store.close()


class FakeQueue:
    def __init__(self, states=()):
        self.submitted = 0
        self.states = list(states)

    def submit(self):
        self.submitted += 1
        return 100 + self.submitted

    def state(self, job):
        return self.states.pop(0) if self.states else "running"


class Clock:
    """Time that passes only when something sleeps."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def run_supervise(store, queue, gate=None, rounds=5, clock=None, **kwargs):
    clock = clock or Clock()
    return supervise(
        store,
        gate or FakeGate(),
        queue,
        sleep=clock.sleep,
        clock=clock,
        load=lambda: 1.0,
        rounds=rounds,
        **kwargs,
    )


def test_supervisor_queues_the_player_once_the_gpu_stays_free(store):
    queue = FakeQueue()
    run_supervise(store, queue, FakeGate(["busy", None, None]), rounds=4)
    assert queue.submitted == 1
    assert store.get_meta("player_job") == 101


def test_supervisor_doesnt_queue_on_one_free_look(store):
    queue = FakeQueue()
    run_supervise(store, queue, FakeGate([None, "busy", None, "busy"]), rounds=4)
    assert queue.submitted == 0
    assert store.get_meta("supervisor")["reason"] == "busy"


def test_supervisor_requeues_after_the_player_yields(store):
    # a long stretch of play, then it hands over
    queue = FakeQueue(["running"] * 20 + ["ended"])
    run_supervise(store, queue, rounds=26)
    assert queue.submitted == 2
    assert "failures" not in str(store.get_meta("supervisor"))


def test_supervisor_backs_off_from_a_player_that_keeps_failing(store):
    class Failing(FakeQueue):
        def state(self, job):
            return "failed"

    queue = Failing()
    clock = Clock()
    run_supervise(store, queue, rounds=40, clock=clock)
    assert 1 < queue.submitted < 8
    # two minutes, then four, then eight...
    assert clock.now - 1000 >= 60 * (2 + 4)


def test_supervisor_stops_after_too_many_failures(store):
    class Failing(FakeQueue):
        def state(self, job):
            return "failed"

    assert run_supervise(store, Failing(), rounds=10_000, max_failures=3) == "failed"
    assert "3 failures" in store.get_meta("supervisor")["reason"]


def test_a_quick_exit_counts_as_a_failure_and_a_long_one_doesnt(store):
    queue = FakeQueue(["running", "ended"])
    run_supervise(store, queue, rounds=4)
    assert queue.submitted == 1
    assert "1 failures" in store.get_meta("supervisor")["reason"]


def test_supervisor_stops_when_its_entry_is_cancelled(store):
    queue = FakeQueue(["cancelled"])
    assert run_supervise(store, queue, rounds=10) == "cancelled"
    assert queue.submitted == 1
    assert store.get_meta("supervisor")["state"] == "stopped"


def test_supervisor_stops_with_the_flag(store):
    store.set_meta("stop", True)
    assert run_supervise(store, FakeQueue()) == "stopped"


def test_supervisor_waits_for_the_machine_load(store):
    queue = FakeQueue()
    supervise(
        store, FakeGate(), queue, sleep=lambda _: None, load=lambda: 99.0, rounds=5
    )
    assert queue.submitted == 0
    assert "load" in store.get_meta("supervisor")["reason"]


def test_supervisor_picks_up_a_running_entry_after_a_restart(store):
    store.set_meta("player_job", 55)
    queue = FakeQueue(["running"])
    run_supervise(store, queue, rounds=1)
    assert queue.submitted == 0
    assert store.get_meta("supervisor")["job"] == 55


def test_supervisor_survives_the_queue_refusing(store):
    class Refusing(FakeQueue):
        def submit(self):
            raise RuntimeError("no such task")

    run_supervise(store, Refusing(), rounds=4)
    assert "no such task" in store.get_meta("supervisor")["reason"]


def board(running=None, queued=()):
    return {
        "resources": [
            {
                "name": "5080",
                "running": running and {"id": running, "about": "training"},
                "queued": [{"id": q, "about": "next"} for q in queued],
            }
        ]
    }


@pytest.fixture
def gpu(monkeypatch):
    state = {"processes": set(), "board": board()}
    monkeypatch.setattr(gate_module, "gpu_processes", lambda: state["processes"])
    monkeypatch.setattr(gate_module, "queue_board", lambda url: state["board"])
    return state


def test_gate_is_free_with_nothing_else(gpu):
    assert SlickGate().busy() is None


def test_gate_sees_processes_outside_the_queue(gpu):
    gpu["processes"] = {7}
    assert "7" in str(SlickGate().busy())
    assert SlickGate(own_pid=7).busy() is None


def test_gate_sees_the_queue(gpu):
    gpu["board"] = board(running=3)
    assert "#3" in str(SlickGate().busy())
    assert SlickGate(own_job=3).busy() is None
    gpu["board"] = board(running=3, queued=[9])
    assert "#9" in str(SlickGate(own_job=3).busy())


def test_gate_counts_what_it_cant_read_as_busy(gpu):
    gpu["processes"] = None
    assert SlickGate().busy() is not None
    gpu["processes"] = set()
    gpu["board"] = None
    assert SlickGate().busy() is not None


def test_throttled_gate_looks_at_most_so_often():
    fake = FakeGate(["first", "second"])
    gate = ThrottledGate(fake, every=1000)
    assert gate.busy() == "first"
    assert gate.busy() == "first"
    assert len(fake.reasons) == 1
