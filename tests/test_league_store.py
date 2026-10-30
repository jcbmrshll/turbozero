"""The league's permanent record of its players and games."""

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.league.ratings import fit_ratings
from core.league.store import GameRecord, Store, decode_moves, encode_moves


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "league")
    yield store
    store.close()


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "7.eqx"
    path.write_bytes(b"weights")
    return path


def game(first, second, result=1, opening=0):
    return GameRecord(first, second, opening, result, [1, 2, 64])


def batch(store, a, b, games):
    return store.record_batch(a, b, 1.0, False, "abc", "bfloat16", games)


def two_players(store):
    return store.add_external_player("x", "edax"), store.add_external_player(
        "y", "edax"
    )


def test_a_registered_checkpoint_outlives_its_source(store, checkpoint):
    store.add_network("n", checkpoint, "resnet:2x8", "run-1")
    player = store.players()
    assert player == []
    store.add_net_player("n", 128)
    checkpoint.unlink()
    player = store.players()[0]
    assert player.name == "n@128"
    assert player.sims == 128
    assert player.run == "run-1"
    assert player.net == "resnet:2x8"
    assert Path(player.path).read_bytes() == b"weights"


def test_the_same_network_at_two_budgets_is_two_players(store, checkpoint):
    store.add_network("n", checkpoint, "resnet:2x8", "run-1")
    first, second = store.add_net_player("n", 128), store.add_net_player("n", 1000)
    assert first != second
    assert [p.network for p in store.players()] == ["n", "n"]


def test_registering_again_changes_nothing(store, checkpoint):
    first = store.add_network("n", checkpoint, "resnet:2x8", "r")
    assert store.add_network("n", checkpoint, "resnet:2x8", "r") == first
    assert store.add_net_player("n", 8) == store.add_net_player("n", 8)
    assert len(store.players()) == 1


def test_a_name_cant_change_its_network(store, checkpoint, tmp_path):
    store.add_network("n", checkpoint, "resnet:2x8", "r")
    other = tmp_path / "other.eqx"
    other.write_bytes(b"different")
    with pytest.raises(ValueError):
        store.add_network("n", other, "resnet:2x8", "r")
    with pytest.raises(ValueError):
        store.add_network("n", checkpoint, "resnet:4x8", "r")


def test_engines_keep_their_settings(store):
    store.add_external_player("edax-L8", "edax", {"level": 8, "args": ["-h", "20"]})
    player = store.players()[0]
    assert player.is_external
    assert player.run == "edax"
    assert player.settings == {"level": 8, "args": ["-h", "20"]}


def test_pair_records_count_each_side_whoever_moved_first(store):
    x, y = two_players(store)
    batch(store, x, y, [game(x, y, 1), game(y, x, 1), game(y, x, -1), game(x, y, 0)])
    record = store.pair_records()[0]
    assert (record.a, record.b) == (x, y)
    # x won as first, lost as second, won as second, drew as first
    assert (record.wins, record.draws, record.losses) == (2, 1, 1)


def test_moves_keep_large_actions(store):
    x, y = two_players(store)
    batch(store, x, y, [GameRecord(x, y, 0, 1, [0, 64, 4671, 65000])])
    moves = store.db.execute("SELECT moves FROM games").fetchone()[0]
    assert decode_moves(moves) == [0, 64, 4671, 65000]
    assert encode_moves([3]) == b"\x03\x00"


def test_played_openings_cover_both_orders(store):
    x, y = two_players(store)
    z = store.add_external_player("z", "edax")
    batch(store, x, y, [game(x, y, opening=4), game(y, x, opening=5)])
    batch(store, x, z, [game(x, z, opening=6)])
    assert store.played_openings(y, x) == {4, 5}


def test_batch_costs_leave_out_compiling(store):
    x, y = two_players(store)
    store.record_batch(x, y, 90.0, True, "abc", "f", [game(x, y)])
    store.record_batch(x, y, 9.0, False, "abc", "f", [game(x, y), game(y, x)])
    assert store.batch_costs() == [(x, y, 2, 9.0)]


def test_a_failed_batch_leaves_nothing(store):
    x, y = two_players(store)
    with pytest.raises(sqlite3.IntegrityError):
        batch(store, x, y, [game(x, y), GameRecord(x, y, 0, 2, [1])])
    assert store.num_games() == 0
    assert store.db.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 0


def test_ratings_recompute_the_same_after_reopening(tmp_path):
    store = Store(tmp_path / "s")
    x, y = two_players(store)
    batch(store, x, y, [game(x, y, 1), game(y, x, -1), game(x, y, 0)])
    before = fit_ratings([x, y], store.pair_records())
    store.close()
    store = Store(tmp_path / "s")
    after = fit_ratings([x, y], store.pair_records())
    assert before.elo.tolist() == after.elo.tolist()
    store.close()


WRITER = """
import sys
from core.league.store import GameRecord, Store

store = Store(sys.argv[1])
x = store.add_external_player("x", "edax")
y = store.add_external_player("y", "edax")
print("ready", flush=True)
while True:
    store.record_batch(x, y, 1.0, False, "abc", "f",
                       [GameRecord(x, y, i, 1, [1, 2, 3]) for i in range(50)])
"""


def test_being_killed_while_writing_leaves_whole_batches(tmp_path):
    root = tmp_path / "s"
    process = subprocess.Popen(
        [sys.executable, "-c", WRITER, str(root)],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"
    time.sleep(0.5)
    process.send_signal(signal.SIGKILL)
    process.wait()
    process.stdout.close()

    store = Store(root)
    assert store.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    batches = store.db.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
    assert batches > 0
    assert store.num_games() == 50 * batches
    store.close()


def test_retiring_needs_a_player(store):
    x, _ = two_players(store)
    store.set_retired("x")
    assert store.players()[0].id == x and store.players()[0].retired
    with pytest.raises(ValueError):
        store.set_retired("nobody")


def test_meta_round_trips(store):
    assert store.get_meta("k", 3) == 3
    store.set_meta("k", {"a": 1})
    assert store.get_meta("k") == {"a": 1}


def test_a_result_from_elsewhere_counts_but_has_no_cost(store):
    x = store.add_external_player("x", "paper", {"url": "https://example.com/paper"})
    y = store.add_external_player("y", "edax")
    games = [GameRecord(x, y, -1, 1, []), GameRecord(x, y, -1, 0, [])]
    store.record_batch(x, y, 0.0, False, "a paper", "", games)
    record = store.pair_records()[0]
    assert (record.wins, record.draws, record.losses) == (1, 1, 0)
    assert store.batch_costs() == []
    assert store.players()[0].url == "https://example.com/paper"
