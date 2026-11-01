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
from core.league.store import (
    SCHEMA_VERSION,
    GameRecord,
    Store,
    decode_moves,
    encode_moves,
)


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


def test_seconds_per_move_leaves_out_compiling_and_results_from_elsewhere(store):
    x = store.add_external_player("x", "edax")
    y = store.add_external_player("y", "edax")
    games = [game(x, y)]
    store.record_batch(x, y, 9.0, False, "abc", "f", games, (6.0, 20), (3.0, 10))
    store.record_batch(y, x, 9.0, False, "abc", "f", games, (2.0, 10), (2.0, 20))
    store.record_batch(x, y, 99.0, True, "abc", "f", games, (90.0, 10), (9.0, 10))
    store.record_batch(x, y, 0.0, False, "a paper", "", games)
    assert store.seconds_per_move() == {
        x: pytest.approx(8.0 / 40),
        y: pytest.approx(5.0 / 20),
    }


def test_a_series_counts_its_batches_and_ends(store):
    x, y = two_players(store)
    request = store.request_match(x, y, 4)
    store.record_batch(x, y, 1.0, False, "abc", "f", [game(x, y)] * 2, request=request)
    assert store.pending_matches()[0].played == 2
    assert store.pending_matches()[0].remaining == 2
    store.record_batch(x, y, 1.0, False, "abc", "f", [game(y, x)] * 2, request=request)
    assert store.pending_matches() == []


def test_a_failed_batch_doesnt_count_towards_its_series(store):
    x, y = two_players(store)
    request = store.request_match(x, y, 4)
    with pytest.raises(sqlite3.IntegrityError):
        store.record_batch(
            x, y, 1.0, False, "abc", "f", [GameRecord(x, y, 0, 2, [1])], request=request
        )
    assert store.pending_matches()[0].played == 0


def test_cancelling_a_series_keeps_its_games(store):
    x, y = two_players(store)
    request = store.request_match(x, y, 4)
    store.record_batch(x, y, 1.0, False, "abc", "f", [game(x, y)] * 2, request=request)
    store.cancel_match(request)
    assert store.pending_matches() == []
    assert store.num_games() == 2
    with pytest.raises(ValueError):
        store.cancel_match(request)
    with pytest.raises(ValueError):
        store.cancel_match(999)


def test_a_batch_finishing_after_its_series_was_cancelled_counts_without_reviving_it(
    store,
):
    x, y = two_players(store)
    request = store.request_match(x, y, 2)
    store.cancel_match(request)
    store.record_batch(x, y, 1.0, False, "abc", "f", [game(x, y)] * 2, request=request)
    row = store.db.execute("SELECT state, played FROM match_requests").fetchone()
    assert tuple(row) == ("cancelled", 2)


# the file as it was before `user_version` was used
VERSION_0 = """
CREATE TABLE networks (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, path TEXT NOT NULL,
    sha256 TEXT NOT NULL, net TEXT NOT NULL, run TEXT NOT NULL, url TEXT, source TEXT NOT NULL,
    added REAL NOT NULL);
CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    network_id INTEGER REFERENCES networks(id), sims INTEGER, external TEXT,
    settings TEXT NOT NULL DEFAULT '{}', retired INTEGER NOT NULL DEFAULT 0, added REAL NOT NULL);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE match_requests (id INTEGER PRIMARY KEY, a_id INTEGER NOT NULL, b_id INTEGER NOT NULL,
    games INTEGER NOT NULL, played INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'pending', added REAL NOT NULL, ended REAL);
INSERT INTO players (id, name, external, added) VALUES (1, 'x', 'edax', 0), (2, 'y', 'edax', 0);
INSERT INTO match_requests (id, a_id, b_id, games, added) VALUES (5, 1, 2, 8, 0), (9, 2, 1, 8, 0);
"""


@pytest.fixture
def old_file(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    db = sqlite3.connect(root / "league.db")
    db.executescript(VERSION_0)
    db.close()
    return root


def test_opening_an_old_file_backs_it_up_and_migrates_it(old_file):
    store = Store(old_file)
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert [r.id for r in store.pending_matches()] == [5, 9]
    assert store.players()[0].description == ""
    # the series keep their order, and new ones go after them
    assert store.request_match(1, 2, 2) == 10
    assert [r.id for r in store.pending_matches()] == [5, 9, 10]
    store.close()
    backup = sqlite3.connect(old_file / "league-backup-before-schema-1.db")
    assert backup.execute("PRAGMA user_version").fetchone()[0] == 0
    assert backup.execute("SELECT COUNT(*) FROM match_requests").fetchone()[0] == 2
    backup.close()


def test_opening_a_migrated_file_again_changes_nothing(old_file):
    Store(old_file).close()
    backup = old_file / "league-backup-before-schema-1.db"
    backup.unlink()
    Store(old_file).close()
    assert not backup.exists()


def test_a_file_newer_than_the_code_isnt_touched(old_file):
    Store(old_file).close()
    db = sqlite3.connect(old_file / "league.db")
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    db.close()
    with pytest.raises(ValueError):
        Store(old_file)
    db = sqlite3.connect(old_file / "league.db")
    assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION + 1
    db.close()


def test_a_backup_of_a_file_migrated_meanwhile_isnt_kept(old_file):
    store = Store(old_file)
    backup = old_file / "league-backup-before-schema-1.db"
    made = backup.read_bytes()
    # a process that read version 0 before this one migrated it
    store._backup(0)
    assert backup.read_bytes() == made
    assert not list(old_file.glob("*.partial"))
    store.close()


def test_a_new_file_needs_no_backup(tmp_path):
    store = Store(tmp_path / "new")
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    store.close()
    assert not list((tmp_path / "new").glob("league-backup*"))


def test_a_networks_budgets_share_its_notes(store, checkpoint):
    store.add_network("n", checkpoint, "resnet:2x8", "run-1")
    low, high = store.add_net_player("n", 128), store.add_net_player("n", 1000)
    store.set_notes(low, description="good", tags=["b", "a", "b", " "])
    players = {p.id: p for p in store.players()}
    assert players[high].description == "good"
    assert players[high].tags == ("a", "b")


def test_an_engines_notes_are_its_own(store):
    x, y = two_players(store)
    store.set_notes(x, description="deep")
    store.set_notes(x, tags=["t"])
    players = {p.id: p for p in store.players()}
    assert (players[x].description, players[x].tags) == ("deep", ("t",))
    assert (players[y].description, players[y].tags) == ("", ())


def test_notes_for_nobody_fail(store):
    with pytest.raises(ValueError):
        store.set_notes(99, description="?")


def test_a_series_moves_in_the_queue(store):
    x, y = two_players(store)
    first, second, third = [store.request_match(x, y, 2) for _ in range(3)]
    store.move_match(third, 0)
    assert [r.id for r in store.pending_matches()] == [third, first, second]
    store.move_match(third, 99)
    assert [r.id for r in store.pending_matches()] == [first, second, third]
    # and one asked for after that still goes last
    fourth = store.request_match(x, y, 2)
    assert [r.id for r in store.pending_matches()][-1] == fourth


def test_moving_a_series_that_isnt_pending_fails(store):
    x, y = two_players(store)
    request = store.request_match(x, y, 2)
    store.cancel_match(request)
    with pytest.raises(ValueError):
        store.move_match(request, 0)


def test_a_series_can_be_lengthened_and_shortened_to_what_its_played(store):
    x, y = two_players(store)
    request = store.request_match(x, y, 4)
    store.record_batch(x, y, 1.0, False, "abc", "", [game(x, y)] * 2, request=request)
    store.set_match_games(request, 8)
    assert store.pending_matches()[0].games == 8
    with pytest.raises(ValueError):
        store.set_match_games(request, 1)
    store.set_match_games(request, 2)
    assert store.pending_matches() == []
