"""What timing_benchmark.py makes of configs and their games, with fake games and a directory as the
store. No networks, no engines."""

import json
import shutil

import pytest
from example_modules import load_example

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")

modal_store, timing_benchmark = load_example(
    "othello", "modal_store", "timing_benchmark"
)


def fake_game(seconds: float = 1.0, kind: str = "net") -> list[dict]:
    moves = []
    for side in (0, 1):
        for ply in range(3):
            move = {"side": side, "ply": ply, "seconds": seconds}
            if kind == "net":
                move |= {
                    "evals": 80,
                    "visits": 100,
                    "tree_seconds": 0.1,
                    "net_seconds": 0.8,
                }
            else:
                move |= {"nodes": 1000}
            moves.append(move)
    return moves


def fake_play(config: dict, _index: int) -> list[dict]:
    return fake_game(kind=config["kind"])


def test_plan_covers_the_league_with_unique_ids():
    plan = timing_benchmark.make_plan()
    ids = [config["id"] for config in plan]
    assert len(ids) == len(set(ids))
    for wanted in ("8x128-16000-b32", "8x128-1000-b16", "4x64-450-b1", "edax-L28-t8"):
        assert wanted in ids
    assert ids != sorted(ids)
    seconds = [timing_benchmark.estimated_seconds(config) for config in plan]
    assert seconds == sorted(seconds)


def test_plotted_batch_and_what_is_known_of_its_strength():
    assert timing_benchmark.plotted_batch(128) == 1
    assert timing_benchmark.plotted_batch(1000) == 16
    assert timing_benchmark.plotted_batch(16000) == 32
    assert timing_benchmark.batch_strength("8x128", 128, 1) == "sequential"
    assert timing_benchmark.batch_strength("8x128", 1000, 16) == "measured_no_loss"
    assert timing_benchmark.batch_strength("8x128", 16000, 32) == "unmeasured"
    assert timing_benchmark.batch_strength("6x96", 1000, 16) == "unmeasured"


def test_summary_charges_one_side_a_game():
    config = timing_benchmark.net_config("4x64", 100, 1, 2)
    summary = timing_benchmark.summarize_config(config, [fake_game(), fake_game(3.0)])
    assert summary["games"] == 2
    assert summary["moves_per_game"] == 3
    assert summary["seconds_per_game"] == pytest.approx(6.0)
    assert summary["seconds_per_game"] == pytest.approx(
        summary["seconds_per_move"] * summary["moves_per_game"]
    )
    assert summary["evals_per_visit"] == pytest.approx(0.8)


def test_fit_recovers_a_line_and_predicts():
    points = [(100.0, 0.2), (1000.0, 1.1), (4000.0, 4.1)]
    overhead, cost, predicted = timing_benchmark.fit_cost(points, 16000.0)
    assert overhead == pytest.approx(0.1)
    assert cost == pytest.approx(0.001)
    assert predicted == pytest.approx(16.1)


def run(store, attempt: int, play=fake_play, limit: float = 1e9, used: float = 0.0):
    plan = [
        timing_benchmark.net_config("4x64", 1, 1, 2),
        timing_benchmark.engine_config("edax", 4, 2),
    ]
    refs = [timing_benchmark.net_config("4x64", 128, 1, 1)]
    timing_benchmark.run_configs(store, "j", attempt, plan, refs, play, limit, used)


def test_run_writes_each_config_and_references_either_end(tmp_path):
    store = modal_store.DirStore(tmp_path)
    run(store, 0)
    keys = set(store.sizes("j/"))
    assert "j/done/4x64-1-b1.json" in keys
    assert "j/done/edax-L4-t8.json" in keys
    assert "j/moves/0/edax-L4-t8.jsonl" in keys
    assert "j/references/a0-r0-4x64-128-b1.json" in keys
    assert "j/references/a0-r1-4x64-128-b1.json" in keys
    assert modal_store.get_json(store, "j/status.json")["state"] == "done"


def test_a_new_attempt_skips_what_is_finished_and_redoes_what_is_not(tmp_path):
    store = modal_store.DirStore(tmp_path)
    played = []

    def stops_in_the_engine(config: dict, index: int) -> list[dict]:
        if config["kind"] == "engine":
            store.put_bytes("j/stop.json", b"true")
        return fake_play(config, index)

    run(store, 0, stops_in_the_engine)
    assert modal_store.get_json(store, "j/status.json")["state"] == "stopped"
    assert "j/done/edax-L4-t8.json" not in store.sizes("j/")
    store.put_bytes("j/stop.json", b"false")

    def counts(config: dict, index: int) -> list[dict]:
        played.append(config["id"])
        return fake_play(config, index)

    run(store, 1, counts)
    assert "4x64-1-b1" not in played
    assert played.count("edax-L4-t8") == 2


def test_the_limit_stops_a_job_between_games(tmp_path):
    store = modal_store.DirStore(tmp_path)
    run(store, 0, limit=10.0, used=11.0)
    assert modal_store.get_json(store, "j/status.json")["state"] == "out of time"
    assert not store.sizes("j/done/")


def record(config: dict, attempt: int, began: float, seconds: float) -> dict:
    return {
        "config": config,
        "attempt": attempt,
        "began": began,
        "ended": began + 1,
        "summary": timing_benchmark.summarize_config(
            config, [fake_game(seconds, config["kind"])]
        ),
    }


def test_drift_notes_a_reference_that_moved_around_a_config():
    ref = timing_benchmark.net_config("4x64", 128, 1, 1)
    refs = [
        record(ref, 0, 0.0, 1.0),
        record(ref, 0, 100.0, 1.05),
        record(ref, 0, 200.0, 1.4),
    ]
    steady = record(timing_benchmark.net_config("4x64", 1, 1, 1), 0, 50.0, 1.0)
    moved = record(timing_benchmark.net_config("4x64", 8, 1, 1), 0, 150.0, 1.0)
    assert timing_benchmark.drift_note(steady, refs) is None
    assert "1.33x" in timing_benchmark.drift_note(moved, refs)


def test_league_run_uses_the_league_keys():
    info = {
        "cpuinfo": {"model": "Some CPU", "flags": ["avx2"]},
        "openvino": {"version": "1"},
        "hostname": "h",
        "memory": ["MemTotal: 1 kB"],
        "kernel": "k",
        "caches": [],
        "thread_layout": {16: "4,16"},
        "precision": "bf16",
    }
    net = record(timing_benchmark.net_config("8x128", 1000, 16, 1), 0, 0.0, 1.0)
    sequential = record(timing_benchmark.net_config("8x128", 1000, 1, 1), 0, 0.0, 1.0)
    engine = record(timing_benchmark.engine_config("edax", 20, 1), 0, 0.0, 1.0)
    scaling = record(
        timing_benchmark.engine_config("edax", 20, 1, threads=1), 0, 0.0, 1.0
    )
    specs = {"8x128": {"net": "resnet:8x128"}}
    run = timing_benchmark.league_run(
        "j", info, [net, sequential, engine, scaling], [], specs, "raw"
    )
    json.dumps(run)
    assert run["name"] == "h-8t-16-23-j"
    assert run["software"]["openvino_precision"] == "bf16"
    assert [
        (c.get("net"), c.get("engine"), c.get("batch"), c["threads"], c["plot"])
        for c in run["configs"]
    ] == [
        ("resnet:8x128", None, 16, 8, True),
        ("resnet:8x128", None, 1, 8, False),
        (None, "edax", None, 8, True),
        (None, "edax", None, 1, False),
    ]
    assert run["configs"][0]["batch_strength"] == "measured_no_loss"


def test_busy_shares_by_group_and_the_note_they_make():
    before = {cpu: (0, 0) for cpu in range(24)}
    after = {cpu: (0, 100) for cpu in range(24)}
    after[4] = (80, 100)
    after[16] = (100, 100)
    shares = timing_benchmark.busy_shares(before, after)
    assert shares["contest"] == pytest.approx(100 / 800)
    assert shares["siblings"] == pytest.approx(80 / 800)
    assert shares["others"] == 0
    noisy = {"machine": [{"busy": shares}]}
    assert "siblings 10% busy" in (timing_benchmark.machine_note(noisy) or "")
    quiet = {"machine": [{"busy": {"siblings": 0.0, "others": 0.04, "contest": 1.0}}]}
    assert timing_benchmark.machine_note(quiet) is None


def test_a_short_noisy_game_counts_for_its_length():
    record = {
        "machine": [
            {"busy": {"siblings": 0.5, "others": 0.0}},
            {"busy": {"siblings": 0.0, "others": 0.0}},
        ]
    }
    assert timing_benchmark.machine_note(record, [0.01, 100.0]) is None
    assert timing_benchmark.machine_note(record, [100.0, 100.0]) == "siblings 25% busy"


def test_busy_cpus_names_only_those_over_the_limit():
    before = {0: (0, 0), 1: (0, 0)}
    after = {0: (1, 100), 1: (50, 100)}
    assert timing_benchmark.busy_cpus(before, after) == {1: 0.5}


def test_watch_is_recorded_per_game(tmp_path):
    store = modal_store.DirStore(tmp_path)
    plan = [timing_benchmark.net_config("4x64", 1, 1, 2)]
    timing_benchmark.run_configs(
        store, "j", 0, plan, [], fake_play, 1e9, watch=lambda: {"busy": {"others": 0.0}}
    )
    done = modal_store.get_json(store, "j/done/4x64-1-b1.json")
    assert len(done["machine"]) == 2
