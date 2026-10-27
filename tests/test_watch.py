import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

import watch


def test_pending_waits_for_training_to_end_then_takes_the_newest():
    assert watch.pending([0, 1, 2], set(), every=0, training=True) == []
    assert watch.pending([0, 1, 2], set(), every=0, training=False) == [2]
    assert watch.pending([0, 1, 2], {2}, every=0, training=False) == []
    assert watch.pending([], set(), every=0, training=False) == []


def test_pending_with_every_takes_unfinished_multiples_past_the_first():
    epochs = [0, 25, 50, 75, 98, 99]
    assert watch.pending(epochs, {25}, every=25, training=True) == [50, 75]
    assert watch.pending(epochs, {25}, every=25, training=False) == [50, 75, 99]
    # the newest is a multiple: it's taken once
    assert watch.pending([0, 25, 50], set(), every=25, training=False) == [25, 50]


def flag_values(cmd: list[str], flag: str) -> list[str]:
    """The values `cmd` passes `flag`, up to the next flag."""
    if flag not in cmd:
        return []
    values = []
    for value in cmd[cmd.index(flag) + 1 :]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def fake_vs_engine(calls: list[list[str]]):
    """A stand-in for `subprocess.run` of vs_engine.py that records its arguments and appends
    a result for each level it's asked to play, as vs_engine.py does."""

    def run(cmd, env, check):
        calls.append(cmd)
        [sims], [openings] = flag_values(cmd, "--sims"), flag_values(cmd, "--openings")
        skip = flag_values(cmd, "--skip-levels")
        with open(cmd[cmd.index("--results") + 1], "a") as f:
            for level in flag_values(cmd, "--levels"):
                if level not in skip:
                    record = {
                        "checkpoint": cmd[2],
                        "engine": "Edax",
                        "level": int(level),
                        "sims": int(sims),
                        "games": 2 * int(openings),
                        "score": 0.5,
                    }
                    f.write(json.dumps(record) + "\n")

    return run


def run_watch(monkeypatch, tmp_path, polls_training: int, *args: str):
    """Runs watch.py's main on `tmp_path` against a training process that saves a checkpoint
    each poll for `polls_training` polls, then exits; returns vs_engine.py's calls."""
    calls = []
    polls = []
    (tmp_path / "0.eqx").touch()

    def sleep(_):
        polls.append(None)
        (tmp_path / f"{len(polls)}.eqx").touch()

    monkeypatch.setattr(watch, "alive", lambda pid: len(polls) < polls_training)
    monkeypatch.setattr(watch.time, "sleep", sleep)
    monkeypatch.setattr(watch.subprocess, "run", fake_vs_engine(calls))
    monkeypatch.setattr(sys, "argv", ["watch.py", str(tmp_path), "--pid", "1", *args])
    watch.main()
    return calls


def test_watch_evaluates_only_the_last_checkpoint_after_training(monkeypatch, tmp_path):
    calls = run_watch(monkeypatch, tmp_path, 5)

    [cmd] = calls
    assert cmd[2] == str(tmp_path / "5.eqx")
    assert flag_values(cmd, "--sims") == ["1000"]
    assert flag_values(cmd, "--openings") == ["256"]
    assert flag_values(cmd, "--levels") == ["4", "6", "8"]


def test_watch_with_every_evaluates_multiples_as_saved_and_the_last(
    monkeypatch, tmp_path
):
    calls = run_watch(monkeypatch, tmp_path, 5, "--every", "2", "--sims", "64")

    assert [Path(cmd[2]).name for cmd in calls] == ["2.eqx", "4.eqx", "5.eqx"]
    assert all(flag_values(cmd, "--sims") == ["64"] for cmd in calls)


def test_watch_plays_only_the_missing_levels_of_the_last_checkpoint(
    monkeypatch, tmp_path
):
    (tmp_path / "3.eqx").touch()
    done = {
        "checkpoint": str(tmp_path / "3.eqx"),
        "engine": "edax",
        "level": 6,
        "sims": 1000,
        "games": 512,
        "score": 0.5,
    }
    (tmp_path / "vs_engine.jsonl").write_text(json.dumps(done) + "\n")
    # without --pid, as for a run that's over
    calls = []
    monkeypatch.setattr(watch.subprocess, "run", fake_vs_engine(calls))
    monkeypatch.setattr(sys, "argv", ["watch.py", str(tmp_path)])
    watch.main()

    [cmd] = calls
    assert cmd[2] == str(tmp_path / "3.eqx")
    assert flag_values(cmd, "--skip-levels") == ["6"]
