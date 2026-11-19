"""student_rack.py: the roster, each segment's train.py command, the ledger and graduation,
with train.py faked."""

import json
import subprocess
from pathlib import Path

import pytest
from example_modules import load_example

(rack_module,) = load_example("othello", "student_rack")

STUDENT = 'name = "{name}"\nnet = "resnet:{size}"\n'


def write_roster(rack: Path, students: list[dict], **settings) -> None:
    lines = [
        f"segment_epochs = {settings.get('segment_epochs', 10)}",
        "lr = 5e-4",
        "weight_decay = 0.5",
        f"keep_states = {settings.get('keep_states', 3)}",
        f"stop = {str(settings.get('stop', False)).lower()}",
        "",
        "[teacher]",
        'name = "teacher"',
        'net = "resnet:9x64"',
        'state = "/teacher/state-160.npz"',
        "",
        "[train]",
        f"args = {json.dumps(settings.get('args', ['--sims', '128']))}",
        "",
    ]
    for s in students:
        lines.append("[[students]]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in s.items()]
        lines.append("")
    (rack / "rack.toml").write_text("\n".join(lines))


def option(cmd: list[str], name: str) -> list[str]:
    """Every value of a train.py option in a command."""
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == name]


@pytest.fixture
def fake_train(monkeypatch):
    """Runs segments without train.py: each "run" saves its state, unless told to fail."""
    calls: list[list[str]] = []
    failing = {"next": False}

    def run(cmd, **kwargs):
        calls.append(cmd)
        if failing["next"]:
            failing["next"] = False
            return subprocess.CompletedProcess(cmd, 1)
        ckpt = Path(option(cmd, "--ckpt-dir")[0])
        ckpt.mkdir(parents=True, exist_ok=True)
        epoch = option(cmd, "--save-state-at")[0]
        (ckpt / f"state-{epoch}.npz").write_bytes(b"state")
        log = kwargs["stdout"]
        for e in range(int(epoch) - 2, int(epoch)):
            log.write(f"Epoch {e}: {{'a/avg4/engine_value_mse': '0.{e:02d}00'}}\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rack_module.subprocess, "run", run)
    return calls, failing


def test_segments_fork_the_teacher_then_branch_with_the_roster_of_the_day(
    tmp_path, fake_train
):
    calls, _ = fake_train
    write_roster(
        tmp_path,
        [
            {"name": "a", "net": "resnet:4x64"},
            {"name": "b", "net": "resnet:6x64", "lr": 5e-5, "weight_decay": 0.05},
        ],
    )

    rack_module.run_segment(tmp_path)

    first = calls[-1]
    assert option(first, "--net") == ["teacher=resnet:9x64"]
    assert option(first, "--grow") == ["a=resnet:4x64", "b=resnet:6x64"]
    assert {"--grow-frozen", "--grow-train-waiting", "--grow-manual"} <= set(first)
    assert option(first, "--init-from") == ["/teacher/state-160.npz"]
    assert option(first, "--branch-from") == []
    assert option(first, "--lr") == option(first, "--lr-final") == ["0.0005"]
    assert option(first, "--lr-of") == ["b=5e-05"]
    assert option(first, "--weight-decay-of") == ["b=0.05"]
    assert option(first, "--epochs") == option(first, "--save-state-at") == ["10"]
    assert option(first, "--sims") == ["128"]

    # b leaves, c joins: the next segment branches from the first one's state
    write_roster(
        tmp_path,
        [{"name": "a", "net": "resnet:4x64"}, {"name": "c", "net": "resnet:7x64"}],
    )
    rack_module.run_segment(tmp_path)

    second = calls[-1]
    assert option(second, "--branch-from") == [str(tmp_path / "ckpt" / "state-10.npz")]
    assert option(second, "--init-from") == []
    assert option(second, "--grow") == ["a=resnet:4x64", "c=resnet:7x64"]
    assert option(second, "--epochs") == ["20"]
    ledger = rack_module.read_ledger(tmp_path)
    assert [
        (s["start"], s["end"], s["added"], s["dropped"]) for s in ledger.segments
    ] == [
        (0, 10, ["a", "b"], []),
        (10, 20, ["c"], ["b"]),
    ]
    assert ledger.students == {
        "a": {"net": "resnet:4x64", "joined": 0},
        "b": {"net": "resnet:6x64", "joined": 0, "left": 10},
        "c": {"net": "resnet:7x64", "joined": 10},
    }


def test_a_name_is_never_reused_nor_a_students_net_changed(tmp_path, fake_train):
    write_roster(
        tmp_path,
        [{"name": "a", "net": "resnet:4x64"}, {"name": "b", "net": "resnet:6x64"}],
    )
    rack_module.run_segment(tmp_path)
    write_roster(tmp_path, [{"name": "a", "net": "resnet:4x64"}])
    rack_module.run_segment(tmp_path)

    write_roster(
        tmp_path,
        [{"name": "a", "net": "resnet:4x64"}, {"name": "b", "net": "resnet:6x64"}],
    )
    with pytest.raises(rack_module.RackError, match="trained in this rack before"):
        rack_module.run_segment(tmp_path)
    write_roster(tmp_path, [{"name": "a", "net": "resnet:5x64"}])
    with pytest.raises(rack_module.RackError, match="net can't change"):
        rack_module.run_segment(tmp_path)


@pytest.mark.parametrize(
    "students, settings, match",
    [
        ([], {}, "no students"),
        ([{"name": "teacher", "net": "resnet:4x64"}], {}, "distinct"),
        ([{"name": "a b", "net": "resnet:4x64"}], {}, "letters, digits"),
        ([{"name": "a", "net": "4x64"}], {}, "net is a spec"),
        ([{"name": "a", "net": "resnet:4x64", "lr": 0}], {}, "positive"),
        ([{"name": "a", "net": "resnet:4x64"}], {"args": ["--lr", "1e-3"]}, "--lr"),
        (
            [{"name": "a", "net": "resnet:4x64"}],
            {"args": ["--save-state-every=10"]},
            "--save-state-every",
        ),
    ],
)
def test_bad_rosters_are_refused(tmp_path, students, settings, match):
    write_roster(tmp_path, students, **settings)
    with pytest.raises(rack_module.RackError, match=match):
        rack_module.read_roster(tmp_path)


def test_a_failed_segment_leaves_the_ledger_and_runs_again(tmp_path, fake_train):
    calls, failing = fake_train
    write_roster(tmp_path, [{"name": "a", "net": "resnet:4x64"}])
    failing["next"] = True
    with pytest.raises(rack_module.RackError, match="segment 0 failed"):
        rack_module.run_segment(tmp_path)
    assert rack_module.read_ledger(tmp_path).segments == []

    rack_module.run_segment(tmp_path)

    assert option(calls[-1], "--init-from") == ["/teacher/state-160.npz"]
    assert len(rack_module.read_ledger(tmp_path).segments) == 1


def test_old_states_are_deleted_and_stop_ends_a_run(tmp_path, fake_train):
    write_roster(tmp_path, [{"name": "a", "net": "resnet:4x64"}], keep_states=2)
    for _ in range(3):
        assert rack_module.run_segment(tmp_path) is False

    assert sorted(p.name for p in (tmp_path / "ckpt").glob("state-*")) == [
        "state-20.npz",
        "state-30.npz",
    ]
    write_roster(tmp_path, [{"name": "a", "net": "resnet:4x64"}], stop=True)
    assert rack_module.run_segment(tmp_path) is True


def test_status_reads_the_segments_logs(tmp_path, fake_train, capsys):
    write_roster(tmp_path, [{"name": "a", "net": "resnet:4x64"}])
    rack_module.run_segment(tmp_path)
    rack_module.run_segment(tmp_path)

    rows = rack_module.epoch_metrics(tmp_path)
    assert sorted(rows) == [8, 9, 18, 19]
    rack_module.status(tmp_path, window=2, age=None)

    out = capsys.readouterr().out
    # the last two epochs' mean, 0.1800 and 0.1900
    assert "0.1850" in out and "training" in out


def test_graduate_forks_a_student_into_its_own_run(tmp_path, fake_train, capsys):
    write_roster(
        tmp_path,
        [{"name": "a", "net": "resnet:4x64", "lr": 5e-5, "weight_decay": 0.05}],
    )
    with pytest.raises(rack_module.RackError, match="no saved state"):
        rack_module.graduate(tmp_path, "a", ["--epochs", "5", "--ckpt-dir", "x"], True)
    rack_module.run_segment(tmp_path)
    with pytest.raises(rack_module.RackError, match="needs --ckpt-dir"):
        rack_module.graduate(tmp_path, "a", ["--epochs", "5"], True)
    with pytest.raises(rack_module.RackError, match="isn't in the newest"):
        rack_module.graduate(tmp_path, "z", ["--epochs", "5", "--ckpt-dir", "x"], True)

    rack_module.graduate(tmp_path, "a", ["--epochs", "5", "--ckpt-dir", "solo"], True)

    cmd = capsys.readouterr().out.strip().splitlines()[-1].split()
    assert option(cmd, "--net") == ["resnet:4x64"]
    assert option(cmd, "--init-from") == [str(tmp_path / "ckpt" / "state-10.npz")]
    assert option(cmd, "--fork-into") == ["a"]
    assert option(cmd, "--lr") == option(cmd, "--lr-final") == ["5e-05"]
    assert option(cmd, "--weight-decay") == ["0.05"]
    assert option(cmd, "--grow") == []
    assert option(cmd, "--ckpt-dir") == ["solo"]


def test_init_writes_a_roster_with_two_controls(tmp_path):
    parser_args = type(
        "Args",
        (),
        {
            "teacher_state": rack_module.TEACHER_STATE,
            "teacher_net": rack_module.TEACHER_NET,
            "segment_epochs": 20,
            "lr": 5e-4,
            "weight_decay": 0.5,
        },
    )
    rack_module.init(tmp_path, parser_args)

    roster = rack_module.read_roster(tmp_path)
    assert roster.teacher_net == rack_module.TEACHER_NET
    assert roster.train_args == rack_module.TRAIN_ARGS
    assert option(roster.train_args, "--sims") == ["128"]
    assert [(s.name, s.net) for s in roster.students] == [
        ("ray7x64", rack_module.TEACHER_NET.replace("9x64", "7x64")),
        ("ray9x64", rack_module.TEACHER_NET),
    ]
    assert roster.taskset == "0-15"
    with pytest.raises(rack_module.RackError, match="exists"):
        rack_module.init(tmp_path, parser_args)


def test_checkpoints_lists_each_students_newest_average(tmp_path, fake_train, capsys):
    write_roster(
        tmp_path,
        [{"name": "a", "net": "resnet:4x64"}],
        args=["--selfplay-average", "4"],
    )
    rack_module.run_segment(tmp_path)
    for epoch in (3, 9):
        path = tmp_path / "ckpt" / "a" / "avg4" / f"{epoch}.eqx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"net")
    capsys.readouterr()

    rack_module.checkpoints(tmp_path, every=False)

    assert capsys.readouterr().out.split() == [
        "a",
        "9",
        "resnet:4x64",
        str(tmp_path / "ckpt" / "a" / "avg4" / "9.eqx"),
    ]
    assert rack_module.read_ledger(tmp_path).segments[0]["train_args"] == [
        "--selfplay-average",
        "4",
    ]
