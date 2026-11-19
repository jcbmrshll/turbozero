"""A rack of student networks trained beside a fixed teacher, in short segments.

The teacher (a trained network, forked from its saved state) plays self-play and never trains;
every student trains on its games, each epoch, from the same replay buffer (train.py's --grow
with --grow-frozen and --grow-train-waiting). The run goes in segments of a few epochs, each a
branch of the last one's saved state. Between segments the roster, `rack.toml` in the rack's
directory, can change: a student left out of it is dropped, a new one starts from scratch, and
every student kept goes on exactly where it was (weights, optimizer, weight averages). A
student's learning rate and weight decay can change too. So candidate architectures are
compared on the same teacher data, the hopeless ones replaced as soon as they show it, and a
promising one graduates: `graduate` forks it into a run of its own self-play.

    uv run examples/othello/student_rack.py init runs/rack
    $EDITOR runs/rack/rack.toml                   # add students to the two controls
    uv run examples/othello/student_rack.py run runs/rack       # segment after segment
    uv run examples/othello/student_rack.py status runs/rack
    uv run examples/othello/student_rack.py graduate runs/rack r7x64 --epochs 200 \\
        --ckpt-dir runs/r7x64-solo

`run` reads the roster before each segment, so edits take effect at the next segment boundary;
`stop = true` in the roster ends it after the segment in progress. The rack's directory holds
the roster, `ledger.json` (the segments done and who trained in each, written by this script),
`ckpt/` (every network's checkpoints, `ckpt/<name>/`, and the segments' saved states) and
`logs/` (each segment's train.py output).
"""

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "examples" / "othello" / "train.py"
NAME = re.compile(r"[A-Za-z0-9_.+-]+")

# the best network as of 2026-11-18: the ray-filter 9x64 at run 860, its weight average over 4
# epochs (its saved state's), and the self-play recipe it trained with
TEACHER_STATE = (
    "~/.cache/turbozero/othello/smallnet-data/runs/"
    "ray9x64-solve20ppend-r701-lr2.5e-5-wd1.25e-2/state-160.npz"
)
TEACHER_NET = (
    "resnet:9x64,pool_every=2,heads=pooled,pool_channels=16,layout=pre,block=convnext,"
    "expand=2,dw_taps=rays,stem_norm=false,expand_norm=false,project_norm=false"
)
TRAIN_ARGS = [
    "--train-batch",
    "4096",
    "--train-steps",
    "256",
    "--buffer",
    "1125",
    "--sampling-moves",
    "20",
    "--dirichlet-alpha",
    "0.3",
    "--dirichlet-epsilon",
    "0.25",
    "--tree-factor",
    "4",
    "--standard-starts",
    "0",
    "--average-epochs",
    "4",
    "--selfplay-average",
    "4",
    "--optimizer",
    "adamw",
    "--sims",
    "128",
    "--seed",
    "0",
    "--keep-every",
    "20",
    "--engine-eval-every",
    "1",
    "--solve-empties",
    "20",
    "--solve-play-past",
    "end",
    "--solve-threads",
    "15",
    "--monitor",
]

TEMPLATE = """\
# A student rack (see examples/othello/student_rack.py). Edit between segments: `run` reads this
# before each one. A student left out is dropped at the next segment (its checkpoints stay in
# ckpt/<name>/); a new one starts from scratch. Names are never reused.

# epochs a segment trains, each ending in a saved state (ckpt/state-<epoch>.npz)
segment_epochs = {segment_epochs}
# the students' learning rate and weight decay, unless a student sets its own (drop both
# together to hold a student's weight norm)
lr = {lr}
weight_decay = {weight_decay}
# saved states kept (the newest), older ones are deleted; each is about 1.2 GB with the buffer
keep_states = 3
# true: `run` stops after the segment in progress
stop = false
# (optional) cores train.py runs on, for taskset, e.g. "0-15"
taskset = "0-15"

[teacher]
name = "teacher"
net = "{teacher_net}"
state = "{teacher_state}"

[train]
# train.py options of every segment (self-play, buffer, testers); the rack sets the networks,
# learning rates, epochs, checkpoints and how each segment starts
args = {train_args}

[env]
XLA_PYTHON_CLIENT_PREALLOCATE = "false"

# one table per student. Each may set its own lr and weight_decay, e.g. lr = 5e-5 and
# weight_decay = 0.05. The two below are controls: in self-play the 9x64 overtook the 7x64 at
# 30 s against engines only after long training, so a rack that ranks candidates by capacity
# should reproduce that.
[[students]]
name = "ray7x64"
net = "{ray7x64}"

[[students]]
name = "ray9x64"
net = "{ray9x64}"
"""

# train.py options the rack sets itself, so the roster's [train] args can't
RESERVED = {
    "--net",
    "--grow",
    "--grow-frozen",
    "--grow-train-waiting",
    "--grow-train-retired",
    "--grow-manual",
    "--grow-lr",
    "--grow-switch-at",
    "--lr-of",
    "--weight-decay-of",
    "--lr",
    "--lr-final",
    "--lr-drops",
    "--lr-plateau",
    "--weight-decay",
    "--epochs",
    "--ckpt-dir",
    "--init-from",
    "--fork-into",
    "--warm-start",
    "--branch-from",
    "--resume",
    "--take-over",
    "--save-state-at",
    "--save-state-every",
    "--name",
}


@dataclass(frozen=True)
class Student:
    name: str
    net: str
    lr: float | None = None
    weight_decay: float | None = None


@dataclass(frozen=True)
class Roster:
    """`rack.toml`, checked."""

    segment_epochs: int
    lr: float
    weight_decay: float
    keep_states: int
    stop: bool
    taskset: str | None
    teacher_name: str
    teacher_net: str
    teacher_state: str
    train_args: list[str]
    env: dict[str, str]
    students: list[Student]

    def student_lr(self, student: Student) -> float:
        return self.lr if student.lr is None else student.lr

    def student_weight_decay(self, student: Student) -> float:
        return (
            self.weight_decay if student.weight_decay is None else student.weight_decay
        )


class RackError(Exception):
    pass


def read_roster(rack: Path) -> Roster:
    path = rack / "rack.toml"
    if not path.exists():
        raise RackError(f"no roster at {path}: start a rack with `init`")
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    teacher = raw.get("teacher", {})
    students = [
        Student(
            name=s["name"],
            net=s["net"],
            lr=float(s["lr"]) if "lr" in s else None,
            weight_decay=float(s["weight_decay"]) if "weight_decay" in s else None,
        )
        for s in raw.get("students", [])
    ]
    roster = Roster(
        segment_epochs=int(raw.get("segment_epochs", 20)),
        lr=float(raw["lr"]),
        weight_decay=float(raw["weight_decay"]),
        keep_states=int(raw.get("keep_states", 3)),
        stop=bool(raw.get("stop", False)),
        taskset=raw.get("taskset"),
        teacher_name=teacher.get("name", "teacher"),
        teacher_net=teacher["net"],
        teacher_state=os.path.expanduser(teacher["state"]),
        train_args=[str(a) for a in raw.get("train", {}).get("args", [])],
        env={k: str(v) for k, v in raw.get("env", {}).items()},
        students=students,
    )
    check_roster(roster)
    return roster


def check_roster(roster: Roster) -> None:
    names = [roster.teacher_name, *(s.name for s in roster.students)]
    bad = [n for n in names if not NAME.fullmatch(n)]
    if bad:
        raise RackError(f"names are letters, digits and _.+- only: {bad}")
    if len(set(names)) != len(names):
        raise RackError(f"the teacher's and students' names must be distinct: {names}")
    if not roster.students:
        raise RackError("the roster has no students")
    if roster.segment_epochs < 1 or roster.keep_states < 1:
        raise RackError("segment_epochs and keep_states are at least 1")
    if roster.lr <= 0 or any(s.lr is not None and s.lr <= 0 for s in roster.students):
        raise RackError("learning rates are positive")
    for s in [*roster.students]:
        if not re.fullmatch(r"[a-z]+:.+", s.net):
            raise RackError(f"{s.name}: net is a spec, KIND:..., e.g. resnet:7x64,...")
    reserved = sorted(
        {a.split("=")[0] for a in roster.train_args if a.split("=")[0] in RESERVED}
    )
    if reserved:
        raise RackError(f"[train] args can't set what the rack sets: {reserved}")


@dataclass
class Ledger:
    """`ledger.json`: the segments done, and every student that ever joined."""

    segments: list[dict]
    students: dict[str, dict]

    @property
    def epoch(self) -> int:
        return self.segments[-1]["end"] if self.segments else 0

    @property
    def state(self) -> str | None:
        return self.segments[-1]["state"] if self.segments else None


def read_ledger(rack: Path) -> Ledger:
    path = rack / "ledger.json"
    if not path.exists():
        return Ledger([], {})
    raw = json.loads(path.read_text())
    return Ledger(raw["segments"], raw["students"])


def write_ledger(rack: Path, ledger: Ledger) -> None:
    path = rack / "ledger.json"
    path.with_suffix(".tmp").write_text(
        json.dumps({"segments": ledger.segments, "students": ledger.students}, indent=1)
    )
    os.replace(path.with_suffix(".tmp"), path)


def segment_plan(rack: Path, roster: Roster, ledger: Ledger) -> tuple[list[str], dict]:
    """The train.py arguments of the next segment, and its ledger entry (without its state)."""
    current = {s.name for s in roster.students}
    previous = set(ledger.segments[-1]["students"]) if ledger.segments else set()
    # a name that left the rack is never reused: its checkpoints and metrics are its own
    reused = sorted(
        n
        for n in current - previous
        if n in ledger.students or n == roster.teacher_name
    )
    if reused:
        raise RackError(
            f"{reused} trained in this rack before: give a new student a new name"
        )
    changed = sorted(
        s.name
        for s in roster.students
        if s.name in ledger.students and ledger.students[s.name]["net"] != s.net
    )
    if changed:
        raise RackError(
            f"{changed}: a student's net can't change; add it under a new name"
        )
    start = ledger.epoch
    end = start + roster.segment_epochs
    index = len(ledger.segments)
    args = [
        "--net",
        f"{roster.teacher_name}={roster.teacher_net}",
        *(a for s in roster.students for a in ("--grow", f"{s.name}={s.net}")),
        "--grow-manual",
        "--grow-frozen",
        "--grow-train-waiting",
        "--lr",
        f"{roster.lr:g}",
        "--lr-final",
        f"{roster.lr:g}",
        "--weight-decay",
        f"{roster.weight_decay:g}",
        *(
            a
            for s in roster.students
            if s.lr is not None
            for a in ("--lr-of", f"{s.name}={s.lr:g}")
        ),
        *(
            a
            for s in roster.students
            if s.weight_decay is not None
            for a in ("--weight-decay-of", f"{s.name}={s.weight_decay:g}")
        ),
        *roster.train_args,
        "--epochs",
        str(end),
        "--save-state-at",
        str(end),
        "--ckpt-dir",
        str(rack / "ckpt"),
        "--name",
        f"{rack.name}-s{index:03d}",
    ]
    if ledger.state is None:
        # the first segment forks the teacher, with its replay buffer and weight averages
        args += ["--init-from", roster.teacher_state]
    else:
        args += ["--branch-from", ledger.state]
    entry = {
        "segment": index,
        "start": start,
        "end": end,
        "students": [s.name for s in roster.students],
        "added": sorted(current - previous),
        "dropped": sorted(previous - current),
        "lr": {s.name: roster.student_lr(s) for s in roster.students},
        "weight_decay": {
            s.name: roster.student_weight_decay(s) for s in roster.students
        },
        "train_args": roster.train_args,
    }
    return args, entry


def command(roster: Roster, args: list[str]) -> list[str]:
    pin = ["taskset", "-c", roster.taskset] if roster.taskset else []
    return [*pin, sys.executable, str(TRAIN), *args]


def run_segment(rack: Path, dry_run: bool = False) -> bool:
    """Runs the next segment; returns whether the roster asks to stop after it."""
    roster = read_roster(rack)
    ledger = read_ledger(rack)
    args, entry = segment_plan(rack, roster, ledger)
    cmd = command(roster, args)
    state = rack / "ckpt" / f"state-{entry['end']}.npz"
    print(
        f"segment {entry['segment']}: epochs {entry['start']}-{entry['end']}, students "
        f"{', '.join(entry['students'])}"
        + (f"; added {', '.join(entry['added'])}" if entry["added"] else "")
        + (f"; dropped {', '.join(entry['dropped'])}" if entry["dropped"] else ""),
        flush=True,
    )
    if dry_run:
        print(shlex.join(cmd))
        return roster.stop
    log = rack / "logs" / f"seg-{entry['segment']:03d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as out:
        out.write(f"$ {shlex.join(cmd)}\n")
        out.flush()
        code = subprocess.run(
            cmd,
            check=False,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1", **roster.env},
            stdout=out,
            stderr=subprocess.STDOUT,
        ).returncode
    if code != 0 or not state.exists():
        raise RackError(
            f"segment {entry['segment']} failed (exit {code}): see {log}; running it again "
            "starts it over from the last segment's state"
        )
    entry["state"] = str(state)
    ledger.segments.append(entry)
    nets = {s.name: s.net for s in roster.students}
    for name in entry["added"]:
        ledger.students[name] = {"net": nets[name], "joined": entry["start"]}
    for name in entry["dropped"]:
        ledger.students[name]["left"] = entry["start"]
    write_ledger(rack, ledger)
    prune_states(rack, ledger, roster.keep_states)
    print(f"segment {entry['segment']} done: {state}", flush=True)
    # the roster as it is now: an edit during the segment may ask to stop
    return read_roster(rack).stop


def prune_states(rack: Path, ledger: Ledger, keep: int) -> None:
    """Deletes the segments' saved states but the newest `keep`."""
    for entry in ledger.segments[:-keep]:
        path = Path(entry["state"])
        if path.exists():
            path.unlink()
            print(f"deleted {path} (keep_states {keep})", flush=True)


def epoch_metrics(rack: Path) -> dict[int, dict[str, float]]:
    """Every epoch's metrics from the segments' logs ("Epoch N: {...}" lines), the later
    segment's where a rerun segment logged an epoch twice."""
    rows: dict[int, dict[str, float]] = {}
    for log in sorted((rack / "logs").glob("seg-*.log")):
        for line in log.read_text().splitlines():
            found = re.match(r"^Epoch (\d+): (\{.*\})\s*$", line)
            if not found:
                continue
            row = rows.setdefault(int(found.group(1)), {})
            for k, v in ast.literal_eval(found.group(2)).items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


STATUS_METRICS = [
    ("avg4/engine_value_mse", "value err"),
    ("avg4/engine_regret", "regret"),
    ("loss", "loss"),
]


def status(rack: Path, window: int, age: int | None) -> None:
    """Prints each student's metrics: means over its last `window` epochs, or with `age`, over
    the `window` epochs before its own epoch `age` (equal training, for students that joined
    at different times)."""
    roster = read_roster(rack)
    ledger = read_ledger(rack)
    rows = epoch_metrics(rack)
    in_roster = {s.name for s in roster.students}
    last = ledger.segments[-1] if ledger.segments else {"students": [], "lr": {}}
    print(
        f"{rack}: {len(ledger.segments)} segments, {ledger.epoch} epochs, teacher "
        f"{roster.teacher_name}"
    )
    where = (
        f"epochs {age - window}-{age - 1} of its own"
        if age
        else f"its last {window} epochs"
    )
    print(f"means over {where} (lower is better)\n")
    print(
        f"{'student':<16}{'joined':>7}{'epochs':>7}{'lr':>9}"
        + "".join(f"{label:>11}" for _, label in STATUS_METRICS)
        + "  state"
    )
    names = [*ledger.students, *(n for n in in_roster if n not in ledger.students)]
    for name in names:
        info = ledger.students.get(name)
        joined = info["joined"] if info else ledger.epoch
        epochs = sorted(
            e for e, row in rows.items() if f"{name}/{STATUS_METRICS[0][0]}" in row
        )
        if age is not None:
            epochs = [e for e in epochs if age - window <= e - joined < age]
        else:
            epochs = epochs[-window:]
        cells = ""
        for metric, _ in STATUS_METRICS:
            values = [
                rows[e][f"{name}/{metric}"]
                for e in epochs
                if f"{name}/{metric}" in rows[e]
            ]
            cells += f"{np.mean(values):>11.4f}" if values else f"{'-':>11}"
        if name in last["students"]:
            state = "training" if name in in_roster else "drops next"
        elif name in in_roster:
            state = "joins next"
        else:
            state = f"dropped at {info['left'] if info else '?'}"
        trained = (info.get("left", ledger.epoch) if info else joined) - joined
        lr = f"{last['lr'][name]:>9.2g}" if name in last["lr"] else f"{'-':>9}"
        print(f"{name:<16}{joined:>7}{trained:>7}{lr}{cells}  {state}")


def checkpoints(rack: Path, every: bool) -> None:
    """Prints each student's newest weight-average checkpoint (with `every`, all of them), as
    NAME EPOCH SPEC PATH lines, e.g. for evaluating a promising student further."""
    roster = read_roster(rack)
    ledger = read_ledger(rack)
    horizon = next(
        (
            roster.train_args[i + 1]
            for i, a in enumerate(roster.train_args)
            if a == "--selfplay-average"
        ),
        None,
    )
    for name, info in ledger.students.items():
        directory = rack / "ckpt" / name
        if horizon is not None:
            directory = directory / f"avg{horizon}"
        epochs = sorted(
            int(p.stem) for p in directory.glob("*.eqx") if p.stem.isdigit()
        )
        for epoch in epochs if every else epochs[-1:]:
            print(f"{name} {epoch} {info['net']} {directory / f'{epoch}.eqx'}")


def graduate(rack: Path, name: str, extra: list[str], dry_run: bool) -> None:
    """Forks a student into a run of its own self-play, from the rack's newest saved state:
    its weights, optimizer state, weight averages, and the replay buffer of the teacher's games
    (which its own games replace), at its learning rate and weight decay in the rack."""
    roster = read_roster(rack)
    ledger = read_ledger(rack)
    if ledger.state is None or not Path(ledger.state).exists():
        raise RackError(f"{rack} has no saved state to graduate from")
    last = ledger.segments[-1]
    if name not in last["students"]:
        raise RackError(
            f"{name} isn't in the newest saved state (it has {', '.join(last['students'])})"
        )
    given = {a.split("=")[0] for a in extra}
    missing = {"--ckpt-dir", "--epochs"} - given
    if missing:
        raise RackError(f"graduate needs {' and '.join(sorted(missing))} (train.py's)")
    lr = last["lr"][name]
    cmd = command(
        roster,
        [
            "--net",
            ledger.students[name]["net"],
            "--init-from",
            ledger.state,
            "--fork-into",
            name,
            "--lr",
            f"{lr:g}",
            "--lr-final",
            f"{lr:g}",
            "--weight-decay",
            f"{last['weight_decay'][name]:g}",
            *roster.train_args,
            *([] if "--name" in given else ["--name", f"{rack.name}-{name}-solo"]),
            *extra,
        ],
    )
    if dry_run:
        print(shlex.join(cmd))
        return
    os.execvpe(cmd[0], cmd, {**os.environ, "PYTHONUNBUFFERED": "1", **roster.env})


def init(rack: Path, args: argparse.Namespace) -> None:
    path = rack / "rack.toml"
    if path.exists():
        raise RackError(f"{path} exists")
    rack.mkdir(parents=True, exist_ok=True)
    train_args = "[\n" + "".join(f"    {json.dumps(a)},\n" for a in TRAIN_ARGS) + "]"
    path.write_text(
        TEMPLATE.format(
            segment_epochs=args.segment_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            teacher_net=args.teacher_net,
            teacher_state=args.teacher_state,
            train_args=train_args,
            ray7x64=TEACHER_NET.replace("9x64", "7x64"),
            ray9x64=TEACHER_NET,
        )
    )
    print(f"wrote {path}: edit its students, then `run`")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write a roster to start a rack from")
    p.add_argument("rack", type=Path)
    p.add_argument("--teacher-state", default=TEACHER_STATE)
    p.add_argument("--teacher-net", default=TEACHER_NET)
    p.add_argument("--segment-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.5)
    p = sub.add_parser("segment", help="train the next segment")
    p.add_argument("rack", type=Path)
    p.add_argument("--dry-run", action="store_true", help="print its train.py command")
    p = sub.add_parser(
        "run", help="train segment after segment, reading the roster each time"
    )
    p.add_argument("rack", type=Path)
    p.add_argument("--segments", type=int, default=None, help="at most this many")
    p = sub.add_parser("status", help="the students' metrics")
    p.add_argument("rack", type=Path)
    p.add_argument("--window", type=int, default=5, help="epochs averaged")
    p.add_argument(
        "--age",
        type=int,
        default=None,
        help="compare at equal training: each student's epochs up to its own epoch AGE",
    )
    p = sub.add_parser("checkpoints", help="each student's newest checkpoint and spec")
    p.add_argument("rack", type=Path)
    p.add_argument("--every", action="store_true", help="every kept checkpoint")
    p = sub.add_parser(
        "graduate",
        help="fork a student into a run of its own self-play (extra train.py args after the "
        "name, e.g. --epochs 200 --ckpt-dir runs/solo)",
    )
    p.add_argument("rack", type=Path)
    p.add_argument("name")
    p.add_argument("--dry-run", action="store_true", help="print its train.py command")
    args, extra = parser.parse_known_args()
    if extra and args.cmd != "graduate":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    rack = args.rack.resolve()
    try:
        if args.cmd == "init":
            init(rack, args)
        elif args.cmd == "segment":
            run_segment(rack, args.dry_run)
        elif args.cmd == "run":
            if read_roster(rack).stop:
                parser.error("the roster says stop = true: set it to false to run")
            done = 0
            while args.segments is None or done < args.segments:
                stop = run_segment(rack)
                done += 1
                if stop:
                    print("the roster says stop", flush=True)
                    break
        elif args.cmd == "status":
            status(rack, args.window, args.age)
        elif args.cmd == "checkpoints":
            checkpoints(rack, args.every)
        else:
            graduate(rack, args.name, extra, args.dry_run)
    except RackError as e:
        sys.exit(f"student_rack: {e}")


if __name__ == "__main__":
    main()
