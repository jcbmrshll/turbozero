"""Drives one long training run on Modal as a mainline of segments, with forks that try the next
replay buffer window or sims count and take over the mainline when they train a better network.

    uv run examples/othello/fork_driver.py init RUN_DIR          # writes RUN_DIR/driver.toml
    $EDITOR RUN_DIR/driver.toml
    uv run examples/othello/fork_driver.py run RUN_DIR           # in tmux: every few minutes
    uv run examples/othello/fork_driver.py status RUN_DIR

The run is train.py's (see README.md), usually a growth run (`--grow`): its networks each follow
a staircase of learning rates, with weight decay a fixed multiple of the rate, stepping down when
the network's held-out value error (`--grow-metric`'s, e.g. avg4/engine_value_mse on the
engine-labelled set) stops improving. It runs in segments of `segment_epochs`, each a step of the
line's Modal job that branches from the last one's final state (`--branch-from`), so settings
change only between segments and the job never leaves its machine (`modal_jobs.py extend`,
`Job.follow_minutes`). The first segment may start from a trained network's saved state
(`start_from`, `--init-from`), with its replay buffer.

Growth: the network playing and the next train (those that played before don't). train.py only
reports when the next has caught up (`--grow-manual`, `growth_caught_up`, by `--grow-metric`'s
rule); the driver takes it over at the mainline's next segment end, and every fork takes it over
at that same epoch (`--take-over`), as it takes the mainline's learning rate steps, so a fork and
the mainline always play the same network. Comparisons start again after each take-over.

Forks try the next step of either ladder, at most one of each at a time: the replay buffer's
sampling window
(`--buffer-window`, the buffer itself kept at its largest, so a wider window has its positions at
once) and the search's sims (with playout cap randomization's fast searches in proportion). A
fork branches from the mainline's newest segment end with one setting a step up, runs beside it
on a Modal job of its own, follows the mainline's learning rates epoch for epoch, and is compared
with it at matched epochs on the playing network's metric:

- it takes over (adopted: the mainline stops, the fork goes on as the mainline) once its mean
  over each of the last `windows_ahead` windows of `window_epochs` is at least `margin` below
  the mainline's, after `min_epochs`. A window fork needs a learning rate step inside it, and
  `after_lr_step` epochs after it, first (or the learning rates at their floor): a wider window's
  gain hides at a constant high rate and shows after it drops;
- it is killed once its own metric has plateaued while its mean lead over those windows is
  under half the margin, or at `max_fork_epochs`; its step may be tried again `retry_after`
  mainline epochs later;
- when the other fork takes over, it starts again from the new mainline (it branched from the
  old one, so it would differ from the new one in two settings).

Every decision goes to RUN_DIR/decisions.jsonl, the driver's state to RUN_DIR/driver.json. The
final checkpoints are decayed offline. Metrics come from the segments' logs, which the Modal sync
brings here.
"""

import argparse
import ast
import json
import re
import shlex
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import modal_jobs
from modal_spec import Job, Step, job_key, parse_step
from modal_store import get_json, put_json

KINDS = ("window", "sims")
EPOCH_LINE = re.compile(r"Epoch (\d+): (\{.*\})\s*$")


@dataclass(frozen=True)
class Rules:
    """When forks start, take over or die, and when learning rates step (see the module)."""

    fork_from_epoch: int = 100
    window_epochs: int = 20
    margin: float = 0.003
    windows_ahead: int = 2
    min_epochs: int = 60
    after_lr_step: int = 40
    max_fork_epochs: int = 300
    retry_after: int = 200
    # a plateau: the least-squares slope of the metric over the last `plateau_epochs` (at the
    # current learning rate) improves it by less than `plateau_per_100` per 100 epochs
    plateau_epochs: int = 60
    plateau_per_100: float = 0.001
    lr_min_epochs: int = 100


@dataclass(frozen=True)
class Config:
    """RUN_DIR/driver.toml."""

    name: str
    # (name, spec) in growth order: the first plays first (train.py's --net, then --grow)
    nets: list[tuple[str, str]]
    # train.py's other options, the same in every segment (not those the driver sets)
    train_args: list[str]
    tags: str
    commit: str
    buffer: int = 9000
    windows: list[int] = field(default_factory=lambda: [1125, 2250, 4500, 9000])
    sims: list[tuple[int, int]] = field(
        default_factory=lambda: [(128, 32), (256, 64), (512, 128)]
    )
    full_share: float = 0.25
    lrs: list[float] = field(default_factory=lambda: [1e-3, 5e-4, 1e-4])
    wd_per_lr: float = 500.0
    segment_epochs: int = 50
    metric: str = "avg4/engine_value_mse"
    # the next segment is added this many epochs before the last one ends
    lead_epochs: int = 3
    gpu: str = "L40S"
    # cores asked for. The exact solver (--solve-threads 31) runs in bursts of about 20 s
    # beside 30-50 s of self-play, and the host is nearly idle otherwise. Modal bills the
    # request or the use, whichever is higher, and lets a container use 16 cores past its
    # request, so the bursts still get 32 cores.
    cpu: float = 16.0
    # GiB of host memory; None sizes it by the buffer (see job_memory_gb)
    memory_gb: int | None = None
    # a state saved to continue from at each segment's end only: with a 9000-position buffer
    # each is about 10 GB, and Modal bills uploads
    resume_every: int = 50
    follow_minutes: float = 30.0
    hours_per_line: float = 100.0
    # (optional) a saved state the first network starts from, with its replay buffer (see
    # train.py's --init-from), and where on the ladders it is: each network's learning rate
    # (an index into lrs, by name; the others start at the first), the window and the sims
    start_from: str | None = None
    start_lrs: dict[str, int] = field(default_factory=dict)
    start_window: int = 0
    start_sims: int = 0
    rules: Rules = field(default_factory=Rules)

    def job_memory_gb(self) -> int:
        """Host memory a line's job asks for: memory_gb if set, otherwise enough for the
        buffer. Saving a state is the peak: a scan 9x64 with a 2250-position buffer held
        about 12 GB and peaked at 15.4 GB in a save, and states grow with the buffer (about
        5 GB at 2250, 12-19 GB at 9000), so 32, 48 and 96 GB for 2250, 4500 and 9000."""
        if self.memory_gb is not None:
            return self.memory_gb
        if self.buffer <= 2250:
            return 32
        if self.buffer <= 4500:
            return 48
        return 96

    @staticmethod
    def load(path: Path) -> "Config":
        data: dict[str, Any] = tomllib.loads(path.read_text())
        data["rules"] = Rules(**data.pop("rules", {}))
        data["nets"] = [(net["name"], net["spec"]) for net in data.pop("nets")]
        if "sims" in data:
            data["sims"] = [(int(a), int(b)) for a, b in data["sims"]]
        config = Config(**data)
        config.check()
        return config

    def check(self) -> None:
        reserved = {
            "--net",
            "--grow",
            "--lr",
            "--lr-final",
            "--lr-drops",
            "--lr-plateau",
            "--weight-decay",
            "--lr-of",
            "--weight-decay-of",
            "--sims",
            "--playout-cap",
            "--buffer",
            "--buffer-window",
            "--epochs",
            "--save-state-at",
            "--save-state-every",
            "--branch-from",
            "--init-from",
            "--resume",
            "--take-over",
            "--grow-manual",
            "--grow-switch-at",
            "--grow-train-retired",
            "--ckpt-dir",
            "--name",
            "--notes",
        }
        taken = sorted(a for a in self.train_args if a.split("=")[0] in reserved)
        if taken:
            raise ValueError(
                f"the driver sets {', '.join(taken)}: leave them out of train_args"
            )
        if not self.nets:
            raise ValueError("no nets")
        if max(self.windows) > self.buffer:
            raise ValueError(f"windows go up to the buffer ({self.buffer})")
        if sorted(self.lrs, reverse=True) != self.lrs:
            raise ValueError("lrs step down")
        names = [name for name, _ in self.nets]
        unknown = sorted(set(self.start_lrs) - set(names))
        if unknown or any(not 0 <= i < len(self.lrs) for i in self.start_lrs.values()):
            raise ValueError(f"start_lrs: indices into lrs, of nets {names}")
        if not 0 <= self.start_window < len(self.windows) or not (
            0 <= self.start_sims < len(self.sims)
        ):
            raise ValueError("start_window and start_sims are indices into the ladders")

    @property
    def growth(self) -> bool:
        return len(self.nets) > 1

    def key(self, net: str | None) -> str:
        """The metric's name in the logs: the playing network's (None), or one network's."""
        if net is None or not self.growth:
            return self.metric
        return f"{net}/{self.metric}"


@dataclass
class Segment:
    start: int
    end: int
    ckpt_dir: str
    log: str
    # the settings it ran with: indices into the config's ladders
    window: int
    sims: int
    lrs: dict[str, int]
    # the next network plays from its first epoch (train.py's --take-over)
    take_over: bool = False


@dataclass
class Line:
    """The mainline, or a fork of it: one Modal job of segments."""

    name: str
    kind: str  # "main" (from scratch, or an adopted fork's), "window" or "sims"
    job: str
    start: int
    segments: list[Segment] = field(default_factory=list)
    status: str = (
        "running"  # running, adopted (now the mainline), killed, replaced, done
    )
    parent: str | None = None
    ended_at: float | None = None

    @property
    def end(self) -> int:
        return self.segments[-1].end


@dataclass
class State:
    """RUN_DIR/driver.json: the lines, which is the mainline, its learning rate steps."""

    lines: dict[str, Line] = field(default_factory=dict)
    main: str = ""
    # the mainline's learning rate steps: (epoch, net, index into lrs), in order
    lr_steps: list[tuple[int, str, int]] = field(default_factory=list)
    # the epochs from which each next network plays, in order: every line takes them over
    # at the same epochs, as it takes the learning rate steps
    takeovers: list[int] = field(default_factory=list)
    # forks started, by kind, and when a killed kind may try again (mainline epoch)
    started: list[str] = field(default_factory=list)
    retry_at: dict[str, int] = field(default_factory=dict)
    done: bool = False

    @staticmethod
    def load(path: Path) -> "State":
        if not path.exists():
            return State()
        data = json.loads(path.read_text())
        lines = {
            name: Line(
                **{
                    **line,
                    "segments": [Segment(**s) for s in line["segments"]],
                }
            )
            for name, line in data.pop("lines").items()
        }
        data["lr_steps"] = [tuple(step) for step in data["lr_steps"]]
        return State(lines=lines, **data)

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1))
        tmp.replace(path)


class Backend(Protocol):
    """What the driver does to Modal (see `ModalBackend`), faked in tests."""

    def launch(self, job: str, steps: list[str]) -> None: ...
    def extend(self, job: str, steps: list[str]) -> None: ...
    def stop(self, job: str) -> None: ...
    def state(self, job: str) -> str: ...
    def resume(self, job: str) -> None: ...
    def has_state(self, path: str) -> bool: ...


def read_log(path: str | Path) -> dict[int, dict[str, float]]:
    """A train.py log's metrics by epoch; an epoch logged twice (a resumed attempt) takes the
    later lines."""
    out: dict[int, dict[str, float]] = {}
    path = Path(path)
    if not path.exists():
        return out
    seen: set[int] = set()
    for line in path.read_text(errors="replace").splitlines():
        match = EPOCH_LINE.match(line)
        if not match:
            continue
        epoch = int(match.group(1))
        try:
            values = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError):
            continue
        if epoch not in seen:
            # a repeated epoch starts over: its first line again
            out[epoch] = {}
            seen.add(epoch)
        elif "moves" in values:
            out[epoch] = {}
        for k, v in values.items():
            try:
                out[epoch][k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def slope_per_100(values: Sequence[float]) -> float:
    """The least-squares slope of `values` (one an epoch), per 100 epochs."""
    n = len(values)
    mean_x, mean_y = (n - 1) / 2, sum(values) / n
    sxx = sum((x - mean_x) ** 2 for x in range(n))
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(values))
    return 100 * sxy / sxx


class Driver:
    def __init__(self, run_dir: Path, config: Config, backend: Backend, log=None):
        self.dir, self.config, self.backend = run_dir, config, backend
        self.log = log or (lambda text: print(text, flush=True))
        self.state_path = run_dir / "driver.json"
        self.state = State.load(self.state_path)

    # --- metrics

    def metrics(self, line: Line) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for segment in line.segments:
            for epoch, values in read_log(segment.log).items():
                if segment.start <= epoch < segment.end:
                    out[epoch] = values
        return out

    def series(self, line: Line, net: str | None) -> dict[int, float]:
        key = self.config.key(net)
        return {e: v[key] for e, v in self.metrics(line).items() if key in v}

    def lineage_series(self, line: Line, net: str | None) -> dict[int, float]:
        """`series` along the line's history: its parent's (and so on) before it forked, its
        own from there. A mainline that was a fork has trained since the run's start."""
        out: dict[int, float] = {}
        parent = self.state.lines.get(line.parent or "")
        if parent is not None:
            out = {
                e: v
                for e, v in self.lineage_series(parent, net).items()
                if e < line.start
            }
        return {**out, **self.series(line, net)}

    def logged_to(self, line: Line) -> int:
        """The epoch after the last the line has logged its playing network's metric for (its
        start, before any)."""
        epochs = self.series(line, None)
        return max(epochs) + 1 if epochs else line.start

    # --- decisions

    def decide(self, kind: str, line: Line | None, **what: Any) -> None:
        record = {
            "time": time.time(),
            "decision": kind,
            "line": line.name if line else None,
            **what,
        }
        with open(self.dir / "decisions.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        self.log(f"{kind}: {line.name if line else ''} {json.dumps(what)}")

    def lrs_at(self, epoch: int) -> dict[str, int]:
        """Each network's learning rate (index into lrs) for epochs from `epoch` on, as the
        mainline's steps have them."""
        lrs = {name: self.config.start_lrs.get(name, 0) for name, _ in self.config.nets}
        for at, net, index in self.state.lr_steps:
            if at <= epoch:
                lrs[net] = index
        return lrs

    def playing_at(self, epoch: int) -> int:
        """The index of the network playing in epoch `epoch`."""
        return sum(1 for at in self.state.takeovers if at <= epoch)

    def training_at(self, epoch: int) -> list[str]:
        """The networks that train in epoch `epoch`: the one playing and the next (those that
        played before don't)."""
        playing = self.playing_at(epoch)
        return [name for name, _ in self.config.nets[playing : playing + 2]]

    def caught_up(self, main: Line) -> bool:
        """Whether the mainline's next network has caught up with the one playing, by train.py's
        rule (--grow-metric, --grow-window, --grow-min-epochs), as of its last epoch logged."""
        metrics = self.metrics(main)
        caught = [
            v["growth_caught_up"]
            for _, v in sorted(metrics.items())
            if "growth_caught_up" in v
        ]
        return bool(caught) and caught[-1] == 1

    def lr_plateaus(self, main: Line, at: int) -> list[str]:
        """The mainline's networks whose learning rate steps down at `at`: plateaued, at least
        `lr_min_epochs` at their rate, a lower one left."""
        rules, out = self.config.rules, []
        lrs = self.lrs_at(at - 1)
        for name in self.training_at(at - 1):
            if lrs[name] + 1 >= len(self.config.lrs):
                continue
            since = max(
                [
                    0,
                    *(e for e, n, _ in self.state.lr_steps if n == name and e < at),
                ]
            )
            series = self.lineage_series(main, name)
            epochs = sorted(e for e in series if e >= since)
            if not epochs or len(epochs) < max(
                rules.lr_min_epochs, rules.plateau_epochs
            ):
                continue
            # a network that joined later counts from its first epoch
            recent = [series[e] for e in epochs[-rules.plateau_epochs :]]
            if slope_per_100(recent) > -rules.plateau_per_100:
                out.append(name)
        return out

    def flags(self, segment: Segment, branch_from: str | None) -> str:
        c = self.config
        first, *grown = c.nets
        sims, fast = c.sims[segment.sims]
        lr = {name: c.lrs[i] for name, i in segment.lrs.items()}
        argv = [
            "examples/othello/train.py",
            "--net",
            f"{first[0]}={first[1]}",
            *(a for name, spec in grown for a in ("--grow", f"{name}={spec}")),
            *c.train_args,
            "--lr",
            f"{lr[first[0]]:g}",
            "--lr-final",
            f"{lr[first[0]]:g}",
            "--weight-decay",
            f"{c.wd_per_lr * lr[first[0]]:g}",
            *(a for name, _ in grown for a in ("--lr-of", f"{name}={lr[name]:g}")),
            *(
                a
                for name, _ in grown
                for a in ("--weight-decay-of", f"{name}={c.wd_per_lr * lr[name]:g}")
            ),
            "--sims",
            str(sims),
            "--playout-cap",
            f"{c.full_share:g}:{fast}",
            "--buffer",
            str(c.buffer),
            "--buffer-window",
            str(c.windows[segment.window]),
            "--epochs",
            str(segment.end),
            "--save-state-at",
            str(segment.end),
            "--ckpt-dir",
            segment.ckpt_dir,
            "--monitor",
            "--name",
            f"{Path(segment.ckpt_dir).parent.name}-{Path(segment.ckpt_dir).name}",
            "--notes",
            f"fork_driver {c.name}: line {Path(segment.ckpt_dir).parent.name}, epochs "
            f"{segment.start}-{segment.end - 1}, window {c.windows[segment.window]}, "
            f"sims {sims}/{fast}, lr " + ", ".join(f"{n} {v:g}" for n, v in lr.items()),
            *(["--grow-manual"] if c.growth else []),
            *(["--take-over"] if segment.take_over else []),
            *(["--branch-from", branch_from] if branch_from else []),
            *(
                ["--init-from", c.start_from]
                if branch_from is None and c.start_from
                else []
            ),
        ]
        return (
            " ".join(shlex.quote(a) for a in argv)
            + f" > {shlex.quote(segment.log)} 2>&1"
        )

    def new_segment(self, line: Line, start: int, window: int, sims: int) -> Segment:
        root = self.dir / "lines" / line.name / f"seg{start}"
        return Segment(
            start=start,
            end=start + self.config.segment_epochs,
            ckpt_dir=str(root),
            log=str(self.dir / "logs" / f"{line.name}-seg{start}.log"),
            window=window,
            sims=sims,
            lrs=self.lrs_at(start),
            take_over=start in self.state.takeovers,
        )

    @staticmethod
    def final_state(segment: Segment) -> str:
        return f"{segment.ckpt_dir}/state-{segment.end}.npz"

    # --- the loop

    def tick(self) -> None:
        """One look at every line: extend, judge, start forks, end the run."""
        s = self.state
        if s.done:
            return
        if not s.main:
            self.start_main()
        main = s.lines[s.main]
        self.extend_main(main)
        for line in [
            l for l in s.lines.values() if l.status == "running" and l.kind != "main"
        ]:
            # (an adoption just now replaces the mainline and restarts the other fork)
            if line.status != "running" or line.kind == "main":
                continue
            self.judge(line, s.lines[s.main])
            if line.status == "running" and line.kind != "main":
                self.extend_fork(line)
        main = s.lines[s.main]
        self.maybe_fork(main)
        self.maybe_finish(main)
        for line in s.lines.values():
            if line.status == "running":
                self.keep_alive(line)
        s.save(self.state_path)

    def start_main(self) -> None:
        line = Line(name="main", kind="main", job=f"{self.config.name}-main", start=0)
        segment = self.new_segment(
            line, 0, window=self.config.start_window, sims=self.config.start_sims
        )
        line.segments.append(segment)
        self.state.lines[line.name] = line
        self.state.main = line.name
        self.backend.launch(line.job, [self.flags(segment, None)])
        self.decide(
            "start",
            line,
            window=self.config.windows[segment.window],
            sims=self.config.sims[segment.sims],
            start_from=self.config.start_from,
        )

    def due(self, line: Line) -> bool:
        return self.logged_to(line) >= line.end - self.config.lead_epochs

    def extend_main(self, main: Line) -> None:
        if main.status != "running" or not self.due(main):
            return
        at = main.end
        # (a step made here by hand, in driver.json, stands: the rule decides the others)
        stepped = {net for e, net, _ in self.state.lr_steps if e == at}
        for net in self.lr_plateaus(main, at):
            if net in stepped:
                continue
            index = self.lrs_at(at - 1)[net] + 1
            self.state.lr_steps.append((at, net, index))
            self.decide("lr step", main, epoch=at, net=net, lr=self.config.lrs[index])
        playing = self.playing_at(at - 1)
        if (
            playing + 1 < len(self.config.nets)
            and at not in self.state.takeovers
            and self.caught_up(main)
        ):
            self.state.takeovers.append(at)
            self.decide(
                "take over",
                main,
                epoch=at,
                net=self.config.nets[playing + 1][0],
                retires=self.config.nets[playing][0],
            )
        last = main.segments[-1]
        self.append(main, last.window, last.sims)

    def append(self, line: Line, window: int, sims: int) -> None:
        last = line.segments[-1]
        segment = self.new_segment(line, last.end, window, sims)
        line.segments.append(segment)
        self.backend.extend(line.job, [self.flags(segment, self.final_state(last))])

    def extend_fork(self, fork: Line) -> None:
        main = self.state.lines[self.state.main]
        # a fork follows the mainline's learning rates: it waits for the mainline's decision
        if not self.due(fork) or main.end <= fork.end and main.status == "running":
            return
        last = fork.segments[-1]
        self.append(fork, last.window, last.sims)

    def maybe_fork(self, main: Line) -> None:
        """Starts a fork of each kind not running, if its ladder has a step left and it isn't
        waiting to try again: at most one window fork and one sims fork at a time."""
        s, c = self.state, self.config
        running = {l.kind for l in s.lines.values() if l.status == "running"}
        done = [seg for seg in main.segments if self.logged_to(main) >= seg.end]
        if not done or done[-1].end < c.rules.fork_from_epoch:
            return
        base = done[-1]
        if main.end <= base.end:
            return  # the mainline hasn't decided its learning rates from there yet
        if not self.backend.has_state(self.final_state(base)):
            return  # still on its way to the bucket
        epoch = self.logged_to(main)
        for kind in KINDS:
            if (
                kind not in running
                and self.next_index(base, kind) is not None
                and s.retry_at.get(kind, 0) <= epoch
            ):
                self.start_fork(main, base, kind)

    def start_fork(self, main: Line, base: Segment, kind: str) -> None:
        s, c = self.state, self.config
        index = self.next_index(base, kind)
        assert index is not None
        name = f"{kind}{len(s.started)}"
        fork = Line(
            name=name,
            kind=kind,
            job=f"{c.name}-{name}",
            start=base.end,
            parent=main.name,
        )
        window, sims = (index, base.sims) if kind == "window" else (base.window, index)
        segment = self.new_segment(fork, base.end, window, sims)
        fork.segments.append(segment)
        s.lines[name] = fork
        s.started.append(kind)
        self.backend.launch(fork.job, [self.flags(segment, self.final_state(base))])
        self.decide(
            "fork",
            fork,
            epoch=base.end,
            window=c.windows[window],
            sims=c.sims[sims],
        )

    def next_index(self, segment: Segment, kind: str) -> int | None:
        current = segment.window if kind == "window" else segment.sims
        ladder = self.config.windows if kind == "window" else self.config.sims
        return current + 1 if current + 1 < len(ladder) else None

    def comparison(self, fork: Line, main: Line) -> list[tuple[float, float]]:
        """The last `windows_ahead` windows of matched epochs: (fork's mean, mainline's)."""
        rules = self.config.rules
        theirs, ours = self.series(fork, None), self.series(main, None)
        # from the last take-over: both lines' playing network is the same one from there
        since = max(
            [0, *(at for at in self.state.takeovers if at <= self.logged_to(fork))]
        )
        matched = sorted(e for e in theirs if e in ours and e >= since)
        out = []
        for w in range(rules.windows_ahead):
            window = matched[len(matched) - (w + 1) * rules.window_epochs :][
                : rules.window_epochs
            ]
            if len(window) < rules.window_epochs:
                break
            out.append(
                (
                    sum(theirs[e] for e in window) / len(window),
                    sum(ours[e] for e in window) / len(window),
                )
            )
        return out

    def ready(self, fork: Line, latest: int) -> bool:
        """Whether the fork has run long enough to judge."""
        rules = self.config.rules
        if latest - fork.start < rules.min_epochs:
            return False
        if fork.kind != "window":
            return True
        # the playing network's steps: its metric is the one compared
        names = [name for name, _ in self.config.nets]
        steps = [
            e
            for e, net, _ in self.state.lr_steps
            if fork.start < e <= latest and net == names[self.playing_at(e)]
        ]
        if steps and latest - steps[-1] >= rules.after_lr_step:
            return True
        floor = len(self.config.lrs) - 1
        return self.lrs_at(latest)[names[self.playing_at(latest)]] == floor

    def judge(self, fork: Line, main: Line) -> None:
        rules = self.config.rules
        latest = self.logged_to(fork)
        windows = self.comparison(fork, main)
        if not self.ready(fork, latest) and latest - fork.start < rules.max_fork_epochs:
            return
        ahead = len(windows) == rules.windows_ahead and all(
            theirs <= ours - rules.margin for theirs, ours in windows
        )
        means = [[round(t, 5), round(o, 5)] for t, o in windows]
        if ahead and self.ready(fork, latest):
            self.adopt(fork, main, latest, means)
            return
        series = self.series(fork, None)
        recent = [series[e] for e in sorted(series)[-rules.plateau_epochs :]]
        plateaued = len(recent) >= rules.plateau_epochs and (
            slope_per_100(recent) > -rules.plateau_per_100
        )
        # a plateau ends it only without a lead worth waiting on: at a constant low learning
        # rate every curve is flat, and a fork just short of the margin may yet clear it
        lead = sum(ours - theirs for theirs, ours in windows) / max(len(windows), 1)
        if latest - fork.start >= rules.max_fork_epochs or (
            self.ready(fork, latest) and plateaued and lead < rules.margin / 2
        ):
            fork.status, fork.ended_at = "killed", time.time()
            self.backend.stop(fork.job)
            self.state.retry_at[fork.kind] = self.logged_to(main) + rules.retry_after
            self.decide(
                "kill",
                fork,
                epoch=latest,
                windows=means,
                plateaued=plateaued,
                lead=round(lead, 5),
            )

    def adopt(self, fork: Line, main: Line, latest: int, means: list) -> None:
        main.status, main.ended_at = "replaced", time.time()
        self.backend.stop(main.job)
        fork.status, fork.kind = "running", "main"
        self.state.main = fork.name
        # the old mainline's steps the fork hasn't taken yet are undone: the new one decides
        applied = fork.segments[-1].start
        self.state.lr_steps = [
            step for step in self.state.lr_steps if step[0] <= applied
        ]
        self.state.takeovers = [at for at in self.state.takeovers if at <= applied]
        self.decide("adopt", fork, epoch=latest, windows=means, replaces=main.name)
        # the other fork branched from the old mainline: it differs from the new one in two
        # settings now, so it starts again from the new one (no wait to try again)
        for other in self.state.lines.values():
            if other.status == "running" and other.kind in KINDS:
                other.status, other.ended_at = "restarted", time.time()
                self.backend.stop(other.job)
                self.decide(
                    "restart", other, epoch=self.logged_to(other), mainline=fork.name
                )

    def maybe_finish(self, main: Line) -> None:
        """The run ends when every network is at its last learning rate and plateaued there,
        and no fork is left to try."""
        if main.status != "running" or not self.due(main):
            return
        floor = len(self.config.lrs) - 1
        playing = self.playing_at(main.end)
        if playing + 1 < len(self.config.nets):
            return  # a bigger network is still to take over
        if self.lrs_at(main.end)[self.config.nets[playing][0]] != floor:
            return
        if any(
            l.status == "running" and l.kind != "main"
            for l in self.state.lines.values()
        ):
            return
        last = main.segments[-1]
        if any(self.next_index(last, kind) is not None for kind in KINDS) and any(
            self.state.retry_at.get(k, 0) <= self.logged_to(main) for k in KINDS
        ):
            return
        rules = self.config.rules
        series = self.lineage_series(main, None)
        recent = [series[e] for e in sorted(series)[-rules.plateau_epochs :]]
        if (
            len(recent) < rules.plateau_epochs
            or slope_per_100(recent) <= -rules.plateau_per_100
        ):
            return
        main.status = "done"
        self.state.done = True
        self.backend.stop(main.job)
        self.decide("done", main, epoch=self.logged_to(main))

    def keep_alive(self, line: Line) -> None:
        """A job that ended short (preempted past its retries, its time limit, a wait for the
        next segment that ran out) goes on."""
        state = self.backend.state(line.job)
        if state in ("done", "timeout", "lost", "failed"):
            if state == "failed":
                self.decide("failed", line, job=line.job)
                line.status = "failed"
                return
            if self.logged_to(line) < line.end:
                self.backend.resume(line.job)
                self.decide("resume", line, job_state=state)


class ModalBackend:
    """Modal jobs, through modal_jobs.py's functions."""

    def __init__(self, config: Config):
        self.config = config
        self.store, _ = modal_jobs.open_store()
        self.home = str(Path.home())

    def _steps(self, steps: list[str]) -> list[Step]:
        return [parse_step(text, self.home) for text in steps]

    def launch(self, job: str, steps: list[str]) -> None:
        c = self.config
        modal_jobs.launch_job(
            self.store,
            Job(
                name=job,
                commit=modal_jobs.resolve_commit(c.commit),
                gpu=c.gpu,
                hours=c.hours_per_line,
                home=self.home,
                steps=self._steps(steps),
                cpu=c.cpu,
                memory_gb=c.job_memory_gb(),
                resume_every=c.resume_every,
                follow_minutes=c.follow_minutes,
                states_in_bucket=True,
                tags=",".join(t for t in (c.tags, "modal") if t),
                created=time.time(),
                launched_from={"driver": c.name},
            ),
        )
        modal_jobs.ensure_sync()

    def extend(self, job: str, steps: list[str]) -> None:
        modal_jobs.extend_job(self.store, job, self._steps(steps))

    def stop(self, job: str) -> None:
        put_json(self.store, job_key(job, "stop.json"), {"stop": True})

    def state(self, job: str) -> str:
        status = get_json(self.store, job_key(job, "status.json")) or {}
        return status.get("state", "queued")

    def resume(self, job: str) -> None:
        """Runs the job again, with `hours_per_line` more than it has spent."""
        data = get_json(self.store, job_key(job, "job.json"))
        status = get_json(self.store, job_key(job, "status.json")) or {}
        spent = sum(a["seconds"] for a in status.get("attempts", [])) / 3600
        data["hours"] = spent + self.config.hours_per_line
        put_json(self.store, job_key(job, "job.json"), data)
        put_json(self.store, job_key(job, "stop.json"), {"stop": False})
        modal_jobs.spawn(Job.from_dict(data), self.store)

    def has_state(self, path: str) -> bool:
        return bool(self.store.sizes(Job("", "", "", 0, self.home, []).key_of(path)))


TEMPLATE = """\
# fork_driver.py's run (see its docstring). Edit before `run`; the ladders and rules are read
# at every tick.
name = "{name}"
commit = "origin/dev"          # what the jobs run: on GitHub
tags = "{tags}"
buffer = 9000                   # kept at its largest; training samples the window's newest
windows = [1125, 2250, 4500, 9000]
sims = [[128, 32], [256, 64], [512, 128]]
full_share = 0.25               # playout cap randomization: full searches' share
lrs = [1e-3, 5e-4, 1e-4]
wd_per_lr = 500.0
segment_epochs = 50
metric = "avg4/engine_value_mse"
gpu = "L40S"
cpu = 16                        # the solver's bursts use up to 16 more
# memory_gb = 96                # host GiB; by default sized by the buffer
resume_every = 50               # a state to continue from at each segment's end
# the 11x64 at 1e-4, 128/32, buffer 2250, line 900: it plays first, at 1e-4, window 2250
start_from = "{home}/.cache/turbozero/othello/smallnet-data/runs/ray11x64n-r800-pcr25x32-lr1e-4-wd5e-2-buf2250/state-100.npz"
start_lrs = {{ r11 = 2 }}
start_window = 1
start_sims = 0
train_args = [{train_args}]

[[nets]]
name = "r11"
spec = "resnet:11x64,pool_every=2,heads=pooled,pool_channels=16,layout=pre,block=convnext,expand=2,dw_taps=rays"

[[nets]]
name = "r13"
spec = "resnet:13x64,pool_every=2,heads=pooled,pool_channels=16,layout=pre,block=convnext,expand=2,dw_taps=rays"

[[nets]]
name = "r15"
spec = "resnet:15x64,pool_every=2,heads=pooled,pool_channels=16,layout=pre,block=convnext,expand=2,dw_taps=rays"

[[nets]]
name = "r17"
spec = "resnet:17x64,pool_every=2,heads=pooled,pool_channels=16,layout=pre,block=convnext,expand=2,dw_taps=rays"

[rules]
fork_from_epoch = 100
window_epochs = 20
margin = 0.003
windows_ahead = 2
min_epochs = 60
after_lr_step = 40
max_fork_epochs = 300
retry_after = 200
plateau_epochs = 60
plateau_per_100 = 0.001
lr_min_epochs = 100
"""

# the 11x64 line 1053 recipe's self-play and training; growth's rule on the averages (the
# driver takes the networks over, --grow-manual, in every line at once)
RECIPE = [
    "--train-batch", "4096", "--train-steps", "256", "--sampling-moves", "20",
    "--dirichlet-alpha", "0.3", "--dirichlet-epsilon", "0.25", "--tree-factor", "4",
    "--standard-starts", "0", "--average-epochs", "4", "--selfplay-average", "4",
    "--optimizer", "adamw", "--seed", "0", "--keep-every", "20", "--engine-eval-every", "1",
    "--solve-empties", "20", "--solve-play-past", "end", "--solve-threads", "31",
    "--grow-metric", "avg4/engine_value_mse", "--grow-window", "20", "--grow-min-epochs", "20",
]  # fmt: skip


def status(driver: Driver) -> None:
    s, c = driver.state, driver.config
    print(f"{c.name}: mainline {s.main or '-'}{', done' if s.done else ''}")
    for line in s.lines.values():
        last = line.segments[-1]
        series = driver.series(line, None)
        recent = [series[e] for e in sorted(series)[-c.rules.window_epochs :]]
        mean = f"{sum(recent) / len(recent):.4f}" if recent else "-"
        print(
            f"  {line.name:10s} {line.kind:6s} {line.status:8s} epochs {line.start}-"
            f"{driver.logged_to(line) - 1} (segment to {line.end}), window "
            f"{c.windows[last.window]}, sims {c.sims[last.sims]}, {c.metric} "
            f"last {c.rules.window_epochs}: {mean}"
        )
    for epoch, net, index in s.lr_steps:
        print(f"  lr step at {epoch}: {net} -> {c.lrs[index]:g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("init", help="write a driver.toml to edit")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--name", required=True)
    p = commands.add_parser(
        "run", help="tick every --every seconds until the run is done"
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--every", type=float, default=120)
    p.add_argument("--once", action="store_true")
    p = commands.add_parser("status", help="the lines and decisions")
    p.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    if args.command == "init":
        args.run_dir.mkdir(parents=True, exist_ok=True)
        path = args.run_dir / "driver.toml"
        if path.exists():
            raise SystemExit(f"{path} exists")
        import os

        path.write_text(
            TEMPLATE.format(
                name=args.name,
                tags=os.environ.get("SEAGLASS_TAGS", ""),
                home=str(Path.home()),
                train_args=", ".join(json.dumps(a) for a in RECIPE),
            )
        )
        print(f"wrote {path}")
        return
    config = Config.load(args.run_dir / "driver.toml")
    if args.command == "status":
        status(Driver(args.run_dir, config, backend=None))  # pyright: ignore[reportArgumentType]
        return
    driver = Driver(args.run_dir, config, ModalBackend(config))
    while True:
        driver.config = Config.load(args.run_dir / "driver.toml")
        driver.tick()
        if args.once or driver.state.done:
            return
        time.sleep(args.every)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
