"""What a Modal job is: its steps, each a train.py command as it would run here, and where
its files go. Imported on both sides (see modal_jobs.py), so only the standard library.

Paths stay the absolute paths of this machine: the container makes the same directories,
under the same $HOME, so a step's flags and its recorded config read as a local run's.
"""

import os
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCRIPT = "examples/othello/train.py"
STATE = r"state-(\d+)\.npz"
# on-demand list prices, per second, read from modal.com/pricing on 2026-10-31
GPU_PER_SECOND = {
    "CPU": 0.0,  # no GPU: for trying out the harness on a tiny net
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "RTX-PRO-6000": 0.000842,
    "H100!": 0.001097,
    "H100": 0.001097,
    "H200": 0.001261,
}
CORE_PER_SECOND = 0.0000131
GIB_PER_SECOND = 0.00000222
# in-run games need Edax, which the image doesn't build
UNSUPPORTED = ("--vs-engine-every", "--vs-engine-last")


def flag(argv: list[str], name: str) -> str | None:
    """The value of `name` in `argv` (`--name value` or `--name=value`), None if absent."""
    for i, arg in enumerate(argv):
        if arg == name:
            return argv[i + 1] if i + 1 < len(argv) else ""
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def without_flag(argv: list[str], name: str, takes_value: bool) -> list[str]:
    """`argv` without `name`, and its value if it takes one."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == name:
            i += 2 if takes_value else 1
        elif argv[i].startswith(name + "="):
            i += 1
        else:
            out.append(argv[i])
            i += 1
    return out


def with_monitor(argv: list[str], url: str) -> list[str]:
    """`argv` logging to `url`, if it logs anywhere: --monitor's value may be absent."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--monitor":
            has_value = i + 1 < len(argv) and not argv[i + 1].startswith("--")
            out += ["--monitor", url]
            i += 2 if has_value else 1
        elif argv[i].startswith("--monitor="):
            out.append(f"--monitor={url}")
            i += 1
        else:
            out.append(argv[i])
            i += 1
    return out


def state_epochs_requested(argv: list[str]) -> set[int]:
    """The epochs of --save-state-at."""
    at = flag(argv, "--save-state-at")
    return {int(epoch) for epoch in at.split(",")} if at else set()


@dataclass(frozen=True)
class Step:
    """One train.py run: its flags, and the file its output goes to."""

    argv: list[str]
    log: str

    @property
    def ckpt_dir(self) -> str:
        path = flag(self.argv, "--ckpt-dir")
        assert path is not None, "parse_step requires it"
        return path


def parse_step(text: str, home: str) -> Step:
    """A step from the command that would run it here, e.g.
    `uv run --extra cuda examples/othello/train.py --epochs 2 --ckpt-dir ~/... > ~/....log`.
    What's before the script (uv run, taskset, variables) is dropped: the image decides it."""
    words = [
        os.path.expandvars(word).replace("~/", home + "/", 1)
        for word in shlex.split(text)
    ]
    if SCRIPT not in words:
        raise ValueError(f"a step runs {SCRIPT}: {text!r}")
    words = words[words.index(SCRIPT) + 1 :]
    log, argv, i = None, [], 0
    while i < len(words):
        if words[i] == ">":
            log = words[i + 1]
            i += 2
        elif words[i] == "<":
            i += 2
        elif words[i] == "2>&1":
            i += 1
        elif words[i].startswith(">"):
            log = words[i][1:]
            i += 1
        else:
            argv.append(words[i])
            i += 1
    if log is None:
        raise ValueError(f"a step sends its output to a file, `> LOG`: {text!r}")
    ckpt_dir = flag(argv, "--ckpt-dir")
    if not ckpt_dir:
        raise ValueError(f"a step needs a --ckpt-dir: {text!r}")
    if "--resume" in argv:
        raise ValueError("no --resume: a job resumes its steps itself")
    init_from = flag(argv, "--init-from")
    if init_from is not None and not re.fullmatch(STATE, Path(init_from).name):
        raise ValueError(f"--init-from names a state-<EPOCH>.npz: {init_from}")
    for name in UNSUPPORTED:
        if flag(argv, name) is not None:
            raise ValueError(
                f"{name}: the Modal image has no Edax; the Elo league measures strength here"
            )
    if flag(argv, "--save-positions-from") is not None:
        raise ValueError(
            "--save-positions-from is for local runs: bringing the positions here from the "
            "bucket costs more in egress than they're worth"
        )
    root = Path(home) / ".cache" / "turbozero"
    for path in (ckpt_dir, log, flag(argv, "--init-from"), flag(argv, "--engine-set")):
        if path is not None and not Path(path).is_relative_to(root):
            raise ValueError(f"{path} isn't under {root}: it wouldn't come back here")
    return Step(argv, log)


@dataclass(frozen=True)
class Job:
    """A sequence of steps on one GPU, at one commit, with a limit on its total time."""

    name: str
    commit: str
    gpu: str
    hours: float
    home: str
    steps: list[Step]
    # 2 cores trained as fast as 4 (49 s/epoch vs 48, 8x128 on an L40S)
    cpu: float = 2.0
    memory_gb: int = 16
    # saves the whole state this often, to continue from if the job is interrupted
    resume_every: int = 10
    tags: str = ""
    created: float = 0.0
    launched_from: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Job":
        steps = [Step(**step) for step in data["steps"]]
        return Job(**{**data, "steps": steps})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dollars_per_hour(self) -> float:
        return 3600 * (
            GPU_PER_SECOND[self.gpu]
            + self.cpu * CORE_PER_SECOND
            + self.memory_gb * GIB_PER_SECOND
        )

    def key_of(self, path: str) -> str:
        """Where a file of this machine lives in the bucket: any job can find it there."""
        return "files/" + str(Path(path).relative_to(self.home))


def job_key(name: str, *rest: str) -> str:
    """A key of the job's own, in the bucket."""
    return "/".join(("jobs", name, *rest))


def path_of(key: str, home: str) -> str:
    """The inverse of `Job.key_of`."""
    assert key.startswith("files/"), key
    return str(Path(home) / key.removeprefix("files/"))
