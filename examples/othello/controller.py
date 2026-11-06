"""A controller for a long Othello run: train.py with --grow, going on in steps, each its own
entry in slick's GPU queue, the controller deciding between steps what the run does next from
games between its networks in the Elo league (see README.md, "The run controller").

Every period:

1. trunk: the run trains on at its current settings, `trunk_turnovers` turnovers of its replay
   buffer.
2. size: if the next network has caught up with the one playing on the engine set's value
   error (growth's rule, which with --grow-manual only reports), the two play a series at equal
   search time a move: the playing network searches as many times more iterations as its
   forward pass is cheaper (by bench_nets.py). The next takes over self-play if it scores
   `switch_margin` Elo or more.
3. sims: two branches of the run, at equal compute. `double` searches twice the iterations
   (both budgets, with playout cap randomization) for `branch_turnovers` turnovers of its
   buffer; `stay` keeps the search for as many epochs as cost the same. The networks playing in
   each play a series, and the run goes on from `double` if it scores `sims_margin` Elo or
   more, else from `stay`.

Compute is counted two ways: in network evaluations (self-play's positions evaluated, and the
samples each network trains on), and in benchmarked GPU time, each network's forward pass at
self-play's batch and its training step at training's (bench_nets.py, run first) times those.
Flat costs (moving the games, writing the buffer, compiling) aren't counted. A branch that loses
counts in what the run spent, not in the line of states it kept.

    uv run examples/othello/controller.py init runs/big --config big.json
    uv run examples/othello/controller.py start runs/big
    uv run examples/othello/controller.py status runs/big
    uv run examples/othello/controller.py stop runs/big          # after the step that runs now
    uv run examples/othello/controller.py retry runs/big p003-double

The config (JSON) names the networks, smallest first, the search to start with, train.py's other
arguments, the periods' lengths and the margins (see `Config`). Each step runs
`controller.py step RUN STEP` in the queue, from the frozen worktree the config names, and queues
the next when it's done, from the slick task's worktree (`queue_cwd`): `stop` lets the step that
runs now finish, and queues nothing after it.
"""

import argparse
import contextlib
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Generator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# train.py's: self-play environments (its Trainer's batch_size), and moves a game takes a step of
# an epoch without --selfplay-cycles
ENVS = 1024
MOVES_PER_EPOCH = 128

# train.py arguments the controller sets itself, step by step
OWN_ARGS = (
    "--net",
    "--grow",
    "--grow-manual",
    "--grow-switch-at",
    "--sims",
    "--sims-schedule",
    "--playout-cap",
    "--selfplay-cycles",
    "--advance-every",
    "--buffer",
    "--train-steps",
    "--train-batch",
    "--epochs",
    "--ckpt-dir",
    "--save-state-at",
    "--save-state-every",
    "--branch-from",
    "--take-over",
    "--resume",
    "--init-from",
    "--name",
    "--monitor",
)


def net_name(spec: str) -> str:
    """A network's short name, as train.py's --grow names its checkpoint directory: its size,
    e.g. 4x64 for resnet:4x64,pool_every=2."""
    found = re.search(r"\d+x\d+", spec)
    return found.group(0) if found else spec.split(",")[0].replace(":", "-")


@dataclass(frozen=True)
class Search:
    """Self-play's search: `full` iterations a move, or with playout cap randomization `full` on
    a `full_probability` share of moves (the moves stored for training) and `fast` on the rest,
    in `cycles` search cycles per environment an epoch (train.py's --selfplay-cycles).

    Doubling it doubles every search's cycles (the root's evaluation and the iterations), and an
    epoch's: an epoch then plays and stores the same moves, at twice the self-play compute, so
    each stored position is trained on as many times."""

    full: int
    fast: int | None = None
    full_probability: float = 1.0
    cycles: int | None = None

    def __post_init__(self):
        if self.full < 1:
            raise ValueError(f"a search needs iterations, got full={self.full}")
        if (self.fast is None) != (self.cycles is None) or (
            self.fast is None and self.full_probability != 1.0
        ):
            raise ValueError(
                "playout cap randomization needs fast, full_probability and cycles together"
            )
        if self.fast is not None:
            assert self.cycles is not None
            if not (0 < self.full_probability < 1 and 0 < self.fast < self.full):
                raise ValueError(
                    f"with playout cap randomization, 0 < full_probability < 1 and "
                    f"0 < fast < full: got {self}"
                )
            if self.cycles % (self.full + 1):
                raise ValueError(
                    f"cycles ({self.cycles}) must be a multiple of full + 1 ({self.full + 1})"
                )

    @property
    def capped(self) -> bool:
        """Whether it's playout cap randomization."""
        return self.fast is not None

    def epoch_cycles(self) -> int:
        """Search cycles per environment an epoch."""
        if self.cycles is not None:
            return self.cycles
        return MOVES_PER_EPOCH * (self.full + 1)

    def cycles_per_move(self) -> float:
        """Search cycles a move takes on average."""
        if self.fast is None:
            return float(self.full + 1)
        p = self.full_probability
        return p * (self.full + 1) + (1 - p) * (self.fast + 1)

    def stored_per_epoch(self) -> float:
        """Positions an environment stores for training an epoch: every move's, or with playout
        cap randomization the full searches'."""
        moves = self.epoch_cycles() / self.cycles_per_move()
        return moves * self.full_probability

    def doubled(self) -> "Search":
        if self.fast is None:
            return Search(2 * (self.full + 1) - 1)
        assert self.cycles is not None
        return Search(
            2 * (self.full + 1) - 1,
            2 * (self.fast + 1) - 1,
            self.full_probability,
            2 * self.cycles,
        )

    def train_args(self) -> list[str]:
        args = ["--sims", str(self.full)]
        if self.fast is not None:
            args += [
                "--playout-cap",
                f"{self.full_probability:g}:{self.fast}",
                "--selfplay-cycles",
                str(self.cycles),
                # play the moves whose searches are done as often as a fast search takes
                "--advance-every",
                str(self.fast + 1),
            ]
        return args

    def label(self) -> str:
        """As run names put it, e.g. 64s, or pcr25x127-31s."""
        if self.fast is None:
            return f"{self.full}s"
        return f"pcr{round(100 * self.full_probability)}x{self.full}-{self.fast}s"


@dataclass(frozen=True)
class Config:
    """A controlled run's settings (the JSON config's keys).

    Attributes:
        name: the run's name: its steps' names and league players start with it
        nets: train.py --net specs, smallest first: the first plays first, each takes over in
            turn (train.py --grow, with --grow-manual)
        search: the search to start with (see `Search`)
        worktree: the frozen worktree the steps run in, with a CUDA environment (./install.sh)
        queue_cwd: the slick task's worktree, where `slick queue` queues the steps
        train_args: train.py's other arguments, e.g. ["--lr-final", "1e-3"] for a constant
            learning rate. Not those the controller sets (`OWN_ARGS`)
        buffer: train.py --buffer, positions kept per environment
        train_steps: train.py --train-steps, training steps an epoch
        train_batch: train.py --train-batch
        trunk_turnovers: a period's trunk, in turnovers of the replay buffer at its search
        branch_turnovers: the `double` branch's length, in turnovers of its replay buffer
        games: games a series (even: both colours of each opening)
        play_sims: the league's search for the series, iterations a move (the playing network
            of a size series searches more, see the module's docstring)
        switch_margin: the Elo the next network must score against the one playing to take over
        sims_margin: the Elo `double` must score against `stay` to be kept
        max_periods: (optional) the periods to run
        budget_hours: (optional) benchmarked GPU hours the run may spend, losing branches
            included: no period starts past it
        save_state_every: train.py --save-state-every, so a step that stops can be retried
        league_worktree: the league's frozen worktree (see the league's OPERATING.md)
        league_root: the league's store
        rate_hours: the longest a rating step plays before it gives up
        priority: the queue entries' priority
        resource: (optional) the queue's resource, when there's more than one
        prune_states: delete saved states no branch will start from (the losing branch's, and
            the trunk's the branches started from), about 1 GB each. Each period's kept state
            stays: the run's line of states, to fork from later (e.g. a final learning rate drop)
        monitor: log each training step to seaglass (train.py --monitor), as its own run
        cores: the CPU cores a step runs on (taskset), so other tasks on the machine keep theirs
    """

    name: str
    nets: list[str]
    search: Search
    worktree: str
    queue_cwd: str
    train_args: list[str] = field(default_factory=list)
    buffer: int = 1125
    train_steps: int = 256
    train_batch: int = 4096
    trunk_turnovers: float = 2.0
    branch_turnovers: float = 1.0
    games: int = 400
    play_sims: int = 128
    switch_margin: float = 0.0
    sims_margin: float = 30.0
    max_periods: int | None = None
    budget_hours: float | None = None
    save_state_every: int = 20
    league_worktree: str = "~/worktrees/turbozero/elo-frozen"
    league_root: str = "~/.cache/turbozero/othello/league"
    rate_hours: float = 2.0
    priority: int = 0
    resource: str | None = None
    prune_states: bool = True
    monitor: bool = True
    cores: str = "0-7"

    def __post_init__(self):
        if not self.nets:
            raise ValueError("a run needs at least one network")
        names = [net_name(n) for n in self.nets]
        if len(set(names)) != len(names):
            raise ValueError(f"the networks' sizes must differ: {names}")
        own = [a for a in self.train_args if a.split("=")[0] in OWN_ARGS]
        if own:
            raise ValueError(f"the controller sets these itself: {own}")
        if self.trunk_turnovers <= 0 or self.branch_turnovers <= 0:
            raise ValueError("trunk_turnovers and branch_turnovers must be positive")
        if self.games < 2 or self.games % 2:
            raise ValueError(f"games must be even, got {self.games}")

    @classmethod
    def from_json(cls, data: dict) -> "Config":
        return cls(**{**data, "search": Search(**data["search"])})

    def to_json(self) -> dict:
        return asdict(self)


# compute


def epoch_compute(
    config: Config, search: Search, playing: int, bench: dict[str, dict]
) -> dict[str, Any]:
    """An epoch's compute: the network playing (index into `config.nets`) evaluates every
    environment's position each search cycle, and it and the next (if any) take
    `config.train_steps` training steps each.

    Returns:
        dict: `seconds`, benchmarked GPU time; `selfplay_evals`, positions evaluated in self-play;
            `train_samples`, by network
    """
    training = [i for i in (playing, playing + 1) if i < len(config.nets)]
    player = bench[config.nets[playing]]
    cycles = search.epoch_cycles()
    samples = config.train_steps * config.train_batch
    seconds = cycles * player["forward_ms"] / 1e3 + sum(
        config.train_steps * bench[config.nets[i]]["train_step_ms"] / 1e3
        for i in training
    )
    return {
        "seconds": seconds,
        "selfplay_evals": cycles * ENVS,
        "train_samples": {net_name(config.nets[i]): samples for i in training},
    }


def scaled(compute: dict[str, Any], epochs: int) -> dict[str, Any]:
    """An epoch's compute (see `epoch_compute`) over `epochs` epochs."""
    return {
        "seconds": compute["seconds"] * epochs,
        "selfplay_evals": compute["selfplay_evals"] * epochs,
        "train_samples": {k: v * epochs for k, v in compute["train_samples"].items()},
    }


def added(total: dict[str, Any], compute: dict[str, Any]) -> dict[str, Any]:
    """Two amounts of compute (see `epoch_compute`) together."""
    samples = dict(total.get("train_samples", {}))
    for k, v in compute["train_samples"].items():
        samples[k] = samples.get(k, 0) + v
    return {
        "seconds": total.get("seconds", 0.0) + compute["seconds"],
        "selfplay_evals": total.get("selfplay_evals", 0) + compute["selfplay_evals"],
        "train_samples": samples,
    }


def turnover_epochs(config: Config, search: Search) -> float:
    """Epochs self-play takes to fill the replay buffer with new positions."""
    return config.buffer / search.stored_per_epoch()


def branch_epochs(
    config: Config, search: Search, playing: int, bench: dict[str, dict]
) -> tuple[int, int]:
    """The epochs of a period's two branches: `double`'s, `branch_turnovers` turnovers of its
    buffer, and `stay`'s, those whose benchmarked GPU time is closest to it."""
    double = max(
        1, math.ceil(config.branch_turnovers * turnover_epochs(config, search))
    )
    stay_cost = epoch_compute(config, search, playing, bench)["seconds"]
    double_cost = epoch_compute(config, search.doubled(), playing, bench)["seconds"]
    stay = max(1, round(double * double_cost / stay_cost))
    return stay, double


# reading saved states and series


def read_meta(path: str) -> dict:
    """A saved state's metadata (see Trainer.save_state), without loading its arrays."""
    with np.load(path) as saved:
        return json.loads(str(saved["meta"]))


def caught_up(meta: dict) -> bool:
    """Whether a saved state's next network has caught up with the one playing: as growth's rule
    (core.training.growth.Growth.caught_up) has it, from the state's progress and the rule's
    window and minimum epochs."""
    growth = meta.get("growth")
    if growth is None:
        return False
    rule = meta["config"]["growth"]
    playing, values, trained = growth["playing"], growth["values"], growth["trained"]
    following = playing + 1
    if following >= len(trained):
        return False
    window = rule["window"]
    if any(len(values[i]) < window for i in (playing, following)):
        return False
    gap = float(np.mean(values[following]) - np.mean(values[playing]))
    return gap <= 0 and trained[following] >= rule["min_epochs"]


def elo_of(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """The Elo difference a series' score implies, and its standard error (by the delta method,
    from the games' own spread). A clean sweep is clipped to a score of 1 / (2 games) from the
    end."""
    n = wins + draws + losses
    if n == 0:
        raise ValueError("no games")
    score = (wins + 0.5 * draws) / n
    clipped = min(max(score, 0.5 / n), 1 - 0.5 / n)
    elo = -400 * math.log10(1 / clipped - 1)
    variance = (
        wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2
    ) / n
    se = math.sqrt(variance / n) * 400 / (math.log(10) * clipped * (1 - clipped))
    return elo, se


# the outside world: the queue, train.py, the benchmark and the league


class Queue(Protocol):
    def submit(self, about: str, estimate_minutes: int, command: str) -> None: ...


class Runner(Protocol):
    def train(self, args: list[str], log: Path) -> None:
        """Runs train.py with `args`, raising if it fails."""
        ...

    def bench(self, nets: list[str], args: list[str], out: Path) -> dict[str, dict]:
        """Runs bench_nets.py on `nets`, returning each one's result by spec."""
        ...


class League(Protocol):
    def register(
        self, name: str, checkpoint: str, net: str, run: str, sims: Sequence[int]
    ) -> None: ...

    def match(self, a: str, b: str, games: int) -> int: ...

    def play(self, requests: Sequence[int], seconds: float, log: Path) -> bool:
        """Plays the series asked for until they're done (True), or until the player gives
        way or runs out of time (False)."""
        ...

    def record(self, a: str, b: str) -> tuple[int, int, int]:
        """`a`'s wins, draws and losses against `b`."""
        ...

    def retire(self, name: str) -> None: ...


class SlickQueue:
    """slick's GPU queue (`slick queue`), from the slick task's worktree."""

    def __init__(self, config: Config):
        self.config = config

    def submit(self, about: str, estimate_minutes: int, command: str) -> None:
        args = [
            "slick",
            "queue",
            "--about",
            about,
            "--estimate",
            f"{estimate_minutes}m",
            "--priority",
            str(self.config.priority),
        ]
        if self.config.resource:
            args += ["--on", self.config.resource]
        subprocess.run(
            [*args, "--", command],
            cwd=os.path.expanduser(self.config.queue_cwd),
            check=True,
        )


def gpu_env() -> dict[str, str]:
    """This process's environment for a child on the GPU (the controller itself stays off it),
    taking GPU memory as it needs it rather than most of it at once."""
    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONUNBUFFERED": "1",
    }
    env.pop("JAX_PLATFORMS", None)
    return env


class LocalRunner:
    """train.py and bench_nets.py, in this process's environment (the frozen worktree's)."""

    def __init__(self, config: Config):
        self.cwd = os.path.expanduser(config.worktree)

    def train(self, args: list[str], log: Path) -> None:
        with open(log, "a") as out:
            subprocess.run(
                [sys.executable, "examples/othello/train.py", *args],
                cwd=self.cwd,
                env=gpu_env(),
                stdout=out,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def bench(self, nets: list[str], args: list[str], out: Path) -> dict[str, dict]:
        out.unlink(missing_ok=True)
        subprocess.run(
            [
                sys.executable,
                "examples/othello/bench_nets.py",
                *nets,
                *args,
                "--json",
                str(out),
            ],
            cwd=self.cwd,
            env=gpu_env(),
            check=True,
        )
        results = [json.loads(line) for line in out.read_text().splitlines()]
        return {r["net"]: r for r in results}


class LocalLeague:
    """The Elo league, through its frozen worktree's elo_league.py (see OPERATING.md): changes
    by its commands, results read from its store."""

    def __init__(self, config: Config):
        self.cwd = os.path.expanduser(config.league_worktree)
        self.root = os.path.expanduser(config.league_root)

    def command(self, *args: str) -> str:
        out = subprocess.run(
            [
                ".venv/bin/python",
                "examples/othello/elo_league.py",
                "--root",
                self.root,
                *args,
            ],
            cwd=self.cwd,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode:
            raise RuntimeError(
                f"elo_league.py {' '.join(args)} failed: {out.stderr[-2000:]}"
            )
        return out.stdout.strip()

    def register(
        self, name: str, checkpoint: str, net: str, run: str, sims: Sequence[int]
    ) -> None:
        with contextlib.closing(self.db()) as db:
            if db.execute("SELECT 1 FROM networks WHERE name = ?", (name,)).fetchone():
                # registered before a retry
                return
        self.command(
            "add-network",
            name,
            checkpoint,
            "--net",
            net,
            "--run",
            run,
            "--sims",
            *map(str, sims),
        )

    def match(self, a: str, b: str, games: int) -> int:
        return int(self.command("match", a, b, "--games", str(games)).lstrip("#"))

    def db(self) -> sqlite3.Connection:
        return sqlite3.connect(
            f"file:{self.root}/league.db?mode=ro", uri=True, timeout=30
        )

    def pending(self, requests: Sequence[int]) -> int:
        marks = ",".join("?" * len(requests))
        with contextlib.closing(self.db()) as db:
            return db.execute(
                f"SELECT COUNT(*) FROM match_requests WHERE state = 'pending' AND id IN ({marks})",
                list(requests),
            ).fetchone()[0]

    def play(self, requests: Sequence[int], seconds: float, log: Path) -> bool:
        """Runs the league's player on the GPU until the series are done: it plays requested
        series only, others' too if theirs come first, and gives way to what's queued at this
        entry's priority or above. Stops it once ours are done."""
        env = {
            **gpu_env(),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_COMPILATION_CACHE_DIR": f"{self.root}/compiled",
        }
        started = time.monotonic()
        with open(log, "a") as out:
            player = subprocess.Popen(
                [
                    "taskset",
                    "-c",
                    os.environ.get("LEAGUE_CORES", "0-7"),
                    ".venv/bin/python",
                    "examples/othello/elo_league.py",
                    "--root",
                    self.root,
                    "play",
                    "--batch-seconds",
                    "300",
                    "--requests-only",
                ],
                cwd=self.cwd,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
            while player.poll() is None:
                if not self.pending(requests) or time.monotonic() - started > seconds:
                    break
                time.sleep(20)
            if player.poll() is None:
                self.command("stop")
                try:
                    player.wait(timeout=600)
                finally:
                    # the stop flag would stop the next player too
                    with contextlib.closing(
                        sqlite3.connect(f"{self.root}/league.db", timeout=30)
                    ) as db:
                        db.execute("UPDATE meta SET value = 'false' WHERE key = 'stop'")
                        db.commit()
        return not self.pending(requests)

    def record(self, a: str, b: str) -> tuple[int, int, int]:
        with contextlib.closing(self.db()) as db:
            ids = dict(
                db.execute(
                    "SELECT name, id FROM players WHERE name IN (?, ?)", (a, b)
                ).fetchall()
            )
            a_id, b_id = ids[a], ids[b]
            counts = [0, 0, 0]
            for first, result, games in db.execute(
                "SELECT first_id, result, COUNT(*) FROM games WHERE "
                "(first_id = ? AND second_id = ?) OR (first_id = ? AND second_id = ?) "
                "GROUP BY first_id, result",
                (a_id, b_id, b_id, a_id),
            ):
                # the result for the first player to move
                ours = result if first == a_id else -result
                counts[1 - ours] += games
        wins, draws, losses = counts
        return wins, draws, losses

    def retire(self, name: str) -> None:
        self.command("retire", name)


# the run's state


@dataclass
class Head:
    """Where the run's kept line of states has got to: the state to go on from (None before
    the first step), its epochs, search and the network playing."""

    state: str | None
    epoch: int
    search: Search
    playing: int


class Controller:
    """A controlled run in `run_dir` (see the module's docstring): its state is `run_dir`'s
    controller.json, its decisions are appended to decisions.jsonl."""

    def __init__(
        self,
        run_dir: str | Path,
        queue: Queue | None = None,
        runner: Runner | None = None,
        league: League | None = None,
    ):
        self.dir = Path(run_dir).resolve()
        self.path = self.dir / "controller.json"
        self.data = json.loads(self.path.read_text())
        self.config = Config.from_json(self.data["config"])
        self.queue = queue or SlickQueue(self.config)
        self.runner = runner or LocalRunner(self.config)
        self.league = league or LocalLeague(self.config)

    @classmethod
    def init(cls, run_dir: str | Path, config: Config, **kwargs) -> "Controller":
        run_dir = Path(run_dir).resolve()
        if (run_dir / "controller.json").exists():
            raise FileExistsError(f"{run_dir} already has a controlled run")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        data = {
            "config": config.to_json(),
            "period": 0,
            "head": asdict(Head(None, 0, config.search, 0)),
            "steps": {},
            "bench": None,
            "spent": {},
            "kept": {},
            "stopped": False,
            "done": None,
        }
        (run_dir / "controller.json").write_text(json.dumps(data, indent=1))
        return cls(run_dir, **kwargs)

    # state

    @contextlib.contextmanager
    def locked(self) -> Generator[None]:
        """Holds the run's lock, with its state read fresh, and saves it after."""
        with open(self.dir / "controller.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.data = json.loads(self.path.read_text())
            yield
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1))
            os.replace(tmp, self.path)

    @property
    def head(self) -> Head:
        head = dict(self.data["head"])
        return Head(**{**head, "search": Search(**head["search"])})

    def set_head(self, head: Head) -> None:
        self.data["head"] = asdict(head)

    @property
    def bench(self) -> dict[str, dict]:
        return self.data["bench"]

    def step(self, name: str) -> dict:
        return self.data["steps"][name]

    def period_steps(self, period: int) -> dict[str, dict]:
        prefix = f"p{period:03d}-"
        return {k: v for k, v in self.data["steps"].items() if k.startswith(prefix)}

    def decide(self, record: dict) -> None:
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
        with open(self.dir / "decisions.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    # the steps

    def start(self) -> None:
        """Queues the run's first step: the networks' benchmark."""
        with self.locked():
            if self.data["steps"]:
                raise RuntimeError("the run has started already")
            self.add({"name": "bench", "kind": "bench"})

    def stop(self) -> None:
        with self.locked():
            self.data["stopped"] = True

    def retry(self, name: str) -> None:
        """Queues a step that failed (or was cancelled) again: a training step goes on from its
        newest saved state, if it saved one."""
        with self.locked():
            step = self.step(name)
            if step["status"] == "done":
                raise RuntimeError(f"{name} is done")
            self.submit(step)

    def add(self, step: dict) -> None:
        """Adds a step to the run and queues it (with the lock held)."""
        if step["name"] in self.data["steps"]:
            raise RuntimeError(f"step {step['name']} exists already")
        self.data["steps"][step["name"]] = step
        self.submit(step)

    def submit(self, step: dict) -> None:
        step["status"] = "queued"
        command = (
            f"cd {shlex.quote(os.path.expanduser(self.config.worktree))} && "
            f"taskset -c {shlex.quote(self.config.cores)} "
            f".venv/bin/python examples/othello/controller.py step "
            f"{shlex.quote(str(self.dir))} {shlex.quote(step['name'])}"
        )
        self.queue.submit(
            f"{self.config.name} {step['name']}: {step.get('about', step['kind'])}",
            self.estimate(step),
            command,
        )

    def estimate(self, step: dict) -> int:
        """Minutes a step should hold the GPU: its benchmarked compute, with half again for what
        the benchmark leaves out, and startup."""
        if step["kind"] == "train":
            return max(5, round(1.5 * step["compute"]["seconds"] / 60 + 3))
        if step["kind"] == "bench":
            return 5
        return 20

    def run(self, name: str) -> None:
        """Runs a step (in its queue entry), records what it did, and queues what's next."""
        with self.locked():
            step = self.step(name)
            if step["status"] == "done":
                raise RuntimeError(f"{name} is done")
            step["status"] = "running"
        try:
            if step["kind"] == "bench":
                result = self.run_bench()
            elif step["kind"] == "train":
                result = self.run_train(step)
            else:
                result = self.run_series(step)
        except BaseException as e:
            with self.locked():
                self.step(name)["status"] = "failed"
                self.step(name)["error"] = repr(e)
            raise
        with self.locked():
            step = self.step(name)
            if result.get("requeue"):
                # the league's player gave way: the step goes back in the queue
                self.submit(step)
                return
            step.update(result, status="done")
            self.after(step)
            if not self.data["stopped"]:
                for new in self.next_steps():
                    self.add(new)

    def run_bench(self) -> dict:
        args = ["--train-batch", str(self.config.train_batch)]
        if "--remat" in self.config.train_args:
            args.append("--remat")
        bench = self.runner.bench(
            self.config.nets, args, self.dir / "logs" / "bench.jsonl"
        )
        for net in self.config.nets:
            if bench[net]["forward_batch"] != ENVS:
                raise ValueError(
                    f"{net} was benchmarked at batch {bench[net]['forward_batch']}"
                )
        return {"bench": bench}

    def train_args(self, step: dict) -> list[str]:
        config = self.config
        search = Search(**step["search"])
        args = ["--net", config.nets[0]]
        for net in config.nets[1:]:
            args += ["--grow", net]
        if len(config.nets) > 1:
            args.append("--grow-manual")
            if "--engine-eval-every" not in config.train_args:
                args += ["--engine-eval-every", "1"]
        args += search.train_args()
        directory = self.dir / step["name"]
        args += [
            "--buffer",
            str(config.buffer),
            "--train-steps",
            str(config.train_steps),
            "--train-batch",
            str(config.train_batch),
            "--epochs",
            str(step["last_epoch"]),
            "--ckpt-dir",
            str(directory),
            "--save-state-at",
            str(step["last_epoch"]),
            "--save-state-every",
            str(config.save_state_every),
            "--name",
            f"{config.name}-{step['name']}",
        ]
        if config.monitor:
            args.append("--monitor")
        if any(directory.glob("state-*.npz")):
            # a retried step goes on from where it stopped
            args.append("--resume")
        elif step["start"] is not None:
            args += ["--branch-from", step["start"]]
            if step["take_over"]:
                args.append("--take-over")
        return [*args, *config.train_args]

    def run_train(self, step: dict) -> dict:
        directory = self.dir / step["name"]
        if directory.exists() and not any(directory.glob("state-*.npz")):
            # a try that stopped before saving a state: start the step again
            shutil.rmtree(directory)
        self.runner.train(
            self.train_args(step), self.dir / "logs" / f"{step['name']}.log"
        )
        state = directory / f"state-{step['last_epoch']}.npz"
        if not state.exists():
            raise FileNotFoundError(f"{step['name']} saved no state at {state}")
        # the states saved on the way, for a retry, aren't needed any more
        for saved in directory.glob("state-*.npz"):
            if saved != state:
                saved.unlink()
        return {"state": str(state)}

    def checkpoint(self, step: dict, net: int) -> str:
        """The last checkpoint of a network (index into `config.nets`) in a training step."""
        directory = self.dir / step["name"]
        if len(self.config.nets) > 1:
            directory = directory / net_name(self.config.nets[net])
        return str(directory / f"{step['last_epoch'] - 1}.eqx")

    def player(self, step: dict, net: int) -> str:
        """The league's name for a network's last checkpoint in a training step."""
        return (
            f"{net_name(self.config.nets[net])}-{self.config.name}-{step['name']}"
            f"-e{step['last_epoch']}"
        )

    def run_series(self, step: dict) -> dict:
        """Plays a rating step's series in the league, and reads each one's result."""
        league = self.league
        if not step.get("requests"):
            requests = []
            for series in step["series"]:
                for side in ("a", "b"):
                    player = series[side]
                    if not player.get("registered"):
                        league.register(
                            player["network"],
                            player["checkpoint"],
                            player["net"],
                            player["run"],
                            [player["sims"]],
                        )
                        player["registered"] = True
                requests.append(
                    league.match(
                        series["a"]["name"], series["b"]["name"], self.config.games
                    )
                )
            with self.locked():
                self.step(step["name"]).update(series=step["series"], requests=requests)
            step["requests"] = requests
        done = league.play(
            step["requests"],
            self.config.rate_hours * 3600,
            self.dir / "logs" / f"{step['name']}.log",
        )
        if not done:
            return {"requeue": True}
        results = []
        for series in step["series"]:
            wins, draws, losses = league.record(
                series["a"]["name"], series["b"]["name"]
            )
            elo, se = elo_of(wins, draws, losses)
            results.append(
                {"wins": wins, "draws": draws, "losses": losses, "elo": elo, "se": se}
            )
            for side in ("a", "b"):
                league.retire(series[side]["name"])
        return {"results": results}

    # deciding what's next

    def after(self, step: dict) -> None:
        """Takes in what a step did (with the lock held): the benchmark, the head of the kept
        line after a trunk, a series' decision."""
        if step["kind"] == "bench":
            self.data["bench"] = step["bench"]
            return
        if step["kind"] == "train":
            self.data["spent"] = added(self.data["spent"], step["compute"])
            if step["branch"] is None:
                self.data["kept"] = added(self.data["kept"], step["compute"])
                head = self.head
                self.set_head(
                    Head(step["state"], step["last_epoch"], head.search, head.playing)
                )
            return
        period = self.data["period"]
        result = step["results"][0]
        if step["kind"] == "size":
            switch = result["elo"] >= self.config.switch_margin
            step["switch"] = switch
            self.decide(
                {
                    "period": period,
                    "question": "size",
                    "a": step["series"][0]["a"]["name"],
                    "b": step["series"][0]["b"]["name"],
                    **result,
                    "margin": self.config.switch_margin,
                    "decision": "switch" if switch else "stay",
                }
            )
            return
        # sims: the run goes on from one branch
        steps = self.period_steps(period)
        stay, double = steps[f"p{period:03d}-stay"], steps[f"p{period:03d}-double"]
        doubled = result["elo"] >= self.config.sims_margin
        kept, lost = (double, stay) if doubled else (stay, double)
        self.data["kept"] = added(self.data["kept"], kept["compute"])
        self.set_head(
            Head(
                kept["state"],
                kept["last_epoch"],
                Search(**kept["search"]),
                kept["playing"],
            )
        )
        self.decide(
            {
                "period": period,
                "question": "sims",
                "a": step["series"][0]["a"]["name"],
                "b": step["series"][0]["b"]["name"],
                **result,
                "others": step["results"][1:],
                "margin": self.config.sims_margin,
                "decision": "double" if doubled else "stay",
                "search": kept["search"],
            }
        )
        if self.config.prune_states:
            for gone in (lost["state"], kept["start"]):
                if gone is not None:
                    Path(gone).unlink(missing_ok=True)
        self.data["period"] = period + 1
        spent_hours = self.data["spent"]["seconds"] / 3600
        if (
            self.config.max_periods is not None
            and self.data["period"] >= self.config.max_periods
        ) or (
            self.config.budget_hours is not None
            and spent_hours >= self.config.budget_hours
        ):
            self.data["done"] = (
                f"after {self.data['period']} periods, "
                f"{spent_hours:.2f} benchmarked GPU hours spent"
            )

    def next_steps(self) -> list[dict]:
        """The steps to queue now that the last one is done (with the lock held)."""
        if self.data["done"] is not None:
            return []
        period = self.data["period"]
        prefix = f"p{period:03d}-"
        steps = self.period_steps(period)
        if any(s["status"] != "done" for s in steps.values()):
            # e.g. one branch done, the other still to come
            return []
        head = self.head
        if not steps:
            return [self.train_step(prefix + "trunk", head, head.search, None, False)]
        if prefix + "sims" in steps:
            raise RuntimeError(f"period {period} is over, but the run hasn't moved on")
        if prefix + "stay" in steps:
            return [self.sims_series(period)]
        size = steps.get(prefix + "size")
        if size is None and caught_up(read_meta(head.state)):  # pyright: ignore[reportArgumentType]
            return [self.size_series(period)]
        take_over = bool(size and size["switch"])
        stay, double = branch_epochs(
            self.config, head.search, head.playing + take_over, self.bench
        )
        return [
            self.train_step(
                prefix + "stay", head, head.search, "stay", take_over, stay
            ),
            self.train_step(
                prefix + "double",
                head,
                head.search.doubled(),
                "double",
                take_over,
                double,
            ),
        ]

    def train_step(
        self,
        name: str,
        head: Head,
        search: Search,
        branch: str | None,
        take_over: bool,
        epochs: int | None = None,
    ) -> dict:
        playing = head.playing + take_over
        if epochs is None:
            epochs = max(
                1,
                math.ceil(
                    self.config.trunk_turnovers * turnover_epochs(self.config, search)
                ),
            )
        compute = scaled(
            epoch_compute(self.config, search, playing, self.bench), epochs
        )
        what = {None: "trunk", "stay": "stay branch", "double": "double branch"}[branch]
        return {
            "name": name,
            "kind": "train",
            "about": (
                f"{what}, {net_name(self.config.nets[playing])} playing at "
                f"{search.label()}, epochs {head.epoch}-{head.epoch + epochs}"
            ),
            "branch": branch,
            "start": head.state,
            "take_over": take_over,
            "search": asdict(search),
            "playing": playing,
            "first_epoch": head.epoch,
            "last_epoch": head.epoch + epochs,
            "compute": compute,
        }

    def network_player(self, step: dict, net: int, sims: int) -> dict:
        network = self.player(step, net)
        return {
            "network": network,
            "name": f"{network}@{sims}",
            "checkpoint": self.checkpoint(step, net),
            "net": self.config.nets[net],
            "run": f"{self.config.name}-{step['name']}",
            "sims": sims,
        }

    def size_series(self, period: int) -> dict:
        """The next network against the one playing, at equal search time a move."""
        head = self.head
        trunk = self.step(f"p{period:03d}-trunk")
        playing, following = head.playing, head.playing + 1
        ratio = (
            self.bench[self.config.nets[following]]["forward_ms"]
            / self.bench[self.config.nets[playing]]["forward_ms"]
        )
        sims = self.config.play_sims
        return {
            "name": f"p{period:03d}-size",
            "kind": "size",
            "about": f"{net_name(self.config.nets[following])} vs "
            f"{net_name(self.config.nets[playing])} at equal search time",
            "series": [
                {
                    "a": self.network_player(trunk, following, sims),
                    "b": self.network_player(trunk, playing, round(sims * ratio)),
                }
            ],
        }

    def sims_series(self, period: int) -> dict:
        """`double`'s networks against `stay`'s, the one playing first (it decides)."""
        steps = self.period_steps(period)
        stay, double = steps[f"p{period:03d}-stay"], steps[f"p{period:03d}-double"]
        sims = self.config.play_sims
        playing = stay["playing"]
        nets = [i for i in (playing, playing + 1) if i < len(self.config.nets)]
        return {
            "name": f"p{period:03d}-sims",
            "kind": "sims",
            "about": f"double vs stay, epochs {double['last_epoch']} and {stay['last_epoch']}",
            "series": [
                {
                    "a": self.network_player(double, net, sims),
                    "b": self.network_player(stay, net, sims),
                }
                for net in nets
            ],
        }

    # reporting

    def status(self) -> str:
        head = self.head
        config = self.config
        lines = [
            f"{config.name}: period {self.data['period']}, epoch {head.epoch}, "
            f"{net_name(config.nets[head.playing])} playing at {head.search.label()}"
            + (", stopped" if self.data["stopped"] else "")
            + (f", done {self.data['done']}" if self.data["done"] else ""),
        ]
        for which in ("spent", "kept"):
            c = self.data[which]
            if c:
                lines.append(
                    f"  {which}: {c['seconds'] / 3600:.2f} benchmarked GPU hours, "
                    f"{c['selfplay_evals'] / 1e9:.2f}G self-play evaluations, "
                    + ", ".join(
                        f"{k} {v / 1e9:.2f}G samples"
                        for k, v in c["train_samples"].items()
                    )
                )
        for name, step in self.data["steps"].items():
            lines.append(
                f"  {name}: {step['status']}, {step.get('about', step['kind'])}"
            )
            for series, result in zip(
                step.get("series", []), step.get("results", []), strict=False
            ):
                lines.append(
                    f"    {series['a']['name']} vs {series['b']['name']}: "
                    f"{result['elo']:+.0f} ± {result['se']:.0f} "
                    f"({result['wins']}-{result['draws']}-{result['losses']})"
                )
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Controls a long Othello run step by step (see the module's docstring)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="set up a controlled run")
    init.add_argument("run")
    init.add_argument("--config", required=True, help="the run's settings, JSON")
    for name, text in [
        ("start", "queue the first step"),
        ("status", "where the run is"),
        ("stop", "queue nothing after the steps queued now"),
    ]:
        commands.add_parser(name, help=text).add_argument("run")
    for name, text in [
        ("step", "run a step (the queue runs this)"),
        ("retry", "queue a step that failed again"),
    ]:
        command = commands.add_parser(name, help=text)
        command.add_argument("run")
        command.add_argument("name")
    args = parser.parse_args(argv)

    if args.command == "init":
        try:
            config = Config.from_json(json.loads(Path(args.config).read_text()))
        except (TypeError, ValueError) as e:
            parser.error(f"--config: {e}")
        Controller.init(args.run, config)
        print(f"{args.run}: ready to start")
        return
    controller = Controller(args.run)
    if args.command == "start":
        controller.start()
    elif args.command == "status":
        print(controller.status())
    elif args.command == "stop":
        controller.stop()
    elif args.command == "step":
        controller.run(args.name)
    elif args.command == "retry":
        controller.retry(args.name)


if __name__ == "__main__":
    main()
