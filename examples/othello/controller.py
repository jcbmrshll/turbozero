"""A controller for a long Othello run: train.py with --grow, going on in steps, each its own
entry in slick's GPU queue, the controller deciding between steps what the run does next from
games between its networks in the Elo league (see README.md, "The run controller").

Every period:

1. trunk: the run trains on at its current settings, `trunk_turnovers` turnovers of its replay
   buffer, the next network training beside the one playing, on its games. No network takes
   over during a trunk (--grow-manual).
2. branches: two branches of the run at equal benchmarked compute, in which no network takes
   over either: `double` searches twice the iterations (both budgets, with playout cap
   randomization) for `branch_turnovers` turnovers of its buffer, and `stay` keeps the search
   for as many epochs as cost the same.
3. test: the branches' networks play series, with their weights averaged over the last
   `average_epochs` epochs (train.py --average-epochs). Growing comes first: if the next
   network, as it trained in `stay`, scores `margin_se` standard errors (of its series) or
   more against `stay`'s playing one, it takes over from `stay`'s state at the next trunk
   (--take-over), at `stay`'s search. Otherwise the search doubles if `double`'s playing
   network scores `margin_se` standard errors or more against `stay`'s, and the run goes on
   from `double`; else from `stay`. Sims only go up while the next network doesn't pay its way.
   Longer trunks (`trunk_turnovers`) test less often.
   With `clock_sims`, the next network plays the one playing at equal time too, each at the
   sims that take it about 1 s of one side's clock a game on the CPU: for the record, no
   decision rests on it.
4. ladder: the network the run kept climbs the league's Edax players (`ladder`), a series each,
   from the rung the last period lost to, on to the next while it scores over half: its games
   put the kept line on the league's scale.
   With `clock_sims`, it climbs a second ladder of its own at its clock's sims (pNNN-clock-ladder).

Compute is counted two ways: in network evaluations (self-play's positions evaluated, and the
samples each network trains on), and in GPU time (see `Costs`): each network's forward pass at
self-play's batch and its training step at training's (bench_nets.py, run first), plus the
search's own bookkeeping per cycle, measured from the run's epochs. Compiling and tests aren't
counted. A branch that loses counts in what the run spent, not in the line of states it kept.

    uv run examples/othello/controller.py init runs/big --config big.json
    uv run examples/othello/controller.py start runs/big
    uv run examples/othello/controller.py status runs/big
    uv run examples/othello/controller.py stop runs/big          # after the step that runs now
    uv run examples/othello/controller.py retry runs/big p003-double
    uv run examples/othello/controller.py advance runs/big   # what a step's end queues

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
from dataclasses import asdict, dataclass, field, replace
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
    "--grow-metric",
    "--grow-margin",
    "--grow-guard",
    "--grow-switch-at",
    "--grow-train-steps",
    "--average-epochs",
    "--selfplay-average",
    "--sims",
    "--sims-schedule",
    "--playout-cap",
    "--selfplay-cycles",
    "--advance-every",
    "--buffer",
    "--lr",
    "--lr-final",
    "--lr-drops",
    "--lr-plateau",
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


# the Edax players of the league's pool, in climbing order (edax-L28 is retired)
EDAX_LADDER = [
    "edax-L1",
    "edax-L2",
    "edax-L4",
    "edax-L6",
    "edax-L8",
    "edax-L10",
    "edax-L12",
    "edax-L16",
    "edax-L20",
    "edax-L24",
]


# sims a move at which each network size takes about 1 s of one side's clock a game on the
# contest CPU path (bf16 OpenVINO, 8 threads, batched C++ search), from the league's timings.json
# (2026-11-01/02): the equal-time games' search
CLOCK_SIMS = {"4x64": 3000, "6x96": 1100, "8x128": 520, "10x128": 390, "12x144": 250}


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
class Window:
    """The replay window by the positions stored per environment so far, n: the replay-window
    task's rule, KataGo's sublinear shape. Its target is n until `c`, then
    c (1 + beta ((n / c)^alpha - 1) / alpha), which grows as n^alpha with slope `beta` at c (the
    defaults reach about 1125 at 30M positions, an 8x128 run's 274 minutes). The buffer is
    resized once the target is `step` times its capacity or more, to the target rounded up to a
    multiple of `multiple`. With playout cap randomization, n counts the positions stored (the
    full searches'), as KataGo's counts the samples it records."""

    c: int = 256
    alpha: float = 0.75
    beta: float = 0.075
    step: float = 1.25
    multiple: int = 32

    def target(self, n: float) -> float:
        if n <= self.c:
            return n
        return self.c * (1 + self.beta * ((n / self.c) ** self.alpha - 1) / self.alpha)

    def capacity(self, n: float, current: int) -> int:
        """The buffer's capacity after n positions per environment, from `current`."""
        target = self.target(n)
        if target < self.step * current:
            return current
        return math.ceil(target / self.multiple) * self.multiple


@dataclass(frozen=True)
class Config:
    """A controlled run's settings (the JSON config's keys).

    Attributes:
        name: the run's name: its steps' names and league players start with it
        nets: train.py --net specs, smallest first: the first plays first, each takes over in
            turn (train.py --grow), as KataGo's rule has it (see the module's docstring)
        search: the search to start with (see `Search`)
        worktree: the frozen worktree the steps run in, with a CUDA environment (./install.sh)
        queue_cwd: the slick task's worktree, where `slick queue` queues the steps
        train_args: train.py's other arguments, e.g. ["--seed", "1"]. Not those the controller
            sets (`OWN_ARGS`)
        lr: the learning rate, constant through the run, as KataGo held its (train.py --lr and
            --lr-final)
        grow_margin: the lead in training loss for train.py's growth rule to report the next
            network caught up, a fraction of the playing one's (--grow-margin; it's logged,
            growth_caught_up, but nothing switches on it: the period's test decides)
        grow_guard: (optional) a metric the next must also be no worse on (--grow-guard)
        buffer: train.py --buffer, positions kept per environment: the replay window, or with
            `window` the one to start with (its `c`)
        window: (optional) grow the replay window by the positions stored so far (see
            `Window`), checked at each step's start; without, `buffer` throughout
        train_steps: train.py --train-steps, training steps an epoch
        train_batch: train.py --train-batch
        trunk_turnovers: a period's trunk, in turnovers of the replay buffer at its search
        branch_turnovers: the `double` branch's length, in turnovers of its replay buffer
        games: games a series (even: both colours of each opening)
        play_sims: the league's search for the series, iterations a move
        margin_se: the standard errors of its own series a challenger must score above the one
            it challenges: the next network as it trained in `stay` against the one playing
            there (to grow), and `double`'s playing network against `stay`'s (to double)
        average_epochs: (optional) train.py --average-epochs: every step keeps each network's
            weights averaged over about this many epochs, and the league rates the averages
            (the test's and the ladder's players). None rates the networks' weights as they are
        selfplay_average: self-play plays with the average too (train.py --selfplay-average)
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
        ladder: the league players each period's kept network climbs, in order, to put the kept
            line on the league's scale: it plays the rung the last period lost to, and the next
            while it scores over half (see `Controller.run_ladder`). None for no ladder
        clock_sims: (optional) sims a move by network size (`net_name`) at equal time on the CPU
            (`CLOCK_SIMS`: about 1 s of one side's clock a game). With them, each test also plays
            the next network against the one playing at their clock's sims, and each period's
            kept network climbs a second ladder at its own: for the record, deciding nothing
        clock_tree_factor: the equal-time players' tree nodes per iteration (league
            add-player --tree-factor): at thousands of sims, the default 2 costs strength
    """

    name: str
    nets: list[str]
    search: Search
    worktree: str
    queue_cwd: str
    train_args: list[str] = field(default_factory=list)
    lr: float = 5e-4
    grow_margin: float = 0.0
    grow_guard: str | None = "engine_value_mse"
    buffer: int = 1125
    window: Window | None = None
    train_steps: int = 256
    train_batch: int = 4096
    trunk_turnovers: float = 4.0
    branch_turnovers: float = 1.0
    games: int = 400
    play_sims: int = 128
    margin_se: float = 2.0
    average_epochs: float | None = 4.0
    selfplay_average: bool = False
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
    ladder: list[str] | None = field(default_factory=lambda: list(EDAX_LADDER))
    clock_sims: dict[str, int] | None = field(default_factory=lambda: dict(CLOCK_SIMS))
    clock_tree_factor: int = 4

    def __post_init__(self):
        if not self.nets:
            raise ValueError("a run needs at least one network")
        names = [net_name(n) for n in self.nets]
        if len(set(names)) != len(names):
            raise ValueError(f"the networks' sizes must differ: {names}")
        own = [a for a in self.train_args if a.split("=")[0] in OWN_ARGS]
        if own:
            raise ValueError(f"the controller sets these itself: {own}")
        if self.lr <= 0:
            raise ValueError(f"the learning rate must be positive, got {self.lr}")
        if not 0 <= self.grow_margin < 1:
            raise ValueError(
                f"grow_margin is a fraction in [0, 1), got {self.grow_margin}"
            )
        if self.trunk_turnovers <= 0 or self.branch_turnovers <= 0:
            raise ValueError("trunk_turnovers and branch_turnovers must be positive")
        if self.margin_se < 0:
            raise ValueError(f"margin_se can't be negative, got {self.margin_se}")
        if self.average_epochs is not None and self.average_epochs < 1:
            raise ValueError(
                f"average_epochs is at least 1 epoch, got {self.average_epochs}"
            )
        if self.selfplay_average and self.average_epochs is None:
            raise ValueError("selfplay_average needs average_epochs")
        if self.games < 2 or self.games % 2:
            raise ValueError(f"games must be even, got {self.games}")
        if self.ladder is not None and (
            not self.ladder or len(set(self.ladder)) != len(self.ladder)
        ):
            raise ValueError(f"the ladder needs distinct rungs, got {self.ladder}")
        if self.clock_sims is not None:
            missing = [n for n in names if n not in self.clock_sims]
            if missing:
                raise ValueError(f"clock_sims has no sims for {missing}")

    @classmethod
    def from_json(cls, data: dict) -> "Config":
        window = data.get("window")
        return cls(
            **{
                **data,
                "search": Search(**data["search"]),
                "window": Window(**window) if window is not None else None,
            }
        )

    def to_json(self) -> dict:
        return asdict(self)


# compute


@dataclass(frozen=True)
class Costs:
    """What an epoch costs in GPU time: each network's forward pass per search cycle at
    self-play's batch and its training step at training's (`bench`, from bench_nets.py), and
    the search's own bookkeeping per cycle, by search (`overhead_ms`, by `Search.label`: the
    time self-play's epochs took per cycle less the playing network's forward pass, measured
    from the run's steps; the benchmark leaves it out). A search not measured yet takes the
    overhead of the measured one with the nearest budget, none before any."""

    bench: dict[str, dict]
    overhead_ms: dict[str, float] = field(default_factory=dict)

    def overhead(self, search: Search) -> float:
        if search.label() in self.overhead_ms:
            return self.overhead_ms[search.label()]
        if not self.overhead_ms:
            return 0.0

        def full(label: str) -> int:
            # 64s, or pcr25x127-31s: the full budget
            found = re.match(r"(?:pcr\d+x)?(\d+)", label)
            assert found is not None, label
            return int(found.group(1))

        nearest = min(
            self.overhead_ms,
            key=lambda label: (abs(math.log(full(label) / search.full)), -full(label)),
        )
        return self.overhead_ms[nearest]

    def cycle_ms(self, net: str, search: Search) -> float:
        """A search cycle's time with `net` playing: its forward pass, and the search's own."""
        return self.bench[net]["forward_ms"] + self.overhead(search)

    def train_ms(self, net: str) -> float:
        return self.bench[net]["train_step_ms"]


def epoch_compute(
    config: Config, search: Search, playing: int, costs: Costs
) -> dict[str, Any]:
    """An epoch's compute: the network playing (index into `config.nets`) evaluates every
    environment's position each search cycle and takes `config.train_steps` training steps, and
    the next (if any) takes `waiting_steps`.

    Returns:
        dict: `seconds`, GPU time (see `Costs`); `selfplay_evals`, positions evaluated in
            self-play; `train_samples`, by network
    """
    cycles = search.epoch_cycles()
    steps = {playing: config.train_steps}
    waiting = waiting_steps(config, search, playing, costs)
    if waiting is not None:
        steps[playing + 1] = waiting
    seconds = cycles * costs.cycle_ms(config.nets[playing], search) / 1e3 + sum(
        n * costs.train_ms(config.nets[i]) / 1e3 for i, n in steps.items()
    )
    return {
        "seconds": seconds,
        "selfplay_evals": cycles * ENVS,
        "train_samples": {
            net_name(config.nets[i]): n * config.train_batch for i, n in steps.items()
        },
    }


def regime_seconds(config: Config, search: Search, playing: int, costs: Costs) -> float:
    """GPU seconds of an epoch while network `playing` plays at `search`: its self-play and
    training, and the training of the next network waiting beside it, at the steps
    `waiting_steps` gives it. Solved from the last network back: with E this cost, the waiting
    network takes E / E_next of the steps, so E = own + (E / E_next) * T_next."""
    nets = config.nets

    def train(i: int) -> float:
        return config.train_steps * costs.train_ms(nets[i]) / 1e3

    own = search.epoch_cycles() * costs.cycle_ms(nets[playing], search) / 1e3 + train(
        playing
    )
    if playing + 1 >= len(nets):
        return own
    after = regime_seconds(config, search, playing + 1, costs)
    return own / (1 - train(playing + 1) / after)


def waiting_steps(
    config: Config, search: Search, playing: int, costs: Costs
) -> int | None:
    """The training steps an epoch of the next network while `playing` plays (None for the
    last): the steps its own play would afford for the same compute, `config.train_steps`
    times the ratio of the two regimes' epoch costs (`regime_seconds`, each with its own
    waiting network). So when the period's test plays it against the one playing, it has
    trained about what the run would have spent on it as the one playing, on that one's games:
    the bigger network must beat the smaller one's training at about equal compute."""
    if playing + 1 >= len(config.nets):
        return None
    share = regime_seconds(config, search, playing, costs) / regime_seconds(
        config, search, playing + 1, costs
    )
    return max(1, round(config.train_steps * share))


def selfplay_overhead(
    log: Path, search: Search, net: str, costs: Costs
) -> dict[str, float] | None:
    """The search's own bookkeeping per cycle in a training step, from its log: the median
    epoch's self-play time per cycle less `net`'s forward pass, leaving out each process's
    first epoch, which compiles. None for a step without such epochs."""
    seconds: dict[int, float] = {}
    compiling: set[int] = set()
    fresh = True
    for line in log.read_text().splitlines() if log.exists() else []:
        if line.startswith(("branching from", "resuming from")):
            fresh = True
        found = re.match(r"^Epoch (\d+): \{'selfplay_seconds': '([0-9.]+)'", line)
        if found:
            epoch = int(found.group(1))
            if fresh:
                compiling.add(epoch)
                fresh = False
            # a retried epoch's last try counts
            seconds[epoch] = float(found.group(2))
    kept = [v for e, v in seconds.items() if e not in compiling]
    if not kept:
        return None
    median = float(np.median(kept))
    per_cycle = median / search.epoch_cycles() * 1e3
    return {
        "selfplay_seconds": median,
        "epochs": len(kept),
        "overhead_ms": max(0.0, per_cycle - costs.bench[net]["forward_ms"]),
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


def turnover_epochs(capacity: int, search: Search) -> float:
    """Epochs self-play takes to fill a replay buffer of `capacity` with new positions."""
    return capacity / search.stored_per_epoch()


def branch_epochs(
    config: Config,
    search: Search,
    playing: int,
    costs: Costs,
    capacity: int,
) -> tuple[int, int]:
    """The epochs of a period's two branches: `double`'s, `branch_turnovers` turnovers of its
    buffer, and `stay`'s, those whose GPU time (see `Costs`) is closest to it."""
    double = max(
        1, math.ceil(config.branch_turnovers * turnover_epochs(capacity, search))
    )
    stay_cost = epoch_compute(config, search, playing, costs)["seconds"]
    double_cost = epoch_compute(config, search.doubled(), playing, costs)["seconds"]
    stay = max(1, round(double * double_cost / stay_cost))
    return stay, double


# reading saved states and series


def read_meta(path: str) -> dict:
    """A saved state's metadata (see Trainer.save_state), without loading its arrays."""
    with np.load(path) as saved:
        return json.loads(str(saved["meta"]))


def players(start: dict | None, end: dict, first: int, last: int) -> list[list[int]]:
    """Who played self-play when in a training step from saved state `start` (None from scratch)
    to `end`, epochs `first` to `last`, from their metadata: [network, first epoch, end epoch]
    spans. A network trains from when it joins (the step's start for the two that train then)
    until the next takes over, and the one after the next joins then (see
    core.training.growth.Growth)."""
    if end.get("growth") is None:
        return [[0, first, last]]
    growth = end["growth"]
    before = start["growth"]["playing"] if start is not None else 0
    trained = (
        start["growth"]["trained"]
        if start is not None
        else [0] * len(growth["trained"])
    )
    done = [e - s for e, s in zip(growth["trained"], trained, strict=True)]
    spans, begin, joined = [], first, {before: first, before + 1: first}
    for net in range(before, growth["playing"]):
        stop = joined[net] + done[net]
        spans.append([net, begin, stop])
        begin, joined[net + 2] = stop, stop
    return [*spans, [growth["playing"], begin, last]]


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
        self,
        name: str,
        checkpoint: str,
        net: str,
        run: str,
        sims: Sequence[int],
        tree_factor: int | None = None,
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

    def resume(self, name: str) -> None: ...


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
    """train.py and bench_nets.py, in this process's environment (the frozen worktree's). A
    step's train.py keeps what it compiles in `cache`, a persistent compilation cache the run's
    steps share: the same networks' training and searches compile once, not once a step."""

    def __init__(self, config: Config, cache: Path | None = None):
        self.cwd = os.path.expanduser(config.worktree)
        self.cache = cache

    def train(self, args: list[str], log: Path) -> None:
        env = gpu_env()
        if self.cache is not None:
            env["JAX_COMPILATION_CACHE_DIR"] = str(self.cache)
        with open(log, "a") as out:
            subprocess.run(
                [sys.executable, "examples/othello/train.py", *args],
                cwd=self.cwd,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def bench(self, nets: list[str], args: list[str], out: Path) -> dict[str, dict]:
        out.unlink(missing_ok=True)
        # a process each: in one, the networks' GPU memory adds up, and 12x144's training step
        # ran out of it after three smaller ones (alone it peaks at 4.7 GB)
        for net in nets:
            subprocess.run(
                [
                    sys.executable,
                    "examples/othello/bench_nets.py",
                    net,
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
        self,
        name: str,
        checkpoint: str,
        net: str,
        run: str,
        sims: Sequence[int],
        tree_factor: int | None = None,
    ) -> None:
        with contextlib.closing(self.db()) as db:
            # registered before a retry, or for another series
            known = db.execute(
                "SELECT 1 FROM networks WHERE name = ?", (name,)
            ).fetchone()
        if tree_factor is not None:
            if not known:
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
            # players of their own, NETWORK@SIMS-tFACTOR; adding one again changes nothing
            self.command(
                "add-player", name, "--sims", *map(str, sims), "--tree-factor",
                str(tree_factor),
            )  # fmt: skip
            return
        if known:
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
                time.sleep(2)
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

    def resume(self, name: str) -> None:
        self.command("resume", name)


# the run's state


@dataclass
class Head:
    """Where the run's kept line of states has got to: the state to go on from (None before
    the first step), its epochs, search, the network playing, the positions it has stored per
    environment, its replay buffer's capacity, the ladder's rung the next ladder step starts at
    (an index into the config's `ladder`), whether `playing` takes over from the state's at the
    next step (the last test grew the run), and the clock ladder's rung."""

    state: str | None
    epoch: int
    search: Search
    playing: int
    stored: float
    capacity: int
    rung: int = 0
    take_over: bool = False
    clock_rung: int = 0


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
        self.runner = runner or LocalRunner(self.config, self.dir / "compiled")
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
            "head": asdict(Head(None, 0, config.search, 0, 0.0, config.buffer)),
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

    @property
    def costs(self) -> Costs:
        return self.costs_with()

    def costs_with(
        self, search: Search | None = None, measured: dict | None = None
    ) -> Costs:
        """The run's cost model (see `Costs`): the benchmark, and each search's overhead, the
        median of what its training steps measured (with `measured`, a step's at `search`,
        that hasn't been recorded yet)."""
        by_label: dict[str, list[float]] = {}
        for step in self.data["steps"].values():
            if step.get("measured") is not None:
                label = Search(**step["search"]).label()
                by_label.setdefault(label, []).append(step["measured"]["overhead_ms"])
        if search is not None and measured is not None:
            by_label.setdefault(search.label(), []).append(measured["overhead_ms"])
        return Costs(
            self.bench, {label: float(np.median(v)) for label, v in by_label.items()}
        )

    @property
    def ladder(self) -> list[str]:
        ladder = self.config.ladder
        assert ladder is not None
        return ladder

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

    def advance(self) -> None:
        """Queues what's next, as a step's end does: e.g. after an update of the controller, the
        ladder steps a run from before it is missing."""
        with self.locked():
            if not self.data["stopped"]:
                for new in self.next_steps():
                    self.add(new)

    def save(self, step: dict, *fields: str) -> None:
        """Saves a running step's progress, its `fields`, holding the lock only to write."""
        with self.locked():
            self.step(step["name"]).update({f: step[f] for f in fields})

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
            elif step["kind"] == "ladder":
                result = self.run_ladder(step)
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
        train_args = self.config.train_args
        if "--remat" in train_args:
            args.append("--remat")
        if "--grad-accum" in train_args:
            # its training steps as train.py takes them, in micro-batches
            args += ["--grad-accum", train_args[train_args.index("--grad-accum") + 1]]
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
            # the training loss on the data both train on, logged; none takes over by itself,
            # only once a period's test grows the run (see `after`)
            args += [
                "--grow-metric",
                "loss",
                "--grow-margin",
                f"{config.grow_margin:g}",
                "--grow-manual",
            ]
            if config.grow_guard is not None:
                args += ["--grow-guard", config.grow_guard]
            if "--engine-eval-every" not in config.train_args:
                args += ["--engine-eval-every", "1"]
        if step["waiting_steps"] is not None:
            args += ["--grow-train-steps", str(step["waiting_steps"])]
        if config.average_epochs is not None:
            args += ["--average-epochs", f"{config.average_epochs:g}"]
            if config.selfplay_average:
                args += ["--selfplay-average", f"{config.average_epochs:g}"]
        args += search.train_args()
        directory = self.dir / step["name"]
        args += [
            # constant: a cosine from it to itself
            "--lr",
            f"{config.lr:g}",
            "--lr-final",
            f"{config.lr:g}",
            "--buffer",
            str(step["buffer"]),
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
        log = self.dir / "logs" / f"{step['name']}.log"
        self.runner.train(self.train_args(step), log)
        state = directory / f"state-{step['last_epoch']}.npz"
        if not state.exists():
            raise FileNotFoundError(f"{step['name']} saved no state at {state}")
        # the states saved on the way, for a retry, aren't needed any more
        for saved in directory.glob("state-*.npz"):
            if saved != state:
                saved.unlink()
        start = read_meta(step["start"]) if step["start"] is not None else None
        spans = players(
            start, read_meta(str(state)), step["first_epoch"], step["last_epoch"]
        )
        # what its self-play cost beyond the benchmark, then its compute, at each network's
        # cost while it played
        search = Search(**step["search"])
        measured = selfplay_overhead(
            log, search, self.config.nets[spans[-1][0]], self.costs
        )
        costs = self.costs_with(search, measured)
        compute: dict[str, Any] = {}
        for net, begin, end in spans:
            compute = added(
                compute,
                scaled(epoch_compute(self.config, search, net, costs), end - begin),
            )
        return {
            "state": str(state),
            "players": spans,
            "compute": compute,
            "measured": measured,
        }

    def checkpoint(self, step: dict, net: int) -> str:
        """The checkpoint the league rates of a network (index into `config.nets`) at a
        training step's end: its weights averaged over `average_epochs`, or as they are."""
        directory = self.dir / step["name"]
        if len(self.config.nets) > 1:
            directory = directory / net_name(self.config.nets[net])
        if self.config.average_epochs is not None:
            directory = directory / f"avg{self.config.average_epochs:g}"
        return str(directory / f"{step['last_epoch'] - 1}.eqx")

    def player(self, step: dict, net: int) -> str:
        """The league's name for the checkpoint it rates of a network in a training step."""
        average = (
            f"-avg{self.config.average_epochs:g}"
            if self.config.average_epochs is not None
            else ""
        )
        return (
            f"{net_name(self.config.nets[net])}-{self.config.name}-{step['name']}"
            f"-e{step['last_epoch']}{average}"
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
                            player["tree_factor"],
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
        # a network may play in more than one series
        for name in dict.fromkeys(
            series[side]["name"] for series in step["series"] for side in ("a", "b")
        ):
            league.retire(name)
        return {"results": results}

    def run_ladder(self, step: dict) -> dict:
        """Plays a ladder step's series a rung at a time: the kept network against the rung it
        starts at, then the next while it scores over half, up to the top. Each series is saved
        as it's asked for and as it ends, so a step that gives way goes on where it was.

        Returns:
            dict: `beaten`, the highest rung it scored over half against this time (None for
                none), and `next_rung`, the one the next period's ladder starts at: the rung it
                lost to, or the top
        """
        league, ladder = self.league, self.ladder
        player = step["player"]
        league.register(
            player["network"],
            player["checkpoint"],
            player["net"],
            player["run"],
            [player["sims"]],
            player["tree_factor"],
        )
        # the period's test retired it
        league.resume(player["name"])
        rung = ladder.index(step["start_rung"]) + len(step["results"])
        log = self.dir / "logs" / f"{step['name']}.log"
        while True:
            if len(step["requests"]) == len(step["results"]):
                step["series"].append({"a": player, "b": {"name": ladder[rung]}})
                step["requests"].append(
                    league.match(player["name"], ladder[rung], self.config.games)
                )
                self.save(step, "series", "requests")
            if not league.play(
                step["requests"][-1:], self.config.rate_hours * 3600, log
            ):
                return {"requeue": True}
            wins, draws, losses = league.record(player["name"], ladder[rung])
            elo, se = elo_of(wins, draws, losses)
            step["results"].append(
                {"wins": wins, "draws": draws, "losses": losses, "elo": elo, "se": se}
            )
            self.save(step, "results")
            won = wins + 0.5 * draws > 0.5 * (wins + draws + losses)
            if not won or rung + 1 == len(ladder):
                break
            rung += 1
        league.retire(player["name"])
        if won:
            beaten = ladder[rung]
        else:
            beaten = ladder[rung - 1] if len(step["results"]) > 1 else None
        return {"beaten": beaten, "next_rung": ladder[rung]}

    # deciding what's next

    def after(self, step: dict) -> None:
        """Takes in what a step did (with the lock held): the benchmark, the head of the kept
        line after a trunk (and any network that took over in it), a series' decision."""
        if step["kind"] == "bench":
            self.data["bench"] = step["bench"]
            return
        if step["kind"] == "train":
            self.data["spent"] = added(self.data["spent"], step["compute"])
            if step["branch"] is None:
                self.data["kept"] = added(self.data["kept"], step["compute"])
                self.set_head(
                    replace(
                        self.head,
                        state=step["state"],
                        epoch=step["last_epoch"],
                        playing=step["players"][-1][0],
                        stored=step["stored"],
                        capacity=step["buffer"],
                        take_over=False,
                    )
                )
            return
        if step["kind"] == "ladder":
            rung = self.ladder.index(step["next_rung"])
            self.set_head(
                replace(self.head, clock_rung=rung)
                if step["clock"]
                else replace(self.head, rung=rung)
            )
            self.decide(
                {
                    "period": int(step["name"][1:4]),
                    "question": "clock ladder" if step["clock"] else "ladder",
                    "a": step["player"]["name"],
                    "rungs": [
                        {"b": series["b"]["name"], **result}
                        for series, result in zip(
                            step["series"], step["results"], strict=True
                        )
                    ],
                    "beaten": step["beaten"],
                    "next": step["next_rung"],
                }
            )
            return
        period = self.data["period"]
        steps = self.period_steps(period)
        stay, double = steps[f"p{period:03d}-stay"], steps[f"p{period:03d}-double"]
        results = {
            series["question"]: result
            for series, result in zip(step["series"], step["results"], strict=True)
        }

        def wins(result: dict) -> bool:
            # by margin_se standard errors of its own series
            return result["elo"] >= self.config.margin_se * result["se"]

        # growing first: the search only doubles if the next network doesn't take over
        grow = "size" in results and wins(results["size"])
        doubled = not grow and wins(results["sims"])
        kept, lost = (double, stay) if doubled else (stay, double)
        decision = "grow" if grow else "double" if doubled else "stay"
        step["kept"], step["decision"] = kept["name"], decision
        self.data["kept"] = added(self.data["kept"], kept["compute"])
        playing = kept["playing"] + int(grow)
        self.set_head(
            replace(
                self.head,
                state=kept["state"],
                epoch=kept["last_epoch"],
                search=Search(**kept["search"]),
                playing=playing,
                stored=kept["stored"],
                capacity=kept["buffer"],
                take_over=grow,
            )
        )
        self.decide(
            {
                "period": period,
                "question": "test",
                "series": [
                    {
                        "question": series["question"],
                        "a": series["a"]["name"],
                        "b": series["b"]["name"],
                        **result,
                    }
                    for series, result in zip(
                        step["series"], step["results"], strict=True
                    )
                ],
                "margin_se": self.config.margin_se,
                "decision": decision,
                "search": kept["search"],
                "playing": net_name(self.config.nets[playing]),
            }
        )
        if self.config.prune_states:
            for gone in (lost["state"], kept["start"]):
                if gone is not None:
                    Path(gone).unlink(missing_ok=True)

    def end_period(self) -> None:
        """Moves the run on to its next period, once this one's decisions (and its ladder) are
        done: the run is done after `max_periods`, or once it has spent `budget_hours`."""
        self.data["period"] += 1
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
        if self.config.ladder is not None:
            # each period's ladder, after its test, in order: a run from before the ladder
            # catches up first
            for name, test in sorted(self.data["steps"].items()):
                if test["kind"] != "test" or test["status"] != "done":
                    continue
                ladders = [(f"{name[:4]}-ladder", False)]
                if self.config.clock_sims is not None:
                    ladders.append((f"{name[:4]}-clock-ladder", True))
                waiting = False
                for ladder_name, clock in ladders:
                    ladder = self.data["steps"].get(ladder_name)
                    if ladder is None:
                        return [self.ladder_step(int(name[1:4]), clock)]
                    if ladder["status"] != "done":
                        # the next waits for its rung
                        waiting = True
                        break
                if waiting:
                    break
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
            return [self.train_step(prefix + "trunk", head, head.search, None)]
        if prefix + "test" in steps:
            # its decisions are made, and its ladder (if any) climbed
            self.end_period()
            return self.next_steps()
        if prefix + "stay" in steps:
            return [self.test_series(period)]
        stay, double = branch_epochs(
            self.config, head.search, head.playing, self.costs, self.capacity(head)
        )
        return [
            self.train_step(prefix + "stay", head, head.search, "stay", stay),
            self.train_step(
                prefix + "double", head, head.search.doubled(), "double", double
            ),
        ]

    def train_step(
        self,
        name: str,
        head: Head,
        search: Search,
        branch: str | None,
        epochs: int | None = None,
    ) -> dict:
        """A training step from `head`: a trunk (`branch` None), in which the network the
        last test grew into takes over, or a period's `stay` or `double` branch. Its compute is
        what it costs with the head's network playing throughout."""
        playing = head.playing
        capacity = self.capacity(head)
        if epochs is None:
            epochs = max(
                1,
                math.ceil(
                    self.config.trunk_turnovers * turnover_epochs(capacity, search)
                ),
            )
        compute = scaled(
            epoch_compute(self.config, search, playing, self.costs), epochs
        )
        what = {None: "trunk", "stay": "stay branch", "double": "double branch"}[branch]
        return {
            "name": name,
            "kind": "train",
            "about": (
                f"{what}, {net_name(self.config.nets[playing])} playing at "
                f"{search.label()}, window {capacity}, epochs {head.epoch}-"
                f"{head.epoch + epochs}"
            ),
            "branch": branch,
            "start": head.state,
            "take_over": branch is None and head.take_over,
            "waiting_steps": waiting_steps(self.config, search, playing, self.costs),
            "search": asdict(search),
            "playing": playing,
            "first_epoch": head.epoch,
            "last_epoch": head.epoch + epochs,
            "buffer": capacity,
            # positions stored per environment along the kept line, at the step's end
            "stored": head.stored + epochs * search.stored_per_epoch(),
            "compute": compute,
        }

    def capacity(self, head: Head) -> int:
        """The replay buffer's capacity for a step from `head`: by the window rule, if any."""
        window = self.config.window
        if window is None:
            return head.capacity
        return window.capacity(head.stored, head.capacity)

    def network_player(
        self, step: dict, net: int, sims: int, tree_factor: int | None = None
    ) -> dict:
        network = self.player(step, net)
        return {
            "network": network,
            "name": f"{network}@{sims}" + (f"-t{tree_factor}" if tree_factor else ""),
            "checkpoint": self.checkpoint(step, net),
            "net": self.config.nets[net],
            "run": f"{self.config.name}-{step['name']}",
            "sims": sims,
            "tree_factor": tree_factor,
        }

    def clock_player(self, step: dict, net: int) -> dict:
        """A network's player at equal time: its clock's sims (see `Config.clock_sims`)."""
        assert self.config.clock_sims is not None
        sims = self.config.clock_sims[net_name(self.config.nets[net])]
        return self.network_player(step, net, sims, self.config.clock_tree_factor)

    def test_series(self, period: int) -> dict:
        """A period's series, after its branches, the second network of each a branch's in
        `stay`: size, the next network against the one playing, both as they trained in
        `stay`; sims, `double`'s playing network against `stay`'s; and `double`'s next network
        against `stay`'s."""
        steps = self.period_steps(period)
        stay, double = steps[f"p{period:03d}-stay"], steps[f"p{period:03d}-double"]
        sims = self.config.play_sims
        playing = stay["playing"]
        series = [
            {
                "question": "sims",
                "a": self.network_player(double, playing, sims),
                "b": self.network_player(stay, playing, sims),
            }
        ]
        if playing + 1 < len(self.config.nets):
            series = [
                {
                    "question": "size",
                    "a": self.network_player(stay, playing + 1, sims),
                    "b": self.network_player(stay, playing, sims),
                },
                *series,
                {
                    "question": "next sims",
                    "a": self.network_player(double, playing + 1, sims),
                    "b": self.network_player(stay, playing + 1, sims),
                },
            ]
            if self.config.clock_sims is not None:
                # for the record: the same two at equal time
                series.append(
                    {
                        "question": "size on the clock",
                        "a": self.clock_player(stay, playing + 1),
                        "b": self.clock_player(stay, playing),
                    }
                )
        return {
            "name": f"p{period:03d}-test",
            "kind": "test",
            "about": f"next vs playing, double vs stay, epochs {double['last_epoch']} and "
            f"{stay['last_epoch']}",
            "series": series,
        }

    def ladder_step(self, period: int, clock: bool = False) -> dict:
        """A period's ladder (see `run_ladder`): its kept network, from the rung the last
        period's lost to; with `clock`, at its clock's sims, on the clock ladder's rungs."""
        prefix = f"p{period:03d}-"
        test = self.step(prefix + "test")
        kept = self.step(test["kept"])
        # the network the run goes on with: the next one in `stay` if the test grew the run
        playing = kept["playing"] + int(test["decision"] == "grow")
        start = self.ladder[self.head.clock_rung if clock else self.head.rung]
        player = (
            self.clock_player(kept, playing)
            if clock
            else self.network_player(kept, playing, self.config.play_sims)
        )
        return {
            "name": prefix + ("clock-ladder" if clock else "ladder"),
            "kind": "ladder",
            "clock": clock,
            "about": f"{net_name(self.config.nets[playing])} at epoch "
            f"{kept['last_epoch']} up the ladder from {start}"
            + (f" at {player['sims']} sims, its clock's" if clock else ""),
            "player": player,
            "start_rung": start,
            "series": [],
            "requests": [],
            "results": [],
        }

    # reporting

    def status(self) -> str:
        head = self.head
        config = self.config
        lines = [
            f"{config.name}: period {self.data['period']}, epoch {head.epoch}, "
            f"{net_name(config.nets[head.playing])} playing at {head.search.label()}, "
            f"window {head.capacity}"
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
        (
            "advance",
            "queue what's next, as a step's end does (e.g. missing ladder steps)",
        ),
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
    elif args.command == "advance":
        controller.advance()


if __name__ == "__main__":
    main()
