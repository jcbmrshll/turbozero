"""Growing the network as training goes: a run that starts with a small network and hands
self-play over to bigger ones in turn (see `Growth`), as KataGo and Leela Zero grew theirs."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import equinox as eqx
import jax
import numpy as np

from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester
from core.training.schedule import EvaluatorSchedule


@dataclass(frozen=True)
class Learner:
    """A network a `Growth` run grows into: what the `Trainer` takes for its own network, for
    another.

    Attributes:
        name: what its metrics (`<name>/<metric>`) and checkpoints (`<ckpt_dir>/<name>`) go by
        nn: the network (an equinox module), training starts from its parameters
        evaluator: its self-play evaluator, which plays once it takes over (or an
            `EvaluatorSchedule` of them, by the run's epochs)
        evaluator_test: the evaluator its testers play with
        testers: its testers, run while it trains
        nn_state: (optional) the network's initial state (e.g. BatchNorm's)
    """

    name: str
    nn: eqx.Module
    evaluator: Evaluator | EvaluatorSchedule
    evaluator_test: Evaluator
    testers: Sequence[BaseTester] = ()
    nn_state: eqx.nn.State | None = None

    @property
    def schedule(self) -> EvaluatorSchedule:
        """Its self-play evaluators, as a schedule (of one, for a single evaluator)."""
        if isinstance(self.evaluator, EvaluatorSchedule):
            return self.evaluator
        return EvaluatorSchedule([(0, self.evaluator)])


@dataclass(frozen=True)
class GrowthProgress:
    """Where a `Growth` run is, on the host, saved with the run's state.

    Attributes:
        playing: index of the learner self-play plays with (the Trainer's own network is 0)
        trained: epochs each learner has trained, one per learner
        values: the newest `Growth.window` values of the switch metric each learner logged
            since it started training, one tuple per learner
        guard_values: the same, of `Growth.guard`'s metric (empty without a guard)
    """

    playing: int = 0
    trained: tuple[int, ...] = ()
    values: tuple[tuple[float, ...], ...] = ()
    guard_values: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class SwitchCheck:
    """What a `Growth` rule made of an epoch (see `Growth.after_epoch`).

    Attributes:
        progress: the run's progress after the epoch, with the switch if there was one
        switched: whether self-play moves to the next learner from the next epoch
        gap: the next learner's mean metric over the window minus the playing one's (negative
            once the next is better), None until both have a full window
        caught_up: whether the next learner has caught up: trained its minimum epochs, with a
            gap at or below 0 (at or below minus `margin` of the playing one's mean, with a
            margin) and no worse on the guard (what the rule switches on, and all a `manual` rule
            reports)
        guard_gap: as `gap`, of the guard's metric (None without a guard)
    """

    progress: GrowthProgress
    switched: bool
    gap: float | None
    caught_up: bool = False
    guard_gap: float | None = None


@dataclass(frozen=True)
class Growth:
    """A run that grows its network: the `Trainer`'s own network plays self-play first, and
    `learners`, bigger networks, take over from it in turn. Two networks train at a time, on the
    same replay buffer: the one playing self-play, and the next, which trains on its games
    without playing (off-policy). When the next takes over, the one that played stops, and the
    one after starts training from scratch on the buffer as it is.

    The next takes over when its `metric` (a metric its testers or its training log every epoch,
    lower being better: e.g. a `ScoredPositionsTester`'s "engine_value_mse" on fixed positions,
    a fair test of a network that doesn't play, or the training "loss" on the data both train
    on, which KataGo compared), averaged over the last `window` epochs, is at or below the
    playing network's, once it has trained `min_epochs` epochs. With a `margin`, it must lead:
    its mean lower than the playing one's by `margin` of it (e.g. 0.01, 1%), so that two
    networks in lockstep don't switch. With a `guard`, it must also be no worse on a second
    metric over the window, e.g. "engine_value_mse" on positions neither trains on, when the
    switch metric is the training loss, which a bigger network can lower by memorising the
    buffer. Or, with `switch_at`, after
    fixed numbers of epochs regardless. Either way it takes over when the playing network's
    learning rate rule (the Trainer's `lr_plateau`) would stop the run: a network done
    learning has nothing more to give.

    A network that waits trains at the learning rate's first value (an `lr_plateau` rule only
    moves the playing network's), and follows the rule from its first rate once it plays.

    With `manual`, nothing switches by itself: the rule only reports when the next has caught up
    (`SwitchCheck.caught_up`), and the next takes over when a saved state of the run is branched
    with `take_over` (see `Trainer.branch`), e.g. by a controller that first plays the two.

    Attributes:
        first: the name of the Trainer's own network (see `Learner.name`)
        learners: the networks to grow into, in order
        metric: what the rule compares, a metric every learner's testers or training log
            every epoch
        window: epochs the metric is averaged over
        min_epochs: epochs the next learner trains before it can take over
        switch_at: (optional) instead of the rule, the numbers of epochs done after which each
            of `learners` takes over, in increasing order, one per learner
        manual: (optional) the rule reports, and switches only on a branch's `take_over`
        margin: (optional) the lead the next needs, a fraction of the playing one's mean metric
        guard: (optional) a second metric the next must be no worse on, logged every epoch
    """

    first: str
    learners: Sequence[Learner]
    metric: str = "engine_value_mse"
    window: int = 10
    min_epochs: int = 10
    switch_at: Sequence[int] = field(default=())
    manual: bool = False
    margin: float = 0.0
    guard: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "learners", tuple(self.learners))
        object.__setattr__(self, "switch_at", tuple(self.switch_at))
        if not self.learners:
            raise ValueError("growth needs at least one network to grow into")
        names = [self.first, *(learner.name for learner in self.learners)]
        if len(set(names)) != len(names) or not all(names):
            raise ValueError(
                f"the networks' names must be distinct and not empty: {names}"
            )
        if any("/" in name for name in names):
            raise ValueError(f"the networks' names can't hold a '/': {names}")
        if self.window < 1 or self.min_epochs < 1:
            raise ValueError("window and min_epochs must be positive")
        if self.switch_at and (
            len(self.switch_at) != len(self.learners)
            or any(e < 1 for e in self.switch_at)
            or any(a >= b for a, b in zip(self.switch_at, self.switch_at[1:]))
        ):
            raise ValueError(
                f"switch_at needs one increasing, positive number of epochs per network to grow "
                f"into ({len(self.learners)}), got {list(self.switch_at)}"
            )
        if self.manual and self.switch_at:
            raise ValueError("a manual rule doesn't switch at fixed epochs: pass one")
        if not 0 <= self.margin < 1:
            raise ValueError(f"the margin is a fraction in [0, 1), got {self.margin}")
        if self.guard is not None and self.guard == self.metric:
            raise ValueError(f"the guard must be another metric than {self.metric!r}")

    @property
    def names(self) -> tuple[str, ...]:
        """Every network's name, the Trainer's own first."""
        return (self.first, *(learner.name for learner in self.learners))

    def start(self) -> GrowthProgress:
        """The progress of a run that hasn't trained yet."""
        n = len(self.learners) + 1
        return GrowthProgress(0, (0,) * n, ((),) * n, ((),) * n if self.guard else ())

    def training(self, progress: GrowthProgress) -> tuple[int, ...]:
        """The learners that train: the one playing, and the next, if there is one."""
        return tuple(
            i
            for i in (progress.playing, progress.playing + 1)
            if i <= len(self.learners)
        )

    def after_epoch(
        self,
        progress: GrowthProgress,
        epochs_done: int,
        values: dict[int, float],
        rule_done: bool = False,
        guard_values: dict[int, float] | None = None,
    ) -> SwitchCheck:
        """Applies the rule after an epoch.

        Args:
            progress: the progress before the epoch
            epochs_done: the run's epochs done, this one included
            values: the switch metric each learner that trained logged this epoch, by index
                (may be empty with `switch_at`)
            rule_done: whether the playing learner's learning rate rule would stop the run
            guard_values: (with a `guard`) its metric, as `values`

        Returns:
            SwitchCheck: the progress after the epoch, and whether self-play moves on
        """
        trained = list(progress.trained)
        kept = list(progress.values)
        guarded: list[tuple[float, ...]] = list(progress.guard_values)
        if self.guard is not None and not guarded:
            guarded = [()] * len(trained)
        for i in self.training(progress):
            trained[i] += 1
            if i in values:
                kept[i] = (*kept[i], float(values[i]))[-self.window :]
            if self.guard is not None and guard_values and i in guard_values:
                guarded[i] = (*guarded[i], float(guard_values[i]))[-self.window :]
        after = GrowthProgress(
            progress.playing, tuple(trained), tuple(kept), tuple(guarded)
        )
        gap = self.gap(after)
        caught_up = self.caught_up(after)
        following = progress.playing + 1
        switched = False
        if following <= len(self.learners) and not self.manual:
            if rule_done:
                switched = True
            elif self.switch_at:
                switched = epochs_done >= self.switch_at[following - 1]
            else:
                switched = caught_up
        # both gaps between the networks the epoch compared, before any switch
        guard_gap = (
            self._gap(after.guard_values, after.playing)
            if self.guard is not None
            else None
        )
        if switched:
            after = self.take_over(after)
        return SwitchCheck(after, switched, gap, caught_up, guard_gap)

    def _gap(
        self, windows: tuple[tuple[float, ...], ...], playing: int
    ) -> float | None:
        following = playing + 1
        if following > len(self.learners) or any(
            len(windows[i]) < self.window for i in (playing, following)
        ):
            return None
        return float(np.mean(windows[following]) - np.mean(windows[playing]))

    def gap(self, progress: GrowthProgress) -> float | None:
        """The next learner's mean metric over the last `window` epochs minus the playing one's,
        None if the last one plays or either hasn't a full window yet."""
        return self._gap(progress.values, progress.playing)

    def caught_up(self, progress: GrowthProgress) -> bool:
        """Whether the next learner has caught up with the one playing: trained `min_epochs`,
        its mean metric over the window lower than the playing one's by at least `margin` of
        it (at or below it, without one), and with a `guard`, no worse on the guard's."""
        gap = self.gap(progress)
        if gap is None or progress.trained[progress.playing + 1] < self.min_epochs:
            return False
        lead = self.margin * abs(float(np.mean(progress.values[progress.playing])))
        if gap > -lead:
            return False
        if self.guard is None:
            return True
        guard_gap = self._gap(progress.guard_values, progress.playing)
        return guard_gap is not None and guard_gap <= 0

    def take_over(self, progress: GrowthProgress) -> GrowthProgress:
        """The progress once the next learner plays (and the one after starts training).

        Raises:
            ValueError: if the last learner already plays
        """
        if progress.playing + 1 > len(self.learners):
            raise ValueError(
                f"{self.names[progress.playing]} is the last network: none takes over from it"
            )
        return replace(progress, playing=progress.playing + 1)

    def get_config(self) -> dict:
        return {
            "first": self.first,
            "learners": [
                {
                    "name": learner.name,
                    "nn": learner.nn.__class__.__name__,
                    "nn_params": sum(
                        x.size
                        for x in jax.tree.leaves(
                            eqx.filter(learner.nn, eqx.is_inexact_array)
                        )
                    ),
                    "evaluator": learner.schedule.get_config(),
                    "evaluator_test": learner.evaluator_test.get_config(),
                    "testers": [tester.get_config() for tester in learner.testers],
                }
                for learner in self.learners
            ],
            "metric": self.metric,
            "window": self.window,
            "min_epochs": self.min_epochs,
            "switch_at": list(self.switch_at) or None,
            "manual": self.manual,
            "margin": self.margin,
            "guard": self.guard,
        }
