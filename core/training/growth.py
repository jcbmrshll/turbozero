"""Growing the network as training goes: a run that starts with a small network and hands
self-play over to bigger ones in turn (see `Growth`), as KataGo and Leela Zero grew theirs."""

from collections.abc import Sequence
from dataclasses import dataclass, field

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
    """

    playing: int = 0
    trained: tuple[int, ...] = ()
    values: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class SwitchCheck:
    """What a `Growth` rule made of an epoch (see `Growth.after_epoch`).

    Attributes:
        progress: the run's progress after the epoch, with the switch if there was one
        switched: whether self-play moves to the next learner from the next epoch
        gap: the next learner's mean metric over the window minus the playing one's (negative
            once the next is better), None until both have a full window
    """

    progress: GrowthProgress
    switched: bool
    gap: float | None


@dataclass(frozen=True)
class Growth:
    """A run that grows its network: the `Trainer`'s own network plays self-play first, and
    `learners`, bigger networks, take over from it in turn. Two networks train at a time, on the
    same replay buffer: the one playing self-play, and the next, which trains on its games
    without playing (off-policy). When the next takes over, the one that played stops, and the
    one after starts training from scratch on the buffer as it is.

    The next takes over when its `metric` (a metric its testers log every epoch, lower being
    better, e.g. a `ScoredPositionsTester`'s "engine_value_mse" on fixed positions: a fair test
    of a network that doesn't play), averaged over the last `window` epochs, is at or below the
    playing network's, once it has trained `min_epochs` epochs. Or, with `switch_at`, after
    fixed numbers of epochs regardless. Either way it takes over when the playing network's
    learning rate rule (the Trainer's `lr_plateau`) would stop the run: a network done
    learning has nothing more to give.

    A network that waits trains at the learning rate's first value (an `lr_plateau` rule only
    moves the playing network's), and follows the rule from its first rate once it plays.

    Attributes:
        first: the name of the Trainer's own network (see `Learner.name`)
        learners: the networks to grow into, in order
        metric: what the rule compares, a metric every learner's testers log every epoch
        window: epochs the metric is averaged over
        min_epochs: epochs the next learner trains before it can take over
        switch_at: (optional) instead of the rule, the numbers of epochs done after which each
            of `learners` takes over, in increasing order, one per learner
    """

    first: str
    learners: Sequence[Learner]
    metric: str = "engine_value_mse"
    window: int = 10
    min_epochs: int = 10
    switch_at: Sequence[int] = field(default=())

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

    @property
    def names(self) -> tuple[str, ...]:
        """Every network's name, the Trainer's own first."""
        return (self.first, *(learner.name for learner in self.learners))

    def start(self) -> GrowthProgress:
        """The progress of a run that hasn't trained yet."""
        n = len(self.learners) + 1
        return GrowthProgress(0, (0,) * n, ((),) * n)

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
    ) -> SwitchCheck:
        """Applies the rule after an epoch.

        Args:
            progress: the progress before the epoch
            epochs_done: the run's epochs done, this one included
            values: the switch metric each learner that trained logged this epoch, by index
                (may be empty with `switch_at`)
            rule_done: whether the playing learner's learning rate rule would stop the run

        Returns:
            SwitchCheck: the progress after the epoch, and whether self-play moves on
        """
        trained = list(progress.trained)
        kept = list(progress.values)
        for i in self.training(progress):
            trained[i] += 1
            if i in values:
                kept[i] = (*kept[i], float(values[i]))[-self.window :]
        playing, following = progress.playing, progress.playing + 1
        gap = None
        if following <= len(self.learners) and all(
            len(kept[i]) == self.window for i in (playing, following)
        ):
            gap = float(np.mean(kept[following]) - np.mean(kept[playing]))
        switched = False
        if following <= len(self.learners):
            if rule_done:
                switched = True
            elif self.switch_at:
                switched = epochs_done >= self.switch_at[following - 1]
            else:
                switched = (
                    trained[following] >= self.min_epochs
                    and gap is not None
                    and gap <= 0
                )
        progress = GrowthProgress(
            playing + 1 if switched else playing, tuple(trained), tuple(kept)
        )
        return SwitchCheck(progress, switched, gap)

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
        }
