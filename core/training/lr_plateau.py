"""A learning rate that steps down a list of rates as training stops improving (see `LRPlateau`)."""

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from core.training.plateau import Plateau, PlateauProgress


class LRPlateauState(NamedTuple):
    """An `LRPlateau` schedule's state, in the optimizer's: the index of the current rate."""

    rate_index: jax.Array


@dataclass(frozen=True)
class LRPlateauStep:
    """What an `LRPlateau` rule made of an epoch (see `LRPlateau.after_epoch`).

    Attributes:
        progress: its plateau's progress after the epoch
        index: the rate to train the next epoch at, its index in `rates`
        improvement: its plateau's fitted improvement (see `Plateau`), None until it has one
        plateaued: whether the current rate plateaued: the rule drops to the next, if there's one
    """

    progress: PlateauProgress
    index: int
    improvement: float | None
    plateaued: bool


@dataclass(frozen=True)
class LRPlateau:
    """A learning rate that drops to the next of `rates` when training stops improving: when
    `plateau` fires (by default when the training loss improved by less than 0.004 over the last 40
    epochs, after at least 40 at the current rate). Its window only holds epochs at the current
    rate.

    It's an optax stateful schedule, to pass as an `optax.inject_hyperparams` optimizer's
    learning rate, e.g. `optax.inject_hyperparams(optax.adamw)(learning_rate=rule)`, and the
    `Trainer`'s `lr_plateau`, which applies the rule after every epoch. The schedule's state is
    the current rate's index, so the rate changes with an array in the optimizer's state (no
    compiling again), saved with it. Restarting the optimizer's schedules (a fork's, see
    `core.training.train.restart_schedules`) starts again from the first rate.

    Attributes:
        rates: the learning rates, the first to start at, each next to drop to
        plateau: when a rate has plateaued: the signal it follows, and how
        stop: end training once the last rate plateaus (else it carries on at it)
    """

    rates: tuple[float, ...]
    plateau: Plateau = field(default_factory=Plateau)
    stop: bool = False

    def __post_init__(self):
        object.__setattr__(self, "rates", tuple(float(r) for r in self.rates))
        if not self.rates or any(r <= 0 for r in self.rates):
            raise ValueError(f"rates must be positive, at least one, got {self.rates}")

    def init(self) -> LRPlateauState:
        return LRPlateauState(rate_index=jnp.zeros([], jnp.int32))

    def update(self, state: LRPlateauState, **extra_args: Any) -> LRPlateauState:  # pylint: disable=unused-argument
        # the rate only moves between epochs, by the trainer (see `after_epoch`)
        return state

    def __call__(self, state: LRPlateauState, **extra_args: Any) -> jax.Array:  # pylint: disable=unused-argument
        return jnp.asarray(self.rates, dtype=jnp.float32)[state.rate_index]

    def after_epoch(
        self, progress: PlateauProgress, index: int, value: float | None
    ) -> LRPlateauStep:
        """Applies the rule after an epoch trained at rate `index`, whose measure of the plateau's
        signal was `value` (None if it wasn't measured).

        When the rate plateaus, the progress starts again (for the next rate, or the last's
        next `min_epochs`)."""
        step = self.plateau.after_epoch(progress, value)
        index = min(index + 1, len(self.rates) - 1) if step.plateaued else index
        return LRPlateauStep(step.progress, index, step.improvement, step.plateaued)

    def get_config(self) -> dict:
        return {
            "rates": list(self.rates),
            **self.plateau.get_config(),
            "stop": self.stop,
        }
