"""A learning rate that steps down a list of rates as training stops improving (see `LRPlateau`)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class LRPlateauState(NamedTuple):
    """An `LRPlateau` schedule's state, in the optimizer's: the index of the current rate."""

    rate_index: jax.Array


@dataclass(frozen=True)
class LRPlateauProgress:
    """What an `LRPlateau` rule has seen at the current rate, on the host: the epochs trained at
    it, and the newest `window` of their training losses."""

    epochs: int = 0
    losses: tuple[float, ...] = ()


@dataclass(frozen=True)
class LRPlateauStep:
    """What an `LRPlateau` rule made of an epoch (see `LRPlateau.after_epoch`).

    Attributes:
        progress: the rule's progress after the epoch
        index: the rate to train the next epoch at, its index in `rates`
        improvement: the fitted improvement over the last `window` epochs at the current rate
            (see `improvement`), None until there are that many
        plateaued: whether the current rate plateaued: the rule drops to the next, if there's one
    """

    progress: LRPlateauProgress
    index: int
    improvement: float | None
    plateaued: bool


def improvement(losses: Sequence[float]) -> float:
    """How much the loss fell over `losses`, one per epoch: minus the slope of the least-squares
    line through them, times their number."""
    y = np.asarray(losses, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    x -= x.mean()
    slope = float((x * (y - y.mean())).sum() / (x * x).sum())
    return -slope * len(y)


@dataclass(frozen=True)
class LRPlateau:
    """A learning rate that drops to the next of `rates` when training stops improving: when the
    training loss improved by less than `threshold` over the last `window` epochs (see
    `improvement`), after at least `min_epochs` at the current rate. The window only holds
    epochs at the current rate.

    It's an optax stateful schedule, to pass as an `optax.inject_hyperparams` optimizer's
    learning rate, e.g. `optax.inject_hyperparams(optax.adamw)(learning_rate=rule)`, and the
    `Trainer`'s `lr_plateau`, which applies the rule after every epoch. The schedule's state is
    the current rate's index, so the rate changes with an array in the optimizer's state (no
    compiling again), saved with it. Restarting the optimizer's schedules (a fork's, see
    `core.training.train.restart_schedules`) starts again from the first rate.

    Attributes:
        rates: the learning rates, the first to start at, each next to drop to
        window: epochs the improvement is measured over
        threshold: the improvement over `window` epochs below which the rate has plateaued
        min_epochs: epochs to train at a rate before it can drop
        stop: end training once the last rate plateaus (else it carries on at it)
    """

    rates: tuple[float, ...]
    window: int = 40
    threshold: float = 0.004
    min_epochs: int = 40
    stop: bool = False

    def __post_init__(self):
        object.__setattr__(self, "rates", tuple(float(r) for r in self.rates))
        if not self.rates or any(r <= 0 for r in self.rates):
            raise ValueError(f"rates must be positive, at least one, got {self.rates}")
        if self.window < 2:
            raise ValueError(f"window must be at least 2 epochs, got {self.window}")
        if self.min_epochs < 1:
            raise ValueError(f"min_epochs must be positive, got {self.min_epochs}")

    def init(self) -> LRPlateauState:
        return LRPlateauState(rate_index=jnp.zeros([], jnp.int32))

    def update(self, state: LRPlateauState, **extra_args: Any) -> LRPlateauState:  # pylint: disable=unused-argument
        # the rate only moves between epochs, by the trainer (see `after_epoch`)
        return state

    def __call__(self, state: LRPlateauState, **extra_args: Any) -> jax.Array:  # pylint: disable=unused-argument
        return jnp.asarray(self.rates, dtype=jnp.float32)[state.rate_index]

    def after_epoch(
        self, progress: LRPlateauProgress, index: int, loss: float
    ) -> LRPlateauStep:
        """Applies the rule after an epoch trained at rate `index` reached training loss `loss`.

        When the rate plateaus, the progress starts again (for the next rate, or the last's
        next `min_epochs`)."""
        epochs = progress.epochs + 1
        losses = (*progress.losses, float(loss))[-self.window :]
        fitted = improvement(losses) if len(losses) == self.window else None
        plateaued = (
            fitted is not None and epochs >= self.min_epochs and fitted < self.threshold
        )
        if plateaued:
            return LRPlateauStep(
                LRPlateauProgress(),
                min(index + 1, len(self.rates) - 1),
                fitted,
                plateaued=True,
            )
        return LRPlateauStep(
            LRPlateauProgress(epochs, losses),
            index,
            fitted,
            plateaued=False,
        )

    def get_config(self) -> dict:
        return {
            "rates": list(self.rates),
            "window": self.window,
            "threshold": self.threshold,
            "min_epochs": self.min_epochs,
            "stop": self.stop,
        }
