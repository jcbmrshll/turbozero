"""A trigger that fires when a measure of training stops improving (see `Plateau`), for levers such
as a learning rate's drops (`core.training.lr_plateau`)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlateauProgress:
    """What a `Plateau` has seen since it last fired (or training started), on the host: the epochs
    since, and the measures taken in the newest `window` of them, as (epoch, value) pairs, epochs
    counting from 1 since it last fired."""

    epochs: int = 0
    points: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class PlateauStep:
    """What a `Plateau` made of an epoch (see `Plateau.after_epoch`).

    Attributes:
        progress: its progress after the epoch, started again if it fired
        improvement: the fitted improvement over the last `window` epochs (see `improvement`), None
            until there are that many and at least two measures in them
        plateaued: whether it fired
    """

    progress: PlateauProgress
    improvement: float | None
    plateaued: bool


def improvement(epochs: Sequence[float], values: Sequence[float], window: int) -> float:
    """How much `values`, measured at `epochs`, rose over `window` epochs: the slope of the
    least-squares line through them, times `window`."""
    x = np.asarray(epochs, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x -= x.mean()
    slope = float((x * (y - y.mean())).sum() / (x * x).sum())
    return slope * window


@dataclass(frozen=True)
class Plateau:
    """Fires when the metric `signal` improved by less than `threshold` over the last `window`
    epochs, by the slope of a least-squares line through its values there (see `improvement`), once
    at least `min_epochs` have passed since it last fired. Improving is falling, or rising with
    `maximize`.

    A signal measured every epoch, such as the training loss, has a value in each of the window's
    epochs; one measured every few epochs, such as games against anchors, has fewer, at least two.
    The window only holds epochs since it last fired: what it measures after, e.g. a learning rate
    drop, starts afresh.

    Attributes:
        signal: the metric it follows, as the trainer logs it, e.g. `loss`
        maximize: whether the signal improves by rising
        window: epochs the improvement is measured over
        threshold: the improvement over `window` epochs below which it fires
        min_epochs: epochs since it last fired before it can again
    """

    signal: str = "loss"
    maximize: bool = False
    window: int = 40
    threshold: float = 0.004
    min_epochs: int = 40

    def __post_init__(self):
        if self.window < 2:
            raise ValueError(f"window must be at least 2 epochs, got {self.window}")
        if self.min_epochs < 1:
            raise ValueError(f"min_epochs must be positive, got {self.min_epochs}")

    def after_epoch(
        self, progress: PlateauProgress, value: float | None
    ) -> PlateauStep:
        """Follows an epoch whose measure of `signal` was `value`, None if it wasn't measured."""
        epochs = progress.epochs + 1
        points = progress.points
        if value is not None:
            points = (*points, (epochs, float(value)))
        points = tuple((e, v) for e, v in points if e > epochs - self.window)
        fitted = None
        if epochs >= self.window and len(points) >= 2:
            rise = improvement(
                [e for e, _ in points], [v for _, v in points], self.window
            )
            fitted = rise if self.maximize else -rise
        plateaued = (
            fitted is not None and epochs >= self.min_epochs and fitted < self.threshold
        )
        if plateaued:
            return PlateauStep(PlateauProgress(), fitted, plateaued=True)
        return PlateauStep(PlateauProgress(epochs, points), fitted, plateaued=False)

    def get_config(self) -> dict:
        return {
            "signal": self.signal,
            "maximize": self.maximize,
            "window": self.window,
            "threshold": self.threshold,
            "min_epochs": self.min_epochs,
        }
