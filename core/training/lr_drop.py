"""A learning rate drop on a schedule: the network playing self-play cools once it has played long
enough (see `LRDrop`)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LRDrop:
    """Scales a network's updates by `factor` (as its learning rate would be) once it has played
    self-play for `after_epochs` epochs: the Trainer's `lr_drop`.

    At a constant learning rate, training stalls where most of each step is noise: the weights
    jitter about where they're heading, and an average of them recovers only part of what a drop
    gains (an Othello 12x144 gained about +42 Elo within 30 epochs of a 10x drop after 200 flat
    epochs). A network that's waiting to take over trains at the full rate, catching up; one that
    takes over counts its own epochs from 0, starting warm on the new data its play brings.
    Without growth, the network plays from the run's first epoch (a fork's, after `fork_from`).

    Attributes:
        after_epochs: epochs a network plays at the full rate
        factor: the learning rate's scale from then on, e.g. 0.1
    """

    after_epochs: int
    factor: float

    def __post_init__(self):
        if self.after_epochs < 0 or not 0 < self.factor <= 1:
            raise ValueError(
                f"after_epochs must be at least 0 and factor in (0, 1], got "
                f"{self.after_epochs} and {self.factor}"
            )

    def scale(self, played: int) -> float:
        """The learning rate's scale for a network that has played `played` epochs."""
        return self.factor if played >= self.after_epochs else 1.0

    def get_config(self) -> dict:
        return {"after_epochs": self.after_epochs, "factor": self.factor}
