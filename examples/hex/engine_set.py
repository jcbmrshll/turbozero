"""Hex positions with every legal move scored by MoHex: a fixed set to measure networks against
a far stronger player (see `label_positions.py` to build one, and train.py's --engine-set, which
scores the network on one as it trains, with `core.testing.scored_positions.ScoredPositionsTester`).

Scores are the value of a move for the player making it, from -1 to 1. At sizes MoHex's solver can
finish they're exact wins and losses (`level` 0); otherwise its search's values (`level` its
playouts).

A set is an .npz of numpy arrays:
- obs: (N, size, size, 4) bool, each position's observation, as the network takes it
- legal: (N, size * size + 1) bool, its legal actions
- scores: (N, size * size + 1) float32, the engine's score of each legal action; ILLEGAL where
  the action isn't legal
- stones: (N,) int16, stones on the board
- engine, level: the engine, and its playouts a position, 0 if it solved them
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from engines import CACHE_DIR

from core.testing.scored_positions import ScoredPositions

# the score of an illegal action
ILLEGAL = -999.0
# the game phases metrics are reported for, by the share of the board's cells filled
PHASES = {"opening": (0.0, 0.2), "midgame": (0.2, 0.4), "endgame": (0.4, 1.0)}


def default_path(size: int) -> Path:
    """Where training runs look for a set unless told otherwise."""
    return CACHE_DIR / "engine-sets" / f"mohex-{size}.npz"


@dataclass(frozen=True)
class EngineSet:
    """A set of positions with each legal action scored by an engine (see the module's docstring
    for the arrays)."""

    obs: np.ndarray
    legal: np.ndarray
    scores: np.ndarray
    stones: np.ndarray
    engine: str
    level: int

    def __post_init__(self):
        if not ((self.scores != ILLEGAL) == self.legal).all():
            raise ValueError("every legal action needs a score, and only those")

    def __len__(self) -> int:
        return len(self.stones)

    def save(self, path: str | Path) -> None:
        """Writes the set to `path` (an .npz), through a temporary file so a reader never sees
        half of one."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp,
            obs=self.obs,
            legal=self.legal,
            scores=self.scores.astype(np.float32),
            stones=self.stones.astype(np.int16),
            engine=np.array(self.engine),
            level=np.array(self.level),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "EngineSet":
        with np.load(path) as f:
            return cls(
                obs=f["obs"],
                legal=f["legal"],
                scores=f["scores"],
                stones=f["stones"],
                engine=str(f["engine"]),
                level=int(f["level"]),
            )

    def groups(self) -> dict[str, np.ndarray]:
        """The subsets metrics are reported for: each game phase."""
        cells = self.obs.shape[1] * self.obs.shape[2]
        return {
            name: (self.stones / cells >= lo) & (self.stones / cells < hi)
            for name, (lo, hi) in PHASES.items()
        }

    def scored_positions(self) -> ScoredPositions:
        """The positions and scores to measure networks on, with the groups of `groups`."""
        return ScoredPositions(
            observations=self.obs,
            legal=self.legal,
            scores=self.scores,
            groups=self.groups(),
        )

    def describe(self) -> dict:
        """What the set is, for a run's configuration."""
        return {"engine": self.engine, "level": self.level, "positions": len(self)}
