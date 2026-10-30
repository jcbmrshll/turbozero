"""Connect Four positions with every legal move scored exactly by the solver: a fixed set to measure
networks against the truth (see `label_positions.py` to build one, `score_solver_set.py` to score
checkpoints on one, and train.py's --engine-set, which scores the network on one as it trains, with
`core.testing.scored_positions.ScoredPositionsTester`).

A set is an .npz of numpy arrays:
- positions: (N,) each position, as the solver writes them
- scores: (N, 7) int16, the solver's score of each column (see `solver.py`): ILLEGAL where it's full
- source: (N,) int8, which of `source_names` the position came from
- source_names: (S,) the sources' names, e.g. how noisily its games were played
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from solver import CACHE_DIR, COLUMNS, ILLEGAL, PERFECT_LEVEL, ROWS, actions

from core.testing.scored_positions import ScoredPositions

# where training runs look for a set unless told otherwise
DEFAULT_PATH = CACHE_DIR / "solver-set.npz"
# the game phases metrics are reported for, by empty squares
PHASES = {"opening": (29, 42), "midgame": (15, 28), "endgame": (1, 14)}


def observation(position: str) -> np.ndarray:
    """A position as the network takes it (see `game.state_to_nn_input`): (6, 7, 2) bool, rows
    from the top, the player to move's discs, then the opponent's."""
    obs = np.zeros((ROWS, COLUMNS, 2), dtype=bool)
    heights = [0] * COLUMNS
    for ply, column in enumerate(actions(position)):
        # the side that made this move: the player to move's if as many moves follow it as an
        # even number
        side = (len(position) - 1 - ply) % 2 == 0
        obs[ROWS - 1 - heights[column], column, int(side)] = True
        heights[column] += 1
    return obs


@dataclass(frozen=True)
class SolverSet:
    """Positions with each legal move scored by the solver (see the module's docstring for the
    arrays).

    Attributes:
        positions: (N,) str
        scores: (N, 7) int16
        source: (N,) int8
        source_names: the sources' names: letters, digits, "-" and "_"
    """

    positions: np.ndarray
    scores: np.ndarray
    source: np.ndarray
    source_names: list[str]

    def __post_init__(self):
        for name in self.source_names:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(
                    f"source names are letters, digits, - and _, not {name!r}"
                )

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def observations(self) -> np.ndarray:
        """(N, 6, 7, 2) bool, as the network takes them."""
        return np.stack([observation(str(p)) for p in self.positions])

    @property
    def legal(self) -> np.ndarray:
        """(N, 7) bool"""
        return self.scores != ILLEGAL

    @property
    def empties(self) -> np.ndarray:
        """(N,) empty squares"""
        return PERFECT_LEVEL - np.array([len(str(p)) for p in self.positions])

    @property
    def outcomes(self) -> np.ndarray:
        """(N, 7) how each move's game ends with perfect play, for the player making it: 1 for a
        win, 0 for a draw, -1 for a loss, however soon. Illegal moves stay ILLEGAL."""
        return np.where(self.legal, np.sign(self.scores), ILLEGAL)

    def save(self, path: str | Path) -> None:
        """Writes the set to `path` (an .npz), through a temporary file so a reader never sees
        half of one."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(
            tmp,
            positions=self.positions,
            scores=self.scores.astype(np.int16),
            source=self.source.astype(np.int8),
            source_names=np.array(self.source_names),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "SolverSet":
        with np.load(path) as f:
            return cls(
                positions=f["positions"],
                scores=f["scores"],
                source=f["source"],
                source_names=[str(s) for s in f["source_names"]],
            )

    def groups(self, by_source: bool = False) -> dict[str, np.ndarray]:
        """The subsets metrics are reported for: each game phase, the positions where the outcome
        depends on the move, and with `by_source` each source too."""
        groups = (
            {name: self.source == i for i, name in enumerate(self.source_names)}
            if by_source
            else {}
        )
        empties = self.empties
        for name, (lo, hi) in PHASES.items():
            groups[name] = (empties >= lo) & (empties <= hi)
        best = np.where(self.legal, self.outcomes, -2).max(axis=1)
        worst = np.where(self.legal, self.outcomes, 2).min(axis=1)
        groups["critical"] = best != worst
        return groups

    def scored_positions(
        self, by_outcome: bool = False, by_source: bool = False
    ) -> ScoredPositions:
        """The positions to measure networks on, with the groups of `groups`: with the solver's
        scores, so that a move that wins later than it could, or loses sooner, gives some up; or
        `by_outcome`, so that only a move that changes how the game ends does."""
        return ScoredPositions(
            observations=self.observations,
            legal=self.legal,
            scores=(self.outcomes if by_outcome else self.scores).astype(np.float32),
            groups=self.groups(by_source),
        )

    def describe(self) -> dict:
        """What the set is, for a run's configuration."""
        return {
            "engine": "solver",
            "sources": {
                name: int((self.source == i).sum())
                for i, name in enumerate(self.source_names)
            },
        }
