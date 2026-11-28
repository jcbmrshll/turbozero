"""Othello positions with every legal move scored by a strong engine: a fixed set to measure
networks against (see `label_positions.py` to build one, `score_engine_set.py` to score
checkpoints on one, and train.py's --engine-set, which validates the weight average on one as it
trains, with `core.training.validation.ScoredPositionsValidator`). Networks are measured on every
position, by game phase (`PHASES`), and score_engine_set.py also on the positions they decide,
between the XOT opening and the exact solver, as filtered_* (`FILTERED_PHASES`).

A set is an .npz of numpy arrays:
- obs: (N, 16) uint8, each position's (8, 8, 2) observation (as the network takes it: the
  player to move's discs, then the opponent's), bit-packed (np.packbits)
- mask: (N, 9) uint8, its 65 legal actions, bit-packed
- scores: (N, 65) int16, the engine's score of each legal action: the final disc margin it
  expects after the move, for the player making it; ILLEGAL where the action isn't legal
- depth: (N,) int8, how many moves ahead the position was searched
- exact: (N,) bool, whether its scores are exact: a search to the end of the game, no pruning
- labeller: (N,) int8, which of `labeller_names` scored it, e.g. "neural-reversi level 26"
- source: (N,) int8, which of `source_names` the position came from
- source_names: (S,) the sources' names
- source_descriptions: (S,) where each source's positions came from, in words
- game: (N,) int64, the game the position is from, in its source (e.g. evals.db's game id;
  positions of one game are alike, so resampling should keep them together); -1 if unknown
- empties: (N,) int8, empty squares
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from engines import CACHE_DIR
from game import PASS

from core.training.validation import ScoredPositions

# where training runs look for a set unless told otherwise
DEFAULT_PATH = CACHE_DIR / "engine-sets" / "nr26-evals-29k.npz"
# the score of an illegal action
ILLEGAL = -999
# positions with this many empty squares or more are XOT openings' (8 moves from the standard
# start), which every game shares
OPENING_EMPTIES = 52
# the game phases metrics are reported for, by empty squares, over every position
PHASES = {"opening": (40, 51), "midgame": (25, 39), "endgame": (1, 24)}
# the phases the filtered metrics are reported for: the positions the network decides, from the
# XOT opening's to the exact solver's (which plays and labels positions with 20 empty squares or
# fewer), in halves; the filtered metrics are over these positions only
FILTERED_PHASES = {"opening": (36, 51), "midgame": (21, 35)}


@dataclass(frozen=True)
class EngineSet:
    """A set of positions with each legal action scored by an engine (see the module's
    docstring for the arrays).

    Attributes:
        obs: (N, 8, 8, 2) bool
        legal: (N, 65) bool
        scores: (N, 65) int16, ILLEGAL where illegal
        depth: (N,) int8
        exact: (N,) bool
        labeller: (N,) int8
        labeller_names: what scored the positions, e.g. "neural-reversi level 26"
        source: (N,) int8
        source_names: the sources' names: letters, digits, "-" and "_" (they name metrics)
        source_descriptions: where each source's positions came from
        game: (N,) int64, -1 where unknown
        empties: (N,) int8
    """

    obs: np.ndarray
    legal: np.ndarray
    scores: np.ndarray
    depth: np.ndarray
    exact: np.ndarray
    labeller: np.ndarray
    labeller_names: list[str]
    source: np.ndarray
    source_names: list[str]
    source_descriptions: list[str]
    game: np.ndarray
    empties: np.ndarray

    def __post_init__(self):
        for name in self.source_names:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(
                    f"source names name metrics: letters, digits, - and _, not {name!r}"
                )
        if len(self.source_descriptions) != len(self.source_names):
            raise ValueError("every source needs its description")
        if not ((self.scores != ILLEGAL) == self.legal).all():
            raise ValueError("every legal action needs a score, and only those")

    def __len__(self) -> int:
        return len(self.empties)

    def save(self, path: str | Path) -> None:
        """Writes the set to `path` (an .npz), through a temporary file so a reader never sees
        half of one."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(
            tmp,
            obs=np.packbits(self.obs.reshape(len(self), -1), axis=1),
            mask=np.packbits(self.legal, axis=1),
            scores=self.scores.astype(np.int16),
            depth=self.depth.astype(np.int8),
            exact=self.exact.astype(bool),
            labeller=self.labeller.astype(np.int8),
            labeller_names=np.array(self.labeller_names),
            source=self.source.astype(np.int8),
            source_names=np.array(self.source_names),
            source_descriptions=np.array(self.source_descriptions),
            game=self.game.astype(np.int64),
            empties=self.empties.astype(np.int8),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "EngineSet":
        with np.load(path) as f:
            n = len(f["empties"])
            return cls(
                obs=np.unpackbits(f["obs"], axis=1)[:, :128]
                .reshape(n, 8, 8, 2)
                .astype(bool),
                legal=np.unpackbits(f["mask"], axis=1)[:, :65].astype(bool),
                scores=f["scores"],
                depth=f["depth"],
                exact=f["exact"],
                labeller=f["labeller"],
                labeller_names=[str(s) for s in f["labeller_names"]],
                source=f["source"],
                source_names=[str(s) for s in f["source_names"]],
                source_descriptions=[str(s) for s in f["source_descriptions"]],
                game=f["game"],
                empties=f["empties"],
            )

    def subset(self, keep: np.ndarray) -> "EngineSet":
        """The positions where `keep` (N,) bool is true, with every source and labeller."""
        per_position = ("obs", "legal", "scores", "depth", "exact", "labeller")
        per_position += ("source", "game", "empties")
        return EngineSet(
            **{**self.__dict__, **{k: getattr(self, k)[keep] for k in per_position}}
        )

    def groups(
        self, by_source: bool = False, phases: dict[str, tuple[int, int]] = PHASES
    ) -> dict[str, np.ndarray]:
        """The subsets metrics are reported for: each game phase of `phases`, and with
        `by_source` each source too (where positions came from matters for checking a set, not
        for reading a network's metrics)."""
        groups = (
            {name: self.source == i for i, name in enumerate(self.source_names)}
            if by_source
            else {}
        )
        for name, (lo, hi) in phases.items():
            groups[name] = (self.empties >= lo) & (self.empties <= hi)
        return groups

    def scored_positions(
        self, by_source: bool = False, filtered: bool = False
    ) -> ScoredPositions:
        """The positions and scores to measure networks on, with the groups of `groups`: every
        position, by `PHASES`; with `filtered`, only those of a phase of `FILTERED_PHASES`."""
        phases = FILTERED_PHASES if filtered else PHASES
        groups = self.groups(by_source, phases)
        keep = (
            np.any([groups[name] for name in phases], axis=0)
            if filtered
            else np.ones(len(self), bool)
        )
        return ScoredPositions(
            observations=self.obs[keep],
            legal=self.legal[keep],
            scores=self.scores[keep].astype(np.float32),
            groups={name: group[keep] for name, group in groups.items()},
        )

    def describe(self) -> dict:
        """What the set is, for a run's configuration."""
        return {
            "labellers": {
                name: int((self.labeller == i).sum())
                for i, name in enumerate(self.labeller_names)
            },
            "exact": int(self.exact.sum()),
            "sources": {
                name: {
                    "positions": int((self.source == i).sum()),
                    "description": description,
                }
                for i, (name, description) in enumerate(
                    zip(self.source_names, self.source_descriptions, strict=True)
                )
            },
        }


def selectable(obs: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Which positions are worth scoring: past the XOT opening, with at least two legal moves
    (a pass is never a choice), and the first of each distinct position.

    Args:
        obs: (N, 8, 8, 2) bool
        legal: (N, 65) bool

    Returns:
        np.ndarray: (N,) bool
    """
    empties = 64 - obs.sum(axis=(1, 2, 3))
    ok = (empties < OPENING_EMPTIES) & (legal[:, :PASS].sum(axis=1) >= 2)
    _, first = np.unique(obs.reshape(len(obs), -1), axis=0, return_index=True)
    distinct = np.zeros(len(obs), bool)
    distinct[first] = True
    return ok & distinct
