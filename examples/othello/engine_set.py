"""Othello positions with every legal move scored by Edax: a fixed set to measure networks
against a far stronger player (see `label_positions.py` to build one, `score_engine_set.py` to
score checkpoints on one, and train.py's --engine-set, which scores the network on one as it
trains, with `core.testing.scored_positions.ScoredPositionsTester`).

A set is an .npz of numpy arrays:
- obs: (N, 16) uint8, each position's (8, 8, 2) observation (as the network takes it: the
  player to move's discs, then the opponent's), bit-packed (np.packbits)
- mask: (N, 9) uint8, its 65 legal actions, bit-packed
- scores: (N, 65) int16, the engine's score of each legal action: the final disc margin it
  expects after the move, for the player making it; ILLEGAL where the action isn't legal
- depth: (N, 65) int16, how many moves ahead each action was searched (an exact solve when it
  reaches the end of the game)
- source: (N,) int8, which of `source_names` the position came from
- source_names: (S,) the sources' names, e.g. the run whose games the positions are from
- source_state_paths: (S,) each source's saved training state (state-<EPOCH>.npz), absolute
- source_state_envs: (S, 2) the environments of it the positions are from, first to last
  (exclusive); (-1, -1) for all
- empties: (N,) int8, empty squares
- engine, level: the engine and the level it searched at
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from engines import CACHE_DIR
from game import PASS

from core.testing.scored_positions import ScoredPositions

# where training runs look for a set unless told otherwise
DEFAULT_PATH = CACHE_DIR / "engine-sets" / "edax-l24.npz"
# the score of an illegal action
ILLEGAL = -999
# positions with this many empty squares or more are XOT openings' (8 moves from the standard
# start), which every game shares
OPENING_EMPTIES = 52
# the game phases metrics are reported for, by empty squares
PHASES = {"opening": (40, 51), "midgame": (25, 39), "endgame": (1, 24)}


@dataclass(frozen=True)
class SourceState:
    """Where a source's positions came from: a saved training state's replay buffer (see
    `Trainer.save_state`), so that whoever trains on that run's games can leave these positions'
    games out.

    Attributes:
        path: the state, state-<EPOCH>.npz, absolute
        envs: (first, last), the environments the positions are from, last exclusive; None for
            all of them
    """

    path: str
    envs: tuple[int, int] | None = field(default=None)


@dataclass(frozen=True)
class EngineSet:
    """A set of positions with each legal action scored by an engine (see the module's
    docstring for the arrays).

    Attributes:
        obs: (N, 8, 8, 2) bool
        legal: (N, 65) bool
        scores: (N, 65) int16, ILLEGAL where illegal
        depth: (N, 65) int16
        source: (N,) int8
        source_names: the sources' names: letters, digits, "-" and "_" (they name metrics)
        source_states: where each source's positions came from
        empties: (N,) int8
        engine: e.g. "edax"
        level: the engine's level
    """

    obs: np.ndarray
    legal: np.ndarray
    scores: np.ndarray
    depth: np.ndarray
    source: np.ndarray
    source_names: list[str]
    source_states: list[SourceState]
    empties: np.ndarray
    engine: str
    level: int

    def __post_init__(self):
        for name in self.source_names:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(
                    f"source names name metrics: letters, digits, - and _, not {name!r}"
                )
        if len(self.source_states) != len(self.source_names):
            raise ValueError("every source needs its state")
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
            depth=self.depth.astype(np.int16),
            source=self.source.astype(np.int8),
            source_names=np.array(self.source_names),
            source_state_paths=np.array([s.path for s in self.source_states]),
            source_state_envs=np.array(
                [s.envs or (-1, -1) for s in self.source_states], dtype=np.int64
            ).reshape(-1, 2),
            empties=self.empties.astype(np.int8),
            engine=np.array(self.engine),
            level=np.array(self.level),
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
                source=f["source"],
                source_names=[str(s) for s in f["source_names"]],
                source_states=[
                    SourceState(
                        str(path), None if first < 0 else (int(first), int(last))
                    )
                    for path, (first, last) in zip(
                        f["source_state_paths"], f["source_state_envs"], strict=True
                    )
                ],
                empties=f["empties"],
                engine=str(f["engine"]),
                level=int(f["level"]),
            )

    def groups(self, by_source: bool = False) -> dict[str, np.ndarray]:
        """The subsets metrics are reported for: each game phase, and with `by_source` each
        source too (where positions came from matters for checking a set, not for reading a
        network's metrics)."""
        groups = (
            {name: self.source == i for i, name in enumerate(self.source_names)}
            if by_source
            else {}
        )
        for name, (lo, hi) in PHASES.items():
            groups[name] = (self.empties >= lo) & (self.empties <= hi)
        return groups

    def scored_positions(self, by_source: bool = False) -> ScoredPositions:
        """The positions and scores to measure networks on, with the groups of `groups`."""
        return ScoredPositions(
            observations=self.obs,
            legal=self.legal,
            scores=self.scores.astype(np.float32),
            groups=self.groups(by_source),
        )

    def describe(self) -> dict:
        """What the set is, for a run's configuration."""
        return {
            "engine": self.engine,
            "level": self.level,
            "sources": {
                name: {
                    "positions": int((self.source == i).sum()),
                    "state": state.path,
                    "envs": state.envs,
                }
                for i, (name, state) in enumerate(
                    zip(self.source_names, self.source_states, strict=True)
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


def boards(obs: np.ndarray) -> np.ndarray:
    """Positions as `EdaxAnalyst.analyse` takes them, with the player to move as black: (N, 64),
    a1, b1, ..., h8: 0 empty, 1 the player to move, 2 the opponent."""
    flat = obs.reshape(len(obs), 64, 2)
    return np.where(flat[..., 0], 1, np.where(flat[..., 1], 2, 0)).astype(np.int8)
