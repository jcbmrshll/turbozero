"""Keeps self-play positions on disk, to train on later without any further search.

A run writes one file per epoch beside its checkpoints, `positions-<epoch>.npy`: a numpy
structured array (see `to_rows`), with every position whose game finished in that epoch. A position
is saved once, when its game finishes, so its outcome is known: positions of games in progress
wait for the next epochs, and those of games truncated at the step limit are never saved. That
also makes the replay buffer's state all there is to know about what's been saved, as everything
before `episode_start_idx` has been:

- a run continued from a saved state starts saving there, as its epoch's files were written before
  the state was; files of the later epochs are replaced (see `drop_positions_from`)
- a fork saves only the games it finishes itself. The positions it starts with in its replay
  buffers are its parent's, in its parent's files (those before the epoch it forked at); see
  `lineage`, which reads the two together
"""

import json
import math
import os
import re
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from core.memory.replay_memory import BaseExperience, ReplayBufferState

POSITIONS_FILE = re.compile(r"positions-(\d+)\.npy")
# beside the files: the run's parent, and the epoch it started saving from
ARCHIVE_META = "positions.json"
# epochs' worth of positions waiting to be written, before training waits for the disk
MAX_PENDING = 2
# slots read from the replay buffers come in multiples of this, to compile few reads
READ_ROUNDING = 64


def positions_path(directory: str | Path, epoch: int) -> str:
    """The file of the positions whose games finished in `epoch`."""
    return os.path.join(directory, f"positions-{epoch}.npy")


def position_epochs(directory: str | Path) -> list[int]:
    """The epochs with a file of positions in `directory`, in order."""
    if not os.path.isdir(directory):
        return []
    matches = (POSITIONS_FILE.fullmatch(name) for name in os.listdir(directory))
    return sorted(int(match.group(1)) for match in matches if match)


def drop_positions_from(directory: str | Path, epoch: int) -> None:
    """Deletes the files of `epoch` and later: a run going back to `epoch` saves them again."""
    for saved in position_epochs(directory):
        if saved >= epoch:
            os.remove(positions_path(directory, saved))


def to_rows(experiences: BaseExperience, epoch: int) -> np.ndarray:
    """Experiences with one leading dimension as a structured array, a field for each of
    `BaseExperience`'s, and `epoch`."""
    arrays = {
        field.name: np.asarray(getattr(experiences, field.name))
        for field in fields(BaseExperience)
    }
    count = len(arrays["reward"])
    rows = np.empty(
        count,
        dtype=[
            *((name, x.dtype, x.shape[1:]) for name, x in arrays.items()),
            ("epoch", np.int32),
        ],
    )
    for name, x in arrays.items():
        rows[name] = x
    rows["epoch"] = epoch
    return rows


def from_rows(rows: np.ndarray) -> BaseExperience:
    """`to_rows`'s inverse, but for the epoch."""
    arrays: dict[str, Any] = {
        field.name: rows[field.name] for field in fields(BaseExperience)
    }
    return BaseExperience(**arrays)


@jax.jit
def _take_slots(buffer: BaseExperience, slots: jax.Array) -> BaseExperience:
    """Each environment's entries at `slots`, (environments, slots). The buffers' first two
    dimensions are environments, capacity."""
    return jax.vmap(lambda env, at: jax.tree.map(lambda x: x[at], env))(buffer, slots)


class PositionArchive:
    """Writes a run's positions (see the module's docstring), on a thread of its own: training
    only waits to read them from the device, or if the disk falls two epochs behind. A failed
    write raises on the training loop's next call."""

    def __init__(
        self, directory: str, capacity: int, from_epoch: int, parent: dict | None
    ):
        """
        Args:
            directory: where to write, the run's checkpoint directory
            capacity: the replay buffers' capacity
            from_epoch: the first epoch to save: its positions are those whose games finish in it
            parent: the state the run forked from (see `Trainer.parent`), None if it didn't
        """
        self.directory, self.capacity, self.from_epoch = directory, capacity, from_epoch
        os.makedirs(directory, exist_ok=True)
        tmp = os.path.join(directory, ARCHIVE_META + ".tmp")
        Path(tmp).write_text(json.dumps({"from_epoch": from_epoch, "parent": parent}))
        os.replace(tmp, os.path.join(directory, ARCHIVE_META))
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="positions")
        self.pending: list[Future] = []

    def start(self, state: ReplayBufferState, moves: jax.Array) -> None:
        """Takes what's in the replay buffers as saved, and those of games in progress as
        waiting: called with the state the run starts with. `moves` counts each environment's."""
        self.saved_to, self.next_idx, self.moves = self._read_counts(state, moves)

    @staticmethod
    def _read_counts(state: ReplayBufferState, moves: jax.Array) -> tuple:
        """Host copies, so that nothing holds on to arrays self-play is donated (see
        `Trainer.collect_steps`)."""
        arrays = (state.episode_start_idx, state.next_idx, moves)
        return jax.device_get(jax.tree.map(jnp.copy, arrays))

    def save(self, epoch: int, state: ReplayBufferState, moves: jax.Array) -> None:
        """Saves the positions of the games finished since the last call, as those of `epoch`.

        Args:
            epoch: the epoch that just ended
            state: the replay buffers
            moves: moves made so far, per environment

        Raises:
            ValueError: if the replay buffer has wrapped around past positions not yet saved
        """
        self._raise_failures()
        finished_to, next_idx, moves_now = self._read_counts(state, moves)
        waiting = (self.next_idx - self.saved_to) % self.capacity
        if (waiting + moves_now - self.moves >= self.capacity).any():
            raise ValueError(
                f"a replay buffer of capacity {self.capacity} can't keep an epoch's moves and "
                "the games in progress until their positions are saved: use a larger one"
            )
        counts = (finished_to - self.saved_to) % self.capacity
        saved_to = self.saved_to
        self.saved_to, self.next_idx, self.moves = finished_to, next_idx, moves_now
        if epoch < self.from_epoch:
            return
        rounded = READ_ROUNDING * math.ceil(max(int(counts.max()), 1) / READ_ROUNDING)
        offsets = np.arange(min(rounded, self.capacity))
        slots = (saved_to[:, None] + offsets) % self.capacity
        taken = _take_slots(state.buffer, jnp.asarray(slots, dtype=jnp.int32))
        # the read is finished before self-play can change the buffers in place
        jax.block_until_ready(taken)
        self.pending.append(
            self.writer.submit(self._write, epoch, taken, offsets < counts[:, None])
        )
        while len(self.pending) > MAX_PENDING:
            self.pending.pop(0).result()

    def _write(self, epoch: int, taken: BaseExperience, finished: np.ndarray) -> None:
        rows = to_rows(jax.tree.map(lambda x: np.asarray(x)[finished], taken), epoch)
        path = positions_path(self.directory, epoch)
        # write to a temporary file first so an interrupted save doesn't leave a partial file
        with open(path + ".tmp", "wb") as f:
            np.save(f, rows)
        os.replace(path + ".tmp", path)

    def _raise_failures(self) -> None:
        for future in [f for f in self.pending if f.done()]:
            self.pending.remove(future)
            future.result()

    def flush(self) -> None:
        """Waits for what's been passed to `save` to be on disk."""
        while self.pending:
            self.pending.pop(0).result()

    def close(self) -> None:
        """Finishes what can be finished, as the run ends for any reason."""
        self.writer.shutdown(wait=True)


@dataclass(frozen=True)
class PositionFile:
    """One epoch's file of a run's positions.

    Attributes:
        path: the file
        epoch: its epoch counted from the start of the run's lineage: a fork's own epoch, plus the
            epochs its ancestors had done where they forked
    """

    path: str
    epoch: int

    def count(self) -> int:
        """Its number of positions."""
        return len(read_memmap(self.path))


def read_memmap(path: str) -> Any:
    """A file of positions, mapped rather than read: only what's used of it is."""
    return np.load(path, mmap_mode="r")


def _forked_at(parent: dict | None) -> int:
    """The epochs of lineage before a run with this parent starts."""
    if parent is None:
        return 0
    return parent["epoch"] + _forked_at(parent["parent"])


def lineage(directory: str | Path) -> list[PositionFile]:
    """The positions behind a run: its own files, and those of its ancestors from before each
    forked, oldest first. An ancestor that saved no positions is reported and skipped.

    The epoch a run forked at divides its parent's games exactly: those finished before it are all
    the fork's replay buffers could hold, and games in progress there aren't carried on in the fork.

    Args:
        directory: the run's checkpoint directory

    Returns:
        list[PositionFile]: the files, with epochs counted along the lineage
    """
    meta_path = os.path.join(directory, ARCHIVE_META)
    if not os.path.exists(meta_path):
        print(f"no positions were saved in {directory}", flush=True)
        return []
    parent = json.loads(Path(meta_path).read_text())["parent"]
    start = _forked_at(parent)
    own = [
        PositionFile(positions_path(directory, epoch), start + epoch)
        for epoch in position_epochs(directory)
    ]
    if parent is None:
        return own
    ancestors = lineage(os.path.dirname(parent["path"]))
    return [file for file in ancestors if file.epoch < start] + own


def select_files(
    directories: Sequence[str | Path],
    from_epoch: int | None = None,
    to_epoch: int | None = None,
) -> list[PositionFile]:
    """The files of the lineages of `directories`, each once (forks share their ancestors'),
    with `from_epoch` <= lineage epoch < `to_epoch`."""
    files = {file.path: file for d in directories for file in lineage(d)}
    return [
        file
        for file in sorted(files.values(), key=lambda file: (file.epoch, file.path))
        if (from_epoch is None or file.epoch >= from_epoch)
        and (to_epoch is None or file.epoch < to_epoch)
    ]
