"""Every self-play move, kept on disk: a dataset of whole games, to label and train on later
without any further search (e.g. a fixed teacher's games, for students).

Self-play writes each move it plays to a log on the device (`MoveLog`, one per environment, as
long as the most moves an environment can play in one call of `Trainer.collect_steps`), whether
its search trains or not (e.g. a fast search of a `PlayoutCap`). After each epoch's self-play the
log is copied to the host, emptied, and written by `MoveLogWriter` as `moves-<epoch>.npy` in its
directory: a numpy structured array (see `row_dtype`) of that epoch's moves, environment by
environment, each environment's in the order they were played.

A row's game is (`env`, `game`), `game` counting the environment's games from 0 in the run, so a
game is put back together from its rows across epochs. Its last row says how it ended (`end`):
- `ENDED`: the game is over, and `result` is the player to move's reward for it (a resigned game
  ends this way too)
- `STOPPED`: the game stopped before this row's move (e.g. at the point exact solves take over,
  see `core.training.endgame_solve`): the row has its search but no move (`action` -1), and its
  result is for whoever labels it
- `TRUNCATED`: the game hit the step limit
Games still in progress at the end of the run have no last row.

The log isn't part of a saved training state, so a run that logs moves doesn't continue from
one into the same directory: `MoveLogWriter` refuses a directory that holds moves already.
"""

import json
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

MOVES_FILE = re.compile(r"moves-(\d+)\.npy")
# beside the files: the shapes of the fields packed into bits, to unpack them (see `unpack`)
META_FILE = "moves.json"
# epochs' worth of moves waiting to be written, before self-play waits for the disk
MAX_PENDING = 2

# `end`: how a row's game went on after it
GOES_ON, ENDED, STOPPED, TRUNCATED = 0, 1, 2, 3


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LoggedMove:
    """A move self-play played, as `MoveLog` keeps it.

    Attributes:
        observation: the network's input, of the position the move was played from
        policy: the search's policy target (as the replay buffer stores it)
        legal: the legal actions
        value: the search's root value, for the player to move
        action: the action played, -1 if the game stopped before it (see `STOPPED`)
        player: the player to move
        full: whether the search makes a training position (see `Evaluator.trains_on`), e.g.
            False for a `PlayoutCap`'s fast search
        ply: the environment's step count at the position
        game: the environment's game it's in, counted from 0 in the run
        end: `GOES_ON`, `ENDED`, `STOPPED` or `TRUNCATED` (see the module's docstring)
        result: with `ENDED`, the player to move's reward for the game; else 0
    """

    observation: jax.Array
    policy: jax.Array
    legal: jax.Array
    value: jax.Array
    action: jax.Array
    player: jax.Array
    full: jax.Array
    ply: jax.Array
    game: jax.Array
    end: jax.Array
    result: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MoveLog:
    """The moves of environments since the log was last emptied.

    Attributes:
        rows: `LoggedMove`s, (environments, length, ...)
        count: (environments,) the rows written of each
        games: (environments,) the game each environment is playing, counted from 0
    """

    rows: LoggedMove
    count: jax.Array
    games: jax.Array


def init_move_log(
    num_envs: int, length: int, observation: jax.Array, num_actions: int
) -> MoveLog:
    """An empty log of `length` rows an environment, its games counted from 0.

    Args:
        num_envs: environments
        length: rows an environment: at least the moves it can play before the log is emptied
        observation: a template of one observation (its shape and dtype)
        num_actions: the policy's actions
    """

    def rows(shape: tuple[int, ...], dtype: Any) -> jax.Array:
        return jnp.zeros((num_envs, length, *shape), dtype)

    return MoveLog(
        rows=LoggedMove(
            observation=rows(observation.shape, observation.dtype),
            policy=rows((num_actions,), jnp.float32),
            legal=rows((num_actions,), jnp.bool_),
            value=rows((), jnp.float32),
            action=rows((), jnp.int32),
            player=rows((), jnp.int32),
            full=rows((), jnp.bool_),
            ply=rows((), jnp.int32),
            game=rows((), jnp.int32),
            end=rows((), jnp.int8),
            result=rows((), jnp.float32),
        ),
        count=jnp.zeros((num_envs,), jnp.int32),
        games=jnp.zeros((num_envs,), jnp.int32),
    )


def log_move(log: MoveLog, move: LoggedMove, played: jax.Array) -> MoveLog:
    """One environment's log (no environments dimension) with `move` written, if `played`; the
    environment's next game starts after it if it ends one (`move.end` other than `GOES_ON`).
    `move.game` is taken from the log."""
    at = jnp.where(played, log.count, log.rows.end.shape[0])
    move = replace(move, game=log.games)
    return MoveLog(
        rows=jax.tree.map(
            lambda rows, x: rows.at[at].set(x.astype(rows.dtype), mode="drop"),
            log.rows,
            move,
        ),
        count=log.count + played,
        games=log.games + (played & (move.end != GOES_ON)),
    )


def emptied(log: MoveLog) -> MoveLog:
    """The log with no rows written, its games' counts kept."""
    return replace(log, count=jnp.zeros_like(log.count))


def row_dtype(rows: LoggedMove) -> np.dtype:
    """The files' rows of a log's (host copies, (environments, length, ...)): booleans with more
    than one element are packed into bits (see `unpack`), the policy is float16, and each row has
    its environment (`env`) and the epoch it was written in (`epoch`)."""
    columns = []
    for field in fields(LoggedMove):
        x = np.asarray(getattr(rows, field.name))
        shape = x.shape[2:]
        if x.dtype == bool and shape:
            columns.append((field.name, np.uint8, ((int(np.prod(shape)) + 7) // 8,)))
        elif field.name == "policy":
            columns.append((field.name, np.float16, shape))
        else:
            columns.append((field.name, x.dtype, shape))
    return np.dtype([*columns, ("env", np.int32), ("epoch", np.int32)])


def to_rows(rows: LoggedMove, count: np.ndarray, epoch: int) -> np.ndarray:
    """A log's written rows (host copies) as a file's (see `row_dtype`), environment by
    environment."""
    written = (
        np.arange(rows.end.shape[1]) < np.minimum(count, rows.end.shape[1])[:, None]
    )
    out = np.empty(int(written.sum()), row_dtype(rows))
    for field in fields(LoggedMove):
        x = np.asarray(getattr(rows, field.name))[written]
        if x.dtype == bool and x.ndim > 1:
            flat = x.reshape(len(x), int(np.prod(x.shape[1:])))
            x = np.packbits(flat, axis=1, bitorder="little")
        out[field.name] = x
    out["env"] = np.nonzero(written)[0]
    out["epoch"] = epoch
    return out


def unpack(rows: np.ndarray, name: str, meta: dict) -> np.ndarray:
    """A field of a file's rows as it was logged: those packed into bits (see `row_dtype`)
    unpacked, with their shape from the directory's meta (see `read_meta`)."""
    shape = meta["packed"].get(name)
    if shape is None:
        return rows[name]
    count = int(np.prod(shape))
    bits = np.unpackbits(rows[name], axis=1, count=count, bitorder="little")
    return bits.astype(bool).reshape(len(rows), *shape)


def moves_path(directory: str | Path, epoch: int) -> str:
    """The file of the moves played in `epoch`."""
    return os.path.join(directory, f"moves-{epoch}.npy")


def move_epochs(directory: str | Path) -> list[int]:
    """The epochs with a file of moves in `directory`, in order."""
    if not os.path.isdir(directory):
        return []
    matches = (MOVES_FILE.fullmatch(name) for name in os.listdir(directory))
    return sorted(int(match.group(1)) for match in matches if match)


def read_meta(directory: str | Path) -> dict:
    """The directory's meta: `num_envs`, and `packed`, each packed field's shape."""
    return json.loads(Path(directory, META_FILE).read_text())


def read_moves(directory: str | Path) -> np.ndarray:
    """All the moves in `directory`, epoch by epoch, as one array of a file's rows."""
    epochs = move_epochs(directory)
    if not epochs:
        raise ValueError(f"no moves in {directory}")
    return np.concatenate([np.load(moves_path(directory, e)) for e in epochs])


class MoveLogWriter:
    """Writes logs (see the module's docstring) on a thread of its own: self-play only waits to
    copy them to the host, or if the disk falls `MAX_PENDING` epochs behind. A failed write
    raises on the next call."""

    def __init__(self, directory: str, num_envs: int):
        """
        Args:
            directory: where to write: a new directory, or one without moves

        Raises:
            ValueError: if `directory` holds moves already
        """
        if move_epochs(directory):
            raise ValueError(
                f"{directory} holds moves already: a run logs its moves to a new directory"
            )
        os.makedirs(directory, exist_ok=True)
        self.directory, self.num_envs = directory, num_envs
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moves")
        self.pending: list[Future] = []

    def write(self, epoch: int, log: MoveLog) -> None:
        """Writes the log's rows as `epoch`'s. They're copied to the host first, so the log can
        be emptied (see `emptied`) and written to as soon as this returns."""
        for future in [f for f in self.pending if f.done()]:
            self.pending.remove(future)
            future.result()
        rows, count = jax.device_get((log.rows, log.count))
        if (count > rows.end.shape[1]).any():
            raise ValueError(
                f"an environment played more moves than its move log's {rows.end.shape[1]} "
                "rows in an epoch: some weren't logged"
            )
        self.pending.append(self.writer.submit(self._write, epoch, rows, count))
        while len(self.pending) > MAX_PENDING:
            self.pending.pop(0).result()

    def _write(self, epoch: int, rows: LoggedMove, count: np.ndarray) -> None:
        meta_path = os.path.join(self.directory, META_FILE)
        if not os.path.exists(meta_path):
            packed = {
                field.name: list(np.asarray(getattr(rows, field.name)).shape[2:])
                for field in fields(LoggedMove)
                if np.asarray(getattr(rows, field.name)).dtype == bool
                and np.asarray(getattr(rows, field.name)).ndim > 2
            }
            meta = {"num_envs": self.num_envs, "packed": packed}
            Path(meta_path).write_text(json.dumps(meta))
        path = moves_path(self.directory, epoch)
        # a temporary file first, so that an interrupted write leaves no partial file
        with open(path + ".tmp", "wb") as f:
            np.save(f, to_rows(rows, count, epoch))
        os.replace(path + ".tmp", path)

    def flush(self) -> None:
        """Waits for what's been passed to `write` to be on disk."""
        while self.pending:
            self.pending.pop(0).result()

    def close(self) -> None:
        """Finishes what can be finished, as the run ends for any reason."""
        self.writer.shutdown(wait=True)
