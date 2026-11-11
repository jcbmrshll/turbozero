"""Exact endgame labels for Othello self-play (train.py's --solve-empties, see
core/training/endgame_solve.py): a game's solve point is its first position with `empties` or
fewer empty squares, and an exact solver's result for it becomes the value target of every
position before it; with --solve-play-past, the game plays on a few moves, each position stored
past the solve point getting its own solved result.

Solvers:
- `native`: endgame.py's C++ solver, a win/draw/loss solve (only the result's sign is needed),
  compiled on first use with g++ (Modal's training image too).
- `edax`: Edax's -solve on a batch file, one process per batch of positions (setup_edax.sh
  builds it; Modal's training image doesn't). For prototyping.
"""

import math
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from engines import Edax

SOLVERS = ("native", "edax")
# positions an Edax process solves at a time
EDAX_BATCH = 32
# a solve's Edax hash table: 2^20 entries, about 32 MB a process
EDAX_HASH_BITS = 20


def stops_at(empties: int) -> Callable[[Any], jax.Array]:
    """The stop function (see `EndgameSolve.stop_fn`) of games stopping at `empties` or fewer empty
    squares."""
    return lambda state: 64 - jnp.sum(state.observation, dtype=jnp.int32) <= empties


def bitboards(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bitboards of the player to move's discs and the opponent's, from (n, 8, 8, 2) observations:
    bit i is square i, a1 = 0 to h8 = 63, row by row (as actions number them)."""
    discs = np.asarray(observation, dtype=bool).reshape(len(observation), 64, 2)
    bits = np.uint64(1) << np.arange(64, dtype=np.uint64)
    return (
        (discs[..., 0] * bits).sum(-1, dtype=np.uint64),
        (discs[..., 1] * bits).sum(-1, dtype=np.uint64),
    )


def results(margins: np.ndarray, current_player: np.ndarray) -> np.ndarray:
    """(n, 2) each player's result, as the environment's rewards at a game's end: 1 for a win, -1
    for a loss, 0 for a draw, from the exact final margins of the players to move."""
    sign = np.sign(np.asarray(margins)).astype(np.float32)
    out = np.empty((len(sign), 2), np.float32)
    rows = np.arange(len(sign))
    player = np.asarray(current_player)
    out[rows, player] = sign
    out[rows, 1 - player] = -sign
    return out


def obf(me: int, opp: int) -> str:
    """A position in Edax's obf format, the player to move as X."""
    return (
        "".join("X" if me >> i & 1 else "O" if opp >> i & 1 else "-" for i in range(64))
        + " X;"
    )


def edax_margins(
    me: np.ndarray, opp: np.ndarray, threads: int, edax_dir: Path | None = None
) -> np.ndarray:
    """Exact final margins for the players to move, from Edax's -solve: `threads` processes at a
    time, each solving a batch of positions."""
    edax_dir = edax_dir or Edax.default_dir()
    if not Edax.installed(edax_dir):
        raise FileNotFoundError(
            f"no Edax in {edax_dir}: run examples/othello/setup_edax.sh"
        )
    boards = [obf(int(m), int(o)) for m, o in zip(me, opp, strict=True)]

    def solve(batch: list[str]) -> list[int]:
        with tempfile.NamedTemporaryFile("w", suffix=".obf") as f:
            f.write("\n".join(batch) + "\n")
            f.flush()
            out = subprocess.run(
                [str(edax_dir / "bin" / "edax"), "-solve", f.name, "-n", "1",
                 "-h", str(EDAX_HASH_BITS), "-book-usage", "off", "-l", "60"],
                cwd=edax_dir, capture_output=True, text=True, check=True,
            ).stdout  # fmt: skip
        rows = re.findall(r"^\s*(\d+)\|\s*(\d+)(@\d+%)?\s+([+-]\d+)", out, re.MULTILINE)
        if [int(r[0]) for r in rows] != list(range(1, len(batch) + 1)) or any(
            r[2] for r in rows
        ):
            raise RuntimeError(f"Edax didn't solve its batch exactly:\n{out[-1000:]}")
        return [int(r[3]) for r in rows]

    batches = [boards[i : i + EDAX_BATCH] for i in range(0, len(boards), EDAX_BATCH)]
    with ThreadPoolExecutor(threads) as pool:
        return np.array([m for b in pool.map(solve, batches) for m in b], np.int32)


def make_solve_fn(solver: str, threads: int) -> Callable[[Any], np.ndarray]:
    """The solve function (see `EndgameSolve.solve_fn`) of `solver`, on `threads` cores."""
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")

    if solver == "native":
        import endgame

        def margins(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
            return endgame.solve_batch(me, opp, threads=threads, exact=False)[0]
    else:

        def margins(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
            return edax_margins(me, opp, threads)

    def solve(positions: Any) -> np.ndarray:
        me, opp = bitboards(positions.observation)
        return results(margins(me, opp), positions.current_player)

    return solve


def solve_slots(
    cycles: int, advance_every: int, empties: int, play_past: float = 0.0
) -> int:
    """`EndgameSolve.slots` for self-play of `cycles` search cycles an epoch, playing moves every
    `advance_every` cycles, with solve points at `empties` and `play_past` moves past them on
    average: two epochs of solves for the most games that can reach their solve points, each at
    least as long as from an XOT opening (52 empty squares), each with a solve there and room for
    twice its mean moves past it (each stored and solved, at most)."""
    moves = -(-cycles // advance_every)
    games = -(-moves // max(52 - empties, 1))
    return 2 * math.ceil(games * (1 + 2 * play_past)) + 2


def default_threads() -> int:
    """The cores this process may run on."""
    return len(os.sched_getaffinity(0))
