"""Othello's endgame solved exactly (`endgame.cpp`, compiled on first use with nothing but g++): the
final disc margin for the player to move under perfect play by both sides, empties going to the
winner, as Edax and Egaroucid count them, and a move that reaches it. Or, about ten times faster,
only whether it's a win, a draw or a loss (`exact=False`): then a margin's sign is right but its
size only a bound (at least it for a win, at most it for a loss), and the move wins or draws if
the position does, and is any move if it's lost.

Positions are bitboards from the side to move's view: bit i is square i, a1 = 0, b1 = 1, ..., h8 = 63,
row by row, as pgx's actions. Passes are handled inside: a player with no move passes, and the game
ends when neither has one.

    margins, moves = solve_batch(me, opp, threads=8)  # many positions, a thread to each
    solver = Solver(threads=4)  # one at a time, on all threads, with a deadline
    solved = solver.solve(me, opp, seconds=1.5)
"""

import ctypes
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from native_target import target_flags

SOURCE = Path(__file__).parent / "endgame.cpp"
CACHE_DIR = Path(
    os.environ.get("TURBOZERO_CACHE", Path.home() / ".cache" / "turbozero" / "othello")
)
PASS = 64
# a move once the game's over
NO_MOVE = -1
# the default table: 2^21 entries of 24 bytes, 48 MB
TABLE_BITS = 21
# a batch's tables, one to a thread: 2^20 entries, 24 MB each
BATCH_TABLE_BITS = 20


def library_path() -> Path:
    """The compiled solver, built for this machine's x86-64 level, named after its source and
    target."""
    target = target_flags()
    digest = hashlib.sha256(SOURCE.read_bytes() + str(target).encode()).hexdigest()
    path = CACHE_DIR / "endgame" / f"endgame_{digest[:16]}.so"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # named per process: several may be compiling at once
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(
            ["g++", "-std=c++17", "-O3", *target, "-shared", "-fPIC"]
            + [str(SOURCE), "-o", str(partial), "-lpthread"],
            check=True,
        )
        partial.replace(path)
    return path


def load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(library_path()))
    u64 = ctypes.c_uint64
    c_int = ctypes.c_int
    pointer = ctypes.c_void_p
    signatures = {
        "endgame_solver_new": (pointer, [c_int, c_int]),
        "endgame_solver_free": (None, [pointer]),
        "endgame_solver_solve": (
            c_int,
            [pointer, u64, u64, ctypes.c_double, c_int]
            + [ctypes.POINTER(c_int), ctypes.POINTER(c_int)],
        ),
        "endgame_solver_nodes": (u64, [pointer]),
        "endgame_solve_batch": (
            None,
            [pointer, pointer, c_int, c_int, c_int, c_int, pointer, pointer],
        ),
        "endgame_move_scores": (None, [u64, u64, c_int, pointer]),
        "endgame_moves": (u64, [u64, u64]),
        "endgame_moves_scalar": (u64, [u64, u64]),
        "endgame_flips": (u64, [c_int, u64, u64]),
        "endgame_flips_scalar": (u64, [c_int, u64, u64]),
        "endgame_stable": (u64, [u64, u64]),
        "endgame_minimax": (c_int, [u64, u64]),
    }
    for name, (restype, argtypes) in signatures.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


LIB = load_library()


def solve_batch(
    me: np.ndarray,
    opp: np.ndarray,
    threads: int = 1,
    exact: bool = True,
    table_bits: int = BATCH_TABLE_BITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Each position of `me` and `opp` (uint64, the discs of the player to move and of the
    opponent) solved, on `threads` threads, a position to a thread at a time, each thread with a
    table of 2^table_bits entries. Returns (margins, moves), int8: the final margin for the
    player to move, and a best move, 0-63, `PASS`, or `NO_MOVE` once the game's over. Releases
    the GIL while it runs."""
    me = np.ascontiguousarray(me, np.uint64)
    opp = np.ascontiguousarray(opp, np.uint64)
    if me.shape != opp.shape or me.ndim != 1:
        raise ValueError(
            f"me {me.shape} and opp {opp.shape} must be the same 1-d shape"
        )
    if np.any(me & opp):
        raise ValueError("a square has both players' discs")
    margins = np.zeros(len(me), np.int8)
    moves = np.zeros(len(me), np.int8)
    LIB.endgame_solve_batch(
        me.ctypes.data,
        opp.ctypes.data,
        len(me),
        threads,
        table_bits,
        exact,
        margins.ctypes.data,
        moves.ctypes.data,
    )
    return margins, moves


def move_margins(me: int, opp: int, table_bits: int = BATCH_TABLE_BITS) -> np.ndarray:
    """(65) the exact final margin after each legal move, from the mover's side; -128 for moves
    that aren't legal. A pass, `PASS`, is legal only with no other move; none is once the game's
    over."""
    out = np.zeros(65, np.int8)
    LIB.endgame_move_scores(me, opp, table_bits, out.ctypes.data)
    return out


@dataclass(frozen=True)
class Solved:
    margin: int  # for the player to move
    move: int  # 0-63, `PASS`, or `NO_MOVE` once the game's over


class Solver:
    """Solves one position at a time on `threads` threads, which share a table of 2^table_bits
    entries kept for the solver's life: within a game, what it learned of one position helps with
    the next."""

    def __init__(self, threads: int = 1, table_bits: int = TABLE_BITS):
        self.handle = LIB.endgame_solver_new(threads, table_bits)

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            LIB.endgame_solver_free(self.handle)

    def solve(
        self, me: int, opp: int, seconds: float = 0.0, exact: bool = True
    ) -> Solved | None:
        """The position solved, or None if it took more than `seconds` (if more than 0)."""
        margin, move = ctypes.c_int(), ctypes.c_int()
        done = LIB.endgame_solver_solve(
            self.handle,
            me,
            opp,
            seconds,
            exact,
            ctypes.byref(margin),
            ctypes.byref(move),
        )
        return Solved(margin.value, move.value) if done else None

    def nodes(self) -> int:
        """Positions searched since it was made."""
        return LIB.endgame_solver_nodes(self.handle)
