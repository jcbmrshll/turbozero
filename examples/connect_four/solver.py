"""Pascal Pons's Connect Four solver (https://github.com/PascalPons/connect4), which scores every
legal move of a position exactly, and the player it makes at each level.

A position is the moves that led to it, as the solver reads them: a string of columns counted
from 1, "" for the empty board. A move's score is for the player making it: 0 for a draw,
positive for a win, the bigger the sooner (see `plies_to_end`), negative for a loss, the more
negative the sooner. Both players are taken to play for that, so the score of a position's best
move says how the game ends under perfect play, and how soon.

Processes run on the cores this one may use (e.g. with `taskset`). Each takes about 120 MB, most
of it its opening book, the solved positions of up to 14 moves.
"""

import os
import queue
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"),
    "turbozero",
    "connect_four",
)
ROWS, COLUMNS = 6, 7
# the score of a move the solver says isn't legal: its column is full
ILLEGAL = -1000
# at this level or above, a player sees to the end of any game
PERFECT_LEVEL = ROWS * COLUMNS


def available_cpus() -> int:
    """The cores this process may run on (and the solver processes it starts, which inherit
    them, e.g. from `taskset`), or all of them where that can't be told."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def wins_now(score: int, ply: int) -> bool:
    """Whether a move, with this score, `ply` moves into a game, ends the game by winning it: no
    later win can score as much."""
    return score == (PERFECT_LEVEL + 1 - ply) // 2


def plies_to_end(scores: np.ndarray, ply: int) -> np.ndarray:
    """How many moves a game lasts from a position `ply` moves into it, the move scored included,
    under perfect play after it. The first player's discs end up on odd counts, the second's on
    even ones, and a win with the board holding T discs scores (43 - T) // 2 + (T even).

    Args:
        scores: the moves' scores, for the player making them (any shape)
        ply: moves played before them
    """
    # the side that wins each move's game, 0 for the first player
    mover = ply % 2
    winner = np.where(scores > 0, mover, 1 - mover)
    size = np.abs(scores)
    total = np.where(winner == 0, 43 - 2 * size, 44 - 2 * size)
    return np.where(scores == 0, PERFECT_LEVEL, total) - ply


def choose_move(
    scores: np.ndarray, ply: int, level: int, rng: np.random.Generator
) -> int:
    """The move of a player that looks `level` moves ahead, this move included, knowing nothing
    but how games that end within them end: the same as a full minimax search that deep, without
    an evaluation. It takes the soonest win it sees, the latest loss if it sees nothing else, and
    otherwise picks at random between the moves it can't tell apart. So level 0 plays at random,
    level 1 takes a win, level 2 also stops the opponent's, and `PERFECT_LEVEL` is perfect.

    Args:
        scores: (7,) the solver's score of each column
        ply: moves played so far
        level: moves it looks ahead
        rng: picks between equal moves
    """
    legal = scores != ILLEGAL
    seen = np.where(plies_to_end(scores, ply) <= level, scores, 0)
    seen = np.where(legal, seen, ILLEGAL)
    return int(rng.choice(np.flatnonzero(seen == seen.max())))


def sequence(moves: Sequence[int]) -> str:
    """Actions (columns from 0) as the solver reads them."""
    return "".join(str(int(move) + 1) for move in moves)


def actions(position: str) -> list[int]:
    """The solver's position as actions (columns from 0)."""
    return [int(c) - 1 for c in position]


def canonical(position: str) -> tuple:
    """What a position's mirror image and every other move order to it share: the squares of each
    side's discs."""
    heights = [0] * COLUMNS
    discs: tuple[list, list] = ([], [])
    for ply, column in enumerate(actions(position)):
        discs[ply % 2].append((column, heights[column]))
        heights[column] += 1
    return min(
        tuple(
            tuple(sorted((COLUMNS - 1 - c if flip else c, r) for c, r in side))
            for side in discs
        )
        for flip in (False, True)
    )


class SolverProcess:
    """A solver process: one position at a time."""

    def __init__(self, directory: Path):
        self.process = subprocess.Popen(
            [str(directory / "c4solver"), "-a", "-b", str(directory / "7x6.book")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def analyse(self, position: str) -> np.ndarray:
        """Each column's score, from a position no game has ended in (the solver doesn't answer
        for any other)."""
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            self.process.stdin.write(position + "\n")
            self.process.stdin.flush()
            reply = self.process.stdout.readline().split()
        except BrokenPipeError:
            reply = []
        if not reply:
            raise RuntimeError(f"the solver stopped on position {position!r}")
        return np.array(reply[-COLUMNS:], dtype=np.int16)

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        self.process.wait()


class Solver:
    """Solver processes scoring positions in parallel, keeping every answer: the games of a match
    share their openings' positions, and every network meets many of the same ones.

    Attributes:
        name: what results against it are logged as
    """

    name = "solver"

    def __init__(self, directory: Path | None = None, workers: int | None = None):
        """
        Args:
            directory: where the solver is installed (default: `default_dir`)
            workers: processes (default: one a core this process may use, up to 8)

        Raises:
            ValueError: if the solver isn't installed
        """
        self.directory = directory or self.default_dir()
        if not self.installed(self.directory):
            raise ValueError(
                f"no solver in {self.directory}: run examples/connect_four/setup_solver.sh"
            )
        self.workers = workers or min(available_cpus(), 8)
        self.processes: list[SolverProcess] = []
        self.known: dict[str, np.ndarray] = {}

    @staticmethod
    def default_dir() -> Path:
        return Path(os.environ.get("SOLVER_DIR", CACHE_DIR / "solver"))

    @staticmethod
    def installed(directory: Path) -> bool:
        return (directory / "c4solver").exists() and (directory / "7x6.book").exists()

    def analyse(self, positions: Sequence[str]) -> np.ndarray:
        """Each position's scores, (len(positions), 7) int16: ILLEGAL where a column is full."""
        todo: queue.SimpleQueue[str] = queue.SimpleQueue()
        for position in dict.fromkeys(positions):
            if position not in self.known:
                todo.put(position)
        if not todo.empty():
            # started when first needed
            if not self.processes:
                self.processes = [
                    SolverProcess(self.directory) for _ in range(self.workers)
                ]

            def drain(process: SolverProcess) -> None:
                while True:
                    try:
                        position = todo.get_nowait()
                    except queue.Empty:
                        return
                    self.known[position] = process.analyse(position)

            with ThreadPoolExecutor(len(self.processes)) as pool:
                list(pool.map(drain, self.processes))
        if not len(positions):
            return np.zeros((0, COLUMNS), dtype=np.int16)
        return np.stack([self.known[position] for position in positions])

    def close(self) -> None:
        for process in self.processes:
            process.close()
        self.processes = []
