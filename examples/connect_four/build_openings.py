"""Builds the list of openings for test games, `openings.txt` (see `openings.py`).

examples/connect_four/setup_solver.sh     # once
uv run examples/connect_four/build_openings.py
"""

from pathlib import Path

import numpy as np
from solver import ILLEGAL, Solver, canonical, wins_now

OPENINGS_PATH = Path(__file__).parent / "openings.txt"
OPENING_PLIES = 6
MAX_SCORE = 2


def continuations(positions: list[str], scores: np.ndarray) -> list[str]:
    """Each position after each move that doesn't end the game, one of each distinct position."""
    seen = {}
    for position, row in zip(positions, scores):
        for column, score in enumerate(row):
            if score == ILLEGAL or wins_now(int(score), len(position)):
                continue
            after = position + str(column + 1)
            seen.setdefault(canonical(after), after)
    return list(seen.values())


def build_openings(
    solver, plies: int = OPENING_PLIES, max_score: int = MAX_SCORE
) -> list[str]:
    """Every distinct `plies`-move opening whose position scores within `max_score`, sorted."""
    positions = [""]
    for _ in range(plies):
        positions = continuations(positions, solver.analyse(positions))
    best = solver.analyse(positions).max(axis=1)
    return sorted(p for p, b in zip(positions, best) if abs(int(b)) <= max_score)


def main():
    solver = Solver()
    openings = build_openings(solver)
    solver.close()
    OPENINGS_PATH.write_text("\n".join(openings) + "\n")
    print(f"{OPENINGS_PATH}: {len(openings)} openings")


if __name__ == "__main__":
    main()
