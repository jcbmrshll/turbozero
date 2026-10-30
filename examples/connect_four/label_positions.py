"""Builds the solver-labelled set (see solver_set.py): positions from games the solver plays with
some of its moves at random, with every legal move scored exactly.

    examples/connect_four/setup_solver.sh     # once
    uv run examples/connect_four/label_positions.py

No training run is needed: --noise 0.1 0.3 1 samples --per-source distinct positions from games in
which 10% of the moves are at random, the sort a strong player meets, from games with 30%, and from
wholly random games, the sort a young network plays. The same --seed gives the same set. Only
positions with a choice of move are kept, and mirror images and move orders that reach the same
position count once.

The solver takes milliseconds a position past its opening book, so 5000 take about 15 seconds on
--workers 8 (one core each; run it under `taskset` to keep it to some cores on a shared machine).
"""

import argparse
import time

import numpy as np
from solver import ILLEGAL, PERFECT_LEVEL, Solver, canonical, wins_now
from solver_set import DEFAULT_PATH, SolverSet

# games to play at a time, until a source has its positions
BATCH_GAMES = 256


def play_noisy_games(
    solver, noise: float, games: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Plays `games` games from the empty board, the solver's best move each turn, or with
    probability `noise` a random legal one.

    Returns:
        every position played from, with its scores: {position: (7,) int16}
    """
    seen = {}
    live = [""] * games
    while live:
        scores = solver.analyse(live)
        seen.update(zip(live, scores))
        after = []
        for position, row in zip(live, scores):
            legal = np.flatnonzero(row != ILLEGAL)
            best = np.flatnonzero(row == row[legal].max())
            column = rng.choice(legal if rng.random() < noise else best)
            if wins_now(int(row[column]), len(position)):
                continue
            if len(position) + 1 < PERFECT_LEVEL:
                after.append(position + str(column + 1))
        live = after
    return seen


def build(
    solver,
    noises: list[float],
    per_source: int,
    seed: int = 0,
    progress=None,
) -> SolverSet:
    """A set of `per_source` positions from each of `noises`, sampled from those with a choice of
    move, and distinct across the sources.

    Args:
        solver: scores positions (see `solver.Solver.analyse`)
        noises: each source's fraction of random moves
        per_source: positions to sample from each
        seed: for the games and the sampling
        progress: (optional) called with each source's name when it's done
    """
    rng = np.random.default_rng(seed)
    positions, scores, source = [], [], []
    taken = set()
    names = [f"noise{round(noise * 100)}" for noise in noises]
    for i, (name, noise) in enumerate(zip(names, noises)):
        found: dict[tuple, tuple[str, np.ndarray]] = {}
        while len(found) < per_source:
            before = len(found)
            for position, row in play_noisy_games(
                solver, noise, BATCH_GAMES, rng
            ).items():
                if (row != ILLEGAL).sum() < 2:
                    continue
                key = canonical(position)
                if key not in taken and key not in found:
                    found[key] = (position, row)
            if len(found) == before:
                raise ValueError(f"{name}: only {before} positions to be found")
        keys = sorted(found)
        for j in sorted(rng.choice(len(keys), per_source, replace=False)):
            position, row = found[keys[j]]
            taken.add(keys[j])
            positions.append(position)
            scores.append(row)
            source.append(i)
        if progress is not None:
            progress(name)
    return SolverSet(
        positions=np.array(positions),
        scores=np.stack(scores),
        source=np.array(source, dtype=np.int8),
        source_names=names,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Builds the solver-labelled set of Connect Four positions."
    )
    parser.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 1.0],
        help="each source's fraction of random moves",
    )
    parser.add_argument("--per-source", type=int, default=1667)
    parser.add_argument("--workers", type=int, default=8, help="solver processes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_PATH))
    args = parser.parse_args()

    start = time.perf_counter()
    solver = Solver(workers=args.workers)
    solver_set = build(
        solver,
        args.noise,
        args.per_source,
        args.seed,
        progress=lambda name: print(
            f"{name}: done at {time.perf_counter() - start:.0f} s", flush=True
        ),
    )
    solver.close()
    solver_set.save(args.out)
    print(f"{args.out}: {len(solver_set)} positions", flush=True)


if __name__ == "__main__":
    main()
