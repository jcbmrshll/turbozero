"""Openings for test games, as XOT's are for Othello: every 6-move opening that leaves the game
close, the position's score within 2 (see `solver.py`), so that with perfect play it's drawn or
won only with the board nearly full. Mirror images and move orders that reach the same position
count once.

Starting games from them gives deterministic players varied games, and each opening is played
twice with the colors swapped. Since the game is solved, each opening's two games have a known
sum: a perfect player scores exactly 0.5 against the perfect solver.

The list, `openings.txt`, is one opening a line, as the solver writes positions; `build_openings.py`
builds it.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from game import env, metadata
from solver import actions

OPENINGS_PATH = Path(__file__).parent / "openings.txt"


def load_openings(path: str | Path = OPENINGS_PATH) -> np.ndarray:
    """The openings as actions, one row per opening (the first player moves first)."""
    lines = Path(path).read_text().split()
    return np.array([actions(line) for line in lines], dtype=np.int32)


def make_opening_init_fn(openings: np.ndarray, standard_start_fraction: float):
    """An env init fn (see game.init_fn) that starts `standard_start_fraction` of the episodes
    from the empty board, and the rest from a random one of `openings`, already played."""
    moves = jnp.asarray(openings)

    def init_fn(key):
        init_key, pick_key, standard_key = jax.random.split(key, 3)
        state = env.init(init_key)
        opening = moves[jax.random.randint(pick_key, (), 0, len(moves))]
        from_opening = jax.random.uniform(standard_key) >= standard_start_fraction

        def play(ply, state):
            # play the opening's moves, or leave the standard start as it is
            return jax.tree.map(
                lambda new, old: jnp.where(from_opening, new, old),
                env.step(state, opening[ply]),
                state,
            )

        state = jax.lax.fori_loop(0, moves.shape[1], play, state)
        return state, metadata(state)

    return init_fn
