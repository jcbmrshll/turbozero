"""XOT openings: 10,784 eight-move Othello openings, each ending in a position Edax
(searching 16 moves ahead) judges nearly even, within 2 discs.

Starting games from them instead of the initial position gives deterministic players
varied, balanced games: the usual way to test Othello programs against each other (e.g.
OLIVAW against Edax, https://arxiv.org/abs/2103.17228), each opening played twice with
the colors swapped.

The list, `xot-openings.txt`, is Matthias Berg's "large" XOT list
(https://berg.earthlingz.de/xot/), as used by Egaroucid, NBoard and others.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from game import action, env, metadata

XOT_PATH = Path(__file__).parent / "xot-openings.txt"


def load_xot(path: str | Path = XOT_PATH) -> np.ndarray:
    """The XOT openings as actions, one row of 8 per opening (black moves first).

    Args:
        path: (optional) an opening list, one opening a line in square names, e.g.
            "f5d6c4d3c2b3b4b5"
    """
    lines = Path(path).read_text().split()
    return np.array(
        [[action(line[i : i + 2]) for i in range(0, len(line), 2)] for line in lines],
        dtype=np.int32,
    )


def make_xot_init_fn(openings: np.ndarray, standard_start_fraction: float):
    """An env init fn (see game.init_fn) that starts most episodes from a random one of
    `openings`, already played, and the rest (`standard_start_fraction` of them) from the
    standard start, so that the network still learns the first moves too."""
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
