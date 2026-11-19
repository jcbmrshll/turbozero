import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from turbozero.evaluators.mcts.state import MCTSTree


def random_argmax(key: jax.Array, scores: jax.Array) -> jax.Array:
    """Index of the highest score over the last axis, exact ties broken at
    random rather than to the lowest index.
    """
    noise = jax.random.uniform(key, scores.shape)
    return jnp.where(scores == scores.max(axis=-1, keepdims=True), noise, -1.0).argmax(
        axis=-1
    )


class MCTSActionSelector:
    """Picks the action to traverse from a node of the search tree."""

    def __call__(
        self, key: jax.Array, tree: MCTSTree, index: ArrayLike, discount: float
    ) -> jax.Array:
        """The action from node `index`, ties broken with `key`."""
        raise NotImplementedError()

    def get_config(self) -> dict:
        """The selector's settings, as checkpoints record them."""
        return {}


class PUCTSelector(MCTSActionSelector):
    """AlphaZero's (and MuZero's) PUCT rule: the action maximizing
    `Q + c * P * sqrt(N) / (1 + n)`, with `c = c_init + log((N + c_base + 1) /
    c_base)` and Q mapped from [-1, 1] to [0, 1], unvisited children at 0, a
    loss, as in AlphaZero's and MuZero's pseudocode for board games. `c_base` is
    the visit count at which `c` has grown by log(2).
    """

    def __init__(self, c_init: float = 1.25, c_base: float = 19652):
        self.c_init = c_init
        self.c_base = c_base

    def get_config(self) -> dict:
        return {"c_init": self.c_init, "c_base": self.c_base}

    def __call__(
        self, key: jax.Array, tree: MCTSTree, index: ArrayLike, discount: float
    ) -> jax.Array:
        node = tree.data_at(index)
        # the children's values are for the player to move at each of them
        q_values = tree.get_child_data("q", index) * discount
        n_values = tree.get_child_data("n", index)
        q_values = jnp.where(n_values > 0, (q_values + 1) / 2, 0.0)
        c = self.c_init + jnp.log((node.n + self.c_base + 1) / self.c_base)
        u_values = c * node.p * jnp.sqrt(node.n) / (n_values + 1)
        return random_argmax(key, q_values + u_values)
