import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from turbozero.evaluators.mcts.state import MCTSTree


def random_argmax(key: jax.Array, scores: jax.Array) -> jax.Array:
    """Index of the highest score, exact ties broken at random rather than to the lowest index
    (as mctx does, though its tiny noise is only a few float32 ulps at scores near 1).

    Args:
        key: rng
        scores: scores to maximize over the last axis

    Returns:
        jax.Array: index of a highest score
    """
    noise = jax.random.uniform(key, scores.shape)
    return jnp.where(scores == scores.max(axis=-1, keepdims=True), noise, -1.0).argmax(
        axis=-1
    )


def bounded_q_values(
    q_values: jax.Array,
    child_n_values: jax.Array,
) -> jax.Array:
    """Map Q-values from the fixed range [-1, 1] to [0, 1], unvisited children to 0.

    AlphaZero's and MuZero's board-game normalization (MuZero's `KnownBounds(-1, 1)`):
    how far apart the children's values are matters, so the search explores more where
    they're close or the position is lost.

    Args:
        q_values: Q-values to normalize, in [-1, 1]
        child_n_values: visit counts of child nodes

    Returns:
        jax.Array: normalized Q-values
    """
    return jnp.where(child_n_values > 0, (q_values + 1) / 2, 0.0)


class MCTSActionSelector:
    """Base class for action selection in MCTS.

    Is callable, selects an action given a search tree state.
    """

    def __call__(
        self, key: jax.Array, tree: MCTSTree, index: ArrayLike, discount: float
    ) -> jax.Array:
        """Selects an action given a search tree state. Implemented by subclasses."""
        raise NotImplementedError()

    def get_config(self) -> dict:
        """Returns the configuration of the action selector. Used for logging."""
        return {}


class PUCTSelector(MCTSActionSelector):
    """PUCT (Polynomial Upper Confidence Trees) action selector, as in AlphaZero.

    Picks the action maximizing `Q + c * P * sqrt(N) / (1 + n)`, with
    `c = c_init + log((N + c_base + 1) / c_base)`: AlphaZero's (and MuZero's) rule. Q is on the
    fixed range [-1, 1] (`bounded_q_values`), as in AlphaZero and MuZero's board games.
    """

    def __init__(
        self,
        c_init: float = 1.25,
        c_base: float = 19652,
    ):
        """Initializes a PUCTSelector.

        Args:
            c_init: exploration constant (larger values encourage exploration)
            c_base: visit count over which the exploration constant grows (by log(2) at N = c_base)
        """
        self.c_init = c_init
        self.c_base = c_base

    def get_config(self) -> dict:
        """Returns the configuration of the PUCT action selector. Used for logging."""
        return {
            "c_init": self.c_init,
            "c_base": self.c_base,
        }

    def __call__(
        self, key: jax.Array, tree: MCTSTree, index: ArrayLike, discount: float
    ) -> jax.Array:
        """Selects an action given a search tree state.

        Args:
            key: rng, to break ties
            tree: search tree
            index: index of the node in the search tree to select an action to take from
            discount: discount factor

        Returns:
            jax.Array: id of action to take
        """
        # get child q-values
        node = tree.data_at(index)
        q_values = tree.get_child_data("q", index)
        # apply discount to q-values
        discounted_q_values = q_values * discount
        # get child visit counts
        n_values = tree.get_child_data("n", index)
        # map q-values to [0, 1]
        q_values = bounded_q_values(discounted_q_values, n_values)
        # calculate U-values
        c = self.c_init + jnp.log((node.n + self.c_base + 1) / self.c_base)
        u_values = c * node.p * jnp.sqrt(node.n) / (n_values + 1)
        # PUCT = Q-value + U-value
        puct_values = q_values + u_values
        # select action with highest PUCT value, ties broken at random
        return random_argmax(key, puct_values)
