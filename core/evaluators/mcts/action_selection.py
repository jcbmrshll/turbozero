import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from core.evaluators.mcts.state import MCTSTree


def normalize_q_values(
    q_values: jax.Array,
    child_n_values: jax.Array,
    parent_q_value: ArrayLike,
    epsilon: ArrayLike,
) -> jax.Array:
    """Normalize Q-values to be in the range [0, 1].

    Args:
        q_values: Q-values to normalize
        child_n_values: visit counts of child nodes
        parent_q_value: Q-value of the parent node
        epsilon: small value to avoid division by zero

    Returns:
        jax.Array: normalized Q-values
    """
    # unvisited children take the parent's value so they don't affect the range
    safe_q_values = jnp.where(child_n_values > 0, q_values, parent_q_value)
    min_value = jnp.minimum(parent_q_value, jnp.min(safe_q_values, axis=-1))
    max_value = jnp.maximum(parent_q_value, jnp.max(safe_q_values, axis=-1))
    completed_by_min = jnp.where(child_n_values > 0, q_values, min_value)
    normalized = (completed_by_min - min_value) / (
        jnp.maximum(max_value - min_value, epsilon)
    )
    return normalized


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
    parent_q_value: ArrayLike,
    epsilon: ArrayLike,
) -> jax.Array:
    """Map Q-values from the fixed range [-1, 1] to [0, 1], unvisited children to 0.

    AlphaZero's and MuZero's board-game normalization (MuZero's `KnownBounds(-1, 1)`):
    unlike `normalize_q_values`, how far apart the children's values are matters, so
    the search explores more where they're close or the position is lost.

    Args:
        q_values: Q-values to normalize, in [-1, 1]
        child_n_values: visit counts of child nodes
        parent_q_value: unused, for the `q_transform` signature
        epsilon: unused, for the `q_transform` signature

    Returns:
        jax.Array: normalized Q-values
    """
    del parent_q_value, epsilon
    return jnp.where(child_n_values > 0, (q_values + 1) / 2, 0.0)


class MCTSActionSelector:
    """Base class for action selection in MCTS.

    Is callable, selects an action given a search tree state.
    """

    def __init__(self, epsilon: float = 1e-8):
        """Initializes an MCTSActionSelector.

        Args:
            epsilon: small value to avoid division by zero
        """
        self.epsilon = epsilon

    def __call__(
        self, key: jax.Array, tree: MCTSTree, index: ArrayLike, discount: float
    ) -> jax.Array:
        """Selects an action given a search tree state. Implemented by subclasses."""
        raise NotImplementedError()

    def get_config(self) -> dict:
        """Returns the configuration of the action selector. Used for logging."""
        return {"epsilon": self.epsilon}


class PUCTSelector(MCTSActionSelector):
    """PUCT (Polynomial Upper Confidence Trees) action selector, as in AlphaZero.

    Picks the action maximizing `Q + c * P * sqrt(N) / (1 + n)`, with
    `c = c_init + log((N + c_base + 1) / c_base)`: AlphaZero's (and MuZero's) rule. By default Q is
    on the fixed range [-1, 1] (`bounded_q_values`), as in AlphaZero and MuZero's board games; use
    `normalize_q_values` (mctx's default, MuZero's Atari setting) for values without known bounds.
    """

    def __init__(
        self,
        c_init: float = 1.25,
        c_base: float = 19652,
        epsilon: float = 1e-8,
        q_transform=bounded_q_values,
    ):
        """Initializes a PUCTSelector.

        Args:
            c_init: exploration constant (larger values encourage exploration)
            c_base: visit count over which the exploration constant grows (by log(2) at N = c_base)
            epsilon: small value to avoid division by zero
            q_transform: function mapping q-values to [0, 1] before selection
        """
        super().__init__(epsilon=epsilon)
        self.c_init = c_init
        self.c_base = c_base
        self.q_transform = q_transform

    def get_config(self) -> dict:
        """Returns the configuration of the PUCT action selector. Used for logging."""
        return {
            "c_init": self.c_init,
            "c_base": self.c_base,
            "q_transform": self.q_transform.__name__,
            **super().get_config(),
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
        # normalize/transform q-values
        q_values = self.q_transform(discounted_q_values, n_values, node.q, self.epsilon)
        # calculate U-values
        c = self.c_init + jnp.log((node.n + self.c_base + 1) / self.c_base)
        u_values = c * node.p * jnp.sqrt(node.n) / (n_values + 1)
        # PUCT = Q-value + U-value
        puct_values = q_values + u_values
        # select action with highest PUCT value, ties broken at random
        return random_argmax(key, puct_values)
