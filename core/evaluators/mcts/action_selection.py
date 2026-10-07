

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from core.evaluators.mcts.state import MCTSTree


def normalize_q_values(
    q_values: jax.Array, 
    child_n_values: jax.Array, 
    parent_q_value: ArrayLike,
    epsilon: ArrayLike
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
        jnp.maximum(max_value - min_value, epsilon))
    return normalized


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


    def __call__(self, tree: MCTSTree, index: ArrayLike, discount: float) -> jax.Array:
        """Selects an action given a search tree state. Implemented by subclasses."""
        raise NotImplementedError()


    def get_config(self) -> dict:
        """Returns the configuration of the action selector. Used for logging."""
        return {
            "epsilon": self.epsilon
        }


class PUCTSelector(MCTSActionSelector):
    """PUCT (Polynomial Upper Confidence Trees) action selector.

    This is the algorithm used for action selection within AlphaZero.
    """

    def __init__(self, 
        c: float = 1.0,
        epsilon: float = 1e-8, 
        q_transform = normalize_q_values
    ):
        """Initializes a PUCTSelector.

        Args:
            c: exploration constant (larger values encourage exploration)
            epsilon: small value to avoid division by zero
            q_transform: function applied to q-values before selection
        """
        super().__init__(epsilon=epsilon)
        self.c = c
        self.q_transform = q_transform


    def get_config(self) -> dict:
        """Returns the configuration of the PUCT action selector. Used for logging."""
        return {
            "c": self.c,
            'q_transform': self.q_transform.__name__,
            **super().get_config()
        }


    def __call__(self, tree: MCTSTree, index: ArrayLike, discount: float) -> jax.Array:
        """Selects an action given a search tree state.

        Args:
            tree: search tree
            index: index of the node in the search tree to select an action to take from
            discount: discount factor

        Returns:
            jax.Array: id of action to take
        """
        # get child q-values
        node = tree.data_at(index)
        q_values = tree.get_child_data('q', index)
        # apply discount to q-values
        discounted_q_values = q_values * discount
        # get child visit counts
        n_values = tree.get_child_data('n', index)
        # normalize/transform q-values
        q_values = self.q_transform(discounted_q_values, n_values, node.q, self.epsilon)
        # calculate U-values
        u_values = self.c * node.p * jnp.sqrt(node.n) / (n_values + 1)
        # PUCT = Q-value + U-value
        puct_values = q_values + u_values
        # select action with highest PUCT value
        return puct_values.argmax()
    

class MuZeroPUCTSelector(MCTSActionSelector):
    """Implements the variant of PUCT used in MuZero."""

    def __init__(self, 
        c1: float = 1.25, 
        c2: float = 19652, 
        epsilon: float = 1e-8,
        q_transform = normalize_q_values
    ):
        """Initializes a MuZeroPUCTSelector.

        Args:
            c1: 1st exploration constant
            c2: 2nd exploration constant
            epsilon: small value to avoid division by zero
            q_transform: function applied to q-values before selection
        """
        super().__init__(epsilon=epsilon)
        self.c1 = c1
        self.c2 = c2
        self.q_transform = q_transform
    

    def get_config(self) -> dict:
        """Returns the configuration of the MuZero PUCT action selector. Used for logging."""
        return {
            "c1": self.c1,
            "c2": self.c2,
            "q_transform": self.q_transform.__name__,
            **super().get_config()
        }

    def __call__(self, tree: MCTSTree, index: ArrayLike, discount: float) -> jax.Array:
        """Selects an action given a search tree state.

        Args:
            tree: search tree
            index: index of the node in the search tree to select an action to take from
            discount: discount factor

        Returns:
            jax.Array: id of action to take
        """
        # get child q-values
        node = tree.data_at(index)
        q_values = tree.get_child_data('q', index)
        # apply discount to q-values
        discounted_q_values = q_values * discount
        # get child visit counts
        n_values = tree.get_child_data('n', index)
        # normalize/transform q-values
        q_values = self.q_transform(discounted_q_values, n_values, node.q, self.epsilon)
        # calculate U-values
        base_term = node.p * jnp.sqrt(node.n) / (n_values + 1)
        log_term = jnp.log((node.n + self.c2 + 1) / self.c2) + self.c1
        u_values = base_term * log_term
        # PUCT = Q-value + U-value
        puct_values = q_values + u_values
        # select action with highest PUCT value
        return puct_values.argmax()
