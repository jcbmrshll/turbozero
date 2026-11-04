import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from core.evaluators.mcts.state import MCTSTree


def mixed_value(
    raw_value: ArrayLike,
    q_values: jax.Array,
    child_n_values: jax.Array,
    prior: jax.Array,
) -> jax.Array:
    """The value of a node's unvisited children in its completed Q-values: the network's value of
    the node, averaged with the prior-weighted mean Q-value of its visited children, weighted by
    their visits. mctx's `_compute_mixed_value`.

    Args:
        raw_value: the network's value of the node
        q_values: (F) its children's Q-values, from the node's player's perspective
        child_n_values: (F) their visit counts
        prior: (F) the network's policy at the node

    Returns:
        jax.Array: the mixed value
    """
    visited = child_n_values > 0
    total_visits = child_n_values.sum(axis=-1)
    # actions the network gives no weight still count, if visited
    prior = jnp.maximum(prior, jnp.finfo(prior.dtype).tiny)
    visited_prior = jnp.where(visited, prior, 0.0).sum(axis=-1)
    mean_q = jnp.where(
        visited, prior * q_values / jnp.where(visited, visited_prior, 1.0), 0.0
    ).sum(axis=-1)
    return (jnp.asarray(raw_value) + total_visits * mean_q) / (total_visits + 1)


def completed_q_values(
    raw_value: ArrayLike,
    q_values: jax.Array,
    child_n_values: jax.Array,
    prior: jax.Array,
    epsilon: float,
) -> jax.Array:
    """A node's children's Q-values, the unvisited ones' taken as the node's `mixed_value`, rescaled
    to [0, 1] by their minimum and maximum. mctx's `qtransform_completed_by_mix_value` before its
    scaling by visits.

    Args: see `mixed_value`
        epsilon: the smallest range to rescale by

    Returns:
        jax.Array: (F) the completed Q-values, in [0, 1]
    """
    value = mixed_value(raw_value, q_values, child_n_values, prior)
    completed = jnp.where(child_n_values > 0, q_values, value)
    low = completed.min(axis=-1, keepdims=True)
    high = completed.max(axis=-1, keepdims=True)
    return (completed - low) / jnp.maximum(high - low, epsilon)


class PolicyTarget:
    """What the policy trains towards, from the root of a finished search."""

    def __call__(self, tree: MCTSTree, discount: float) -> jax.Array:
        """The target, (F) a distribution over the actions. Implemented by subclasses."""
        raise NotImplementedError()

    def get_config(self) -> dict:
        """Returns the configuration of the policy target. Used for logging."""
        return {"policy_target": type(self).__name__}


def visit_distribution(n: jax.Array) -> jax.Array:
    """(F) visits normalized to a distribution; uniform if there are none."""
    total = n.sum(axis=-1, keepdims=True)
    return jnp.where(total > 0, n / jnp.maximum(total, 1), 1 / n.shape[-1])


class VisitCounts(PolicyTarget):
    """The root's visit distribution, as in AlphaZero; uniform if its children have none.

    With `snapshot`, the distribution of the visits after the search's first
    `MCTS.snapshot_iterations` iterations (`MCTSTree.root_snapshot_n`) instead: a shallower
    search's target from the same search, whose full visits can still choose the move played.
    """

    def __init__(self, snapshot: bool = False):
        self.snapshot = snapshot

    def get_config(self) -> dict:
        return {"snapshot": self.snapshot, **super().get_config()}

    def __call__(self, tree: MCTSTree, discount: float) -> jax.Array:
        if self.snapshot:
            return visit_distribution(tree.root_snapshot_n)
        return visit_distribution(tree.get_child_data("n", tree.ROOT_INDEX))


class CompletedQ(PolicyTarget):
    """Gumbel MuZero's improved policy (Danihelka et al. 2022, https://openreview.net/forum?id=bERaNdoegnO):
    `softmax(logits + sigma(completed Q))`, with the network's logits at the root, before any noise,
    and `sigma(q) = (maxvisit_init + max_a N(a)) * value_scale * q`, the completed Q-values
    (`completed_q_values`) scaled by the most visited child's visits. The defaults are mctx's; the
    paper's Go and chess agents have `value_scale = 1`.

    Root noise then only steers which actions the search explores, not the target, and the target
    is defined at any budget: with no visits it is the network's policy.
    """

    def __init__(
        self, value_scale: float = 0.1, maxvisit_init: float = 50.0, epsilon=1e-8
    ):
        """Initializes a CompletedQ policy target.

        Args:
            value_scale: scale of the completed Q-values against the logits, per visit
            maxvisit_init: visits added to the most visited child's
            epsilon: the smallest range to rescale Q-values by
        """
        self.value_scale = value_scale
        self.maxvisit_init = maxvisit_init
        self.epsilon = epsilon

    def get_config(self) -> dict:
        return {
            "value_scale": self.value_scale,
            "maxvisit_init": self.maxvisit_init,
            "epsilon": self.epsilon,
            **super().get_config(),
        }

    def __call__(self, tree: MCTSTree, discount: float) -> jax.Array:
        # the children's values are for the player to move at each of them
        q = tree.get_child_data("q", tree.ROOT_INDEX) * discount
        n = tree.get_child_data("n", tree.ROOT_INDEX)
        logits = tree.root_logits
        completed = completed_q_values(
            tree.root_value, q, n, jax.nn.softmax(logits), self.epsilon
        )
        scale = (self.maxvisit_init + n.max(axis=-1)) * self.value_scale
        return jax.nn.softmax(logits + scale * completed)
