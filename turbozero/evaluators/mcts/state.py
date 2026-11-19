from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from turbozero.trees.tree import Tree, init_tree


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MCTSNode:
    """A search tree node's data."""

    n: jax.Array  # () visit count
    p: jax.Array  # (F,) prior over the node's actions
    q: jax.Array  # () mean value, for the player to move at the node
    terminated: jax.Array  # () whether the environment state is terminal
    embedding: Any  # the environment state


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MCTSTree(Tree[MCTSNode]):
    """A search tree, with the noise its root's prior takes (e.g. AlphaZero's
    Dirichlet noise), drawn once when the search starts.
    """

    root_noise: jax.Array  # (F,)


def init_mcts_tree(
    max_nodes: int, branching_factor: int, template_data: MCTSNode
) -> MCTSTree:
    """An empty MCTSTree, see `init_tree`."""
    return MCTSTree(
        **vars(init_tree(max_nodes, branching_factor, template_data)),
        root_noise=jnp.zeros((branching_factor,), dtype=jnp.float32),
    )
