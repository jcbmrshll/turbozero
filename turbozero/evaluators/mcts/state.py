from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from turbozero.trees.tree import Tree, init_tree


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MCTSNode:
    """Base MCTS node data strucutre.

    Attributes:
        n: visit count
        p: policy vector
        q: cumulative value estimate / visit count
        terminated: whether the environment state is terminal
        embedding: environment state
    """

    n: jax.Array
    p: jax.Array
    q: jax.Array
    terminated: jax.Array
    embedding: Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MCTSTree(Tree[MCTSNode]):
    """A Tree of MCTSNodes, with the noise its root's prior takes.

    Attributes:
        root_noise: (F) the noise the root's prior takes (e.g. AlphaZero's Dirichlet noise), drawn
            when the search starts (see `MCTS.start_search`)
    """

    root_noise: jax.Array


def init_mcts_tree(
    max_nodes: int, branching_factor: int, template_data: MCTSNode
) -> MCTSTree:
    """An empty MCTSTree, see `init_tree`."""
    tree = init_tree(max_nodes, branching_factor, template_data)
    return MCTSTree(
        next_free_idx=tree.next_free_idx,
        parents=tree.parents,
        edge_map=tree.edge_map,
        data=tree.data,
        root_noise=jnp.zeros((branching_factor,), dtype=jnp.float32),
    )
