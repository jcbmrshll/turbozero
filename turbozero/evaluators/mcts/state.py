from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from turbozero.evaluators.evaluator import EvalOutput
from turbozero.trees.tree import Tree, init_tree
from turbozero.types import StepMetadata


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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TraversalState:
    """State used during traversal step of MCTS.

    Attributes:
        parent: parent node index
        action: action taken from parent
    """

    parent: ArrayLike
    action: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Leaf:
    """Where an iteration's traversal ended, and the state there it evaluates.

    Attributes:
        parent: index of the node the leaf is under
        action: action from the parent to the leaf
        embedding: the leaf's environment state
        metadata: its metadata
    """

    parent: ArrayLike
    action: jax.Array
    embedding: Any
    metadata: StepMetadata


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BackpropState:
    """State used during backpropagation step of MCTS.

    Attributes:
        node_idx: current node
        value: value to backpropagate
        stats: the search tree's node data without the fields backpropagation never changes
            (see `backprop_stats`)
    """

    node_idx: ArrayLike
    value: ArrayLike
    stats: MCTSNode


def backprop_stats(tree: MCTSTree) -> MCTSNode:
    """The node data backpropagation reads and updates: everything but the policies and embeddings.

    These statistics are a few numbers per node, while the policies and embeddings (a whole environment
    state per node) are most of the tree, so backpropagation should touch only them. Where it's a
    while_loop up the path to the root (`WeightedMCTS`), this matters most: under vmap, a while_loop whose
    condition differs across the batch selects between the old and the new loop state at every
    iteration, for every tree in the batch. The loop reads everything it doesn't change (structure
    included) from the tree outside it.

    Returns:
        MCTSNode: the tree's node data, with `p` and `embedding` set to None
    """
    return replace(tree.data, p=None, embedding=None)


def with_backprop_stats(tree: MCTSTree, stats: MCTSNode) -> MCTSTree:
    """The tree with its node data replaced by `stats`, its policies and embeddings put back."""
    return replace(
        tree, data=replace(stats, p=tree.data.p, embedding=tree.data.embedding)
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MCTSOutput(EvalOutput):
    """Output of an MCTS evaluation. See EvalOutput.

    Attributes:
        eval_state: The updated internal state of the Evaluator.
        policy_weights: The policy weights assigned to each action.
    """

    eval_state: MCTSTree
    policy_weights: jax.Array
