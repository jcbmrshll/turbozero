from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from core.evaluators.evaluator import EvalOutput
from core.trees.tree import Tree
from core.types import StepMetadata


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

    @property
    def w(self) -> jax.Array:
        """Cumulative value estimate."""
        return self.q * self.n


# an MCTSTree is a Tree containing MCTSNodes
MCTSTree = Tree[MCTSNode]


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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SearchRecord:
    """What a search explored: its tree's structure and statistics, without the environment states
    (most of the tree), for the monitor to show. Node indices are the tree's.

    Attributes:
        parents: (N) each node's parent, NULL_INDEX for the root and unused nodes
        actions: (N) the action from each node's parent to it, NULL_INDEX for the root and unused nodes
        n: (N) visit counts
        q: (N) value estimates, for the player to move at each node
        prior: (N) the policy's weight on the action from each node's parent to it
        terminated: (N) whether each node's state is terminal
        policy: (F) the root's policy
        num_nodes: () nodes in use, 0 to `num_nodes - 1`
    """

    parents: jax.Array
    actions: jax.Array
    n: jax.Array
    q: jax.Array
    prior: jax.Array
    terminated: jax.Array
    policy: jax.Array
    num_nodes: jax.Array


def search_record(tree: MCTSTree) -> SearchRecord:
    """The record of what the search in `tree` explored."""
    capacity, branching = tree.capacity, tree.branching_factor
    # each child's action is the edge that points at it; missing edges point out of bounds
    edges = tree.edge_map.reshape(-1)
    actions = (
        jnp.full((capacity,), tree.NULL_INDEX, dtype=jnp.int32)
        .at[jnp.where(edges == tree.NULL_INDEX, capacity, edges)]
        .set(jnp.tile(jnp.arange(branching, dtype=jnp.int32), capacity), mode="drop")
    )
    has_parent = tree.parents != tree.NULL_INDEX
    prior = jnp.where(
        has_parent,
        tree.data.p[jnp.maximum(tree.parents, 0), jnp.maximum(actions, 0)],
        0.0,
    )
    return SearchRecord(
        parents=tree.parents,
        actions=actions,
        n=tree.data.n,
        q=tree.data.q,
        prior=prior,
        terminated=tree.data.terminated,
        policy=tree.data.p[tree.ROOT_INDEX],
        num_nodes=tree.next_free_idx,
    )
