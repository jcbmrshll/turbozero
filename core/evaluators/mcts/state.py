from dataclasses import dataclass, replace
from operator import itemgetter
from typing import Any

import graphviz
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from core.evaluators.evaluator import EvalOutput
from core.trees.tree import Tree


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


def tree_to_graph(tree, batch_id=0):
    """Converts a search tree to a graphviz graph."""
    graph = graphviz.Digraph()

    def get_child_visits_no_batch(tree, index):
        mapping = tree.edge_map[batch_id, index]
        child_data = tree.data.n[batch_id, mapping]
        return jnp.where(
            (mapping == Tree.NULL_INDEX).reshape((-1,) + (1,) * (child_data.ndim - 1)),
            0,
            child_data,
        )

    for n_i in range(tree.parents.shape[1]):
        node = jax.tree_util.tree_map(itemgetter((batch_id, n_i)), tree.data)
        if node.n.item() > 0:
            graph.node(
                str(n_i),
                str(
                    {
                        "i": str(n_i),
                        "n": str(node.n.item()),
                        "q": f"{node.q.item():.2f}",
                        "t": str(node.terminated.item()),
                    }
                ),
            )

            child_visits = get_child_visits_no_batch(tree, n_i)
            mapping = tree.edge_map[batch_id, n_i]
            for a_i in range(tree.edge_map.shape[2]):
                v_a = child_visits[a_i].item()
                if v_a > 0:
                    graph.edge(str(n_i), str(mapping[a_i]), f"{a_i}:{node.p[a_i]:.4f}")
        else:
            break

    return graph
