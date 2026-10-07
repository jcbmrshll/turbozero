
from dataclasses import dataclass
from typing import Any

import graphviz
import jax
import jax.numpy as jnp

from core.evaluators.evaluator import EvalOutput
from core.trees.tree import Tree


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MCTSNode:
    """Base MCTS node data strucutre.
    - `n`: visit count
    - `p`: policy vector
    - `q`: cumulative value estimate / visit count
    - `terminated`: whether the environment state is terminal
    - `embedding`: environment state
    """
    n: jnp.number
    p: jax.Array
    q: jnp.number
    terminated: jnp.number
    embedding: Any

    @property
    def w(self) -> jnp.number:
        """cumulative value estimate"""
        return self.q * self.n


# an MCTSTree is a Tree containing MCTSNodes
MCTSTree = Tree[MCTSNode] 


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TraversalState:
    """State used during traversal step of MCTS.
    - `parent`: parent node index
    - `action`: action taken from parent
    """
    parent: int
    action: int


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BackpropState:
    """State used during backpropagation step of MCTS.
    - `node_idx`: current node
    - `value`: value to backpropagate
    - `tree`: search tree
    """
    node_idx: int
    value: float
    tree: MCTSTree


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MCTSOutput(EvalOutput):
    """Output of an MCTS evaluation. See EvalOutput.
    - `eval_state`: The updated internal state of the Evaluator.
    - `policy_weights`: The policy weights assigned to each action.
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
        node = jax.tree_util.tree_map(lambda x: x[batch_id, n_i], tree.data)
        if node.n.item() > 0:
            graph.node(str(n_i), str({
                "i": str(n_i),
                "n": str(node.n.item()),
                "q": f"{node.q.item():.2f}",
                "t": str(node.terminated.item())
            }))

            child_visits = get_child_visits_no_batch(tree, n_i)
            mapping = tree.edge_map[batch_id, n_i]
            for a_i in range(tree.edge_map.shape[2]):
                v_a = child_visits[a_i].item()
                if v_a > 0:
                    graph.edge(str(n_i), str(mapping[a_i]), f'{a_i}:{node.p[a_i]:.4f}')
        else:
            break
    
    return graph
