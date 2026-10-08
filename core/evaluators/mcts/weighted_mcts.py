from dataclasses import dataclass, replace
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from core.evaluators.mcts.action_selection import normalize_q_values
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.state import (
    BackpropState,
    MCTSNode,
    MCTSTree,
    backprop_stats,
    with_backprop_stats,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WeightedMCTSNode(MCTSNode):
    # Weighted MCTS needs access to the original raw value returned by the leaf evaluation
    r: jax.Array  # raw value


class WeightedMCTS(MCTS):
    """Weighted MCTS implementation.

    https://twitter.com/ptrschmdtnlsn/status/1748800529608888362
    """

    def __init__(self, q_temperature: float = 1.0, *args, **kwargs):
        """Initializes a WeightedMCTS evaluator.

        Args:
            q_temperature: temperature to apply to child q-values when backpropagating
        """
        super().__init__(*args, **kwargs)
        self.q_temperature = q_temperature

    def get_config(self) -> dict:
        """Returns the configuration of the WeightedMCTS evaluator. Used for logging."""
        return {"q_temperature": self.q_temperature, **super().get_config()}

    @staticmethod
    def new_node(
        policy: jax.Array, value: ArrayLike, embedding: Any, terminated: ArrayLike
    ) -> WeightedMCTSNode:
        """Create a new WeightedMCTSNode.

        Args:
            policy: policy vector
            value: value estimate
            embedding: environment state
            terminated: whether the environment state is terminal

        Returns:
            WeightedMCTSNode: new node
        """
        return WeightedMCTSNode(
            n=jnp.array(1, dtype=jnp.int32),
            p=policy,
            q=jnp.array(value, dtype=jnp.float32),
            r=jnp.array(value, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool_),
            embedding=embedding,
        )

    @staticmethod
    def update_root_node(
        root_node: MCTSNode,
        root_policy: jax.Array,
        root_value: ArrayLike,
        root_embedding: Any,
    ) -> WeightedMCTSNode:
        """Updates the root node.

        - if the tree is empty, create a new node
        - otherwise, update the existing root node

        Args:
            root_node: root node
            root_policy: root policy
            root_value: root value
            root_embedding: root environment state

        Returns:
            WeightedMCTSNode: updated root node
        """
        # WeightedMCTS trees hold WeightedMCTSNodes (see `new_node`), which the MCTSTree annotation can't express
        root_node = cast(WeightedMCTSNode, root_node)
        visited = root_node.n > 0
        return replace(
            root_node,
            p=root_policy,
            q=jnp.where(visited, root_node.q, root_value),
            r=jnp.where(visited, root_node.r, root_value),
            n=jnp.where(visited, root_node.n, 1),
            embedding=root_embedding,
        )

    def backpropagate(
        self, key: jax.Array, tree: MCTSTree, parent: ArrayLike, value: ArrayLike
    ) -> MCTSTree:
        """Backpropagate weighted sums of child q-values and update visit counts.

        Args:
            key: rng
            tree: The search tree.
            parent: index of the parent node (in most cases, this is the new node added to the tree this iteration)
            value: expanded node value estimate

        Returns:
            MCTSTree: updated search tree
        """

        def body_fn(state: BackpropState) -> BackpropState:
            node_idx = state.node_idx
            # the loop carries only the node statistics (see `backprop_stats`); read them through
            # the tree outside it, whose structure doesn't change
            view = replace(tree, data=state.stats)
            # get node data (WeightedMCTS trees hold WeightedMCTSNodes, see `new_node`)
            node = cast(WeightedMCTSNode, view.data_at(node_idx))
            # get q values, visit counts of children
            child_q_values = view.get_child_data("q", node_idx) * self.discount
            child_n_values = view.get_child_data("n", node_idx)

            # normalize q-values to [0, 1]
            # (only used to compute the weights, the backed-up value stays on the network's value scale)
            normalized_q_values = normalize_q_values(
                child_q_values, child_n_values, node.q, jnp.finfo(node.q).eps
            )

            if self.q_temperature > 0:
                # if temperature > 0, apply temperature to the softmax logits
                # mask out unvisited action a[i] so softmax(a)[i] = 0.0
                q_values_masked = jnp.where(
                    child_n_values > 0,
                    normalized_q_values / self.q_temperature,
                    jnp.finfo(normalized_q_values).min,
                )
            else:
                # if temperature == 0, select max q-value amongst visited children
                # apply random noise to break ties amongst nodes w/ same number of visits
                # (fold in the node index so each node on the path gets independent noise)
                node_key = jax.random.fold_in(key, node_idx)
                noise = jax.random.uniform(
                    node_key,
                    shape=normalized_q_values.shape,
                    maxval=self.tiebreak_noise,
                )
                noisy_q_values = jnp.where(
                    child_n_values > 0,
                    normalized_q_values + noise,
                    jnp.finfo(normalized_q_values).min,
                )

                # mask out all values except for max value index
                # so softmax output at a[max_index] = 1 and 0 everywhere else
                max_vector = jnp.full_like(
                    noisy_q_values, jnp.finfo(noisy_q_values).min
                )
                index_of_max = jnp.argmax(noisy_q_values)
                max_vector = max_vector.at[index_of_max].set(1)
                q_values_masked = max_vector

            # compute weights
            child_weights = jax.nn.softmax(q_values_masked, axis=-1)
            # compute weighted sum of (discounted) child q-values
            weighted_value = jnp.sum(child_weights * child_q_values)
            # update node with weighted value
            node = replace(node, q=weighted_value)
            # adjust node value to ((weighted_value * node_visits) + raw_value) / (node_visits + 1)
            # and increment visit count
            node = self.visit_node(node, node.r)
            # update the node's statistics
            stats = view.update_node(node_idx, node).data
            # backprop to parent node
            return BackpropState(
                node_idx=tree.parents[node_idx], value=value, stats=stats
            )

        state = jax.lax.while_loop(
            lambda s: s.node_idx != tree.NULL_INDEX,
            body_fn,
            BackpropState(node_idx=parent, value=value, stats=backprop_stats(tree)),
        )
        return with_backprop_stats(tree, state.stats)
