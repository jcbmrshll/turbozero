from dataclasses import replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import (
    Leaf,
    MCTSNode,
    MCTSOutput,
    MCTSTree,
    SearchRecord,
    TraversalState,
    backprop_stats,
    search_record,
    with_backprop_stats,
)
from core.trees.tree import init_tree
from core.types import EnvStepFn, EvalFn, StepMetadata


class MCTS(Evaluator):
    """Batched implementation of Monte Carlo Tree Search (MCTS).

    Not stateful. This class operates on 'MCTSTree' state objects.

    Compatible with `jax.vmap`, `jax.pmap`, `jax.jit`, etc.
    """

    def __init__(
        self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        branching_factor: int,
        max_nodes: int,
        num_iterations: int,
        discount: float = -1.0,
        temperature: float = 1.0,
        persist_tree: bool = True,
    ):
        """Initializes an MCTS evaluator.

        Args:
            eval_fn: leaf node evaluation function (env_state -> (policy_logits, value))
            action_selector: action selection function (eval_state -> action)
            branching_factor: max number of actions (== children per node)
            max_nodes: allocated size of MCTS tree. A search that needs a new node when the tree is
                full makes room by evicting a leaf (see `eviction_candidate`), so it searches all its
                iterations whatever the size; but evictions forget part of the tree, and with
                `persist_tree` the tree a search starts from can fill most of it, so allow for
                `num_iterations` on top of the subtree a search typically keeps
            num_iterations: number of MCTS iterations to perform per evaluate call
            discount: discount factor for MCTS (default: -1.0)
                - use a negative discount in two-player games (e.g. -1.0)
                - use a positive discount in single-player games (e.g. 1.0)
            temperature: temperature for root action selection (default: 1.0); at 0, the most visited
                action, ties broken at random
            persist_tree: whether to persist search tree state between calls to `evaluate` (default: True)
        """
        super().__init__(discount=discount)
        self.eval_fn = eval_fn
        self.num_iterations = num_iterations
        self.branching_factor = branching_factor
        self.max_nodes = max_nodes
        self.action_selector = action_selector
        self.temperature = temperature
        self.persist_tree = persist_tree

    def get_config(self) -> dict:
        """Returns a config object for checkpoints."""
        return {
            "eval_fn": self.eval_fn.__name__,
            "num_iterations": self.num_iterations,
            "branching_factor": self.branching_factor,
            "max_nodes": self.max_nodes,
            "action_selection_config": self.action_selector.get_config(),
            "discount": self.discount,
            "temperature": self.temperature,
            "persist_tree": self.persist_tree,
        }

    def evaluate(
        self,  # pylint: disable=arguments-differ
        key: jax.Array,
        eval_state: MCTSTree,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
        **kwargs,
    ) -> MCTSOutput:
        """Performs `self.num_iterations` MCTS iterations on an `MCTSTree`.

        Samples an action to take from the root node after search is completed.

        Args:
            key: rng
            eval_state: `MCTSTree` to evaluate, could be empty or partially complete
            env_state: current environment state
            root_metadata: metadata for the root node of the tree
            params: parameters to pass to the the leaf evaluation function
            env_step_fn: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
            MCTSOutput: contains new tree state, selected action, root value, and policy weights
        """
        # store current state metadata in the root node
        root_key, iterate_key, sample_key = jax.random.split(key, 3)
        eval_state = self.update_root(
            root_key, eval_state, env_state, params, root_metadata=root_metadata
        )
        # perform 'num_iterations' iterations of MCTS
        iterate = partial(self.iterate, params=params, env_step_fn=env_step_fn)

        iterate_keys = jax.random.split(iterate_key, self.num_iterations)
        eval_state, _ = jax.lax.scan(
            lambda state, k: (iterate(k, state), None), eval_state, iterate_keys
        )
        # sample action based on root visit counts
        # (also get normalized policy weights for training purposes)
        action, policy_weights = self.sample_root_action(sample_key, eval_state)
        return MCTSOutput(
            eval_state=eval_state, action=action, policy_weights=policy_weights
        )

    def search_cycles(self) -> int:
        """The root's evaluation, then `num_iterations` iterations."""
        return self.num_iterations + 1

    def search_done(self, state: MCTSTree, cycles: jax.Array) -> jax.Array:
        """Done after `search_cycles`."""
        return cycles >= self.search_cycles()

    def search_cycle(
        self,
        key: jax.Array,
        state: MCTSTree,
        cycle: jax.Array,
        active: jax.Array,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
    ) -> MCTSTree:
        """The root's evaluation (cycle 0), or an iteration: what `evaluate` does, a step at a
        time, with the same keys. Either way, one network evaluation, so that a batch of searches
        at different points evaluates the network once a cycle.

        Args: see `Evaluator.search_cycle`.

        Returns:
            MCTSTree: updated MCTSTree
        """
        root_key, iterate_key, _ = jax.random.split(key, 3)
        # the iteration's key, as `evaluate` splits it off (the split's i-th key doesn't depend on
        # how many it makes)
        iteration = jnp.clip(cycle - 1, 0, self.num_iterations - 1)
        iteration_key = jax.random.split(iterate_key, self.num_iterations)[iteration]
        traverse_key, leaf_eval_key, expand_key = jax.random.split(iteration_key, 3)
        root_eval_key, _ = jax.random.split(root_key)
        at_root = cycle == 0
        # the network evaluates the root or the leaf; then the tree takes one of the two updates.
        # Both are computed, as under vmap a lax.cond would, but they write only what they change,
        # where a cond would select between two whole trees
        leaf = self.find_leaf(traverse_key, state, env_step_fn)
        policy_logits, value = self.eval_fn(
            jax.tree.map(
                lambda r, l: jnp.where(at_root, r, l), env_state, leaf.embedding
            ),
            params,
            jnp.where(at_root, root_eval_key, leaf_eval_key),
        )
        state = self.expand_root(
            root_key,
            state,
            env_state,
            root_metadata,
            policy_logits,
            value,
            write=active & at_root,
        )
        return self.expand(
            expand_key, state, leaf, policy_logits, value, write=active & ~at_root
        )

    def search_output(
        self,
        key: jax.Array,
        state: MCTSTree,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
    ) -> MCTSOutput:
        """Samples the action to take from the root, as `evaluate` does after its iterations.

        Args: see `Evaluator.search_output`.

        Returns:
            MCTSOutput: as `evaluate`'s
        """
        _, _, sample_key = jax.random.split(key, 3)
        action, policy_weights = self.sample_root_action(sample_key, state)
        return MCTSOutput(
            eval_state=state, action=action, policy_weights=policy_weights
        )

    def get_value(self, state: MCTSTree) -> jax.Array:
        """Returns value estimate of the environment state stored in the root node of the tree.

        Args:
            state: MCTSTree to evaluate

        Returns:
            jax.Array: value estimate of the environment state stored in the root node of the tree
        """
        return state.data_at(state.ROOT_INDEX).q

    def search_record(self, state: MCTSTree) -> SearchRecord:
        """The record of what the search in `state` explored, see `SearchRecord`."""
        return search_record(state)

    def update_root(
        self,
        key: jax.Array,
        tree: MCTSTree,
        root_embedding: Any,
        params: Any,
        root_metadata: StepMetadata,
        **kwargs,
    ) -> MCTSTree:  # pylint: disable=unused-argument
        """Populates the root node of an MCTSTree.

        Args:
            key: rng
            tree: MCTSTree to update
            root_embedding: root environment state
            params: nn parameters
            root_metadata: metadata of the root environment state

        Returns:
            MCTSTree: updated MCTSTree
        """
        eval_key, _ = jax.random.split(key)
        policy_logits, value = self.eval_fn(root_embedding, params, eval_key)
        return self.expand_root(
            key, tree, root_embedding, root_metadata, policy_logits, value
        )

    def expand_root(
        self,
        key: jax.Array,
        tree: MCTSTree,
        root_embedding: Any,
        root_metadata: StepMetadata,
        policy_logits: jax.Array,
        value: ArrayLike,
        write: ArrayLike = True,
    ) -> MCTSTree:
        """Populates the root node of an MCTSTree with the network's evaluation of it.

        Args:
            key: rng, `update_root`'s
            tree: MCTSTree to update
            root_embedding: root environment state
            root_metadata: metadata of the root environment state
            policy_logits: the root's policy logits
            value: the root's value
            write: (optional) whether to; if not, the tree is left as it is (see `Tree.set_root`)

        Returns:
            MCTSTree: updated MCTSTree
        """
        _, prior_key = jax.random.split(key)
        policy_logits = jnp.where(
            root_metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min
        )
        policy = self.root_prior(
            prior_key, jax.nn.softmax(policy_logits), root_metadata.action_mask
        )
        root_node = self.update_root_node(
            tree.data_at(tree.ROOT_INDEX), policy, value, root_embedding
        )
        return tree.set_root(root_node, write=write)

    def root_prior(
        self, key: jax.Array, policy: jax.Array, action_mask: jax.Array
    ) -> jax.Array:  # pylint: disable=unused-argument
        """The root's prior, from the network's policy over the legal actions: the policy itself.

        Args:
            key: rng
            policy: the network's policy, 0 for illegal actions
            action_mask: the legal actions

        Returns:
            jax.Array: the prior
        """
        return policy

    def iterate(
        self, key: jax.Array, tree: MCTSTree, params: Any, env_step_fn: EnvStepFn
    ) -> MCTSTree:
        """Performs one iteration of MCTS.

        1. Traverse to leaf node.
        2. Evaluate Leaf Node
        3. Expand Leaf Node (add to tree)
        4. Backpropagate

        Args:
            key: rng
            tree: MCTSTree to evaluate
            params: parameters to pass to the the leaf evaluation function
            env_step_fn: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
            MCTSTree: updated MCTSTree
        """
        traverse_key, eval_key, key = jax.random.split(key, 3)
        leaf = self.find_leaf(traverse_key, tree, env_step_fn)
        policy_logits, value = self.eval_fn(leaf.embedding, params, eval_key)
        return self.expand(key, tree, leaf, policy_logits, value)

    def find_leaf(self, key: jax.Array, tree: MCTSTree, env_step_fn: EnvStepFn) -> Leaf:
        """Traverses from the root to the leaf an iteration evaluates, and steps to its state.

        Args:
            key: rng, for the traversal
            tree: MCTSTree to search
            env_step_fn: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
            Leaf: the leaf
        """
        traversal_state = self.traverse(key, tree)
        parent, action = traversal_state.parent, traversal_state.action
        embedding, metadata = env_step_fn(tree.data_at(parent).embedding, action)
        return Leaf(
            parent=parent, action=action, embedding=embedding, metadata=metadata
        )

    def expand(
        self,
        key: jax.Array,
        tree: MCTSTree,
        leaf: Leaf,
        policy_logits: jax.Array,
        value: ArrayLike,
        write: ArrayLike = True,
    ) -> MCTSTree:
        """Adds a leaf with the network's evaluation of it to the tree, and backpropagates its value.

        Args:
            key: rng, for backpropagation
            tree: MCTSTree to update
            leaf: the leaf (see `find_leaf`)
            policy_logits: the leaf's policy logits
            value: the leaf's value
            write: (optional) whether to; if not, the tree is left as it is (see `Tree.set_child`)

        Returns:
            MCTSTree: updated MCTSTree
        """
        parent, action = leaf.parent, leaf.action
        new_embedding, metadata = leaf.embedding, leaf.metadata
        player_reward = metadata.rewards[metadata.cur_player_id]
        policy_logits = jnp.where(
            metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min
        )
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        # add leaf node to tree, or update it if it's already there (e.g. a terminal node)
        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]
        # both are computed and one picked node by node: under vmap a lax.cond on `node_exists`
        # would run both anyway, as a select between whole trees (see `Tree.set_child`)
        visited = self.visit_node(
            node=tree.data_at(node_idx),
            value=value,
            p=policy,
            terminated=metadata.terminated,
            embedding=new_embedding,
        )
        new = self.new_node(
            policy=policy,
            value=value,
            embedding=new_embedding,
            terminated=metadata.terminated,
        )
        node_data = jax.tree.map(
            lambda v, n: jnp.where(node_exists, v, n), visited, new
        )
        # when the tree is full, a new leaf takes the slot of the least-visited leaf
        tree = tree.set_child(
            parent,
            action,
            node_data,
            evict=self.eviction_candidate(tree, parent),
            write=write,
        )
        # backpropagate, only if the leaf is in the tree: otherwise (a full tree with nothing to
        # evict) its value would reach the root but no child of it, and PUCT would choose the same
        # leaf again and again. Every later iteration then does nothing, and the search ends early
        landed = write & tree.is_edge(parent, action)
        return self.backpropagate(
            key, tree, jnp.where(landed, parent, tree.NULL_INDEX), value
        )

    def eviction_candidate(self, tree: MCTSTree, parent: ArrayLike) -> jax.Array:
        """The leaf to evict if the tree is full and an iteration needs a new node under `parent`:
        the least-visited leaf, other than the root and `parent`, and not a child of the root unless
        every leaf is (the root's children hold the policy target); of those, the one whose parent
        has the fewest visits, which the search is least likely to come back to. The evicted leaf's
        visits stay counted in its ancestors, as if it had never been expanded; PUCT can expand it
        again.

        Args:
            tree: MCTSTree, full
            parent: index of the node the new leaf goes under

        Returns:
            jax.Array: the leaf's index, or NULL_INDEX if there is none
        """
        index = jnp.arange(tree.capacity)
        leaves = tree.leaves() & (index != tree.ROOT_INDEX) & (index != parent)
        deep = leaves & (tree.parents != tree.ROOT_INDEX)
        candidates = jnp.where(deep.any(), deep, leaves)
        most = jnp.iinfo(tree.data.n.dtype).max
        n = jnp.where(candidates, tree.data.n, most)
        # (only the root's parent is NULL_INDEX, and the root isn't a candidate)
        parent_n = tree.data.n[jnp.maximum(tree.parents, 0)]
        # of the least visited, the first with the least visited parent
        leaf = jnp.argmin(jnp.where(n == n.min(), parent_n, most))
        return jnp.where(candidates.any(), leaf, tree.NULL_INDEX)

    def traverse(self, key: jax.Array, tree: MCTSTree) -> TraversalState:
        """Traverse from the root node until an unvisited leaf node is reached.

        Args:
            key: rng, for the action selector
            tree: MCTSTree to evaluate

        Returns:
            TraversalState: state of the traversal
                - parent: index of the parent node
                - action: action to take from the parent node
        """

        # continue while:
        # - there is an existing edge corresponding to the chosen action
        # - AND the child node connected to that edge is not terminal
        def cond_fn(state: TraversalState) -> jax.Array:
            return jnp.logical_and(
                tree.is_edge(state.parent, state.action),
                ~(tree.data_at(tree.edge_map[state.parent, state.action]).terminated),
                # TODO: maximum depth
            )

        # a traversal visits each node at most once, so each selection gets its own key
        def select(node_idx: ArrayLike) -> jax.Array:
            node_key = jax.random.fold_in(key, node_idx)
            return self.action_selector(node_key, tree, node_idx, self.discount)

        # each iterration:
        # - get the index of the child node connected to the chosen action
        # - choose the action to take from the child node
        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
            action = select(node_idx)
            return TraversalState(parent=node_idx, action=action)

        # choose the action to take from the root
        root_action = select(tree.ROOT_INDEX)
        # traverse from root to leaf
        return jax.lax.while_loop(
            cond_fn, body_fn, TraversalState(parent=tree.ROOT_INDEX, action=root_action)
        )

    def backpropagate(
        self, key: jax.Array, tree: MCTSTree, parent: ArrayLike, value: ArrayLike
    ) -> MCTSTree:  # pylint: disable=unused-argument
        """Backpropagate the value estimate from the leaf node to the root node and update visit counts.

        Args:
            key: rng
            tree: MCTSTree to evaluate
            parent: index of the parent node (in most cases, this is the new node added to the tree this iteration),
                or NULL_INDEX to back up nothing
            value: value estimate of the leaf node

        Returns:
            MCTSTree: updated search tree
        """

        # every node from `parent` up to the root is visited once, all at the same time (rather than
        # in a loop up the path, see `Tree.path_to_root`)
        path = tree.path_to_root(parent)
        # the value is discounted once more at each level up
        if abs(self.discount) == 1:
            # flip signs rather than multiply: exactly the values a loop up the path computes, and
            # without a product XLA could fuse into the update below (an FMA rounds differently)
            levels = jnp.arange(1, tree.capacity + 1)
            values = jnp.where(
                (self.discount == -1) & (levels % 2 == 1), jnp.negative(value), value
            )
        else:
            values = value * jnp.cumprod(jnp.full(path.shape, self.discount))
        # entries past the root point out of bounds, so their writes are dropped
        index = jnp.where(path == tree.NULL_INDEX, tree.capacity, path)
        stats = backprop_stats(tree)
        nodes = jax.tree.map(lambda x: x[index], stats)
        # increment visit counts and update value estimates
        visited = self.visit_node(nodes, values)
        stats = jax.tree.map(
            lambda x, y: x.at[index].set(y, mode="drop"), stats, visited
        )
        return with_backprop_stats(tree, stats)

    def sample_root_action(
        self, key: jax.Array, tree: MCTSTree
    ) -> tuple[jax.Array, jax.Array]:
        """Sample an action based on the root visit counts.

        Args:
            key: rng
            tree: MCTSTree to evaluate

        Returns:
            Tuple[jax.Array, jax.Array]: sampled action, normalized policy weights
        """
        # get root visit counts
        action_visits = tree.get_child_data("n", tree.ROOT_INDEX)
        # normalize visit counts to get policy weights
        total_visits = action_visits.sum(axis=-1)
        policy_weights = action_visits / jnp.maximum(total_visits, 1)
        policy_weights = jnp.where(
            total_visits > 0, policy_weights, 1 / self.branching_factor
        )

        # zero temperature == argmax
        if self.temperature == 0:
            # of the most visited actions, one at random: equal visits give exactly equal weights
            noise = jax.random.uniform(key, shape=policy_weights.shape)
            action = jnp.argmax(
                jnp.where(policy_weights == policy_weights.max(), noise, -1.0)
            )
            return action, policy_weights

        # apply temperature
        policy_weights_t = policy_weights ** (1 / self.temperature)
        # re-normalize
        policy_weights_t /= policy_weights_t.sum()
        # sample action
        action = jax.random.choice(key, policy_weights_t.shape[-1], p=policy_weights_t)
        # return original policy weights (we train on the policy before temperature is applied)
        return action, policy_weights

    @staticmethod
    def visit_node(
        node: MCTSNode,
        value: ArrayLike,
        p: jax.Array | None = None,
        terminated: ArrayLike | None = None,
        embedding: Any | None = None,
    ) -> MCTSNode:
        """Update the visit counts and value estimate of a node.

        Args:
            node: MCTSNode to update
            value: value estimate to update the node with
            p: (optional) policy weights to overwrite the node's policy with
            terminated: (optional) whether the node is terminal
            embedding: (optional) embedding to overwrite the node's embedding with

        Returns:
            MCTSNode: updated MCTSNode
        """
        # update running value estimate
        q_value = ((node.q * node.n) + value) / (node.n + 1)
        # update other node attributes
        if p is None:
            p = node.p
        if terminated is None:
            terminated = node.terminated
        if embedding is None:
            embedding = node.embedding
        return replace(
            node,
            n=node.n + 1,  # increment visit count
            q=q_value,
            p=p,
            terminated=terminated,
            embedding=embedding,
        )

    @staticmethod
    def new_node(
        policy: jax.Array, value: ArrayLike, embedding: Any, terminated: ArrayLike
    ) -> MCTSNode:
        """Create a new MCTSNode.

        Args:
            policy: policy weights
            value: value estimate
            embedding: environment state embedding
                - 'embedding' because in some MCTS use-cases, e.g. MuZero, we store an embedding of the state
                  rather than the state itself. In AlphaZero, this is just the entire environment state.
            terminated: whether the state is terminal

        Returns:
            MCTSNode: initialized MCTSNode
        """
        return MCTSNode(
            n=jnp.array(1, dtype=jnp.int32),  # init visit count to 1
            p=policy,
            q=jnp.array(value, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool_),
            embedding=embedding,
        )

    @staticmethod
    def update_root_node(
        root_node: MCTSNode,
        root_policy: jax.Array,
        root_value: ArrayLike,
        root_embedding: Any,
    ) -> MCTSNode:
        """Update the root node of the search tree.

        Args:
            root_node: node to update
            root_policy: policy weights
            root_value: value estimate
            root_embedding: environment state embedding

        Returns:
            MCTSNode: updated root node
        """
        visited = root_node.n > 0
        return replace(
            root_node,
            p=root_policy,
            # keep old value estimate if the node has already been visited
            q=jnp.where(visited, root_node.q, root_value),
            # keep old visit count if the node has already been visited
            n=jnp.where(visited, root_node.n, 1),
            embedding=root_embedding,
        )

    def reset(self, state: MCTSTree) -> MCTSTree:
        """Resets the internal state of MCTS.

        Args:
            state: evaluator state

        Returns:
            MCTSTree: reset evaluator state
        """
        return state.reset()

    def step(self, state: MCTSTree, action: jax.Array) -> MCTSTree:
        """Update the internal state of MCTS after taking an action in the environment.

        Args:
            state: evaluator state
            action: action taken in the environment

        Returns:
            MCTSTree: updated evaluator state
        """

        if self.persist_tree:
            # get subtree corresponding to action taken if persist_tree is True
            return state.get_subtree(action)
        # just reset to an empty tree if persist_tree is False
        return state.reset()

    def init(self, template_embedding: Any, *args, **kwargs) -> MCTSTree:  # pylint: disable=arguments-differ
        """Initializes the internal state of the MCTS evaluator.

        Args:
            template_embedding: template environment state embedding
                - not stored, just used to initialize data structures to the correct shape

        Returns:
            MCTSTree: initialized MCTSTree
        """
        return init_tree(
            self.max_nodes,
            self.branching_factor,
            self.new_node(
                policy=jnp.zeros((self.branching_factor,)),
                value=0.0,
                embedding=template_embedding,
                terminated=False,
            ),
        )
