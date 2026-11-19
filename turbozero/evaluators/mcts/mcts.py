from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from turbozero.evaluators.evaluator import EvalOutput, Evaluator
from turbozero.evaluators.mcts.action_selection import MCTSActionSelector
from turbozero.evaluators.mcts.state import MCTSNode, MCTSTree, init_mcts_tree
from turbozero.types import EnvStepFn, LeafEvalFn, StepMetadata


class MCTS(Evaluator):
    """Batched implementation of Monte Carlo Tree Search (MCTS).

    Not stateful. This class operates on 'MCTSTree' state objects.

    Compatible with `jax.vmap`, `jax.pmap`, `jax.jit`, etc.
    """

    def __init__(
        self,
        leaf_eval_fn: LeafEvalFn,
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
            leaf_eval_fn: leaf node evaluation function (env_state -> (policy_logits, value))
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
        self.leaf_eval_fn = leaf_eval_fn
        self.num_iterations = num_iterations
        self.branching_factor = branching_factor
        self.max_nodes = max_nodes
        self.action_selector = action_selector
        self.temperature = temperature
        self.persist_tree = persist_tree

    def get_config(self) -> dict:
        """Returns a config object for checkpoints."""
        return {
            "leaf_eval_fn": self.leaf_eval_fn.__name__,
            "num_iterations": self.num_iterations,
            "branching_factor": self.branching_factor,
            "max_nodes": self.max_nodes,
            "action_selection_config": self.action_selector.get_config(),
            "discount": self.discount,
            "temperature": self.temperature,
            "persist_tree": self.persist_tree,
        }

    def evaluate(
        self,
        key: jax.Array,
        eval_state: MCTSTree,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
    ) -> EvalOutput:
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
            EvalOutput: contains new tree state, selected action, and policy weights
        """
        tree = self.start_search(key, eval_state)
        root_key, iterate_key, sample_key = jax.random.split(key, 3)
        eval_key, _ = jax.random.split(root_key)
        policy_logits, value = self.leaf_eval_fn(env_state, params, eval_key)
        tree = self.expand_root(
            root_key, tree, env_state, root_metadata, policy_logits, value
        )

        def iterate(tree: MCTSTree, key: jax.Array) -> tuple[MCTSTree, None]:
            traverse_key, eval_key, key = jax.random.split(key, 3)
            leaf = self.find_leaf(traverse_key, tree, env_step_fn)
            policy_logits, value = self.leaf_eval_fn(leaf[2], params, eval_key)
            return self.expand(key, tree, leaf, policy_logits, value), None

        keys = jax.random.split(iterate_key, self.num_iterations)
        tree, _ = jax.lax.scan(iterate, tree, keys)
        action, policy_weights = self.sample_root_action(sample_key, tree)
        return EvalOutput(eval_state=tree, action=action, policy_weights=policy_weights)

    def search_cycles(self) -> int:
        """The root's evaluation, then `num_iterations` iterations."""
        return self.num_iterations + 1

    def start_search(
        self, key: jax.Array, state: MCTSTree, active: ArrayLike = True
    ) -> MCTSTree:
        """Draws the root's noise (see `root_noise`) for the search for a move with key `key`, as
        `evaluate` does first: a search a cycle at a time draws it once, rather than in every
        cycle (under vmap, the root's update is computed in every cycle, see `search_cycle`).

        Args: see `Evaluator.start_search`.

        Returns:
            MCTSTree: the tree, with its `root_noise`
        """
        root_key, _, _ = jax.random.split(key, 3)
        _, noise_key = jax.random.split(root_key)
        noise = self.root_noise(noise_key, state.root_noise.shape)
        return replace(state, root_noise=jnp.where(active, noise, state.root_noise))

    def root_noise(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:  # pylint: disable=unused-argument
        """The noise the root's prior takes (see `root_prior`): none.

        Args:
            key: rng
            shape: (F,)

        Returns:
            jax.Array: the noise, float32
        """
        return jnp.zeros(shape, dtype=jnp.float32)

    def search_cycle(
        self,
        key: jax.Array,
        state: MCTSTree,
        cycle: jax.Array,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
    ) -> MCTSTree:
        """The root's evaluation (cycle 0), or an iteration: what `evaluate` does after
        `start_search`, a step at a time, with the same keys. Either way, one network evaluation, so that a batch of searches
        at different points evaluates the network once a cycle. Past `num_iterations`, more
        iterations, each with a key of its own.

        Args: see `Evaluator.search_cycle`.

        Returns:
            MCTSTree: updated MCTSTree
        """
        root_key, iterate_key, _ = jax.random.split(key, 3)
        # the iteration's key, as `evaluate` splits it off (the split's i-th key doesn't depend on
        # how many it makes); past `num_iterations`, folded in
        iteration = jnp.maximum(cycle - 1, 0)
        iteration_key = jnp.where(
            iteration < self.num_iterations,
            jax.random.split(iterate_key, self.num_iterations)[
                jnp.minimum(iteration, self.num_iterations - 1)
            ],
            jax.random.fold_in(iterate_key, iteration),
        )
        traverse_key, leaf_eval_key, expand_key = jax.random.split(iteration_key, 3)
        root_eval_key, _ = jax.random.split(root_key)
        at_root = cycle == 0
        # the network evaluates the root or the leaf; then the tree takes one of the two updates.
        # Both are computed, as under vmap a lax.cond would, but they write only what they change,
        # where a cond would select between two whole trees
        leaf = self.find_leaf(traverse_key, state, env_step_fn)
        policy_logits, value = self.leaf_eval_fn(
            jax.tree.map(lambda r, l: jnp.where(at_root, r, l), env_state, leaf[2]),
            params,
            jnp.where(at_root, root_eval_key, leaf_eval_key),
        )
        # the leaf's update first: the root's writes the root's state, which doesn't depend on
        # the traversal's reads of the tree, so XLA would copy the arrays they read (whole trees
        # of states, every cycle) rather than order them before it. Either way round, only one
        # of the two writes anything
        state = self.expand(
            expand_key, state, leaf, policy_logits, value, write=~at_root
        )
        return self.expand_root(
            root_key,
            state,
            env_state,
            root_metadata,
            policy_logits,
            value,
            write=at_root,
        )

    def search_output(
        self,
        key: jax.Array,
        state: MCTSTree,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
    ) -> EvalOutput:
        """Samples the action to take from the root, as `evaluate` does after its iterations.

        Args: see `Evaluator.search_output`.

        Returns:
            EvalOutput: as `evaluate`'s
        """
        _, _, sample_key = jax.random.split(key, 3)
        action, policy_weights = self.sample_root_action(sample_key, state)
        return EvalOutput(
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
            key: rng, `update_root`'s (the root's noise is `start_search`'s)
            tree: MCTSTree to update
            root_embedding: root environment state
            root_metadata: metadata of the root environment state
            policy_logits: the root's policy logits
            value: the root's value
            write: (optional) whether to; if not, the tree is left as it is (see `Tree.set_root`)

        Returns:
            MCTSTree: updated MCTSTree
        """
        policy = self.prior(policy_logits, root_metadata.action_mask)
        prior = self.root_prior(tree.root_noise, policy, root_metadata.action_mask)
        root_node = tree.data_at(tree.ROOT_INDEX)
        visited = root_node.n > 0
        root_node = replace(
            root_node,
            p=prior,
            # keep old value estimate if the node has already been visited
            q=jnp.where(visited, root_node.q, value),
            # keep old visit count if the node has already been visited
            n=jnp.where(visited, root_node.n, 1),
            embedding=root_embedding,
        )
        return tree.set_root(root_node, write=write)

    def prior(self, policy_logits: jax.Array, action_mask: jax.Array) -> jax.Array:
        """A node's prior over its legal actions: the softmax of the network's policy logits,
        0 for illegal actions.

        Args:
            policy_logits: the network's policy logits
            action_mask: the legal actions

        Returns:
            jax.Array: the prior
        """
        return jax.nn.softmax(
            jnp.where(action_mask, policy_logits, jnp.finfo(policy_logits).min)
        )

    def root_prior(
        self, noise: jax.Array, policy: jax.Array, action_mask: jax.Array
    ) -> jax.Array:  # pylint: disable=unused-argument
        """The root's prior, from the network's policy over the legal actions: the policy itself.

        Args:
            noise: the search's `root_noise`
            policy: the network's policy, 0 for illegal actions
            action_mask: the legal actions

        Returns:
            jax.Array: the prior
        """
        return policy

    def find_leaf(
        self, key: jax.Array, tree: MCTSTree, env_step_fn: EnvStepFn
    ) -> tuple[jax.Array, jax.Array, Any, StepMetadata]:
        """Traverses from the root to the leaf an iteration evaluates, and steps to its state.

        Args:
            key: rng, for the traversal
            tree: MCTSTree to search
            env_step_fn: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
            the leaf: the index of the node it's under, the action from there to it, its
            environment state and its metadata
        """

        # continue while:
        # - there is an existing edge corresponding to the chosen action
        # - AND the child node connected to that edge is not terminal
        def cond_fn(state: tuple[jax.Array, jax.Array]) -> jax.Array:
            parent, action = state
            return jnp.logical_and(
                tree.is_edge(parent, action),
                ~(tree.data_at(tree.edge_map[parent, action]).terminated),
            )

        # a traversal visits each node at most once, so each selection gets its own key
        def select(node_idx: ArrayLike) -> jax.Array:
            node_key = jax.random.fold_in(key, node_idx)
            return self.action_selector(node_key, tree, node_idx, self.discount)

        # each step, unless the traversal is already at its leaf:
        # - get the index of the child node connected to the chosen action
        # - choose the action to take from the child node
        def step(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            parent, action = state
            descend = cond_fn(state)
            node_idx = tree.edge_map[parent, action]
            return (
                jnp.where(descend, node_idx, parent),
                jnp.where(descend, select(node_idx), action),
            )

        # traverse from root to leaf, from the action to take from the root
        root = jnp.asarray(tree.ROOT_INDEX)
        parent, action = jax.lax.while_loop(cond_fn, step, (root, select(root)))
        embedding, metadata = env_step_fn(tree.data_at(parent).embedding, action)
        return parent, action, embedding, metadata

    def expand(
        self,
        key: jax.Array,
        tree: MCTSTree,
        leaf: tuple[ArrayLike, ArrayLike, Any, StepMetadata],
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
        parent, action, new_embedding, metadata = leaf
        player_reward = metadata.rewards[metadata.cur_player_id]
        policy = self.prior(policy_logits, metadata.action_mask)
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
        # when the tree is full, a new leaf takes the slot of the least-visited leaf, other than
        # the root and `parent`, and not a child of the root unless every leaf is (the root's
        # children hold the policy target); of those, the one whose parent has the fewest visits,
        # which the search is least likely to come back to. The evicted leaf's visits stay
        # counted in its ancestors, as if it had never been expanded; PUCT can expand it again
        index = jnp.arange(tree.capacity)
        leaves = tree.leaves() & (index != tree.ROOT_INDEX) & (index != parent)
        deep = leaves & (tree.parents != tree.ROOT_INDEX)
        candidates = jnp.where(deep.any(), deep, leaves)
        most = jnp.iinfo(tree.data.n.dtype).max
        n = jnp.where(candidates, tree.data.n, most)
        # (only the root's parent is NULL_INDEX, and the root isn't a candidate)
        parent_n = tree.data.n[jnp.maximum(tree.parents, 0)]
        # of the least visited, the first with the least visited parent
        evict = jnp.argmin(jnp.where(n == n.min(), parent_n, most))
        tree = tree.set_child(
            parent,
            action,
            node_data,
            evict=jnp.where(candidates.any(), evict, tree.NULL_INDEX),
            write=write,
        )
        # backpropagate, only if the leaf is in the tree: otherwise (a full tree with nothing to
        # evict) its value would reach the root but no child of it, and PUCT would choose the same
        # leaf again and again. Every later iteration then does nothing, and the search ends early
        landed = write & tree.is_edge(parent, action)
        # every node from `parent` up to the root is visited once, all at the same time (rather than
        # in a loop up the path, see `Tree.path_to_root`)
        path = tree.path_to_root(jnp.where(landed, parent, tree.NULL_INDEX))
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
        # only the statistics: the policies and embeddings (a whole environment state per node)
        # are most of the tree, and backpropagation doesn't change them
        stats = replace(tree.data, p=None, embedding=None)
        nodes = jax.tree.map(lambda x: x[index], stats)
        # increment visit counts and update value estimates
        visited = self.visit_node(nodes, values)
        stats = jax.tree.map(
            lambda x, y: x.at[index].set(y, mode="drop"), stats, visited
        )
        return replace(
            tree, data=replace(stats, p=tree.data.p, embedding=tree.data.embedding)
        )

    def sample_root_action(
        self, key: jax.Array, tree: MCTSTree
    ) -> tuple[jax.Array, jax.Array]:
        """Sample an action based on the root visit counts.

        Args:
            key: rng
            tree: MCTSTree to evaluate

        Returns:
            Tuple[jax.Array, jax.Array]: sampled action, the root's visit distribution (uniform
                if its children have no visits)
        """
        n = tree.get_child_data("n", tree.ROOT_INDEX)
        total = n.sum(axis=-1)
        policy_weights = jnp.where(
            total > 0, n / jnp.maximum(total, 1), 1 / n.shape[-1]
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

    def next_state(
        self,
        state: MCTSTree,
        action: jax.Array,
        reset: ArrayLike,
        active: ArrayLike = True,
    ) -> MCTSTree:
        """`step` or `reset`, in a single pass over the tree (see `Tree.get_subtree`).

        Args: see `Evaluator.next_state`.

        Returns:
            MCTSTree: the tree for the next move
        """
        return state.get_subtree(
            action, reset=jnp.logical_or(reset, not self.persist_tree), active=active
        )

    def init(self, template_embedding: Any) -> MCTSTree:
        """Initializes the internal state of the MCTS evaluator.

        Args:
            template_embedding: template environment state embedding
                - not stored, just used to initialize data structures to the correct shape

        Returns:
            MCTSTree: initialized MCTSTree
        """
        return init_mcts_tree(
            self.max_nodes,
            self.branching_factor,
            self.new_node(
                policy=jnp.zeros((self.branching_factor,)),
                value=0.0,
                embedding=template_embedding,
                terminated=False,
            ),
        )
