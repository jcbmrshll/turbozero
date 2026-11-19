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
    """Monte Carlo tree search that evaluates its leaves with `leaf_eval_fn`,
    a policy and a value (e.g. a network's, as in AlphaZero), rather than with
    rollouts. Its state is an `MCTSTree`; every method is a pure function of
    it, so searches jit and vmap.

    Each iteration traverses the tree with `action_selector` to a leaf, steps
    the environment to it, evaluates it with `leaf_eval_fn` and backs its value
    up to the root. Subclasses can add noise to the root's prior (`root_noise`,
    `root_prior`).
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
        """
        Args:
            branching_factor: the number of actions
            max_nodes: the tree's size, by convention 4 * `num_iterations`. A
                search needing a new node in a full tree evicts a leaf, so it
                runs all its iterations whatever the size, but forgets part of
                the tree, and with `persist_tree` the reused subtree takes room
                too. Smaller trees (e.g. 2 * `num_iterations`) have been
                observed to cost strength in some settings
            num_iterations: iterations per search, after the root's evaluation
            discount: see `Evaluator`
            temperature: each move is played with probability proportional to
                its root visits to the power of 1 / temperature; at 0, the most
                visited move is played, ties broken at random
            persist_tree: whether the next move's search starts from the
                subtree of the move played, rather than an empty tree
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
        """Searches `env_state` from the tree `eval_state` (empty, or the
        subtree reused from the last move): the root's evaluation, then
        `num_iterations` iterations, then the move sampled from the root's
        visits, with the keys a search a cycle at a time uses.
        """
        tree = self.start_search(key, eval_state)
        root_key, iterate_key, _ = jax.random.split(key, 3)
        eval_key, _ = jax.random.split(root_key)
        policy_logits, value = self.leaf_eval_fn(env_state, params, eval_key)
        tree = self.expand_root(tree, env_state, root_metadata, policy_logits, value)

        def iterate(tree: MCTSTree, key: jax.Array) -> tuple[MCTSTree, None]:
            traverse_key, eval_key, _ = jax.random.split(key, 3)
            leaf = self.find_leaf(traverse_key, tree, env_step_fn)
            policy_logits, value = self.leaf_eval_fn(leaf[2], params, eval_key)
            return self.expand(tree, leaf, policy_logits, value), None

        keys = jax.random.split(iterate_key, self.num_iterations)
        tree, _ = jax.lax.scan(iterate, tree, keys)
        return self.search_output(
            key, tree, env_state, root_metadata, params, env_step_fn
        )

    def search_cycles(self) -> int:
        """The root's evaluation, then `num_iterations` iterations."""
        return self.num_iterations + 1

    def start_search(
        self, key: jax.Array, state: MCTSTree, active: ArrayLike = True
    ) -> MCTSTree:
        """Draws the root's noise (see `root_noise`) for the search with key
        `key`, once: under vmap, a search a cycle at a time computes the root's
        update in every cycle (see `search_cycle`).
        """
        root_key, _, _ = jax.random.split(key, 3)
        _, noise_key = jax.random.split(root_key)
        noise = self.root_noise(noise_key, state.root_noise.shape)
        return replace(state, root_noise=jnp.where(active, noise, state.root_noise))

    def root_noise(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """The noise, float32 of shape (F,), that `root_prior` mixes into the
        root's prior: none.
        """
        return jnp.zeros(shape, dtype=jnp.float32)

    def root_prior(
        self, noise: jax.Array, policy: jax.Array, action_mask: jax.Array
    ) -> jax.Array:
        """The root's prior, from the search's `noise` and the leaf evaluation's
        policy (0 for illegal actions): the policy itself.
        """
        return policy

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
        """The root's evaluation (cycle 0) or an iteration, as `evaluate` makes
        them, with the same keys. Either way one call of `leaf_eval_fn`, so a
        batch of searches at different cycles makes one batched call. Past
        `num_iterations`, more iterations, each with a key of its own.
        """
        root_key, iterate_key, _ = jax.random.split(key, 3)
        # the iteration's key, as `evaluate` splits it off (a split's i-th key
        # doesn't depend on how many it makes); past `num_iterations`, folded in
        iteration = jnp.maximum(cycle - 1, 0)
        iteration_key = jnp.where(
            iteration < self.num_iterations,
            jax.random.split(iterate_key, self.num_iterations)[
                jnp.minimum(iteration, self.num_iterations - 1)
            ],
            jax.random.fold_in(iterate_key, iteration),
        )
        traverse_key, leaf_eval_key, _ = jax.random.split(iteration_key, 3)
        root_eval_key, _ = jax.random.split(root_key)
        at_root = cycle == 0
        # `leaf_eval_fn` evaluates the root or the leaf, then the tree takes one
        # of the two updates. Both are computed, as under vmap a lax.cond would,
        # but each writes only what it changes, where a cond would select
        # between two whole trees
        leaf = self.find_leaf(traverse_key, state, env_step_fn)
        policy_logits, value = self.leaf_eval_fn(
            jax.tree.map(lambda r, l: jnp.where(at_root, r, l), env_state, leaf[2]),
            params,
            jnp.where(at_root, root_eval_key, leaf_eval_key),
        )
        # the leaf's update first: the root's writes the root's state, which
        # doesn't depend on the traversal's reads of the tree, so XLA would copy
        # the arrays they read (whole trees of states, every cycle) rather than
        # order them before it. Either way round, only one of the two writes
        state = self.expand(state, leaf, policy_logits, value, write=~at_root)
        return self.expand_root(
            state, env_state, root_metadata, policy_logits, value, write=at_root
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
        """The move, sampled from the root's visits (see `temperature`), and the
        policy weights: the root's visit distribution, uniform if its children
        have none.
        """
        _, _, sample_key = jax.random.split(key, 3)
        n = state.get_child_data("n", state.ROOT_INDEX)
        total = n.sum(axis=-1)
        weights = jnp.where(total > 0, n / jnp.maximum(total, 1), 1 / n.shape[-1])
        if self.temperature == 0:
            # of the most visited actions, one at random: equal visits give
            # exactly equal weights
            noise = jax.random.uniform(sample_key, shape=weights.shape)
            action = jnp.argmax(jnp.where(weights == weights.max(), noise, -1.0))
            return EvalOutput(state, action, weights)
        tempered = weights ** (1 / self.temperature)
        tempered /= tempered.sum()
        action = jax.random.choice(sample_key, tempered.shape[-1], p=tempered)
        return EvalOutput(state, action, weights)

    def get_value(self, state: MCTSTree) -> jax.Array:
        """The root's value: the mean of the values backed up to it."""
        return state.data_at(state.ROOT_INDEX).q

    def expand_root(
        self,
        tree: MCTSTree,
        root_embedding: Any,
        root_metadata: StepMetadata,
        policy_logits: jax.Array,
        value: ArrayLike,
        write: ArrayLike = True,
    ) -> MCTSTree:
        """The tree with `leaf_eval_fn`'s evaluation of the root: its prior (see
        `root_prior`), and its value and one visit unless a reused root has
        them already. Unchanged unless `write` (see `Tree.set_root`).
        """
        mask = root_metadata.action_mask
        policy = jax.nn.softmax(
            jnp.where(mask, policy_logits, jnp.finfo(policy_logits).min)
        )
        prior = self.root_prior(tree.root_noise, policy, mask)
        node = tree.data_at(tree.ROOT_INDEX)
        visited = node.n > 0
        node = replace(
            node,
            p=prior,
            q=jnp.where(visited, node.q, value),
            n=jnp.where(visited, node.n, 1),
            embedding=root_embedding,
        )
        return tree.set_root(node, write=write)

    def find_leaf(
        self, key: jax.Array, tree: MCTSTree, env_step_fn: EnvStepFn
    ) -> tuple[jax.Array, jax.Array, Any, StepMetadata]:
        """The leaf an iteration evaluates: the node the traversal stops at,
        the action from it to a new or terminal child, and that child's
        environment state and metadata.
        """

        # on along the chosen action while it leads to a child that isn't
        # terminal
        def descend(state: tuple[jax.Array, jax.Array]) -> jax.Array:
            parent, action = state
            child = tree.edge_map[parent, action]
            return tree.is_edge(parent, action) & ~tree.data_at(child).terminated

        # a traversal visits each node at most once, so each selection gets its
        # own key
        def select(index: ArrayLike) -> jax.Array:
            node_key = jax.random.fold_in(key, index)
            return self.action_selector(node_key, tree, index, self.discount)

        def step(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            child = tree.edge_map[state[0], state[1]]
            return child, select(child)

        root = jnp.asarray(tree.ROOT_INDEX)
        parent, action = jax.lax.while_loop(descend, step, (root, select(root)))
        embedding, metadata = env_step_fn(tree.data_at(parent).embedding, action)
        return parent, action, embedding, metadata

    def expand(
        self,
        tree: MCTSTree,
        leaf: tuple[ArrayLike, ArrayLike, Any, StepMetadata],
        policy_logits: jax.Array,
        value: ArrayLike,
        write: ArrayLike = True,
    ) -> MCTSTree:
        """The tree with `leaf` (see `find_leaf`) added, or visited again if
        it's a terminal node already there, and its value (the outcome, if
        terminal) backed up to the root. Unchanged unless `write`.
        """
        parent, action, embedding, metadata = leaf
        player_reward = metadata.rewards[metadata.cur_player_id]
        value = jnp.where(metadata.terminated, player_reward, value)
        # the new node and the visited one, picked field by field: under vmap a
        # lax.cond on `exists` would compute both anyway, as a select between
        # whole trees (see `Tree.set_child`)
        exists = tree.is_edge(parent, action)
        node = tree.data_at(tree.edge_map[parent, action])
        node = MCTSNode(
            n=jnp.where(exists, node.n + 1, 1),
            q=jnp.where(exists, (node.q * node.n + value) / (node.n + 1), value),
            p=jax.nn.softmax(
                jnp.where(
                    metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min
                )
            ),
            terminated=metadata.terminated,
            embedding=embedding,
        )
        # in a full tree, the new node takes the slot of the least visited leaf
        # other than the root and `parent`, and not a child of the root unless
        # every leaf is (the root's children make the policy weights); of those,
        # the one whose parent has the fewest visits, which the search is least
        # likely to come back to. Its visits stay counted in its ancestors, as
        # if it had never been expanded; PUCT can expand it again
        index = jnp.arange(tree.capacity)
        leaves = tree.leaves() & (index != tree.ROOT_INDEX) & (index != parent)
        deep = leaves & (tree.parents != tree.ROOT_INDEX)
        candidates = jnp.where(deep.any(), deep, leaves)
        most = jnp.iinfo(tree.data.n.dtype).max
        n = jnp.where(candidates, tree.data.n, most)
        # (only the root's parent is NULL_INDEX, and the root isn't a candidate)
        parent_n = tree.data.n[jnp.maximum(tree.parents, 0)]
        evict = jnp.argmin(jnp.where(n == n.min(), parent_n, most))
        tree = tree.set_child(
            parent,
            action,
            node,
            evict=jnp.where(candidates.any(), evict, tree.NULL_INDEX),
            write=write,
        )
        # back up only if the leaf is in the tree: otherwise (a full tree with
        # nothing to evict) its value would reach the root but no child of it,
        # and PUCT would choose the same leaf again and again. Every later
        # iteration then does nothing, and the search ends early
        landed = write & tree.is_edge(parent, action)
        # every node from `parent` up to the root at once, rather than in a loop
        # up the path (see `Tree.path_to_root`)
        path = tree.path_to_root(jnp.where(landed, parent, tree.NULL_INDEX))
        # discounted once more at each level up
        if abs(self.discount) == 1:
            # signs flipped rather than multiplied: exactly the values a loop up
            # the path computes, and no product for XLA to fuse into the update
            # below (an FMA rounds differently)
            levels = jnp.arange(1, tree.capacity + 1)
            values = jnp.where(
                (self.discount == -1) & (levels % 2 == 1), jnp.negative(value), value
            )
        else:
            values = value * jnp.cumprod(jnp.full(path.shape, self.discount))
        # entries past the root point out of bounds, so their writes are dropped
        index = jnp.where(path == tree.NULL_INDEX, tree.capacity, path)
        # only n and q: the rest (the states most of all) doesn't change
        n, q = tree.data.n[index], tree.data.q[index]
        return replace(
            tree,
            data=replace(
                tree.data,
                n=tree.data.n.at[index].set(n + 1, mode="drop"),
                q=tree.data.q.at[index].set((q * n + values) / (n + 1), mode="drop"),
            ),
        )

    def reset(self, state: MCTSTree) -> MCTSTree:
        return state.reset()

    def step(self, state: MCTSTree, action: jax.Array) -> MCTSTree:
        """The subtree of `action`, or with `persist_tree` off, an empty tree."""
        if self.persist_tree:
            return state.get_subtree(action)
        return state.reset()

    def next_state(
        self,
        state: MCTSTree,
        action: jax.Array,
        reset: ArrayLike,
        active: ArrayLike = True,
    ) -> MCTSTree:
        """`step` or `reset`, in a single pass over the tree (see
        `Tree.get_subtree`).
        """
        return state.get_subtree(
            action, reset=jnp.logical_or(reset, not self.persist_tree), active=active
        )

    def init(self, template_embedding: Any) -> MCTSTree:
        return init_mcts_tree(
            self.max_nodes,
            self.branching_factor,
            MCTSNode(
                n=jnp.array(0, dtype=jnp.int32),
                p=jnp.zeros((self.branching_factor,), dtype=jnp.float32),
                q=jnp.array(0.0, dtype=jnp.float32),
                terminated=jnp.array(False),
                embedding=template_embedding,
            ),
        )
