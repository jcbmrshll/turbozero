from dataclasses import replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from turbozero.evaluators.evaluator import EvalOutput, Evaluator
from turbozero.evaluators.mcts.action_selection import MCTSActionSelector
from turbozero.evaluators.mcts.state import MCTSNode, MCTSTree, init_mcts_tree
from turbozero.types import EnvStepFn, LeafEvalFn, StepMetadata


class MCTS(Evaluator):
    """Monte Carlo tree search.

    Each iteration walks down the tree from the root, choosing actions with
    `action_selector`, until it reaches an action with no child yet (or a
    terminal one). It steps the environment along that action, evaluates the
    new state with `leaf_eval_fn` (policy logits and a value, e.g. from a
    network as in AlphaZero, instead of rollouts), adds it to the tree and backs
    the value up to the root. After `num_iterations` iterations, the move is
    sampled from the root's visit counts. As in AlphaZero, the root's prior can
    take Dirichlet noise, so that self-play tries moves the network rates low.
    Options can make some searches shorter (see `search_done`).

    The search's state is an `MCTSTree`, which methods take and return rather
    than store, so searches can be jitted and vmapped.
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
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.0,
        full_probability: float = 1.0,
        fast_iterations: int = 0,
        count_reused_visits: bool = False,
        min_iterations: int = 0,
        traversal_steps: int = 1,
        prior_temperature: float = 1.0,
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
            dirichlet_alpha: the concentration of the root's Dirichlet noise:
                lower puts it on fewer actions (AlphaZero: 0.3 in chess, 0.03
                in Go)
            dirichlet_epsilon: the share of the root's prior that is the noise,
                0 for none (AlphaZero: 0.25)
            full_probability: KataGo's playout cap randomization
                (https://arxiv.org/abs/1902.10565, section 3.1): the share of
                searches that are full ones, of `num_iterations` iterations. The
                others are fast ones, of `fast_iterations`, whose roots take no
                noise and whose positions don't train (see `trains_on`). For the
                same compute self-play then plays more moves, so more games for
                the value to learn from, while the policy learns only from full
                searches
            fast_iterations: the iterations of a fast search
            count_reused_visits: with `persist_tree`, a search is done once its
                root has `num_iterations` + 1 visits (`fast_iterations` + 1 for
                a fast search), counting those of the subtree it reused, rather
                than after `num_iterations` iterations. Searches from
                well-explored subtrees then take fewer iterations, and every
                root ends with the visits of a search from scratch. The tree
                never holds more than `num_iterations` + 1 + `min_iterations`
                nodes, plus one per cycle self-play runs past `search_done`
            min_iterations: with `count_reused_visits`, the fewest iterations a
                search runs however many visits it reused, at most
                `num_iterations`: the root's noise only steers the visits of the
                search that drew it
            traversal_steps: the steps down the tree a traversal takes between
                checks for its leaf. The search is the same whatever the number,
                but on GPU each check waits on the host, and under vmap every
                traversal steps until the deepest in the batch is done: more
                steps a check wait less and compute more, and which is fastest
                depends on the trees' depth
            prior_temperature: every node's prior is the softmax of
                `leaf_eval_fn`'s policy logits divided by this (Leela Chess
                Zero's policy softmax temperature). Above 1 flattens the prior,
                so the search tries more of the actions the network rates low;
                the root's noise is mixed into the flattened prior
        """
        assert 0 < full_probability <= 1
        assert full_probability == 1 or 0 < fast_iterations < num_iterations
        assert 0 <= min_iterations <= num_iterations
        assert traversal_steps >= 1
        assert prior_temperature > 0
        super().__init__(discount=discount)
        self.leaf_eval_fn = leaf_eval_fn
        self.num_iterations = num_iterations
        self.branching_factor = branching_factor
        self.max_nodes = max_nodes
        self.action_selector = action_selector
        self.temperature = temperature
        self.persist_tree = persist_tree
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.full_probability = full_probability
        self.fast_iterations = fast_iterations
        self.count_reused_visits = count_reused_visits
        self.min_iterations = min_iterations
        self.traversal_steps = traversal_steps
        self.prior_temperature = prior_temperature

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
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            "full_probability": self.full_probability,
            "fast_iterations": self.fast_iterations,
            "count_reused_visits": self.count_reused_visits,
            "min_iterations": self.min_iterations,
            "traversal_steps": self.traversal_steps,
            "prior_temperature": self.prior_temperature,
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
        """Runs a whole search for one move and picks the move.

        Evaluates the root, runs `num_iterations` iterations (or fewer, see
        `search_done`), then samples the move from the root's visit counts (see
        `search_output`). `eval_state` is an empty tree or the subtree kept from
        the previous move. Uses the same keys as running the search a cycle at a
        time (`start_search`, `search_cycle` until `search_done`,
        `search_output`), so both give the same result.
        """
        tree = self.start_search(key, eval_state)
        root_key, iterate_key, _ = jax.random.split(key, 3)
        eval_key, _ = jax.random.split(root_key)
        policy_logits, value = self.leaf_eval_fn(env_state, params, eval_key)
        tree = self.expand_root(tree, env_state, root_metadata, policy_logits, value)

        def iterate(
            tree: MCTSTree, i_key: tuple[jax.Array, jax.Array]
        ) -> tuple[MCTSTree, None]:
            i, key = i_key
            traverse_key, eval_key, _ = jax.random.split(key, 3)
            leaf = self.find_leaf(traverse_key, tree, env_step_fn)
            policy_logits, value = self.leaf_eval_fn(leaf[2], params, eval_key)
            # iterations after the search is done change nothing (it has run
            # the root's evaluation and `i` iterations, `i` + 1 cycles)
            write = True
            if self.full_probability < 1 or self.count_reused_visits:
                write = ~self.search_done(tree, i + 1)
            return self.expand(tree, leaf, policy_logits, value, write=write), None

        keys = jax.random.split(iterate_key, self.num_iterations)
        tree, _ = jax.lax.scan(iterate, tree, (jnp.arange(self.num_iterations), keys))
        return self.search_output(
            key, tree, env_state, root_metadata, params, env_step_fn
        )

    def search_cycles(self) -> int:
        """The number of cycles in a full search: one to evaluate the root,
        then one per iteration.
        """
        return self.num_iterations + 1

    def search_done(self, state: MCTSTree, cycles: jax.Array) -> jax.Array:
        """Returns whether the search in `state` is done after `cycles` cycles:
        once it has evaluated the root and run `num_iterations` iterations, or
        `fast_iterations` if it's a fast one (see `full_probability`). With
        `count_reused_visits`, also once the root has one visit more than that
        and the search has evaluated the root and run `min_iterations`
        iterations.
        """
        iterations = jnp.where(state.fast, self.fast_iterations, self.num_iterations)
        done = cycles >= iterations + 1
        if self.count_reused_visits:
            visited = state.data.n[state.ROOT_INDEX] > iterations
            done |= visited & (cycles >= 1 + self.min_iterations)
        return done

    def trains_on(self, state: MCTSTree) -> jax.Array:
        """Returns whether self-play trains on the position searched in `state`:
        unless the search is a fast one (see `full_probability`).
        """
        return ~state.fast

    def start_search(
        self, key: jax.Array, state: MCTSTree, active: ArrayLike = True
    ) -> MCTSTree:
        """Draws the root's noise for a new search and stores it in the tree:
        log-Gamma(`dirichlet_alpha`) draws, one per action, which `expand_root`
        turns into a Dirichlet sample over the legal actions (zeros without
        noise). With `full_probability` below 1, also draws whether the search
        is a fast one.

        Done once per search, before its cycles, rather than in `search_cycle`:
        under vmap every cycle computes the root's update, so the draw would be
        repeated every cycle. Trees where `active` is False are left unchanged.
        """
        # the first 3 keys are those `evaluate` splits off: the i-th key of a
        # split doesn't depend on how many it makes
        root_key, _, _, fast_key = jax.random.split(key, 4)
        _, noise_key = jax.random.split(root_key)
        shape = state.root_noise.shape
        noise = jnp.zeros(shape, dtype=jnp.float32)
        if self.dirichlet_epsilon:
            # JAX's exact sampler loops until every draw in the batch is
            # accepted, and on GPU each trip waits on the host. The approximate
            # one is a fixed computation, and its draws have been observed to be
            # as close to the exact sampler's as two sets of exact draws (KS
            # distance at alpha 0.03-2)
            noise = jax.random.loggamma(
                noise_key, self.dirichlet_alpha, shape, method="approximate"
            )
        state = replace(state, root_noise=jnp.where(active, noise, state.root_noise))
        if self.full_probability == 1:
            return state
        fast = ~jax.random.bernoulli(fast_key, self.full_probability)
        return replace(state, fast=jnp.where(active, fast, state.fast))

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
        """Runs one cycle of a search: cycle 0 evaluates the root, every later
        cycle runs one iteration, with the same keys as `evaluate`.

        Every cycle calls `leaf_eval_fn` exactly once, so a batch of searches at
        different cycles shares one batched evaluation. Cycles past
        `search_done` keep iterating (self-play runs them while it waits to
        play its moves), each with a key of its own.
        """
        root_key, iterate_key, _ = jax.random.split(key, 3)
        # the same key `evaluate` gives this iteration: the i-th key of a split
        # doesn't depend on how many keys it makes. Iterations past
        # `num_iterations` fold their number into the key instead
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
        # evaluate either the root (cycle 0) or the leaf, then apply either the
        # root's update or the leaf's. Both updates are computed, as under vmap
        # a lax.cond would anyway, but each writes only what it changes, where a
        # cond would select between two whole trees
        leaf = self.find_leaf(traverse_key, state, env_step_fn)
        policy_logits, value = self.leaf_eval_fn(
            jax.tree.map(lambda r, l: jnp.where(at_root, r, l), env_state, leaf[2]),
            params,
            jnp.where(at_root, root_eval_key, leaf_eval_key),
        )
        # the leaf's update goes first. The root's update writes the root's
        # state, which doesn't depend on what the traversal read from the tree,
        # so if it came first XLA would copy the arrays the traversal reads
        # (whole trees of states, every cycle) instead of ordering the reads
        # before the write. Only one of the two updates writes anything
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
        """Picks the move once a search is done, and returns it with the
        policy weights.

        The weights are the root's children's visit counts, normalized (uniform
        if none has been visited). The move is sampled in proportion to the
        weights to the power of 1 / `temperature`; at temperature 0 it's the
        most visited action, ties broken at random.
        """
        _, _, sample_key = jax.random.split(key, 3)
        n = state.get_child_data("n", state.ROOT_INDEX)
        total = n.sum(axis=-1)
        weights = jnp.where(total > 0, n / jnp.maximum(total, 1), 1 / n.shape[-1])
        if self.temperature == 0:
            # pick one of the most visited actions at random (equal visits give
            # exactly equal weights)
            noise = jax.random.uniform(sample_key, shape=weights.shape)
            action = jnp.argmax(jnp.where(weights == weights.max(), noise, -1.0))
            return EvalOutput(state, action, weights)
        tempered = weights ** (1 / self.temperature)
        tempered /= tempered.sum()
        action = jax.random.choice(sample_key, tempered.shape[-1], p=tempered)
        return EvalOutput(state, action, weights)

    def get_value(self, state: MCTSTree) -> jax.Array:
        """Returns the root's value estimate: the mean of the values backed up
        to it.
        """
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
        """Writes the evaluation of the root into the tree.

        The root gets the `prior` of `policy_logits`, with the search's
        Dirichlet noise mixed in, as its prior, and `root_embedding` as its
        state. A new root also gets `value` and one visit; a root kept from the
        previous move keeps its visits and value. Does nothing if `write` is
        False (see `Tree.set_root`).
        """
        mask = root_metadata.action_mask
        prior = self.prior(policy_logits, mask)
        if self.dirichlet_epsilon:
            # the softmax of log-Gamma(alpha) draws is a Dirichlet(alpha) sample,
            # here over the legal actions only, as in AlphaZero. Both are
            # distributions over the legal actions, so their mix is too
            lowest = jnp.finfo(policy_logits).min
            noise = jax.nn.softmax(jnp.where(mask, tree.root_noise, lowest))
            epsilon = self.dirichlet_epsilon
            mixed = (1 - epsilon) * prior + epsilon * noise
            # fast searches' roots take no noise. The barrier keeps XLA from
            # fusing the mix into the select, which it does differently in
            # `evaluate` and in a search a cycle at a time, rounding differently
            if self.full_probability < 1:
                mixed = jnp.where(tree.fast, prior, jax.lax.optimization_barrier(mixed))
            prior = mixed
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

    def prior(self, policy_logits: jax.Array, action_mask: jax.Array) -> jax.Array:
        """Computes a node's prior: the softmax of `policy_logits` divided by
        `prior_temperature`, over the legal actions (0 for the others).
        """
        # divided before masking: the lowest float divided by a temperature
        # below 1 would overflow
        logits = policy_logits / self.prior_temperature
        return jax.nn.softmax(jnp.where(action_mask, logits, jnp.finfo(logits).min))

    def find_leaf(
        self, key: jax.Array, tree: MCTSTree, env_step_fn: EnvStepFn
    ) -> tuple[jax.Array, jax.Array, Any, StepMetadata]:
        """Walks down from the root to the leaf the next iteration evaluates.

        At each node it chooses an action with `action_selector`, and moves on
        to that child if it exists and isn't terminal. It stops at the first
        action without a child (or with a terminal one) and steps the
        environment along it. Returns the parent's index, the action, and the
        resulting environment state and metadata.
        """

        # keep going while the chosen action leads to an existing child that
        # isn't terminal
        def descend(state: tuple[jax.Array, jax.Array]) -> jax.Array:
            parent, action = state
            child = tree.edge_map[parent, action]
            return tree.is_edge(parent, action) & ~tree.data_at(child).terminated

        # a traversal visits each node at most once, so folding the node's index
        # into the key gives each selection its own key
        def select(index: ArrayLike) -> jax.Array:
            node_key = jax.random.fold_in(key, index)
            return self.action_selector(node_key, tree, index, self.discount)

        def step(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            child = tree.edge_map[state[0], state[1]]
            return child, select(child)

        # `traversal_steps` steps a check (see `__init__`): the check holds for
        # the first, and the others stay put once the traversal is at its leaf
        def steps(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            state = step(state)
            for _ in range(self.traversal_steps - 1):
                down = descend(state)
                state = jax.tree.map(partial(jnp.where, down), step(state), state)
            return state

        root = jnp.asarray(tree.ROOT_INDEX)
        parent, action = jax.lax.while_loop(descend, steps, (root, select(root)))
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
        """Adds `leaf` (from `find_leaf`) to the tree and backs its value up
        to the root.

        The new node's prior is the `prior` of `policy_logits`, and its value is `value`, or the reward if the leaf is
        terminal. A terminal node already in the tree is visited again instead.
        In a full tree the new node replaces an evicted leaf; if there's none to
        evict, nothing changes. Then every node from the parent up to the root
        gets one more visit and the value, negated (or discounted) once per
        level. Does nothing if `write` is False.
        """
        parent, action, embedding, metadata = leaf
        player_reward = metadata.rewards[metadata.cur_player_id]
        value = jnp.where(metadata.terminated, player_reward, value)
        # compute both the new node and the revisited one and pick field by
        # field: under vmap a lax.cond on `exists` would compute both anyway, as
        # a select between whole trees (see `Tree.set_child`)
        exists = tree.is_edge(parent, action)
        node = tree.data_at(tree.edge_map[parent, action])
        node = MCTSNode(
            n=jnp.where(exists, node.n + 1, 1),
            q=jnp.where(exists, (node.q * node.n + value) / (node.n + 1), value),
            p=self.prior(policy_logits, metadata.action_mask),
            terminated=metadata.terminated,
            embedding=embedding,
        )
        # choose the leaf to evict if the tree is full: the least visited leaf
        # other than the root and `parent`, avoiding the root's children unless
        # every leaf is one (their visits make the policy weights). Ties go to
        # the leaf whose parent has the fewest visits, which the search is least
        # likely to come back to. The evicted leaf's visits stay counted in its
        # ancestors, as if it had never been expanded, and it can be expanded
        # again later
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
        # back the value up only if the leaf made it into the tree. Otherwise (a
        # full tree with nothing to evict) the value would reach the root but no
        # child of it, and the selector would choose the same leaf again and
        # again. Every later iteration then does nothing
        landed = write & tree.is_edge(parent, action)
        # update every node from `parent` up to the root at once, rather than in
        # a loop up the path (see `Tree.path_to_root`)
        path = tree.path_to_root(jnp.where(landed, parent, tree.NULL_INDEX))
        # the value each node on the path gets: discounted once per level up
        if abs(self.discount) == 1:
            # flip signs rather than multiply: the same values a loop up the
            # path computes, and no product for XLA to fuse into the update below
            # (an FMA rounds differently)
            levels = jnp.arange(1, tree.capacity + 1)
            values = jnp.where(
                (self.discount == -1) & (levels % 2 == 1), jnp.negative(value), value
            )
        else:
            values = value * jnp.cumprod(jnp.full(path.shape, self.discount))
        # entries past the root point out of bounds, so their writes are dropped
        index = jnp.where(path == tree.NULL_INDEX, tree.capacity, path)
        # one more visit, and the value averaged in. Only n and q are read and
        # written: the rest of the node data (the states above all) doesn't
        # change
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
        """Empties the tree."""
        return state.reset()

    def step(self, state: MCTSTree, action: jax.Array) -> MCTSTree:
        """Moves the tree on after `action` is played: keeps the action's
        subtree, or empties the tree if `persist_tree` is off.
        """
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
        """Does `step`, or `reset` if `reset` is True, in a single pass over the
        tree (see `Tree.get_subtree`). Trees where `active` is False are left
        unchanged.
        """
        return state.get_subtree(
            action, reset=jnp.logical_or(reset, not self.persist_tree), active=active
        )

    def init(self, template_embedding: Any) -> MCTSTree:
        """Creates an empty tree for environment states shaped like
        `template_embedding`.
        """
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
