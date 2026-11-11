from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts import learned_stop
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.learned_stop import LearnedStop
from core.evaluators.mcts.policy_targets import PolicyTarget, VisitCounts
from core.evaluators.mcts.state import (
    Leaf,
    MCTSNode,
    MCTSOutput,
    MCTSTree,
    SearchRecord,
    TraversalState,
    backprop_stats,
    init_mcts_tree,
    search_record,
    with_backprop_stats,
)
from core.types import EnvStepFn, EvalFn, StepMetadata


@dataclass(frozen=True)
class PlayoutCap:
    """Playout cap randomization (KataGo, https://arxiv.org/abs/1902.10565, section 3.1): each
    move's search is a full one, with probability `full_probability`, or else a fast one of
    `fast_iterations` iterations. Only full searches make training positions (see
    `MCTS.trains_on`); fast ones play their move without the root's noise (e.g. AlphaZero's
    Dirichlet noise), for strength. More moves for the same compute means more games, each with
    one outcome for the value to learn from, while the policy learns from full searches only.

    Attributes:
        full_probability: the share of moves searched in full, `num_iterations` iterations
        fast_iterations: the iterations of a fast search
    """

    full_probability: float
    fast_iterations: int

    def get_config(self) -> dict:
        return {
            "full_probability": self.full_probability,
            "fast_iterations": self.fast_iterations,
        }


@dataclass(frozen=True)
class SmartPruning:
    """STOP (Baier and Winands 2016), Leela Chess Zero's smart pruning: a search stops once
    `factor` times the lead in visits of the root's most visited child over the runner-up
    exceeds the iterations the search has left. The visits counted include those the root kept
    from earlier searches (see `MCTS`'s `persist_tree`). At `factor` 1 the most visited child
    can't change, so the move is the one the whole search would play greedily (most visits),
    only its visit distribution differs; above 1, it stops sooner and the move may differ.

    It applies to some searches only (`searches`):
    - "fast-greedy": fast searches (see `PlayoutCap`) whose moves are played greedily (see
      `Evaluator.search_done`'s `greedy`, or the MCTS's temperature 0): neither their visit
      distribution nor their move's sampling then matters, only the move and the subtree the
      next search keeps
    - "fast": every fast search, whose moves may be sampled from its visits
    - "all": every search, those that make training positions included

    Attributes:
        searches: which searches it applies to, see above
        factor: how readily it stops, higher sooner; 1 never changes a greedy move
        min_full_iterations: the fewest iterations a search that makes a training position takes
            before the rule may stop it, so that the root's noise (e.g. AlphaZero's) steers some
            of its visits
    """

    searches: str
    factor: float = 1.0
    min_full_iterations: int = 0

    SEARCHES = ("fast-greedy", "fast", "all")

    def __post_init__(self):
        if self.searches not in self.SEARCHES:
            raise ValueError(
                f"smart pruning's searches must be one of {self.SEARCHES}, got {self.searches!r}"
            )
        if self.factor <= 0:
            raise ValueError(
                f"smart pruning's factor must be positive, got {self.factor}"
            )
        if self.min_full_iterations < 0:
            raise ValueError(
                "smart pruning's min_full_iterations can't be negative, got "
                f"{self.min_full_iterations}"
            )

    def get_config(self) -> dict:
        return {
            "searches": self.searches,
            "factor": self.factor,
            "min_full_iterations": self.min_full_iterations,
        }


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
        count_reused_visits: bool = False,
        min_iterations: int = 0,
        policy_target: PolicyTarget | None = None,
        traversal_steps: int = 1,
        playout_cap: PlayoutCap | None = None,
        smart_pruning: SmartPruning | None = None,
        learned_stop: LearnedStop | None = None,
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
            count_reused_visits: (optional) with `persist_tree`, stop a search once its root has
                `num_iterations + 1` visits, those of the subtree it reused from the last search
                counted, rather than after `num_iterations` iterations (default: False). A search from
                a well-explored subtree then takes fewer iterations, and every root ends with the visits
                of a search from scratch. The tree then never holds more than
                `num_iterations + 1 + min_iterations` nodes, so `max_nodes` needs no more
            min_iterations: (optional) with `count_reused_visits`, the fewest iterations a search
                takes, however many visits it reused: the root's noise (e.g. AlphaZero's) only steers
                the visits of its own search (default: 0). At most `num_iterations`
            policy_target: (optional) what the policy trains towards, from the root of a finished
                search (default: `VisitCounts`, its visit distribution)
            traversal_steps: (optional) steps down the tree a traversal takes between checks for
                having reached its leaf (default: 1). Steps past it change nothing, so the search is
                the same whatever the number; but on GPU each check waits on the host, and under
                vmap a traversal goes on until the deepest in the batch is done: more steps a check
                wait less and compute more, and which is fastest depends on the trees' depth
            playout_cap: (optional) search each move in full (`num_iterations` iterations) only
                some of the time, and otherwise with fewer, see `PlayoutCap`. With
                `count_reused_visits`, a fast search stops at `fast_iterations + 1` root visits
            smart_pruning: (optional) stop some searches once their most visited move is
                settled, see `SmartPruning`. Its "fast" choices need a `playout_cap`
            learned_stop: (optional) stop some searches once a network fit during self-play
                predicts that searching on would gain little, see `LearnedStop`; with
                `smart_pruning`, a search stops when either does. Its "fast" choices need a
                `playout_cap`
        """
        if traversal_steps < 1:
            raise ValueError(
                f"traversal_steps must be at least 1, got {traversal_steps}"
            )
        if min_iterations < 0 or min_iterations > num_iterations:
            raise ValueError(
                f"min_iterations must be 0 to num_iterations ({num_iterations}), "
                f"got {min_iterations}"
            )
        if playout_cap is not None and not (
            0 < playout_cap.full_probability <= 1
            and 0 < playout_cap.fast_iterations < num_iterations
        ):
            raise ValueError(
                "a playout cap's full_probability must be in (0, 1] and its fast_iterations "
                f"between 1 and num_iterations ({num_iterations}) - 1, got {playout_cap}"
            )
        if (
            smart_pruning is not None
            and smart_pruning.searches != "all"
            and playout_cap is None
        ):
            raise ValueError(
                f"smart pruning of {smart_pruning.searches!r} searches needs a playout_cap"
            )
        if (
            smart_pruning is not None
            and smart_pruning.min_full_iterations > num_iterations
        ):
            raise ValueError(
                "smart pruning's min_full_iterations must be at most num_iterations "
                f"({num_iterations}), got {smart_pruning.min_full_iterations}"
            )
        if (
            learned_stop is not None
            and learned_stop.searches != "all"
            and playout_cap is None
        ):
            raise ValueError(
                f"a learned stop of {learned_stop.searches!r} searches needs a playout_cap"
            )
        super().__init__(discount=discount)
        self.playout_cap = playout_cap
        self.smart_pruning = smart_pruning
        self.learned_stop = learned_stop
        self.count_reused_visits = count_reused_visits
        self.min_iterations = min_iterations
        self.eval_fn = eval_fn
        self.num_iterations = num_iterations
        self.branching_factor = branching_factor
        self.max_nodes = max_nodes
        self.action_selector = action_selector
        self.temperature = temperature
        self.persist_tree = persist_tree
        self.policy_target = policy_target or VisitCounts()
        self.traversal_steps = traversal_steps

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
            "count_reused_visits": self.count_reused_visits,
            "min_iterations": self.min_iterations,
            "policy_target_config": self.policy_target.get_config(),
            "traversal_steps": self.traversal_steps,
            "playout_cap": self.playout_cap.get_config()
            if self.playout_cap is not None
            else None,
            "smart_pruning": self.smart_pruning.get_config()
            if self.smart_pruning is not None
            else None,
            "learned_stop": self.learned_stop.get_config()
            if self.learned_stop is not None
            else None,
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
        """Performs `self.num_iterations` MCTS iterations on an `MCTSTree` (fewer with
        `count_reused_visits`, a `playout_cap` or `smart_pruning`, see `search_done`).

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
        eval_state = self.start_search(key, eval_state)
        # store current state metadata in the root node
        root_key, iterate_key, sample_key = jax.random.split(key, 3)
        eval_state = self.update_root(
            root_key, eval_state, env_state, params, root_metadata=root_metadata
        )
        # perform 'num_iterations' iterations of MCTS
        iterate = partial(self.iterate, params=params, env_step_fn=env_step_fn)

        iterate_keys = jax.random.split(iterate_key, self.num_iterations)

        def iteration(
            state: MCTSTree, i_key: tuple[jax.Array, jax.Array]
        ) -> tuple[MCTSTree, None]:
            i, k = i_key
            if (
                not self.count_reused_visits
                and self.playout_cap is None
                and self.smart_pruning is None
                and self.learned_stop is None
            ):
                return iterate(k, state), None
            # after the root's evaluation and `i` iterations, as `search_cycle` counts them
            return iterate(k, state, write=~self.search_done(state, i + 1)), None

        eval_state, _ = jax.lax.scan(
            iteration, eval_state, (jnp.arange(self.num_iterations), iterate_keys)
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

    def mean_iterations(self) -> float:
        """The iterations a search runs on average: `num_iterations`, or with a `playout_cap`,
        its mix of full and fast searches' (counting reused visits, at most these)."""
        cap = self.playout_cap
        if cap is None:
            return float(self.num_iterations)
        return (
            cap.full_probability * self.num_iterations
            + (1 - cap.full_probability) * cap.fast_iterations
        )

    def iterations(self, state: MCTSTree) -> jax.Array:
        """The iterations of the search in `state`: `num_iterations`, or a fast search's (see
        `PlayoutCap`)."""
        if self.playout_cap is None:
            return jnp.array(self.num_iterations)
        return jnp.where(
            state.fast, self.playout_cap.fast_iterations, self.num_iterations
        )

    def search_done(
        self, state: MCTSTree, cycles: jax.Array, greedy: ArrayLike = False
    ) -> jax.Array:
        """Done after the root's evaluation and its `iterations`; with `count_reused_visits`,
        also once the root has `iterations + 1` visits, after its evaluation and
        `min_iterations` iterations; with `smart_pruning`, also once it stops the search (see
        `pruned`). At temperature 0, every move is played greedily."""
        iterations = self.iterations(state)
        done = cycles >= iterations + 1
        if self.count_reused_visits:
            visited = state.data.n[state.ROOT_INDEX] > iterations
            done = done | (visited & (cycles >= 1 + self.min_iterations))
        stop = jnp.array(False)
        if self.smart_pruning is not None:
            stop = self.pruned(state, cycles, greedy)
        if self.learned_stop is not None:
            stopper = state.stopper
            features = learned_stop.root_features(
                state, cycles, iterations, self.discount
            )
            gain = learned_stop.predict(stopper.params, features)
            stop = stop | (stopper.applies & (cycles >= 1) & (gain < stopper.threshold))
            # a search labelling the rule runs to its end
            stop = stop & ~(stopper.label & stopper.applies)
        return done | stop

    def applies(
        self, searches: str, state: MCTSTree, greedy: ArrayLike = False
    ) -> jax.Array:
        """Whether a stop rule of `searches` (see `SmartPruning`) applies to the search in
        `state`, whose move is played greedily or not."""
        if searches == "all":
            return jnp.array(True)
        if searches == "fast-greedy" and self.temperature != 0:
            return state.fast & jnp.asarray(greedy)
        return state.fast

    def pruned(
        self, state: MCTSTree, cycles: ArrayLike, greedy: ArrayLike = False
    ) -> jax.Array:
        """Whether `smart_pruning` stops the search in `state` after `cycles` cycles: once the
        root is evaluated, if the rule applies to the search (see `SmartPruning`) and the lead
        of its most visited child over the runner-up, times the rule's factor, exceeds the
        iterations it has left. A search that makes a training position takes the rule's
        `min_full_iterations` first."""
        rule = self.smart_pruning
        assert rule is not None
        cycles = jnp.asarray(cycles)
        visits = state.get_child_data("n", state.ROOT_INDEX)
        top, runner_up = jax.lax.top_k(visits, 2)[0]
        left = self.iterations(state) - (cycles - 1)
        settled = rule.factor * (top - runner_up) > left
        applies = self.applies(rule.searches, state, greedy)
        if rule.searches == "all":
            applies = ~self.trains_on(state) | (cycles - 1 >= rule.min_full_iterations)
        return settled & applies & (cycles >= 1)

    def search_stats(self, state: MCTSTree, cycles: jax.Array) -> dict[str, jax.Array]:
        """Besides its `cycles` (see `Evaluator.search_stats`): `stopped`, 1 if the search
        stopped before its `iterations` were up (e.g. by `smart_pruning`); and
        `inherited_visits`, the visits its root had from earlier searches as it started (see
        `persist_tree`; a root searched from scratch counts 1, its own evaluation)."""
        cycles = jnp.asarray(cycles)
        root_visits = state.data.n[state.ROOT_INDEX]
        return {
            "cycles": cycles.astype(jnp.float32),
            "stopped": (cycles < self.iterations(state) + 1).astype(jnp.float32),
            # every cycle after the root's evaluation visits the root once
            "inherited_visits": jnp.maximum(root_visits - (cycles - 1), 0).astype(
                jnp.float32
            ),
        }

    def trains_on(self, state: MCTSTree) -> jax.Array:
        """Whether the search in `state` makes a training position: unless it's a fast one
        (see `PlayoutCap`)."""
        return ~state.fast

    def start_search(
        self,
        key: jax.Array,
        state: MCTSTree,
        active: ArrayLike = True,
        greedy: ArrayLike = False,
    ) -> MCTSTree:
        """Draws the root's noise (see `root_noise`) for the search for a move with key `key`, as
        `evaluate` does first: a search a cycle at a time draws it once, rather than in every
        cycle (under vmap, the root's update is computed in every cycle, see `search_cycle`).
        With a `playout_cap`, also draws whether the search is a fast one; with a
        `learned_stop`, whether the search labels it.

        Args: see `Evaluator.start_search`.

        Returns:
            MCTSTree: the tree, with its `root_noise` (and `fast`, and its stop rule's state)
        """
        state = self._start_search(key, state, active)
        if self.learned_stop is None:
            return state
        # the 5th key of the split (see `_start_search`)
        label_key = jax.random.split(key, 5)[4]
        label = jax.random.bernoulli(label_key, self.learned_stop.label_probability)
        stopper = state.stopper
        applies = self.applies(self.learned_stop.searches, state, greedy)
        active = jnp.asarray(active)
        return replace(
            state,
            stopper=replace(
                stopper,
                label=jnp.where(active, label, stopper.label),
                greedy=jnp.where(active, jnp.asarray(greedy), stopper.greedy),
                applies=jnp.where(active, applies, stopper.applies),
                recorded=stopper.recorded & ~active,
            ),
        )

    def _start_search(
        self, key: jax.Array, state: MCTSTree, active: ArrayLike = True
    ) -> MCTSTree:
        """`start_search`'s root noise and playout cap's draw."""
        # the 4th key of the split: the first 3 are those `evaluate` splits off (the split's i-th
        # key doesn't depend on how many it makes)
        root_key, _, _, cap_key = jax.random.split(key, 4)
        _, noise_key = jax.random.split(root_key)
        noise = self.root_noise(noise_key, state.root_noise.shape)
        state = replace(state, root_noise=jnp.where(active, noise, state.root_noise))
        if self.playout_cap is None:
            return state
        fast = ~jax.random.bernoulli(cap_key, self.playout_cap.full_probability)
        return replace(state, fast=jnp.where(active, fast, state.fast))

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
        at different points evaluates the network once a cycle. Past `search_done`, more
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
        policy_logits, value = self.eval_fn(
            jax.tree.map(
                lambda r, l: jnp.where(at_root, r, l), env_state, leaf.embedding
            ),
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
        state = self.expand_root(
            root_key,
            state,
            env_state,
            root_metadata,
            policy_logits,
            value,
            write=at_root,
        )
        return self.record_check(state, cycle + 1)

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
        policy_logits = jnp.where(
            root_metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min
        )
        policy = jax.nn.softmax(policy_logits)
        prior = self.root_prior(tree.root_noise, policy, root_metadata.action_mask)
        if self.playout_cap is not None:
            # a fast search's root takes no noise. The barrier keeps XLA from fusing the noise's
            # mix into the select, which it does differently in `evaluate` and in a search a cycle
            # at a time, rounding the prior differently
            prior = jnp.where(tree.fast, policy, jax.lax.optimization_barrier(prior))
        root_node = self.update_root_node(
            tree.data_at(tree.ROOT_INDEX), prior, value, root_embedding
        )
        # the network's evaluation, before the prior's noise and the search's averaging
        tree = replace(
            tree,
            root_logits=jnp.where(write, policy_logits, tree.root_logits),
            root_value=jnp.where(write, value, tree.root_value),
        )
        return tree.set_root(root_node, write=write)

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

    def iterate(
        self,
        key: jax.Array,
        tree: MCTSTree,
        params: Any,
        env_step_fn: EnvStepFn,
        write: ArrayLike = True,
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
            write: (optional) whether to; if not, the tree is left as it is (see `expand`)

        Returns:
            MCTSTree: updated MCTSTree
        """
        traverse_key, eval_key, key = jax.random.split(key, 3)
        leaf = self.find_leaf(traverse_key, tree, env_step_fn)
        policy_logits, value = self.eval_fn(leaf.embedding, params, eval_key)
        return self.expand(key, tree, leaf, policy_logits, value, write=write)

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

        # each step, unless the traversal is already at its leaf:
        # - get the index of the child node connected to the chosen action
        # - choose the action to take from the child node
        def step(state: TraversalState) -> TraversalState:
            descend = cond_fn(state)
            node_idx = tree.edge_map[state.parent, state.action]
            action = select(node_idx)
            return TraversalState(
                parent=jnp.where(descend, node_idx, state.parent),
                action=jnp.where(descend, action, state.action),
            )

        # `traversal_steps` steps between checks of the loop's condition: on GPU each check waits
        # on the host, and under vmap the loop runs until the deepest traversal in the batch is done
        def body_fn(state: TraversalState) -> TraversalState:
            for _ in range(self.traversal_steps):
                state = step(state)
            return state

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
            Tuple[jax.Array, jax.Array]: sampled action, the policy target (see `policy_target`)
        """
        target = self.policy_target(tree, self.discount)
        policy_weights = VisitCounts()(tree, self.discount)

        # zero temperature == argmax
        if self.temperature == 0:
            # of the most visited actions, one at random: equal visits give exactly equal weights
            noise = jax.random.uniform(key, shape=policy_weights.shape)
            action = jnp.argmax(
                jnp.where(policy_weights == policy_weights.max(), noise, -1.0)
            )
            return action, target

        # apply temperature
        policy_weights_t = policy_weights ** (1 / self.temperature)
        # re-normalize
        policy_weights_t /= policy_weights_t.sum()
        # sample action
        action = jax.random.choice(key, policy_weights_t.shape[-1], p=policy_weights_t)
        return action, target

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
        if self.learned_stop is not None:
            state = self.label_checks(state, active)
        return state.get_subtree(
            action, reset=jnp.logical_or(reset, not self.persist_tree), active=active
        )

    def checks(self) -> int:
        """With a `learned_stop`, the most checks a search it applies to makes."""
        assert self.learned_stop is not None
        cap = self.num_iterations
        if self.learned_stop.searches != "all":
            assert self.playout_cap is not None
            cap = self.playout_cap.fast_iterations
        return (cap + 1) // self.learned_stop.interval

    def record_check(self, state: MCTSTree, cycles: ArrayLike) -> MCTSTree:
        """With a `learned_stop`, the root's statistics after the search's `cycles`-th cycle, if
        it's one of the rule's checks (every `interval`) and the search labels the rule."""
        rule = self.learned_stop
        if rule is None:
            return state
        stopper = state.stopper
        cycles = jnp.asarray(cycles)
        index = cycles // rule.interval - 1
        at = (cycles % rule.interval == 0) & (index >= 0) & (index < self.checks())
        at = at & stopper.label & stopper.applies
        index = jnp.clip(index, 0, self.checks() - 1)
        features = learned_stop.root_features(
            state, cycles, self.iterations(state), self.discount
        )
        pruned = (
            self.pruned(state, cycles, stopper.greedy)
            if self.smart_pruning is not None
            else jnp.array(False)
        )

        def put(array, value):
            return array.at[index].set(jnp.where(at, value, array[index]))

        return replace(
            state,
            stopper=replace(
                stopper,
                features=put(stopper.features, features),
                top=put(stopper.top, learned_stop.most_visited(state)),
                pruned=put(stopper.pruned, pruned),
                recorded=put(stopper.recorded, True),
            ),
        )

    def label_checks(self, state: MCTSTree, active: ArrayLike) -> MCTSTree:
        """With a `learned_stop`, the search in `state` done (if `active`): if it labels the
        rule, its checks before its end become examples (see `learned_stop.write_examples`)."""
        assert self.learned_stop is not None
        stopper = state.stopper
        end = self.iterations(state) + 1
        check_cycles = (jnp.arange(self.checks()) + 1) * self.learned_stop.interval
        valid = (
            stopper.recorded
            & (check_cycles < end)
            & stopper.label
            & stopper.applies
            & jnp.asarray(active)
        )
        gain = learned_stop.gains(state, stopper.top, self.discount)
        return replace(
            state, stopper=learned_stop.write_examples(stopper, gain, valid, end)
        )

    def adapt(self, state: MCTSTree, key: jax.Array) -> tuple[MCTSTree, dict]:
        """With a `learned_stop`, fits its network to the searches that labelled it since the
        last fit, sets its threshold, and starts collecting anew (see
        `learned_stop.fit_and_calibrate`). Without, nothing.

        Args: see `Evaluator.adapt`.
        """
        if self.learned_stop is None:
            return state, {}
        stopper = state.stopper
        params, threshold, metrics = learned_stop.fit_and_calibrate(
            key, self.learned_stop, stopper
        )
        envs = stopper.ex_count.shape
        stopper = replace(
            stopper,
            params=jax.tree.map(
                lambda x: jnp.broadcast_to(x, envs + x.shape).copy(), params
            ),
            threshold=jnp.full(envs, threshold, dtype=jnp.float32),
            ex_count=jnp.zeros_like(stopper.ex_count),
            ex_valid=jnp.zeros_like(stopper.ex_valid),
        )
        return replace(state, stopper=stopper), metrics

    def init(self, template_embedding: Any, *args, **kwargs) -> MCTSTree:  # pylint: disable=arguments-differ
        """Initializes the internal state of the MCTS evaluator.

        Args:
            template_embedding: template environment state embedding
                - not stored, just used to initialize data structures to the correct shape

        Returns:
            MCTSTree: initialized MCTSTree
        """
        tree = init_mcts_tree(
            self.max_nodes,
            self.branching_factor,
            self.new_node(
                policy=jnp.zeros((self.branching_factor,)),
                value=0.0,
                embedding=template_embedding,
                terminated=False,
            ),
        )
        if self.learned_stop is None:
            return tree
        return replace(
            tree, stopper=learned_stop.init_stopper(self.learned_stop, self.checks())
        )
