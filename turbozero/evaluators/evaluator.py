from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from turbozero.pytree_utils import where_tree
from turbozero.types import EnvStepFn, StepMetadata


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EvalOutput:
    """What an evaluation returns: the evaluator's updated state, the action
    to take, and the policy's weight for each action.
    """

    eval_state: Any
    action: jax.Array
    policy_weights: jax.Array


class Evaluator:
    """Picks an action for an environment state, and weights every action (the
    policy), keeping a state of its own (e.g. a search tree) across moves.

    `evaluate` does a move's work in one call. Self-play instead runs a search
    a cycle at a time, so that environments at different points of their
    searches evaluate the network together: `start_search`, then
    `search_cycles()` calls of `search_cycle` (or more, while self-play waits to
    play its moves), then `search_output`, all with the move's key. An evaluator
    that doesn't search does all of it in `search_output`.
    """

    def __init__(self, discount: float):
        """`discount` multiplies the value of the next state: negative in
        two-player games (the next state's value is the opponent's), positive in
        single-player ones.
        """
        self.discount = discount

    def init(self, template_embedding: Any) -> Any:
        """The evaluator's state before an episode, for environment states shaped
        like `template_embedding`.
        """
        raise NotImplementedError()

    def reset(self, state: Any) -> Any:
        """The state for a new episode."""
        return state

    def evaluate(
        self,
        key: jax.Array,
        eval_state: Any,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
    ) -> EvalOutput:
        """Evaluates `env_state`, whose metadata is `root_metadata`, with the
        network's `params`.
        """
        raise NotImplementedError()

    def search_cycles(self) -> int:
        """The cycles a move's search takes: self-play's unit of compute."""
        return 1

    def search_done(self, state: Any, cycles: jax.Array) -> jax.Array:
        """Returns whether the search in `state` is done after `cycles` cycles,
        so self-play can play its move: after `search_cycles()`, unless the
        evaluator's searches vary in length.
        """
        return cycles >= self.search_cycles()

    def trains_on(self, state: Any) -> jax.Array:
        """Returns whether self-play trains on the position searched in `state`
        (stores it in the replay buffer): always, unless the evaluator says
        otherwise (e.g. MCTS's fast searches).
        """
        return jnp.array(True)

    def start_search(self, key: jax.Array, state: Any, active: ArrayLike = True) -> Any:
        """What a search does once before its cycles rather than in each (under
        vmap, all of a cycle is computed in every cycle); `state` as it is
        unless `active`.
        """
        return state

    def search_cycle(
        self,
        key: jax.Array,
        state: Any,
        cycle: jax.Array,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
    ) -> Any:
        """Runs cycle `cycle` (counting from 0, possibly past `search_cycles()`)
        of the search for a move.
        """
        return state

    def search_output(
        self,
        key: jax.Array,
        state: Any,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
    ) -> EvalOutput:
        """The output of a search after its cycles, as `evaluate`'s."""
        return self.evaluate(key, state, env_state, root_metadata, params, env_step_fn)

    def step(self, state: Any, action: jax.Array) -> Any:
        """The state after `action` is played."""
        return state

    def next_state(
        self, state: Any, action: jax.Array, reset: ArrayLike, active: ArrayLike = True
    ) -> Any:
        """The state for the next move: `step` with `action`, or `reset` if
        `reset` (e.g. at the end of an episode); `state` as it is unless `active`.
        """
        moved = jax.lax.cond(reset, self.reset, lambda s: self.step(s, action), state)
        return where_tree(active, moved, state)

    def get_value(self, state: Any) -> jax.Array:
        """The value estimate of the state evaluated (the search's root)."""
        raise NotImplementedError()

    def get_config(self) -> dict:
        """The evaluator's settings, as checkpoints record them."""
        return {"discount": self.discount}
