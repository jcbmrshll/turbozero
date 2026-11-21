from dataclasses import dataclass
from typing import Any

import jax
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
        """Creates the evaluator's initial state, for environment states shaped
        like `template_embedding`.
        """
        raise NotImplementedError()

    def reset(self, state: Any) -> Any:
        """Resets the state for a new episode. By default it's kept as it is."""
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
        """Picks the action to play in `env_state` and returns it with the
        policy weights and the updated state, doing a whole move's work in one
        call (see the class docstring for running it a cycle at a time).

        `root_metadata` is `env_state`'s metadata, `params` the network's, and
        `env_step_fn` lets a search step the environment.
        """
        raise NotImplementedError()

    def search_cycles(self) -> int:
        """Returns the number of cycles a move's search takes. Cycles are
        self-play's unit of compute: it plays a move once its search has run
        this many.
        """
        return 1

    def start_search(self, key: jax.Array, state: Any, active: ArrayLike = True) -> Any:
        """Does the work a search needs only once, before its cycles (e.g.
        drawing root noise).

        Under vmap every branch of a cycle is computed in every cycle, so work
        done in `search_cycle` only when the search starts would be repeated
        every cycle. States where `active` is False are left unchanged. By
        default there's nothing to do.
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
        """Picks the move once a search's cycles are done, returning what
        `evaluate` would. By default it calls `evaluate`, so an evaluator that
        doesn't search does all its work here.
        """
        return self.evaluate(key, state, env_state, root_metadata, params, env_step_fn)

    def step(self, state: Any, action: jax.Array) -> Any:
        """Moves the state on after `action` is played (e.g. a search keeps the
        action's subtree). By default it's kept as it is.
        """
        return state

    def next_state(
        self, state: Any, action: jax.Array, reset: ArrayLike, active: ArrayLike = True
    ) -> Any:
        """Moves the state on to the next move: does `step` with `action`, or
        `reset` if `reset` is True (e.g. at the end of an episode). States where
        `active` is False are left unchanged.
        """
        moved = jax.lax.cond(reset, self.reset, lambda s: self.step(s, action), state)
        return where_tree(active, moved, state)

    def get_value(self, state: Any) -> jax.Array:
        """Returns the value estimate of the last state evaluated (a search's
        root), for the player to move there.
        """
        raise NotImplementedError()

    def get_config(self) -> dict:
        """Returns the evaluator's settings, for checkpoints to record."""
        return {"discount": self.discount}
