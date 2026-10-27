from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from core.pytree import where_tree
from core.types import EnvStepFn, StepMetadata


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EvalOutput:
    """Output of an evaluation.

    Attributes:
        eval_state: The updated internal state of the Evaluator.
        action: The action to take.
        policy_weights: The policy weights assigned to each action.
    """

    eval_state: Any
    action: jax.Array
    policy_weights: jax.Array


class Evaluator:
    """Base class for Evaluators.

    An Evaluator *evaluates* an environment state, and returns an action to take, as well as a 'policy', assigning a weight to each action.
    Evaluators may maintain an internal state, which is updated by the `step` method.
    """

    def __init__(self, discount: float, *args, **kwargs):  # pylint: disable=unused-argument
        """Initializes an Evaluator.

        Args:
            discount: The discount factor applied to future rewards/value estimates.
        """
        self.discount = discount

    def init(self, *args, **kwargs) -> Any:
        """Initializes the internal state of the Evaluator."""
        raise NotImplementedError()

    def init_batched(self, batch_size: int, *args, **kwargs) -> Any:
        """Initializes the internal state of the Evaluator across a batch dimension."""
        tree = self.init(*args, **kwargs)
        return jax.tree.map(
            lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), tree
        )

    def reset(self, state: Any) -> Any:
        """Resets the internal state of the Evaluator."""
        raise NotImplementedError()

    def evaluate(
        self, key: jax.Array, eval_state: Any, env_state: Any, *args, **kwargs
    ) -> EvalOutput:
        """Evaluates the environment state.

        Args:
            key: rng
            eval_state: The internal state of the Evaluator.
            env_state: The environment state to evaluate.

        Returns:
            EvalOutput: The output of the evaluation.
                - eval_state: The updated internal state of the Evaluator.
                - action: The action to take.
                - policy_weights: The policy weights assigned to each action.
        """
        raise NotImplementedError()

    # A search can also run a cycle at a time, as self-play runs it (see
    # `core.training.train.Trainer.collect_steps`): environments at different points of their
    # searches then make their network evaluations together, a cycle at a time. A move's search is
    # `search_cycle` until `search_done` and its move is played (self-play checks for those every
    # few cycles, and a search done in between goes on), then `search_output`, all with the move's
    # key. An evaluator that doesn't search does all of it in `search_output`.

    def search_cycles(self) -> int:
        """The most cycles a move's search takes: self-play's unit of compute."""
        return 1

    def search_done(self, state: Any, cycles: jax.Array) -> jax.Array:  # pylint: disable=unused-argument
        """Whether the search in `state`, which has run `cycles` cycles, is done.

        Args:
            state: The internal state of the Evaluator.
            cycles: the cycles the search has run so far

        Returns:
            jax.Array: whether to stop searching and play a move
        """
        return jnp.array(True)

    def search_cycle(  # pylint: disable=unused-argument
        self,
        key: jax.Array,
        state: Any,
        cycle: jax.Array,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
    ) -> Any:
        """Runs cycle `cycle` (counting from 0) of the search for a move.

        Args:
            key: rng, the move's (as `evaluate`'s)
            state: The internal state of the Evaluator.
            cycle: the cycles the search has run so far, possibly past `search_done`
            env_state: the environment state to evaluate
            root_metadata: its metadata
            params: nn parameters
            env_step_fn: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
            pytree: The updated internal state of the Evaluator.
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
        """The output of a search that `search_done`, as `evaluate`'s.

        Args:
            key: rng, the move's (as `evaluate`'s)
            state: The internal state of the Evaluator, after the search's cycles.
            env_state: the environment state evaluated
            root_metadata: its metadata
            params: nn parameters
            env_step_fn: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
            EvalOutput: as `evaluate`'s
        """
        return self.evaluate(
            key=key,
            eval_state=state,
            env_state=env_state,
            root_metadata=root_metadata,
            params=params,
            env_step_fn=env_step_fn,
        )

    def step(self, state: Any, action: jax.Array) -> Any:  # pylint: disable=unused-argument
        """Updates the internal state of the Evaluator.

        Args:
            state: The internal state of the Evaluator.
            action: The action taken in the environment.

        Returns:
            pytree: The updated internal state of the Evaluator.
        """
        return state

    def next_state(
        self,
        state: Any,
        action: jax.Array,
        reset: ArrayLike,
        active: ArrayLike = True,
    ) -> Any:
        """The internal state of the Evaluator for the next move: `step` with `action`, or `reset`
        if `reset` (e.g. at the end of an episode); `state` as it is unless `active`.

        Args:
            state: The internal state of the Evaluator.
            action: The action taken in the environment.
            reset: whether to reset instead
            active: (optional) whether to move on at all

        Returns:
            pytree: The internal state for the next move.
        """
        moved = jax.lax.cond(reset, self.reset, lambda s: self.step(s, action), state)
        return where_tree(active, moved, state)

    def get_value(self, state: Any) -> jax.Array:
        """Extracts the state value estimate (for the current/root environment state) from the internal state of the Evaluator.

        Args:
            state: The internal state of the Evaluator.

        Returns:
            jax.Array: The value estimate.
        """
        raise NotImplementedError()

    def search_record(self, state: Any) -> Any:  # pylint: disable=unused-argument
        """What the evaluation that left `state` explored, for the monitor to show (see
        `core.evaluators.mcts.state.SearchRecord`), or None for an evaluator that doesn't search.

        Args:
            state: The internal state of the Evaluator, as an evaluation left it (before `step`).

        Returns:
            pytree: The record, or None.
        """
        return None

    def get_config(self) -> dict:
        """Returns the configuration of the Evaluator. Used for logging."""
        return {"discount": self.discount}
