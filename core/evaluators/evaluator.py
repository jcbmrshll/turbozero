
from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EvalOutput:
    """Output of an evaluation.
    - `eval_state`: The updated internal state of the Evaluator.
    - `action`: The action to take.
    - `policy_weights`: The policy weights assigned to each action.
    """
    eval_state: Any
    action: int
    policy_weights: jax.Array


class Evaluator:
    """Base class for Evaluators.
    An Evaluator *evaluates* an environment state, and returns an action to take, as well as a 'policy', assigning a weight to each action.
    Evaluators may maintain an internal state, which is updated by the `step` method.
    """

    def __init__(self, discount: float, *args, **kwargs):  # pylint: disable=unused-argument
        """Initializes an Evaluator.

        Args:
        - `discount`: The discount factor applied to future rewards/value estimates.
        """
        self.discount = discount


    def init(self, *args, **kwargs) -> Any:
        """Initializes the internal state of the Evaluator."""
        raise NotImplementedError()


    def init_batched(self, batch_size: int, *args, **kwargs) -> Any:
        """Initializes the internal state of the Evaluator across a batch dimension."""
        tree = self.init(*args, **kwargs)
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), tree)


    def reset(self, state: Any) -> Any:
        """Resets the internal state of the Evaluator."""
        raise NotImplementedError()


    def evaluate(self, key: jax.Array, eval_state: Any, env_state: Any, **kwargs) -> EvalOutput:
        """Evaluates the environment state.

        Args:
        - `key`: rng
        - `eval_state`: The internal state of the Evaluator.
        - `env_state`: The environment state to evaluate.

        Returns:
        - `EvalOutput`: The output of the evaluation.
            - `eval_state`: The updated internal state of the Evaluator.
            - `action`: The action to take.
            - `policy_weights`: The policy weights assigned to each action.
        """
        raise NotImplementedError()


    def step(self, state: Any, action: jax.Array) -> Any:  # pylint: disable=unused-argument
        """Updates the internal state of the Evaluator.

        Args:
        - `state`: The internal state of the Evaluator.
        - `action`: The action taken in the environment.

        Returns:
        - (pytree): The updated internal state of the Evaluator.
        """
        return state


    def get_value(self, state: Any) -> jax.Array:
        """Extracts the state value estimate (for the current/root environment state) from the internal state of the Evaluator.

        Args:
        - `state`: The internal state of the Evaluator.

        Returns:
        - `jax.Array`: The value estimate.
        """
        raise NotImplementedError()


    def get_config(self) -> Dict:
        """Returns the configuration of the Evaluator. Used for logging."""
        return {'discount': self.discount}
