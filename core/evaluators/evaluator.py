
from typing import Dict

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


@dataclass(frozen=True)
class EvalOutput:
    """Output of an evaluation.

    Attributes:
        eval_state: The updated internal state of the Evaluator.
        action: The action to take.
        policy_weights: The policy weights assigned to each action.
    """
    eval_state: chex.ArrayTree
    action: int
    policy_weights: chex.Array


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


    def init(self, *args, **kwargs) -> chex.ArrayTree:
        """Initializes the internal state of the Evaluator."""
        raise NotImplementedError()


    def init_batched(self, batch_size: int, *args, **kwargs) -> chex.ArrayTree:
        """Initializes the internal state of the Evaluator across a batch dimension."""
        tree = self.init(*args, **kwargs)
        return jax.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), tree)


    def reset(self, state: chex.ArrayTree) -> chex.ArrayTree:
        """Resets the internal state of the Evaluator."""
        raise NotImplementedError()


    def evaluate(self, key: chex.PRNGKey, eval_state: chex.ArrayTree, env_state: chex.ArrayTree, **kwargs) -> EvalOutput:
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


    def step(self, state: chex.ArrayTree, action: chex.Array) -> chex.ArrayTree:  # pylint: disable=unused-argument
        """Updates the internal state of the Evaluator.

        Args:
            state: The internal state of the Evaluator.
            action: The action taken in the environment.

        Returns:
            chex.ArrayTree: The updated internal state of the Evaluator.
        """
        return state


    def get_value(self, state: chex.ArrayTree) -> chex.Array:
        """Extracts the state value estimate (for the current/root environment state) from the internal state of the Evaluator.

        Args:
            state: The internal state of the Evaluator.

        Returns:
            chex.Array: The value estimate.
        """
        raise NotImplementedError()


    def get_config(self) -> Dict:
        """Returns the configuration of the Evaluator. Used for logging."""
        return {'discount': self.discount}
