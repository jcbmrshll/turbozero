from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from core.evaluators.evaluator import EvalOutput, Evaluator
from core.types import StepMetadata


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RandomState:
    """RandomEvaluator has no state of its own; this placeholder keeps it a pytree of arrays."""

    placeholder: jax.Array


class RandomEvaluator(Evaluator):
    """Plays a uniformly random legal action: the weakest sensible baseline, which any
    trained agent should beat."""

    def __init__(self, discount: float = -1.0):
        """Initializes a RandomEvaluator.

        Args:
            discount: discount factor applied to future rewards (-1 for two-player zero-sum games)
        """
        super().__init__(discount=discount)

    def init(self, *args, **kwargs) -> RandomState:  # pylint: disable=unused-argument
        return RandomState(placeholder=jnp.zeros((), dtype=jnp.int32))

    def reset(self, state: RandomState) -> RandomState:
        return state

    def evaluate(
        self,
        key: jax.Array,
        eval_state: RandomState,
        env_state: Any,  # pylint: disable=unused-argument
        root_metadata: StepMetadata,
        **kwargs,
    ) -> EvalOutput:
        """Picks a legal action uniformly at random.

        Args:
            key: rng
            eval_state: internal state (unused)
            env_state: environment state (unused)
            root_metadata: metadata of the environment state, whose `action_mask` marks the legal actions

        Returns:
            EvalOutput: the action, and a uniform policy over the legal actions
        """
        mask = root_metadata.action_mask
        policy_weights = mask / mask.sum()
        action = jax.random.choice(key, mask.shape[-1], p=policy_weights)
        return EvalOutput(
            eval_state=eval_state, action=action, policy_weights=policy_weights
        )

    def get_value(self, state: RandomState) -> jax.Array:  # pylint: disable=unused-argument
        # it has no idea who is winning
        return jnp.array(0.0)
