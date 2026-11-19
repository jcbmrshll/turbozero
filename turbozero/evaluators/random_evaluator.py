from typing import Any

import jax
import jax.numpy as jnp

from turbozero.evaluators.evaluator import EvalOutput, Evaluator
from turbozero.types import EnvStepFn, StepMetadata


class RandomEvaluator(Evaluator):
    """Plays a uniformly random legal action: the weakest sensible baseline,
    which any trained agent should beat.
    """

    def __init__(self, discount: float = -1.0):
        super().__init__(discount)

    def init(self, template_embedding: Any) -> jax.Array:
        # no state of its own: a placeholder keeps it a pytree of arrays
        return jnp.zeros((), dtype=jnp.int32)

    def evaluate(
        self,
        key: jax.Array,
        eval_state: jax.Array,
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
    ) -> EvalOutput:
        mask = root_metadata.action_mask
        policy_weights = mask / mask.sum()
        action = jax.random.choice(key, mask.shape[-1], p=policy_weights)
        return EvalOutput(
            eval_state=eval_state, action=action, policy_weights=policy_weights
        )

    def get_value(self, state: jax.Array) -> jax.Array:
        # it has no idea who is winning
        return jnp.array(0.0)
