from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

from turbozero.networks.nn_utils import apply_nn
from turbozero.types import LeafEvalFn


def make_nn_leaf_eval_fn(
    nn: eqx.Module, state_to_nn_input_fn: Callable[[Any], jax.Array]
) -> LeafEvalFn:
    """MCTS's leaf evaluation with network `nn` (see `apply_nn`): (state, params,
    key) -> (policy_logits, value), where `params` is (nn_params, nn_state), the
    trained arrays to combine with `nn`'s structure.
    """
    static = eqx.filter(nn, eqx.is_inexact_array, inverse=True)

    def leaf_eval_fn(state, params, key):
        nn_params, nn_state = params
        nn = eqx.nn.inference_mode(eqx.combine(nn_params, static))
        (policy_logits, value), _ = apply_nn(
            nn, nn_state, state_to_nn_input_fn(state)[None, ...]
        )
        # the raw logits: MCTS applies the (masked) softmax
        return policy_logits.squeeze(0), value.squeeze()

    return leaf_eval_fn
