from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

from core.networks.utils import apply_nn


def make_nn_eval_fn(
    nn: eqx.Module,
    state_to_nn_input_fn: Callable[[Any], jax.Array]
) -> Callable[[Any, Any, jax.Array], tuple[jax.Array, jax.Array]]:
    """Creates a leaf evaluation function using a neural network (state, params) -> (policy_logits, value).

    Args:
        nn: The neural network (an equinox module, see core.networks.utils.apply_nn).
            Only its structure is used, the parameters it is evaluated with are passed as `params`.
        state_to_nn_input_fn: A function that converts the state to the input format expected by the neural network.

    Returns:
        Callable: A function that evaluates the state using the neural network (state, params) -> (policy_logits, value)
            - `params` is (nn_params, nn_state), as returned by core.training.train.extract_params
    """
    static = eqx.filter(nn, eqx.is_inexact_array, inverse=True)

    def eval_fn(state, params, *args):
        nn_params, nn_state = params
        # get the policy and value from the neural network
        nn = eqx.nn.inference_mode(eqx.combine(nn_params, static))
        (policy_logits, value), _ = apply_nn(nn, nn_state, state_to_nn_input_fn(state)[None,...])
        # return the raw policy logits, MCTS applies the (masked) softmax
        return policy_logits.squeeze(0), value.squeeze()

    return eval_fn


def make_nn_eval_fn_no_params_callable(
    nn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    state_to_nn_input_fn: Callable[[Any], jax.Array]
) -> Callable[[Any, Any, jax.Array], tuple[jax.Array, jax.Array]]:
    """Creates a leaf evaluation function that uses a stateless neural net evaluation function (state) -> (policy, value).

    Args:
        nn: The stateless evaluation function.
        state_to_nn_input_fn: A function that converts the state to the input format expected by the neural network

    Returns:
        Callable: A function that evaluates the state using the neural network (state) -> (policy_logits, value)
    """

    def eval_fn(state, *args):
        # get the policy and value from the neural network
        policy_logits, value = nn(state_to_nn_input_fn(state)[None,...])
        # return the raw policy logits, MCTS applies the (masked) softmax
        return policy_logits.squeeze(0), value.squeeze()
            
    return eval_fn
