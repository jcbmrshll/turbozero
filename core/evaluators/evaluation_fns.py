
from typing import Callable, Tuple

import chex
import flax
import jax


def make_nn_eval_fn(
    nn: flax.linen.Module,
    state_to_nn_input_fn: Callable[[chex.ArrayTree], chex.Array]
) -> Callable[[chex.ArrayTree, chex.ArrayTree, chex.PRNGKey], Tuple[chex.Array, chex.Array]]:
    """Creates a leaf evaluation function using a neural network (state, params) -> (policy, value).

    Args:
        nn: The neural network module.
        state_to_nn_input_fn: A function that converts the state to the input format expected by the neural network.

    Returns:
        Callable: A function that evaluates the state using the neural network (state, params) -> (policy, value)
    """
    
    def eval_fn(state, params, *args):
        # get the policy and value from the neural network
        policy_logits, value = nn.apply(params, state_to_nn_input_fn(state)[None,...], train=False)
        # apply softmax to the policy logits
        return jax.nn.softmax(policy_logits, axis=-1).squeeze(0), value.squeeze()

    return eval_fn


def make_nn_eval_fn_no_params_callable(
    nn: Callable[[chex.Array], Tuple[chex.Array, chex.Array]],
    state_to_nn_input_fn: Callable[[chex.ArrayTree], chex.Array]
) -> Callable[[chex.ArrayTree, chex.ArrayTree, chex.PRNGKey], Tuple[chex.Array, chex.Array]]:
    """Creates a leaf evaluation function that uses a stateless neural net evaluation function (state) -> (policy, value).

    Args:
        nn: The stateless evaluation function.
        state_to_nn_input_fn: A function that converts the state to the input format expected by the neural network

    Returns:
        Callable: A function that evaluates the state using the neural network (state) -> (policy, value)
    """

    def eval_fn(state, *args):
        # get the policy and value from the neural network
        policy_logits, value = nn(state_to_nn_input_fn(state)[None,...])
        # apply softmax to the policy logits
        return jax.nn.softmax(policy_logits, axis=-1).squeeze(0), value.squeeze()
            
    return eval_fn
