from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

# name of the vmapped axis networks are applied over;
# layers that compute statistics across the batch (e.g. eqx.nn.BatchNorm) should use it as their `axis_name`
BATCH_AXIS = "batch"


def apply_nn(
    nn: Callable, nn_state: eqx.nn.State | None, x: jax.Array
) -> tuple[tuple[Any, ...], eqx.nn.State | None]:
    """Applies a neural network to a batch of inputs.

    As is the convention in equinox, networks act on a single (unbatched) input:
    - stateless networks: `nn(x) -> (policy_logits, value)`
    - stateful networks (e.g. with BatchNorm), created with `eqx.nn.make_with_state`:
        `nn(x, nn_state) -> ((policy_logits, value), nn_state)`

    Args:
        nn: the neural network
        nn_state: state of the network, None for stateless networks
        x: batch of inputs

    Returns:
        Tuple[Tuple, Optional[eqx.nn.State]]: ((policy_logits, value), nn_state), the batched
            outputs (whatever the network returns, e.g. more with `AZResnet`'s `train_outputs`)
            and the updated network state
    """
    if nn_state is None:
        return jax.vmap(nn)(x), None
    return jax.vmap(nn, axis_name=BATCH_AXIS, in_axes=(0, None), out_axes=(0, None))(
        x, nn_state
    )


def outcome_value(logits: jax.Array, scores: Sequence[float] | jax.Array) -> jax.Array:
    """The value of a distribution over a game's final scores, `logits` (..., scores) a logit for
    each of `scores`: the probability of a positive score less that of a negative one, the expected
    result of a game won on a positive score."""
    signs = jnp.sign(jnp.asarray(scores, jnp.float32))
    return jax.nn.softmax(logits.astype(jnp.float32), axis=-1) @ signs
