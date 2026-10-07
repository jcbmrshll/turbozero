from typing import Callable, Optional, Tuple

import equinox as eqx
import jax

# name of the vmapped axis networks are applied over;
# layers that compute statistics across the batch (e.g. eqx.nn.BatchNorm) should use it as their `axis_name`
BATCH_AXIS = 'batch'


def apply_nn(
    nn: Callable,
    nn_state: Optional[eqx.nn.State],
    x: jax.Array
) -> Tuple[Tuple[jax.Array, jax.Array], Optional[eqx.nn.State]]:
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
        Tuple[Tuple[jax.Array, jax.Array], Optional[eqx.nn.State]]: ((policy_logits, value), nn_state),
            the batched outputs and the updated network state
    """
    if nn_state is None:
        return jax.vmap(nn)(x), None
    return jax.vmap(nn, axis_name=BATCH_AXIS, in_axes=(0, None), out_axes=(0, None))(x, nn_state)
