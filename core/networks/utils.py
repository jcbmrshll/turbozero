from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

# name of the vmapped axis networks are applied over;
# layers that compute statistics across the batch (e.g. eqx.nn.BatchNorm) should use it as their `axis_name`
BATCH_AXIS = "batch"


def apply_nn(
    nn: Callable, nn_state: eqx.nn.State | None, x: jax.Array
) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State | None]:
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
    return jax.vmap(nn, axis_name=BATCH_AXIS, in_axes=(0, None), out_axes=(0, None))(
        x, nn_state
    )


def freeze_batchnorm(nn: Any) -> Any:
    """`nn` with every BatchNorm in inference mode, so that it normalizes by its running statistics
    in training too and they stay as they are, as fine-tuning commonly does; its scales and offsets
    still train. (A network may also pick its compute dtype by its BatchNorm's mode: see
    `core.networks.azresnet.AZResnet`.)"""

    def is_batchnorm(x: Any) -> bool:
        return isinstance(x, eqx.nn.BatchNorm)

    return jax.tree.map(
        lambda x: (
            eqx.tree_at(lambda bn: bn.inference, x, True) if is_batchnorm(x) else x
        ),
        nn,
        is_leaf=is_batchnorm,
    )
