from collections.abc import Callable

import equinox as eqx
import jax

# the vmapped axis networks are applied over: layers that compute statistics
# across the batch (e.g. eqx.nn.BatchNorm) take it as their `axis_name`
BATCH_AXIS = "batch"


def apply_nn(
    nn: Callable, nn_state: eqx.nn.State | None, x: jax.Array
) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State | None]:
    """((policy_logits, value), nn_state) of `nn` for a batch of inputs `x`.

    As equinox networks do, `nn` acts on one input: `nn(x) -> (policy_logits,
    value)` for a stateless network (`nn_state` None), and `nn(x, nn_state) ->
    ((policy_logits, value), nn_state)` for a stateful one (e.g. with
    BatchNorm), made with `eqx.nn.make_with_state`.
    """
    if nn_state is None:
        return jax.vmap(nn)(x), None
    return jax.vmap(nn, axis_name=BATCH_AXIS, in_axes=(0, None), out_axes=(0, None))(
        x, nn_state
    )
