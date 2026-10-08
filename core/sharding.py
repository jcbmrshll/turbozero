from collections.abc import Callable
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# name of the mesh axis that batches (self-play environments, test episodes, training minibatches) are split along
AXIS = "d"


def make_mesh(num_devices: int) -> Mesh:
    """Creates a 1-D mesh over the first `num_devices` devices, with axis `AXIS`.

    Args:
        num_devices: number of devices to use

    Returns:
        Mesh: the mesh
    """
    return Mesh(jax.devices()[:num_devices], (AXIS,))


def shard(data: Any, mesh: Mesh) -> Any:
    """Splits each array in a data structure across the devices of `mesh` along its first axis.

    Args:
        data: pytree of arrays, the first axis of each must be divisible by the number of devices
        mesh: mesh to shard across

    Returns:
        pytree: `data`, sharded along `AXIS`
    """
    return jax.device_put(data, NamedSharding(mesh, PartitionSpec(AXIS)))


def replicate(data: Any, mesh: Mesh) -> Any:
    """Places a copy of each array in a data structure on every device of `mesh`.

    Args:
        data: pytree of arrays
        mesh: mesh to replicate across

    Returns:
        pytree: `data`, replicated across `mesh`
    """
    return jax.device_put(data, NamedSharding(mesh, PartitionSpec()))


def shard_map(f: Callable, mesh: Mesh, in_specs: Any, out_specs: Any) -> Callable:
    """`jax.shard_map` over `mesh`, for per-device code written like the body of a `jax.pmap`.

    Doesn't check that outputs specified as replicated (`PartitionSpec()`) are the same on every device
    (`check_vma=False`), like `jax.pmap`, so code that runs per device doesn't need varying/invariant annotations.
    The checks also reject code that this library runs per device:
    - loops whose carry starts as a constant and becomes per-device (e.g. the MCTS loops)
    - collectives over vmapped axes (e.g. eqx.nn.BatchNorm's pmean over the batch) fail an assertion in JAX 0.11

    Args:
        f: function to run on every device
        mesh: mesh to run on
        in_specs: partition specs of the inputs
        out_specs: partition specs of the outputs

    Returns:
        Callable: `f`, mapped over the devices of `mesh`
    """
    return jax.shard_map(
        f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )
