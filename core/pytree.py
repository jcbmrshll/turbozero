from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def where_tree(condition: ArrayLike, new: Any, old: Any) -> Any:
    """`jnp.where(condition, new, old)`, leaf by leaf, for two pytrees of the same structure.
    Leaves that are the same array in both (e.g. those `dataclasses.replace` left as they were)
    are kept, without a select: under vmap, it would read and write the whole leaf."""
    return jax.tree.map(
        lambda n, o: o if n is o else jnp.where(condition, n, o), new, old
    )


def map_where(
    fn: Callable[..., Any], active: jax.Array, tree: Any, *args: Any, size: int
) -> Any:
    """`where_tree(active, jax.vmap(fn)(tree, *args), tree)` for a batch of `tree`s (and of
    `args`), where `fn` maps a `tree` (with its `args`) to one of the same structure, computed
    only for the entries where `active`: they're gathered `size` at a time, mapped through `fn`
    and written back in place, while the others aren't read at all. Under vmap, or with a
    select, `fn` would read and write every entry, however few are active.

    Args:
        fn: maps an entry of `tree`, and of each of `args`, to the entry's new value
        active: (batch,) the entries to map
        tree: pytree whose leaves have a leading batch dimension
        args: (optional) pytrees whose leaves have the same leading batch dimension
        size: entries mapped at a time (each time costs a loop trip, and `size` entries' work
            whether or not that many are left)

    Returns:
        Any: `tree` with its active entries mapped
    """
    batch = active.shape[0]
    size = max(1, min(size, batch))
    trips = -(-batch // size)
    # the active entries' indices, then `batch` (out of bounds) for the rest of the trips
    order = jnp.nonzero(active, size=trips * size, fill_value=batch)[0]
    count = jnp.sum(active)

    def trip(carry: tuple[jax.Array, Any]) -> tuple[jax.Array, Any]:
        i, tree = carry
        index = jax.lax.dynamic_slice(order, (i * size,), (size,))
        # every write depends on the reads (a gathered entry goes into what's written), so XLA
        # writes in place rather than copying the arrays to read from; the padding entries read
        # the last one and write nowhere
        taken = jax.tree.map(lambda x: x[jnp.minimum(index, batch - 1)], (tree, *args))
        mapped = jax.vmap(fn)(*taken)
        tree = jax.tree.map(lambda x, y: x.at[index].set(y, mode="drop"), tree, mapped)
        return i + 1, tree

    _, tree = jax.lax.while_loop(
        lambda carry: carry[0] * size < count, trip, (jnp.array(0), tree)
    )
    return tree
