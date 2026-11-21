from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def where_tree(condition: ArrayLike, new: Any, old: Any) -> Any:
    """Selects between two pytrees of the same structure leaf by leaf, as
    `jnp.where(condition, new, old)`. A leaf that is the same array in both
    (e.g. one that `dataclasses.replace` left as it was) is kept without a
    select, which under vmap would read and write the whole leaf.
    """
    return jax.tree.map(
        lambda n, o: o if n is o else jnp.where(condition, n, o), new, old
    )


def map_where(
    fn: Callable[..., Any], active: jax.Array, tree: Any, *args: Any, size: int
) -> Any:
    """Applies `fn` to the entries of `tree` where `active` and leaves the
    rest unchanged, as `where_tree(active, jax.vmap(fn)(tree, *args), tree)`
    would, but computing `fn` only for the active entries (leaves have a
    leading batch dimension).

    The active entries are gathered `size` at a time, mapped and written back
    in place, and the others aren't read at all; under vmap, or with a select,
    `fn` would read and write every entry however few are active. Each gather
    is a loop trip and costs `size` entries' work whether or not that many are
    left.
    """
    batch = active.shape[0]
    size = min(size, batch)
    trips = -(-batch // size)
    # the active entries' indices, then `batch` (out of bounds) for the rest
    # of the trips
    order = jnp.nonzero(active, size=trips * size, fill_value=batch)[0]
    count = jnp.sum(active)

    def trip(carry: tuple[jax.Array, Any]) -> tuple[jax.Array, Any]:
        i, tree = carry
        index = jax.lax.dynamic_slice(order, (i * size,), (size,))
        # every write depends on the reads (a gathered entry goes into what's
        # written), so XLA writes in place rather than copying the arrays to
        # read from; the padding entries read the last one and write nowhere
        taken = jax.tree.map(lambda x: x[jnp.minimum(index, batch - 1)], (tree, *args))
        mapped = jax.vmap(fn)(*taken)
        tree = jax.tree.map(lambda x, y: x.at[index].set(y, mode="drop"), tree, mapped)
        return i + 1, tree

    _, tree = jax.lax.while_loop(
        lambda carry: carry[0] * size < count, trip, (jnp.array(0), tree)
    )
    return tree
