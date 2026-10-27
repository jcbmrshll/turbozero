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
