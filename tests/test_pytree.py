import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.pytree import map_where


def step(tree: dict, by: jax.Array) -> dict:
    """An entry's update: reads every leaf, as an evaluator's `next_state` would."""
    return {"a": tree["a"] * 2 + by, "b": tree["b"][::-1] + tree["a"][0]}


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8])
@pytest.mark.parametrize(
    "active",
    [
        [True, False, True, True, False],
        [False] * 5,
        [True] * 5,
    ],
)
def test_map_where_maps_the_active_entries_only(size, active):
    tree = {
        "a": jnp.arange(15, dtype=jnp.float32).reshape(5, 3),
        "b": jnp.arange(10, dtype=jnp.float32).reshape(5, 2),
    }
    by = jnp.arange(5, dtype=jnp.float32)
    active = jnp.array(active)

    mapped = jax.jit(lambda *x: map_where(step, *x, size=size))(active, tree, by)

    expected = jax.tree.map(
        lambda new, old: np.where(active.reshape(-1, 1), new, old),
        jax.vmap(step)(tree, by),
        tree,
    )
    jax.tree.map(np.testing.assert_array_equal, mapped, expected)
