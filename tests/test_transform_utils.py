from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from turbozero.memory.replay_memory import BaseExperience
from turbozero.memory.transform_utils import transform_randomly


def shift(k):
    """A data transform that adds 10**6 * k to the observation, to show which was drawn, and
    rolls the policy's actions k places."""

    def transform_fn(e):
        return replace(
            e,
            observation_nn=e.observation_nn + 10**6 * k,
            policy_weights=jnp.roll(e.policy_weights, k),
            policy_mask=jnp.roll(e.policy_mask, k),
        )

    return transform_fn


def test_transform_randomly_draws_a_transform_for_each_sample_uniformly():
    n = 4000
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    experiences = BaseExperience(
        reward=jax.random.normal(keys[0], (n, 2)),
        policy_weights=jax.random.uniform(keys[1], (n, 5)),
        policy_mask=jax.random.bernoulli(keys[2], 0.5, (n, 5)),
        observation_nn=jnp.arange(n, dtype=jnp.int32),
        cur_player_id=jnp.arange(n, dtype=jnp.int32) % 2,
        search_value=jnp.linspace(-1.0, 1.0, n),
    )

    out = jax.jit(transform_randomly, static_argnums=2)(
        jax.random.PRNGKey(1), experiences, (shift(1), shift(2), shift(3))
    )

    drawn = np.asarray(out.observation_nn) // 10**6
    # the identity, and each of the 3 transforms, about a quarter of the time each
    np.testing.assert_allclose(np.bincount(drawn) / n, 0.25, atol=0.03)
    # each sample is its stored experience's exact image under the drawn transform
    np.testing.assert_array_equal(out.observation_nn - 10**6 * drawn, np.arange(n))
    # rolled k places: action j holds what action j - k did
    rolled = (np.arange(5) - drawn[:, None]) % 5
    for field in ("policy_weights", "policy_mask"):
        np.testing.assert_array_equal(
            getattr(out, field),
            np.take_along_axis(np.asarray(getattr(experiences, field)), rolled, axis=1),
        )
    # the value targets and the player to move are left as they are
    for field in ("reward", "search_value", "cur_player_id"):
        np.testing.assert_array_equal(getattr(out, field), getattr(experiences, field))
    # without transforms, the experiences are left as they are
    same = transform_randomly(jax.random.PRNGKey(1), experiences, ())
    assert same is experiences
