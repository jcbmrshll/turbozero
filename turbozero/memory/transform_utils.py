from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from turbozero.memory.replay_memory import BaseExperience

# a data transform: maps an experience, without batch dimensions, to another
# that's as valid a training sample, its observation transformed with its
# policy target and legal actions to match. The value targets (`reward`,
# `search_value`) and `cur_player_id` stay as they are.
DataTransformFn = Callable[[BaseExperience], BaseExperience]


def transform_randomly(
    key: jax.Array,
    experiences: BaseExperience,
    transform_fns: Sequence[DataTransformFn],
) -> BaseExperience:
    """Maps each of a batch of experiences through a transform drawn uniformly
    at random, independently for each, from the identity and `transform_fns`.
    """
    if not transform_fns:
        return experiences
    branches = [lambda e: e, *transform_fns]
    choice = jax.random.randint(
        key, experiences.reward.shape[:1], 0, len(branches), dtype=jnp.int32
    )
    return jax.vmap(lambda c, e: jax.lax.switch(c, branches, e))(choice, experiences)
