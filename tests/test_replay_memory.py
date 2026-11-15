from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from turbozero.memory.replay_memory import (
    BaseExperience,
    EpisodeReplayBuffer,
    transform_randomly,
)


def experience(obs_id):
    """An experience identified by its observation."""
    return BaseExperience(
        reward=jnp.zeros((2,)),
        policy_weights=jnp.full((3,), 1 / 3),
        policy_mask=jnp.ones((3,), dtype=jnp.bool_),
        observation_nn=jnp.array(obs_id, dtype=jnp.float32),
        cur_player_id=jnp.array(0, dtype=jnp.int32),
        search_value=jnp.array(0.0),
    )


def init_single(buffer):
    """Buffer state for a single environment (no batch dimension)."""
    return jax.tree.map(lambda x: x[0], buffer.init(1, experience(0)))


def add(buffer, state, obs_ids):
    for obs_id in obs_ids:
        state = buffer.add_experience(state, experience(obs_id))
    return state


def test_add_experience_and_assign_rewards():
    buffer = EpisodeReplayBuffer(capacity=5)
    state = add(buffer, init_single(buffer), [1, 2, 3])

    assert state.next_idx == 3
    np.testing.assert_array_equal(state.buffer.observation_nn, [1, 2, 3, 0, 0])
    np.testing.assert_array_equal(state.populated, [True, True, True, False, False])
    np.testing.assert_array_equal(state.has_reward, [False, False, False, True, True])

    state = buffer.assign_rewards(state, jnp.array([1.0, -1.0]))

    assert state.episode_start_idx == 3
    assert state.has_reward.all()
    np.testing.assert_array_equal(state.buffer.reward[:3], [[1.0, -1.0]] * 3)
    np.testing.assert_array_equal(state.buffer.reward[3:], 0.0)


def test_add_experience_that_is_not_valid_leaves_the_buffer_as_it_is():
    buffer = EpisodeReplayBuffer(capacity=3)
    # the next write wraps around, over a finished entry
    state = buffer.assign_rewards(
        add(buffer, init_single(buffer), [1, 2, 3]), jnp.ones(2)
    )

    after = buffer.add_experience(state, experience(4), valid=False)

    jax.tree.map(np.testing.assert_array_equal, after, state)


def test_assign_rewards_wraps_around_mid_episode():
    buffer = EpisodeReplayBuffer(capacity=5)
    state = add(buffer, init_single(buffer), [1, 2, 3])
    state = buffer.assign_rewards(state, jnp.array([1.0, -1.0]))

    # the second episode starts at index 3 and wraps around to overwrite indices 0 and 1
    state = add(buffer, state, [4, 5, 6, 7])
    assert state.next_idx == 2
    np.testing.assert_array_equal(state.buffer.observation_nn, [6, 7, 3, 4, 5])

    state = buffer.assign_rewards(state, jnp.array([-1.0, 1.0]))

    assert state.episode_start_idx == 2
    np.testing.assert_array_equal(
        state.buffer.reward,
        [[-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]],
    )


def test_truncate_wraps_around_mid_episode():
    buffer = EpisodeReplayBuffer(capacity=5)
    state = add(buffer, init_single(buffer), [1, 2, 3])
    state = buffer.assign_rewards(state, jnp.array([1.0, -1.0]))
    state = add(buffer, state, [4, 5, 6, 7])

    state = buffer.truncate(state)

    # only the finished first episode's surviving entry (index 2) is left
    assert state.next_idx == 3
    np.testing.assert_array_equal(state.populated, [False, False, True, False, False])
    assert state.has_reward.all()
    np.testing.assert_array_equal(state.buffer.reward[2], [1.0, -1.0])

    # the next episode is written from where the truncated one started
    state = add(buffer, state, [8])
    assert state.buffer.observation_nn[3] == 8


def test_sample_indices_with_replacement_from_fewer_entries():
    buffer = EpisodeReplayBuffer(capacity=8)
    mask = jnp.zeros((2, 8), dtype=bool).at[1, 3].set(True)

    indices = buffer.sample_indices(jax.random.PRNGKey(0), mask, 5, replace=True)

    np.testing.assert_array_equal(indices, [8 + 3] * 5)


def test_sampling_draws_only_finished_populated_entries():
    buffer = EpisodeReplayBuffer(capacity=4)
    # two environments, as the trainer stores them (batch, capacity, ...)
    states = [init_single(buffer), init_single(buffer)]
    # env 0: a finished episode (1, 2), then an episode in progress (3)
    states[0] = buffer.assign_rewards(
        add(buffer, states[0], [1, 2]), jnp.array([1.0, -1.0])
    )
    states[0] = add(buffer, states[0], [3])
    # env 1: a finished episode (4), then a truncated one (5, 6)
    states[1] = buffer.assign_rewards(
        add(buffer, states[1], [4]), jnp.array([-1.0, 1.0])
    )
    states[1] = buffer.truncate(add(buffer, states[1], [5, 6]))
    state = jax.tree.map(lambda *x: jnp.stack(x), *states)

    mask = buffer.sample_mask(state)
    for seed in range(10):
        indices = buffer.sample_indices(jax.random.PRNGKey(seed), mask, 3)
        sample = state.buffer.observation_nn.reshape(-1)[indices]
        assert sorted(sample.tolist()) == [1, 2, 4]


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


def sampleable(buffer, state):
    """Observations of the entries `sample_mask` marks, sorted."""
    mask = np.asarray(buffer.sample_mask(state))
    return sorted(np.asarray(state.buffer.observation_nn)[mask].tolist())


@pytest.mark.parametrize("capacity", [3, 5, 9])
def test_resize_keeps_the_newest_entries_oldest_first(capacity):
    buffer = EpisodeReplayBuffer(capacity=5)
    # wrapped around: index 0 holds 6, index 1 holds 7
    state = buffer.assign_rewards(
        add(buffer, init_single(buffer), [1, 2, 3, 4, 5, 6, 7]), jnp.array([1.0, -1.0])
    )

    resized = buffer.resize(state, capacity)

    assert resized.capacity == capacity
    assert sampleable(buffer, resized) == [3, 4, 5, 6, 7][-capacity:]
    # every array keeps its dtype (e.g. the boolean policy mask)
    assert [x.dtype for x in jax.tree.leaves(resized)] == [
        x.dtype for x in jax.tree.leaves(state)
    ]
    # and it goes on as a buffer of the new capacity
    resized = buffer.assign_rewards(add(buffer, resized, [8]), jnp.array([1.0, -1.0]))
    assert sampleable(buffer, resized) == [3, 4, 5, 6, 7, 8][-capacity:]


@pytest.mark.parametrize("capacity", [4, 6, 12])
def test_resize_keeps_an_episode_in_progress(capacity):
    buffer = EpisodeReplayBuffer(capacity=7)
    # a finished episode 1-5, then 6-7 in progress
    state = buffer.assign_rewards(
        add(buffer, init_single(buffer), [1, 2, 3, 4, 5]), jnp.array([1.0, -1.0])
    )
    state = add(buffer, state, [6, 7])

    resized = buffer.resize(state, capacity)

    # the episode in progress can't be sampled, and is kept
    assert sampleable(buffer, resized) == [1, 2, 3, 4, 5][-(capacity - 2) :]
    # when it ends, its entries get its reward
    ended = buffer.assign_rewards(add(buffer, resized, [8]), jnp.array([-1.0, 1.0]))
    assert sampleable(buffer, ended) == [1, 2, 3, 4, 5, 6, 7, 8][-capacity:]
    rewards = {
        int(obs): float(r[0])
        for obs, r, ok in zip(
            ended.buffer.observation_nn,
            ended.buffer.reward,
            ended.populated,
            strict=True,
        )
        if ok
    }
    assert {obs: rewards[obs] for obs in (6, 7, 8)} == {6: -1.0, 7: -1.0, 8: -1.0}
    assert all(rewards[obs] == 1.0 for obs in rewards if obs < 6)
    # or truncating discards just its entries
    truncated = buffer.truncate(add(buffer, resized, [8]))
    assert sampleable(buffer, truncated) == [1, 2, 3, 4, 5][-(capacity - 3) :]
    assert int(truncated.populated.sum()) == min(5, capacity - 3)


def test_resize_keeps_the_newest_entries_of_an_episode_longer_than_it():
    buffer = EpisodeReplayBuffer(capacity=6)
    state = add(buffer, init_single(buffer), [1, 2, 3, 4, 5])

    resized = buffer.assign_rewards(
        add(buffer, buffer.resize(state, 3), [6]), jnp.ones(2)
    )

    assert sampleable(buffer, resized) == [4, 5, 6]


def test_resize_under_jit_and_vmap():
    buffer = EpisodeReplayBuffer(capacity=4)
    states = [
        buffer.assign_rewards(add(buffer, init_single(buffer), ids), jnp.ones(2))
        for ids in ([1, 2, 3, 4, 5], [6, 7])
    ]
    state = jax.tree.map(lambda *x: jnp.stack(x), *states)

    resized = jax.jit(jax.vmap(buffer.resize, in_axes=(0, None)), static_argnums=1)(
        state, 2
    )

    assert resized.populated.shape == (2, 2)
    for i, expected in enumerate([[4, 5], [6, 7]]):
        env = jax.tree.map(lambda x, i=i: x[i], resized)
        assert sampleable(buffer, env) == expected


def test_buffer_operations_take_the_capacity_of_the_state():
    # a buffer configured for 5 entries, on a state of 3
    buffer = EpisodeReplayBuffer(capacity=5)
    state = init_single(EpisodeReplayBuffer(capacity=3))

    state = buffer.assign_rewards(add(buffer, state, [1, 2, 3, 4]), jnp.ones(2))
    assert state.populated.shape == (3,)
    assert sampleable(buffer, state) == [2, 3, 4]
    assert state.next_idx == 1
