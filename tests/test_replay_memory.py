from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.memory.replay_memory import BaseExperience, EpisodeReplayBuffer


def experience(obs_id):
    """An experience identified by its observation."""
    return BaseExperience(
        reward=jnp.zeros((2,)),
        policy_weights=jnp.full((3,), 1 / 3),
        policy_mask=jnp.ones((3,), dtype=jnp.bool_),
        observation_nn=jnp.array(obs_id, dtype=jnp.float32),
        cur_player_id=jnp.array(0, dtype=jnp.int32),
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


def test_sample_only_returns_finished_populated_entries():
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

    for seed in range(10):
        sample = buffer.sample(state, jax.random.PRNGKey(seed), 3)
        assert sorted(sample.observation_nn.tolist()) == [1, 2, 4]


def test_sample_before_any_episode_finished_raises():
    buffer = EpisodeReplayBuffer(capacity=4)
    state = add(buffer, init_single(buffer), [1, 2])
    state = jax.tree.map(lambda x: x[None], state)

    with pytest.raises(ValueError):
        buffer.sample(state, jax.random.PRNGKey(0), 2)


def test_count_distinct_observations_counts_only_sampleable_experiences():
    buffer = EpisodeReplayBuffer(capacity=8)
    state = add(buffer, init_single(buffer), [1, 2, 1, 3, 2])
    state = buffer.assign_rewards(state, jnp.array([1.0, -1.0]))
    # an unfinished episode can't be sampled yet, so its new observation doesn't count
    state = add(buffer, state, [4])

    distinct, total = buffer.count_distinct_observations(state)

    assert (distinct, total) == (3, 5)


def test_count_distinct_observations_across_batch_dimensions():
    buffer = EpisodeReplayBuffer(capacity=4)
    template = experience(0)
    template = replace(template, observation_nn=jnp.zeros((2,), dtype=jnp.bool_))
    # two devices, two environments each
    state = jax.tree.map(
        lambda x: x.reshape(2, 2, *x.shape[1:]), buffer.init(4, template)
    )
    obs = jnp.array([[1, 0], [0, 1], [1, 1], [1, 0]], dtype=jnp.bool_)
    state = replace(
        state,
        buffer=replace(
            state.buffer,
            # environment i holds observation i in its first slot, and observation 0 in its second
            observation_nn=state.buffer.observation_nn.at[:, :, 0]
            .set(obs.reshape(2, 2, 2))
            .at[:, :, 1]
            .set(obs[0]),
        ),
        populated=state.populated.at[:, :, :2].set(True),
    )

    distinct, total = jax.jit(buffer.count_distinct_observations)(state)

    # [1, 0] and [0, 1] hold the same values in different places, so they differ
    assert (distinct, total) == (3, 8)
