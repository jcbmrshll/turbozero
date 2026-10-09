from dataclasses import replace
from functools import partial

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


def stacked(obs_ids):
    """Experiences identified by their observations, stacked along a leading dimension."""
    return jax.tree.map(lambda *x: jnp.stack(x), *[experience(i) for i in obs_ids])


def test_add_experiences_adds_the_valid_ones_ready_to_sample():
    buffer = EpisodeReplayBuffer(capacity=5)
    state = init_single(buffer)

    state = buffer.add_experiences(
        state, stacked([1, 2, 3, 4]), jnp.array([True, False, True, False])
    )

    # written one after another, without gaps, and they can be sampled at once
    assert state.next_idx == 2
    np.testing.assert_array_equal(state.buffer.observation_nn, [1, 3, 0, 0, 0])
    np.testing.assert_array_equal(
        buffer.sample_mask(state), [True, True, False, False, False]
    )

    # wrapping around the end
    state = buffer.add_experiences(
        state, stacked([5, 6, 7, 8]), jnp.array([True, True, True, True])
    )
    state = buffer.add_experiences(state, stacked([9, 10]), jnp.array([False, True]))
    assert state.next_idx == 2
    np.testing.assert_array_equal(state.buffer.observation_nn, [8, 10, 5, 6, 7])
    assert buffer.sample_mask(state).all()

    # none valid: nothing changes
    unchanged = buffer.add_experiences(state, stacked([11, 12]), jnp.zeros(2, bool))
    assert jax.tree.all(jax.tree.map(np.array_equal, unchanged, state))


def test_sample_indices_with_replacement_from_fewer_entries():
    buffer = EpisodeReplayBuffer(capacity=8)
    mask = jnp.zeros((2, 8), dtype=bool).at[1, 3].set(True)

    indices = buffer.sample_indices(jax.random.PRNGKey(0), mask, 5, replace=True)

    np.testing.assert_array_equal(indices, [8 + 3] * 5)


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


def sampleable(buffer, state, window=None):
    """Observations of the entries `sample_mask` marks, sorted."""
    mask = np.asarray(buffer.sample_mask(state, window))
    return sorted(np.asarray(state.buffer.observation_nn)[mask].tolist())


def test_window_holds_the_newest_entries():
    buffer = EpisodeReplayBuffer(capacity=8)
    state = buffer.assign_rewards(
        add(buffer, init_single(buffer), [1, 2, 3, 4, 5]), jnp.array([1.0, -1.0])
    )

    assert sampleable(buffer, state, 1) == [5]
    assert sampleable(buffer, state, 3) == [3, 4, 5]
    assert sampleable(buffer, state, 5) == [1, 2, 3, 4, 5]


def test_window_wraps_around():
    buffer = EpisodeReplayBuffer(capacity=5)
    state = buffer.assign_rewards(
        add(buffer, init_single(buffer), [1, 2, 3, 4, 5, 6, 7]), jnp.array([1.0, -1.0])
    )
    # 6 and 7 wrapped around to indices 0 and 1, so the newest entries straddle the end
    np.testing.assert_array_equal(state.buffer.observation_nn, [6, 7, 3, 4, 5])
    assert state.next_idx == 2

    assert sampleable(buffer, state, 1) == [7]
    assert sampleable(buffer, state, 2) == [6, 7]
    assert sampleable(buffer, state, 3) == [5, 6, 7]
    assert sampleable(buffer, state, 5) == [3, 4, 5, 6, 7]


def test_window_wider_than_the_filled_buffer():
    buffer = EpisodeReplayBuffer(capacity=8)
    state = buffer.assign_rewards(
        add(buffer, init_single(buffer), [1, 2, 3]), jnp.array([1.0, -1.0])
    )

    # the unwritten entries are in the window, but can't be sampled
    for window in (3, 6, 8, 100):
        assert sampleable(buffer, state, window) == [1, 2, 3]
    assert sampleable(buffer, state) == [1, 2, 3]


def test_window_counts_an_unfinished_episode():
    buffer = EpisodeReplayBuffer(capacity=8)
    state = buffer.assign_rewards(
        add(buffer, init_single(buffer), [1, 2, 3]), jnp.array([1.0, -1.0])
    )
    # an episode in progress at the newest end of the window takes up room in it
    state = add(buffer, state, [4, 5])

    assert sampleable(buffer, state, 3) == [3]
    assert sampleable(buffer, state, 4) == [2, 3]
    assert sampleable(buffer, state, 2) == []
    state = jax.tree.map(lambda x: x[None], state)
    # an episode has finished, but none in the window
    with pytest.raises(ValueError, match="only 0 can be sampled"):
        buffer.sample(state, jax.random.PRNGKey(0), 1, window=2)
    with pytest.raises(ValueError, match="only 1 can be sampled"):
        buffer.sample(state, jax.random.PRNGKey(0), 2, window=3)


def test_window_after_truncation_counts_back_from_the_truncated_episode():
    buffer = EpisodeReplayBuffer(capacity=8)
    state = buffer.assign_rewards(
        add(buffer, init_single(buffer), [1, 2, 3]), jnp.array([1.0, -1.0])
    )
    state = buffer.truncate(add(buffer, state, [4, 5]))

    # the discarded episode doesn't push older entries out of the window
    assert sampleable(buffer, state, 2) == [2, 3]


@pytest.mark.parametrize("seed", range(5))
def test_window_matches_a_buffer_of_its_capacity(seed):
    # a window of w over a bigger buffer holds what a buffer of capacity w would, through any
    # sequence of entries and finished episodes
    rng = np.random.default_rng(seed)
    big, window = EpisodeReplayBuffer(capacity=13), int(rng.integers(1, 13))
    small = EpisodeReplayBuffer(capacity=window)
    big_state, small_state = init_single(big), init_single(small)
    obs_id = 1
    for _ in range(40):
        if rng.random() < 0.3:
            reward = jnp.array([1.0, -1.0])
            big_state = big.assign_rewards(big_state, reward)
            small_state = small.assign_rewards(small_state, reward)
        else:
            big_state = big.add_experience(big_state, experience(obs_id))
            small_state = small.add_experience(small_state, experience(obs_id))
            obs_id += 1
        assert sampleable(big, big_state, window) == sampleable(small, small_state)


def test_sampling_never_draws_outside_the_window():
    buffer = EpisodeReplayBuffer(capacity=6)
    # three environments at different points of the buffer: one wrapped around, one partly
    # filled, one with an episode in progress
    states = [
        buffer.assign_rewards(
            add(buffer, init_single(buffer), [1, 2, 3, 4, 5, 6, 7, 8]),
            jnp.array([1.0, -1.0]),
        ),
        buffer.assign_rewards(
            add(buffer, init_single(buffer), [11, 12]), jnp.array([1.0, -1.0])
        ),
        add(
            buffer,
            buffer.assign_rewards(
                add(buffer, init_single(buffer), [21, 22, 23, 24]),
                jnp.array([1.0, -1.0]),
            ),
            [25],
        ),
    ]
    state = jax.tree.map(lambda *x: jnp.stack(x), *states)
    expected = {3: [6, 7, 8, 11, 12, 23, 24], 4: [5, 6, 7, 8, 11, 12, 22, 23, 24]}

    @partial(jax.jit, static_argnums=2)
    def sample(key, window, sample_size):
        mask = buffer.sample_mask(state, window)
        indices = buffer.sample_indices(key, mask, sample_size)
        return state.buffer.observation_nn.reshape(-1)[indices]

    for window, allowed in expected.items():
        drawn = set()
        for seed in range(30):
            obs = sample(jax.random.PRNGKey(seed), jnp.array(window), 3).tolist()
            assert set(obs) <= set(allowed)
            assert len(set(obs)) == 3
            drawn |= set(obs)
        assert drawn == set(allowed)
        # a sample as big as the window takes all of it
        whole = sample(jax.random.PRNGKey(0), jnp.array(window), len(allowed))
        assert sorted(whole.tolist()) == allowed


def test_count_distinct_observations_in_the_window():
    buffer = EpisodeReplayBuffer(capacity=8)
    state = buffer.assign_rewards(
        add(buffer, init_single(buffer), [1, 2, 1, 3, 2]), jnp.array([1.0, -1.0])
    )

    assert buffer.count_distinct_observations(state, 3) == (3, 3)
    assert buffer.count_distinct_observations(state, 4) == (3, 4)
    assert buffer.count_distinct_observations(state) == (3, 5)
