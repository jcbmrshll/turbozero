"""Resignation in self-play (core.training.resignation), through a `Trainer`'s collection.

Uses `make_fixed_length_env` (see conftest), whose games end after exactly `length` moves with
fixed rewards, and an evaluator whose root value depends only on the player to move."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.evaluators.evaluator import EvalOutput, Evaluator
from core.training.resignation import Resignation
from core.training.train import CollectionStats


class PlayerValueEvaluator(Evaluator):
    """Plays action 0; its root value is `values[player to move]`."""

    def __init__(self, values):
        super().__init__(discount=-1.0)
        self.values = jnp.asarray(values, dtype=jnp.float32)

    def init(self, *args, **kwargs):
        return jnp.array(0.0)

    def reset(self, state):
        return state

    def evaluate(self, key, eval_state, env_state, root_metadata, **kwargs):
        action = jnp.array(0)
        return EvalOutput(
            eval_state=self.values[root_metadata.cur_player_id],
            action=action,
            policy_weights=jax.nn.one_hot(action, root_metadata.action_mask.shape[-1]),
        )

    def get_value(self, state):
        return state


def collect(make_collector, env, values, resignation, num_steps, tmp_path):
    return make_collector(
        env,
        PlayerValueEvaluator(values),
        max_episode_steps=16,
        ckpt_dir=tmp_path,
        resignation=resignation,
    )(num_steps)


def sampleable(state):
    buffer = state.buffer_state
    return np.flatnonzero(np.asarray(buffer.populated & buffer.has_reward)[0])


def test_the_player_to_move_resigns_at_the_threshold_and_loses(
    fixed_length_env, make_collector, tmp_path
):
    env = fixed_length_env(length=6, rewards=(-1.0, 1.0))
    # player 1 is lost by its own search, though it would win the game played out
    state = collect(
        make_collector, env, (0.0, -0.9), Resignation(0.9, 0.0), 2, tmp_path
    )

    # both positions are kept, the resigning one included, scored as player 1's loss
    np.testing.assert_array_equal(sampleable(state), [0, 1])
    np.testing.assert_array_equal(
        state.buffer_state.buffer.reward[0, :2], [[1.0, -1.0], [1.0, -1.0]]
    )
    np.testing.assert_array_equal(state.episodes, [1])
    np.testing.assert_array_equal(state.resign_state.resigned, [1])
    # a new game started
    np.testing.assert_array_equal(state.metadata.step, [0])


def test_a_value_above_the_threshold_plays_the_game_out(
    fixed_length_env, make_collector, tmp_path
):
    env = fixed_length_env(length=4)
    state = collect(
        make_collector, env, (-0.85, -0.85), Resignation(0.9, 0.0), 4, tmp_path
    )

    np.testing.assert_array_equal(sampleable(state), np.arange(4))
    np.testing.assert_array_equal(
        state.buffer_state.buffer.reward[0, :4], np.tile(env.rewards, (4, 1))
    )
    np.testing.assert_array_equal(state.resign_state.resigned, [0])


@pytest.mark.parametrize(
    "rewards, false", [((1.0, -1.0), 0), ((-1.0, 1.0), 1), ((0.0, 0.0), 1)]
)
def test_calibration_games_play_out_and_count_wrong_resignations(
    fixed_length_env, make_collector, tmp_path, rewards, false
):
    env = fixed_length_env(length=4, rewards=rewards)
    state = collect(
        make_collector, env, (0.0, -0.95), Resignation(0.9, 1.0), 4, tmp_path
    )

    # the game was played to its end and scored by the environment
    np.testing.assert_array_equal(sampleable(state), np.arange(4))
    np.testing.assert_array_equal(
        state.buffer_state.buffer.reward[0, :4], np.tile(env.rewards, (4, 1))
    )
    resign = state.resign_state
    np.testing.assert_array_equal(resign.resigned, [0])
    np.testing.assert_array_equal(resign.calibration_games, [1])
    np.testing.assert_array_equal(resign.calibration_below, [1])
    # player 1 fell to the threshold: it was wrong to resign unless it lost
    np.testing.assert_array_equal(resign.calibration_false, [false])
    np.testing.assert_array_equal(resign.first_below, [-1])


def test_collection_metrics_report_resignations(
    fixed_length_env, make_collector, tmp_path
):
    env = fixed_length_env(length=6)
    collect = make_collector(
        env,
        PlayerValueEvaluator((0.0, -0.95)),
        max_episode_steps=16,
        ckpt_dir=tmp_path,
        resignation=Resignation(0.9, 0.0),
    )

    metrics = collect.trainer.collection_metrics(
        CollectionStats.of(collect(0)), collect(6)
    )

    # every game ends in player 1's resignation, on its first move
    assert int(metrics["episodes"]) == 3
    assert float(metrics["resigned"]) == 1.0
    assert int(metrics["resign_calibration_games"]) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0.0},
        {"threshold": 1.5},
        {"threshold": 0.9, "calibration_fraction": 1.5},
    ],
)
def test_settings_out_of_range_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Resignation(**kwargs)
