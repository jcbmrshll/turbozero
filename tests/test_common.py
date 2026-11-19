"""Stepping an environment and an evaluator together, and two-player games between evaluators."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from turbozero.common import step_env_and_evaluator, two_player_game


def play_games(env, evaluator_1, evaluator_2, max_steps, num_games=16, p1_first=None):
    game = jax.jit(
        jax.vmap(
            partial(
                two_player_game,
                evaluator_1=evaluator_1,
                evaluator_2=evaluator_2,
                params_1=None,
                params_2=None,
                env_step_fn=env.step_fn,
                env_init_fn=env.init_fn,
                max_steps=max_steps,
                p1_first=p1_first,
            )
        )
    )
    outcomes, frames, p_ids = game(jax.random.split(jax.random.PRNGKey(0), num_games))
    # evaluator_1 moved first if it plays as the player to move in the initial frame
    evaluator_1_first = p_ids[:, 0] == frames.env_state.current_player[:, 0]
    return outcomes, frames, evaluator_1_first


@pytest.mark.parametrize("winner", [1, 2])
def test_two_player_game_attributes_outcomes_to_evaluators(ttt, scripted, winner):
    # the resigning evaluator loses whoever moves first: unlike in a game decided by who moves
    # first, the outcomes show which evaluator they're credited to
    evaluators = (
        (scripted.first_legal, scripted.resign)
        if winner == 1
        else (scripted.resign, scripted.first_legal)
    )
    outcomes, frames, evaluator_1_first = play_games(ttt, *evaluators, max_steps=10)

    # both move orders are covered
    assert evaluator_1_first.any() and (~evaluator_1_first).any()
    expected = [1.0, -1.0] if winner == 1 else [-1.0, 1.0]
    np.testing.assert_array_equal(outcomes, np.tile(expected, (len(outcomes), 1)))
    assert frames.completed[:, -1].all()


@pytest.mark.parametrize("p1_first", [None, True, False])
def test_two_player_game_is_won_by_whoever_moves_first(
    fixed_length_env, scripted, p1_first
):
    # a game of 3 moves, won by the first to move; the step limit is odd too
    env = fixed_length_env(length=3)
    max_steps = 5

    outcomes, frames, evaluator_1_first = play_games(
        env, scripted.first_legal, scripted.first_legal, max_steps, p1_first=p1_first
    )

    if p1_first is None:
        # chosen at random
        assert evaluator_1_first.any() and (~evaluator_1_first).any()
    else:
        assert (evaluator_1_first == p1_first).all()
    expected = jnp.where(
        evaluator_1_first[:, None], jnp.array([1.0, -1.0]), jnp.array([-1.0, 1.0])
    )
    np.testing.assert_array_equal(outcomes, expected)
    # the initial frame, then one per step
    assert frames.completed.shape[1] == max_steps + 1


def step_once(env, evaluator, max_steps, reset=True):
    """One step of `env` from its initial state, and of `evaluator` from its initial state."""
    env_state, meta = env.init_fn(jax.random.PRNGKey(0))
    return jax.jit(
        partial(
            step_env_and_evaluator,
            params=None,
            evaluator=evaluator,
            env_step_fn=env.step_fn,
            env_init_fn=env.init_fn,
            max_steps=max_steps,
            reset=reset,
        )
    )(jax.random.PRNGKey(0), env_state, meta, evaluator.init())


@pytest.mark.parametrize(
    "length, max_steps, ending",
    [(1, 10, "terminated"), (10, 1, "truncated")],
)
def test_step_resets_env_and_evaluator_when_episode_ends(
    fixed_length_env, scripted, length, max_steps, ending
):
    env = fixed_length_env(length=length)

    output, env_state, _, terminated, truncated, rewards = step_once(
        env, scripted.first_legal, max_steps
    )

    assert (bool(terminated), bool(truncated)) == (
        ending == "terminated",
        ending == "truncated",
    )
    np.testing.assert_array_equal(
        rewards, env.rewards if ending == "terminated" else [0.0, 0.0]
    )
    # a new game, and the evaluator reset rather than stepped
    assert int(env_state.step) == 0
    eval_state = output.eval_state
    assert int(eval_state.resets) == 1 and int(eval_state.steps) == 0


def test_step_without_reset_keeps_terminal_state(fixed_length_env, scripted):
    env = fixed_length_env(length=1)

    output, env_state, meta, terminated, _, _ = step_once(
        env, scripted.first_legal, max_steps=10, reset=False
    )

    assert terminated and meta.terminated and int(env_state.step) == 1
    # the evaluator's state as the evaluation left it, neither reset nor stepped
    eval_state = output.eval_state
    assert int(eval_state.moves) == 1
    assert int(eval_state.resets) == 0 and int(eval_state.steps) == 0


def test_step_mid_episode_advances_env_and_steps_evaluator(fixed_length_env, scripted):
    env = fixed_length_env(length=10)

    output, env_state, _, terminated, truncated, _ = step_once(
        env, scripted.first_legal, max_steps=10
    )

    assert not terminated and not truncated
    assert int(env_state.step) == 1
    # the evaluator follows the move played (e.g. to the subtree of a search tree)
    eval_state = output.eval_state
    assert int(eval_state.steps) == 1 and int(eval_state.last_step) == output.action
    assert int(eval_state.resets) == 0
