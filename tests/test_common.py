"""Stepping an environment and an evaluator together, and two-player games between evaluators."""

from functools import partial
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from turbozero.common import step_env_and_evaluator, two_player_game


def play_games(env, evaluator_1, evaluator_2, max_steps, p1_first, num_games=16):
    game = partial(
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
    keys = jax.random.split(jax.random.PRNGKey(0), num_games)
    return jax.jit(jax.vmap(game))(keys)


@pytest.mark.parametrize("p1_first", [None, True, False])
@pytest.mark.parametrize("winner", [1, 2])
def test_two_player_game_attributes_outcomes_to_evaluators(
    ttt, scripted, winner, p1_first
):
    # the winner wins whoever moves first: unlike in a game decided by who moves
    # first, the outcomes show which evaluator they're credited to
    evaluators = (
        (scripted.beats_first_legal, scripted.first_legal)
        if winner == 1
        else (scripted.first_legal, scripted.beats_first_legal)
    )
    env = SimpleNamespace(step_fn=ttt.step_fn, init_fn=lambda key: ttt.init_fn(0))

    outcomes = play_games(env, *evaluators, max_steps=9, p1_first=p1_first)

    expected = [1.0, -1.0] if winner == 1 else [-1.0, 1.0]
    np.testing.assert_array_equal(outcomes, np.tile(expected, (len(outcomes), 1)))


@pytest.mark.parametrize("p1_first", [None, True, False])
def test_two_player_game_is_won_by_whoever_moves_first(
    fixed_length_env, scripted, p1_first
):
    # a game of 3 moves, won by the first to move; the step limit is odd too
    env = fixed_length_env(length=3)

    outcomes = play_games(
        env, scripted.first_legal, scripted.first_legal, 5, p1_first=p1_first
    )

    evaluator_1_first = outcomes[:, 0] == 1.0
    np.testing.assert_array_equal(outcomes[:, 1], -outcomes[:, 0])
    if p1_first is None:
        # chosen at random
        assert evaluator_1_first.any() and (~evaluator_1_first).any()
    else:
        assert (evaluator_1_first == p1_first).all()


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
    )(jax.random.PRNGKey(0), env_state, meta, evaluator.init(None))


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


def run_steps(env, evaluator, max_steps, num_steps, reset=True):
    """Steps `env` from its initial state `num_steps` times. Returns per-step (step count, terminated, truncated, rewards)."""
    step = jax.jit(
        partial(
            step_env_and_evaluator,
            params=None,
            evaluator=evaluator,
            env_step_fn=env.step_fn,
            env_init_fn=env.init_fn,
            max_steps=max_steps,
            reset=reset,
        )
    )
    env_state, meta = env.init_fn(jax.random.PRNGKey(0))
    eval_state = evaluator.init(None)
    history = []
    for i in range(num_steps):
        output, env_state, meta, terminated, truncated, rewards = step(
            jax.random.PRNGKey(i),
            env_state=env_state,
            env_state_metadata=meta,
            eval_state=eval_state,
        )
        eval_state = output.eval_state
        history.append(
            (int(meta.step), bool(terminated), bool(truncated), np.asarray(rewards))
        )
    return history


@pytest.mark.parametrize("max_steps", [1, 2, 5])
def test_episode_is_truncated_on_its_max_steps_th_step(
    fixed_length_env, scripted, max_steps
):
    env = fixed_length_env(length=10)

    history = run_steps(
        env, scripted.first_legal, max_steps, num_steps=max_steps, reset=False
    )

    truncated = [h[2] for h in history]
    assert truncated == [False] * (max_steps - 1) + [True]
    assert not any(h[1] for h in history)
    assert history[-1][0] == max_steps


def test_truncated_episode_resets_after_max_steps_steps(fixed_length_env, scripted):
    env = fixed_length_env(length=10)

    history = run_steps(env, scripted.first_legal, max_steps=3, num_steps=7)

    # two full truncated episodes of 3 steps, then the first step of a third
    assert [h[2] for h in history] == [False, False, True] * 2 + [False]
    assert [h[0] for h in history] == [1, 2, 0, 1, 2, 0, 1]


def test_episode_terminating_on_its_last_allowed_step_is_terminated_not_truncated(
    fixed_length_env, scripted
):
    env = fixed_length_env(length=4)

    *_, (_, terminated, truncated, rewards) = run_steps(
        env, scripted.first_legal, max_steps=4, num_steps=4, reset=False
    )

    assert terminated and not truncated
    np.testing.assert_array_equal(rewards, env.rewards)


@pytest.mark.parametrize("max_steps", [4, 5])
def test_two_player_game_finishing_on_exactly_max_steps_is_scored(
    fixed_length_env, scripted, max_steps
):
    env = fixed_length_env(length=max_steps)

    outcomes = play_games(
        env, scripted.first_legal, scripted.first_legal, max_steps, p1_first=True
    )

    # evaluator_1 moves first, as player 0
    np.testing.assert_array_equal(outcomes, np.tile(env.rewards, (len(outcomes), 1)))


def test_two_player_game_hitting_the_step_limit_is_completed_as_a_draw(
    fixed_length_env, scripted
):
    env = fixed_length_env(length=5)

    outcomes = play_games(
        env, scripted.first_legal, scripted.first_legal, 4, p1_first=True
    )

    np.testing.assert_array_equal(outcomes, np.zeros_like(outcomes))
