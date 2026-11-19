"""Two-player games between evaluators."""

from functools import partial
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from turbozero.common import two_player_game


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
