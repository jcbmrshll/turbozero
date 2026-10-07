from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.common import step_env_and_evaluator, two_player_game
from core.evaluators.alphazero import AlphaZero
from core.evaluators.mcts.mcts import MCTS

# a drawn game with one move left: X 0 2 3 7 (8 to play), O 1 4 5 6
DRAW_MOVES, LAST_ACTION = [0, 1, 2, 4, 3, 5, 7, 6], 8


def play_games(ttt, evaluator_1, evaluator_2, max_steps, num_games=16):
    game = jax.jit(
        jax.vmap(
            partial(
                two_player_game,
                evaluator_1=evaluator_1,
                evaluator_2=evaluator_2,
                params_1=None,
                params_2=None,
                env_step_fn=ttt.step_fn,
                env_init_fn=ttt.init_fn,
                max_steps=max_steps,
            )
        )
    )
    outcomes, frames, p_ids = game(jax.random.split(jax.random.PRNGKey(0), num_games))
    # evaluator_1 moved first if it plays as the player to move in the initial frame
    evaluator_1_first = p_ids[:, 0] == frames.env_state.current_player[:, 0]
    return outcomes, frames, evaluator_1_first


@pytest.mark.parametrize("winner", [1, 2])
def test_two_player_game_attributes_outcomes_to_evaluators(ttt, scripted, winner):
    # the resigning evaluator loses whoever moves first
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


def test_two_player_game_first_mover_wins_with_first_legal_play(ttt, scripted):
    # both players take the lowest free square, so X completes the 2-4-6 diagonal on move 7
    outcomes, _, evaluator_1_first = play_games(
        ttt, scripted.first_legal, scripted.first_legal, max_steps=10
    )

    expected = jnp.where(
        evaluator_1_first[:, None], jnp.array([1.0, -1.0]), jnp.array([-1.0, 1.0])
    )
    np.testing.assert_array_equal(outcomes, expected)


def test_two_player_game_with_odd_max_steps(ttt, scripted):
    max_steps = 7
    outcomes, frames, evaluator_1_first = play_games(
        ttt, scripted.first_legal, scripted.first_legal, max_steps
    )

    expected = jnp.where(
        evaluator_1_first[:, None], jnp.array([1.0, -1.0]), jnp.array([-1.0, 1.0])
    )
    np.testing.assert_array_equal(outcomes, expected)
    # the initial frame, then one per step
    assert frames.completed.shape[1] == max_steps + 1


@pytest.fixture(scope="module")
def stepper(make_search, ttt):
    search = make_search(AlphaZero(MCTS))

    @partial(jax.jit, static_argnames=("max_steps", "reset"))
    def step(key, env_state, meta, eval_state, max_steps, reset=True):
        return step_env_and_evaluator(
            key,
            env_state,
            meta,
            eval_state,
            None,
            search.evaluator,
            ttt.step_fn,
            ttt.init_fn,
            max_steps,
            reset=reset,
        )

    return search, step


def assert_is_initial_state(env_state):
    assert env_state._step_count == 0
    assert not env_state.terminated
    assert env_state.legal_action_mask.all()


def test_step_resets_env_and_evaluator_when_episode_terminates(ttt, stepper):
    search, step = stepper
    state, meta = ttt.play(DRAW_MOVES)

    output, env_state, _, terminated, truncated, rewards = step(
        jax.random.PRNGKey(0), state, meta, search.init(), max_steps=20
    )

    assert output.action == LAST_ACTION
    assert terminated and not truncated
    np.testing.assert_array_equal(rewards, [0.0, 0.0])
    assert_is_initial_state(env_state)
    assert output.eval_state.next_free_idx == 0
    assert root_n(output.eval_state) == 0


def test_step_resets_env_and_evaluator_when_episode_is_truncated(ttt, stepper):
    search, step = stepper
    state, meta = ttt.play([])

    output, env_state, _, terminated, truncated, _ = step(
        jax.random.PRNGKey(0), state, meta, search.init(), max_steps=0
    )

    assert truncated and not terminated
    assert_is_initial_state(env_state)
    assert output.eval_state.next_free_idx == 0


def test_step_without_reset_keeps_terminal_state(ttt, stepper):
    search, step = stepper
    state, meta = ttt.play(DRAW_MOVES)

    output, env_state, _, terminated, _, _ = step(
        jax.random.PRNGKey(0), state, meta, search.init(), max_steps=20, reset=False
    )

    assert terminated
    assert env_state.terminated
    assert output.eval_state.next_free_idx > 0


def test_step_mid_episode_advances_env_and_reuses_subtree(ttt, stepper):
    search, step = stepper
    state, meta = ttt.play([4])

    output, env_state, _, terminated, truncated, _ = step(
        jax.random.PRNGKey(0), state, meta, search.init(), max_steps=20
    )

    assert not terminated and not truncated
    assert env_state._step_count == 2
    assert env_state._x.board[output.action] != -1
    # the evaluator moved to the searched subtree under the chosen action, not to an empty tree
    assert root_n(output.eval_state) > 1


def root_n(tree):
    return tree.data_at(tree.ROOT_INDEX).n
