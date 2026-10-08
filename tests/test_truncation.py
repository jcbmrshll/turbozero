"""Episode step limits: `max_steps` is the maximum number of steps an episode may take.

Uses `make_fixed_length_env` (see conftest), which terminates after exactly `length` steps,
so every episode length below is deterministic."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from core.common import step_env_and_evaluator, two_player_game
from core.memory.replay_memory import EpisodeReplayBuffer
from core.testing.two_player_tester import TwoPlayerTester, TwoPlayerTestState
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer


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
    eval_state = evaluator.init()
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


def make_collector(env, evaluator, max_episode_steps, ckpt_dir):
    """A single-environment Trainer on `env`, returning (jitted collect step, initial collection state)."""
    nn = eqx.nn.Linear(2, env.num_actions, key=jax.random.PRNGKey(0))
    trainer = Trainer(
        batch_size=1,
        train_batch_size=1,
        warmup_steps=0,
        collection_steps_per_epoch=1,
        train_steps_per_epoch=1,
        nn=nn,
        loss_fn=az_default_loss_fn,
        optimizer=optax.sgd(1e-3),
        evaluator=evaluator,
        memory_buffer=EpisodeReplayBuffer(capacity=16),
        max_episode_steps=max_episode_steps,
        env_step_fn=env.step_fn,
        env_init_fn=env.init_fn,
        state_to_nn_input_fn=env.state_to_nn_input,
        testers=[],
        ckpt_dir=str(ckpt_dir),
        num_devices=1,
    )
    collect = jax.jit(jax.vmap(partial(trainer.collect, params=None)))
    return collect, trainer.init_collection_state(jax.random.PRNGKey(0), batch_size=1)


def collect_steps(collect, state, num_steps):
    for i in range(num_steps):
        state = collect(jax.random.split(jax.random.PRNGKey(i), 1), state)
    return state.buffer_state


def sampleable(buffer_state):
    return np.asarray(buffer_state.populated & buffer_state.has_reward)[0]


def test_collect_keeps_episode_that_terminates_on_its_last_allowed_step(
    fixed_length_env, scripted, tmp_path
):
    env = fixed_length_env(length=4)
    collect, state = make_collector(
        env, scripted.first_legal, max_episode_steps=4, ckpt_dir=tmp_path
    )

    buffer_state = collect_steps(collect, state, num_steps=4)

    np.testing.assert_array_equal(
        np.flatnonzero(sampleable(buffer_state)), np.arange(4)
    )
    np.testing.assert_array_equal(
        buffer_state.buffer.reward[0, :4], np.tile(env.rewards, (4, 1))
    )
    np.testing.assert_array_equal(
        buffer_state.buffer.cur_player_id[0, :4], [0, 1, 0, 1]
    )
    # the next episode is written after it
    np.testing.assert_array_equal(buffer_state.next_idx, [4])
    np.testing.assert_array_equal(buffer_state.episode_start_idx, [4])


def test_collect_discards_episode_that_hits_the_step_limit(
    fixed_length_env, scripted, tmp_path
):
    env = fixed_length_env(length=4)
    collect, state = make_collector(
        env, scripted.first_legal, max_episode_steps=3, ckpt_dir=tmp_path
    )

    buffer_state = collect_steps(collect, state, num_steps=3)

    assert not sampleable(buffer_state).any()
    # the next episode overwrites the discarded one
    np.testing.assert_array_equal(buffer_state.next_idx, [0])


def play_game(env, evaluator, max_steps):
    game = jax.jit(
        partial(
            two_player_game,
            evaluator_1=evaluator,
            evaluator_2=evaluator,
            params_1=None,
            params_2=None,
            env_step_fn=env.step_fn,
            env_init_fn=env.init_fn,
            max_steps=max_steps,
        )
    )
    return game(jax.random.PRNGKey(0))


@pytest.mark.parametrize("max_steps", [4, 5])
def test_two_player_game_finishing_on_exactly_max_steps_is_scored(
    fixed_length_env, scripted, max_steps
):
    env = fixed_length_env(length=max_steps)

    outcomes, frames, p_ids = play_game(env, scripted.first_legal, max_steps)

    np.testing.assert_array_equal(outcomes, env.rewards[p_ids])
    # the initial frame, then one per step; the game completes on the last one
    np.testing.assert_array_equal(frames.env_state.step, np.arange(max_steps + 1))
    np.testing.assert_array_equal(frames.completed, [False] * max_steps + [True])
    np.testing.assert_array_equal(frames.outcomes[-1], env.rewards)


def test_two_player_game_hitting_the_step_limit_is_completed_as_a_draw(
    fixed_length_env, scripted
):
    max_steps = 4
    env = fixed_length_env(length=max_steps + 1)

    outcomes, frames, _ = play_game(env, scripted.first_legal, max_steps)

    np.testing.assert_array_equal(outcomes, [0.0, 0.0])
    np.testing.assert_array_equal(frames.env_state.step, np.arange(max_steps + 1))
    np.testing.assert_array_equal(frames.completed, [False] * max_steps + [True])


def test_two_player_game_on_tic_tac_toe_hitting_the_step_limit(ttt, scripted):
    # with first-legal play X wins on move 7, so 6 steps cut the game short
    outcomes, frames, _ = play_game(ttt, scripted.first_legal, max_steps=6)

    np.testing.assert_array_equal(outcomes, [0.0, 0.0])
    assert frames.completed[-1] and not frames.completed[:-1].any()
    assert not frames.env_state.terminated[-1]


def test_tester_renders_every_frame_up_to_the_final_state(fixed_length_env, scripted):
    max_steps = 4
    env = fixed_length_env(length=max_steps)
    rendered = []

    def render_fn(frames, p_ids, title, render_dir):  # pylint: disable=unused-argument
        rendered.extend(frames)
        return "rendering"

    tester = TwoPlayerTester(num_episodes=2, render_fn=render_fn)
    state = TwoPlayerTestState(best_params={"w": jnp.zeros(3)})

    _, _, path = tester.run(
        key=jax.random.PRNGKey(0),
        epoch_num=0,
        max_steps=max_steps,
        num_devices=2,
        env_step_fn=env.step_fn,
        env_init_fn=env.init_fn,
        evaluator=scripted.first_legal,
        state=state,
        params={"w": jnp.ones(3)},
    )

    assert path == "rendering"
    assert [int(f.env_state.step) for f in rendered] == list(range(max_steps + 1))
    assert rendered[-1].completed
