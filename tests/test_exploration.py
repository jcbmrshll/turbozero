"""Self-play exploration: the move self-play plays, chosen from the evaluator's output,
while the evaluator's policy weights stay the training target."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.common import step_env_and_evaluator
from core.evaluators.evaluator import EvalOutput
from core.training.exploration import Exploration, MoveTemperature
from core.types import StepMetadata

NUM_SAMPLES = 4000


def eval_output(action, policy_weights):
    return EvalOutput(
        eval_state=jnp.zeros(()),
        action=jnp.array(action),
        policy_weights=jnp.asarray(policy_weights, dtype=jnp.float32),
    )


def step_metadata(action_mask, step=0):
    return StepMetadata(
        rewards=jnp.zeros((2,)),
        action_mask=jnp.asarray(action_mask, dtype=jnp.bool_),
        terminated=jnp.array(False),
        cur_player_id=jnp.array(0),
        step=jnp.array(step),
    )


def choose(exploration, output, metadata, num_samples=NUM_SAMPLES):
    """The actions `exploration` plays across `num_samples` keys."""
    keys = jax.random.split(jax.random.PRNGKey(0), num_samples)
    choose_action = jax.vmap(
        partial(exploration.choose_action, output=output, metadata=metadata)
    )
    return np.asarray(jax.jit(choose_action)(keys))


def test_default_plays_the_evaluators_move():
    output = eval_output(2, [0.1, 0.6, 0.3])

    actions = choose(Exploration(), output, step_metadata([1, 1, 1], step=50))

    assert (actions == 2).all()


def test_random_move_prob_one_plays_uniformly_random_legal_moves():
    mask = [True, False, True, False, False, True]
    output = eval_output(0, [1.0, 0, 0, 0, 0, 0])

    actions = choose(Exploration(random_move_prob=1.0), output, step_metadata(mask))

    counts = np.bincount(actions, minlength=len(mask))
    assert (counts[~np.array(mask)] == 0).all()
    np.testing.assert_allclose(counts[np.array(mask)] / NUM_SAMPLES, 1 / 3, atol=0.03)


def test_random_move_prob_replaces_that_fraction_of_moves():
    # 4 legal moves: a random move differs from the evaluator's 3/4 of the time
    output = eval_output(1, [0.0, 1.0, 0.0, 0.0])

    actions = choose(
        Exploration(random_move_prob=0.2), output, step_metadata([1, 1, 1, 1])
    )

    assert (actions != 1).mean() == pytest.approx(0.2 * 3 / 4, abs=0.03)


@pytest.mark.parametrize("step, expected", [(0, 0), (2, 0), (3, 1), (10, 1)])
def test_num_sampling_moves_plays_the_evaluators_move_then_the_best(step, expected):
    # the evaluator sampled move 0, but move 1 has the most weight
    output = eval_output(0, [0.2, 0.5, 0.3])

    actions = choose(
        Exploration(num_sampling_moves=3),
        output,
        step_metadata([1, 1, 1], step=step),
    )

    assert (actions == expected).all()


def test_greedy_moves_break_ties_at_random_among_legal_moves():
    # move 3 has the most weight but is illegal; moves 0 and 2 tie
    output = eval_output(1, [0.4, 0.1, 0.4, 0.9])

    actions = choose(
        Exploration(num_sampling_moves=0),
        output,
        step_metadata([1, 1, 1, 0], step=5),
    )

    assert set(actions.tolist()) == {0, 2}
    assert (actions == 0).mean() == pytest.approx(0.5, abs=0.05)


def test_step_plays_the_chosen_move_and_keeps_the_evaluators_policy(ttt, scripted):
    step = jax.jit(
        partial(
            step_env_and_evaluator,
            params=None,
            evaluator=scripted.first_legal,
            env_step_fn=ttt.step_fn,
            env_init_fn=ttt.init_fn,
            max_steps=9,
            choose_action=lambda key, output, metadata: jnp.array(4),
        )
    )
    env_state, meta = ttt.init_fn(jax.random.PRNGKey(0))

    output, env_state, *_ = step(
        jax.random.PRNGKey(1),
        env_state=env_state,
        env_state_metadata=meta,
        eval_state=scripted.first_legal.init(),
    )

    assert output.action == 4
    # the center square is taken, not the first legal one the evaluator played
    assert not env_state.legal_action_mask[4]
    assert env_state.legal_action_mask[0]
    np.testing.assert_array_equal(output.policy_weights, np.eye(9)[0])
    # and the evaluator follows the move played (e.g. to its subtree), not its own
    assert int(output.eval_state.last_step) == 4


def test_collect_stores_the_evaluators_policy_not_the_move_played(
    fixed_length_env, scripted, make_collector, tmp_path
):
    env = fixed_length_env(length=4, num_actions=3)
    collect = make_collector(
        env,
        scripted.first_legal,
        max_episode_steps=4,
        ckpt_dir=tmp_path,
        exploration=Exploration(random_move_prob=1.0),
    )

    buffer = collect(num_steps=4).buffer_state.buffer

    np.testing.assert_array_equal(buffer.policy_weights[0, :4], [[1, 0, 0]] * 4)


def test_move_temperature_decays_from_early_towards_late():
    schedule = MoveTemperature(early=0.75, late=0.15, halflife=8, start=8)

    temperatures = [float(schedule.at(jnp.array(step))) for step in (0, 8, 16, 24, 200)]

    assert temperatures[:4] == pytest.approx([0.75, 0.75, 0.45, 0.3])
    assert temperatures[4] == pytest.approx(0.15, abs=1e-6)


@pytest.mark.parametrize("kwargs", [{"late": 0.0}, {"late": 1.0}, {"halflife": 0}])
def test_move_temperature_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        MoveTemperature(**kwargs)


def test_move_temperature_samples_the_sharpened_weights_whatever_the_evaluators_move():
    # at temperature 0.5 the weights [0.6, 0.0, 0.4, 0.3] (move 3 illegal) become 0.36 : 0.16
    output = eval_output(1, [0.6, 0.0, 0.4, 0.3])
    temperature = MoveTemperature(early=0.5, late=0.5)

    actions = choose(
        Exploration(temperature=temperature), output, step_metadata([1, 1, 1, 0])
    )

    counts = np.bincount(actions, minlength=4) / NUM_SAMPLES
    np.testing.assert_allclose(counts, [0.36 / 0.52, 0, 0.16 / 0.52, 0], atol=0.03)


def test_move_temperature_never_turns_greedy():
    # late in the game, at 0.15, a move with half the best's weight is still played sometimes
    output = eval_output(0, [0.5, 0.25, 0.25])

    actions = choose(
        Exploration(temperature=MoveTemperature()),
        output,
        step_metadata([1, 1, 1], step=200),
    )

    # 0.5 ** (1 / 0.15) = 0.0098 of the best's weight, each
    assert (actions != 0).mean() == pytest.approx(2 * 0.0098 / 1.0196, abs=0.006)
    assert (actions != 0).any()


def test_move_temperature_with_no_weights_plays_uniformly_among_legal_moves():
    output = eval_output(0, [0.0, 0.0, 0.0])

    actions = choose(
        Exploration(temperature=MoveTemperature()), output, step_metadata([1, 0, 1])
    )

    assert set(actions.tolist()) == {0, 2}


def test_sampling_moves_and_a_move_temperature_dont_go_together():
    with pytest.raises(ValueError):
        Exploration(num_sampling_moves=20, temperature=MoveTemperature())


@dataclass(frozen=True)
class AlwaysSecond(Exploration):
    """Plays action 1 every move: off line whenever the policy's top move is another."""

    def choose_action(self, key, output, metadata):
        return jnp.array(1)


def test_store_until_offline_stops_storing_a_game_after_its_first_off_line_move(
    fixed_length_env, scripted, make_collector, tmp_path
):
    # games of 4 moves; the scripted policy's top move is 0, so every move here is off line. From
    # move 2 on that counts: moves 0, 1 and 2 are stored, 3 isn't, and the next game starts on line
    env = fixed_length_env(length=4, num_actions=3)
    collect = make_collector(
        env,
        scripted.first_legal,
        max_episode_steps=4,
        ckpt_dir=tmp_path,
        exploration=AlwaysSecond(store_until_offline=2),
    )

    state = collect(num_steps=8)

    assert int(state.buffer_state.populated.sum()) == 6
    assert not bool(state.offline[0])


def test_store_until_offline_keeps_games_played_on_line_whole(
    fixed_length_env, scripted, make_collector, tmp_path
):
    env = fixed_length_env(length=4, num_actions=3)
    collect = make_collector(
        env,
        scripted.first_legal,
        max_episode_steps=4,
        ckpt_dir=tmp_path,
        exploration=Exploration(num_sampling_moves=0, store_until_offline=0),
    )

    assert int(collect(num_steps=8).buffer_state.populated.sum()) == 8


def test_goes_offline_counts_ties_for_the_top_move_as_on_line():
    exploration = Exploration(store_until_offline=3)
    output = eval_output(2, [0.4, 0.2, 0.4])

    assert (
        bool(exploration.goes_offline(output, step_metadata([1, 1, 1], step=3)))
        is False
    )
    off = eval_output(1, [0.4, 0.2, 0.4])
    assert bool(exploration.goes_offline(off, step_metadata([1, 1, 1], step=3)))
    assert not bool(exploration.goes_offline(off, step_metadata([1, 1, 1], step=2)))
