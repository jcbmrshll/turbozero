from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.common import two_player_game
from core.evaluators.random_evaluator import RandomEvaluator
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.testing.two_player_tester import TwoPlayerTester, TwoPlayerTestState

MAX_STEPS = 10


@pytest.mark.parametrize(
    # every game pays both players the same, so the agent's mean outcome is the reward
    "reward, adopted",
    [(0.0, False), (-1.0, False), (1.0, True)],
)
def test_two_player_tester_adopts_the_params_only_if_they_win_on_average(
    fixed_length_env, scripted, reward, adopted
):
    env = fixed_length_env(length=3, rewards=(reward, reward))
    tester = TwoPlayerTester(num_episodes=2)
    state = TwoPlayerTestState(best_params={"w": jnp.zeros(3)})

    state, metrics, _, _ = tester.test(
        MAX_STEPS,
        env.step_fn,
        env.init_fn,
        scripted.first_legal,
        jax.random.split(jax.random.PRNGKey(0), 2),
        state,
        {"w": jnp.ones(3)},
    )

    np.testing.assert_array_equal(metrics["TwoPlayerTester_avg_outcome"], reward)
    np.testing.assert_array_equal(state.best_params["w"], np.full(3, float(adopted)))


def test_tester_run_on_skipped_epoch_returns_state_unchanged(ttt, scripted):
    tester = TwoPlayerTester(num_episodes=2, epochs_per_test=2)
    state = TwoPlayerTestState(best_params={"w": jnp.zeros(3)})

    new_state, metrics, rendered = tester.run(
        key=jax.random.PRNGKey(0),
        epoch_num=1,
        max_steps=MAX_STEPS,
        env_step_fn=ttt.step_fn,
        env_init_fn=ttt.init_fn,
        evaluator=scripted.first_legal,
        state=state,
        params={"w": jnp.ones(3)},
    )

    assert new_state is state
    assert metrics == {}
    assert rendered is None


def test_random_evaluator_plays_legal_moves_uniformly(ttt):
    evaluator = RandomEvaluator()
    state, meta = ttt.play([0, 4])
    eval_state = evaluator.init()
    keys = jax.random.split(jax.random.PRNGKey(0), 7000)

    actions = jax.jit(
        jax.vmap(
            lambda k: (
                evaluator.evaluate(k, eval_state, state, root_metadata=meta).action
            )
        )
    )(keys)

    counts = np.bincount(np.asarray(actions), minlength=ttt.num_actions)
    assert counts[0] == counts[4] == 0
    # 7 legal squares, 1000 picks each expected
    np.testing.assert_allclose(counts[np.asarray(meta.action_mask)], 1000, rtol=0.1)


@pytest.mark.parametrize(
    "rewards, win_rate, loss_rate",
    [
        # won by whoever moves first: with the agent first in exactly half the games, it wins half
        ((1.0, -1.0), 0.5, 0.5),
        # drawn
        ((0.0, 0.0), 0.0, 0.0),
    ],
)
def test_two_player_baseline_reports_win_and_loss_rates(
    fixed_length_env, scripted, rewards, win_rate, loss_rate
):
    env = fixed_length_env(length=3, rewards=rewards)
    tester = TwoPlayerBaseline(
        num_episodes=8,
        baseline_evaluator=scripted.first_legal,
        balance_first_player=True,
        name="baseline",
    )

    _, metrics, _, _ = tester.test(
        MAX_STEPS,
        env.step_fn,
        env.init_fn,
        scripted.first_legal,
        jax.random.split(jax.random.PRNGKey(0), 8),
        tester.init(params=None),
        {"w": jnp.ones(3)},
    )

    np.testing.assert_array_equal(metrics["baseline_win_rate"], win_rate)
    np.testing.assert_array_equal(metrics["baseline_loss_rate"], loss_rate)


def test_balanced_two_player_baseline_needs_an_even_number_of_episodes(scripted):
    TwoPlayerBaseline(
        num_episodes=6,
        baseline_evaluator=scripted.first_legal,
        balance_first_player=True,
    )
    with pytest.raises(ValueError, match="even number of episodes"):
        TwoPlayerBaseline(
            num_episodes=3,
            baseline_evaluator=scripted.first_legal,
            balance_first_player=True,
        )


def test_two_player_game_records_each_players_tree_of_each_position(ttt, make_search):
    # two searches of different sizes, so their records differ in shape too
    big = make_search().evaluator
    small = make_search(num_iterations=16, max_nodes=40).evaluator
    game = partial(
        two_player_game,
        evaluator_1=big,
        evaluator_2=small,
        params_1=None,
        params_2=None,
        env_step_fn=ttt.step_fn,
        env_init_fn=ttt.init_fn,
        max_steps=MAX_STEPS,
        p1_first=True,
    )
    _, frames, _ = jax.device_get(
        jax.jit(partial(game, record_search=True))(jax.random.PRNGKey(0))
    )
    # (tracing is enough to see that a game without records keeps none)
    _, unrecorded, _ = jax.eval_shape(game, jax.random.PRNGKey(0))
    assert unrecorded.p1_search is None and unrecorded.p2_search is None

    num_moves = int(np.argmax(frames.completed))
    board = frames.env_state._x.board
    # the cell each move filled
    played = [
        int(np.flatnonzero(board[t + 1] != board[t])[0]) for t in range(num_moves)
    ]
    records = (frames.p1_search, frames.p2_search)
    at = lambda records, t: jax.tree.map(lambda x: x[t], records)
    for t in range(num_moves):
        mover, other = (0, 1) if t % 2 == 0 else (1, 0)
        searched = at(records[mover], t + 1)
        # the mover's record is its search of the position, at least a search's worth
        assert searched.n[0] >= (big if mover == 0 else small).num_iterations
        carried = at(records[other], t + 1)
        if t == 0:
            # the other player hasn't searched yet
            assert carried.num_nodes == 0
            continue
        # the other's record is the subtree it kept from its search of the position before,
        # under the move it played
        before = at(records[other], t)
        child = np.flatnonzero(
            (before.parents == 0) & (before.actions == played[t - 1])
        )[0]
        assert carried.n[0] == before.n[child]
        assert carried.q[0] == before.q[child]


def test_tester_episode_has_every_frame_up_to_the_final_state(
    fixed_length_env, scripted
):
    max_steps = 4
    env = fixed_length_env(length=max_steps)
    packed = []

    def episode_fn(frames, p_ids, env_step_fn):  # pylint: disable=unused-argument
        packed.append(frames)
        return "episode"

    tester = TwoPlayerTester(num_episodes=2, episode_fn=episode_fn)
    state = TwoPlayerTestState(best_params={"w": jnp.zeros(3)})

    _, _, episode = tester.run(
        key=jax.random.PRNGKey(0),
        epoch_num=0,
        max_steps=max_steps,
        env_step_fn=env.step_fn,
        env_init_fn=env.init_fn,
        evaluator=scripted.first_legal,
        state=state,
        params={"w": jnp.ones(3)},
    )

    assert episode == "episode"
    [frames] = packed
    # one transfer off the device: numpy arrays stacked along time
    assert isinstance(frames.completed, np.ndarray)
    assert frames.env_state.step.tolist() == list(range(max_steps + 1))
    assert frames.completed[-1]
