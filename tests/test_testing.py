import xml.etree.ElementTree as ET
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.common import two_player_game
from core.evaluators.random_evaluator import RandomEvaluator
from core.monitor.renderers import _caption
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.testing.two_player_tester import TwoPlayerTester, TwoPlayerTestState

MAX_STEPS = 10


def win_and_loss_keys(ttt, scripted):
    """Game keys the current params win and lose.

    Both sides take the lowest free square, so whoever moves first wins."""
    game = jax.vmap(
        partial(
            two_player_game,
            evaluator_1=scripted.first_legal,
            evaluator_2=scripted.first_legal,
            params_1=None,
            params_2=None,
            env_step_fn=ttt.step_fn,
            env_init_fn=ttt.init_fn,
            max_steps=MAX_STEPS,
        )
    )
    candidates = jax.random.split(jax.random.PRNGKey(0), 16)
    outcomes, _, _ = game(candidates)
    win_key = candidates[np.flatnonzero(outcomes[:, 0] == 1)[0]]
    loss_key = candidates[np.flatnonzero(outcomes[:, 0] == -1)[0]]
    return win_key, loss_key


def test_two_player_tester_keeps_best_params_when_mean_outcome_is_not_positive(
    ttt, scripted
):
    win_key, loss_key = win_and_loss_keys(ttt, scripted)
    keys = jnp.stack([win_key, loss_key])

    tester = TwoPlayerTester(num_episodes=2)
    state = TwoPlayerTestState(best_params={"w": jnp.zeros(3)})

    state, metrics, _, _ = tester.test(
        MAX_STEPS,
        ttt.step_fn,
        ttt.init_fn,
        scripted.first_legal,
        keys,
        state,
        {"w": jnp.ones(3)},
    )

    np.testing.assert_array_equal(metrics["TwoPlayerTester_avg_outcome"], 0.0)
    np.testing.assert_array_equal(state.best_params["w"], np.zeros(3))


def test_two_player_tester_adopts_params_when_mean_outcome_is_positive(ttt, scripted):
    win_key, loss_key = win_and_loss_keys(ttt, scripted)
    # the params lose the first game, but win on average
    keys = jnp.stack([loss_key, win_key, win_key])

    tester = TwoPlayerTester(num_episodes=3)
    state = TwoPlayerTestState(best_params={"w": jnp.zeros(3)})

    state, metrics, _, _ = tester.test(
        MAX_STEPS,
        ttt.step_fn,
        ttt.init_fn,
        scripted.first_legal,
        keys,
        state,
        {"w": jnp.ones(3)},
    )

    np.testing.assert_allclose(metrics["TwoPlayerTester_avg_outcome"], 1 / 3)
    np.testing.assert_array_equal(state.best_params["w"], np.ones(3))


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


def test_caption_handles_svgs_with_a_viewbox():
    # pgx's own SVGs only set width/height; an SVG with a viewBox (and no width) must work too
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50" height="50">'
        b'<rect width="100" height="50" fill="black"/></svg>'
    )

    root = ET.fromstring(_caption(svg, ["agent", "opponent"]))

    # the viewBox and height grow by a fifth, for a strip holding a line per player
    assert root.attrib["viewBox"] == "0.0 0.0 100.0 60.0"
    assert float(root.attrib["height"]) == 60.0
    texts = [el.text for el in root.iter("{http://www.w3.org/2000/svg}text")]
    assert texts == ["agent", "opponent"]


def test_caption_handles_pgx_svgs():
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="240.0" height="240.0">'
        b'<rect width="240" height="240" fill="black"/></svg>'
    )

    root = ET.fromstring(_caption(svg, ["agent", "opponent"]))

    assert float(root.attrib["height"]) == 288.0
    assert "viewBox" not in root.attrib


def test_random_evaluator_plays_legal_moves_uniformly(ttt):
    evaluator = RandomEvaluator()
    state, meta = ttt.play([0, 4])
    eval_state = evaluator.init()
    keys = jax.random.split(jax.random.PRNGKey(0), 7000)

    actions = jax.vmap(
        lambda k: evaluator.evaluate(k, eval_state, state, root_metadata=meta).action
    )(keys)

    counts = np.bincount(np.asarray(actions), minlength=ttt.num_actions)
    assert counts[0] == counts[4] == 0
    # 7 legal squares, 1000 picks each expected
    np.testing.assert_allclose(counts[np.asarray(meta.action_mask)], 1000, rtol=0.1)


def test_two_player_baseline_reports_win_and_loss_rates(ttt, scripted):
    # both sides take the lowest free square, so whoever moves first wins: the agent wins the
    # first two episodes and loses the third
    game = jax.vmap(
        partial(
            two_player_game,
            evaluator_1=scripted.first_legal,
            evaluator_2=scripted.first_legal,
            params_1=None,
            params_2=None,
            env_step_fn=ttt.step_fn,
            env_init_fn=ttt.init_fn,
            max_steps=MAX_STEPS,
        )
    )
    candidates = jax.random.split(jax.random.PRNGKey(0), 16)
    outcomes, _, _ = game(candidates)
    win_key = candidates[np.flatnonzero(outcomes[:, 0] == 1)[0]]
    loss_key = candidates[np.flatnonzero(outcomes[:, 0] == -1)[0]]
    keys = jnp.stack([win_key, win_key, loss_key])

    tester = TwoPlayerBaseline(
        num_episodes=3, baseline_evaluator=scripted.first_legal, name="baseline"
    )
    _, metrics, _, _ = tester.test(
        MAX_STEPS,
        ttt.step_fn,
        ttt.init_fn,
        scripted.first_legal,
        keys,
        tester.init(params=None),
        {"w": jnp.ones(3)},
    )

    np.testing.assert_allclose(metrics["baseline_win_rate"], 2 / 3)
    np.testing.assert_allclose(metrics["baseline_loss_rate"], 1 / 3)


def test_two_player_game_with_a_fixed_first_player(ttt, scripted):
    # both sides take the lowest free square, so whoever moves first wins, whatever the key
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    for p1_first, expected in ((True, 1.0), (False, -1.0)):
        outcomes, _, _ = jax.vmap(
            partial(
                two_player_game,
                evaluator_1=scripted.first_legal,
                evaluator_2=scripted.first_legal,
                params_1=None,
                params_2=None,
                env_step_fn=ttt.step_fn,
                env_init_fn=ttt.init_fn,
                max_steps=MAX_STEPS,
                p1_first=p1_first,
            )
        )(keys)
        np.testing.assert_array_equal(outcomes[:, 0], expected)


def test_two_player_baseline_balances_the_first_player(ttt, scripted):
    # whoever moves first wins, so with the agent first in exactly half the games it wins half
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    tester = TwoPlayerBaseline(
        num_episodes=8,
        baseline_evaluator=scripted.first_legal,
        balance_first_player=True,
        name="baseline",
    )
    _, metrics, _, _ = tester.test(
        MAX_STEPS,
        ttt.step_fn,
        ttt.init_fn,
        scripted.first_legal,
        keys,
        tester.init(params=None),
        {"w": jnp.ones(3)},
    )

    np.testing.assert_array_equal(metrics["baseline_win_rate"], 0.5)
    np.testing.assert_array_equal(metrics["baseline_loss_rate"], 0.5)


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
