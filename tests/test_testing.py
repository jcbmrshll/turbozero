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


def replicate(tree, num_devices):
    return jax.tree.map(lambda x: jnp.stack([x] * num_devices), tree)


def test_two_player_tester_keeps_best_params_in_sync_across_devices(ttt, scripted):
    assert jax.local_device_count() >= 2
    # both sides take the lowest free square, so whoever moves first wins: pick one game key the
    # current params win and one they lose, and give one to each device
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
    keys = jnp.stack([win_key, loss_key])[:, None]  # (devices, episodes per device, 2)

    tester = TwoPlayerTester(num_episodes=2)
    params = replicate({"w": jnp.ones(3)}, 2)
    state = TwoPlayerTestState(best_params=replicate({"w": jnp.zeros(3)}, 2))

    state, metrics, _, _ = tester.test(
        MAX_STEPS, ttt.step_fn, ttt.init_fn, scripted.first_legal, keys, state, params
    )

    # the setup gives the devices opposite results
    np.testing.assert_array_equal(metrics["TwoPlayerTester_avg_outcome"], [1.0, -1.0])
    best = state.best_params["w"]
    np.testing.assert_array_equal(best[0], best[1])


def test_two_player_tester_adopts_params_on_every_device_when_overall_mean_is_positive(
    ttt, scripted
):
    assert jax.local_device_count() >= 2
    # device 0 wins both its games, device 1 wins one and loses one: device 1's own mean is 0,
    # but the mean over all episodes is positive, so both devices should adopt the new params
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
    keys = jnp.stack([jnp.stack([win_key, win_key]), jnp.stack([win_key, loss_key])])

    tester = TwoPlayerTester(num_episodes=4)
    params = replicate({"w": jnp.ones(3)}, 2)
    state = TwoPlayerTestState(best_params=replicate({"w": jnp.zeros(3)}, 2))

    state, metrics, _, _ = tester.test(
        MAX_STEPS, ttt.step_fn, ttt.init_fn, scripted.first_legal, keys, state, params
    )

    np.testing.assert_array_equal(metrics["TwoPlayerTester_avg_outcome"], [1.0, 0.0])
    np.testing.assert_array_equal(state.best_params["w"], np.ones((2, 3)))


def test_tester_run_on_skipped_epoch_returns_state_unchanged(ttt, scripted):
    tester = TwoPlayerTester(num_episodes=2, epochs_per_test=2)
    state = TwoPlayerTestState(best_params=replicate({"w": jnp.zeros(3)}, 2))

    new_state, metrics, rendered = tester.run(
        key=jax.random.PRNGKey(0),
        epoch_num=1,
        max_steps=MAX_STEPS,
        num_devices=2,
        env_step_fn=ttt.step_fn,
        env_init_fn=ttt.init_fn,
        evaluator=scripted.first_legal,
        state=state,
        params=replicate({"w": jnp.ones(3)}, 2),
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
    assert jax.local_device_count() >= 2
    # both sides take the lowest free square, so whoever moves first wins: device 0 gets a key
    # the agent wins, device 1 one it loses
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
    keys = jnp.stack([win_key, loss_key])[:, None]

    tester = TwoPlayerBaseline(
        num_episodes=2, baseline_evaluator=scripted.first_legal, name="baseline"
    )
    _, metrics, _, _ = tester.test(
        MAX_STEPS,
        ttt.step_fn,
        ttt.init_fn,
        scripted.first_legal,
        keys,
        replicate(tester.init(params=None), 2),
        replicate({"w": jnp.ones(3)}, 2),
    )

    np.testing.assert_array_equal(metrics["baseline_win_rate"], [1.0, 0.0])
    np.testing.assert_array_equal(metrics["baseline_loss_rate"], [0.0, 1.0])


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
    keys = jax.random.split(jax.random.PRNGKey(0), 8).reshape(2, 4, -1)
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
        replicate(tester.init(params=None), 2),
        replicate({"w": jnp.ones(3)}, 2),
    )

    np.testing.assert_array_equal(metrics["baseline_win_rate"], [0.5, 0.5])
    np.testing.assert_array_equal(metrics["baseline_loss_rate"], [0.5, 0.5])


def test_balanced_two_player_baseline_needs_an_even_number_of_episodes_per_device(
    scripted,
):
    tester = TwoPlayerBaseline(
        num_episodes=6,
        baseline_evaluator=scripted.first_legal,
        balance_first_player=True,
    )
    tester.check_size_compatibilities(3)  # 2 per device
    with pytest.raises(ValueError, match="even number of episodes per device"):
        tester.check_size_compatibilities(2)  # 3 per device
