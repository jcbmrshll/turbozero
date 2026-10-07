from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from core.common import two_player_game
from core.testing.two_player_tester import TwoPlayerTester, TwoPlayerTestState

MAX_STEPS = 10


def replicate(tree, num_devices):
    return jax.tree.map(lambda x: jnp.stack([x] * num_devices), tree)


def test_two_player_tester_keeps_best_params_in_sync_across_devices(ttt, scripted):
    assert jax.local_device_count() >= 2
    # both sides take the lowest free square, so whoever moves first wins: pick one game key the
    # current params win and one they lose, and give one to each device
    game = jax.vmap(partial(two_player_game, evaluator_1=scripted.first_legal, evaluator_2=scripted.first_legal,
                            params_1=None, params_2=None, env_step_fn=ttt.step_fn, env_init_fn=ttt.init_fn,
                            max_steps=MAX_STEPS))
    candidates = jax.random.split(jax.random.PRNGKey(0), 16)
    outcomes, _, _ = game(candidates)
    win_key = candidates[np.flatnonzero(outcomes[:, 0] == 1)[0]]
    loss_key = candidates[np.flatnonzero(outcomes[:, 0] == -1)[0]]
    keys = jnp.stack([win_key, loss_key])[:, None]  # (devices, episodes per device, 2)

    tester = TwoPlayerTester(num_episodes=2)
    params = replicate({"w": jnp.ones(3)}, 2)
    state = TwoPlayerTestState(best_params=replicate({"w": jnp.zeros(3)}, 2))

    state, metrics, _, _ = tester.test(MAX_STEPS, ttt.step_fn, ttt.init_fn, scripted.first_legal, keys, state, params)

    # the setup gives the devices opposite results
    np.testing.assert_array_equal(metrics["TwoPlayerTester_avg_outcome"], [1.0, -1.0])
    best = state.best_params["w"]
    np.testing.assert_array_equal(best[0], best[1])


def test_two_player_tester_adopts_params_on_every_device_when_overall_mean_is_positive(ttt, scripted):
    assert jax.local_device_count() >= 2
    # device 0 wins both its games, device 1 wins one and loses one: device 1's own mean is 0,
    # but the mean over all episodes is positive, so both devices should adopt the new params
    game = jax.vmap(partial(two_player_game, evaluator_1=scripted.first_legal, evaluator_2=scripted.first_legal,
                            params_1=None, params_2=None, env_step_fn=ttt.step_fn, env_init_fn=ttt.init_fn,
                            max_steps=MAX_STEPS))
    candidates = jax.random.split(jax.random.PRNGKey(0), 16)
    outcomes, _, _ = game(candidates)
    win_key = candidates[np.flatnonzero(outcomes[:, 0] == 1)[0]]
    loss_key = candidates[np.flatnonzero(outcomes[:, 0] == -1)[0]]
    keys = jnp.stack([jnp.stack([win_key, win_key]), jnp.stack([win_key, loss_key])])

    tester = TwoPlayerTester(num_episodes=4)
    params = replicate({"w": jnp.ones(3)}, 2)
    state = TwoPlayerTestState(best_params=replicate({"w": jnp.zeros(3)}, 2))

    state, metrics, _, _ = tester.test(MAX_STEPS, ttt.step_fn, ttt.init_fn, scripted.first_legal, keys, state, params)

    np.testing.assert_array_equal(metrics["TwoPlayerTester_avg_outcome"], [1.0, 0.0])
    np.testing.assert_array_equal(state.best_params["w"], np.ones((2, 3)))
