from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from core import sharding
from core.common import two_player_game
from core.testing.two_player_tester import TwoPlayerTester, TwoPlayerTestState
from core.testing.utils import render_pgx_2p

MAX_STEPS = 10


def device_copies(x):
    """Each device's copy of a replicated array."""
    return [np.asarray(shard.data) for shard in x.addressable_shards]


def test_two_player_tester_keeps_best_params_in_sync_across_devices(ttt, scripted):
    assert jax.local_device_count() >= 2
    mesh = sharding.make_mesh(2)
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
    keys = sharding.shard(
        jnp.stack([win_key, loss_key]), mesh
    )  # one episode per device

    tester = TwoPlayerTester(num_episodes=2)
    params = sharding.replicate({"w": jnp.ones(3)}, mesh)
    state = TwoPlayerTestState(
        best_params=sharding.replicate({"w": jnp.zeros(3)}, mesh)
    )

    state, metrics, _, _ = tester.test_on_devices(
        mesh,
        MAX_STEPS,
        ttt.step_fn,
        ttt.init_fn,
        scripted.first_legal,
        keys,
        state,
        params,
    )

    # the setup gives the devices opposite results, so the mean over all episodes is 0: keep the old params
    np.testing.assert_array_equal(metrics["TwoPlayerTester_avg_outcome"], [1.0, -1.0])
    for best in device_copies(state.best_params["w"]):
        np.testing.assert_array_equal(best, np.zeros(3))


def test_two_player_tester_adopts_params_on_every_device_when_overall_mean_is_positive(
    ttt, scripted
):
    assert jax.local_device_count() >= 2
    mesh = sharding.make_mesh(2)
    # device 0 wins one game and loses one, device 1 wins both: device 0's own mean is 0,
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
    keys = sharding.shard(
        jnp.stack([win_key, loss_key, win_key, win_key]), mesh
    )  # two episodes per device

    tester = TwoPlayerTester(num_episodes=4)
    params = sharding.replicate({"w": jnp.ones(3)}, mesh)
    state = TwoPlayerTestState(
        best_params=sharding.replicate({"w": jnp.zeros(3)}, mesh)
    )

    state, metrics, _, _ = tester.test_on_devices(
        mesh,
        MAX_STEPS,
        ttt.step_fn,
        ttt.init_fn,
        scripted.first_legal,
        keys,
        state,
        params,
    )

    np.testing.assert_array_equal(metrics["TwoPlayerTester_avg_outcome"], [0.0, 1.0])
    for best in device_copies(state.best_params["w"]):
        np.testing.assert_array_equal(best, np.ones(3))


def test_tester_run_on_skipped_epoch_returns_state_unchanged(ttt, scripted):
    tester = TwoPlayerTester(num_episodes=2, epochs_per_test=2)
    state = TwoPlayerTestState(best_params={"w": jnp.zeros(3)})

    new_state, metrics, rendered = tester.run(
        key=jax.random.PRNGKey(0),
        epoch_num=1,
        max_steps=MAX_STEPS,
        num_devices=2,
        env_step_fn=ttt.step_fn,
        env_init_fn=ttt.init_fn,
        evaluator=scripted.first_legal,
        state=state,
        params={"w": jnp.ones(3)},
    )

    assert new_state is state
    assert metrics == {}
    assert rendered is None


def test_render_pgx_2p_handles_svgs_with_a_viewbox(tmp_path):
    # pgx's own SVGs only set width/height; an SVG with a viewBox used to hit an unbound `original_width`
    class ViewBoxState:
        current_player = 0

        def save_svg(self, path, color_theme):  # pylint: disable=unused-argument
            with open(path, "w") as f:
                f.write(
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50" height="50">'
                    '<rect width="100" height="50" fill="black"/></svg>'
                )

    frames = [
        SimpleNamespace(
            env_state=ViewBoxState(),
            completed=np.array(done),
            outcomes=np.array([1.0, -1.0]),
            p1_value_estimate=0.5,
            p2_value_estimate=-0.5,
        )
        for done in (False, True)
    ]

    render_pgx_2p(frames, p_ids=[0, 1], title="viewbox", frame_dir=str(tmp_path))

    assert (tmp_path / "viewbox.gif").exists()
