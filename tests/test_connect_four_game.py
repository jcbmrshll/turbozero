from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from example_modules import load_example

from core.memory.replay_memory import BaseExperience

(game,) = load_example("connect_four", "game")


def save_checkpoint(path: Path, net: str):
    """A checkpoint as the trainer saves one: params and network state first, then the
    optimizer's state."""
    network, network_state = game.make_network(net)
    params = eqx.filter(network, eqx.is_inexact_array)
    opt_state = game.make_optimizer(total_steps=1).init(params)
    eqx.tree_serialise_leaves(path, (params, network_state, opt_state))
    return params


def test_load_checkpoint_loads_its_net(tmp_path):
    path = tmp_path / "0.eqx"
    saved = save_checkpoint(path, "resnet:1x4")
    _, (params, _) = game.load_checkpoint(str(path), "resnet:1x4")
    assert jax.tree.all(jax.tree.map(lambda a, b: (a == b).all(), params, saved))


def test_load_checkpoint_names_the_net_on_mismatch(tmp_path):
    path = tmp_path / "0.eqx"
    save_checkpoint(path, "resnet:1x4")
    with pytest.raises(ValueError, match="doesn't hold a resnet:1x4,pool_every=1"):
        game.load_checkpoint(str(path), "resnet:1x4,pool_every=1,pool_channels=2")


def test_env_is_connect_four():
    assert game.env.observation_shape == (6, 7, 2)
    assert game.env.num_actions == 7


def experience(state, weights=None) -> BaseExperience:
    mask = state.legal_action_mask
    return BaseExperience(
        reward=jnp.array([1.0, -1.0]),
        policy_weights=jnp.where(mask, jnp.arange(7.0), 0.0)
        if weights is None
        else weights,
        policy_mask=mask,
        observation_nn=state.observation,
        cur_player_id=state.current_player,
        search_value=jnp.array(0.5),
    )


def test_mirror_is_exactly_the_game_played_in_mirrored_columns():
    rng = np.random.default_rng(0)
    step = jax.jit(game.env.step)
    key = jax.random.PRNGKey(0)
    state, mirrored = game.env.init(key), game.env.init(key)
    # a whole game, stacking discs high so that some columns fill up
    while not state.terminated:
        image = game.mirror(experience(state))
        np.testing.assert_array_equal(image.observation_nn, mirrored.observation)
        np.testing.assert_array_equal(image.policy_mask, mirrored.legal_action_mask)
        move = rng.choice(np.flatnonzero(state.legal_action_mask))
        # the action's weight goes with its column
        assert image.policy_weights[6 - move] == experience(state).policy_weights[move]
        state, mirrored = step(state, int(move)), step(mirrored, int(6 - move))
    assert (state.rewards == mirrored.rewards).all()


def test_mirror_is_its_own_inverse_and_leaves_the_rest_alone():
    key = jax.random.PRNGKey(0)
    state = game.env.init(key)
    for move in (0, 1, 1, 2):
        state = game.env.step(state, move)
    sample = experience(state)
    image = game.mirror(sample)
    # a position that isn't its own mirror image
    assert not (image.observation_nn == sample.observation_nn).all()
    twice = game.mirror(image)
    for field in ("observation_nn", "policy_mask", "policy_weights"):
        np.testing.assert_array_equal(getattr(twice, field), getattr(sample, field))
    for field in ("reward", "cur_player_id", "search_value"):
        np.testing.assert_array_equal(getattr(image, field), getattr(sample, field))
    assert game.SYMMETRY_TRANSFORM_FNS == [game.mirror]


def test_mirror_maps_a_batch():
    state = jax.vmap(game.env.init)(jax.random.split(jax.random.PRNGKey(0), 3))
    weights = jnp.tile(jnp.arange(7.0), (3, 1))
    image = game.mirror(jax.vmap(experience)(state, weights))
    np.testing.assert_array_equal(image.policy_weights[0], np.arange(7.0)[::-1])
    assert image.observation_nn.shape == (3, 6, 7, 2)


def test_episode_metrics_says_whether_a_game_was_drawn():
    assert game.episode_metrics(jnp.zeros(2))["draw"]
    assert not game.episode_metrics(jnp.array([1.0, -1.0]))["draw"]
