import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from game import (
    PASS,
    SYMMETRY_TRANSFORM_FNS,
    env,
    load_checkpoint,
    make_network,
    make_optimizer,
)

from core.memory.replay_memory import BaseExperience


def save_checkpoint(path: Path, net: str):
    """A checkpoint as the trainer saves one: params and network state first, then the
    optimizer's state."""
    network, network_state = make_network(net)
    params = eqx.filter(network, eqx.is_inexact_array)
    opt_state = make_optimizer(total_steps=1).init(params)
    eqx.tree_serialise_leaves(path, (params, network_state, opt_state))
    return params


def test_load_checkpoint_loads_its_net(tmp_path):
    path = tmp_path / "0.eqx"
    saved = save_checkpoint(path, "resnet:1x4")
    _, (params, _) = load_checkpoint(str(path), "resnet:1x4")
    assert jax.tree.all(jax.tree.map(lambda a, b: (a == b).all(), params, saved))


@pytest.mark.parametrize(
    "trained, net",
    [
        ("resnet:1x4", "resnet:1x4,heads=pooled"),
        ("resnet:1x4", "resnet:1x4,pool_every=1,pool_channels=2"),
        ("resnet:1x4,pool_every=1,pool_channels=2,heads=pooled", "resnet:1x4"),
    ],
)
def test_load_checkpoint_names_the_net_on_mismatch(tmp_path, trained, net):
    path = tmp_path / "0.eqx"
    save_checkpoint(path, trained)
    with pytest.raises(ValueError, match=f"doesn't hold a {net} network.*--net"):
        load_checkpoint(str(path), net)


def legal_moves(obs: np.ndarray) -> np.ndarray:
    """The legal actions of a position, from its observation (the player to move's discs, then
    the opponent's): a move flanks a line of the opponent's discs; a pass, when there's none."""
    me, opp = obs[..., 0], obs[..., 1]
    legal = np.zeros(65, bool)
    directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    for r in range(8):
        for c in range(8):
            if me[r, c] or opp[r, c]:
                continue
            for dr, dc in directions:
                i, j, flanked = r + dr, c + dc, 0
                while 0 <= i < 8 and 0 <= j < 8 and opp[i, j]:
                    i, j, flanked = i + dr, j + dc, flanked + 1
                if flanked and 0 <= i < 8 and 0 <= j < 8 and me[i, j]:
                    legal[r * 8 + c] = True
    legal[PASS] = not legal[:PASS].any()
    return legal


def test_symmetries_map_a_position_to_a_consistent_one():
    # a position with no symmetry of its own, after a few random moves
    rng = np.random.default_rng(0)
    step = jax.jit(env.step)
    state = env.init(jax.random.PRNGKey(0))
    for _ in range(10):
        state = step(state, rng.choice(np.flatnonzero(state.legal_action_mask)))
    mask = np.asarray(state.legal_action_mask)
    np.testing.assert_array_equal(mask, legal_moves(np.asarray(state.observation)))
    experience = BaseExperience(
        reward=jnp.array([1.0, -1.0]),
        policy_weights=jnp.where(mask, jnp.arange(65.0), 0.0),
        policy_mask=jnp.asarray(mask),
        observation_nn=state.observation,
        cur_player_id=state.current_player,
        search_value=jnp.array(0.5),
    )
    # a disc on each square in turn, and each square's action
    squares = np.eye(64, dtype=bool).reshape(64, 8, 8)
    probes = BaseExperience(
        reward=jnp.zeros((64, 2)),
        policy_weights=jnp.tile(jnp.arange(65.0), (64, 1)),
        policy_mask=jnp.ones((64, 65), bool),
        observation_nn=jnp.asarray(np.stack([squares, ~squares], axis=-1)),
        cur_player_id=jnp.zeros(64, jnp.int32),
        search_value=jnp.zeros(64),
    )

    images = [experience]
    for transform_fn in SYMMETRY_TRANSFORM_FNS:
        image = transform_fn(experience)
        obs = np.asarray(image.observation_nn)
        # the legal moves of the board's image are the image of its legal moves
        np.testing.assert_array_equal(image.policy_mask, legal_moves(obs))
        np.testing.assert_array_equal(image.policy_weights[~image.policy_mask], 0.0)
        assert image.policy_weights[PASS] == experience.policy_weights[PASS]
        for field in ("reward", "cur_player_id", "search_value"):
            np.testing.assert_array_equal(
                getattr(image, field), getattr(experience, field)
            )
        # the square a disc moves to is the one whose action takes its square's
        moved = transform_fn(probes)
        to = np.asarray(moved.observation_nn[..., 0]).reshape(64, 64).argmax(axis=1)
        np.testing.assert_array_equal(
            np.asarray(moved.policy_weights)[np.arange(64), to], np.arange(64)
        )
        images.append(image)

    # the 8 symmetries give 8 different boards, and map each of them to another of them
    boards = {np.asarray(i.observation_nn).tobytes() for i in images}
    assert len(boards) == 8
    for transform_fn in SYMMETRY_TRANSFORM_FNS:
        for image in images:
            assert np.asarray(transform_fn(image).observation_nn).tobytes() in boards
