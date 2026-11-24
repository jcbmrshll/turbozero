import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from game import (
    MARGINS,
    PASS,
    RAY_BRANCHES,
    RAY_TAPS,
    SYMMETRY_TRANSFORM_FNS,
    WEIGHT_DECAY,
    env,
    load_checkpoint,
    make_network,
    make_optimizer,
    move_planes,
    scores,
    weight_decay_mask,
)

from core.memory.replay_memory import BaseExperience
from core.networks.azresnet import (
    AZResnet,
    BranchedDepthwise,
    Depthwise,
    GatedScan,
    PreActConvNeXt,
)


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


def test_lr_drops_hold_then_step():
    params = {"w": jnp.zeros((2, 2))}
    optimizer = make_optimizer(100, 1e-3, lr_drops=[(0.7, 1e-4), (0.9, 1e-5)])
    state: Any = optimizer.init(params)
    rates = []
    for _ in range(100):
        _, state = optimizer.update(jax.tree.map(jnp.ones_like, params), state, params)
        rates.append(float(state.hyperparams["learning_rate"]))
    assert rates[0] == pytest.approx(1e-3) and rates[69] == pytest.approx(1e-3)
    assert rates[70] == pytest.approx(1e-4) and rates[89] == pytest.approx(1e-4)
    assert rates[90] == pytest.approx(1e-5) and rates[99] == pytest.approx(1e-5)


def test_constant_rate_is_the_cosine_to_itself_and_its_program_ignores_run_length():
    # a run in steps (controller.py) compiles each step's training once into the persistent
    # compilation cache only if the program doesn't depend on --epochs; the updates, and the
    # optimizer state's structure (saved states load), are a cosine from the rate to itself's
    params = {"w": jnp.linspace(-1.0, 1.0, 12).reshape(3, 4), "b": jnp.ones(3)}
    grads = jax.tree.map(lambda x: jnp.sin(3 * x) + 0.1, params)

    def run(optimizer):
        update = jax.jit(optimizer.update)
        state: Any = optimizer.init(params)
        p = params
        for _ in range(3):
            updates, state = update(grads, state, p)
            p = optax.apply_updates(p, updates)
        return p, state, update.lower(grads, optimizer.init(params), p).as_text()

    constant = [run(make_optimizer(steps, 5e-4, 5e-4)) for steps in (100, 200)]
    cosine = optax.inject_hyperparams(optax.adamw, static_args=("mask",))(
        learning_rate=optax.cosine_decay_schedule(5e-4, decay_steps=100, alpha=1.0),
        weight_decay=WEIGHT_DECAY,
        mask=weight_decay_mask,
    )
    expected, expected_state, _ = run(cosine)
    assert constant[0][2] == constant[1][2]
    for p, state, _ in constant:
        assert jax.tree.structure(state) == jax.tree.structure(expected_state)
        for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(a, b)


def on_a_line(dy: int, dx: int) -> bool:
    return dy == 0 or dx == 0 or abs(dy) == abs(dx)


def test_ray_taps_are_the_lines_through_a_square():
    taps = set(RAY_TAPS)
    assert len(taps) == len(RAY_TAPS) == 57
    assert taps == {
        (dy, dx) for dy in range(-7, 8) for dx in range(-7, 8) if on_a_line(dy, dx)
    }


RAY_NET = (
    "resnet:2x8,heads=pooled,layout=pre,block=convnext,expand=2,"
    "pool_every=2,pool_channels=4,dw_taps=rays"
)


def test_a_spec_names_the_ray_taps():
    network, state = make_network(RAY_NET)
    assert isinstance(network, AZResnet)
    for block in network.blocks:
        assert isinstance(block, PreActConvNeXt)
        assert isinstance(block.dw, Depthwise) and block.dw.taps == RAY_TAPS
    obs = jnp.zeros((3, 8, 8, 2), bool)
    (policy, value), _ = jax.vmap(lambda o: network(o, state), axis_name="batch")(obs)
    assert policy.shape == (3, 65) and value.shape == (3, 1)
    with pytest.raises(ValueError, match="rays"):
        make_network(RAY_NET.replace("=rays", "=knights"))


def ray_filter(channels: int = 2, seed: int = 0) -> Depthwise:
    """A filter of the ray taps, every weight random and nonzero."""
    dw = Depthwise(channels, RAY_TAPS, key=jax.random.PRNGKey(seed))
    weight = np.random.default_rng(seed).uniform(0.5, 1.5, dw.weight.shape)
    return eqx.tree_at(lambda d: d.weight, dw, jnp.asarray(weight, jnp.float32))


def test_ray_filters_reach_a_square_from_exactly_its_lines():
    dw = ray_filter()
    for i, j in [(0, 0), (3, 4), (7, 2)]:
        x = np.zeros((2, 8, 8), np.float32)
        x[:, i, j] = 1
        y = np.asarray(dw(jnp.asarray(x), jnp.dtype("float32")))
        reached = {(a, b) for a, b in zip(*np.nonzero(y[0]), strict=True)}
        assert reached == {
            (a, b) for a in range(8) for b in range(8) if on_a_line(a - i, b - j)
        }


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("quarter_turns", range(4))
def test_a_symmetry_of_a_ray_filter_is_a_ray_filter(quarter_turns, transpose):
    # training maps positions through the board's symmetries (SYMMETRY_TRANSFORM_FNS): a ray
    # filter on the image of a board is the image of another ray filter on the board, its weights
    # the same, moved to the image's taps. The set of taps is the same under every symmetry; the
    # weights aren't tied, so the net isn't invariant, but the augmentation stays in its family
    def image(a):
        a = np.swapaxes(a, -2, -1) if transpose else a
        return np.rot90(a, quarter_turns, axes=(-2, -1))

    dw = ray_filter(seed=1)
    kernel = image(np.asarray(dw.dense())[:, 0])
    r = dw.kernel_size // 2
    off_taps = np.ones(kernel.shape[1:], bool)
    for dy, dx in RAY_TAPS:
        off_taps[dy + r, dx + r] = False
    assert not kernel[:, off_taps].any()
    moved = eqx.tree_at(
        lambda d: d.weight,
        dw,
        jnp.asarray(np.stack([kernel[:, dy + r, dx + r] for dy, dx in RAY_TAPS], 1)),
    )
    x = np.random.default_rng(0).normal(size=(2, 8, 8)).astype(np.float32)
    f32 = jnp.dtype("float32")

    np.testing.assert_allclose(
        moved(jnp.asarray(image(x).copy()), f32),
        image(np.asarray(dw(jnp.asarray(x), f32))),
        atol=1e-5,
    )


def position(mover: int, opponent: int, current_player: int = 0):
    """A state of `mover` discs for the player to move and `opponent` for the other, as `scores`
    reads one."""
    discs = np.zeros((64, 2), bool)
    discs[:mover, 0] = True
    discs[64 - opponent :, 1] = True
    return SimpleNamespace(
        observation=jnp.asarray(discs.reshape(8, 8, 2)),
        current_player=jnp.int32(current_player),
    )


@pytest.mark.parametrize(
    "mover, opponent, current_player, expected",
    [
        # the 14 empty squares go to the one ahead
        (30, 20, 0, (24.0, -24.0)),
        (20, 30, 1, (24.0, -24.0)),
        (32, 32, 0, (0.0, 0.0)),
        (0, 13, 0, (-64.0, 64.0)),
        (40, 24, 1, (-16.0, 16.0)),
    ],
)
def test_scores_are_the_margins_with_the_empty_squares_to_the_one_ahead(
    mover, opponent, current_player, expected
):
    np.testing.assert_array_equal(
        scores(position(mover, opponent, current_player)), expected
    )


def test_a_spec_names_a_value_head_over_the_final_margin():
    network, state = make_network("resnet:1x16,heads=pooled,outcomes=margin")
    assert isinstance(network, AZResnet)
    assert network.outcome_scores == MARGINS
    assert MARGINS[0] == 64 and MARGINS[-1] == -64 and len(MARGINS) == 65
    network = eqx.nn.inference_mode(network)
    obs = jnp.zeros((3, 8, 8, 2), bool)
    (_, value), _ = jax.vmap(lambda o: network(o, state), axis_name="batch")(obs)
    (_, logits), _ = jax.vmap(
        lambda o: network(o, state, outcome_logits=True), axis_name="batch"
    )(obs)
    assert value.shape == (3, 1) and logits.shape == (3, 65)
    with pytest.raises(ValueError, match="margin"):
        make_network("resnet:1x16,heads=pooled,outcomes=parity")


def test_move_planes_are_the_legal_moves_of_both_players():
    # along a random game: the first plane is the player to move's legal moves as pgx has them,
    # the second the first plane of the same board with the players' discs swapped
    rng = np.random.default_rng(0)
    state = env.init(jax.random.PRNGKey(0))
    moves = 0
    while not bool(state.terminated):
        planes = np.asarray(move_planes(state.observation))
        mask = np.asarray(state.legal_action_mask)
        np.testing.assert_array_equal(planes[..., 0].reshape(-1), mask[:64])
        swapped = np.asarray(move_planes(state.observation[..., ::-1]))
        np.testing.assert_array_equal(planes[..., 1], swapped[..., 0])
        state = env.step(state, int(rng.choice(np.flatnonzero(mask))))
        moves += 1
    assert moves > 50


def test_the_ray_branches_fold_into_the_ray_taps():
    assert set().union(*map(set, RAY_BRANCHES)) == set(RAY_TAPS)


@pytest.mark.parametrize(
    "setting, mixer",
    [("dw_branches=rays", BranchedDepthwise), ("scan=rays", GatedScan)],
)
def test_a_spec_names_othellos_steps_between_squares(setting, mixer):
    network, state = make_network(RAY_NET.replace("dw_taps=rays", setting))
    assert isinstance(network, AZResnet)
    for block in network.blocks:
        assert isinstance(block, PreActConvNeXt) and isinstance(block.dw, mixer)
    obs = jnp.zeros((3, 8, 8, 2), bool)
    (policy, value), _ = jax.vmap(lambda o: network(o, state), axis_name="batch")(obs)
    assert policy.shape == (3, 65) and value.shape == (3, 1)


def test_a_spec_names_the_legal_move_planes():
    network, state = make_network(RAY_NET + ",inputs=moves")
    assert isinstance(network, AZResnet)
    assert network.stem_conv.weight.shape[1] == 4
    obs = jnp.zeros((3, 8, 8, 2), bool)
    (policy, value), _ = jax.vmap(lambda o: network(o, state), axis_name="batch")(obs)
    assert policy.shape == (3, 65) and value.shape == (3, 1)
    with pytest.raises(ValueError, match="moves"):
        make_network(RAY_NET + ",inputs=stability")
