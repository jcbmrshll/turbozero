"""Hex's game.py: pgx's Hex at any size, and that its symmetries are exact on real games."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from example_modules import load_example

game_module, openings_module = load_example("hex", "game", "openings")
HexGame = game_module.HexGame

from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.memory.replay_memory import BaseExperience
from core.monitoring.episodes import BOARDS

SIZE = 5
GAME = HexGame(SIZE)
STEP = jax.jit(GAME.env.step)
INIT = jax.jit(GAME.env.init)
ROTATE, REFLECT, ROTATE_REFLECT = GAME.symmetry_transform_fns


def play(moves, game=GAME, step=STEP, init=INIT):
    state = init(jax.random.PRNGKey(0))
    for move in moves:
        state = step(state, move)
    return state


def experience(state) -> BaseExperience:
    mask = jnp.asarray(state.legal_action_mask)
    return BaseExperience(
        reward=jnp.array([1.0, -1.0]),
        policy_weights=jnp.where(mask, jnp.arange(mask.shape[0], dtype=float), 0.0),
        policy_mask=mask,
        observation_nn=state.observation,
        cur_player_id=state.current_player,
        search_value=jnp.array(0.5),
    )


def random_moves(rng, state):
    return int(rng.choice(np.flatnonzero(np.asarray(state.legal_action_mask))))


@pytest.mark.parametrize("size", [3, 7, 9, 11, 13])
def test_any_size_has_its_shapes(size):
    game = HexGame(size)
    assert game.env.observation_shape == (size, size, 4)
    assert game.env.num_actions == size * size + 1
    state = game.env.init(jax.random.PRNGKey(0))
    assert state._x.board.shape == (size * size,)
    assert state.legal_action_mask.tolist() == [True] * size * size + [False]


def test_who_connects_what():
    # colour 0 fills the left column, top to bottom, which connects its sides
    state = play([0, 4, 5, 9, 10, 14, 15, 19, 20])
    assert bool(state.terminated)
    assert state.rewards[state._player_order[0]] == 1
    # along the top row, colour 1 connects left to right, with colour 0 elsewhere
    state = play([20, 0, 21, 1, 22, 2, 23, 3, 12, 4])
    assert bool(state.terminated)
    assert state.rewards[state._player_order[1]] == 1
    # the short diagonal is a chain of neighbours, and the long diagonal isn't
    assert bool(play([4, 0, 8, 1, 12, 2, 16, 3, 20]).terminated)
    assert not bool(play([0, 1, 6, 2, 12, 3, 18, 4, 24]).terminated)


def test_a_game_always_ends_and_nobody_draws():
    rng = np.random.default_rng(0)
    for _ in range(5):
        state = INIT(jax.random.PRNGKey(0))
        plies = 0
        while not state.terminated:
            state = STEP(state, random_moves(rng, state))
            plies += 1
        assert plies <= GAME.max_moves
        assert sorted(np.asarray(state.rewards).tolist()) == [-1, 1]


def test_swap_is_legal_only_on_the_second_move_and_reflects_the_stone():
    assert not bool(play([])._x.step_count) and not bool(
        play([]).legal_action_mask[GAME.swap]
    )
    state = play([7])
    assert bool(state.legal_action_mask[GAME.swap])
    assert bool(state.observation[0, 0, 3])
    # (1, 2) -> (2, 1): the opponent's now, with colour 0 to move again
    state = STEP(state, GAME.swap)
    assert GAME.reflected(7) == 11
    assert np.argwhere(np.asarray(state.observation[..., 1])).tolist() == [[2, 1]]
    assert not np.asarray(state.observation[..., 0]).any()
    assert not bool(state.observation[0, 0, 2])
    assert not bool(state.observation[0, 0, 3])
    assert not bool(state.legal_action_mask[GAME.swap])
    # the cell it left is free again, and the one it took isn't
    assert bool(state.legal_action_mask[7]) and not bool(state.legal_action_mask[11])
    # a stone on the diagonal stays where it is
    state = STEP(play([6]), GAME.swap)
    assert np.argwhere(np.asarray(state.observation[..., 1])).tolist() == [[1, 1]]


def test_a_swap_on_a_real_game_keeps_playing_to_a_connection():
    # colour 1 swaps c1, so owns a3, on its left-right row 3
    state = play([2, GAME.swap, 0, 11, 1, 12, 5, 13, 6, 14])
    assert bool(state.terminated)
    assert state.rewards[state._player_order[1]] == 1


def image_state(state, action_ids):
    """The state whose board is `state`'s through a symmetry: a cell's group id follows its
    cell, as each is named after a cell of its group."""
    board = np.asarray(state._x.board)
    image = np.zeros_like(board)
    image[np.arange(board.size)] = board[action_ids[: board.size]]
    # relabel each group by its id's image, keeping its sign
    inverse = np.argsort(action_ids[: board.size])
    new = np.sign(image) * (inverse[np.abs(image) - 1] + 1)
    new[image == 0] = 0
    return state.replace(_x=state._x._replace(board=jnp.asarray(new)))


def action_ids(transform_fn):
    """The action each action of the image comes from."""
    probe = BaseExperience(
        reward=jnp.zeros(2),
        policy_weights=jnp.arange(GAME.swap + 1.0),
        policy_mask=jnp.ones(GAME.swap + 1, bool),
        observation_nn=jnp.zeros((SIZE, SIZE, 4), bool),
        cur_player_id=jnp.int32(0),
        search_value=jnp.array(0.0),
    )
    return np.asarray(transform_fn(probe).policy_weights).astype(int)


@pytest.mark.parametrize("seed", range(8))
def test_the_half_turn_is_exact_on_real_games(seed):
    """A game and its image under the half turn, both played through pgx from the empty board,
    stay each other's image, a swap and the end of the game included."""
    rng = np.random.default_rng(seed)
    ids = action_ids(ROTATE)
    state, image = INIT(jax.random.PRNGKey(0)), INIT(jax.random.PRNGKey(0))
    swap_early = seed % 2 == 0
    while not state.terminated:
        mapped = ROTATE(experience(state))
        np.testing.assert_array_equal(mapped.observation_nn, image.observation)
        np.testing.assert_array_equal(mapped.policy_mask, image.legal_action_mask)
        if swap_early and state._step_count == 1:
            move = GAME.swap
        else:
            move = random_moves(rng, state)
        # the image plays the action whose image-action is `move`: argwhere(ids == move)
        image_move = int(np.flatnonzero(ids == move)[0])
        state, image = STEP(state, move), STEP(image, image_move)
    assert bool(image.terminated)
    np.testing.assert_array_equal(state.rewards, image.rewards)


@pytest.mark.parametrize(
    "transform_fn", [REFLECT, ROTATE_REFLECT], ids=["reflect", "rotate-reflect"]
)
@pytest.mark.parametrize("seed", range(8))
def test_a_reflection_with_the_colours_exchanged_is_exact_on_real_games(
    seed, transform_fn
):
    """From the third move on (when no more swap can happen), a game and its image, with the
    colours exchanged by being a move further on, both played through pgx, stay each other's
    image, whoever wins: the player to move in one is the player to move in the other."""
    rng = np.random.default_rng(seed)
    ids = action_ids(transform_fn)
    swapped = seed % 2 == 0
    state = play([rng.integers(GAME.swap)])
    state = STEP(state, GAME.swap if swapped else random_moves(rng, state))
    image = image_state(state, ids)
    # a move further on is the other colour's turn, with the other player id to move, and only
    # the colour decides the observation
    x = image._x._replace(step_count=image._x.step_count + 1)
    image = image.replace(
        _x=x,
        _step_count=image._step_count + 1,
        current_player=1 - image.current_player,
        legal_action_mask=GAME.env._game.legal_action_mask(x),
    )
    image = image.replace(observation=GAME.env.observe(image))
    mover, image_mover = state.current_player, image.current_player
    while True:
        mapped = transform_fn(experience(state))
        np.testing.assert_array_equal(mapped.observation_nn, image.observation)
        if state.terminated:
            break
        np.testing.assert_array_equal(mapped.policy_mask, image.legal_action_mask)
        mover = state.current_player
        image_mover = image.current_player
        move = random_moves(rng, state)
        state, image = (
            STEP(state, move),
            STEP(image, int(np.flatnonzero(ids == move)[0])),
        )
    assert bool(image.terminated)
    # the one who moved last won in both
    assert state.rewards[mover] == image.rewards[image_mover] == 1


def test_a_swap_is_its_own_position_reflected():
    """Taking a swap gives the position before it, reflected with the colours exchanged, but for
    the swap no longer being available: why reflection leaves the swap action where it is."""
    before = play([7])
    after = STEP(before, GAME.swap)
    mapped = np.asarray(REFLECT(experience(after)).observation_nn)
    expected = np.asarray(before.observation).copy()
    expected[..., 3] = False
    np.testing.assert_array_equal(mapped, expected)


def test_symmetries_form_the_group_they_should():
    rng = np.random.default_rng(0)
    state = play([3])
    for _ in range(6):
        state = STEP(state, random_moves(rng, state))
    sample = experience(state)

    def same(a, b):
        np.testing.assert_array_equal(a.observation_nn, b.observation_nn)
        np.testing.assert_array_equal(a.policy_weights, b.policy_weights)
        np.testing.assert_array_equal(a.policy_mask, b.policy_mask)

    for fn in GAME.symmetry_transform_fns:
        same(fn(fn(sample)), sample)
    same(ROTATE(REFLECT(sample)), ROTATE_REFLECT(sample))
    same(REFLECT(ROTATE(sample)), ROTATE_REFLECT(sample))
    # four different images of a position with no symmetry of its own
    images = {
        np.asarray(fn(sample).observation_nn).tobytes()
        for fn in [lambda x: x, *GAME.symmetry_transform_fns]
    }
    assert len(images) == 4
    # the value isn't touched
    for fn in GAME.symmetry_transform_fns:
        np.testing.assert_array_equal(fn(sample).reward, sample.reward)
        np.testing.assert_array_equal(fn(sample).search_value, sample.search_value)
        assert fn(sample).policy_weights[GAME.swap] == sample.policy_weights[GAME.swap]


def test_a_batch_of_samples_transforms_along_its_last_axes():
    state = play([3, 8])
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), experience(state))
    for fn in GAME.symmetry_transform_fns:
        one = fn(experience(state))
        two = fn(batch)
        np.testing.assert_array_equal(two.observation_nn[1], one.observation_nn)
        np.testing.assert_array_equal(two.policy_weights[0], one.policy_weights)


def test_shortest_path_eval():
    game = HexGame(3)
    env = game.env
    # colour 0 has the top two of the middle column; colour 1 the top row's corners
    state = env.init(jax.random.PRNGKey(0))
    for move in (1, 0, 4, 2):
        state = env.step(state, move)
    # colour 0, to move, needs one more stone; colour 1 needs three, along the bottom
    logits, value = game.shortest_path_eval(state.observation[None])
    assert logits.shape == (1, env.num_actions)
    # the cells that finish colour 0's path are the policy's favourites
    assert {6, 7} <= set(np.flatnonzero(logits[0][:9] == logits[0][:9].max()))
    assert value.shape == (1,)
    assert float(value[0]) == pytest.approx(2 / 3)
    # the same counts, from the other player's turn
    state = env.step(state, 3)
    _, value = game.shortest_path_eval(state.observation[None])
    assert float(value[0]) == pytest.approx(-2 / 3)


def test_cell_names():
    game = HexGame(9)
    assert game.cell_name(19) == "b3"
    assert game.action("b3") == 19
    assert game.action("B3") == 19
    assert game.action("i9") == 80
    assert game.cell_name(game.swap) == "swap"
    assert game.action("swap") == game.swap
    eleven = HexGame(11)
    assert eleven.cell_name(120) == "k11"
    assert eleven.action("k11") == 120


def test_load_checkpoint_names_the_net_on_mismatch(tmp_path):
    import equinox as eqx

    game = HexGame(5)
    network, network_state = game.make_network("resnet:1x4")
    params = eqx.filter(network, eqx.is_inexact_array)
    path = tmp_path / "0.eqx"
    eqx.tree_serialise_leaves(path, (params, network_state))
    _, (loaded, _) = game.load_checkpoint(str(path), "resnet:1x4")
    assert jax.tree.all(jax.tree.map(lambda a, b: (a == b).all(), loaded, params))
    with pytest.raises(ValueError, match="doesn't hold a resnet:1x4,heads=pooled"):
        game.load_checkpoint(str(path), "resnet:1x4,heads=pooled")


def test_board_for_the_views():
    moves = [7, GAME.swap, 0]
    state = STEP(STEP(STEP(INIT(jax.random.PRNGKey(0)), 7), GAME.swap), 0)
    batch = jax.tree.map(lambda x: np.asarray(x)[None], state._x)
    board, to_move = BOARDS["hex"](batch)
    assert board.shape == (1, SIZE, SIZE)
    assert int(to_move[0]) == 1
    # colour 0's a1, colour 1's stone reflected from c2 to b3
    assert board[0, 0, 0] == 1
    assert board[0, 2, 1] == 2
    assert (board != 0).sum() == len(moves) - 1


def test_openings_start_games_after_their_moves(tmp_path):
    game = HexGame(5)
    path = tmp_path / "openings.txt"
    path.write_text("c3\na1b2\n")
    openings = openings_module.load_openings(game, path)
    assert openings.tolist() == [[12, openings_module.NO_MOVE], [0, 6]]
    init_fn = openings_module.make_opening_init_fn(game, openings, 0.0)
    plies = {int(init_fn(jax.random.PRNGKey(i))[0]._step_count) for i in range(16)}
    assert plies == {1, 2}
    # all from the empty board
    init_fn = openings_module.make_opening_init_fn(game, openings, 1.0)
    assert int(init_fn(jax.random.PRNGKey(0))[0]._step_count) == 0


@pytest.mark.parametrize("size", [7, 9, 11])
def test_the_opening_lists_are_legal_and_cover_every_first_stone(size):
    game = HexGame(size)
    openings = openings_module.load_openings(game)
    one = openings[openings[:, 1] == openings_module.NO_MOVE, 0]
    assert sorted(one.tolist()) == list(range(size * size))
    two = openings[openings[:, 1] != openings_module.NO_MOVE]
    assert len(two) > 0
    assert (two[:, 0] != two[:, 1]).all()
    assert Path(openings_module.openings_path(size)).exists()


def test_one_tiny_search_runs_on_hex_with_a_network():
    game = HexGame(3)
    network, network_state = game.make_network("resnet:1x4")
    import equinox as eqx

    params = (eqx.filter(network, eqx.is_inexact_array), network_state)
    evaluator = game.make_test_evaluator(
        make_nn_eval_fn(network, game_module.state_to_nn_input), 4
    )
    state, meta = game.init_fn(jax.random.PRNGKey(0))
    tree = evaluator.init(template_embedding=state)
    out = evaluator.evaluate(
        key=jax.random.PRNGKey(1),
        eval_state=tree,
        env_state=state,
        root_metadata=meta,
        params=params,
        env_step_fn=game.step_fn,
    )
    assert bool(state.legal_action_mask[int(out.action)])
