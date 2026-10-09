"""The Othello example's helpers: its board symmetries must map a position's legal moves to
the legal moves of the transformed position (search tree positions' too), and its XOT openings
must be legal games."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# the example's scripts import their sibling modules
sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "othello"))
import game as othello
from xot import load_xot, make_xot_init_fn

from core.evaluators.alphazero import AlphaZero
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.training.tree_positions import select_nodes, tree_experiences


def legal_moves(obs):
    """Legal moves for the player to move, computed from the observation alone."""
    mine, theirs = np.asarray(obs[..., 0]) > 0, np.asarray(obs[..., 1]) > 0
    legal = np.zeros(65, dtype=bool)
    for r in range(8):
        for c in range(8):
            if mine[r, c] or theirs[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc, flipped = r + dr, c + dc, 0
                    while 0 <= rr < 8 and 0 <= cc < 8 and theirs[rr, cc]:
                        rr, cc, flipped = rr + dr, cc + dc, flipped + 1
                    if flipped and 0 <= rr < 8 and 0 <= cc < 8 and mine[rr, cc]:
                        legal[r * 8 + c] = True
    legal[64] = not legal.any()
    return legal


def test_there_are_seven_distinct_symmetries():
    state = othello.env.init(jax.random.PRNGKey(0))
    for _ in range(6):
        state = othello.env.step(state, np.argmax(np.asarray(state.legal_action_mask)))
    boards = {
        np.asarray(
            fn(state.legal_action_mask, state.legal_action_mask, state)[2].observation
        ).tobytes()
        for fn in othello.SYMMETRY_TRANSFORM_FNS
    }
    boards.add(np.asarray(state.observation).tobytes())
    assert len(othello.SYMMETRY_TRANSFORM_FNS) == 7
    assert len(boards) == 8


def test_symmetries_map_legal_moves_and_policy_with_the_board():
    key = jax.random.PRNGKey(0)
    state = othello.env.init(key)
    step = jax.jit(othello.env.step)
    for _ in range(30):
        key, action_key = jax.random.split(key)
        mask = np.asarray(state.legal_action_mask)
        assert np.array_equal(legal_moves(state.observation), mask)
        policy = np.arange(65, dtype=np.float32)
        for fn in othello.SYMMETRY_TRANSFORM_FNS:
            t_mask, t_policy, t_state = fn(state.legal_action_mask, policy, state)
            assert np.array_equal(legal_moves(t_state.observation), np.asarray(t_mask))
            # the policy moves with the mask: each legal move keeps its weight
            assert np.array_equal(
                np.sort(np.asarray(t_policy)[np.asarray(t_mask)]), np.sort(policy[mask])
            )
            # passing stays the last action
            assert t_policy[64] == 64
        action = jax.random.choice(action_key, 65, p=mask / mask.sum())
        state = step(state, action)


def test_the_player_to_move_changes_every_step_passes_included():
    """MCTS keeps each node's value from the perspective of the player to move there, negating it
    from one level to the next (discount -1), so it relies on the players taking turns: in pgx's
    Othello a player with no move passes (action 64), and the turn passes too."""

    @jax.jit
    @jax.vmap
    def game(key):
        def step(carry, key):
            state, done = carry
            action = jax.random.categorical(
                key, jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
            )
            next_state = othello.env.step(state, action)
            out = (
                ~done & (next_state.current_player == state.current_player),
                ~done & (action == othello.PASS),
            )
            return (next_state, done | next_state.terminated), out

        state = othello.env.init(key)
        _, (same_player, passed) = jax.lax.scan(
            step, (state, jnp.array(False)), jax.random.split(key, 80)
        )
        return same_player, passed

    same_player, passed = game(jax.random.split(jax.random.PRNGKey(0), 64))
    assert passed.any()
    assert not same_player.any()


def test_tree_positions_symmetric_copies_are_legal():
    def uniform(state, params, key):
        return jnp.zeros((65,)), jnp.array(0.0)

    evaluator = AlphaZero(MCTS)(
        eval_fn=uniform,
        num_iterations=32,
        max_nodes=48,
        branching_factor=65,
        action_selector=PUCTSelector(),
    )
    state, metadata = make_xot_init_fn(load_xot()[:1], 0.0)(jax.random.PRNGKey(0))
    tree = jax.jit(evaluator.evaluate, static_argnames="env_step_fn")(
        key=jax.random.PRNGKey(0),
        eval_state=evaluator.init(template_embedding=state),
        env_state=state,
        root_metadata=metadata,
        params=None,
        env_step_fn=othello.step_fn,
    ).eval_state
    indices, valid = select_nodes(jax.random.PRNGKey(0), tree, num=4, min_visits=2)
    samples, sample_valid = tree_experiences(
        tree,
        indices,
        valid,
        othello.step_fn,
        othello.state_to_nn_input,
        othello.SYMMETRY_TRANSFORM_FNS,
    )
    samples = jax.tree.map(np.asarray, samples)
    assert valid.sum() == 4 and sample_valid.shape == (32,)

    def sample(i):
        return jax.tree.map(lambda x: x[i], samples)

    for i in range(4):
        original = sample(i)
        for copy in map(sample, range(i, 32, 4)):
            assert np.array_equal(legal_moves(copy.observation_nn), copy.policy_mask)
            assert not copy.policy_weights[~copy.policy_mask].any()
            # each legal move keeps its weight
            assert np.array_equal(
                np.sort(copy.policy_weights[copy.policy_mask]),
                np.sort(original.policy_weights[original.policy_mask]),
            )
            assert copy.search_value == original.search_value
            assert np.array_equal(copy.reward, original.reward)
            assert copy.cur_player_id == original.cur_player_id


def test_square_names_round_trip():
    assert othello.square(19) == "d3" and othello.action("D3") == 19
    assert (
        othello.square(othello.PASS) == "pass"
        and othello.action("pass") == othello.PASS
    )
    for a in range(65):
        assert othello.action(othello.square(a)) == a


def test_xot_openings_are_legal_eight_move_games():
    openings = load_xot()
    assert openings.shape == (10784, 8)
    # every opening, from the start position, playing each move only if it's legal
    states = jax.vmap(othello.env.init)(
        jax.random.split(jax.random.PRNGKey(0), len(openings))
    )
    step = jax.jit(jax.vmap(othello.env.step))
    for ply in range(8):
        moves = jnp.asarray(openings[:, ply])
        legal = states.legal_action_mask[jnp.arange(len(openings)), moves]
        assert bool(legal.all()), f"an illegal move at ply {ply}"
        states = step(states, moves)
    assert not bool(states.terminated.any())
    # the openings are distinct
    assert len({tuple(o) for o in openings}) == len(openings)


def test_xot_init_fn_starts_from_openings_or_the_standard_start():
    openings = load_xot()[:50]
    keys = jax.random.split(jax.random.PRNGKey(0), 400)
    # every game from an opening: 8 moves in, on one of the openings' boards
    states, meta = jax.vmap(make_xot_init_fn(openings, 0.0))(keys)
    assert (np.asarray(meta.step) == 8).all()
    played = jax.vmap(othello.env.init)(
        jax.random.split(jax.random.PRNGKey(1), len(openings))
    )
    step = jax.jit(jax.vmap(othello.env.step))
    for ply in range(openings.shape[1]):
        played = step(played, jnp.asarray(openings[:, ply]))
    boards = {np.asarray(o).tobytes() for o in played.observation}
    assert all(np.asarray(o).tobytes() in boards for o in states.observation)
    # every game from the standard start
    _, meta = jax.vmap(make_xot_init_fn(openings, 1.0))(keys)
    assert (np.asarray(meta.step) == 0).all()
    # a mix
    _, meta = jax.vmap(make_xot_init_fn(openings, 0.25))(keys)
    assert 0.15 < (np.asarray(meta.step) == 0).mean() < 0.35


def test_engines_read_their_boards(monkeypatch):
    from engines import Edax, Egaroucid

    # what each engine's showboard prints (after GTPEngine.command strips the "= ")
    edax_board = """A B C D E F G H
1 - - - - - - - - 1
2 - - - - - - - - 2 * to move
3 - . O * - - - - 3
4 - - . O * - - - 4 *: discs =  3    moves =  4
5 - - - * O . - - 5 O: discs =  3    moves =  5"""
    egaroucid_board = "a b c d e f g h\n" + "\n".join(
        f"  {r} " + " ".join(row)
        for r, row in enumerate(
            ["........", "........", "..OX....", "...OX...", "...XO...",
             "........", "........", "XXXXXXXX"],
            start=1,
        )
    )  # fmt: skip
    for cls, board, expected in (
        (Edax, edax_board, (3, 3)),
        (Egaroucid, egaroucid_board, (11, 3)),
    ):
        engine = cls.__new__(cls)  # no process
        monkeypatch.setattr(engine, "command", lambda _, board=board: board)
        assert engine.discs() == expected


def test_engines_are_never_sent_passes(monkeypatch):
    from engines import Edax

    engine = Edax.__new__(Edax)
    sent = []
    monkeypatch.setattr(engine, "command", sent.append)
    engine.play("b", othello.action("d3"))
    engine.play("w", othello.PASS)
    assert sent == ["play b d3"]
