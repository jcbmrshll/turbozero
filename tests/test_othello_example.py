"""The Othello example's helpers: its board symmetries must map a position's legal moves to
the legal moves of the transformed position, and its XOT openings must be legal games."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# the example's scripts import their sibling modules
sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "othello"))
import game as othello
from xot import load_xot


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
