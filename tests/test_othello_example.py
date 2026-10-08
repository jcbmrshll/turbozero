"""The Othello example's symmetry transforms: each must map a position's legal moves to the
legal moves of the transformed position."""

import importlib.util
from pathlib import Path

import jax
import numpy as np

spec = importlib.util.spec_from_file_location(
    "othello_example", Path(__file__).parents[1] / "examples" / "othello.py"
)
assert spec is not None and spec.loader is not None
othello = importlib.util.module_from_spec(spec)
spec.loader.exec_module(othello)


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
