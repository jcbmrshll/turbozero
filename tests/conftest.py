"""Shared setup for the turbozero test suite."""

import os

# tests run on the CPU unless JAX_PLATFORMS asks for another platform, e.g.
# JAX_PLATFORMS=cuda; without preallocation the pytest-xdist workers share a GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dataclasses import dataclass
from functools import cache
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from turbozero.evaluators.mcts.action_selection import PUCTSelector
from turbozero.evaluators.mcts.mcts import MCTS
from turbozero.types import StepMetadata

# compiling dominates the suite's runtime, so compiled programs persist across
# runs and worktrees; JAX_COMPILATION_CACHE_DIR overrides where
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        os.path.expanduser("~/.cache/turbozero/jax-tests"),
    )
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FixedLengthState:
    """State of `make_fixed_length_env`: steps taken so far, and the player to move."""

    step: jax.Array
    current_player: jax.Array


def make_fixed_length_env(length, rewards=(1.0, -1.0), num_actions=2, action_mask=None):
    """Two-player environment that terminates after exactly `length` steps, whatever the actions.

    Players alternate, starting with player 0. The legal actions are `action_mask`'s at every step
    (by default, every action), and the terminal step pays `rewards` (indexed by player id); every
    other step pays 0.
    """
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    action_mask = (
        jnp.ones((num_actions,), dtype=jnp.bool_)
        if action_mask is None
        else jnp.asarray(action_mask, dtype=jnp.bool_)
    )

    def metadata(state):
        terminated = state.step >= length
        return StepMetadata(
            rewards=jnp.where(terminated, rewards, 0.0),
            action_mask=action_mask,
            terminated=terminated,
            cur_player_id=state.current_player,
            step=state.step,
        )

    def init_fn(key):
        state = FixedLengthState(
            step=jnp.array(0, dtype=jnp.int32),
            current_player=jnp.array(0, dtype=jnp.int32),
        )
        return state, metadata(state)

    def step_fn(state, action):
        state = FixedLengthState(
            step=state.step + 1, current_player=1 - state.current_player
        )
        return state, metadata(state)

    return SimpleNamespace(
        length=length,
        rewards=rewards,
        num_actions=num_actions,
        init_fn=init_fn,
        step_fn=step_fn,
    )


@pytest.fixture(scope="session")
def fixed_length_env():
    """Factory for deterministic fixed-length environments, see `make_fixed_length_env`."""
    return make_fixed_length_env


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TicTacToeState:
    """Tic-tac-toe: each square's mark (0 empty, else 1 + the id of the player who
    marked it), the player to move, and the moves made so far."""

    board: jax.Array  # (9,)
    current_player: jax.Array
    step: jax.Array
    rewards: jax.Array  # (2,) the last move's reward for each player
    terminated: jax.Array


TTT_LINES = jnp.array(
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]
)


def ttt_metadata(state):
    return StepMetadata(
        rewards=state.rewards,
        action_mask=state.board == 0,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state.step,
    )


def ttt_init_fn(first_player):
    """The empty board, `first_player` (0 or 1) to move."""
    state = TicTacToeState(
        board=jnp.zeros((9,), dtype=jnp.int32),
        current_player=jnp.asarray(first_player, dtype=jnp.int32),
        step=jnp.array(0, dtype=jnp.int32),
        rewards=jnp.zeros((2,), dtype=jnp.float32),
        terminated=jnp.array(False),
    )
    return state, ttt_metadata(state)


def ttt_step_fn(state, action):
    """Marks square `action` (which must be empty) for the player to move: +1 to
    them and -1 to the other if it makes a line, and the game ends on a line or a
    full board."""
    player = state.current_player
    board = state.board.at[action].set(player + 1)
    won = (board[TTT_LINES] == player + 1).all(axis=-1).any()
    state = TicTacToeState(
        board=board,
        current_player=1 - player,
        step=state.step + 1,
        rewards=jnp.where(won, jnp.where(jnp.arange(2) == player, 1.0, -1.0), 0.0),
        terminated=won | (board != 0).all(),
    )
    return state, ttt_metadata(state)


def ttt_play(moves, first_player=0):
    """Plays `moves` from the empty board. Returns (state, metadata)."""
    state, meta = ttt_init_fn(first_player)
    for action in moves:
        state, meta = ttt_step_fn(state, action)
    return state, meta


@pytest.fixture(scope="session")
def ttt():
    """Tic-tac-toe, see `TicTacToeState`."""
    return SimpleNamespace(
        num_actions=9, init_fn=ttt_init_fn, step_fn=ttt_step_fn, play=ttt_play
    )


def ttt_uniform_eval_fn(state, params, key):
    """Uniform policy logits and a value of 0 for every tic-tac-toe state."""
    return jnp.zeros((9,)), jnp.array(0.0)


@pytest.fixture(scope="session")
def make_search():
    """Cached factory for tic-tac-toe MCTS evaluators and their jitted
    `evaluate`/`step`: compiling a search dominates the suite's runtime, so tests
    with the same configuration share one evaluator (and compiled function).

    `make_search(cls=MCTS, leaf_eval_fn=ttt_uniform_eval_fn, **evaluator_kwargs)`
    returns a namespace with:
    - `evaluator`: the evaluator
    - `init()`: an empty search tree
    - `evaluate(key, tree, env_state, metadata, params=None)`: jitted
      `evaluator.evaluate`
    - `step(tree, action)`: jitted `evaluator.step`
    """

    @cache
    def make(cls=MCTS, leaf_eval_fn=ttt_uniform_eval_fn, **kwargs):
        config: dict[str, Any] = {
            "leaf_eval_fn": leaf_eval_fn,
            "action_selector": PUCTSelector(),
            "branching_factor": 9,
            "max_nodes": 80,
            "num_iterations": 32,
            "temperature": 0.0,
        }
        evaluator = cls(**(config | kwargs))
        template, _ = ttt_init_fn(0)
        return SimpleNamespace(
            evaluator=evaluator,
            init=lambda: evaluator.init(template_embedding=template),
            evaluate=jax.jit(
                lambda key, tree, env_state, meta, params=None: evaluator.evaluate(
                    key, tree, env_state, meta, params, ttt_step_fn
                )
            ),
            step=jax.jit(evaluator.step),
        )

    return make
