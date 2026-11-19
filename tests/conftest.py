"""Shared setup for the turbozero test suite."""

import os
import re

# tests run on the CPU unless JAX_PLATFORMS asks for another platform, e.g.
# JAX_PLATFORMS=cuda; without preallocation the pytest-xdist workers share a GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dataclasses import dataclass, replace
from functools import cache
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from turbozero.evaluators.evaluator import EvalOutput, Evaluator
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


def outer_loop_bodies(hlo: str) -> list[str]:
    """The bodies of the outermost while loops of compiled `hlo` with a known trip
    count (e.g. a scan; not the loops within them)."""
    computations = re.split(r"\n(?=\S)", hlo)
    (entry,) = [c for c in computations if c.startswith("ENTRY")]
    bodies = {
        body
        for line in entry.splitlines()
        if " while(" in line and "known_trip_count" in line
        for body in re.findall(r"body=%?([\w.-]+)", line)
    }
    return [c for c in computations if c.split(" ", 1)[0].lstrip("%") in bodies]


def copies_in_loop(hlo: str, shape: str) -> list[str]:
    """The copies of arrays whose shape starts with `shape` (e.g. "5,24") in
    `outer_loop_bodies`: each makes the loop's every trip cost the whole array."""
    return [
        copy
        for body in outer_loop_bodies(hlo)
        for copy in re.findall(rf"= \w+\[{shape}[,\]]\S* copy\(.*", body)
    ]


def open_loops_in_loop(hlo: str) -> int:
    """The while loops in `outer_loop_bodies` whose trip count XLA doesn't know: on
    GPU, each of their trips waits on the host for their condition."""
    return sum(
        "known_trip_count" not in line
        for body in outer_loop_bodies(hlo)
        for line in body.splitlines()
        if " while(" in line
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScriptedState:
    """State of `ScriptedEvaluator`: the moves it has made, the times it has been
    stepped and the action it was last stepped with (-1 for none), and the times
    it has been reset."""

    moves: jax.Array
    steps: jax.Array
    last_step: jax.Array
    resets: jax.Array


class ScriptedEvaluator(Evaluator):
    """Plays the first legal action in `order` (by default, the lowest legal
    action), without a search. Its state records how it's used (see
    `ScriptedState`)."""

    def __init__(self, order=None, discount=-1.0):
        super().__init__(discount)
        self.order = order

    def init(self, template_embedding):
        zero = jnp.array(0, dtype=jnp.int32)
        return ScriptedState(moves=zero, steps=zero, last_step=zero - 1, resets=zero)

    def reset(self, state):
        return replace(self.init(None), resets=state.resets + 1)

    def step(self, state, action):
        return replace(state, steps=state.steps + 1, last_step=action.astype(jnp.int32))

    def evaluate(self, key, eval_state, env_state, root_metadata, params, env_step_fn):
        mask = root_metadata.action_mask
        if self.order is None:
            action = jnp.argmax(mask)
        else:
            order = jnp.array(self.order)
            action = order[jnp.argmax(mask[order])]
        return EvalOutput(
            eval_state=replace(eval_state, moves=eval_state.moves + 1),
            action=action,
            policy_weights=jax.nn.one_hot(action, mask.shape[-1]),
        )

    def get_value(self, state):
        return jnp.array(0.0)


@pytest.fixture(scope="session")
def scripted():
    """Scripted evaluators (see `ScriptedEvaluator`): `first_legal`, and
    `beats_first_legal`, which wins tic-tac-toe against it whoever moves first
    (with the centre, then corners 2 and 6)."""
    return SimpleNamespace(
        first_legal=ScriptedEvaluator(),
        beats_first_legal=ScriptedEvaluator(order=(4, 2, 6, 0, 8, 1, 3, 5, 7)),
    )
