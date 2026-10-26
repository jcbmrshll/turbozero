"""Shared setup for the turbozero test suite: tic-tac-toe, stub evaluation functions and evaluator factories."""

import os

# tests always run on CPU, never the GPU
os.environ["JAX_PLATFORMS"] = "cpu"
# pytest-xdist workers already use one core each, so each runs XLA single-threaded
if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
    ).strip()

import json
import threading
import urllib.request
from dataclasses import dataclass, replace
from functools import cache, partial
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pgx
import pytest
from seaglass.server import make_server

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
from core.types import StepMetadata

# compiling dominates the suite's runtime, so compiled programs persist across runs (and
# worktrees); JAX_COMPILATION_CACHE_DIR overrides where
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        os.path.expanduser("~/.cache/turbozero/jax-tests"),
    )
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_compilation_cache_max_size", 2**30)

env = pgx.make("tic_tac_toe")


def metadata(state) -> StepMetadata:
    return StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count,
    )


def step_fn(state, action):
    state = env.step(state, action)
    return state, metadata(state)


def init_fn(key):
    state = env.init(key)
    return state, metadata(state)


def state_to_nn_input(state):
    return state.observation


def uniform_eval_fn(state, params, key):  # pylint: disable=unused-argument
    """Uniform policy logits and a value of 0 for every state."""
    return jnp.zeros((env.num_actions,)), jnp.array(0.0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScriptedState:
    """Evaluator state for `ScriptedEvaluator`: how many moves it has made and how many times it
    has been stepped this game, the action it was last stepped with (-1 for none), and how many
    times it has been reset."""

    moves: jax.Array
    steps: jax.Array
    last_step: jax.Array
    resets: jax.Array


class ScriptedEvaluator(Evaluator):
    """Deterministic, search-free evaluator.

    - `first_legal`: always plays the lowest-index legal action.
    - `resign`: plays square 0 on its first move and repeats it on its second, which is illegal,
      so it loses on its second move (pgx ends the game and gives -1 to whoever moves illegally).

    Its state records how it's stepped and reset (see `ScriptedState`).
    """

    def __init__(self, strategy: str, discount: float = -1.0):
        super().__init__(discount=discount)
        assert strategy in ("first_legal", "resign")
        self.strategy = strategy

    def init(self, *args, **kwargs) -> ScriptedState:
        zero = jnp.array(0, dtype=jnp.int32)
        return ScriptedState(moves=zero, steps=zero, last_step=zero - 1, resets=zero)

    def reset(self, state: ScriptedState) -> ScriptedState:
        return replace(self.init(), resets=state.resets + 1)

    def step(self, state: ScriptedState, action: jax.Array) -> ScriptedState:
        return replace(state, steps=state.steps + 1, last_step=action.astype(jnp.int32))

    def evaluate(
        self, key, eval_state, env_state, root_metadata, params, env_step_fn, **kwargs
    ) -> EvalOutput:
        if self.strategy == "first_legal":
            action = jnp.argmax(root_metadata.action_mask)
        else:
            action = jnp.array(0)
        return EvalOutput(
            eval_state=replace(eval_state, moves=eval_state.moves + 1),
            action=action,
            policy_weights=jax.nn.one_hot(action, root_metadata.action_mask.shape[-1]),
        )

    def get_value(self, state: ScriptedState) -> jax.Array:
        return jnp.array(0.0)


def play(moves, key=None):
    """Plays `moves` from the initial tic-tac-toe position. Returns (state, metadata)."""
    if key is None:
        key = jax.random.PRNGKey(0)
    state, meta = init_fn(key)
    for action in moves:
        state, meta = step_fn(state, action)
    return state, meta


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

    def init_fn(key):  # pylint: disable=unused-argument
        state = FixedLengthState(
            step=jnp.array(0, dtype=jnp.int32),
            current_player=jnp.array(0, dtype=jnp.int32),
        )
        return state, metadata(state)

    def step_fn(state, action):  # pylint: disable=unused-argument
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
        state_to_nn_input=lambda state: jnp.stack(
            [state.step, state.current_player]
        ).astype(jnp.float32),
    )


@pytest.fixture(scope="session")
def fixed_length_env():
    """Factory for deterministic fixed-length environments, see `make_fixed_length_env`."""
    return make_fixed_length_env


@pytest.fixture(scope="session")
def ttt():
    """Tic-tac-toe environment helpers."""
    # init keys that give each player the first move
    first_player_keys = {}
    seed = 0
    while len(first_player_keys) < 2:
        key = jax.random.PRNGKey(seed)
        first_player_keys.setdefault(int(env.init(key).current_player), key)
        seed += 1
    return SimpleNamespace(
        env=env,
        num_actions=env.num_actions,
        step_fn=step_fn,
        init_fn=init_fn,
        metadata=metadata,
        state_to_nn_input=state_to_nn_input,
        play=play,
        first_player_keys=first_player_keys,
    )


@pytest.fixture(scope="session")
def make_search():
    """Cached factory for MCTS evaluators and their jitted `evaluate`/`step`.

    Compiling a search dominates the runtime of the suite, so tests that use the same
    configuration share one evaluator (and therefore one compiled function).

    `make_search(cls=AlphaZero(MCTS), eval_fn=uniform_eval_fn, **evaluator_kwargs)` returns a namespace with:
    - `evaluator`: the evaluator
    - `init()`: an empty search tree
    - `evaluate(key, tree, env_state, metadata, params=None)`: jitted `evaluator.evaluate`
    - `step(tree, action)`: jitted `evaluator.step`
    """

    @cache
    def factory(cls=None, eval_fn=uniform_eval_fn, **kwargs):
        cls = AlphaZero(MCTS) if cls is None else cls
        config: dict[str, Any] = {
            "eval_fn": eval_fn,
            "action_selector": PUCTSelector(),
            "branching_factor": env.num_actions,
            "max_nodes": 80,
            "num_iterations": 32,
            "temperature": 0.0,
        }
        config.update(kwargs)
        evaluator = cls(**config)
        template, _ = init_fn(jax.random.PRNGKey(0))
        return SimpleNamespace(
            evaluator=evaluator,
            init=lambda: evaluator.init(template_embedding=template),
            evaluate=jax.jit(
                lambda key, tree, env_state, meta, params=None: evaluator.evaluate(
                    key=key,
                    eval_state=tree,
                    env_state=env_state,
                    root_metadata=meta,
                    params=params,
                    env_step_fn=step_fn,
                )
            ),
            step=jax.jit(evaluator.step),
        )

    def make(cls=None, eval_fn=uniform_eval_fn, **kwargs):
        return factory(cls, eval_fn, **kwargs)

    return make


@pytest.fixture(scope="session")
def make_collector():
    """Factory for self-play collection in a single environment, through a `Trainer`'s `collect`.

    `make_collector(env, evaluator, max_episode_steps, ckpt_dir, **trainer_kwargs)` returns
    `collect(num_steps)`: the `CollectionState` after `num_steps` steps from the initial state.
    """

    def make(env, evaluator, max_episode_steps, ckpt_dir, **trainer_kwargs):
        nn = eqx.nn.Linear(2, env.num_actions, key=jax.random.PRNGKey(0))
        trainer = Trainer(
            batch_size=1,
            train_batch_size=1,
            warmup_steps=0,
            collection_steps_per_epoch=1,
            train_steps_per_epoch=1,
            nn=nn,
            loss_fn=az_default_loss_fn,
            optimizer=optax.sgd(1e-3),
            evaluator=evaluator,
            memory_buffer=EpisodeReplayBuffer(capacity=16),
            max_episode_steps=max_episode_steps,
            env_step_fn=env.step_fn,
            env_init_fn=env.init_fn,
            state_to_nn_input_fn=env.state_to_nn_input,
            testers=[],
            ckpt_dir=str(ckpt_dir),
            **trainer_kwargs,
        )
        step = jax.jit(jax.vmap(partial(trainer.collect, params=None)))

        def collect(num_steps):
            state = trainer.init_collection_state(jax.random.PRNGKey(0), batch_size=1)
            for i in range(num_steps):
                state = step(jax.random.split(jax.random.PRNGKey(i), 1), state)
            return state

        return collect

    return make


@pytest.fixture(scope="session")
def scripted():
    """Scripted (search-free) evaluators: `first_legal` and `resign`. See `ScriptedEvaluator`."""
    return SimpleNamespace(
        first_legal=ScriptedEvaluator("first_legal"),
        resign=ScriptedEvaluator("resign"),
    )


@pytest.fixture
def seaglass_server(tmp_path):
    """A seaglass server on a free port, storing runs in a temporary directory.

    Returns a namespace with its `url`, `dir`, `get(path)` to read its API as JSON, and
    `file(path)` to read a file's bytes.
    """
    server = make_server("127.0.0.1", 0, str(tmp_path / "runs"))
    # polls for shutdown often, so that it takes no time
    thread = threading.Thread(
        target=partial(server.serve_forever, poll_interval=0.01), daemon=True
    )
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def file(path):
        with urllib.request.urlopen(url + path) as resp:
            return resp.read()

    def get(path):
        return json.loads(file(path))

    yield SimpleNamespace(url=url, dir=tmp_path / "runs", get=get, file=file)
    server.shutdown()
    server.server_close()
