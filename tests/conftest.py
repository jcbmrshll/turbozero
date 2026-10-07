"""Shared setup for the turbozero test suite: tic-tac-toe, stub evaluation functions and evaluator factories."""
import os

# tests always run on CPU, never the GPU
os.environ["JAX_PLATFORMS"] = "cpu"
# simulate two devices so the pmap code paths run with more than one device
os.environ["XLA_FLAGS"] = " ".join(
    [os.environ.get("XLA_FLAGS", ""), "--xla_force_host_platform_device_count=2"]).strip()

from dataclasses import dataclass, replace
from functools import cache
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pgx
import pytest

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.types import StepMetadata


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


def make_logits_eval_fn(logits, value=0.0):
    """Eval fn returning fixed policy logits and a fixed value for every state."""
    logits = jnp.asarray(logits, dtype=jnp.float32)

    def eval_fn(state, params, key):  # pylint: disable=unused-argument
        return logits, jnp.array(value, dtype=jnp.float32)

    return eval_fn


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScriptedState:
    """Evaluator state for `ScriptedEvaluator`: how many moves it has made this game."""
    moves: jax.Array


class ScriptedEvaluator(Evaluator):
    """Deterministic, search-free evaluator.

    - `first_legal`: always plays the lowest-index legal action.
    - `resign`: plays square 0 on its first move and repeats it on its second, which is illegal,
      so it loses on its second move (pgx ends the game and gives -1 to whoever moves illegally).
    """

    def __init__(self, strategy: str, discount: float = -1.0):
        super().__init__(discount=discount)
        assert strategy in ("first_legal", "resign")
        self.strategy = strategy

    def init(self, *args, **kwargs) -> ScriptedState:
        return ScriptedState(moves=jnp.array(0, dtype=jnp.int32))

    def reset(self, state: ScriptedState) -> ScriptedState:
        return ScriptedState(moves=jnp.zeros_like(state.moves))

    def evaluate(self, key, eval_state, env_state, root_metadata, params, env_step_fn, **kwargs) -> EvalOutput:
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
        config = dict(
            eval_fn=eval_fn,
            action_selector=PUCTSelector(),
            branching_factor=env.num_actions,
            max_nodes=80,
            num_iterations=32,
            temperature=0.0,
        )
        config.update(kwargs)
        evaluator = cls(**config)
        template, _ = init_fn(jax.random.PRNGKey(0))
        return SimpleNamespace(
            evaluator=evaluator,
            init=lambda: evaluator.init(template_embedding=template),
            evaluate=jax.jit(lambda key, tree, env_state, meta, params=None: evaluator.evaluate(
                key=key, eval_state=tree, env_state=env_state, root_metadata=meta,
                params=params, env_step_fn=step_fn)),
            step=jax.jit(evaluator.step),
        )

    def make(cls=None, eval_fn=uniform_eval_fn, **kwargs):
        return factory(cls, eval_fn, **kwargs)

    return make


@pytest.fixture(scope="session")
def scripted():
    """Scripted (search-free) evaluators: `first_legal` and `resign`. See `ScriptedEvaluator`."""
    return SimpleNamespace(
        first_legal=ScriptedEvaluator("first_legal"),
        resign=ScriptedEvaluator("resign"),
    )
