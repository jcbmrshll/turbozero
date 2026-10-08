"""Othello pieces shared by the scripts in this directory: the environment and how
turbozero talks to it, board symmetries, the network and optimizer, evaluators for
test games, square names, and loading checkpoints."""

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pgx

from core.evaluators.alphazero import AlphaZero
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.types import StepMetadata, TrainState

# vectorized environments pair well with batched AlphaZero; pgx has many more:
# https://sotets.uk/pgx/othello/
env = pgx.make("othello")

# actions 0-63 are the squares, row by row from a1 to h8 (a-h are columns, 1-8 rows);
# action 64 is a pass, legal only when there is no other move
PASS = 64


# turbozero interfaces with an environment through a step fn (state, action) and an
# init fn (key), each returning the new state along with the StepMetadata it needs:
# rewards for each player, a mask of legal actions, whether the episode has
# terminated, the id of the player to move, and the step number
def step_fn(state, action):
    state = env.step(state, action)
    return state, metadata(state)


def init_fn(key):
    state = env.init(key)
    return state, metadata(state)


def metadata(state) -> StepMetadata:
    return StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count,
    )


def state_to_nn_input(state):
    """Converts an environment state to the network's input. pgx provides this as
    `state.observation`: the player to move's discs, then the opponent's."""
    return state.observation


def square(action: int) -> str:
    """The name of an action's square, e.g. 19 -> "d3", or "pass"."""
    return "pass" if action == PASS else "abcdefgh"[action % 8] + str(action // 8 + 1)


def action(name: str) -> int:
    """The action for a square's name, e.g. "d3" (or "D3") -> 19; "pass" -> 64."""
    name = name.strip().lower()
    if name in ("pass", "ps"):
        return PASS
    return (int(name[1]) - 1) * 8 + "abcdefgh".index(name[0])


def greedy_eval(obs):
    """Values a position by the active player's lead in tiles, with a uniform policy:
    a baseline that doesn't use a neural network at all."""
    value = (obs[..., 0].sum() - obs[..., 1].sum()) / 64
    return jnp.ones((1, env.num_actions)), jnp.array([value])


def make_symmetry_transform_fn(quarter_turns: int, transpose: bool):
    """A DataTransformFn that maps the board through one of its symmetries (an optional
    transpose, then `quarter_turns` quarter turns), to generate an extra training sample
    from each self-play step. Othello's rules don't change under any of the board's 8
    symmetries. The policy mask and weights are mapped to match: only the first 64
    actions are board squares, the 65th (pass) stays where it is."""

    def transform_fn(mask, policy, state):
        # we only use state.observation, no need to update the rest of the state fields
        new_obs = state.observation
        # idxs[r, c] is the square that ends up at (r, c)
        idxs = jnp.arange(64).reshape(8, 8)
        if transpose:
            new_obs = jnp.swapaxes(new_obs, -3, -2)
            idxs = idxs.T
        new_obs = jnp.rot90(new_obs, quarter_turns, axes=(-3, -2))
        idxs = jnp.rot90(idxs, quarter_turns, axes=(0, 1))
        action_ids = jnp.arange(65).at[:64].set(idxs.flatten())
        return (
            mask[..., action_ids],
            policy[..., action_ids],
            state.replace(observation=new_obs),
        )

    return transform_fn


# every symmetry but the identity: 7 extra samples per self-play step
SYMMETRY_TRANSFORM_FNS = [
    make_symmetry_transform_fn(quarter_turns, transpose)
    for transpose in (False, True)
    for quarter_turns in range(4)
    if quarter_turns or transpose
]


def make_network(num_blocks: int, num_channels: int, seed: int = 0):
    """The residual network from the AlphaZero paper; any equinox module works (see
    core.networks.utils.apply_nn). It uses BatchNorm, so it's created along with its
    state: returns (network, state)."""
    return eqx.nn.make_with_state(AZResnet)(
        AZResnetConfig(
            policy_head_out_size=env.num_actions,
            num_blocks=num_blocks,
            num_channels=num_channels,
        ),
        # pgx types observation_shape as Tuple[int, ...]; for board games it's (height, width, channels)
        cast(tuple[int, int, int], env.observation_shape),
        key=jax.random.PRNGKey(seed),
    )


def make_optimizer(total_steps: int) -> optax.GradientTransformation:
    """Adam, with the learning rate decaying from 1e-3 to a tenth of that over
    `total_steps` training steps."""
    return optax.adam(
        optax.cosine_decay_schedule(1e-3, decay_steps=max(total_steps, 1), alpha=0.1)
    )


def load_checkpoint(path: str, num_blocks: int, num_channels: int):
    """Loads the network saved in a training checkpoint (see `train.py`).

    Returns:
        (network, (params, network state)): the network, and the parameters and state
        to evaluate it with, as `core.evaluators.evaluation_fns.make_nn_eval_fn` takes them
    """
    network, network_state = make_network(num_blocks, num_channels)
    params = eqx.filter(network, eqx.is_inexact_array)
    # the optimizer state's structure doesn't depend on the schedule's length
    template = TrainState(
        params=params,
        nn_state=network_state,
        opt_state=make_optimizer(1).init(params),
        step=jnp.array(0, dtype=jnp.int32),
    )
    with open(path, "rb") as f:
        train_state = eqx.tree_deserialise_leaves(f, template)
    return network, (train_state.params, train_state.nn_state)


def make_test_evaluator(eval_fn, num_iterations: int = 64, noise: bool = True) -> MCTS:
    """Evaluator used in test games: temperature 0 to always play the most-visited
    action. Opponents share these settings, so that only the quality of their
    policy/value estimates and their search budget differ.

    With `noise`, AlphaZero's Dirichlet noise is mixed into the root policy, so that
    two deterministic players don't replay the same game; turn it off to measure an
    evaluator at full strength when the games get their variety elsewhere (e.g. from
    XOT openings, see `xot.py`)."""
    return AlphaZero(MCTS)(
        eval_fn=eval_fn,
        num_iterations=num_iterations,
        max_nodes=num_iterations + 16,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=0.0,
        dirichlet_epsilon=0.25 if noise else 0.0,
    )
