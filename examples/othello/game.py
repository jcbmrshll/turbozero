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
from core.networks.spec import make_network as build_network
from core.networks.spec import network_config
from core.training.train import weight_decay_mask
from core.types import StepMetadata

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
    """The action for a square's name, e.g. "d3" (or "D3") -> 19; "pass" (or "ps", "pa") -> 64."""
    name = name.strip().lower()
    if name in ("pass", "ps", "pa"):
        return PASS
    return (int(name[1]) - 1) * 8 + "abcdefgh".index(name[0])


def greedy_eval(obs):
    """Values a position by the active player's lead in tiles, with a uniform policy:
    a baseline that doesn't use a neural network at all."""
    value = (obs[..., 0].sum() - obs[..., 1].sum()) / 64
    return jnp.ones((1, env.num_actions)), jnp.array([value])


def episode_metrics(rewards):
    """What train.py tracks about each finished self-play game (the Trainer's
    `episode_metrics_fn`): whether it was a draw, logged as its fraction `episode_draw`."""
    return {"draw": (rewards == 0).all()}


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


# AdamW's decay of a weight each step is learning rate * WEIGHT_DECAY * weight. An L2 penalty
# of 1e-4 in the loss, which training used before, came to about this under Adam on the
# convolutions (from 0.3 to 0.8, Adam's normalization making it uneven between weights)
WEIGHT_DECAY = 0.5

# steps down the tree an MCTS traversal takes between checks for its leaf (see
# MCTS.traversal_steps): the fastest on GPU of 1, 2, 4, 8 and 16, with 1024 games of 64 to 256
# sims (1.2x to 1.25x faster self-play than 1; 16 was slower again)
TRAVERSAL_STEPS = 8

# the network every script here defaults to: a ResNet of 6 blocks of 128 channels with
# KataGo's global pooling, in every 2nd block and in the policy and value heads. Pooling
# beat the AlphaZero paper's plain ResNet, resnet:6x128 before, against Edax and on the
# engine-labelled set (see README.md)
DEFAULT_NET = "resnet:6x128,pool_every=2,heads=pooled"


def make_network(
    net: str = DEFAULT_NET,
    seed: int = 0,
    inference_dtype: str = "float32",
    train_dtype: str = "float32",
    remat: bool = False,
):
    """The network a spec names (see core.networks.spec.network_config), e.g.
    DEFAULT_NET, or "resnet:10x128", the residual network from the AlphaZero paper with
    10 blocks of 128 channels; any equinox module works (see core.networks.utils.apply_nn).
    It's created along with its state (BatchNorm's, if it has any): returns (network,
    state). It computes in `inference_dtype` when it isn't training, e.g. in self-play,
    and in `train_dtype` in training (mixed precision, see AZResnetConfig). With `remat`,
    training computes each block's activations again in the backward pass instead of
    keeping them."""
    config = network_config(
        net,
        env.num_actions,
        inference_dtype=inference_dtype,
        train_dtype=train_dtype,
        remat=remat,
    )
    # pgx types observation_shape as Tuple[int, ...]; for board games it's (height, width, channels)
    shape = cast(tuple[int, int, int], env.observation_shape)
    return build_network(config, shape, jax.random.PRNGKey(seed))


def make_optimizer(
    total_steps: int,
    lr: float = 1e-3,
    lr_final: float = 1e-4,
    weight_decay: float = WEIGHT_DECAY,
) -> optax.GradientTransformation:
    """AdamW, with the learning rate decaying from `lr` to `lr_final` over
    `total_steps` training steps, along a cosine, and weight decay on the network's weights,
    not its biases or BatchNorm scales and offsets (see `weight_decay_mask`). Its
    hyperparameters are injected, for the trainer to record them and log the learning
    rate."""
    return optax.inject_hyperparams(optax.adamw, static_args=("mask",))(
        learning_rate=optax.cosine_decay_schedule(
            lr, decay_steps=max(total_steps, 1), alpha=lr_final / lr
        ),
        weight_decay=weight_decay,
        mask=weight_decay_mask,
    )


def load_checkpoint(path: str, net: str, inference_dtype: str = "float32"):
    """Loads the network saved in a training checkpoint (see `train.py`). `net` must name
    the network it was trained with, the run's --net.

    Raises:
        ValueError: if the checkpoint doesn't hold a `net` network

    Returns:
        (network, (params, network state)): the network, and the parameters and state
        to evaluate it with, as `core.evaluators.evaluation_fns.make_nn_eval_fn` takes them
    """
    network, network_state = make_network(net, inference_dtype=inference_dtype)
    params = eqx.filter(network, eqx.is_inexact_array)
    # a checkpoint is a TrainState's arrays in order, params and network state first: read
    # just those, so that it loads whatever optimizer trained it
    with open(path, "rb") as f:
        try:
            params, network_state = eqx.tree_deserialise_leaves(
                f, (params, network_state)
            )
        # a mismatch fails wherever the layouts first differ, as a shape error or worse
        except Exception as e:
            raise ValueError(
                f"{path} doesn't hold a {net} network: pass the --net its run trained "
                f"with (before {DEFAULT_NET} became the default, runs trained "
                "resnet:6x128)"
            ) from e
    return network, (params, network_state)


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
        # the tree is kept between moves: room for the subtree a search reuses, and its own
        # iterations on top (as self-play's, see train.py's --tree-factor)
        max_nodes=2 * num_iterations,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=0.0,
        dirichlet_epsilon=0.25 if noise else 0.0,
        traversal_steps=TRAVERSAL_STEPS,
    )
