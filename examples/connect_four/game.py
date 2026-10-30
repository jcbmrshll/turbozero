"""Connect Four pieces shared by the scripts in this directory: the environment and how
turbozero talks to it, the board's symmetry, the network and optimizer, evaluators for test
games, and loading checkpoints."""

from dataclasses import replace
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pgx

from core.evaluators.alphazero import AlphaZero
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import BaseExperience
from core.networks.spec import make_network as build_network
from core.networks.spec import network_config
from core.training.train import weight_decay_mask
from core.types import StepMetadata

# vectorized environments pair well with batched AlphaZero; pgx has many more:
# https://sotets.uk/pgx/connect_four/
env = pgx.make("connect_four")

# a game lasts at most one move per square
MAX_STEPS = 42


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
    `state.observation`, (6, 7, 2), rows from the top: the player to move's discs, then the
    opponent's."""
    return state.observation


def episode_metrics(rewards):
    """What train.py tracks about each finished self-play game (the Trainer's
    `episode_metrics_fn`): whether it was a draw, logged as its fraction `episode_draw`."""
    return {"draw": (rewards == 0).all()}


# the columns, mirrored: action a is column a
MIRRORED_ACTIONS = np.arange(env.num_actions)[::-1]


def mirror(experience: BaseExperience) -> BaseExperience:
    """A DataTransformFn that maps a training sample through the board's one symmetry, its
    left-right mirror. Connect Four's rules don't change under it, so training draws it or the
    identity at random for each sample it takes from the replay buffer. The policy mask and
    weights are mapped to match the observation's board planes."""
    return replace(
        experience,
        observation_nn=jnp.flip(experience.observation_nn, axis=-2),
        policy_mask=experience.policy_mask[..., MIRRORED_ACTIONS],
        policy_weights=experience.policy_weights[..., MIRRORED_ACTIONS],
    )


# every symmetry but the identity, which training draws from along with it
SYMMETRY_TRANSFORM_FNS = [mirror]


# AdamW's decay of a weight each step is learning rate * WEIGHT_DECAY * weight (see the Othello
# example's game.py)
WEIGHT_DECAY = 0.5

# steps down the tree an MCTS traversal takes between checks for its leaf (see
# MCTS.traversal_steps), as in the Othello example
TRAVERSAL_STEPS = 8

# AlphaZero's Dirichlet noise at the root scales as 10 / (legal moves): 0.3 in chess, 0.03 in Go.
# Connect Four has 7 moves at most, and 1.0 is what Oracle's study of AlphaZero on it found best
# ("Lessons from AlphaZero: Connect Four")
DIRICHLET_ALPHA = 1.0

# the network every script here defaults to: AlphaZero's ResNet with 4 blocks of 64 channels,
# whose 9 squares of receptive field nearly span the 6 x 7 board. KataGo's pooled heads, the
# Othello example's default, don't apply: they output a logit per square, and here an action is a
# column
DEFAULT_NET = "resnet:4x64"


def make_network(
    net: str = DEFAULT_NET,
    seed: int = 0,
    inference_dtype: str = "float32",
    train_dtype: str = "float32",
    remat: bool = False,
):
    """The network a spec names (see core.networks.spec.network_config), e.g. DEFAULT_NET, or
    "resnet:10x128", the residual network from the AlphaZero paper with 10 blocks of 128
    channels; any equinox module works (see core.networks.utils.apply_nn). It's created along
    with its state (BatchNorm's, if it has any): returns (network, state). It computes in
    `inference_dtype` when it isn't training, e.g. in self-play, and in `train_dtype` in training
    (mixed precision, see AZResnetConfig). With `remat`, training computes each block's
    activations again in the backward pass instead of keeping them."""
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
                f"{path} doesn't hold a {net} network: pass the --net its run trained with"
            ) from e
    return network, (params, network_state)


def make_test_evaluator(
    eval_fn,
    num_iterations: int = 64,
    noise: bool = True,
    search: type[MCTS] = MCTS,
    **search_kwargs,
) -> MCTS:
    """Evaluator used in test games: temperature 0 to always play the most-visited
    action. Opponents share these settings, so that only the quality of their
    policy/value estimates and their search budget differ.

    With `noise`, AlphaZero's Dirichlet noise is mixed into the root policy, so that
    two deterministic players don't replay the same game; turn it off to measure an
    evaluator at full strength when the games get their variety elsewhere (e.g. from
    openings, see `openings.py`). `search` is the search backend, with its own settings in
    `search_kwargs`."""
    return AlphaZero(search)(
        eval_fn=eval_fn,
        num_iterations=num_iterations,
        # the tree is kept between moves: room for the subtree a search reuses, and its own
        # iterations on top (as self-play's, see train.py's --tree-factor)
        max_nodes=2 * num_iterations,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=0.0,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=0.25 if noise else 0.0,
        traversal_steps=TRAVERSAL_STEPS,
        **search_kwargs,
    )
