"""Othello pieces shared by the scripts in this directory: the environment and how
turbozero talks to it, board symmetries, the network and optimizer, evaluators for
test games, square names, and loading checkpoints."""

import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
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
from core.training.lr_plateau import LRPlateau
from core.training.train import checkpoint_epochs, checkpoint_path, weight_decay_mask
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
    """A DataTransformFn that maps a training sample through one of the board's symmetries
    (an optional transpose, then `quarter_turns` quarter turns). Othello's rules don't
    change under any of the board's 8 symmetries, so training draws one at random for each
    sample it takes from the replay buffer. The policy mask and weights are mapped to match
    the observation's board planes: only the first 64 actions are board squares, the 65th
    (pass) stays where it is."""
    # idxs[r, c] is the square that ends up at (r, c)
    idxs = np.arange(64).reshape(8, 8)
    if transpose:
        idxs = idxs.T
    idxs = np.rot90(idxs, quarter_turns, axes=(0, 1))
    action_ids = np.append(idxs.flatten(), PASS)

    def transform_fn(experience: BaseExperience) -> BaseExperience:
        obs = experience.observation_nn
        if transpose:
            obs = jnp.swapaxes(obs, -3, -2)
        return replace(
            experience,
            observation_nn=jnp.rot90(obs, quarter_turns, axes=(-3, -2)),
            policy_mask=experience.policy_mask[..., action_ids],
            policy_weights=experience.policy_weights[..., action_ids],
        )

    return transform_fn


# every symmetry but the identity, which training draws from along with it
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
    lr_drops: Sequence[tuple[float, float]] = (),
    warmup_steps: int = 0,
    lr_plateau: LRPlateau | None = None,
) -> optax.GradientTransformation:
    """AdamW, with the learning rate decaying from `lr` to `lr_final` over
    `total_steps` training steps, along a cosine, or with `lr_drops`, held at `lr` and then
    dropping in steps: each (fraction of `total_steps`, the learning rate from there on), or
    with `lr_plateau`, its rates, dropping when training stops improving (the trainer's
    `lr_plateau` too; `lr`, `lr_final`, `lr_drops` and `warmup_steps` are unused). With
    `warmup_steps`, it first rises linearly from 0 to `lr` over that many steps, and the
    schedule runs over the rest. And
    weight decay on the network's weights, not its biases or BatchNorm scales and offsets (see
    `weight_decay_mask`). Its hyperparameters are injected, for the trainer to record them and
    log the learning rate."""
    if not 0 <= warmup_steps < max(total_steps, 1):
        raise ValueError(f"{warmup_steps} warmup steps don't fit in {total_steps}")
    total_steps -= warmup_steps
    schedule = optax.cosine_decay_schedule(
        lr, decay_steps=max(total_steps, 1), alpha=lr_final / lr
    )
    if lr_drops:
        rates = [lr, *(rate for _, rate in lr_drops)]
        schedule = optax.piecewise_constant_schedule(
            lr,
            {
                round(fraction * total_steps): after / before
                for (fraction, _), before, after in zip(
                    lr_drops, rates[:-1], rates[1:], strict=True
                )
            },
        )
    if warmup_steps:
        schedule = optax.join_schedules(
            [optax.linear_schedule(0.0, lr, warmup_steps), schedule], [warmup_steps]
        )
    return optax.inject_hyperparams(optax.adamw, static_args=("mask",))(
        learning_rate=schedule if lr_plateau is None else lr_plateau,
        weight_decay=weight_decay,
        mask=weight_decay_mask,
    )


def export_last(ckpt_dir: str, net: str) -> None:
    """Exports a finished run's last checkpoint to ONNX beside it. A failure only says so: the
    run's results are safe by now, and the checkpoint can be exported later (`export_onnx.py`)."""
    try:
        from export_onnx import save_onnx

        last = checkpoint_path(ckpt_dir, max(checkpoint_epochs(ckpt_dir)))
        save_onnx(last, net, Path(last).with_suffix(".onnx"))
    except Exception as e:  # noqa: BLE001
        print(f"no ONNX export: {e!r}", file=sys.stderr)


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
    network, network_state = load_weights(path, net, network, network_state)
    return network, (eqx.filter(network, eqx.is_inexact_array), network_state)


def load_weights(path: str, net: str, network, network_state):
    """`network` and `network_state` (as `make_network` makes them, for `net`) with the
    parameters and state saved in a training checkpoint instead.

    Raises:
        ValueError: if the checkpoint doesn't hold a `net` network
    """
    params, static = eqx.partition(network, eqx.is_inexact_array)
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
    return eqx.combine(params, static), network_state


def make_test_evaluator(
    eval_fn, num_iterations: int = 64, noise: bool = True, tree_factor: int = 2
) -> MCTS:
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
        max_nodes=tree_factor * num_iterations,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=0.0,
        dirichlet_epsilon=0.25 if noise else 0.0,
        traversal_steps=TRAVERSAL_STEPS,
    )
