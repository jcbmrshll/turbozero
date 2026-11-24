from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import optax

from core.memory.replay_memory import BaseExperience


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class StepMetadata:
    """Metadata for a step in the environment.

    Attributes:
        rewards: rewards received by the players
        action_mask: mask of valid actions
        terminated: whether the environment is terminated
        cur_player_id: current player id
        step: step number
        scores: (optional) for a game decided by score, each player's final score once it has
            terminated (e.g. its disc margin): the value targets of a network that predicts the
            score (see `Trainer`'s `score_targets`)
        ownership: (optional) for a game played on the network's input locations, (players,
            locations) who owns each location once it has terminated, from each player's side (1
            the player, -1 another, 0 no one): the targets of an ownership head (see `Trainer`'s
            `ownership_targets`)
    """

    rewards: jax.Array
    action_mask: jax.Array
    terminated: jax.Array
    cur_player_id: jax.Array
    step: jax.Array
    scores: jax.Array | None = None
    ownership: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState:
    """Training state of the neural network.

    Attributes:
        params: trainable parameters, the floating point arrays of the network (`eqx.filter(nn, eqx.is_inexact_array)`)
        nn_state: state of the network (e.g. BatchNorm statistics), None for stateless networks
        opt_state: optimizer state
        step: number of training steps taken
    """

    params: Any
    nn_state: eqx.nn.State | None
    opt_state: optax.OptState
    step: jax.Array


EnvStepFn = Callable[[Any, jax.Array], tuple[Any, StepMetadata]]
EnvInitFn = Callable[[jax.Array], tuple[Any, StepMetadata]]
Params = Any
EvalFn = Callable[[Any, Params, jax.Array], tuple[jax.Array, jax.Array]]
LossFn = Callable[
    [Any, eqx.nn.State | None, BaseExperience],
    tuple[jax.Array, tuple[dict, eqx.nn.State | None]],
]
ExtractModelParamsFn = Callable[[TrainState], Any]
StateToNNInputFn = Callable[[Any], jax.Array]
