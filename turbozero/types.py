from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import optax


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
    """

    rewards: jax.Array
    action_mask: jax.Array
    terminated: jax.Array
    cur_player_id: jax.Array
    step: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState:
    """Training state of the neural network.

    Attributes:
        params: trainable parameters, the floating point arrays of the network (`eqx.filter(nn, eqx.is_inexact_array)`)
        nn_state: state of the network (e.g. BatchNorm statistics), None for stateless networks
        opt_state: optimizer state
    """

    params: Any
    nn_state: eqx.nn.State | None
    opt_state: optax.OptState


EnvStepFn = Callable[[Any, jax.Array], tuple[Any, StepMetadata]]
EnvInitFn = Callable[[jax.Array], tuple[Any, StepMetadata]]
EvalFn = Callable[[Any, Any, jax.Array], tuple[jax.Array, jax.Array]]
StateToNNInputFn = Callable[[Any], jax.Array]
