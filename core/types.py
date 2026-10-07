from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

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
    """
    rewards: jax.Array
    action_mask: jax.Array
    terminated: bool
    cur_player_id: int
    step: int


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
    nn_state: Optional[eqx.nn.State]
    opt_state: optax.OptState
    step: jax.Array


EnvStepFn = Callable[[Any, int], Tuple[Any, StepMetadata]]
EnvInitFn = Callable[[jax.Array], Tuple[Any, StepMetadata]]  
DataTransformFn = Callable[[jax.Array, jax.Array, Any], Tuple[jax.Array, jax.Array, Any]]
Params = Any
EvalFn = Callable[[Any, Params, jax.Array], Tuple[jax.Array, float]]
LossFn = Callable[[Any, Optional[eqx.nn.State], BaseExperience], Tuple[jax.Array, Tuple[dict, Optional[eqx.nn.State]]]]
ExtractModelParamsFn = Callable[[TrainState], Any]
StateToNNInputFn = Callable[[Any], jax.Array]
