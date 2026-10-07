
from dataclasses import dataclass
from typing import Any, Callable, Tuple

from flax.training.train_state import TrainState
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
    

EnvStepFn = Callable[[Any, int], Tuple[Any, StepMetadata]]
EnvInitFn = Callable[[jax.Array], Tuple[Any, StepMetadata]]  
DataTransformFn = Callable[[jax.Array, jax.Array, Any], Tuple[jax.Array, jax.Array, Any]]
Params = Any
EvalFn = Callable[[Any, Params, jax.Array], Tuple[jax.Array, float]]
LossFn = Callable[[Any, TrainState, BaseExperience], Tuple[jax.Array, Tuple[Any, optax.OptState]]]
ExtractModelParamsFn = Callable[[TrainState], Any]
StateToNNInputFn = Callable[[Any], jax.Array]
