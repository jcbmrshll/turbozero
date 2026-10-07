
from typing import Callable, Tuple

import chex
from flax.training.train_state import TrainState
import jax
import optax

from core.memory.replay_memory import BaseExperience

@chex.dataclass(frozen=True)
class StepMetadata:
    """Metadata for a step in the environment.

    Attributes:
        rewards: rewards received by the players
        action_mask: mask of valid actions
        terminated: whether the environment is terminated
        cur_player_id: current player id
        step: step number
    """
    rewards: chex.Array
    action_mask: chex.Array
    terminated: bool
    cur_player_id: int
    step: int
    

EnvStepFn = Callable[[chex.ArrayTree, int], Tuple[chex.ArrayTree, StepMetadata]]
EnvInitFn = Callable[[jax.random.PRNGKey], Tuple[chex.ArrayTree, StepMetadata]]  
DataTransformFn = Callable[[chex.Array, chex.Array, chex.ArrayTree], Tuple[chex.Array, chex.Array, chex.ArrayTree]]
Params = chex.ArrayTree
EvalFn = Callable[[chex.ArrayTree, Params, jax.random.PRNGKey], Tuple[chex.Array, float]]
LossFn = Callable[[chex.ArrayTree, TrainState, BaseExperience], Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]]
ExtractModelParamsFn = Callable[[TrainState], chex.ArrayTree]
StateToNNInputFn = Callable[[chex.ArrayTree], chex.Array]
