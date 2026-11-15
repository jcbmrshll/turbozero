from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import optax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class StepMetadata:
    """What an environment reports with each state it steps to."""

    rewards: jax.Array  # (players,) this step's reward for each player
    action_mask: jax.Array  # (actions,) the legal actions
    terminated: jax.Array  # () whether the episode is over
    cur_player_id: jax.Array  # () the player to move
    step: jax.Array  # () steps taken so far in the episode


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState:
    """What training updates: the network's floating point arrays
    (`eqx.filter(nn, eqx.is_inexact_array)`), its state (e.g. BatchNorm
    statistics, None for stateless networks) and the optimizer's state.
    """

    params: Any
    nn_state: eqx.nn.State | None
    opt_state: optax.OptState


# (env_state, action) -> (next env_state, its metadata)
EnvStepFn = Callable[[Any, jax.Array], tuple[Any, StepMetadata]]
# key -> (initial env_state, its metadata)
EnvInitFn = Callable[[jax.Array], tuple[Any, StepMetadata]]
# (env_state, params, key) -> (policy logits, value), for one state
EvalFn = Callable[[Any, Any, jax.Array], tuple[jax.Array, jax.Array]]
# env_state -> the network's input
StateToNNInputFn = Callable[[Any], jax.Array]
