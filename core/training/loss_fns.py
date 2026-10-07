
from typing import Any, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from core.memory.replay_memory import BaseExperience
from core.networks.utils import apply_nn


def az_default_loss_fn(nn: Any, nn_state: Optional[eqx.nn.State], experience: BaseExperience,
                       l2_reg_lambda: float = 0.0001) -> Tuple[jax.Array, Tuple[dict, Optional[eqx.nn.State]]]:
    """ Implements the default AlphaZero loss function.
    
    = Policy Loss + Value Loss + L2 Regularization
    Policy Loss: Cross-entropy loss between predicted policy and target policy
    Value Loss: L2 loss between predicted value and target value
    
    Args:
    - `nn`: the neural network (an equinox module, see core.networks.utils.apply_nn), differentiated with respect to its floating point arrays
    - `nn_state`: state of the neural network (e.g. BatchNorm statistics), None for stateless networks
    - `experience`: experience sampled from replay buffer
        - stores the observation, target policy, target value
    - `l2_reg_lambda`: L2 regularization weight (default = 1e-4)

    Returns:
    - (loss, (aux_metrics, nn_state))
        - `loss`: total loss
        - `aux_metrics`: auxiliary metrics (policy_loss, value_loss)
        - `nn_state`: updated state of the neural network
    """

    # get predictions
    (pred_policy, pred_value), nn_state = apply_nn(nn, nn_state, experience.observation_nn)

    # set invalid actions in policy to -inf
    pred_policy = jnp.where(
        experience.policy_mask,
        pred_policy,
        jnp.finfo(jnp.float32).min
    )

    # compute policy loss
    policy_loss = optax.softmax_cross_entropy(pred_policy, experience.policy_weights).mean()
    # select appropriate value from experience.reward
    current_player = experience.cur_player_id
    target_value = experience.reward[jnp.arange(experience.reward.shape[0]), current_player]
    # compute MSE value loss
    value_loss = optax.l2_loss(pred_value.squeeze(), target_value).mean()

    # compute L2 regularization
    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree.map(
            lambda x: (x ** 2).sum(),
            eqx.filter(nn, eqx.is_inexact_array)
        )
    )

    # total loss
    loss = policy_loss + value_loss + l2_reg
    aux_metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss
    }
    return loss, (aux_metrics, nn_state)
