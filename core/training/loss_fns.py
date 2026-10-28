from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from core.memory.replay_memory import BaseExperience
from core.networks.utils import apply_nn


def az_default_loss_fn(
    nn: Any,
    nn_state: eqx.nn.State | None,
    experience: BaseExperience,
    value_target_q: float = 0.0,
) -> tuple[jax.Array, tuple[dict, eqx.nn.State | None]]:
    """Implements the default AlphaZero loss function.

    = Policy Loss + Value Loss
    Policy Loss: Cross-entropy loss between predicted policy and target policy
    Value Loss: L2 loss between predicted value and target value

    The target value is the episode's outcome for the player to move, z, or with `value_target_q`
    a mix of it and the search's value of the position, q: (1 - value_target_q) * z + value_target_q * q.
    z is a noisy signal for a position far from the end of its game; q is better informed, if limited
    by the search's horizon.

    Args:
        nn: the neural network (an equinox module, see core.networks.utils.apply_nn), differentiated with respect to its floating point arrays
        nn_state: state of the neural network (e.g. BatchNorm statistics), None for stateless networks
        experience: experience sampled from replay buffer
            - stores the observation, target policy, target value
        value_target_q: weight of the search's value in the target value, from 0 (the outcome alone,
            the default) to 1 (the search's value alone)

    Returns:
        Tuple[jax.Array, Tuple[dict, Optional[eqx.nn.State]]]: (loss, (aux_metrics, nn_state))
            - loss: total loss
            - aux_metrics: auxiliary metrics (policy_loss, value_loss)
            - nn_state: updated state of the neural network
    """

    # get predictions
    (pred_policy, pred_value), nn_state = apply_nn(
        nn, nn_state, experience.observation_nn
    )

    # set invalid actions in policy to -inf
    pred_policy = jnp.where(
        experience.policy_mask, pred_policy, jnp.finfo(jnp.float32).min
    )

    # compute policy loss
    policy_loss = optax.softmax_cross_entropy(
        pred_policy, experience.policy_weights
    ).mean()
    # select appropriate value from experience.reward
    current_player = experience.cur_player_id
    target_value = experience.reward[
        jnp.arange(experience.reward.shape[0]), current_player
    ]
    if value_target_q:
        target_value = (
            1 - value_target_q
        ) * target_value + value_target_q * experience.search_value
    # compute MSE value loss
    value_loss = optax.l2_loss(pred_value.squeeze(), target_value).mean()

    # the policy loss is the targets' entropy, which no network can lower, plus the KL divergence
    # from the targets to the network's policy: of the two, only the KL says how well the network
    # fits its targets whatever they are (a search that changes them changes the entropy)
    targets = experience.policy_weights
    target_entropy = (
        -jnp.where(targets > 0, targets * jnp.log(targets), 0.0).sum(-1).mean()
    )

    # total loss
    loss = policy_loss + value_loss
    aux_metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "policy_target_entropy": target_entropy,
        "policy_kl": policy_loss - target_entropy,
    }
    return loss, (aux_metrics, nn_state)
