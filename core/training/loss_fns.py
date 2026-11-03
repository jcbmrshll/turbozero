from collections.abc import Callable
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

    (pred_policy, pred_value), nn_state = apply_nn(
        nn, nn_state, experience.observation_nn
    )
    loss, metrics = _az_loss(pred_policy, pred_value, experience, value_target_q)
    return loss, (metrics, nn_state)


def _az_loss(
    pred_policy: jax.Array,
    pred_value: jax.Array,
    experience: BaseExperience,
    value_target_q: float,
) -> tuple[jax.Array, dict]:
    """`az_default_loss_fn`'s loss and metrics, from the network's outputs on the batch."""
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
    return loss, aux_metrics


# a frozen network's view of a batch: (observations, legal actions) -> (policy over legal actions,
# value for the player to move)
Reference = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]


def anchored_loss_fn(
    nn: Any,
    nn_state: eqx.nn.State | None,
    experience: BaseExperience,
    reference: Reference,
    policy_weight: float,
    value_weight: float,
    value_target_q: float = 0.0,
) -> tuple[jax.Array, tuple[dict, eqx.nn.State | None]]:
    """`az_default_loss_fn`, with targets anchored to a frozen reference network (e.g. the network
    training started from, or a teacher): the policy target is `policy_weight` of the reference's
    policy and the rest the experience's, and the value target `value_weight` of the reference's
    value and the rest the outcome's (or the outcome and search value's, with `value_target_q`).

    Training on these blended targets is the same, but for the loss's scale, as adding a penalty
    of policy_weight / (1 - policy_weight) times KL(reference's policy || network's policy) to the
    policy loss, and the like for the value: a pull back toward the reference wherever the network's
    own targets lead it away. Gradients don't reach the reference.

    Logs, besides `az_default_loss_fn`'s metrics, `anchor_policy_kl` and `anchor_value_mse`: how far
    the network is from the reference.

    Raises:
        ValueError: if a value weight and `value_target_q` are both set
    """
    if value_weight and value_target_q:
        raise ValueError("anchor the value or mix in the search's value, not both")
    ref_policy, ref_value = jax.lax.stop_gradient(
        reference(experience.observation_nn, experience.policy_mask)
    )
    targets = (
        1 - policy_weight
    ) * experience.policy_weights + policy_weight * ref_policy
    anchored = BaseExperience(
        reward=experience.reward,
        policy_weights=targets,
        policy_mask=experience.policy_mask,
        observation_nn=experience.observation_nn,
        cur_player_id=experience.cur_player_id,
        search_value=ref_value.reshape(experience.search_value.shape)
        if value_weight
        else experience.search_value,
    )
    (logits, value), nn_state = apply_nn(nn, nn_state, experience.observation_nn)
    loss, metrics = _az_loss(logits, value, anchored, value_weight or value_target_q)
    # how far the network is from the reference, by its outputs on this batch
    log_policy = jax.nn.log_softmax(
        jnp.where(experience.policy_mask, logits, jnp.finfo(jnp.float32).min)
    )
    ref_entropy = jnp.where(ref_policy > 0, ref_policy * jnp.log(ref_policy), 0.0)
    metrics["anchor_policy_kl"] = jax.lax.stop_gradient(
        (ref_entropy - jnp.where(ref_policy > 0, ref_policy * log_policy, 0.0))
        .sum(-1)
        .mean()
    )
    metrics["anchor_value_mse"] = jax.lax.stop_gradient(
        ((value.reshape(-1) - ref_value.reshape(-1)) ** 2).mean()
    )
    return loss, (metrics, nn_state)
