from collections.abc import Sequence
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from core.memory.replay_memory import BaseExperience
from core.networks.utils import apply_nn, outcome_value


def score_target(scores: Sequence[float], target: jax.Array) -> jax.Array:
    """(batch, scores) the distribution over `scores` (distinct, in any order) a value head over
    them trains toward, for (batch,) `target` scores: all on the score equal to it, or for one
    between two scores, shared between them in proportion to its nearness to each (a draw, 0,
    half a win at 1 and half a loss at -1, as KataGo's value targets have it); all on the nearest
    end beyond them."""
    order = np.argsort(scores)
    ascending = jnp.asarray(np.asarray(scores, np.float32)[order])
    # each score's share is its hat function: 1 there, falling to 0 at its neighbours
    shares = jax.vmap(
        lambda column: jnp.interp(target, ascending, column), out_axes=-1
    )(jnp.eye(len(scores), dtype=jnp.float32))
    return shares[:, np.argsort(order)]


def az_default_loss_fn(
    nn: Any,
    nn_state: eqx.nn.State | None,
    experience: BaseExperience,
    value_target_q: float = 0.0,
    ownership_weight: float = 0.02,
    aux_score_weight: float = 0.02,
) -> tuple[jax.Array, tuple[dict, eqx.nn.State | None]]:
    """Implements the default AlphaZero loss function.

    = Policy Loss + Value Loss
    Policy Loss: Cross-entropy loss between predicted policy and target policy
    Value Loss: L2 loss between predicted value and target value

    The target value is the episode's outcome for the player to move, z, or with `value_target_q`
    a mix of it and the search's value of the position, q: (1 - value_target_q) * z + value_target_q * q.
    z is a noisy signal for a position far from the end of its game; q is better informed, if limited
    by the search's horizon.

    A network whose value head predicts the game's final score (see `AZResnetConfig.outcome_scores`)
    trains on the score instead, which the experience's `reward` holds (see `Trainer`'s
    `score_targets`): the value loss is the cross-entropy of its distribution over the scores with
    the score the game ended on (see `score_target`). Its scores may be the results themselves,
    e.g. a win and a loss, the experience's `reward` then as it is. `value_loss` is then logged as the mean squared error of its value
    against the game's result (the score's sign), comparable to a network's that predicts the result
    directly, and `score_error` as the mean absolute error of its expected score.

    A network with an ownership head (see `AZResnetConfig.ownership`) also trains it on the game's
    final ownership, from the side of the player to move (the experience's `ownership`, see
    `Trainer`'s `ownership_targets`), as KataGo does: `ownership_weight` times the sum over
    locations of the cross-entropy of (1 + o) / 2 with (1 + tanh(x)) / 2, o the target and x the
    head's output. KataGo's weight is 1.5 / b^2 on a board of side b. `ownership_loss` is logged
    as that cross-entropy's mean over locations, unweighted.

    A network with an auxiliary score head (see `AZResnetConfig.aux_scores`), beside a value head
    over results, trains it on the final score as a value head over scores trains, at
    `aux_score_weight`: its value head's targets are then the scores' results (a score beyond the
    results' range is the nearest result, see `score_target`). Logged as `aux_score_loss`, its
    cross-entropy unweighted, and `aux_score_error`, its expected score's mean absolute error.

    Args:
        nn: the neural network (an equinox module, see core.networks.utils.apply_nn), differentiated with respect to its floating point arrays
        nn_state: state of the neural network (e.g. BatchNorm statistics), None for stateless networks
        experience: experience sampled from replay buffer
            - stores the observation, target policy, target value
        value_target_q: weight of the search's value in the target value, from 0 (the outcome alone,
            the default) to 1 (the search's value alone)
        ownership_weight: weight of the ownership head's loss, summed over locations
        aux_score_weight: weight of the auxiliary score head's cross-entropy

    Returns:
        Tuple[jax.Array, Tuple[dict, Optional[eqx.nn.State]]]: (loss, (aux_metrics, nn_state))
            - loss: total loss
            - aux_metrics: auxiliary metrics (policy_loss, value_loss)
            - nn_state: updated state of the neural network
    """

    scores = getattr(nn, "outcome_scores", None)
    aux_scores = getattr(nn, "aux_scores", None)
    ownership = getattr(nn, "ownership", False)
    if scores is not None and value_target_q:
        raise ValueError(
            "a value head over scores trains on them alone: no value_target_q"
        )
    if ownership and experience.ownership is None:
        raise ValueError("an ownership head needs experiences with their ownership")
    # get predictions; with heads training sees more of (the value head's logits over the scores,
    # an ownership head), what it needs
    aux = {}
    if scores is None and not ownership and aux_scores is None:
        (pred_policy, pred_value), nn_state = apply_nn(
            nn, nn_state, experience.observation_nn
        )
    else:
        (pred_policy, pred_value, aux), nn_state = apply_nn(
            partial(nn, train_outputs=True), nn_state, experience.observation_nn
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
    metrics = {}
    if scores is None:
        # compute MSE value loss
        value_loss = optax.l2_loss(pred_value.squeeze(), target_value).mean()
        loss_value = value_loss
    else:
        # the target is the score's bin, or between two the two nearest in proportion; the
        # value's error is against the result, its sign
        loss_value = optax.softmax_cross_entropy(
            pred_value, score_target(scores, target_value)
        ).mean()
        scores = jnp.asarray(scores, jnp.float32)
        value_loss = optax.l2_loss(
            outcome_value(pred_value, scores), jnp.sign(target_value)
        ).mean()
        expected = jax.nn.softmax(pred_value, axis=-1) @ scores
        # against the target as the head can express it: a score beyond its scores' range is
        # the nearest end (e.g. a margin's result, for a head over a win and a loss)
        reachable = jnp.clip(target_value, scores.min(), scores.max())
        metrics = {
            "outcome_loss": loss_value,
            "score_error": jnp.abs(expected - reachable).mean(),
        }

    # the policy loss is the targets' entropy, which no network can lower, plus the KL divergence
    # from the targets to the network's policy: of the two, only the KL says how well the network
    # fits its targets whatever they are (a search that changes them changes the entropy)
    targets = experience.policy_weights
    target_entropy = (
        -jnp.where(targets > 0, targets * jnp.log(targets), 0.0).sum(-1).mean()
    )

    if aux_scores is not None:
        aux_loss = optax.softmax_cross_entropy(
            aux["scores"], score_target(aux_scores, target_value)
        ).mean()
        aux_expected = jax.nn.softmax(aux["scores"], axis=-1) @ jnp.asarray(
            aux_scores, jnp.float32
        )
        loss_value = loss_value + aux_score_weight * aux_loss
        metrics["aux_score_loss"] = aux_loss
        metrics["aux_score_error"] = jnp.abs(aux_expected - target_value).mean()

    if ownership:
        assert experience.ownership is not None
        # the player to move's ownership; (1 + tanh(x)) / 2 = sigmoid(2x)
        target = experience.ownership[
            jnp.arange(experience.ownership.shape[0]), current_player
        ].astype(jnp.float32)
        per_location = optax.sigmoid_binary_cross_entropy(
            2 * aux["ownership"], (1 + target) / 2
        )
        ownership_loss = per_location.mean()
        loss_value = loss_value + ownership_weight * per_location.sum(-1).mean()
        metrics["ownership_loss"] = ownership_loss

    # total loss
    loss = policy_loss + loss_value
    aux_metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        **metrics,
        "policy_target_entropy": target_entropy,
        "policy_kl": policy_loss - target_entropy,
    }
    return loss, (aux_metrics, nn_state)
