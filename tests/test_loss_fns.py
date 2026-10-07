from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from core.memory.replay_memory import BaseExperience
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.training.loss_fns import az_default_loss_fn


class OutputAsParams(eqx.Module):
    """Network whose params are its output (policy logits, value) for each example in the batch,
    so gradients flow straight to the outputs. Its input is the example's index in the batch."""
    logits: jax.Array
    value: jax.Array

    def __init__(self, policy_logits, value):
        self.logits = jnp.asarray(policy_logits, dtype=jnp.float32)
        self.value = jnp.asarray(value, dtype=jnp.float32)

    def __call__(self, x):
        return self.logits[x], self.value[x]


def experience(rewards, cur_player_id, policy_weights, policy_mask, observation=None):
    batch = len(cur_player_id)
    return BaseExperience(
        reward=jnp.asarray(rewards, dtype=jnp.float32),
        policy_weights=jnp.asarray(policy_weights, dtype=jnp.float32),
        policy_mask=jnp.asarray(policy_mask),
        observation_nn=jnp.arange(batch) if observation is None else observation,
        cur_player_id=jnp.asarray(cur_player_id, dtype=jnp.int32),
    )


def test_masked_policy_loss_is_finite_and_ignores_illegal_logits():
    mask = jnp.array([[True, False, True], [False, True, True]])
    batch = experience(rewards=[[1, -1], [1, -1]], cur_player_id=[0, 1],
                       policy_weights=[[0.25, 0.0, 0.75], [0.0, 0.5, 0.5]], policy_mask=mask)
    logits = jnp.array([[1.0, 50.0, -1.0], [-50.0, 0.5, 0.0]])

    loss_fn = partial(az_default_loss_fn, l2_reg_lambda=0.0)
    net = OutputAsParams(logits, jnp.zeros((2, 1)))
    (loss, (metrics, _)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(net, None, batch)

    assert jnp.isfinite(loss)
    # cross entropy against the softmax over legal actions only
    log_probs = jax.nn.log_softmax(jnp.where(mask, logits, -jnp.inf))
    expected = -jnp.where(mask, batch.policy_weights * log_probs, 0.0).sum(-1).mean()
    np.testing.assert_allclose(metrics["policy_loss"], expected, rtol=1e-6)
    assert all(jnp.isfinite(g).all() for g in jax.tree.leaves(grads))
    np.testing.assert_array_equal(grads.logits[~mask], 0.0)


def test_value_target_is_outcome_for_player_to_move():
    # the same game seen from each player: player 0 won
    batch = experience(rewards=[[1, -1], [1, -1]], cur_player_id=[0, 1],
                       policy_weights=[[0.5, 0.5]] * 2, policy_mask=[[True, True]] * 2)
    logits = jnp.zeros((2, 2))

    def value_loss(predicted):
        net = OutputAsParams(logits, jnp.array(predicted)[:, None])
        _, (metrics, _) = az_default_loss_fn(net, None, batch, l2_reg_lambda=0.0)
        return float(metrics["value_loss"])

    assert value_loss([1.0, -1.0]) == 0.0
    # optax.l2_loss is 0.5 * squared error
    np.testing.assert_allclose(value_loss([-1.0, 1.0]), 0.5 * 4, rtol=1e-6)


def test_a_few_optimizer_steps_lower_the_loss():
    config = AZResnetConfig(policy_head_out_size=9, num_blocks=1, num_channels=4)
    net, nn_state = eqx.nn.make_with_state(AZResnet)(config, (3, 3, 2), key=jax.random.PRNGKey(1))
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    batch_size = 16
    observation = jax.random.bernoulli(keys[0], shape=(batch_size, 3, 3, 2)).astype(jnp.float32)
    mask = jax.random.bernoulli(keys[1], p=0.7, shape=(batch_size, 9)).at[:, 0].set(True)
    policy = jax.nn.softmax(jnp.where(mask, jax.random.normal(keys[2], (batch_size, 9)), -jnp.inf))
    outcome = jnp.where(jax.random.bernoulli(keys[3], shape=(batch_size,)), 1.0, -1.0)
    batch = experience(rewards=jnp.stack([outcome, -outcome], -1), cur_player_id=jnp.arange(batch_size) % 2,
                       policy_weights=policy, policy_mask=mask, observation=observation)

    params, static = eqx.partition(net, eqx.is_inexact_array)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, nn_state, opt_state):
        grad_fn = eqx.filter_value_and_grad(az_default_loss_fn, has_aux=True)
        (loss, (_, nn_state)), grads = grad_fn(eqx.combine(params, static), nn_state, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), nn_state, opt_state, loss

    losses = []
    for _ in range(20):
        params, nn_state, opt_state, loss = train_step(params, nn_state, opt_state)
        losses.append(float(loss))

    assert all(np.isfinite(losses))
    assert losses[-1] < 0.8 * losses[0]
