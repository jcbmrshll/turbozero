from functools import partial

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from core.memory.replay_memory import BaseExperience
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.training.loss_fns import az_default_loss_fn
from core.training.train import TrainStateWithBS


def output_as_params(policy_logits, value):
    """Params and a TrainState for a network that ignores its input and returns its params
    (policy logits, value) as the output, so gradients flow straight to the outputs."""
    def apply_fn(variables, x, train, mutable):  # pylint: disable=unused-argument
        return (variables["params"]["logits"], variables["params"]["value"]), {}
    params = {"logits": jnp.asarray(policy_logits, dtype=jnp.float32),
              "value": jnp.asarray(value, dtype=jnp.float32)}
    return params, TrainState.create(apply_fn=apply_fn, params=params, tx=optax.sgd(0.0))


def experience(rewards, cur_player_id, policy_weights, policy_mask, observation=None):
    batch = len(cur_player_id)
    return BaseExperience(
        reward=jnp.asarray(rewards, dtype=jnp.float32),
        policy_weights=jnp.asarray(policy_weights, dtype=jnp.float32),
        policy_mask=jnp.asarray(policy_mask),
        observation_nn=jnp.zeros((batch, 1)) if observation is None else observation,
        cur_player_id=jnp.asarray(cur_player_id, dtype=jnp.int32),
    )


def test_masked_policy_loss_is_finite_and_ignores_illegal_logits():
    mask = jnp.array([[True, False, True], [False, True, True]])
    batch = experience(rewards=[[1, -1], [1, -1]], cur_player_id=[0, 1],
                       policy_weights=[[0.25, 0.0, 0.75], [0.0, 0.5, 0.5]], policy_mask=mask)
    logits = jnp.array([[1.0, 50.0, -1.0], [-50.0, 0.5, 0.0]])

    loss_fn = partial(az_default_loss_fn, l2_reg_lambda=0.0)
    params, state = output_as_params(logits, jnp.zeros((2, 1)))
    (loss, (metrics, _)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, batch)

    assert jnp.isfinite(loss)
    # cross entropy against the softmax over legal actions only
    log_probs = jax.nn.log_softmax(jnp.where(mask, logits, -jnp.inf))
    expected = -jnp.where(mask, batch.policy_weights * log_probs, 0.0).sum(-1).mean()
    np.testing.assert_allclose(metrics["policy_loss"], expected, rtol=1e-6)
    assert all(jnp.isfinite(g).all() for g in jax.tree.leaves(grads))
    np.testing.assert_array_equal(grads["logits"][~mask], 0.0)


def test_value_target_is_outcome_for_player_to_move():
    # the same game seen from each player: player 0 won
    batch = experience(rewards=[[1, -1], [1, -1]], cur_player_id=[0, 1],
                       policy_weights=[[0.5, 0.5]] * 2, policy_mask=[[True, True]] * 2)
    logits = jnp.zeros((2, 2))

    def value_loss(predicted):
        params, state = output_as_params(logits, jnp.array(predicted)[:, None])
        _, (metrics, _) = az_default_loss_fn(params, state, batch, l2_reg_lambda=0.0)
        return float(metrics["value_loss"])

    assert value_loss([1.0, -1.0]) == 0.0
    # optax.l2_loss is 0.5 * squared error
    np.testing.assert_allclose(value_loss([-1.0, 1.0]), 0.5 * 4, rtol=1e-6)


def test_a_few_optimizer_steps_lower_the_loss():
    net = AZResnet(AZResnetConfig(policy_head_out_size=9, num_blocks=1, num_channels=4))
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    batch_size = 16
    observation = jax.random.bernoulli(keys[0], shape=(batch_size, 3, 3, 2)).astype(jnp.float32)
    mask = jax.random.bernoulli(keys[1], p=0.7, shape=(batch_size, 9)).at[:, 0].set(True)
    policy = jax.nn.softmax(jnp.where(mask, jax.random.normal(keys[2], (batch_size, 9)), -jnp.inf))
    outcome = jnp.where(jax.random.bernoulli(keys[3], shape=(batch_size,)), 1.0, -1.0)
    batch = experience(rewards=jnp.stack([outcome, -outcome], -1), cur_player_id=jnp.arange(batch_size) % 2,
                       policy_weights=policy, policy_mask=mask, observation=observation)

    variables = net.init(jax.random.PRNGKey(1), observation[:1], train=False)
    state = TrainStateWithBS.create(apply_fn=net.apply, params=variables["params"],
                                    batch_stats=variables["batch_stats"], tx=optax.adam(1e-2))

    @jax.jit
    def train_step(state):
        grad_fn = jax.value_and_grad(az_default_loss_fn, has_aux=True)
        (loss, (_, updates)), grads = grad_fn(state.params, state, batch)
        state = state.apply_gradients(grads=grads).replace(batch_stats=updates["batch_stats"])
        return state, loss

    losses = []
    for _ in range(20):
        state, loss = train_step(state)
        losses.append(float(loss))

    assert all(np.isfinite(losses))
    assert losses[-1] < 0.8 * losses[0]
