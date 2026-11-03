import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.memory.replay_memory import BaseExperience
from core.training.loss_fns import anchored_loss_fn, az_default_loss_fn


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


class CountingCalls(eqx.Module):
    """Stateful `OutputAsParams`: its state counts the batches it has been applied to."""

    outputs: OutputAsParams
    calls: eqx.nn.StateIndex

    def __init__(self, policy_logits, value):
        self.outputs = OutputAsParams(policy_logits, value)
        self.calls = eqx.nn.StateIndex(jnp.array(0))

    def __call__(self, x, state):
        return self.outputs(x), state.set(self.calls, state.get(self.calls) + 1)


def experience(
    rewards,
    cur_player_id,
    policy_weights,
    policy_mask,
    observation=None,
    search_value=None,
):
    batch = len(cur_player_id)
    return BaseExperience(
        reward=jnp.asarray(rewards, dtype=jnp.float32),
        policy_weights=jnp.asarray(policy_weights, dtype=jnp.float32),
        policy_mask=jnp.asarray(policy_mask),
        observation_nn=jnp.arange(batch) if observation is None else observation,
        cur_player_id=jnp.asarray(cur_player_id, dtype=jnp.int32),
        search_value=jnp.zeros((batch,))
        if search_value is None
        else jnp.asarray(search_value, dtype=jnp.float32),
    )


def test_masked_policy_loss_is_finite_and_ignores_illegal_logits():
    mask = jnp.array([[True, False, True], [False, True, True]])
    batch = experience(
        rewards=[[1, -1], [1, -1]],
        cur_player_id=[0, 1],
        policy_weights=[[0.25, 0.0, 0.75], [0.0, 0.5, 0.5]],
        policy_mask=mask,
    )
    logits = jnp.array([[1.0, 50.0, -1.0], [-50.0, 0.5, 0.0]])

    net = OutputAsParams(logits, jnp.zeros((2, 1)))
    (loss, (metrics, _)), grads = eqx.filter_jit(
        eqx.filter_value_and_grad(az_default_loss_fn, has_aux=True)
    )(net, None, batch)

    assert jnp.isfinite(loss)
    # cross entropy against the softmax over legal actions only
    log_probs = jax.nn.log_softmax(jnp.where(mask, logits, -jnp.inf))
    expected = -jnp.where(mask, batch.policy_weights * log_probs, 0.0).sum(-1).mean()
    np.testing.assert_allclose(metrics["policy_loss"], expected, rtol=1e-6)
    # the loss splits into the targets' entropy and the KL divergence from them (zero targets
    # count for nothing in the entropy)
    w = np.asarray(batch.policy_weights, dtype=np.float64)
    entropy = -np.where(w > 0, w * np.log(np.where(w > 0, w, 1)), 0).sum(-1).mean()
    np.testing.assert_allclose(metrics["policy_target_entropy"], entropy, rtol=1e-6)
    np.testing.assert_allclose(
        metrics["policy_kl"], float(expected) - entropy, rtol=1e-5, atol=1e-6
    )
    assert metrics["policy_kl"] > 0
    assert all(jnp.isfinite(g).all() for g in jax.tree.leaves(grads))
    np.testing.assert_array_equal(grads.logits[~mask], 0.0)


def test_value_target_is_outcome_for_player_to_move():
    # the same game seen from each player: player 0 won
    batch = experience(
        rewards=[[1, -1], [1, -1]],
        cur_player_id=[0, 1],
        policy_weights=[[0.5, 0.5]] * 2,
        policy_mask=[[True, True]] * 2,
    )
    logits = jnp.zeros((2, 2))

    def value_loss(predicted):
        net = OutputAsParams(logits, jnp.array(predicted)[:, None])
        _, (metrics, _) = az_default_loss_fn(net, None, batch)
        return float(metrics["value_loss"])

    assert value_loss([1.0, -1.0]) == 0.0
    # optax.l2_loss is 0.5 * squared error
    np.testing.assert_allclose(value_loss([-1.0, 1.0]), 0.5 * 4, rtol=1e-6)


def test_value_target_mixes_in_the_search_value():
    # player 0 won; the search valued the positions at 0.5 and 0.2 for the player to move
    batch = experience(
        rewards=[[1, -1], [1, -1]],
        cur_player_id=[0, 1],
        policy_weights=[[0.5, 0.5]] * 2,
        policy_mask=[[True, True]] * 2,
        search_value=[0.5, 0.2],
    )
    net = OutputAsParams(jnp.zeros((2, 2)), jnp.zeros((2, 1)))

    def value_loss(value_target_q):
        _, (metrics, _) = az_default_loss_fn(
            net, None, batch, value_target_q=value_target_q
        )
        return float(metrics["value_loss"])

    def expected(targets):
        return 0.5 * np.mean(np.square(targets))

    # the outcome alone by default, the search value alone at 1, and an even mix at 0.5
    np.testing.assert_allclose(value_loss(0.0), expected([1.0, -1.0]), rtol=1e-6)
    np.testing.assert_allclose(value_loss(1.0), expected([0.5, 0.2]), rtol=1e-6)
    np.testing.assert_allclose(value_loss(0.5), expected([0.75, -0.4]), rtol=1e-6)


def two_positions():
    """A batch of two positions, with a network's (made-up) outputs for them."""
    batch = experience(
        rewards=[[1, -1], [1, -1]],
        cur_player_id=[0, 1],
        policy_weights=[[0.25, 0.75], [1.0, 0.0]],
        policy_mask=[[True, True]] * 2,
    )
    return batch, jnp.array([[0.5, -1.0], [2.0, 0.0]]), jnp.array([[0.5], [-0.25]])


def test_the_loss_is_the_policy_loss_plus_the_value_loss():
    batch, logits, value = two_positions()

    loss, (metrics, _) = az_default_loss_fn(OutputAsParams(logits, value), None, batch)

    np.testing.assert_allclose(
        loss, metrics["policy_loss"] + metrics["value_loss"], rtol=1e-6
    )


def test_network_state_comes_back_updated():
    batch, logits, value = two_positions()
    net, nn_state = eqx.nn.make_with_state(CountingCalls)(logits, value)

    _, (_, nn_state) = az_default_loss_fn(net, nn_state, batch)

    assert nn_state is not None
    assert int(nn_state.get(net.calls)) == 1


def reference_of(policy, value):
    """A frozen reference that gives every batch these outputs."""
    return lambda obs, mask: (jnp.asarray(policy), jnp.asarray(value))


def test_an_anchor_of_weight_zero_changes_nothing():
    batch, logits, value = two_positions()
    net = OutputAsParams(logits, value)
    ref = reference_of([[0.9, 0.1], [0.5, 0.5]], [0.3, -0.3])

    plain, (plain_metrics, _) = az_default_loss_fn(net, None, batch)
    anchored, (metrics, _) = anchored_loss_fn(net, None, batch, ref, 0.0, 0.0)

    np.testing.assert_allclose(anchored, plain, rtol=1e-6)
    for name in plain_metrics:
        np.testing.assert_allclose(metrics[name], plain_metrics[name], rtol=1e-6)


def test_an_anchor_blends_the_references_outputs_into_the_targets():
    batch, logits, value = two_positions()
    net = OutputAsParams(logits, value)
    ref_policy, ref_value = jnp.array([[0.9, 0.1], [0.5, 0.5]]), jnp.array([0.3, -0.3])

    _, (metrics, _) = anchored_loss_fn(
        net, None, batch, reference_of(ref_policy, ref_value), 0.25, 0.5
    )

    # the same as training on a quarter of the reference's policy and half of its value
    blended = experience(
        rewards=[[1, -1], [1, -1]],
        cur_player_id=[0, 1],
        policy_weights=0.75 * batch.policy_weights + 0.25 * ref_policy,
        policy_mask=[[True, True]] * 2,
        search_value=ref_value,
    )
    _, (expected, _) = az_default_loss_fn(net, None, blended, value_target_q=0.5)
    for name in ("policy_loss", "value_loss"):
        np.testing.assert_allclose(metrics[name], expected[name], rtol=1e-6)


def test_the_anchors_distance_is_zero_when_the_network_is_the_reference():
    batch, logits, value = two_positions()
    net = OutputAsParams(logits, value)
    ref = reference_of(jax.nn.softmax(logits), value.reshape(-1))

    _, (metrics, _) = anchored_loss_fn(net, None, batch, ref, 0.5, 0.5)

    np.testing.assert_allclose(metrics["anchor_policy_kl"], 0.0, atol=1e-6)
    np.testing.assert_allclose(metrics["anchor_value_mse"], 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="not both"):
        anchored_loss_fn(net, None, batch, ref, 0.5, 0.5, value_target_q=0.5)
