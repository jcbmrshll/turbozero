"""The leaf evaluation functions MCTS calls: a network's policy logits and value for one state."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from turbozero.evaluators.leaf_eval import make_nn_leaf_eval_fn
from turbozero.networks.nn_utils import BATCH_AXIS


class OffsetNet(eqx.Module):
    """A linear policy head, and a value that's its state: a stateful network."""

    linear: eqx.nn.Linear
    offset: eqx.nn.StateIndex

    def __init__(self, key):
        self.linear = eqx.nn.Linear(2, 3, key=key)
        self.offset = eqx.nn.StateIndex(jnp.array(0.0))

    def __call__(self, x, state):
        return (self.linear(x), state.get(self.offset)[None]), state


def test_nn_leaf_eval_fn_applies_the_params_and_state_it_is_given():
    net, _ = eqx.nn.make_with_state(OffsetNet)(jax.random.PRNGKey(0))
    other, state = eqx.nn.make_with_state(OffsetNet)(jax.random.PRNGKey(1))
    state = state.set(other.offset, jnp.array(3.0))
    leaf_eval_fn = make_nn_leaf_eval_fn(net, lambda s: s["obs"])
    x = jnp.array([1.0, -2.0])

    # the params and state of the trained network, not those it was made with
    logits, value = leaf_eval_fn(
        {"obs": x},
        (eqx.filter(other, eqx.is_inexact_array), state),
        jax.random.PRNGKey(0),
    )

    # for one state: no batch dimension
    assert logits.shape == (3,) and value.shape == ()
    np.testing.assert_allclose(logits, other.linear(x), rtol=1e-6)
    assert float(value) == 3.0


class NormNet(eqx.Module):
    """A linear layer, then BatchNorm: the policy logits, and their sum the value."""

    linear: eqx.nn.Linear
    norm: eqx.nn.BatchNorm

    def __init__(self, key):
        self.linear = eqx.nn.Linear(2, 3, key=key)
        self.norm = eqx.nn.BatchNorm(3, axis_name=BATCH_AXIS, mode="ema")

    def __call__(self, x, state):
        logits, state = self.norm(self.linear(x), state)
        return (logits, logits.sum()), state


def test_nn_leaf_eval_fn_normalizes_with_the_running_statistics():
    net, state = eqx.nn.make_with_state(NormNet)(jax.random.PRNGKey(0))
    mean, var = jnp.array([1.0, -1.0, 0.5]), jnp.array([4.0, 0.25, 1.0])
    assert net.norm.ema_state_index is not None
    state = state.set(net.norm.ema_state_index, (mean, var))
    x = jnp.array([1.0, -2.0])

    logits, value = make_nn_leaf_eval_fn(net, lambda s: s)(
        x, (eqx.filter(net, eqx.is_inexact_array), state), jax.random.PRNGKey(0)
    )

    # the batch of one's own statistics would normalize every logit to 0
    expected = (net.linear(x) - mean) / jnp.sqrt(var + net.norm.eps)
    np.testing.assert_allclose(logits, expected, rtol=1e-5)
    np.testing.assert_allclose(value, expected.sum(), rtol=1e-5)


class LinearNet(eqx.Module):
    """A stateless network: linear policy logits, and their sum the value."""

    linear: eqx.nn.Linear

    def __init__(self, key):
        self.linear = eqx.nn.Linear(2, 3, key=key)

    def __call__(self, x):
        logits = self.linear(x)
        return logits, logits.sum()


def test_nn_leaf_eval_fn_applies_a_stateless_network():
    net, other = LinearNet(jax.random.PRNGKey(0)), LinearNet(jax.random.PRNGKey(1))
    x = jnp.array([1.0, -2.0])

    logits, value = make_nn_leaf_eval_fn(net, lambda s: s)(
        x, (eqx.filter(other, eqx.is_inexact_array), None), jax.random.PRNGKey(0)
    )

    assert logits.shape == (3,) and value.shape == ()
    np.testing.assert_allclose(logits, other.linear(x), rtol=1e-6)
    np.testing.assert_allclose(value, other.linear(x).sum(), rtol=1e-5)
