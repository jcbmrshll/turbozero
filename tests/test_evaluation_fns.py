"""The leaf evaluation functions MCTS calls: a network's policy logits and value for one state."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from turbozero.evaluators.evaluation_fns import (
    make_nn_eval_fn,
    make_nn_eval_fn_no_params_callable,
)


class OffsetNet(eqx.Module):
    """A linear policy head, and a value that's its state: a stateful network."""

    linear: eqx.nn.Linear
    offset: eqx.nn.StateIndex

    def __init__(self, key):
        self.linear = eqx.nn.Linear(2, 3, key=key)
        self.offset = eqx.nn.StateIndex(jnp.array(0.0))

    def __call__(self, x, state):
        return (self.linear(x), state.get(self.offset)[None]), state


def test_nn_eval_fn_applies_the_params_and_state_it_is_given():
    net, _ = eqx.nn.make_with_state(OffsetNet)(jax.random.PRNGKey(0))
    other, state = eqx.nn.make_with_state(OffsetNet)(jax.random.PRNGKey(1))
    state = state.set(other.offset, jnp.array(3.0))
    eval_fn = make_nn_eval_fn(net, lambda s: s["obs"])
    x = jnp.array([1.0, -2.0])

    # the params and state of the trained network, not those it was made with
    logits, value = eval_fn(
        {"obs": x},
        (eqx.filter(other, eqx.is_inexact_array), state),
        jax.random.PRNGKey(0),
    )

    # for one state: no batch dimension
    assert logits.shape == (3,) and value.shape == ()
    np.testing.assert_allclose(logits, other.linear(x), rtol=1e-6)
    assert float(value) == 3.0


def test_no_params_eval_fn_calls_the_function_on_a_batch_of_one():
    def nn(x):
        assert x.shape == (1, 2)
        return x * 2, x.sum(axis=-1, keepdims=True)

    logits, value = make_nn_eval_fn_no_params_callable(nn, lambda s: s["obs"])(
        {"obs": jnp.array([1.0, 2.0])}, None, jax.random.PRNGKey(0)
    )

    np.testing.assert_array_equal(logits, [2.0, 4.0])
    assert value.shape == () and float(value) == 3.0
