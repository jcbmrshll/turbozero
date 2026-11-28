"""Tests for what the trainer records of its optimizer and loss: injected hyperparameters, and
describing functions."""

from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from core.training.loss_fns import az_default_loss_fn
from core.training.train import (
    _flatten,
    _to_config,
    describe_fn,
    injected_hyperparams,
    sgdw,
    with_hyperparams,
)

PARAMS = {"w": jnp.ones(3)}


def run(optimizer, steps) -> Any:
    """`optimizer`'s state after `steps` updates of `PARAMS`."""
    return run_from(optimizer, optimizer.init(PARAMS), steps)


def run_from(optimizer, state, steps) -> Any:
    """`state` after `steps` more updates of `PARAMS`."""
    for _ in range(steps):
        _, state = optimizer.update(PARAMS, state, PARAMS)
    return state


def test_injected_hyperparams():
    state = run(optax.inject_hyperparams(optax.adam)(learning_rate=0.5, b1=0.8), 2)

    everything = injected_hyperparams(state)
    assert set(everything) == {"learning_rate", "b1", "b2", "eps", "eps_root"}
    assert np.isclose(everything["b1"], 0.8)
    assert np.isclose(everything["learning_rate"], 0.5)
    # nested in another transform
    assert set(injected_hyperparams((optax.EmptyState(), state))) == set(everything)


def test_with_hyperparams_sets_what_the_next_update_uses():
    def sgd(weight_decay):
        return optax.inject_hyperparams(sgdw, static_args=("mask", "decoupled"))(
            learning_rate=1.0,
            momentum=0.0,
            weight_decay=weight_decay,
        )

    state = run(sgd(0.5), 2)

    changed: Any = with_hyperparams(
        state, {"learning_rate": 0.9, "weight_decay": 0.1, "missing": 3.0}
    )

    assert float(changed.hyperparams["weight_decay"]) == pytest.approx(0.1)
    assert (
        changed.hyperparams["weight_decay"].dtype
        == state.hyperparams["weight_decay"].dtype
    )
    assert set(changed.hyperparams) == set(state.hyperparams)
    # the restored optimizer's next update takes the new values: a zero gradient's update is
    # -learning rate * weight decay * w
    zeros = {"w": jnp.zeros(3)}
    updates: Any = sgd(0.5).update(zeros, changed, PARAMS)[0]
    np.testing.assert_allclose(updates["w"], -0.9 * 0.1 * np.ones(3), rtol=1e-6)


def test_injected_hyperparams_without_injection():
    assert injected_hyperparams(run(optax.adam(1e-3), 1)) == {}


def test_describe_fn():
    def loss(nn, nn_state, experience, scale=1.0, weight=2.0):
        return az_default_loss_fn(nn, nn_state, experience)

    assert describe_fn(partial(loss, weight=0.5)) == {
        "name": "loss",
        "scale": 1.0,
        "weight": 0.5,
    }
    assert describe_fn(az_default_loss_fn) == {"name": "az_default_loss_fn"}
    assert describe_fn(len) == {"name": "len"}


def test_flatten():
    assert _flatten(
        {"a": 1, "args": {"lr": 1e-3, "nested": {"x": 2}}, "empty": {}}
    ) == {
        "a": 1,
        "args.lr": 1e-3,
        "args.nested.x": 2,
        "empty": {},
    }


def test_to_config_gives_floats_as_their_shortest_decimals():
    assert _to_config(jnp.float32(0.9)) == 0.9
    assert _to_config(jnp.array([0.1, 1e-8], jnp.float32)) == [0.1, 1e-8]
    assert _to_config(jnp.int32(3)) == 3
