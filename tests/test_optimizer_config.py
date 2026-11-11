"""Tests for what the trainer records of its optimizer and loss: restarting learning rate
schedules, injected hyperparameters, and describing functions."""

from functools import partial
from typing import Any, cast

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
    restart_schedules,
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


def test_restart_schedules_restarts_a_learning_rate_schedule():
    adam, schedule = run(optax.adam(optax.linear_schedule(1.0, 0.0, 10)), 3)

    restarted_adam, restarted_schedule = cast(
        tuple, restart_schedules((adam, schedule))
    )

    assert int(schedule.count) == 3 and int(restarted_schedule.count) == 0
    # Adam's moments and bias correction carry on
    assert int(restarted_adam.count) == 3
    np.testing.assert_array_equal(restarted_adam.mu["w"], adam.mu["w"])


def test_restart_schedules_restarts_injected_schedules():
    optimizer = optax.inject_hyperparams(optax.adam)(
        learning_rate=optax.linear_schedule(1.0, 0.0, 10)
    )
    state = run(optimizer, 3)

    restarted: Any = run_from(optimizer, restart_schedules(state), 1)

    # the update after the restart uses the schedule's first learning rate
    assert float(restarted.hyperparams["learning_rate"]) == 1.0
    adam = restarted.inner_state[0]
    assert int(adam.count) == 4


def test_injected_hyperparams():
    state = run(
        optax.inject_hyperparams(optax.adam)(
            learning_rate=optax.linear_schedule(1.0, 0.0, 10), b1=0.8
        ),
        2,
    )

    # the learning rate the last update used
    assert {
        k: float(v) for k, v in injected_hyperparams(state, scheduled_only=True).items()
    } == {"learning_rate": pytest.approx(0.9)}
    everything = injected_hyperparams(state)
    assert set(everything) == {"learning_rate", "b1", "b2", "eps", "eps_root"}
    assert np.isclose(everything["b1"], 0.8)
    # nested in another transform
    assert set(injected_hyperparams((optax.EmptyState(), state))) == set(everything)


def test_with_hyperparams_sets_what_the_next_update_uses():
    def sgd(weight_decay):
        return optax.inject_hyperparams(sgdw, static_args=("mask", "decoupled"))(
            learning_rate=optax.linear_schedule(1.0, 0.5, 10),
            momentum=0.0,
            weight_decay=weight_decay,
        )

    state = run(sgd(0.5), 2)

    changed: Any = with_hyperparams(state, {"weight_decay": 0.1, "missing": 3.0})

    assert float(changed.hyperparams["weight_decay"]) == pytest.approx(0.1)
    assert changed.hyperparams["weight_decay"].dtype == state.hyperparams["weight_decay"].dtype
    assert set(changed.hyperparams) == set(state.hyperparams)
    # the restored optimizer's next update decays at the new rate: a zero gradient's update is
    # -learning rate * weight decay * w (the schedule's third rate, 0.9)
    zeros = {"w": jnp.zeros(3)}
    updates, _ = sgd(0.5).update(zeros, changed, PARAMS)
    np.testing.assert_allclose(updates["w"], -0.9 * 0.1 * np.ones(3), rtol=1e-6)


def test_injected_hyperparams_without_injection():
    assert injected_hyperparams(run(optax.adam(1e-3), 1)) == {}


def test_describe_fn():
    assert describe_fn(partial(az_default_loss_fn, value_target_q=0.5)) == {
        "name": "az_default_loss_fn",
        "value_target_q": 0.5,
    }
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
