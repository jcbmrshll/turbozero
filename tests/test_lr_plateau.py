"""Tests of `LRPlateau`: the rule, on made-up losses, and the learning rate schedule it is."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest

from core.training.lr_plateau import (
    LRPlateau,
    LRPlateauProgress,
    improvement,
)
from core.training.train import lr_plateau_states, restart_schedules, with_rate_index

PARAMS = {"w": jnp.ones(3)}


def test_improvement_is_the_fitted_lines_fall_over_the_window():
    assert improvement([1.0 - 0.001 * i for i in range(40)]) == pytest.approx(0.04)
    assert improvement([0.5] * 10) == pytest.approx(0.0)
    # a rising loss improves by less than nothing
    assert improvement([0.0, 1.0]) == pytest.approx(-2.0)
    # the line through noise, not its ends
    assert improvement([1.0, 0.0, 1.0, 0.0, 1.0]) == pytest.approx(0.0)


def run_rule(rule, losses, index=0):
    """Applies `rule` to `losses`, one an epoch, from a fresh start at rate `index`: the epochs
    at which a rate plateaued, and the rate index after each epoch."""
    progress, plateaus, indices = LRPlateauProgress(), [], []
    for epoch, loss in enumerate(losses):
        step = rule.after_epoch(progress, index, loss)
        progress, index = step.progress, step.index
        if step.plateaued:
            plateaus.append(epoch)
        indices.append(index)
    return plateaus, indices


def test_a_rate_drops_when_its_loss_stops_falling():
    rule = LRPlateau((1e-3, 1e-4, 1e-5), window=4, threshold=0.1, min_epochs=4)
    # falling 0.1 an epoch for 6 epochs (0.4 over 4), then flat
    losses = [1.0 - 0.1 * min(e, 6) for e in range(20)]

    plateaus, indices = run_rule(rule, losses)

    # the first window whose line falls less than 0.1 is epochs 6-9 (epochs 5-8's falls 0.12);
    # then the window starts again at the new rate, and plateaus as soon as it's full
    assert plateaus == [9, 13, 17]
    assert indices[8] == 0 and indices[9] == 1 and indices[13] == 2
    # the last rate stays
    assert indices[-1] == 2


def test_a_rate_is_kept_for_min_epochs():
    rule = LRPlateau((1e-3, 1e-4), window=2, threshold=0.1, min_epochs=5)

    plateaus, _ = run_rule(rule, [1.0] * 6)

    # flat from the start, its window full at epoch 1: it waits for epoch 4, the fifth
    assert plateaus == [4]


def test_the_window_only_holds_epochs_at_the_current_rate():
    rule = LRPlateau((1e-3, 1e-4, 1e-5), window=3, threshold=0.1, min_epochs=1)
    # flat, then falling fast right after the first drop
    losses = [1.0, 1.0, 1.0, 0.5, 0.0, -0.5, -0.5, -0.5]

    plateaus, _ = run_rule(rule, losses)

    # epochs 3-5 fall, so the next plateau is at the first flat window, epochs 5-7
    assert plateaus == [2, 7]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"rates": ()}, "rates must be positive"),
        ({"rates": (1e-3, 0.0)}, "rates must be positive"),
        ({"rates": (1e-3,), "window": 1}, "window must be at least 2"),
        ({"rates": (1e-3,), "min_epochs": 0}, "min_epochs must be positive"),
    ],
)
def test_bad_settings_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LRPlateau(**kwargs)


def optimizer_and_state(rule) -> tuple[Any, Any]:
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=rule)
    return optimizer, optimizer.init(PARAMS)


def test_the_schedule_trains_at_the_current_rate_without_compiling_again():
    rule = LRPlateau((1e-3, 1e-4, 1e-5))
    optimizer, state = optimizer_and_state(rule)
    traces = []

    @jax.jit
    def update(updates, state, params):
        traces.append(state)
        return optimizer.update(updates, state, params)

    rates = []
    for rate_index in (0, 0, 1, 2):
        state = with_rate_index(state, rate_index)
        _, state = update(PARAMS, state, PARAMS)
        rates.append(float(state.hyperparams["learning_rate"]))

    assert rates == [pytest.approx(r) for r in (1e-3, 1e-3, 1e-4, 1e-5)]
    assert len(traces) == 1
    (plateau_state,) = lr_plateau_states(state)
    assert int(plateau_state.rate_index) == 2


def test_restarting_the_schedules_starts_again_at_the_first_rate():
    optimizer, state = optimizer_and_state(LRPlateau((1e-3, 1e-4)))
    state = with_rate_index(state, 1)

    _, restarted = optimizer.update(PARAMS, restart_schedules(state), PARAMS)

    assert float(restarted.hyperparams["learning_rate"]) == pytest.approx(1e-3)


def test_a_plateau_optimizers_state_has_a_fixed_schedules_arrays():
    """So that a fork can be either, from a state saved by either."""
    _, plateau = optimizer_and_state(LRPlateau((1e-3, 1e-4)))
    _, schedule = optimizer_and_state(optax.linear_schedule(1e-3, 0.0, 10))

    def described(tree):
        return [(x.shape, x.dtype) for x in jax.tree.leaves(tree)]

    assert described(plateau) == described(schedule)
