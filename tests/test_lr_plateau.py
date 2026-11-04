"""Tests of `LRPlateau`: the rule, on made-up losses, and the learning rate schedule it is (its\ntrigger, `Plateau`, has its own tests in test_plateau.py)."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest

from core.training.lr_plateau import LRPlateau
from core.training.plateau import Plateau, PlateauProgress
from core.training.train import lr_plateau_states, restart_schedules, with_rate_index

PARAMS = {"w": jnp.ones(3)}


def run_rule(rule, losses, index=0):
    """Applies `rule` to `losses`, one an epoch, from a fresh start at rate `index`: the epochs
    at which a rate plateaued, and the rate index after each epoch."""
    progress, plateaus, indices = PlateauProgress(), [], []
    for epoch, loss in enumerate(losses):
        step = rule.after_epoch(progress, index, loss)
        progress, index = step.progress, step.index
        if step.plateaued:
            plateaus.append(epoch)
        indices.append(index)
    return plateaus, indices


def test_a_rate_drops_to_the_next_when_its_plateau_fires():
    rule = LRPlateau((1e-3, 1e-4, 1e-5), Plateau(window=4, threshold=0.1, min_epochs=4))
    # falling 0.1 an epoch for 6 epochs (0.4 over 4), then flat
    losses = [1.0 - 0.1 * min(e, 6) for e in range(20)]

    plateaus, indices = run_rule(rule, losses)

    # see test_plateau.py: it fires at 9, 13 and 17
    assert plateaus == [9, 13, 17]
    assert indices[8] == 0 and indices[9] == 1 and indices[13] == 2
    # the last rate stays
    assert indices[-1] == 2


def test_by_default_it_follows_the_training_loss():
    rule = LRPlateau((1e-3, 1e-4))
    assert (rule.plateau.signal, rule.plateau.maximize) == ("loss", False)
    assert (rule.plateau.window, rule.plateau.threshold) == (40, 0.004)
    assert rule.get_config() == {
        "rates": [1e-3, 1e-4],
        "signal": "loss",
        "maximize": False,
        "window": 40,
        "threshold": 0.004,
        "min_epochs": 40,
        "stop": False,
    }


@pytest.mark.parametrize("rates", [(), (1e-3, 0.0)])
def test_bad_rates_are_refused(rates):
    with pytest.raises(ValueError, match="rates must be positive"):
        LRPlateau(rates)


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
