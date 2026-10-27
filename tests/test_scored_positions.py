import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.testing import tester as tester_module
from core.testing.scored_positions import (
    METRICS,
    ScoredPositions,
    ScoredPositionsTester,
    scored_position_metrics,
)


def softmax(x):
    e = np.exp(np.asarray(x, dtype=np.float64) - np.max(x))
    return e / e.sum()


def test_metrics_measure_the_policy_and_value_against_the_engine():
    legal = np.array([[True, True, False], [True, True, True], [True, True, False]])
    scores = np.array([[5, 1, 0], [-2, -2, -6], [0, -3, 0]], dtype=np.float32)
    # 0: picks the move giving up 4; 1: picks one of two equally best; 2: picks the best
    # (the illegal action's huge logit is ignored)
    logits = np.array([[0, 1, 0], [3, 0, 1], [2, 0, 9]], dtype=np.float32)
    value = np.array([0.5, 0.2, -0.4], dtype=np.float32)
    groups = np.array([[True, True, True], [True, False, False]])

    table = np.asarray(
        scored_position_metrics(
            jnp.asarray(logits), jnp.asarray(value), legal, scores, groups
        )
    )

    regret = np.array([4.0, 0.0, 0.0])
    exp_regret = np.array(
        [
            softmax([0, 1]) @ [0, 4],
            softmax([3, 0, 1]) @ [0, 0, 4],
            softmax([2, 0]) @ [0, 3],
        ]
    )
    # the engine's predicted outcomes: win, loss, draw
    value_se = np.array([(0.5 - 1) ** 2, (0.2 + 1) ** 2, (-0.4 - 0) ** 2])
    expected_all = [
        2 / 3,
        regret.mean(),
        exp_regret.mean(),
        value_se.mean(),
        # the draw doesn't count: 1 of the 2 decided positions has the right sign
        0.5,
    ]
    np.testing.assert_allclose(table[0], expected_all, rtol=1e-5)
    np.testing.assert_allclose(
        table[1], [0, 4, exp_regret[0], value_se[0], 1], rtol=1e-5
    )


class Echo(eqx.Module):
    """A fake network whose outputs are in its input: the logits, then the value."""

    scale: jax.Array

    def __init__(self):
        self.scale = jnp.ones(())

    def __call__(self, x, state):
        return (x[:3] * self.scale, x[3] * self.scale), state


def make_positions(rng, n):
    legal = rng.random((n, 3)) < 0.7
    legal[:, 0] = True
    scores = rng.integers(-10, 10, size=(n, 3)).astype(np.float32)
    outputs = rng.normal(size=(n, 4)).astype(np.float32)
    outputs[:, 3] = np.tanh(outputs[:, 3])
    groups = {"odd": np.arange(n) % 2 == 1}
    return ScoredPositions(outputs, legal, scores, groups)


@pytest.mark.parametrize("batch_size", [2, 4, 16])
def test_the_tester_batches_and_pads_without_changing_the_metrics(batch_size):
    positions = make_positions(np.random.default_rng(0), 7)
    net, state = eqx.nn.make_with_state(Echo)()
    tester = ScoredPositionsTester(positions, net, batch_size=batch_size, name="edax")
    params = (eqx.filter(net, eqx.is_inexact_array), state)

    metrics = tester.evaluate(params)

    obs = positions.observations
    groups = np.stack([np.ones(7, bool), positions.groups["odd"]])
    table = np.asarray(
        scored_position_metrics(
            jnp.asarray(obs[:, :3]),
            jnp.asarray(obs[:, 3]),
            positions.legal,
            positions.scores,
            groups,
        )
    )
    assert set(metrics) == {f"edax_{m}" for m in METRICS} | {
        f"edax_{m}_odd" for m in METRICS
    }
    for i, m in enumerate(METRICS):
        np.testing.assert_allclose(metrics[f"edax_{m}"], table[0, i], rtol=1e-5)
        np.testing.assert_allclose(metrics[f"edax_{m}_odd"], table[1, i], rtol=1e-5)


def test_the_tester_measures_the_parameters_it_is_given():
    # the network's structure is fixed at construction, its parameters come with each run
    positions = make_positions(np.random.default_rng(1), 5)
    net, state = eqx.nn.make_with_state(Echo)()
    tester = ScoredPositionsTester(positions, net, epochs_per_test=3)
    params = eqx.filter(net, eqx.is_inexact_array)
    flipped = eqx.tree_at(lambda n: n.scale, params, -jnp.ones(()))

    def run(params, epoch):
        _, metrics, episode = tester.run(
            jax.random.PRNGKey(0),
            epoch,
            80,
            None,
            None,
            None,
            tester_module.TestState(),
            (params, state),
        )
        assert episode is None
        return metrics

    assert run(params, 1) == {}
    same, other = run(params, 3), run(flipped, 3)
    assert (
        set(same) == set(other)
        and same["engine_value_mse"] != other["engine_value_mse"]
    )
    assert tester.get_config()["positions"] == 5
    assert tester.get_config()["groups"] == ["odd"]
