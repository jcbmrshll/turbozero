import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.training.validation import (
    METRICS,
    ScoredPositions,
    ScoredPositionsValidator,
    TwoPlayerValidator,
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
    # 0's second move gives up 4 but still wins, and every move of 1 loses as the best does:
    # only 2's second move (a draw to a loss) worsens the outcome
    result_change = np.array([0.0, 0.0, softmax([2, 0])[1]])
    # the engine's predicted outcomes: win, loss, draw
    value_se = np.array([(0.5 - 1) ** 2, (0.2 + 1) ** 2, (-0.4 - 0) ** 2])
    expected_all = [
        2 / 3,
        regret.mean(),
        exp_regret.mean(),
        value_se.mean(),
        # the draw doesn't count: 1 of the 2 decided positions has the right sign
        0.5,
        result_change.mean(),
    ]
    np.testing.assert_allclose(table[0], expected_all, rtol=1e-5)
    np.testing.assert_allclose(
        table[1], [0, 4, exp_regret[0], value_se[0], 1, 0], rtol=1e-5
    )


def test_result_change_is_measured_under_the_flattened_prior():
    # a win: the second move draws and the third loses, both worsen the outcome
    legal = np.array([[True, True, True]])
    scores = np.array([[3, 0, -1]], dtype=np.float32)
    logits = jnp.asarray([[2.0, 0.0, 0.0]])
    groups = np.array([[True]])
    result_change = METRICS.index("result_change")

    for temperature in (1.0, 2.0):
        table = scored_position_metrics(
            logits, jnp.zeros(1), legal, scores, groups, temperature
        )
        np.testing.assert_allclose(
            table[0, result_change],
            softmax([2 / temperature, 0, 0])[1:].sum(),
            rtol=1e-5,
        )


class Echo(eqx.Module):
    """A fake network whose outputs are in its input: the logits, then the value."""

    scale: jax.Array

    def __init__(self):
        self.scale = jnp.ones(())

    def __call__(self, x, state):
        return (x[:3] * self.scale, x[3] * self.scale), state


class RoundingEcho(eqx.Module):
    """Echo computing in a dtype it holds as a static field, as a network's inference dtype
    is."""

    scale: jax.Array
    dtype: str = eqx.field(static=True)

    def __init__(self, dtype):
        self.scale = jnp.ones(())
        self.dtype = dtype

    def __call__(self, x, state):
        x = (x * self.scale).astype(self.dtype)
        return (x[:3].astype(jnp.float32), x[3].astype(jnp.float32)), state


def make_positions(rng, n):
    legal = rng.random((n, 3)) < 0.7
    legal[:, 0] = True
    scores = rng.integers(-10, 10, size=(n, 3)).astype(np.float32)
    outputs = rng.normal(size=(n, 4)).astype(np.float32)
    outputs[:, 3] = np.tanh(outputs[:, 3])
    groups = {"odd": np.arange(n) % 2 == 1}
    return ScoredPositions(outputs, legal, scores, groups)


@pytest.mark.parametrize("batch_size", [2, 4, 16])
def test_the_validator_batches_and_pads_without_changing_the_metrics(batch_size):
    positions = make_positions(np.random.default_rng(0), 7)
    net, state = eqx.nn.make_with_state(Echo)()
    validator = ScoredPositionsValidator(
        positions, net, batch_size=batch_size, name="edax", prior_temperature=2.0
    )
    params = (eqx.filter(net, eqx.is_inexact_array), state)

    metrics = validator(params)

    obs = positions.observations
    groups = np.stack([np.ones(7, bool), positions.groups["odd"]])
    table = np.asarray(
        scored_position_metrics(
            jnp.asarray(obs[:, :3]),
            jnp.asarray(obs[:, 3]),
            positions.legal,
            positions.scores,
            groups,
            2.0,
        )
    )
    assert set(metrics) == {f"edax_{m}" for m in METRICS} | {
        f"edax_{m}_odd" for m in METRICS
    }
    for i, m in enumerate(METRICS):
        np.testing.assert_allclose(metrics[f"edax_{m}"], table[0, i], rtol=1e-5)
        np.testing.assert_allclose(metrics[f"edax_{m}_odd"], table[1, i], rtol=1e-5)


def test_the_validator_measures_the_parameters_it_is_given():
    # the network's structure is fixed at construction, its parameters come with each call
    positions = make_positions(np.random.default_rng(1), 5)
    net, state = eqx.nn.make_with_state(Echo)()
    validator = ScoredPositionsValidator(positions, net)
    params = eqx.filter(net, eqx.is_inexact_array)
    flipped = eqx.tree_at(lambda n: n.scale, params, -jnp.ones(()))

    same, other = validator((params, state)), validator((flipped, state))

    assert (
        set(same) == set(other)
        and same["engine_value_mse"] != other["engine_value_mse"]
    )
    assert validator.get_config()["positions"] == 5
    assert validator.get_config()["groups"] == ["odd"]


def test_the_validator_scores_weights_of_a_copy_computing_in_another_dtype():
    # a run's weights carry its network's bfloat16 setting; the validator's float32 copy
    # scores them in float32
    positions = make_positions(np.random.default_rng(2), 6)
    run_net, state = eqx.nn.make_with_state(RoundingEcho)("bfloat16")
    params = (eqx.filter(run_net, eqx.is_inexact_array), state)
    float32_net, _ = eqx.nn.make_with_state(RoundingEcho)("float32")
    echo, _ = eqx.nn.make_with_state(Echo)()

    in_float32 = ScoredPositionsValidator(positions, float32_net)(params)
    in_bfloat16 = ScoredPositionsValidator(positions, run_net)(params)
    exact = ScoredPositionsValidator(positions, echo)(
        (eqx.filter(echo, eqx.is_inexact_array), state)
    )

    assert in_float32["engine_value_mse"] == exact["engine_value_mse"]
    assert in_bfloat16["engine_value_mse"] != exact["engine_value_mse"]


MAX_STEPS = 10


@pytest.mark.parametrize(
    "rewards, win_rate, loss_rate",
    [
        # won by whoever moves first: with the agent first in exactly half the games, it wins half
        ((1.0, -1.0), 0.5, 0.5),
        # drawn
        ((0.0, 0.0), 0.0, 0.0),
    ],
)
def test_two_player_validator_reports_win_and_loss_rates(
    fixed_length_env, scripted, rewards, win_rate, loss_rate
):
    env = fixed_length_env(length=3, rewards=rewards)
    validator = TwoPlayerValidator(
        scripted.first_legal,
        scripted.first_legal,
        env.step_fn,
        env.init_fn,
        MAX_STEPS,
        num_episodes=8,
        name="baseline",
    )

    metrics = validator({"w": jnp.ones(3)})

    np.testing.assert_array_equal(metrics["baseline_win_rate"], win_rate)
    np.testing.assert_array_equal(metrics["baseline_loss_rate"], loss_rate)
    np.testing.assert_array_equal(
        metrics["baseline_avg_outcome"], (rewards[0] + rewards[1]) / 2
    )


def test_two_player_validator_plays_the_same_games_every_call(ttt):
    from core.evaluators.random_evaluator import RandomEvaluator

    validator = TwoPlayerValidator(
        RandomEvaluator(),
        RandomEvaluator(),
        ttt.step_fn,
        ttt.init_fn,
        9,
        num_episodes=64,
        name="random",
    )

    first, second = validator(None), validator(None)

    assert {k: float(v) for k, v in first.items()} == {
        k: float(v) for k, v in second.items()
    }


def test_two_player_validator_needs_an_even_number_of_episodes(ttt, scripted):
    with pytest.raises(ValueError, match="even"):
        TwoPlayerValidator(
            scripted.first_legal,
            scripted.first_legal,
            ttt.step_fn,
            ttt.init_fn,
            MAX_STEPS,
            num_episodes=3,
            name="baseline",
        )
