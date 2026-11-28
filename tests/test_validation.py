import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.evaluators.evaluator import EvalOutput, Evaluator
from core.training.selfplay_data import KL_FLOOR
from core.training.validation import (
    METRICS,
    ScoredPositions,
    ScoredPositionsValidator,
    SearchTargetValidator,
    TwoPlayerValidator,
    cluster_index,
    clustered_means,
    scored_position_metrics,
)
from core.types import StepMetadata


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


class TableSearch(Evaluator):
    """A fake search: on the position whose state is index i, its target is `targets[i]`, with
    `noise` times a random draw added, and its value `values[i]`."""

    def __init__(self, targets, values, noise=0.0):
        super().__init__(discount=-1.0)
        self.targets = jnp.asarray(targets, jnp.float32)
        self.values = jnp.asarray(values, jnp.float32)
        self.noise = noise

    def init(self, *args, **kwargs):
        return jnp.zeros((), jnp.float32)

    def reset(self, state):
        return state

    def evaluate(self, key, eval_state, env_state, root_metadata, **kwargs):
        target = self.targets[env_state] + self.noise * jax.random.uniform(
            key, self.targets.shape[1:]
        )
        return EvalOutput(
            eval_state=self.values[env_state],
            action=jnp.argmax(target),
            policy_weights=target / target.sum(),
        )

    def get_value(self, state):
        return state


def search_positions(n, clusters=None):
    """`n` positions of 3 legal actions, the first the engine's best (a win by 4), the others
    giving up 2 and 6; observations whose logits are uniform (see `Echo`), every other one in
    the group "odd"."""
    observations = np.zeros((n, 4), np.float32)
    scores = np.tile(np.array([4.0, 2.0, -2.0], np.float32), (n, 1))
    legal = np.ones((n, 3), bool)
    odd = {"odd": np.arange(n) % 2 == 1}
    return ScoredPositions(observations, legal, scores, odd, clusters)


def search_validator(evaluators, n=5, batch_size=2, clusters=None):
    positions = search_positions(n, clusters)
    net, state = eqx.nn.make_with_state(Echo)()
    metadata = StepMetadata(
        rewards=jnp.zeros((n, 2)),
        action_mask=jnp.asarray(positions.legal),
        terminated=jnp.zeros(n, bool),
        cur_player_id=jnp.zeros(n, jnp.int32),
        step=jnp.zeros(n, jnp.int32),
    )
    validator = SearchTargetValidator(
        positions,
        jnp.arange(n),
        metadata,
        evaluators,
        env_step_fn=lambda s, a: (s, metadata),
        nn=net,
        every=5,
        batch_size=batch_size,
    )
    return validator, (eqx.filter(net, eqx.is_inexact_array), state)


def test_search_targets_are_scored_against_the_engine_and_each_other():
    n = 5
    # the small budget's target is all on the second action, the bigger one's on the best
    small = TableSearch(np.tile([0.0, 1.0, 0.0], (n, 1)), np.full(n, 0.5))
    big = TableSearch(np.tile([1.0, 0.0, 0.0], (n, 1)), np.full(n, 1.0))
    validator, params = search_validator({"8": small, "16": big}, n=n)

    metrics = validator(params)

    assert metrics["search8_top1"] == pytest.approx(0.0)
    assert metrics["search8_regret"] == pytest.approx(2.0)
    assert metrics["search16_top1"] == pytest.approx(1.0)
    assert metrics["search16_regret"] == pytest.approx(0.0)
    assert metrics["search8_value_mse"] == pytest.approx(0.25)
    assert metrics["search16_value_mse_odd"] == pytest.approx(0.0)
    # all of the bigger budget's target is on an action the smaller gives only the floor
    assert metrics["search_kl_16_8"] == pytest.approx(-np.log(KL_FLOOR / 3), rel=1e-4)
    # a search without noise repeats itself exactly: only the floor's spread costs anything
    assert metrics["search_kl_8_repeat"] == pytest.approx(
        -np.log(1 - KL_FLOOR + KL_FLOOR / 3), rel=1e-3
    )
    # the prior is uniform
    assert metrics["search_kl_8_prior"] == pytest.approx(
        -np.log((1 - KL_FLOOR) / 3 + KL_FLOOR / 3), rel=1e-4
    )
    assert validator.every == 5
    assert set(validator.get_config()["evaluators"]) == {"8", "16"}


def test_search_validator_searches_with_the_same_keys_every_call():
    n = 5
    noisy = TableSearch(np.ones((n, 3)), np.zeros(n), noise=1.0)
    validator, params = search_validator({"8": noisy}, n=n)

    first, second = validator(params), validator(params)

    for name in first:
        assert float(first[name]) == pytest.approx(float(second[name]))
    # the repeat search draws other noise
    assert first["search_kl_8_repeat"] > 0


def clustered_reference(d, clusters):
    """The mean of `d` and its standard error, the positions of a cluster resampled together,
    each negative id alone: sqrt(sum over clusters of (sum of d - mean)^2) / N."""
    clusters = np.where(clusters < 0, clusters.max() + 1 + np.arange(len(d)), clusters)
    _, clusters = np.unique(clusters, return_inverse=True)
    sums = np.bincount(clusters, weights=d - d.mean())
    return d.mean(), np.sqrt((sums**2).sum()) / len(d)


def test_a_subset_keeps_each_positions_groups_and_cluster():
    positions = search_positions(4, clusters=np.array([2, 2, -1, 5]))

    subset = positions.subset(np.array([1, 3]))

    assert subset.groups["odd"].tolist() == [True, True]
    assert subset.clusters is not None and subset.clusters.tolist() == [2, 5]
    assert search_positions(4).subset(np.array([0])).clusters is None


def test_cluster_index_keeps_shared_ids_together_and_unknown_ones_apart():
    ids = np.array([5, 5, -1, 7, -1, 5])
    assert cluster_index(ids, 6).tolist() == [0, 0, 2, 1, 3, 0]
    assert cluster_index(np.array([-1, -1]), 2).tolist() == [0, 1]
    assert cluster_index(None, 3).tolist() == [0, 1, 2]


def test_clustered_means_resample_whole_clusters():
    rng = np.random.default_rng(0)
    n = 40
    values = rng.normal(size=(n, 2))
    # clusters of unequal sizes, some positions alone, one column counting on only some
    ids = rng.integers(-1, 6, size=n)
    weights = np.ones((n, 2))
    weights[:, 1] = rng.random(n) < 0.6
    groups = np.stack([np.ones(n, bool), np.arange(n) % 3 == 0])

    clusters = cluster_index(ids, n)
    means, errors = clustered_means(
        values, weights, groups, clusters, int(clusters.max()) + 1
    )

    for g, group in enumerate(groups):
        for m in range(2):
            keep = group & (weights[:, m] > 0)
            mean, error = clustered_reference(values[keep, m], ids[keep])
            assert means[g, m] == pytest.approx(mean, abs=1e-5)
            assert errors[g, m] == pytest.approx(error, rel=1e-4)
    # positions that move together widen the error past an independent-samples one
    alike = np.repeat(np.arange(4), 10)
    together = np.repeat(rng.normal(size=4), 10)[:, None]
    _, clustered = clustered_means(together, np.ones((n, 1)), groups[:1], alike, 4)
    _, alone = clustered_means(together, np.ones((n, 1)), groups[:1], np.arange(n), n)
    assert clustered[0, 0] == pytest.approx(np.sqrt(10) * alone[0, 0], rel=1e-4)


def test_search_budgets_are_compared_position_by_position_with_clustered_errors():
    n = 7
    # the small budget finds the engine's best on some positions, the big one on all
    small_found = np.array([1, 0, 0, 1, 0, 1, 0], bool)
    small = TableSearch(
        np.where(small_found[:, None], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        np.linspace(-0.5, 0.9, n),
    )
    big_values = np.linspace(0.2, 1.0, n)
    big = TableSearch(np.tile([1.0, 0.0, 0.0], (n, 1)), big_values)
    # clusters of 2 and 3 positions, two positions alone
    clusters = np.array([3, 3, -1, 8, 8, 8, -1])
    validator, params = search_validator(
        {"8": small, "16": big}, n=n, batch_size=2, clusters=clusters
    )

    metrics = validator(params)

    odd = np.arange(n) % 2 == 1
    differences = {
        "top1": 1.0 - small_found,
        "regret": -2.0 * ~small_found,
        "value_mse": (big_values - 1) ** 2 - (np.linspace(-0.5, 0.9, n) - 1) ** 2,
    }
    for metric, d in differences.items():
        for suffix, keep in (("", np.ones(n, bool)), ("_odd", odd)):
            mean, error = clustered_reference(d[keep], clusters[keep])
            name = f"search16_minus_8_{metric}{suffix}"
            assert metrics[name] == pytest.approx(mean, abs=1e-5)
            assert metrics[f"{name}_se"] == pytest.approx(error, rel=1e-4)
    # the KL's per-position values: the floor alone where the budgets agree
    kl = np.where(
        small_found, -np.log(1 - KL_FLOOR + KL_FLOOR / 3), -np.log(KL_FLOOR / 3)
    )
    mean, error = clustered_reference(kl, clusters)
    assert metrics["search_kl_16_8"] == pytest.approx(mean, rel=1e-4)
    assert metrics["search_kl_16_8_se"] == pytest.approx(error, rel=1e-4)
    # a search without noise repeats itself: every position's KL is the same
    assert metrics["search_kl_8_repeat_se"] == pytest.approx(0.0, abs=1e-5)
    assert validator.get_config()["clusters"] == 4
