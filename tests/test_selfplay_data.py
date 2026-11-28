import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.training.selfplay_data import (
    KL_FLOOR,
    PAIR_METRICS,
    POLICY_METRICS,
    VALUE_METRICS,
    floored_kl,
    group_sample,
    named,
    pair_metrics,
    policy_metrics,
    value_metrics,
)


def column(table, metrics, name):
    return np.asarray(table)[:, metrics.index(name)]


def test_policy_metrics_compare_the_targets_with_the_prior_over_legal_actions():
    mask = np.array([[True, True, False], [True, True, True]])
    # 0: the target is the prior (the illegal action's logit is ignored); 1: search moved the
    # whole target to the prior's least likely action
    target = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]], np.float32)
    logits = np.array([[0.0, 0.0, 9.0], [2.0, 0.0, 0.0]], np.float32)
    search_value = np.array([0.95, 0.1], np.float32)
    groups = np.array([[True, True], [True, False], [False, True]])

    table = policy_metrics(
        jnp.asarray(target),
        jnp.asarray(logits),
        jnp.asarray(mask),
        jnp.asarray(search_value),
        jnp.ones(2),
        jnp.asarray(groups),
    )

    prior_1 = np.exp([2.0, 0.0, 0.0]) / np.exp([2.0, 0.0, 0.0]).sum()
    floored = (1 - KL_FLOOR) * prior_1[2] + KL_FLOOR / 3
    np.testing.assert_allclose(
        column(table, POLICY_METRICS, "target_entropy"),
        [np.log(2) / 2, np.log(2), 0],
        atol=1e-6,
    )
    entropy_1 = -(prior_1 * np.log(prior_1)).sum()
    np.testing.assert_allclose(
        column(table, POLICY_METRICS, "prior_entropy")[1:],
        [np.log(2), entropy_1],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        column(table, POLICY_METRICS, "target_prior_kl")[1:],
        [0, -np.log(floored)],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        column(table, POLICY_METRICS, "top_move_change"), [0.5, 0, 1]
    )
    np.testing.assert_allclose(
        column(table, POLICY_METRICS, "decided_share"), [0.5, 1, 0]
    )


def test_padding_and_empty_groups_count_for_nothing():
    mask = np.ones((2, 2), bool)
    target = np.array([[1.0, 0.0], [0.5, 0.5]], np.float32)
    groups = np.array([[True, True], [False, False]])

    table = policy_metrics(
        jnp.asarray(target),
        jnp.zeros((2, 2)),
        jnp.asarray(mask),
        jnp.zeros(2),
        jnp.array([1.0, 0.0]),
        jnp.asarray(groups),
    )

    # only the first position counts; the empty group reports zeros, not NaN
    assert column(table, POLICY_METRICS, "target_entropy")[0] == pytest.approx(
        0, abs=1e-6
    )
    np.testing.assert_array_equal(np.asarray(table)[1], 0)


def test_value_metrics_compare_the_values_with_the_outcomes():
    search_value = np.array([0.5, -0.5, 0.0], np.float32)
    value = np.array([1.0, 0.2, -0.4], np.float32)
    outcome = np.array([1.0, -1.0, 0.0], np.float32)
    groups = np.array([[True, True, True], [True, False, True]])

    table = np.asarray(
        value_metrics(
            jnp.asarray(search_value),
            jnp.asarray(value),
            jnp.asarray(outcome),
            jnp.ones(3),
            jnp.asarray(groups),
        )
    )

    np.testing.assert_allclose(
        table[:, VALUE_METRICS.index("label_value_mse")], [0.5 / 3, 0.125], rtol=1e-6
    )
    np.testing.assert_allclose(
        table[:, VALUE_METRICS.index("value_bias")], [0.8 / 3, -0.2], rtol=1e-5
    )
    np.testing.assert_allclose(
        table[:, VALUE_METRICS.index("outcome")], [0, 0.5], atol=1e-7
    )


def test_pairs_are_compared_at_their_closest_transform():
    # 0 and 1 are one position, the second reversed (a transform maps one to the other); 2
    # differs from 0 in two entries at either transform
    observations = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 0]], np.float32)
    images = jnp.asarray(np.stack([observations, observations[:, ::-1]]))

    distance, collision = np.asarray(pair_metrics(images, jnp.ones(3, bool)))

    # ordered pairs: (0, 1) and (1, 0) at 0, the four with 2 at 2
    assert distance == pytest.approx(8 / 6)
    assert collision == pytest.approx(2 / 6)
    # without the reversal, 0 and 1 differ in two entries too, and nothing collides
    distance, collision = np.asarray(pair_metrics(images[:1], jnp.ones(3, bool)))
    assert collision == 0
    assert distance == pytest.approx((2 + 2 + 2 + 2 + 4 + 4) / 6)


def test_pairs_skip_padding():
    observations = jnp.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    distance, collision = np.asarray(
        pair_metrics(observations[None], jnp.array([True, True, False]))
    )

    assert (distance, collision) == (0, 1)


def test_a_group_sample_takes_members_only_and_pads_the_rest():
    members = jnp.array([False, True, True, False, True])

    indices, valid = group_sample(jax.random.PRNGKey(0), members, 4)

    indices, valid = np.asarray(indices), np.asarray(valid)
    assert sorted(indices[valid]) == [1, 2, 4]
    assert valid.sum() == 3


def test_floored_kl_is_zero_for_equal_distributions_and_finite_off_support():
    mask = jnp.array([[True, True, True]])
    p = jnp.array([[0.2, 0.8, 0.0]])

    assert float(floored_kl(p, p, mask)[0]) == pytest.approx(0, abs=1e-3)
    assert np.isfinite(float(floored_kl(p, jnp.array([[1.0, 0.0, 0.0]]), mask)[0]))


def test_names_leave_all_positions_unsuffixed():
    table = jnp.arange(4.0).reshape(2, 2)

    assert named(table, PAIR_METRICS, ["all", "late"]) == {
        "data_pair_distance": 0.0,
        "data_collision": 1.0,
        "data_pair_distance_late": 2.0,
        "data_collision_late": 3.0,
    }
