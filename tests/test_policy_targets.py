"""Policy targets (core.evaluators.mcts.policy_targets): the visit distribution, and Gumbel MuZero's
completed-Q improved policy, and visits with root noise's forced visits pruned, on hand-built roots."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.policy_targets import (
    CompletedQ,
    PrunedVisitCounts,
    VisitCounts,
    completed_q_values,
    mixed_value,
)
from core.evaluators.mcts.state import MCTSNode, init_mcts_tree

MIN = float(jnp.finfo(jnp.float32).min)

# a root with prior [0.5, 0.3, 0.2] and network value 0.2, whose children along actions 0 and 1
# have 3 and 1 visits and are worth 0.6 and -0.2 to the root's player:
PRIOR = np.array([0.5, 0.3, 0.2])
RAW_VALUE = 0.2
VISITS = np.array([3, 1, 0])
Q = np.array([0.6, -0.2, 0.0])
# the visited children's prior-weighted Q, (0.5 * 0.6 + 0.3 * -0.2) / 0.8 = 0.3, mixed with the
# network's value by visits, (0.2 + 4 * 0.3) / 5
MIXED = 0.28
# [0.6, -0.2, 0.28], rescaled to [0, 1]
COMPLETED = np.array([1.0, 0.0, 0.6])
# (50 + 3) * 0.1 per unit of completed Q
SIGMA = 5.3 * COMPLETED


def node(n, q, num_actions=3):
    return MCTSNode(
        n=jnp.array(n, dtype=jnp.int32),
        p=jnp.full((num_actions,), 1 / num_actions),
        q=jnp.array(q, dtype=jnp.float32),
        terminated=jnp.array(False),
        embedding=jnp.array(0),
    )


def root_tree(logits, raw_value, visits, q, discount=-1.0):
    """A tree whose root has the network's `logits` and `raw_value` and a child along each action
    with visits, worth `q` to the root's player (so `q / discount` to the child's)."""
    tree = init_mcts_tree(8, len(logits), node(0, 0.0, len(logits)))
    tree = tree.set_root(node(1 + sum(visits), 0.0, len(logits)))
    for action, (n, value) in enumerate(zip(visits, q)):
        if n > 0:
            tree = tree.add_node(0, action, node(n, value / discount, len(logits)))
    return replace(
        tree,
        root_logits=jnp.asarray(logits, dtype=jnp.float32),
        root_value=jnp.array(raw_value, dtype=jnp.float32),
    )


def test_mixed_value_by_hand():
    value = mixed_value(
        RAW_VALUE, jnp.asarray(Q), jnp.asarray(VISITS), jnp.asarray(PRIOR)
    )

    assert float(value) == pytest.approx(MIXED, rel=1e-6)


def test_without_visits_the_mixed_value_is_the_networks():
    value = mixed_value(
        RAW_VALUE, jnp.zeros(3), jnp.zeros(3, dtype=jnp.int32), jnp.asarray(PRIOR)
    )

    assert float(value) == pytest.approx(RAW_VALUE)


def test_completed_q_values_by_hand():
    completed = completed_q_values(
        RAW_VALUE, jnp.asarray(Q), jnp.asarray(VISITS), jnp.asarray(PRIOR), 1e-8
    )

    np.testing.assert_allclose(completed, COMPLETED, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("discount", [-1.0, 1.0])
def test_the_completed_q_target_by_hand(discount):
    tree = root_tree(np.log(PRIOR), RAW_VALUE, VISITS, Q, discount)

    target = CompletedQ()(tree, discount)

    expected = PRIOR * np.exp(SIGMA)
    np.testing.assert_allclose(target, expected / expected.sum(), rtol=1e-5)


def test_value_scale_and_maxvisit_init_scale_the_completed_q_values():
    tree = root_tree(np.log(PRIOR), RAW_VALUE, VISITS, Q)

    target = CompletedQ(value_scale=1.0, maxvisit_init=0.0)(tree, -1.0)

    # (0 + 3) * 1.0 per unit of completed Q
    expected = PRIOR * np.exp(3.0 * COMPLETED)
    np.testing.assert_allclose(target, expected / expected.sum(), rtol=1e-5)


def test_without_visits_the_completed_q_target_is_the_networks_policy():
    tree = root_tree(np.log(PRIOR), RAW_VALUE, [0, 0, 0], [0.0, 0.0, 0.0])

    np.testing.assert_allclose(CompletedQ()(tree, -1.0), PRIOR, rtol=1e-6)


def test_the_completed_q_target_gives_illegal_actions_nothing():
    logits = [0.0, MIN, 1.0, MIN]
    tree = root_tree(logits, 0.0, [2, 0, 1, 0], [-0.5, 0.0, 0.5, 0.0])

    target = CompletedQ()(tree, -1.0)

    np.testing.assert_allclose(target.sum(), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(target[np.array([1, 3])], 0.0)
    # the better child gains on its prior, the worse one loses
    prior = jax.nn.softmax(jnp.array([0.0, 1.0]))
    assert target[2] > prior[1] and target[0] < prior[0]


def test_the_visit_counts_target():
    tree = root_tree(np.log(PRIOR), RAW_VALUE, VISITS, Q)

    np.testing.assert_allclose(VisitCounts()(tree, -1.0), VISITS / 4)


def pruned_by_hand(prior, visits, q, c_init=1.25, c_base=19652):
    """KataGo's pruning one visit at a time, from each child but the most visited: take a visit
    away while that leaves its PUCT below the most visited child's; children left with one visit
    are dropped. The root has 1 + sum(visits) visits, and value 0."""
    visits = np.asarray(visits, dtype=float)
    total = 1 + visits.sum()
    c = c_init + np.log((total + c_base + 1) / c_base)

    def puct(action, n):
        value = (q[action] + 1) / 2 if visits[action] > 0 else 0.0
        return value + c * prior[action] * np.sqrt(total) / (n + 1)

    best = int(np.argmax(visits))
    kept = visits.copy()
    for action in range(len(visits)):
        if action == best:
            continue
        while kept[action] > 0 and puct(action, kept[action] - 1) < puct(
            best, visits[best]
        ):
            kept[action] -= 1
        if kept[action] <= 1 and kept[action] < visits[action]:
            kept[action] = 0
    return kept / kept.sum()


def test_pruning_takes_visits_the_clean_prior_wouldnt_have_given():
    # action 2 has a small prior and a poor value but many visits, as root noise would give it
    prior = np.array([0.6, 0.3, 0.1])
    visits = [20, 6, 12]
    q = np.array([0.2, 0.1, -0.5])
    tree = root_tree(np.log(prior), 0.0, visits, q)

    target = PrunedVisitCounts(PUCTSelector())(tree, -1.0)

    expected = pruned_by_hand(prior, visits, q)
    np.testing.assert_allclose(target, expected, rtol=1e-6)
    # the most visited child gains, the forced one loses most of its visits
    assert target[0] > 20 / 38 and target[2] < 0.5 * 12 / 38


@pytest.mark.parametrize("seed", range(20))
def test_pruning_matches_taking_visits_one_at_a_time(seed):
    rng = np.random.default_rng(seed)
    num_actions = 6
    prior = rng.dirichlet(np.ones(num_actions))
    visits = rng.integers(0, 30, num_actions)
    visits[0] = max(visits[0], 1)
    q = rng.uniform(-1, 1, num_actions)
    tree = root_tree(np.log(prior), 0.0, visits, q)

    target = PrunedVisitCounts(PUCTSelector())(tree, -1.0)

    np.testing.assert_allclose(target, pruned_by_hand(prior, visits, q), rtol=1e-5)


def test_pruning_leaves_a_child_better_than_the_most_visited_alone():
    # action 1's value beats the most visited child's PUCT outright: nothing is taken from it
    prior = np.array([0.5, 0.25, 0.25])
    tree = root_tree(np.log(prior), 0.0, [10, 9, 0], [0.0, 0.9, 0.0])

    target = PrunedVisitCounts(PUCTSelector())(tree, -1.0)

    np.testing.assert_allclose(target, [10 / 19, 9 / 19, 0.0], rtol=1e-6)


def test_pruning_uses_the_prior_at_the_searchs_temperature():
    # flattened at temperature 4, action 2's prior is high enough to justify its visits
    logits = np.log(np.array([0.85, 0.1, 0.05]))
    visits, q = [20, 3, 6], np.array([0.1, 0.0, -0.2])
    tree = root_tree(logits, 0.0, visits, q)

    sharp = PrunedVisitCounts(PUCTSelector())(tree, -1.0)
    flat = PrunedVisitCounts(PUCTSelector(), prior_temperature=4.0)(tree, -1.0)

    flat_prior = np.exp(logits / 4) / np.exp(logits / 4).sum()
    np.testing.assert_allclose(flat, pruned_by_hand(flat_prior, visits, q), rtol=1e-5)
    assert flat[2] > sharp[2]


def test_pruning_gives_illegal_actions_nothing():
    logits = [0.0, MIN, 1.0, MIN]
    tree = root_tree(logits, 0.0, [2, 0, 5, 0], [-0.5, 0.0, 0.5, 0.0])

    target = PrunedVisitCounts(PUCTSelector())(tree, -1.0)

    np.testing.assert_allclose(target.sum(), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(target[np.array([1, 3])], 0.0)
