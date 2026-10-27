"""Policy targets (core.evaluators.mcts.policy_targets): the visit distribution, and Gumbel MuZero's
completed-Q improved policy, on hand-built roots."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.evaluators.mcts.policy_targets import (
    CompletedQ,
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
