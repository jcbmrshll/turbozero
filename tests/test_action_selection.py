from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.evaluators.alphazero import AlphaZero
from core.evaluators.mcts.action_selection import (
    MuZeroPUCTSelector,
    PUCTSelector,
    normalize_q_values,
)
from core.evaluators.mcts.mcts import MCTS
from core.trees.tree import init_tree

EPS = 1e-8


def mctx_qtransform(q, n, parent_q, eps):
    """mctx's `qtransform_by_parent_and_siblings`: unvisited children take the parent's value
    when computing the range, then map to the bottom of it."""
    safe_q = jnp.where(n > 0, q, parent_q)
    lo = jnp.minimum(parent_q, safe_q.min())
    hi = jnp.maximum(parent_q, safe_q.max())
    return (jnp.where(n > 0, q, lo) - lo) / jnp.maximum(hi - lo, eps)


def random_q_cases(num_cases, unvisited):
    rng = np.random.default_rng(0)
    for _ in range(num_cases):
        q = rng.uniform(-1, 1, size=6).astype(np.float32)
        n = rng.integers(1, 5, size=6)
        if unvisited:
            n[rng.choice(6, size=3, replace=False)] = 0
            # unvisited children read as the tree's null value
            q = np.where(n > 0, q, 0.0).astype(np.float32)
        yield jnp.array(q), jnp.array(n), jnp.float32(rng.uniform(-1, 1))


def test_normalize_q_values_matches_mctx_when_all_children_visited():
    for q, n, parent_q in random_q_cases(20, unvisited=False):
        np.testing.assert_allclose(
            normalize_q_values(q, n, parent_q, EPS),
            mctx_qtransform(q, n, parent_q, EPS),
            atol=1e-6,
        )


def test_normalize_q_values_ignores_unvisited_children():
    q, n, parent_q = jnp.array([0.6, 0.9, 0.0, 0.0]), jnp.array([3, 3, 0, 0]), 0.75
    np.testing.assert_allclose(
        normalize_q_values(q, n, parent_q, EPS), [0.0, 1.0, 0.0, 0.0], atol=1e-6
    )

    for q, n, parent_q in random_q_cases(20, unvisited=True):
        np.testing.assert_allclose(
            normalize_q_values(q, n, parent_q, EPS),
            mctx_qtransform(q, n, parent_q, EPS),
            atol=1e-6,
        )


def root_only_tree(prior):
    node = MCTS.new_node(
        policy=jnp.asarray(prior, dtype=jnp.float32),
        value=0.0,
        embedding=jnp.zeros(()),
        terminated=False,
    )
    return init_tree(8, len(prior), node).set_root(node)


def test_puct_with_no_visits_picks_highest_prior():
    for prior in ([0.1, 0.2, 0.6, 0.1], [0.7, 0.1, 0.1, 0.1], [0.05, 0.05, 0.1, 0.8]):
        tree = root_only_tree(prior)
        assert PUCTSelector()(tree, tree.ROOT_INDEX, -1.0) == np.argmax(prior)


MUZERO = MuZeroPUCTSelector()


def test_muzero_selector_runs_in_search(make_search, ttt):
    search = make_search(AlphaZero(MCTS), action_selector=MUZERO)
    state, meta = ttt.play([4, 0])

    out = search.evaluate(jax.random.PRNGKey(0), search.init(), state, meta)

    np.testing.assert_allclose(out.policy_weights.sum(), 1.0, rtol=1e-6)
    assert meta.action_mask[out.action]


def visited_tree(prior, root_q, child_n, child_q):
    """Root with every child visited: `child_q` is from each child's perspective, the root's `n` is 1 + the children's."""
    prior, child_n, child_q = (np.asarray(x) for x in (prior, child_n, child_q))
    root = replace(
        MCTS.new_node(
            policy=jnp.asarray(prior, dtype=jnp.float32),
            value=0.0,
            embedding=jnp.zeros(()),
            terminated=False,
        ),
        n=jnp.array(1 + child_n.sum(), dtype=jnp.int32),
        q=jnp.float32(root_q),
    )
    tree = init_tree(8, len(prior), root).set_root(root)
    for action, (n, q) in enumerate(zip(child_n, child_q)):
        child = replace(
            MCTS.new_node(
                policy=jnp.zeros(len(prior)),
                value=0.0,
                embedding=jnp.zeros(()),
                terminated=False,
            ),
            n=jnp.array(n, dtype=jnp.int32),
            q=jnp.float32(q),
        )
        tree = tree.add_node(tree.ROOT_INDEX, action, child)
    return tree


@pytest.mark.parametrize(
    "c1, c2, expected", [(1.25, 19652, 0), (1.25, 1.0, 3), (3.0, 19652, 3)]
)
def test_muzero_selector_matches_paper_ucb_score(c1, c2, expected):
    """MuZero pseudocode `ucb_score`, with values min-max normalized over the parent and its children."""
    prior = np.array([0.1, 0.5, 0.15, 0.25])
    child_n = np.array([4, 10, 1, 3])
    child_q = np.array([-0.5, 0.2, 0.3, -0.1])
    root_q, discount = 0.1, -1.0
    tree = visited_tree(prior, root_q, child_n, child_q)

    parent_n = 1 + child_n.sum()
    value = discount * child_q
    lo, hi = min(root_q, value.min()), max(root_q, value.max())
    value_score = (value - lo) / (hi - lo)
    pb_c = (np.log((parent_n + c2 + 1) / c2) + c1) * np.sqrt(parent_n) / (child_n + 1)
    ucb = value_score + pb_c * prior
    assert np.argmax(ucb) == expected

    assert MuZeroPUCTSelector(c1=c1, c2=c2)(tree, tree.ROOT_INDEX, discount) == expected
