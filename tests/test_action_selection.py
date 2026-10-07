import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.evaluators.alphazero import AlphaZero
from core.evaluators.mcts.action_selection import MuZeroPUCTSelector, PUCTSelector, normalize_q_values
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
        np.testing.assert_allclose(normalize_q_values(q, n, parent_q, EPS), mctx_qtransform(q, n, parent_q, EPS),
                                   atol=1e-6)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#3: normalize_q_values takes min/max over unvisited children, which read as 0")
def test_normalize_q_values_ignores_unvisited_children():
    q, n, parent_q = jnp.array([0.6, 0.9, 0.0, 0.0]), jnp.array([3, 3, 0, 0]), 0.75
    np.testing.assert_allclose(normalize_q_values(q, n, parent_q, EPS), [0.0, 1.0, 0.0, 0.0], atol=1e-6)

    for q, n, parent_q in random_q_cases(20, unvisited=True):
        np.testing.assert_allclose(normalize_q_values(q, n, parent_q, EPS), mctx_qtransform(q, n, parent_q, EPS),
                                   atol=1e-6)


def root_only_tree(prior):
    node = MCTS.new_node(policy=jnp.asarray(prior, dtype=jnp.float32), value=0.0,
                         embedding=jnp.zeros(()), terminated=False)
    return init_tree(8, len(prior), node).set_root(node)


def test_puct_with_no_visits_picks_highest_prior():
    for prior in ([0.1, 0.2, 0.6, 0.1], [0.7, 0.1, 0.1, 0.1], [0.05, 0.05, 0.1, 0.8]):
        tree = root_only_tree(prior)
        assert PUCTSelector()(tree, tree.ROOT_INDEX, -1.0) == np.argmax(prior)


MUZERO = MuZeroPUCTSelector()


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="#7: MuZeroPUCTSelector calls q_transform with 5 arguments; normalize_q_values takes 4")
def test_muzero_selector_runs_in_search(make_search, ttt):
    search = make_search(AlphaZero(MCTS), action_selector=MUZERO)
    state, meta = ttt.play([4, 0])

    out = search.evaluate(jax.random.PRNGKey(0), search.init(), state, meta)

    np.testing.assert_allclose(out.policy_weights.sum(), 1.0, rtol=1e-6)
    assert meta.action_mask[out.action]
