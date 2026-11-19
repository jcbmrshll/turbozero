from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from turbozero.evaluators.mcts.action_selection import PUCTSelector
from turbozero.evaluators.mcts.state import MCTSNode, init_mcts_tree

KEY = jax.random.PRNGKey(0)


def new_node(policy, value, embedding, terminated):
    """An evaluated node, visited once (dev's `MCTS.new_node`)."""
    return MCTSNode(
        n=jnp.array(1, dtype=jnp.int32),
        p=policy,
        q=jnp.array(value, dtype=jnp.float32),
        terminated=jnp.array(terminated, dtype=jnp.bool_),
        embedding=embedding,
    )


def root_only_tree(prior):
    node = new_node(
        policy=jnp.asarray(prior, dtype=jnp.float32),
        value=0.0,
        embedding=jnp.zeros(()),
        terminated=False,
    )
    return init_mcts_tree(8, len(prior), node).set_root(node)


def test_puct_with_no_visits_picks_highest_prior():
    for prior in ([0.1, 0.2, 0.6, 0.1], [0.7, 0.1, 0.1, 0.1], [0.05, 0.05, 0.1, 0.8]):
        tree = root_only_tree(prior)
        assert PUCTSelector()(KEY, tree, tree.ROOT_INDEX, -1.0) == np.argmax(prior)


def visited_tree(prior, root_q, child_n, child_q):
    """A root whose children have `child_n` visits (unvisited ones aren't in the tree) and values
    `child_q`, from each child's perspective; the root's `n` is 1 + the children's."""
    prior, child_n, child_q = (np.asarray(x) for x in (prior, child_n, child_q))
    root = replace(
        new_node(
            policy=jnp.asarray(prior, dtype=jnp.float32),
            value=0.0,
            embedding=jnp.zeros(()),
            terminated=False,
        ),
        n=jnp.array(1 + child_n.sum(), dtype=jnp.int32),
        q=jnp.float32(root_q),
    )
    tree = init_mcts_tree(8, len(prior), root).set_root(root)
    for action, (n, q) in enumerate(zip(child_n, child_q)):
        if n == 0:
            continue
        child = replace(
            new_node(
                policy=jnp.zeros(len(prior)),
                value=0.0,
                embedding=jnp.zeros(()),
                terminated=False,
            ),
            n=jnp.array(n, dtype=jnp.int32),
            q=jnp.float32(q),
        )
        tree = tree.set_child(tree.ROOT_INDEX, action, child)
    return tree


def test_puct_never_chooses_a_move_without_prior():
    selector = PUCTSelector()
    # illegal moves have no prior; however bad the legal moves' values are for the player to move
    # (they're the children's, from their side), their exploration bonus keeps them ahead
    tree = visited_tree(
        prior=[0.5, 0.0, 0.5, 0.0],
        root_q=0.9,
        child_n=[20, 0, 20, 0],
        child_q=[1, 0, 1, 0],
    )

    assert int(selector(KEY, tree, tree.ROOT_INDEX, -1.0)) in (0, 2)


def test_puct_breaks_exact_ties_at_random():
    selector = PUCTSelector()
    # every child has the same prior, visits and value: only the key decides
    tree = visited_tree(
        prior=[0.25] * 4, root_q=0.0, child_n=[3] * 4, child_q=[0.2] * 4
    )
    select = jax.jit(lambda key: selector(key, tree, tree.ROOT_INDEX, -1.0))

    assert {int(select(jax.random.PRNGKey(seed))) for seed in range(32)} == {0, 1, 2, 3}
    assert int(select(KEY)) == int(select(KEY))


def test_tie_breaking_never_overturns_a_strict_preference():
    selector = PUCTSelector()
    # child 2 is ahead by 1e-5 in value: far more than the noise
    tree = visited_tree(
        prior=[0.25] * 4, root_q=0.0, child_n=[3] * 4, child_q=[0.2, 0.2, 0.19999, 0.2]
    )
    select = jax.jit(lambda key: selector(key, tree, tree.ROOT_INDEX, -1.0))

    assert {int(select(jax.random.PRNGKey(seed))) for seed in range(32)} == {2}


ALL_VISITED = [4, 10, 1, 3]
ONE_UNVISITED = [4, 10, 0, 3]


@pytest.mark.parametrize(
    "child_n, discount, c1, c2, expected",
    [
        (ALL_VISITED, -1.0, 1.25, 19652, 3),
        (ALL_VISITED, -1.0, 1.25, 1.0, 2),
        (ALL_VISITED, -1.0, 0.1, 19652, 0),
        # a positive discount (single-player): the children's values are the root's
        (ALL_VISITED, 1.0, 1.25, 19652, 2),
        # an unvisited child scores 0 for its value, a loss; at 0.5 it would win
        # each of these
        (ONE_UNVISITED, -1.0, 1.25, 19652, 3),
        (ONE_UNVISITED, -1.0, 0.5, 19652, 0),
        (ONE_UNVISITED, 1.0, 1.25, 19652, 1),
    ],
)
def test_puct_matches_alphazero_and_muzero_ucb_score(
    child_n, discount, c1, c2, expected
):
    """MuZero pseudocode `ucb_score` for board games, whose values are normalized by
    `KnownBounds(-1, 1)`; an unvisited child's value score is 0."""
    prior = np.array([0.1, 0.5, 0.15, 0.25])
    child_n = np.array(child_n)
    child_q = np.array([-0.5, 0.2, 0.3, -0.1])
    root_q = 0.1
    tree = visited_tree(prior, root_q, child_n, child_q)

    parent_n = 1 + child_n.sum()
    value_score = np.where(child_n > 0, (discount * child_q + 1) / 2, 0.0)
    pb_c = (np.log((parent_n + c2 + 1) / c2) + c1) * np.sqrt(parent_n) / (child_n + 1)
    ucb = value_score + pb_c * prior
    assert np.argmax(ucb) == expected

    selector = PUCTSelector(c_init=c1, c_base=c2)
    assert selector(KEY, tree, tree.ROOT_INDEX, discount) == expected
