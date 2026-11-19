"""MCTS tree state (turbozero.evaluators.mcts.state): what MCTSTree adds to
Tree (see test_tree.py for the tree itself)."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from turbozero.evaluators.mcts.state import MCTSNode, MCTSTree, init_mcts_tree


def node(n):
    return MCTSNode(
        n=jnp.array(n, dtype=jnp.int32),
        p=jnp.full((3,), 1 / 3),
        q=jnp.array(0.0, dtype=jnp.float32),
        terminated=jnp.array(False),
        embedding=jnp.zeros((2, 2)),
    )


def test_init_mcts_tree_starts_without_root_noise():
    tree = init_mcts_tree(5, 3, node(0))

    assert isinstance(tree, MCTSTree)
    np.testing.assert_array_equal(tree.root_noise, jnp.zeros(3, dtype=jnp.float32))


def test_tree_methods_keep_the_mcts_tree_and_its_root_noise():
    noise = jnp.array([0.2, 0.3, 0.5])
    tree = init_mcts_tree(5, 3, node(0))
    tree = replace(tree, root_noise=noise)

    tree = tree.set_root(node(2)).set_child(0, 1, node(1))
    tree = jax.jit(lambda t: t.set_child(0, 2, node(1)))(tree)

    assert isinstance(tree, MCTSTree)
    assert int(tree.next_free_idx) == 3
    np.testing.assert_array_equal(tree.root_noise, noise)
    for subtree in (tree.get_subtree(1), tree.reset()):
        assert isinstance(subtree, MCTSTree)
        np.testing.assert_array_equal(subtree.root_noise, noise)
