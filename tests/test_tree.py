import random
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.trees.tree import Tree, init_tree


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NodeData:
    value: jax.Array
    vec: jax.Array


def node(value):
    return NodeData(
        value=jnp.array(value, dtype=jnp.float32),
        vec=jnp.array([value, -value], dtype=jnp.float32),
    )


def empty_tree(max_nodes=4, branching_factor=3):
    return init_tree(max_nodes, branching_factor, node(0.0))


def assert_trees_equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_add_node_sets_parent_edge_and_data():
    tree = empty_tree().set_root(node(1.0))
    tree = tree.add_node(parent_index=0, edge_index=2, data=node(5.0))

    assert tree.next_free_idx == 2
    assert tree.parents[1] == 0
    assert tree.edge_map[0, 2] == 1
    assert tree.is_edge(0, 2)
    assert not tree.is_edge(0, 0)
    assert_trees_equal(tree.data_at(1), node(5.0))


def test_add_node_is_a_noop_when_full():
    tree = empty_tree(max_nodes=3).set_root(node(1.0))
    tree = tree.add_node(0, 0, node(2.0))
    full = tree.add_node(0, 1, node(3.0))
    assert full.next_free_idx == full.capacity

    after = full.add_node(0, 2, node(4.0))

    assert after.edge_map[0, 2] == Tree.NULL_INDEX
    assert_trees_equal(after, full)


def test_get_child_data_fills_missing_children_with_null_value():
    tree = empty_tree().set_root(node(1.0))
    tree = tree.add_node(0, 0, node(2.0))
    tree = tree.add_node(0, 2, node(3.0))

    np.testing.assert_array_equal(
        tree.get_child_data("value", 0), [2.0, Tree.NULL_VALUE, 3.0]
    )
    np.testing.assert_array_equal(
        tree.get_child_data("value", 0, null_value=-7.0), [2.0, -7.0, 3.0]
    )
    np.testing.assert_array_equal(
        tree.get_child_data("vec", 0), [[2.0, -2.0], [Tree.NULL_VALUE] * 2, [3.0, -3.0]]
    )
    # a leaf has no children at all
    np.testing.assert_array_equal(
        tree.get_child_data("value", 1), [Tree.NULL_VALUE] * 3
    )


def test_reset_empties_the_tree():
    tree = empty_tree().set_root(node(1.0)).add_node(0, 1, node(2.0))

    assert_trees_equal(tree.reset(), empty_tree())


# get_subtree vs. a plain-Python reference on random trees

MAX_NODES, BRANCHING = 12, 3


def random_tree(rng: random.Random):
    """Grows a random tree the way MCTS does: each new node goes in the next free slot,
    under a random existing node, along one of that node's free edges."""
    num_nodes = rng.randint(1, MAX_NODES)
    parents = [Tree.NULL_INDEX] * MAX_NODES
    edge_map = [[Tree.NULL_INDEX] * BRANCHING for _ in range(MAX_NODES)]
    for idx in range(1, num_nodes):
        free = [
            (p, e)
            for p in range(idx)
            for e in range(BRANCHING)
            if edge_map[p][e] == Tree.NULL_INDEX
        ]
        p, e = rng.choice(free)
        parents[idx] = p
        edge_map[p][e] = idx
    values = [rng.uniform(1, 2) for _ in range(num_nodes)] + [0.0] * (
        MAX_NODES - num_nodes
    )
    return num_nodes, parents, edge_map, values


def reference_subtree(parents, edge_map, values, action):
    """Plain-Python get_subtree: keeps the subtree under the root's child along `action`,
    renumbering nodes by their old index so the new root lands at 0."""
    child = edge_map[0][action]
    kept = []
    if child != Tree.NULL_INDEX:
        stack = [child]
        while stack:
            idx = stack.pop()
            kept.append(idx)
            stack.extend(c for c in edge_map[idx] if c != Tree.NULL_INDEX)
    kept.sort()
    new_idx = {old: new for new, old in enumerate(kept)}

    new_parents = [Tree.NULL_INDEX] * MAX_NODES
    new_edge_map = [[Tree.NULL_INDEX] * BRANCHING for _ in range(MAX_NODES)]
    new_values = [0.0] * MAX_NODES
    for old, new in new_idx.items():
        new_parents[new] = new_idx.get(parents[old], Tree.NULL_INDEX)
        new_edge_map[new] = [
            new_idx[c] if c != Tree.NULL_INDEX else Tree.NULL_INDEX
            for c in edge_map[old]
        ]
        new_values[new] = values[old]
    return len(kept), new_parents, new_edge_map, new_values


def build_tree(num_nodes, parents, edge_map, values):
    values = jnp.array(values, dtype=jnp.float32)
    return Tree(
        next_free_idx=jnp.array(num_nodes, dtype=jnp.int32),
        parents=jnp.array(parents, dtype=jnp.int32),
        edge_map=jnp.array(edge_map, dtype=jnp.int32),
        data=NodeData(value=values, vec=jnp.stack([values, -values], axis=-1)),
    )


@pytest.fixture(scope="module")
def get_subtree():
    return jax.jit(lambda tree, action: tree.get_subtree(action))


@pytest.mark.parametrize("seed", range(40))
def test_get_subtree_matches_reference(get_subtree, seed):
    rng = random.Random(seed)
    num_nodes, parents, edge_map, values = random_tree(rng)
    tree = build_tree(num_nodes, parents, edge_map, values)

    for action in range(BRANCHING):
        expected = build_tree(*reference_subtree(parents, edge_map, values, action))
        assert_trees_equal(get_subtree(tree, action), expected)


def test_get_subtree_of_unexpanded_child_is_empty(get_subtree):
    tree = empty_tree(MAX_NODES, BRANCHING).set_root(node(1.0))
    tree = tree.add_node(0, 0, node(2.0)).add_node(1, 1, node(3.0))

    assert_trees_equal(get_subtree(tree, 2), empty_tree(MAX_NODES, BRANCHING))
