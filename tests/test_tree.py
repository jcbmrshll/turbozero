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


def random_tree(
    rng: random.Random, max_nodes=MAX_NODES, branching=BRANCHING, num_nodes=None
):
    """Grows a random tree the way MCTS does: each new node goes in the next free slot,
    under a random existing node, along one of that node's free edges."""
    if num_nodes is None:
        num_nodes = rng.randint(1, max_nodes)
    parents = [Tree.NULL_INDEX] * max_nodes
    edge_map = [[Tree.NULL_INDEX] * branching for _ in range(max_nodes)]
    values = [rng.uniform(1, 2)] + [0.0] * (max_nodes - 1)
    for idx in range(1, num_nodes):
        grow(rng, parents, edge_map, values, idx)
    return num_nodes, parents, edge_map, values


def grow(rng: random.Random, parents, edge_map, values, idx):
    """Adds node `idx` under a random node below it, along a random free edge. Returns (parent, edge, value)."""
    free = [
        (p, e)
        for p in range(idx)
        for e in range(len(edge_map[0]))
        if edge_map[p][e] == Tree.NULL_INDEX
    ]
    p, e = rng.choice(free)
    parents[idx] = p
    edge_map[p][e] = idx
    values[idx] = rng.uniform(1, 2)
    return p, e, values[idx]


def reference_subtree(parents, edge_map, values, action):
    """Plain-Python get_subtree: keeps the subtree under the root's child along `action`,
    renumbering nodes by their old index so the new root lands at 0."""
    max_nodes, branching = len(parents), len(edge_map[0])
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

    new_parents = [Tree.NULL_INDEX] * max_nodes
    new_edge_map = [[Tree.NULL_INDEX] * branching for _ in range(max_nodes)]
    new_values = [0.0] * max_nodes
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


# get_subtree stress cases


def assert_subtree_matches_reference(get_subtree, num_nodes, parents, edge_map, values):
    tree = build_tree(num_nodes, parents, edge_map, values)
    for action in range(len(edge_map[0])):
        expected = build_tree(*reference_subtree(parents, edge_map, values, action))
        assert_trees_equal(get_subtree(tree, action), expected)


@pytest.mark.parametrize("max_nodes", [2, 3, 64, 257])
def test_get_subtree_of_deep_chain(get_subtree, max_nodes):
    # one long path from the root: depth == capacity - 1
    parents = [Tree.NULL_INDEX] + list(range(max_nodes - 1))
    edge_map = [[Tree.NULL_INDEX] * BRANCHING for _ in range(max_nodes)]
    for idx in range(1, max_nodes):
        edge_map[idx - 1][idx % BRANCHING] = idx
    values = [float(idx + 1) for idx in range(max_nodes)]

    tree = build_tree(max_nodes, parents, edge_map, values)
    subtree = get_subtree(tree, 1 % BRANCHING)
    assert subtree.next_free_idx == max_nodes - 1
    np.testing.assert_array_equal(subtree.data.value[: max_nodes - 1], values[1:])
    assert_subtree_matches_reference(get_subtree, max_nodes, parents, edge_map, values)


@pytest.mark.parametrize("depth", [1, 2])
def test_get_subtree_of_wide_shallow_tree(get_subtree, depth):
    # every node has a child along every edge, down to `depth`
    branching = 8
    max_nodes = sum(branching**d for d in range(depth + 1))
    parents = [Tree.NULL_INDEX] * max_nodes
    edge_map = [[Tree.NULL_INDEX] * branching for _ in range(max_nodes)]
    for idx in range(1, max_nodes):
        parents[idx] = (idx - 1) // branching
        edge_map[parents[idx]][(idx - 1) % branching] = idx
    values = [float(idx + 1) for idx in range(max_nodes)]

    assert_subtree_matches_reference(get_subtree, max_nodes, parents, edge_map, values)


@pytest.mark.parametrize("seed", range(10))
def test_get_subtree_of_full_tree(get_subtree, seed):
    rng = random.Random(seed)
    max_nodes = rng.choice([16, 33, 128])
    tree = random_tree(rng, max_nodes=max_nodes, branching=4, num_nodes=max_nodes)

    assert_subtree_matches_reference(get_subtree, *tree)


def test_get_subtree_of_unexpanded_child_of_full_tree_is_empty(get_subtree):
    max_nodes = 16
    # root has children on edges 0 and 1 only; edge 2 is never expanded
    tree = empty_tree(max_nodes, BRANCHING).set_root(node(1.0))
    tree = tree.add_node(0, 0, node(2.0)).add_node(0, 1, node(3.0))
    for idx in range(3, max_nodes):
        tree = tree.add_node(idx - 1, 0, node(float(idx + 1)))
    assert tree.next_free_idx == tree.capacity

    assert_trees_equal(get_subtree(tree, 2), empty_tree(max_nodes, BRANCHING))


@pytest.mark.parametrize("seed", range(5))
def test_get_subtree_reused_over_several_steps(get_subtree, seed):
    # like self-play with persist_tree: search grows the tree, an action is taken, the subtree is kept
    rng = random.Random(seed)
    max_nodes, branching = 48, 3
    num_nodes, parents, edge_map, values = random_tree(
        rng, max_nodes, branching, num_nodes=max_nodes // 2
    )
    tree = build_tree(num_nodes, parents, edge_map, values)

    for _ in range(6):
        # a random action: usually an expanded child, sometimes an unexpanded one (empty tree)
        action = rng.randrange(branching)
        num_nodes, parents, edge_map, values = reference_subtree(
            parents, edge_map, values, action
        )
        tree = get_subtree(tree, action)
        assert_trees_equal(tree, build_tree(num_nodes, parents, edge_map, values))

        if num_nodes == 0:
            # an empty tree gets a new root, as MCTS does on the next search
            values[0] = rng.uniform(1, 2)
            num_nodes = 1
            tree = tree.set_root(node(values[0]))
        for idx in range(
            num_nodes, min(max_nodes, num_nodes + rng.randint(0, max_nodes // 2))
        ):
            p, e, v = grow(rng, parents, edge_map, values, idx)
            tree = tree.add_node(p, e, node(v))
            num_nodes = idx + 1
        assert_trees_equal(tree, build_tree(num_nodes, parents, edge_map, values))
