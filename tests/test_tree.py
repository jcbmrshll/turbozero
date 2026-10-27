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


def test_set_child_adds_a_missing_child_like_add_node():
    tree = empty_tree().set_root(node(1.0))

    assert_trees_equal(tree.set_child(0, 2, node(5.0)), tree.add_node(0, 2, node(5.0)))


def test_set_child_updates_an_existing_child_in_place():
    tree = empty_tree().set_root(node(1.0)).add_node(0, 2, node(5.0))

    assert_trees_equal(tree.set_child(0, 2, node(7.0)), tree.update_node(1, node(7.0)))


def test_set_child_is_a_noop_for_a_missing_child_of_a_full_tree():
    tree = empty_tree(max_nodes=2).set_root(node(1.0)).add_node(0, 0, node(2.0))
    assert tree.next_free_idx == tree.capacity

    assert_trees_equal(tree.set_child(0, 1, node(9.0)), tree)
    # but an existing child of a full tree can still be updated
    assert_trees_equal(tree.set_child(0, 0, node(9.0)), tree.update_node(1, node(9.0)))


def test_set_child_under_vmap_matches_each_tree_alone():
    # one tree has the child already, one doesn't, one is full
    trees = [
        empty_tree(max_nodes=2).set_root(node(1.0)).add_node(0, 1, node(2.0)),
        empty_tree(max_nodes=2).set_root(node(1.0)),
        empty_tree(max_nodes=2).set_root(node(1.0)).add_node(0, 0, node(2.0)),
    ]
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *trees)

    out = jax.vmap(lambda t: t.set_child(0, 1, node(9.0)))(batch)

    for i, tree in enumerate(trees):
        assert_trees_equal(
            jax.tree.map(lambda x, i=i: x[i], out), tree.set_child(0, 1, node(9.0))
        )


@pytest.mark.parametrize(
    "tree",
    [
        empty_tree(),
        empty_tree().set_root(node(1.0)),
        empty_tree().set_root(node(1.0)).add_node(0, 2, node(5.0)),
    ],
    ids=["empty", "missing child", "existing child"],
)
def test_set_child_and_set_root_without_write_are_noops(tree):
    assert_trees_equal(tree.set_child(0, 2, node(9.0), write=False), tree)
    assert_trees_equal(tree.set_root(node(9.0), write=False), tree)


def test_set_child_without_write_does_not_evict():
    tree = full_tree()

    assert_trees_equal(tree.set_child(0, 2, node(9.0), evict=2, write=False), tree)


def full_tree():
    """A full tree: the root with children on edges 0 and 1, and a grandchild under edge 0's."""
    tree = empty_tree(max_nodes=4).set_root(node(1.0))
    tree = tree.add_node(0, 0, node(2.0)).add_node(1, 0, node(3.0))
    return tree.add_node(0, 1, node(4.0))


def test_set_child_evicts_a_leaf_for_a_missing_child_of_a_full_tree():
    tree = full_tree()

    after = tree.set_child(0, 2, node(9.0), evict=2)

    # node 2 (the grandchild) is cut from its parent, and its slot holds the new child
    assert after.next_free_idx == tree.capacity
    assert after.edge_map[1, 0] == Tree.NULL_INDEX
    assert after.edge_map[0, 2] == 2
    assert after.parents[2] == 0
    assert_trees_equal(after.data_at(2), node(9.0))
    # nothing else changes
    np.testing.assert_array_equal(after.edge_map[0, :2], tree.edge_map[0, :2])
    assert_trees_equal(
        after.data_at(np.array([0, 1, 3])), tree.data_at(np.array([0, 1, 3]))
    )


def test_set_child_ignores_evict_unless_the_tree_is_full():
    # room left: the child goes in the next free slot, as without `evict`
    tree = empty_tree().set_root(node(1.0)).add_node(0, 0, node(2.0))
    assert_trees_equal(
        tree.set_child(0, 1, node(9.0), evict=1), tree.set_child(0, 1, node(9.0))
    )
    # an existing child of a full tree is updated in place
    tree = full_tree()
    assert_trees_equal(
        tree.set_child(0, 1, node(9.0), evict=2), tree.update_node(3, node(9.0))
    )
    # nothing to evict: a no-op, as without `evict`
    assert_trees_equal(tree.set_child(0, 2, node(9.0), evict=Tree.NULL_INDEX), tree)


def test_leaves():
    tree = full_tree()
    np.testing.assert_array_equal(tree.leaves(), [False, False, True, True])
    # unused slots aren't leaves; a lone root is
    tree = empty_tree().set_root(node(1.0))
    np.testing.assert_array_equal(tree.leaves(), [True, False, False, False])


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
    renumbering the new root to 0 and the other nodes after it by their old index."""
    max_nodes, branching = len(parents), len(edge_map[0])
    child = edge_map[0][action]
    kept = []
    if child != Tree.NULL_INDEX:
        stack = [c for c in edge_map[child] if c != Tree.NULL_INDEX]
        while stack:
            idx = stack.pop()
            kept.append(idx)
            stack.extend(c for c in edge_map[idx] if c != Tree.NULL_INDEX)
        kept = [child] + sorted(kept)
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


def evict_and_regrow(rng: random.Random, parents, edge_map, values):
    """Moves a random leaf (not the root) to a random free edge of another node, as `set_child`
    does when it evicts the leaf for a new child of a full tree. Returns (parent, edge, value, leaf)."""
    has_child = {p for p in parents if p != Tree.NULL_INDEX}
    leaf = rng.choice([i for i in range(1, len(parents)) if i not in has_child])
    old_parent = parents[leaf]
    edge_map[old_parent][edge_map[old_parent].index(leaf)] = Tree.NULL_INDEX
    p, e = rng.choice(
        [
            (p, e)
            for p in range(len(parents))
            for e in range(len(edge_map[0]))
            if p != leaf and edge_map[p][e] == Tree.NULL_INDEX
        ]
    )
    parents[leaf] = p
    edge_map[p][e] = leaf
    values[leaf] = rng.uniform(1, 2)
    return p, e, values[leaf], leaf


# compiled once for each shape of tree
set_child = jax.jit(lambda tree, p, e, v, leaf: tree.set_child(p, e, node(v), leaf))


@pytest.mark.parametrize("seed", range(20))
def test_get_subtree_after_evictions_matches_reference(get_subtree, seed):
    # evicted slots hold nodes added later, so a node's parent or the new root can come after it
    rng = random.Random(seed)
    max_nodes = rng.choice([4, 12, 33])
    num_nodes, parents, edge_map, values = random_tree(
        rng, max_nodes, BRANCHING, num_nodes=max_nodes
    )
    tree = build_tree(num_nodes, parents, edge_map, values)
    for _ in range(rng.randint(1, 3 * max_nodes)):
        p, e, v, leaf = evict_and_regrow(rng, parents, edge_map, values)
        tree = set_child(tree, p, e, v, leaf)
    assert_trees_equal(tree, build_tree(num_nodes, parents, edge_map, values))

    assert_subtree_matches_reference(get_subtree, num_nodes, parents, edge_map, values)


# path_to_root vs. following parents one at a time


def reference_path(parents, index):
    path = []
    while index != Tree.NULL_INDEX:
        path.append(index)
        index = parents[index]
    return path + [Tree.NULL_INDEX] * (len(parents) - len(path))


path_to_root = jax.jit(lambda tree, index: tree.path_to_root(index))


@pytest.mark.parametrize("seed", range(20))
def test_path_to_root_matches_reference(seed):
    rng = random.Random(seed)
    tree = random_tree(rng, max_nodes=rng.choice([1, 2, 12, 33]))
    num_nodes, parents = tree[0], tree[1]
    built = build_tree(*tree)
    for index in range(num_nodes):
        np.testing.assert_array_equal(
            path_to_root(built, index), reference_path(parents, index)
        )


@pytest.mark.parametrize("max_nodes", [2, 3, 64, 257])
def test_path_to_root_of_deep_chain(max_nodes):
    # one long path from the root: depth == capacity - 1
    parents = [Tree.NULL_INDEX] + list(range(max_nodes - 1))
    edge_map = [[Tree.NULL_INDEX] * BRANCHING for _ in range(max_nodes)]
    tree = build_tree(max_nodes, parents, edge_map, [0.0] * max_nodes)
    np.testing.assert_array_equal(
        tree.path_to_root(max_nodes - 1), list(range(max_nodes - 1, -1, -1))
    )


def test_path_to_root_of_null_index_is_empty():
    tree = build_tree(*random_tree(random.Random(0), num_nodes=MAX_NODES))
    np.testing.assert_array_equal(
        tree.path_to_root(Tree.NULL_INDEX), [Tree.NULL_INDEX] * MAX_NODES
    )


@pytest.mark.parametrize("seed", range(10))
def test_root_subtrees_match_reference(seed):
    rng = random.Random(seed)
    tree = random_tree(rng, max_nodes=rng.choice([1, 2, 12, 33]))
    num_nodes, parents = tree[0], tree[1]
    capacity = len(parents)

    def root_child(index):
        # the last node before the root on the path up, or the node itself for the root, its
        # children and unused nodes
        path = [i for i in reference_path(parents, index) if i != Tree.NULL_INDEX]
        return path[-2] if len(path) > 1 else index

    expected = [root_child(i) if i < num_nodes else i for i in range(capacity)]
    np.testing.assert_array_equal(build_tree(*tree).root_subtrees(), expected)
