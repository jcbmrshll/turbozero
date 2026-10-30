"""Search tree training positions (core.training.tree_positions): which nodes are chosen, and the targets
they train on."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.state import init_mcts_tree
from core.training.tree_positions import (
    TreePositions,
    child_visits,
    select_nodes,
    tree_experiences,
)


def hand_built(nodes, capacity=16):
    """A search tree of `nodes`, (parent, action, n, terminated) for each node in the order they're
    added, the root's (None, None, n, False). `select_nodes` reads nothing else, so the nodes hold
    no positions."""

    def node(n, terminated):
        new = MCTS.new_node(
            policy=jnp.full((9,), 1 / 9),
            value=0.0,
            embedding=jnp.zeros(()),
            terminated=terminated,
        )
        return replace(new, n=jnp.array(n, dtype=jnp.int32))

    tree = init_mcts_tree(capacity, 9, node(0, False))
    for parent, action, n, terminated in nodes:
        data = node(n, terminated)
        tree = (
            tree.set_root(data)
            if parent is None
            else tree.add_node(parent, action, data)
        )
    return tree


def synthetic_tree(visits, terminated=()):
    """A search tree whose root has a child along action i for each of `visits`, visited `visits[i]`
    times, which has a child of its own along action 0, visited one time fewer (so with no visited
    children of its own). The children whose actions are in `terminated` are terminal.

    Returns (tree, index of each root child)."""
    nodes = [(None, None, sum(visits) + 1, False)]
    children = []
    for action, n in enumerate(visits):
        children.append(len(nodes))
        nodes += [(0, action, n, action in terminated), (len(nodes), 0, n - 1, False)]
    return hand_built(nodes), children


def selected(indices, valid):
    return sorted(int(i) for i, v in zip(indices, valid, strict=True) if v)


def test_select_nodes_takes_qualifying_nodes_only():
    # the grandchildren are visited too, but have no visited children to make a policy target of
    tree, children = synthetic_tree([30, 20, 10, 3], terminated=(2,))
    key = jax.random.PRNGKey(0)

    # the root, the terminal child, the grandchildren and the child below the threshold don't qualify
    indices, valid = select_nodes(key, tree, num=6, min_visits=5)
    assert selected(indices, valid) == sorted(children[:2])
    # padding fills the rest
    assert valid.sum() == 2 and indices.shape == valid.shape == (6,)

    indices, valid = select_nodes(key, tree, num=6, min_visits=31)
    assert not valid.any()


def test_select_nodes_most_visited():
    tree, children = synthetic_tree([10, 30, 20, 20])

    for seed in range(5):
        indices, valid = select_nodes(
            jax.random.PRNGKey(seed), tree, num=1, min_visits=2, most_visited=True
        )
        assert selected(indices, valid) == [children[1]]
    # ties are broken at random
    seconds = set()
    for seed in range(20):
        picks = selected(
            *select_nodes(
                jax.random.PRNGKey(seed), tree, num=2, min_visits=2, most_visited=True
            )
        )
        assert children[1] in picks
        seconds |= set(picks) - {children[1]}
    assert seconds == {children[2], children[3]}


def test_select_nodes_samples_in_proportion_to_visits():
    visits = [30, 20, 10]
    tree, children = synthetic_tree(visits)
    keys = jax.random.split(jax.random.PRNGKey(0), 6000)

    indices, valid = jax.jit(
        jax.vmap(lambda k: select_nodes(k, tree, num=1, min_visits=2))
    )(keys)

    assert valid.all()
    freq = [(indices[:, 0] == c).mean() for c in children]
    np.testing.assert_allclose(freq, np.array(visits) / sum(visits), atol=0.02)
    # without replacement
    indices, valid = jax.vmap(lambda k: select_nodes(k, tree, num=3, min_visits=2))(
        keys[:50]
    )
    assert valid.all()
    assert all(sorted(row.tolist()) == sorted(children) for row in indices)


@pytest.fixture(scope="module")
def lost_search(ttt, make_search):
    """A tic-tac-toe search from a lost position: X (at 0, 1 and 8) is to move, and O (at 2, 3 and 4)
    threatens both 5 and 6, so whichever of 5, 6 and 7 X plays, O wins next move."""
    search = make_search()
    state, metadata = ttt.play([0, 2, 1, 3, 8, 4])
    output = search.evaluate(jax.random.PRNGKey(0), search.init(), state, metadata)
    return state, output.eval_state


def test_tree_targets_take_the_perspective_of_the_player_to_move(ttt, lost_search):
    root_state, tree = lost_search
    # O's winning moves after each of X's moves
    o_wins = {5: {6}, 6: {5}, 7: {5, 6}}
    indices, valid = select_nodes(jax.random.PRNGKey(0), tree, num=16, min_visits=2)
    samples = jax.jit(
        lambda tree, indices: tree_experiences(
            tree, indices, ttt.step_fn, ttt.state_to_nn_input
        )
    )(tree, indices)

    # the network's values are all 0, so only game outcomes move q away from 0: the root is lost for X
    assert float(tree.data.q[tree.ROOT_INDEX]) < -0.5
    o_player = 1 - int(root_state.current_player)
    root_children = 0
    for i, node_index in enumerate(np.asarray(indices)):
        if not valid[i]:
            continue
        node = tree.data_at(node_index)
        state = node.embedding
        player = int(samples.cur_player_id[i])
        q = float(node.q)
        # the player to move, legal moves and observation are the node's
        assert player == int(state.current_player)
        np.testing.assert_array_equal(samples.policy_mask[i], state.legal_action_mask)
        np.testing.assert_array_equal(samples.observation_nn[i], state.observation)
        # the policy target is the children's visit distribution, on legal moves only
        visits = np.asarray(child_visits(tree)[node_index], dtype=np.float32)
        np.testing.assert_allclose(samples.policy_weights[i], visits / visits.sum())
        assert not np.asarray(samples.policy_weights[i])[~state.legal_action_mask].any()
        # the value target is q for the player to move there, and -q for the other player
        assert float(samples.search_value[i]) == pytest.approx(q)
        assert float(samples.reward[i, player]) == pytest.approx(q)
        assert float(samples.reward[i, 1 - player]) == pytest.approx(-q)

        if int(tree.parents[node_index]) == tree.ROOT_INDEX:
            # O is to move, and wins: the node is good for O, who mostly searched a winning move
            action = int(
                np.argmax(np.asarray(tree.edge_map[tree.ROOT_INDEX]) == node_index)
            )
            assert player == o_player
            assert q > 0.5, (action, q)
            assert int(np.argmax(samples.policy_weights[i])) in o_wins[action]
            root_children += 1
    # every one of X's moves is visited often enough
    assert root_children == 3


def test_select_nodes_can_leave_out_the_reused_subtree():
    # root children 1 and 4; 1's child 2 and grandchild 3, 4's child 5. 1, 2 and 4 have visited
    # children
    tree = hand_built(
        [
            (None, None, 19, False),
            (0, 0, 10, False),
            (1, 0, 6, False),
            (2, 0, 5, False),
            (0, 1, 8, False),
            (4, 0, 7, False),
        ]
    )
    key = jax.random.PRNGKey(0)

    def chosen(exclude_subtree):
        return selected(
            *select_nodes(
                key,
                tree,
                num=8,
                min_visits=2,
                exclude_subtree=jnp.array(exclude_subtree),
            )
        )

    # everything but the reused subtree, the child of the root included
    assert chosen(1) == [4]
    assert chosen(4) == [1, 2]
    # nothing to leave out
    assert chosen(tree.NULL_INDEX) == [1, 2, 4]


def test_tree_positions_batch_count_and_decay():
    tree = TreePositions(per_move=2, min_visits=4, capacity=64)
    # 1:1 by default
    assert tree.batch_count(0, 4096) == 2048
    assert tree.batch_count(100, 4096) == 2048

    decaying = replace(tree, ratio=1.0, half_life=10.0)
    assert decaying.ratio_at(0) == 1.0
    assert decaying.ratio_at(10) == pytest.approx(0.5)
    assert decaying.ratio_at(20) == pytest.approx(0.25)
    # ratio 1/2 is a third of the batch
    assert decaying.batch_count(10, 300) == 100
    assert decaying.batch_count(1000, 4096) == 0

    assert replace(tree, ratio=3.0).batch_count(0, 8) == 6

    with pytest.raises(ValueError):
        TreePositions(per_move=0, min_visits=4, capacity=64)
    with pytest.raises(ValueError):
        TreePositions(per_move=1, min_visits=1, capacity=64)
