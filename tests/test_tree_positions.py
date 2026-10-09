"""Search tree training positions (core.training.tree_positions): which nodes are chosen, and the targets
they train on."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.training.tree_positions import (
    TreePositions,
    child_visits,
    select_nodes,
    tree_experiences,
)


def synthetic_tree(make_search, visits, terminated=()):
    """A tic-tac-toe search tree whose root has a child along action i for each of `visits`, visited
    `visits[i]` times, which has a child of its own along action 0, visited one time fewer (so with no
    visited children of its own). The children whose actions are in `terminated` are terminal.

    Returns (tree, index of each root child)."""
    search = make_search()
    evaluator = search.evaluator
    tree = search.init()
    template = tree.data_at(tree.ROOT_INDEX).embedding

    def node(n, terminal=False):
        new = evaluator.new_node(
            policy=jnp.full((9,), 1 / 9),
            value=0.0,
            embedding=template,
            terminated=terminal,
        )
        return replace(new, n=jnp.array(n, dtype=jnp.int32))

    tree = tree.set_root(node(sum(visits) + 1))
    children = []
    for action, n in enumerate(visits):
        tree = tree.add_node(tree.ROOT_INDEX, action, node(n, action in terminated))
        child = int(tree.edge_map[tree.ROOT_INDEX, action])
        tree = tree.add_node(child, 0, node(n - 1))
        children.append(child)
    return tree, children


def selected(indices, valid):
    return sorted(int(i) for i, v in zip(indices, valid, strict=True) if v)


def test_select_nodes_takes_qualifying_nodes_only(make_search):
    # the grandchildren are visited too, but have no visited children to make a policy target of
    tree, children = synthetic_tree(make_search, [30, 20, 10, 3], terminated=(2,))
    key = jax.random.PRNGKey(0)

    # the root, the terminal child, the grandchildren and the child below the threshold don't qualify
    indices, valid = select_nodes(key, tree, num=6, min_visits=5)
    assert selected(indices, valid) == sorted(children[:2])
    # padding fills the rest
    assert valid.sum() == 2 and indices.shape == valid.shape == (6,)

    indices, valid = select_nodes(key, tree, num=6, min_visits=31)
    assert not valid.any()


def test_select_nodes_most_visited(make_search):
    tree, children = synthetic_tree(make_search, [10, 30, 20, 20])

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


def test_select_nodes_samples_in_proportion_to_visits(make_search):
    visits = [30, 20, 10]
    tree, children = synthetic_tree(make_search, visits)
    keys = jax.random.split(jax.random.PRNGKey(0), 6000)

    indices, valid = jax.vmap(lambda k: select_nodes(k, tree, num=1, min_visits=2))(
        keys
    )

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
    search = make_search(num_iterations=64, max_nodes=80)
    state, metadata = ttt.play([0, 2, 1, 3, 8, 4])
    output = search.evaluate(jax.random.PRNGKey(0), search.init(), state, metadata)
    return state, output.eval_state


def test_tree_targets_take_the_perspective_of_the_player_to_move(ttt, lost_search):
    root_state, tree = lost_search
    # O's winning moves after each of X's moves
    o_wins = {5: {6}, 6: {5}, 7: {5, 6}}
    indices, valid = select_nodes(jax.random.PRNGKey(0), tree, num=16, min_visits=2)
    samples, sample_valid = tree_experiences(
        tree, indices, valid, ttt.step_fn, ttt.state_to_nn_input
    )
    np.testing.assert_array_equal(sample_valid, valid)

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


def test_select_nodes_can_leave_out_the_reused_subtree(lost_search):
    _, tree = lost_search
    subtrees = np.asarray(tree.root_subtrees())
    key = jax.random.PRNGKey(0)
    everything = selected(*select_nodes(key, tree, num=32, min_visits=2))
    reused = int(tree.edge_map[tree.ROOT_INDEX, 6])
    assert reused in everything and (subtrees[everything] == reused).sum() > 1

    left = selected(
        *select_nodes(
            key, tree, num=32, min_visits=2, exclude_subtree=jnp.array(reused)
        )
    )

    # everything else, and nothing from the reused subtree, the node itself included
    assert left == [i for i in everything if subtrees[i] != reused]
    # nothing to leave out
    assert (
        selected(
            *select_nodes(
                key,
                tree,
                num=32,
                min_visits=2,
                exclude_subtree=jnp.array(tree.NULL_INDEX),
            )
        )
        == everything
    )


def test_tree_experiences_transformed_copies(ttt, lost_search):
    _, tree = lost_search
    indices, valid = select_nodes(jax.random.PRNGKey(0), tree, num=4, min_visits=2)

    def transpose(mask, policy, state):
        # the board's transpose: square (r, c) <-> (c, r)
        perm = jnp.arange(9).reshape(3, 3).T.reshape(-1)
        return (
            mask[perm],
            policy[perm],
            state.replace(observation=jnp.swapaxes(state.observation, 0, 1)),
        )

    samples, sample_valid = tree_experiences(
        tree, indices, valid, ttt.step_fn, ttt.state_to_nn_input, [transpose]
    )
    perm = np.arange(9).reshape(3, 3).T.reshape(-1)
    assert samples.policy_weights.shape[0] == 8
    np.testing.assert_array_equal(sample_valid, np.tile(valid, 2))
    original = jax.tree.map(lambda x: x[:4], samples)
    copy = jax.tree.map(lambda x: x[4:], samples)
    np.testing.assert_array_equal(copy.policy_weights, original.policy_weights[:, perm])
    np.testing.assert_array_equal(copy.policy_mask, original.policy_mask[:, perm])
    np.testing.assert_array_equal(
        copy.observation_nn, np.swapaxes(original.observation_nn, 1, 2)
    )
    for field in ("reward", "search_value", "cur_player_id"):
        np.testing.assert_array_equal(getattr(copy, field), getattr(original, field))


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
