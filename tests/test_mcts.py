"""MCTS: each step of a search (root update, traversal, expansion,
backpropagation, eviction, action choice, tree reuse) run on small hand-built
trees of a fixed-length environment, and a few whole searches on tic-tac-toe for
what only a search shows: tactics, and its bookkeeping over many iterations, in
a tree with room to spare and in one too small for it."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import copies_in_loop, open_loops_in_loop

from turbozero.evaluators.mcts.action_selection import MCTSActionSelector, PUCTSelector
from turbozero.evaluators.mcts.mcts import MCTS
from turbozero.evaluators.mcts.state import MCTSNode

NULL = -1


# AlphaZero's root noise
NOISE = {"dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25}


@pytest.fixture(scope="module")
def env(fixed_length_env):
    """A two-player environment with 3 actions, whose games last longer than any test's tree."""
    return fixed_length_env(length=10, num_actions=3)


def fixed_eval_fn(logits, value=0.0):
    """An evaluation function giving every state `logits` and `value`."""
    logits = jnp.asarray(logits, dtype=jnp.float32)

    def eval_fn(state, params, key):
        return logits, jnp.array(value, dtype=jnp.float32)

    return eval_fn


def varying_eval_fn(state, params, key):
    """An evaluation function whose value differs from one move to the next."""
    return jnp.array([0.5, 0.0, -0.5]), jnp.sin(state.step.astype(jnp.float32))


def state_eval_fn(state, params, key):
    """An evaluation function whose policy and value both differ from one move to the next (none
    of its arithmetic is constant, for XLA to fold where it would compute it at runtime)."""
    step = state.step.astype(jnp.float32)
    return jnp.array([0.5, 0.0, -0.5]) * jnp.cos(step), jnp.sin(step)


class FirstAction(MCTSActionSelector):
    """Always chooses action 0, so that a test decides where a traversal goes."""

    def __call__(self, key, tree, index, discount):
        return jnp.array(0, dtype=jnp.int32)


def make_mcts(
    cls=MCTS,
    num_actions=3,
    max_nodes=8,
    leaf_eval_fn=None,
    logits=None,
    value=0.0,
    selector=None,
    num_iterations=1,
    **kwargs,
):
    """An MCTS evaluator (of class `cls`) whose evaluation function gives every state `logits`
    (uniform by default) and `value`, unless it's `leaf_eval_fn`."""
    if leaf_eval_fn is None:
        leaf_eval_fn = fixed_eval_fn(
            logits if logits is not None else jnp.zeros(num_actions), value
        )
    return cls(
        leaf_eval_fn=leaf_eval_fn,
        action_selector=selector if selector is not None else PUCTSelector(),
        branching_factor=num_actions,
        max_nodes=max_nodes,
        num_iterations=num_iterations,
        **kwargs,
    )


def build_tree(search, env, nodes):
    """Builds a tree of `search` holding `env`'s positions: `nodes` are (parent,
    action, n, q) for each node in the order they're added, the root's (None, None,
    n, q). Each node holds the position its moves lead to, and a uniform policy over
    its legal moves."""
    root_state, root_meta = env.init_fn(jax.random.PRNGKey(0))
    tree = search.init(template_embedding=root_state)
    states = []
    for parent, action, n, q in nodes:
        if parent is None:
            state, meta = root_state, root_meta
        else:
            state, meta = env.step_fn(states[parent], action)
        states.append(state)
        node = MCTSNode(
            n=jnp.array(n, dtype=jnp.int32),
            p=meta.action_mask / meta.action_mask.sum(),
            q=jnp.array(q, dtype=jnp.float32),
            terminated=meta.terminated,
            embedding=state,
        )
        tree = (
            tree.set_root(node)
            if parent is None
            else tree.set_child(parent, action, node)
        )
    return tree


# a full tree of 6 nodes: the root (10 visits) has children 1 (4 visits) along action 0 and 2 (5
# visits) along action 1; 1 has leaves 3 (1 visit) and 4 (2 visits), and 2 has leaf 5 (1 visit)
FULL = [
    (None, None, 10, 0.0),
    (0, 0, 4, 0.0),
    (0, 1, 5, 0.0),
    (1, 0, 1, 0.0),
    (1, 1, 2, 0.0),
    (2, 0, 1, 0.0),
]
# a single path of 4 nodes along action 0
CHAIN = [(None, None, 4, 0.0), (0, 0, 3, 0.0), (1, 0, 2, 0.0), (2, 0, 1, 0.0)]


def root(tree):
    return tree.data_at(tree.ROOT_INDEX)


def root_child_visits(tree):
    return tree.get_child_data("n", tree.ROOT_INDEX)


def masked_softmax(logits, mask):
    return jax.nn.softmax(jnp.where(jnp.asarray(mask), jnp.asarray(logits), -jnp.inf))


def assert_consistent(tree):
    """Parents and edges agree, and every node has at least one visit more than its children
    (exactly one more, but for the visits of evicted leaves)."""
    parents, edge_map, n = map(np.asarray, (tree.parents, tree.edge_map, tree.data.n))
    used = int(tree.next_free_idx)
    for i in range(used):
        children = edge_map[i][edge_map[i] != tree.NULL_INDEX]
        assert (children < used).all() and (parents[children] == i).all()
        assert n[i] >= 1 + n[children].sum()
        if i != tree.ROOT_INDEX:
            assert (edge_map[parents[i]] == i).sum() == 1


def assert_trees_equal(a, b):
    jax.tree.map(np.testing.assert_array_equal, a, b)


# the root


def evaluate_root(search, env, tree, state, meta):
    """Runs the search's root cycle (cycle 0) on `tree`, for `state`."""
    return search.search_cycle(
        jax.random.PRNGKey(0), tree, jnp.array(0), state, meta, None, env.step_fn
    )


def test_the_root_update_evaluates_an_empty_root_with_a_masked_prior(fixed_length_env):
    env = fixed_length_env(length=10, num_actions=3, action_mask=(True, False, True))
    search = make_mcts(logits=[1.0, 5.0, 0.0], value=0.3)
    state, meta = env.init_fn(jax.random.PRNGKey(0))
    tree = search.init(template_embedding=state)

    tree = evaluate_root(search, env, tree, state, meta)

    node = root(tree)
    np.testing.assert_allclose(
        node.p, masked_softmax([1.0, 5.0, 0.0], meta.action_mask), rtol=1e-6
    )
    assert int(node.n) == 1 and float(node.q) == pytest.approx(0.3)
    assert int(tree.next_free_idx) == 1


def test_the_root_update_keeps_a_reused_roots_visits_and_value(env):
    search = make_mcts(logits=[0.0, 1.0, 2.0], value=0.9)
    tree = build_tree(search, env, [(None, None, 5, -0.2), (0, 1, 4, 0.1)])
    state, meta = env.init_fn(jax.random.PRNGKey(0))

    tree = evaluate_root(search, env, tree, state, meta)

    node = root(tree)
    assert int(node.n) == 5 and float(node.q) == pytest.approx(-0.2)
    # the prior is the new evaluation's
    np.testing.assert_allclose(node.p, jax.nn.softmax(jnp.array([0.0, 1.0, 2.0])))


def test_the_roots_prior_takes_the_noise_its_search_drew(fixed_length_env):
    mask = np.array([True, False, True, True])
    env = fixed_length_env(length=10, num_actions=4, action_mask=mask)
    logits = jnp.array([2.0, 1.0, 0.5, -1.0])
    search = make_mcts(num_actions=4, logits=logits, **NOISE)
    state, meta = env.init_fn(jax.random.PRNGKey(0))
    tree = search.init(template_embedding=state)

    started = search.start_search(jax.random.PRNGKey(1), tree)
    kept = search.start_search(jax.random.PRNGKey(2), started, active=False)
    other = search.start_search(jax.random.PRNGKey(2), tree)
    updated = evaluate_root(search, env, kept, state, meta)

    # a search that doesn't start keeps its noise, and another key draws another
    np.testing.assert_array_equal(kept.root_noise, started.root_noise)
    assert not np.allclose(other.root_noise, started.root_noise)
    np.testing.assert_allclose(
        root(updated).p,
        0.75 * masked_softmax(logits, mask)
        + 0.25 * masked_softmax(started.root_noise, mask),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    # a small alpha's Gamma draws underflow to 0 outside log space
    "alpha",
    [0.03, 0.3, 2.0],
)
def test_dirichlet_noise_is_mixed_into_the_root_prior_over_legal_moves_only(
    fixed_length_env, alpha
):
    mask = np.array([True, True, False, True, False, True])
    env = fixed_length_env(length=10, num_actions=6, action_mask=mask)
    logits = jnp.array([2.0, -1.0, 0.5, 0.0, 3.0, -2.0])
    epsilon = 0.25
    search = make_mcts(
        num_actions=6,
        logits=logits,
        dirichlet_alpha=alpha,
        dirichlet_epsilon=epsilon,
    )
    state, meta = env.init_fn(jax.random.PRNGKey(0))
    tree = search.init(template_embedding=state)
    keys = jax.random.split(jax.random.PRNGKey(0), 1024)

    priors = jax.jit(
        jax.vmap(
            lambda key: (
                root(
                    search.search_cycle(
                        key,
                        search.start_search(key, tree),
                        jnp.array(0),
                        state,
                        meta,
                        None,
                        env.step_fn,
                    )
                ).p
            )
        )
    )(keys)

    np.testing.assert_allclose(priors.sum(axis=-1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(priors[:, ~mask], 0.0)
    # the noise differs from key to key
    assert not np.allclose(priors[0], priors[1])
    # what's left after taking out the network's share is the noise: a distribution
    # over the legal moves, carrying exactly `epsilon` of the root prior
    noise = (priors - (1 - epsilon) * masked_softmax(logits, mask)) / epsilon
    assert (noise > -1e-5).all()
    np.testing.assert_allclose(noise.sum(axis=-1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(noise[:, ~mask], 0.0, atol=1e-6)
    # a symmetric Dirichlet: each legal move gets an equal share on average, spread as
    # Dirichlet(alpha)'s, (1/k)(1 - 1/k) / (k alpha + 1) over k legal moves
    k = mask.sum()
    np.testing.assert_allclose(noise[:, mask].mean(axis=0), 1 / k, atol=0.06)
    np.testing.assert_allclose(
        noise[:, mask].var(), (1 / k) * (1 - 1 / k) / (k * alpha + 1), rtol=0.15
    )


def test_drawing_the_root_noise_has_no_open_loops(env):
    # on GPU, each trip of a loop whose trip count XLA doesn't know waits on the host: self-play
    # starts searches (and draws their noise) for every environment at every check
    search = make_mcts(**NOISE)
    tree = search.init(template_embedding=env.init_fn(None)[0])
    keys = jax.random.split(jax.random.PRNGKey(0), 5)

    hlo = (
        jax.jit(jax.vmap(lambda key: search.start_search(key, tree)))
        .lower(keys)
        .compile()
        .as_text()
    )

    assert hlo is not None
    # (the random bits' hash rounds may stay a loop, of a known trip count)
    loops = [line for line in hlo.splitlines() if " while(" in line]
    assert all("known_trip_count" in line for line in loops)


# choosing the move to play


def sample(search, key, tree):
    """The move `search` plays from `tree`'s root with key `key`, and its policy weights."""
    out = search.search_output(key, tree, None, None, None, None)
    return out.action, out.policy_weights


def root_with_child_visits(search, env, visits):
    """A tree whose root has a child along action i visited `visits[i]` times (none for 0)."""
    return build_tree(
        search,
        env,
        [(None, None, 1 + sum(visits), 0.0)]
        + [(0, action, n, 0.0) for action, n in enumerate(visits) if n > 0],
    )


def test_temperature_zero_plays_a_most_visited_move(fixed_length_env):
    env = fixed_length_env(length=10, num_actions=4)
    search = make_mcts(num_actions=4, temperature=0.0)
    tree = root_with_child_visits(search, env, [3, 0, 5, 5])
    keys = jax.random.split(jax.random.PRNGKey(0), 64)

    actions, weights = jax.vmap(lambda k: sample(search, k, tree))(keys)

    # ties are broken at random
    assert set(np.asarray(actions).tolist()) == {2, 3}
    np.testing.assert_allclose(weights[0], np.array([3, 0, 5, 5]) / 13, rtol=1e-6)


@pytest.mark.parametrize(
    "temperature, expected", [(1.0, [3, 0, 5, 5]), (0.5, [9, 0, 25, 25])]
)
def test_moves_are_sampled_by_their_visits_to_the_power_of_one_over_the_temperature(
    fixed_length_env, temperature, expected
):
    env = fixed_length_env(length=10, num_actions=4)
    search = make_mcts(num_actions=4, temperature=temperature)
    tree = root_with_child_visits(search, env, [3, 0, 5, 5])
    keys = jax.random.split(jax.random.PRNGKey(0), 4000)

    actions, weights = jax.jit(jax.vmap(lambda k: sample(search, k, tree)))(keys)

    frequencies = np.bincount(np.asarray(actions), minlength=4) / len(keys)
    np.testing.assert_allclose(
        frequencies, np.array(expected) / sum(expected), atol=0.03
    )
    # the policy weights are the visits, without the temperature
    np.testing.assert_allclose(weights[0], np.array([3, 0, 5, 5]) / 13, rtol=1e-6)


def test_a_root_without_visited_children_has_a_uniform_policy(fixed_length_env):
    env = fixed_length_env(length=10, num_actions=4)
    search = make_mcts(num_actions=4)
    tree = root_with_child_visits(search, env, [0, 0, 0, 0])

    _, weights = sample(search, jax.random.PRNGKey(0), tree)

    np.testing.assert_allclose(weights, 0.25)


# traversal, expansion and backpropagation


def test_traversal_follows_the_selector_to_an_unexpanded_edge(env):
    search = make_mcts(selector=FirstAction())
    tree = build_tree(search, env, CHAIN[:3])

    parent, action, _, _ = search.find_leaf(jax.random.PRNGKey(0), tree, env.step_fn)

    assert [int(parent), int(action)] == [2, 0]


def test_traversal_breaks_ties_with_its_key(env):
    # a root with no visited children and a uniform prior: every action scores the same
    search = make_mcts()
    tree = build_tree(search, env, CHAIN[:1])
    find_leaf = jax.jit(lambda key: search.find_leaf(key, tree, env.step_fn))

    actions = {int(find_leaf(jax.random.PRNGKey(seed))[1]) for seed in range(16)}

    assert actions == {0, 1, 2}


def test_traversal_stops_at_a_terminal_child(fixed_length_env):
    # node 2, two moves in, ends the game
    env = fixed_length_env(length=2, num_actions=3)
    search = make_mcts(selector=FirstAction())
    tree = build_tree(search, env, CHAIN[:3])
    assert bool(tree.data_at(2).terminated)

    parent, action, _, _ = search.find_leaf(jax.random.PRNGKey(0), tree, env.step_fn)

    assert [int(parent), int(action)] == [1, 0]


def iterate(search, env):
    """Runs one iteration of `search` (its cycle 1) on a tree of `env`, compiled."""
    root_state, root_meta = env.init_fn(jax.random.PRNGKey(0))
    return jax.jit(
        lambda key, tree: search.search_cycle(
            key, tree, jnp.array(1), root_state, root_meta, None, env.step_fn
        )
    )


def test_an_iteration_expands_a_leaf_and_backs_its_value_up(fixed_length_env):
    env = fixed_length_env(length=10, num_actions=3, action_mask=(True, False, True))
    search = make_mcts(logits=[1.0, 5.0, 0.0], value=0.3, selector=FirstAction())
    tree = build_tree(search, env, [(None, None, 1, 0.5)])

    tree = iterate(search, env)(jax.random.PRNGKey(0), tree)

    assert int(tree.edge_map[tree.ROOT_INDEX, 0]) == 1 and int(tree.parents[1]) == 0
    child = tree.data_at(1)
    assert int(child.n) == 1 and float(child.q) == pytest.approx(0.3)
    # the position after the move, and its evaluation's prior over its own legal moves
    assert int(child.embedding.step) == 1
    np.testing.assert_allclose(
        child.p, masked_softmax([1.0, 5.0, 0.0], [True, False, True]), rtol=1e-6
    )
    # the root gets the child's value from the other player's side
    assert int(root(tree).n) == 2
    assert float(root(tree).q) == pytest.approx((0.5 - 0.3) / 2)


def test_an_iteration_values_a_new_terminal_leaf_by_its_outcome(fixed_length_env):
    # the root's child along action 0 ends the game, won by player 0: it's worth -1
    # to player 1, who would be to move there, whatever the evaluation says
    env = fixed_length_env(length=1, num_actions=3)
    search = make_mcts(value=0.3, selector=FirstAction())
    tree = build_tree(search, env, [(None, None, 1, 0.0)])

    tree = iterate(search, env)(jax.random.PRNGKey(0), tree)

    assert bool(tree.data_at(1).terminated)
    assert int(tree.data_at(1).n) == 1 and float(tree.data_at(1).q) == pytest.approx(-1)
    assert float(root(tree).q) == pytest.approx(0.5)


def test_an_iteration_revisits_a_terminal_child_with_its_outcome(fixed_length_env):
    # the root's child along action 0 ends the game, won by player 0: it's worth -1 to player 1,
    # who would be to move there
    env = fixed_length_env(length=1, num_actions=3)
    search = make_mcts(value=0.3, selector=FirstAction())
    tree = build_tree(search, env, [(None, None, 2, 1.0), (0, 0, 1, -1.0)])

    tree = iterate(search, env)(jax.random.PRNGKey(0), tree)

    # no new node: the child is visited again, with the outcome rather than the evaluated value
    assert int(tree.next_free_idx) == 2
    assert int(tree.data_at(1).n) == 2 and float(tree.data_at(1).q) == pytest.approx(-1)
    assert int(root(tree).n) == 3 and float(root(tree).q) == pytest.approx(1)


def test_an_iteration_in_a_full_tree_takes_an_evicted_leafs_place(env):
    search = make_mcts(max_nodes=6, selector=FirstAction())
    tree = build_tree(search, env, FULL)

    # it reaches leaf 3 (along action 0 from the root, then from 1) and expands it
    # in place of leaf 5: the least visited leaf below the root's children, other
    # than 3 itself, the node the new leaf goes under
    tree = iterate(search, env)(jax.random.PRNGKey(0), tree)

    assert int(tree.edge_map[3, 0]) == 5 and int(tree.parents[5]) == 3
    assert (np.asarray(tree.edge_map[2]) == NULL).all()
    assert int(tree.data.n[5]) == 1 and int(root(tree).n) == 11
    assert_consistent(tree)


def test_an_iteration_in_a_full_tree_with_nothing_to_evict_changes_nothing(env):
    # the only leaf is the one the iteration expands under. Backing its value up anyway would
    # reach the root but no child of it, and PUCT would choose the same leaf again and again
    search = make_mcts(max_nodes=4, selector=FirstAction())
    tree = build_tree(search, env, CHAIN)

    assert_trees_equal(iterate(search, env)(jax.random.PRNGKey(0), tree), tree)


@pytest.mark.parametrize(
    "nodes, num_actions, parent, action, evicted",
    [
        # the least-visited leaf below the root's children: of 3 and 5, the one whose parent has
        # fewer visits
        (FULL, 3, 1, 2, 3),
        # a child of the root only when every leaf is one: the least visited
        (
            [(None, None, 7, 0.0), (0, 0, 3, 0.0), (0, 1, 1, 0.0), (0, 2, 2, 0.0)],
            4,
            0,
            3,
            2,
        ),
    ],
)
def test_a_new_leaf_in_a_full_tree_evicts(
    fixed_length_env, nodes, num_actions, parent, action, evicted
):
    env = fixed_length_env(length=10, num_actions=num_actions)
    search = make_mcts(num_actions=num_actions, max_nodes=len(nodes))
    tree = build_tree(search, env, nodes)
    leaf = (parent, action, *env.step_fn(tree.data_at(parent).embedding, action))

    expanded = search.expand(tree, leaf, jnp.zeros(num_actions), 0.0)

    assert int(expanded.edge_map[parent, action]) == evicted
    assert int(expanded.parents[evicted]) == parent
    assert_consistent(expanded)


@pytest.mark.parametrize(
    "discount, values",
    [
        # the value each node gets, from node 2 up to the root
        # two players: the sign flips at each level up
        (-1.0, [-0.8, 0.8, -0.8]),
        # one player: the same value at every level
        (1.0, [0.8, 0.8, 0.8]),
        # one player, discounted: once more at each level up
        (0.5, [0.4, 0.2, 0.1]),
    ],
)
def test_backpropagation_visits_every_node_up_to_the_root(env, discount, values):
    search = make_mcts(discount=discount)
    # q of 0.3 at the root, 0.2 at node 1, and 0.1 at node 2
    tree = build_tree(
        search, env, [(None, None, 3, 0.3), (0, 0, 2, 0.2), (1, 0, 1, 0.1)]
    )
    leaf = (2, 0, *env.step_fn(tree.data_at(2).embedding, 0))

    # the value 0.8 of a new leaf under node 2
    backed_up = search.expand(tree, leaf, jnp.zeros(3), 0.8)

    new = backed_up.data_at(3)
    assert int(new.n) == 1 and float(new.q) == pytest.approx(0.8)
    np.testing.assert_array_equal(backed_up.data.n[:3], [4, 3, 2])
    old_q, old_n = np.array([0.3, 0.2, 0.1]), np.array([3, 2, 1])
    np.testing.assert_allclose(
        backed_up.data.q[:3],
        (old_q * old_n + np.array(values[::-1])) / (old_n + 1),
        rtol=1e-6,
    )


@pytest.mark.parametrize("persist_tree", [True, False])
def test_step_keeps_the_played_moves_subtree_or_starts_over(env, persist_tree):
    search = make_mcts(persist_tree=persist_tree)
    tree = build_tree(search, env, FULL)

    stepped = search.step(tree, jnp.array(1))

    if persist_tree:
        # node 2 and its child 5
        assert int(stepped.next_free_idx) == 2
        assert_trees_equal(root(stepped), tree.data_at(2))
        assert_trees_equal(stepped.data_at(stepped.edge_map[0, 0]), tree.data_at(5))
    else:
        assert int(stepped.next_free_idx) == 0


@pytest.mark.parametrize("persist_tree", [True, False])
def test_next_state_steps_resets_or_keeps_each_tree(env, persist_tree):
    # as self-play moves a batch of trees on: one played a move, one ended its game,
    # and one didn't play
    search = make_mcts(persist_tree=persist_tree)
    tree = build_tree(search, env, FULL)
    trees = jax.tree.map(lambda x: jnp.stack([x] * 3), tree)

    moved = jax.vmap(search.next_state)(
        trees,
        jnp.array([1, 1, 1]),
        jnp.array([False, True, False]),
        jnp.array([True, True, False]),
    )

    def at(i):
        return jax.tree.map(lambda x: x[i], moved)

    assert_trees_equal(at(0), search.step(tree, jnp.array(1)))
    assert_trees_equal(at(1), search.reset(tree))
    assert_trees_equal(at(2), tree)


class RecordingSelector(PUCTSelector):
    """PUCT, recording each key it receives with `record`."""

    def __init__(self, record):
        super().__init__()
        self.record = record

    def __call__(self, key, tree, index, discount):
        self.record("select", key)
        return super().__call__(key, tree, index, discount)


class KeyRecordingMCTS(MCTS):
    """MCTS whose leaf evaluations and selections record the key each one receives."""

    def __init__(self, **kwargs):
        def leaf_eval_fn(state, params, key):
            self.record("leaf_eval_fn", key)
            return jnp.zeros(self.branching_factor), jnp.array(0.0)

        super().__init__(
            leaf_eval_fn=leaf_eval_fn,
            action_selector=RecordingSelector(self.record),
            **kwargs,
        )
        self.recorded = []

    def record(self, name, key):
        jax.debug.callback(lambda k: self.recorded.append((name, np.asarray(k))), key)


def test_evaluate_gives_every_consumer_an_independent_key(env):
    search = KeyRecordingMCTS(branching_factor=3, max_nodes=8, num_iterations=4)
    state, meta = env.init_fn(jax.random.PRNGKey(0))
    tree = search.init(template_embedding=state)
    key = jax.random.PRNGKey(0)

    search.evaluate(key, tree, state, meta, None, env.step_fn)
    jax.effects_barrier()

    names = [name for name, _ in search.recorded]
    assert names.count("leaf_eval_fn") == 1 + search.num_iterations
    # a selection at the root and one more for each level down
    assert names.count("select") >= search.num_iterations
    # and the move is sampled with the third key of the move's
    recorded = [*search.recorded, ("sample", jax.random.split(key, 3)[2])]
    keys = [tuple(np.asarray(k).tolist()) for _, k in recorded]
    assert tuple(np.asarray(key).tolist()) not in keys
    assert len(set(keys)) == len(keys)
    # no consumer's key is a split of another consumer's key
    # (the search used to split the action-sampling key into the iteration keys)
    for _, k in recorded:
        children = {
            tuple(c.tolist())
            for n in range(2, search.num_iterations + 1)
            for c in np.asarray(jax.random.split(k, n))
        }
        assert not children & set(keys)


# trees of different depths
SHAPES = [
    [(None, None, 1, 0.0)],
    [(None, None, 3, 0.1), (0, 0, 2, -0.2), (1, 0, 1, 0.3)],
    [
        (None, None, 6, 0.2),
        (0, 0, 2, -0.1),
        (0, 1, 3, 0.4),
        (2, 2, 2, -0.3),
        (3, 1, 1, 0.5),
    ],
]


def search_cycles(search, env, cycles):
    """Runs `start_search` and then `search`'s cycles `cycles` on a tree of
    `env`, with key `key`, in a scan (unrolled, XLA could compile each cycle
    differently and round differently)."""
    root_state, root_meta = env.init_fn(jax.random.PRNGKey(0))

    def run(key, tree):
        def cycle(tree, c):
            tree = search.search_cycle(
                key, tree, c, root_state, root_meta, None, env.step_fn
            )
            return tree, None

        tree = search.start_search(key, tree)
        return jax.lax.scan(cycle, tree, jnp.asarray(cycles, dtype=jnp.int32))[0]

    return run


def test_iterations_do_not_depend_on_the_trees_capacity(env):
    # with room to spare, a larger tree changes nothing: arrays of the tree's
    # capacity (e.g. the path to the root) mustn't change the results, e.g. by XLA
    # fusing the arithmetic differently
    small, large = (
        make_mcts(max_nodes=m, leaf_eval_fn=varying_eval_fn) for m in (24, 40)
    )

    def search_step_search(search):
        run = search_cycles(search, env, jnp.arange(1, 7))

        @jax.jit
        def go(key, tree):
            first_key, second_key = jax.random.split(key)
            searched = run(first_key, tree)
            stepped = search.step(searched, jnp.array(0))
            return searched, stepped, run(second_key, stepped)

        return go(jax.random.PRNGKey(0), build_tree(search, env, SHAPES[2]))

    for a, b in zip(search_step_search(small), search_step_search(large), strict=True):
        assert int(a.next_free_idx) < a.capacity
        np.testing.assert_array_equal(a.next_free_idx, b.next_free_idx)
        for x, y in zip(
            jax.tree.leaves((a.parents, a.edge_map, a.data)),
            jax.tree.leaves((b.parents, b.edge_map, b.data)),
            strict=True,
        ):
            np.testing.assert_array_equal(x, y[: a.capacity])


# a search a cycle at a time, as self-play runs it


def search_until_done(search, env):
    """The search for a move with key `key` of the position in the tree's root: `start_search`,
    then its cycles until it's done."""
    root_state, root_meta = env.init_fn(jax.random.PRNGKey(0))

    def run(key, tree):
        def cycle(carry):
            tree, c = carry
            tree = search.search_cycle(
                key, tree, c, root_state, root_meta, None, env.step_fn
            )
            return tree, c + 1

        return jax.lax.while_loop(
            lambda carry: ~search.search_done(*carry),
            cycle,
            (search.start_search(key, tree), jnp.array(0, dtype=jnp.int32)),
        )[0]

    return run


@pytest.mark.parametrize("reused", [False, True], ids=["empty", "reused"])
@pytest.mark.parametrize(
    "budget",
    [
        {},
        # key 3 draws a full search, key 10 a fast one
        {"full_probability": 0.5, "fast_iterations": 2},
        {"full_probability": 0.5, "fast_iterations": 2, "key": 10},
        {"count_reused_visits": True},
        {
            "full_probability": 0.5,
            "fast_iterations": 2,
            "count_reused_visits": True,
            "key": 10,
        },
    ],
    ids=[
        "iterations",
        "playout-cap-full",
        "playout-cap-fast",
        "visits",
        "playout-cap-fast-visits",
    ],
)
def test_a_search_a_cycle_at_a_time_is_evaluate(env, reused, budget):
    # with the root's evaluation first, the same keys, a tree too small for every iteration, and
    # searches that stop early
    budget = dict(budget)
    key = jax.random.PRNGKey(budget.pop("key", 3))
    search = make_mcts(
        max_nodes=8, leaf_eval_fn=state_eval_fn, num_iterations=6, **NOISE, **budget
    )
    root_state, root_meta = env.init_fn(jax.random.PRNGKey(0))
    tree = (
        build_tree(search, env, SHAPES[2])
        if reused
        else search.init(template_embedding=root_state)
    )
    assert search.search_cycles() == 7
    if "full_probability" in budget:
        # the key draws the kind of search the test is named for
        fast = bool(search.start_search(key, tree).fast)
        assert fast == (key[-1] == 10)

    @jax.jit
    def by_cycles(key, tree):
        tree = search_until_done(search, env)(key, tree)
        return search.search_output(key, tree, root_state, root_meta, None, env.step_fn)

    expected = jax.jit(search.evaluate, static_argnames="env_step_fn")(
        key, tree, root_state, root_meta, None, env_step_fn=env.step_fn
    )

    assert_trees_equal(by_cycles(key, tree), expected)


def test_a_search_past_its_iterations_goes_on_iterating(env):
    # as self-play's does, until its move is played: each cycle past the 4th iteration visits
    # the tree once more
    search = make_mcts(max_nodes=12, leaf_eval_fn=state_eval_fn, num_iterations=4)
    root_state, _ = env.init_fn(jax.random.PRNGKey(0))
    tree = search.init(template_embedding=root_state)
    run = jax.jit(
        lambda key, tree, cycles: search_cycles(search, env, cycles)(key, tree)
    )
    key = jax.random.PRNGKey(0)

    done = run(key, tree, list(range(5)))
    on = run(key, tree, list(range(8)))

    assert int(root(done).n) == 5
    assert int(root(on).n) == 8
    assert_consistent(on)


def test_batched_cycles_match_each_trees_cycle_alone(env):
    # under vmap the root's evaluation and the iteration are both computed, and each
    # tree must take only its own; traversals loop until the deepest in the batch is
    # done, and the shallower ones must come out as if alone; past its iterations, a
    # search goes on
    search = make_mcts(max_nodes=8, leaf_eval_fn=state_eval_fn, num_iterations=4)
    trees = [build_tree(search, env, nodes) for nodes in [*SHAPES, SHAPES[0]]]
    cycles = jnp.array([0, 1, 6, 2])
    keys = jax.random.split(jax.random.PRNGKey(0), len(trees))
    root_state, root_meta = env.init_fn(jax.random.PRNGKey(0))

    def run(key, tree, cycle):
        return search.search_cycle(
            key, tree, cycle, root_state, root_meta, None, env.step_fn
        )

    batched = jax.jit(jax.vmap(run))(
        keys, jax.tree.map(lambda *x: jnp.stack(x), *trees), cycles
    )

    alone = jax.jit(run)
    for i, tree in enumerate(trees):
        assert_trees_equal(
            jax.tree.map(lambda x, i=i: x[i], batched),
            alone(keys[i], tree, cycles[i]),
        )


def test_search_cycles_write_trees_in_place_and_loop_only_to_traverse(env):
    # a copy of an array of the trees' nodes in a cycle makes each cycle's cost grow
    # with the trees' capacity: e.g. where the root's update reads what the iteration
    # writes (or the other way round) and XLA doesn't order the reads first, it
    # copies the arrays to read from. And on GPU each trip of a loop whose trip count
    # XLA doesn't know waits on the host: the traversal's is the only one (the root's
    # noise is drawn once, before the cycles)
    batch, capacity = 5, 24
    search = make_mcts(
        max_nodes=capacity, leaf_eval_fn=state_eval_fn, num_iterations=6, **NOISE
    )
    trees = jax.vmap(lambda _: search.init(template_embedding=env.init_fn(None)[0]))(
        jnp.arange(batch)
    )
    run = jax.vmap(search_cycles(search, env, jnp.arange(search.search_cycles())))
    keys = jax.random.split(jax.random.PRNGKey(0), batch)

    hlo = jax.jit(run).lower(keys, trees).compile().as_text()

    assert hlo is not None
    assert copies_in_loop(hlo, f"{batch},{capacity}") == []
    assert open_loops_in_loop(hlo) == 1


def test_a_search_is_done_after_the_roots_evaluation_and_its_iterations(env):
    search = make_mcts(num_iterations=3)
    tree = build_tree(search, env, SHAPES[2])

    done = [bool(search.search_done(tree, jnp.array(c))) for c in range(6)]

    assert done == [False, False, False, False, True, True]


@pytest.mark.parametrize(
    "nodes, done_from",
    [
        # the root has 1 visit: done after its evaluation and 4 iterations, as without
        (SHAPES[0], 5),
        # 6 visits: done once evaluated
        (SHAPES[2], 1),
    ],
)
def test_a_search_counting_reused_visits_is_done_with_the_roots_visits(
    env, nodes, done_from
):
    search = make_mcts(num_iterations=4, count_reused_visits=True)
    tree = build_tree(search, env, nodes)

    done = [bool(search.search_done(tree, jnp.array(c))) for c in range(7)]

    assert done == [c >= done_from for c in range(7)]


@pytest.mark.parametrize("nodes", [SHAPES[0], SHAPES[2]])
def test_a_search_counting_reused_visits_ends_with_as_many_root_visits(env, nodes):
    # 8 iterations of a search from scratch: 9 visits, from the root's 1 or the reused 6
    search = make_mcts(
        max_nodes=16,
        leaf_eval_fn=state_eval_fn,
        num_iterations=8,
        count_reused_visits=True,
    )
    tree = build_tree(search, env, nodes)
    state, meta = env.init_fn(jax.random.PRNGKey(0))

    searched = search.evaluate(
        jax.random.PRNGKey(0), tree, state, meta, None, env_step_fn=env.step_fn
    ).eval_state

    assert int(root(searched).n) == 9
    assert_consistent(searched)


# whole searches, on tic-tac-toe

# X: 0, 1   O: 4, 8   -> X to move, wins at 2
WIN_MOVES, WINNING_ACTION = [0, 4, 1, 8], 2
# X: 0, 8   O: 4, 2   -> X to move, must block at 6
BLOCK_MOVES, BLOCKING_ACTION = [0, 4, 8, 2], 6
# a few moves in, so some actions are illegal
MIDGAME_MOVES = [4, 0, 8, 2]


def run_search(search, key, moves, ttt, first_player=0):
    state, meta = ttt.play(moves, first_player)
    return search.evaluate(key, search.init(), state, meta), meta


def test_a_search_and_the_next_one_from_its_subtree(make_search, ttt):
    search = make_search()
    k = search.evaluator.num_iterations
    state, meta = ttt.play(MIDGAME_MOVES)

    out = search.evaluate(jax.random.PRNGKey(0), search.init(), state, meta)

    tree = out.eval_state
    # every iteration is counted at the root, and in its children
    assert root(tree).n == 1 + k == 1 + root_child_visits(tree).sum()
    assert_consistent(tree)
    # the policy weights are over the legal moves, and so is the move played
    np.testing.assert_allclose(out.policy_weights.sum(), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(out.policy_weights[~meta.action_mask], 0.0)
    assert meta.action_mask[out.action]
    # no node has a child along an illegal move
    legal = np.asarray(tree.data.embedding.board == 0)
    has_child = np.asarray(tree.edge_map != tree.NULL_INDEX)
    assert not (has_child & ~legal).any()

    # the next search goes on from the subtree of the move played
    child_n = tree.data_at(tree.edge_map[tree.ROOT_INDEX, out.action]).n
    state, meta = ttt.step_fn(state, out.action)
    out = search.evaluate(
        jax.random.PRNGKey(1), search.step(tree, out.action), state, meta
    )

    assert root(out.eval_state).n == child_n + k
    assert_consistent(out.eval_state)


@pytest.mark.parametrize("first_player", [0, 1])
def test_finds_forced_win(make_search, ttt, first_player):
    out, _ = run_search(
        make_search(), jax.random.PRNGKey(0), WIN_MOVES, ttt, first_player
    )

    assert out.action == WINNING_ACTION
    # a won position is worth more than a draw to the player to move
    assert root(out.eval_state).q > 0


@pytest.mark.parametrize("first_player", [0, 1])
def test_blocks_forced_loss(make_search, ttt, first_player):
    out, _ = run_search(
        make_search(), jax.random.PRNGKey(0), BLOCK_MOVES, ttt, first_player
    )

    assert out.action == BLOCKING_ACTION


def test_searches_in_a_tree_too_small_for_them(make_search, ttt):
    # 10 nodes for 32 iterations: room for the root's 5 children and deeper leaves to evict
    search = make_search(max_nodes=10)
    k = search.evaluator.num_iterations
    state, meta = ttt.play(MIDGAME_MOVES)

    out = search.evaluate(jax.random.PRNGKey(0), search.init(), state, meta)

    tree = out.eval_state
    assert tree.next_free_idx == tree.capacity
    # every iteration lands on a node, and is counted in the policy weights
    assert root(tree).n == 1 + k
    assert root_child_visits(tree).sum() == k
    assert jnp.isfinite(tree.data.q).all()
    assert_consistent(tree)
    # the next search, from the subtree of the move played, fills the tree again
    state, meta = ttt.step_fn(state, out.action)
    out = search.evaluate(
        jax.random.PRNGKey(1), search.step(tree, out.action), state, meta
    )
    assert_consistent(out.eval_state)
    # and values still reach the root
    out, _ = run_search(search, jax.random.PRNGKey(0), WIN_MOVES, ttt)
    assert out.action == WINNING_ACTION and root(out.eval_state).q > 0


# playout cap randomization


def capped(full_probability=0.25, fast_iterations=2, **kwargs):
    return make_mcts(
        num_iterations=kwargs.pop("num_iterations", 5),
        full_probability=full_probability,
        fast_iterations=fast_iterations,
        **kwargs,
    )


def test_a_playout_cap_draws_a_full_search_with_its_probability(env):
    search = capped(full_probability=0.25)
    root_state, _ = env.init_fn(jax.random.PRNGKey(0))
    tree = search.init(template_embedding=root_state)
    keys = jax.random.split(jax.random.PRNGKey(0), 4000)

    fast = jax.vmap(lambda k: search.start_search(k, tree).fast)(keys)
    kept = jax.vmap(lambda k: search.start_search(k, tree, active=False).fast)(keys)

    assert np.mean(fast) == pytest.approx(0.75, abs=0.03)
    # a search that doesn't start keeps its draw
    assert not np.asarray(kept).any()


def test_without_a_playout_cap_starting_a_search_draws_the_same_noise(env):
    # the cap's draw takes a key of its own: searches without one are as before it
    search = make_mcts(**NOISE)
    root_state, _ = env.init_fn(jax.random.PRNGKey(0))
    tree = search.init(template_embedding=root_state)
    key = jax.random.PRNGKey(7)

    started = search.start_search(key, tree)

    root_key, _, _ = jax.random.split(key, 3)
    _, noise_key = jax.random.split(root_key)
    np.testing.assert_array_equal(
        started.root_noise,
        jax.random.loggamma(noise_key, 0.3, (3,), method="approximate"),
    )
    assert not bool(started.fast)


@pytest.mark.parametrize("fast, done_from", [(False, 6), (True, 3)])
def test_a_fast_search_is_done_after_its_own_iterations(env, fast, done_from):
    search = capped(num_iterations=5, fast_iterations=2)
    tree = replace(build_tree(search, env, SHAPES[0]), fast=jnp.array(fast))

    done = [bool(search.search_done(tree, jnp.array(c))) for c in range(8)]

    assert done == [c >= done_from for c in range(8)]
    assert search.search_cycles() == 6


def test_a_fast_search_counting_reused_visits_stops_at_its_own_visits(env):
    # 6 reused visits: past a fast search's 2 + 1, done once the root is evaluated
    search = capped(num_iterations=8, fast_iterations=2, count_reused_visits=True)
    tree = build_tree(search, env, SHAPES[2])

    fast = replace(tree, fast=jnp.array(True))
    full = replace(tree, fast=jnp.array(False))

    assert bool(search.search_done(fast, jnp.array(1)))
    assert not bool(search.search_done(full, jnp.array(1)))


def test_only_full_searches_roots_take_noise(fixed_length_env):
    mask = np.array([True, False, True, True])
    env = fixed_length_env(length=10, num_actions=4, action_mask=mask)
    logits = jnp.array([2.0, 1.0, 0.5, -1.0])
    search = capped(
        num_actions=4,
        logits=logits,
        dirichlet_epsilon=0.5,
        full_probability=0.5,
    )
    state, meta = env.init_fn(jax.random.PRNGKey(0))
    tree = search.start_search(
        jax.random.PRNGKey(0), search.init(template_embedding=state)
    )

    def prior(fast):
        searched = replace(tree, fast=jnp.array(fast))
        return root(evaluate_root(search, env, searched, state, meta)).p

    np.testing.assert_allclose(prior(True), masked_softmax(logits, mask), rtol=1e-6)
    assert not np.allclose(prior(False), masked_softmax(logits, mask))


def test_only_full_searches_make_training_positions(env):
    root_state, _ = env.init_fn(jax.random.PRNGKey(0))
    search = capped()
    tree = search.init(template_embedding=root_state)

    assert bool(search.trains_on(tree))
    assert not bool(search.trains_on(replace(tree, fast=jnp.array(True))))
    # without a cap, every search
    plain = make_mcts()
    assert bool(plain.trains_on(plain.init(template_embedding=root_state)))


@pytest.mark.parametrize("probability, fast", [(0.0, 2), (1.5, 2), (0.5, 0), (0.5, 5)])
def test_a_playout_cap_must_be_a_probability_and_fewer_iterations(probability, fast):
    with pytest.raises(AssertionError):
        capped(num_iterations=5, full_probability=probability, fast_iterations=fast)
