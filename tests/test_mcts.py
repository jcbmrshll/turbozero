"""Behaviour of the MCTS evaluators on tic-tac-toe, using stub evaluation functions."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import (
    make_nn_eval_fn,
    make_nn_eval_fn_no_params_callable,
)
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.weighted_mcts import WeightedMCTS

AZ_MCTS = AlphaZero(MCTS)
AZ_WEIGHTED = AlphaZero(WeightedMCTS)

# X: 0, 1   O: 4, 8   -> X to move, wins at 2
WIN_MOVES, WINNING_ACTION = [0, 4, 1, 8], 2
# X: 0, 8   O: 4, 2   -> X to move, must block at 6
BLOCK_MOVES, BLOCKING_ACTION = [0, 4, 8, 2], 6
# a few moves in, so some actions are illegal
MIDGAME_MOVES = [4, 0, 8, 2]


ALL_SEARCHES = [
    pytest.param(AZ_MCTS, id="AlphaZero(MCTS)"),
    pytest.param(AZ_WEIGHTED, id="AlphaZero(WeightedMCTS)"),
    pytest.param(MCTS, id="MCTS"),
]


def root(tree):
    return tree.data_at(tree.ROOT_INDEX)


def root_child_visits(tree):
    return tree.get_child_data("n", tree.ROOT_INDEX)


def run_search(search, key, moves, ttt, key_init=None):
    state, meta = ttt.play(moves, key_init)
    return search.evaluate(key, search.init(), state, meta), meta


@pytest.mark.parametrize("cls", ALL_SEARCHES)
def test_visit_counts_add_up(make_search, ttt, cls):
    search = make_search(cls)
    k = search.evaluator.num_iterations
    state, meta = ttt.play(MIDGAME_MOVES)

    out = search.evaluate(jax.random.PRNGKey(0), search.init(), state, meta)
    assert root(out.eval_state).n == 1 + k
    assert root(out.eval_state).n == 1 + root_child_visits(out.eval_state).sum()

    out = search.evaluate(jax.random.PRNGKey(1), out.eval_state, state, meta)
    assert root(out.eval_state).n == 1 + 2 * k
    assert root(out.eval_state).n == 1 + root_child_visits(out.eval_state).sum()


@pytest.mark.parametrize("cls", [
    pytest.param(AZ_MCTS, id="AlphaZero(MCTS)"),
    pytest.param(AZ_WEIGHTED, id="AlphaZero(WeightedMCTS)"),
    pytest.param(MCTS, id="MCTS"),
])
def test_policy_weights_are_normalised_and_legal(make_search, ttt, cls):
    out, meta = run_search(make_search(cls), jax.random.PRNGKey(0), MIDGAME_MOVES, ttt)

    np.testing.assert_allclose(out.policy_weights.sum(), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(out.policy_weights[~meta.action_mask], 0.0)
    assert meta.action_mask[out.action]


@pytest.mark.parametrize("cls", [
    pytest.param(AZ_MCTS, id="AlphaZero(MCTS)"),
    pytest.param(AZ_WEIGHTED, id="AlphaZero(WeightedMCTS)"),
    pytest.param(MCTS, id="MCTS"),
])
def test_no_root_visits_to_illegal_moves(make_search, ttt, cls):
    out, meta = run_search(make_search(cls), jax.random.PRNGKey(0), MIDGAME_MOVES, ttt)

    tree = out.eval_state
    np.testing.assert_array_equal(root_child_visits(tree)[~meta.action_mask], 0)
    np.testing.assert_array_equal(tree.edge_map[tree.ROOT_INDEX][~meta.action_mask], tree.NULL_INDEX)


@pytest.mark.parametrize("cls", ALL_SEARCHES)
def test_no_visits_to_illegal_moves_below_root(make_search, ttt, cls):
    out, _ = run_search(make_search(cls), jax.random.PRNGKey(0), MIDGAME_MOVES, ttt)

    tree = out.eval_state
    # pgx marks every action legal in terminal states, and terminal nodes are never expanded
    legal = np.asarray(tree.data.embedding.legal_action_mask)
    has_child = np.asarray(tree.edge_map != tree.NULL_INDEX)
    below_root = np.arange(tree.capacity) != tree.ROOT_INDEX
    assert not (has_child & ~legal)[below_root].any()


TACTICS_SEARCHES = [
    pytest.param(AZ_MCTS, id="AlphaZero(MCTS)"),
    pytest.param(AZ_WEIGHTED, id="AlphaZero(WeightedMCTS)"),
]


@pytest.mark.parametrize("first_player", [0, 1])
@pytest.mark.parametrize("cls", TACTICS_SEARCHES)
def test_finds_forced_win(make_search, ttt, cls, first_player):
    out, _ = run_search(make_search(cls), jax.random.PRNGKey(0), WIN_MOVES, ttt,
                        key_init=ttt.first_player_keys[first_player])

    assert out.action == WINNING_ACTION
    # a won position is worth more than a draw to the player to move
    assert out.eval_state.data_at(0).q > 0


@pytest.mark.parametrize("first_player", [0, 1])
@pytest.mark.parametrize("cls", [
    pytest.param(AZ_MCTS, id="AlphaZero(MCTS)"),
    pytest.param(AZ_WEIGHTED, id="AlphaZero(WeightedMCTS)"),
])
def test_blocks_forced_loss(make_search, ttt, cls, first_player):
    out, _ = run_search(make_search(cls), jax.random.PRNGKey(0), BLOCK_MOVES, ttt,
                        key_init=ttt.first_player_keys[first_player])

    assert out.action == BLOCKING_ACTION


@pytest.mark.parametrize("cls", ALL_SEARCHES)
def test_tree_reuse_keeps_child_stats(make_search, ttt, cls):
    search = make_search(cls)
    out, _ = run_search(search, jax.random.PRNGKey(0), MIDGAME_MOVES, ttt)
    tree = out.eval_state
    action = out.action
    child = tree.data_at(tree.edge_map[tree.ROOT_INDEX, action])

    new_root = root(search.step(tree, action))

    assert child.n > 1
    np.testing.assert_array_equal(new_root.n, child.n)
    np.testing.assert_array_equal(new_root.q, child.q)
    np.testing.assert_array_equal(new_root.p, child.p)


def test_search_continues_from_reused_tree(make_search, ttt):
    search = make_search(AZ_MCTS)
    state, meta = ttt.play(MIDGAME_MOVES)
    out = search.evaluate(jax.random.PRNGKey(0), search.init(), state, meta)
    tree = out.eval_state
    child_n = tree.data_at(tree.edge_map[tree.ROOT_INDEX, out.action]).n

    tree = search.step(tree, out.action)
    state, meta = ttt.step_fn(state, out.action)
    out = search.evaluate(jax.random.PRNGKey(1), tree, state, meta)

    assert root(out.eval_state).n == child_n + search.evaluator.num_iterations


@pytest.mark.parametrize("cls", ALL_SEARCHES)
def test_tiny_tree_does_not_crash_and_counts_every_iteration(make_search, ttt, cls):
    search = make_search(cls, max_nodes=4)
    out, _ = run_search(search, jax.random.PRNGKey(0), MIDGAME_MOVES, ttt)

    tree = out.eval_state
    assert tree.next_free_idx == tree.capacity
    # every iteration is still counted at the root, even when its leaf could not be stored
    assert root(tree).n == 1 + search.evaluator.num_iterations
    assert jnp.isfinite(tree.data.q).all()


@pytest.mark.parametrize("cls", ALL_SEARCHES)
def test_tiny_tree_still_backs_up_values(make_search, ttt, cls):
    search = make_search(cls, max_nodes=4)
    out, _ = run_search(search, jax.random.PRNGKey(0), WIN_MOVES, ttt)

    # the winning move is the first one PUCT expands, and its value reaches the root
    assert root(out.eval_state).q > 0
    assert out.action == WINNING_ACTION


def test_temperature_zero_picks_most_visited_move(make_search, ttt):
    search = make_search(AZ_MCTS, temperature=0.0)
    state, meta = ttt.play(MIDGAME_MOVES)
    keys = jax.random.split(jax.random.PRNGKey(0), 8)

    out = jax.vmap(search.evaluate, in_axes=(0, None, None, None))(keys, search.init(), state, meta)

    for action, weights in zip(out.action, out.policy_weights):
        assert weights[action] == weights.max()


def test_dirichlet_noise_keeps_prior_normalised_and_masked(make_search, ttt):
    search = make_search(AZ_MCTS, dirichlet_epsilon=0.5)
    state, meta = ttt.play(MIDGAME_MOVES)
    keys = jax.random.split(jax.random.PRNGKey(0), 8)

    out = jax.vmap(search.evaluate, in_axes=(0, None, None, None))(keys, search.init(), state, meta)

    priors = out.eval_state.data.p[:, 0]
    np.testing.assert_allclose(priors.sum(axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(priors[:, ~meta.action_mask], 0.0)
    # the noise actually changes the prior between keys
    assert not np.allclose(priors[0], priors[1])


# the prior MCTS stores is softmax(masked logits) of the network output

LOGITS = jnp.array([2.0, -1.0, 0.5, 0.0, 3.0, -2.0, 1.0, 0.25, -0.5])
VALUE = 0.1


class FixedLogitsNet(eqx.Module):
    """Parameter-free network that outputs LOGITS and VALUE for every input."""

    def __call__(self, x):  # pylint: disable=unused-argument
        return LOGITS, jnp.array([VALUE])


def fixed_logits(x):
    return LOGITS[None], jnp.array([VALUE])


# module-level so each eval fn is a single object, which lets make_search reuse compiled searches
EVAL_FNS = {
    "make_nn_eval_fn": make_nn_eval_fn(FixedLogitsNet(), lambda s: s.observation),
    "make_nn_eval_fn_no_params_callable": make_nn_eval_fn_no_params_callable(fixed_logits, lambda s: s.observation),
}


def masked_softmax(logits, mask):
    return jax.nn.softmax(jnp.where(mask, logits, -jnp.inf))


@pytest.mark.parametrize("eval_fn_name", EVAL_FNS)
@pytest.mark.parametrize("cls", [pytest.param(AZ_MCTS, id="AlphaZero(MCTS)"), pytest.param(MCTS, id="MCTS")])
def test_prior_is_softmax_of_masked_logits(make_search, ttt, cls, eval_fn_name):
    # the search runs from the initial position, where every move is legal, so plain MCTS's
    # missing root mask (#4) does not affect this test; children are masked
    no_noise = {"dirichlet_epsilon": 0.0} if cls is AZ_MCTS else {}
    search = make_search(cls, eval_fn=EVAL_FNS[eval_fn_name], **no_noise)
    state, meta = ttt.play([])
    # (nn_params, nn_state) for make_nn_eval_fn, ignored by the other eval fn
    params = (eqx.filter(FixedLogitsNet(), eqx.is_inexact_array), None)
    out = search.evaluate(jax.random.PRNGKey(0), search.init(), state, meta, params)
    tree = out.eval_state
    assert meta.action_mask.all()

    np.testing.assert_allclose(root(tree).p, jax.nn.softmax(LOGITS), atol=1e-6)
    num_nodes = int(tree.next_free_idx)
    for idx in range(1, num_nodes):
        node = tree.data_at(idx)
        if not node.terminated:
            np.testing.assert_allclose(node.p, masked_softmax(LOGITS, node.embedding.legal_action_mask), atol=1e-6)


class KeyRecordingMCTS(MCTS):
    """MCTS whose root update, iterations and action sampling only record the key each one receives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded = []

    def record(self, name, key):
        jax.debug.callback(lambda k: self.recorded.append((name, np.asarray(k))), key)

    def update_root(self, key, tree, root_embedding, params, root_metadata, **kwargs):
        self.record("update_root", key)
        return tree

    def iterate(self, key, tree, params, env_step_fn):
        self.record("iterate", key)
        return tree

    def sample_root_action(self, key, tree):
        self.record("sample_root_action", key)
        return jnp.array(0), jnp.zeros((self.branching_factor,))


def test_evaluate_gives_every_consumer_an_independent_key(ttt):
    search = KeyRecordingMCTS(eval_fn=lambda *_: (jnp.zeros(ttt.num_actions), 0.0), action_selector=PUCTSelector(),
                              branching_factor=ttt.num_actions, max_nodes=8, num_iterations=4)
    state, meta = ttt.play([])
    tree = search.init(template_embedding=state)
    key = jax.random.PRNGKey(0)

    search.evaluate(key, tree, state, meta, params=None, env_step_fn=ttt.step_fn)
    jax.effects_barrier()

    names = [name for name, _ in search.recorded]
    assert sorted(names) == sorted(["update_root", "sample_root_action"] + ["iterate"] * search.num_iterations)
    keys = [tuple(k.tolist()) for _, k in search.recorded]
    assert tuple(np.asarray(key).tolist()) not in keys
    assert len(set(keys)) == len(keys)
    # no consumer's key is a split of another consumer's key
    # (the search used to split the action-sampling key into the iteration keys)
    for _, k in search.recorded:
        children = {tuple(c.tolist()) for n in range(2, search.num_iterations + 1)
                    for c in np.asarray(jax.random.split(k, n))}
        assert not children & set(keys)
