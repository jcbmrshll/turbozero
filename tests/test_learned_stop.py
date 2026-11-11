"""The learned stop rule: its labels, its replay and threshold, and its place in a search."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_mcts import LEAD_OF_4, build_tree, capped

from core.evaluators.mcts import learned_stop
from core.evaluators.mcts.learned_stop import LearnedStop
from core.evaluators.mcts.mcts import SmartPruning


@pytest.fixture(scope="module")
def env(fixed_length_env):
    return fixed_length_env(length=10, num_actions=3)


def learned(searches="fast", budget=4.0, interval=2, **kwargs):
    return capped(
        num_iterations=10,
        fast_iterations=7,
        learned_stop=LearnedStop(searches, budget, interval=interval),
        **kwargs,
    )


def with_stopper(search, tree, **fields):
    stopper = learned_stop.init_stopper(search.learned_stop, search.checks())
    return replace(tree, stopper=replace(stopper, **fields))


def test_replay_stops_at_the_first_check_below_the_threshold_or_pruned():
    pred = np.array([[0.5, 0.1, 0.0], [0.5, 0.5, 0.5]])
    pruned = np.array([[False, False, False], [False, True, False]])
    gain = np.array([[0.3, 0.2, 0.0], [0.4, 0.0, 0.0]])
    valid = np.ones_like(pruned)
    end = np.array([8, 8])

    # the first stops at its 2nd check (4 cycles, giving up 0.2), the second where pruned (4)
    assert learned_stop.replay(pred, pruned, gain, valid, 0.2, 2, end) == (4.0, 0.1)
    # without the rule: the first runs to its end
    assert learned_stop.replay(pred, pruned, gain, valid, -np.inf, 2, end) == (6.0, 0.0)


def test_calibrate_finds_the_threshold_for_the_budget():
    rng = np.random.default_rng(0)
    pred = rng.random((200, 8))
    pruned = np.zeros((200, 8), dtype=bool)
    valid = np.ones((200, 8), dtype=bool)
    gain = rng.random((200, 8))
    end = np.full(200, 18)

    threshold = learned_stop.calibrate(pred, pruned, valid, gain, 9.0, 2, end)

    cycles, _ = learned_stop.replay(pred, pruned, gain, valid, threshold, 2, end)
    assert cycles >= 9.0
    # the next prediction up stops sooner than the budget
    above = np.unique(pred)[np.unique(pred) > threshold][0]
    assert learned_stop.replay(pred, pruned, gain, valid, above, 2, end)[0] < 9.0
    # a budget the searches already meet needs no rule
    assert learned_stop.calibrate(pred, pruned, valid, gain, 20.0, 2, end) == -np.inf


def test_root_features_are_finite_and_bounded(env):
    search = learned()
    tree = build_tree(search, env, LEAD_OF_4)

    features = learned_stop.root_features(tree, jnp.array(3), 7, -1.0)

    assert features.shape == (learned_stop.NUM_FEATURES,)
    assert np.isfinite(features).all() and (np.abs(features) <= 4).all()


def test_the_gain_is_the_value_the_end_puts_on_the_move_then_most_visited(env):
    # children along actions 0 and 1, of values (to their player) 0.2 and -0.5
    search = learned()
    tree = build_tree(
        search, env, [(None, None, 9, 0.0), (0, 0, 6, 0.2), (0, 1, 2, -0.5)]
    )

    gain = learned_stop.gains(tree, jnp.array([0, 1]), -1.0)

    # to the root's player 0 is worth -0.2 and 1 is worth 0.5; the end's most visited is 0, so
    # a check whose most visited was 1 would have played the better move (a negative gain)
    np.testing.assert_allclose(gain, [0.0, -0.7], atol=1e-6)


@pytest.mark.parametrize(
    "threshold, label, stops",
    [(-np.inf, False, False), (np.inf, False, True), (np.inf, True, False)],
)
def test_the_rule_stops_below_its_threshold_unless_the_search_labels_it(
    env, threshold, label, stops
):
    search = learned()
    tree = with_stopper(
        search,
        replace(build_tree(search, env, LEAD_OF_4), fast=jnp.array(True)),
        threshold=jnp.array(threshold, dtype=jnp.float32),
        label=jnp.array(label),
        applies=jnp.array(True),
    )

    assert bool(search.search_done(tree, jnp.array(2))) == stops


def test_a_search_labels_the_rule_with_its_probability_where_it_applies(env):
    search = learned("fast-greedy")
    tree = replace(build_tree(search, env, LEAD_OF_4), fast=jnp.array(True))
    tree = with_stopper(search, tree)
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)

    started = jax.vmap(lambda k: search.start_search(k, tree, greedy=True).stopper)(
        keys
    )
    sampled = jax.vmap(lambda k: search.start_search(k, tree, greedy=False).stopper)(
        keys
    )

    assert np.mean(started.label) == pytest.approx(0.1, abs=0.02)
    # the playout cap's draw decides fast, and the rule applies to fast greedy ones
    fast = jax.vmap(lambda k: search.start_search(k, tree).fast)(keys)
    np.testing.assert_array_equal(started.applies, fast)
    assert not np.asarray(sampled.applies).any()


def test_a_labelled_search_records_its_checks_and_writes_them_when_its_move_is_played(
    env,
):
    search = learned(interval=2)
    tree = replace(build_tree(search, env, LEAD_OF_4), fast=jnp.array(True))
    tree = with_stopper(search, tree, label=jnp.array(True), applies=jnp.array(True))

    for cycles in range(1, 9):
        tree = search.record_check(tree, cycles)
    # a fast search of 7 iterations ends after 8 cycles: checks at 2, 4, 6 (and 8, its end)
    np.testing.assert_array_equal(tree.stopper.recorded, [True] * 4)

    played = search.next_state(tree, jnp.array(0), jnp.array(False))
    stopper = played.stopper
    assert int(stopper.ex_count) == 1
    np.testing.assert_array_equal(stopper.ex_valid[0], [True, True, True, False])
    assert int(stopper.ex_end[0]) == 8
    # a search not played writes nothing
    kept = search.next_state(tree, jnp.array(0), jnp.array(False), active=False)
    assert int(kept.stopper.ex_count) == 0


def test_adapt_fits_and_sets_the_threshold_for_the_budget():
    rule = LearnedStop("fast", budget=6.0, interval=2, fit_steps=50, fit_batch=64)
    envs, slots, checks = 4, 32, 4
    stopper = jax.vmap(lambda _: learned_stop.init_stopper(rule, checks))(
        jnp.arange(envs)
    )
    rng = np.random.default_rng(0)
    features = rng.random((envs, slots, checks, learned_stop.NUM_FEATURES)).astype(
        np.float32
    )
    # the gain falls with the first feature: a rule that reads it stops where it's small
    gain = (1 - features[..., 0]) * np.linspace(1, 0, checks)
    stopper = replace(
        stopper,
        ex_features=jnp.asarray(features),
        ex_gain=jnp.asarray(gain, dtype=jnp.float32),
        ex_valid=jnp.ones((envs, slots, checks), dtype=bool),
        ex_end=jnp.full((envs, slots), 10, dtype=jnp.int32),
        ex_count=jnp.full((envs,), slots, dtype=jnp.int32),
    )

    params, threshold, metrics = learned_stop.fit_and_calibrate(
        jax.random.PRNGKey(0), rule, stopper
    )

    assert np.isfinite(threshold)
    assert metrics["learned_stop_cycles"] == pytest.approx(6.0, abs=1.0)
    assert metrics["learned_stop_holdout_loss"] < metrics["learned_stop_gain_variance"]
    assert params["w1"].shape == (learned_stop.NUM_FEATURES, rule.hidden)


def test_without_labelled_searches_adapt_leaves_the_rule_as_it_was():
    rule = LearnedStop("fast", budget=6.0)
    stopper = jax.vmap(lambda _: learned_stop.init_stopper(rule, 4))(jnp.arange(2))

    _, threshold, metrics = learned_stop.fit_and_calibrate(
        jax.random.PRNGKey(0), rule, stopper
    )

    assert threshold == -np.inf
    assert metrics == {"learned_stop_searches": 0}


def test_a_learned_stop_goes_with_smart_pruning(env):
    search = learned(smart_pruning=SmartPruning("fast"))
    tree = replace(build_tree(search, env, LEAD_OF_4), fast=jnp.array(True))
    tree = with_stopper(search, tree, applies=jnp.array(True))

    # the lead of 4 exceeds a fast search's 7 iterations left from cycle 5: STOP stops it
    assert bool(search.search_done(tree, jnp.array(6)))
    assert not bool(search.search_done(tree, jnp.array(2)))


def test_a_learned_stop_needs_a_valid_setting():
    with pytest.raises(ValueError):
        LearnedStop("greedy", 4.0)
    with pytest.raises(ValueError):
        LearnedStop("fast", 0.0)
    with pytest.raises(ValueError):
        LearnedStop("fast", 4.0, label_probability=1.0)
