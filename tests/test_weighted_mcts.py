"""WeightedMCTS backup: a node's value is a softmax-weighted average of its children's values,
sharpened by `q_temperature`, mixed with the network's raw value for the node."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.state import MCTSTree
from core.evaluators.mcts.weighted_mcts import WeightedMCTS
from core.trees.tree import init_tree


def weighted_mcts(q_temperature):
    return WeightedMCTS(
        q_temperature=q_temperature,
        eval_fn=lambda *_: (jnp.zeros(3), 0.0),
        action_selector=PUCTSelector(),
        branching_factor=3,
        max_nodes=8,
        num_iterations=1,
    )


def root_with_children(root_q, root_r, child_values):
    """A root with one visit per child. `child_values` are from the root player's perspective,
    so children store their negation (the value for the player to move at the child)."""

    def node(q, r, n):
        new = WeightedMCTS.new_node(
            policy=jnp.full((3,), 1 / 3),
            value=q,
            embedding=jnp.zeros(()),
            terminated=False,
        )
        return replace(
            new, r=jnp.array(r, dtype=jnp.float32), n=jnp.array(n, dtype=jnp.int32)
        )

    tree: MCTSTree = init_tree(8, 3, node(0.0, 0.0, 0))
    tree = tree.set_root(node(root_q, root_r, 1 + len(child_values)))
    for action, value in enumerate(child_values):
        tree = tree.add_node(tree.ROOT_INDEX, action, node(-value, -value, 1))
    return tree


def backed_up_root_q(q_temperature, tree):
    tree = weighted_mcts(q_temperature).backpropagate(
        jax.random.PRNGKey(0), tree, tree.ROOT_INDEX, 0.0
    )
    return float(tree.data_at(tree.ROOT_INDEX).q)


def test_lower_q_temperature_weights_the_best_child_more():
    # child values already span [0, 1], so normalising them changes nothing and only the
    # temperature differs between the runs
    tree = root_with_children(root_q=0.5, root_r=0.0, child_values=[0.0, 1.0, 0.5])

    hot, neutral, cold = (backed_up_root_q(t, tree) for t in (4.0, 1.0, 0.25))

    assert hot < neutral < cold


@pytest.mark.parametrize("q_temperature", [0.5, 1.0])
def test_backed_up_value_stays_on_the_network_value_scale(q_temperature):
    # every child is losing for the root player, and so is the network's own value for the root
    child_values = [-0.8, -0.6]
    tree = root_with_children(root_q=-0.7, root_r=-0.7, child_values=child_values)

    root_q = backed_up_root_q(q_temperature, tree)

    assert min(child_values) <= root_q <= max(child_values)


@pytest.mark.parametrize(
    "child_values, root_value, expected_weighted",
    [
        ([-0.8, -0.6], -0.7, -0.6),
        # all visited children tie: the unvisited third child must never be picked
        ([-0.5, -0.5], -0.5, -0.5),
    ],
)
def test_zero_q_temperature_backs_up_the_best_visited_child(
    child_values, root_value, expected_weighted
):
    tree = root_with_children(
        root_q=root_value, root_r=root_value, child_values=child_values
    )
    n = 1 + len(child_values)

    root_q = backed_up_root_q(0.0, tree)

    assert root_q == pytest.approx((expected_weighted * n + root_value) / (n + 1))


def test_zero_q_temperature_draws_independent_tiebreak_noise_at_each_node(monkeypatch):
    tree = root_with_children(root_q=0.0, root_r=0.0, child_values=[0.0, 0.0])
    recorded = []
    uniform = jax.random.uniform

    def recording_uniform(key, *args, **kwargs):
        jax.debug.callback(
            lambda k: recorded.append(tuple(np.asarray(k).tolist())), key
        )
        return uniform(key, *args, **kwargs)

    monkeypatch.setattr(jax.random, "uniform", recording_uniform)
    # backpropagate from a child, so the loop visits the child and then the root
    weighted_mcts(0.0).backpropagate(jax.random.PRNGKey(0), tree, 1, 0.0)
    jax.effects_barrier()

    assert len(recorded) == 2
    assert recorded[0] != recorded[1]
