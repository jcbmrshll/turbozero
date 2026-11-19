"""The Evaluator base class's own logic: moving its state on to the next move, and when
its search is done and trains."""

import jax
import jax.numpy as jnp
import numpy as np

from turbozero.evaluators.evaluator import Evaluator


class Counter(Evaluator):
    """Counts the actions played; a reset zeroes the counts."""

    def init(self, template_embedding):
        return jnp.zeros(3)

    def reset(self, state):
        return jnp.zeros_like(state)

    def step(self, state, action):
        return state.at[action].add(1.0)


def test_next_state_steps_resets_or_keeps_the_state():
    evaluator = Counter(discount=-1.0)
    state = jnp.ones((4, 3))
    action = jnp.array([0, 1, 2, 0])
    reset = jnp.array([False, True, False, True])
    active = jnp.array([True, True, False, False])

    moved = jax.jit(jax.vmap(evaluator.next_state))(state, action, reset, active)

    np.testing.assert_array_equal(moved, [[2, 1, 1], [0, 0, 0], [1, 1, 1], [1, 1, 1]])


def test_a_search_is_done_after_its_cycles_and_trains():
    evaluator = Counter(discount=-1.0)
    state = evaluator.init(None)

    done = [bool(evaluator.search_done(state, jnp.array(c))) for c in range(3)]

    assert done == [False, True, True]
    assert bool(evaluator.trains_on(state))
