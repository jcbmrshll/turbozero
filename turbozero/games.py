from typing import Any

import jax
import jax.numpy as jnp

from turbozero.evaluators.evaluator import Evaluator
from turbozero.pytree_utils import where_tree
from turbozero.types import EnvInitFn, EnvStepFn


def two_player_game(
    key: jax.Array,
    evaluator_1: Evaluator,
    evaluator_2: Evaluator,
    params_1: Any,
    params_2: Any,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int,
    p1_first: bool | None = None,
) -> jax.Array:
    """Plays a game between two evaluators and returns the outcomes (final
    rewards) of `evaluator_1` and `evaluator_2`, whichever seat each played.

    A game still going after `max_steps` steps is truncated: it's scored with
    the rewards of its last step (0 in games that only reward their end).
    `p1_first` is whether `evaluator_1` moves first, at random if None. Under
    vmap, a random first player makes whose turn it is differ across games, so
    every step runs both evaluators and keeps one move from each game; with a
    fixed `p1_first`, only the evaluator to move runs.
    """
    env_key, turn_key, key = jax.random.split(key, 3)
    env_state, metadata = env_init_fn(env_key)
    evaluators = (evaluator_1, evaluator_2)
    params = (params_1, params_2)
    eval_states = tuple(evaluator.init(env_state) for evaluator in evaluators)
    p1_moves_first = p1_first
    if p1_moves_first is None:
        p1_moves_first = jax.random.randint(turn_key, (), 0, 2) == 0
    p1_id = jnp.where(
        p1_moves_first, metadata.cur_player_id, 1 - metadata.cur_player_id
    )

    def move(state: tuple, p: int) -> tuple:
        """Has evaluator `p` (0 or 1) choose and play a move; both evaluators
        then step with it, e.g. to reuse their search trees."""
        key, env_state, metadata, eval_states, outcomes, _ = state
        step_key, key = jax.random.split(key)
        _, evaluate_key = jax.random.split(step_key)
        output = evaluators[p].evaluate(
            evaluate_key, eval_states[p], env_state, metadata, params[p], env_step_fn
        )
        env_state, metadata = env_step_fn(env_state, output.action)
        eval_states = list(eval_states)
        eval_states[p] = output.eval_state
        eval_states = tuple(
            evaluator.step(eval_state, output.action)
            for evaluator, eval_state in zip(evaluators, eval_states)
        )
        # `step` counts the steps taken so far
        ended = metadata.terminated | (metadata.step >= max_steps)
        outcomes = jnp.where(ended, metadata.rewards, outcomes)
        return key, env_state, metadata, eval_states, outcomes, ended

    def turn(state: tuple, step: jax.Array) -> tuple[tuple, None]:
        p1_moves = (step % 2 == 0) == p1_moves_first
        # Branches whose predicate can differ across a vmapped batch of games
        # are written as selects rather than lax.cond. vmap turns such a cond
        # into a select anyway, running both branches, but it does so by
        # batching every operand of the branches, the evaluators' parameters
        # included: their convolutions then run per game, which has been seen
        # to run about an order of magnitude slower.
        if isinstance(p1_moves_first, bool):
            moved = jax.lax.cond(
                p1_moves, lambda s: move(s, 0), lambda s: move(s, 1), state
            )
        else:
            moved = where_tree(p1_moves, move(state, 0), move(state, 1))
        # a completed game keeps its final state
        return where_tree(state[-1], state, moved), None

    completed = jnp.zeros((), dtype=jnp.bool_)
    outcomes = jnp.zeros((2,), dtype=jnp.float32)
    state = (key, env_state, metadata, eval_states, outcomes, completed)
    state, _ = jax.lax.scan(turn, state, jnp.arange(max_steps))
    outcomes = state[4]
    return jnp.array([outcomes[p1_id], outcomes[1 - p1_id]])
