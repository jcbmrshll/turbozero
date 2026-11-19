from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from turbozero.evaluators.evaluator import Evaluator
from turbozero.types import EnvInitFn, EnvStepFn, StepMetadata


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TwoPlayerGameState:
    """Stores the state of a two player game using two different evaluators.

    Attributes:
        key: rng
        env_state: The environment state.
        env_state_metadata: Metadata associated with the environment state.
        p1_eval_state: The internal state of the first evaluator.
        p2_eval_state: The internal state of the second evaluator.
        outcomes: The outcomes of the game (final rewards) for each player.
        completed: Whether the game is completed.
    """

    key: jax.Array
    env_state: Any
    env_state_metadata: StepMetadata
    p1_eval_state: Any
    p2_eval_state: Any
    outcomes: jax.Array
    completed: jax.Array


def _select(pred: jax.Array, on_true: Any, on_false: Any) -> Any:
    """Picks `on_true` or `on_false` leaf by leaf, by a scalar predicate."""
    return jax.tree.map(lambda t, f: jnp.where(pred, t, f), on_true, on_false)


def two_player_game_step(
    state: TwoPlayerGameState,
    p1_evaluator: Evaluator,
    p2_evaluator: Evaluator,
    params: Any,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    use_p1: bool,
    max_steps: int,
) -> TwoPlayerGameState:
    """Make a single step in a two player game.

    Args:
        state: The current game state.
        p1_evaluator: The first evaluator.
        p2_evaluator: The second evaluator.
        params: The parameters of the active evaluator.
        env_step_fn: The environment step function.
        env_init_fn: The environment initialization function.
        use_p1: Whether to use the first evaluator.
        max_steps: The maximum number of steps per episode (see `two_player_game`).

    Returns:
        TwoPlayerGameState: The updated game state.
    """
    # determine which evaluator to use based on the current player
    if use_p1:
        active_evaluator = p1_evaluator
        other_evaluator = p2_evaluator
        active_eval_state = state.p1_eval_state
        other_eval_state = state.p2_eval_state
    else:
        active_evaluator = p2_evaluator
        other_evaluator = p1_evaluator
        active_eval_state = state.p2_eval_state
        other_eval_state = state.p1_eval_state

    step_key, key = jax.random.split(state.key)
    _, evaluate_key = jax.random.split(step_key)
    output = active_evaluator.evaluate(
        evaluate_key,
        active_eval_state,
        state.env_state,
        state.env_state_metadata,
        params,
        env_step_fn,
    )
    env_state, env_state_metadata = env_step_fn(state.env_state, output.action)
    # `step` counts the steps taken so far: a game still going after its
    # `max_steps`-th step is truncated
    ended = env_state_metadata.terminated | (env_state_metadata.step >= max_steps)
    # the evaluator follows the move played, e.g. to its search tree's subtree
    active_eval_state = active_evaluator.step(output.eval_state, output.action)
    # update the other evaluator
    other_eval_state = other_evaluator.step(other_eval_state, output.action)
    # update the game state
    if use_p1:
        p1_eval_state, p2_eval_state = active_eval_state, other_eval_state
    else:
        p1_eval_state, p2_eval_state = other_eval_state, active_eval_state
    return replace(
        state,
        key=key,
        env_state=env_state,
        env_state_metadata=env_state_metadata,
        p1_eval_state=p1_eval_state,
        p2_eval_state=p2_eval_state,
        outcomes=jnp.where(
            (ended & ~state.completed)[..., None],
            env_state_metadata.rewards,
            state.outcomes,
        ),
        completed=state.completed | ended,
    )


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
    """Play a two player game between two evaluators.

    Args:
        key: rng
        evaluator_1: The first evaluator.
        evaluator_2: The second evaluator.
        params_1: The parameters of the first evaluator.
        params_2: The parameters of the second evaluator.
        env_step_fn: The environment step function.
        env_init_fn: The environment initialization function.
        max_steps: The maximum number of steps per episode. A game still in progress after `max_steps` steps
            is truncated: it is marked completed and scored with the rewards of its last step (0 in games that only
            reward their end).
        p1_first: (optional) whether the first evaluator moves first. Chosen at random from `key` if not given.
            - Under `vmap`, a random first player makes whose turn it is differ across games, so every step runs
              both evaluators and keeps one move from each game. Pass it as a fixed value for a batch of games
              to run only the active evaluator at each step.

    Returns:
        jax.Array: The outcomes of the game (final rewards) of the first and second evaluator.
    """
    # init rng
    env_key, turn_key, key = jax.random.split(key, 3)
    # init env state
    env_state, metadata = env_init_fn(env_key)
    # init evaluator states
    p1_eval_state = evaluator_1.init(template_embedding=env_state)
    p2_eval_state = evaluator_2.init(template_embedding=env_state)
    # compile step functions
    game_step = partial(
        two_player_game_step,
        p1_evaluator=evaluator_1,
        p2_evaluator=evaluator_2,
        env_step_fn=env_step_fn,
        env_init_fn=env_init_fn,
        max_steps=max_steps,
    )
    step_p1 = partial(game_step, params=params_1, use_p1=True)
    step_p2 = partial(game_step, params=params_2, use_p1=False)

    # determine who goes first
    p1_moves_first: bool | jax.Array = (
        jax.random.randint(turn_key, (), 0, 2) == 0 if p1_first is None else p1_first
    )
    p1_id, p2_id = jax.lax.cond(
        p1_moves_first,
        lambda _: (metadata.cur_player_id, 1 - metadata.cur_player_id),
        lambda _: (1 - metadata.cur_player_id, metadata.cur_player_id),
        None,
    )
    # init game state
    state = TwoPlayerGameState(
        key=key,
        env_state=env_state,
        env_state_metadata=metadata,
        p1_eval_state=p1_eval_state,
        p2_eval_state=p2_eval_state,
        outcomes=jnp.zeros((2,), dtype=jnp.float32),
        completed=jnp.zeros((), dtype=jnp.bool_),
    )

    # takes a turn for the active player
    def take_turn(
        state: TwoPlayerGameState, step_num: jax.Array
    ) -> tuple[TwoPlayerGameState, None]:
        # players alternate turns, starting with the first player
        use_p1 = (step_num % 2 == 0) == p1_moves_first
        # Branches whose predicate can differ across a vmapped batch of games are written as selects
        # rather than lax.cond. vmap turns such a cond into a select anyway, running both branches,
        # but it does so by batching every operand of the branches, the evaluators' parameters
        # included: their convolutions then run per game, an order of magnitude slower.
        if isinstance(p1_moves_first, bool):
            # the same for every game, so only the active player's evaluator runs
            new_state = jax.lax.cond(use_p1, step_p1, step_p2, state)
        else:
            new_state = _select(use_p1, step_p1(state), step_p2(state))
        # a completed game keeps its final state
        state = _select(state.completed, state, new_state)
        return state, None

    # play the game
    state, _ = jax.lax.scan(take_turn, state, xs=jnp.arange(max_steps))
    return jnp.array([state.outcomes[p1_id], state.outcomes[p2_id]])
