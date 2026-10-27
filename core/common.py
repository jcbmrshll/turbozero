from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from core.evaluators.evaluator import EvalOutput, Evaluator
from core.types import EnvInitFn, EnvStepFn, StepMetadata

# (key, evaluator output, metadata of the state being evaluated) -> action to play
ChooseActionFn = Callable[[jax.Array, EvalOutput, StepMetadata], jax.Array]


def step_env_and_evaluator(
    key: jax.Array,
    env_state: Any,
    env_state_metadata: StepMetadata,
    eval_state: Any,
    params: Any,
    evaluator: Evaluator,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int,
    reset: bool = True,
    choose_action: ChooseActionFn | None = None,
) -> tuple[EvalOutput, Any, StepMetadata, jax.Array, jax.Array, jax.Array]:
    """Steps the environment and evaluator.

    - Evaluates the environment state with the Evaluator and selects an action.
    - Optionally replaces the selected action with one chosen by `choose_action`.
    - Performs a step in the environment with the selected action.
    - Updates the internal state of the Evaluator.
    - Optionally resets the environment and evaluator state if the episode is terminated or truncated.

    Args:
        key: rng
        env_state: The environment state to evaluate.
        env_state_metadata: Metadata associated with the environment state.
        eval_state: The internal state of the Evaluator.
        params: nn parameters used by the Evaluator.
        evaluator: The Evaluator.
        env_step_fn: The environment step function.
        env_init_fn: The environment initialization function.
        max_steps: The maximum number of environment steps per episode. An episode that has taken
            `max_steps` steps without terminating is truncated, so no episode runs longer than `max_steps` steps.
        reset: Whether to reset the environment and evaluator state if the episode is terminated or truncated.
        choose_action: (optional) chooses the action to play from the evaluator's output instead,
            e.g. `core.training.exploration.Exploration.choose_action`. The environment and
            evaluator step with the action it returns; the policy weights are left as they are.

    Returns:
        Tuple[EvalOutput, Any, StepMetadata, jax.Array, jax.Array, jax.Array]:
            - `output`: The output of the evaluation, with the action that was played.
            - `env_state`: The updated environment state.
            - `env_state_metadata`: Metadata associated with the updated environment state.
            - `terminated`: Whether the episode is terminated.
            - `truncated`: Whether the episode reached `max_steps` steps without terminating.
              Never true together with `terminated`: an episode that terminates on its last allowed step is terminated.
            - `rewards`: Rewards emitted by the environment.
    """
    return search_and_step(
        key,
        env_state,
        env_state_metadata,
        eval_state,
        params,
        evaluator,
        env_step_fn,
        env_init_fn,
        max_steps,
        reset,
        choose_action,
    )[:6]


def search_and_step(
    key: jax.Array,
    env_state: Any,
    env_state_metadata: StepMetadata,
    eval_state: Any,
    params: Any,
    evaluator: Evaluator,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int,
    reset: bool = True,
    choose_action: ChooseActionFn | None = None,
) -> tuple[EvalOutput, Any, StepMetadata, jax.Array, jax.Array, jax.Array, Any]:
    """`step_env_and_evaluator`, also returning the evaluator's state as the evaluation left it, before
    the evaluator stepped: e.g. the whole search tree, and its root value (`Evaluator.get_value`).

    Args: see `step_env_and_evaluator`.

    Returns:
        Tuple[EvalOutput, Any, StepMetadata, jax.Array, jax.Array, jax.Array, Any]:
            what `step_env_and_evaluator` returns, then the evaluator's state after the evaluation
    """
    key, evaluate_key = jax.random.split(key)
    # evaluate the environment state
    output = evaluator.evaluate(
        key=evaluate_key,
        eval_state=eval_state,
        env_state=env_state,
        root_metadata=env_state_metadata,
        params=params,
        env_step_fn=env_step_fn,
    )
    return step_with_output(
        key,
        output,
        env_state,
        env_state_metadata,
        evaluator,
        env_step_fn,
        env_init_fn,
        max_steps,
        reset,
        choose_action,
    )


def step_with_output(
    key: jax.Array,
    output: EvalOutput,
    env_state: Any,
    env_state_metadata: StepMetadata,
    evaluator: Evaluator,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int,
    reset: bool = True,
    choose_action: ChooseActionFn | None = None,
    active: ArrayLike = True,
) -> tuple[EvalOutput, Any, StepMetadata, jax.Array, jax.Array, jax.Array, Any]:
    """`search_and_step` after its evaluation: steps the environment and evaluator with the
    evaluator's `output` for `env_state`.

    Args:
        key: rng, `search_and_step`'s after it splits off the evaluation's
        output: the evaluator's output
        active: (optional) if not, the evaluator's state stays as `output` left it (see
            `Evaluator.next_state`); the environment's is stepped all the same, for the caller
            to keep or not
        others: see `step_env_and_evaluator`.

    Returns:
        Tuple[EvalOutput, Any, StepMetadata, jax.Array, jax.Array, jax.Array, Any]: see
            `search_and_step`
    """
    if choose_action is not None:
        choose_key, key = jax.random.split(key)
        output = replace(
            output, action=choose_action(choose_key, output, env_state_metadata)
        )
    # take the selected action
    env_state, env_state_metadata = env_step_fn(env_state, output.action)
    # check for termination and truncation
    terminated = env_state_metadata.terminated
    # `step` counts the steps taken so far, so this is the episode's `max_steps`-th step
    truncated = ~terminated & (env_state_metadata.step >= max_steps)
    # reset the environment and evaluator state if the episode is terminated or truncated
    # else, update the evaluator state
    rewards = env_state_metadata.rewards
    if reset:
        eval_state = evaluator.next_state(
            output.eval_state, output.action, terminated | truncated, active=active
        )
    else:
        eval_state = evaluator.next_state(
            output.eval_state,
            output.action,
            False,
            active=active & ~(terminated | truncated),
        )
    # reset the environment if the episode is terminated or truncated
    env_state, env_state_metadata = jax.lax.cond(
        terminated | truncated,
        lambda _: env_init_fn(key) if reset else (env_state, env_state_metadata),
        lambda _: (env_state, env_state_metadata),
        None,
    )
    searched_eval_state = output.eval_state
    output = replace(output, eval_state=eval_state)
    return (
        output,
        env_state,
        env_state_metadata,
        terminated,
        truncated,
        rewards,
        searched_eval_state,
    )


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
        p1_value_estimate: The current state value estimate of the first evaluator.
        p2_value_estimate: The current state value estimate of the second evaluator.
        outcomes: The outcomes of the game (final rewards) for each player.
        completed: Whether the game is completed.
        p1_search: What the first evaluator made of the last position played from, see `GameFrame`.
        p2_search: The same for the second evaluator.
    """

    key: jax.Array
    env_state: Any
    env_state_metadata: StepMetadata
    p1_eval_state: Any
    p2_eval_state: Any
    p1_value_estimate: jax.Array
    p2_value_estimate: jax.Array
    outcomes: jax.Array
    completed: jax.Array
    p1_search: Any = None
    p2_search: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GameFrame:
    """Stores information necessary for rendering the environment state in a two-player game.

    Attributes:
        env_state: The environment state.
        p1_value_estimate: The current state value estimate of the first evaluator.
        p2_value_estimate: The current state value estimate of the second evaluator.
        completed: Whether the game is completed.
        outcomes: The outcomes of the game (final rewards) for each player.
        p1_search: What the first evaluator made of the position the last move was played from
            (`Evaluator.search_record`), None unless the game records searches: its search, if it
            played the move, else the tree it had for the position from its own previous search.
        p2_search: The same for the second evaluator.
    """

    env_state: Any
    p1_value_estimate: jax.Array
    p2_value_estimate: jax.Array
    completed: jax.Array
    outcomes: jax.Array
    p1_search: Any = None
    p2_search: Any = None


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
    record_search: bool = False,
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
        max_steps: The maximum number of steps per episode (see `step_env_and_evaluator`).
        record_search: Whether to keep a record of what the active evaluator's search explored
            (`Evaluator.search_record`).

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

    # step
    step_key, key = jax.random.split(state.key)
    output, env_state, env_state_metadata, terminated, truncated, rewards, searched = (
        search_and_step(
            key=step_key,
            env_state=state.env_state,
            env_state_metadata=state.env_state_metadata,
            eval_state=active_eval_state,
            params=params,
            evaluator=active_evaluator,
            env_step_fn=env_step_fn,
            env_init_fn=env_init_fn,
            max_steps=max_steps,
            reset=False,
        )
    )

    active_eval_state = output.eval_state
    active_value_estimate = active_evaluator.get_value(active_eval_state)
    active_value_estimate = jax.lax.cond(
        terminated | truncated,
        lambda a: a,
        lambda a: active_evaluator.discount * a,
        active_value_estimate,
    )
    # update the other evaluator
    other_eval_state = other_evaluator.step(other_eval_state, output.action)
    other_value_estimate = other_evaluator.get_value(other_eval_state)
    # what each player made of the position: the active one's search, and what the other had
    # explored of it before (searching its own last move, if it keeps its tree)
    p1_search, p2_search = state.p1_search, state.p2_search
    if record_search:
        active_search = active_evaluator.search_record(searched)
        other_search = other_evaluator.search_record(
            state.p2_eval_state if use_p1 else state.p1_eval_state
        )
        p1_search, p2_search = (
            (active_search, other_search) if use_p1 else (other_search, active_search)
        )
    # update the game state
    if use_p1:
        p1_eval_state, p2_eval_state = active_eval_state, other_eval_state
        p1_value_estimate, p2_value_estimate = (
            active_value_estimate,
            other_value_estimate,
        )
    else:
        p1_eval_state, p2_eval_state = other_eval_state, active_eval_state
        p1_value_estimate, p2_value_estimate = (
            other_value_estimate,
            active_value_estimate,
        )
    return replace(
        state,
        key=key,
        env_state=env_state,
        env_state_metadata=env_state_metadata,
        p1_eval_state=p1_eval_state,
        p2_eval_state=p2_eval_state,
        p1_value_estimate=p1_value_estimate,
        p2_value_estimate=p2_value_estimate,
        p1_search=p1_search,
        p2_search=p2_search,
        outcomes=jnp.where(
            ((terminated | truncated) & ~state.completed)[..., None],
            rewards,
            state.outcomes,
        ),
        completed=state.completed | terminated | truncated,
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
    record_search: bool = False,
) -> tuple[jax.Array, GameFrame, jax.Array]:
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
        record_search: Whether the frames record what each move's search explored (`GameFrame.p1_search`,
            `p2_search`), for the monitor to show. Each frame then holds a record of a whole search
            tree from each player, so leave it off unless something reads them.

    Returns:
        Tuple[jax.Array, GameFrame, jax.Array]:
            - `outcomes`: The outcomes of the game (final rewards) for each player.
            - `frames`: Frames collected from the game (used for rendering), `max_steps + 1` of them:
              the initial state, then the state after each step.
            - `p_ids`: The player ids of the two evaluators. [evaluator_1_id, evaluator_2_id]
    """
    # init rng
    env_key, turn_key, key = jax.random.split(key, 3)
    # init env state
    env_state, metadata = env_init_fn(env_key)
    # init evaluator states
    p1_eval_state = evaluator_1.init(template_embedding=env_state)
    p2_eval_state = evaluator_2.init(template_embedding=env_state)
    # a record of a search has the same shape before the first search as after it
    p1_search = evaluator_1.search_record(p1_eval_state) if record_search else None
    p2_search = evaluator_2.search_record(p2_eval_state) if record_search else None
    # compile step functions
    game_step = partial(
        two_player_game_step,
        p1_evaluator=evaluator_1,
        p2_evaluator=evaluator_2,
        env_step_fn=env_step_fn,
        env_init_fn=env_init_fn,
        max_steps=max_steps,
        record_search=record_search,
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
        p1_value_estimate=jnp.array(0.0, dtype=jnp.float32),
        p2_value_estimate=jnp.array(0.0, dtype=jnp.float32),
        outcomes=jnp.zeros((2,), dtype=jnp.float32),
        completed=jnp.zeros((), dtype=jnp.bool_),
        p1_search=p1_search,
        p2_search=p2_search,
    )
    # make initial render frame
    initial_game_frame = GameFrame(
        env_state=state.env_state,
        p1_value_estimate=state.p1_value_estimate,
        p2_value_estimate=state.p2_value_estimate,
        completed=state.completed,
        outcomes=state.outcomes,
        p1_search=state.p1_search,
        p2_search=state.p2_search,
    )

    # takes a turn for the active player
    def take_turn(
        state: TwoPlayerGameState, step_num: jax.Array
    ) -> tuple[TwoPlayerGameState, GameFrame]:
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
        # collect render frame
        frame = GameFrame(
            env_state=state.env_state,
            p1_value_estimate=state.p1_value_estimate,
            p2_value_estimate=state.p2_value_estimate,
            completed=state.completed,
            outcomes=state.outcomes,
            p1_search=state.p1_search,
            p2_search=state.p2_search,
        )
        # return game state and render frame
        return state, frame

    # play the game
    state, frames = jax.lax.scan(take_turn, state, xs=jnp.arange(max_steps))
    # append initial state to front of frames
    frames = jax.tree.map(
        lambda i, x: jnp.concatenate([jnp.expand_dims(i, 0), x]),
        initial_game_frame,
        frames,
    )
    # return outcome, frames, player ids
    return (
        jnp.array([state.outcomes[p1_id], state.outcomes[p2_id]]),
        frames,
        jnp.array([p1_id, p2_id]),
    )
