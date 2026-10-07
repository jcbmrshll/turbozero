
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.types import EnvInitFn, EnvStepFn, StepMetadata

def partition(
    data: Any,
    num_partitions: int
) -> Any:
    """Partition each array in a data structure into num_partitions along the first axis.
    e.g. partitions an array of shape (N, ...) into (num_partitions, N//num_partitions, ...)

    Args:
    - `data`: ArrayTree to partition
    - `num_partitions`: number of partitions

    Returns:
    - (pytree): partitioned pytree
    """
    return jax.tree.map(
        lambda x: x.reshape(num_partitions, x.shape[0] // num_partitions, *x.shape[1:]),
        data
    )


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
    reset: bool = True
) -> Tuple[EvalOutput, Any,  StepMetadata, bool, bool, jax.Array]:
    """
    - Evaluates the environment state with the Evaluator and selects an action.
    - Performs a step in the environment with the selected action.
    - Updates the internal state of the Evaluator.
    - Optionally resets the environment and evaluator state if the episode is terminated or truncated.

    Args:
    - `key`: rng
    - `env_state`: The environment state to evaluate.
    - `env_state_metadata`: Metadata associated with the environment state.
    - `eval_state`: The internal state of the Evaluator.
    - `params`: nn parameters used by the Evaluator.
    - `evaluator`: The Evaluator.
    - `env_step_fn`: The environment step function.
    - `env_init_fn`: The environment initialization function.
    - `max_steps`: The maximum number of environment steps per episode.
    - `reset`: Whether to reset the environment and evaluator state if the episode is terminated or truncated.

    Returns:
    - (EvalOutput, Any, StepMetadata, bool, bool, jax.Array)
        - `output`: The output of the evaluation.
        - `env_state`: The updated environment state.
        - `env_state_metadata`: Metadata associated with the updated environment state.
        - `terminated`: Whether the episode is terminated.
        - `truncated`: Whether the episode is truncated.
        - `rewards`: Rewards emitted by the environment.
    """
    key, evaluate_key = jax.random.split(key)
    # evaluate the environment state
    output = evaluator.evaluate(
        key=evaluate_key,
        eval_state=eval_state,
        env_state=env_state,
        root_metadata=env_state_metadata,
        params=params,
        env_step_fn=env_step_fn
    )
    # take the selected action
    env_state, env_state_metadata = env_step_fn(env_state, output.action)
    # check for termination and truncation
    terminated = env_state_metadata.terminated
    truncated = env_state_metadata.step > max_steps 
    # reset the environment and evaluator state if the episode is terminated or truncated
    # else, update the evaluator state
    rewards = env_state_metadata.rewards
    eval_state = jax.lax.cond(
        terminated | truncated,
        evaluator.reset if reset else lambda s: s,
        lambda s: evaluator.step(s, output.action),
        output.eval_state
    )
    # reset the environment if the episode is terminated or truncated
    env_state, env_state_metadata = jax.lax.cond(
        terminated | truncated,
        lambda _: env_init_fn(key) if reset else (env_state, env_state_metadata),
        lambda _: (env_state, env_state_metadata),
        None
    )
    output = replace(output, eval_state=eval_state)
    return output, env_state, env_state_metadata, terminated, truncated, rewards


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TwoPlayerGameState:
    """Stores the state of a two player game using two different evaluators.
    - `key`: rng
    - `env_state`: The environment state.
    - `env_state_metadata`: Metadata associated with the environment state.
    - `p1_eval_state`: The internal state of the first evaluator.
    - `p2_eval_state`: The internal state of the second evaluator.
    - `p1_value_estimate`: The current state value estimate of the first evaluator.
    - `p2_value_estimate`: The current state value estimate of the second evaluator.
    - `outcomes`: The outcomes of the game (final rewards) for each player
    - `completed`: Whether the game is completed.
    """
    key: jax.Array
    env_state: Any
    env_state_metadata: StepMetadata
    p1_eval_state: Any
    p2_eval_state: Any
    p1_value_estimate: jax.Array
    p2_value_estimate: jax.Array
    outcomes: float
    completed: bool


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GameFrame:
    """Stores information necessary for rendering the environment state in a two-player game.
    - `env_state`: The environment state.
    - `p1_value_estimate`: The current state value estimate of the first evaluator.
    - `p2_value_estimate`: The current state value estimate of the second evaluator.
    - `completed`: Whether the game is completed.
    - `outcomes`: The outcomes of the game (final rewards) for each player
    """
    env_state: Any
    p1_value_estimate: jax.Array
    p2_value_estimate: jax.Array
    completed: jax.Array
    outcomes: jax.Array


def two_player_game_step(
    state: TwoPlayerGameState,
    p1_evaluator: Evaluator,
    p2_evaluator: Evaluator,
    params: Any,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    use_p1: bool,
    max_steps: int
) -> TwoPlayerGameState:
    """Make a single step in a two player game.
    
    Args:
    - `state`: The current game state.
    - `p1_evaluator`: The first evaluator.
    - `p2_evaluator`: The second evaluator.
    - `params`: The parameters of the active evaluator.
    - `env_step_fn`: The environment step function.
    - `env_init_fn`: The environment initialization function.
    - `use_p1`: Whether to use the first evaluator.
    - `max_steps`: The maximum number of steps per episode.
    
    Returns:
    - (TwoPlayerGameState): The updated game state.
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
    output, env_state, env_state_metadata, terminated, truncated, rewards = step_env_and_evaluator(
        key = step_key,
        env_state = state.env_state,
        env_state_metadata = state.env_state_metadata,
        eval_state = active_eval_state,
        params = params,
        evaluator = active_evaluator,
        env_step_fn = env_step_fn,
        env_init_fn = env_init_fn,
        max_steps = max_steps,
        reset = False
    )

    
    active_eval_state = output.eval_state
    active_value_estimate = active_evaluator.get_value(active_eval_state)
    active_value_estimate = jax.lax.cond(
        terminated | truncated,
        lambda a: a,
        lambda a: active_evaluator.discount * a,
        active_value_estimate
    )
    # update the other evaluator
    other_eval_state = other_evaluator.step(other_eval_state, output.action)
    other_value_estimate = other_evaluator.get_value(other_eval_state)
    # update the game state
    if use_p1:
        p1_eval_state, p2_eval_state = active_eval_state, other_eval_state
        p1_value_estimate, p2_value_estimate = active_value_estimate, other_value_estimate
    else:
        p1_eval_state, p2_eval_state = other_eval_state, active_eval_state
        p1_value_estimate, p2_value_estimate = other_value_estimate, active_value_estimate
    return replace(state,
        key = key,
        env_state = env_state,
        env_state_metadata = env_state_metadata,
        p1_eval_state = p1_eval_state,
        p2_eval_state = p2_eval_state,
        p1_value_estimate = p1_value_estimate,
        p2_value_estimate = p2_value_estimate,
        outcomes=jnp.where(
            ((terminated | truncated) & ~state.completed)[..., None],
            rewards,
            state.outcomes
        ),
        completed = state.completed | terminated | truncated,
    )


def two_player_game(
    key: jax.Array,
    evaluator_1: Evaluator,
    evaluator_2: Evaluator,
    params_1: Any,
    params_2: Any,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int
) -> Tuple[jax.Array, TwoPlayerGameState, jax.Array]:
    """
    Play a two player game between two evaluators.

    Args:
    - `key`: rng
    - `evaluator_1`: The first evaluator.
    - `evaluator_2`: The second evaluator.
    - `params_1`: The parameters of the first evaluator.
    - `params_2`: The parameters of the second evaluator.
    - `env_step_fn`: The environment step function.
    - `env_init_fn`: The environment initialization function.
    - `max_steps`: The maximum number of steps per episode.

    Returns:
    - (jax.Array, TwoPlayerGameState, jax.Array, jax.Array)
        - `outcomes`: The outcomes of the game (final rewards) for each player.
        - `frames`: Frames collected from the game (used for rendering)
        - `p_ids`: The player ids of the two evaluators. [evaluator_1_id, evaluator_2_id]
    """
    # init rng
    env_key, turn_key, key = jax.random.split(key, 3)
    # init env state
    env_state, metadata = env_init_fn(env_key) 
    # init evaluator states
    p1_eval_state = evaluator_1.init(template_embedding=env_state)
    p2_eval_state = evaluator_2.init(template_embedding=env_state)
    # compile step functions
    game_step = partial(two_player_game_step,
        p1_evaluator=evaluator_1,
        p2_evaluator=evaluator_2,
        env_step_fn=env_step_fn,
        env_init_fn=env_init_fn,
        max_steps=max_steps
    )
    step_p1 = partial(game_step, params=params_1, use_p1=True)
    step_p2 = partial(game_step, params=params_2, use_p1=False)
    
    # determine who goes first
    first_player = jax.random.randint(turn_key, (), 0, 2)
    p1_first = first_player == 0
    p1_id, p2_id = jax.lax.cond(
        p1_first,
        lambda _: (metadata.cur_player_id, 1 - metadata.cur_player_id),
        lambda _: (1 - metadata.cur_player_id, metadata.cur_player_id),
        None
    )
    # init game state
    state = TwoPlayerGameState(
        key = key,
        env_state = env_state,
        env_state_metadata = metadata,
        p1_eval_state = p1_eval_state,
        p2_eval_state = p2_eval_state,
        p1_value_estimate = jnp.array(0.0, dtype=jnp.float32),
        p2_value_estimate = jnp.array(0.0, dtype=jnp.float32),
        outcomes = jnp.zeros((2,), dtype=jnp.float32),
        completed = jnp.zeros((), dtype=jnp.bool_)
    )     
    # make initial render frame
    initial_game_frame = GameFrame(
        env_state = state.env_state,
        p1_value_estimate = state.p1_value_estimate,
        p2_value_estimate = state.p2_value_estimate,
        completed = state.completed,
        outcomes = state.outcomes
    )

    # takes a turn for the active player
    def take_turn(state: TwoPlayerGameState, step_num: jax.Array) -> TwoPlayerGameState:
        # players alternate turns, starting with the first player
        use_p1 = (step_num % 2 == 0) == p1_first
        state = jax.lax.cond(
            state.completed,
            lambda s: s,
            lambda s: jax.lax.cond(
                use_p1,
                step_p1,
                step_p2,
                s
            ),
            state
        )
        # collect render frame
        frame = GameFrame(
            env_state = state.env_state,
            p1_value_estimate = state.p1_value_estimate,
            p2_value_estimate = state.p2_value_estimate,
            completed = state.completed,
            outcomes = state.outcomes
        )
        # return game state and render frame
        return state, frame
    
    # play the game
    state, frames = jax.lax.scan(
        take_turn,
        state,
        xs=jnp.arange(max_steps)
    )
    # append initial state to front of frames
    frames = jax.tree.map(lambda i, x: jnp.concatenate([jnp.expand_dims(i, 0), x]), initial_game_frame, frames)
    # return outcome, frames, player ids
    return jnp.array([state.outcomes[p1_id], state.outcomes[p2_id]]), frames, jnp.array([p1_id, p2_id])
