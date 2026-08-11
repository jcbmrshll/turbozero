
from functools import partial
from typing import Tuple
import chex
from chex import dataclass

import jax
import jax.numpy as jnp
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.types import EnvInitFn, EnvStepFn, StepMetadata

def partition(
    data: chex.ArrayTree,
    num_partitions: int
) -> chex.ArrayTree:
    """Partition each array in a data structure into num_partitions along the first axis.
    e.g. partitions an array of shape (N, ...) into (num_partitions, N//num_partitions, ...)

    Args:
    - `data`: ArrayTree to partition
    - `num_partitions`: number of partitions

    Returns:
    - (chex.ArrayTree): partitioned ArrayTree
    """
    return jax.tree_map(
        lambda x: x.reshape(num_partitions, x.shape[0] // num_partitions, *x.shape[1:]),
        data
    )


def step_env_and_evaluator(
    key: jax.random.PRNGKey,
    env_state: chex.ArrayTree,
    env_state_metadata: StepMetadata,
    eval_state: chex.ArrayTree,
    params: chex.ArrayTree,
    evaluator: Evaluator,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int,
    reset: bool = True
) -> Tuple[EvalOutput, chex.ArrayTree,  StepMetadata, bool, bool, chex.Array]:
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
    - (EvalOutput, chex.ArrayTree, StepMetadata, bool, bool, chex.Array)
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
    output = output.replace(eval_state=eval_state)
    return output, env_state, env_state_metadata, terminated, truncated, rewards

@dataclass(frozen=True)
class SinglePlayerGameState:
    """Stores the state of a single-player game for two evaluators playing independently.
    - `key`: rng
    - `env_state`: The initial environment state.
    - `env_state_metadata`: Metadata associated with the initial environment state.
    - `eval_state_1`: The internal state of the first evaluator.
    - `eval_state_2`: The internal state of the second evaluator.
    - `completed_1`: Whether the first evaluator's game is completed.
    - `completed_2`: Whether the second evaluator's game is completed.
    - `outcome_1`: The final reward of the first evaluator.
    - `outcome_2`: The final reward of the second evaluator.
    """
    key: jax.random.PRNGKey
    env_state_1: chex.ArrayTree
    env_state_2: chex.ArrayTree
    env_state_metadata_1: StepMetadata
    env_state_metadata_2: StepMetadata
    eval_state_1: chex.ArrayTree
    eval_state_2: chex.ArrayTree
    completed_1: bool
    completed_2: bool
    outcome_1: float
    outcome_2: float

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
    key: jax.random.PRNGKey
    env_state: chex.ArrayTree
    env_state_metadata: StepMetadata
    p1_eval_state: chex.ArrayTree
    p2_eval_state: chex.ArrayTree
    p1_value_estimate: chex.Array
    p2_value_estimate: chex.Array
    outcomes: float
    completed: bool


@dataclass(frozen=True)
class GameFrame:
    """Stores information necessary for rendering the environment state in a two-player game.
    - `env_state`: The environment state.
    - `p1_value_estimate`: The current state value estimate of the first evaluator.
    - `p2_value_estimate`: The current state value estimate of the second evaluator.
    - `completed`: Whether the game is completed.
    - `outcomes`: The outcomes of the game (final rewards) for each player
    """
    env_state: chex.ArrayTree
    p1_value_estimate: chex.Array
    p2_value_estimate: chex.Array
    completed: chex.Array
    outcomes: chex.Array

def single_player_play(
    key: jax.random.PRNGKey,
    env_state: chex.ArrayTree,
    env_state_metadata: StepMetadata,
    eval_state: chex.ArrayTree,
    evaluator: Evaluator,
    params: chex.ArrayTree,
    env_step_fn: EnvStepFn,
    max_steps: int
) -> Tuple[chex.ArrayTree, StepMetadata, chex.ArrayTree, bool, float]:
    """
    Executes a single player game until completion.
    
    Args:
    - `key`: RNG.
    - `env_state`: Initial environment state.
    - `env_state_metadata`: Metadata for the environment state.
    - `eval_state`: Evaluator's internal state.
    - `evaluator`: Evaluator to step through the environment.
    - `params`: Evaluator's parameters.
    - `env_step_fn`: Function to step the environment.
    - `max_steps`: Maximum steps to take.

    Returns:
    - `env_state`: Final environment state.
    - `env_state_metadata`: Final metadata.
    - `eval_state`: Final evaluator state.
    - `completed`: Whether the episode is completed.
    - `outcome`: The total reward for the game.
    """
    def step_fn(carry, _):
        env_state, env_metadata, eval_state, key = carry
        step_key, key = jax.random.split(key)

        # Evaluate and take action
        output = evaluator.evaluate(
            key=step_key,
            eval_state=eval_state,
            env_state=env_state,
            root_metadata=env_metadata,
            params=params,
            env_step_fn=env_step_fn
        )
        next_env_state, next_env_metadata = env_step_fn(env_state, output.action)
        terminated = next_env_metadata.terminated
        truncated = next_env_metadata.step > max_steps
        completed = terminated | truncated
        rewards = next_env_metadata.rewards
        eval_state = jax.lax.cond(
            completed,
            lambda _: eval_state,
            lambda _: evaluator.step(eval_state, output.action),
            None
        )
        return (next_env_state, next_env_metadata, eval_state, step_key), (completed, rewards)

    # Scan through the steps
    (env_state, final_metadata, final_eval_state, _), (completed, rewards) = jax.lax.scan(
        step_fn,
        (env_state, env_state_metadata, eval_state, key),
        xs=None,
        length=max_steps
    )

    # Determine the outcome
    final_outcome = jnp.where(completed, rewards.flatten(), 0.0)
    final_completed = jnp.any(completed)
    return env_state, final_metadata, final_eval_state, final_completed, final_outcome

def single_player_game(
    key: jax.random.PRNGKey,
    evaluator_1: Evaluator,
    evaluator_2: Evaluator,
    params_1: chex.ArrayTree,
    params_2: chex.ArrayTree,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int
) -> Tuple[chex.Array, SinglePlayerGameState, chex.Array]:
    """
    Simulates a single-player game for two evaluators playing independently.

    Args:
    - `key`: RNG.
    - `evaluator_1`: The first evaluator.
    - `evaluator_2`: The second evaluator.
    - `params_1`: Parameters for the first evaluator.
    - `params_2`: Parameters for the second evaluator.
    - `env_step_fn`: The environment step function.
    - `env_init_fn`: The environment initialization function.
    - `max_steps`: Maximum number of steps.

    Returns:
    - (chex.Array): A tuple of final outcomes for both evaluators.
    - (SinglePlayerGameState): The final state of the single-player game.
    """
    # Initialize RNG
    init_key, key_1, key_2 = jax.random.split(key, 3)

    # Initialize environment and evaluators
    env_state, metadata = env_init_fn(init_key)
    eval_state_1 = evaluator_1.init(template_embedding=env_state)
    eval_state_2 = evaluator_2.init(template_embedding=env_state)

    # No compilation as function is called only once
    # Evaluate the game for both evaluators independently
    env_state_1, metadata_1, eval_state_1, completed_1, outcome_1 = single_player_play(
        key_1, env_state, metadata, eval_state_1, evaluator_1, params_1, env_step_fn, max_steps
    )
    env_state_2, metadata_2, eval_state_2, completed_2, outcome_2 = single_player_play(
        key_2, env_state, metadata, eval_state_2, evaluator_2, params_2, env_step_fn, max_steps
    )

    # Create the final game state
    game_state = SinglePlayerGameState(
        key=key,
        env_state_1=env_state_1,
        env_state_2=env_state_2,
        env_state_metadata_1=metadata_1,
        env_state_metadata_2=metadata_2,
        eval_state_1=eval_state_1,
        eval_state_2=eval_state_2,
        completed_1=completed_1,
        completed_2=completed_2,
        outcome_1=outcome_1,
        outcome_2=outcome_2
    )

    return jnp.array([outcome_1, outcome_2]), game_state, jnp.array([0, 1])


def two_player_game_step(
    state: TwoPlayerGameState,
    p1_evaluator: Evaluator,
    p2_evaluator: Evaluator,
    params: chex.ArrayTree,
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
    return state.replace(
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
    key: jax.random.PRNGKey,
    evaluator_1: Evaluator,
    evaluator_2: Evaluator,
    params_1: chex.ArrayTree,
    params_2: chex.ArrayTree,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int
) -> Tuple[chex.Array, TwoPlayerGameState, chex.Array]:
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
    - (chex.Array, TwoPlayerGameState, chex.Array, chex.Array)
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

    # takes a turn for each player
    def step_step(state: TwoPlayerGameState, _) -> TwoPlayerGameState:
        # take a turn for the active player
        state = jax.lax.cond(
            state.completed,
            lambda s: s,
            lambda s: jax.lax.cond(
                p1_first,
                step_p1,
                step_p2,
                s
            ),
            state
        )
        # collect render frame
        frame1 = GameFrame(
            env_state = state.env_state,
            p1_value_estimate = state.p1_value_estimate,
            p2_value_estimate = state.p2_value_estimate,
            completed = state.completed,
            outcomes = state.outcomes
        )
        # take a turn for the other player
        state = jax.lax.cond(
            state.completed,
            lambda s: s,
            lambda s: jax.lax.cond(
                p1_first,
                step_p2,
                step_p1,
                s
            ),
            state
        )
        # collect render frame
        frame2 = GameFrame(
            env_state = state.env_state,
            p1_value_estimate = state.p1_value_estimate,
            p2_value_estimate = state.p2_value_estimate,
            completed = state.completed,
            outcomes = state.outcomes
        )
        # return game state and render frames
        return state, jax.tree_map(lambda x, y: jnp.stack([x, y]), frame1, frame2)
    
    # play the game
    state, frames = jax.lax.scan(
        step_step,
        state,
        xs=jnp.arange(max_steps//2)
    )
    # reshape frames
    frames = jax.tree_map(lambda x: x.reshape(max_steps, *x.shape[2:]), frames)
    # append initial state to front of frames
    frames = jax.tree_map(lambda i, x: jnp.concatenate([jnp.expand_dims(i, 0), x]), initial_game_frame, frames)
    # return outcome, frames, player ids
    return jnp.array([state.outcomes[p1_id], state.outcomes[p2_id]]), frames, jnp.array([p1_id, p2_id])
