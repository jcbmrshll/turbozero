
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BaseExperience:
    """Experience data structure. Stores a training sample.
    - `reward`: reward for each player in the episode this sample belongs to
    - `policy_weights`: policy weights
    - `policy_mask`: mask for policy weights (mask out invalid/illegal actions)
    - `observation_nn`: observation for neural network input
    - `cur_player_id`: current player id
    """
    reward: jax.Array
    policy_weights: jax.Array
    policy_mask: jax.Array
    observation_nn: jax.Array
    cur_player_id: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayBufferState:
    """State of the replay buffer. Stores objects stored in the buffer 
    and metadata used to determine where to store the next object, as well as 
    which objects are valid to sample from.
    - `next_idx`: index where the next experience will be stored
    - `episode_start_idx`: index where the current episode started, samples are placed in order
    - `buffer`: buffer of experiences
    - `populated`: mask for populated buffer indices
    - `has_reward`: mask for buffer indices that have been assigned a reward
        - we store samples from in-progress episodes, but don't want to be able to sample them 
        until the episode is complete
    """
    next_idx: int
    episode_start_idx: int
    buffer: BaseExperience
    populated: jax.Array
    has_reward: jax.Array


class EpisodeReplayBuffer:
    """Replay buffer, stores trajectories from episodes for training.
    
    Compatible with `jax.jit`, `jax.vmap`, and `jax.pmap`."""

    def __init__(self,
        capacity: int,
    ):
        """
        Args:
        - `capacity`: number of experiences to store in the buffer
        """
        self.capacity = capacity


    def get_config(self):
        """Returns the configuration of the replay buffer. Used for logging."""
        return {
            'capacity': self.capacity,
        }


    def add_experience(self, state: ReplayBufferState, experience: BaseExperience) -> ReplayBufferState:
        """Adds an experience to the replay buffer.
        
        Args:
        - `state`: replay buffer state
        - `experience`: experience to add
        
        Returns:
        - (ReplayBufferState): updated replay buffer state"""
        return replace(state,
            buffer = jax.tree_util.tree_map(
                lambda x, y: x.at[state.next_idx].set(y),
                state.buffer,
                experience
            ),
            next_idx = (state.next_idx + 1) % self.capacity,
            populated = state.populated.at[state.next_idx].set(True),
            has_reward = state.has_reward.at[state.next_idx].set(False)
        )
    

    def assign_rewards(self, state: ReplayBufferState, reward: jax.Array) -> ReplayBufferState:
        """ Assign rewards to the current episode.
        
        Args:
        - `state`: replay buffer state
        - `reward`: rewards to assign (for each player)

        Returns:
        - (ReplayBufferState): updated replay buffer state
        """
        return replace(state,
            episode_start_idx = state.next_idx,
            has_reward = jnp.full_like(state.has_reward, True),
            buffer = replace(state.buffer,
                reward = jnp.where(
                    ~state.has_reward[..., None],
                    reward[None, ...],
                    state.buffer.reward
                )
            )
        )
    

    def truncate(self,
        state: ReplayBufferState,
    ) -> ReplayBufferState:
        """Truncates the replay buffer, removing all experiences from the current episode.
        Use this if we want to discard all experiences from the current episode.
        
        Args:
        - `state`: replay buffer state
        
        Returns:
        - (ReplayBufferState): updated replay buffer state
        """
        # un-assigned trajectory indices have populated set to False
        # so their buffer contents will be overwritten (eventually)
        # and cannot be sampled
        # so there's no need to overwrite them with zeros here
        return replace(state,
            next_idx = state.episode_start_idx,
            has_reward = jnp.full_like(state.has_reward, True),
            populated = jnp.where(
                ~state.has_reward,
                False,
                state.populated 
            )
        )
    
    # assumes input is batched!! (dont vmap/pmap)
    def sample(self,
        state: ReplayBufferState,
        key: jax.Array,
        sample_size: int
    ) -> BaseExperience:
        """Samples experiences from the replay buffer.

        Assumes the buffer has two batch dimensions, so shape = (devices, batch_size, capacity, ...)
        Perhaps there is a dimension-agnostic way to do this?

        Samples across all batch dimensions, not per-batch/device.
        
        Args:
        - `state`: replay buffer state
        - `key`: rng
        - `sample_size`: size of minibatch to sample

        Returns:
        - (BaseExperience): minibatch of size (sample_size, ...)
        """
        masked_weights = jnp.logical_and(
            state.populated,
            state.has_reward
        ).reshape(-1)

        num_partitions = state.populated.shape[0]
        num_batches = state.populated.shape[1]

        indices = jax.random.choice(
            key,
            self.capacity * num_partitions * num_batches,
            shape=(sample_size,),
            replace=False,
            p = masked_weights / masked_weights.sum()
        )

        partition_indices, batch_indices, item_indices = jnp.unravel_index(
            indices,
            (num_partitions, num_batches, self.capacity)
        )
        
        sampled_buffer_items = jax.tree_util.tree_map(
            lambda x: x[partition_indices, batch_indices, item_indices],
            state.buffer
        )

        return sampled_buffer_items
    
    
    def init(self, batch_size: int, template_experience: BaseExperience) -> ReplayBufferState:
        """Initializes the replay buffer state.
        
        Args:
        - `batch_size`: number of parallel environments
        - `template_experience`: template experience data structure
            - just used to determine the shape of the replay buffer data

        Returns:
        - (ReplayBufferState): initialized replay buffer state
        """
        return ReplayBufferState(
            next_idx = jnp.zeros((batch_size,), dtype=jnp.int32),
            episode_start_idx = jnp.zeros((batch_size,), dtype=jnp.int32),
            buffer = jax.tree_util.tree_map(
                lambda x: jnp.zeros((batch_size, self.capacity, *x.shape), dtype=x.dtype),
                template_experience
            ),
            populated = jnp.full((batch_size, self.capacity,), fill_value=False, dtype=jnp.bool_),
            has_reward = jnp.full((batch_size, self.capacity,), fill_value=True, dtype=jnp.bool_),
        )
