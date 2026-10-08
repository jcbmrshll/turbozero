from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BaseExperience:
    """Experience data structure. Stores a training sample.

    Attributes:
        reward: reward for each player in the episode this sample belongs to
        policy_weights: policy weights
        policy_mask: mask for policy weights (mask out invalid/illegal actions)
        observation_nn: observation for neural network input
        cur_player_id: current player id
    """

    reward: jax.Array
    policy_weights: jax.Array
    policy_mask: jax.Array
    observation_nn: jax.Array
    cur_player_id: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayBufferState:
    """State of the replay buffer.

    Stores objects stored in the buffer and metadata used to determine where to store
    the next object, as well as which objects are valid to sample from.

    Attributes:
        next_idx: index where the next experience will be stored
        episode_start_idx: index where the current episode started, samples are placed in order
        buffer: buffer of experiences
        populated: mask for populated buffer indices
        has_reward: mask for buffer indices that have been assigned a reward
            - we store samples from in-progress episodes, but don't want to be able to sample them
              until the episode is complete
    """

    next_idx: jax.Array
    episode_start_idx: jax.Array
    buffer: BaseExperience
    populated: jax.Array
    has_reward: jax.Array


class EpisodeReplayBuffer:
    """Replay buffer, stores trajectories from episodes for training.

    Compatible with `jax.jit` and `jax.vmap`.
    """

    def __init__(
        self,
        capacity: int,
    ):
        """Initializes an EpisodeReplayBuffer.

        Args:
            capacity: number of experiences to store in the buffer
        """
        self.capacity = capacity

    def get_config(self):
        """Returns the configuration of the replay buffer. Used for logging."""
        return {
            "capacity": self.capacity,
        }

    def add_experience(
        self, state: ReplayBufferState, experience: BaseExperience
    ) -> ReplayBufferState:
        """Adds an experience to the replay buffer.

        Args:
            state: replay buffer state
            experience: experience to add

        Returns:
            ReplayBufferState: updated replay buffer state
        """
        return replace(
            state,
            buffer=jax.tree_util.tree_map(
                lambda x, y: x.at[state.next_idx].set(y), state.buffer, experience
            ),
            next_idx=(state.next_idx + 1) % self.capacity,
            populated=state.populated.at[state.next_idx].set(True),
            has_reward=state.has_reward.at[state.next_idx].set(False),
        )

    def assign_rewards(
        self, state: ReplayBufferState, reward: jax.Array
    ) -> ReplayBufferState:
        """Assign rewards to the current episode.

        Args:
            state: replay buffer state
            reward: rewards to assign (for each player)

        Returns:
            ReplayBufferState: updated replay buffer state
        """
        return replace(
            state,
            episode_start_idx=state.next_idx,
            has_reward=jnp.full_like(state.has_reward, True),
            buffer=replace(
                state.buffer,
                reward=jnp.where(
                    ~state.has_reward[..., None], reward[None, ...], state.buffer.reward
                ),
            ),
        )

    def truncate(
        self,
        state: ReplayBufferState,
    ) -> ReplayBufferState:
        """Truncates the replay buffer, removing all experiences from the current episode.

        Use this if we want to discard all experiences from the current episode.

        Args:
            state: replay buffer state

        Returns:
            ReplayBufferState: updated replay buffer state
        """
        # un-assigned trajectory indices have populated set to False
        # so their buffer contents will be overwritten (eventually)
        # and cannot be sampled
        # so there's no need to overwrite them with zeros here
        return replace(
            state,
            next_idx=state.episode_start_idx,
            has_reward=jnp.full_like(state.has_reward, True),
            populated=jnp.where(~state.has_reward, False, state.populated),
        )

    def sample_mask(self, state: ReplayBufferState) -> jax.Array:
        """Marks the buffer entries that can be sampled: populated, and from a finished episode.

        Args:
            state: replay buffer state

        Returns:
            jax.Array: boolean mask, same shape as `state.populated`
        """
        return jnp.logical_and(state.populated, state.has_reward)

    def check_can_sample(self, state: ReplayBufferState) -> None:
        """Checks on the host that at least one episode has finished, so there is something to sample.

        Not compatible with `jax.jit`: `state` must hold concrete arrays.

        Args:
            state: replay buffer state

        Raises:
            ValueError: if no episode has finished yet
        """
        if not self.sample_mask(state).any():
            raise ValueError(
                "Cannot sample from the replay buffer: no episodes have finished yet. "
                "Collect more self-play steps before training (e.g. increase `warmup_steps`)."
            )

    def sample_indices(
        self, key: jax.Array, mask: jax.Array, sample_size: int
    ) -> jax.Array:
        """Samples entries uniformly, without replacement, from those marked in `mask`.

        Compatible with `jax.jit`. `mask` must mark at least `sample_size` entries,
        use `check_can_sample` to check that it marks any.

        Args:
            key: rng
            mask: mask of entries that can be sampled (see `sample_mask`), any shape
            sample_size: number of entries to sample

        Returns:
            jax.Array: indices into the flattened `mask`, shape (sample_size,)
        """
        mask = mask.reshape(-1)
        return jax.random.choice(
            key, mask.size, shape=(sample_size,), replace=False, p=mask / mask.sum()
        )

    def sample(
        self, state: ReplayBufferState, key: jax.Array, sample_size: int
    ) -> BaseExperience:
        """Samples experiences from the replay buffer.

        The buffer may have any number of batch dimensions, e.g. (batch_size, capacity, ...);
        samples are drawn across all of them, not per batch.

        Not compatible with `jax.jit`: checks on the host that at least one episode has finished
        (see `check_can_sample`). Use `sample_mask` and `sample_indices` to sample under `jax.jit`.

        Args:
            state: replay buffer state
            key: rng
            sample_size: size of minibatch to sample

        Returns:
            BaseExperience: minibatch of size (sample_size, ...)

        Raises:
            ValueError: if no episode has finished yet, so there is nothing to sample
        """
        self.check_can_sample(state)
        mask = self.sample_mask(state)
        indices = jnp.unravel_index(
            self.sample_indices(key, mask, sample_size), mask.shape
        )
        return jax.tree_util.tree_map(lambda x: x[indices], state.buffer)

    def init(
        self, batch_size: int, template_experience: BaseExperience
    ) -> ReplayBufferState:
        """Initializes the replay buffer state.

        Args:
            batch_size: number of parallel environments
            template_experience: template experience data structure
                - just used to determine the shape of the replay buffer data

        Returns:
            ReplayBufferState: initialized replay buffer state
        """
        return ReplayBufferState(
            next_idx=jnp.zeros((batch_size,), dtype=jnp.int32),
            episode_start_idx=jnp.zeros((batch_size,), dtype=jnp.int32),
            buffer=jax.tree_util.tree_map(
                lambda x: jnp.zeros(
                    (batch_size, self.capacity, *x.shape), dtype=x.dtype
                ),
                template_experience,
            ),
            populated=jnp.full(
                (
                    batch_size,
                    self.capacity,
                ),
                fill_value=False,
                dtype=jnp.bool_,
            ),
            has_reward=jnp.full(
                (
                    batch_size,
                    self.capacity,
                ),
                fill_value=True,
                dtype=jnp.bool_,
            ),
        )
