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
        search_value: the search's value of the position, for the player to move (for a position
            played in self-play, the root value of the search that chose its move)
    """

    reward: jax.Array
    policy_weights: jax.Array
    policy_mask: jax.Array
    observation_nn: jax.Array
    cur_player_id: jax.Array
    search_value: jax.Array


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

    def add_experiences(
        self, state: ReplayBufferState, experiences: BaseExperience, valid: jax.Array
    ) -> ReplayBufferState:
        """Adds experiences whose targets are already known (e.g. positions from a search tree, see
        `core.training.tree_positions`), which can be sampled right away.

        Only the experiences marked in `valid` are added, in order, one after another. Keep them in a
        buffer of their own: `truncate` would discard them along with an episode in progress.

        Args:
            state: replay buffer state
            experiences: experiences to add, with a leading dimension of at most `capacity`
            valid: (num experiences,) which of them to add

        Returns:
            ReplayBufferState: updated replay buffer state
        """
        index = self.write_indices(state, valid)
        next_idx = (state.next_idx + valid.sum()) % self.capacity
        return replace(
            state,
            buffer=jax.tree_util.tree_map(
                lambda x, y: x.at[index].set(y, mode="drop"), state.buffer, experiences
            ),
            next_idx=next_idx,
            episode_start_idx=next_idx,
            populated=state.populated.at[index].set(True, mode="drop"),
            has_reward=state.has_reward.at[index].set(True, mode="drop"),
        )

    def write_indices(self, state: ReplayBufferState, valid: jax.Array) -> jax.Array:
        """Where `add_experiences` writes each of the experiences marked in `valid`.

        Args:
            state: replay buffer state
            valid: which experiences are added

        Returns:
            jax.Array: each experience's index in the buffer; `capacity` (out of bounds) for those
                not added
        """
        # the i-th valid experience goes i places after `next_idx`
        offsets = jnp.cumsum(valid) - 1
        return jnp.where(
            valid, (state.next_idx + offsets) % self.capacity, self.capacity
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

    def sample_mask(
        self, state: ReplayBufferState, window: jax.Array | int | None = None
    ) -> jax.Array:
        """Marks the buffer entries that can be sampled: populated, from a finished episode, and
        with `window`, among the `window` newest entries written.

        An entry's age is how many entries have been written after it, `(next_idx - 1 - idx) mod
        capacity`, so a window of `w` holds what a buffer of capacity `w` would (an episode
        discarded by `truncate` aside: it doesn't push older entries out of the window). Entries
        of an episode still in progress count towards the window, though they can't be sampled.

        Compatible with `jax.jit`: `window` may be a traced value, so changing it doesn't
        recompile. Works with any number of batch dimensions in front of the capacity dimension.

        Args:
            state: replay buffer state
            window: (optional) entries per environment, newest first, to sample from; by default
                the whole buffer. At least `capacity` means the whole buffer too.

        Returns:
            jax.Array: boolean mask, same shape as `state.populated`
        """
        mask = jnp.logical_and(state.populated, state.has_reward)
        if window is None:
            return mask
        age = (
            state.next_idx[..., None] - 1 - jnp.arange(self.capacity)
        ) % self.capacity
        return mask & (age < window)

    def check_can_sample(
        self,
        state: ReplayBufferState,
        window: int | None = None,
        sample_size: int = 1,
    ) -> None:
        """Checks on the host that an episode has finished, and that there are at least
        `sample_size` entries to sample (in `window`, see `sample_mask`).

        Not compatible with `jax.jit`: `state` must hold concrete arrays.

        Args:
            state: replay buffer state
            window: (optional) entries per environment to sample from, see `sample_mask`
            sample_size: entries a sample takes

        Raises:
            ValueError: if no episode has finished yet, or fewer than `sample_size` entries can be
                sampled
        """
        if not self.sample_mask(state).any():
            raise ValueError(
                "Cannot sample from the replay buffer: no episodes have finished yet. "
                "Collect more self-play steps before training (e.g. increase `warmup_steps`)."
            )
        count = int(self.sample_mask(state, window).sum())
        if count < sample_size:
            raise ValueError(
                f"Cannot sample {sample_size} entries from the replay buffer: only {count} can be "
                "sampled. Collect more self-play steps before training (e.g. increase "
                "`warmup_steps`), widen the replay window, or sample fewer."
            )

    def sample_indices(
        self, key: jax.Array, mask: jax.Array, sample_size: int, replace: bool = False
    ) -> jax.Array:
        """Samples entries uniformly (by default without replacement) from those marked in `mask`.

        Compatible with `jax.jit`. Without replacement, `mask` must mark at least `sample_size` entries,
        else unmarked ones fill the rest of the sample; use `check_can_sample` to check.

        Args:
            key: rng
            mask: mask of entries that can be sampled (see `sample_mask`), any shape
            sample_size: number of entries to sample
            replace: sample with replacement instead (default: False)

        Returns:
            jax.Array: indices into the flattened `mask`, shape (sample_size,)
        """
        mask = mask.reshape(-1)
        return jax.random.choice(
            key, mask.size, shape=(sample_size,), replace=replace, p=mask / mask.sum()
        )

    def sample(
        self,
        state: ReplayBufferState,
        key: jax.Array,
        sample_size: int,
        window: int | None = None,
    ) -> BaseExperience:
        """Samples experiences from the replay buffer.

        The buffer may have any number of batch dimensions, e.g. (batch_size, capacity, ...);
        samples are drawn across all of them, not per batch.

        Not compatible with `jax.jit`: checks on the host that there are enough entries to sample
        (see `check_can_sample`). Use `sample_mask` and `sample_indices` to sample under `jax.jit`.

        Args:
            state: replay buffer state
            key: rng
            sample_size: size of minibatch to sample
            window: (optional) entries per environment, newest first, to sample from (see
                `sample_mask`); by default the whole buffer

        Returns:
            BaseExperience: minibatch of size (sample_size, ...)

        Raises:
            ValueError: if fewer than `sample_size` entries can be sampled
        """
        self.check_can_sample(state, window, sample_size)
        mask = self.sample_mask(state, window)
        indices = jnp.unravel_index(
            self.sample_indices(key, mask, sample_size), mask.shape
        )
        return jax.tree_util.tree_map(lambda x: x[indices], state.buffer)

    def count_distinct_observations(
        self, state: ReplayBufferState, window: jax.Array | int | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Counts the experiences that can be sampled (in `window`, see `sample_mask`), and how
        many distinct observations they hold.

        A measure of how varied the replay data is. Observations are compared by a 64-bit hash
        of their contents, so collisions are possible but vanishingly rare at buffer sizes.
        Works with any number of batch dimensions in front of the capacity dimension.

        Args:
            state: replay buffer state
            window: (optional) entries per environment, newest first, to count (see `sample_mask`);
                by default the whole buffer

        Returns:
            Tuple[jax.Array, jax.Array]: (number of distinct observations, number of experiences)
        """
        valid = self.sample_mask(state, window).reshape(-1)
        obs = state.buffer.observation_nn.reshape(valid.shape[0], -1)
        # compare raw bits: floats as float32, everything else as int32
        if jnp.issubdtype(obs.dtype, jnp.floating):
            bits = jax.lax.bitcast_convert_type(obs.astype(jnp.float32), jnp.uint32)
        else:
            bits = jax.lax.bitcast_convert_type(obs.astype(jnp.int32), jnp.uint32)
        position = jnp.arange(1, bits.shape[1] + 1, dtype=jnp.uint32)

        def hash_rows(seed: int) -> jax.Array:
            # mix each element with a constant for its position (murmur3's finalizer),
            # then sum (mod 2^32), so the hash depends on every element and where it is
            h = bits ^ (position * jnp.uint32(0x9E3779B9) + jnp.uint32(seed))
            h = (h ^ (h >> 16)) * jnp.uint32(0x85EBCA6B)
            h = (h ^ (h >> 13)) * jnp.uint32(0xC2B2AE35)
            return (h ^ (h >> 16)).sum(axis=-1, dtype=jnp.uint32)

        h1, h2 = hash_rows(0x2545F491), hash_rows(0x6C8E9CF5)
        # sort valid experiences first, then by hash, so equal observations end up adjacent
        order = jnp.lexsort((h2, h1, ~valid))
        valid, h1, h2 = valid[order], h1[order], h2[order]
        new = jnp.concatenate(
            [jnp.array([True]), (h1[1:] != h1[:-1]) | (h2[1:] != h2[:-1])]
        )
        return (valid & new).sum(), valid.sum()

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
