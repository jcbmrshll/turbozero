from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BaseExperience:
    """A training sample: a position, its search's policy target and value, and
    the result of the game it was played in.
    """

    reward: jax.Array  # the game's reward for each player
    policy_weights: jax.Array  # the search's policy target
    policy_mask: jax.Array  # the legal actions
    observation_nn: jax.Array  # the network's input
    cur_player_id: jax.Array  # the player to move
    # the search's value of the position, for the player to move (in self-play,
    # the root value of the search that chose its move)
    search_value: jax.Array


# a data transform (e.g. one of a board's symmetries): maps an experience,
# without batch dimensions, to another as valid, e.g. the board rotated with its
# policy target and legal actions to match. The value targets (`reward`,
# `search_value`) and `cur_player_id` stay as they are.
DataTransformFn = Callable[[BaseExperience], BaseExperience]


def transform_randomly(
    key: jax.Array,
    experiences: BaseExperience,
    transform_fns: Sequence[DataTransformFn],
) -> BaseExperience:
    """Maps each of a batch of experiences through a transform drawn uniformly
    at random, independently for each, from the identity and `transform_fns`
    (e.g. a board's other symmetries).
    """
    if not transform_fns:
        return experiences
    branches = [lambda e: e, *transform_fns]
    choice = jax.random.randint(
        key, experiences.reward.shape[:1], 0, len(branches), dtype=jnp.int32
    )
    return jax.vmap(lambda c, e: jax.lax.switch(c, branches, e))(choice, experiences)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayBufferState:
    """A replay buffer's entries, written in order around a ring of `capacity`
    places. The current episode's entries can't be sampled until it ends and
    they get its reward.
    """

    next_idx: jax.Array  # where the next entry goes
    episode_start_idx: jax.Array  # where the current episode's entries start
    buffer: BaseExperience  # leaves of shape (capacity, ...)
    populated: jax.Array  # (capacity,) which places hold an entry
    has_reward: jax.Array  # (capacity,) false for the current episode's entries


class EpisodeReplayBuffer:
    """Stores self-play episodes' experiences for training, `capacity` per
    environment, overwriting the oldest. Its methods take a single
    environment's state, so self-play vmaps them over environments.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity

    def add_experience(
        self,
        state: ReplayBufferState,
        experience: BaseExperience,
        valid: ArrayLike = True,
    ) -> ReplayBufferState:
        """Adds an experience to the current episode. Nothing changes if not
        `valid`, which is cheaper under vmap than a lax.cond: that would select
        between two whole buffers.
        """
        # an out-of-bounds write is dropped
        index = jnp.where(valid, state.next_idx, self.capacity)
        return replace(
            state,
            buffer=jax.tree.map(
                lambda x, y: x.at[index].set(y, mode="drop"), state.buffer, experience
            ),
            next_idx=jnp.where(
                valid, (state.next_idx + 1) % self.capacity, state.next_idx
            ),
            populated=state.populated.at[index].set(True, mode="drop"),
            has_reward=state.has_reward.at[index].set(False, mode="drop"),
        )

    def add_experiences(
        self, state: ReplayBufferState, experiences: BaseExperience, valid: jax.Array
    ) -> ReplayBufferState:
        """Adds the experiences marked in `valid`, in order, one after another,
        ready to sample: their targets are already known (e.g. positions from a
        search tree). Keep them in a buffer of their own, as `truncate` would
        discard them along with an episode in progress.

        Args:
            experiences: leading dimension of at most `capacity`
        """
        index = self.write_indices(state, valid)
        next_idx = (state.next_idx + valid.sum()) % self.capacity
        return replace(
            state,
            buffer=jax.tree.map(
                lambda x, y: x.at[index].set(y, mode="drop"), state.buffer, experiences
            ),
            next_idx=next_idx,
            episode_start_idx=next_idx,
            populated=state.populated.at[index].set(True, mode="drop"),
            has_reward=state.has_reward.at[index].set(True, mode="drop"),
        )

    def write_indices(self, state: ReplayBufferState, valid: jax.Array) -> jax.Array:
        """Where `add_experiences` writes each experience marked in `valid`;
        `capacity` (out of bounds) for the others.
        """
        # the i-th valid experience goes i places after `next_idx`
        offsets = jnp.cumsum(valid) - 1
        return jnp.where(
            valid, (state.next_idx + offsets) % self.capacity, self.capacity
        )

    def assign_rewards(
        self, state: ReplayBufferState, reward: jax.Array
    ) -> ReplayBufferState:
        """Ends the current episode, giving its entries `reward` (one for each
        player).
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

    def truncate(self, state: ReplayBufferState) -> ReplayBufferState:
        """Ends the current episode, discarding its entries: they're left
        unpopulated, so they can't be sampled, and the next episode writes over
        them.
        """
        return replace(
            state,
            next_idx=state.episode_start_idx,
            has_reward=jnp.full_like(state.has_reward, True),
            populated=jnp.where(~state.has_reward, False, state.populated),
        )

    def sample_mask(self, state: ReplayBufferState) -> jax.Array:
        """The entries that can be sampled: populated, from a finished episode.
        Works with any batch dimensions in front of the capacity dimension.
        """
        return jnp.logical_and(state.populated, state.has_reward)

    def sample_indices(
        self, key: jax.Array, mask: jax.Array, sample_size: int, replace: bool = False
    ) -> jax.Array:
        """(sample_size,) indices into the flattened `mask`, drawn uniformly
        from the entries it marks (see `sample_mask`). Without `replace`, `mask`
        must mark at least `sample_size` entries, else unmarked ones fill the
        rest of the sample.
        """
        mask = mask.reshape(-1)
        return jax.random.choice(
            key, mask.size, shape=(sample_size,), replace=replace, p=mask / mask.sum()
        )

    def count_distinct_observations(
        self,
        state: ReplayBufferState,
        transform_fns: Sequence[DataTransformFn] = (),
    ) -> tuple[jax.Array, jax.Array]:
        """(distinct observations, experiences) among the experiences that can
        be sampled: a measure of how varied the replay data is. Works with any
        batch dimensions in front of the capacity dimension.

        Observations are compared by a 64-bit hash of their contents, so
        collisions are possible but vanishingly rare at buffer sizes. With
        `transform_fns` (e.g. a board's symmetries, which training applies to
        sampled experiences), observations that one maps to another count as
        one: each is compared by the least hash of its images under the identity
        and the transforms, which must be a group (as a board's symmetries are).
        """
        valid = self.sample_mask(state).reshape(-1)
        experiences = jax.tree.map(
            lambda x: x.reshape(valid.shape[0], *x.shape[state.populated.ndim :]),
            state.buffer,
        )

        def hashes(obs: jax.Array) -> tuple[jax.Array, jax.Array]:
            obs = obs.reshape(valid.shape[0], -1)
            # compare raw bits: floats as float32, everything else as int32
            if jnp.issubdtype(obs.dtype, jnp.floating):
                bits = jax.lax.bitcast_convert_type(obs.astype(jnp.float32), jnp.uint32)
            else:
                bits = jax.lax.bitcast_convert_type(obs.astype(jnp.int32), jnp.uint32)
            position = jnp.arange(1, bits.shape[1] + 1, dtype=jnp.uint32)

            def hash_rows(seed: int) -> jax.Array:
                # mix each element with a constant for its position (murmur3's
                # finalizer), then sum (mod 2^32), so the hash depends on every
                # element and where it is
                h = bits ^ (position * jnp.uint32(0x9E3779B9) + jnp.uint32(seed))
                h = (h ^ (h >> 16)) * jnp.uint32(0x85EBCA6B)
                h = (h ^ (h >> 13)) * jnp.uint32(0xC2B2AE35)
                return (h ^ (h >> 16)).sum(axis=-1, dtype=jnp.uint32)

            return hash_rows(0x2545F491), hash_rows(0x6C8E9CF5)

        h1, h2 = hashes(experiences.observation_nn)
        for transform_fn in transform_fns:
            t1, t2 = hashes(jax.vmap(transform_fn)(experiences).observation_nn)
            # keep the least (h1, h2) pair
            less = (t1 < h1) | ((t1 == h1) & (t2 < h2))
            h1, h2 = jnp.where(less, t1, h1), jnp.where(less, t2, h2)
        # valid experiences first, then by hash, so equal observations are
        # adjacent
        order = jnp.lexsort((h2, h1, ~valid))
        valid, h1, h2 = valid[order], h1[order], h2[order]
        new = jnp.concatenate(
            [jnp.array([True]), (h1[1:] != h1[:-1]) | (h2[1:] != h2[:-1])]
        )
        return (valid & new).sum(), valid.sum()

    def init(
        self, batch_size: int, template_experience: BaseExperience
    ) -> ReplayBufferState:
        """Empty states for `batch_size` environments, stacked, holding
        experiences shaped as `template_experience`.
        """
        shape = (batch_size, self.capacity)
        return ReplayBufferState(
            next_idx=jnp.zeros((batch_size,), dtype=jnp.int32),
            episode_start_idx=jnp.zeros((batch_size,), dtype=jnp.int32),
            buffer=jax.tree.map(
                lambda x: jnp.zeros(shape + x.shape, dtype=x.dtype),
                template_experience,
            ),
            populated=jnp.zeros(shape, dtype=jnp.bool_),
            has_reward=jnp.ones(shape, dtype=jnp.bool_),
        )
