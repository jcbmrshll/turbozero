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


# a data transform: maps an experience, without batch dimensions, to another
# that's as valid a training sample, its observation transformed with its
# policy target and legal actions to match. The value targets (`reward`,
# `search_value`) and `cur_player_id` stay as they are.
DataTransformFn = Callable[[BaseExperience], BaseExperience]


def transform_randomly(
    key: jax.Array,
    experiences: BaseExperience,
    transform_fns: Sequence[DataTransformFn],
) -> BaseExperience:
    """Maps each of a batch of experiences through a transform drawn uniformly
    at random, independently for each, from the identity and `transform_fns`.
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

    @property
    def capacity(self) -> int:
        """The most entries the buffer holds."""
        return self.populated.shape[-1]


class EpisodeReplayBuffer:
    """Stores self-play experiences for training: each environment's newest
    entries, in a ring that overwrites the oldest.

    Writing and resizing work on one environment's state; vmap them over
    environments. Sampling works on all environments' states stacked, and
    draws from all of them at once.

    `capacity` is the size of the states `init` makes. The other methods read
    the size from the state, so a resized state keeps its new size.
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
        index = jnp.where(valid, state.next_idx, state.capacity)
        return replace(
            state,
            buffer=jax.tree.map(
                lambda x, y: x.at[index].set(y, mode="drop"), state.buffer, experience
            ),
            next_idx=jnp.where(
                valid, (state.next_idx + 1) % state.capacity, state.next_idx
            ),
            populated=state.populated.at[index].set(True, mode="drop"),
            has_reward=state.has_reward.at[index].set(False, mode="drop"),
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

    def resize(self, state: ReplayBufferState, capacity: int) -> ReplayBufferState:
        """`state` with `capacity` places, keeping its newest entries, as many
        as fit, oldest first, so the next entry written overwrites the oldest.
        Growing leaves the oldest places empty. An episode in progress carries
        on: its entries are the newest, so they're kept (its newest, if it has
        more than fit), and it ends as it would have.
        """
        # place j gets the entry of age capacity - 1 - j
        age = capacity - 1 - jnp.arange(capacity)
        source = (state.next_idx - 1 - age) % state.capacity
        kept = age < state.capacity

        def copy(x, empty):
            return jnp.where(
                kept.reshape((-1,) + (1,) * (x.ndim - 1)),
                x[source],
                jnp.asarray(empty, x.dtype),
            )

        # the episode in progress is the newest entries, in the last places
        in_progress = jnp.sum(state.populated & ~state.has_reward)
        start = (capacity - jnp.minimum(in_progress, capacity)) % capacity
        return ReplayBufferState(
            next_idx=jnp.zeros_like(state.next_idx),
            episode_start_idx=start,
            buffer=jax.tree.map(lambda x: copy(x, 0), state.buffer),
            populated=copy(state.populated, False),
            has_reward=copy(state.has_reward, True),
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
        from all the entries it marks, whatever environment they're in: `mask`
        is `sample_mask` of the stacked states, and
        `jnp.unravel_index(indices, mask.shape)` indexes their buffers. Without
        `replace`, `mask` must mark at least `sample_size` entries, else
        unmarked ones fill the rest of the sample.
        """
        # over every environment's stacked states: on a single environment's
        # (e.g. vmapped), each environment would draw its own sample
        assert mask.ndim >= 2 and mask.dtype == jnp.bool_
        mask = mask.reshape(-1)
        return jax.random.choice(
            key, mask.size, shape=(sample_size,), replace=replace, p=mask / mask.sum()
        )

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
