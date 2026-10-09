"""Training positions taken from self-play search trees, as in OLIVAW (Norelli & Panconesi, 2022,
https://arxiv.org/abs/2103.17228, section IV-B).

Besides the positions self-play plays, which train on (the search's visit distribution, the game's
outcome), the network trains on positions its searches explored often: each on the visit distribution
over its children and its value in the search, q. The game's outcome is a noisy signal for a position
far from the end of the game; q is a better informed one, if limited by the search's horizon (and early
in training, mostly the network's own estimates).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from core.evaluators.mcts.state import MCTSTree
from core.memory.replay_memory import BaseExperience
from core.types import DataTransformFn, EnvStepFn, StateToNNInputFn


@dataclass(frozen=True)
class TreePositions:
    """Which search tree positions self-play stores, and how many of them training batches hold.

    Attributes:
        per_move: most tree positions stored from each self-play search (before data transforms add
            their copies)
        min_visits: visits a node needs before it can be stored. Its policy target comes from the
            visits it passed on to its children, one fewer than its own.
        capacity: tree positions (data transforms' copies included) the replay buffer keeps per
            environment. Training samples those stored in the replay window's span of self-play (see
            `Trainer.tree_sample_mask`), so K times the largest replay window always holds it.
        ratio: tree positions per played position in training batches (OLIVAW's is 1)
        half_life: (optional) epochs over which `ratio` halves; by default it stays constant
        most_visited: store each search's most-visited nodes, as OLIVAW did, rather than sampling
            nodes in proportion to their visits
        discarded_only: only store nodes the search tree is about to discard: not those in the
            subtree of the move played, which the next search reuses (when the evaluator persists its
            tree). Each node can then be stored once, by the last search it's in, with all its visits;
            otherwise the same position can be stored by several searches in a row, and is often
            played next too.
    """

    per_move: int
    min_visits: int
    capacity: int
    ratio: float = 1.0
    half_life: float | None = None
    most_visited: bool = False
    discarded_only: bool = False

    def __post_init__(self):
        if self.per_move < 1 or self.min_visits < 2:
            raise ValueError(
                "TreePositions needs per_move >= 1, and min_visits >= 2 for a policy target"
            )

    def ratio_at(self, epoch: int) -> float:
        """Tree positions per played position in the training batches of `epoch`."""
        if self.half_life is None:
            return self.ratio
        return self.ratio * 0.5 ** (epoch / self.half_life)

    def batch_count(self, epoch: int, batch_size: int) -> int:
        """Tree positions in each training batch of `batch_size` samples in `epoch`."""
        ratio = self.ratio_at(epoch)
        return min(batch_size, math.floor(batch_size * ratio / (1 + ratio) + 0.5))

    def get_config(self) -> dict:
        """Returns the configuration. Used for logging."""
        return {
            "per_move": self.per_move,
            "min_visits": self.min_visits,
            "capacity": self.capacity,
            "ratio": self.ratio,
            "half_life": self.half_life,
            "most_visited": self.most_visited,
            "discarded_only": self.discarded_only,
        }


def child_visits(tree: MCTSTree) -> jax.Array:
    """Visit counts of every node's children.

    Returns:
        jax.Array: (capacity, branching factor), 0 where there's no child
    """
    return jnp.where(
        tree.edge_map == tree.NULL_INDEX, 0, tree.data.n[tree.edge_map]
    ).astype(tree.data.n.dtype)


def select_nodes(
    key: jax.Array,
    tree: MCTSTree,
    num: int,
    min_visits: int,
    most_visited: bool = False,
    exclude_subtree: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Chooses up to `num` nodes of a search tree to train on.

    A node qualifies if it isn't the root, has been visited at least `min_visits` times, isn't terminal,
    and has visited children (and, with `exclude_subtree`, isn't in that subtree). Of those, `num`
    are sampled without replacement in proportion to their visit counts, or with `most_visited`, the
    `num` most visited are taken (ties broken at random).

    Args:
        key: rng
        tree: search tree, after the search
        num: nodes to choose, at most the tree's capacity
        min_visits: visits a node needs to qualify
        most_visited: take the most-visited nodes rather than sampling them
        exclude_subtree: (optional) a child of the root whose subtree to leave out, or NULL_INDEX for
            none: e.g. the move played, whose subtree the next search reuses, so that its nodes can be
            chosen then (with the visits they gain) rather than now as well

    Returns:
        Tuple[jax.Array, jax.Array]: the chosen nodes' indices (num,), and which of them are valid (num,):
            when fewer than `num` nodes qualify, the rest are padding
    """
    n = tree.data.n
    index = jnp.arange(tree.capacity)
    qualifies = (
        (index != tree.ROOT_INDEX)
        & (index < tree.next_free_idx)
        & (n >= min_visits)
        & ~tree.data.terminated
        & (child_visits(tree).sum(axis=-1) > 0)
    )
    if exclude_subtree is not None:
        qualifies &= (exclude_subtree == tree.NULL_INDEX) | (
            tree.root_subtrees() != exclude_subtree
        )
    if most_visited:
        # visit counts are whole numbers, so noise below 1 only breaks ties
        score = n + 0.5 * jax.random.uniform(key, n.shape)
    else:
        # the Gumbel top-k trick: the top `num` of log-weights plus Gumbel noise are a sample
        # without replacement in proportion to the weights
        score = jnp.log(jnp.maximum(n, 1)) + jax.random.gumbel(key, n.shape)
    score, indices = jax.lax.top_k(jnp.where(qualifies, score, -jnp.inf), num)
    return indices, score > -jnp.inf


def tree_experiences(
    tree: MCTSTree,
    indices: jax.Array,
    valid: jax.Array,
    env_step_fn: EnvStepFn,
    state_to_nn_input_fn: StateToNNInputFn,
    transform_fns: Sequence[DataTransformFn] = (),
) -> tuple[BaseExperience, jax.Array]:
    """Training samples for nodes of a search tree (see `select_nodes`), with their transformed copies.

    Each node's policy target is the visit distribution over its children, and its value target its
    value in the search, q, for the player to move there (MCTS keeps each node's value from the
    perspective of the player to move at it). Its legal actions and the player to move come from
    stepping its parent's state along the edge to it again.

    Its `reward` holds q for the player to move, and -q for the other (two-player zero-sum games), the
    targets an episode's outcome would give, so it trains on q whatever weight
    `core.training.loss_fns.az_default_loss_fn` gives the search's value; its `search_value` is q too.

    Args:
        tree: search tree, after the search
        indices: (K,) nodes to make samples of
        valid: (K,) which of `indices` are nodes, rather than padding
        env_step_fn: environment step function
        state_to_nn_input_fn: converts an environment state to the network's input
        transform_fns: data transforms (e.g. board symmetries), each adding a copy of every sample

    Returns:
        Tuple[BaseExperience, jax.Array]: K * (1 + len(transform_fns)) samples: the nodes', then each
            transform's copies of them, and which of them are valid
    """
    nodes = tree.data_at(indices)
    parents = tree.parents[indices]
    # the action along the edge from the parent to the node
    actions = jnp.argmax(tree.edge_map[parents] == indices[:, None], axis=-1)
    _, metadata = jax.vmap(env_step_fn)(tree.data_at(parents).embedding, actions)
    mask = metadata.action_mask
    # only legal actions have children, so the mask only guards against a malformed tree
    visits = jnp.where(mask, child_visits(tree)[indices], 0)
    policy = visits / jnp.maximum(visits.sum(axis=-1, keepdims=True), 1)
    q = nodes.q
    players = jnp.arange(metadata.rewards.shape[-1])
    reward = jnp.where(
        players == metadata.cur_player_id[:, None], q[:, None], -q[:, None]
    ).astype(metadata.rewards.dtype)

    def experiences(mask, policy, env_state):
        return BaseExperience(
            observation_nn=jax.vmap(state_to_nn_input_fn)(env_state),
            policy_mask=mask,
            policy_weights=policy,
            reward=reward,
            cur_player_id=metadata.cur_player_id,
            search_value=q,
        )

    samples = [experiences(mask, policy, nodes.embedding)]
    for transform_fn in transform_fns:
        samples.append(
            experiences(*jax.vmap(transform_fn)(mask, policy, nodes.embedding))
        )
    return (
        jax.tree.map(lambda *x: jnp.concatenate(x), *samples),
        jnp.tile(valid, len(samples)),
    )
