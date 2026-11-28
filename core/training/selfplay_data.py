"""What an epoch's self-play data is like: how sharp its policy targets are and how far search
moved them from the network's prior, how its value labels agree with the games' outcomes, and
how varied its positions are. Computed on a sample of the replay buffer's newest positions (see
`Trainer`'s `data_positions`), for all of them and for groups of them a game defines (e.g. its
phases), so that a change of self-play's settings shows in the data it makes, epoch by epoch.

Every function here is pure and compiles under `jax.jit`: positions come as arrays with a
leading sample axis, `weights` marking the real ones (a sample may be padded), `groups` (G, N)
marking each group's.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

# a search value counts as decided past this: an expected score of 95% or 5%
DECIDED = 0.9
# the share of probability spread over the legal actions before a KL divergence's log, so that an
# action the second distribution gives no weight to costs a finite amount
KL_FLOOR = 1e-3

# what `policy_metrics` reports, in its order
POLICY_METRICS = (
    "target_entropy",
    "prior_entropy",
    "target_prior_kl",
    "top_move_change",
    "decided_share",
)
# what `value_metrics` reports, in its order
VALUE_METRICS = ("label_value_mse", "value_bias", "outcome")
# what `pair_metrics` reports, in its order
PAIR_METRICS = ("pair_distance", "collision")


def entropy(p: jax.Array, mask: jax.Array) -> jax.Array:
    """(N,) the entropy of each row of `p` (N, A), in nats, over the actions `mask` marks."""
    return -jnp.sum(
        jnp.where(mask & (p > 0), p * jnp.log(jnp.where(p > 0, p, 1)), 0), -1
    )


def floored_kl(p: jax.Array, q: jax.Array, mask: jax.Array) -> jax.Array:
    """(N,) KL(p || q') of each row, over the actions `mask` marks, q' being q with `KL_FLOOR` of
    its probability spread uniformly over them (a search can visit an action its prior all but
    rules out)."""
    legal = mask.sum(-1, keepdims=True)
    q = (1 - KL_FLOOR) * q + KL_FLOOR * mask / jnp.maximum(legal, 1)
    terms = p * (jnp.log(jnp.where(p > 0, p, 1)) - jnp.log(jnp.where(mask, q, 1)))
    return jnp.sum(jnp.where(mask & (p > 0), terms, 0), -1)


def group_means(values: jax.Array, weights: jax.Array, groups: jax.Array) -> jax.Array:
    """(G, M) each group's weighted mean of each column of `values` (N, M); 0 for an empty
    group."""
    w = groups.astype(jnp.float32) * weights.astype(jnp.float32)[None]
    return (w @ values.astype(jnp.float32)) / jnp.maximum(
        w.sum(-1, keepdims=True), 1e-9
    )


def policy_metrics(
    target: jax.Array,
    logits: jax.Array,
    mask: jax.Array,
    search_value: jax.Array,
    weights: jax.Array,
    groups: jax.Array,
) -> jax.Array:
    """(G, len(POLICY_METRICS)) each group's means of:
    - `target_entropy`: the stored policy target's entropy
    - `prior_entropy`: the network's prior's entropy, softmax(logits) over the legal actions
    - `target_prior_kl`: KL(target || prior), how far search moved the target from the prior:
      what a position can teach the policy beyond what the network already says
    - `top_move_change`: how often the target's most likely action isn't the prior's
    - `decided_share`: how often the search's value is past ±`DECIDED`

    Args:
        target: (N, A) the stored policy targets
        logits: (N, A) the network's policy logits on the same positions
        mask: (N, A) bool, the legal actions
        search_value: (N,) the search's value, for the player to move
        weights: (N,) the positions' weights (0 for padding)
        groups: (G, N) bool
    """
    target = jnp.where(mask, target.astype(jnp.float32), 0)
    target = target / jnp.maximum(target.sum(-1, keepdims=True), 1e-9)
    logits = jnp.where(mask, logits.astype(jnp.float32), -jnp.inf)
    prior = jax.nn.softmax(logits, -1)
    changed = jnp.argmax(target, -1) != jnp.argmax(logits, -1)
    columns = jnp.stack(
        [
            entropy(target, mask),
            entropy(prior, mask),
            floored_kl(target, prior, mask),
            changed.astype(jnp.float32),
            (jnp.abs(search_value) > DECIDED).astype(jnp.float32),
        ],
        -1,
    )
    return group_means(columns, weights, groups)


def value_metrics(
    search_value: jax.Array,
    value: jax.Array,
    outcome: jax.Array,
    weights: jax.Array,
    groups: jax.Array,
) -> jax.Array:
    """(G, len(VALUE_METRICS)) each group's means of, all for the player to move:
    - `label_value_mse`: (search value - outcome)^2: how far the outcome training learns from
      is from what search expected, label noise included
    - `value_bias`: the network's value minus the outcome: a group whose bias differs from
      another's (e.g. by which side is to move) has an offset the outcomes don't support
    - `outcome`: the outcome

    Args:
        search_value: (N,) the search's value
        value: (N,) the network's value
        outcome: (N,) the game's result
        weights: (N,) the positions' weights (0 for padding or no outcome yet)
        groups: (G, N) bool
    """
    outcome = outcome.astype(jnp.float32)
    columns = jnp.stack(
        [
            jnp.square(search_value.astype(jnp.float32) - outcome),
            value.astype(jnp.float32) - outcome,
            outcome,
        ],
        -1,
    )
    return group_means(columns, weights, groups)


def pair_metrics(images: jax.Array, valid: jax.Array) -> jax.Array:
    """(len(PAIR_METRICS),) how alike a sample of positions are, over its pairs of distinct
    valid positions:
    - `pair_distance`: the mean count of observation entries that differ between the two, at the
      closest of the data transforms (e.g. a board's symmetries): how far apart positions are
    - `collision`: how often the two are the same position (Simpson's index: unbiased at any
      sample size, where a share of duplicates depends on how many positions there are)

    Args:
        images: (T, P, D) each position's observation, flattened, under each data transform,
            the identity first
        valid: (P,) bool, the sample's positions (others are padding)
    """
    images = images.astype(jnp.float32)
    first = images[0]

    def distance(image: jax.Array) -> jax.Array:
        return jnp.sum(jnp.abs(first[:, None, :] - image[None, :, :]), -1)

    nearest = jnp.min(jax.lax.map(distance, images), axis=0)
    p = valid.shape[0]
    pairs = valid[:, None] & valid[None, :] & ~jnp.eye(p, dtype=bool)
    count = jnp.maximum(pairs.sum(), 1)
    mean_distance = jnp.sum(jnp.where(pairs, nearest, 0)) / count
    collision = jnp.sum(pairs & (nearest == 0)) / count
    return jnp.stack([mean_distance, collision])


def group_sample(
    key: jax.Array, members: jax.Array, size: int
) -> tuple[jax.Array, jax.Array]:
    """At most `size` of the positions `members` (N,) marks, at random: their indices and
    whether each is one (fewer members than `size` leave the rest padding)."""
    priority = jnp.where(members, jax.random.uniform(key, members.shape), -1.0)
    top, indices = jax.lax.top_k(priority, size)
    return indices, top >= 0


def named(
    table: jax.Array, metrics: Sequence[str], group_names: Sequence[str]
) -> dict[str, jax.Array]:
    """`data_<metric>` for the first group (all positions), `data_<metric>_<group>` for each
    other, from `table` (G, M)."""
    return {
        f"data_{metric}" + ("" if g == 0 else f"_{group}"): table[g, m]
        for g, group in enumerate(group_names)
        for m, metric in enumerate(metrics)
    }
