"""A learned stop rule for MCTS (see `LearnedStop`): a small network, fit during self-play, that
predicts what a search would still gain on its move by searching on, and stops it once that is
small.

Its labels come from self-play's own searches: a share of the searches it applies to ignore it
and run their whole budget, recording the root's statistics at every check. When such a search
ends, the gain at each check is G(t) = Q_T(a_T) - Q_T(a_t): the value, by the root's children's
Q-values at the end T, of the move the whole search would play (a_T, the most visited at T) less
that of the move most visited at t. Each epoch the network is fit to those labels, and its
threshold set so that the searches it applies to take a chosen number of cycles on average, by
replaying the rule on the labelled searches.
"""

from dataclasses import dataclass, replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.typing import ArrayLike

from core.evaluators.mcts.state import MCTSTree

# the root's statistics the network reads (see `root_features`)
NUM_FEATURES = 16


@dataclass(frozen=True)
class LearnedStop:
    """A learned stop rule's settings (see the module's docstring).

    Attributes:
        searches: which searches it applies to, as `SmartPruning`'s: "fast-greedy", "fast" or
            "all"
        budget: the cycles (the root's evaluation and each iteration) the searches it applies to
            should take on average; each epoch's threshold is set to it
        interval: cycles between its checks, from a search's start: the cycles between a
            self-play's plays of the moves whose searches are done (`Trainer`'s
            `advance_every`), where the rule is read
        label_probability: the share of the searches it applies to that ignore it (and any
            `SmartPruning`), run their whole budget, and label its checks
        slots: labelled searches each environment keeps between fits; more in an epoch replace
            the oldest
        hidden: the network's hidden units, in each of its two layers
        fit_steps: Adam steps each fit takes, from the last fit's weights
        fit_batch: examples a step
        learning_rate: Adam's
        holdout: the share of labelled searches the threshold is set on and the rule is measured
            on, rather than fit to
    """

    searches: str
    budget: float
    interval: int = 8
    label_probability: float = 0.1
    slots: int = 32
    hidden: int = 32
    fit_steps: int = 300
    fit_batch: int = 1024
    learning_rate: float = 1e-3
    holdout: float = 0.2

    def __post_init__(self):
        if self.searches not in ("fast-greedy", "fast", "all"):
            raise ValueError(
                f"a learned stop's searches are fast-greedy, fast or all, got {self.searches!r}"
            )
        if self.budget < 1 or self.interval < 1:
            raise ValueError(
                f"a learned stop's budget and interval must be at least 1, got {self.budget}, "
                f"{self.interval}"
            )
        if not 0 < self.label_probability < 1 or not 0 < self.holdout < 1:
            raise ValueError(
                "a learned stop's label_probability and holdout must be in (0, 1), got "
                f"{self.label_probability}, {self.holdout}"
            )

    def get_config(self) -> dict:
        return dict(vars(self))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class StopperState:
    """A learned stop rule's state, in each environment's tree (see `LearnedStop`).

    Attributes:
        params: the network's weights (the same in every environment)
        threshold: () it stops a search once its prediction is below this; -inf (never) until
            its first fit
        label: () whether the search labels the rule's checks, drawn when it starts
        greedy: () whether the search's move is played greedily, told when it starts
        applies: () whether the rule applies to the search, decided when it starts
        features: (K, F) the root's statistics at each of the search's checks
        top: (K) the most visited action at each
        pruned: (K) whether `SmartPruning` would have stopped it there
        recorded: (K) which checks the search has reached
        ex_features, ex_gain, ex_pruned, ex_valid: (S, K, ...) the labelled searches since the
            last fit: their checks' statistics, gains, `pruned` and which are examples
        ex_end: (S) the cycles each of those searches ran
        ex_count: () labelled searches written since the last fit
    """

    params: dict
    threshold: jax.Array
    label: jax.Array
    greedy: jax.Array
    applies: jax.Array
    features: jax.Array
    top: jax.Array
    pruned: jax.Array
    recorded: jax.Array
    ex_features: jax.Array
    ex_gain: jax.Array
    ex_pruned: jax.Array
    ex_valid: jax.Array
    ex_end: jax.Array
    ex_count: jax.Array


def init_params(key: jax.Array, hidden: int) -> dict:
    """The network's first weights: two tanh layers of `hidden` units and a linear output."""
    k1, k2 = jax.random.split(key)

    def dense(k, n_in, n_out):
        return jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(1.0 / n_in)

    return {
        "w1": dense(k1, NUM_FEATURES, hidden),
        "b1": jnp.zeros((hidden,)),
        "w2": dense(k2, hidden, hidden),
        "b2": jnp.zeros((hidden,)),
        "w3": jnp.zeros((hidden,)),
        "b3": jnp.zeros(()),
    }


def predict(params: dict, features: jax.Array) -> jax.Array:
    """The predicted gain of searching on, for (..., F) features."""
    h = jnp.tanh(features @ params["w1"] + params["b1"])
    h = jnp.tanh(h @ params["w2"] + params["b2"])
    return h @ params["w3"] + params["b3"]


def init_stopper(rule: LearnedStop, checks: int) -> StopperState:
    """A stop rule's state before any search, with room for `checks` checks a search: its
    network's first weights, never stopping."""
    k, s = checks, rule.slots
    return StopperState(
        params=init_params(jax.random.PRNGKey(0), rule.hidden),
        threshold=jnp.array(-jnp.inf, dtype=jnp.float32),
        label=jnp.array(False),
        greedy=jnp.array(False),
        applies=jnp.array(False),
        features=jnp.zeros((k, NUM_FEATURES), dtype=jnp.float32),
        top=jnp.zeros((k,), dtype=jnp.int32),
        pruned=jnp.zeros((k,), dtype=jnp.bool_),
        recorded=jnp.zeros((k,), dtype=jnp.bool_),
        ex_features=jnp.zeros((s, k, NUM_FEATURES), dtype=jnp.float32),
        ex_gain=jnp.zeros((s, k), dtype=jnp.float32),
        ex_pruned=jnp.zeros((s, k), dtype=jnp.bool_),
        ex_valid=jnp.zeros((s, k), dtype=jnp.bool_),
        ex_end=jnp.zeros((s,), dtype=jnp.int32),
        ex_count=jnp.array(0, dtype=jnp.int32),
    )


def root_values(tree: MCTSTree, discount: float) -> jax.Array:
    """The root's children's Q-values from the root player's perspective."""
    return tree.get_child_data("q", tree.ROOT_INDEX) * discount


def root_features(
    tree: MCTSTree, cycles: ArrayLike, iterations: ArrayLike, discount: float
) -> jax.Array:
    """The root's statistics after `cycles` cycles of a search of `iterations` iterations: (F)
    float32, each roughly within [-1, 4]."""
    cycles = jnp.asarray(cycles, dtype=jnp.float32)
    iterations = jnp.asarray(iterations, dtype=jnp.float32)
    done = jnp.maximum(cycles - 1, 0.0)
    left = jnp.maximum(iterations - done, 0.0)
    n = tree.get_child_data("n", tree.ROOT_INDEX).astype(jnp.float32)
    q = root_values(tree, discount)
    root = tree.data_at(tree.ROOT_INDEX)
    prior = root.p
    total = jnp.maximum(n.sum(), 1.0)
    (n1, n2), (i1, i2) = jax.lax.top_k(n, 2)
    visited = n > 0
    q1 = jnp.where(n1 > 0, q[i1], 0.0)
    q2 = jnp.where(n2 > 0, q[i2], q1)
    best_q = jnp.where(visited, q, -jnp.inf).max()
    best_q = jnp.where(visited.any(), best_q, 0.0)
    best_q_share = n[jnp.where(visited, q, -jnp.inf).argmax()] / total
    share = n / total
    entropy = -jnp.where(share > 0, share * jnp.log(share), 0.0).sum()
    legal = (prior > 0).sum().astype(jnp.float32)
    inherited = jnp.maximum(root.n.astype(jnp.float32) - done, 0.0)
    return jnp.stack(
        [
            done / iterations,
            left / iterations,
            jnp.log1p(root.n.astype(jnp.float32)) / 8,
            jnp.log1p(inherited) / 8,
            n1 / total,
            n2 / total,
            jnp.minimum((n1 - n2) / jnp.maximum(left, 1.0), 4.0),
            q1,
            q2,
            best_q - q1,
            best_q_share,
            prior[i1],
            prior.max(),
            jnp.asarray(tree.root_value, dtype=jnp.float32),
            entropy / 4,
            jnp.log(jnp.maximum(legal, 1.0)) / 4,
        ]
    ).astype(jnp.float32)


def most_visited(tree: MCTSTree) -> jax.Array:
    """The root's most visited action (the first, of equals)."""
    return jnp.argmax(tree.get_child_data("n", tree.ROOT_INDEX)).astype(jnp.int32)


def gains(tree: MCTSTree, top: jax.Array, discount: float) -> jax.Array:
    """(K) the gain of searching on from each check whose most visited action was `top`, to the
    end of the search in `tree`: Q_T(a_T) - Q_T(a_t)."""
    q = root_values(tree, discount)
    return q[most_visited(tree)] - q[top]


def write_examples(
    stopper: StopperState, gain: jax.Array, valid: jax.Array, end: ArrayLike
) -> StopperState:
    """`stopper` with its search's checks, their `gain`s and which are `valid` examples written
    to its next slot, if any is, with the cycles the search ran (`end`)."""
    write = valid.any()
    slot = stopper.ex_count % stopper.ex_valid.shape[0]

    def put(buffer, rows):
        return buffer.at[slot].set(jnp.where(write, rows, buffer[slot]))

    return replace(
        stopper,
        ex_features=put(stopper.ex_features, stopper.features),
        ex_gain=put(stopper.ex_gain, gain),
        ex_pruned=put(stopper.ex_pruned, stopper.pruned),
        ex_valid=put(stopper.ex_valid, valid),
        ex_end=put(stopper.ex_end, jnp.asarray(end, dtype=jnp.int32)),
        ex_count=stopper.ex_count + write,
    )


@jax.jit
def _fit(
    params: dict, features: jax.Array, gain: jax.Array, learning_rate: float
) -> tuple:
    """Adam steps on (steps, batch, F) features and (steps, batch) gains; the weights and the
    mean squared error of each step."""
    optimizer = optax.adam(learning_rate)

    def step(carry, batch):
        params, opt_state = carry
        x, y = batch

        def loss_fn(p):
            return jnp.mean((predict(p, x) - y) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (cast(dict, optax.apply_updates(params, updates)), opt_state), loss

    (params, _), losses = jax.lax.scan(
        step, (params, optimizer.init(params)), (features, gain)
    )
    return params, losses


def replay(
    pred: np.ndarray,
    pruned: np.ndarray,
    gain: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    interval: int,
    end_cycles: np.ndarray,
) -> tuple[float, float]:
    """The rule replayed on labelled searches: (searches, checks) predictions, `pruned`, gains
    and which checks are valid. A search stops at its first valid check with `pruned` or a
    prediction below `threshold`, else at its end (`end_cycles`). Returns the mean cycles the
    searches take, and the mean gain they give up (0 for one that runs to its end)."""
    stops = valid & (pruned | (pred < threshold))
    first = np.where(stops.any(axis=1), stops.argmax(axis=1), -1)
    stopped = first >= 0
    cycles = np.where(stopped, (first + 1) * interval, end_cycles)
    lost = np.where(stopped, gain[np.arange(len(first)), np.maximum(first, 0)], 0.0)
    return float(cycles.mean()), float(lost.mean())


def calibrate(
    pred: np.ndarray,
    pruned: np.ndarray,
    valid: np.ndarray,
    gain: np.ndarray,
    budget: float,
    interval: int,
    end_cycles: np.ndarray,
) -> float:
    """The threshold at which the rule replayed on the labelled searches takes `budget` cycles
    on average (the highest that takes at least as many): -inf if they take less without it."""

    def mean_cycles(threshold):
        return replay(pred, pruned, gain, valid, threshold, interval, end_cycles)[0]

    if mean_cycles(-np.inf) <= budget:
        return -np.inf
    candidates = np.unique(pred[valid])
    lo, hi = 0, len(candidates) - 1
    # the mean cycles fall as the threshold rises: the highest candidate still above budget
    if mean_cycles(candidates[0]) <= budget:
        return float(candidates[0]) - 1e-6
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mean_cycles(candidates[mid]) >= budget:
            lo = mid
        else:
            hi = mid - 1
    return float(candidates[lo])


def fit_and_calibrate(
    key: jax.Array, rule: LearnedStop, stopper: StopperState
) -> tuple[dict, float, dict[str, Any]]:
    """Fits the network to the labelled searches since the last fit, from every environment's
    `stopper` (batched), and sets its threshold. Returns the weights, the threshold and what to
    log; with too few labelled searches, the network and threshold as they were.

    Args:
        key: rng
        rule: the rule's settings
        stopper: every environment's state
    """
    count = np.asarray(stopper.ex_count)
    slots = stopper.ex_valid.shape[1]
    filled = np.arange(slots)[None, :] < np.minimum(count, slots)[:, None]
    valid = np.asarray(stopper.ex_valid)[filled]
    features = np.asarray(stopper.ex_features)[filled]
    gain = np.asarray(stopper.ex_gain)[filled]
    pruned = np.asarray(stopper.ex_pruned)[filled]
    ends = np.asarray(stopper.ex_end)[filled]
    params = jax.tree.map(lambda x: x[0], stopper.params)
    threshold = float(np.asarray(stopper.threshold).reshape(-1)[0])
    searches = len(valid)
    metrics: dict[str, Any] = {"learned_stop_searches": searches}
    if searches < 20 or not valid.any():
        return params, threshold, metrics
    rng = np.random.default_rng(np.asarray(jax.random.key_data(key)).ravel()[-1])
    held = rng.random(searches) < rule.holdout
    fit_rows = ~held[:, None] & valid
    x, y = features[fit_rows], gain[fit_rows]
    if len(y) < rule.fit_batch:
        return params, threshold, metrics
    batches = rng.integers(0, len(y), size=(rule.fit_steps, rule.fit_batch))
    params, losses = _fit(
        params, jnp.asarray(x[batches]), jnp.asarray(y[batches]), rule.learning_rate
    )
    pred = np.asarray(predict(params, jnp.asarray(features[held])))
    h_valid, h_gain, h_pruned, h_end = valid[held], gain[held], pruned[held], ends[held]
    threshold = calibrate(
        pred, h_pruned, h_valid, h_gain, rule.budget, rule.interval, h_end
    )
    cycles, lost = replay(
        pred, h_pruned, h_gain, h_valid, threshold, rule.interval, h_end
    )
    stop_cycles, stop_lost = replay(
        pred, h_pruned, h_gain, h_valid, -np.inf, rule.interval, h_end
    )
    # a fixed cut at the same budget: stop at the check nearest it, or when pruned
    cut = max(1, round(rule.budget / rule.interval))
    fixed_pred = np.where(np.arange(pred.shape[1])[None, :] >= cut - 1, -1.0, 1.0)
    cut_cycles, cut_lost = replay(
        fixed_pred, h_pruned, h_gain, h_valid, 0.0, rule.interval, h_end
    )
    held_rows = h_valid
    metrics.update(
        {
            "learned_stop_examples": len(y),
            "learned_stop_fit_loss": float(np.asarray(losses[-20:]).mean()),
            "learned_stop_holdout_loss": float(
                np.mean((pred[held_rows] - h_gain[held_rows]) ** 2)
            ),
            "learned_stop_gain_variance": float(np.var(h_gain[held_rows])),
            "learned_stop_threshold": threshold,
            "learned_stop_cycles": cycles,
            "learned_stop_lost": lost,
            "learned_stop_pruning_cycles": stop_cycles,
            "learned_stop_pruning_lost": stop_lost,
            "learned_stop_cut_cycles": cut_cycles,
            "learned_stop_cut_lost": cut_lost,
        }
    )
    return params, threshold, metrics
