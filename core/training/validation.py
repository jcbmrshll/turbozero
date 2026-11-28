from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from core.common import two_player_game
from core.evaluators.evaluator import Evaluator
from core.networks.utils import apply_nn
from core.training.selfplay_data import floored_kl, group_means
from core.types import EnvInitFn, EnvStepFn, StepMetadata

# what `scored_position_metrics` reports, for all positions and for each group
METRICS = ("top1", "regret", "exp_regret", "value_mse", "value_sign", "result_change")


@dataclass(frozen=True)
class ScoredPositions:
    """Fixed positions with a strong engine's score for every legal action, to measure a network
    against (see `ScoredPositionsValidator`).

    Attributes:
        observations: (N, ...) the positions, as the network takes them
        legal: (N, A) bool, which actions are legal
        scores: (N, A) each legal action's score from the engine for the player to move, higher
            better, e.g. the final margin it expects after the action; ignored where illegal
        groups: subsets to report metrics for besides all positions, by name: (N,) bool each
    """

    observations: np.ndarray
    legal: np.ndarray
    scores: np.ndarray
    groups: dict[str, np.ndarray]


def scored_position_metrics(
    logits: ArrayLike,
    value: ArrayLike,
    legal: ArrayLike,
    scores: ArrayLike,
    groups: ArrayLike,
    prior_temperature: float = 1.0,
) -> jax.Array:
    """How a network's raw outputs (no search) measure up to an engine's scores of positions'
    actions, for each of several groups of the positions:
    - `top1`: how often its most likely action is one of the engine's best
    - `regret`: how much score its most likely action gives up against the engine's best (mean)
    - `exp_regret`: the same, expected under its policy: smoother, so less noisy
    - `value_mse`: (value - z)^2, z the sign of the engine's best score (its predicted outcome)
    - `value_sign`: how often its value has z's sign, over positions where z isn't 0
    - `result_change`: the probability that its prior picks an action that worsens the outcome
      the engine predicts (an action whose score has a lower sign than the best's: a win to a
      draw or a loss, or a draw to a loss), the prior being softmax(logits / prior_temperature).
      Unlike regret, it doesn't count score given up by actions that keep the outcome, which a
      network trained on outcomes has no reason to avoid

    Args:
        logits: (N, A) policy logits
        value: (N,) values
        legal: (N, A) bool, legal actions
        scores: (N, A) the engine's scores, higher better (ignored where illegal)
        groups: (G, N) bool, which positions each group has
        prior_temperature: the temperature of the prior `result_change` samples from, e.g. the
            one a search flattens the policy with

    Returns:
        jax.Array: (G, len(METRICS)), each group's metrics, in the order of `METRICS`
    """
    logits, value, legal = jnp.asarray(logits), jnp.asarray(value), jnp.asarray(legal)
    scores, groups = jnp.asarray(scores, dtype=jnp.float32), jnp.asarray(groups)
    best = jnp.max(jnp.where(legal, scores, -jnp.inf), axis=1)
    # what each action gives up against the best
    loss = jnp.where(legal, best[:, None] - scores, 0.0)
    logits = jnp.where(legal, logits.astype(jnp.float32), -jnp.inf)
    top = jnp.argmax(logits, axis=1)
    regret = jnp.take_along_axis(loss, top[:, None], axis=1)[:, 0]
    exp_regret = jnp.sum(jax.nn.softmax(logits, axis=1) * loss, axis=1)
    z = jnp.sign(best)
    decided = z != 0
    worsens = legal & (jnp.sign(scores) < z[:, None])
    prior = jax.nn.softmax(logits / prior_temperature, axis=1)
    result_change = jnp.sum(jnp.where(worsens, prior, 0.0), axis=1)
    per_position = jnp.stack(
        [
            (regret == 0).astype(jnp.float32),
            regret,
            exp_regret,
            (value - z) ** 2,
            (jnp.sign(value) == z).astype(jnp.float32),
            result_change,
        ],
        axis=1,
    )
    # value_sign only counts the positions the engine doesn't call a draw
    weights = groups.astype(jnp.float32)
    counted = [weights] * 4 + [weights * decided, weights]
    counts = jnp.stack([w.sum(axis=1) for w in counted], axis=1)
    sums = jnp.stack(
        [w @ per_position[:, i] for i, w in enumerate(counted)],
        axis=1,
    )
    return sums / jnp.maximum(counts, 1)


class ScoredPositionsValidator:
    """Scores a network against an engine on fixed positions (see `ScoredPositions` and
    `scored_position_metrics`): one forward pass over them, no games, so it's cheap enough to
    run every epoch. A `Trainer` validator: called with the weights, it returns
    `<name>_<metric>` for all positions and `<name>_<metric>_<group>` for each group."""

    def __init__(
        self,
        positions: ScoredPositions,
        nn: eqx.Module,
        batch_size: int = 1024,
        name: str = "engine",
        config: dict | None = None,
        prior_temperature: float = 1.0,
    ):
        """
        Args:
            positions: the positions and the engine's scores
            nn: the network (an equinox module, see core.networks.utils.apply_nn); only its
                structure is used, the weights it's scored with are passed in each call. They
                may come from a copy of it that differs only in static settings, e.g. one
                computing in bfloat16 where `nn` computes in float32: they're put in `nn`'s
                structure, leaf for leaf
            batch_size: positions per forward pass
            name: prefixes the metrics
            config: (optional) recorded with the validator's configuration, e.g. where the
                positions came from
            prior_temperature: the temperature of the prior `result_change` is measured under
                (see `scored_position_metrics`): the one the play search flattens the policy
                with, so that it measures the prior the search starts from
        """
        self.name = name
        n = len(positions.legal)
        # padded to whole batches; the padding belongs to no group
        padded = -(-n // batch_size) * batch_size

        def pad(x: np.ndarray) -> jax.Array:
            return jnp.asarray(
                np.concatenate([x, np.repeat(x[:1], padded - n, axis=0)])
            )

        self.observations = pad(positions.observations)
        self.legal = pad(positions.legal)
        self.scores = pad(positions.scores)
        self.group_names = ["all", *positions.groups]
        groups = np.stack([np.ones(n, bool), *positions.groups.values()])
        self.groups = jnp.asarray(
            np.concatenate([groups, np.zeros((len(groups), padded - n), bool)], axis=1)
        )
        self.num_positions = n
        self.batch_size = batch_size
        self.extra_config = config or {}
        self.prior_temperature = prior_temperature
        static = eqx.filter(nn, eqx.is_inexact_array, inverse=True)
        structure = jax.tree.structure(eqx.filter(nn, eqx.is_inexact_array))

        @jax.jit
        def evaluate(params, observations, legal, scores, groups):
            nn_params, nn_state = params
            nn_params = jax.tree.unflatten(structure, jax.tree.leaves(nn_params))
            net = eqx.nn.inference_mode(eqx.combine(nn_params, static))
            batches = observations.reshape(-1, batch_size, *observations.shape[1:])
            logits, value = jax.lax.map(
                lambda x: apply_nn(net, nn_state, x)[0], batches
            )
            logits = logits.reshape(-1, logits.shape[-1])
            value = value.reshape(-1)
            return scored_position_metrics(
                logits, value, legal, scores, groups, prior_temperature
            )

        self._evaluate = evaluate

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "positions": self.num_positions,
            "groups": self.group_names[1:],
            "prior_temperature": self.prior_temperature,
            **self.extra_config,
        }

    def __call__(self, params: Any) -> dict[str, jax.Array]:
        """The metrics of the network with `params`, its (parameters, network state)."""
        table = self._evaluate(
            params, self.observations, self.legal, self.scores, self.groups
        )
        return {
            f"{self.name}_{metric}" + ("" if group == "all" else f"_{group}"): table[
                g, m
            ]
            for g, group in enumerate(self.group_names)
            for m, metric in enumerate(METRICS)
        }


class TwoPlayerValidator:
    """Plays `num_episodes` games of `evaluator`, with the weights it's called with, against
    `baseline_evaluator`. A `Trainer` validator: it returns `<name>_avg_outcome` (the mean
    final reward), `<name>_win_rate` and `<name>_loss_rate` (the rest are draws).

    The games' keys are drawn once from `seed`, so every call plays from the same starts with
    the same noise, and calls differ only by the weights. The agent moves first in exactly half
    of the games: with the turn order fixed across each half, each step runs only the
    evaluator whose turn it is."""

    def __init__(
        self,
        evaluator: Evaluator,
        baseline_evaluator: Evaluator,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        max_steps: int,
        num_episodes: int,
        name: str,
        baseline_params: Any = None,
        seed: int = 0,
    ):
        """
        Args:
            evaluator: the agent's evaluator, e.g. a search with the network's evaluation
            baseline_evaluator: the opponent's
            env_step_fn: (env_state, action) -> (new_env_state, metadata)
            env_init_fn: (key) -> (env_state, metadata), the games' starts
            max_steps: games still running after this many moves are scored as they stand
                (see `core.common.two_player_game`)
            num_episodes: games per call, an even number
            name: prefixes the metrics
            baseline_params: (optional) the opponent's parameters
            seed: draws the games' keys
        """
        if num_episodes % 2:
            raise ValueError(f"num_episodes must be even, got {num_episodes}")
        self.name = name
        self.num_episodes = num_episodes
        self.evaluator = evaluator
        self.baseline_evaluator = baseline_evaluator
        self.seed = seed
        keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
        baseline_params = jnp.array([]) if baseline_params is None else baseline_params
        game_fn = partial(
            two_player_game,
            evaluator_1=evaluator,
            evaluator_2=baseline_evaluator,
            params_2=baseline_params,
            env_step_fn=env_step_fn,
            env_init_fn=env_init_fn,
            max_steps=max_steps,
        )
        half = num_episodes // 2

        @jax.jit
        def play(params: Any) -> jax.Array:
            first, _, _ = jax.vmap(
                lambda k: game_fn(k, params_1=params, p1_first=True)
            )(keys[:half])
            second, _, _ = jax.vmap(
                lambda k: game_fn(k, params_1=params, p1_first=False)
            )(keys[half:])
            return jnp.concatenate([first, second])

        self._play = play

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "num_episodes": self.num_episodes,
            "seed": self.seed,
            "evaluator": self.evaluator.__class__.__name__,
            "evaluator_config": self.evaluator.get_config(),
            "baseline_evaluator": self.baseline_evaluator.__class__.__name__,
            "baseline_evaluator_config": self.baseline_evaluator.get_config(),
        }

    def __call__(self, params: Any) -> dict[str, jax.Array]:
        """The games' results with `params`, the agent's (parameters, network state)."""
        results = self._play(params)
        return {
            f"{self.name}_avg_outcome": results[:, 0].mean(),
            f"{self.name}_win_rate": (results[:, 0] > results[:, 1]).mean(),
            f"{self.name}_loss_rate": (results[:, 0] < results[:, 1]).mean(),
        }


class SearchTargetValidator:
    """Scores the policy targets self-play's search makes, at several budgets, against an
    engine on fixed positions (see `ScoredPositions`): the headroom of its labels. Each position
    is searched from a fresh tree by each of `evaluators` (e.g. self-play's search at N and 2N
    simulations, root noise included), and its visit distribution and root value are scored as
    `ScoredPositionsValidator` scores a network's raw outputs (see `scored_position_metrics`).
    Where a bigger budget's targets beat a smaller's, more search would teach more; where the
    network's raw scores have caught up with its own targets', its labels have little left to
    teach at that budget.

    A `Trainer` validator, costly (a search per position and budget, and one more at the first),
    so it runs every `every` epochs. It returns, for each evaluator `<k>`, `<name><k>_<metric>`
    for all positions and `<name><k>_<metric>_<group>` for each group; and, over the same
    groups, `<name>_kl_<k>_<first>`, the KL divergence of each later evaluator's targets from the
    first's (how much the extra search changes them); `<name>_kl_<first>_repeat`, of the first's
    from a second search of its own with other noise (the floor those differences sit on); and
    `<name>_kl_<first>_prior`, of the first's from the network's prior (how far search moves the
    policy). Each KL spreads 0.1% of the second distribution over the legal actions (see
    `core.training.selfplay_data.floored_kl`).

    The searches' keys are drawn once from `seed`, so calls differ only by the weights."""

    def __init__(
        self,
        positions: ScoredPositions,
        states: Any,
        metadata: StepMetadata,
        evaluators: Mapping[str, Evaluator],
        env_step_fn: EnvStepFn,
        nn: eqx.Module,
        every: int,
        batch_size: int = 1024,
        name: str = "search",
        seed: int = 0,
        config: dict | None = None,
    ):
        """
        Args:
            positions: the positions, as the network takes them, and the engine's scores
            states: the same positions as environment states, with a leading axis
            metadata: their metadata, with a leading axis
            evaluators: the searches, by the names their metrics take, e.g. {"128": ...,
                "256": ...}: the first is the one the others are compared with
            env_step_fn: (env_state, action) -> (new_env_state, metadata)
            nn: the network the searches evaluate with (only its structure is used, for the
                prior, as `ScoredPositionsValidator`'s)
            every: epochs between calls (see `core.training.train.Validator`)
            batch_size: positions searched at once
            name: prefixes the metrics
            seed: draws the searches' keys
            config: (optional) recorded with the validator's configuration
        """
        if not evaluators:
            raise ValueError("evaluators can't be empty")
        if every < 1:
            raise ValueError(f"every must be positive, got {every}")
        self.name = name
        self.every = every
        self.seed = seed
        self.batch_size = batch_size
        self.extra_config = config or {}
        self.evaluators = dict(evaluators)
        n = len(positions.legal)
        self.num_positions = n
        padded = -(-n // batch_size) * batch_size

        def pad(x: Any) -> Any:
            x = jnp.asarray(x)
            return jnp.concatenate([x, jnp.repeat(x[:1], padded - n, axis=0)])

        self.observations = pad(positions.observations)
        self.legal = pad(positions.legal)
        self.scores = pad(positions.scores)
        self.states = jax.tree.map(pad, states)
        self.metadata = jax.tree.map(pad, metadata)
        self.group_names = ["all", *positions.groups]
        groups = np.stack([np.ones(n, bool), *positions.groups.values()])
        self.groups = jnp.asarray(
            np.concatenate([groups, np.zeros((len(groups), padded - n), bool)], axis=1)
        )
        # a key per position for each search, the first evaluator's twice
        searches = [*self.evaluators, f"{next(iter(self.evaluators))}_repeat"]
        self.keys = dict(
            zip(
                searches,
                jax.random.split(jax.random.PRNGKey(seed), (len(searches), padded)),
                strict=True,
            )
        )
        static = eqx.filter(nn, eqx.is_inexact_array, inverse=True)

        @jax.jit
        def prior_logits(params: Any, observations: jax.Array) -> jax.Array:
            nn_params, nn_state = params
            net = eqx.nn.inference_mode(eqx.combine(nn_params, static))
            batches = observations.reshape(-1, batch_size, *observations.shape[1:])
            logits = jax.lax.map(lambda x: apply_nn(net, nn_state, x)[0][0], batches)
            return logits.reshape(-1, logits.shape[-1])

        self._prior_logits = prior_logits

        def make_search(evaluator: Evaluator):
            def one(params: Any, key: jax.Array, state: Any, metadata: StepMetadata):
                tree = evaluator.init(template_embedding=state)
                output = evaluator.evaluate(
                    key=key,
                    eval_state=tree,
                    env_state=state,
                    root_metadata=metadata,
                    params=params,
                    env_step_fn=env_step_fn,
                )
                return output.policy_weights, evaluator.get_value(output.eval_state)

            return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0)))

        self._searches = {k: make_search(e) for k, e in self.evaluators.items()}

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "positions": self.num_positions,
            "groups": self.group_names[1:],
            "every": self.every,
            "seed": self.seed,
            "evaluators": {
                k: {"class": e.__class__.__name__, **e.get_config()}
                for k, e in self.evaluators.items()
            },
            **self.extra_config,
        }

    def search(
        self, params: Any, evaluator: str, keys: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """`evaluator`'s targets (visit distributions) and root values on every position, with
        `keys`, a batch at a time."""
        search = self._searches[evaluator]
        targets, values = [], []
        for i in range(0, len(self.legal), self.batch_size):
            batch = slice(i, i + self.batch_size)
            target, value = search(
                params,
                keys[batch],
                jax.tree.map(lambda x, b=batch: x[b], self.states),
                jax.tree.map(lambda x, b=batch: x[b], self.metadata),
            )
            targets.append(target)
            values.append(value)
        target = jnp.where(self.legal, jnp.concatenate(targets).astype(jnp.float32), 0)
        target = target / jnp.maximum(target.sum(-1, keepdims=True), 1e-9)
        return target, jnp.concatenate(values).astype(jnp.float32)

    def __call__(self, params: Any) -> dict[str, jax.Array]:
        """The searches' metrics with `params`, the network's (parameters, network state)."""
        out: dict[str, jax.Array] = {}
        targets = {}
        for k in self.evaluators:
            target, value = self.search(params, k, self.keys[k])
            targets[k] = target
            # the target as logits: its most visited action is the one played greedily, and
            # its softmax is the target itself
            table = scored_position_metrics(
                jnp.log(jnp.maximum(target, 1e-30)),
                value,
                self.legal,
                self.scores,
                self.groups,
            )
            out.update(self._named(f"{self.name}{k}", table, METRICS))
        first = next(iter(self.evaluators))
        repeat, _ = self.search(params, first, self.keys[f"{first}_repeat"])
        prior = jax.nn.softmax(
            jnp.where(
                self.legal,
                self._prior_logits(params, self.observations).astype(jnp.float32),
                -jnp.inf,
            ),
            -1,
        )
        pairs = {
            **{
                f"kl_{k}_{first}": (targets[k], targets[first])
                for k in list(self.evaluators)[1:]
            },
            f"kl_{first}_repeat": (targets[first], repeat),
            f"kl_{first}_prior": (targets[first], prior),
        }
        columns = jnp.stack(
            [floored_kl(p, q, self.legal) for p, q in pairs.values()], -1
        )
        table = group_means(columns, jnp.ones(len(self.legal)), self.groups)
        out.update(self._named(self.name, table, list(pairs)))
        return out

    def _named(
        self, prefix: str, table: jax.Array, metrics: Any
    ) -> dict[str, jax.Array]:
        return {
            f"{prefix}_{metric}" + ("" if group == "all" else f"_{group}"): table[g, m]
            for g, group in enumerate(self.group_names)
            for m, metric in enumerate(metrics)
        }
