from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from core.networks.utils import apply_nn
from core.testing.tester import BaseTester, TestState

# what `scored_position_metrics` reports, for all positions and for each group
METRICS = ("top1", "regret", "exp_regret", "value_mse", "value_sign")


@dataclass(frozen=True)
class ScoredPositions:
    """Fixed positions with a strong engine's score for every legal action, to measure a network
    against (see `ScoredPositionsTester`).

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
) -> jax.Array:
    """How a network's raw outputs (no search) measure up to an engine's scores of positions'
    actions, for each of several groups of the positions:
    - `top1`: how often its most likely action is one of the engine's best
    - `regret`: how much score its most likely action gives up against the engine's best (mean)
    - `exp_regret`: the same, expected under its policy: smoother, so less noisy
    - `value_mse`: (value - z)^2, z the sign of the engine's best score (its predicted outcome)
    - `value_sign`: how often its value has z's sign, over positions where z isn't 0

    Args:
        logits: (N, A) policy logits
        value: (N,) values
        legal: (N, A) bool, legal actions
        scores: (N, A) the engine's scores, higher better (ignored where illegal)
        groups: (G, N) bool, which positions each group has

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
    per_position = jnp.stack(
        [
            (regret == 0).astype(jnp.float32),
            regret,
            exp_regret,
            (value - z) ** 2,
            (jnp.sign(value) == z).astype(jnp.float32),
        ],
        axis=1,
    )
    # value_sign only counts the positions the engine doesn't call a draw
    weights = groups.astype(jnp.float32)
    counts = jnp.stack(
        [weights.sum(axis=1)] * 4 + [(weights * decided).sum(axis=1)], axis=1
    )
    sums = jnp.stack(
        [weights @ per_position[:, i] for i in range(4)]
        + [(weights * decided) @ per_position[:, 4]],
        axis=1,
    )
    return sums / jnp.maximum(counts, 1)


class ScoredPositionsTester(BaseTester):
    """Measures the network against an engine on fixed positions (see `ScoredPositions` and
    `scored_position_metrics`): a forward pass over them, no games, so it's cheap enough to run
    every epoch. Logs `<name>_<metric>` for all positions and `<name>_<metric>_<group>` for each
    group."""

    def __init__(
        self,
        positions: ScoredPositions,
        nn: eqx.Module,
        batch_size: int = 1024,
        epochs_per_test: int = 1,
        name: str = "engine",
        config: dict | None = None,
    ):
        """
        Args:
            positions: the positions and the engine's scores
            nn: the network (an equinox module, see core.networks.utils.apply_nn); only its
                structure is used, the parameters it's measured with are passed to `run`
            batch_size: positions per forward pass
            epochs_per_test: epochs between measurements
            name: prefixes the metrics
            config: (optional) recorded with the tester's configuration, e.g. where the
                positions came from
        """
        super().__init__(num_keys=1, epochs_per_test=epochs_per_test, name=name)
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
        static = eqx.filter(nn, eqx.is_inexact_array, inverse=True)

        @jax.jit
        def evaluate(params, observations, legal, scores, groups):
            nn_params, nn_state = params
            net = eqx.nn.inference_mode(eqx.combine(nn_params, static))
            batches = observations.reshape(-1, batch_size, *observations.shape[1:])
            logits, value = jax.lax.map(
                lambda x: apply_nn(net, nn_state, x)[0], batches
            )
            logits = logits.reshape(-1, logits.shape[-1])
            value = value.reshape(-1)
            return scored_position_metrics(logits, value, legal, scores, groups)

        self._evaluate = evaluate

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "positions": self.num_positions,
            "groups": self.group_names[1:],
            **self.extra_config,
        }

    def evaluate(self, params: Any) -> dict[str, jax.Array]:
        """The metrics of the network with `params` ((nn_params, nn_state), as
        core.training.train.extract_params gives them)."""
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

    def run(
        self,
        key: jax.Array,
        epoch_num: int,
        max_steps: int,
        env_step_fn: Any,
        env_init_fn: Any,
        evaluator: Any,
        state: TestState,
        params: Any,
        *args,
        **kwargs,
    ) -> tuple[TestState, dict, None]:
        """Measures the network every `epochs_per_test` epochs (see `BaseTester.run`); plays no
        games, so the episode is always None."""
        if epoch_num % self.epochs_per_test != 0:
            return state, {}, None
        return state, self.evaluate(params), None
