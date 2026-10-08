import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.types import EnvInitFn, EnvStepFn


@dataclass(frozen=True)
class Rung:
    """One opponent on a `LadderTester`'s ladder.

    Attributes:
        name: name of the opponent, used in metric names
        evaluator: the opponent's evaluator
        params: (optional) parameters of the opponent's evaluator
    """

    name: str
    evaluator: Evaluator
    params: Any = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LadderTestState(TestState):
    """Internal state of a LadderTester.

    Attributes:
        rung: index of the rung the agent plays next: the number of rungs it has beaten so far
    """

    rung: jax.Array


class LadderTester(BaseTester):
    """Implements a tester that climbs a ladder of two-player baselines, ordered from easiest to hardest.

    The agent moves first in exactly half the games against each rung (see `TwoPlayerBaseline`'s
    `balance_first_player`).

    Each test plays the agent against the lowest rung it hasn't beaten yet. Once it beats that rung
    (scores at least `promote_score`, counting a draw as half a win) it moves up, and with `climb`
    plays the next rung in the same test, until it fails to beat one or runs out of rungs. Rungs
    are never revisited.
    """

    def __init__(
        self,
        num_episodes: int,
        rungs: Sequence[Rung],
        promote_score: float = 0.55,
        climb: bool = True,
        episode_fn: Callable | None = None,
        name: str | None = "ladder",
        **kwargs,
    ):
        """Initializes a LadderTester.

        Args:
            num_episodes: number of episodes to play against each rung, in each test
            rungs: the opponents, ordered from easiest to hardest
            promote_score: score (wins plus half the draws, as a fraction of games) the agent needs
                against a rung to move past it
            climb: when the agent beats a rung, play the next one in the same test
            episode_fn: (optional) packs the first episode against each rung played for the monitor
                to render, see `BaseTester`
            name: (optional) name of the tester, prefixes its metrics
        """
        super().__init__(
            num_keys=num_episodes, episode_fn=episode_fn, name=name, **kwargs
        )
        if not rungs:
            raise ValueError(f"{self.__class__.__name__}: needs at least one rung")
        self.num_episodes = num_episodes
        self.rungs = list(rungs)
        self.promote_score = promote_score
        self.climb = climb
        # each rung is played with its own TwoPlayerBaseline, compiled the first time it's reached
        self.baselines = [
            TwoPlayerBaseline(
                num_episodes=num_episodes,
                baseline_evaluator=rung.evaluator,
                baseline_params=rung.params,
                balance_first_player=True,
                name=rung.name,
            )
            for rung in self.rungs
        ]

    def init(self, *args, **kwargs) -> LadderTestState:  # pylint: disable=unused-argument
        """Initializes the internal state of the LadderTester: at the bottom of the ladder."""
        return LadderTestState(rung=jnp.array(0, dtype=jnp.int32))

    def check_size_compatibilities(self, num_devices: int) -> None:
        """Checks if tester configuration is compatible with number of devices being utilized.

        Args:
            num_devices: number of devices
        """
        # every rung plays the same number of episodes, half with the agent moving first
        self.baselines[0].check_size_compatibilities(num_devices)

    def run(
        self,
        key: jax.Array,
        epoch_num: int,
        max_steps: int,
        num_devices: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        state: TestState,
        params: Any,
        *args,
        log_fn: Callable[[dict], None] | None = None,
        activity_fn: Callable[[str], None] | None = None,
    ) -> tuple[LadderTestState, dict, Any]:
        """Plays the agent against the current rung, moving up the ladder for each rung it beats.

        Args:
            key: rng
            epoch_num: current epoch number
            max_steps: maximum number of steps per episode
            num_devices: number of devices
            env_step_fn: environment step function
            env_init_fn: environment initialization function
            evaluator: evaluator used by agent
            state: internal state of the tester, replicated across devices
            params: nn parameters used by agent
            log_fn: (optional) logs each rung's metrics as soon as it's played, instead of
                returning them all at the end
            activity_fn: (optional) told which rung is being played, before each one

        Returns:
            Tuple[LadderTestState, Dict, Any]:
                - updated internal state of the tester
                - metrics: `{name}_rung` (rungs beaten so far), and for each rung played
                  `{name}_{rung name}_{score, win_rate, loss_rate, seconds}` (unless already
                  logged with `log_fn`)
                - the first episode against the last rung played, packed by `episode_fn` (None without one)
                - on epochs that are not tested, returns `state` unchanged, empty metrics, and None
        """
        assert isinstance(state, LadderTestState)
        if epoch_num % self.epochs_per_test != 0:
            return state, {}, None

        rung = int(np.asarray(state.rung).reshape(-1)[0])
        metrics = {}
        episode = None
        while rung < len(self.rungs):
            rung_key, key = jax.random.split(key)
            baseline = self.baselines[rung]
            if activity_fn is not None:
                activity_fn(
                    f"{self.name}: playing {baseline.name} (rung {rung + 1}/{len(self.rungs)}, "
                    f"{self.num_episodes} games)"
                )
            start = time.perf_counter()
            keys = baseline.split_keys(rung_key, num_devices)
            _, rung_metrics, frames, p_ids = baseline.test(
                max_steps,
                env_step_fn,
                env_init_fn,
                evaluator,
                keys,
                # a TwoPlayerBaseline keeps no state of its own
                TestState(),
                params,
            )
            win_rate = float(rung_metrics[f"{baseline.name}_win_rate"].mean())
            loss_rate = float(rung_metrics[f"{baseline.name}_loss_rate"].mean())
            # a draw counts as half a win
            score = (1.0 + win_rate - loss_rate) / 2
            prefix = f"{self.name}_{baseline.name}"
            rung_metrics = {
                f"{prefix}_score": np.float32(score),
                f"{prefix}_win_rate": np.float32(win_rate),
                f"{prefix}_loss_rate": np.float32(loss_rate),
                # includes compiling, the first time a rung is played
                f"{prefix}_seconds": np.float32(time.perf_counter() - start),
            }
            if log_fn is not None:
                log_fn(rung_metrics)
            else:
                metrics.update(rung_metrics)
            if self.episode_fn is not None:
                frames, p_ids = jax.device_get(
                    (jax.tree.map(lambda x: x[0], frames), p_ids[0])
                )
                episode = self.episode_fn(frames, p_ids)
            if score < self.promote_score:
                break
            rung += 1
            if not self.climb:
                break

        metrics[f"{self.name}_rung"] = np.float32(rung)
        new_state = LadderTestState(rung=jnp.full_like(state.rung, rung))
        return new_state, metrics, episode
