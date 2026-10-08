from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax

from core.evaluators.evaluator import Evaluator
from core.types import EnvInitFn, EnvStepFn


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TestState:
    """Base class for TestState."""


class BaseTester:
    """Base class for Testers.

    A Tester is used to evaluate the performance of an agent in an environment,
    in some cases against one or more opponents.

    A Tester may maintain its own internal state.
    """

    def __init__(
        self,
        num_keys: int,
        epochs_per_test: int = 1,
        episode_fn: Callable | None = None,
        name: str | None = None,
    ):
        """Initializes a Tester.

        Args:
            num_keys: number of keys to use for tester
                - often equal to number of episodes
            epochs_per_test: number of epochs between each test
            episode_fn: (optional) packs the first episode of each test for the monitor to render, as
                `episode_fn(frames, p_ids)`: the episode's `GameFrame`s stacked along a leading time axis
                (`max_steps + 1` of them) and its player ids, both as numpy arrays.
                - e.g. `core.monitor.renderers.pgx_two_player_episode()`: the monitor server draws the
                  episode, so the training loop never renders anything
            name: (optional) name of the tester (used for logging and differentiating between testers)
                - defaults to the class name
        """
        self.num_keys = num_keys
        self.epochs_per_test = epochs_per_test
        self.episode_fn = episode_fn
        if name is None:
            name = self.__class__.__name__
        self.name = name

    def init(self, *args, **kwargs) -> TestState:  # pylint: disable=unused-argument
        """Initializes the internal state of the Tester."""
        return TestState()

    def split_keys(self, key: jax.Array) -> jax.Array:
        """Splits a key into `num_keys` keys, one per episode.

        Args:
            key: rng

        Returns:
            jax.Array: `num_keys` keys
        """
        return jax.random.split(key, self.num_keys)

    def run(
        self,
        key: jax.Array,
        epoch_num: int,
        max_steps: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        state: TestState,
        params: Any,
        *args,
        log_fn: Callable[[dict], None] | None = None,  # pylint: disable=unused-argument
        activity_fn: Callable[[str], None] | None = None,  # pylint: disable=unused-argument
    ) -> tuple[TestState, dict, Any]:
        """Runs the test, if the current epoch is an epoch that should be tested on (i.e. `epoch_num % epochs_per_test == 0`).

        If an `episode_fn` is provided, packs the first episode of the test for the monitor.

        Args:
            key: rng
            epoch_num: current epoch number
            max_steps: maximum number of steps per episode
            env_step_fn: environment step function
            env_init_fn: environment initialization function
            evaluator: evaluator used by agent
            state: internal state of the tester
            params: nn parameters used by agent
            log_fn: (optional) logs metrics right away, for testers that run in stages and have
                results before they finish; whatever they log this way they don't also return
            activity_fn: (optional) reports what a long test is doing now, e.g. which opponent
                it's playing

        Returns:
            Tuple[TestState, Dict, Any]:
                - updated internal state of the tester
                - metrics from the test
                - the first episode of the test, packed by `episode_fn` (None without one)
                - on epochs that are not tested, returns `state` unchanged, empty metrics, and None
        """
        keys = self.split_keys(key)

        if epoch_num % self.epochs_per_test == 0:
            # run test
            state, metrics, frames, p_ids = self.test(
                max_steps, env_step_fn, env_init_fn, evaluator, keys, state, params
            )

            if self.episode_fn is not None:
                # the first episode, copied off the device in one transfer: the initial state,
                # then one frame per step
                frames, p_ids = jax.device_get((frames, p_ids))
                episode = self.episode_fn(frames, p_ids)
            else:
                episode = None
            return state, metrics, episode
        return state, {}, None

    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def test(
        self,
        max_steps: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        keys: jax.Array,
        state: TestState,
        params: Any,
    ) -> tuple[TestState, dict, Any, jax.Array]:
        """Run the test implemented by the Tester.

        Implemented by subclasses, which should jit it (with `self` and the functions as static arguments).

        Args:
            max_steps: maximum number of steps per episode
            env_step_fn: environment step function
            env_init_fn: environment initialization function
            evaluator: evaluator used by agent
            keys: rng, one key per episode
            state: internal state of the tester
            params: nn parameters used by agent

        Returns:
            Tuple[TestState, Dict, Any, jax.Array]:
                - updated internal state of the tester
                - metrics from the test
                - frames from the first episode of the test (used to produce renderings)
                - player ids from the first episode of the test (used to produce renderings)
        """
        raise NotImplementedError()
