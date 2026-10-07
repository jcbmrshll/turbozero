from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax

from core.common import two_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TwoPlayerTestState(TestState):
    """Internal state of a TwoPlayerTester. Stores the best parameters found so far.

    Attributes:
        best_params: best performing parameters
    """

    best_params: Any


class TwoPlayerTester(BaseTester):
    """Implements a tester that evaluates an agent against the best performing parameters found so far in a two-player game."""

    def __init__(self, num_episodes: int, *args, **kwargs):
        """Initializes a TwoPlayerTester.

        Args:
            num_episodes: number of episodes to play in each test
        """
        super().__init__(*args, num_keys=num_episodes, **kwargs)
        self.num_episodes = num_episodes

    def init(self, params: Any, **kwargs) -> TwoPlayerTestState:  # pylint: disable=unused-argument
        """Initializes the internal state of the TwoPlayerTester.

        Args:
            params: initial parameters to store as the best performing
                - can just be the initial parameters of the agent
        """
        return TwoPlayerTestState(best_params=params)

    def check_size_compatibilities(self, num_devices: int) -> None:
        """Checks if tester configuration is compatible with number of devices being utilized.

        Args:
            num_devices: number of devices
        """
        if self.num_episodes % num_devices != 0:
            raise ValueError(
                f"{self.__class__.__name__}: number of episodes ({self.num_episodes}) must be divisible by number of devices ({num_devices})"
            )

    @partial(jax.pmap, axis_name="d", static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(
        self,
        max_steps: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        keys: jax.Array,
        state: TwoPlayerTestState,
        params: Any,
    ) -> tuple[TwoPlayerTestState, dict, Any, jax.Array]:
        """Test the agent against the best performing parameters found so far in a two-player game.

        Args:
            max_steps: maximum number of steps per episode
            env_step_fn: environment step function
            env_init_fn: environment initialization function
            evaluator: the agent evaluator
            keys: rng
            state: internal state of the tester
            params: nn parameters used by agent

        Returns:
            Tuple[TwoPlayerTestState, Dict, Any, jax.Array]:
                - updated internal state of the tester
                - metrics from the test
                - frames from the test (used for rendering)
                - player ids from the test (used for rendering)
        """

        game_fn = partial(
            two_player_game,
            evaluator_1=evaluator,
            evaluator_2=evaluator,
            params_1=params,
            params_2=state.best_params,
            env_step_fn=env_step_fn,
            env_init_fn=env_init_fn,
            max_steps=max_steps,
        )

        results, frames, p_ids = jax.vmap(game_fn)(keys)
        frames = jax.tree.map(lambda x: x[0], frames)
        p_ids = p_ids[0]

        avg = results[:, 0].mean()

        metrics = {f"{self.name}_avg_outcome": avg}

        # decide on the mean across all devices so best_params stays identical on every device
        best_params = jax.lax.cond(
            jax.lax.pmean(avg, axis_name="d") > 0.0,
            lambda _: params,
            lambda _: state.best_params,
            None,
        )

        return replace(state, best_params=best_params), metrics, frames, p_ids
