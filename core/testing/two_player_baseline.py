
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from core.common import GameFrame, two_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn


class TwoPlayerBaseline(BaseTester):
    """Implements a tester that evaluates an agent against a baseline evaluator in a two-player game."""

    def __init__(self, num_episodes: int, baseline_evaluator: Evaluator, baseline_params: Optional[Any] = None, 
                 *args, **kwargs):
        """Initializes a TwoPlayerBaseline tester.

        Args:
            num_episodes: number of episodes to evaluate against the baseline
            baseline_evaluator: the baseline evaluator to evaluate against
            baseline_params: (optional) the parameters of the baseline evaluator
        """
        super().__init__(num_keys=num_episodes, *args, **kwargs)
        self.num_episodes = num_episodes
        self.baseline_evaluator = baseline_evaluator
        if baseline_params is None:
            baseline_params = jnp.array([])
        self.baseline_params = baseline_params
        

    def check_size_compatibilities(self, num_devices: int) -> None:
        """Checks if tester configuration is compatible with number of devices being utilized.

        Args:
            num_devices: number of devices
        """
        if self.num_episodes % num_devices != 0:
            raise ValueError(f"{self.__class__.__name__}: number of episodes ({self.num_episodes}) must be divisible by number of devices ({num_devices})")


    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(self, max_steps: int, env_step_fn: EnvStepFn, env_init_fn: EnvInitFn, evaluator: Evaluator,
        keys: jax.Array, state: TestState, params: Any) -> Tuple[TestState, Dict, GameFrame, jax.Array]:
        """Test the agent against the baseline evaluator in a two-player game.

        Args:
            max_steps: maximum number of steps per episode
            env_step_fn: environment step function
            env_init_fn: environment initialization function
            evaluator: the agent evaluator
            keys: rng
            state: internal state of the tester
            params: nn parameters used by agent

        Returns:
            Tuple[TestState, Dict, GameFrame, jax.Array]:
                - updated internal state of the tester
                - metrics from the test
                - frames from the first episode of the test
                - player ids from the first episode of the test
        """

        game_fn = partial(two_player_game,
            evaluator_1 = evaluator,
            evaluator_2 = self.baseline_evaluator,
            params_1 = params,
            params_2 = self.baseline_params,
            env_step_fn = env_step_fn,
            env_init_fn = env_init_fn,
            max_steps = max_steps
        )

        results, frames, p_ids = jax.vmap(game_fn)(keys)
        frames = jax.tree.map(lambda x: x[0], frames)
        p_ids = p_ids[0]
        
        avg = results[:, 0].mean()

        metrics = {
            f"{self.name}_avg_outcome": avg
        }

        return state, metrics, frames, p_ids
