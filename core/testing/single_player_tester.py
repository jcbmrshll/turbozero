from functools import partial
from typing import Dict, Tuple

import chex
from chex import dataclass
import jax
import jax.numpy as jnp

from core.common import single_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn


@dataclass(frozen=True)
class SinglePlayerTestState(TestState):
    """Internal state of a SinglePlayerTester. Stores the best parameters found so far.
    - `best_params`: best performing parameters
    """
    best_params: chex.ArrayTree


class SinglePlayerTester(BaseTester):
    """Implements a tester that evaluates an agent against the best performing parameters 
    found so far in a two-player game."""
    def __init__(self, num_episodes: int, *args, **kwargs):
        """
        Args:
        - `num_episodes`: number of episodes to play in each test
        """
        super().__init__(*args, num_keys=num_episodes, **kwargs)
        self.num_episodes = num_episodes


    def init(self, params: chex.ArrayTree, **kwargs) -> SinglePlayerTestState: #pylint: disable=unused-argument
        """Initializes the internal state of the TwoPlayerTester.
        Args:
        - `params`: initial parameters to store as the best performing
            - can just be the initial parameters of the agent
        """
        return SinglePlayerTestState(best_params=params)
    

    def check_size_compatibilities(self, num_devices: int) -> None:
        """Checks if tester configuration is compatible with number of devices being utilized.
        
        Args:
        - `num_devices`: number of devices
        """
        if self.num_episodes % num_devices != 0:
            raise ValueError(f"{self.__class__.__name__}: number of episodes ({self.num_episodes}) must be divisible by number of devices ({num_devices})")


    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(self, max_steps: int, env_step_fn: EnvStepFn, env_init_fn: EnvInitFn, evaluator: Evaluator,
        keys: chex.PRNGKey, state: SinglePlayerTestState, params: chex.ArrayTree) -> Tuple[SinglePlayerTestState, Dict, chex.ArrayTree, chex.Array]:
        """Test the agent against the best performing parameters found so far in a single-player game.
        
        Args:
        - `max_steps`: maximum number of steps per episode
        - `env_step_fn`: environment step function
        - `env_init_fn`: environment initialization function
        - `evaluator`: the agent evaluator
        - `keys`: rng
        - `state`: internal state of the tester
        - `params`: nn parameters used by agent
        
        Returns:
        - (SinglePlayerTestState, Dict, chex.ArrayTree, chex.Array)
            - updated internal state of the tester
            - metrics from the test
            - frames from the test (used for rendering)
            - player ids from the test (used for rendering)
        """

        game_fn = partial(single_player_game,
            evaluator_1 = evaluator,
            evaluator_2 = evaluator,
            params_1 = params,
            params_2 = state.best_params,
            env_step_fn = env_step_fn,
            env_init_fn = env_init_fn,
            max_steps = max_steps
        )

        results, frames, p_ids = jax.vmap(game_fn)(keys)
        frames = jax.tree_map(lambda x: x[0], frames)
        p_ids = p_ids[0]
        
        num = keys.shape[-2]
        new = jnp.sum(results[...,0,:]) / num
        best = jnp.sum(results[...,1,:]) / num
        avg = new/best

        metrics = {
            f"{self.name}_new_params": new,
            f"{self.name}_best_params": best,
            f"{self.name}_ratio": avg,
        }

        best_params = jax.lax.cond(
            new > best,
            lambda _: params,
            lambda _: state.best_params,
            None
        )

        return state.replace(best_params=best_params), metrics, frames, p_ids
