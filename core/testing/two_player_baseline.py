from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from core.common import GameFrame, two_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn


class TwoPlayerBaseline(BaseTester):
    """Implements a tester that evaluates an agent against a baseline evaluator in a two-player game."""

    def __init__(
        self,
        num_episodes: int,
        baseline_evaluator: Evaluator,
        baseline_params: Any | None = None,
        *args,
        balance_first_player: bool = False,
        **kwargs,
    ):
        """Initializes a TwoPlayerBaseline tester.

        Args:
            num_episodes: number of episodes to evaluate against the baseline
            baseline_evaluator: the baseline evaluator to evaluate against
            baseline_params: (optional) the parameters of the baseline evaluator
            balance_first_player: the agent moves first in exactly half of the episodes, instead of
                a random half. Only the active player's evaluator runs at each step, which
                about halves the cost of a test, and the results don't vary with who happened to move
                first. Needs an even number of episodes.
        """
        super().__init__(*args, num_keys=num_episodes, **kwargs)
        if balance_first_player and num_episodes % 2 != 0:
            raise ValueError(
                f"{self.__class__.__name__}: balance_first_player needs an even number of episodes, got {num_episodes}"
            )
        self.num_episodes = num_episodes
        self.balance_first_player = balance_first_player
        self.baseline_evaluator = baseline_evaluator
        if baseline_params is None:
            baseline_params = jnp.array([])
        self.baseline_params = baseline_params

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
    ) -> tuple[TestState, dict, GameFrame, jax.Array]:
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

        game_fn = partial(
            two_player_game,
            evaluator_1=evaluator,
            evaluator_2=self.baseline_evaluator,
            params_1=params,
            params_2=self.baseline_params,
            env_step_fn=env_step_fn,
            env_init_fn=env_init_fn,
            max_steps=max_steps,
        )

        if self.balance_first_player:
            # the agent moves first in the first half, second in the rest: with the turn order
            # fixed across each half, each step only runs the evaluator whose turn it is
            half = keys.shape[0] // 2
            first, frames, p_ids = jax.vmap(partial(game_fn, p1_first=True))(
                keys[:half]
            )
            second, _, _ = jax.vmap(partial(game_fn, p1_first=False))(keys[half:])
            results = jnp.concatenate([first, second])
        else:
            results, frames, p_ids = jax.vmap(game_fn)(keys)
        frames = jax.tree.map(lambda x: x[0], frames)
        p_ids = p_ids[0]

        avg = results[:, 0].mean()

        metrics = {
            f"{self.name}_avg_outcome": avg,
            # the rest of the games are draws
            f"{self.name}_win_rate": (results[:, 0] > results[:, 1]).mean(),
            f"{self.name}_loss_rate": (results[:, 0] < results[:, 1]).mean(),
        }

        return state, metrics, frames, p_ids
