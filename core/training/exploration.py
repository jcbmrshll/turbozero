from dataclasses import dataclass

import jax
import jax.numpy as jnp

from core.evaluators.evaluator import EvalOutput
from core.types import StepMetadata


@dataclass(frozen=True)
class SelfPlayExploration:
    """Chooses the move self-play actually plays, given the evaluator's output.

    The evaluator's `policy_weights` (for MCTS, the root visit distribution) stay the
    policy training target either way: only the move played changes, so these settings
    change which positions self-play reaches, not the targets it trains on there.

    Applied by the `Trainer` to self-play only; test games play the evaluator's own move.

    Attributes:
        num_sampling_moves: play the evaluator's move for the first `num_sampling_moves`
            moves of an episode, then the action with the highest policy weight (ties broken
            at random). With an MCTS evaluator at temperature 1 this is AlphaZero's schedule
            (`num_sampling_moves = 30` in its pseudocode): sample in proportion to visit counts
            early, play the most-visited move after.
            It explores *less* than sampling every move, trading late-game variety for value
            targets that reflect good play. None (the default) plays the evaluator's move
            throughout.
        random_move_prob: probability of playing a uniformly random legal move instead,
            on every move. Not part of AlphaZero, whose only other source of variety is root
            Dirichlet noise: this is epsilon-greedy exploration on top. Keeps self-play visiting
            positions a sharpened policy never reaches on its own (e.g. after an opponent's
            blunder), at the cost of value targets that partly reflect the random moves.

    Both are applied with traced arithmetic, so a schedule or adaptive controller can later
    pass them in as traced values. Changing the attributes of a `Trainer`'s instance between
    epochs has no effect: its self-play is compiled once, with these values baked in.
    """

    num_sampling_moves: int | None = None
    random_move_prob: float = 0.0

    def choose_action(
        self, key: jax.Array, output: EvalOutput, metadata: StepMetadata
    ) -> jax.Array:
        """Chooses the move to play.

        Args:
            key: rng
            output: the evaluator's output for the current environment state
            metadata: metadata of the current environment state; `step` counts the moves
                played so far this episode, `action_mask` marks the legal actions

        Returns:
            jax.Array: the action to play
        """
        greedy_key, random_key, explore_key = jax.random.split(key, 3)
        mask = metadata.action_mask
        action = output.action
        if self.num_sampling_moves is not None:
            weights = jnp.where(mask, output.policy_weights, -jnp.inf)
            best = mask & (weights == weights.max())
            greedy = jax.random.choice(greedy_key, mask.shape[-1], p=best / best.sum())
            action = jnp.where(metadata.step < self.num_sampling_moves, action, greedy)
        random_action = jax.random.choice(
            random_key, mask.shape[-1], p=mask / mask.sum()
        )
        explore = jax.random.uniform(explore_key) < self.random_move_prob
        return jnp.where(explore, random_action, action)

    def get_config(self) -> dict:
        """Returns the configuration of the exploration settings. Used for logging."""
        return {
            "num_sampling_moves": self.num_sampling_moves,
            "random_move_prob": self.random_move_prob,
        }
