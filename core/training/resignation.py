from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResignState:
    """One environment's resignation state (see `Resignation`).

    Attributes:
        calibrating: whether the game in progress is a calibration game, played to the end
        first_below: the player whose search value first fell to the threshold this game, -1 if
            none has
        resigned: games ended by resignation so far
        calibration_games: calibration games played to the end so far
        calibration_below: those in which a player's search value fell to the threshold
        calibration_false: those in which that player then didn't lose
    """

    calibrating: jax.Array
    first_below: jax.Array
    resigned: jax.Array
    calibration_games: jax.Array
    calibration_below: jax.Array
    calibration_false: jax.Array


@dataclass(frozen=True)
class Resignation:
    """AlphaGo Zero's resignation in self-play: the player to move resigns, ending the game as a
    loss, when its search's root value is at most `-threshold`.

    The resigning position is stored with its search's policy target, and every position of the
    game gets the resignation's result: -1 for the player who resigned, +1 for the other (two-player
    zero-sum games only). The environment starts a new game straight away, so self-play's compute
    per epoch stays the same and goes into more games.

    A `calibration_fraction` of games, drawn as each starts, never resign. They measure how often
    resigning would have been wrong: of those in which a player's value fell to the threshold, the
    fraction that player didn't go on to lose (AlphaGo Zero kept this under 5%, with 10% of games
    played out; AlphaZero played out 20%).

    Attributes:
        threshold: resign at a root value of `-threshold` or below, e.g. 0.9 for an expected score
            of 5% with values in [-1, 1]
        calibration_fraction: the fraction of games played to the end
    """

    threshold: float
    calibration_fraction: float = 0.1

    def __post_init__(self):
        if not 0 < self.threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if not 0 <= self.calibration_fraction <= 1:
            raise ValueError(
                f"calibration_fraction must be in [0, 1], got {self.calibration_fraction}"
            )

    def init(self, key: jax.Array) -> ResignState:
        """The state of one environment, whose first game starts now."""
        zero = jnp.array(0, dtype=jnp.int32)
        return ResignState(
            calibrating=self.draw_calibrating(key),
            first_below=zero - 1,
            resigned=zero,
            calibration_games=zero,
            calibration_below=zero,
            calibration_false=zero,
        )

    def draw_calibrating(self, key: jax.Array) -> jax.Array:
        """Whether a game starting now is a calibration game."""
        return jax.random.uniform(key) < self.calibration_fraction

    def resigns(self, state: ResignState, value: jax.Array) -> jax.Array:
        """Whether the player to move resigns, given its search's root `value`."""
        return ~state.calibrating & (value <= -self.threshold)

    def update(
        self,
        key: jax.Array,
        state: ResignState,
        moved: jax.Array,
        value: jax.Array,
        player: jax.Array,
        resigned: jax.Array,
        terminated: jax.Array,
        ended: jax.Array,
        rewards: jax.Array,
    ) -> ResignState:
        """The state after a move's search (one environment).

        Args:
            key: rng, to draw whether the next game calibrates if this one ended
            state: the state before the move
            moved: whether a move was played (if not, the state is left as it is)
            value: its search's root value, for the player to move
            player: the player to move
            resigned: whether that player resigned
            terminated: whether the move ended the game at a terminal state
            ended: whether the game ended, by resignation, termination or truncation
            rewards: the terminal state's rewards, by player (used if `terminated`)

        Returns:
            ResignState: the state for the next move
        """
        below = moved & (value <= -self.threshold) & (state.first_below < 0)
        first_below = jnp.where(below, player, state.first_below)
        # a calibration game that reached its end, scored against its first player at the threshold
        scored = moved & state.calibrating & terminated
        was_below = first_below >= 0
        lost = rewards[jnp.maximum(first_below, 0)] < 0
        ended = moved & ended
        return ResignState(
            calibrating=jnp.where(ended, self.draw_calibrating(key), state.calibrating),
            first_below=jnp.where(ended, -1, first_below),
            resigned=state.resigned + (moved & resigned),
            calibration_games=state.calibration_games + scored,
            calibration_below=state.calibration_below + (scored & was_below),
            calibration_false=state.calibration_false + (scored & was_below & ~lost),
        )

    def get_config(self) -> dict:
        """Returns the resignation settings. Used for logging."""
        return {
            "threshold": self.threshold,
            "calibration_fraction": self.calibration_fraction,
        }
