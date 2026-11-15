from dataclasses import dataclass

import jax
import jax.numpy as jnp

from core.evaluators.evaluator import EvalOutput
from core.types import StepMetadata


@dataclass(frozen=True)
class MoveTemperature:
    """A temperature for the move played that decays through the game, as KataGo's self-play
    (`chosenMoveTemperatureEarly` 0.75, `chosenMoveTemperature` 0.15 and
    `chosenMoveTemperatureHalflife` 19 moves on 19x19): `late + (early - late) * 0.5 ** (moves /
    halflife)`, where `moves` counts from `start`. It never reaches 0, so every move is sampled.

    Attributes:
        early: the temperature at `start`
        late: the temperature it decays towards
        halflife: moves over which its distance from `late` halves
        start: the move it starts decaying from (`early` before it); moves count as
            `StepMetadata.step` does, e.g. from the standard start including an opening's moves
    """

    early: float = 0.75
    late: float = 0.15
    halflife: float = 8.0
    start: int = 0

    def __post_init__(self):
        if not (0 < self.late <= self.early and self.halflife > 0 and self.start >= 0):
            raise ValueError(
                "MoveTemperature needs 0 < late <= early, a positive halflife and start "
                f"at least 0, got {self}"
            )

    def at(self, step: jax.Array) -> jax.Array:
        """The temperature at move `step`."""
        moves = jnp.maximum(step - self.start, 0)
        return self.late + (self.early - self.late) * 0.5 ** (moves / self.halflife)

    def get_config(self) -> dict:
        """Returns the schedule's settings. Used for logging."""
        return {
            "early": self.early,
            "late": self.late,
            "halflife": self.halflife,
            "start": self.start,
        }


@dataclass(frozen=True)
class Exploration:
    """Chooses the move data collection actually plays, given the evaluator's output.

    The evaluator's `policy_weights` (for MCTS, the root visit distribution) stay the
    policy training target either way: only the move played changes, so these settings
    change which positions collection reaches, not the targets it trains on there.

    Applied by the `Trainer` to data collection only; test games play the evaluator's own move.

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
            Dirichlet noise: this is epsilon-greedy exploration on top. Keeps collection visiting
            positions a sharpened policy never reaches on its own (e.g. after an opponent's
            blunder), at the cost of value targets that partly reflect the random moves.
        temperature: instead of `num_sampling_moves`' schedule, sample every move from the
            policy weights raised to 1 / `temperature.at(step)` (see `MoveTemperature`): KataGo's
            schedule, sharpening through the game without ever turning greedy. The policy weights
            are the evaluator's target (for MCTS by default the root's visits), whatever move it
            chose itself.
        store_until_offline: if set, a game stops storing positions for training once it has
            played a move other than the policy weights' top one at or after this move (counted as
            `StepMetadata.step` is); its later positions are played and count towards its outcome,
            but aren't stored. Separates what late off-line moves do to the outcomes stored before
            them (kept) from the lopsided positions they lead to (not stored). Applied by the
            `Trainer`.

    All are applied with traced arithmetic, so a schedule or adaptive controller can later
    pass them in as traced values. Changing the attributes of a `Trainer`'s instance between
    epochs has no effect: its data collection is compiled once, with these values baked in.
    """

    num_sampling_moves: int | None = None
    random_move_prob: float = 0.0
    temperature: MoveTemperature | None = None
    store_until_offline: int | None = None

    def __post_init__(self):
        if self.temperature is not None and self.num_sampling_moves is not None:
            raise ValueError("pass num_sampling_moves or temperature, not both")
        if self.store_until_offline is not None and self.store_until_offline < 0:
            raise ValueError("store_until_offline must be at least 0")

    def goes_offline(self, output: EvalOutput, metadata: StepMetadata) -> jax.Array:
        """Whether the move `output` played (its `action`, as `choose_action` chose it) from the
        position of `metadata` takes its game off line: with `store_until_offline`, a move at or
        after it that isn't the policy weights' top one (any of them, on ties). False without."""
        if self.store_until_offline is None:
            return jnp.array(False)
        weights = jnp.where(metadata.action_mask, output.policy_weights, -jnp.inf)
        top = weights[output.action] == weights.max()
        return (metadata.step >= self.store_until_offline) & ~top

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
        if self.temperature is not None:
            temperature = self.temperature.at(metadata.step)
            # in logs, so that a sharp temperature doesn't underflow the small weights to 0
            logits = jnp.where(
                mask & (output.policy_weights > 0),
                jnp.log(jnp.maximum(output.policy_weights, 1e-30)) / temperature,
                -jnp.inf,
            )
            # all-zero weights over the legal moves (no visits): uniform over them
            logits = jnp.where(
                jnp.isneginf(logits).all(), jnp.where(mask, 0.0, -jnp.inf), logits
            )
            action = jax.random.categorical(greedy_key, logits)
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
            "temperature": None
            if self.temperature is None
            else self.temperature.get_config(),
            "store_until_offline": self.store_until_offline,
        }
