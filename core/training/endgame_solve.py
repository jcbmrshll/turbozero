"""Exact endgame labels for self-play (see `Trainer`'s `endgame_solve`): once a game reaches a
position an exact solver can solve (e.g. Othello's with 20 or fewer empty squares, its solve
point), every position of the game before it gets the solved result as its value target, instead
of the result its players went on to get. The game stops there, or plays on a few moves, each
position stored past the solve point getting its own solved result. Self-play skips the searches
for the rest of the game, and the labels lose the noise of its endgame mistakes.

The solver runs on the host, a batch at a time, while the device plays: the solves waiting at
the end of an epoch are solved during the next epoch's self-play, and their positions are
labelled after it (see `Trainer._train_loop`). Until then, the positions are in the replay
buffer, unpopulated so that training doesn't sample them, and the positions to solve are kept in
an environment's `PendingSolves`.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PendingSolves:
    """One environment's solves waiting for their results (see `EndgameSolve`), in a ring of
    slots: the `added`-th solve is in slot `added % slots`. A solve is a position to solve and the
    replay buffer entries its result labels: a game's entries up to its solve point, or one
    position stored past it.

    Attributes:
        positions: the positions to solve, environment states with a leading slot dimension
        whole: (slots,) whether each solve is a game's, at its solve point (its result is the
            game's), rather than a position's past it
        added: solves added so far
        labelled: solves labelled so far: those from `labelled` to `added` are waiting
        entry_solve: (replay buffer capacity,) for each buffer entry, the number of the solve
            (counting from 0, as `added` does) whose result it waits for; -1 for none
        stops: whether the game in progress stops for solves (see `EndgameSolve.share`)
        past: whether the game in progress has passed its solve point
        missed: solves that found every slot full (see `EndgameSolve.slots`)
    """

    positions: Any
    whole: jax.Array
    added: jax.Array
    labelled: jax.Array
    entry_solve: jax.Array
    stops: jax.Array
    past: jax.Array
    missed: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Move:
    """What `EndgameSolve.move` decided for a move about to be played (an environment's).

    Attributes:
        solve: the position is solved: it labels the game's entries since the last solve, and
            its own if it's stored
        whole: that solve is the game's, at its solve point
        end: the game ends before the move
        drop: the game's entries since the last solve are discarded, unlabelled (a solve past the
            solve point found every slot full)
        missed: a solve found every slot full
    """

    solve: jax.Array
    whole: jax.Array
    end: jax.Array
    drop: jax.Array
    missed: jax.Array


@dataclass(frozen=True)
class Solved:
    """Solves done on the host (see `EndgameSolve.solve`).

    Attributes:
        results: (environments, slots, players) the results, by the slots the solves are in
            (zeros in the others)
        counts: (environments,) how many solves each environment had waiting: its oldest, those
            `EndgameSolve.label` labels with these results
        solves: (solves, players) the results, a solve at a time
        whole: (solves,) whether each is a game's, at its solve point
        seconds: the solve's wall time
    """

    results: np.ndarray
    counts: np.ndarray
    solves: np.ndarray
    whole: np.ndarray
    seconds: float = 0.0


@dataclass(frozen=True)
class EndgameSolve:
    """Exact endgame labels for self-play's games (see the module's docstring).

    A game's solve point is the first position `stop_fn` marks: that position is searched and
    stored like any other, then solved, and its result labels every stored position of the game
    up to it. With `play_past` 0, the game ends there, before its move. Otherwise it plays on,
    ending before each further move with probability 1 / (1 + `play_past`), so a geometric number
    of moves past the solve point, `play_past` on average; each position it stores there is
    solved on its own, and labelled with its own result. A game played to its end past the solve
    point needs nothing more. The environment starts a new game as one ends, so self-play's
    compute per epoch stays the same and goes into more games. Games count as episodes, and in
    `episode_metrics_fn`'s metrics, when their solve point is labelled, with its result.

    Attributes:
        stop_fn: (environment state) -> whether it's at or past the solve point; compatible with
            `jax.jit`
        solve_fn: on the host, a batch of positions, environment states as a pytree of numpy
            arrays with a leading batch dimension -> (batch, players) each one's exact result for
            each player, as the environment's rewards for the game's end would be
        slots: solves each environment can hold while they wait for their results. Results come
            once an epoch, for the solves added before its self-play, so this needs two epochs'
            worth. A game whose solve point finds every slot full is played to the end instead,
            unsolved; one past it ends there, its unsolved position dropped (both counted in
            `solve_missed`, see `Trainer.collection_metrics`)
        share: the fraction of games solved, each drawn as it starts; the others are played to
            the end, as without solving
        play_past: the mean number of moves a game plays past its solve point (see above)
    """

    stop_fn: Callable[[Any], jax.Array]
    solve_fn: Callable[[Any], np.ndarray]
    slots: int
    share: float = 1.0
    play_past: float = 0.0

    def __post_init__(self):
        if self.slots < 1:
            raise ValueError(f"slots must be positive, got {self.slots}")
        if not 0 <= self.share <= 1:
            raise ValueError(f"share must be in [0, 1], got {self.share}")
        if self.play_past < 0:
            raise ValueError(f"play_past must be at least 0, got {self.play_past}")

    def get_config(self) -> dict:
        return {"slots": self.slots, "share": self.share, "play_past": self.play_past}

    def init(self, key: jax.Array, env_state: Any, capacity: int) -> PendingSolves:
        """An environment's `PendingSolves`, with nothing waiting.

        Args:
            key: rng, to draw whether the game in progress is solved
            env_state: an environment state (one environment's), for the positions' shapes
            capacity: the replay buffer's capacity, entries per environment
        """
        return PendingSolves(
            positions=jax.tree.map(
                lambda x: jnp.zeros((self.slots, *jnp.shape(x)), jnp.asarray(x).dtype),
                env_state,
            ),
            whole=jnp.zeros((self.slots,), dtype=jnp.bool_),
            added=jnp.array(0, dtype=jnp.int32),
            labelled=jnp.array(0, dtype=jnp.int32),
            entry_solve=jnp.full((capacity,), -1, dtype=jnp.int32),
            stops=self.draw(key),
            past=jnp.array(False),
            missed=jnp.array(0, dtype=jnp.int32),
        )

    def draw(self, key: jax.Array) -> jax.Array:
        """Whether a game starting now is solved (see `share`)."""
        if self.share >= 1:
            return jnp.array(True)
        return jax.random.uniform(key) < self.share

    def move(
        self,
        key: jax.Array,
        pending: PendingSolves,
        env_state: Any,
        moving: jax.Array,
        stored: jax.Array,
    ) -> Move:
        """Decides what a move about to be played does (an environment's).

        Args:
            key: rng, to draw whether a game past its solve point ends
            pending: the environment's pending solves
            env_state: the position the move is played from
            moving: whether a move is played (if not, nothing happens)
            stored: whether its position is stored in the replay buffer
        """
        at = moving & pending.stops & self.stop_fn(env_state)
        first = at & ~pending.past
        wanted = first | (at & stored)
        full = pending.added - pending.labelled >= self.slots
        # a game whose solve point finds no slot plays on unsolved (see `slots`)
        late = wanted & ~first & full
        ends = (
            jnp.array(True)
            if self.play_past == 0
            else jax.random.uniform(key) < 1 / (1 + self.play_past)
        )
        return Move(
            solve=wanted & ~full,
            whole=first & ~full,
            end=at & ~(first & full) & (ends | late),
            drop=late,
            missed=wanted & full,
        )

    def add(
        self, pending: PendingSolves, env_state: Any, entries: jax.Array, move: Move
    ) -> PendingSolves:
        """Records `move`'s solve, if it has one, of the position `env_state` labelling the
        buffer's `entries` (an environment's)."""
        slot = jnp.where(move.solve, pending.added % self.slots, self.slots)
        first_missed = move.missed & ~move.drop
        return replace(
            pending,
            positions=jax.tree.map(
                lambda p, x: p.at[slot].set(x, mode="drop"),
                pending.positions,
                env_state,
            ),
            whole=pending.whole.at[slot].set(move.whole, mode="drop"),
            entry_solve=jnp.where(
                move.solve & entries, pending.added, pending.entry_solve
            ),
            added=pending.added + move.solve,
            stops=pending.stops & ~first_missed,
            past=pending.past | move.whole,
            missed=pending.missed + move.missed,
        )

    def next_game(
        self, key: jax.Array, pending: PendingSolves, ended: jax.Array
    ) -> PendingSolves:
        """Starts the next game's state, if the game in progress `ended` (an environment's):
        whether it's solved, drawn, and not past its solve point."""
        return replace(
            pending,
            stops=jnp.where(ended, self.draw(key), pending.stops),
            past=pending.past & ~ended,
        )

    def waiting(
        self, pending: PendingSolves
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        """On the host: the positions waiting for their solves, across environments.

        Args:
            pending: every environment's pending solves (a leading environment dimension)

        Returns:
            Tuple[Any, np.ndarray, np.ndarray, np.ndarray]: the positions, a pytree of numpy
                arrays with a leading dimension of one per solve; each one's environment, slot,
                and whether it's a game's at its solve point
        """
        labelled = np.asarray(pending.labelled)
        counts = np.asarray(pending.added) - labelled
        env = np.repeat(np.arange(len(counts)), counts)
        # each environment's solves from its oldest
        first = np.cumsum(counts) - counts
        nth = np.arange(counts.sum()) - first[env]
        slot = (labelled[env] + nth) % self.slots
        positions = jax.tree.map(lambda x: np.asarray(x)[env, slot], pending.positions)
        return positions, env, slot, np.asarray(pending.whole)[env, slot]

    def solve(self, pending: PendingSolves) -> Solved:
        """On the host: solves every environment's waiting positions (see `waiting`)."""
        started = time.perf_counter()
        positions, env, slot, whole = self.waiting(pending)
        counts = np.asarray(pending.added) - np.asarray(pending.labelled)
        solves = (
            np.asarray(self.solve_fn(positions), dtype=np.float32)
            if len(env)
            else np.zeros((0, 0), np.float32)
        )
        results = np.zeros((len(counts), self.slots, solves.shape[-1]), np.float32)
        results[env, slot] = solves
        return Solved(
            results,
            counts.astype(np.int32),
            solves,
            whole,
            time.perf_counter() - started,
        )

    def label(
        self,
        pending: PendingSolves,
        rewards: jax.Array,
        populated: jax.Array,
        results: jax.Array,
        count: jax.Array,
    ) -> tuple[PendingSolves, jax.Array, jax.Array, jax.Array]:
        """Labels the oldest `count` waiting solves' entries with their `results` (an
        environment's).

        Args:
            pending: the environment's pending solves
            rewards: (capacity, players) its replay buffer's rewards
            populated: (capacity,) its replay buffer's populated mask
            results: (slots, players) the results, by slot
            count: how many of the oldest waiting solves to label

        Returns:
            Tuple[PendingSolves, jax.Array, jax.Array, jax.Array]: the pending solves, the
                buffer's rewards and populated mask with the solves' entries labelled and
                populated, and (slots,) the slots labelled that are games' solves at their solve
                points
        """
        nth = (jnp.arange(self.slots) - pending.labelled) % self.slots
        labelled_slots = nth < count
        solve = pending.labelled + nth
        # (slots, capacity): the entries of each solve labelled
        match = labelled_slots[:, None] & (
            pending.entry_solve[None, :] == solve[:, None]
        )
        hit = match.any(0)
        result = jnp.einsum("sc,sp->cp", match.astype(results.dtype), results)
        return (
            replace(
                pending,
                labelled=pending.labelled + count,
                entry_solve=jnp.where(hit, -1, pending.entry_solve),
            ),
            jnp.where(hit[:, None], result.astype(rewards.dtype), rewards),
            populated | hit,
            labelled_slots & pending.whole,
        )
