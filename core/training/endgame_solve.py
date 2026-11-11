"""Exact endgame labels for self-play (see `Trainer`'s `endgame_solve`): a game stops at a
position an exact solver can solve (e.g. Othello's with 20 or fewer empty squares), and every
position of the game gets the solved result as its value target, instead of the result its
players went on to get. Self-play skips the searches for the rest of the game, and the labels
lose the noise of its endgame mistakes.

The solver runs on the host, a batch at a time, while the device plays: the games stopped by the
end of an epoch are solved during the next epoch's self-play, and their positions are labelled
after it (see `Trainer._train_loop`). Until then, a stopped game's positions are in the replay
buffer, unpopulated so that training doesn't sample them, and its stopped position is kept in an
environment's `PendingSolves`.
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
    """One environment's games stopped for a solve and waiting for their results (see
    `EndgameSolve`), in a ring of slots: the `added`-th game stopped is in slot `added % slots`.

    Attributes:
        positions: the stopped positions, environment states with a leading slot dimension
        added: games stopped so far
        labelled: games labelled so far: those from `labelled` to `added` are waiting
        entry_game: (replay buffer capacity,) for each buffer entry, the number of the stopped game
            (counting from 0, as `added` does) it belongs to while it waits for its result; -1
            for none
        stops: whether the game in progress stops (see `EndgameSolve.share`)
        missed: games that would have stopped with every slot full, so played on
    """

    positions: Any
    added: jax.Array
    labelled: jax.Array
    entry_game: jax.Array
    stops: jax.Array
    missed: jax.Array


@dataclass(frozen=True)
class Solved:
    """Games solved on the host (see `EndgameSolve.solve`).

    Attributes:
        results: (environments, slots, players) the results, by the slots the games are in (zeros
            in the others)
        counts: (environments,) how many games each environment had waiting: its oldest, those
            `EndgameSolve.label` labels with these results
        games: (games, players) the results, a game at a time
        seconds: the solve's wall time
    """

    results: np.ndarray
    counts: np.ndarray
    games: np.ndarray
    seconds: float = 0.0


@dataclass(frozen=True)
class EndgameSolve:
    """Exact endgame labels for self-play's games (see the module's docstring).

    A game stops before its move from the first position `stop_fn` marks: that position is
    searched and stored as the game's last (its search's policy is a target like any other), and
    the game ends, with no result yet. The environment starts a new game straight away, so
    self-play's compute per epoch stays the same and goes into more games. Once `solve_fn` has
    solved the stopped position, every stored position of the game gets its result. Stopped games
    count as episodes, and in `episode_metrics_fn`'s metrics, when they're labelled.

    Attributes:
        stop_fn: (environment state) -> whether a game stops at it, before its move; compatible
            with `jax.jit`
        solve_fn: on the host, a batch of stopped positions, environment states as a pytree of
            numpy arrays with a leading batch dimension -> (batch, players) the exact result for
            each player, as the environment's rewards for the game's end would be
        slots: games each environment can hold while they wait for their results. Results come
            once an epoch, for the games stopped before its self-play, so this needs two epochs'
            worth: a game that would stop with every slot full is played to the end instead (and
            counted in `solve_missed`, see `Trainer.collection_metrics`)
        share: the fraction of games that stop, each drawn as it starts; the others are played to
            the end, as without solving (e.g. to keep training on positions past the stop)
    """

    stop_fn: Callable[[Any], jax.Array]
    solve_fn: Callable[[Any], np.ndarray]
    slots: int
    share: float = 1.0

    def __post_init__(self):
        if self.slots < 1:
            raise ValueError(f"slots must be positive, got {self.slots}")
        if not 0 <= self.share <= 1:
            raise ValueError(f"share must be in [0, 1], got {self.share}")

    def get_config(self) -> dict:
        return {"slots": self.slots, "share": self.share}

    def init(self, key: jax.Array, env_state: Any, capacity: int) -> PendingSolves:
        """An environment's `PendingSolves`, with nothing waiting.

        Args:
            key: rng, to draw whether the game in progress stops
            env_state: an environment state (one environment's), for the positions' shapes
            capacity: the replay buffer's capacity, entries per environment
        """
        return PendingSolves(
            positions=jax.tree.map(
                lambda x: jnp.zeros((self.slots, *jnp.shape(x)), jnp.asarray(x).dtype),
                env_state,
            ),
            added=jnp.array(0, dtype=jnp.int32),
            labelled=jnp.array(0, dtype=jnp.int32),
            entry_game=jnp.full((capacity,), -1, dtype=jnp.int32),
            stops=self.draw(key),
            missed=jnp.array(0, dtype=jnp.int32),
        )

    def draw(self, key: jax.Array) -> jax.Array:
        """Whether a game starting now stops (see `share`)."""
        if self.share >= 1:
            return jnp.array(True)
        return jax.random.uniform(key) < self.share

    def stopping(
        self, pending: PendingSolves, env_state: Any
    ) -> tuple[jax.Array, jax.Array]:
        """Whether the game at `env_state` stops before its move, and whether it would but every
        slot is full (an environment's)."""
        due = pending.stops & self.stop_fn(env_state)
        full = pending.added - pending.labelled >= self.slots
        return due & ~full, due & full

    def stop(
        self,
        pending: PendingSolves,
        env_state: Any,
        entries: jax.Array,
        stopped: jax.Array,
        missed: jax.Array,
    ) -> PendingSolves:
        """Records a stopped game (an environment's), if `stopped`, and a missed one, if `missed`
        (see `stopping`): the game then plays on, and doesn't stop again.

        Args:
            pending: the environment's pending solves
            env_state: the position the game stopped at
            entries: (capacity,) the game's entries in the replay buffer
            stopped: whether it stopped
            missed: whether it would have, but every slot was full
        """
        slot = jnp.where(stopped, pending.added % self.slots, self.slots)
        return replace(
            pending,
            positions=jax.tree.map(
                lambda p, x: p.at[slot].set(x, mode="drop"),
                pending.positions,
                env_state,
            ),
            entry_game=jnp.where(stopped & entries, pending.added, pending.entry_game),
            added=pending.added + stopped,
            stops=pending.stops & ~missed,
            missed=pending.missed + missed,
        )

    def next_game(
        self, key: jax.Array, pending: PendingSolves, ended: jax.Array
    ) -> PendingSolves:
        """Draws whether the next game stops, if the game in progress `ended` (an environment's)."""
        return replace(pending, stops=jnp.where(ended, self.draw(key), pending.stops))

    def waiting(self, pending: PendingSolves) -> tuple[Any, np.ndarray, np.ndarray]:
        """On the host: the stopped positions waiting for their results, across environments.

        Args:
            pending: every environment's pending solves (a leading environment dimension)

        Returns:
            Tuple[Any, np.ndarray, np.ndarray]: the positions, a pytree of numpy arrays with a
                leading dimension of one per game; each game's environment; and its slot
        """
        labelled = np.asarray(pending.labelled)
        counts = np.asarray(pending.added) - labelled
        env = np.repeat(np.arange(len(counts)), counts)
        # each environment's games from its oldest
        first = np.cumsum(counts) - counts
        nth = np.arange(counts.sum()) - first[env]
        slot = (labelled[env] + nth) % self.slots
        positions = jax.tree.map(lambda x: np.asarray(x)[env, slot], pending.positions)
        return positions, env, slot

    def solve(self, pending: PendingSolves) -> Solved:
        """On the host: solves every environment's waiting games (see `waiting`)."""
        started = time.perf_counter()
        positions, env, slot = self.waiting(pending)
        counts = np.asarray(pending.added) - np.asarray(pending.labelled)
        games = (
            np.asarray(self.solve_fn(positions), dtype=np.float32)
            if len(env)
            else np.zeros((0, 0), np.float32)
        )
        results = np.zeros((len(counts), self.slots, games.shape[-1]), np.float32)
        results[env, slot] = games
        return Solved(
            results, counts.astype(np.int32), games, time.perf_counter() - started
        )

    def label(
        self,
        pending: PendingSolves,
        rewards: jax.Array,
        populated: jax.Array,
        results: jax.Array,
        count: jax.Array,
    ) -> tuple[PendingSolves, jax.Array, jax.Array, jax.Array]:
        """Labels the oldest `count` waiting games with their `results` (an environment's).

        Args:
            pending: the environment's pending solves
            rewards: (capacity, players) its replay buffer's rewards
            populated: (capacity,) its replay buffer's populated mask
            results: (slots, players) the results, by slot
            count: how many of the oldest waiting games to label

        Returns:
            Tuple[PendingSolves, jax.Array, jax.Array, jax.Array]: the pending solves, the
                buffer's rewards and populated mask with the games' entries labelled and
                populated, and (slots,) which slots were labelled
        """
        nth = (jnp.arange(self.slots) - pending.labelled) % self.slots
        labelled_slots = nth < count
        game = pending.labelled + nth
        # (slots, capacity): the entries of each game labelled
        match = labelled_slots[:, None] & (pending.entry_game[None, :] == game[:, None])
        hit = match.any(0)
        result = jnp.einsum("sc,sp->cp", match.astype(results.dtype), results)
        return (
            replace(
                pending,
                labelled=pending.labelled + count,
                entry_game=jnp.where(hit, -1, pending.entry_game),
            ),
            jnp.where(hit[:, None], result.astype(rewards.dtype), rewards),
            populated | hit,
            labelled_slots,
        )
