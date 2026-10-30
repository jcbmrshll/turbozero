"""Plays a batch of games between two players, each only seeing the moves.

Each network plays with its own evaluator, parameters and search trees, and its own random
keys; no cache or state passes between the two. The same program serves any network of the
same spec, batch size and search budget, since parameters are its input, but programs hold
nothing between calls, so this shares compile time and no information.

The batch's games move in step, and only the side to move searches, so that a batch's cost is
one search per position. Half the games are played with each moving first, opening by opening
(each opening played with both colours), one half after the other.
"""

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from core.league.game import ExternalPlayer, LeagueGame
from core.league.store import GameRecord, Player


class Stopped(Exception):
    """Someone else needs the GPU, or the league's being stopped: the batch is abandoned."""


def select(which, on_true, on_false):
    """Game by game, `on_true` where `which` is set, else `on_false`."""
    which = jnp.asarray(which)
    return jax.tree.map(
        lambda t, f: jnp.where(which.reshape((-1,) + (1,) * (t.ndim - 1)), t, f),
        on_true,
        on_false,
    )


@dataclass(frozen=True)
class Program:
    """A network spec's compiled search, for one batch size and search budget."""

    init: Callable
    search: Callable
    advance: Callable


class Programs:
    """The programs compiled so far, and the networks loaded so far, most recent kept. Each
    program compiles when first called; `compiled` says whether any new one has been asked
    for since it was last cleared."""

    def __init__(
        self, game: LeagueGame, dtype: str, keep: int = 32, networks: int = 16
    ):
        self.game = game
        self.dtype = dtype
        self.keep = keep
        self.keep_networks = networks
        self.programs: OrderedDict[tuple, Program] = OrderedDict()
        self.networks: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self.compiled = False

    def network(self, player: Player) -> tuple[Any, Any]:
        assert player.path is not None and player.net is not None
        found = self.networks.get(player.path)
        if found is None:
            found = self.game.load_network(player.path, player.net, self.dtype)
            self.networks[player.path] = found
            if len(self.networks) > self.keep_networks:
                self.networks.popitem(last=False)
        self.networks.move_to_end(player.path)
        return found

    def program(self, player: Player, games: int) -> Program:
        key = (player.net, player.sims, games)
        found = self.programs.get(key)
        if found is None:
            assert player.sims is not None
            network, _ = self.network(player)
            found = self._compile(network, player.sims)
            self.programs[key] = found
            self.compiled = True
            if len(self.programs) > self.keep:
                self.programs.popitem(last=False)
        self.programs.move_to_end(key)
        return found

    def _compile(self, network: Any, sims: int) -> Program:
        game = self.game
        evaluator = game.make_evaluator(network, sims)
        return Program(
            init=jax.jit(
                jax.vmap(lambda state: evaluator.init(template_embedding=state))
            ),
            search=jax.jit(
                jax.vmap(
                    lambda key, tree, state, params: evaluator.evaluate(
                        key=key,
                        eval_state=tree,
                        env_state=state,
                        root_metadata=game.metadata(state),
                        params=params,
                        env_step_fn=game.step_fn,
                    ),
                    in_axes=(0, 0, 0, None),
                )
            ),
            advance=jax.jit(jax.vmap(evaluator.step)),
        )


class NetworkSide:
    """A network's side of a batch's games: its parameters, and its own tree of each game,
    kept between its moves."""

    def __init__(self, program: Program, params: Any, key: jax.Array):
        self.program = program
        self.params = params
        self.key = key
        self.trees: Any = None

    def start(self, states: Any, first: np.ndarray) -> None:
        self.trees = self.program.init(states)
        self.keys = jax.random.split(self.key, len(first))

    def move(self, states: Any, movers: np.ndarray) -> np.ndarray:
        keys = jax.vmap(jax.random.fold_in)(self.keys, states._step_count)
        out = self.program.search(keys, self.trees, states, self.params)
        self.trees = select(movers, out.eval_state, self.trees)
        return np.asarray(out.action)

    def observe(self, actions: np.ndarray, movers: np.ndarray) -> None:
        trees = self.program.advance(self.trees, jnp.asarray(actions))
        self.trees = select(movers, trees, self.trees)

    def finish(self, states: Any, results: np.ndarray) -> None:
        pass

    def close(self) -> None:
        self.trees = None


Side = NetworkSide | ExternalPlayer


@cache
def _env_init(env):
    return jax.jit(jax.vmap(env.init))


@cache
def _env_step(env):
    return jax.jit(jax.vmap(env.step))


def play_games(
    game: LeagueGame,
    first: Side,
    second: Side,
    openings: np.ndarray,
    key: jax.Array,
    should_stop: Callable[[], bool] = lambda: False,
) -> tuple[np.ndarray, list[list[int]], np.ndarray | None]:
    """Plays a game per row of `openings` between two sides, the moves known to both.

    Args:
        game: the game
        first: the side that moves first in the game (the opening's first move included)
        second: the other
        openings: (games, plies) the moves played for them first
        key: the environment's rng
        should_stop: checked before each move; once it's true the games are abandoned

    Returns:
        the first player's result in each game (+1, 0, -1); each game's moves, from the
        start; the game's scores of them (see `LeagueGame.scores`), if it has any

    Raises:
        Stopped: if `should_stop` said so
    """
    env = game.env
    num_games = len(openings)
    step = _env_step(env)
    states = _env_init(env)(jax.random.split(key, num_games))
    first_id = np.asarray(states.current_player)
    moves: list[list[int]] = [[] for _ in range(num_games)]
    first.start(states, first_id)
    second.start(states, first_id)

    def played(actions: np.ndarray, movers: np.ndarray, told: tuple[Side, ...]) -> None:
        """Plays `actions` in the games marked `movers`, telling the sides `told`."""
        nonlocal states
        mask = np.asarray(states.legal_action_mask)
        assert mask[movers, actions[movers]].all(), "an illegal move"
        for side in told:
            side.observe(actions, movers)
        states = select(movers, step(states, jnp.asarray(actions)), states)
        for g in np.flatnonzero(movers):
            moves[g].append(int(actions[g]))

    # nobody searched for these moves: both sides are told
    for ply in range(openings.shape[1]):
        if should_stop():
            raise Stopped
        played(
            openings[:, ply].astype(np.int32),
            np.ones(num_games, dtype=bool),
            (first, second),
        )

    while not (done := np.asarray(states.terminated)).all():
        if should_stop():
            raise Stopped
        to_move = np.asarray(states.current_player)
        for side, other, movers in (
            (first, second, (to_move == first_id) & ~done),
            (second, first, (to_move != first_id) & ~done),
        ):
            if movers.any():
                # an external player has already played its own move on its board
                own = (side,) if isinstance(side, NetworkSide) else ()
                played(side.move(states, movers), movers, (*own, other))
    results = np.sign(np.asarray(states.rewards)[np.arange(num_games), first_id])
    results = results.astype(int)
    first.finish(states, results)
    second.finish(states, results)
    return results, moves, game.scores(states, first_id)


@dataclass(frozen=True)
class BatchResult:
    """A batch played: its games, how long it took and whether that includes compiling."""

    games: list[GameRecord]
    seconds: float
    compiled: bool


def play_batch(
    game: LeagueGame,
    programs: Programs,
    a: Player,
    b: Player,
    openings: np.ndarray,
    seed: int,
    should_stop: Callable[[], bool] = lambda: False,
) -> BatchResult:
    """Plays 2 * len(openings) games between `a` and `b`, each of `openings` (their indices)
    with each moving first.

    Raises:
        Stopped: if `should_stop` said so: none of the batch's games are kept, since those
            finished by then would be the shorter ones
    """
    start = time.perf_counter()
    programs.compiled = False
    moves_of = game.openings(openings)

    def side(player: Player, key: jax.Array) -> Side:
        if player.is_external:
            return game.external_player(player)
        program = programs.program(player, len(openings))
        return NetworkSide(program, programs.network(player)[1], key)

    games: list[GameRecord] = []
    for half, (one, other) in enumerate(((a, b), (b, a))):
        env_key, first_key, second_key = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(seed), half), 3
        )
        first, second = side(one, first_key), side(other, second_key)
        try:
            results, moves, scores = play_games(
                game,
                first,
                second,
                moves_of,
                env_key,
                should_stop,
            )
        finally:
            first.close()
            second.close()
        games += [
            GameRecord(
                first=one.id,
                second=other.id,
                opening=int(openings[g]),
                result=int(results[g]),
                moves=moves[g],
                first_score=None if scores is None else float(scores[g, 0]),
                second_score=None if scores is None else float(scores[g, 1]),
            )
            for g in range(len(openings))
        ]
    return BatchResult(games, time.perf_counter() - start, programs.compiled)
