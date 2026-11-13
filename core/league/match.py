"""Plays a batch of games between two players, each only seeing the moves.

Each network plays with its own evaluator, parameters and search trees, and its own random
keys; no cache or state passes between the two. The same program serves any network of the
same spec, batch size and search budget, since parameters are its input, but programs hold
nothing between calls, so this shares compile time and no information.

The batch's games move in step, and only the side to move searches, so that a batch's cost is
one search per position. Half the games are played with each moving first, opening by opening
(each opening played with both colours), one half after the other.

What a side's search made of each move is read from its finished tree by a program of its own
(`summarise_searches`), rather than added to the search's outputs: the search is the same program
whatever is recorded, so the moves can't depend on it.
"""

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from core.evaluators.mcts.state import MCTSTree
from core.league.game import ExternalPlayer, LeagueGame
from core.league.store import (
    TOP_MOVES,
    BatchSettings,
    GameRecord,
    NetSearch,
    Player,
    Search,
    TopMove,
    changed_settings,
)


class Stopped(Exception):
    """Someone else needs the GPU, or the league's being stopped: the batch is abandoned."""


class SettingsRefused(Exception):
    """A player would have played with other settings than its own: no longer the same player,
    so the batch is abandoned before its first move.

    Attributes:
        player: who
        settings: what it would have played with
        changed: which of them differ
    """

    def __init__(self, player: Player, settings: dict[str, Any], changed: list[str]):
        super().__init__(f"{player.name}: {', '.join(changed)}")
        self.player = player
        self.settings = settings
        self.changed = changed


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
    """A network spec's compiled search, for one batch size, search budget and tree size."""

    init: Callable
    search: Callable
    advance: Callable
    # everything the evaluator was given, as `Evaluator.get_config` says
    config: dict[str, Any]


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
        key = (
            player.net,
            player.sims,
            player.tree_factor,
            player.prior_temperature,
            games,
        )
        found = self.programs.get(key)
        if found is None:
            assert player.sims is not None
            network, _ = self.network(player)
            found = self._compile(
                network, player.sims, player.tree_factor, player.prior_temperature
            )
            self.programs[key] = found
            self.compiled = True
            if len(self.programs) > self.keep:
                self.programs.popitem(last=False)
        self.programs.move_to_end(key)
        return found

    def _compile(
        self, network: Any, sims: int, tree_factor: int, prior_temperature: float
    ) -> Program:
        game = self.game
        evaluator = game.make_evaluator(network, sims, tree_factor, prior_temperature)
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
            config=evaluator.get_config(),
        )

    def settings(self, player: Player, games: int) -> dict[str, Any]:
        """Everything `player` plays with here, but for what it's playing on."""
        assert player.path is not None
        return {
            **self.program(player, games).config,
            "dtype": self.dtype,
            "net": player.net,
            "checkpoint_sha256": Path(player.path).stem,
        }


def _summarise_search(before: MCTSTree, tree: MCTSTree) -> dict[str, jax.Array]:
    """What a `NetSearch` keeps of a finished tree, and of the one it started from. Values are as
    they're stored at their nodes: a child's is for the player to move there."""
    children = tree.edge_map[tree.ROOT_INDEX]
    visits = jnp.where(children != tree.NULL_INDEX, tree.data.n[children], 0)
    top_visits, actions = jax.lax.top_k(visits, TOP_MOVES)
    return {
        "value": tree.data.q[tree.ROOT_INDEX],
        "net_value": tree.root_value,
        "visits": tree.data.n[tree.ROOT_INDEX],
        "reused": before.data.n[before.ROOT_INDEX],
        "tree": tree.next_free_idx,
        "actions": actions,
        "top_visits": top_visits,
        "top_q": tree.data.q[children[actions]],
    }


# one program for all of a tree's shape, reading a few numbers a game however big the trees: what
# it doesn't read of them stays on the device
summarise_searches = jax.jit(jax.vmap(_summarise_search))


class NetworkSide:
    """A network's side of a batch's games: its parameters, and its own tree of each game,
    kept between its moves."""

    def __init__(
        self,
        program: Program,
        params: Any,
        key: jax.Array,
        settings: dict[str, Any] | None = None,
    ):
        self.program = program
        self.params = params
        self.key = key
        self.played_settings = settings or {}
        self.trees: Any = None
        self.summary: dict[str, np.ndarray] = {}

    def start(self, states: Any, first: np.ndarray) -> None:
        self.trees = self.program.init(states)
        self.keys = jax.random.split(self.key, len(first))

    def move(self, states: Any, movers: np.ndarray) -> np.ndarray:
        keys = jax.vmap(jax.random.fold_in)(self.keys, states._step_count)
        out = self.program.search(keys, self.trees, states, self.params)
        summary = summarise_searches(self.trees, out.eval_state)
        self.trees = select(movers, out.eval_state, self.trees)
        self.summary = jax.device_get(summary)
        return np.asarray(out.action)

    def searches(self, movers: np.ndarray) -> dict[int, Search]:
        """What the search of the last `move` made of the games marked `movers`, by game."""
        s, discount = self.summary, self.program.config["discount"]
        return {
            g: NetSearch(
                value=float(s["value"][g]),
                net_value=float(s["net_value"][g]),
                visits=int(s["visits"][g]),
                # a root with visits to reuse is still evaluated, without counting as a visit
                evals=int(s["visits"][g] - s["reused"][g] + (s["reused"][g] > 0)),
                tree=int(s["tree"][g]),
                top=tuple(
                    TopMove(int(a), int(v), discount * float(q))
                    for a, v, q in zip(
                        s["actions"][g], s["top_visits"][g], s["top_q"][g]
                    )
                    if v > 0
                ),
            )
            for g in np.flatnonzero(movers).tolist()
        }

    def settings(self) -> dict[str, Any]:
        return self.played_settings

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


@dataclass
class Clock:
    """How long a player took to choose its moves. Games move together, so a move's share of
    a step's time is what it took: this is throughput rather than the time of one search."""

    seconds: float = 0.0
    moves: int = 0

    def add(self, seconds: float, moves: int) -> None:
        self.seconds += seconds
        self.moves += moves


def play_games(
    game: LeagueGame,
    first: Side,
    second: Side,
    openings: np.ndarray,
    key: jax.Array,
    should_stop: Callable[[], bool] = lambda: False,
    started: Callable[[], None] = lambda: None,
) -> tuple[
    np.ndarray,
    list[list[int]],
    np.ndarray | None,
    Clock,
    Clock,
    list[dict[int, Search]],
    list[tuple[int, float, int]],
]:
    """Plays a game per row of `openings` between two sides, the moves known to both.

    Args:
        game: the game
        first: the side that moves first in the game (the opening's first move included)
        second: the other
        openings: (games, plies) the moves played for them first
        key: the environment's rng
        should_stop: checked before each move; once it's true the games are abandoned
        started: called once the sides have started, before the first move

    Returns:
        the first player's result in each game (+1, 0, -1); each game's moves, from the
        start; the game's scores of them (see `LeagueGame.scores`), if it has any; how long
        the first and the second took to choose their moves; what each game's searches made of
        its moves, by ply; each step of the games: (ply, seconds, games moving)

    Raises:
        Stopped: if `should_stop` said so
    """
    env = game.env
    num_games = len(openings)
    step = _env_step(env)
    states = _env_init(env)(jax.random.split(key, num_games))
    first_id = np.asarray(states.current_player)
    moves: list[list[int]] = [[] for _ in range(num_games)]
    searches: list[dict[int, Search]] = [{} for _ in range(num_games)]
    steps: list[tuple[int, float, int]] = []
    first.start(states, first_id)
    second.start(states, first_id)
    started()
    clocks = {id(first): Clock(), id(second): Clock()}

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
                start = time.perf_counter()
                actions = side.move(states, movers)
                seconds = time.perf_counter() - start
                clocks[id(side)].add(seconds, int(movers.sum()))
                # every game's at the same ply: passes count, and the openings are one length
                ply = len(moves[int(np.argmax(movers))])
                steps.append((ply, round(seconds, 4), int(movers.sum())))
                for g, search in side.searches(movers).items():
                    searches[g][len(moves[g])] = search
                played(actions, movers, (*own, other))
    results = np.sign(np.asarray(states.rewards)[np.arange(num_games), first_id])
    results = results.astype(int)
    first.finish(states, results)
    second.finish(states, results)
    return (
        results,
        moves,
        game.scores(states, first_id),
        clocks[id(first)],
        clocks[id(second)],
        searches,
        steps,
    )


def check_settings(sides: tuple[tuple[Player, Side], ...]) -> None:
    """Raises `SettingsRefused` if a player, once started, has other settings than its own."""
    for player, side in sides:
        settings = side.settings()
        changed = changed_settings(player.played_settings or settings, settings)
        if changed:
            raise SettingsRefused(player, settings, changed)


@dataclass(frozen=True)
class BatchResult:
    """A batch played: its games, how long it took and whether that includes compiling, how
    long each player took to choose its moves, and what it was played with."""

    games: list[GameRecord]
    seconds: float
    compiled: bool
    a_clock: Clock
    b_clock: Clock
    settings: BatchSettings | None = None


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
        SettingsRefused: if either would play with other settings than its own
    """
    start = time.perf_counter()
    programs.compiled = False
    moves_of = game.openings(openings)

    def side(player: Player, key: jax.Array) -> Side:
        if player.is_external:
            return game.external_player(player)
        return NetworkSide(
            programs.program(player, len(openings)),
            programs.network(player)[1],
            key,
            programs.settings(player, len(openings)),
        )

    games: list[GameRecord] = []
    clock_of = {a.id: Clock(), b.id: Clock()}
    settings_of: dict[int, dict[str, Any]] = {}
    steps_of_halves = []
    for half, (one, other) in enumerate(((a, b), (b, a))):
        env_key, first_key, second_key = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(seed), half), 3
        )
        first, second = side(one, first_key), side(other, second_key)
        try:
            results, moves, scores, first_clock, second_clock, searches, steps = (
                play_games(
                    game,
                    first,
                    second,
                    moves_of,
                    env_key,
                    should_stop,
                    partial(check_settings, ((one, first), (other, second))),
                )
            )
            settings_of[one.id] = first.settings()
            settings_of[other.id] = second.settings()
        finally:
            first.close()
            second.close()
        clock_of[one.id].add(first_clock.seconds, first_clock.moves)
        clock_of[other.id].add(second_clock.seconds, second_clock.moves)
        steps_of_halves.append(tuple(steps))
        games += [
            GameRecord(
                first=one.id,
                second=other.id,
                opening=int(openings[g]),
                result=int(results[g]),
                moves=moves[g],
                first_score=None if scores is None else float(scores[g, 0]),
                second_score=None if scores is None else float(scores[g, 1]),
                searches=searches[g],
            )
            for g in range(len(openings))
        ]
    return BatchResult(
        games,
        time.perf_counter() - start,
        programs.compiled,
        clock_of[a.id],
        clock_of[b.id],
        BatchSettings(
            settings_of[a.id],
            settings_of[b.id],
            len(openings),
            jax.devices()[0].device_kind,
            tuple(steps_of_halves),
        ),
    )
