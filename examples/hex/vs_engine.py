"""Plays a checkpoint from `train.py` against MoHex, at a series of levels.

    examples/hex/setup_mohex.sh         # once: builds MoHex
    uv run examples/hex/vs_engine.py CHECKPOINT --size 11 --levels 100 1000 10000 --sims 400

Games start from openings (see `openings.py`), each played twice with the colours swapped, so
that two deterministic players give varied, balanced games. The same `--seed` always draws the
same openings, so checkpoints play the same games. At level L MoHex searches L playouts a move,
on one thread (see `engines.py`). Our agent runs `--sims` MCTS iterations a move at temperature
0, keeping its search tree between moves, without the root noise used in self-play, its network
computing in `--dtype` (bfloat16, as in self-play, unless told otherwise).

Games are played in batches of equal size, each game with its own engine process: as many at a
time as fit in a quarter of the available memory, up to 128, since the machine may be shared. The
games of a batch move in step: the engine moves in all of them it's to move in, then our agent's
moves are searched together on the accelerator. Each game's result is checked against the
engine's own board. The engine processes run on the cores this process may use (e.g. with
`taskset`), as many searching at once as there are cores.

train.py plays these games as it trains (`EngineMatchTester`, see its --vs-engine-every and
--vs-engine-last), logging them to the run. Here, with --monitor and --run, a checkpoint's
results are logged to its training run afterwards, at the checkpoint's epoch, as
`{engine}_level{L}_sims{S}` (the score, and `_se`, `_win`, `_draw` and `_loss`), the first game at
each level with them, as `{engine}_level{L}_sims{S}_game`: its positions, what our agent's search
made of each, and what the engine's made of those it moved from (see `engines.MoHex.analysis`).
"""

import argparse
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from operator import methodcaller
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from engines import ENGINES, Analysis, MoHex
from game import SIZES, HexGame, metadata, state_to_nn_input
from openings import NO_MOVE, load_openings
from seaglass import Arrays

from core.common import GameFrame
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.monitoring import DEFAULT_URL, Monitor
from core.monitoring.episodes import line_record, pgx_two_player_episode
from core.testing.tester import BaseTester, TestState


def select(which, on_true, on_false):
    """Game by game, `on_true` where `which` is set, else `on_false`."""
    which = jnp.asarray(which)
    return jax.tree.map(
        lambda t, f: jnp.where(which.reshape((-1,) + (1,) * (t.ndim - 1)), t, f),
        on_true,
        on_false,
    )


def stack(trees):
    """Trees stacked along a new leading axis."""
    return jax.tree.map(lambda *x: np.stack(x), *trees)


def first_stones(states) -> np.ndarray:
    """Each game's first stone, for a game on its second move (see `MoHex.play`); else 0."""
    # the opponent's, the only one on the board
    return np.asarray(states._x.board < 0).argmax(axis=1)


class GameRecorder:
    """One game of a batch, recorded for the monitor: each position, what our agent's search made
    of it (its search, if it moved, else the tree it kept from its last one) and what the
    engine's made of those it moved from. Its engine has to be analysing (`analyse`)."""

    def __init__(self, game: int, hex_game: HexGame, evaluator):
        self.game = game
        self.hex_game = hex_game
        self.search_record = jax.jit(evaluator.search_record)
        self.states = []
        # what each made of each position played from, None for nothing
        self.trees, self.analyses = [], []

    def take(self, batch):
        return jax.device_get(jax.tree.map(lambda x: x[self.game], batch))

    def start(self, states, our_id: int) -> None:
        """The game's first position, in `states`, where our agent is player `our_id`."""
        self.our_id = our_id
        self.states.append(self.take(states))

    def moved(self, states, trees=None, analysis: list[Analysis] | None = None) -> None:
        """The game's next position, in `states`, after a move from the last, which our agent
        had `trees` of and the engine made `analysis` of."""
        tree = None if trees is None else self.search_record(self.take(trees))
        self.trees.append(None if tree is None else jax.device_get(tree))
        self.analyses.append(analysis)
        self.states.append(self.take(states))

    def episode(self, opening: int) -> Arrays:
        """The game, for the monitor; the players started playing after `opening` moves played
        for them."""
        num_actions = self.hex_game.env.num_actions
        capacity = 1 + max(
            (sum(len(a.line) for a in moves) for moves in self.analyses if moves),
            default=0,
        )
        records = [
            line_record(
                moves and [(a.line, a.score, 0) for a in moves],
                num_actions=num_actions,
                capacity=capacity,
            )
            for moves in self.analyses
        ]
        # each one's value of each position, from its own side, as of its last look; a root's q
        # is for the player to move
        values = np.zeros((len(self.states), 2), dtype=np.float32)
        for t, state in enumerate(self.states):
            values[t] = values[t - 1] if t else 0
            ours = int(state.current_player) == self.our_id
            if t < len(self.trees) and self.trees[t] is not None:
                q = float(self.trees[t].q[0])
                values[t, 0] = q if ours else -q
            if t < len(records) and self.analyses[t]:
                q = float(records[t].q[0])
                values[t, 1] = -q if ours else q
        empty = jax.tree.map(
            np.zeros_like, next(t for t in self.trees if t is not None)
        )
        # frame t + 1 holds what each made of position t
        frames = GameFrame(
            env_state=stack(self.states),
            p1_value_estimate=jnp.asarray(values[:, 0]),
            p2_value_estimate=jnp.asarray(values[:, 1]),
            completed=jnp.array([bool(s.terminated) for s in self.states]),
            outcomes=jnp.stack([s.rewards for s in self.states]),
            p1_search=stack(
                [empty] + [t if t is not None else empty for t in self.trees]
            ),
            p2_search=stack(
                [line_record(None, num_actions=num_actions, capacity=capacity)]
                + records
            ),
        )
        return pgx_two_player_episode()(
            frames,
            np.array([self.our_id, 1 - self.our_id]),
            self.hex_game.step_fn,
            start=opening,
        )


def play_games(
    engines: list[MoHex],
    openings,
    we_first,
    search,
    advance,
    evaluator,
    pool,
    keys,
    game: HexGame,
    recorder=None,
):
    """Plays one game per engine process, from `openings` (one per game, padded with NO_MOVE),
    with our agent as colour 0 in the games marked `we_first`, recording one of them with
    `recorder`. Each game has its own key, of `keys`, so that its moves don't depend on the other
    games.

    Returns:
        our result in each game: +1 win, -1 loss
    """
    num_games = len(engines)
    states: Any = jax.vmap(game.env.init)(keys)
    trees = jax.vmap(lambda s: evaluator.init(template_embedding=s))(states)
    # pgx picks which player id is colour 0 at random
    first_id = np.asarray(states.current_player)
    our_id = np.where(we_first, first_id, 1 - first_id)
    if recorder is not None:
        recorder.start(states, int(our_id[recorder.game]))

    # play the openings
    for ply in range(openings.shape[1]):
        moves = openings[:, ply]
        playing = moves != NO_MOVE
        colour = "b" if ply % 2 == 0 else "w"
        moved = np.flatnonzero(playing)
        assert np.asarray(states.legal_action_mask)[moved, moves[moved]].all()
        list(
            pool.map(
                lambda engine, *args: engine.play(*args),
                [engines[g] for g in moved],
                [colour] * len(moved),
                moves[moved].tolist(),
            )
        )
        new_trees, new_states = advance(trees, states, jnp.asarray(moves.clip(0)))
        trees = select(playing, new_trees, trees)
        states = select(playing, new_states, states)
        if recorder is not None and playing[recorder.game]:
            recorder.moved(states)

    # the games step in turn: wherever it's the engine's move, the engine moves, and only once it
    # has moved in every game is our agent's move searched, in all of them at once. The players
    # alternate, so games that start in step stay in step
    while not np.asarray(states.terminated).all():
        done = np.asarray(states.terminated)
        mask = np.asarray(states.legal_action_mask)
        to_move = np.asarray(states.current_player)
        colours = np.where(np.asarray(states._x.color) == 0, "b", "w")
        stones = first_stones(states)
        theirs = (to_move != our_id) & ~done
        moves = np.zeros(num_games, dtype=np.int32)
        # our agent's tree of each position moved from: the one it kept, or its search
        trees_before = trees
        if theirs.any():
            # the engine's moves, all games at once
            movers = theirs
            asking = np.flatnonzero(theirs)
            replies = pool.map(
                lambda engine, *args: engine.genmove(*args),
                [engines[g] for g in asking],
                colours[asking].tolist(),
                stones[asking].tolist(),
            )
            for g, move in zip(asking, replies):
                moves[g] = move
        else:
            # our moves, every game in one batch
            movers = ~done
            # a key for each game and ply
            out = search(
                jax.vmap(jax.random.fold_in)(keys, states._step_count), trees, states
            )
            moves[movers] = np.asarray(out.action)[movers]
            moved = np.flatnonzero(movers)
            list(
                pool.map(
                    lambda engine, *args: engine.play(*args),
                    [engines[g] for g in moved],
                    colours[moved].tolist(),
                    moves[moved].tolist(),
                    stones[moved].tolist(),
                )
            )
            trees = trees_before = select(movers, out.eval_state, trees)
        assert mask[movers, moves[movers]].all(), "an illegal move"
        # step the games that moved; the others stay as they are
        new_trees, new_states = advance(trees, states, jnp.asarray(moves))
        trees = select(movers, new_trees, trees)
        states = select(movers, new_states, states)
        if recorder is not None and movers[recorder.game]:
            g = recorder.game
            recorder.moved(
                states, trees_before, engines[g].analysis if theirs[g] else None
            )

    results = np.asarray(states.rewards)[np.arange(num_games), our_id]
    # check every result against the engine's own board
    for g, winner in enumerate(pool.map(methodcaller("winner"), engines)):
        ours = "b" if we_first[g] else "w"
        assert winner == (ours if results[g] > 0 else "bw"[ours == "b"]), (
            f"game {g}: the engine disagrees on the result"
        )
    return results


def score(results) -> tuple[float, float]:
    """Mean score and its standard error."""
    points = (1 + np.sign(results)) / 2
    return points.mean(), points.std() / np.sqrt(len(points))


def available_memory() -> int:
    """Bytes of memory available to new processes: free memory plus what the kernel could reclaim
    from caches (MemAvailable on Linux), or just free memory elsewhere."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def available_cpus() -> int:
    """The cores this process may run on (and the engine processes it starts, which inherit them,
    e.g. from `taskset`), or all of them where that can't be told."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def max_parallel_games(engine: type[MoHex], limit: int = 128) -> int:
    """How many engine processes to run at once: as many as fit in a quarter of the memory
    available now (the machine may be shared), up to `limit`."""
    fit = int(0.25 * available_memory() / engine.memory_per_process)
    return max(2, min(fit, limit) // 2 * 2)


@dataclass(frozen=True)
class LevelResult:
    """Our agent's games against the engine at one level.

    Attributes:
        level: the engine's level
        results: our result in each game: +1 win, -1 loss
        we_first: which games we played colour 0 in
        seconds: how long the games took (and recording the first, if it was)
        episode: the first game, for the monitor, if it was recorded
    """

    level: int
    results: np.ndarray
    we_first: np.ndarray
    seconds: float
    episode: Arrays | None = None

    def metrics(self, name: str) -> dict[str, Any]:
        """What the monitor logs of it, under `name` (`EngineMatch.name`): the score, its
        standard error, the fractions of games won and lost, none being drawn, and the game."""
        mean, se = score(self.results)
        metrics: dict[str, Any] = {
            name: float(mean),
            f"{name}_se": float(se),
            f"{name}_win": float((self.results > 0).mean()),
            f"{name}_draw": 0.0,
            f"{name}_loss": float((self.results < 0).mean()),
        }
        if self.episode is not None:
            metrics[f"{name}_game"] = self.episode
        return metrics

    def record(self, **extra: Any) -> dict[str, Any]:
        """The result as a line of a results file: `extra` (e.g. which checkpoint played), then
        the games' score, standard error, outcomes and duration."""
        mean, se = score(self.results)
        return {
            **extra,
            "level": self.level,
            "games": len(self.results),
            "score": float(mean),
            "se": float(se),
            "win": float((self.results > 0).mean()),
            "loss": float((self.results < 0).mean()),
            "as_first": float(score(self.results[self.we_first])[0]),
            "as_second": float(score(self.results[~self.we_first])[0]),
            "seconds": round(self.seconds, 1),
        }

    def summary(self) -> str:
        """A line for the console."""
        mean, se = score(self.results)
        first = score(self.results[self.we_first])[0]
        second = score(self.results[~self.we_first])[0]
        return (
            f"level {self.level:6d}: score {mean:.3f} ± {se:.3f}  (as colour 0 {first:.2f}, "
            f"as colour 1 {second:.2f})  {self.seconds:.0f}s"
        )


class EngineMatch:
    """Our agent, `network` searching `sims` MCTS iterations a move, against an engine at each of
    `levels`: 2 * `openings` games a level, from openings each played with both colours. The
    openings are drawn level by level from `seed`, so every network it plays plays the same
    games, and so does every match with the same levels and seed, here or in train.py (see
    `EngineMatchTester`).

    The search is compiled once, for whatever parameters it's given: train.py plays the network
    as it trains, epoch after epoch, without compiling again.
    """

    def __init__(
        self,
        game: HexGame,
        network,
        engine: type[MoHex],
        levels: list[int],
        sims: int,
        openings: int,
        engine_dir: Path | None = None,
        seed: int = 0,
        opening_list: str | Path | None = None,
        parallel: int | None = None,
    ):
        """
        Args:
            game: the game at its board size
            network: the network (see HexGame.make_network), computing in the dtype to play in
            engine: the engine, of `engines.ENGINES`
            levels: the levels to play at
            sims: our MCTS iterations a move
            openings: openings a level, each played with both colours
            engine_dir: where the engine is installed (default: its own default)
            seed: draws the openings, and seeds the games
            opening_list: an opening list (default: the board size's)
            parallel: games at a time (default: as many as fit in a quarter of the memory
                available now, up to 128), in batches of equal size: the search compiles for
                each size

        Raises:
            ValueError: if the engine isn't installed
        """
        self.game = game
        self.engine = engine
        self.engine_dir = engine_dir or engine.default_dir()
        if not engine.installed(self.engine_dir):
            raise ValueError(
                f"no {engine.name} in {self.engine_dir}: run "
                f"examples/hex/{engine.setup_script}"
            )
        self.levels = list(levels)
        self.sims = sims
        self.num_openings = openings
        self.num_games = 2 * openings
        self.seed = seed
        parallel = min(self.num_games, parallel or max_parallel_games(engine))
        # batches of equal size: a smaller last one would take about as long as a full one
        self.parallel = -(-self.num_games // -(-self.num_games // parallel))
        self.evaluator = game.make_test_evaluator(
            make_nn_eval_fn(network, state_to_nn_input), sims, noise=False
        )
        evaluator = self.evaluator
        self.search = jax.jit(
            jax.vmap(
                lambda k, tree, state, params: evaluator.evaluate(
                    key=k,
                    eval_state=tree,
                    env_state=state,
                    root_metadata=metadata(state),
                    params=params,
                    env_step_fn=game.step_fn,
                ),
                in_axes=(0, 0, 0, None),
            )
        )
        self.advance = jax.jit(
            jax.vmap(
                lambda tree, state, a: (
                    evaluator.step(tree, a),
                    game.env.step(state, a),
                )
            )
        )
        all_openings = load_openings(game, opening_list)
        rng = np.random.default_rng(seed)
        # game 2i plays opening i with us as colour 0, game 2i+1 with us as colour 1
        self.openings = {
            level: np.repeat(
                all_openings[
                    rng.choice(
                        len(all_openings),
                        size=openings,
                        replace=openings > len(all_openings),
                    )
                ],
                2,
                axis=0,
            )
            for level in self.levels
        }

    def name(self, level: int) -> str:
        """What a level's result is logged as: `mohex_level1000_sims1000` and so on."""
        return f"{self.engine.name}_level{level}_sims{self.sims}"

    def play(self, params: Any, level: int, record: bool = False) -> LevelResult:
        """Plays the games at `level`, our agent's network with `params` (as `make_nn_eval_fn`
        takes them); with `record`, records the first for the monitor."""
        start = time.perf_counter()
        num_games = self.num_games
        openings = self.openings[level]
        we_first = np.arange(num_games) % 2 == 0
        results = np.zeros(num_games, dtype=np.int32)
        keys = jax.random.split(
            jax.random.PRNGKey(self.seed * 100_000 + level), num_games
        )

        def search(k, trees, states):
            return self.search(k, trees, states, params)

        parallel = self.parallel
        # the first game of the first batch, its engine analysing
        recorder = GameRecorder(0, self.game, self.evaluator) if record else None
        with ThreadPoolExecutor(max_workers=available_cpus()) as pool:
            for begin in range(0, num_games, parallel):
                games = slice(begin, min(begin + parallel, num_games))
                count = games.stop - games.start
                recording = recorder if begin == 0 else None
                engines = list(
                    pool.map(
                        self.engine,
                        [self.engine_dir] * count,
                        [self.game] * count,
                        [level] * count,
                        [
                            recording is not None and g == recording.game
                            for g in range(count)
                        ],
                        range(1 + begin, 1 + begin + count),
                    )
                )
                try:
                    results[games] = play_games(
                        engines,
                        openings[games],
                        we_first[games],
                        search,
                        self.advance,
                        self.evaluator,
                        pool,
                        keys[games],
                        self.game,
                        recorder=recording,
                    )
                finally:
                    for e in engines:
                        e.close()
        episode = None
        if recorder is not None:
            episode = recorder.episode(int((openings[0] != NO_MOVE).sum()))
            episode.attrs = {"opponent": f"{self.engine.name} level {level}"}
        return LevelResult(
            level, results, we_first, time.perf_counter() - start, episode
        )

    def describe(self) -> dict[str, Any]:
        """The match's settings, for a run's configuration."""
        return {
            "engine": self.engine.name,
            "levels": self.levels,
            "sims": self.sims,
            "openings": self.num_openings,
            "seed": self.seed,
        }


def describe_scores(monitor: Monitor, names: list[str]) -> None:
    """Tells `monitor` that the metrics `names`, scores against an engine, are better higher and
    headline numbers. declared.json describes them, but can't say which way is better: its
    patterns for them also match the standard errors logged beside them, which would take it
    up."""
    for name in names:
        monitor.describe(name, better="higher", headline=True)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EngineMatchState(TestState):
    """What an `EngineMatchTester` keeps of the last networks of a run it's played, to log their
    mean once it has played them all. Kept with the run's saved state, so a resumed run has them
    too.

    Attributes:
        results: (last, levels, games) our result in each game (+1, -1) of the network after each
            of the run's last epochs, the oldest first
        played: (last,) which of them it has played
    """

    results: jax.Array
    played: jax.Array


class EngineMatchTester(BaseTester):
    """Plays an `EngineMatch` as a run trains: on the network after every `every` epochs (epochs
    every - 1, 2 * every - 1, ... counting from 0, as the run logs them), and after each of the
    last `last` epochs of a run of `num_epochs`, logging the mean of those too, at the last epoch,
    as `<name>_last<last>`, with its standard error (`_last<last>_se`): how much it would move
    with other openings, each game's results averaged over the networks first, since they all
    play the same games.

    Each level's result is logged as soon as it's played (see `LevelResult.metrics`), at the epoch
    of the network that played, then the time all of them took, as `<engine>_seconds`, so that a
    run's training time can be told from its time playing. With `results`, each is also appended
    to that file, as a line of JSON.

    It plays in the training process, on the same accelerator, between epochs: the search takes
    its memory from what the run has, and the engine processes it starts run on the cores the run
    may use (see `available_cpus`).
    """

    def __init__(
        self,
        match: EngineMatch,
        num_epochs: int,
        every: int = 0,
        last: int = 0,
        record: bool = False,
        results: Path | None = None,
        config: dict | None = None,
    ):
        """
        Args:
            match: the games to play
            num_epochs: epochs in the run
            every: play after every this many epochs (0 for never)
            last: play after each of the run's last this many epochs (0 for none)
            record: record a game at each level for the monitor
            results: (optional) also append each level's result to this file
            config: (optional) recorded with the tester's configuration, e.g. the dtype the
                network plays in; added to each result in `results` too
        """
        super().__init__(num_keys=1, name=match.engine.name)
        if not 0 <= last <= num_epochs:
            raise ValueError(f"last must be between 0 and {num_epochs}, got {last}")
        self.match = match
        self.num_epochs = num_epochs
        self.every = every
        self.last = last
        self.record = record
        self.results = results
        self.extra_config = config or {}

    def plays_at(self, epoch: int) -> bool:
        """Whether it plays the network after `epoch` (counting from 0)."""
        return (self.every > 0 and (epoch + 1) % self.every == 0) or (
            epoch >= self.num_epochs - self.last
        )

    def mean_name(self, level: int) -> str:
        return f"{self.match.name(level)}_last{self.last}"

    def score_names(self) -> list[str]:
        """The scores it logs (see `describe_scores`)."""
        names = [self.match.name(level) for level in self.match.levels]
        if self.last > 0:
            names += [self.mean_name(level) for level in self.match.levels]
        return names

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            **self.match.describe(),
            "every": self.every,
            "last": self.last,
            **self.extra_config,
        }

    def init(self, *args, **kwargs) -> EngineMatchState:
        shape = (self.last, len(self.match.levels), self.match.num_games)
        return EngineMatchState(
            results=jnp.zeros(shape, dtype=jnp.int8),
            played=jnp.zeros((self.last,), dtype=bool),
        )

    def last_metrics(self, state: EngineMatchState) -> dict[str, float]:
        """The mean of each level's scores over the last networks played, and its standard
        error."""
        played = np.asarray(state.played)
        results = np.asarray(state.results)[played]
        metrics = {}
        for i, level in enumerate(self.match.levels):
            # each game's points, averaged over the networks
            points = ((1 + np.sign(results[:, i])) / 2).mean(axis=0)
            name = self.mean_name(level)
            metrics[name] = float(points.mean())
            metrics[f"{name}_se"] = float(points.std() / np.sqrt(len(points)))
        return metrics

    def run(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        key: jax.Array,
        epoch_num: int,
        max_steps: int,
        env_step_fn: Any,
        env_init_fn: Any,
        evaluator: Any,
        state: EngineMatchState,
        params: Any,
        *args,
        log_fn: Callable[[dict], None] | None = None,
        activity_fn: Callable[[str], None] | None = None,
    ) -> tuple[EngineMatchState, dict, None]:
        """Plays the match at the epochs it's set to (see `BaseTester.run`), logging its results
        with `log_fn` as they come; returns no metrics or episode of its own."""
        if not self.plays_at(epoch_num):
            return state, {}, None
        start = time.perf_counter()
        results = []
        for level in self.match.levels:
            if activity_fn is not None:
                activity_fn(
                    f"{self.match.num_games} games against {self.match.engine.name} "
                    f"level {level}, {self.match.sims} sims"
                )
            result = self.match.play(params, level, record=self.record)
            print(f"epoch {epoch_num}: {self.match.engine.name} {result.summary()}")
            if log_fn is not None:
                log_fn(result.metrics(self.match.name(level)))
            if self.results is not None:
                with open(self.results, "a") as f:
                    record = result.record(
                        epoch=epoch_num,
                        engine=self.match.engine.name,
                        sims=self.match.sims,
                        **self.extra_config,
                    )
                    f.write(json.dumps(record) + "\n")
            results.append(result.results)
        metrics: dict[str, Any] = {
            f"{self.match.engine.name}_seconds": time.perf_counter() - start
        }
        slot = epoch_num - (self.num_epochs - self.last)
        if 0 <= slot < self.last:
            state = EngineMatchState(
                results=state.results.at[slot].set(np.stack(results).astype(np.int8)),
                played=state.played.at[slot].set(True),
            )
            if epoch_num == self.num_epochs - 1:
                metrics.update(self.last_metrics(state))
        if log_fn is not None:
            log_fn(metrics)
        return state, {}, None


def main():
    parser = argparse.ArgumentParser(
        description="Plays a train.py checkpoint against MoHex at fixed levels."
    )
    parser.add_argument("checkpoint", help="a checkpoint saved by train.py")
    parser.add_argument(
        "--size", type=int, choices=sorted(SIZES), default=11, help="the board size"
    )
    parser.add_argument("--engine", choices=sorted(ENGINES), default="mohex")
    parser.add_argument(
        "--net",
        default=None,
        help="the checkpoint's network, as train.py's --net names it (default: the size's)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="dtype the network computes in (bfloat16 is about twice as fast as float32, and "
        "self-play computes in it too, see train.py's --inference-dtype)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[100, 1000, 10000],
        help="MoHex's playouts a move",
    )
    parser.add_argument(
        "--sims", type=int, default=64, help="our MCTS iterations a move"
    )
    parser.add_argument(
        "--openings",
        type=int,
        default=128,
        help="openings per level, each played with both colours",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="games at a time (default: as many as fit in a quarter of the available memory, "
        "up to 128)",
    )
    parser.add_argument(
        "--skip-levels",
        type=int,
        nargs="+",
        default=[],
        help="levels of --levels not to play (e.g. done already), whose openings are still "
        "drawn, so that the other levels play the same games as without it",
    )
    parser.add_argument(
        "--stop-below",
        type=float,
        default=0.1,
        help="stop once our score against a level drops below this",
    )
    parser.add_argument(
        "--engine-dir", type=Path, default=None, help="where the engine is installed"
    )
    parser.add_argument(
        "--opening-list",
        default=None,
        help="an opening list (default: the board size's, see openings.py)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="also append each level's result to this file, as a line of JSON",
    )
    parser.add_argument(
        "--monitor",
        nargs="?",
        const=DEFAULT_URL,
        default=None,
        metavar="URL",
        help="also log each level's result, and a game at it, to the checkpoint's training run "
        f"on a seaglass server (default {DEFAULT_URL}), at the checkpoint's epoch",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="with --monitor, the training run's id on the server",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    epoch = None
    if args.monitor:
        if not args.run:
            parser.error("--monitor needs --run, the id in train.py's seaglass link")
        if not Path(args.checkpoint).stem.isdigit():
            parser.error(
                f"{args.checkpoint} isn't named after its epoch (as train.py's N.eqx)"
            )
        epoch = int(Path(args.checkpoint).stem)

    game = HexGame(args.size)
    network, params = game.load_checkpoint(args.checkpoint, args.net, args.dtype)
    try:
        match = EngineMatch(
            game,
            network,
            ENGINES[args.engine],
            args.levels,
            args.sims,
            args.openings,
            engine_dir=args.engine_dir,
            seed=args.seed,
            opening_list=args.opening_list,
            parallel=args.parallel,
        )
    except ValueError as e:
        parser.error(str(e))
    monitor = None
    if args.monitor:
        monitor = Monitor(args.monitor, project="hex")
        monitor.run_id = args.run
        describe_scores(monitor, [match.name(level) for level in args.levels])
    print(
        f"{args.checkpoint}: {args.sims} MCTS iterations a move in {args.dtype}, against "
        f"{args.engine} at levels {args.levels}; {args.openings} openings x both colours per "
        f"level, {match.parallel} games at a time",
        flush=True,
    )
    for level in args.levels:
        if level in args.skip_levels:
            continue
        result = match.play(params, level, record=monitor is not None)
        if args.results is not None:
            with open(args.results, "a") as f:
                record = result.record(
                    checkpoint=args.checkpoint,
                    engine=args.engine,
                    sims=args.sims,
                    dtype=args.dtype,
                )
                f.write(json.dumps(record) + "\n")
        if monitor is not None and epoch is not None:
            monitor.log(epoch, result.metrics(match.name(level)))
        print(f"  {args.engine} {result.summary()}", flush=True)
        if score(result.results)[0] < args.stop_below:
            break
    if monitor is not None:
        monitor.flush()


if __name__ == "__main__":
    main()
