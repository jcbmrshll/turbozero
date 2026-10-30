"""Plays a checkpoint from `train.py` against the solver at a series of levels.

    examples/connect_four/setup_solver.sh     # once: builds the solver
    uv run examples/connect_four/vs_solver.py CHECKPOINT --levels 0 2 4 8 42 --sims 256

At level L the solver looks L moves ahead, as a minimax search without an evaluation would, and
otherwise plays at random (`solver.choose_move`): level 0 is a random player, and level 42
sees every game to its end, playing perfectly. Games start from the openings of `openings.py`,
each played twice with the colors swapped, so that a deterministic player gets varied games, and
the same `--seed` always draws the same openings. Our agent runs `--sims` MCTS iterations a move
at temperature 0, keeping its search tree between moves, without the root noise used in
self-play, its network computing in `--dtype` (bfloat16, as in self-play, unless told otherwise).

Each opening's two games have a known result under perfect play: one side's win and the other's
loss, or two draws. So a perfect player scores exactly 0.5 against level 42, whatever the openings,
and anything less is the games our agent let slip.

All the games of a level move in step: the solver moves in all of them, then our agent's moves
are searched together on the accelerator. The solver's processes run on the cores this process may
use (e.g. with `taskset`).

train.py plays these games as it trains (`SolverMatchTester`, see its --vs-engine-every and
--vs-engine-last), logging them to the run. Here, with --monitor and --run, a checkpoint's
results are logged to its training run afterwards, at the checkpoint's epoch, as
`solver_level{L}_sims{S}` (the score, and `_se`, `_win`, `_draw` and `_loss`).

Logged to a run, the first game at each level goes with them, as `solver_level{L}_sims{S}_game`:
its positions, what our agent's search made of each, and the truth about each: how every legal
move's game ends under perfect play, with the line that gets there.
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from game import (
    DEFAULT_NET,
    env,
    load_checkpoint,
    make_test_evaluator,
    metadata,
    state_to_nn_input,
    step_fn,
)
from openings import OPENINGS_PATH, load_openings
from seaglass import Arrays
from solver import ILLEGAL, PERFECT_LEVEL, Solver, choose_move, sequence, wins_now

from core.common import GameFrame
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.mcts import MCTS
from core.monitoring import DEFAULT_URL, Monitor
from core.monitoring.episodes import LineRecord, line_record, pgx_two_player_episode
from core.testing.tester import BaseTester, TestState

# moves of each line a recorded game shows
LINE_PLIES = 8


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


def perfect_lines(
    solver, position: str, max_plies: int = LINE_PLIES
) -> list[tuple[list[int], float, int]]:
    """What the solver knows of a position: for each legal move, the line it expects after it
    (each side's best move, the lowest column of those equally good), up to `max_plies` moves, and
    how the game ends, for the player making the move (+1, 0 or -1, always exact), as
    `line_record` takes them."""
    scores = solver.analyse([position])[0]
    ply = len(position)
    lines = []
    for column in np.flatnonzero(scores != ILLEGAL):
        column = int(column)
        line = [column]
        score = int(scores[column])
        at = position
        step = ply
        while len(line) < max_plies and not (
            wins_now(score, step) or step + 1 == PERFECT_LEVEL
        ):
            at += sequence(line[-1:])
            step += 1
            next_scores = solver.analyse([at])[0]
            best = int(np.argmax(next_scores))
            line.append(best)
            score = int(next_scores[best])
        lines.append((line, float(np.sign(scores[column])), 0))
    return lines


class GameRecorder:
    """One game of a batch, recorded for the monitor: each position, what our agent's search
    made of it (its search, if it moved, else the tree it kept from its last one), and the
    solver's verdict on it."""

    def __init__(self, game: int, evaluator):
        self.game = game
        self.search_record = jax.jit(evaluator.search_record)
        self.states = []
        # what our agent made of each position played from, None for nothing
        self.trees = []

    def take(self, batch):
        return jax.device_get(jax.tree.map(lambda x: x[self.game], batch))

    def start(self, states, our_id: int, opening: int) -> None:
        """The game's first position, in `states`, where our agent is player `our_id`, and
        the players start playing after `opening` moves played for them."""
        self.our_id = our_id
        self.opening = opening
        self.states.append(self.take(states))

    def moved(self, states, trees=None) -> None:
        """The game's next position, in `states`, after a move from the last, which our
        agent had `trees` of."""
        tree = None if trees is None else self.search_record(self.take(trees))
        self.trees.append(None if tree is None else jax.device_get(tree))
        self.states.append(self.take(states))

    def records(self, solver, position: str) -> list[LineRecord]:
        """The solver's verdict on each position played from, `position` being the whole game's
        moves."""
        lines = [
            perfect_lines(solver, position[:t]) for t in range(len(self.states) - 1)
        ]
        capacity = 1 + max(sum(len(line) for line, _, _ in ls) for ls in lines)
        return [
            line_record(
                ls,
                PERFECT_LEVEL - t,
                num_actions=env.num_actions,
                capacity=capacity,
            )
            for t, ls in enumerate(lines)
        ]

    def episode(self, solver, position: str) -> Arrays:
        """The game, for the monitor, from the whole game's moves as `position`."""
        records = self.records(solver, position)
        # each one's value of each position, from its own side, as of its last look; a
        # root's q is for the player to move
        values = np.zeros((len(self.states), 2), dtype=np.float32)
        for t, state in enumerate(self.states):
            values[t] = values[t - 1] if t else 0
            ours = int(state.current_player) == self.our_id
            if t < len(self.trees) and self.trees[t] is not None:
                q = float(self.trees[t].q[0])
                values[t, 0] = q if ours else -q
            if t < len(records):
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
                [
                    line_record(
                        None,
                        num_actions=env.num_actions,
                        capacity=records[0].parents.shape[0],
                    )
                ]
                + records
            ),
        )
        return pgx_two_player_episode()(
            frames,
            np.array([self.our_id, 1 - self.our_id]),
            step_fn,
            start=self.opening,
        )


def play_games(
    solver,
    level,
    openings,
    we_first,
    search,
    advance,
    evaluator,
    keys,
    rngs,
    recorder=None,
) -> tuple[np.ndarray, list[str]]:
    """Plays one game per row of `openings` against the solver at `level`, with our agent moving
    first in the games marked `we_first`, recording one of them with `recorder`. Each game has
    its own key, of `keys`, and generator, of `rngs`, so that its moves don't depend on the other
    games.

    Returns:
        our result in each game (+1 win, 0 draw, -1 loss), and each game's moves as the solver
        writes positions
    """
    num_games = len(openings)
    states = jax.vmap(env.init)(keys)
    trees = jax.vmap(lambda s: evaluator.init(template_embedding=s))(states)
    # pgx picks which player id moves first at random
    first_id = np.asarray(states.current_player)
    our_id = np.where(we_first, first_id, 1 - first_id)
    if recorder is not None:
        recorder.start(states, int(our_id[recorder.game]), openings.shape[1])

    positions = [""] * num_games
    # play the openings
    for ply in range(openings.shape[1]):
        moves = openings[:, ply]
        assert np.asarray(states.legal_action_mask)[np.arange(num_games), moves].all()
        trees, states = advance(trees, states, jnp.asarray(moves))
        positions = [p + sequence([m]) for p, m in zip(positions, moves)]
        if recorder is not None:
            recorder.moved(states)

    # the games step in turn: wherever it's the solver's move, the solver moves, and only once it
    # has moved in every game is our agent's move searched, in all of them at once
    while not np.asarray(states.terminated).all():
        done = np.asarray(states.terminated)
        mask = np.asarray(states.legal_action_mask)
        to_move = np.asarray(states.current_player)
        theirs = (to_move != our_id) & ~done
        moves = np.zeros(num_games, dtype=np.int32)
        # our agent's tree of each position moved from: the one it kept, or its search
        trees_before = trees
        if theirs.any():
            movers = theirs
            games = np.flatnonzero(theirs)
            scores = solver.analyse([positions[g] for g in games])
            for g, row in zip(games, scores):
                moves[g] = choose_move(row, len(positions[g]), level, rngs[g])
            # the solver's idea of the moves that end the game has to be the environment's
            won = [
                wins_now(int(r[moves[g]]), len(positions[g]))
                for g, r in zip(games, scores)
            ]
        else:
            # our moves, every game in one batch
            movers = ~done
            won = None
            games = np.flatnonzero(movers)
            # a key for each game and ply
            out = search(
                jax.vmap(jax.random.fold_in)(keys, states._step_count), trees, states
            )
            moves[movers] = np.asarray(out.action)[movers]
            trees = trees_before = select(movers, out.eval_state, trees)
        assert mask[movers, moves[movers]].all(), "an illegal move"
        # step the games that moved; the others stay as they are
        new_trees, new_states = advance(trees, states, jnp.asarray(moves))
        trees = select(movers, new_trees, trees)
        states = select(movers, new_states, states)
        for g in games:
            positions[g] += sequence([moves[g]])
        if won is not None:
            ended = np.asarray(states.terminated)[games]
            full = np.array([len(positions[g]) == PERFECT_LEVEL for g in games])
            assert (ended == (np.array(won) | full)).all(), (
                "the solver and the environment disagree on how a game ended"
            )
        if recorder is not None and movers[recorder.game]:
            recorder.moved(states, trees_before)

    return np.asarray(states.rewards)[np.arange(num_games), our_id], positions


def score(results) -> tuple[float, float]:
    """Mean score (a draw counts half) and its standard error."""
    points = (1 + np.sign(results)) / 2
    return points.mean(), points.std() / np.sqrt(len(points))


@dataclass(frozen=True)
class LevelResult:
    """Our agent's games against the solver at one level.

    Attributes:
        level: the solver's level
        results: our result in each game: +1 win, 0 draw, -1 loss
        we_first: which games we moved first in
        seconds: how long the games took (and recording the first, if it was)
        episode: the first game, for the monitor, if it was recorded
    """

    level: int
    results: np.ndarray
    we_first: np.ndarray
    seconds: float
    episode: Arrays | None = None

    def metrics(self, name: str) -> dict[str, Any]:
        """What the monitor logs of it, under `name` (`SolverMatch.name`): the score, its
        standard error, the fractions of games won, drawn and lost, and the game."""
        mean, se = score(self.results)
        metrics: dict[str, Any] = {
            name: float(mean),
            f"{name}_se": float(se),
            f"{name}_win": float((self.results > 0).mean()),
            f"{name}_draw": float((self.results == 0).mean()),
            f"{name}_loss": float((self.results < 0).mean()),
        }
        if self.episode is not None:
            metrics[f"{name}_game"] = self.episode
        return metrics

    def record(self, **extra: Any) -> dict[str, Any]:
        """The result as a line of a results file: `extra` (e.g. which checkpoint played),
        then the games' score, standard error, outcomes and duration."""
        mean, se = score(self.results)
        return {
            **extra,
            "level": self.level,
            "games": len(self.results),
            "score": float(mean),
            "se": float(se),
            "win": float((self.results > 0).mean()),
            "draw": float((self.results == 0).mean()),
            "loss": float((self.results < 0).mean()),
            "seconds": round(self.seconds, 1),
        }

    def summary(self) -> str:
        """A line for the console."""
        mean, se = score(self.results)
        first = score(self.results[self.we_first])[0]
        second = score(self.results[~self.we_first])[0]
        return (
            f"level {self.level:2d}: score {mean:.3f} ± {se:.3f}  "
            f"(win {(self.results > 0).mean():.3f}, draw {(self.results == 0).mean():.3f}, "
            f"loss {(self.results < 0).mean():.3f}; moving first {first:.2f}, "
            f"second {second:.2f})  {self.seconds:.0f}s"
        )


class SolverMatch:
    """Our agent, `network` searching `sims` MCTS iterations a move, against the solver at each
    of `levels`: 2 * `openings` games a level, from openings each played with both colors. The
    openings are drawn level by level from `seed`, so every network it plays plays them all, and
    so does every match with the same levels and seed, here or in train.py (see
    `SolverMatchTester`).

    The search is compiled once, for whatever parameters it's given: train.py plays the
    network as it trains, epoch after epoch, without compiling again.
    """

    def __init__(
        self,
        network,
        solver: Solver,
        levels: list[int],
        sims: int,
        openings: int,
        seed: int = 0,
        openings_path: str | Path = OPENINGS_PATH,
        search_backend: type[MCTS] = MCTS,
        **search_kwargs,
    ):
        """
        Args:
            network: the network (see game.make_network), computing in the dtype to play in
            solver: the solver
            levels: the levels to play at
            sims: our MCTS iterations a move
            openings: openings a level, each played with both colors
            seed: draws the openings, and seeds the games
            openings_path: the opening list
            search_backend: our agent's search, with its own settings in `search_kwargs`
        """
        self.solver = solver
        self.levels = list(levels)
        self.sims = sims
        self.num_openings = openings
        self.num_games = 2 * openings
        self.seed = seed
        self.evaluator = make_test_evaluator(
            make_nn_eval_fn(network, state_to_nn_input),
            sims,
            noise=False,
            search=search_backend,
            **search_kwargs,
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
                    env_step_fn=step_fn,
                ),
                in_axes=(0, 0, 0, None),
            )
        )
        self.advance = jax.jit(
            jax.vmap(
                lambda tree, state, a: (evaluator.step(tree, a), env.step(state, a))
            )
        )
        opening_list = load_openings(openings_path)
        rng = np.random.default_rng(seed)
        # game 2i plays opening i with us first, game 2i+1 with us second
        self.openings = {
            level: np.repeat(
                opening_list[
                    rng.choice(len(opening_list), size=openings, replace=False)
                ],
                2,
                axis=0,
            )
            for level in self.levels
        }

    def name(self, level: int) -> str:
        """What a level's result is logged as: `solver_level42_sims256` and so on."""
        return f"{self.solver.name}_level{level}_sims{self.sims}"

    def play(self, params: Any, level: int, record: bool = False) -> LevelResult:
        """Plays the games at `level`, our agent's network with `params` (as
        `make_nn_eval_fn` takes them); with `record`, records the first for the monitor."""
        start = time.perf_counter()
        we_first = np.arange(self.num_games) % 2 == 0
        keys = jax.random.split(
            jax.random.PRNGKey(self.seed * 100_000 + level * 1000), self.num_games
        )
        rngs = [
            np.random.default_rng([self.seed, level, g]) for g in range(self.num_games)
        ]

        def search(k, trees, states):
            return self.search(k, trees, states, params)

        recorder = GameRecorder(0, self.evaluator) if record else None
        results, positions = play_games(
            self.solver,
            level,
            self.openings[level],
            we_first,
            search,
            self.advance,
            self.evaluator,
            keys,
            rngs,
            recorder=recorder,
        )
        episode = None
        if recorder is not None:
            episode = recorder.episode(self.solver, positions[recorder.game])
            episode.attrs = {"opponent": f"{self.solver.name} level {level}"}
        return LevelResult(
            level, results, we_first, time.perf_counter() - start, episode
        )

    def describe(self) -> dict[str, Any]:
        """The match's settings, for a run's configuration."""
        return {
            "engine": self.solver.name,
            "levels": self.levels,
            "sims": self.sims,
            "openings": self.num_openings,
            "seed": self.seed,
        }


def describe_scores(monitor: Monitor, names: list[str]) -> None:
    """Tells `monitor` that the metrics `names`, scores against the solver, are better
    higher and headline numbers. declared.json describes them, but can't say which way is
    better: its patterns for them also match the standard errors and fractions of draws
    logged beside them, which would take it up."""
    for name in names:
        monitor.describe(name, better="higher", headline=True)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolverMatchState(TestState):
    """What a `SolverMatchTester` keeps of the last networks of a run it's played, to log
    their mean once it has played them all. Kept with the run's saved state, so a resumed
    run has them too.

    Attributes:
        results: (last, levels, games) our result in each game (+1, 0, -1) of the network
            after each of the run's last epochs, the oldest first
        played: (last,) which of them it has played
    """

    results: jax.Array
    played: jax.Array


class SolverMatchTester(BaseTester):
    """Plays a `SolverMatch` as a run trains: on the network after every `every` epochs
    (epochs every - 1, 2 * every - 1, ... counting from 0, as the run logs them), and after
    each of the last `last` epochs of a run of `num_epochs`, logging the mean of those too,
    at the last epoch, as `<name>_last<last>`, with its standard error (`_last<last>_se`):
    how much it would move with other openings, each game's results averaged over the
    networks first, since they all play the same openings.

    Each level's result is logged as soon as it's played (see `LevelResult.metrics`), at the
    epoch of the network that played, then the time all of them took, as `solver_seconds`,
    so that a run's training time can be told from its time playing. With `results`, each is
    also appended to that file, as a line of JSON.

    It plays in the training process, on the same accelerator, between epochs: the search
    takes its memory from what the run has, and the solver's processes run on the cores the
    run may use.
    """

    def __init__(
        self,
        match: SolverMatch,
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
            config: (optional) recorded with the tester's configuration, e.g. the dtype
                the network plays in; added to each result in `results` too
        """
        super().__init__(num_keys=1, name=match.solver.name)
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

    def init(self, *args, **kwargs) -> SolverMatchState:
        shape = (self.last, len(self.match.levels), self.match.num_games)
        return SolverMatchState(
            results=jnp.zeros(shape, dtype=jnp.int8),
            played=jnp.zeros((self.last,), dtype=bool),
        )

    def last_metrics(self, state: SolverMatchState) -> dict[str, float]:
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
        state: SolverMatchState,
        params: Any,
        *args,
        log_fn: Callable[[dict], None] | None = None,
        activity_fn: Callable[[str], None] | None = None,
    ) -> tuple[SolverMatchState, dict, None]:
        """Plays the match at the epochs it's set to (see `BaseTester.run`), logging its
        results with `log_fn` as they come; returns no metrics or episode of its own."""
        if not self.plays_at(epoch_num):
            return state, {}, None
        start = time.perf_counter()
        results = []
        for level in self.match.levels:
            if activity_fn is not None:
                activity_fn(
                    f"{self.match.num_games} games against the solver at level {level}, "
                    f"{self.match.sims} sims"
                )
            result = self.match.play(params, level, record=self.record)
            print(f"epoch {epoch_num}: solver {result.summary()}")
            if log_fn is not None:
                log_fn(result.metrics(self.match.name(level)))
            if self.results is not None:
                with open(self.results, "a") as f:
                    record = result.record(
                        epoch=epoch_num,
                        engine=self.match.solver.name,
                        sims=self.match.sims,
                        **self.extra_config,
                    )
                    f.write(json.dumps(record) + "\n")
            results.append(result.results)
        metrics: dict[str, Any] = {
            f"{self.match.solver.name}_seconds": time.perf_counter() - start
        }
        slot = epoch_num - (self.num_epochs - self.last)
        if 0 <= slot < self.last:
            state = SolverMatchState(
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
        description="Plays a train.py checkpoint against the Connect Four solver at fixed levels."
    )
    parser.add_argument("checkpoint", help="a checkpoint saved by train.py")
    parser.add_argument(
        "--net",
        default=DEFAULT_NET,
        help="the checkpoint's network, as train.py's --net names it",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="dtype the network computes in (self-play computes in bfloat16 too, see "
        "train.py's --inference-dtype)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[0, 2, 4, 6, 8, PERFECT_LEVEL],
        help=f"the solver's levels: moves it looks ahead, 0 for a random player, "
        f"{PERFECT_LEVEL} for a perfect one",
    )
    parser.add_argument(
        "--sims", type=int, default=256, help="our MCTS iterations a move"
    )
    parser.add_argument(
        "--openings",
        type=int,
        default=256,
        help="openings per level, each played with both colors",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="solver processes (default: one a core this process may use, up to 8)",
    )
    parser.add_argument(
        "--solver-dir", type=Path, default=None, help="where the solver is installed"
    )
    parser.add_argument(
        "--openings-path", default=OPENINGS_PATH, help="the opening list"
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
        help="also log each level's result, and a game at it, to the checkpoint's training "
        f"run on a seaglass server (default {DEFAULT_URL}), at the checkpoint's epoch",
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

    network, params = load_checkpoint(args.checkpoint, args.net, args.dtype)
    try:
        solver = Solver(args.solver_dir, args.workers)
    except ValueError as e:
        parser.error(str(e))
    match = SolverMatch(
        network,
        solver,
        args.levels,
        args.sims,
        args.openings,
        seed=args.seed,
        openings_path=args.openings_path,
    )
    monitor = None
    if args.monitor:
        monitor = Monitor(args.monitor, project="connect_four")
        monitor.run_id = args.run
        describe_scores(monitor, [match.name(level) for level in args.levels])
    print(
        f"{args.checkpoint}: {args.sims} MCTS iterations a move in {args.dtype}, against the "
        f"solver at levels {args.levels}; {args.openings} openings x both colors per level",
        flush=True,
    )
    for level in args.levels:
        result = match.play(params, level, record=monitor is not None)
        if args.results is not None:
            with open(args.results, "a") as f:
                record = result.record(
                    checkpoint=args.checkpoint,
                    engine=solver.name,
                    sims=args.sims,
                    dtype=args.dtype,
                )
                f.write(json.dumps(record) + "\n")
        if monitor is not None and epoch is not None:
            monitor.log(epoch, result.metrics(match.name(level)))
        print(f"  solver {result.summary()}", flush=True)
    if monitor is not None:
        monitor.flush()
    solver.close()


if __name__ == "__main__":
    main()
