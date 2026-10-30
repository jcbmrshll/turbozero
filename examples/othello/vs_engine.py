"""Plays a checkpoint from `train.py` against a strong Othello engine, Edax or Egaroucid,
at a series of levels.

    examples/othello/setup_edax.sh         # once: builds Edax
    uv run examples/othello/vs_engine.py CHECKPOINT --engine edax --levels 2 4 6 8 --sims 400

Games start from XOT openings (see `xot.py`), each played twice with the colors
swapped, so that two deterministic players give varied, balanced games. The same
`--seed` always draws the same openings, so checkpoints play the same games. The engine plays
at the given level with its opening book off: at level L (up to 10) it searches L moves ahead,
and solves the game exactly once 2L or fewer squares are empty. Our agent runs
`--sims` MCTS iterations a move at temperature 0, keeping its search tree between moves,
without the root noise used in self-play, its network computing in `--dtype` (bfloat16,
as in self-play, unless told otherwise). This is how OLIVAW was compared with Edax
(https://arxiv.org/abs/2103.17228).

Games are played in batches of equal size, each game with its own engine process (see
`engines.py`): as many at a time as fit in a quarter of the available memory, up to 128,
since the machine may be shared. The games of a batch move in step: the engine moves in
all of them, then our agent's moves are searched together on the accelerator. Each game's
result is checked against the engine's own board.

With --episodes, the first game at each level is also saved for the monitor (see `watch.py`),
as `{engine}_level{L}_sims{S}.npz`: its positions, what our agent's search made of each, and
what the engine's made of those it moved from (see `engines.GTPEngine.analysis`), read from
its logs as it plays: Edax's scores of every legal move, each with the line it expects after
it, Egaroucid's of the move it plays. Edax's analyst (`engines.EdaxAnalyst`), a process of
its own, then scores every legal move of the positions it didn't move from, after the games
at the level, so that searching them doesn't change the moves Edax plays.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from operator import methodcaller
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from engines import ANALYSTS, ENGINES, Analysis, EdaxAnalyst, GTPEngine
from game import (
    DEFAULT_NET,
    PASS,
    env,
    load_checkpoint,
    make_test_evaluator,
    metadata,
    state_to_nn_input,
    step_fn,
)
from seaglass import Arrays
from xot import XOT_PATH, load_xot

from core.common import GameFrame
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.monitoring.episodes import BOARDS, line_record, pgx_two_player_episode

# the most discs a game can be won by: an engine's score, a disc margin, over this is a value
MAX_MARGIN = 64


def select(which, on_true, on_false):
    """Game by game, `on_true` where `which` is set, else `on_false`."""
    which = jnp.asarray(which)
    return jax.tree.map(
        lambda t, f: jnp.where(which.reshape((-1,) + (1,) * (t.ndim - 1)), t, f),
        on_true,
        on_false,
    )


def color(ply: int) -> str:
    """Who moves at `ply`: black ("b") first. Passes count as moves."""
    return "b" if ply % 2 == 0 else "w"


def stack(trees):
    """Trees stacked along a new leading axis."""
    return jax.tree.map(lambda *x: np.stack(x), *trees)


class GameRecorder:
    """One game of a batch, recorded for the monitor: each position, what our agent's search
    made of it (its search, if it moved, else the tree it kept from its last one) and what
    the engine's made of those it moved from. Its engine has to be analysing (`analyse`)."""

    def __init__(self, game: int, evaluator):
        self.game = game
        self.search_record = jax.jit(evaluator.search_record)
        self.states = []
        # what each made of each position played from, None for nothing
        self.trees, self.analyses = [], []

    def take(self, batch):
        return jax.device_get(jax.tree.map(lambda x: x[self.game], batch))

    def start(self, states, our_id: int, opening: int) -> None:
        """The game's first position, in `states`, where our agent is player `our_id`, and
        the players start playing after `opening` moves played for them."""
        self.our_id = our_id
        self.opening = opening
        self.states.append(self.take(states))

    def moved(self, states, trees=None, analysis: list[Analysis] | None = None) -> None:
        """The game's next position, in `states`, after a move from the last, which our
        agent had `trees` of and the engine made `analysis` of."""
        tree = None if trees is None else self.search_record(self.take(trees))
        self.trees.append(None if tree is None else jax.device_get(tree))
        self.analyses.append(analysis)
        self.states.append(self.take(states))

    def episode(self, analyst: EdaxAnalyst | None = None) -> Arrays:
        """The game, for the monitor, with `analyst`'s analysis of the positions the engine
        didn't move from, if there is one."""
        num_actions = env.num_actions
        for t, state in enumerate(self.states[:-1]):
            # a position with only a pass has nothing to score
            if (
                analyst is None
                or self.analyses[t]
                or not state.legal_action_mask[:PASS].any()
            ):
                continue
            board, to_move = BOARDS["othello"](
                jax.tree.map(lambda x: x[None], state._x)
            )
            color = "b" if int(to_move[0]) == 0 else "w"
            self.analyses[t] = analyst.analyse(board.reshape(-1), color)
        capacity = 1 + max(
            (sum(len(a.line) for a in moves) for moves in self.analyses if moves),
            default=0,
        )
        records = [
            line_record(
                moves
                and [
                    (a.line, float(np.clip(a.score / MAX_MARGIN, -1, 1)), a.bound)
                    for a in moves
                ],
                max((a.depth for a in moves), default=0) if moves else 0,
                num_actions=num_actions,
                capacity=capacity,
            )
            for moves in self.analyses
        ]
        # each one's value of each position, from its own side, as of its last look; a
        # root's q is for the player to move
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
            step_fn,
            start=self.opening,
        )


def play_games(
    engines, openings, we_black, search, advance, evaluator, pool, keys, recorder=None
):
    """Plays one game per engine process, from `openings` (one per game), with our agent
    as black in the games marked `we_black`, recording one of them with `recorder`. Each
    game has its own key, of `keys`, so that its moves don't depend on the other games.

    Returns:
        our result in each game: +1 win, 0 draw, -1 loss
    """
    num_games = len(engines)
    states = jax.vmap(env.init)(keys)
    trees = jax.vmap(lambda s: evaluator.init(template_embedding=s))(states)
    # pgx picks which player id moves first (black) at random
    black_id = np.asarray(states.current_player)
    our_id = np.where(we_black, black_id, 1 - black_id)
    if recorder is not None:
        recorder.start(states, int(our_id[recorder.game]), openings.shape[1])

    # play the openings
    for ply in range(openings.shape[1]):
        moves = openings[:, ply]
        assert np.asarray(states.legal_action_mask)[np.arange(num_games), moves].all()
        list(pool.map(GTPEngine.play, engines, [color(ply)] * num_games, moves))
        trees, states = advance(trees, states, jnp.asarray(moves))
        if recorder is not None:
            recorder.moved(states)

    # the games step in turn: wherever it's the engine's move, the engine moves, and only once
    # it has moved in every game is our agent's move searched, in all of them at once. The
    # players alternate (a pass is a move), so after the engine's first move as black the
    # games stay in step, every search a move of ours
    while not np.asarray(states.terminated).all():
        done = np.asarray(states.terminated)
        mask = np.asarray(states.legal_action_mask)
        to_move = np.asarray(states.current_player)
        colors = np.where(to_move == black_id, "b", "w")
        theirs = (to_move != our_id) & ~done
        moves = np.full(num_games, PASS, dtype=np.int32)
        # our agent's tree of each position moved from: the one it kept, or its search
        trees_before = trees
        asking = []
        if theirs.any():
            # the engine's moves, all games at once (a side with no move passes without
            # asking)
            movers = theirs
            asking = [g for g in np.flatnonzero(theirs) if not mask[g, PASS]]
            replies = pool.map(
                GTPEngine.genmove, [engines[g] for g in asking], colors[asking]
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
                    GTPEngine.play,
                    [engines[g] for g in moved],
                    colors[moved],
                    moves[moved],
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
                states, trees_before, engines[g].analysis if g in asking else None
            )

    results = np.asarray(states.rewards)[np.arange(num_games), our_id]
    # check every result against the engine's own board
    for g, (black, white) in enumerate(pool.map(methodcaller("discs"), engines)):
        engine_result = np.sign(black - white) * (1 if we_black[g] else -1)
        assert engine_result == np.sign(results[g]), (
            f"game {g}: the engine disagrees on the result"
        )
    return results


def score(results) -> tuple[float, float]:
    """Mean score (a draw counts half) and its standard error."""
    points = (1 + np.sign(results)) / 2
    return points.mean(), points.std() / np.sqrt(len(points))


def available_memory() -> int:
    """Bytes of memory available to new processes: free memory plus what the kernel
    could reclaim from caches (MemAvailable on Linux), or just free memory elsewhere."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def max_parallel_games(engine: type[GTPEngine], limit: int = 128) -> int:
    """How many engine processes to run at once: as many as fit in a quarter of the
    memory available now (the machine may be shared), up to `limit`."""
    fit = int(0.25 * available_memory() / engine.memory_per_process)
    return max(2, min(fit, limit) // 2 * 2)


def main():
    parser = argparse.ArgumentParser(
        description="Plays a train.py checkpoint against Edax or Egaroucid at fixed levels."
    )
    parser.add_argument("checkpoint", help="a checkpoint saved by train.py")
    parser.add_argument("--engine", choices=sorted(ENGINES), default="edax")
    parser.add_argument(
        "--net",
        default=DEFAULT_NET,
        help="the checkpoint's network, as train.py's --net names it"
        " (resnet:6x128 for a run trained with the old default)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="dtype the network computes in (bfloat16 is about twice as fast as float32, "
        "and self-play computes in it too, see train.py's --inference-dtype)",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    parser.add_argument(
        "--sims", type=int, default=64, help="our MCTS iterations a move"
    )
    parser.add_argument(
        "--openings",
        type=int,
        default=128,
        help="XOT openings per level, each played with both colors",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="games at a time (default: as many as fit in a quarter of the available "
        "memory, up to 128)",
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
    parser.add_argument("--xot", default=XOT_PATH, help="the XOT opening list")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="also append each level's result to this file, as a line of JSON",
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=None,
        help="also save the first game at each level here, for the monitor",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    engine = ENGINES[args.engine]
    engine_dir = args.engine_dir or engine.default_dir()
    if not engine.installed(engine_dir):
        parser.error(
            f"no {engine.name} in {engine_dir}: run examples/othello/{engine.setup_script}"
        )
    network, params = load_checkpoint(args.checkpoint, args.net, args.dtype)
    evaluator = make_test_evaluator(
        make_nn_eval_fn(network, state_to_nn_input), args.sims, noise=False
    )
    search = jax.jit(
        jax.vmap(
            lambda k, tree, state: evaluator.evaluate(
                key=k,
                eval_state=tree,
                env_state=state,
                root_metadata=metadata(state),
                params=params,
                env_step_fn=step_fn,
            )
        )
    )
    advance = jax.jit(
        jax.vmap(lambda tree, state, a: (evaluator.step(tree, a), env.step(state, a)))
    )
    xot = load_xot(args.xot)
    num_games = 2 * args.openings
    parallel = min(num_games, args.parallel or max_parallel_games(engine))
    # batches of equal size: a smaller last one would take about as long as a full one
    parallel = -(-num_games // -(-num_games // parallel))
    rng = np.random.default_rng(args.seed)
    print(
        f"{args.checkpoint}: {args.sims} MCTS iterations a move in {args.dtype}, against "
        f"{engine.name} at "
        f"levels {args.levels}; {args.openings} XOT openings x both colors per level, "
        f"{parallel} games at a time",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for level in args.levels:
            start = time.perf_counter()
            # game 2i plays opening i with us as black, game 2i+1 with us as white
            openings = np.repeat(
                xot[rng.choice(len(xot), size=args.openings, replace=False)], 2, axis=0
            )
            if level in args.skip_levels:
                continue
            we_black = np.arange(num_games) % 2 == 0
            results = np.zeros(num_games, dtype=np.int32)
            keys = jax.random.split(
                jax.random.PRNGKey(args.seed * 100_000 + level * 1000), num_games
            )
            # the first game of the first batch, its engine analysing
            recorder = GameRecorder(0, evaluator) if args.episodes else None
            for begin in range(0, num_games, parallel):
                games = slice(begin, min(begin + parallel, num_games))
                count = games.stop - games.start
                recording = recorder if begin == 0 else None
                engines = list(
                    pool.map(
                        engine,
                        [engine_dir] * count,
                        [level] * count,
                        [
                            recording is not None and g == recording.game
                            for g in range(count)
                        ],
                    )
                )
                try:
                    results[games] = play_games(
                        engines,
                        openings[games],
                        we_black[games],
                        search,
                        advance,
                        evaluator,
                        pool,
                        keys[games],
                        recorder=recording,
                    )
                finally:
                    for e in engines:
                        e.close()
            if recorder is not None:
                analyst = ANALYSTS.get(engine.name)
                analyst = analyst and analyst(engine_dir, level)
                try:
                    episode = recorder.episode(analyst)
                finally:
                    if analyst is not None:
                        analyst.close()
                args.episodes.mkdir(parents=True, exist_ok=True)
                name = f"{engine.name}_level{level}_sims{args.sims}.npz"
                (args.episodes / name).write_bytes(episode.encode())
            mean, se = score(results)
            seconds = time.perf_counter() - start
            if args.results is not None:
                with open(args.results, "a") as f:
                    record = {
                        "checkpoint": args.checkpoint,
                        "engine": engine.name,
                        "level": level,
                        "sims": args.sims,
                        "dtype": args.dtype,
                        "games": num_games,
                        "score": float(mean),
                        "se": float(se),
                        "win": float((results > 0).mean()),
                        "draw": float((results == 0).mean()),
                        "loss": float((results < 0).mean()),
                        "seconds": round(seconds, 1),
                    }
                    f.write(json.dumps(record) + "\n")
            print(
                f"  {engine.name} level {level:2d}: score {mean:.3f} ± {se:.3f}  "
                f"(win {(results > 0).mean():.3f}, draw {(results == 0).mean():.3f}, "
                f"loss {(results < 0).mean():.3f}; as black {score(results[we_black])[0]:.2f}, "
                f"as white {score(results[~we_black])[0]:.2f})  {seconds:.0f}s",
                flush=True,
            )
            if mean < args.stop_below:
                break


if __name__ == "__main__":
    main()
