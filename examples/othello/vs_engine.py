"""Plays a checkpoint from `train.py` against a strong Othello engine, Edax or Egaroucid,
at a series of fixed search depths.

    examples/othello/setup_edax.sh         # once: builds Edax
    uv run examples/othello/vs_engine.py CHECKPOINT --engine edax --levels 2 4 6 8 --sims 400

Games start from XOT openings (see `xot.py`), each played twice with the colors
swapped, so that two deterministic players give varied, balanced games. The same
`--seed` always draws the same openings, so checkpoints play the same games. The engine searches
to the given depth ("level", in its own terms) with its opening book off; our agent runs
`--sims` MCTS iterations a move at temperature 0, keeping its search tree between moves,
without the root noise used in self-play. This is how OLIVAW was compared with Edax
(https://arxiv.org/abs/2103.17228).

Games are played in batches: our agent's moves are searched together on the
accelerator, and each game has its own engine process (see `engines.py`): as many at a
time as fit in a quarter of the available memory, up to 128, since the machine may be
shared. Each game's result is checked against the engine's own board.
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
from engines import ENGINES, GTPEngine
from game import (
    PASS,
    env,
    load_checkpoint,
    make_test_evaluator,
    metadata,
    state_to_nn_input,
    step_fn,
)
from xot import XOT_PATH, load_xot

from core.evaluators.evaluation_fns import make_nn_eval_fn


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


def play_games(engines, openings, we_black, search, advance, evaluator, pool, key):
    """Plays one game per engine process, from `openings` (one per game), with our agent
    as black in the games marked `we_black`.

    Returns:
        our result in each game: +1 win, 0 draw, -1 loss
    """
    num_games = len(engines)
    init_key, key = jax.random.split(key)
    states = jax.vmap(env.init)(jax.random.split(init_key, num_games))
    trees = jax.vmap(lambda s: evaluator.init(template_embedding=s))(states)
    # pgx picks which player id moves first (black) at random
    black_id = np.asarray(states.current_player)
    our_id = np.where(we_black, black_id, 1 - black_id)

    # play the openings
    for ply in range(openings.shape[1]):
        moves = openings[:, ply]
        assert np.asarray(states.legal_action_mask)[np.arange(num_games), moves].all()
        list(pool.map(GTPEngine.play, engines, [color(ply)] * num_games, moves))
        trees, states = advance(trees, states, jnp.asarray(moves))

    ply = openings.shape[1]
    while not np.asarray(states.terminated).all():
        done = np.asarray(states.terminated)
        mask = np.asarray(states.legal_action_mask)
        ours = we_black == (ply % 2 == 0)
        # our moves: every game is searched in one batch, the ones where it's the
        # engine's turn are thrown away
        search_key, key = jax.random.split(key)
        out = search(jax.random.split(search_key, num_games), trees, states)
        moves = np.where(ours, np.asarray(out.action), PASS).astype(np.int32)
        # the engine's moves, all games at once (a side with no move passes without asking)
        asking = [
            g
            for g in range(num_games)
            if not ours[g] and not done[g] and not mask[g, PASS]
        ]
        replies = pool.map(
            GTPEngine.genmove, [engines[g] for g in asking], [color(ply)] * len(asking)
        )
        for g, move in zip(asking, replies):
            moves[g] = move
        live = [g for g in range(num_games) if not done[g]]
        assert mask[live, moves[live]].all(), "an illegal move"
        # tell the engines our moves
        moved = [g for g in live if ours[g]]
        list(
            pool.map(
                GTPEngine.play,
                [engines[g] for g in moved],
                [color(ply)] * len(moved),
                moves[moved],
            )
        )
        # keep the search trees of the games we just moved in, then step everything
        trees = select(ours, out.eval_state, trees)
        new_trees, new_states = advance(trees, states, jnp.asarray(moves))
        # finished games stay as they are
        trees = select(done, trees, new_trees)
        states = select(done, states, new_states)
        ply += 1

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
        description="Plays a train.py checkpoint against Edax or Egaroucid at fixed search depths."
    )
    parser.add_argument("checkpoint", help="a checkpoint saved by train.py")
    parser.add_argument("--engine", choices=sorted(ENGINES), default="edax")
    parser.add_argument(
        "--blocks", type=int, default=6, help="the checkpoint's network size"
    )
    parser.add_argument(
        "--channels", type=int, default=128, help="the checkpoint's network size"
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    engine = ENGINES[args.engine]
    engine_dir = args.engine_dir or engine.default_dir()
    if not engine.installed(engine_dir):
        parser.error(
            f"no {engine.name} in {engine_dir}: run examples/othello/{engine.setup_script}"
        )
    network, params = load_checkpoint(args.checkpoint, args.blocks, args.channels)
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
    rng = np.random.default_rng(args.seed)
    print(
        f"{args.checkpoint}: {args.sims} MCTS iterations a move, against {engine.name} at "
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
            we_black = np.arange(num_games) % 2 == 0
            results = np.zeros(num_games, dtype=np.int32)
            for begin in range(0, num_games, parallel):
                games = slice(begin, min(begin + parallel, num_games))
                count = games.stop - games.start
                engines = list(pool.map(engine, [engine_dir] * count, [level] * count))
                try:
                    results[games] = play_games(
                        engines,
                        openings[games],
                        we_black[games],
                        search,
                        advance,
                        evaluator,
                        pool,
                        jax.random.PRNGKey(args.seed * 100_000 + level * 1000 + begin),
                    )
                finally:
                    for e in engines:
                        e.close()
            mean, se = score(results)
            seconds = time.perf_counter() - start
            if args.results is not None:
                with open(args.results, "a") as f:
                    record = {
                        "checkpoint": args.checkpoint,
                        "engine": engine.name,
                        "level": level,
                        "sims": args.sims,
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
