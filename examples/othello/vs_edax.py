"""Plays a checkpoint from `train.py` against Edax, a strong open-source Othello engine,
at a series of fixed search depths.

    examples/othello/setup_edax.sh       # once: builds Edax
    uv run examples/othello/vs_edax.py CHECKPOINT --levels 2 4 6 8 --sims 400

Games start from XOT openings (see `xot.py`), each played twice with the colors
swapped, so that two deterministic players give varied, balanced games. Edax searches
to the given depth with its opening book off ("level" in Edax's terms); our agent runs
`--sims` MCTS iterations a move at temperature 0, keeping its search tree between moves,
without the root noise used in self-play. This is how OLIVAW was compared with Edax
(https://arxiv.org/abs/2103.17228).

All games against one level are played at once: our agent's moves are searched in a
batch on the accelerator, and each game has its own Edax process, talking GTP. Each
game's result is checked against Edax's own board.
"""

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from game import (
    PASS,
    action,
    env,
    load_checkpoint,
    make_test_evaluator,
    metadata,
    square,
    state_to_nn_input,
    step_fn,
)
from xot import XOT_PATH, load_xot

from core.evaluators.evaluation_fns import make_nn_eval_fn

DEFAULT_EDAX_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "turbozero"
    / "othello"
    / "edax"
)


class Edax:
    """One Edax process, talking GTP (the Go Text Protocol, which Edax speaks for Othello)."""

    def __init__(self, edax_dir: Path, level: int):
        self.process = subprocess.Popen(
            # one thread each: games run in parallel instead
            [
                str(edax_dir / "bin" / "edax"),
                "-gtp",
                "-q",
                "-l",
                str(level),
                "-n",
                "1",
                "-book-usage",
                "off",
            ],
            cwd=edax_dir,  # it finds its weights in data/eval.dat
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.command("boardsize 8")
        self.command("clear_board")

    def command(self, command: str) -> str:
        """Sends a command and returns the response (without its leading "= ")."""
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        # a response is one or more lines, then a blank line
        lines = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(f"Edax exited after {command!r}")
            if line.strip():
                lines.append(line.strip())
            elif lines:
                break
        if lines[0].startswith("?"):
            raise RuntimeError(f"Edax rejected {command!r}: {lines[0]}")
        return "\n".join([lines[0].lstrip("=").strip(), *lines[1:]])

    def play(self, color: str, move: int) -> None:
        # Edax passes by itself: a command for one color passes for the other if it has
        # no move, and an explicit pass is rejected
        if move != PASS:
            self.command(f"play {color} {square(move)}")

    def genmove(self, color: str) -> int:
        return action(self.command(f"genmove {color}"))

    def discs(self) -> tuple[int, int]:
        """(black, white) discs on Edax's board."""
        counts = {}
        for line in self.command("showboard").splitlines():
            if "discs =" in line:
                counts[line.split(":")[0].split()[-1]] = int(
                    line.split("discs =")[1].split()[0]
                )
        return counts["*"], counts["O"]

    def close(self) -> None:
        self.process.kill()
        self.process.wait()


def select(which, on_true, on_false):
    """Game by game, `on_true` where `which` is set, else `on_false`."""
    which = jnp.asarray(which)
    return jax.tree.map(
        lambda t, f: jnp.where(which.reshape((-1,) + (1,) * (t.ndim - 1)), t, f),
        on_true,
        on_false,
    )


def play_level(level, openings, evaluator, params, edax_dir, pool, key):
    """Plays one game per opening and color against Edax at `level`.

    Returns:
        our result in each game (+1 win, 0 draw, -1 loss), and whether we played black
    """
    num_games = 2 * len(openings)
    # game 2i plays opening i with us as black, game 2i+1 with us as white
    we_black = np.arange(num_games) % 2 == 0
    openings = np.repeat(openings, 2, axis=0)
    edaxes = list(pool.map(lambda _: Edax(edax_dir, level), range(num_games)))
    try:
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
            jax.vmap(
                lambda tree, state, a: (evaluator.step(tree, a), env.step(state, a))
            )
        )
        init_key, key = jax.random.split(key)
        states = jax.vmap(env.init)(jax.random.split(init_key, num_games))
        trees = jax.vmap(lambda s: evaluator.init(template_embedding=s))(states)
        # pgx picks which player id moves first (black) at random
        black_id = np.asarray(states.current_player)
        our_id = np.where(we_black, black_id, 1 - black_id)

        def color(ply):
            return "b" if ply % 2 == 0 else "w"

        # play the openings
        for ply in range(openings.shape[1]):
            moves = openings[:, ply]
            assert np.asarray(states.legal_action_mask)[
                np.arange(num_games), moves
            ].all()
            list(pool.map(Edax.play, edaxes, [color(ply)] * num_games, moves))
            trees, states = advance(trees, states, jnp.asarray(moves))

        ply = openings.shape[1]
        while not np.asarray(states.terminated).all():
            done = np.asarray(states.terminated)
            mask = np.asarray(states.legal_action_mask)
            ours = we_black == (ply % 2 == 0)
            # our moves: every game is searched in one batch, the ones where it's
            # Edax's turn are thrown away
            search_key, key = jax.random.split(key)
            out = search(jax.random.split(search_key, num_games), trees, states)
            moves = np.where(ours, np.asarray(out.action), PASS).astype(np.int32)
            # Edax's moves, all games at once (a side with no move passes without asking)
            asking = [
                g
                for g in range(num_games)
                if not ours[g] and not done[g] and not mask[g, PASS]
            ]
            replies = pool.map(
                Edax.genmove, [edaxes[g] for g in asking], [color(ply)] * len(asking)
            )
            for g, move in zip(asking, replies):
                moves[g] = move
            live = [g for g in range(num_games) if not done[g]]
            assert mask[live, moves[live]].all(), "an illegal move"
            # tell Edax our moves
            moved = [g for g in live if ours[g]]
            list(
                pool.map(
                    Edax.play,
                    [edaxes[g] for g in moved],
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
        # check every result against Edax's own board
        for g, (black, white) in enumerate(pool.map(lambda e: e.discs(), edaxes)):
            edax_result = np.sign(black - white) * (1 if we_black[g] else -1)
            assert edax_result == np.sign(results[g]), (
                f"game {g}: Edax disagrees on the result"
            )
        return results, we_black
    finally:
        for e in edaxes:
            e.close()


def score(results) -> tuple[float, float]:
    """Mean score (a draw counts half) and its standard error."""
    points = (1 + np.sign(results)) / 2
    return points.mean(), points.std() / np.sqrt(len(points))


def main():
    parser = argparse.ArgumentParser(
        description="Plays a train.py checkpoint against Edax at fixed search depths."
    )
    parser.add_argument("checkpoint", help="a checkpoint saved by train.py")
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
        "--stop-below",
        type=float,
        default=0.1,
        help="stop once our score against a level drops below this",
    )
    parser.add_argument("--edax-dir", type=Path, default=DEFAULT_EDAX_DIR)
    parser.add_argument("--xot", default=XOT_PATH, help="the XOT opening list")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (args.edax_dir / "bin" / "edax").exists():
        parser.error(f"no Edax in {args.edax_dir}: run examples/othello/setup_edax.sh")
    network, params = load_checkpoint(args.checkpoint, args.blocks, args.channels)
    evaluator = make_test_evaluator(
        make_nn_eval_fn(network, state_to_nn_input), args.sims, noise=False
    )
    xot = load_xot(args.xot)
    rng = np.random.default_rng(args.seed)
    print(
        f"{args.checkpoint}: {args.sims} MCTS iterations a move, against Edax at depths "
        f"{args.levels}; {args.openings} XOT openings x both colors per depth",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for level in args.levels:
            start = time.perf_counter()
            openings = xot[rng.choice(len(xot), size=args.openings, replace=False)]
            results, we_black = play_level(
                level,
                openings,
                evaluator,
                params,
                args.edax_dir,
                pool,
                jax.random.PRNGKey(args.seed * 1000 + level),
            )
            mean, se = score(results)
            print(
                f"  Edax depth {level:2d}: score {mean:.3f} ± {se:.3f}  "
                f"(win {(results > 0).mean():.3f}, draw {(results == 0).mean():.3f}, "
                f"loss {(results < 0).mean():.3f}; as black {score(results[we_black])[0]:.2f}, "
                f"as white {score(results[~we_black])[0]:.2f})  {time.perf_counter() - start:.0f}s",
                flush=True,
            )
            if mean < args.stop_below:
                break


if __name__ == "__main__":
    main()
