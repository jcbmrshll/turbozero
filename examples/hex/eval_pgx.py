"""Plays a checkpoint from `train.py` against pgx's pretrained Hex model at a series of search
budgets, with more games than the test games in training, for tighter estimates. pgx's model
plays 11 x 11 only.

    uv run examples/hex/eval_pgx.py CHECKPOINT --pgx-sims 64 256 1024 --games 512

Games start from openings (see `openings.py`), as in `vs_engine.py`. Both sides search with MCTS
at temperature 0, with AlphaZero's root noise for variety; our agent moves first in exactly half
the games.
"""

import argparse
import time

import jax
import numpy as np
import pgx
from engines import CACHE_DIR
from game import HexGame, state_to_nn_input
from openings import load_openings, make_opening_init_fn

from core.evaluators.evaluation_fns import (
    make_nn_eval_fn,
    make_nn_eval_fn_no_params_callable,
)
from core.testing.tester import TestState
from core.testing.two_player_baseline import TwoPlayerBaseline


def main():
    parser = argparse.ArgumentParser(
        description="Plays a train.py checkpoint against pgx's pretrained Hex model."
    )
    parser.add_argument("checkpoint", help="a checkpoint saved by train.py")
    parser.add_argument(
        "--net",
        default=None,
        help="the checkpoint's network, as train.py's --net names it (default: 11 x 11's)",
    )
    parser.add_argument(
        "--sims", type=int, default=64, help="our MCTS iterations a move"
    )
    parser.add_argument(
        "--pgx-sims",
        type=int,
        nargs="+",
        default=[64, 256],
        help="pgx's model's MCTS iterations a move, one match each",
    )
    parser.add_argument("--games", type=int, default=512, help="games per match (even)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    game = HexGame(11)
    init_fn = make_opening_init_fn(game, load_openings(game), standard_start_fraction=0)
    network, params = game.load_checkpoint(args.checkpoint, args.net)
    agent = game.make_test_evaluator(
        make_nn_eval_fn(network, state_to_nn_input), args.sims
    )
    pretrained = make_nn_eval_fn_no_params_callable(
        pgx.make_baseline_model("hex_v0", download_dir=str(CACHE_DIR / "baselines")),
        state_to_nn_input,
    )
    print(
        f"{args.checkpoint}: {args.sims} MCTS iterations a move, {args.games} games "
        "per match",
        flush=True,
    )
    for sims in args.pgx_sims:
        tester = TwoPlayerBaseline(
            num_episodes=args.games,
            baseline_evaluator=game.make_test_evaluator(pretrained, sims),
            balance_first_player=True,
            name="pgx",
        )
        start = time.perf_counter()
        keys = tester.split_keys(jax.random.PRNGKey(args.seed))
        _, metrics, _, _ = tester.test(
            game.max_moves,
            game.step_fn,
            init_fn,
            agent,
            keys,
            TestState(),
            params,
        )
        win = float(metrics["pgx_win_rate"].mean())
        # standard error of the mean score
        se = np.sqrt(win * (1 - win) / args.games)
        print(
            f"  pgx at {sims:4d} iterations: score {win:.3f} ± {se:.3f}  "
            f"{time.perf_counter() - start:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
