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

import numpy as np
import pgx
from engines import CACHE_DIR
from game import HexGame, state_to_nn_input
from openings import load_openings, make_opening_init_fn

from core.evaluators.evaluation_fns import (
    make_nn_eval_fn,
    make_nn_eval_fn_no_params_callable,
)
from core.training.validation import TwoPlayerValidator


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
        match = TwoPlayerValidator(
            agent,
            game.make_test_evaluator(pretrained, sims),
            game.step_fn,
            init_fn,
            game.max_moves,
            args.games,
            name="pgx",
            seed=args.seed,
        )
        start = time.perf_counter()
        metrics = match(params)
        win = float(metrics["pgx_win_rate"])
        # standard error of the mean score
        se = np.sqrt(win * (1 - win) / args.games)
        print(
            f"  pgx at {sims:4d} iterations: score {win:.3f} ± {se:.3f}  "
            f"{time.perf_counter() - start:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
