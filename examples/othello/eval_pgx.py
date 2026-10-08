"""Plays a checkpoint from `train.py` against pgx's pretrained Othello model at a series
of search budgets, with more games than the ladder in training plays, for tighter
estimates.

    uv run examples/othello/eval_pgx.py CHECKPOINT --pgx-sims 64 256 1024 --games 512

Both sides search with MCTS at temperature 0, with AlphaZero's root noise for variety;
our agent moves first in exactly half the games.
"""

import argparse
import time

import jax
import numpy as np
import pgx
from game import (
    init_fn,
    load_checkpoint,
    make_test_evaluator,
    state_to_nn_input,
    step_fn,
)

from core.evaluators.evaluation_fns import (
    make_nn_eval_fn,
    make_nn_eval_fn_no_params_callable,
)
from core.testing.tester import TestState
from core.testing.two_player_baseline import TwoPlayerBaseline


def main():
    parser = argparse.ArgumentParser(
        description="Plays a train.py checkpoint against pgx's pretrained Othello model."
    )
    parser.add_argument("checkpoint", help="a checkpoint saved by train.py")
    parser.add_argument(
        "--blocks", type=int, default=6, help="the checkpoint's network size"
    )
    parser.add_argument(
        "--channels", type=int, default=128, help="the checkpoint's network size"
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

    network, params = load_checkpoint(args.checkpoint, args.blocks, args.channels)
    # the tester runs on every device with pmap: one device here, so add its axis
    params = jax.tree.map(lambda x: x[None], params)
    agent = make_test_evaluator(make_nn_eval_fn(network, state_to_nn_input), args.sims)
    pretrained = make_nn_eval_fn_no_params_callable(
        pgx.make_baseline_model("othello_v0"), state_to_nn_input
    )
    print(
        f"{args.checkpoint}: {args.sims} MCTS iterations a move, {args.games} games "
        "per match",
        flush=True,
    )
    for sims in args.pgx_sims:
        tester = TwoPlayerBaseline(
            num_episodes=args.games,
            baseline_evaluator=make_test_evaluator(pretrained, sims),
            balance_first_player=True,
            name="pgx",
        )
        start = time.perf_counter()
        keys = tester.split_keys(jax.random.PRNGKey(args.seed), num_devices=1)
        _, metrics, _, _ = tester.test(
            80, step_fn, init_fn, agent, keys, TestState(), params
        )
        win, loss = (
            float(metrics["pgx_win_rate"].mean()),
            float(metrics["pgx_loss_rate"].mean()),
        )
        draw = 1 - win - loss
        score = win + draw / 2
        # standard error of the mean score, a game scoring 1, 1/2 or 0
        se = np.sqrt((win + draw / 4 - score**2) / args.games)
        print(
            f"  pgx at {sims:4d} iterations: score {score:.3f} ± {se:.3f}  (win {win:.3f}, "
            f"draw {draw:.3f}, loss {loss:.3f})  {time.perf_counter() - start:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
