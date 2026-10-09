"""AlphaZero on Othello, tested on a ladder of opponents: a random player, a greedy
tile counter, then pgx's pretrained Othello model searching more and more.

Self-play games are collected in parallel across a batch of environments (and across
every available GPU), with Monte Carlo Tree Search run on each of them; the network
then trains on minibatches sampled from replay memory. Self-play games start from XOT
openings (see xot.py), which keeps them varied.

    uv run examples/othello/train.py
    uv run examples/othello/train.py --epochs 20 --monitor
    uv run examples/othello/train.py --sims-schedule 0:32,50:64,150:128
    uv run examples/othello/train.py --buffer-schedule 0:1000,20:3000,35:6000

Start the monitor first, in another shell, with `uv run turbozero-monitor`. It shows
the metrics, which rung of the ladder the agent has reached, and a game against the
last opponent it played, which it renders itself.

The first epoch is slow: nearly all of the training loop is JIT-compiled the first
time it runs. On one RTX 5080 the rest take about 63s each (36s with
--inference-dtype bfloat16, see README.md). With these settings, one
200-epoch run passed every rung by epoch 65; at the end, in 512 games against pgx's
model (our agent searching 64 iterations a move), it scored 0.82 against the model
searching 64, 0.74 against it searching 256, and 0.57 against it searching 1024
(draws count half).

Checkpoints go to --ckpt-dir; `eval_pgx.py` and `vs_engine.py` evaluate them further (see
README.md). The hyperparameters here are only an example; tune them for your task and
hardware.
"""

import argparse
import tempfile
from functools import partial

import pgx
from game import (
    SYMMETRY_TRANSFORM_FNS,
    env,
    greedy_eval,
    make_network,
    make_optimizer,
    make_test_evaluator,
    state_to_nn_input,
    step_fn,
)
from xot import load_xot, make_xot_init_fn

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import (
    make_nn_eval_fn,
    make_nn_eval_fn_no_params_callable,
)
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.random_evaluator import RandomEvaluator
from core.memory.replay_memory import EpisodeReplayBuffer
from core.monitor import DEFAULT_URL, Monitor
from core.monitor.renderers import pgx_two_player_episode
from core.testing.ladder import LadderTester, Rung
from core.training.loss_fns import az_default_loss_fn
from core.training.schedule import EvaluatorSchedule, Schedule, parse_schedule
from core.training.train import Trainer
from core.training.tree_positions import TreePositions


def positive_schedule(text: str) -> list[tuple[int, int]]:
    """Parses --sims-schedule or --buffer-schedule, whose values must be positive."""
    try:
        schedule = parse_schedule(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    if any(value < 1 for _, value in schedule):
        raise argparse.ArgumentTypeError(f"values must be positive, got {text!r}")
    return schedule


def main():
    parser = argparse.ArgumentParser(description="AlphaZero on Othello.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=6, help="residual blocks")
    parser.add_argument("--channels", type=int, default=128, help="channels per block")
    sims = parser.add_mutually_exclusive_group()
    sims.add_argument(
        "--sims", type=int, default=64, help="MCTS iterations per self-play move"
    )
    sims.add_argument(
        "--sims-schedule",
        type=positive_schedule,
        default=None,
        metavar="EPOCH:SIMS,...",
        help="MCTS iterations per self-play move by epoch instead, e.g. 0:32,50:64,150:128 "
        "for 32 from epoch 0, 64 from epoch 50 and 128 from epoch 150. Each change "
        "compiles self-play again, so keep them few",
    )
    parser.add_argument(
        "--train-batch",
        type=int,
        default=4096,
        help="samples per training step; with a big network, halve it (and double "
        "--train-steps) if training runs out of memory",
    )
    parser.add_argument(
        "--train-steps", type=int, default=128, help="training steps per epoch"
    )
    buffer = parser.add_mutually_exclusive_group()
    buffer.add_argument(
        "--buffer",
        type=int,
        default=3000,
        help="replay memory: samples kept per environment (with the 7 symmetric copies "
        "of each, 3000 is about 3 epochs of self-play)",
    )
    buffer.add_argument(
        "--buffer-schedule",
        type=positive_schedule,
        default=None,
        metavar="EPOCH:SAMPLES,...",
        help="replay window by epoch instead: training samples from each environment's "
        "newest SAMPLES, e.g. 0:1000,20:3000,35:6000 for 1000 from epoch 0, 3000 from "
        "epoch 20 and 6000 from epoch 35. Replay memory keeps the largest; changing the "
        "window doesn't compile anything again",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="the initial learning rate"
    )
    parser.add_argument(
        "--lr-final",
        type=float,
        default=1e-4,
        help="the learning rate at the end of the run (it decays along a cosine)",
    )
    parser.add_argument(
        "--inference-dtype",
        default="float32",
        help="dtype the network computes in during self-play and test games, e.g. "
        "bfloat16 (about twice as fast); it always trains in float32",
    )
    parser.add_argument(
        "--standard-starts",
        type=float,
        default=0.0,
        help="fraction of self-play games from the standard start; the rest start from "
        "XOT openings. Every test game starts from an XOT opening, so by default none "
        "do, and the network never learns the first 8 moves",
    )
    parser.add_argument(
        "--value-target-q",
        type=float,
        default=0.0,
        help="train the value of played positions on a mix of the game's outcome z and the "
        "search's root value q: (1 - this) * z + this * q (0, the default, is z alone)",
    )
    parser.add_argument(
        "--tree-positions",
        type=int,
        default=0,
        metavar="K",
        help="also train on up to K positions from each self-play search tree, as OLIVAW "
        "did: each on its children's visit distribution and its search value q (0, the "
        "default, for none; see core/training/tree_positions.py)",
    )
    parser.add_argument(
        "--tree-min-visits",
        type=int,
        default=16,
        help="visits a search tree node needs to be stored",
    )
    parser.add_argument(
        "--tree-select",
        choices=("sample", "most-visited"),
        default="sample",
        help="store nodes sampled in proportion to their visits, or the most-visited "
        "ones (OLIVAW's)",
    )
    parser.add_argument(
        "--tree-discarded-only",
        action="store_true",
        help="only store nodes outside the played move's subtree (which the next search "
        "reuses), so that each position is stored at most once",
    )
    parser.add_argument(
        "--tree-ratio",
        type=float,
        default=1.0,
        help="tree positions per played position in training batches (OLIVAW's 1:1 by "
        "default: half of each batch)",
    )
    parser.add_argument(
        "--tree-half-life",
        type=float,
        default=None,
        help="epochs over which --tree-ratio halves (by default it stays constant)",
    )
    parser.add_argument(
        "--tree-buffer",
        type=int,
        default=None,
        help="tree positions (with their symmetric copies) kept per environment (default: "
        "K times --buffer, or the largest window of --buffer-schedule). Training samples "
        "those stored in the replay window's span of self-play",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="epochs between test games on the ladder (0 for none, e.g. when "
        "watch.py plays checkpoints against an engine instead)",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="where to save checkpoints (default: a new temporary directory)",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=None,
        help="also keep every checkpoint whose epoch is a multiple of this (otherwise "
        "only the 2 newest are kept)",
    )
    parser.add_argument("--name", default=None, help="the run's name on the monitor")
    parser.add_argument(
        "--monitor",
        nargs="?",
        const=DEFAULT_URL,
        default=None,
        metavar="URL",
        help=f"log to a turbozero monitor (default {DEFAULT_URL}); start it with `uv run turbozero-monitor`",
    )
    args = parser.parse_args()

    # self-play starts games from XOT openings (see xot.py) for varied, balanced
    # games
    selfplay_init_fn = make_xot_init_fn(load_xot(), args.standard_starts)
    # test games all start from XOT openings
    test_init_fn = make_xot_init_fn(load_xot(), standard_start_fraction=0)

    # the residual network from the AlphaZero paper (see game.py), with its BatchNorm state
    resnet, resnet_state = make_network(
        args.blocks, args.channels, seed=args.seed, inference_dtype=args.inference_dtype
    )

    # AlphaZero takes an arbitrary search backend, here classic MCTS. Temperature 1.0
    # samples moves in proportion to visit counts, for exploration during self-play.
    # With --sims-schedule, a search for each stage: each takes over at its epoch,
    # starting from empty trees sized for it (see core.training.schedule)
    selfplay_eval_fn = make_nn_eval_fn(resnet, state_to_nn_input)
    evaluator = EvaluatorSchedule(
        [
            (
                epoch,
                AlphaZero(MCTS)(
                    eval_fn=selfplay_eval_fn,
                    num_iterations=sims,
                    max_nodes=2 * sims,
                    branching_factor=env.num_actions,
                    action_selector=PUCTSelector(),
                    temperature=1.0,
                ),
            )
            for epoch, sims in args.sims_schedule or [(0, args.sims)]
        ]
    )
    evaluator_test = make_test_evaluator(make_nn_eval_fn(resnet, state_to_nn_input))

    # opponents: the greedy tile counter and pgx's pretrained model (others are listed
    # at https://sotets.uk/pgx/api/#pgx.BaselineModelId)
    pretrained = make_nn_eval_fn_no_params_callable(
        pgx.make_baseline_model("othello_v0"), state_to_nn_input
    )
    greedy = make_nn_eval_fn_no_params_callable(greedy_eval, state_to_nn_input)

    # with a monitor, each test sends a game for the monitor server to render
    # (drawing it needs the cairo system library there, not here)
    episode_fn = (
        pgx_two_player_episode(p1_label="Black", p2_label="White")
        if args.monitor
        else None
    )

    # the ladder, easiest first: each test plays the lowest rung the agent hasn't
    # beaten, and moves up past every rung it scores at least 55% against. Our agent
    # searches 64 iterations a move throughout; pgx's model searches more and more
    rungs = [
        Rung("random", RandomEvaluator()),
        Rung("greedy", make_test_evaluator(greedy)),
    ] + [
        Rung(f"pgx{n}", make_test_evaluator(pretrained, n)) for n in (1, 4, 16, 64, 256)
    ]
    testers = (
        [LadderTester(num_episodes=128, rungs=rungs, episode_fn=episode_fn)]
        if args.eval_every > 0
        else []
    )

    # with --buffer-schedule, training samples from fewer of the newest samples early on
    replay_window = args.buffer_schedule or [(0, args.buffer)]
    buffer_capacity = max(w for _, w in replay_window)

    # each epoch collects `collection_steps_per_epoch` self-play steps in each of
    # `batch_size` environments, then takes `train_steps_per_epoch` training steps
    trainer = Trainer(
        batch_size=1024,
        train_batch_size=args.train_batch,
        warmup_steps=0,
        collection_steps_per_epoch=128,
        train_steps_per_epoch=args.train_steps,
        nn=resnet,
        nn_state=resnet_state,
        loss_fn=partial(
            az_default_loss_fn,
            l2_reg_lambda=1e-4,
            value_target_q=args.value_target_q,
        ),
        # decays from --lr to --lr-final over the run
        optimizer=make_optimizer(
            args.epochs * args.train_steps, args.lr, args.lr_final
        ),
        evaluator=evaluator,
        # stores `capacity` samples for each of the `batch_size` environments: the
        # largest window, which training samples from the newest of
        memory_buffer=EpisodeReplayBuffer(capacity=buffer_capacity),
        replay_window=Schedule(replay_window),
        max_episode_steps=80,
        env_step_fn=step_fn,
        env_init_fn=selfplay_init_fn,
        test_env_init_fn=test_init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=testers,
        evaluator_test=evaluator_test,
        # add each sample's 7 symmetric copies
        data_transform_fns=SYMMETRY_TRANSFORM_FNS,
        tree_positions=TreePositions(
            per_move=args.tree_positions,
            min_visits=args.tree_min_visits,
            capacity=args.tree_buffer or args.tree_positions * buffer_capacity,
            ratio=args.tree_ratio,
            half_life=args.tree_half_life,
            most_visited=args.tree_select == "most-visited",
            discarded_only=args.tree_discarded_only,
        )
        if args.tree_positions > 0
        else None,
        monitor=Monitor(args.monitor, project="othello", name=args.name)
        if args.monitor
        else None,
        ckpt_dir=args.ckpt_dir or tempfile.mkdtemp(prefix="turbozero-othello-"),
        keep_every=args.keep_every,
        extra_config={"value_target_q": args.value_target_q},
    )
    trainer.train_loop(
        seed=args.seed, num_epochs=args.epochs, eval_every=max(args.eval_every, 1)
    )


if __name__ == "__main__":
    main()
