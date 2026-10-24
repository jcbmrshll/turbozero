"""AlphaZero on Othello, tested on a ladder of opponents: a random player, a greedy
tile counter, then pgx's pretrained Othello model searching more and more.

Self-play games are collected in parallel across a batch of environments (and across
every available GPU), with Monte Carlo Tree Search run on each of them; the network
then trains on minibatches sampled from replay memory. Self-play games start from XOT
openings (see xot.py), which keeps them varied.

    uv run examples/othello/train.py
    uv run examples/othello/train.py --epochs 20 --monitor
    uv run examples/othello/train.py --sims-schedule 0:32,50:64,150:128

Start the monitor first, in another shell, with `uv run turbozero-monitor`. It shows
the metrics, which rung of the ladder the agent has reached, and a game against the
last opponent it played, which it draws in the browser.

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

With --save-state-at or --save-state-every, the whole training state (network, optimizer,
replay buffers, games in progress, rng) is saved there too, to continue the run exactly
with --resume, or to fork short runs from with --init-from (see README.md):

    uv run examples/othello/train.py --epochs 150 --lr-final 1e-3 --save-state-at 150 --ckpt-dir runs/trunk
    uv run examples/othello/train.py --epochs 30 --init-from runs/trunk --ckpt-dir runs/fork-control
    uv run examples/othello/train.py --epochs 200 --save-state-every 10 --ckpt-dir runs/long
    uv run examples/othello/train.py --epochs 200 --save-state-every 10 --ckpt-dir runs/long --resume
"""

import argparse
import tempfile
from functools import partial

import pgx
from game import (
    SYMMETRY_TRANSFORM_FNS,
    WEIGHT_DECAY,
    env,
    episode_metrics,
    greedy_eval,
    make_network,
    make_optimizer,
    make_test_evaluator,
    state_to_nn_input,
    step_fn,
)
from xot import load_xot, make_xot_init_fn, xot_episode_fn

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import (
    make_nn_eval_fn,
    make_nn_eval_fn_no_params_callable,
)
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.random_evaluator import RandomEvaluator
from core.memory.replay_memory import EpisodeReplayBuffer
from core.testing.ladder import LadderTester, Rung
from core.training.exploration import Exploration
from core.training.loss_fns import az_default_loss_fn
from core.training.schedule import EvaluatorSchedule, parse_schedule
from core.training.train import Trainer
from core.training.tree_positions import TreePositions
from monitor import DEFAULT_URL, Monitor
from monitor.episodes import pgx_two_player_episode


def epoch_list(text: str) -> list[int]:
    """Parses --save-state-at: epochs separated by commas."""
    try:
        epochs = [int(epoch) for epoch in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected epochs separated by commas, e.g. 50,150, got {text!r}"
        ) from None
    if any(epoch < 1 for epoch in epochs):
        raise argparse.ArgumentTypeError(f"epochs must be positive, got {text!r}")
    return epochs


def positive_schedule(text: str) -> list[tuple[int, int]]:
    """Parses --sims-schedule, whose values must be positive."""
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
        "--tree-factor",
        type=float,
        default=2.0,
        help="self-play search tree capacity, in nodes per MCTS iteration (the tree is "
        "kept between moves, so it needs room for the reused subtree too)",
    )
    parser.add_argument(
        "--sampling-moves",
        type=int,
        default=None,
        metavar="N",
        help="self-play samples moves in proportion to visit counts for the first N "
        "moves of a game, then plays the most-visited move, as AlphaZero (30) and "
        "OLIVAW (20) did. N counts from the standard start, so it includes an XOT "
        "opening's 8 moves. By default every move is sampled",
    )
    parser.add_argument(
        "--train-batch",
        type=int,
        default=4096,
        help="samples per training step; if training runs out of memory with a big "
        "network, see --remat and --grad-accum, which keep the batch",
    )
    parser.add_argument(
        "--train-steps", type=int, default=128, help="training steps per epoch"
    )
    parser.add_argument(
        "--remat",
        action="store_true",
        help="train with much less memory, for roughly a third more training compute: "
        "keep only each residual block's input for the backward pass, and compute its "
        "activations again (gradient checkpointing). Training is the same, up to float "
        "rounding",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        metavar="M",
        help="split each training batch into M micro-batches, one after another, and make "
        "one update with their mean gradient: activations take 1/M the memory. BatchNorm "
        "normalizes each micro-batch by its own statistics, so training differs a little "
        "from whole batches (see README.md). Must divide --train-batch",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=3000,
        help="replay memory: samples kept per environment (with the 7 symmetric copies "
        "of each, 3000 is about 3 epochs of self-play)",
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
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help="AdamW's weight decay, of the network's weights (not its biases or BatchNorm)",
    )
    parser.add_argument(
        "--inference-dtype",
        default="float32",
        help="dtype the network computes in during self-play and test games, e.g. "
        "bfloat16 (about twice as fast)",
    )
    parser.add_argument(
        "--train-dtype",
        default="float32",
        help="dtype the network's convolutions compute in during training, e.g. bfloat16 "
        "for mixed precision: its parameters, BatchNorm and heads stay in float32",
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
        "K times --buffer). Training samples those stored in the replay memory's span "
        "of self-play",
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
    parser.add_argument(
        "--save-state-at",
        type=epoch_list,
        default=[],
        metavar="EPOCHS",
        help="also save the whole training state (with the replay buffers: a few GB) to "
        "--ckpt-dir as state-<EPOCH>.npz once this many epochs are done, e.g. 150 or "
        "50,150, to fork runs from with --init-from",
    )
    parser.add_argument(
        "--save-state-every",
        type=int,
        default=None,
        metavar="N",
        help="also save the whole training state every N epochs, keeping the newest, to "
        "continue the run with --resume if it stops",
    )
    init = parser.add_mutually_exclusive_group()
    init.add_argument(
        "--init-from",
        default=None,
        metavar="PATH",
        help="fork a new run from a saved state (state-<EPOCH>.npz, or a --ckpt-dir to take "
        "the newest from): it starts at epoch 0, with its own schedules, games and seed, "
        "from the saved network, optimizer state (the learning rate decaying anew from "
        "--lr) and replay buffers. Needs the same --blocks and --channels; --buffer may "
        "differ (each environment's newest entries are kept)",
    )
    parser.add_argument(
        "--init-without-buffers",
        action="store_true",
        help="with --init-from, start with empty replay buffers instead",
    )
    init.add_argument(
        "--resume",
        action="store_true",
        help="continue the run in --ckpt-dir from its newest saved state, exactly as if it "
        "hadn't stopped; pass the run's own arguments",
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
    if args.resume and args.ckpt_dir is None:
        parser.error("--resume needs the run's --ckpt-dir")
    if args.init_without_buffers and args.init_from is None:
        parser.error("--init-without-buffers needs --init-from")

    # self-play starts games from XOT openings (see xot.py) for varied, balanced
    # games
    selfplay_init_fn = make_xot_init_fn(load_xot(), args.standard_starts)
    # test games all start from XOT openings
    test_init_fn = make_xot_init_fn(load_xot(), standard_start_fraction=0)

    # the residual network from the AlphaZero paper (see game.py), with its BatchNorm state
    resnet, resnet_state = make_network(
        args.blocks,
        args.channels,
        seed=args.seed,
        inference_dtype=args.inference_dtype,
        train_dtype=args.train_dtype,
        remat=args.remat,
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
                    max_nodes=int(args.tree_factor * sims),
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

    # with a monitor, each test sends a game for the dashboard to play, its XOT opening's
    # moves before it
    episode_fn = (
        xot_episode_fn(pgx_two_player_episode(), load_xot()) if args.monitor else None
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

    # each epoch collects `collection_steps_per_epoch` self-play steps in each of
    # `batch_size` environments, then takes `train_steps_per_epoch` training steps
    trainer = Trainer(
        batch_size=1024,
        train_batch_size=args.train_batch,
        warmup_steps=0,
        collection_steps_per_epoch=128,
        train_steps_per_epoch=args.train_steps,
        grad_accum_steps=args.grad_accum,
        nn=resnet,
        nn_state=resnet_state,
        loss_fn=partial(az_default_loss_fn, value_target_q=args.value_target_q),
        # decays from --lr to --lr-final over the run
        optimizer=make_optimizer(
            args.epochs * args.train_steps, args.lr, args.lr_final, args.weight_decay
        ),
        evaluator=evaluator,
        # stores `capacity` samples for each of the `batch_size` environments
        memory_buffer=EpisodeReplayBuffer(capacity=args.buffer),
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
            capacity=args.tree_buffer or args.tree_positions * args.buffer,
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
        # how often self-play games are drawn
        episode_metrics_fn=episode_metrics,
        exploration=Exploration(num_sampling_moves=args.sampling_moves)
        if args.sampling_moves is not None
        else None,
        save_state_at=args.save_state_at,
        save_state_every=args.save_state_every,
        # every argument but how the run starts (a fork records its parent)
        extra_config={
            "args": {
                k: v
                for k, v in vars(args).items()
                if k not in ("resume", "init_from", "init_without_buffers")
            }
        },
    )
    trainer.train_loop(
        seed=args.seed,
        num_epochs=args.epochs,
        eval_every=max(args.eval_every, 1),
        initial_state=trainer.resume() if args.resume else None,
        fork_from=args.init_from,
        fork_replay_buffers=not args.init_without_buffers,
    )


if __name__ == "__main__":
    main()
