"""AlphaZero on Othello, tracked by how close a weight average of the network comes to a strong
engine's moves and values on fixed positions.

Self-play games are collected in parallel across a batch of environments, with Monte Carlo
Tree Search run on each of them, playing with a weight average of the network (see
--average-epochs); the network then trains on batches sampled from replay memory. Self-play
games start from XOT openings (see xot.py), which keeps them varied.

    uv run examples/othello/train.py
    uv run examples/othello/train.py --epochs 20 --monitor

Every epoch it scores the average on the engine-labelled set, if there is one (see
engine_set.py and --engine-set), logged as `engine_*`. Games measure strength:
`evaluate.py` plays checkpoints against engines and each other, and the Elo league rates them.

With --monitor the run logs to slick's seaglass server ($SEAGLASS_URL, set in a slick task;
--monitor URL for another, e.g. `uv run seaglass serve --dir runs` at http://localhost:8008).
The run's page shows the metrics.

The first epoch is slow: nearly all of the training loop is JIT-compiled the first time it
runs.

The average's weights go to --ckpt-dir/avg4/<epoch>.eqx every --save-weights-every epochs;
`evaluate.py` evaluates them (see README.md). With --save-state-at or --save-state-every, the
whole training state (network, optimizer, weight average, replay buffer, games in progress,
rng) is saved there too, to continue the run exactly with --resume, to continue it with other
settings with --branch-from (e.g. a lower --lr), or to fork new runs from with --init-from
(see README.md):

    uv run examples/othello/train.py --epochs 150 --save-state-at 150 --ckpt-dir runs/trunk
    uv run examples/othello/train.py --epochs 250 --lr 1e-4 --branch-from runs/trunk --ckpt-dir runs/trunk-1e-4
    uv run examples/othello/train.py --epochs 30 --init-from runs/trunk --ckpt-dir runs/fork
    uv run examples/othello/train.py --epochs 200 --save-state-every 10 --ckpt-dir runs/long --resume
"""

import argparse
import math
import tempfile
import warnings
from pathlib import Path

from engine_set import DEFAULT_PATH as DEFAULT_ENGINE_SET
from engine_set import EngineSet
from exact_labels import SOLVERS, default_threads, make_solve_fn, solve_slots, stops_at
from game import (
    DEFAULT_NET,
    PLAY_PRIOR_TEMPERATURE,
    SGD_WEIGHT_DECAY,
    SYMMETRY_TRANSFORM_FNS,
    TRAVERSAL_STEPS,
    WEIGHT_DECAY,
    env,
    episode_metrics,
    make_network,
    make_optimizer,
    state_to_nn_input,
    step_fn,
)
from xot import load_xot, make_xot_init_fn

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS, PlayoutCap
from core.memory.replay_memory import EpisodeReplayBuffer
from core.monitoring import DEFAULT_URL, Monitor
from core.training.endgame_solve import EndgameSolve
from core.training.exploration import Exploration
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
from core.training.validation import ScoredPositionsValidator

# self-play moves' compute an epoch, in each environment (see --selfplay-cycles)
COLLECTION_STEPS = 128


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


def solve_empties(text: str) -> int | None:
    """Parses --solve-empties: empty squares in [0, 60), or "off" (None)."""
    if text == "off":
        return None
    try:
        empties = int(text)
    except ValueError:
        empties = -1
    if not 0 <= empties < 60:
        raise argparse.ArgumentTypeError(
            f"empty squares in [0, 60) or off, got {text!r}"
        )
    return empties


def play_past(text: str) -> float:
    """Parses --solve-play-past: a mean number of moves of at least 0, or "end" (inf)."""
    try:
        moves = math.inf if text == "end" else float(text)
    except ValueError:
        moves = math.nan
    if not moves >= 0:
        raise argparse.ArgumentTypeError(f"a mean of at least 0 or end, got {text!r}")
    return moves


def playout_cap(text: str) -> PlayoutCap:
    """Parses --playout-cap, FULL_PROBABILITY:FAST_SIMS, e.g. 0.25:43."""
    try:
        probability, fast = text.split(":")
        cap = PlayoutCap(float(probability), int(fast))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected FULL_PROBABILITY:FAST_SIMS, e.g. 0.25:43"
        ) from None
    if not (0 < cap.full_probability <= 1 and cap.fast_iterations > 0):
        raise argparse.ArgumentTypeError(
            f"{text!r}: the probability is in (0, 1], the fast sims positive"
        )
    return cap


def load_engine_set(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> EngineSet | None:
    """Loads the engine-labelled set to validate on, None if there's to be none. A missing
    default set is skipped with a warning (build one with label_positions.py); a missing
    --engine-set is an error."""
    if args.validate_every <= 0:
        return None
    if args.engine_set is not None:
        if not Path(args.engine_set).exists():
            parser.error(f"--engine-set {args.engine_set}: no such file")
        return EngineSet.load(args.engine_set)
    if not DEFAULT_ENGINE_SET.exists():
        warnings.warn(
            f"no engine-labelled set at {DEFAULT_ENGINE_SET}, so the network won't be scored "
            "against an engine: build one with label_positions.py, or pass --engine-set",
            stacklevel=2,
        )
        return None
    return EngineSet.load(DEFAULT_ENGINE_SET)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaZero on Othello.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--net",
        default=DEFAULT_NET,
        help="the network: KIND:DEPTHxWIDTH[,FIELD=VALUE...], e.g. resnet:10x128 for "
        "AlphaZero's ResNet with 10 blocks of 128 channels, resnet:10x128,pool_every=3,"
        "heads=pooled with KataGo's global pooling, or transformer:8x256 (see "
        "core.networks.spec.network_config)",
    )
    # self-play
    parser.add_argument(
        "--envs",
        type=int,
        default=1024,
        help="self-play environments: games played at once, each with --buffer positions of "
        "replay memory. On an L40S, 2048 played about 29%% more moves a second than 1024 "
        "(scan 11x64), 4096 fewer again. A saved state forks into another count only with "
        "--init-without-buffers",
    )
    parser.add_argument(
        "--sims", type=int, default=128, help="MCTS iterations per self-play move"
    )
    parser.add_argument(
        "--tree-factor",
        type=float,
        default=4.0,
        help="self-play search tree capacity, in nodes per MCTS iteration (the tree is "
        "kept between moves, so it needs room for the reused subtree too; at 2, reused "
        "trees fill and searches get clipped)",
    )
    parser.add_argument(
        "--traversal-steps",
        type=int,
        default=TRAVERSAL_STEPS,
        help="steps down the tree a self-play search's traversal takes between checks for "
        "having reached its leaf: the same search whatever the number, but on GPU each check "
        "waits on the host, and more steps a check compute more past the leaf",
    )
    parser.add_argument(
        "--advance-every",
        type=int,
        default=None,
        help="self-play cycles between plays of the moves whose searches are done (a search "
        "done in between goes on until then): by default a whole search's, or 5 with "
        "--playout-cap",
    )
    parser.add_argument(
        "--playout-cap",
        type=playout_cap,
        default=None,
        metavar="P:FAST",
        help="playout cap randomization (KataGo): search each self-play move with --sims "
        "iterations with probability P, and otherwise with FAST, without the root's noise; "
        "only the --sims searches' positions are stored for training, e.g. 0.25:43. Size "
        "--buffer by the share stored, and see --selfplay-cycles and --advance-every",
    )
    parser.add_argument(
        "--selfplay-cycles",
        type=int,
        default=None,
        metavar="CYCLES",
        help="self-play's compute per epoch, in search cycles per environment (a cycle is a "
        f"network evaluation of the root or of an iteration's leaf): by default "
        f"{COLLECTION_STEPS} moves' searches, {COLLECTION_STEPS} x (--sims + 1). Holds "
        "self-play's compute fixed across settings, e.g. with --playout-cap, whose moves take "
        "fewer on average. A multiple of --sims + 1",
    )
    parser.add_argument(
        "--sampling-moves",
        type=int,
        default=20,
        metavar="N",
        help="self-play samples moves in proportion to visit counts for the first N "
        "moves of a game, then plays the most-visited move, as AlphaZero (30) and "
        "OLIVAW (20) did. N counts from the standard start, so it includes an XOT "
        "opening's 8 moves. 0 samples none; one past a game's length, every move",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="concentration of the Dirichlet noise mixed into each self-play root's prior: "
        "lower puts it on fewer moves (AlphaZero: 0.3 chess, 0.15 shogi, 0.03 Go)",
    )
    parser.add_argument(
        "--dirichlet-epsilon",
        type=float,
        default=0.25,
        help="share of each self-play root's prior that is Dirichlet noise (0 for none)",
    )
    parser.add_argument(
        "--standard-starts",
        type=float,
        default=0.0,
        help="fraction of self-play games from the standard start; the rest start from "
        "XOT openings",
    )
    parser.add_argument(
        "--inference-dtype",
        default="bfloat16",
        help="dtype the network computes in during self-play (bfloat16 is about twice as "
        "fast as float32, and plays as well)",
    )
    # exact endgame labels
    parser.add_argument(
        "--solve-empties",
        type=solve_empties,
        default=20,
        metavar="N",
        help="exact endgame labels: once a self-play game reaches N or fewer empty squares, "
        "an exact solver's result for that position is every earlier position's value "
        "target, and with the default --solve-play-past end each stored position from there "
        "on gets its own exact result (see exact_labels.py). A game cut in an epoch is "
        "solved during the next epoch's self-play and trained on from then. Default 20; off "
        "plays games to the end and labels them with the played result",
    )
    parser.add_argument(
        "--solve-play-past",
        type=play_past,
        default=math.inf,
        metavar="MEAN",
        help="with --solve-empties, games play on past N empty squares, ending before each "
        "further move with probability 1 / (1 + MEAN): MEAN moves on average; end (or inf) "
        "plays every game to its end. Each position stored past N is solved and labelled on "
        "its own, so the network still trains on positions past N, which searches from above "
        "N reach. 0 stops games at N. Default end",
    )
    parser.add_argument(
        "--solver",
        choices=SOLVERS,
        default="native",
        help="with --solve-empties, the exact solver: native (endgame.py) or edax (needs "
        "setup_edax.sh, not on Modal)",
    )
    parser.add_argument(
        "--solve-threads",
        type=int,
        default=None,
        help="with --solve-empties, the solver's cores (default: every core this process "
        "may run on; it runs while self-play does, which mostly waits on the GPU)",
    )
    # replay buffer and training
    parser.add_argument(
        "--buffer",
        type=int,
        default=1125,
        help="replay memory: positions kept per environment (1125 is about 9 epochs of "
        "self-play). Training maps each position it samples through a random symmetry",
    )
    parser.add_argument(
        "--buffer-window",
        type=int,
        default=None,
        metavar="N",
        help="train on only each environment's newest N positions of the --buffer kept, as if "
        "the buffer held N: a branch (--branch-from) may widen it at once onto positions the "
        "run already has, e.g. --buffer 9000 --buffer-window 1125. Default all of --buffer",
    )
    parser.add_argument(
        "--train-batch",
        type=int,
        default=4096,
        help="samples per training step; if training runs out of memory with a big "
        "network, see --grad-accum and --remat, which keep the batch",
    )
    parser.add_argument(
        "--train-steps", type=int, default=256, help="training steps per epoch"
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
        "--remat",
        action="store_true",
        help="train with much less memory, for roughly a third more training compute: "
        "keep only each residual block's input for the backward pass, and compute its "
        "activations again (gradient checkpointing). --grad-accum is faster for the same "
        "memory",
    )
    parser.add_argument(
        "--train-dtype",
        default="bfloat16",
        help="dtype the network's convolutions compute in during training: bfloat16 is mixed "
        "precision, its parameters, BatchNorm and heads staying in float32",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate")
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",
        help="AdamW, or SGD with --momentum, as AlphaZero and KataGo trained (they used L2 "
        "weight decay, a rate about 0.2-0.25 at batch 4096 and momentum 0.9). Under SGD a "
        "batch's size trades against the learning rate linearly",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="with --optimizer sgd, the momentum",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="weight decay of the network's weights (not its biases or normalization layers): "
        f"AdamW's (default {WEIGHT_DECAY}), or with --optimizer sgd, w times this added to "
        f"the gradient, an L2 penalty of half this (default {SGD_WEIGHT_DECAY}; AlphaZero's "
        "penalty 1e-4 is 2e-4 here). A branch at a lower --lr holds the weight norm only if "
        "this rises by as much",
    )
    parser.add_argument(
        "--decoupled-decay",
        action="store_true",
        help="with --optimizer sgd, apply weight decay to the step after momentum, as AdamW "
        "does, instead of adding it to the gradient (L2)",
    )
    # the weight average
    parser.add_argument(
        "--average-epochs",
        type=float,
        default=4.0,
        metavar="H",
        help="self-play plays with, validation scores and checkpoints save an average of the "
        "network's weights over about H epochs (an exponential moving average whose horizon "
        "grows from 1 early in the run; training is unchanged); 1 for the network's own "
        "weights",
    )
    parser.add_argument(
        "--average-norm-batches",
        type=int,
        default=32,
        metavar="N",
        help="the weight average gets BatchNorm statistics of its own each epoch, from N "
        "training micro-batches (--train-batch / --grad-accum) of the replay buffer through "
        "its weights (0: it uses the network's statistics, which fit only its own weights)",
    )
    # memorization and validation
    parser.add_argument(
        "--holdout-envs",
        type=int,
        default=0,
        metavar="N",
        help="never train on the games of N of the self-play environments (e.g. 32 of 1024, "
        "about 3%% of the games), and every epoch score the network and its average on their "
        "positions and on the trained-on games' of the same ages: heldout_*, seen_*, and the "
        "difference memorized_*, what training fits in its replay buffer that doesn't carry "
        "over to new games. Default 0, none",
    )
    parser.add_argument(
        "--holdout-positions",
        type=int,
        default=32768,
        metavar="N",
        help="with --holdout-envs, the positions scored from each side every epoch, rounded up "
        "to whole micro-batches: 32768 adds about 5%% to training at the defaults",
    )
    parser.add_argument(
        "--engine-set",
        default=None,
        metavar="PATH",
        help="the engine-labelled set to validate the average on (see engine_set.py; default "
        f"{DEFAULT_ENGINE_SET}, skipped with a warning if it doesn't exist)",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="epochs between scoring the average on the engine-labelled set (0 for never). "
        "A forward pass over its positions: a fraction of a second",
    )
    # saves
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="where to save the weights and states (default: a new temporary directory)",
    )
    parser.add_argument(
        "--save-weights-every",
        type=int,
        default=1,
        metavar="N",
        help="save the weight average (its weights and BatchNorm statistics, what a player "
        "loads) to --ckpt-dir/avg<H>/<epoch>.eqx every N epochs, keeping them all",
    )
    parser.add_argument(
        "--save-raw-every",
        type=int,
        default=None,
        metavar="N",
        help="also save the network's own weights (its raw weights, which the average takes "
        "in) to --ckpt-dir/raw/<epoch>.eqx at the start of every N epochs, all kept: "
        "avg<H>/<epoch>.eqx's raw counterpart",
    )
    parser.add_argument(
        "--save-state-at",
        type=epoch_list,
        default=[],
        metavar="EPOCHS",
        help="also save the whole training state (with the replay buffer: a few GB) to "
        "--ckpt-dir as state-<EPOCH>.npz once this many epochs are done, e.g. 150 or "
        "50,150, to branch or fork runs from",
    )
    parser.add_argument(
        "--save-state-every",
        type=int,
        default=None,
        metavar="N",
        help="also save the whole training state every N epochs, keeping the newest, to "
        "continue the run with --resume if it stops",
    )
    # how the run starts
    init = parser.add_mutually_exclusive_group()
    init.add_argument(
        "--init-from",
        default=None,
        metavar="PATH",
        help="fork a new run from a saved state (state-<EPOCH>.npz, or a --ckpt-dir to take "
        "the newest from): it starts at epoch 0, with its own games and seed, from the saved "
        "network, optimizer state, weight average and replay buffer. Needs the same --net and "
        "--envs; --buffer may differ (each environment's newest entries are kept)",
    )
    parser.add_argument(
        "--init-without-buffers",
        action="store_true",
        help="with --init-from, start with an empty replay buffer instead",
    )
    init.add_argument(
        "--resume",
        action="store_true",
        help="continue the run in --ckpt-dir from its newest saved state, exactly as if it "
        "hadn't stopped; pass the run's own arguments",
    )
    init.add_argument(
        "--branch-from",
        default=None,
        metavar="PATH",
        help="continue a run from a saved state (state-<EPOCH>.npz, or a directory to take the "
        "newest from) in a new --ckpt-dir with these arguments, e.g. a lower --lr or other "
        "--sims: the run goes on at its epoch, with its network, optimizer, weight average, "
        "buffer, games and rng; only the moves in progress are searched again if the search "
        "changed, and a different --buffer resizes the replay buffer. --epochs counts the "
        "run's epochs before the branch too",
    )
    # logging
    parser.add_argument("--name", default=None, help="the run's name on the server")
    parser.add_argument(
        "--notes",
        default=None,
        help="what the run is testing and why, shown on its seaglass page (else $SEAGLASS_NOTES)",
    )
    parser.add_argument(
        "--monitor",
        nargs="?",
        const=DEFAULT_URL,
        default=None,
        metavar="URL",
        help=f"log to a seaglass server (default {DEFAULT_URL}: $SEAGLASS_URL, else slick's)",
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.resume and args.ckpt_dir is None:
        parser.error("--resume needs the run's --ckpt-dir")
    if args.optimizer != "sgd" and args.decoupled_decay:
        parser.error("--decoupled-decay goes with --optimizer sgd")
    if args.weight_decay is None:
        args.weight_decay = (
            SGD_WEIGHT_DECAY if args.optimizer == "sgd" else WEIGHT_DECAY
        )
    if args.average_epochs < 1:
        parser.error("--average-epochs must be at least 1")
    micro = args.train_batch // args.grad_accum
    if args.holdout_envs < 0 or args.holdout_positions <= 0:
        parser.error("--holdout-envs is at least 0, --holdout-positions positive")
    args.holdout_positions = -(-args.holdout_positions // micro) * micro
    if args.init_without_buffers and args.init_from is None:
        parser.error("--init-without-buffers needs --init-from")
    if args.advance_every is None and args.playout_cap is not None:
        # searches of varying lengths: see Trainer's advance_every
        args.advance_every = 5
    if args.playout_cap is not None and args.playout_cap.fast_iterations >= args.sims:
        parser.error("--playout-cap's fast sims must be fewer than --sims")
    collection_steps = COLLECTION_STEPS
    if args.selfplay_cycles is not None:
        if args.selfplay_cycles < 1 or args.selfplay_cycles % (args.sims + 1):
            parser.error(
                f"--selfplay-cycles must be a positive multiple of --sims + 1 ({args.sims + 1})"
            )
        # self-play runs as many cycles as this many moves' searches take at most
        collection_steps = args.selfplay_cycles // (args.sims + 1)
    if args.buffer_window is not None and not 0 < args.buffer_window <= args.buffer:
        parser.error("--buffer-window is 1 to --buffer positions")
    if args.dirichlet_alpha <= 0:
        parser.error("--dirichlet-alpha must be positive")
    if not 0 <= args.dirichlet_epsilon <= 1:
        parser.error("--dirichlet-epsilon must be between 0 and 1")
    if args.save_weights_every < 1:
        parser.error("--save-weights-every must be positive")
    if args.save_raw_every is not None and args.save_raw_every < 1:
        parser.error("--save-raw-every must be positive")

    network, network_state = make_network(
        args.net,
        seed=args.seed,
        inference_dtype=args.inference_dtype,
        train_dtype=args.train_dtype,
        remat=args.remat,
    )
    # AlphaZero takes an arbitrary search backend, here classic MCTS. Temperature 1.0 samples
    # moves in proportion to visit counts, for exploration during self-play
    evaluator = AlphaZero(MCTS)(
        eval_fn=make_nn_eval_fn(network, state_to_nn_input),
        num_iterations=args.sims,
        max_nodes=int(args.tree_factor * args.sims),
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=1.0,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        traversal_steps=args.traversal_steps,
        playout_cap=args.playout_cap,
    )

    # the average's raw moves and values against an engine's, on the engine-labelled set, in
    # float32: in bfloat16 a checkpoint's value MSE is off by up to 0.006, more than ten epochs
    # of a slow run's progress
    engine_set = load_engine_set(parser, args)
    validators = (
        [
            ScoredPositionsValidator(
                engine_set.scored_positions(),
                make_network(args.net, inference_dtype="float32")[0],
                name="engine",
                config={
                    "path": str(args.engine_set or DEFAULT_ENGINE_SET),
                    **engine_set.describe(),
                },
                prior_temperature=PLAY_PRIOR_TEMPERATURE,
            )
        ]
        if engine_set is not None
        else []
    )

    endgame_solve = None
    if args.solve_empties is not None:
        endgame_solve = EndgameSolve(
            stop_fn=stops_at(args.solve_empties),
            solve_fn=make_solve_fn(
                args.solver, args.solve_threads or default_threads()
            ),
            slots=solve_slots(
                collection_steps * evaluator.search_cycles(),
                args.advance_every or evaluator.search_cycles(),
                args.solve_empties,
                args.solve_play_past,
            ),
            play_past=args.solve_play_past,
        )

    ckpt_dir = args.ckpt_dir or tempfile.mkdtemp(prefix="turbozero-othello-")
    trainer = Trainer(
        batch_size=args.envs,
        train_batch_size=args.train_batch,
        collection_steps_per_epoch=collection_steps,
        train_steps_per_epoch=args.train_steps,
        nn=network,
        nn_state=network_state,
        loss_fn=az_default_loss_fn,
        optimizer=make_optimizer(
            args.lr,
            args.weight_decay,
            sgd=args.optimizer == "sgd",
            momentum=args.momentum,
            decoupled=args.decoupled_decay,
        ),
        evaluator=evaluator,
        # stores `capacity` positions for each of the `batch_size` environments
        memory_buffer=EpisodeReplayBuffer(
            capacity=args.buffer, window=args.buffer_window
        ),
        max_episode_steps=80,
        env_step_fn=step_fn,
        # self-play starts games from XOT openings (see xot.py) for varied, balanced games
        env_init_fn=make_xot_init_fn(load_xot(), args.standard_starts),
        state_to_nn_input_fn=state_to_nn_input,
        grad_accum_steps=args.grad_accum,
        average_epochs=args.average_epochs,
        average_norm_batches=args.average_norm_batches,
        holdout_envs=args.holdout_envs,
        holdout_positions=args.holdout_positions if args.holdout_envs else 0,
        validators=validators,
        validate_every=max(args.validate_every, 1),
        exploration=Exploration(num_sampling_moves=args.sampling_moves),
        endgame_solve=endgame_solve,
        # how often self-play games are drawn
        episode_metrics_fn=episode_metrics,
        # training maps each sample through one of the board's 8 symmetries, at random
        data_transform_fns=SYMMETRY_TRANSFORM_FNS,
        advance_every=args.advance_every,
        monitor=Monitor(
            args.monitor, project="othello", name=args.name, notes=args.notes
        )
        if args.monitor
        else None,
        ckpt_dir=ckpt_dir,
        save_weights_every=args.save_weights_every,
        save_raw_every=args.save_raw_every,
        save_state_every=args.save_state_every,
        save_state_at=args.save_state_at,
        # every argument but how the run starts (a fork or branch records its parent)
        extra_config={
            "args": {
                k: v.get_config() if isinstance(v, PlayoutCap) else v
                for k, v in vars(args).items()
                if k
                not in (
                    "resume",
                    "init_from",
                    "init_without_buffers",
                    "branch_from",
                    "notes",
                )
            }
        },
    )
    initial_state = None
    if args.resume:
        initial_state = trainer.resume()
    elif args.branch_from is not None:
        try:
            initial_state = trainer.branch(args.branch_from)
        except ValueError as e:
            parser.error(f"--branch-from: {e}")
    trainer.train_loop(
        seed=args.seed,
        num_epochs=args.epochs,
        initial_state=initial_state,
        fork_from=args.init_from,
        fork_replay_buffer=not args.init_without_buffers,
    )


if __name__ == "__main__":
    main()
