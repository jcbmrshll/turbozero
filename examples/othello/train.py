"""AlphaZero on Othello, tracked by how close the raw network's moves and values come to
Edax's on fixed positions.

Self-play games are collected in parallel across a batch of environments (and across
every available GPU), with Monte Carlo Tree Search run on each of them; the network
then trains on minibatches sampled from replay memory. Self-play games start from XOT
openings (see xot.py), which keeps them varied.

    uv run examples/othello/train.py
    uv run examples/othello/train.py --epochs 20 --monitor
    uv run examples/othello/train.py --sims-schedule 0:32,50:64,150:128

Every epoch it scores the network on the engine-labelled set, if there is one (see
engine_set.py and --engine-set), logged as `engine_*`: `engine_exp_regret`, the discs its
policy gives up against Edax's best move, is the one to watch. Games against Edax measure
strength: --vs-engine-every N plays them after every N epochs, and --vs-engine-last K after
each of the last K epochs, logging the mean of those too (by default 512 games at level 8,
searching 1000 iterations a move; see vs_engine.py and README.md):

    uv run examples/othello/train.py --epochs 50 --monitor --vs-engine-last 3

Each level's score is logged at the epoch of the network that played, as
`edax_level8_sims1000` (with `_se`, `_win`, `_draw` and `_loss`), the mean as
`edax_level8_sims1000_last3`, and the time the games took as `edax_seconds`, to take out of
the run's wall time. --eval-every N also plays test
games every N epochs, on a ladder of opponents: a random player, a greedy tile counter, then
pgx's pretrained Othello model searching more and more.

With --monitor the run logs to slick's seaglass server ($SEAGLASS_URL, set in a slick task;
--monitor URL for another, e.g. `uv run seaglass serve --dir runs` at http://localhost:8008).
The run's page shows the metrics, and with
--eval-every which rung of the ladder the agent has reached, and a game against the last
opponent it played, which it draws in the browser.

The first epoch is slow: nearly all of the training loop is JIT-compiled the first
time it runs. On one RTX 5080 the rest take about 63s each (36s with
--inference-dtype bfloat16, see README.md). With these settings and the ladder every 5
epochs, one 200-epoch run passed every rung by epoch 65; at the end, in 512 games against
pgx's model (our agent searching 64 iterations a move), it scored 0.82 against the model
searching 64, 0.74 against it searching 256, and 0.57 against it searching 1024
(draws count half).

Checkpoints go to --ckpt-dir, with the results of games against the engine in
`vs_engine.jsonl`, and the last is exported to ONNX beside itself when the run ends; `eval_pgx.py` and `evaluate.py` evaluate checkpoints further (see
README.md). The hyperparameters here are only an example; tune them for your task and
hardware.

With --save-state-at or --save-state-every, the whole training state (network, optimizer,
replay buffers, games in progress, rng) is saved there too, to continue the run exactly
with --resume, or to fork short runs from with --init-from (see README.md):

    uv run examples/othello/train.py --epochs 150 --lr-final 1e-3 --save-state-at 150 --ckpt-dir runs/trunk
    uv run examples/othello/train.py --epochs 30 --init-from runs/trunk --ckpt-dir runs/fork-control
    uv run examples/othello/train.py --epochs 200 --save-state-every 10 --ckpt-dir runs/long
    uv run examples/othello/train.py --epochs 200 --save-state-every 10 --ckpt-dir runs/long --resume

With --lr-plateau, the learning rate drops when the training loss stops improving instead,
from --lr to each of its rates in turn, --epochs only capping the run (see README.md):

    uv run examples/othello/train.py --epochs 600 --lr-plateau --save-state-every 10 --ckpt-dir runs/plateau
"""

import argparse
import os
import re
import tempfile
import warnings
from functools import partial
from pathlib import Path

import pgx
from engine_set import DEFAULT_PATH as DEFAULT_ENGINE_SET
from engine_set import EngineSet
from engines import ENGINES
from exact_labels import SOLVERS, default_threads, make_solve_fn, solve_slots, stops_at
from game import (
    DEFAULT_NET,
    SGD_WEIGHT_DECAY,
    SYMMETRY_TRANSFORM_FNS,
    TRAVERSAL_STEPS,
    WEIGHT_DECAY,
    env,
    episode_metrics,
    export_last,
    greedy_eval,
    make_network,
    make_optimizer,
    make_test_evaluator,
    state_to_nn_input,
    step_fn,
)
from vs_engine import EngineMatch, EngineMatchTester, describe_scores
from xot import load_xot, make_xot_init_fn, xot_episode_fn

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import (
    make_nn_eval_fn,
    make_nn_eval_fn_no_params_callable,
)
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS, PlayoutCap
from core.evaluators.mcts.policy_targets import CompletedQ, VisitCounts
from core.evaluators.random_evaluator import RandomEvaluator
from core.memory.replay_memory import EpisodeReplayBuffer
from core.monitoring import DEFAULT_URL, Monitor
from core.monitoring.episodes import pgx_two_player_episode
from core.testing.ladder import LadderTester, Rung
from core.testing.scored_positions import ScoredPositionsTester
from core.testing.tester import BaseTester
from core.training.endgame_solve import EndgameSolve
from core.training.exploration import Exploration
from core.training.growth import Growth, Learner
from core.training.loss_fns import az_default_loss_fn
from core.training.lr_drop import LRDrop
from core.training.lr_plateau import LRPlateau
from core.training.resignation import Resignation
from core.training.schedule import EvaluatorSchedule, parse_schedule
from core.training.train import Trainer
from core.training.tree_positions import TreePositions


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


# training holds --lr, then drops 10x late in the run and again near its end: AlphaGo Zero's three
# levels, and about where Lc0's example configuration and KataGo drop. At 110 epochs of 4x64 it
# played within noise of one drop at 83%, and far above a constant learning rate
DEFAULT_LR_DROPS = [(0.7, 1e-4), (0.9, 1e-5)]


def lr_drops(text: str) -> list[tuple[float, float]]:
    """Parses --lr-drops."""
    try:
        drops = [
            (float(a), float(b)) for a, b in (d.split(":") for d in text.split(","))
        ]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} isn't FRACTION:LR,...") from e
    fractions = [fraction for fraction, _ in drops]
    if fractions != sorted(set(fractions)) or not 0 < fractions[0] <= fractions[-1] < 1:
        raise argparse.ArgumentTypeError(f"{text!r}: fractions rise, between 0 and 1")
    return drops


def warm_start_spec(text: str) -> tuple[str, str]:
    """Parses --warm-start: SIZE=PATH."""
    size, sep, path = text.partition("=")
    if not (size and sep and path):
        raise argparse.ArgumentTypeError(
            f"{text!r} isn't SIZE=PATH, e.g. 4x64=run/state-100.npz"
        )
    return size, path


def lr_drop_spec(text: str) -> LRDrop:
    """Parses --lr-drop: EPOCHS:FACTOR."""
    try:
        epochs, factor = text.split(":")
        return LRDrop(int(epochs), float(factor))
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"{text!r} isn't EPOCHS:FACTOR, epochs at least 0 and a factor in (0, 1]"
        ) from e


# the rates --lr-plateau drops to from --lr's 1e-3, as --lr-drops' do. On the training losses of
# six rated 6x96 and 8x128 phases at 1e-3 and 1e-4, the rule (40 epochs, 0.004) fired between
# about 7 epochs before and 36 after their Elo stopped rising
DEFAULT_LR_PLATEAU = [1e-4, 1e-5]


def rate_list(text: str) -> list[float]:
    """Parses --lr-plateau: learning rates separated by commas, each below the last."""
    try:
        rates = [float(rate) for rate in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected learning rates separated by commas, e.g. 1e-4,1e-5, got {text!r}"
        ) from None
    if any(rate <= 0 for rate in rates) or rates != sorted(set(rates), reverse=True):
        raise argparse.ArgumentTypeError(f"{text!r}: rates fall, and are positive")
    return rates


def positive_schedule(text: str) -> list[tuple[int, int]]:
    """Parses --sims-schedule, whose values must be positive."""
    try:
        schedule = parse_schedule(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    if any(value < 1 for _, value in schedule):
        raise argparse.ArgumentTypeError(f"values must be positive, got {text!r}")
    return schedule


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
    """The engine-labelled set to score the network on, None if it isn't to be. A missing
    default set is skipped with a warning (build one with label_positions.py); a missing
    --engine-set is an error."""
    if args.engine_eval_every <= 0:
        return None
    if args.engine_set is not None:
        if not Path(args.engine_set).exists():
            parser.error(f"--engine-set {args.engine_set}: no such file")
        return EngineSet.load(args.engine_set)
    if not DEFAULT_ENGINE_SET.exists():
        warnings.warn(
            f"no engine-labelled set at {DEFAULT_ENGINE_SET}, so the network won't be scored "
            "against Edax: build one with label_positions.py, or pass --engine-set",
            stacklevel=2,
        )
        return None
    return EngineSet.load(DEFAULT_ENGINE_SET)


def net_name(spec: str) -> str:
    """A network's short name, for --grow's metrics and checkpoint directories: its size, e.g.
    4x64 for resnet:4x64,pool_every=2."""
    found = re.search(r"\d+x\d+", spec)
    return found.group(0) if found else spec.split(",")[0].replace(":", "-")


def main():
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
    parser.add_argument(
        "--grow",
        action="append",
        default=[],
        metavar="NET",
        help="a bigger network to grow into (repeat for several, smallest first), as --net: "
        "--net plays self-play first, and each takes over in turn, training alongside the one "
        "playing until it does, on the same replay buffer (see core/training/growth.py). Each "
        "network's checkpoints go to --ckpt-dir/<its size>, and its metrics are logged as "
        "<its size>/<metric>",
    )
    parser.add_argument(
        "--grow-metric",
        default="engine_value_mse",
        choices=["engine_value_mse", "engine_exp_regret", "loss"],
        help="with --grow, what decides when the next network takes over: its error on the "
        "engine-labelled set (scored every epoch, --engine-eval-every 1), or its training loss "
        "on the data both train on (KataGo's rule), averaged over --grow-window epochs, at or "
        "below the playing network's",
    )
    parser.add_argument(
        "--grow-margin",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="with --grow, the lead the next network needs to take over: its mean --grow-metric "
        "lower than the playing one's by this fraction of it, e.g. 0.01 for 1%%, so that two "
        "networks in lockstep don't switch (default 0: at or below)",
    )
    parser.add_argument(
        "--grow-train-retired",
        action="store_true",
        help="with --grow, the networks that played before the one playing keep training on "
        "its games, --train-steps each an epoch (for the record: not compute-optimal); a "
        "resumed or branched run can turn it on or off",
    )
    parser.add_argument(
        "--grow-train-steps",
        type=int,
        default=None,
        metavar="N",
        help="with --grow, the training steps an epoch of the network that trains without "
        "playing (default --train-steps): fewer charge it what the run could spend on it",
    )
    parser.add_argument(
        "--grow-guard",
        default=None,
        choices=["engine_value_mse", "engine_exp_regret"],
        help="with --grow, a second metric the next network must be no worse on over "
        "--grow-window epochs, e.g. engine_value_mse on positions neither network trains on, "
        "when --grow-metric is the training loss, which a bigger network can lower by "
        "memorising the buffer (needs --engine-eval-every 1)",
    )
    parser.add_argument(
        "--grow-window",
        type=int,
        default=10,
        help="with --grow, the epochs --grow-metric is averaged over",
    )
    parser.add_argument(
        "--grow-min-epochs",
        type=int,
        default=10,
        help="with --grow, the epochs the next network trains before it can take over",
    )
    parser.add_argument(
        "--grow-switch-at",
        type=epoch_list,
        default=None,
        metavar="EPOCHS",
        help="with --grow, instead of --grow-metric's rule, the numbers of epochs done after "
        "which each network takes over, e.g. 80 (one per --grow)",
    )
    parser.add_argument(
        "--grow-manual",
        action="store_true",
        help="with --grow, nothing takes over by itself: the rule only logs when the next "
        "network has caught up (growth_caught_up), and it takes over in a branch of the run "
        "with --take-over (see controller.py)",
    )
    sims = parser.add_mutually_exclusive_group()
    sims.add_argument(
        "--sims", type=int, default=128, help="MCTS iterations per self-play move"
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
        "--count-reused-visits",
        action="store_true",
        help="a self-play search stops once its root has --sims + 1 visits, those of the "
        "subtree it kept from the last move's search included, so it searches less from a "
        "well-explored subtree (the tree is then sized for it, and --tree-factor is unused)",
    )
    parser.add_argument(
        "--no-persist-tree",
        action="store_true",
        help="start every self-play search from an empty tree, rather than from the subtree "
        "of the move played that the last search explored (by default kept, so a target is "
        "built on its visits plus --sims more): an ablation of tree reuse",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=0,
        help="with --count-reused-visits, the fewest iterations a search takes",
    )
    parser.add_argument(
        "--advance-every",
        type=int,
        default=None,
        help="self-play cycles between plays of the moves whose searches are done (a search "
        "done in between goes on until then): by default a whole search's, or 5 with "
        "--count-reused-visits",
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
        "network evaluation of the root or of an iteration's leaf): by default 128 moves' "
        "searches, 128 x (--sims + 1). Holds self-play's compute fixed across settings, e.g. "
        "with --playout-cap, whose moves take fewer on average. A multiple of --sims + 1",
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
        "--resign",
        type=float,
        nargs="?",
        const=0.95,
        default=None,
        metavar="THRESHOLD",
        help="self-play's player to move resigns when its search's root value is at most "
        "-THRESHOLD: 0.95 (an expected score of 2.5%%) if no threshold is given, 0.9 for "
        "5%%. Saves the searches of decided endgames for more games at "
        "the same compute: on 4x64 at 0.9, 1.26x the games per epoch, +7 ± 17 Elo. "
        "Default: games are played to the end",
    )
    parser.add_argument(
        "--resign-calibration",
        type=float,
        default=0.1,
        help="with --resign, the fraction of games played to the end anyway, to measure how "
        "often resigning would have been wrong (logged as resign_false)",
    )
    parser.add_argument(
        "--solve-empties",
        type=int,
        default=None,
        metavar="N",
        help="exact endgame labels: self-play's games stop at N or fewer empty squares, and "
        "an exact solver's result for that position is every position's value target (see "
        "exact_labels.py). A game stopped in an epoch is solved during the next epoch's "
        "self-play and trained on from then. Default: games are played to the end",
    )
    parser.add_argument(
        "--solve-share",
        type=float,
        default=1.0,
        help="with --solve-empties, the fraction of games that stop; the others are played "
        "to the end, so that training still sees positions past N",
    )
    parser.add_argument(
        "--solve-play-past",
        type=float,
        default=0.0,
        metavar="MEAN",
        help="with --solve-empties, games play on past N empty squares, ending before each "
        "further move with probability 1 / (1 + MEAN): MEAN moves on average. Each position "
        "stored past N is solved and labelled on its own, so the network still trains on "
        "positions just past N, which searches from above N reach. Default 0: games stop at N",
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
    parser.add_argument(
        "--train-batch",
        type=int,
        default=4096,
        help="samples per training step; if training runs out of memory with a big "
        "network, see --remat and --grad-accum, which keep the batch",
    )
    parser.add_argument(
        "--train-steps", type=int, default=256, help="training steps per epoch"
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
        default=1125,
        help="replay memory: positions kept per environment (1125 is about 9 epochs of "
        "self-play). Training maps each position it samples through a random symmetry",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="the initial learning rate"
    )
    parser.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="instead of --lr-drops' schedule, decay along a cosine to this learning rate; "
        "equal to --lr for a constant learning rate",
    )
    parser.add_argument(
        "--lr-drops",
        type=lr_drops,
        default=None,
        help="hold --lr and drop in steps: FRACTION:LR,... of the run's training steps "
        "(default: 0.7:1e-4,0.9:1e-5, two drops of 10x)",
    )
    parser.add_argument(
        "--lr-plateau",
        type=rate_list,
        nargs="?",
        const=DEFAULT_LR_PLATEAU,
        default=None,
        metavar="LR,...",
        help="instead of --lr-drops' schedule, hold --lr and drop to each of these in turn when "
        "the training loss stops improving: when it improved by less than "
        "--lr-plateau-threshold over the last --lr-plateau-window epochs, by the slope of a "
        "least-squares line through the epochs' losses, after at least --lr-plateau-min-epochs "
        "epochs at the rate (default: 1e-4,1e-5). A fork starts again at --lr",
    )
    parser.add_argument(
        "--lr-plateau-window",
        type=int,
        default=40,
        metavar="EPOCHS",
        help="with --lr-plateau, the epochs the loss's improvement is measured over",
    )
    parser.add_argument(
        "--lr-plateau-threshold",
        type=float,
        default=0.004,
        help="with --lr-plateau, the improvement over the window below which the rate drops",
    )
    parser.add_argument(
        "--lr-plateau-min-epochs",
        type=int,
        default=40,
        metavar="EPOCHS",
        help="with --lr-plateau, the epochs to train at a rate before it can drop",
    )
    parser.add_argument(
        "--lr-plateau-metric",
        default="loss",
        choices=["loss", "engine_value_mse", "engine_exp_regret"],
        help="with --lr-plateau, what the rule watches: the training loss, or the network's raw value "
        "error (engine_value_mse) or expected regret (engine_exp_regret) on the engine-labelled set "
        "(see --engine-set), fixed positions whose targets don't move as the network improves. "
        "Scored every epoch for it (--engine-eval-every 1). A conservative setting that waits for "
        "Elo to stop rising (small Othello nets): engine_value_mse with --lr-plateau-window 60 "
        "--lr-plateau-threshold 0.005",
    )
    parser.add_argument(
        "--lr-plateau-stop",
        action="store_true",
        help="with --lr-plateau, end the run once the last rate plateaus, before --epochs "
        "(--vs-engine-last's games are then not played)",
    )
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
        f"AdamW's (default {WEIGHT_DECAY}), or with --optimizer sgd, w times this added to the "
        f"gradient, an L2 penalty of half this (default {SGD_WEIGHT_DECAY}; AlphaZero's penalty "
        "1e-4 is 2e-4 here)",
    )
    parser.add_argument(
        "--decoupled-decay",
        action="store_true",
        help="with --optimizer sgd, apply weight decay to the step after momentum, as AdamW "
        "does, instead of adding it to the gradient (L2)",
    )
    parser.add_argument(
        "--lr-drop",
        type=lr_drop_spec,
        default=None,
        metavar="EPOCHS:FACTOR",
        help="scale the playing network's learning rate by FACTOR once it has played self-play "
        "for EPOCHS epochs (with --grow, each network counts its own, from taking over; a "
        "network waiting to take over trains at the full rate), e.g. 150:0.1",
    )
    parser.add_argument(
        "--noise-measure",
        type=int,
        default=0,
        metavar="K",
        help="every epoch, measure each training network's gradient noise scale at its weight "
        "average (--selfplay-average's, else the first of --average-epochs) from K "
        "micro-batches, logged as gns_avg_*: K 64 adds about 6%% to training. For watching; "
        "nothing follows it",
    )
    parser.add_argument(
        "--fresh-positions",
        type=int,
        default=0,
        metavar="N",
        help="every epoch, score the playing network on N of the positions its self-play just "
        "stored, before it trains on them: its raw weights' and its self-play average's policy "
        "KL and value loss (fresh_*), and the raw weights' excess (fresh_*_gap), what the "
        "step's jitter costs on new data. N a multiple of the micro-batch (--train-batch / "
        "--grad-accum), e.g. 16 of them. For watching; nothing follows it. Default 0, none",
    )
    parser.add_argument(
        "--inference-dtype",
        default="bfloat16",
        help="dtype the network computes in during self-play and test games (bfloat16 is "
        "about twice as fast as float32, and plays as well)",
    )
    parser.add_argument(
        "--train-dtype",
        default="bfloat16",
        help="dtype the network's convolutions compute in during training: bfloat16 is mixed "
        "precision, its parameters, BatchNorm and heads staying in float32",
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
        "--policy-target",
        choices=["visits", "completed-q"],
        default="visits",
        help="train the policy of played positions on the search's visit distribution "
        "(the default, as AlphaZero), or on Gumbel MuZero's improved policy: the "
        "network's policy at the root, before the root's noise, shifted towards the "
        "children the search found better (see core/evaluators/mcts/policy_targets.py)",
    )
    parser.add_argument(
        "--value-scale",
        type=float,
        default=0.1,
        help="with --policy-target completed-q, how far the children's values shift the "
        "policy, per visit of the most visited child (mctx's default; Gumbel MuZero's "
        "Go and chess agents used 1)",
    )
    parser.add_argument(
        "--maxvisit-init",
        type=float,
        default=50.0,
        help="with --policy-target completed-q, visits added to the most visited "
        "child's in scaling the children's values",
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
        help="tree positions kept per environment (default: "
        "K times --buffer). Training samples those stored in the replay memory's span "
        "of self-play",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="epochs between test games on the ladder (0 for none: the engine-labelled "
        "set tracks the run, see --engine-set)",
    )
    parser.add_argument(
        "--engine-set",
        default=None,
        metavar="PATH",
        help="the engine-labelled set to score the network on (see engine_set.py; default "
        f"{DEFAULT_ENGINE_SET}, skipped with a warning if it doesn't exist)",
    )
    parser.add_argument(
        "--engine-eval-every",
        type=int,
        default=1,
        help="epochs between scoring the network on the engine-labelled set (0 for never). "
        "A forward pass over its positions: a fraction of a second",
    )
    parser.add_argument(
        "--vs-engine-every",
        type=int,
        default=0,
        metavar="N",
        help="play games against --vs-engine after every N epochs, between epochs (0, the "
        "default, for never), and log them to the run at the epoch of the network that "
        "played (see vs_engine.py). At the defaults, 512 games at level 8 take minutes",
    )
    parser.add_argument(
        "--vs-engine-last",
        type=int,
        default=0,
        metavar="K",
        help="play the games after each of the run's last K epochs too, and log their mean "
        "at the last, as <name>_last<K>: one network's score drifts from its neighbours' "
        "by more than the games' own noise (0, the default, for none)",
    )
    parser.add_argument(
        "--vs-engine",
        choices=sorted(ENGINES),
        default="edax",
        help="the engine to play",
    )
    parser.add_argument(
        "--vs-engine-levels",
        type=int,
        nargs="+",
        default=[8],
        help="the engine's levels to play at",
    )
    parser.add_argument(
        "--vs-engine-sims",
        type=int,
        default=1000,
        help="our MCTS iterations a move in games against the engine",
    )
    parser.add_argument(
        "--vs-engine-openings",
        type=int,
        default=256,
        help="XOT openings a level in games against the engine, each played with both "
        "colors: the same ones every time, and in every run",
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
        "--average-epochs",
        type=float,
        nargs="+",
        default=[],
        metavar="H",
        help="also keep averages of each network's weights over about H epochs (exponential "
        "moving averages, see --average-warmup; training is unchanged), "
        "checkpointed to avg<H> in its checkpoint directory and scored on the engine-labelled "
        "set as avg<H>/engine_*",
    )
    parser.add_argument(
        "--selfplay-average",
        type=float,
        default=None,
        metavar="H",
        help="self-play plays with the playing network's average over H epochs (one of "
        "--average-epochs) rather than its current weights",
    )
    parser.add_argument(
        "--average-warmup",
        type=float,
        default=10.0,
        metavar="W",
        help="an average's horizon starts at 1 epoch and grows by one every W epochs it takes "
        "in, up to its own: early in training, when every epoch improves the network a lot, a "
        "long average lags it (1: the plain mean of the epochs so far until the horizon)",
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
        "--lr) and replay buffers. Needs the same --net; --buffer may "
        "differ (each environment's newest entries are kept)",
    )
    parser.add_argument(
        "--fork-into",
        default=None,
        metavar="SIZE",
        help="with --init-from: the saved run's network to fork, by size (e.g. 8x128), from a "
        "--grow run's state too, with its weight averages. With --grow, it forks into this "
        "run's network of that size, which plays from the start; the ones before it are retired "
        "(training beside it with --grow-train-retired), the next waits, and the rest start from "
        "scratch. Without --grow, it is this run's --net, trained alone",
    )
    parser.add_argument(
        "--warm-start",
        type=warm_start_spec,
        action="append",
        default=[],
        metavar="SIZE=PATH",
        help="with --grow, in a new run: this run's network of that size (e.g. 4x64) starts from "
        "a saved state's (state-<EPOCH>.npz, or a --ckpt-dir to take the newest from; a --grow "
        "run's network of that size, or a run's one network) instead of from scratch, with its "
        "optimizer state (the learning rate decaying anew) and a --grow run's weight averages, "
        "e.g. a student beside a teacher forked with --init-from and --fork-into. Repeatable; "
        "not the network --init-from forks into",
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
    init.add_argument(
        "--branch-from",
        default=None,
        metavar="PATH",
        help="continue a run from a saved state (state-<EPOCH>.npz, or a directory to take the "
        "newest from) in a new --ckpt-dir, with these arguments' self-play, e.g. other --sims: "
        "the run goes on at its epoch, with its networks, optimizers, buffers, games and rng; "
        "only the moves in progress are searched again if the search changed, and a different "
        "--buffer resizes the replay buffer. --epochs counts the run's epochs before the branch "
        "too (see Trainer.branch and controller.py)",
    )
    parser.add_argument(
        "--take-over",
        action="store_true",
        help="with --branch-from and --grow, the next network plays from the branch's first epoch",
    )
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
    args = parser.parse_args()
    if args.resume and args.ckpt_dir is None:
        parser.error("--resume needs the run's --ckpt-dir")
    if args.optimizer != "sgd" and args.decoupled_decay:
        parser.error("--decoupled-decay goes with --optimizer sgd")
    if args.weight_decay is None:
        args.weight_decay = (
            SGD_WEIGHT_DECAY if args.optimizer == "sgd" else WEIGHT_DECAY
        )
    micro = args.train_batch // args.grad_accum
    if args.fresh_positions < 0 or args.fresh_positions % micro:
        parser.error(
            f"--fresh-positions must be a multiple of the micro-batch ({micro}, --train-batch / "
            "--grad-accum)"
        )
    if args.noise_measure < 0:
        parser.error("--noise-measure must be at least 0")
    if args.init_without_buffers and args.init_from is None:
        parser.error("--init-without-buffers needs --init-from")
    if args.fork_into is not None and args.init_from is None:
        parser.error("--fork-into needs --init-from")
    if args.warm_start and not args.grow:
        parser.error("--warm-start is for a run with --grow")
    if args.warm_start and (args.resume or args.branch_from is not None):
        parser.error("--warm-start is for a new run, not --resume or --branch-from")
    if args.grow_train_steps is not None and (
        not args.grow or args.grow_train_steps < 1
    ):
        parser.error("--grow-train-steps is a positive number of steps, with --grow")
    if args.grow_train_retired and not args.grow:
        parser.error("--grow-train-retired is for a run with --grow")
    if any(h < 1 for h in args.average_epochs) or len(set(args.average_epochs)) != len(
        args.average_epochs
    ):
        parser.error("--average-epochs are distinct horizons of at least 1 epoch")
    if (
        args.selfplay_average is not None
        and args.selfplay_average not in args.average_epochs
    ):
        parser.error("--selfplay-average must be one of --average-epochs")
    if args.average_warmup <= 0:
        parser.error("--average-warmup must be positive")
    if sum(x is not None for x in (args.lr_final, args.lr_drops, args.lr_plateau)) > 1:
        parser.error(
            "--lr-final, --lr-drops and --lr-plateau are different schedules: pass one"
        )
    lr_plateau = None
    if args.lr_plateau is not None:
        if args.lr_plateau[0] >= args.lr:
            parser.error("--lr-plateau's rates must be below --lr")
        try:
            lr_plateau = LRPlateau(
                (args.lr, *args.lr_plateau),
                window=args.lr_plateau_window,
                threshold=args.lr_plateau_threshold,
                min_epochs=args.lr_plateau_min_epochs,
                stop=args.lr_plateau_stop,
            )
        except ValueError as e:
            parser.error(str(e))
    if args.lr_plateau_metric != "loss":
        if args.lr_plateau is None:
            parser.error("--lr-plateau-metric goes with --lr-plateau")
        if args.engine_eval_every != 1:
            parser.error(
                f"--lr-plateau-metric {args.lr_plateau_metric} needs --engine-eval-every 1"
            )
    if args.no_persist_tree and args.count_reused_visits:
        parser.error(
            "--count-reused-visits counts the visits of the subtree a search keeps, which "
            "--no-persist-tree discards"
        )
    if args.advance_every is None and (
        args.count_reused_visits or args.playout_cap is not None
    ):
        # searches of varying lengths: see Trainer's advance_every
        args.advance_every = 5
    if args.playout_cap is not None:
        if args.sims_schedule is not None:
            parser.error("--playout-cap goes with --sims, not --sims-schedule")
        if args.playout_cap.fast_iterations >= args.sims:
            parser.error("--playout-cap's fast sims must be fewer than --sims")
    collection_steps = 128
    if args.selfplay_cycles is not None:
        if args.sims_schedule is not None:
            parser.error("--selfplay-cycles goes with --sims, not --sims-schedule")
        if args.selfplay_cycles < 1 or args.selfplay_cycles % (args.sims + 1):
            parser.error(
                f"--selfplay-cycles must be a positive multiple of --sims + 1 ({args.sims + 1})"
            )
        # self-play runs as many cycles as this many moves' searches take at most
        collection_steps = args.selfplay_cycles // (args.sims + 1)
    if args.resign is not None and not 0 < args.resign <= 1:
        parser.error("--resign must be in (0, 1]")
    if not 0 <= args.resign_calibration <= 1:
        parser.error("--resign-calibration must be between 0 and 1")
    if args.solve_empties is not None:
        if not 0 <= args.solve_empties < 60:
            parser.error("--solve-empties must be in [0, 60)")
        if args.resign is not None:
            parser.error("--solve-empties doesn't go with --resign")
    if not 0 <= args.solve_share <= 1:
        parser.error("--solve-share must be between 0 and 1")
    if args.solve_play_past < 0:
        parser.error("--solve-play-past must be at least 0")
    if args.dirichlet_alpha <= 0:
        parser.error("--dirichlet-alpha must be positive")
    if not 0 <= args.dirichlet_epsilon <= 1:
        parser.error("--dirichlet-epsilon must be between 0 and 1")
    if args.vs_engine_last > args.epochs:
        parser.error("--vs-engine-last can't be more than --epochs")
    if args.policy_target == "completed-q" and args.tree_positions > 0:
        # a tree position's target is its visits: the network's policy before the
        # search isn't kept below the root
        parser.error("--policy-target completed-q can't go with --tree-positions")

    if not args.grow and (
        args.grow_switch_at is not None
        or args.grow_manual
        or any(
            getattr(args, k) != parser.get_default(k)
            for k in (
                "grow_metric",
                "grow_window",
                "grow_min_epochs",
                "grow_margin",
                "grow_guard",
            )
        )
    ):
        parser.error(
            "--grow-metric, --grow-window, --grow-min-epochs, --grow-margin, --grow-guard, "
            "--grow-switch-at and --grow-manual go with --grow"
        )
    if not 0 <= args.grow_margin < 1:
        parser.error("--grow-margin is a fraction in [0, 1)")
    if args.grow_manual and args.grow_switch_at is not None:
        parser.error("--grow-manual and --grow-switch-at are different rules: pass one")
    # the engine set's metrics need it every epoch; the training loss is logged every epoch
    engine_rule = (
        args.grow
        and args.grow_switch_at is None
        and (args.grow_metric != "loss" or args.grow_guard is not None)
    )
    if engine_rule and args.engine_eval_every != 1:
        parser.error(
            "--grow's rule needs the engine set every epoch: --engine-eval-every 1"
        )
    if args.take_over and (args.branch_from is None or not args.grow):
        parser.error("--take-over goes with --branch-from and --grow")

    # self-play starts games from XOT openings (see xot.py) for varied, balanced
    # games
    selfplay_init_fn = make_xot_init_fn(load_xot(), args.standard_starts)
    # test games all start from XOT openings
    test_init_fn = make_xot_init_fn(load_xot(), standard_start_fraction=0)

    def make_net(spec: str):
        """A network (see game.py), with its state (BatchNorm's, if it has any)."""
        return make_network(
            spec,
            seed=args.seed,
            inference_dtype=args.inference_dtype,
            train_dtype=args.train_dtype,
            remat=args.remat,
        )

    def make_selfplay(network) -> EvaluatorSchedule:
        """A network's self-play search. AlphaZero takes an arbitrary search backend, here
        classic MCTS. Temperature 1.0 samples moves in proportion to visit counts, for
        exploration during self-play. With --sims-schedule, a search for each stage: each
        takes over at its epoch, starting from empty trees sized for it (see
        core.training.schedule)"""
        selfplay_eval_fn = make_nn_eval_fn(network, state_to_nn_input)
        return EvaluatorSchedule(
            [
                (
                    epoch,
                    AlphaZero(MCTS)(
                        eval_fn=selfplay_eval_fn,
                        num_iterations=sims,
                        # counting reused visits, a search's root never has more visits than
                        # this, those of the cycles it goes on for until its move is played included
                        max_nodes=sims + args.min_iterations + args.advance_every
                        if args.count_reused_visits
                        else int(args.tree_factor * sims),
                        branching_factor=env.num_actions,
                        action_selector=PUCTSelector(),
                        temperature=1.0,
                        dirichlet_alpha=args.dirichlet_alpha,
                        dirichlet_epsilon=args.dirichlet_epsilon,
                        persist_tree=not args.no_persist_tree,
                        count_reused_visits=args.count_reused_visits,
                        min_iterations=args.min_iterations,
                        traversal_steps=args.traversal_steps,
                        policy_target=CompletedQ(args.value_scale, args.maxvisit_init)
                        if args.policy_target == "completed-q"
                        else VisitCounts(),
                        playout_cap=args.playout_cap,
                    ),
                )
                for epoch, sims in args.sims_schedule or [(0, args.sims)]
            ]
        )

    # the network's raw moves and values against Edax's, on the engine-labelled set
    engine_set = load_engine_set(parser, args)
    if engine_set is None and args.lr_plateau_metric != "loss":
        parser.error(
            f"--lr-plateau-metric {args.lr_plateau_metric} needs an engine-labelled set"
        )
    if engine_set is None and engine_rule:
        parser.error("--grow's rule needs an engine-labelled set")
    ckpt_dir = args.ckpt_dir or tempfile.mkdtemp(prefix="turbozero-othello-")
    monitor = (
        Monitor(args.monitor, project="othello", name=args.name, notes=args.notes)
        if args.monitor
        else None
    )

    def make_testers(network, results_dir: str) -> list[BaseTester]:
        """A network's testers: the ladder, the engine-labelled set, games against Edax."""
        testers: list[BaseTester] = []
        if args.eval_every > 0:
            # opponents: the greedy tile counter and pgx's pretrained model (others are listed
            # at https://sotets.uk/pgx/api/#pgx.BaselineModelId)
            pretrained = make_nn_eval_fn_no_params_callable(
                pgx.make_baseline_model("othello_v0"), state_to_nn_input
            )
            greedy = make_nn_eval_fn_no_params_callable(greedy_eval, state_to_nn_input)

            # with a monitor, each test sends a game for the run's page to play, its XOT
            # opening's moves before it
            episode_fn = (
                xot_episode_fn(pgx_two_player_episode(), load_xot())
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
                Rung(f"pgx{n}", make_test_evaluator(pretrained, n))
                for n in (1, 4, 16, 64, 256)
            ]
            testers.append(
                LadderTester(
                    num_episodes=128,
                    rungs=rungs,
                    episode_fn=episode_fn,
                    epochs_per_test=args.eval_every,
                )
            )
        if engine_set is not None:
            testers.append(
                ScoredPositionsTester(
                    engine_set.scored_positions(),
                    network,
                    epochs_per_test=args.engine_eval_every,
                    config={
                        "path": str(args.engine_set or DEFAULT_ENGINE_SET),
                        **engine_set.describe(),
                    },
                )
            )
        # games against Edax (or Egaroucid), between epochs, in this process (see vs_engine.py)
        if args.vs_engine_every > 0 or args.vs_engine_last > 0:
            try:
                match = EngineMatch(
                    network,
                    ENGINES[args.vs_engine],
                    args.vs_engine_levels,
                    args.vs_engine_sims,
                    args.vs_engine_openings,
                )
            except ValueError as e:
                parser.error(str(e))
            match_tester = EngineMatchTester(
                match,
                args.epochs,
                every=args.vs_engine_every,
                last=args.vs_engine_last,
                # with a monitor, a game at each level for the run's page
                record=monitor is not None,
                results=Path(results_dir) / "vs_engine.jsonl",
                config={"dtype": args.inference_dtype},
            )
            testers.append(match_tester)
            if monitor is not None:
                describe_scores(monitor, match_tester.score_names())
        return testers

    network, network_state = make_net(args.net)
    evaluator = make_selfplay(network)
    evaluator_test = make_test_evaluator(make_nn_eval_fn(network, state_to_nn_input))
    # with --grow, the bigger networks, each with its own searches and testers
    growth = None
    if args.grow:
        learners = []
        for spec in args.grow:
            grown, grown_state = make_net(spec)
            learners.append(
                Learner(
                    name=net_name(spec),
                    nn=grown,
                    nn_state=grown_state,
                    evaluator=make_selfplay(grown),
                    evaluator_test=make_test_evaluator(
                        make_nn_eval_fn(grown, state_to_nn_input)
                    ),
                    testers=make_testers(grown, os.path.join(ckpt_dir, net_name(spec))),
                )
            )
        try:
            growth = Growth(
                first=net_name(args.net),
                learners=learners,
                metric=args.grow_metric,
                window=args.grow_window,
                min_epochs=args.grow_min_epochs,
                switch_at=args.grow_switch_at or (),
                manual=args.grow_manual,
                margin=args.grow_margin,
                guard=args.grow_guard,
            )
        except ValueError as e:
            parser.error(f"--grow: {e}")
    testers = make_testers(
        network,
        os.path.join(ckpt_dir, net_name(args.net)) if args.grow else ckpt_dir,
    )

    endgame_solve = None
    if args.solve_empties is not None:
        stages = [e for _, e in evaluator.stages]
        endgame_solve = EndgameSolve(
            stop_fn=stops_at(args.solve_empties),
            solve_fn=make_solve_fn(
                args.solver, args.solve_threads or default_threads()
            ),
            slots=solve_slots(
                max(collection_steps * e.search_cycles() for e in stages),
                args.advance_every or min(e.search_cycles() for e in stages),
                args.solve_empties,
                args.solve_play_past,
            ),
            share=args.solve_share,
            play_past=args.solve_play_past,
        )

    # each epoch collects `collection_steps_per_epoch` self-play steps' compute (see
    # --selfplay-cycles) in each of `batch_size` environments, then takes
    # `train_steps_per_epoch` training steps
    trainer = Trainer(
        batch_size=1024,
        train_batch_size=args.train_batch,
        warmup_steps=0,
        collection_steps_per_epoch=collection_steps,
        train_steps_per_epoch=args.train_steps,
        advance_every=args.advance_every,
        grad_accum_steps=args.grad_accum,
        lr_drop=args.lr_drop,
        noise_measure=args.noise_measure,
        fresh_positions=args.fresh_positions,
        nn=network,
        nn_state=network_state,
        loss_fn=partial(az_default_loss_fn, value_target_q=args.value_target_q),
        # decays from --lr to --lr-final over the run, drops at --lr-drops, or on plateaus
        optimizer=make_optimizer(
            args.epochs * args.train_steps,
            args.lr,
            args.lr if args.lr_final is None else args.lr_final,
            args.weight_decay,
            args.lr_drops or (DEFAULT_LR_DROPS if args.lr_final is None else ()),
            lr_plateau=lr_plateau,
            sgd=args.optimizer == "sgd",
            momentum=args.momentum,
            decoupled=args.decoupled_decay,
        ),
        lr_plateau=lr_plateau,
        lr_plateau_metric=args.lr_plateau_metric,
        growth=growth,
        waiting_train_steps=args.grow_train_steps,
        train_retired=args.grow_train_retired,
        averages=args.average_epochs,
        selfplay_average=args.selfplay_average,
        average_warmup=args.average_warmup,
        evaluator=evaluator,
        # stores `capacity` positions for each of the `batch_size` environments
        memory_buffer=EpisodeReplayBuffer(capacity=args.buffer),
        max_episode_steps=80,
        env_step_fn=step_fn,
        env_init_fn=selfplay_init_fn,
        test_env_init_fn=test_init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=testers,
        evaluator_test=evaluator_test,
        # training maps each sample through one of the board's 8 symmetries, at random
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
        monitor=monitor,
        ckpt_dir=ckpt_dir,
        keep_every=args.keep_every,
        # how often self-play games are drawn
        episode_metrics_fn=episode_metrics,
        exploration=Exploration(num_sampling_moves=args.sampling_moves),
        resignation=Resignation(args.resign, args.resign_calibration)
        if args.resign is not None
        else None,
        endgame_solve=endgame_solve,
        save_state_at=args.save_state_at,
        save_state_every=args.save_state_every,
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
                    "fork_into",
                    "warm_start",
                    "branch_from",
                    "take_over",
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
            initial_state = trainer.branch(args.branch_from, take_over=args.take_over)
        except ValueError as e:
            parser.error(f"--branch-from: {e}")
    trainer.train_loop(
        seed=args.seed,
        num_epochs=args.epochs,
        # each tester keeps its own cadence (--eval-every, --engine-eval-every,
        # --vs-engine-every and --vs-engine-last)
        eval_every=1,
        initial_state=initial_state,
        fork_from=args.init_from,
        fork_replay_buffers=not args.init_without_buffers,
        fork_into=args.fork_into,
        warm_start=args.warm_start,
    )
    for spec, net_dir in zip([args.net, *args.grow], trainer.ckpt_dirs, strict=True):
        export_last(net_dir, spec)


if __name__ == "__main__":
    main()
