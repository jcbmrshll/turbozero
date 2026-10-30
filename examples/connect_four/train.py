"""AlphaZero on Connect Four, tracked by how close the raw network's moves and values come to
the truth on fixed positions: the game is solved.

Self-play games are collected in parallel across a batch of environments (and across
every available GPU), with Monte Carlo Tree Search run on each of them; the network
then trains on minibatches sampled from replay memory.

    examples/connect_four/setup_solver.sh     # once: the solver and its labelled positions
    uv run examples/connect_four/train.py --monitor
    uv run examples/connect_four/train.py --search weighted --monitor

Every epoch it scores the network on the solver-labelled set, if there is one (see
solver_set.py and --engine-set), logged as `engine_*`, against the solver's scores, and
`outcome_*`, against how each move's game ends: `outcome_top1_critical`, how often its first
choice keeps the game's result where the move matters, and `engine_value_mse`, its value's
squared error against the true result, are the ones to watch. Games against the solver measure
strength: --vs-engine-every N plays them after every N epochs, and --vs-engine-last K after each
of the last K epochs, logging the mean of those too (by default 512 games at each of levels 0,
4, 8 and 42, searching 256 iterations a move; see vs_solver.py and README.md):

    uv run examples/connect_four/train.py --epochs 50 --monitor --vs-engine-last 3

Each level's score is logged at the epoch of the network that played, as
`solver_level42_sims256` (with `_se`, `_win`, `_draw` and `_loss`), the mean as
`solver_level42_sims256_last3`, and the time the games took as `solver_seconds`, to take out of
the run's wall time. --eval-every N also plays test games against a random player every N epochs,
logged as `random_avg_outcome`, `random_win_rate` and `random_loss_rate`.

With --monitor the run logs to slick's seaglass server ($SEAGLASS_URL, set in a slick task;
--monitor URL for another, e.g. `uv run seaglass serve --dir runs` at http://localhost:8008).
The run's page shows the metrics, and games against the solver and the random player, which it
draws in the browser with what each move's search explored.

Checkpoints go to --ckpt-dir, with the results of games against the solver in
`vs_solver.jsonl`; `vs_solver.py` evaluates checkpoints further (see README.md). The
hyperparameters here are only an example; tune them for your task and hardware.

With --save-state-at or --save-state-every, the whole training state (network, optimizer,
replay buffers, games in progress, rng) is saved there too, to continue the run exactly
with --resume, or to fork short runs from with --init-from (see README.md):

    uv run examples/connect_four/train.py --epochs 100 --lr-final 1e-3 --save-state-at 100 --ckpt-dir runs/trunk
    uv run examples/connect_four/train.py --epochs 30 --init-from runs/trunk --ckpt-dir runs/fork-control
    uv run examples/connect_four/train.py --epochs 100 --save-state-every 10 --ckpt-dir runs/long
    uv run examples/connect_four/train.py --epochs 100 --save-state-every 10 --ckpt-dir runs/long --resume
"""

import argparse
import tempfile
import warnings
from functools import partial
from pathlib import Path

from game import (
    DEFAULT_NET,
    DIRICHLET_ALPHA,
    MAX_STEPS,
    SYMMETRY_TRANSFORM_FNS,
    TRAVERSAL_STEPS,
    WEIGHT_DECAY,
    env,
    episode_metrics,
    init_fn,
    make_network,
    make_optimizer,
    make_test_evaluator,
    state_to_nn_input,
    step_fn,
)
from openings import load_openings, make_opening_init_fn
from solver import PERFECT_LEVEL, Solver
from solver_set import DEFAULT_PATH as DEFAULT_ENGINE_SET
from solver_set import SolverSet
from vs_solver import SolverMatch, SolverMatchTester, describe_scores

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.policy_targets import CompletedQ, VisitCounts
from core.evaluators.mcts.weighted_mcts import WeightedMCTS
from core.evaluators.random_evaluator import RandomEvaluator
from core.memory.replay_memory import EpisodeReplayBuffer
from core.monitoring import DEFAULT_URL, Monitor
from core.monitoring.episodes import pgx_two_player_episode
from core.testing.scored_positions import ScoredPositionsTester
from core.testing.tester import BaseTester
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.training.exploration import Exploration
from core.training.loss_fns import az_default_loss_fn
from core.training.schedule import EvaluatorSchedule, parse_schedule
from core.training.train import Trainer
from core.training.tree_positions import TreePositions

# self-play: games at once, moves each plays an epoch
SELFPLAY_GAMES = 1024
COLLECTION_STEPS = 32


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


def load_engine_set(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> SolverSet | None:
    """The solver-labelled set to score the network on, None if it isn't to be. A missing
    default set is skipped with a warning (build one with setup_solver.sh); a missing
    --engine-set is an error."""
    if args.engine_eval_every <= 0:
        return None
    if args.engine_set is not None:
        if not Path(args.engine_set).exists():
            parser.error(f"--engine-set {args.engine_set}: no such file")
        return SolverSet.load(args.engine_set)
    if not DEFAULT_ENGINE_SET.exists():
        warnings.warn(
            f"no solver-labelled set at {DEFAULT_ENGINE_SET}, so the network won't be scored "
            "against the solver: build one with setup_solver.sh, or pass --engine-set",
            stacklevel=2,
        )
        return None
    return SolverSet.load(DEFAULT_ENGINE_SET)


def main():
    parser = argparse.ArgumentParser(description="AlphaZero on Connect Four.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--net",
        default=DEFAULT_NET,
        help="the network: KIND:DEPTHxWIDTH[,FIELD=VALUE...], e.g. resnet:4x64 for "
        "AlphaZero's ResNet with 4 blocks of 64 channels, resnet:4x64,pool_every=2 with "
        "KataGo's global pooling, or transformer:4x64 (see "
        "core.networks.spec.network_config)",
    )
    parser.add_argument(
        "--search",
        choices=["mcts", "weighted"],
        default="mcts",
        help="the search: AlphaZero's PUCT MCTS, or weighted MCTS, which backs up a "
        "softmax-weighted sum of child Q-values instead of a plain average "
        "(https://twitter.com/ptrschmdtnlsn/status/1748800529608888362)",
    )
    parser.add_argument(
        "--q-temperature",
        type=float,
        default=1.0,
        help="with --search weighted, the temperature applied to child Q-values when backing "
        "them up",
    )
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
        "--sampling-moves",
        type=int,
        default=None,
        metavar="N",
        help="self-play samples moves in proportion to visit counts for the first N "
        "moves of a game, then plays the most-visited move, as AlphaZero (30) did. N counts "
        "from the empty board. By default every move is sampled",
    )
    parser.add_argument(
        "--train-batch",
        type=int,
        default=4096,
        help="samples per training step; if training runs out of memory with a big "
        "network, see --remat and --grad-accum, which keep the batch",
    )
    parser.add_argument(
        "--train-steps", type=int, default=32, help="training steps per epoch"
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
        default=256,
        help="replay memory: positions kept per environment (256 is 8 epochs of self-play). "
        "Training maps each position it samples through a random symmetry",
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
        default=1.0,
        help="fraction of self-play games from the empty board; the rest start from the "
        "openings of openings.py. Sampling moves in self-play keeps its games varied, so by "
        "default all of them do",
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
        default=10,
        help="epochs between test games against a random player (0 for none: the "
        "solver-labelled set tracks the run, see --engine-set)",
    )
    parser.add_argument(
        "--engine-set",
        default=None,
        metavar="PATH",
        help="the solver-labelled set to score the network on (see solver_set.py; default "
        f"{DEFAULT_ENGINE_SET}, skipped with a warning if it doesn't exist)",
    )
    parser.add_argument(
        "--engine-eval-every",
        type=int,
        default=1,
        help="epochs between scoring the network on the solver-labelled set (0 for never). "
        "A forward pass over its positions: a fraction of a second",
    )
    parser.add_argument(
        "--vs-engine-every",
        type=int,
        default=10,
        metavar="N",
        help="play games against the solver after every N epochs, between epochs (0 for "
        "never), and log them to the run at the epoch of the network that played (see "
        "vs_solver.py). Skipped with a warning if the solver isn't installed",
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
        "--vs-engine-levels",
        type=int,
        nargs="+",
        default=[0, 4, 8, PERFECT_LEVEL],
        help="the solver's levels to play at: moves it looks ahead, 0 for a random player, "
        f"{PERFECT_LEVEL} for a perfect one, against which a perfect player scores 0.5",
    )
    parser.add_argument(
        "--vs-engine-sims",
        type=int,
        default=256,
        help="our MCTS iterations a move in games against the solver",
    )
    parser.add_argument(
        "--vs-engine-openings",
        type=int,
        default=256,
        help="openings a level in games against the solver, each played with both "
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
    if args.init_without_buffers and args.init_from is None:
        parser.error("--init-without-buffers needs --init-from")
    if args.no_persist_tree and args.count_reused_visits:
        parser.error(
            "--count-reused-visits counts the visits of the subtree a search keeps, which "
            "--no-persist-tree discards"
        )
    if args.advance_every is None and args.count_reused_visits:
        # searches of varying lengths: see Trainer's advance_every
        args.advance_every = 5
    if args.vs_engine_last > args.epochs:
        parser.error("--vs-engine-last can't be more than --epochs")
    if args.policy_target == "completed-q" and args.tree_positions > 0:
        # a tree position's target is its visits: the network's policy before the
        # search isn't kept below the root
        parser.error("--policy-target completed-q can't go with --tree-positions")

    # self-play starts games from the empty board, or as many as asked from openings (see
    # openings.py)
    selfplay_init_fn = (
        init_fn
        if args.standard_starts >= 1
        else make_opening_init_fn(load_openings(), args.standard_starts)
    )
    # the search, for self-play and our agent's test games
    if args.search == "weighted":
        search_backend, search_kwargs = (
            WeightedMCTS,
            {"q_temperature": args.q_temperature},
        )
    else:
        search_backend, search_kwargs = MCTS, {}

    # the network (see game.py), with its state (BatchNorm's, if it has any)
    network, network_state = make_network(
        args.net,
        seed=args.seed,
        inference_dtype=args.inference_dtype,
        train_dtype=args.train_dtype,
        remat=args.remat,
    )

    # AlphaZero takes an arbitrary search backend, by default classic MCTS. Temperature 1.0
    # samples moves in proportion to visit counts, for exploration during self-play.
    # With --sims-schedule, a search for each stage: each takes over at its epoch,
    # starting from empty trees sized for it (see core.training.schedule)
    selfplay_eval_fn = make_nn_eval_fn(network, state_to_nn_input)
    evaluator = EvaluatorSchedule(
        [
            (
                epoch,
                AlphaZero(search_backend)(
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
                    dirichlet_alpha=DIRICHLET_ALPHA,
                    persist_tree=not args.no_persist_tree,
                    count_reused_visits=args.count_reused_visits,
                    min_iterations=args.min_iterations,
                    traversal_steps=args.traversal_steps,
                    policy_target=CompletedQ(args.value_scale, args.maxvisit_init)
                    if args.policy_target == "completed-q"
                    else VisitCounts(),
                    **search_kwargs,
                ),
            )
            for epoch, sims in args.sims_schedule or [(0, args.sims)]
        ]
    )
    evaluator_test = make_test_evaluator(
        make_nn_eval_fn(network, state_to_nn_input),
        search=search_backend,
        **search_kwargs,
    )

    testers: list[BaseTester] = []
    if args.eval_every > 0:
        testers.append(
            TwoPlayerBaseline(
                num_episodes=128,
                baseline_evaluator=RandomEvaluator(),
                balance_first_player=True,
                name="random",
                # with a monitor, each test sends a game for the run's page to play
                episode_fn=pgx_two_player_episode() if args.monitor else None,
                epochs_per_test=args.eval_every,
            )
        )
    # the network's raw moves and values against the truth, on the solver-labelled set: by the
    # solver's scores, in which a slower win gives some up, and by outcomes alone
    solver_set = load_engine_set(parser, args)
    if solver_set is not None:
        testers += [
            ScoredPositionsTester(
                solver_set.scored_positions(by_outcome),
                network,
                epochs_per_test=args.engine_eval_every,
                name=name,
                config={
                    "path": str(args.engine_set or DEFAULT_ENGINE_SET),
                    **solver_set.describe(),
                },
            )
            for name, by_outcome in (("engine", False), ("outcome", True))
        ]
    ckpt_dir = args.ckpt_dir or tempfile.mkdtemp(prefix="turbozero-connect-four-")
    monitor = (
        Monitor(args.monitor, project="connect_four", name=args.name, notes=args.notes)
        if args.monitor
        else None
    )
    # games against the solver, between epochs, in this process (see vs_solver.py)
    if args.vs_engine_every > 0 or args.vs_engine_last > 0:
        try:
            solver = Solver()
        except ValueError as e:
            warnings.warn(f"no games against the solver: {e}", stacklevel=2)
        else:
            match = SolverMatch(
                network,
                solver,
                args.vs_engine_levels,
                args.vs_engine_sims,
                args.vs_engine_openings,
                search_backend=search_backend,
                **search_kwargs,
            )
            match_tester = SolverMatchTester(
                match,
                args.epochs,
                every=args.vs_engine_every,
                last=args.vs_engine_last,
                # with a monitor, a game at each level for the run's page
                record=monitor is not None,
                results=Path(ckpt_dir) / "vs_solver.jsonl",
                config={"dtype": args.inference_dtype},
            )
            testers.append(match_tester)
            if monitor is not None:
                describe_scores(monitor, match_tester.score_names())

    # each epoch collects `collection_steps_per_epoch` self-play steps in each of
    # `batch_size` environments, then takes `train_steps_per_epoch` training steps
    trainer = Trainer(
        batch_size=SELFPLAY_GAMES,
        train_batch_size=args.train_batch,
        warmup_steps=0,
        collection_steps_per_epoch=COLLECTION_STEPS,
        train_steps_per_epoch=args.train_steps,
        advance_every=args.advance_every,
        grad_accum_steps=args.grad_accum,
        nn=network,
        nn_state=network_state,
        loss_fn=partial(az_default_loss_fn, value_target_q=args.value_target_q),
        # decays from --lr to --lr-final over the run
        optimizer=make_optimizer(
            args.epochs * args.train_steps, args.lr, args.lr_final, args.weight_decay
        ),
        evaluator=evaluator,
        # stores `capacity` positions for each of the `batch_size` environments
        memory_buffer=EpisodeReplayBuffer(capacity=args.buffer),
        max_episode_steps=MAX_STEPS,
        env_step_fn=step_fn,
        env_init_fn=selfplay_init_fn,
        test_env_init_fn=init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=testers,
        evaluator_test=evaluator_test,
        # training maps each sample through the board's mirror image or leaves it, at random
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
                if k not in ("resume", "init_from", "init_without_buffers", "notes")
            }
        },
    )
    trainer.train_loop(
        seed=args.seed,
        num_epochs=args.epochs,
        # each tester keeps its own cadence (--eval-every, --engine-eval-every,
        # --vs-engine-every and --vs-engine-last)
        eval_every=1,
        initial_state=trainer.resume() if args.resume else None,
        fork_from=args.init_from,
        fork_replay_buffers=not args.init_without_buffers,
    )


if __name__ == "__main__":
    main()
