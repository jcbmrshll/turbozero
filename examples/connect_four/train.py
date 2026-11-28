"""AlphaZero on Connect Four, tracked by how close the raw network's moves and values come to
the truth on fixed positions: the game is solved.

Self-play games are collected in parallel across a batch of environments (and across
every available GPU), with Monte Carlo Tree Search run on each of them by a weight average of
the network (--average-epochs); the network then trains on minibatches sampled from replay
memory.

    examples/connect_four/setup_solver.sh     # once: the solver and its labelled positions
    uv run examples/connect_four/train.py --monitor
    uv run examples/connect_four/train.py --search weighted --monitor

Every --validate-every epochs it scores the average on the solver-labelled set, if there is one
(see solver_set.py and --engine-set), logged as `engine_*`, against the solver's scores,
and `outcome_*`, against how each move's game ends: `outcome_top1_critical`, how often
its first choice keeps the game's result where the move matters, and `engine_value_mse`, its
value's squared error against the true result, are the ones to watch. Games against the solver
(vs_solver.py) measure the saved weights' strength afterwards (see README.md).

With --monitor the run logs to slick's seaglass server ($SEAGLASS_URL, set in a slick task;
--monitor URL for another, e.g. `uv run seaglass serve --dir runs` at http://localhost:8008).

The average's weights go to --ckpt-dir/avg<H>/<epoch>.eqx every --save-weights-every epochs. The
hyperparameters here are only an example; tune them for your task and hardware.

With --save-state-at or --save-state-every, the whole training state (network, optimizer,
replay buffer, games in progress, rng) is saved there too, to continue the run exactly
with --resume, or to fork short runs from with --init-from (see README.md):

    uv run examples/connect_four/train.py --epochs 100 --save-state-at 100 --ckpt-dir runs/trunk
    uv run examples/connect_four/train.py --epochs 30 --init-from runs/trunk --ckpt-dir runs/fork-control
    uv run examples/connect_four/train.py --epochs 100 --save-state-every 10 --ckpt-dir runs/long
    uv run examples/connect_four/train.py --epochs 100 --save-state-every 10 --ckpt-dir runs/long --resume
"""

import argparse
import tempfile
import warnings
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
    state_to_nn_input,
    step_fn,
)
from openings import load_openings, make_opening_init_fn
from solver_set import DEFAULT_PATH as DEFAULT_ENGINE_SET
from solver_set import SolverSet

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.weighted_mcts import WeightedMCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.monitoring import DEFAULT_URL, Monitor
from core.training.exploration import Exploration
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
from core.training.validation import ScoredPositionsValidator

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


def load_engine_set(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> SolverSet | None:
    """The solver-labelled set to score the network on, None if it isn't to be. A missing
    default set is skipped with a warning (build one with setup_solver.sh); a missing
    --engine-set is an error."""
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
    parser.add_argument(
        "--sims", type=int, default=64, help="MCTS iterations per self-play move"
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
        "--lr", type=float, default=1e-3, help="the learning rate, constant"
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
        "--engine-set",
        default=None,
        metavar="PATH",
        help="the solver-labelled set to score the network on (see solver_set.py; default "
        f"{DEFAULT_ENGINE_SET}, skipped with a warning if it doesn't exist)",
    )
    parser.add_argument(
        "--average-epochs",
        type=float,
        default=1.0,
        metavar="H",
        help="self-play plays with, and validation scores, an average of the network's "
        "weights over about H epochs (see the Trainer's average_epochs); 1, the default, is "
        "the network's own weights",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="epochs between scoring the average on the solver-labelled set: a forward pass "
        "over its positions, a fraction of a second",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="where to save checkpoints (default: a new temporary directory)",
    )
    parser.add_argument(
        "--save-weights-every",
        type=int,
        default=1,
        metavar="N",
        help="save the average's weights to --ckpt-dir/avg<H>/<epoch>.eqx every N epochs, "
        "all kept",
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
        "the newest from): it starts at epoch 0, with its own games and seed, from the "
        "saved network, optimizer state (at this run's --lr and --weight-decay), weight "
        "average and replay buffer. Needs the same --net; --buffer may differ (each "
        "environment's newest entries are kept)",
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
    # self-play starts games from the empty board, or as many as asked from openings (see
    # openings.py)
    selfplay_init_fn = (
        init_fn
        if args.standard_starts >= 1
        else make_opening_init_fn(load_openings(), args.standard_starts)
    )
    # the search, for self-play
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
    # samples moves in proportion to visit counts, for exploration during self-play
    evaluator = AlphaZero(search_backend)(
        eval_fn=make_nn_eval_fn(network, state_to_nn_input),
        num_iterations=args.sims,
        max_nodes=int(args.tree_factor * args.sims),
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=1.0,
        dirichlet_alpha=DIRICHLET_ALPHA,
        traversal_steps=args.traversal_steps,
        **search_kwargs,
    )
    # the average's raw moves and values against the truth, on the solver-labelled set: by
    # the solver's scores, in which a slower win gives some up, and by outcomes alone. In
    # float32, so that one checkpoint's numbers compare with the next's
    solver_set = load_engine_set(parser, args)
    validators = (
        [
            ScoredPositionsValidator(
                solver_set.scored_positions(by_outcome),
                make_network(args.net, inference_dtype="float32")[0],
                name=name,
                config={
                    "path": str(args.engine_set or DEFAULT_ENGINE_SET),
                    **solver_set.describe(),
                },
            )
            for name, by_outcome in (("engine", False), ("outcome", True))
        ]
        if solver_set is not None
        else []
    )
    ckpt_dir = args.ckpt_dir or tempfile.mkdtemp(prefix="turbozero-connect-four-")
    monitor = (
        Monitor(args.monitor, project="connect_four", name=args.name, notes=args.notes)
        if args.monitor
        else None
    )

    # each epoch collects `collection_steps_per_epoch` self-play steps in each of
    # `batch_size` environments, then takes `train_steps_per_epoch` training steps
    trainer = Trainer(
        batch_size=SELFPLAY_GAMES,
        train_batch_size=args.train_batch,
        collection_steps_per_epoch=COLLECTION_STEPS,
        train_steps_per_epoch=args.train_steps,
        grad_accum_steps=args.grad_accum,
        nn=network,
        nn_state=network_state,
        loss_fn=az_default_loss_fn,
        optimizer=make_optimizer(args.lr, args.weight_decay),
        evaluator=evaluator,
        # stores `capacity` positions for each of the `batch_size` environments
        memory_buffer=EpisodeReplayBuffer(capacity=args.buffer),
        max_episode_steps=MAX_STEPS,
        env_step_fn=step_fn,
        env_init_fn=selfplay_init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        # training maps each sample through the board's mirror image or leaves it, at random
        data_transform_fns=SYMMETRY_TRANSFORM_FNS,
        monitor=monitor,
        ckpt_dir=ckpt_dir,
        average_epochs=args.average_epochs,
        validators=validators,
        validate_every=args.validate_every,
        save_weights_every=args.save_weights_every,
        save_raw_every=args.save_raw_every,
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
        initial_state=trainer.resume() if args.resume else None,
        fork_from=args.init_from,
        fork_replay_buffer=not args.init_without_buffers,
    )


if __name__ == "__main__":
    main()
