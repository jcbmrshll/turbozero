"""AlphaZero on Hex, at board sizes 7, 9 and 11.

Self-play games are collected in parallel across a batch of environments (and across every
available GPU), with Monte Carlo Tree Search run on each of them, by a weight average of the
network (--average-epochs); the network then trains on minibatches sampled from replay memory.
Self-play games start from the empty board, with the swap rule, or from openings (see
openings.py and --standard-starts).

    uv run --extra cuda examples/hex/train.py --size 11 --monitor

If there's an engine-labelled set (see engine_set.py and --engine-set), every --validate-every
epochs the average is scored on it, logged as `engine_*`. Games against MoHex
(vs_engine.py) and pgx's pretrained model (eval_pgx.py) evaluate saved weights afterwards (see
README.md).

With --monitor the run logs to slick's seaglass server ($SEAGLASS_URL, set in a slick task;
--monitor URL for another, e.g. `uv run seaglass serve --dir runs` at http://localhost:8008).

The first epoch is slow: nearly all of the training loop is JIT-compiled the first time it runs.
The average's weights go to --ckpt-dir/avg<H>/<epoch>.eqx every --save-weights-every epochs.
With --save-state-at or --save-state-every, the whole training state (network, optimizer, replay
buffer, games in progress, rng) is saved there too, to continue the run exactly with --resume, or
to fork short runs from with --init-from (see README.md).
"""

import argparse
import tempfile
import warnings
from pathlib import Path

from engine_set import EngineSet, default_path
from game import (
    SIZES,
    TRAVERSAL_STEPS,
    WEIGHT_DECAY,
    HexGame,
    make_optimizer,
    state_to_nn_input,
)
from openings import load_openings, make_opening_init_fn

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.monitoring import DEFAULT_URL, Monitor
from core.training.exploration import Exploration
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
from core.training.validation import ScoredPositionsValidator


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
    parser: argparse.ArgumentParser, args: argparse.Namespace, size: int
) -> EngineSet | None:
    """The engine-labelled set to score the network on, None if it isn't to be. A missing
    default set is skipped with a warning (build one with label_positions.py); a missing
    --engine-set is an error."""
    if args.engine_set is not None:
        if not Path(args.engine_set).exists():
            parser.error(f"--engine-set {args.engine_set}: no such file")
        return EngineSet.load(args.engine_set)
    if not default_path(size).exists():
        warnings.warn(
            f"no engine-labelled set at {default_path(size)}, so the network won't be scored "
            "against MoHex: build one with label_positions.py, or pass --engine-set",
            stacklevel=2,
        )
        return None
    return EngineSet.load(default_path(size))


def main():
    parser = argparse.ArgumentParser(description="AlphaZero on Hex.")
    parser.add_argument(
        "--size", type=int, choices=sorted(SIZES), default=11, help="the board size"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epochs in the run (default: the size's, see game.SIZES)",
    )
    parser.add_argument(
        "--net",
        default=None,
        help="the network, by default the size's (see game.SIZES): KIND:DEPTHxWIDTH[,FIELD=VALUE...], e.g. resnet:10x128 for "
        "AlphaZero's ResNet with 10 blocks of 128 channels, resnet:10x128,pool_every=3,"
        "heads=pooled with KataGo's global pooling, or transformer:8x256 (see "
        "core.networks.spec.network_config)",
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
        "moves of a game, then plays the most-visited move, as AlphaZero (30) and "
        "OLIVAW (20) did. N counts from the empty board, so it includes an opening's "
        "moves. By default every move is sampled",
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
        default=375,
        help="replay memory: positions kept per environment (375 is about 3 epochs of "
        "self-play). Training maps each position it samples through a random symmetry",
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
        help="fraction of self-play games from the empty board, with the swap rule; the rest "
        "start from the openings test games start from (see openings.py)",
    )
    parser.add_argument(
        "--engine-set",
        default=None,
        metavar="PATH",
        help="the engine-labelled set to score the network on (see engine_set.py; default "
        f"{default_path(0).parent}/mohex-<size>.npz, skipped with a warning if it doesn't exist)",
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
        help="epochs between scoring the average on the engine-labelled set: a forward "
        "pass over its positions, a fraction of a second",
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
    game = HexGame(args.size)
    defaults = SIZES[args.size]
    args.net = args.net or defaults.net
    args.epochs = args.epochs or defaults.epochs
    if args.resume and args.ckpt_dir is None:
        parser.error("--resume needs the run's --ckpt-dir")
    if args.init_without_buffers and args.init_from is None:
        parser.error("--init-without-buffers needs --init-from")

    # self-play starts games from the empty board by default (see openings.py)
    selfplay_init_fn = make_opening_init_fn(
        game, load_openings(game), args.standard_starts
    )

    # the network (see game.py), with its state (BatchNorm's, if it has any)
    network, network_state = game.make_network(
        args.net,
        seed=args.seed,
        inference_dtype=args.inference_dtype,
        train_dtype=args.train_dtype,
        remat=args.remat,
    )

    # AlphaZero takes an arbitrary search backend, here classic MCTS. Temperature 1.0
    # samples moves in proportion to visit counts, for exploration during self-play
    evaluator = AlphaZero(MCTS)(
        eval_fn=make_nn_eval_fn(network, state_to_nn_input),
        num_iterations=args.sims,
        max_nodes=int(args.tree_factor * args.sims),
        branching_factor=game.env.num_actions,
        action_selector=PUCTSelector(),
        temperature=1.0,
        traversal_steps=args.traversal_steps,
    )
    # the average's raw moves and values against MoHex's, on the engine-labelled set, in
    # float32, so that one checkpoint's numbers compare with the next's
    validators = []
    engine_set = load_engine_set(parser, args, args.size)
    if engine_set is not None:
        validators.append(
            ScoredPositionsValidator(
                engine_set.scored_positions(),
                game.make_network(args.net, inference_dtype="float32")[0],
                config={
                    "path": str(args.engine_set or default_path(args.size)),
                    **engine_set.describe(),
                },
            )
        )
    ckpt_dir = args.ckpt_dir or tempfile.mkdtemp(prefix=f"turbozero-hex{args.size}-")
    monitor = (
        Monitor(args.monitor, project="hex", name=args.name, notes=args.notes)
        if args.monitor
        else None
    )

    # each epoch collects `collection_steps_per_epoch` self-play steps in each of
    # `batch_size` environments, then takes `train_steps_per_epoch` training steps
    trainer = Trainer(
        batch_size=1024,
        train_batch_size=args.train_batch,
        collection_steps_per_epoch=128,
        train_steps_per_epoch=args.train_steps,
        grad_accum_steps=args.grad_accum,
        nn=network,
        nn_state=network_state,
        loss_fn=az_default_loss_fn,
        optimizer=make_optimizer(args.lr, args.weight_decay),
        evaluator=evaluator,
        # stores `capacity` positions for each of the `batch_size` environments
        memory_buffer=EpisodeReplayBuffer(capacity=args.buffer),
        max_episode_steps=game.max_moves,
        env_step_fn=game.step_fn,
        env_init_fn=selfplay_init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        # training maps each sample through one of the board's 4 symmetries, at random
        data_transform_fns=game.symmetry_transform_fns,
        monitor=monitor,
        ckpt_dir=ckpt_dir,
        average_epochs=args.average_epochs,
        validators=validators,
        validate_every=args.validate_every,
        save_weights_every=args.save_weights_every,
        save_raw_every=args.save_raw_every,
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
