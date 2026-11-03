"""Trains a network on another run's games, with no self-play: supervised on the search policies
in a saved training state's replay buffer (see `Trainer.save_state`), and on a value target of the
game's outcome, the search's value stored with each position, the value a teacher network gives it
(see the `values` command), or a mix of the outcome and one of those. Or on a labelled corpus of the
league's games instead (see league_corpus.py, and --corpus): on its teacher's policies, and its
teacher's values or the outcome, measured on the corpus's held-out openings.

    uv run examples/othello/distill.py values --teacher 37.eqx --teacher-net NET \\
        --data state-188.npz state-3.npz --out teacher-values
    uv run examples/othello/distill.py train --data state-188.npz --heldout state-3.npz \\
        --net resnet:4x64,pool_every=2,heads=pooled,pool_channels=16 \\
        --value-source teacher --value-target-q 1 --teacher-values teacher-values \\
        --ckpt-dir runs/student --monitor --notes "does a student learn from its teacher?"
    uv run examples/othello/distill.py train --corpus corpus --passes 4 --eval-every 1 \\
        --net resnet:4x64,pool_every=2,heads=pooled,pool_channels=16 \\
        --value-source teacher --value-target-q 1 --ckpt-dir runs/pretrained

Its checkpoints load with `game.load_checkpoint`, like train.py's.
"""

import argparse
import dataclasses
import json
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from game import (
    SYMMETRY_TRANSFORM_FNS,
    WEIGHT_DECAY,
    export_last,
    load_checkpoint,
    make_network,
    make_optimizer,
)
from league_corpus import Corpus

from core.memory.replay_memory import BaseExperience, transform_randomly
from core.monitoring import DEFAULT_URL, Monitor
from core.networks.utils import apply_nn
from core.training.loss_fns import az_default_loss_fn

# positions evaluated at once when measuring a network, not training it
EVAL_BATCH = 4096


def load_positions(path: str) -> BaseExperience:
    """The positions of a saved training state's replay buffer that can be sampled, in a fixed
    order, as host arrays with one leading dimension."""
    with np.load(path) as saved:
        meta = json.loads(str(saved["meta"]))
        index = {
            leaf[0]: i for i, leaf in enumerate(meta["leaves"]["collection_state"])
        }

        def leaf(name: str) -> np.ndarray:
            return saved[f"collection_state.{index['.buffer_state' + name]}"]

        valid = leaf(".populated") & leaf(".has_reward")
        fields: dict[str, Any] = {
            field.name: leaf(f".buffer.{field.name}")[valid]
            for field in dataclasses.fields(BaseExperience)
        }
        return BaseExperience(**fields)


def outcomes(positions: BaseExperience) -> np.ndarray:
    """Each position's game outcome for the player to move, z."""
    return np.asarray(positions.reward)[
        np.arange(len(positions.reward)), positions.cur_player_id
    ]


@eqx.filter_jit
def _evaluate(net, nn_state, batch: BaseExperience) -> tuple[jax.Array, jax.Array]:
    """Each position's policy KL, scored as in `az_default_loss_fn`, and value."""
    (logits, value), _ = apply_nn(net, nn_state, batch.observation_nn)
    logits = jnp.where(batch.policy_mask, logits, jnp.finfo(jnp.float32).min)
    targets = batch.policy_weights.astype(jnp.float32)
    entropy = -jnp.where(targets > 0, targets * jnp.log(targets), 0.0).sum(-1)
    return optax.softmax_cross_entropy(logits, targets) - entropy, value.reshape(-1)


def measure(
    network, params, positions: BaseExperience, teacher_values: np.ndarray | None = None
) -> tuple[dict[str, float], np.ndarray]:
    """How a network does on positions, in inference mode: the KL divergence from their search
    policies to its policy, and its value's MSE against the outcome and (if given) the teacher's
    values.

    Returns:
        (metrics, the network's value of each position)
    """
    nn_params, nn_state = params
    net = eqx.nn.inference_mode(eqx.combine(nn_params, network))
    kls, values = [], []
    for start in range(0, len(positions.reward), EVAL_BATCH):
        indices = np.arange(start, min(start + EVAL_BATCH, len(positions.reward)))
        kl, value = _evaluate(
            net,
            nn_state,
            jax.tree.map(lambda x, i=indices: jnp.asarray(x[i]), positions),
        )
        kls.append(np.asarray(kl))
        values.append(np.asarray(value))
    value = np.concatenate(values)
    metrics = {
        "policy_kl": float(np.concatenate(kls).mean()),
        "value_mse_z": float(((value - outcomes(positions)) ** 2).mean()),
    }
    if teacher_values is not None:
        metrics["value_mse_teacher"] = float(((value - teacher_values) ** 2).mean())
    return metrics, value


def with_value_source(
    positions: BaseExperience, source: str, teacher_values: np.ndarray | None
) -> BaseExperience:
    """`positions` with the value `az_default_loss_fn` mixes into its target (`search_value`) taken
    from `source`: "root", the value stored with the position, or "teacher"."""
    if source == "root":
        return positions
    if teacher_values is None:
        raise ValueError("the teacher value source needs teacher values")
    return replace(positions, search_value=teacher_values)


def make_train_step(optimizer, value_target_q: float):
    """One supervised step on a batch, under a random symmetry for each position; the loss is
    self-play training's (see `az_default_loss_fn`), so the steps differ only in their data."""
    loss_fn = partial(az_default_loss_fn, value_target_q=value_target_q)

    @eqx.filter_jit
    def step(params, static, nn_state, opt_state, positions, indices, key):
        batch = jax.tree.map(lambda x: x[indices], positions)
        # a corpus keeps its policies in float16
        batch = replace(batch, policy_weights=batch.policy_weights.astype(jnp.float32))
        batch = transform_randomly(key, batch, SYMMETRY_TRANSFORM_FNS)
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
        (_, (metrics, nn_state)), grads = grad_fn(
            eqx.combine(params, static), nn_state, batch
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), nn_state, opt_state, metrics

    return step


def train(
    network,
    nn_state,
    positions: BaseExperience,
    heldout: BaseExperience,
    heldout_teacher: np.ndarray | None,
    *,
    passes: int,
    batch_size: int,
    lr: float,
    lr_final: float,
    weight_decay: float,
    value_target_q: float,
    seed: int,
    eval_every: int,
    train_probe_size: int,
    ckpt_dir: Path | None,
    log=print,
):
    """Trains `network` for `passes` passes over `positions`, with self-play training's optimizer.
    Each `eval_every` passes, and after the last, it measures the network on `heldout` and on a
    probe of `train_probe_size` of the positions it trains on, whose gap to `heldout` shows
    memorisation, and saves a checkpoint to `ckpt_dir`.

    Returns:
        (params, nn_state): the trained parameters and network state
    """
    steps_per_pass = len(positions.reward) // batch_size
    if steps_per_pass < 1:
        raise ValueError(
            f"{len(positions.reward)} positions are too few for batches of {batch_size}"
        )
    optimizer = make_optimizer(passes * steps_per_pass, lr, lr_final, weight_decay)
    params, static = eqx.partition(network, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    step = make_train_step(optimizer, value_target_q)
    on_device = jax.tree.map(jnp.asarray, positions)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    probe_size = min(train_probe_size, len(positions.reward))
    probe_indices = rng.choice(len(positions.reward), probe_size, replace=False)
    probe = jax.tree.map(lambda x: x[probe_indices], positions)
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train_metrics: dict[str, jax.Array] = {}
    for done in range(passes):
        order = rng.permutation(len(positions.reward))
        for i in range(steps_per_pass):
            key, step_key = jax.random.split(key)
            indices = jnp.asarray(order[i * batch_size : (i + 1) * batch_size])
            params, nn_state, opt_state, train_metrics = step(
                params, static, nn_state, opt_state, on_device, indices, step_key
            )
        finished = done + 1
        if finished % eval_every and finished != passes:
            continue
        model = (params, nn_state)
        metrics = {
            f"heldout_{name}": value
            for name, value in measure(network, model, heldout, heldout_teacher)[
                0
            ].items()
        } | {
            f"probe_{name}": value
            for name, value in measure(network, model, probe)[0].items()
        }
        metrics |= {
            f"train_{name}": float(value) for name, value in train_metrics.items()
        }
        metrics["minutes"] = (time.time() - started) / 60
        log(finished, metrics)
        if ckpt_dir is not None:
            eqx.tree_serialise_leaves(ckpt_dir / f"{finished}.eqx", model)
    return params, nn_state


def values_command(args) -> None:
    """Computes a teacher's value of each state's positions, into `args.out`."""
    network, params = load_checkpoint(args.teacher, args.teacher_net)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in args.data:
        positions = load_positions(path)
        metrics, values = measure(network, params, positions)
        np.save(teacher_values_path(out, path), values)
        z = outcomes(positions)
        print(
            json.dumps(
                {
                    "state": path,
                    "positions": len(z),
                    **{f"teacher_{name}": value for name, value in metrics.items()},
                    "root_value_mse_z": float(
                        ((positions.search_value - z) ** 2).mean()
                    ),
                    "teacher_mse_root": float(
                        ((values - positions.search_value) ** 2).mean()
                    ),
                }
            ),
            flush=True,
        )


def teacher_values_path(directory: Path, state: str) -> Path:
    """Where the `values` command keeps a teacher's values of a state's positions: named by the
    state's run directory and file, so that one directory can hold several runs' states."""
    state_path = Path(state)
    return directory / f"{state_path.parent.name}-{state_path.stem}.npy"


def state_positions(args):
    """--data's positions and --heldout's (--heldout-size of them), with their teacher's values
    if --teacher-values has them."""
    positions = load_positions(args.data)
    heldout = load_positions(args.heldout)
    if args.heldout_size < len(heldout.reward):
        chosen = np.random.default_rng(0).choice(
            len(heldout.reward), args.heldout_size, replace=False
        )
        heldout = jax.tree.map(lambda x: x[chosen], heldout)
    else:
        chosen = np.arange(len(heldout.reward))
    teacher_values = heldout_teacher = None
    if args.teacher_values is not None:
        directory = Path(args.teacher_values)
        teacher_values = np.load(teacher_values_path(directory, args.data))
        heldout_teacher = np.load(teacher_values_path(directory, args.heldout))[chosen]
    return positions, heldout, teacher_values, heldout_teacher


def corpus_positions(
    corpus: Corpus, heldout_size: int
) -> tuple[BaseExperience, BaseExperience]:
    """A labelled corpus's training split, and `heldout_size` of its held-out positions."""
    heldout = corpus.split("heldout")
    if heldout_size < len(heldout):
        heldout = np.random.default_rng(0).choice(heldout, heldout_size, replace=False)
    return corpus.experiences(corpus.split("train")), corpus.experiences(heldout)


def train_command(args) -> None:
    if args.corpus is not None:
        if args.value_source == "root":
            raise SystemExit(
                "a corpus has no search values: pass --value-source teacher"
            )
        positions, heldout = corpus_positions(Corpus(args.corpus), args.heldout_size)
        teacher_values = np.asarray(positions.search_value)
        heldout_teacher = np.asarray(heldout.search_value)
    elif args.data is None or args.heldout is None:
        raise SystemExit("pass --corpus, or --data and --heldout")
    else:
        positions, heldout, teacher_values, heldout_teacher = state_positions(args)
    positions = with_value_source(positions, args.value_source, teacher_values)
    network, nn_state = make_network(
        args.net,
        seed=args.seed,
        inference_dtype="float32",
        train_dtype=args.train_dtype,
    )
    monitor = (
        Monitor(args.monitor, project="othello", name=args.name, notes=args.notes)
        if args.monitor
        else None
    )
    if monitor is not None:
        monitor.start(config={"args": vars(args)}, total=args.passes)

    def log(finished: int, metrics: dict[str, float]) -> None:
        print(f"pass {finished}: {json.dumps(metrics)}", flush=True)
        if monitor is not None:
            monitor.log(finished, metrics)

    try:
        train(
            network,
            nn_state,
            positions,
            heldout,
            heldout_teacher,
            passes=args.passes,
            batch_size=args.batch,
            lr=args.lr,
            lr_final=args.lr_final,
            weight_decay=args.weight_decay,
            value_target_q=args.value_target_q,
            seed=args.seed,
            eval_every=args.eval_every,
            train_probe_size=args.heldout_size,
            ckpt_dir=Path(args.ckpt_dir) if args.ckpt_dir else None,
            log=log,
        )
        if args.ckpt_dir:
            export_last(args.ckpt_dir, args.net)
    except BaseException as e:
        if monitor is not None:
            monitor.finish("stopped" if isinstance(e, KeyboardInterrupt) else "crashed")
        raise
    if monitor is not None:
        monitor.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Trains a network on the games in a saved training state."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    values = commands.add_parser(
        "values", help="compute a teacher's value of positions"
    )
    values.add_argument("--teacher", required=True, help="the teacher's checkpoint")
    values.add_argument("--teacher-net", required=True, help="the teacher's --net")
    values.add_argument(
        "--data", nargs="+", required=True, help="saved training states"
    )
    values.add_argument("--out", required=True, help="directory to write the values to")
    values.set_defaults(run=values_command)

    fit = commands.add_parser("train", help="train a network on a state's positions")
    data = fit.add_mutually_exclusive_group(required=True)
    data.add_argument("--data", help="the saved training state to learn from")
    data.add_argument(
        "--corpus",
        help="a labelled corpus to learn from instead (see league_corpus.py), its training "
        "split, measured on its held-out one",
    )
    fit.add_argument(
        "--heldout",
        help="with --data, a saved training state of games the network and teacher never "
        "trained on",
    )
    fit.add_argument("--heldout-size", type=int, default=100_000)
    fit.add_argument("--net", required=True, help="the network, as train.py's --net")
    fit.add_argument(
        "--value-target-q",
        type=float,
        default=0.0,
        help="weight of --value-source in the value target, the rest being the game's outcome",
    )
    fit.add_argument("--value-source", choices=("root", "teacher"), default="root")
    fit.add_argument(
        "--teacher-values",
        default=None,
        help="the `values` command's directory: needed with --value-source teacher, else "
        "just reported against",
    )
    fit.add_argument("--passes", type=int, default=32, help="passes over the positions")
    fit.add_argument("--batch", type=int, default=4096, help="samples per step")
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--lr-final", type=float, default=1e-4)
    fit.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    fit.add_argument("--train-dtype", default="bfloat16")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--eval-every", type=int, default=2, help="in passes")
    fit.add_argument("--ckpt-dir", default=None)
    fit.add_argument("--name", default=None, help="the run's name on the server")
    fit.add_argument("--notes", default=None, help="what the run tests and why")
    fit.add_argument(
        "--monitor",
        nargs="?",
        const=DEFAULT_URL,
        default=None,
        metavar="URL",
        help=f"log to a seaglass server (default {DEFAULT_URL})",
    )
    fit.set_defaults(run=train_command)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
