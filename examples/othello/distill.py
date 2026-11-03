"""Trains a network on other runs' games, with no self-play: supervised on the search policies in
saved training states' replay buffers (see `Trainer.save_state`) or the positions a run kept with
train.py's --save-positions-from, and on a value target of the game's outcome, the search's value
stored with each position, the value a teacher network gives it (see the `values` command), or a
mix of the outcome and one of those.

    uv run examples/othello/distill.py values --teacher 37.eqx --teacher-net NET \\
        --data state-188.npz state-3.npz --out teacher-values
    uv run examples/othello/distill.py train --data state-188.npz --heldout state-3.npz \\
        --net resnet:4x64,pool_every=2,heads=pooled,pool_channels=16 \\
        --value-source teacher --value-target-q 1 --teacher-values teacher-values \\
        --ckpt-dir runs/student --monitor --notes "does a student learn from its teacher?"

--data takes kept positions by run directory, a fork's with its ancestors' from before it forked.
They stay on disk, however many: training reads --chunk-positions at a time, in shuffled blocks.

    uv run examples/othello/distill.py train --data runs/fork --from-epoch 100 --heldout state-3.npz \\
        --net resnet:4x64,pool_every=2,heads=pooled,pool_channels=16 --ckpt-dir runs/student

A state's positions that are also among kept ones given with it count once. Its checkpoints load with `game.load_checkpoint`, like train.py's.
"""

import argparse
import dataclasses
import json
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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

from core.memory.position_archive import (
    PositionFile,
    from_rows,
    read_memmap,
    select_files,
    to_rows,
)
from core.memory.replay_memory import BaseExperience, transform_randomly
from core.monitoring import DEFAULT_URL, Monitor
from core.networks.utils import apply_nn
from core.training.loss_fns import az_default_loss_fn

# positions evaluated at once when measuring a network, not training it
EVAL_BATCH = 4096
# positions read from disk together, in training: enough for chunks to mix games from across a
# large dataset (a chunk of 4M holds about 490 of them), small enough to seek little
BLOCK = 8192


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


def row_hashes(rows: np.ndarray) -> np.ndarray:
    """A 64-bit hash of each row's position and search results, to recognize it elsewhere."""
    hashes = np.zeros(len(rows), np.uint64)
    for name in ("observation_nn", "policy_weights", "search_value"):
        x = rows[name].reshape(len(rows), -1)
        words = x.astype(np.uint8) if x.dtype == bool else x.view(np.uint32)
        odd = np.random.default_rng(len(name)).integers(
            0, 2**63, words.shape[1], np.uint64
        )
        hashes = hashes * np.uint64(0x9E3779B97F4A7C15) + (words * (2 * odd + 1)).sum(
            axis=1, dtype=np.uint64
        )
    return hashes


def already_kept(rows: np.ndarray, files: list[PositionFile]) -> np.ndarray:
    """Which of a state's rows are also in `files`. A state's positions are those finished last,
    so this looks from the newest file back, until it finds no more."""
    hashes = row_hashes(rows)
    found = np.zeros(len(rows), bool)
    for file in reversed(files):
        matched = np.isin(hashes, row_hashes(read_memmap(file.path)))
        if found.any() and not matched.any():
            break
        found |= matched
        if found.all():
            break
    return found


@dataclass(frozen=True)
class Source:
    """Rows to train on (see `core.memory.position_archive.to_rows`).

    Attributes:
        rows: the rows in memory, as a state's, or the file they're in
        values: (optional) a teacher's values of the rows of a file, the file they're in
    """

    rows: np.ndarray | str
    values: str | None = None

    def __len__(self) -> int:
        return len(self._open())

    def _open(self) -> np.ndarray:
        return read_memmap(self.rows) if isinstance(self.rows, str) else self.rows

    def read(self, at: slice | np.ndarray) -> np.ndarray:
        """New rows in memory. Files stay mapped only for as long as this takes."""
        rows = np.array(self._open()[at])
        if self.values is not None:
            rows["search_value"] = read_memmap(self.values)[at]
        return rows


class Dataset:
    """Positions to train on, from any number of sources, in memory only as they're used."""

    def __init__(self, sources: list[Source], chunk_positions: int | None = None):
        """
        Args:
            sources: where the positions are
            chunk_positions: at most how many to hold at once in training (see `batches`). If
                not given, all of them.
        """
        self.sources = sources
        self.starts = np.cumsum([0, *(len(source) for source in sources)])
        self.chunk_positions = chunk_positions

    @classmethod
    def of(cls, positions: BaseExperience) -> "Dataset":
        """Positions in memory, as a whole."""
        return cls([Source(to_rows(positions, 0))])

    def __len__(self) -> int:
        return int(self.starts[-1])

    def take(self, indices: np.ndarray) -> BaseExperience:
        """Those positions, read in order of where they are."""
        ranks = np.argsort(indices, kind="stable")
        ordered = indices[ranks]
        at_source = np.searchsorted(self.starts, ordered, side="right") - 1
        rows = np.concatenate(
            [
                self.sources[i].read(ordered[at_source == i] - self.starts[i])
                for i in np.unique(at_source)
            ]
        )
        restore = np.empty_like(ranks)
        restore[ranks] = np.arange(len(ranks))
        return from_rows(rows[restore])

    def _blocks(self, rng: np.random.Generator) -> Iterator[np.ndarray]:
        """Rows in blocks, in a new random order of all of them each pass, without end."""
        blocks = [
            (source, start, min(start + BLOCK, len(source)))
            for source in self.sources
            for start in range(0, len(source), BLOCK)
        ]
        while True:
            for i in rng.permutation(len(blocks)):
                source, start, stop = blocks[i]
                yield source.read(slice(start, stop))

    def batches(
        self, batch_size: int, rng: np.random.Generator
    ) -> Iterator[tuple[BaseExperience, jax.Array]]:
        """Training's batches without end, each as positions on the device and a batch's indices
        in them.

        If it's allowed to hold all the positions, it does, and each pass goes through them in a
        new random order. Otherwise, it holds a chunk at a time: blocks of `BLOCK` positions from
        across the whole dataset, shuffled together. A pass of a chunk at a time still sees every
        block once. The next chunk is read while the last trains.
        """
        total = len(self)
        if self.chunk_positions is None or total <= self.chunk_positions:
            on_device = jax.tree.map(jnp.asarray, self.take(np.arange(total)))
            while True:
                order = rng.permutation(total)
                for i in range(total // batch_size):
                    yield (
                        on_device,
                        jnp.asarray(order[i * batch_size : (i + 1) * batch_size]),
                    )
        # a whole number of batches, all chunks one size: training compiles for each size
        size = max(self.chunk_positions // batch_size, 1) * batch_size
        blocks = self._blocks(rng)
        rest = next(blocks)

        def next_chunk(rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            parts, count = [rest], len(rest)
            while count < size:
                parts.append(next(blocks))
                count += len(parts[-1])
            rows = np.concatenate(parts)
            return rows[:size], rows[size:]

        with ThreadPoolExecutor(max_workers=1) as reader:
            coming = reader.submit(next_chunk, rest)
            while True:
                rows, rest = coming.result()
                coming = reader.submit(next_chunk, rest)
                on_device = jax.tree.map(jnp.asarray, from_rows(rows))
                order = rng.permutation(size)
                for i in range(size // batch_size):
                    yield (
                        on_device,
                        jnp.asarray(order[i * batch_size : (i + 1) * batch_size]),
                    )
                del on_device


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
    targets = batch.policy_weights
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


def make_train_step(optimizer, value_target_q: float):
    """One supervised step on a batch, under a random symmetry for each position; the loss is
    self-play training's (see `az_default_loss_fn`), so the steps differ only in their data."""
    loss_fn = partial(az_default_loss_fn, value_target_q=value_target_q)

    @eqx.filter_jit
    def step(params, static, nn_state, opt_state, positions, indices, key):
        batch = jax.tree.map(lambda x: x[indices], positions)
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
    positions: Dataset,
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
    steps_per_pass = len(positions) // batch_size
    if steps_per_pass < 1:
        raise ValueError(
            f"{len(positions)} positions are too few for batches of {batch_size}"
        )
    optimizer = make_optimizer(passes * steps_per_pass, lr, lr_final, weight_decay)
    params, static = eqx.partition(network, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    step = make_train_step(optimizer, value_target_q)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    probe_size = min(train_probe_size, len(positions))
    probe = positions.take(rng.choice(len(positions), probe_size, replace=False))
    batches = positions.batches(batch_size, rng)
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train_metrics: dict[str, jax.Array] = {}
    for done in range(passes):
        for _ in range(steps_per_pass):
            key, step_key = jax.random.split(key)
            on_device, indices = next(batches)
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
    states, directories = split_data(args.data)
    files = select_files(directories, args.from_epoch, args.to_epoch)
    for path in [*states, *(file.path for file in files)]:
        target = teacher_values_path(out, path)
        # an interrupted run goes on from where it was
        if target.exists():
            continue
        positions = (
            load_positions(path)
            if path in states
            else from_rows(np.array(read_memmap(path)))
        )
        metrics, values = measure(network, params, positions)
        with open(f"{target}.tmp", "wb") as f:
            np.save(f, values)
        os.replace(f"{target}.tmp", target)
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


def split_data(data: list[str]) -> tuple[list[str], list[str]]:
    """--data's saved states, and its run directories."""
    return [p for p in data if not os.path.isdir(p)], [
        p for p in data if os.path.isdir(p)
    ]


def load_dataset(args) -> Dataset:
    """The positions of --data: its saved states', but for those also kept in its run directories,
    and those kept in its run directories, within --from-epoch and --to-epoch."""
    states, directories = split_data(args.data)
    files = select_files(directories, args.from_epoch, args.to_epoch)
    if directories and not files:
        raise ValueError(f"no positions kept in {directories} in those epochs")
    teacher = args.value_source == "teacher"
    directory = Path(args.teacher_values) if teacher else None
    if teacher and args.teacher_values is None:
        raise ValueError("the teacher value source needs teacher values")
    sources = []
    for path in states:
        rows = to_rows(load_positions(path), 0)
        repeated = already_kept(rows, files)
        if directory is not None:
            rows["search_value"] = np.load(teacher_values_path(directory, path))
        print(f"{path}: {(~repeated).sum()} of {len(rows)} positions are new")
        sources.append(Source(rows[~repeated]))
    sources += [
        Source(
            file.path,
            str(teacher_values_path(directory, file.path)) if directory else None,
        )
        for file in files
    ]
    dataset = Dataset(sources, args.chunk_positions if directories else None)
    print(f"{len(dataset)} positions, from {len(sources)} sources", flush=True)
    return dataset


def train_command(args) -> None:
    positions = load_dataset(args)
    heldout = load_positions(args.heldout)
    if args.heldout_size < len(heldout.reward):
        chosen = np.random.default_rng(0).choice(
            len(heldout.reward), args.heldout_size, replace=False
        )
        heldout = jax.tree.map(lambda x: x[chosen], heldout)
    else:
        chosen = np.arange(len(heldout.reward))
    heldout_teacher = None
    if args.teacher_values is not None:
        directory = Path(args.teacher_values)
        heldout_teacher = np.load(teacher_values_path(directory, args.heldout))[chosen]
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


def add_epoch_filter(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--from-epoch",
        type=int,
        default=None,
        help="of kept positions, only those of games finished from this epoch on, counted along "
        "the lineage: a fork forked after E epochs goes on from E",
    )
    parser.add_argument(
        "--to-epoch",
        type=int,
        default=None,
        help="of kept positions, only those of games finished before this epoch",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Trains a network on other runs' games."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    values = commands.add_parser(
        "values", help="compute a teacher's value of positions"
    )
    values.add_argument("--teacher", required=True, help="the teacher's checkpoint")
    values.add_argument("--teacher-net", required=True, help="the teacher's --net")
    values.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="saved training states, and run directories of positions kept",
    )
    add_epoch_filter(values)
    values.add_argument("--out", required=True, help="directory to write the values to")
    values.set_defaults(run=values_command)

    fit = commands.add_parser("train", help="train a network on positions")
    fit.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="what to learn from: saved training states, and directories of runs that kept their "
        "positions (train.py's --save-positions-from), a fork's with its ancestors'",
    )
    add_epoch_filter(fit)
    fit.add_argument(
        "--chunk-positions",
        type=int,
        default=4_000_000,
        help="with kept positions, at most how many to train on at once, in memory and on the "
        "device: about 470 MB a million, twice that in memory as the next is read",
    )
    fit.add_argument(
        "--heldout",
        required=True,
        help="a saved training state of games the network and teacher never trained on",
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
