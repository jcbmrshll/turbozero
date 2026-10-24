"""Evaluates a training run's checkpoints against Edax or Egaroucid as they're saved, so a
long run's strength shows before it ends.

    uv run examples/othello/train.py --net resnet:10x128 --keep-every 25 --ckpt-dir runs/othello &
    uv run examples/othello/watch.py runs/othello --net resnet:10x128 --pid $!

Runs `vs_engine.py` on every checkpoint whose epoch is a multiple of --every (keep them
with train.py's --keep-every, since it deletes all but the newest 2), and on the newest
one once the training process --pid has exited. `N.eqx` is the network after N + 1 epochs.
Results are appended to `vs_engine.jsonl` in the checkpoint directory, a line per level,
and with --monitor logged to the training run's page as `edax_level4_sims64` and so on
(`{engine}_level{L}_sims{S}`), each with a game at that level, `edax_level4_sims64_game`, to
explore move by move: our agent's searches, and the engine's scores: Edax's of every legal
move of every position, each with the line it expects after it, Egaroucid's of the moves it
played (see vs_engine.py). Without --pid it evaluates the checkpoints already there and
exits.

A checkpoint is done once `vs_engine.jsonl` has a result for each of --levels at --sims
and --openings. One evaluated at other levels or with other openings, or whose evaluation
stopped partway, is evaluated at its missing levels (with the openings they'd have had in
a full evaluation, see vs_engine.py's --skip-levels).

Our agent's search runs on the accelerator next to training, but it only takes what it
needs (XLA_PYTHON_CLIENT_PREALLOCATE=false); --cpu keeps it off the accelerator.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.training.train import checkpoint_epochs, checkpoint_path
from monitor import DEFAULT_URL, Monitor
from monitor.client import Episode

VS_ENGINE = Path(__file__).with_name("vs_engine.py")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def done_levels(
    results: Path, sims: int, engine: str, games: int
) -> defaultdict[str, set[int]]:
    """The levels each checkpoint has a result for in `results`, at these settings."""
    done = defaultdict(set)
    if not results.exists():
        return done
    with open(results) as f:
        records = [json.loads(line) for line in f if line.strip()]
    for r in records:
        if (
            r["sims"] == sims
            and r["engine"].lower() == engine
            and r.get("games") == games
        ):
            done[r["checkpoint"]].add(r["level"])
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates a training run's checkpoints against an engine as they're saved."
    )
    parser.add_argument("ckpt_dir", type=Path, help="train.py's --ckpt-dir")
    parser.add_argument("--net", default="resnet:6x128", help="the run's --net")
    parser.add_argument("--engine", default="edax")
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--openings", type=int, default=128)
    parser.add_argument(
        "--every",
        type=int,
        default=25,
        help="evaluate epochs that are multiples of this",
    )
    parser.add_argument(
        "--pid", type=int, default=None, help="the training process to wait for"
    )
    parser.add_argument("--poll", type=float, default=30, help="seconds between looks")
    parser.add_argument("--cpu", action="store_true", help="search on the CPU")
    parser.add_argument(
        "--monitor",
        nargs="?",
        const=DEFAULT_URL,
        default=None,
        metavar="URL",
        help=f"log results to a turbozero monitor (default {DEFAULT_URL})",
    )
    parser.add_argument(
        "--run", default=None, help="the training run's id on the monitor"
    )
    args = parser.parse_args()
    if args.monitor and not args.run:
        parser.error("--monitor needs --run, the id in train.py's monitor link")

    results = args.ckpt_dir / "vs_engine.jsonl"
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if args.cpu:
        env["JAX_PLATFORMS"] = "cpu"
    monitor = None
    if args.monitor:
        monitor = Monitor(args.monitor, project="othello")
        monitor.run_id = args.run

    def evaluate(epoch: int, skip: list[int]) -> None:
        path = checkpoint_path(str(args.ckpt_dir), epoch)
        print(
            f"epoch {epoch}: evaluating {path}"
            + (f" (levels {skip} are done)" if skip else ""),
            flush=True,
        )
        start = time.perf_counter()
        # vs_engine.py appends its results to `results`; read back the new lines. With a
        # monitor, it saves a game per level to show there
        seen = results.stat().st_size if results.exists() else 0
        episodes = tempfile.TemporaryDirectory(prefix="watch-")
        subprocess.run(
            [sys.executable, str(VS_ENGINE), path]
            + ["--engine", args.engine, "--net", args.net, "--sims", str(args.sims)]
            + ["--openings", str(args.openings), "--stop-below", "-1"]
            + ["--results", str(results), "--levels"]
            + [str(level) for level in args.levels]
            + (["--skip-levels"] + [str(level) for level in skip] if skip else [])
            + (["--episodes", episodes.name] if monitor is not None else []),
            env=env,
            check=True,
        )
        print(f"epoch {epoch}: done in {time.perf_counter() - start:.0f}s", flush=True)
        if monitor is not None:
            with open(results) as f:
                f.seek(seen)
                records = [json.loads(line) for line in f if line.strip()]
            logged = {}
            for r in records:
                key = f"{args.engine}_level{r['level']}_sims{r['sims']}"
                logged[key] = r["score"]
                game = Path(episodes.name) / f"{key}.npz"
                if game.exists():
                    with np.load(game) as arrays:
                        logged[f"{key}_game"] = Episode(
                            "othello",
                            opponent=f"{args.engine} level {r['level']}",
                            **arrays,
                        )
            monitor.log(epoch, logged)
            monitor.flush()
        episodes.cleanup()

    while True:
        # read the process's state before the directory, so a checkpoint saved
        # just before it exited is seen
        training = args.pid is not None and alive(args.pid)
        epochs = checkpoint_epochs(str(args.ckpt_dir))
        done = done_levels(results, args.sims, args.engine, 2 * args.openings)
        # the levels asked for that each checkpoint has a result for
        done_at = {
            e: [
                level
                for level in args.levels
                if level in done[checkpoint_path(str(args.ckpt_dir), e)]
            ]
            for e in epochs
        }
        todo = [
            e
            for e in epochs
            # 0.eqx, after a single epoch, is too early to be worth evaluating
            if e > 0 and e % args.every == 0 and len(done_at[e]) < len(args.levels)
        ]
        if not training and epochs:
            last = epochs[-1]
            if len(done_at[last]) < len(args.levels) and last not in todo:
                todo.append(last)
        if todo:
            # the newest first: it says most about the run now
            evaluate(todo[-1], done_at[todo[-1]])
            continue
        if not training:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
