"""Evaluates a training run's checkpoints against Edax or Egaroucid as they're saved, so a
long run's strength shows before it ends.

    uv run examples/othello/train.py --blocks 10 --keep-every 25 --ckpt-dir runs/othello &
    uv run examples/othello/watch.py runs/othello --blocks 10 --pid $!

Runs `vs_engine.py` on every checkpoint whose epoch is a multiple of --every (keep them
with train.py's --keep-every, since it deletes all but the newest 2), and on the newest
one once the training process --pid has exited. Results are appended to
`vs_engine.jsonl` in the checkpoint directory, a line per level, and with --monitor
logged to the training run's page as `edax_level4` and so on. Without --pid it
evaluates the checkpoints already there and exits.

Our agent's search runs on the accelerator next to training, but it only takes what it
needs (XLA_PYTHON_CLIENT_PREALLOCATE=false); --cpu keeps it off the accelerator.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from core.monitor import DEFAULT_URL, Monitor
from core.training.train import checkpoint_epochs, checkpoint_path

VS_ENGINE = Path(__file__).with_name("vs_engine.py")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def done_checkpoints(results: Path, sims: int, engine: str) -> set[str]:
    """Checkpoints already in `results` at these settings."""
    if not results.exists():
        return set()
    with open(results) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {
        r["checkpoint"]
        for r in records
        if r["sims"] == sims and r["engine"].lower() == engine
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates a training run's checkpoints against an engine as they're saved."
    )
    parser.add_argument("ckpt_dir", type=Path, help="train.py's --ckpt-dir")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=128)
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

    def evaluate(epoch: int) -> None:
        path = checkpoint_path(str(args.ckpt_dir), epoch)
        print(f"epoch {epoch}: evaluating {path}", flush=True)
        start = time.perf_counter()
        # vs_engine.py appends its results to `results`; read back the new lines
        seen = results.stat().st_size if results.exists() else 0
        subprocess.run(
            [sys.executable, str(VS_ENGINE), path]
            + ["--engine", args.engine, "--blocks", str(args.blocks)]
            + ["--channels", str(args.channels), "--sims", str(args.sims)]
            + ["--openings", str(args.openings), "--stop-below", "-1"]
            + ["--results", str(results), "--levels"]
            + [str(level) for level in args.levels],
            env=env,
            check=True,
        )
        print(f"epoch {epoch}: done in {time.perf_counter() - start:.0f}s", flush=True)
        if monitor is not None:
            with open(results) as f:
                f.seek(seen)
                records = [json.loads(line) for line in f if line.strip()]
            monitor.log(
                epoch,
                {
                    f"{args.engine}_level{r['level']}_sims{r['sims']}": r["score"]
                    for r in records
                },
            )
            monitor.flush()

    while True:
        # read the process's state before the directory, so a checkpoint saved
        # just before it exited is seen
        training = args.pid is not None and alive(args.pid)
        epochs = checkpoint_epochs(str(args.ckpt_dir))
        done = done_checkpoints(results, args.sims, args.engine)
        todo = [
            e
            for e in epochs
            # epoch 0 is the untrained network
            if e > 0
            and e % args.every == 0
            and checkpoint_path(str(args.ckpt_dir), e) not in done
        ]
        if not training and epochs:
            last = checkpoint_path(str(args.ckpt_dir), epochs[-1])
            if last not in done and epochs[-1] not in todo:
                todo.append(epochs[-1])
        if todo:
            # the newest first: it says most about the run now
            evaluate(todo[-1])
            continue
        if not training:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
