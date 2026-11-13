"""How fast each engine solves endgames: a set of positions solved exactly, one after another, at
some threads, each engine in its own way of doing so, timed and its nodes counted where it says.

    taskset -c 0 uv run examples/othello/bench_engines.py --set ffo20-39 --threads 1 --rounds 3
    taskset -c 0-7 uv run examples/othello/bench_engines.py --set ffo40-59 --threads 8

The sets are OBF files (a board of 64 squares a1..h8 and who's to move, a line each) in
`--dir`, by default engine-fairness's: ffo20-39 and ffo40-59 (FFO's endgame test positions,
20-34 empties), positions_end (40 positions of 12-24 empties from real games). Engines alternate
round by round, so that a slow drift of the machine falls on all; each's median round is
reported. The nodes are each engine's own count: they measure different things, and only the
times compare across engines.

- Edax, Edax-FM: `-solve FILE` at level 60.
- Egaroucid: `-solve FILE` at level 60, no book.
- Melody: its benchmark mode, its command line's FILE: each position solved exactly, its hash
  cleared before each.
- Neural Reversi: `solve FILE --exact`.
- ymatioun, which has no such mode: each position given to a fresh process with all the time in
  the world, so that it searches until its search is exact; timed here, from the command that
  starts it, nodes unknown. A position with one move it answers without searching. It deepens
  its midgame search before it solves, and on some positions takes many minutes to be exact
  (one of ffo20-39's, at 26 empties, a 64-0 win, over 9 minutes, which Edax solves at once), so a
  position it hasn't solved in `CAP` seconds counts as `CAP`, and the result says how many.

Hash tables: 2^25 entries for Edax, Edax-FM, Egaroucid and Melody (their benchmarks' setting);
1 GB for Neural Reversi; ymatioun's, set when it's built, 128 MB a thread. The last JSON line
printed is every result.
"""

import argparse
import json
import platform
import re
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from engines import (
    CACHE_DIR,
    ClockedYmatioun,
    Edax,
    EdaxFM,
    Egaroucid,
    Melody,
    NeuralReversi,
)
from native_target import cpus

SETS_DIR = CACHE_DIR / "engine-fairness"
# a solve's seconds, nodes (None where the engine doesn't say), and how many positions it gave up
# on at `CAP` seconds each, counted at that
Solve = tuple[float, int | None, int]
# what ymatioun, which doesn't solve as the others do, gets a position at most
CAP = 60.0


def clock(text: str) -> float:
    """Seconds from "1:02:03.4", "2:03.4" or "03.4"."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def run(command: list[str], cwd: Path, timeout: float) -> str:
    done = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    return done.stdout + done.stderr


def edax_solve(engine_dir: Path, path: Path, threads: int, timeout: float) -> Solve:
    text = run(
        [str(engine_dir / "bin" / "edax"), "-l", "60", "-h", "25", "-n", str(threads),
         "-solve", str(path)],
        engine_dir,
        timeout,
    )  # fmt: skip
    found = re.search(r"(\d+) nodes in\s+([\d:.]+)", text)
    assert found, text[-800:]
    return clock(found[2]), int(found[1]), 0


def egaroucid_solve(
    engine_dir: Path, path: Path, threads: int, timeout: float
) -> Solve:
    text = run(
        [str(engine_dir / "bin" / "Egaroucid_for_Console.out"), "-level", "60",
         "-hash", "25", "-nobook", "-threads", str(threads), "-solve", str(path)],
        engine_dir / "bin",
        timeout,
    )  # fmt: skip
    found = re.search(r"total (\d+) nodes in ([\d.]+)s", text)
    assert found, text[-800:]
    return float(found[2]), int(found[1]), 0


def melody_solve(engine_dir: Path, path: Path, threads: int, timeout: float) -> Solve:
    text = run(
        [
            str(engine_dir / "build" / "Melody"),
            "-h",
            "25",
            "-t",
            str(threads),
            str(path),
        ],
        engine_dir,
        timeout,
    )
    # "Total time  : 00:00:01.23", "Total nodes : 1 412 590"
    seconds = re.search(r"Total time\s*:\s*([\d:.]+)", text)
    nodes = re.search(r"Total nodes\s*:\s*([\d ]+)", text)
    assert seconds and nodes, text[-800:]
    return clock(seconds[1]), int(nodes[1].replace(" ", "")), 0


def neural_reversi_solve(
    engine_dir: Path, path: Path, threads: int, timeout: float
) -> Solve:
    text = run(
        [str(engine_dir / "target" / "release" / "cli"), "solve", str(path), "--exact",
         "--threads", str(threads), "--hash-size", "1024"],
        engine_dir,
        timeout,
    )  # fmt: skip
    # "| Total |        |       |    0:00.034 |           1,131,376 | ..."
    found = re.search(
        r"\|\s*Total\s*\|[^|]*\|[^|]*\|\s*([\d:.]+)\s*\|\s*([\d,]+)", text
    )
    assert found, text[-800:]
    return clock(found[1]), int(found[2].replace(",", "")), 0


def ymatioun_solve(engine_dir: Path, path: Path, threads: int, timeout: float) -> Solve:
    seconds, capped = 0.0, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        board, side = line.split()[:2]
        # a process a position: it carries what it last searched into the next search, copying
        # its whole hash table the first time the position isn't the one it expected
        engine = ClockedYmatioun(engine_dir, 0, threads)
        assert engine.process.stdout is not None
        stop = threading.Timer(CAP, engine.process.kill)
        try:
            engine.send(f"position fen {board} {'*' if side == 'X' else 'O'}")
            start = time.perf_counter()
            stop.start()
            engine.send(f"go time {10**9}")
            while (reply := engine.process.stdout.readline()) and not reply.startswith(
                "bestmove"
            ):
                pass
            stop.cancel()
            seconds += min(time.perf_counter() - start, CAP)
        finally:
            stop.cancel()
            engine.close()
        if not reply:
            capped += 1
            continue
        # "bestmove 62 score 2 time 54 depth 17.10": the search exact (ProbCut's width 10,
        # its infinity), or a single move answered without one
        assert reply.split()[2] == "score0" or reply.split()[-1].endswith(".10"), reply
    return seconds, None, capped


SOLVERS: dict[str, tuple[Callable[[Path, Path, int, float], Solve], Path]] = {
    "edax": (edax_solve, Edax.default_dir()),
    "edax-fm": (edax_solve, EdaxFM.default_dir()),
    "egaroucid": (egaroucid_solve, Egaroucid.default_dir()),
    "melody": (melody_solve, Melody.default_dir()),
    "neural-reversi": (neural_reversi_solve, NeuralReversi.default_dir()),
    "ymatioun": (ymatioun_solve, ClockedYmatioun.default_dir()),
}


def cpu() -> str | None:
    text = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    found = re.search(r"^model name\s*:\s*(.*)$", text, re.MULTILINE)
    return found[1] if found else platform.processor() or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Engines' exact solves, timed.")
    parser.add_argument("--set", required=True, help="an OBF file's name in --dir")
    parser.add_argument("--dir", type=Path, default=SETS_DIR)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--engines", default=",".join(SOLVERS))
    parser.add_argument("--timeout", type=float, default=3600, help="seconds a solve")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="engines built in ROOT/<name> instead of where their setup scripts put them",
    )
    args = parser.parse_args()

    path = args.dir / f"{args.set}.obf"
    names = args.engines.split(",")
    times: dict[str, list[Solve]] = {name: [] for name in names}
    for round_ in range(args.rounds):
        for name in names:
            solve, engine_dir = SOLVERS[name]
            if args.root is not None:
                engine_dir = args.root / name
            times[name].append(solve(engine_dir, path, args.threads, args.timeout))
            seconds, nodes, capped = times[name][-1]
            print(
                json.dumps({"round": round_, "engine": name, "seconds": seconds,
                            "nodes": nodes, "capped": capped}),
                flush=True,
            )  # fmt: skip
    results = {}
    for name, solves in times.items():
        seconds = statistics.median(s for s, _, _ in solves)
        nodes = [n for _, n, _ in solves if n is not None]
        results[name] = {
            "seconds": round(seconds, 3),
            "seconds_all": [round(s, 3) for s, _, _ in solves],
            "capped": max(c for _, _, c in solves),
            "nodes": int(statistics.median(nodes)) if nodes else None,
            "nodes_per_second": (
                round(statistics.median(nodes) / seconds) if nodes else None
            ),
        }
    print(
        json.dumps({"set": args.set, "threads": args.threads, "rounds": args.rounds,
                    "cpu": cpu(), "cpus": sorted(cpus()),
                    "results": results}),
        flush=True,
    )  # fmt: skip


if __name__ == "__main__":
    main()
