"""Network evaluations a second of quantized models against bf16, on the path play uses
(`cpu_search.openvino_eval`), net alone and with the search. Arms take turns, round after round, so
drifting load falls on all of them.

    taskset -c 16-23 uv run examples/othello/bench_quantized.py \
        --net 8x128-128s-8p-w1125-274m --results out.jsonl

An arm is "bf16", "f32", or "int8-<variant>" (of `quantize.py`), computing what isn't quantized in
f32, or "int8-<variant>-bf16", in bf16.
"""

import argparse
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
from bench_cpu_backends import network_path
from cpu_match import make_eval, model_and_precision
from cpu_search import Search, SearchSettings, compile_model
from quantize import BEST, engine_positions
from xot import XOT_PATH, load_xot


def primitives(arm: str, name: str, cores: int) -> dict[str, int]:
    """How many of the compiled graph's convolutions run on each kernel."""
    import openvino as ov

    compiled = compile_model(*model_and_precision(name, arm), cores, 1, ov.Core())
    counts: Counter[str] = Counter()
    for op in compiled.get_runtime_model().get_ops():
        info = op.get_rt_info()
        if info["layerType"].astype(str) == "Convolution":
            kernel = info["primitiveType"].astype(str)
            counts[f"{kernel}/{info['runtimePrecision'].astype(str)}"] += 1
    return dict(counts)


def net_only(evaluate, x: np.ndarray, batch: int, seconds: float) -> float:
    """Evals a second, cycling through the positions."""
    start = time.perf_counter()
    calls = 0
    n = len(x) - batch
    while (elapsed := time.perf_counter() - start) < seconds:
        offset = calls * batch % n
        evaluate(x[offset : offset + batch])
        calls += 1
    return calls * batch / elapsed


def with_search(evaluate, openings: np.ndarray, batch: int, visits: int) -> dict:
    """Whole games against itself, one at a time, as `cpu_match.py speed` plays them."""
    settings = SearchSettings(batch=batch, virtual_loss="visit")
    buffer = np.empty((batch, 2, 8, 8), np.float32)
    leaves = 0
    eval_seconds = 0.0
    begin = time.perf_counter()
    for opening in openings:
        search = Search(settings)
        for action in opening:
            search.play(int(action))
        while not search.terminated():
            if search.num_legal() == 1:
                search.play(search.first_legal())
                continue
            search.begin(visits)
            while search.remaining > 0:
                count = search.collect(buffer)
                t0 = time.perf_counter()
                logits, values = evaluate(buffer[:count])
                eval_seconds += time.perf_counter() - t0
                search.apply(logits, values)
                leaves += count
            search.play(search.best_action())
    seconds = time.perf_counter() - begin
    return {"evals_per_s": leaves / seconds, "eval_share": eval_seconds / seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--net", nargs="+", required=True)
    parser.add_argument("--arm", nargs="+", default=["bf16", f"int8-{BEST}"])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--search-batch", type=int, nargs="*", default=[16])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--visits", type=int, default=1000)
    parser.add_argument("--openings", type=int, default=2)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    cores = len(os.sched_getaffinity(0))
    positions = engine_positions(4096)[512:]
    pool = load_xot(XOT_PATH)
    openings = pool[
        np.random.default_rng(0).choice(len(pool), args.openings, replace=False)
    ]
    print(f"{cores} cores, load {os.getloadavg()[0]:.1f}", flush=True)

    def emit(record: dict) -> None:
        record["load"] = os.getloadavg()[0]
        print(json.dumps(record), flush=True)
        if args.results is not None:
            with open(args.results, "a") as f:
                f.write(json.dumps(record) + "\n")

    for name in args.net:
        network_path(name)
        for arm in args.arm:
            emit({"net": name, "arm": arm, "kernels": primitives(arm, name, cores)})
        # batch 1 on one stream, the others on a stream per core, as play has them
        evals = {
            (arm, streams): make_eval(name, f"{arm}-s{streams}", cores)
            for arm in args.arm
            for streams in sorted({1 if b == 1 else cores for b in args.batch})
        }
        rates: dict[tuple, list[float]] = {}
        for _ in range(args.rounds):
            for batch in args.batch:
                for arm in args.arm:
                    evaluate = evals[(arm, 1 if batch == 1 else cores)]
                    evaluate(positions[:batch])
                    rate = net_only(evaluate, positions, batch, args.seconds)
                    rates.setdefault((arm, batch), []).append(rate)
            for batch in args.search_batch:
                for arm in args.arm:
                    r = with_search(evals[(arm, cores)], openings, batch, args.visits)
                    rates.setdefault((arm, f"search{batch}"), []).append(
                        r["evals_per_s"]
                    )
                    rates.setdefault((arm, f"search{batch}_share"), []).append(
                        r["eval_share"]
                    )
        for (arm, batch), values in rates.items():
            emit(
                {
                    "net": name,
                    "arm": arm,
                    "batch": batch,
                    "median": round(statistics.median(values), 3),
                    "all": [round(v, 3) for v in values],
                }
            )


if __name__ == "__main__":
    main()
