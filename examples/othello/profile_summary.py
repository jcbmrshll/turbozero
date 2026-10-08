"""Summarizes a GPU profile written by `jax.profiler.trace(dir, create_perfetto_trace=True)`
(e.g. by `bench_selfplay.py --trace DIR`): where the GPU's time goes, and how much of
the time it is idle.

    uv run examples/othello/profile_summary.py DIR [--steps 16] [--top 25]

With --steps, times are per self-play step (the number of steps the trace covers).
"""

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

# kernel name patterns, first match wins
CATEGORIES = [
    ("layout conversion", r"convertTensor|nchwToNhwc|nhwcToNchw|transpose"),
    ("conv", r"fprop|conv|xmma|implicit_gemm"),
    ("matmul", r"gemm|gemv|dot"),
    ("device->host copy", r"MemcpyD2H"),
    ("host->device copy", r"MemcpyH2D"),
    ("device copy", r"memcpy|MemcpyD2D|memset"),
]


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return "other (XLA fusions)"


def load_gpu_events(trace_dir: str) -> list[dict]:
    path = max(
        Path(trace_dir).rglob("perfetto_trace.json.gz"), key=lambda p: p.stat().st_mtime
    )
    with gzip.open(path) as f:
        events = json.load(f)["traceEvents"]
    gpu_pids = {
        e["pid"]
        for e in events
        if e.get("ph") == "M"
        and e["name"] == "process_name"
        and "/device:GPU" in e["args"]["name"]
    }
    return [e for e in events if e.get("ph") == "X" and e["pid"] in gpu_pids]


def busy_time(events: list[dict]) -> float:
    """Total time at least one kernel or copy runs (the union of their intervals), in µs."""
    total, end = 0.0, float("-inf")
    for start, stop in sorted((e["ts"], e["ts"] + e["dur"]) for e in events):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def main():
    parser = argparse.ArgumentParser(description="Summarize a GPU profile.")
    parser.add_argument("trace_dir")
    parser.add_argument("--steps", type=int, default=1, help="divide times by this")
    parser.add_argument("--top", type=int, default=25, help="kernels to list")
    args = parser.parse_args()

    events = load_gpu_events(args.trace_dir)
    # the profiled call is the long one: skip small jitted calls around it (key splits etc.)
    module_time = Counter()
    for e in events:
        module_time[e.get("args", {}).get("hlo_module", "")] += e["dur"]
    module = module_time.most_common(1)[0][0]
    events = [e for e in events if e.get("args", {}).get("hlo_module", "") == module]

    n = args.steps
    span = max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)
    busy = busy_time(events)
    kernel_time = sum(e["dur"] for e in events)
    print(f"module {module}: {len(events) / n:,.0f} GPU events per step")
    print(
        f"wall {span / n / 1e3:.1f} ms/step, GPU busy {busy / n / 1e3:.1f} ms/step "
        f"({100 * busy / span:.0f}%), kernel time {kernel_time / n / 1e3:.1f} ms/step"
    )

    by_category, count_by_category = Counter(), Counter()
    by_name, count_by_name = Counter(), Counter()
    for e in events:
        c = category(e["name"])
        by_category[c] += e["dur"]
        count_by_category[c] += 1
        by_name[e["name"][:100]] += e["dur"]
        count_by_name[e["name"][:100]] += 1
    print("\nby category (ms/step, count/step):")
    for c, t in by_category.most_common():
        print(f"  {t / n / 1e3:8.1f}  {count_by_category[c] / n:9.0f}  {c}")
    print(f"\ntop {args.top} kernels (ms/step, count/step):")
    for name, t in by_name.most_common(args.top):
        print(f"  {t / n / 1e3:8.1f}  {count_by_name[name] / n:9.0f}  {name}")


if __name__ == "__main__":
    main()
