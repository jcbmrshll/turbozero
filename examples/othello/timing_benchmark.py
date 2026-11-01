"""How much CPU each Othello player uses, whatever it plays against: configs timed one after
another on the contest's CPUs, each game a player against itself from an XOT opening.

    uv run examples/othello/timing_benchmark.py prepare         (the nets, as ONNX, once)
    uv run examples/othello/timing_benchmark.py plan
    taskset -c 16-23 uv run examples/othello/timing_benchmark.py work run-1 --hours 3.5
    uv run examples/othello/timing_benchmark.py summarize run-1 --publish

Everything else on the machine should be stopped: other load costs engines 30-45%, and a job won't
start if any CPU is doing real work. Results go to `TIMING_DIR` as each game ends, and each game
logs what else the machine was doing. Run it again to continue a job: finished configs are kept.

Each side of a game keeps its own tree or engine process, as in a league game, and only the side
to move is charged. Nets play from `cpu_search.py`, engines at a fixed level with their books off.
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from clock_match import Board
from cpu_search import Eval, Search, SearchSettings, openvino_eval
from engines import Edax, Egaroucid, GTPEngine
from game import action, square
from modal_store import DirStore, Store, get_json, put_json
from xot import XOT_PATH, load_xot

TIMING_DIR = Path.home() / ".cache" / "turbozero" / "othello" / "timing"
ONNX_DIR = TIMING_DIR / "onnx"
# a network of each architecture: those of one architecture cost the same
NETWORKS = {
    "4x64": "4x64-128s-8p-w1125-137m",
    "6x96": "6x96-128s-8p-w1125-137m",
    "8x128": "8x128-128s-8p-w1125-274m",
    "10x128": "10x128-128s-8p-w1125-274m",
}
# the builds of `engines.ENGINE_BUILDS`, as the task engine-fairness made them
ENGINE_BUILDS = {
    "edax": "okuhara/edax-reversi-AVX 8a2b3ea, x86-64-v4",
    "egaroucid": "Nyanyan/Egaroucid 3b1c6ad, -march=native -DHAS_AVX512 -DHAS_AMD_PROCESSOR",
}
OPENINGS_SEED = 0
THREADS = 8
# the contest's: 8 physical cores, whose other threads are 4-11
CONTEST_CPUS = set(range(16, 24))
SIBLING_CPUS = set(range(4, 12))
# a net's sequential evals a second and how much a batch multiplies it; only for budgeting
NET_SPEED = {"4x64": 10000.0, "6x96": 5600.0, "8x128": 2800.0, "10x128": 2200.0}
BATCH_EVALS = {1: 1.0, 4: 2.0, 16: 3.9, 32: 4.4}
# over which a config's timings are in doubt; everything else should be stopped for a run
BUSY_SIBLINGS = 0.05
BUSY_OTHERS = 0.05
# over which a job won't start, on any CPU, and how long it looks: a job shows well above it, and
# open agent sessions, a percent or so each, don't
BUSY_TO_START = 0.15
QUIET_SECONDS = 15.0
ENGINE_LEVELS = {
    "edax": (4, 6, 8, 10, 12, 16, 20, 24, 28),
    "egaroucid": (8, 12, 16, 20, 24, 28),
}
LEAGUE_TIMINGS = TIMING_DIR.parent / "league" / "timings.json"
REFERENCE_EVERY = 1800.0
FIT_SIMS = (128, 1000, 4000)
HELD_OUT = 16000
# how many rounds of work that fills a core's floating-point units a process gets through in some
# seconds: two threads of a core would have to share them
FMA_SOURCE = """
#include <immintrin.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
int main(int, char** argv) {
    auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(atof(argv[1]));
    __m256d a[8];
    for (auto& v : a) v = _mm256_set1_pd(1.0);
    const __m256d m = _mm256_set1_pd(0.999), c = _mm256_set1_pd(0.001);
    uint64_t rounds = 0;
    while (std::chrono::steady_clock::now() < end) {
        for (int i = 0; i < 1 << 18; i++)
            for (auto& v : a) v = _mm256_fmadd_pd(v, m, c);
        rounds++;
    }
    double out[4];
    _mm256_storeu_pd(out, a[0]);
    printf("%llu %f\\n", (unsigned long long)rounds, out[0]);
}
"""


def read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def run_text(*command: str) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_cpuinfo(text: str) -> dict[str, Any]:
    """The first processor of /proc/cpuinfo: its model, flags and the like."""
    first = {}
    for line in text.split("\n\n")[0].splitlines():
        key, _, value = line.partition(":")
        first[key.strip()] = value.strip()
    flags = first.get("flags", "").split()
    wanted = ("avx2", "avx512f", "avx512_vnni", "avx512_bf16", "amx_bf16", "amx_tile")
    return {
        "model": first.get("model name"),
        "family": first.get("cpu family"),
        "model_id": first.get("model"),
        "stepping": first.get("stepping"),
        "mhz": first.get("cpu MHz"),
        "cache_size": first.get("cache size"),
        "flags": [flag for flag in wanted if flag in flags],
        "all_flags": flags,
    }


def thread_layout(cpus: list[int]) -> dict[int, str | None]:
    """Each of `cpus`' threads of its physical core, as sysfs writes them."""
    return {
        cpu: read_text(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        for cpu in cpus
    }


def caches() -> list[dict[str, str | None]]:
    root = Path("/sys/devices/system/cpu/cpu0/cache")
    return [
        {
            name: read_text(index / name)
            for name in ("level", "type", "size", "shared_cpu_list")
        }
        for index in sorted(root.glob("index*"))
    ]


def sysinfo() -> dict[str, Any]:
    """What this machine is, as far as it can tell."""
    cpus = sorted(os.sched_getaffinity(0))
    meminfo = read_text("/proc/meminfo") or ""
    return {
        "time": time.time(),
        "cpuinfo": parse_cpuinfo(read_text("/proc/cpuinfo") or ""),
        "lscpu": run_text("lscpu"),
        "affinity": cpus,
        "cpu_count": os.cpu_count(),
        "thread_layout": thread_layout(cpus),
        "caches": caches(),
        "memory": meminfo.splitlines()[:2],
        "kernel": platform.platform(),
        "uname": platform.uname()._asdict(),
        "hostname": platform.node(),
        "load": os.getloadavg(),
        "gxx": (run_text("g++", "--version") or "").split("\n")[0],
        "engine_builds": ENGINE_BUILDS,
    }  # fmt: skip


def build_spin(directory: str) -> str:
    source = Path(directory) / "spin.cpp"
    source.write_text(FMA_SOURCE)
    out = str(Path(directory) / "spin")
    subprocess.run(
        ["g++", "-O2", "-std=c++17", "-mavx2", "-mfma", str(source), "-o", out],
        check=True,
    )
    return out


def spin(binary: str, count: int, seconds: float) -> dict[str, Any]:
    """`count` processes of compute at once, how many rounds each got through."""
    processes = [
        subprocess.Popen([binary, str(seconds)], stdout=subprocess.PIPE, text=True)
        for _ in range(count)
    ]
    rounds = [int(p.communicate()[0].split()[0]) for p in processes]
    return {"processes": count, "rounds": rounds, "total": sum(rounds)}


def net_path(arch: str) -> Path:
    return ONNX_DIR / f"{arch}.onnx"


def openvino_info(path: Path, threads: int, streams: int, precision: str) -> dict:
    """What OpenVINO decided: the precision it will compute in, and the primitives it chose."""
    import openvino as ov

    core = ov.Core()
    network = core.read_model(str(path))
    network.reshape({"input": [-1, 2, 8, 8]})
    compiled = core.compile_model(
        network,
        "CPU",
        {
            "INFERENCE_NUM_THREADS": threads,
            "NUM_STREAMS": streams,
            "INFERENCE_PRECISION_HINT": precision,
        },
    )
    kinds: dict[str, int] = {}
    precisions: dict[str, int] = {}
    for op in compiled.get_runtime_model().get_ops():
        info = op.get_rt_info()
        if "primitiveType" in info:
            kind = info["primitiveType"].astype(str)
            kinds[kind] = kinds.get(kind, 0) + 1
        if "runtimePrecision" in info:
            value = info["runtimePrecision"].astype(str)
            precisions[value] = precisions.get(value, 0) + 1

    def asked(name: str) -> str | None:
        try:
            return str(compiled.get_property(name))
        except RuntimeError:
            return None

    return {
        "version": ov.__version__,
        "device": core.get_property("CPU", "FULL_DEVICE_NAME"),
        "capabilities": [str(c) for c in core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")],
        "asked": precision,
        "precision": asked("INFERENCE_PRECISION_HINT"),
        "threads": asked("INFERENCE_NUM_THREADS"),
        "streams": asked("NUM_STREAMS"),
        "pinning": asked("ENABLE_CPU_PINNING"),
        "scheduling": asked("SCHEDULING_CORE_TYPE"),
        "primitives": kinds,
        "runtime_precisions": precisions,
    }  # fmt: skip


def make_openings(count: int) -> np.ndarray:
    """The openings every config plays, the first of them if it plays fewer."""
    pool = load_xot(XOT_PATH)
    rng = np.random.default_rng(OPENINGS_SEED)
    return pool[rng.choice(len(pool), size=count, replace=False)]


def real_leaves(evaluate: Eval, opening: np.ndarray, count: int) -> np.ndarray:
    """`count` observations a search of a game asks to evaluate."""
    search = Search(SearchSettings(batch=16, virtual_loss="visit"))
    for move in opening:
        search.play(int(move))
    buffer = np.empty((16, 2, 8, 8), np.float32)
    leaves = []
    total = 0
    while not search.terminated() and total < count:
        search.begin(128)
        while search.remaining > 0:
            found = search.collect(buffer)
            leaves.append(buffer[:found].copy())
            total += found
            logits, values = evaluate(buffer[:found])
            search.apply(logits, values)
        search.play(search.best_action())
    return np.concatenate(leaves)[:count]


def net_only(evaluate: Eval, leaves: np.ndarray, batch: int, seconds: float) -> float:
    """Evals a second of the network alone."""
    evaluate(leaves[:batch])
    begun = time.perf_counter()
    done = at = 0
    while time.perf_counter() - begun < seconds:
        if at + batch > len(leaves):
            at = 0
        evaluate(leaves[at : at + batch])
        at += batch
        done += batch
    return done / (time.perf_counter() - begun)


def net_game(
    evaluate: Eval,
    settings: SearchSettings,
    visits: int,
    opening: np.ndarray,
) -> list[dict[str, Any]]:
    """A game of a player against itself, a record for each move searched."""
    searches = (Search(settings), Search(settings))
    for move in opening:
        for search in searches:
            search.play(int(move))
    buffer = np.empty((settings.batch, 2, 8, 8), np.float32)
    records = []
    ply = len(opening)
    mover = 0
    while not searches[0].terminated():
        search, other = searches[mover], searches[1 - mover]
        if search.num_legal() == 1:
            move = search.first_legal()
        else:
            nodes_before = search.num_nodes()
            collisions_before = search.collisions()
            me, opp, _, _ = search.position()
            evals = calls = 0
            tree = network = 0.0
            cpu_begun = sum(os.times()[:2])
            begun = time.perf_counter()
            search.begin(visits)
            while search.remaining > 0:
                t0 = time.perf_counter()
                count = search.collect(buffer)
                t1 = time.perf_counter()
                logits, values = evaluate(buffer[:count])
                t2 = time.perf_counter()
                search.apply(logits, values)
                t3 = time.perf_counter()
                tree += t1 - t0 + t3 - t2
                network += t2 - t1
                evals += count
                calls += 1
            move = search.best_action()
            records.append(
                {
                    "side": mover,
                    "ply": ply,
                    "empties": 64 - (me | opp).bit_count(),
                    "legal": search.num_legal(),
                    "seconds": time.perf_counter() - begun,
                    "cpu": sum(os.times()[:2]) - cpu_begun,
                    "visits": visits,
                    "evals": evals,
                    "calls": calls,
                    "tree_seconds": tree,
                    "net_seconds": network,
                    "new_nodes": search.num_nodes() - nodes_before,
                    "collisions": search.collisions() - collisions_before,
                }
            )
        search.play(move)
        other.play(move)
        ply += 1
        mover = 1 - mover
    return records


class CountingEdax(Edax):
    """Edax, saying the nodes of its last search."""

    nodes: int | None = None

    def read_analysis(self, move: int) -> None:
        assert self.search_log is not None
        found = re.findall(r"nodes = (\d+) N", self.search_log.read())
        self.nodes = int(found[-1]) if found else None


class CountingEgaroucid(Egaroucid):
    """Egaroucid, saying the nodes of its last search."""

    nodes: int | None = None

    def read_analysis(self, move: int) -> None:
        found = [m for line in self.log if (m := re.search(r"n_nodes (\d+)", line))]
        self.nodes = int(found[-1][1]) if found else None


COUNTING: dict[str, type[GTPEngine]] = {
    "edax": CountingEdax,
    "egaroucid": CountingEgaroucid,
}
PLAIN: dict[str, type[GTPEngine]] = {"edax": Edax, "egaroucid": Egaroucid}


def process_cpu(pid: int) -> float | None:
    """Seconds of CPU a process has used, to a tick."""
    text = read_text(f"/proc/{pid}/stat")
    if text is None:
        return None
    fields = text.rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def engine_game(
    name: str, level: int, threads: int, opening: np.ndarray, count_nodes: bool = True
) -> list[dict[str, Any]]:
    """A game of an engine against itself at `level`, by fresh processes, a record per move."""
    kind = (COUNTING if count_nodes else PLAIN)[name]
    engines = [
        kind(
            kind.default_dir(),
            level,
            analyse=count_nodes,
            threads=threads,
        )
        for _ in range(2)
    ]
    records = []
    try:
        board = Board()
        for move in opening:
            color = board.color
            board.play(int(move))
            for engine in engines:
                engine.play(color, int(move))
        ply = len(opening)
        while board.legal():
            side = int(not board.black_to_move)
            color = board.color
            legal = len(board.legal())
            empties = 64 - (board.discs[0] | board.discs[1]).bit_count()
            engine = engines[side]
            cpu_begun = process_cpu(engine.process.pid)
            begun = time.perf_counter()
            reply = engine.command(f"genmove {color}")
            seconds = time.perf_counter() - begun
            cpu_ended = process_cpu(engine.process.pid)
            move = action(reply)
            if count_nodes:
                engine.read_analysis(move)
            records.append(
                {
                    "side": side,
                    "ply": ply,
                    "empties": empties,
                    "legal": legal,
                    "seconds": seconds,
                    "cpu": None
                    if cpu_begun is None or cpu_ended is None
                    else cpu_ended - cpu_begun,
                    "nodes": getattr(engine, "nodes", None),
                    "move": square(move),
                }
            )
            board.play(move)
            engines[1 - side].play(color, move)
            ply += 1
    finally:
        for engine in engines:
            engine.close()
    return records


def side_seconds(games: list[list[dict[str, Any]]]) -> list[float]:
    """One side's seconds of each side of each game."""
    out = []
    for game in games:
        out += [sum(m["seconds"] for m in game if m["side"] == s) for s in (0, 1)]
    return out


def cpu_ticks() -> dict[int, tuple[int, int]]:
    """Each CPU's (busy, all) ticks so far."""
    ticks = {}
    for line in (read_text("/proc/stat") or "").splitlines():
        found = re.match(r"cpu(\d+) (.*)", line)
        if found:
            fields = [int(field) for field in found[2].split()]
            idle = fields[3] + fields[4]
            ticks[int(found[1])] = sum(fields[:8]) - idle, sum(fields[:8])
    return ticks


def busy_shares(
    before: dict[int, tuple[int, int]], after: dict[int, tuple[int, int]]
) -> dict[str, float]:
    """How busy the contest's CPUs, their other threads and the rest were between two readings."""
    groups = {
        "contest": CONTEST_CPUS,
        "siblings": SIBLING_CPUS,
        "others": set(after) - CONTEST_CPUS - SIBLING_CPUS,
    }
    shares = {}
    for name, cpus in groups.items():
        busy = sum(after[c][0] - before[c][0] for c in cpus if c in after)
        total = sum(after[c][1] - before[c][1] for c in cpus if c in after)
        shares[name] = busy / max(total, 1)
    return shares


def others_running(cpus: set[int]) -> list[str]:
    """What's running on `cpus` right now and isn't this process or its children."""
    me = os.getpid()
    found = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        text = read_text(path)
        if text is None:
            continue
        comm = text[text.find("(") + 1 : text.rfind(")")]
        fields = text.rsplit(")", 1)[1].split()
        pid, state, parent, cpu = (
            int(path.parent.name),
            fields[0],
            int(fields[1]),
            int(fields[36]),
        )
        if state == "R" and cpu in cpus and me not in (pid, parent):
            found.append(f"{comm} on {cpu}")
    return found


def busy_cpus(
    before: dict[int, tuple[int, int]], after: dict[int, tuple[int, int]]
) -> dict[int, float]:
    """The CPUs busier than a job can start beside, and how busy, between two readings."""
    shares = {
        cpu: (after[cpu][0] - before[cpu][0]) / max(after[cpu][1] - before[cpu][1], 1)
        for cpu in after
    }
    return {cpu: share for cpu, share in shares.items() if share > BUSY_TO_START}


def machine_watch() -> Callable[[], dict[str, Any]]:
    """What reports, each time it's called, what the machine did since the last time."""
    last = cpu_ticks()

    def watch() -> dict[str, Any]:
        nonlocal last
        now = cpu_ticks()
        shares = busy_shares(last, now)
        last = now
        return {
            "busy": shares,
            "load": os.getloadavg()[0],
            "running": others_running(CONTEST_CPUS | SIBLING_CPUS),
        }

    return watch


def net_config(
    arch: str, sims: int, batch: int, games: int, max_seconds: float = 1500
) -> dict[str, Any]:
    return {
        "id": f"{arch}-{sims}-b{batch}",
        "kind": "net",
        "arch": arch,
        "sims": sims,
        "batch": batch,
        "games": games,
        "max_seconds": max_seconds,
    }


def engine_config(
    name: str, level: int, games: int, max_seconds: float = 1500, threads: int = THREADS
) -> dict[str, Any]:
    return {
        "id": f"{name}-L{level}-t{threads}",
        "kind": "engine",
        "engine": name,
        "level": level,
        "threads": threads,
        "games": games,
        "max_seconds": max_seconds,
    }


def estimated_seconds(config: dict[str, Any]) -> float:
    """About how long a config takes, to order and budget the plan."""
    if config["kind"] == "net":
        rate = NET_SPEED[config["arch"]] * BATCH_EVALS[config["batch"]]
        seconds = config["games"] * 51 * config["sims"] * 0.8 / rate
    else:
        # one side's seconds at L16 on 8 threads, and how much more a level above it takes
        seconds = config["games"] * 2 * 0.25 * 1.35 ** (config["level"] - 16)
        seconds *= THREADS / config["threads"]
    return min(seconds, config["max_seconds"])


def make_plan() -> list[dict[str, Any]]:
    """Every config of the main run, cheapest first, so that what's cheap is there early."""
    configs = []
    for arch in NETWORKS:
        rows = [(128, 1, 20), (128, 4, 12), (128, 32, 8)]
        rows += [(1000, 1, 12), (1000, 16, 20), (1000, 32, 8)]
        rows += [(4000, 1, 6), (4000, 16, 8), (4000, 32, 8)]
        rows += [(16000, 32, 2 if arch == "10x128" else 4)]
        if arch != "10x128":
            rows += [(1, 1, 20), (8, 1, 20), (32, 1, 20)]
        if arch == "4x64":
            rows += [(450, 1, 12)]
        if arch in ("4x64", "8x128"):
            rows += [(16000, 1, 1)]
        configs += [net_config(arch, *row) for row in rows]
    for name, levels in ENGINE_LEVELS.items():
        for level in levels:
            # the deepest levels are spiky: as many games as a limit allows
            configs.append(
                engine_config(name, level, 40, 1800 if level >= 24 else 1500)
            )
        configs += [engine_config(name, 20, 8, threads=t) for t in (1, 2, 4)]
    return sorted(configs, key=estimated_seconds)


def references() -> list[dict[str, Any]]:
    """Configs run again and again through a job: nets are deterministic, so what changes in them
    is the machine's speed."""
    return [
        net_config("8x128", 1000, 16, 4),
        net_config("4x64", 128, 1, 4),
        engine_config("edax", 16, 8),
        engine_config("egaroucid", 16, 8),
    ]


def plotted_batch(sims: int) -> int:
    """The batch a net's point is plotted at: the largest with no strength lost where it's been
    measured, and as if it scaled with visits elsewhere (`batch_strength` says which)."""
    if sims < 1000:
        return 1
    return 16 if sims < 4000 else 32


def batch_strength(arch: str, sims: int, batch: int) -> str:
    """What's known of the strength a batch costs. Measured on 8x128 only."""
    if batch == 1:
        return "sequential"
    if arch == "8x128" and (sims, batch) in ((1000, 16), (4000, 32)):
        return "measured_no_loss"
    return "unmeasured"


def summarize_config(
    config: dict[str, Any], games: list[list[dict[str, Any]]]
) -> dict[str, Any]:
    """What a config's games came to. A game has two sides, each one sample of a side's time."""
    moves = [m for game in games for m in game]
    seconds = [m["seconds"] for m in moves]
    sides = side_seconds(games)
    # the two sides of a game aren't independent: the error is from the games'
    means = [sum(m["seconds"] for m in game) / 2 for game in games]
    out: dict[str, Any] = {
        "games": len(games),
        "threads": config.get("threads", THREADS),
        "seconds_per_game": statistics.mean(sides),
        "seconds_per_game_se": statistics.stdev(means) / len(means) ** 0.5
        if len(means) > 1
        else None,
        "moves_per_game": len(moves) / len(sides),
        "seconds_per_move": statistics.mean(seconds),
        "p50_move": float(np.percentile(seconds, 50)),
        "p95_move": float(np.percentile(seconds, 95)),
        "p99_move": float(np.percentile(seconds, 99)),
        "max_move": max(seconds),
    }
    if config["kind"] == "net":
        total = sum(seconds)
        evals = sum(m["evals"] for m in moves)
        out |= {
            "evals_per_move": evals / len(moves),
            "evals_per_visit": evals / sum(m["visits"] for m in moves),
            "evals_per_s": evals / total,
            "tree_share": sum(m["tree_seconds"] for m in moves) / total,
            "net_share": sum(m["net_seconds"] for m in moves) / total,
        }
    else:
        nodes = [m["nodes"] for m in moves if m["nodes"] is not None]
        out["nodes_per_move"] = statistics.mean(nodes) if nodes else None
    return out


def startup(job: str) -> dict[str, Any]:
    """What an attempt found, and how the machine scales on work that fills a core."""
    out = sysinfo()
    with tempfile.TemporaryDirectory() as directory:
        fma = build_spin(directory)
        out["fma_scaling"] = [spin(fma, n, 5.0) for n in (1, 2, 4, 8)]
    path = net_path("8x128")
    bf16 = openvino_info(path, THREADS, THREADS, "bf16")
    out["precision"] = "bf16" if "BF16" in bf16["capabilities"] else "f32"
    out["openvino"] = openvino_info(path, THREADS, THREADS, out["precision"])
    leaves = real_leaves(
        openvino_eval(str(path), THREADS, out["precision"]), make_openings(1)[0], 1024
    )
    out["net_scaling"] = [
        {
            "batch": batch,
            "threads": threads,
            "streams": streams,
            "evals_per_s": net_only(
                openvino_eval(str(path), threads, out["precision"], streams),
                leaves,
                batch,
                4.0,
            ),
        }
        for batch, threads, streams in [
            *[(1, t, 1) for t in (1, 2, 4, 8)],
            *[(16, t, t) for t in (1, 2, 4, 8)],
        ]
    ]
    out["job"] = job
    return out


class OutOfTime(Exception):
    """The job's limit, or a stop asked for: what's finished is kept."""


def run_configs(
    store: Store,
    job: str,
    attempt: int,
    plan: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    play: Callable[[dict[str, Any], int], list[dict[str, Any]]],
    limit: float,
    used: float = 0.0,
    every: float = REFERENCE_EVERY,
    clock: Callable[[], float] = time.time,
    watch: Callable[[], dict[str, Any]] = dict,
) -> None:
    """The configs not yet finished, in order, with `refs` at the start, every `every` seconds and
    at the end. Each game's moves are written as it ends. What an attempt left unfinished is
    done again, whole, by the next."""
    begun = clock()
    finished = store.sizes(timing_key(job, "done/"))
    rounds = 0

    def spent() -> float:
        return used + clock() - begun

    def status(state: str, doing: str | None = None) -> None:
        put_json(
            store,
            timing_key(job, "status.json"),
            {
                "state": state,
                "attempt": attempt,
                "doing": doing,
                "seconds": spent(),
                "heartbeat": clock(),
            },
        )

    def one(config: dict[str, Any], folder: str, name: str) -> None:
        started = clock()
        games: list[list[dict[str, Any]]] = []
        machine = []
        key = timing_key(job, "moves", str(attempt), f"{name}.jsonl")
        for index in range(config["games"]):
            stopped = get_json(store, timing_key(job, "stop.json"))
            if spent() > limit or stopped:
                status("out of time" if spent() > limit else "stopped")
                raise OutOfTime
            if games and clock() - started > config["max_seconds"]:
                break
            status("running", name)
            watch()
            games.append(play(config, index))
            machine.append(watch())
            lines = [
                json.dumps({"game": g, **move})
                for g, game in enumerate(games)
                for move in game
            ]
            store.put_bytes(key, "\n".join(lines).encode())
        put_json(
            store,
            timing_key(job, folder, f"{name}.json"),
            {
                "config": config,
                "attempt": attempt,
                "began": started,
                "ended": clock(),
                "summary": summarize_config(config, games),
                "machine": machine,
            },
        )

    def round_of_references() -> None:
        nonlocal rounds
        for config in refs:
            one(config, "references", f"a{attempt}-r{rounds}-{config['id']}")
        rounds += 1

    try:
        round_of_references()
        last = clock()
        for config in plan:
            if timing_key(job, "done", f"{config['id']}.json") in finished:
                continue
            if clock() - last > every:
                round_of_references()
                last = clock()
            one(config, "done", config["id"])
        round_of_references()
    except OutOfTime:
        return
    status("done")


def make_play(precision: str) -> Callable[[dict[str, Any], int], list[dict[str, Any]]]:
    """What plays a config's games, from the same openings for every config."""
    openings = make_openings(max(c["games"] for c in make_plan()))
    evals: dict[tuple[str, int], Eval] = {}

    def play(config: dict[str, Any], index: int) -> list[dict[str, Any]]:
        if config["kind"] == "engine":
            return engine_game(
                config["engine"], config["level"], config["threads"], openings[index]
            )
        key = config["arch"], config["batch"]
        if key not in evals:
            # one stream to a core is the fastest way found at batches of 8 and up, one in all
            # at a batch of 1
            streams = 1 if config["batch"] == 1 else min(THREADS, config["batch"])
            evals[key] = openvino_eval(
                str(net_path(config["arch"])), THREADS, precision, streams
            )
        settings = SearchSettings(batch=config["batch"], virtual_loss="visit")
        return net_game(evals[key], settings, config["sims"], openings[index])

    return play


def work(args: argparse.Namespace) -> None:
    """The main run. Each attempt says what it found before it starts."""
    if os.sched_getaffinity(0) != CONTEST_CPUS:
        raise SystemExit(
            f"run this under taskset -c 16-23, not {os.sched_getaffinity(0)}"
        )
    before = cpu_ticks()
    time.sleep(QUIET_SECONDS)
    busy = busy_cpus(before, cpu_ticks())
    if busy and not args.not_quiet:
        running = others_running(set(range(os.cpu_count() or 0)))
        raise SystemExit(f"the machine isn't quiet: {busy}, {running}")
    store = DirStore(TIMING_DIR)
    attempt = len(store.sizes(timing_key(args.job, "attempts/")))
    status = get_json(store, timing_key(args.job, "status.json")) or {}
    info = startup(args.job)
    put_json(store, timing_key(args.job, "attempts", f"{attempt}.json"), info)
    print(f"attempt {attempt}, {info['precision']}", flush=True)
    run_configs(
        store,
        args.job,
        attempt,
        make_plan(),
        references(),
        make_play(info["precision"]),
        args.hours * 3600,
        status.get("seconds", 0.0),
        watch=machine_watch(),
    )


def machine_note(
    record: dict[str, Any], game_seconds: list[float] | None = None
) -> str | None:
    """Whether the machine was too busy during a config for its timings to stand, its games counted
    by how long they took: a game of milliseconds reads little from a CPU's ticks."""
    windows = [w["busy"] for w in record.get("machine", []) if "busy" in w]
    if not windows:
        return None
    weights = game_seconds or [1.0] * len(windows)
    notes = []
    for name, limit in (("siblings", BUSY_SIBLINGS), ("others", BUSY_OTHERS)):
        share = sum(w * b[name] for w, b in zip(weights, windows)) / sum(weights)
        if share > limit:
            notes.append(f"{name} {share:.0%} busy")
    return "; ".join(notes) or None


def reference_drift(records: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    """Each reference's seconds a game through an attempt, by its id, relative to its first."""
    series: dict[str, list[dict[str, float]]] = {}
    for record in sorted(records, key=lambda r: r["began"]):
        key = f"a{record['attempt']}-{record['config']['id']}"
        rows = series.setdefault(key, [])
        seconds = record["summary"]["seconds_per_game"]
        first = rows[0]["seconds_per_game"] if rows else seconds
        rows.append(
            {
                "began": record["began"],
                "seconds_per_game": seconds,
                "relative": seconds / first,
            }
        )
    return series


def drift_note(
    record: dict[str, Any], references: list[dict[str, Any]], limit: float = 1.10
) -> str | None:
    """Whether a net reference run moved by over `limit` between the rounds either side of a
    config, or None. Engine references are too noisy to say so from a few games."""
    notes = []
    for key, rows in reference_drift(
        [
            r
            for r in references
            if r["attempt"] == record["attempt"] and r["config"]["kind"] == "net"
        ]
    ).items():
        before = [r for r in rows if r["began"] <= record["began"]][-1:]
        after = [r for r in rows if r["began"] >= record["ended"]][:1]
        if before and after:
            ratio = after[0]["seconds_per_game"] / before[0]["seconds_per_game"]
            if max(ratio, 1 / ratio) > limit:
                notes.append(f"{key} {ratio:.2f}x")
    return "; ".join(notes) or None


def fit_cost(
    points: list[tuple[float, float]], at: float
) -> tuple[float, float, float]:
    """Seconds a move = overhead + cost x work, weighted to fit each point's share, not its size.

    Args:
        points: (work, seconds a move), where work is sims or evals a move
        at: the work to predict for

    Returns:
        the overhead, the cost per unit of work and the seconds predicted at `at`
    """
    work = np.array([p[0] for p in points], float)
    seconds = np.array([p[1] for p in points], float)
    design = np.stack([np.ones_like(work), work], axis=1) / seconds[:, None]
    (overhead, cost), *_ = np.linalg.lstsq(design, np.ones_like(work), rcond=None)
    return float(overhead), float(cost), float(overhead + cost * at)


def held_out_fits(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each (architecture, batch) timed at `HELD_OUT` sims and at those below: what fitting the
    rest predicted for it, from sims alone and from the evals it really took."""
    by_key: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for record in records:
        config = record["config"]
        if config["kind"] == "net":
            key = config["arch"], config["batch"]
            by_key.setdefault(key, {})[config["sims"]] = record["summary"]
    out = []
    for (arch, batch), rows in sorted(by_key.items()):
        fitted = [rows[s] for s in FIT_SIMS if s in rows]
        if HELD_OUT not in rows or len(fitted) < len(FIT_SIMS):
            continue
        real = rows[HELD_OUT]
        _, per_sim, by_sims = fit_cost(
            [(s, rows[s]["seconds_per_move"]) for s in FIT_SIMS], HELD_OUT
        )
        overhead, per_eval, by_evals = fit_cost(
            [(r["evals_per_move"], r["seconds_per_move"]) for r in fitted],
            real["evals_per_move"],
        )
        out.append(
            {
                "arch": arch,
                "batch": batch,
                "measured_seconds_per_move": real["seconds_per_move"],
                "seconds_per_sim": per_sim,
                "by_sims": by_sims,
                "by_sims_error": by_sims / real["seconds_per_move"] - 1,
                "overhead": overhead,
                "seconds_per_eval": per_eval,
                "by_evals": by_evals,
                "by_evals_error": by_evals / real["seconds_per_move"] - 1,
            }
        )
    return out


def league_config(
    record: dict[str, Any],
    specs: dict[str, dict[str, str]],
    references: list[dict[str, Any]],
    game_seconds: list[float] | None = None,
) -> dict[str, Any]:
    """A config as the league's panel reads it (league/timings.json)."""
    config, summary = record["config"], record["summary"]
    notes = [drift_note(record, references), machine_note(record, game_seconds)]
    drift = {
        "source": "measured",
        "drift": "; ".join(note for note in notes if note) or None,
    }
    if config["kind"] == "net":
        arch, sims, batch = config["arch"], config["sims"], config["batch"]
        return {
            "net": specs[arch]["net"],
            "sims": sims,
            "batch": batch,
            "virtual_loss": "visit",
            "batch_strength": batch_strength(arch, sims, batch),
            "plot": batch == plotted_batch(sims),
            **drift,
            **summary,
        }
    return {
        "engine": config["engine"],
        "level": config["level"],
        "plot": config["threads"] == THREADS,
        **drift,
        **summary,
    }


def game_seconds(raw: str, record: dict[str, Any]) -> list[float] | None:
    """How long each game of a finished config took, from its moves, if they're here."""
    path = Path(raw) / str(record["attempt"]) / f"{record['config']['id']}.jsonl"
    if not path.exists():
        return None
    seconds: dict[int, float] = {}
    for line in path.read_text().splitlines():
        move = json.loads(line)
        seconds[move["game"]] = seconds.get(move["game"], 0.0) + move["seconds"]
    return [seconds[game] for game in sorted(seconds)]


def league_run(
    job: str,
    info: dict[str, Any],
    records: list[dict[str, Any]],
    references: list[dict[str, Any]],
    specs: dict[str, dict[str, str]],
    raw: str,
) -> dict[str, Any]:
    """A job as the panel reads it: one machine, whatever attempts it took."""
    return {
        "name": f"{info['hostname']}-8t-16-23-{job}",
        "raw": raw,
        "hardware": {
            "provider": "local",
            "cpu": info["cpuinfo"]["model"],
            "host_id": info["hostname"],
            "cores": THREADS,
            "threads": THREADS,
            "flags": info["cpuinfo"]["flags"],
            "memory": info["memory"][0],
            "kernel": info["kernel"],
            "cache": info["caches"],
            "smt_layout": info["thread_layout"],
        },
        "software": {
            **ENGINE_BUILDS,
            "openvino": info["openvino"]["version"],
            "openvino_precision": info["precision"],
        },
        "configs": [
            league_config(r, specs, references, game_seconds(raw, r)) for r in records
        ],
    }


def write_atomically(path: Path, text: str) -> None:
    partial = path.with_suffix(f".{os.getpid()}.tmp")
    partial.write_text(text)
    partial.replace(path)


def read_records(folder: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]


def summarize_job(args: argparse.Namespace) -> None:
    """A pulled job's table, fit and drift, and with --publish the league's file."""
    local = TIMING_DIR / args.job
    records = read_records(local / "done")
    refs = read_records(local / "references")
    info = json.loads((local / "attempts" / "0.json").read_text())
    specs = json.loads((ONNX_DIR / "nets.json").read_text())
    run = league_run(args.job, info, records, refs, specs, str(local / "moves"))
    summary = {
        "runs": [run],
        "fits": held_out_fits(records),
        "drift": reference_drift(refs),
    }
    write_atomically(local / "summary.json", json.dumps(summary, indent=1))
    if args.publish:
        write_atomically(LEAGUE_TIMINGS, json.dumps({"runs": [run]}, indent=1))
    for record in records:
        s = record["summary"]
        print(
            f"{record['config']['id']:22} a{record['attempt']} games {s['games']:3} "
            f"s/game {s['seconds_per_game']:8.3f} s/move {s['seconds_per_move']:7.4f} "
            f"p95 {s['p95_move']:7.3f} max {s['max_move']:7.3f}"
        )
    for fit in summary["fits"]:
        print(
            f"fit {fit['arch']} b{fit['batch']}: by sims {fit['by_sims_error']:+.1%}, "
            f"by evals {fit['by_evals_error']:+.1%}"
        )
    for key, rows in summary["drift"].items():
        print(key, " ".join(f"{row['relative']:.3f}" for row in rows))


def show_plan(_args: argparse.Namespace) -> None:
    plan = make_plan()
    total = sum(estimated_seconds(c) for c in plan)
    rounds = 2 + total // REFERENCE_EVERY
    total += rounds * sum(estimated_seconds(c) for c in references())
    for config in plan:
        print(
            f"{config['id']:22} games {config['games']:3} about {estimated_seconds(config):6.0f} s"
        )
    print(f"{len(plan)} configs, about {total / 3600:.1f} h with {rounds:.0f} rounds of references")  # fmt: skip


def timing_key(job: str, *rest: str) -> str:
    return "/".join((job, *rest))


def prepare(_args: argparse.Namespace) -> None:
    """The networks as ONNX, here, for the image, and what the league calls their architectures."""
    from bench_cpu_backends import network_path
    from export_onnx import export_checkpoint

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    specs = {}
    for arch, name in NETWORKS.items():
        path, net = network_path(name)
        specs[arch] = {"network": name, "net": net}
        target = ONNX_DIR / f"{arch}.onnx"
        if not target.exists():
            target.write_bytes(export_checkpoint(path, net).SerializeToString())
        print(f"{arch}: {name} {target.stat().st_size} bytes")
    (ONNX_DIR / "nets.json").write_text(json.dumps(specs, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("prepare", help="the networks as ONNX, once")
    p.set_defaults(fn=prepare)

    p = commands.add_parser("work", help="the main run, under taskset -c 16-23")
    p.add_argument("job")
    p.add_argument("--hours", type=float, required=True, help="over all attempts")
    p.add_argument(
        "--not-quiet", action="store_true", help="start even with other work running"
    )
    p.set_defaults(fn=work)

    p = commands.add_parser("plan", help="the main run's configs, and how long")
    p.set_defaults(fn=show_plan)

    p = commands.add_parser("summarize", help="a pulled job's table, fit and drift")
    p.add_argument("job")
    p.add_argument("--publish", action="store_true", help="and league/timings.json")
    p.set_defaults(fn=summarize_job)

    args = parser.parse_args()
    args.fn(args)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
