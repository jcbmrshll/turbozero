"""Builds an engine-labelled set (see engine_set.py): positions from the games of the evals store
(evals.db: networks' and engines' evaluation games, which nothing trains on), each legal move of
each scored by Neural Reversi.

    # 1. sample the positions: 9000 from networks' 30 s games against engines, and so on
    uv run examples/othello/label_positions.py positions --out ~/.cache/.../positions.npz \\
        --source 'vs-engine-ray11=1800:net-engine:ray11x64' \\
        --source 'engines=5000:engine-engine' \\
        --source 'ray11-ray9n=4000:net-net:ray11x64/ray9x64n'
    # 2. score them, on Modal's CPUs (a container of 8 cores a host), or here with `label`
    uv run examples/othello/label_positions.py cloud ~/.cache/.../positions.npz --name es2 \\
        --hosts 100 --out ~/.cache/.../labels/results.jsonl
    # 3. the set
    uv run examples/othello/label_positions.py assemble ~/.cache/.../positions.npz \\
        ~/.cache/.../labels/results-files --out ~/.cache/turbozero/othello/engine-sets/new.npz

A source is NAME=COUNT:KIND[:A[/B]], its positions from the games of one kind of match:
`net-engine` (a network against an engine on a --clock, the network's label containing A),
`engine-engine` (two engines on a --clock), or `net-net` (two networks, one's label containing A
and the other's B). From each it samples COUNT positions worth scoring (see
`engine_set.selectable`) with --min-empties to 51 empty squares, spread over the empties as
--bands asks (over the bands its games reach) and at most --per-game from a game (4), none twice.

Neural Reversi at --level 26 scores every legal move: exactly (to the end of the game, no
pruning) up to about 29 empty squares, 26 moves ahead with ProbCut beyond. On one core a
position takes about 8 s at 21-26 empties, 20-160 s at 27-29, and 4 s past that.
"""

import argparse
import json
import random
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from engine_set import ILLEGAL, OPENING_EMPTIES, EngineSet, selectable
from engines import CACHE_DIR, Board, MoveScores, NeuralReversiAnalyst
from exact_labels import bitboards

EVALS_DB = CACHE_DIR / "evals" / "evals.db"
LEVEL = 26
# empty squares from which positions are bands of, and the share of a source's positions each
# band gets by default: most where games against engines are decided
BANDS = ((21, 27), (28, 35), (36, 43), (44, 51))
BAND_SHARES = (0.20, 0.35, 0.30, 0.15)
KINDS = ("net-engine", "engine-engine", "net-net")


class Analyst(Protocol):
    """Scores every legal move of a position given as bitboards (the mover's discs, the
    opponent's): `NeuralReversiAnalyst`, or a fake in tests."""

    def analyse(self, me: int, opp: int) -> MoveScores: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Source:
    """Where some of a set's positions come from: games of one kind of match in the store.

    Attributes:
        name: names its metrics: letters, digits, "-" and "_"
        count: positions to take
        kind: one of KINDS
        a, b: what the players' labels contain (see the module's docstring)
    """

    name: str
    count: int
    kind: str
    a: str = ""
    b: str = ""

    def description(self, clock: float) -> str:
        what = {
            "net-engine": f"a network ({self.a!r}) against an engine at {clock:g} s",
            "engine-engine": f"engines against each other at {clock:g} s",
            "net-net": f"networks ({self.a!r} against {self.b!r})",
        }[self.kind]
        return f"evals.db games: {what}"


def parse_source(text: str) -> Source:
    """NAME=COUNT:KIND[:A[/B]] -> a Source."""
    name, sep, rest = text.partition("=")
    parts = rest.split(":")
    if not sep or len(parts) not in (2, 3) or parts[1] not in KINDS:
        raise argparse.ArgumentTypeError(
            f"a source is NAME=COUNT:KIND[:A[/B]], KIND one of {KINDS}, not {text!r}"
        )
    a, _, b = (parts[2] if len(parts) == 3 else "").partition("/")
    return Source(name, int(parts[0]), parts[1], a, b)


def source_games(
    db: str | Path, sources: list[Source], clock: float
) -> dict[str, list[tuple[int, list[int]]]]:
    """Each source's games in the store `db` (read only): (game id, moves from the standard
    start), by source name. A game goes to the first source it fits."""
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    players = {
        pid: (label, kind, json.loads(config))
        for pid, label, kind, config in connection.execute(
            "SELECT id, label, kind, config FROM players"
        )
    }
    on_clock = {"clock": {"ext": None, "inc": 0.0, "ini": clock}}

    def fits(source: Source, black: str, white: str) -> bool:
        (la, ka, ca), (lb, kb, cb) = players[black], players[white]
        kinds = sorted((ka, kb))
        if source.kind == "engine-engine":
            return kinds == ["engine", "engine"] and ca.get("budget") == on_clock
        if source.kind == "net-engine":
            if kinds != ["engine", "net"]:
                return False
            net, engine = (la, cb) if ka == "net" else (lb, ca)
            return engine.get("budget") == on_clock and source.a in net
        return kinds == ["net", "net"] and (
            (source.a in la and source.b in lb) or (source.a in lb and source.b in la)
        )

    games: dict[str, list[tuple[int, list[int]]]] = {s.name: [] for s in sources}
    for gid, black, white, moves in connection.execute(
        "SELECT id, black_id, white_id, moves FROM games ORDER BY id"
    ):
        for source in sources:
            if fits(source, black, white):
                games[source.name].append((gid, json.loads(moves)))
                break
    connection.close()
    return games


def game_positions(moves: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each position a game passes through, as (obs (8, 8, 2) bool, legal (65,) bool), the
    player to move's discs first."""
    board = Board()
    out = []
    for move in moves:
        mover = 0 if board.black_to_move else 1
        me, opp = board.discs[mover], board.discs[1 - mover]
        obs = np.zeros((64, 2), bool)
        obs[:, 0] = [me >> i & 1 for i in range(64)]
        obs[:, 1] = [opp >> i & 1 for i in range(64)]
        legal = np.zeros(65, bool)
        legal[board.legal()] = True
        out.append((obs.reshape(8, 8, 2), legal))
        # the store's games have no passes: `play` passes for a player without a move
        board.play(move)
    return out


def sample(
    games: list[tuple[int, list[int]]],
    count: int,
    rng: random.Random,
    taken: set[bytes],
    min_empties: int = 21,
    per_game: int = 4,
    bands: tuple[tuple[int, int], ...] = BANDS,
    shares: tuple[float, ...] = BAND_SHARES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`count` positions from `games`, each worth scoring, with `min_empties` to 51 empty
    squares and not in `taken` (which gets them), at most `per_game` from a game, spread over
    `bands` by `shares` (renormalized over the bands the games reach, e.g. none under 28 in
    games cut there). Games are visited in a random order, so a large source is only read as
    far as needed.

    Returns:
        tuple: obs (count, 8, 8, 2) bool, legal (count, 65) bool, game (count,) int64
    """
    order = list(range(len(games)))
    rng.shuffle(order)
    # what the games reach, judged from the first few hundred
    reached = set()
    for i in order[:300]:
        for obs, legal in game_positions(games[i][1]):
            empties = 64 - int(obs.sum())
            reached |= {b for b, (lo, hi) in enumerate(bands) if lo <= empties <= hi}
    reached = sorted(b for b in reached if bands[b][1] >= min_empties)
    total = sum(shares[b] for b in reached)
    quota: dict[int, int] = {b: round(count * shares[b] / total) for b in reached}
    quota[reached[-1]] += count - sum(quota.values())
    picked: list[tuple[np.ndarray, np.ndarray, int]] = []
    per = dict.fromkeys(range(len(games)), 0)
    # a pass by the bands' quotas, then, if games ran out in some band, one taking from any
    for by_band in (True, False):
        if not by_band:
            # -1: any band
            quota = {-1: sum(quota.values())}
        for i in order:
            if not any(quota.values()):
                break
            if per[i] == per_game:
                continue
            gid, moves = games[i]
            positions = game_positions(moves)
            obs = np.stack([o for o, _ in positions])
            legal = np.stack([m for _, m in positions])
            empties = 64 - obs.sum(axis=(1, 2, 3))
            ok = selectable(obs, legal) & (empties >= min_empties)
            candidates = [
                int(j) for j in np.nonzero(ok & (empties < OPENING_EMPTIES))[0]
            ]
            rng.shuffle(candidates)
            for j in candidates:
                band = next(
                    (b for b, (lo, hi) in enumerate(bands) if lo <= empties[j] <= hi),
                    None,
                )
                key = np.packbits(obs[j]).tobytes()
                if by_band and (band is None or not quota.get(band)) or key in taken:
                    continue
                taken.add(key)
                quota[band if by_band and band is not None else -1] -= 1
                picked.append((obs[j], legal[j], gid))
                per[i] += 1
                if per[i] == per_game or not any(quota.values()):
                    break
    if any(quota.values()):
        raise ValueError(
            f"{len(games)} games give {len(picked)} positions, not {count}"
        )
    return (
        np.stack([o for o, _, _ in picked]),
        np.stack([m for _, m, _ in picked]),
        np.array([g for _, _, g in picked], np.int64),
    )


def save_positions(path: Path, obs, legal, game, source, names, descriptions) -> None:
    np.savez(
        path,
        obs=np.packbits(obs.reshape(len(obs), -1), axis=1),
        mask=np.packbits(legal, axis=1),
        game=game,
        source=source.astype(np.int8),
        source_names=np.array(names),
        source_descriptions=np.array(descriptions),
    )


def load_positions(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        n = len(f["game"])
        return {
            "obs": np.unpackbits(f["obs"], axis=1)[:, :128]
            .reshape(n, 8, 8, 2)
            .astype(bool),
            "legal": np.unpackbits(f["mask"], axis=1)[:, :65].astype(bool),
            **{
                k: f[k]
                for k in ("game", "source", "source_names", "source_descriptions")
            },
        }


def label(
    obs: np.ndarray,
    workers: int,
    analyst: Callable[[], Analyst],
    progress: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every legal move of each position scored, by `workers` analysts from `analyst` at once,
    each taking the next position as it finishes one.

    Returns:
        tuple: scores (N, 65) int16 (ILLEGAL where not scored), depth (N,) int8, exact (N,) bool
    """
    me, opp = bitboards(obs)
    n = len(obs)
    scores = np.full((n, 65), ILLEGAL, np.int16)
    depth = np.zeros(n, np.int8)
    exact = np.zeros(n, bool)
    lock = threading.Lock()
    nxt = [0]

    def work() -> None:
        scorer = analyst()
        try:
            while True:
                with lock:
                    i = nxt[0]
                    nxt[0] += 1
                if i >= n:
                    return
                found = scorer.analyse(int(me[i]), int(opp[i]))
                for move, score in found.scores.items():
                    scores[i, move] = score
                depth[i], exact[i] = found.depth, found.exact
                if progress is not None:
                    with lock:
                        progress(i)
        finally:
            scorer.close()

    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(work) for _ in range(workers)]:
            future.result()
    return scores, depth, exact


def neural_reversi(level: int, engine_dir: Path | None) -> Callable[[], Analyst]:
    return lambda: NeuralReversiAnalyst(level, engine_dir)


def assemble(
    positions: dict[str, np.ndarray], labels: list[Path], level: int
) -> EngineSet:
    """The set from the positions and their label files (`label`'s outputs, which together
    cover every position once)."""
    n = len(positions["game"])
    scores = np.full((n, 65), ILLEGAL, np.int16)
    depth = np.zeros(n, np.int8)
    exact = np.zeros(n, bool)
    done = np.zeros(n, bool)
    for path in labels:
        with np.load(path) as f:
            index = f["index"]
            if done[index].any():
                raise ValueError(f"{path} labels positions already labelled")
            scores[index], depth[index], exact[index] = (
                f["scores"],
                f["depth"],
                f["exact"],
            )
            done[index] = True
    if not done.all():
        raise ValueError(f"{(~done).sum()} positions have no labels")
    obs = positions["obs"]
    return EngineSet(
        obs=obs,
        legal=positions["legal"],
        scores=scores,
        depth=depth,
        exact=exact,
        labeller=np.zeros(n, np.int8),
        labeller_names=[f"neural-reversi level {level}"],
        source=positions["source"],
        source_names=[str(s) for s in positions["source_names"]],
        source_descriptions=[str(s) for s in positions["source_descriptions"]],
        game=positions["game"],
        empties=(64 - obs.sum(axis=(1, 2, 3))).astype(np.int8),
    )


def in_cache(path: str) -> Path:
    """A path as given, or else under CACHE_DIR (how a cloud host is told where its files are)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else CACHE_DIR / p


def cmd_positions(args: argparse.Namespace) -> None:
    names = [s.name for s in args.source]
    if len(set(names)) != len(names):
        raise SystemExit("source names must be unique")
    games = source_games(args.db, args.source, args.clock)
    rng = random.Random(args.seed)
    taken: set[bytes] = set()
    obs, legal, game, source = [], [], [], []
    for i, s in enumerate(args.source):
        o, m, g = sample(
            games[s.name], s.count, rng, taken, args.min_empties, args.per_game
        )
        print(f"{s.name}: {len(o)} positions from {len(set(g))} of {len(games[s.name])} games",
              flush=True)  # fmt: skip
        for parts, part in zip((obs, legal, game), (o, m, g), strict=True):
            parts.append(part)
        source.append(np.full(len(o), i, np.int8))
    obs, legal, game, source = (np.concatenate(x) for x in (obs, legal, game, source))
    # shuffled, so that any range of them is a fair share of the work
    order = np.random.default_rng(args.seed).permutation(len(obs))
    save_positions(
        Path(args.out), obs[order], legal[order], game[order], source[order], names,
        [s.description(args.clock) for s in args.source],
    )  # fmt: skip
    print(f"{args.out}: {len(obs)} positions", flush=True)


def cmd_label(args: argparse.Namespace) -> None:
    positions = load_positions(in_cache(args.positions))
    last = len(positions["game"]) if args.last is None else args.last
    index = np.arange(args.first, last)
    start = time.perf_counter()
    count = [0]

    def progress(_: int) -> None:
        count[0] += 1
        if count[0] % 25 == 0:
            print(json.dumps({"done": count[0], "of": len(index),
                              "seconds": round(time.perf_counter() - start)}), flush=True)  # fmt: skip

    engine_dir = in_cache(args.engine_dir) if args.engine_dir else None
    scores, depth, exact = label(
        positions["obs"][index],
        args.workers,
        neural_reversi(args.level, engine_dir),
        progress,
    )
    missing = (scores != ILLEGAL) != positions["legal"][index]
    if missing.any():
        raise SystemExit(
            f"{missing.any(axis=1).sum()} positions' scored moves aren't their legal moves"
        )
    np.savez(args.out, index=index, scores=scores, depth=depth, exact=exact)
    print(json.dumps({"positions": len(index), "exact": int(exact.sum()),
                      "seconds": round(time.perf_counter() - start, 1)}), flush=True)  # fmt: skip


def cmd_cloud(args: argparse.Namespace) -> None:
    from gcp_cpu_jobs import launch
    from modal_hosts import ModalHosts
    from modal_jobs import open_store

    path = Path(args.positions).expanduser().resolve()
    relative = str(path.relative_to(CACHE_DIR.resolve()))
    engine = (
        Path(args.engine_dir).expanduser().resolve().relative_to(CACHE_DIR.resolve())
    )
    n = len(load_positions(path)["game"])
    jobs = []
    parts = args.hosts * args.jobs_per_host
    for k in range(parts):
        first, last = n * k // parts, n * (k + 1) // parts
        if args.limit is not None:
            last = min(last, first + args.limit)
        out = f"labels-{k}.npz"
        jobs.append({
            "name": f"{args.name}-{k}",
            "cmd": ["python", "label_positions.py", "label", relative, "--first", str(first),
                    "--last", str(last), "--workers", str(args.cpu), "--level", str(args.level),
                    "--engine-dir", str(engine), "--out", out],
            "outputs": [out],
        })  # fmt: skip
    store, config = open_store()
    launch(
        store,
        ModalHosts("CPU", args.cpu, args.memory),
        args.name,
        jobs,
        Path(args.out).expanduser(),
        vms=args.hosts,
        hours=args.hours,
        files=[relative, f"{engine}/target/release/cli"],
        bucket=config["bucket"],
        with_engines=False,
    )


def cmd_assemble(args: argparse.Namespace) -> None:
    labels = sorted(p for d in args.labels for p in Path(d).rglob("labels-*.npz"))
    engine_set = assemble(load_positions(args.positions), labels, args.level)
    engine_set.save(args.out)
    print(
        json.dumps(
            {"out": args.out, "positions": len(engine_set), **engine_set.describe()}
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Builds an engine-labelled set of positions."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "positions", help="sample positions from the evals store's games"
    )
    p.add_argument("--source", type=parse_source, action="append", required=True,
                   metavar="NAME=COUNT:KIND[:A[/B]]")  # fmt: skip
    p.add_argument("--db", default=str(EVALS_DB))
    p.add_argument(
        "--clock", type=float, default=30.0, help="engine games' seconds a side"
    )
    p.add_argument("--min-empties", type=int, default=21)
    p.add_argument("--per-game", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("label", help="score a range of positions here")
    p.add_argument("positions", help="a positions file (under the cache if relative)")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=None)
    p.add_argument(
        "--workers", type=int, default=1, help="Neural Reversi processes, a core each"
    )
    p.add_argument("--level", type=int, default=LEVEL)
    p.add_argument(
        "--engine-dir",
        default=None,
        help="a Neural Reversi build (under the cache if relative)",
    )
    p.add_argument("--out", required=True)
    p = sub.add_parser("cloud", help="score the positions on Modal's CPUs")
    p.add_argument("positions", help="a positions file under the cache")
    p.add_argument("--name", required=True, help="the job's name, unique")
    p.add_argument("--hosts", type=int, default=1, help="containers at once")
    p.add_argument("--jobs-per-host", type=int, default=1)
    p.add_argument("--cpu", type=int, default=8, help="a container's cores")
    p.add_argument("--memory", type=int, default=16, help="a container's GiB")
    p.add_argument("--hours", type=float, default=2.0, help="a container's limit")
    p.add_argument("--level", type=int, default=LEVEL)
    p.add_argument("--engine-dir", default=str(CACHE_DIR / "engines-2026-11" / "neural-reversi-v3"),
                   help="a Neural Reversi build under the cache, for any x86 CPU (x86-64-v3)")  # fmt: skip
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="at most this many positions a job (a trial)",
    )
    p.add_argument(
        "--out", required=True, help="results, a JSON line a job; label files beside it"
    )
    p = sub.add_parser("assemble", help="the set from the positions and their labels")
    p.add_argument("positions")
    p.add_argument("labels", nargs="+", help="directories of label files")
    p.add_argument("--level", type=int, default=LEVEL)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"positions": cmd_positions, "label": cmd_label, "cloud": cmd_cloud,
     "assemble": cmd_assemble}[args.command](args)  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
