"""Builds an engine-labelled set (see engine_set.py): positions from training runs' replay buffers,
with every legal move scored by Edax.

    examples/othello/setup_edax.sh                       # once
    uv run examples/othello/label_positions.py \\
        --source az10x128=runs/trunk-10x128-az/state-150.npz:896-1024 \\
        --source pooled4x64=runs/p4x64/state-450.npz --per-source 2500 --level 24

Each --source is NAME=STATE[:FIRST-LAST], a state saved by train.py (--save-state-at), and
optionally a range of its environments (e.g. those held out from supervised training): the
positions of its replay buffer's finished games. From each source it samples --per-source of
the positions worth scoring (past the XOT opening, at least two legal moves, distinct; see
`engine_set.selectable`), and Edax scores every legal move of each at --level, searching that
many moves ahead and solving the endgame exactly once it's close enough (from about 31 empty
squares at level 24).

At level 24 a position takes about 2 seconds on average, but up to a minute; 5000 positions
took 84 minutes on 8 workers. --workers caps the Edax processes (one core each; run it under
`taskset` to keep it to some cores on a shared machine).
"""

import argparse
import json
import multiprocessing as mp
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from engine_set import DEFAULT_PATH, ILLEGAL, EngineSet, SourceState, boards, selectable
from engines import Analysis, Edax, EdaxAnalyst

# what scores a position: Edax's analyst at a level, or a fake in tests
AnalystFactory = Callable[[int], Callable[[np.ndarray], list[Analysis]]]


def edax_analyst(level: int) -> Callable[[np.ndarray], list[Analysis]]:
    """Edax at `level`, scoring every legal move of a position whose player to move is black."""
    analyst = EdaxAnalyst(Edax.default_dir(), level)
    return lambda board: analyst.analyse(board, "b")


def buffer_positions(
    path: str | Path, envs: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The positions of a saved state's replay buffer (see `Trainer.save_state`): those of
    finished games, without the data transforms' copies (each move stores its position, then
    the copies).

    Args:
        path: the state, state-<EPOCH>.npz
        envs: (optional) only the environments from `first` to `last`, exclusive

    Returns:
        tuple: obs (N, 8, 8, 2) bool, legal (N, 65) bool
    """
    with np.load(path) as saved:
        meta = json.loads(str(saved["meta"]))
        leaves = {
            name: i for i, (name, _, _) in enumerate(meta["leaves"]["collection_state"])
        }

        def get(name: str) -> np.ndarray:
            return saved[f"collection_state.{leaves['.buffer_state.' + name]}"]

        stride = 1 + meta["config"]["data_transforms"]
        if (get("next_idx") % stride).any():
            raise ValueError(
                f"{path}: the buffer's positions aren't in groups of {stride} (a position and "
                "its copies)"
            )
        keep = (get("has_reward") & get("populated"))[:, ::stride]
        if envs is not None:
            keep[: envs[0]] = False
            keep[envs[1] :] = False
        obs = get("buffer.observation_nn")[:, ::stride][keep]
        legal = get("buffer.policy_mask")[:, ::stride][keep]
    return obs, legal


_analyse = None


def _init_worker(factory: AnalystFactory, level: int) -> None:
    global _analyse
    _analyse = factory(level)


def _label_one(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert _analyse is not None
    scores = np.full(65, ILLEGAL, np.int16)
    depth = np.zeros(65, np.int16)
    for a in _analyse(board):
        scores[a.line[0]] = a.score
        depth[a.line[0]] = a.depth
    return scores, depth


def label(
    positions: np.ndarray,
    level: int,
    workers: int,
    factory: AnalystFactory = edax_analyst,
    progress: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Every legal move of each position, scored by an analyst from `factory`.

    Args:
        positions: (N, 64) boards (see `engine_set.boards`)
        level: the engine's level
        workers: analyst processes; 0 scores them in this process
        factory: makes an analyst, `factory(level)(board) -> [Analysis, ...]`
        progress: (optional) called with the number of positions scored so far

    Returns:
        tuple: scores (N, 65) int16 (ILLEGAL where not scored), depth (N, 65) int16
    """
    results = []
    if workers == 0:
        _init_worker(factory, level)
        it = map(_label_one, positions)
    else:
        # spawned, not forked: this process has JAX's threads (game.py imports it), which a fork
        # would copy mid-flight
        pool = mp.get_context("spawn").Pool(
            workers, initializer=_init_worker, initargs=(factory, level)
        )
        it = pool.imap(_label_one, positions, chunksize=4)
    try:
        for r in it:
            results.append(r)
            if progress is not None:
                progress(len(results))
    finally:
        if workers:
            pool.close()  # pyright: ignore[reportPossiblyUnboundVariable]
            pool.join()  # pyright: ignore[reportPossiblyUnboundVariable]
    scores, depth = (np.stack(x) for x in zip(*results))
    return scores, depth


def build(
    sources: list[tuple[str, SourceState, np.ndarray, np.ndarray]],
    per_source: int,
    level: int,
    workers: int,
    seed: int = 0,
    factory: AnalystFactory = edax_analyst,
    progress: Callable[[int], None] | None = None,
) -> EngineSet:
    """An engine set of `per_source` positions from each source, sampled from those worth
    scoring (see `engine_set.selectable`) and distinct across sources.

    Args:
        sources: each source's name, where its positions came from, and their obs (N, 8, 8, 2)
            and legal (N, 65)
        per_source: positions to sample from each
        level: the engine's level
        workers: analyst processes (see `label`)
        seed: for the sampling
        factory: makes analysts (see `label`)
        progress: (optional) called with the number of positions scored so far
    """
    rng = np.random.default_rng(seed)
    obs, legal, source = [], [], []
    seen: set[bytes] = set()
    for i, (name, _, o, m) in enumerate(sources):
        ok = selectable(o, m)
        # not already taken from an earlier source
        packed = np.packbits(o.reshape(len(o), -1), axis=1)
        ok &= np.array([p.tobytes() not in seen for p in packed])
        idx = np.nonzero(ok)[0]
        if len(idx) < per_source:
            raise ValueError(
                f"{name}: {len(idx)} positions worth scoring, fewer than {per_source}"
            )
        idx = np.sort(rng.choice(idx, per_source, replace=False))
        seen.update(p.tobytes() for p in packed[idx])
        obs.append(o[idx])
        legal.append(m[idx])
        source.append(np.full(per_source, i, np.int8))
    obs, legal, source = (np.concatenate(x) for x in (obs, legal, source))
    scores, depth = label(boards(obs), level, workers, factory, progress)
    return EngineSet(
        obs=obs,
        legal=legal,
        scores=scores,
        depth=depth,
        source=source,
        source_names=[name for name, _, _, _ in sources],
        source_states=[state for _, state, _, _ in sources],
        empties=(64 - obs.sum(axis=(1, 2, 3))).astype(np.int8),
        engine="edax",
        level=level,
    )


def parse_source(text: str) -> tuple[str, str, tuple[int, int] | None]:
    """NAME=STATE[:FIRST-LAST] -> (name, state, (first, last) or None)."""
    name, sep, rest = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"a source is NAME=STATE[:FIRST-LAST], not {text!r}"
        )
    path, sep, envs = rest.rpartition(":") if ":" in rest else (rest, "", "")
    if not sep:
        return name, rest, None
    first, _, last = envs.partition("-")
    return name, path, (int(first), int(last))


def main():
    parser = argparse.ArgumentParser(
        description="Builds an engine-labelled set of Othello positions."
    )
    parser.add_argument(
        "--source",
        type=parse_source,
        action="append",
        required=True,
        metavar="NAME=STATE[:FIRST-LAST]",
        help="a saved state's replay buffer (and optionally only environments FIRST to LAST, "
        "exclusive); repeat for more sources",
    )
    parser.add_argument("--per-source", type=int, default=2500)
    parser.add_argument("--level", type=int, default=24, help="Edax's level")
    parser.add_argument("--workers", type=int, default=8, help="Edax processes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_PATH))
    args = parser.parse_args()

    sources = []
    for name, path, envs in args.source:
        obs, legal = buffer_positions(path, envs)
        print(f"{name}: {len(obs)} positions from {path}", flush=True)
        sources.append((name, SourceState(str(Path(path).resolve()), envs), obs, legal))
    total = args.per_source * len(sources)
    start = time.perf_counter()

    def progress(done: int) -> None:
        if done % 250 == 0 or done == total:
            elapsed = time.perf_counter() - start
            print(f"{done}/{total} scored in {elapsed:.0f} s", flush=True)

    engine_set = build(
        sources, args.per_source, args.level, args.workers, args.seed, progress=progress
    )
    engine_set.save(args.out)
    print(f"{args.out}: {len(engine_set)} positions", flush=True)


if __name__ == "__main__":
    main()
