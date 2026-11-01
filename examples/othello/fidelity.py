"""How closely a quantized model's outputs follow bf16's, on the engine-labelled set and on real
search leaves: value error (all, and near zero) and rank correlation, policy top-move agreement and
KL over the legal moves. Leaves are drawn from games of the reference net against itself.

    taskset -c 18-23 uv run examples/othello/fidelity.py leaves \
        --net 8x128-128s-8p-w1125-274m --out DIR
    taskset -c 18-23 uv run examples/othello/fidelity.py compare \
        --net 8x128-128s-8p-w1125-274m --leaves DIR/test.npy --arm f32 int8-performance_leaves_heads_minmax
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from cpu_match import make_eval
from cpu_search import Search, SearchSettings
from quantize import engine_positions
from xot import XOT_PATH, load_xot

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """(N, 8, 8) squares moved by a step, what leaves the board dropped."""
    out = np.zeros_like(a)
    to = slice(max(dy, 0), 8 + min(dy, 0)), slice(max(dx, 0), 8 + min(dx, 0))
    source = slice(max(-dy, 0), 8 + min(-dy, 0)), slice(max(-dx, 0), 8 + min(-dx, 0))
    out[:, to[0], to[1]] = a[:, source[0], source[1]]
    return out


def legal_squares(x: np.ndarray) -> np.ndarray:
    """(N, 64) which squares the player to move can play, of (N, 2, 8, 8) observations."""
    me, opp = x[:, 0] > 0.5, x[:, 1] > 0.5
    empty = ~(me | opp)
    legal = np.zeros_like(empty)
    for dy, dx in DIRECTIONS:
        # from each disc of ours, through a line of theirs, to an empty square
        reach = shift(me, dy, dx) & opp
        for _ in range(5):
            reach |= shift(reach, dy, dx) & opp
        legal |= shift(reach, dy, dx) & empty
    return legal.reshape(len(x), 64)


def collect_leaves(
    name: str, games: int, visits: int, seed: int, cores: int, keep: int
) -> np.ndarray:
    """A sample of the leaves evaluated in `games` games of the network against itself, at batch 16
    with virtual visits."""
    evaluate = make_eval(name, "bf16", cores)
    pool = load_xot(XOT_PATH)
    rng = np.random.default_rng(seed)
    openings = pool[rng.choice(len(pool), games, replace=False)]
    settings = SearchSettings(batch=16, virtual_loss="visit")
    buffer = np.empty((16, 2, 8, 8), np.float32)
    leaves = []
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
                logits, values = evaluate(buffer[:count])
                search.apply(logits, values)
                leaves.append(buffer[:count].copy())
            search.play(search.best_action())
    all_leaves = np.concatenate(leaves)
    return all_leaves[rng.permutation(len(all_leaves))[:keep]]


def in_batches(evaluate, x: np.ndarray, batch: int = 16):
    parts = [evaluate(x[i : i + batch]) for i in range(0, len(x), batch)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def compare(x: np.ndarray, reference, outputs) -> dict[str, float]:
    """The errors of `outputs` against `reference`, each (policy logits, values) on `x`."""
    ref_logits, ref_value = reference
    logits, value = outputs
    error = value - ref_value
    near = np.abs(ref_value) < 0.2
    legal = legal_squares(x)
    choice = legal.sum(axis=1) >= 2
    legal, p_ref, p = legal[choice], ref_logits[choice, :64], logits[choice, :64]

    def log_softmax(a: np.ndarray) -> np.ndarray:
        a = np.where(legal, a, -np.inf)
        return a - np.logaddexp.reduce(a, axis=1, keepdims=True)

    lp_ref, lp = log_softmax(p_ref), log_softmax(p)
    kl = (np.exp(lp_ref) * np.where(legal, lp_ref - lp, 0)).sum(1)
    return {
        "positions": len(x),
        "value_mae": float(np.abs(error).mean()),
        "value_bias": float(error.mean()),
        "value_p99": float(np.quantile(np.abs(error), 0.99)),
        "near_zero_share": float(near.mean()),
        "near_zero_mae": float(np.abs(error[near]).mean()),
        "near_zero_bias": float(error[near].mean()),
        "spearman": spearman(ref_value, value),
        "sign_agree": float((np.sign(ref_value) == np.sign(value)).mean()),
        "top_move_agree": float((lp_ref.argmax(1) == lp.argmax(1)).mean()),
        "policy_kl": float(kl.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    leaves = commands.add_parser("leaves")
    leaves.add_argument("--net", required=True)
    leaves.add_argument("--out", type=Path, required=True)
    leaves.add_argument("--games", type=int, default=10)
    leaves.add_argument("--visits", type=int, default=500)
    leaves.add_argument("--keep", type=int, default=4000)
    run = commands.add_parser("compare")
    run.add_argument("--net", required=True)
    run.add_argument("--leaves", type=Path, required=True)
    run.add_argument("--arm", nargs="+", required=True)
    run.add_argument("--reference", default="bf16")
    run.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    cores = len(os.sched_getaffinity(0))

    if args.command == "leaves":
        args.out.mkdir(parents=True, exist_ok=True)
        for split, seed in (("calibration", 101), ("test", 202)):
            x = collect_leaves(
                args.net, args.games, args.visits, seed, cores, args.keep
            )
            np.save(args.out / f"{split}.npy", x)
            print(split, x.shape, flush=True)
        return

    sets = {"engine": engine_positions(4096)[512:], "leaves": np.load(args.leaves)}
    evals = {
        arm: make_eval(args.net, arm, cores) for arm in {args.reference, *args.arm}
    }
    for label, x in sets.items():
        reference = in_batches(evals[args.reference], x)
        for arm in args.arm:
            record = {
                "net": args.net,
                "arm": arm,
                "reference": args.reference,
                "set": label,
                **compare(x, reference, in_batches(evals[arm], x)),
            }
            print(json.dumps(record), flush=True)
            if args.results is not None:
                with open(args.results, "a") as f:
                    f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
