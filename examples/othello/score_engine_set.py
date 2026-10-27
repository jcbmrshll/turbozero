"""Scores train.py checkpoints on an engine-labelled set (see engine_set.py): the raw network, no
search, against Edax's scores of every legal move.

    uv run examples/othello/score_engine_set.py runs/othello/49.eqx runs/othello/99.eqx \\
        --net resnet:10x128,pool_every=3,heads=pooled

Prints each checkpoint's metrics, for all positions and by source and game phase (see
`core.testing.scored_positions.scored_position_metrics`); with --out, also appends them to a
.jsonl file.
"""

import argparse
import json
from pathlib import Path

from engine_set import DEFAULT_PATH, EngineSet
from game import load_checkpoint

from core.testing.scored_positions import METRICS, ScoredPositionsTester


def main():
    parser = argparse.ArgumentParser(
        description="Scores checkpoints on an engine-labelled set of positions."
    )
    parser.add_argument("checkpoints", nargs="+", help="checkpoints saved by train.py")
    parser.add_argument(
        "--net",
        default="resnet:6x128",
        help="the checkpoints' network, as train.py's --net",
    )
    parser.add_argument(
        "--set", default=str(DEFAULT_PATH), help="the engine-labelled set"
    )
    parser.add_argument(
        "--dtype", default="float32", help="dtype the network computes in"
    )
    parser.add_argument("--out", default=None, help="append the results to this .jsonl")
    args = parser.parse_args()

    if not Path(args.set).exists():
        parser.error(
            f"no engine-labelled set at {args.set}: build one with label_positions.py"
        )
    engine_set = EngineSet.load(args.set)
    tester = None
    for path in args.checkpoints:
        network, params = load_checkpoint(path, args.net, args.dtype)
        if tester is None:
            tester = ScoredPositionsTester(engine_set.scored_positions(), network)
        metrics = {k: float(v) for k, v in tester.evaluate(params).items()}
        print(
            f"{path} ({len(engine_set)} positions, {engine_set.engine} level {engine_set.level})"
        )
        for group in tester.group_names:
            suffix = "" if group == "all" else f"_{group}"
            values = "  ".join(
                f"{m} {metrics[f'{tester.name}_{m}{suffix}']:.4f}" for m in METRICS
            )
            print(f"  {group:>24}: {values}")
        if args.out:
            with open(args.out, "a") as f:
                f.write(
                    json.dumps({"checkpoint": path, "net": args.net, **metrics}) + "\n"
                )


if __name__ == "__main__":
    main()
