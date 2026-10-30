"""Scores train.py checkpoints on the solver-labelled set (see solver_set.py): the raw network, no
search, against the solver's scores of every legal move.

    uv run examples/connect_four/score_solver_set.py runs/connect_four/49.eqx runs/connect_four/99.eqx

Prints each checkpoint's metrics, for all positions and by game phase (and with --by-source, by where
the positions came from; see `core.testing.scored_positions.scored_position_metrics`): `engine_*`
against the solver's scores, in which winning later than a move could gives some up, and
`outcome_*` against how each move's game ends. With --out, also appends them to a .jsonl file.
"""

import argparse
import json
from pathlib import Path

from game import DEFAULT_NET, load_checkpoint
from solver_set import DEFAULT_PATH, SolverSet

from core.testing.scored_positions import METRICS, ScoredPositionsTester


def main():
    parser = argparse.ArgumentParser(
        description="Scores checkpoints on the solver-labelled set of positions."
    )
    parser.add_argument("checkpoints", nargs="+", help="checkpoints saved by train.py")
    parser.add_argument(
        "--net",
        default=DEFAULT_NET,
        help="the checkpoints' network, as train.py's --net",
    )
    parser.add_argument(
        "--set", default=str(DEFAULT_PATH), help="the solver-labelled set"
    )
    parser.add_argument(
        "--dtype", default="float32", help="dtype the network computes in"
    )
    parser.add_argument(
        "--by-source",
        action="store_true",
        help="also report each source of the set's positions",
    )
    parser.add_argument("--out", default=None, help="append the results to this .jsonl")
    args = parser.parse_args()

    if not Path(args.set).exists():
        parser.error(
            f"no solver-labelled set at {args.set}: build one with label_positions.py"
        )
    solver_set = SolverSet.load(args.set)
    testers = None
    for path in args.checkpoints:
        network, params = load_checkpoint(path, args.net, args.dtype)
        if testers is None:
            testers = [
                ScoredPositionsTester(
                    solver_set.scored_positions(by_outcome, args.by_source),
                    network,
                    name=name,
                )
                for name, by_outcome in (("engine", False), ("outcome", True))
            ]
        metrics = {k: float(v) for t in testers for k, v in t.evaluate(params).items()}
        print(f"{path} ({len(solver_set)} positions)")
        for tester in testers:
            for group in tester.group_names:
                suffix = "" if group == "all" else f"_{group}"
                values = "  ".join(
                    f"{m} {metrics[f'{tester.name}_{m}{suffix}']:.4f}" for m in METRICS
                )
                print(f"  {tester.name:>8} {group:>10}: {values}")
        if args.out:
            with open(args.out, "a") as f:
                f.write(
                    json.dumps({"checkpoint": path, "net": args.net, **metrics}) + "\n"
                )


if __name__ == "__main__":
    main()
