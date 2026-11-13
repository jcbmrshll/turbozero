"""Our player on the CPU, a `cpu_search.py` search of one game (`NetPlayer`, the `net:` player of
`evaluate.py`), the networks it evaluates with (`make_eval`, `make_engine`), and its speed:

    taskset -c 16-23 uv run examples/othello/cpu_match.py speed --net NET --batch 1 8 16

`speed` plays one game at a time, as in a contest, and reports evals a second including everything
the search does. Games between players are `evaluate.py`'s.
"""

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# a CPU script: JAX would otherwise take the GPU's memory, and the league would give way
if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from bench_cpu_backends import network_path
from cpu_search import (
    PASS,
    Engine,
    Eval,
    Search,
    SearchSettings,
    openvino_eval,
    ort_eval,
    ort_session,
)
from export_onnx import export_checkpoint
from native_target import cpus
from quantize import quantized_path
from xot import XOT_PATH, load_xot

# traversals in a row that ended the game after which a search on a clock stops
ENDED_VISITS = 64


@dataclass(frozen=True)
class Player:
    """A way of playing: `visits` visits of the tree a move, searching by `settings`, of the
    network named `net` on `backend` (see `make_eval`)."""

    net: str
    backend: str
    visits: int
    settings: SearchSettings

    @classmethod
    def parse(cls, spec: str, net: str, backend: str) -> "Player":
        """From e.g. "visits=1000,batch=8,virtual_loss=visit,net=6x96-...,backend=f32"."""
        fields = dict(item.split("=") for item in spec.split(","))
        net = fields.pop("net", net)
        backend = fields.pop("backend", backend)
        visits = int(fields.pop("visits"))
        settings = SearchSettings(
            batch=int(fields.pop("batch", 1)),
            virtual_loss=fields.pop("virtual_loss", "loss"),
            weight=float(fields.pop("weight", 1)),
            c_init=float(fields.pop("c_init", 1.25)),
            prior_temperature=float(
                fields.pop("prior_temperature", SearchSettings.prior_temperature)
            ),
            max_collisions=int(fields["max_collisions"])
            if "max_collisions" in fields
            else None,
            random_symmetry=bool(int(fields.pop("random_symmetry", 0))),
            reuse=bool(int(fields.pop("reuse", 1))),
        )
        return cls(net, backend, visits, settings)


def model_and_precision(name: str, arm: str) -> tuple[bytes | str, str]:
    """The model of the network `name` for an OpenVINO `arm`: "bf16" or "f32", or a quantized one,
    "int8-<variant>" (see `quantize.py`), computing what isn't quantized in f32, or
    "int8-<variant>-bf16"."""
    if arm.startswith("int8-"):
        _, variant, *rest = arm.split("-")
        path = quantized_path(name, variant)
        return str(path), rest[0] if rest else "f32"
    path, net = network_path(name)
    return export_checkpoint(path, net).SerializeToString(), arm


def make_eval(name: str, backend: str, cores: int) -> Eval:
    """The network `name` on `backend`: "ort", ONNX Runtime; "native", `cpu_net.py`'s; else an arm
    of `model_and_precision`, a stream to a core, or e.g. "bf16-s1", on that many streams."""
    if backend == "native":
        from cpu_net import load_native

        network = load_native(name)
        return lambda x: network.forward(x, cores)
    if backend == "ort":
        path, net = network_path(name)
        model = export_checkpoint(path, net).SerializeToString()
        return ort_eval(ort_session(model, cores))
    arm, _, streams = backend.partition("-s")
    model, precision = model_and_precision(name, arm)
    return openvino_eval(model, cores, precision, int(streams) if streams else None)


def make_engine(name: str, backend: str, cores: int, batch: int) -> Engine:
    """The network `name` on `backend`, "native" (see `cpu_net.py`) or an arm of
    `model_and_precision`, as an `Engine` of a worker to each of `cores`, each evaluating up to
    `batch` leaves at a time."""
    if backend == "native":
        from cpu_net import load_native

        return Engine.native(load_native(name), cores, batch)
    model, precision = model_and_precision(name, backend)
    if isinstance(model, str):
        return Engine.openvino(model, cores, batch, precision)
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        f.write(model)
        f.flush()
        return Engine.openvino(f.name, cores, batch, precision)


def timed_move(
    search: Search, seconds: float, batch: np.ndarray, evaluate: Eval
) -> tuple[int, int]:
    """Searches until `seconds` have passed, finishing the batch in flight. Returns the leaves
    evaluated and the visits."""
    start = time.perf_counter()
    before = search.total_visits()
    leaves = ended = 0
    search.begin(1 << 30)
    while time.perf_counter() - start < seconds:
        count = search.collect(batch)
        logits, values = evaluate(batch[:count])
        search.apply(logits, values)
        leaves += count
        # a run of traversals that all ended the game: the tree has little left to evaluate
        ended = ended + len(batch) if count == 0 else 0
        if ended >= ENDED_VISITS:
            break
    search.begin(0)
    return leaves, search.total_visits() - before


def move_budget(
    seconds_left: float,
    empties: int,
    taper: int = 0,
    solve_from: int = 0,
    reserve: float = 0.0,
) -> float:
    """Seconds to search a move on a game's clock: its share of what's left over the moves still to
    play, each side playing every other empty square, with 5% held back for what isn't searching.
    Moves count alike, or with `taper`, those with fewer empties than that less and less, by the
    square of the fraction of it left: positions near the end are small, and their values good.
    With `solve_from`, the game is to be solved from so many empties: only the moves before it are
    searched, and `reserve` seconds are kept for the solve. Once there (a solve gave up), the
    moves to the end share what's left."""

    def weight(e: int) -> float:
        return min(1.0, e / taper) ** 2 if taper else 1.0

    if empties <= solve_from:
        solve_from, reserve = 0, 0.0
    total = sum(weight(e) for e in range(empties, solve_from, -2))
    return 0.95 * max(seconds_left - reserve, 0.0) * weight(empties) / max(total, 1e-9)


@dataclass(frozen=True)
class Solving:
    """When a player solves the endgame rather than searching it (see `endgame.py`).

    On a clock, a game plans its solve as Edax does: from the most empties, up to `empties`,
    whose expected solve takes at most `share` of the game's clock; the moves before it share the
    clock less that, and the solve may take all but a tenth of what's left before it gives up and
    the move is searched. A solve that succeeds leaves the game's later moves almost nothing to do.
    At fixed visits, positions with `empties` or fewer are solved, taking what they take.

    Attributes:
        empties: positions with this many empty squares or fewer are solved, at most
        exact: solve for the best final margin; else, about ten times faster, only for a win, a
            draw or a loss, which is all a game's result needs: a position solved as lost is
            then searched as usual, the network's best try against a fallible opponent
        share: of the game's clock, the most a planned solve may be expected to take
        threads: the solver's, by default one
        cost_at_24: a solve's expected seconds at 24 empties on 8 threads, twice as long each
            empty more, and shorter in proportion to more threads. By default Edax's exact solver
            on positions of real games on the contest's cores (0.07, 0.19 and 0.80 s at 22, 24
            and 26 empties, 2026-11-10): the speed we assume ours will have. Ours took about 2.0 s
            at 24 for a win, a draw or a loss then; plan with that to play with it as it is
    """

    empties: int
    exact: bool = False
    share: float = 0.25
    threads: int = 1
    cost_at_24: float = 0.19

    def cost(self, empties: int) -> float:
        """A solve's expected seconds at `empties`."""
        return self.cost_at_24 * 2.0 ** (empties - 24) * 8 / self.threads

    def depth(self, clock: float) -> int:
        """The empties a game on a clock of `clock` seconds plans to solve from; 0 for none."""
        fits = [
            e for e in range(1, self.empties + 1) if self.cost(e) <= self.share * clock
        ]
        return max(fits, default=0)


class NetPlayer:
    """Our side of a game on a per-game clock (`clock_match.Player`): a `Search` with its own
    tree, searched a batch at a time through `evaluate`, or by an `Engine`'s workers; with
    `solving`, the endgame solved instead."""

    def __init__(
        self,
        settings: SearchSettings,
        evaluate: Eval | Engine,
        taper: int = 0,
        seed: int = 0,
        visits: int = 0,
        solving: Solving | None = None,
    ):
        """`visits`, if any, a move, ignoring the clock."""
        self.search = Search(settings, seed)
        self.evaluate = evaluate
        self.taper = taper
        self.visits = visits
        self.batch = np.empty((settings.batch, 2, 8, 8), np.float32)
        self.to_move = "b"
        self.solving = solving
        # the game's solver, its table kept from one move to the next
        self.solver = None
        if solving is not None:
            from endgame import Solver

            self.solver = Solver(solving.threads)
        # moves solved, and solves given up, in this game
        self.solved = 0
        self.gave_up = 0
        # the empties the game is solved from, planned on the game's clock at its first move
        self.solve_from: int | None = None
        # what the last move() made of its position, for the player to move there: ("q", the
        # search's root value, -1 to 1), ("margin", the solved final margin in discs), or
        # ("wdl", a solve's sign only, with `exact` off); None if it didn't search
        self.last_value: tuple[str, float] | None = None

    def sync(self, color: str) -> None:
        """Passes for the other color if `color` moves out of turn: the game passes for a player
        with no move without saying so, and the tree has passes as moves."""
        if color != self.to_move:
            self.search.play(PASS)
        self.to_move = "w" if color == "b" else "b"

    def play(self, color: str, move: int) -> None:
        """A move by `color`."""
        self.sync(color)
        self.search.play(move)

    def set_position(self, black: int, white: int, color: str) -> None:
        """Starts the game from a position given whole (see `engines.Board.at`)."""
        me, opp = (black, white) if color == "b" else (white, black)
        self.search.set_position(me, opp)
        self.to_move = color

    def solve(self, me: int, opp: int, seconds_left: float) -> int | None:
        """The move the endgame's solution plays, if it's solved in time (and with `exact` off,
        isn't lost)."""
        solving = self.solving
        assert solving is not None and self.solver is not None
        # a solve at fixed visits has no deadline; one with no time left, the least of one
        seconds = 0.0 if self.visits else max(0.9 * seconds_left, 1e-3)
        solved = self.solver.solve(me, opp, seconds, solving.exact)
        if solved is None:
            self.gave_up += 1
            return None
        self.last_value = ("margin" if solving.exact else "wdl", float(solved.margin))
        if not solving.exact and solved.margin < 0:
            return None
        self.solved += 1
        return solved.move

    def move(self, color: str, seconds_left: float) -> int:
        """Our move, which it also plays. A position with one legal move takes no time."""
        self.sync(color)
        me, opp, _, _ = self.search.position()
        empties = 64 - (me | opp).bit_count()
        if self.solving is not None and self.solve_from is None:
            # the first move sees the game's whole clock
            self.solve_from = (
                self.solving.empties
                if self.visits
                else self.solving.depth(seconds_left)
            )
        solve_from = self.solve_from or 0
        action = None
        self.last_value = None
        if self.search.num_legal() == 1:
            action = self.search.first_legal()
        elif self.solving is not None and empties <= solve_from:
            start = time.perf_counter()
            action = self.solve(me, opp, seconds_left)
            seconds_left -= time.perf_counter() - start
        if action is None:
            reserve = (
                self.solving.cost(solve_from)
                if self.solving is not None and solve_from
                else 0.0
            )
            seconds = move_budget(
                max(seconds_left, 0.0), empties, self.taper, solve_from, reserve
            )
            if isinstance(self.evaluate, Engine):
                self.evaluate.search(
                    self.search,
                    visits=self.visits,
                    seconds=seconds,
                    stop_ended=0 if self.visits else ENDED_VISITS,
                )
            elif self.visits:
                self.search.run(self.visits, self.evaluate)
            else:
                timed_move(self.search, seconds, self.batch, self.evaluate)
            action = self.search.best_action()
            self.last_value = ("q", self.search.root_q())
        self.search.play(action)
        return action

    def close(self) -> None:
        """Nothing outlives the player."""


def speed(args: argparse.Namespace) -> None:
    """One game at a time against itself, reporting evals a second and where the time goes."""
    cores = len(cpus())
    evaluate = make_eval(args.net, args.backend, cores)
    pool = load_xot(XOT_PATH)
    rng = np.random.default_rng(args.seed)
    openings = pool[rng.choice(len(pool), size=args.openings, replace=False)]
    for batch in args.batch:
        settings = SearchSettings(
            batch=batch,
            virtual_loss=args.virtual_loss,
            random_symmetry=args.random_symmetry,
        )
        buffer = np.empty((batch, 2, 8, 8), np.float32)
        visits = leaves = calls = 0
        collect = call = apply = 0.0
        begin = time.perf_counter()
        for opening in openings:
            search = Search(settings)
            for action in opening:
                search.play(int(action))
            while not search.terminated():
                if search.num_legal() == 1:
                    search.play(search.first_legal())
                    continue
                search.begin(args.visits)
                while search.remaining > 0:
                    t0 = time.perf_counter()
                    count = search.collect(buffer)
                    t1 = time.perf_counter()
                    logits, values = evaluate(buffer[:count])
                    t2 = time.perf_counter()
                    search.apply(logits, values)
                    t3 = time.perf_counter()
                    collect += t1 - t0
                    call += t2 - t1
                    apply += t3 - t2
                    leaves += count
                    calls += 1
                search.play(search.best_action())
            visits += search.total_visits()
        seconds = time.perf_counter() - begin
        record = {
            "net": args.net,
            "backend": args.backend,
            "batch": batch,
            "virtual_loss": args.virtual_loss,
            "random_symmetry": args.random_symmetry,
            "visits_per_move": args.visits,
            "evals_per_s": round(leaves / seconds),
            "visits_per_s": round(visits / seconds),
            "mean_batch": round(leaves / max(calls, 1), 2),
            "tree_share": round((collect + apply) / seconds, 4),
            "eval_share": round(call / seconds, 4),
            "tree_us_per_visit": round(1e6 * (collect + apply) / visits, 2),
            "seconds": round(seconds, 1),
            "load": os.getloadavg()[0],
        }
        print(json.dumps(record), flush=True)
        if args.results is not None:
            with open(args.results, "a") as f:
                f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(required=True)
    s = commands.add_parser("speed")
    s.add_argument("--net", required=True)
    s.add_argument("--backend", default="bf16")
    s.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    s.add_argument("--visits", type=int, default=1000)
    s.add_argument("--virtual-loss", default="loss")
    s.add_argument("--random-symmetry", action="store_true")
    s.add_argument("--openings", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--results", type=Path, default=None)
    s.set_defaults(fn=speed)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
