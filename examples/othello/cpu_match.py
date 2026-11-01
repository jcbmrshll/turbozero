"""Games between two `cpu_search.py` players at fixed visits a move, or one's speed.

    taskset -c 16-23 uv run examples/othello/cpu_match.py match --net NET \
        --a visits=1000,batch=1 --b visits=1000,batch=8,virtual_loss=visit --openings 200

Games start from XOT openings, each played with both colors, sharing nothing but the network's
session. Many are played at once, their leaves evaluated in one call: at fixed visits, how fast
doesn't change the moves. A position with one legal move is played without a search.

`speed` plays one game at a time, as in a contest, and reports evals a second including everything
the search does. `clock_match.py` plays on a clock, with a `NetPlayer` as "net:".
"""

import argparse
import json
import os
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
    Eval,
    Search,
    SearchSettings,
    openvino_eval,
    ort_eval,
    ort_session,
)
from export_onnx import export_checkpoint
from quantize import quantized_path
from xot import XOT_PATH, load_xot

# the most leaves one call of the network evaluates
MAX_CALL = 64
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
            max_collisions=int(fields["max_collisions"])
            if "max_collisions" in fields
            else None,
            random_symmetry=bool(int(fields.pop("random_symmetry", 0))),
        )
        return cls(net, backend, visits, settings)


def chunked(evaluate: Eval) -> Eval:
    """`evaluate`, calling it with at most `MAX_CALL` leaves at a time."""

    def run(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x) <= MAX_CALL:
            return evaluate(x)
        parts = [evaluate(x[i : i + MAX_CALL]) for i in range(0, len(x), MAX_CALL)]
        return (
            np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
        )

    return run


class Game:
    """A game between players a and b, from an opening. Each keeps its own tree, seeded apart."""

    def __init__(
        self, a: Player, b: Player, opening: np.ndarray, a_black: bool, seed: int = 0
    ):
        self.players = (a, b)
        self.searches = (Search(a.settings, 2 * seed), Search(b.settings, 2 * seed + 1))
        self.a_black = a_black
        self.ply = 0
        for action in opening:
            self.play(int(action))
        self.begin()

    @property
    def evaluator(self) -> tuple[str, str]:
        """The network and backend of the player to move."""
        player = self.players[self.mover]
        return player.net, player.backend

    @property
    def mover(self) -> int:
        """Which of the players moves: 0, a, or 1, b."""
        return 0 if (self.ply % 2 == 0) == self.a_black else 1

    def play(self, action: int) -> None:
        for search in self.searches:
            search.play(action)
        self.ply += 1

    def begin(self) -> None:
        """Starts the search of the next move, past moves with only one choice and to the game's
        end."""
        search = self.searches[self.mover]
        while not search.terminated() and search.num_legal() == 1:
            self.play(search.first_legal())
            search = self.searches[self.mover]
        if not search.terminated():
            search.begin(self.players[self.mover].visits)

    def over(self) -> bool:
        return self.searches[0].terminated()

    def result(self) -> float:
        """The result for a: +1, 0 or -1."""
        reward = self.searches[0].reward()
        return reward if self.mover == 0 else -reward

    def collect(self, out: np.ndarray) -> int:
        """The next leaves to evaluate, past any moves whose search is done. Returns how many, 0
        if none this time, the game then over or the batch all game's ends."""
        search = self.searches[self.mover]
        while search.remaining == 0:
            self.play(search.best_action())
            self.begin()
            if self.over():
                return 0
            search = self.searches[self.mover]
        return search.collect(out)

    def apply(self, logits: np.ndarray, values: np.ndarray) -> None:
        self.searches[self.mover].apply(logits, values)


def play_match(
    a: Player,
    b: Player,
    openings: np.ndarray,
    evals: dict[tuple[str, str], Eval],
    parallel: int,
    seed: int = 0,
) -> np.ndarray:
    """Plays each opening twice, a black in the first, `parallel` games at once. Each game has a
    seed of its own, from `seed`: how many are played at once doesn't change them.

    Returns:
        (openings, 2) a's result in each game.
    """
    results = np.zeros((len(openings), 2))
    queue = [(o, side) for o in range(len(openings)) for side in range(2)]
    queue.reverse()
    active: list[tuple[Game, int, int]] = []
    capacity = max(a.settings.batch, b.settings.batch)
    buffers = {
        key: np.empty((parallel * capacity, 2, 8, 8), np.float32) for key in evals
    }
    start = time.perf_counter()
    done = 0
    while queue or active:
        while queue and len(active) < parallel:
            o, side = queue.pop()
            game_seed = (seed << 20) + 2 * o + side
            active.append((Game(a, b, openings[o], side == 0, game_seed), o, side))
        filled = dict.fromkeys(evals, 0)
        pending = []
        still = []
        for game, o, side in active:
            key = game.evaluator
            count = game.collect(buffers[key][filled[key] :])
            if game.over():
                results[o, side] = game.result()
                done += 1
                if done % 40 == 0:
                    rate = done / (time.perf_counter() - start)
                    print(f"  {done} games, {rate:.2f} a second", flush=True)
                continue
            # `collect` may have moved on to the other player's move, whose leaves it wrote to the
            # buffer of the player it started with: move them
            moved = game.evaluator
            if moved != key:
                buffers[moved][filled[moved] : filled[moved] + count] = buffers[key][
                    filled[key] : filled[key] + count
                ]
                key = moved
            pending.append((game, key, filled[key], count))
            filled[key] += count
            still.append((game, o, side))
        active = still
        outs = {
            key: chunked(evaluate)(buffers[key][: filled[key]])
            for key, evaluate in evals.items()
        }
        for game, key, offset, count in pending:
            logits, values = outs[key]
            game.apply(logits[offset : offset + count], values[offset : offset + count])
    return results


def elo(score: float) -> float:
    score = min(max(score, 1e-3), 1 - 1e-3)
    return 400 * np.log10(score / (1 - score))


def summarize(results: np.ndarray) -> dict[str, float]:
    """a's score, Elo over b and its standard error, from the openings' mean scores."""
    points = (1 + results) / 2
    pairs = points.mean(axis=1)
    score = float(pairs.mean())
    se = float(pairs.std() / np.sqrt(len(pairs)))
    return {
        "games": int(results.size),
        "score": score,
        "se": se,
        "elo": float(elo(score)),
        "elo_se": float((elo(score + se) - elo(score - se)) / 2),
        "win": float((results > 0).mean()),
        "draw": float((results == 0).mean()),
    }


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
    """The network `name` on `backend`: "ort", ONNX Runtime; else an arm of `model_and_precision`,
    a stream to a core, or e.g. "bf16-s1", on that many streams."""
    if backend == "ort":
        path, net = network_path(name)
        model = export_checkpoint(path, net).SerializeToString()
        return ort_eval(ort_session(model, cores))
    arm, _, streams = backend.partition("-s")
    model, precision = model_and_precision(name, arm)
    return openvino_eval(model, cores, precision, int(streams) if streams else None)


def match(args: argparse.Namespace) -> None:
    a = Player.parse(args.a, args.net, args.backend)
    b = Player.parse(args.b, args.net, args.backend)
    cores = len(os.sched_getaffinity(0))
    evals = {(p.net, p.backend): make_eval(p.net, p.backend, cores) for p in (a, b)}
    pool = load_xot(XOT_PATH)
    rng = np.random.default_rng(args.seed)
    openings = pool[rng.choice(len(pool), size=args.openings, replace=False)]
    start = time.perf_counter()
    results = play_match(a, b, openings, evals, args.parallel, args.seed)
    record = {
        "a": args.a,
        "b": args.b,
        "net": args.net,
        "seed": args.seed,
        "seconds": round(time.perf_counter() - start),
        "load": os.getloadavg()[0],
        **summarize(results),
    }
    print(json.dumps(record), flush=True)
    if args.results is not None:
        with open(args.results, "a") as f:
            f.write(json.dumps(record) + "\n")


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


def move_budget(seconds_left: float, empties: int, taper: int = 0) -> float:
    """Seconds to search a move on a game's clock: its share of what's left over the moves still to
    play, each side playing every other empty square, with 5% held back for what isn't searching.
    Moves count alike, or with `taper`, those with fewer empties than that less and less, by the
    square of the fraction of it left: positions near the end are small, and their values good."""

    def weight(e: int) -> float:
        return min(1.0, e / taper) ** 2 if taper else 1.0

    total = sum(weight(e) for e in range(empties, 0, -2))
    return 0.95 * seconds_left * weight(empties) / max(total, 1e-9)


class NetPlayer:
    """Our side of a game on a per-game clock (`clock_match.Player`): a `Search` with its own
    tree."""

    def __init__(
        self, settings: SearchSettings, evaluate: Eval, taper: int = 0, seed: int = 0
    ):
        self.search = Search(settings, seed)
        self.evaluate = evaluate
        self.taper = taper
        self.batch = np.empty((settings.batch, 2, 8, 8), np.float32)
        self.to_move = "b"

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

    def move(self, color: str, seconds_left: float) -> int:
        """Our move, which it also plays. A position with one legal move takes no time."""
        self.sync(color)
        if self.search.num_legal() == 1:
            action = self.search.first_legal()
        else:
            me, opp, _, _ = self.search.position()
            empties = 64 - (me | opp).bit_count()
            seconds = move_budget(seconds_left, empties, self.taper)
            timed_move(self.search, seconds, self.batch, self.evaluate)
            action = self.search.best_action()
        self.search.play(action)
        return action

    def close(self) -> None:
        """Nothing outlives the player."""


def speed(args: argparse.Namespace) -> None:
    """One game at a time against itself, reporting evals a second and where the time goes."""
    cores = len(os.sched_getaffinity(0))
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
    m = commands.add_parser("match")
    m.add_argument("--net", required=True)
    m.add_argument("--backend", default="bf16")
    m.add_argument("--a", required=True, help="e.g. visits=1000,batch=1")
    m.add_argument("--b", required=True)
    m.add_argument("--openings", type=int, default=200)
    m.add_argument("--parallel", type=int, default=64)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--results", type=Path, default=None)
    m.set_defaults(fn=match)
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
