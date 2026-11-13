"""Evaluation games between any two players, here, on the GPU, on Google Cloud or on Modal, all
kept in one store (`core.evals.store`, in ~/.cache/turbozero/othello/evals/evals.db).

    uv run examples/othello/evaluate.py run --a net:net=NAME,visits=26400,engine=1,backend=native \\
        --b edax:level=24 --openings 50 --solve-at 24 --cores 16-23
    uv run examples/othello/evaluate.py run --a net:net=NAME,clock=10,engine=1,backend=native \\
        --b edax:clock=10 --openings 50 --on gcp --jobs 4
    uv run --extra cuda examples/othello/evaluate.py run \\
        --a net:net=NAME,device=gpu-batched,visits=1000 --b edax:level=20 --openings 256
    uv run examples/othello/evaluate.py run --config match.toml
    uv run examples/othello/evaluate.py show MATCH          # what a match showed; matches, players

A player is `kind:key=value,...` (see `eval_players.py`): a network (`net:`) at `visits` a move
or on a `clock`, on the CPU, the GPU a game at a time, or the GPU in batches; or an engine at a
`level` or on a `clock`. Each side has its own budget. Its id is its config's digest, so the same
settings are the same player wherever they play, and a changed one is a new player.

A match (see `eval_games.py`) plays `--openings` starts, each with both colors, from `--starts`
(`xot`, `random:N`, `file:PATH`), drawn by `--seed`; with `--solve-at N`, every game stops at N
empties and is solved for its result. A match's id is its setup's and where it's played: running
it again continues it, and `--tag` makes another.

Where (`--on`):

- `local`: here, on `--cores` (else all this process may use; quiet ones, for timed games)
- `gcp`: Spot VMs of one CPU model (`gcp_cpu_jobs.py`, 8 cores of AMD Turin), `--jobs` of them
- `modal`: Modal containers, `--gpu CPU` or a GPU, `--modal-cpu` cores each (a mix of CPU models:
  don't compare their seconds)

Cloud hosts write their games to the bucket as they go, and they're brought into the store while
the match runs; `sync MATCH` brings in what's there later. Every game records its moves, each
move's seconds (when timed alone), how it ended, and the host it ran on.
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any



def wants_gpu(argv: Sequence[str]) -> bool:
    """Whether a player of the match asked for on the command line, in its `--config` or in its
    `--match @FILE`, plays on the GPU."""
    texts = list(argv)
    for flag in ("--config", "--match"):
        if flag in argv[:-1]:
            value = argv[list(argv).index(flag) + 1]
            path = Path(value[1:] if value.startswith("@") else value)
            if (flag == "--config" or value.startswith("@")) and path.is_file():
                texts.append(path.read_text())
    return any(GPU_DEVICE.search(text) for text in texts)


GPU_DEVICE = re.compile(r"""device["']?\s*[=:]\s*["']?gpu""")

# JAX picks its devices when the players' modules are imported: off the GPU unless a player is on
# it, where loading a checkpoint would take its memory and other work there (the league) gives way
if __name__ == "__main__" and not wants_gpu(sys.argv[1:]):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from clock_match import Game, elo, pair_summary
from eval_games import MatchSpec, numbers_of, play
from eval_players import CACHE_DIR, parse, resolve

from core.evals import hardware
from core.evals.store import GameRecord, Store

DEFAULT_STORE = CACHE_DIR / "evals" / "evals.db"
# where a cloud job writes its games, from examples/othello
OUT_DIR = "eval-out"
# what a match's setup is made of, in a config file or on the command line
MATCH_KEYS = (
    "openings",
    "starts",
    "seed",
    "solve_at",
    "solve_exact",
    "synchro",
    "match_type",
    "since",
    "tag",
)
SYNC_SECONDS = 60.0


def parse_cores(text: str) -> list[int]:
    """ "16-23" or "0-5,12,13" as CPUs."""
    cores: list[int] = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        cores += list(range(int(low), int(high or low) + 1))
    return cores


def check_devices(spec: MatchSpec) -> None:
    """Refuses a GPU player when JAX was kept to the CPU (see `wants_gpu`)."""
    if spec.uses_gpu and os.environ.get("JAX_PLATFORMS") == "cpu":
        raise SystemExit("a player is on the GPU, and JAX_PLATFORMS is cpu")


def placement_of(args: argparse.Namespace) -> tuple[str, int]:
    """Where the games are played, and the cores a player has by default there."""
    if args.on == "local":
        # each job's share of the cores
        return "local", max(len(os.sched_getaffinity(0)) // args.jobs, 1)
    if args.on == "gcp":
        # one thread a core: half the machine's vCPUs (gcp_cpu_jobs.machine_options)
        return f"gcp:{args.machine}", int(args.machine.split("-")[-1]) // 2
    if args.on == "modal":
        return f"modal:{args.gpu}:{args.modal_cpu}c", args.modal_cpu
    raise SystemExit(f"--on {args.on}: local, gcp or modal")


def match_setup(args: argparse.Namespace) -> dict[str, Any]:
    """The match's settings, from `--config` and then the command line, which wins."""
    setup: dict[str, Any] = {}
    if args.config is not None:
        path = Path(args.config)
        text = path.read_text()
        setup = tomllib.loads(text) if path.suffix == ".toml" else json.loads(text)
    for key in ("a", "b", "note", *MATCH_KEYS):
        value = getattr(args, key, None)
        if value is not None and value is not False:
            setup[key] = value
    for side in ("a", "b"):
        if side not in setup:
            raise SystemExit(f"no player {side}: --{side} or the config's {side}")
    return setup


def build_match(setup: dict[str, Any], cores: int) -> MatchSpec:
    batched = any(
        kind == "net" and options.get("device") == "gpu-batched"
        for kind, options in (parse(setup["a"]), parse(setup["b"]))
    )
    a, b = (resolve(setup[side], cores, batched) for side in ("a", "b"))
    return MatchSpec(
        a,
        b,
        openings=int(setup.get("openings", 100)),
        **{k: setup[k] for k in MATCH_KEYS if k in setup and k != "openings"},
    )


def register(store: Store, spec: MatchSpec, placement: str, note: str) -> str:
    match_id = spec.id(placement)
    for player in (spec.a, spec.b):
        store.add_player(player.config, player.label)
    store.start_match(
        match_id,
        spec.a.id,
        spec.b.id,
        spec.to_dict(),
        placement,
        hardware.code_version(),
        spec.games,
        note,
    )
    return match_id


def run(args: argparse.Namespace) -> None:
    if args.cores is not None:
        if args.on != "local":
            raise SystemExit("--cores is for --on local")
        os.sched_setaffinity(0, parse_cores(args.cores))
    placement, cores = placement_of(args)
    setup = match_setup(args)
    spec = build_match(setup, cores)
    match_id = spec.id(placement)
    print(
        json.dumps(
            {
                "match": match_id,
                "placement": placement,
                "a": {"id": spec.a.id, "label": spec.a.label},
                "b": {"id": spec.b.id, "label": spec.b.label},
                "games": spec.games,
            }
        ),
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(spec.to_dict(), indent=2))
        return
    store = Store(args.store)
    register(store, spec, placement, setup.get("note", ""))
    if args.on == "local" and args.jobs > 1:
        play_here_in_jobs(store, spec, match_id, args.jobs, args.store)
    elif args.on == "local":
        play_here(store, spec, match_id)
    else:
        play_in_cloud(store, spec, match_id, placement, args)
    report(store, match_id)


def play_here(store: Store, spec: MatchSpec, match_id: str) -> None:
    check_devices(spec)
    done = store.numbers(match_id)
    numbers = [n for n in numbers_of(spec.openings) if n not in done]
    if done:
        print(f"{len(done)} games already played: {len(numbers)} to go", flush=True)

    def emit(record: GameRecord) -> None:
        store.record_games([record])
        played = len(store.numbers(match_id))
        if played % 10 == 0 or played == spec.games:
            print(json.dumps(summary(store, match_id, brief=True)), flush=True)

    play(
        spec,
        match_id,
        numbers,
        emit,
        lambda info, host: store.add_run(match_id, info, host),
    )


def play_here_in_jobs(
    store: Store, spec: MatchSpec, match_id: str, jobs: int, store_path: Path
) -> None:
    """The match in `jobs` processes here, each a share of its starts on its share of the cores,
    their games brought into the store as they're written."""
    import subprocess
    import tempfile

    if spec.uses_gpu:
        raise SystemExit("--jobs here is for CPU players: GPU players share one GPU")
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < jobs:
        raise SystemExit(f"{jobs} jobs on {len(cores)} cores")
    if any(p.clock is not None for p in (spec.a, spec.b)):
        print(
            "warning: timed games in parallel share the machine's caches and memory",
            flush=True,
        )
    done = store.numbers(match_id)
    share = len(cores) // jobs
    with tempfile.TemporaryDirectory() as directory:
        match_file = Path(directory) / "match.json"
        match_file.write_text(json.dumps({"id": match_id, "spec": spec.to_dict()}))
        processes, outs = [], []
        for k in range(jobs):
            out = Path(directory) / f"{k}.jsonl"
            outs.append(out)
            mine = ",".join(str(c) for c in cores[k * share : (k + 1) * share])
            command = [
                *("taskset", "-c", mine, sys.executable, __file__),
                *("--store", str(store_path), "play", "--match", f"@{match_file}"),
                *("--slice", f"{k}/{jobs}", "--out", str(out)),
                *("--skip", ",".join(str(n) for n in sorted(done))),
            ]
            processes.append(subprocess.Popen(command))
        offsets = [0] * jobs

        def collect() -> None:
            for k, out in enumerate(outs):
                if not out.exists():
                    continue
                data = out.read_bytes()
                complete = data[offsets[k] : data.rfind(b"\n") + 1]
                offsets[k] += len(complete)
                if complete and sync_lines(store, complete.decode().splitlines()):
                    print(json.dumps(summary(store, match_id, brief=True)), flush=True)

        try:
            while any(p.poll() is None for p in processes):
                time.sleep(5)
                collect()
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            collect()
        failed = [k for k, p in enumerate(processes) if p.returncode != 0]
        if failed:
            raise SystemExit(f"jobs {failed} failed")


def job_files(spec: MatchSpec) -> list[str]:
    """What a cloud host needs of `CACHE_DIR` for the match's players."""
    files = sorted({f for p in (spec.a, spec.b) for f in p.files})
    outside = [f for f in files if f.startswith("/")]
    if outside:
        raise SystemExit(
            f"not under {CACHE_DIR}, so a cloud host can't have them: {outside}"
        )
    if spec.starts.startswith("file:"):
        raise SystemExit("cloud matches start from xot or random starts")
    return files


def cloud_jobs(
    spec: MatchSpec, match_id: str, placement: str, jobs: int, skip: set[int]
) -> list[dict[str, Any]]:
    """The match as `jobs` jobs for `gcp_cpu_jobs`' hosts, each a share of its starts."""
    match = json.dumps({"id": match_id, "spec": spec.to_dict()})
    out = []
    for k in range(jobs):
        path = f"{OUT_DIR}/{match_id}-{k}.jsonl"
        out.append(
            {
                "name": f"{match_id}-{k}",
                "cmd": [
                    "env",
                    f"{hardware.PLACEMENT_ENV}={placement}",
                    f"{hardware.CODE_ENV}={hardware.code_version()}",
                    "python",
                    "evaluate.py",
                    "play",
                    "--match",
                    match,
                    "--slice",
                    f"{k}/{jobs}",
                    "--skip",
                    ",".join(str(n) for n in sorted(skip)),
                    "--out",
                    path,
                ],
                "outputs": [path],
            }
        )
    return out


def sync_lines(store: Store, lines: Sequence[str]) -> int:
    """Brings a host's lines into the store: games, and runs' timings. How many games were new."""
    games, added = [], 0
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        data = json.loads(line)
        if "run" in data:
            store.add_run(data["match"], data["run"], data["host"])
        else:
            games.append(GameRecord(**data))
    if games:
        added = store.record_games(games)
    return added


def sync_bucket(store: Store, bucket: Any, match_id: str) -> int:
    """Brings what a cloud match's hosts have written so far into the store: the outputs of
    its jobs, those of every run of it (`--fresh-jobs`) too."""
    from gcp_cpu_jobs import PREFIX

    lines: list[str] = []
    for key in sorted(bucket.sizes(f"{PREFIX}/{match_id}")):
        if "/outputs/" in key:
            lines += (bucket.get_bytes(key) or b"").decode().splitlines()
    return sync_lines(store, lines)


def play_in_cloud(
    store: Store,
    spec: MatchSpec,
    match_id: str,
    placement: str,
    args: argparse.Namespace,
) -> None:
    from gcp_cpu_jobs import Gcloud, launch, project
    from modal_jobs import open_store

    bucket, config = open_store()
    sync_bucket(store, bucket, match_id)
    done = store.numbers(match_id)
    if len(done) >= spec.games:
        return
    # a match's jobs, continued as they were when run again, unless asked for new ones for
    # what's left (e.g. after a job failed): the store keeps a game once whichever ran it
    name = match_id if not args.fresh_jobs else f"{match_id}-r{int(time.time())}"
    jobs = cloud_jobs(spec, match_id, placement, args.jobs, done)
    if args.on == "gcp":
        cloud: Any = Gcloud(project(), args.machine)
    else:
        from modal_hosts import ModalHosts

        cloud = ModalHosts(args.gpu, args.modal_cpu, args.modal_memory)
    finished = threading.Event()

    def keep_syncing() -> None:
        # its own connection: SQLite's aren't shared between threads
        mine = Store(args.store)
        while not finished.wait(SYNC_SECONDS):
            if sync_bucket(mine, bucket, match_id):
                print(json.dumps(summary(mine, match_id, brief=True)), flush=True)
        mine.close()

    syncing = threading.Thread(target=keep_syncing, daemon=True)
    syncing.start()
    try:
        launch(
            bucket,
            cloud,
            name,
            jobs,
            CACHE_DIR / "evals" / "cloud" / f"{name}.jsonl",
            vms=args.jobs,
            hours=args.hours,
            attempts=args.attempts,
            files=job_files(spec),
            machine=args.machine,
            bucket=config["bucket"],
            with_engines=False,
        )
    finally:
        finished.set()
        syncing.join()
        sync_bucket(store, bucket, match_id)


def report(store: Store, match_id: str) -> None:
    match = store.match(match_id)
    assert match is not None
    if match.played >= match.planned and match.finished is None:
        store.finish_match(match_id)
    print(json.dumps(summary(store, match_id)), flush=True)


def summary(store: Store, match_id: str, brief: bool = False) -> dict[str, Any]:
    """What a match's games showed, for player a against b."""
    match = store.match(match_id)
    if match is None:
        raise SystemExit(f"no match {match_id}")
    games = store.games(match_id)
    result: dict[str, Any] = {
        "match": match_id,
        "games": len(games),
        "planned": match.planned,
    }
    if not games:
        return result
    a_black = [g.black == match.a for g in games]

    def points(value: int, black: bool) -> float:
        return 0.5 + 0.5 * float(np.sign(value if black else -value))

    final = np.array([points(g.result, b) for g, b in zip(games, a_black)])
    board = np.array([points(g.margin, b) for g, b in zip(games, a_black)])
    score = float(final.mean())
    result.update(
        {
            "score": round(score, 4),
            "stderr": round(float(final.std() / math.sqrt(len(final))), 4),
            "elo": round(elo(score), 1),
            "wins": int((final == 1).sum()),
            "draws": int((final == 0.5).sum()),
            "losses": int((final == 0).sum()),
        }
    )
    if brief:
        return result
    a = store.player(match.a)
    b = store.player(match.b)
    result.update(
        {
            "a": a.label if a else match.a,
            "b": b.label if b else match.b,
            "placement": match.placement,
            "board_score": round(float(board.mean()), 4),
            "solved": sum(g.ending == "solved" for g in games),
            "timeouts": {
                side: sum(
                    g.flagged is not None
                    and (g.flagged == "b") == (black == (side == "a"))
                    for g, black in zip(games, a_black)
                )
                for side in ("a", "b")
            },
            # a's discs less b's, game by game, as in `margins` of clock_match
            "margins": [
                g.margin if black else -g.margin for g, black in zip(games, a_black)
            ],
            "hosts": sorted({g.host.get("cpu") or "?" for g in games}),
        }
    )
    for side in ("a", "b"):
        per_game, per_move = [], []
        for g, black in zip(games, a_black):
            mine_black = black == (side == "a")
            seconds = g.black_seconds if mine_black else g.white_seconds
            if seconds is None:
                continue
            per_game.append(seconds)
            first_black = _black_moves(g)
            per_move += [
                s
                for s, by_black in zip(g.move_seconds, first_black)
                if s is not None and by_black == mine_black
            ]
        if per_game:
            result[side] = {
                "seconds_a_game": round(float(np.mean(per_game)), 3),
                "seconds_a_game_max": round(float(np.max(per_game)), 3),
                "seconds_a_move": round(float(np.mean(per_move)), 4)
                if per_move
                else None,
                "seconds_a_move_p99": round(float(np.percentile(per_move, 99)), 3)
                if per_move
                else None,
            }
    # pairs: both games of a start, in order
    by_number = {g.number: (g, black) for g, black in zip(games, a_black)}
    paired_games, paired_black = [], []
    for number in sorted(by_number):
        if number % 2 == 0 and number + 1 in by_number:
            for n in (number, number + 1):
                g, black = by_number[n]
                paired_games.append(Game(margin=g.margin, result=g.result))
                paired_black.append(black)
    result["pairs"] = pair_summary(paired_games, paired_black)
    return result


def _black_moves(game: GameRecord) -> list[bool]:
    """Whether each move after the opening was black's, replaying the game for passes."""
    from engines import Board
    from ggs import Start

    board = (
        Start.parse(game.start[len("board:") :]).board()
        if game.start.startswith("board:")
        else Board()
    )
    out = []
    for ply, move in enumerate(game.moves):
        if ply >= game.opening_moves:
            out.append(board.black_to_move)
        board.play(move)
    return out


def host_play(args: argparse.Namespace) -> None:
    """A host's share of a match, its games written to `--out` a line each as they end."""
    text = args.match
    data = json.loads(Path(text[1:]).read_text() if text.startswith("@") else text)
    spec, match_id = MatchSpec.from_dict(data["spec"]), data["id"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {int(n) for n in args.skip.split(",") if n} if args.skip else set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.startswith("{") and '"number"' in line:
                done.add(json.loads(line)["number"])
    k, n = (int(x) for x in args.slice.split("/"))
    numbers = [m for m in numbers_of(spec.openings, (k, n)) if m not in done]
    check_devices(spec)
    played = 0

    def write(line: str) -> None:
        with out.open("a") as f:
            f.write(line + "\n")

    def emit(record: GameRecord) -> None:
        nonlocal played
        write(record.to_json())
        played += 1

    def record_run(info: dict[str, Any], host: dict[str, Any]) -> None:
        write(json.dumps({"match": match_id, "run": info, "host": host}))

    play(spec, match_id, numbers, emit, record_run)
    print(json.dumps({"match": match_id, "slice": args.slice, "played": played}))


def sync(args: argparse.Namespace) -> None:
    from modal_jobs import open_store

    store = Store(args.store)
    bucket, _ = open_store()
    added = sync_bucket(store, bucket, args.match)
    print(f"{added} new games")
    report(store, args.match)


def import_files(args: argparse.Namespace) -> None:
    store = Store(args.store)
    for path in args.files:
        added = sync_lines(store, Path(path).read_text().splitlines())
        print(f"{path}: {added} new games")


def show(args: argparse.Namespace) -> None:
    store = Store(args.store)
    result = summary(store, args.match)
    if args.games:
        result["games_detail"] = [
            {
                "number": g.number,
                "black": g.black,
                "start": g.start,
                "ending": g.ending,
                "margin": g.margin,
                "result": g.result,
                "black_seconds": g.black_seconds,
                "white_seconds": g.white_seconds,
            }
            for g in store.games(args.match)
        ]
    print(json.dumps(result, indent=2))


def list_players(args: argparse.Namespace) -> None:
    store = Store(args.store)
    for p in store.players():
        print(f"{p.id}  {p.label}")
        if args.verbose:
            print(f"    {json.dumps(p.config, sort_keys=True)}")


def list_matches(args: argparse.Namespace) -> None:
    store = Store(args.store)
    labels = {p.id: p.label for p in store.players()}
    for m in store.matches():
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.started))
        state = "done" if m.finished else f"{m.played}/{m.planned}"
        print(
            f"{m.id}  {when}  {m.placement:24}  {state:>9}  "
            f"{labels.get(m.a, m.a)}  vs  {labels.get(m.b, m.b)}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("run", help="play a match, or the rest of one")
    p.add_argument("--config", help="a TOML or JSON file of the match's settings")
    p.add_argument("--a", help="a player, kind:key=value,...")
    p.add_argument("--b")
    p.add_argument("--openings", type=int, help="starts, each played with both colors")
    p.add_argument("--starts", help="xot (default), random:N or file:PATH")
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--solve-at", type=int, help="stop games at N empties and solve them"
    )
    p.add_argument(
        "--solve-exact",
        action="store_true",
        default=None,
        help="solve for the margin, not only win, draw or loss (about 10x slower)",
    )
    p.add_argument(
        "--synchro",
        action="store_true",
        default=None,
        help="a start's two games at once, on the same cores, as GGS plays them",
    )
    p.add_argument("--match-type", help="a file's starts of (GGS archives)")
    p.add_argument("--since", help="a file's starts from, e.g. 2024")
    p.add_argument("--tag", help="sets the match apart from one otherwise the same")
    p.add_argument("--note", help="what the match is for")
    p.add_argument("--on", default="local", choices=("local", "gcp", "modal"))
    p.add_argument("--cores", help="here, e.g. 16-23: the CPUs to play on")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="processes here (each on a share of the cores), or cloud hosts: each a share",
    )
    p.add_argument("--hours", type=float, default=2.0, help="a cloud host's limit")
    p.add_argument("--attempts", type=int, default=3, help="hosts a job may start")
    p.add_argument("--machine", default="n4d-highcpu-16", help="a GCP machine type")
    p.add_argument("--gpu", default="CPU", help="Modal's GPU, or CPU")
    p.add_argument("--modal-cpu", type=int, default=8, help="a Modal host's cores")
    p.add_argument("--modal-memory", type=int, default=32, help="its GiB")
    p.add_argument(
        "--fresh-jobs",
        action="store_true",
        help="cloud: new jobs for what's left, even if this match's jobs have run before",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="resolve the players and stop"
    )
    p.set_defaults(fn=run)

    p = commands.add_parser(
        "play", help="a host's share of a match (cloud jobs run it)"
    )
    p.add_argument("--match", required=True, help="the match's JSON, or @FILE")
    p.add_argument("--slice", default="0/1", help="k/n: every nth start from the kth")
    p.add_argument("--skip", default="", help="games already played, by number")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=host_play)

    p = commands.add_parser("sync", help="bring a cloud match's games here")
    p.add_argument("match")
    p.set_defaults(fn=sync)

    p = commands.add_parser("import", help="bring in games written by `play`")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=import_files)

    p = commands.add_parser("show", help="what a match showed")
    p.add_argument("match")
    p.add_argument("--games", action="store_true", help="each game too")
    p.set_defaults(fn=show)

    p = commands.add_parser("matches", help="every match")
    p.set_defaults(fn=list_matches)

    p = commands.add_parser("players", help="every player")
    p.add_argument("--verbose", "-v", action="store_true", help="with its config")
    p.set_defaults(fn=list_players)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
