"""The league's command line: a game's script hands its `LeagueGame` to `main`.

add-network NAME CHECKPOINT --net SPEC --run RUN --sims 128 1000 [--url URL]
link NAME URL
add-external NAME ENGINE --settings '{"level": 8}'
add-result A B WINS DRAWS LOSSES --source SOURCE
anchor NAME; retire NAME; resume NAME
ratings
start; stop
dashboard
"""

import argparse
import json
import os
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from core.league.game import LeagueGame
from core.league.ratings import Ratings
from core.league.store import GameRecord, Player, Store


def leaderboard(players: list[Player], ratings: Ratings) -> str:
    """The ratings as text, best first."""
    order = np.argsort(-ratings.elo)
    width = max((len(p.name) for p in players), default=0)
    lines = [
        f"{players[i].name:<{width}}  {ratings.elo[i]:8.1f} ± {ratings.sd[i]:5.1f}  "
        f"{ratings.games[i]:6d} games"
        for i in order
    ]
    if not ratings.connected:
        lines.append(f"{len(set(ratings.components.tolist()))} separate parts")
    return "\n".join(lines)


def queue_command(root: Path, script: str) -> str:
    """What the queue runs to play: from this code, on capped cores, taking GPU memory
    only as needed, with compiled programs kept between entries."""
    env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_COMPILATION_CACHE_DIR": str(root / "compiled"),
    }
    assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    cores = os.environ.get("LEAGUE_CORES", "0-7")
    return (
        f"cd {shlex.quote(os.getcwd())} && {assignments} taskset -c {cores} "
        f"uv run --extra cuda {shlex.quote(script)} --root {shlex.quote(str(root))} play"
    )


def main(make_game: Callable[[], LeagueGame], default_root: Path, argv=None) -> None:
    parser = argparse.ArgumentParser(description="A continuously running Elo league.")
    parser.add_argument("--root", type=Path, default=default_root)
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add-network", help="register a checkpoint")
    add.add_argument("name")
    add.add_argument("checkpoint")
    add.add_argument("--net", required=True, help="the network's spec")
    add.add_argument("--run", required=True, help="the training run it came from")
    add.add_argument("--sims", type=int, nargs="+", required=True)
    add.add_argument("--url", help="where to read about it, e.g. its run's page")

    link = commands.add_parser("link", help="link a network to any URL")
    link.add_argument("name")
    link.add_argument("url")

    external = commands.add_parser("add-external", help="register an engine")
    external.add_argument("name")
    external.add_argument("engine")
    external.add_argument("--settings", type=json.loads, default={})

    result = commands.add_parser(
        "add-result", help="record a series played elsewhere, without its games' moves"
    )
    result.add_argument("a")
    result.add_argument("b")
    result.add_argument("wins", type=int, help="A's")
    result.add_argument("draws", type=int)
    result.add_argument("losses", type=int, help="A's")
    result.add_argument("--source", required=True, help="where it's reported")

    anchor = commands.add_parser("anchor", help="the player the scale's zero is at")
    anchor.add_argument("name")
    retire = commands.add_parser("retire", help="no more new games for a player")
    retire.add_argument("name")
    resume = commands.add_parser("resume", help="new games for a retired player again")
    resume.add_argument("name")

    engines = commands.add_parser("engines", help="whether engines play new games")
    engines.add_argument("state", choices=["on", "off"])

    commands.add_parser("ratings", help="print the leaderboard")
    start = commands.add_parser("start", help="run the supervisor (stays running)")
    start.add_argument("--queue-cwd", required=True, help="the slick task's worktree")
    start.add_argument("--resource", default="5080")
    commands.add_parser("stop", help="stop the worker, at once")
    play_cmd = commands.add_parser("play", help="the player itself: the queue runs it")
    play_cmd.add_argument("--max-batches", type=int, default=None)
    dashboard_cmd = commands.add_parser("dashboard", help="serve the web page")
    dashboard_cmd.add_argument("--host", default="0.0.0.0")
    dashboard_cmd.add_argument("--port", type=int, default=8031)

    args = parser.parse_args(argv)
    store = Store(args.root)
    players = {p.name: p for p in store.players()}

    if args.command == "add-network":
        store.add_network(args.name, args.checkpoint, args.net, args.run)
        for sims in args.sims:
            store.add_net_player(args.name, sims)
        if args.url:
            store.set_network_url(args.name, args.url)
    elif args.command == "link":
        store.set_network_url(args.name, args.url)
    elif args.command == "add-external":
        store.add_external_player(args.name, args.engine, args.settings)
    elif args.command == "add-result":
        a, b = players[args.a].id, players[args.b].id
        outcomes = [1] * args.wins + [0] * args.draws + [-1] * args.losses
        games = [GameRecord(a, b, -1, outcome, []) for outcome in outcomes]
        store.record_batch(a, b, 0.0, False, args.source, "", games)
    elif args.command == "anchor":
        store.set_meta("anchor", players[args.name].id)
    elif args.command == "retire":
        store.set_retired(args.name)
    elif args.command == "resume":
        store.set_retired(args.name, False)
    elif args.command == "engines":
        store.set_meta("engines", args.state == "on")
    elif args.command == "ratings":
        from core.league.worker import current_ratings

        print(leaderboard(store.players(), current_ratings(store)))
    elif args.command == "stop":
        store.set_meta("stop", True)
    elif args.command == "start":
        from core.league.gate import SlickGate
        from core.league.worker import Queue, supervise

        store.set_meta("stop", False)
        command = queue_command(args.root.resolve(), str(Path(sys.argv[0]).resolve()))
        queue = Queue(command, cwd=args.queue_cwd, resource=args.resource)
        print(supervise(store, SlickGate(args.resource), queue))
    elif args.command == "play":
        from core.league.gate import SlickGate, ThrottledGate
        from core.league.worker import play

        own_job = os.environ.get("SLICK_RUN")
        gate = SlickGate(own_job=int(own_job) if own_job else None, own_pid=os.getpid())
        reason = play(
            store, make_game(), ThrottledGate(gate), max_batches=args.max_batches
        )
        print(reason)
        if reason == "no two players can play":
            sys.exit(1)
    elif args.command == "dashboard":
        from core.league.dashboard import serve

        # only what draws its boards: nothing here touches the GPU
        serve(store, make_game, args.host, args.port)
