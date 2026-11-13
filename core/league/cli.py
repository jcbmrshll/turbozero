"""The league's command line: a game's script hands its `LeagueGame` to `main`.

add-network NAME CHECKPOINT --net SPEC --run RUN --sims 128 1000 [--url URL]
add-player NETWORK --sims 4000 16000 [--tree-factor 4]
link NAME URL
add-external NAME ENGINE --settings '{"level": 8}'
add-result A B WINS DRAWS LOSSES --source SOURCE
match A B --games N; matches; cancel-match ID
anchor NAME; retire NAME; resume NAME
ratings
settings-changes; accept-settings NAME; variant SUFFIX [NAME ...]
start; stop
dashboard
"""

import argparse
import json
import os
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from core.league.game import LeagueGame, check_match
from core.league.ratings import Ratings
from core.league.scheduler import SchedulerConfig
from core.league.store import (
    DEFAULT_TREE_FACTOR,
    GameRecord,
    Player,
    Store,
    changed_settings,
)

# seconds a player rides out slick being unreachable, as when it restarts
PLAYER_QUEUE_OUTAGE = 300.0


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


def pending_matches(store: Store) -> str:
    """The series still to finish as text, next first."""
    by_id = {p.id: p for p in store.players()}
    lines = []
    for r in store.pending_matches():
        a, b = by_id[r.a], by_id[r.b]
        waiting = "  retired" if a.retired or b.retired else ""
        lines.append(f"#{r.id}  {a.name} vs {b.name}  {r.played}/{r.games}{waiting}")
    return "\n".join(lines)


def _differences(before: dict[str, Any], now: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}: {before.get(key)!r} -> {now.get(key)!r}"
        for key in changed_settings(before, now)
    )


def settings_changes(store: Store) -> str:
    """As text, the players refused for changed settings and what changed, then any who've played
    under more than one set."""
    players = store.players()
    lines = [
        f"{p.name} refused  {_differences(p.played_settings or {}, p.refused_settings)}"
        for p in players
        if p.refused_settings is not None
    ]
    by_id = {p.id: p for p in players}
    for player, periods in store.settings_changes().items():
        lines.append(by_id[player].name)
        for before, period in zip([None, *periods], periods):
            changed = (
                ""
                if before is None
                else "  " + _differences(before.settings, period.settings)
            )
            started = time.strftime("%Y-%m-%d %H:%M", time.localtime(period.first))
            lines.append(f"  from {started}, {period.batches} batches{changed}")
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

    player = commands.add_parser(
        "add-player", help="more players of a network: NETWORK@SIMS[-tFACTOR]"
    )
    player.add_argument("network")
    player.add_argument("--sims", type=int, nargs="+", required=True)
    player.add_argument(
        "--tree-factor",
        type=int,
        default=DEFAULT_TREE_FACTOR,
        help="the tree's nodes per iteration of the search",
    )

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

    match = commands.add_parser("match", help="ask the worker to play a series next")
    match.add_argument("a")
    match.add_argument("b")
    match.add_argument(
        "--games", type=int, required=True, help="even: half each colour"
    )
    commands.add_parser("matches", help="the series still to finish")
    cancel = commands.add_parser("cancel-match", help="drop a series")
    cancel.add_argument("id", type=int)

    anchor = commands.add_parser("anchor", help="the player the scale's zero is at")
    anchor.add_argument("name")
    retire = commands.add_parser("retire", help="no more new games for a player")
    retire.add_argument("name")
    resume = commands.add_parser("resume", help="new games for a retired player again")
    resume.add_argument("name")

    engines = commands.add_parser("engines", help="whether engines play new games")
    engines.add_argument("state", choices=["on", "off"])

    commands.add_parser("ratings", help="print the leaderboard")
    accept = commands.add_parser(
        "accept-settings",
        help="a refused player's new settings don't affect play: it stays the same player",
    )
    accept.add_argument("name")
    variant = commands.add_parser(
        "variant", help="a new player of each one named, by default each one refused"
    )
    variant.add_argument("suffix", help="what sets it apart, e.g. c2")
    variant.add_argument("names", nargs="*")
    commands.add_parser(
        "settings-changes",
        help="players who've played under more than one set of settings",
    )
    start = commands.add_parser("start", help="run the supervisor (stays running)")
    start.add_argument("--queue-cwd", required=True, help="the slick task's worktree")
    start.add_argument("--resource", default="5080")
    commands.add_parser("stop", help="stop the worker, at once")
    play_cmd = commands.add_parser("play", help="the player itself: the queue runs it")
    play_cmd.add_argument("--max-batches", type=int, default=None)
    play_cmd.add_argument(
        "--batch-seconds",
        type=float,
        default=SchedulerConfig.batch_seconds,
        help="how long a batch should take at most: longer ones run at larger batches, faster, "
        "but lose more when it gives way",
    )
    play_cmd.add_argument(
        "--max-tree-nodes",
        type=int,
        default=SchedulerConfig.max_tree_nodes,
        help="the most tree nodes a batch may hold at once",
    )
    play_cmd.add_argument(
        "--requests-only",
        action="store_true",
        help="stop once no requested series can be played",
    )
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
    elif args.command == "add-player":
        default = args.tree_factor == DEFAULT_TREE_FACTOR
        for sims in args.sims:
            store.add_net_player(
                args.network,
                sims,
                "" if default else f"t{args.tree_factor}",
                {} if default else {"tree_factor": args.tree_factor},
            )
    elif args.command == "link":
        store.set_network_url(args.name, args.url)
    elif args.command == "add-external":
        store.add_external_player(args.name, args.engine, args.settings)
    elif args.command == "add-result":
        a, b = players[args.a].id, players[args.b].id
        outcomes = [1] * args.wins + [0] * args.draws + [-1] * args.losses
        games = [GameRecord(a, b, -1, outcome, []) for outcome in outcomes]
        store.record_batch(a, b, 0.0, False, args.source, "", games)
    elif args.command == "match":
        for name in (args.a, args.b):
            if name not in players:
                parser.error(f"no player {name}")
        a, b = players[args.a], players[args.b]
        try:
            check_match(make_game(), a, b, args.games)
        except ValueError as e:
            parser.error(str(e))
        print(f"#{store.request_match(a.id, b.id, args.games)}")
    elif args.command == "matches":
        print(pending_matches(store))
    elif args.command == "cancel-match":
        try:
            store.cancel_match(args.id)
        except ValueError as e:
            parser.error(str(e))
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
    elif args.command == "settings-changes":
        print(settings_changes(store))
    elif args.command == "accept-settings":
        store.accept_settings(args.name)
    elif args.command == "variant":
        chosen = [players[name] for name in args.names] or [
            p for p in players.values() if p.refused_settings is not None
        ]
        for player in chosen:
            print(store.add_variant(player, args.suffix))
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
        gate = SlickGate(
            own_job=int(own_job) if own_job else None,
            own_pid=os.getpid(),
            queue_outage=PLAYER_QUEUE_OUTAGE,
        )
        reason = play(
            store,
            make_game(),
            ThrottledGate(gate),
            SchedulerConfig(
                batch_seconds=args.batch_seconds, max_tree_nodes=args.max_tree_nodes
            ),
            max_batches=args.max_batches,
            requests_only=args.requests_only,
        )
        print(reason)
        if reason == "no two players can play":
            sys.exit(1)
    elif args.command == "dashboard":
        from core.league.dashboard import serve

        serve(store, make_game().name, args.host, args.port)
