"""Keeps the league playing whenever the GPU is free, and out of the way whenever it isn't.

Two processes, so that the GPU's memory goes back whenever the league isn't playing:

- The supervisor (`supervise`) runs all the time, off the GPU. When nothing else is on the GPU
  or in slick's run queue, it puts one entry in the queue, at low priority: the player.
- The player (`play`), the queue's entry, plays batch after batch. It's in the queue, so
  everyone sees it, and the queue starts nothing else meanwhile. As soon as anything else
  appears in the queue or on the GPU, it abandons its batch and exits, handing over within
  seconds, however long the batch had left. The supervisor queues it again once they're done.

It can stop at any moment: games are only ever written as whole batches, and starting again
needs nothing but the store. An entry cancelled from the queue stops the supervisor too, since
that's someone saying so.

A batch is short (`SchedulerConfig.batch_seconds`) so that little is lost when one's abandoned.
"""

import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from core.league.game import LeagueGame
from core.league.gate import SLICK_URL, Gate, lane, queue_board
from core.league.match import Programs, Stopped, play_batch
from core.league.ratings import Ratings, fit_ratings
from core.league.scheduler import (
    Pairing,
    SchedulerConfig,
    choose_openings,
    next_pairing,
    seconds_per_game,
)
from core.league.store import Player, Store


def code_version() -> str:
    """The commit of the code that's running, marked if it's been changed since."""
    root = os.path.dirname(__file__)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=root, check=False
        ).returncode
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def current_ratings(store: Store, players: list[Player] | None = None) -> Ratings:
    """The fit to every game so far."""
    players = store.players() if players is None else players
    return fit_ratings(
        [p.id for p in players], store.pair_records(), anchor=store.get_meta("anchor")
    )


def pick_batch(
    store: Store,
    game: LeagueGame,
    config: SchedulerConfig,
    rng: np.random.Generator,
) -> tuple[Pairing, Player, Player, np.ndarray] | None:
    """The next batch to play: the pairing, its players and the openings to play."""
    config = replace(config, engines=store.get_meta("engines", True))
    players = store.players()
    ratings = current_ratings(store, players)
    external = {p.id: game.external_seconds(p) for p in players if p.is_external}
    costs = seconds_per_game(players, external, store.batch_costs())
    pairing = next_pairing(config, ratings, players, costs, rng)
    if pairing is None:
        return None
    by_id = {p.id: p for p in players}
    a, b = by_id[pairing.a], by_id[pairing.b]
    num_openings = pairing.games // 2
    for player in (a, b):
        if player.is_external:
            num_openings = min(num_openings, game.external_parallel(player))
    openings = choose_openings(
        game.num_openings(), store.played_openings(a.id, b.id), num_openings, rng
    )
    return pairing, a, b, openings


@dataclass
class Signals:
    """Whether a player process has been told to stop."""

    received: bool = False

    def handle(self, signum: int, frame: Any) -> None:
        self.received = True


def play(
    store: Store,
    game: LeagueGame,
    gate: Gate,
    config: SchedulerConfig | None = None,
    dtype: str = "bfloat16",
    seed: int | None = None,
    max_batches: int | None = None,
    playing: Callable = play_batch,
    version: str | None = None,
) -> str:
    """Plays batches until the GPU is wanted, or the league's stopped, whichever's first.

    Args:
        store: where the games go
        game: the game
        gate: says when the GPU is wanted
        config: how to pick batches
        dtype: the dtype networks compute in
        seed: for the draws and games (default: a fresh one)
        max_batches: stop after this many
        playing: plays a batch (`play_batch`)
        version: the code's version (default: `code_version`)

    Returns:
        why it stopped
    """
    config = config or SchedulerConfig()
    rng = np.random.default_rng(seed)
    version = version or code_version()
    programs = Programs(game, dtype)
    signals = Signals()
    # a cancel from the queue is a hangup
    previous = {
        signum: signal.signal(signum, signals.handle)
        for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    }

    stop_reason: str | None = None

    def should_stop() -> bool:
        nonlocal stop_reason
        if signals.received:
            stop_reason = "signalled"
        elif store.get_meta("stop", False):
            stop_reason = "stopped"
        else:
            stop_reason = gate.busy()
        return stop_reason is not None

    batches = 0
    try:
        while True:
            if max_batches is not None and batches >= max_batches:
                stop_reason = "played its batches"
                break
            if should_stop():
                break
            picked = pick_batch(store, game, config, rng)
            if picked is None:
                stop_reason = "no two players can play"
                break
            pairing, a, b, openings = picked
            store.set_meta(
                "player",
                {
                    "state": "playing",
                    "a": a.name,
                    "b": b.name,
                    "games": 2 * len(openings),
                    "mode": pairing.mode,
                    "started": time.time(),
                    "pid": os.getpid(),
                },
            )
            try:
                result = playing(
                    game,
                    programs,
                    a,
                    b,
                    openings,
                    int(rng.integers(1 << 31)),
                    should_stop,
                )
            except Stopped:
                break
            store.record_batch(
                a.id,
                b.id,
                pairing.mode,
                result.seconds,
                result.compiled,
                version,
                dtype,
                result.games,
            )
            batches += 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    store.set_meta(
        "player", {"state": "idle", "reason": stop_reason, "stopped": time.time()}
    )
    return stop_reason or "stopped"


@dataclass
class Queue:
    """Puts the player in slick's run queue, as the task in `cwd` (`SLICK_TASK` names it)."""

    command: str
    cwd: str
    resource: str = "5080"
    priority: int = -10
    estimate: str = "10m"
    url: str = SLICK_URL

    def submit(self) -> int:
        """Queues the player; returns its entry's number.

        Raises:
            RuntimeError: if the queue refused it
        """
        out = subprocess.run(
            ["slick", "queue", "--on", self.resource, "--priority", str(self.priority),
             "--estimate", self.estimate, "--about", "Elo league: playing until anyone needs the GPU",
             "--", self.command],
            cwd=self.cwd, capture_output=True, text=True, check=False,
        )  # fmt: skip
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or out.stdout.strip())
        return int(out.stdout.split("#")[1].split()[0])

    def state(self, job: int) -> str | None:
        """The entry's state: "queued", "running", "cancelled", "failed" (it exited with an
        error, or was lost) or "ended"; None if it can't be told."""
        board = queue_board(self.url)
        if board is None:
            return None
        found = lane(board, self.resource)
        running = found.get("running")
        if running and running.get("id") == job:
            return "running"
        if any(entry.get("id") == job for entry in found.get("queued") or []):
            return "queued"
        for entry in board.get("ended") or []:
            if entry.get("id") == job:
                if entry.get("state") == "cancelled":
                    return "cancelled"
                return "ended" if entry.get("exit") == 0 else "failed"
        return "ended"


def supervise(
    store: Store,
    gate: Gate,
    queue: Any,
    every: float = 30.0,
    settle: int = 2,
    max_load: float = 16.0,
    quick_exit: float = 180.0,
    max_failures: int = 8,
    sleep: Callable[[float], None] = time.sleep,
    load: Callable[[], float] = lambda: os.getloadavg()[0],
    clock: Callable[[], float] = time.time,
    rounds: int | None = None,
) -> str:
    """Keeps the player queued whenever the GPU is free, until the league's stopped.

    An entry that fails or ends soon after it starts is a failure, and each in a row doubles
    the wait before the next (from 2 minutes to at most an hour): a queue entry's end
    prompts the agent of the slick task it's queued from. After `max_failures` in a row,
    something's wrong that waiting won't fix, and it stops.

    Args:
        store: the league's store
        gate: says when the GPU is free, not counting the player's own entry
        queue: `Queue`
        every: seconds between looks
        settle: how many looks in a row must find the GPU free: jobs follow each other
        max_load: the machine's load above which it waits (24 cores, shared)
        quick_exit: an entry that ends within this many seconds of starting has failed
        max_failures: failures in a row to stop after
        sleep: waits
        load: the machine's load
        clock: the time
        rounds: looks to take, at most

    Returns:
        why it stopped
    """

    def status(state: str, **extra: Any) -> None:
        store.set_meta(
            "supervisor", {"state": state, "pid": os.getpid(), "time": clock(), **extra}
        )

    job: int | None = store.get_meta("player_job")
    free = 0
    round_ = 0
    started: float | None = None
    failures = 0
    hold_until = 0.0
    while rounds is None or round_ < rounds:
        round_ += 1
        if store.get_meta("stop", False):
            status("stopped")
            return "stopped"
        if job is not None:
            state = queue.state(job)
            if state in ("queued", "running"):
                if state == "running" and started is None:
                    started = clock()
                status("playing" if state == "running" else "queued", job=job)
                sleep(every)
                continue
            if state is None:
                status("waiting", reason="can't read the queue")
                sleep(every)
                continue
            quick = started is None or clock() - started < quick_exit
            job, started = None, None
            store.set_meta("player_job", None)
            if state == "cancelled":
                status("stopped", reason="its queue entry was cancelled")
                return "cancelled"
            failures = failures + 1 if state == "failed" or quick else 0
            if failures >= max_failures:
                status("stopped", reason=f"{failures} failures in a row")
                return "failed"
            if failures:
                hold_until = clock() + min(3600.0, 60.0 * 2**failures)
            free = 0
        if clock() < hold_until:
            status("waiting", reason=f"{failures} failures in a row")
            sleep(every)
            continue
        reason = gate.busy()
        if reason is None and load() > max_load:
            reason = f"machine load {load():.0f}"
        if reason is not None:
            free = 0
            status("waiting", reason=reason)
            sleep(every)
            continue
        free += 1
        if free < settle:
            status("waiting", reason="checking the GPU stays free")
            sleep(every)
            continue
        try:
            job = queue.submit()
        except (RuntimeError, OSError, ValueError, IndexError) as e:
            status("waiting", reason=f"can't queue: {e}")
            sleep(10 * every)
            continue
        store.set_meta("player_job", job)
        free = 0
        status("queued", job=job)
        sleep(every)
    return "rounds"
