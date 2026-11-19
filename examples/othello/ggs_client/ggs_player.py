"""Our player's side of a GGS game: what to play in a position the server sends, on its clock.

`NetBridge` drives the CPU player of `clock_match.py`'s `net:` (`cpu_match.NetPlayer`, its
network searched by the native int8 engine, the endgame solved from `solve=N` empties): a
player to each game, started from the game's start position (`set_position`: GGS's random
starts aren't reachable from the standard one), told every move, so its tree carries over
from one move to the next while what it was told agrees with the server's board. When it
doesn't (a reconnect, an undo, a desync) the player starts again from the server's board.

Its cores go to the games it's moving in at the time, split evenly: an `Engine` of so many
workers for each search under way. In a synchro match the games move in rounds and each player
moves in one of the two games a round (see ggs_client.py), so a lone match gets all of them.

With `ponder=1`, the cores no search of ours is using search the trees of the games where the
opponent is to move (split evenly between them), each on a thread of its own until the game
changes or one of our searches starts: the opponent's move, when it comes, is played into a tree
that has already looked at it. A ponder stops at once (`Search.halt`), finishing only the
batches in flight.

A move's eval goes to the server in discs, for the player moving: a solve's margin, or the
search's root value turned into discs by `value_discs`.
"""

import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from engines import Board

# a pass, in a game's list of moves
PASS = None


@dataclass(frozen=True)
class GameView:
    """A game where it's our turn, as the server has it.

    Attributes:
        game_id: the server's, e.g. ".12.0"
        start: the start position (black's discs, white's, who's to move, "b" or "w")
        moves: every move since, a square (bit 0 is a1) or `PASS`
        black, white, color: the position now, and who's to move: us
        seconds_left: our clock before its extension
        extension: the extension not yet used (all of it until a soft timeout)
        extended: whether we've had a soft timeout, and are on the extension
        increment: added to our clock after each move
        komi: whether this is a komi game's first move, which both players send, each with
            the result it expects for the player to move
        mine: our colour, "b" or "w" (None before a komi game's colours are set)
    """

    game_id: str
    start: tuple[int, int, str]
    moves: tuple[int | None, ...]
    black: int
    white: int
    color: str
    seconds_left: float
    extension: float = 0.0
    extended: bool = False
    increment: float = 0.0
    komi: bool = False
    mine: str | None = None

    @property
    def empties(self) -> int:
        return 64 - (self.black | self.white).bit_count()


@dataclass(frozen=True)
class Decision:
    """A move (`PASS` for a pass), its eval in discs for the player moving, and what it took."""

    square: int | None
    eval: float
    seconds: float
    info: dict = field(default_factory=dict)


class Bridge(Protocol):
    """What the client asks of a player, whatever it runs on. One serves all of a client's
    games; it keeps their trees, by `game_id`.

    `decide` is our move, called on a thread of its own for each game (two can overlap, with
    two matches); `cores` is the CPU share the client gives the search. `observe` is told every
    board of our games as the server shows it, whoever's to move, on the client's thread: it
    must return at once (a player that ponders hands it to a thread of its own). `forget` is a
    game's end.
    """

    def decide(self, view: GameView, cores: int) -> Decision: ...

    def observe(self, view: GameView) -> None: ...

    def forget(self, game_id: str) -> None: ...

    def close(self) -> None: ...


def replay(start: tuple[int, int, str], moves: Sequence[int | None]) -> Board:
    """The board after `moves` from `start`. `Board.play` passes for a player with no move by
    itself, so a pass in the list only matters when the player to move has none."""
    board = Board.at(*start)
    for move in moves:
        if move is None:
            if not board.legal():
                board.black_to_move = not board.black_to_move
            continue
        board.play(move)
    return board


def colored_moves(
    start: tuple[int, int, str], moves: Sequence[int | None]
) -> list[tuple[str, int]]:
    """Each move that isn't a pass, with the colour that made it."""
    board = Board.at(*start)
    out = []
    for move in moves:
        if move is None:
            if not board.legal():
                board.black_to_move = not board.black_to_move
            continue
        out.append((board.color, move))
        board.play(move)
    return out


# --- time -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class TimePolicy:
    """How much of the clock a move may plan on. What's kept for the endgame's solve is the
    player's to keep (`cpu_match.Solving.kept`): a clock already planned to the solve, told to
    it, would keep it twice.

    Attributes:
        reserve: seconds of the clock never planned on, for the network and the server's whole
            seconds (it prints clocks truncated to the second)
        reserve_share: and this share of what's left
        extension_share: on the extension (a soft timeout: the game is lost by at least the
            least margin already), this share of it a move, so as never to run out of it too
    """

    reserve: float = 1.5
    reserve_share: float = 0.03
    extension_share: float = 0.1

    def seconds_left(self, view: GameView) -> float:
        """What the player may plan the rest of the game on."""
        if view.extended:
            # no planning on the extension: a fixed share a move
            return 0.0
        usable = view.seconds_left * (1 - self.reserve_share) - self.reserve
        # an increment comes after each move, the game's half of the empties ours
        usable += view.increment * max(view.empties // 2 - 1, 0)
        return max(usable, 0.0)

    def extension_move(self, view: GameView) -> float:
        """A move's seconds on the extension."""
        return max(view.extension - self.reserve, 0.0) * self.extension_share


# --- evals ----------------------------------------------------------------------------------

# discs for a root value: discs = DISC_SCALE * atanh(value), capped at 64. The value is about
# P(win) - P(loss); a margin of 10 discs is, for a net and Edax alike, about a 90% win
DISC_SCALE = 6.5


def value_discs(value: float) -> float:
    """A search's root value, -1 to 1, as discs for the player to move."""
    value = min(max(value, -0.999999), 0.999999)
    return max(-64.0, min(64.0, DISC_SCALE * math.atanh(value)))


def eval_discs(last_value: tuple[str, float] | None, fallback: float = 0.0) -> float:
    """`NetPlayer.last_value` in discs."""
    if last_value is None:
        return fallback
    kind, value = last_value
    if kind == "margin":
        return value
    if kind == "wdl":
        # a win, draw or loss, no more: the least margin each way
        return 2.0 * (value > 0) - 2.0 * (value < 0)
    return value_discs(value)


class FirstLegal:
    """A `Bridge` that plays the first legal move at once: for testing the protocol."""

    def __init__(self, seconds: float = 0.0):
        self.seconds = seconds
        self.forgotten: list[str] = []

    def decide(self, view: GameView, cores: int) -> Decision:
        time.sleep(self.seconds)
        legal = Board.at(view.black, view.white, view.color).legal()
        return Decision(legal[0] if legal else PASS, 0.0, self.seconds)

    def observe(self, view: GameView) -> None:
        pass

    def forget(self, game_id: str) -> None:
        self.forgotten.append(game_id)

    def close(self) -> None:
        pass


# --- the net player -------------------------------------------------------------------------


class EnginePool:
    """Engines of the network, one per search under way, each of the cores it was given: an
    `Engine` searches one tree at a time."""

    def __init__(self, make: Callable[[int], Any]):
        self.make = make
        self.idle: dict[int, list[Any]] = {}
        self.lock = threading.Lock()

    def acquire(self, cores: int) -> Any:
        with self.lock:
            free = self.idle.get(cores)
            if free:
                return free.pop()
        return self.make(cores)

    def release(self, cores: int, engine: Any) -> None:
        with self.lock:
            self.idle.setdefault(cores, []).append(engine)


# a ponder searches in slices of this many seconds, checking its tree's size between them
PONDER_SLICE = 0.5


class _Ponder:
    """A game's tree searched by an engine on a thread of its own, until `stop` (or the tree has
    `max_nodes` nodes, or nothing left to evaluate)."""

    def __init__(self, game: "_Game", engine: Any, cores: int, max_nodes: int = 0):
        self.game = game
        self.engine = engine
        self.cores = cores
        self.max_nodes = max_nodes
        self.leaves = 0
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="ponder", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        search = self.game.player.search
        while not self.stopping:
            if self.max_nodes and search.num_nodes() >= self.max_nodes:
                break
            leaves = self.engine.search(search, seconds=PONDER_SLICE)
            self.leaves += leaves
            if leaves == 0:
                # halted, or the tree's all games' ends
                break

    def stop(self) -> int:
        """Stops it, the batches in flight finished. Returns the leaves it evaluated."""
        self.stopping = True
        search = self.game.player.search
        search.halt(True)
        self.thread.join()
        search.halt(False)
        return self.leaves


class PlayerBridge:
    """`Bridge` for players like `cpu_match.NetPlayer`, made by `new_player(seed)`, each
    searching with an engine of `make_engine(cores)` (set as its `evaluate` for each move).

    With `ponder`, the games where the opponent is to move are searched meanwhile (see the
    module's doc): a thread of its own starts and stops the ponders, so `observe` returns at
    once, and a `decide` stops them all before it searches."""

    def __init__(
        self,
        new_player: Callable[[int], Any],
        make_engine: Callable[[int], Any],
        cores: int,
        taper: int = 0,
        time_policy: TimePolicy | None = None,
        solve_from: int = 0,
        ponder: bool = False,
        ponder_nodes: int = 0,
    ):
        """`solve_from`: the empties from which the player solves the endgame (its `solve`); the
        moves before it share the clock evenly (no `taper`), the player keeping the solve's
        reserve (`cpu_match.Solving.kept`); no ponder when our next move is solved.
        `ponder_nodes`: a ponder stops once its game's tree has this many nodes (0: never)."""
        self.new_player = new_player
        self.solve_from = solve_from
        self.pool = EnginePool(make_engine)
        self.cores = cores
        self.taper = taper
        self.time = time_policy or TimePolicy()
        self.players: dict[str, _Game] = {}
        self.seed = 0
        self.ponders = ponder
        self.ponder_nodes = ponder_nodes
        # guards the games, the ponders and what's being decided; held only briefly (a ponder
        # stops within a batch)
        self.lock = threading.RLock()
        # games a decide has under way
        self.deciding: set[str] = set()
        # games whose player is at the opponent's turn, told every move: those to ponder
        self.ready: set[str] = set()
        self.pondering: dict[str, _Ponder] = {}
        self.wake = threading.Event()
        self.closed = False
        if ponder:
            self.scheduler = threading.Thread(
                target=self._schedule_loop, name="ponder-scheduler", daemon=True
            )
            self.scheduler.start()

    # -- pondering

    def _schedule_loop(self) -> None:
        while True:
            self.wake.wait()
            self.wake.clear()
            if self.closed:
                return
            with self.lock:
                self._schedule()

    def _worth_pondering(self, game: "_Game") -> bool:
        """Not if our next move is solved, or the game's over."""
        me, opp, _, terminated = game.player.search.position()
        if terminated:
            return False
        # the depth the game plans to solve from, once its first move has seen the clock
        solve_from = getattr(game.player, "solve_from", None)
        if solve_from is None:
            solve_from = self.solve_from
        empties = 64 - (me | opp).bit_count()
        return not (solve_from and empties - 1 <= solve_from)

    def _schedule(self) -> None:
        """Ponders the games ready, the cores split between them, unless we're deciding."""
        if self.deciding or self.closed:
            self._stop_all()
            return
        # a game pondering now was worth it when it started, its root the same since (and
        # its tree not to be read while an engine searches it)
        wanted = sorted(
            gid
            for gid in self.ready
            if gid in self.players
            and (gid in self.pondering or self._worth_pondering(self.players[gid]))
        )[: self.cores]
        share = max(1, self.cores // max(len(wanted), 1))
        for gid, ponder in list(self.pondering.items()):
            if gid not in wanted or ponder.cores != share:
                self._stop(gid)
        for gid in wanted:
            if gid not in self.pondering:
                game = self.players[gid]
                engine = self.pool.acquire(share)
                game.player.evaluate = engine
                self.pondering[gid] = _Ponder(game, engine, share, self.ponder_nodes)

    def _stop(self, game_id: str) -> None:
        ponder = self.pondering.pop(game_id, None)
        if ponder is not None:
            ponder.game.pondered += ponder.stop()
            self.pool.release(ponder.cores, ponder.engine)

    def _stop_all(self) -> None:
        for gid in list(self.pondering):
            self._stop(gid)

    def _game(self, view: GameView) -> "_Game":
        """The game's player, told every move so far: its tree kept if what it was told before
        is how the game went, else started afresh from the server's position."""
        game = self.players.get(view.game_id)
        moves = list(view.moves)
        if (
            game is None
            or game.start != view.start
            or moves[: len(game.moves)] != game.moves
        ):
            self.seed += 1
            game = _Game(self.new_player(self.seed), view.start)
            game.player.set_position(*view.start)
            self.players[view.game_id] = game
        for color, move in colored_moves(view.start, moves)[game.told :]:
            game.player.play(color, move)
            game.told += 1
        game.moves = moves
        # what the player was told should be the server's board; if not, start from that
        board = replay(view.start, moves)
        if board.discs != [view.black, view.white]:
            game.desyncs += 1
            game.player.set_position(view.black, view.white, view.color)
            game.start = (view.black, view.white, view.color)
            game.moves, game.told = [], 0
        return game

    def decide(self, view: GameView, cores: int) -> Decision:
        start = time.perf_counter()
        board = Board.at(view.black, view.white, view.color)
        legal = board.legal()
        if not legal:
            return Decision(PASS, 0.0, time.perf_counter() - start, {"pass": True})
        with self.lock:
            # every core to our searches: no ponder while one's under way
            self.deciding.add(view.game_id)
            self.ready.discard(view.game_id)
            self._stop_all()
            game = self._game(view)
        pondered, game.pondered = game.pondered, 0
        # the visits below the root when our search starts: what reuse and pondering kept
        carried = int(game.player.search.visits().sum()) if self.ponders else 0
        cores = max(1, min(cores, self.cores))
        engine = self.pool.acquire(cores)
        try:
            game.player.evaluate = engine
            if view.extended:
                # on the extension: a fixed share of it a move
                move = game.player.move(
                    view.color, 0.0, per_move=self.time.extension_move(view)
                )
            else:
                # solving, the moves before the solve share the clock evenly: solved, those
                # after it take little (about a second, from 20 empties)
                game.player.taper = 0 if self.solve_from else self.taper
                move = game.player.move(view.color, self.time.seconds_left(view))
        finally:
            self.pool.release(cores, engine)
            with self.lock:
                self.deciding.discard(view.game_id)
            self.wake.set()
        game.told += 1
        game.moves = list(view.moves) + [move]
        value = game.player.last_value
        game.last_eval = eval_discs(value, game.last_eval)
        info = {"cores": cores, "value": value, "desyncs": game.desyncs}
        if self.ponders:
            info["pondered"] = pondered
            info["carried"] = carried
            with self.lock:
                if self.players.get(view.game_id) is game:
                    # the opponent's turn now, in the player's tree: ponder it
                    self.ready.add(view.game_id)
            self.wake.set()
        return Decision(move, game.last_eval, time.perf_counter() - start, info)

    def observe(self, view: GameView) -> None:
        """With `ponder`, a game where the opponent's to move is told its moves, to be pondered;
        one where it's our turn isn't pondered any more. Returns at once (a ponder stops within
        a batch). Without, nothing: the move it's told is played in its tree at our turn (and
        its cores are the opponent's meanwhile, in a contest on one machine)."""
        if not self.ponders:
            return
        with self.lock:
            if view.game_id in self.deciding:
                return
            self._stop(view.game_id)
            if view.mine is None or view.komi or view.color == view.mine:
                self.ready.discard(view.game_id)
            else:
                self._game(view)
                self.ready.add(view.game_id)
        self.wake.set()

    def forget(self, game_id: str) -> None:
        with self.lock:
            self._stop(game_id)
            self.ready.discard(game_id)
            self.players.pop(game_id, None)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self._stop_all()
            self.players.clear()
        self.wake.set()


class NetBridge(PlayerBridge):
    """`PlayerBridge` for `cpu_match.NetPlayer`s of one network, from `clock_match.py`'s `net:`
    options (`net`, `batch`, `solve`, `solve_exact`, `solve_share`, `taper`, ... ; `engine=1`
    is implied: the native int8 engine, a worker to a core). Unlike there, `solve_exact`
    defaults to 1: a game's score is its disc margin, so a won endgame should win by the most
    and a lost one lose by the least, which the network's win/loss value doesn't see."""

    def __init__(
        self,
        options: dict[str, str],
        cores: int | None = None,
        time_policy: TimePolicy | None = None,
    ):
        from cpu_match import NetPlayer, Solving, make_engine
        from cpu_match import Player as NetSpec

        options = dict(options)
        options.pop("engine", None)
        cores = cores or int(options.pop("threads", len(os.sched_getaffinity(0))))
        options.pop("threads", None)
        ponder = bool(int(options.pop("ponder", 0)))
        ponder_nodes = int(options.pop("ponder_nodes", 0))
        solving = None
        if "solve" in options:
            solving = Solving(
                empties=int(options.pop("solve")),
                # GGS scores games by discs: solve for the best margin, not just the win
                exact=bool(int(options.pop("solve_exact", 1))),
                share=float(options.pop("solve_share", Solving.share)),
                threads=cores,
            )
        taper = int(options.pop("taper", 0))
        fields = ",".join(f"{k}={v}" for k, v in options.items())
        spec = NetSpec.parse(f"visits=0,{fields}", options["net"], "native")

        def new_player(seed: int) -> NetPlayer:
            # its engine is set for each move (see `decide`)
            player = NetPlayer(spec.settings, None, taper, seed, 0, solving)  # type: ignore[arg-type]
            return player

        super().__init__(
            new_player,
            lambda n: make_engine(spec.net, spec.backend, n, spec.settings.batch),
            cores,
            taper,
            time_policy,
            solve_from=solving.empties if solving else 0,
            ponder=ponder,
            ponder_nodes=ponder_nodes,
        )
        # an engine of every core, made now: the first move shouldn't wait for it
        self.pool.release(cores, self.pool.acquire(cores))
        if ponder and cores > 1:
            # and two of half of them: two games pondered at once (a synchro match)
            halves = [self.pool.acquire(cores // 2) for _ in range(2)]
            for engine in halves:
                self.pool.release(cores // 2, engine)


@dataclass
class _Game:
    player: Any  # cpu_match.NetPlayer
    start: tuple[int, int, str]
    moves: list[int | None] = field(default_factory=list)
    # the moves (not passes) the player has been told, its own included
    told: int = 0
    last_eval: float = 0.0
    desyncs: int = 0
    # leaves pondered since our last move
    pondered: int = 0


def make_bridge(options: dict[str, str], cores: int) -> Bridge:
    """The player `--player` names: `net=NAME,...` on the CPU's native engine (`NetBridge`)."""
    options = dict(options)
    device = options.pop("device", "cpu")
    if device == "cpu":
        return NetBridge(options, cores)
    raise ValueError(f"no GGS player for device={device}")
