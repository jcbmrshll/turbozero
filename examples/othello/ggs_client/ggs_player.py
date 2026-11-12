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
    """What the client asks of a player."""

    def decide(self, view: GameView, cores: int) -> Decision: ...

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
    """How much of the clock a move may plan on.

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


class NetBridge:
    """`Bridge` for `cpu_match.NetPlayer`s of one network, from `clock_match.py`'s `net:`
    options (`net`, `batch`, `solve`, `solve_exact`, `solve_share`, `taper`, ... ; `engine=1`
    is implied: the native int8 engine, a worker to a core)."""

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
        self.cores = cores or int(options.pop("threads", len(os.sched_getaffinity(0))))
        options.pop("threads", None)
        self.solving = None
        if "solve" in options:
            self.solving = Solving(
                empties=int(options.pop("solve")),
                exact=bool(int(options.pop("solve_exact", 0))),
                share=float(options.pop("solve_share", Solving.share)),
                threads=self.cores,
            )
        self.taper = int(options.pop("taper", 0))
        fields = ",".join(f"{k}={v}" for k, v in options.items())
        self.spec = NetSpec.parse(f"visits=0,{fields}", options["net"], "native")
        self.pool = EnginePool(
            lambda n: make_engine(
                self.spec.net, self.spec.backend, n, self.spec.settings.batch
            )
        )
        # an engine of every core, made now: the first move shouldn't wait for it
        self.pool.release(self.cores, self.pool.acquire(self.cores))
        self.new_player = lambda seed: NetPlayer(
            self.spec.settings,
            None,
            self.taper,
            seed,
            0,
            self.solving,  # type: ignore[arg-type]
        )
        self.time = time_policy or TimePolicy()
        self.players: dict[str, _Game] = {}
        self.seed = 0

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
        game = self._game(view)
        cores = max(1, min(cores, self.cores))
        engine = self.pool.acquire(cores)
        try:
            game.player.evaluate = engine
            if view.extended:
                # on the extension: a fixed share of it a move, which `move_budget` gives
                # from this much over the moves left, untapered
                game.player.taper = 0
                per_move = self.time.extension_move(view)
                move = game.player.move(
                    view.color, per_move * (view.empties + 1) / 2 / 0.95
                )
            else:
                game.player.taper = self.taper
                move = game.player.move(view.color, self.time.seconds_left(view))
        finally:
            self.pool.release(cores, engine)
        game.told += 1
        game.moves = list(view.moves) + [move]
        value = game.player.last_value
        game.last_eval = eval_discs(value, game.last_eval)
        info = {"cores": cores, "value": value, "desyncs": game.desyncs}
        return Decision(move, game.last_eval, time.perf_counter() - start, info)

    def forget(self, game_id: str) -> None:
        self.players.pop(game_id, None)

    def close(self) -> None:
        self.players.clear()


@dataclass
class _Game:
    player: Any  # cpu_match.NetPlayer
    start: tuple[int, int, str]
    moves: list[int | None] = field(default_factory=list)
    # the moves (not passes) the player has been told, its own included
    told: int = 0
    last_eval: float = 0.0
    desyncs: int = 0
