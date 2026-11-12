"""A GGS client for our Othello player: it logs in to a GGS server, accepts the match requests
its policy allows, and plays each game's moves with the CPU player (see ggs_player.py). The
server is the source of truth for every board and clock; see README.md for running it.

    JAX_PLATFORMS=cpu taskset -c 0-7 uv run examples/othello/ggs_client/ggs_client.py \
        --host 127.0.0.1 --port 15000 --login tzbot --password-file ~/.ggs-tzbot \
        --player net=NET,batch=8,solve=20

`Session` is the client's logic and state, fed the server's messages and the player's
decisions, sending commands through a callback; `run` connects it to a server, keeps the
connection alive, and reconnects (with backoff) when it drops. A dropped connection doesn't end
a game: the server keeps it (the player becomes a "ghost", its clock running) and sends its
board again on login.

Synchro matches (`s8...`) are two games at once from the same start with colours swapped. The
server plays them in rounds: a round's clocks start once both games' moves of the round before
are in, and the opponent's move in the other game is shown only then. Passes are moves (sent as
"pass"), so each player moves in one game a round.
"""

import argparse
import logging
import os
import queue
import random
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# examples/othello, whose modules the player imports
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

import ggs_protocol as gp
from ggs_player import Bridge, Decision, GameView, replay

log = logging.getLogger("ggs")


# --- which requests to accept ---------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """Which match requests to accept; the rest are declined.

    Attributes:
        types: match types (as the server writes them, e.g. "s8r14", "8", "8k")
        min_seconds, max_seconds: our clock's initial time, in seconds
        max_increment: the most seconds a move adds
        min_extension: the least extension (a soft timeout's time to finish)
        same_clocks: only matches where both players have the same clock
        rated: "any", "rated" or "unrated"
        max_games: games at once (a synchro match is 2); also the server's `open`
        opponents: if any, only these
        blocked: never these
        resume_stored: accept a stored (adjourned) match being resumed, whatever its settings
    """

    types: tuple[str, ...] = ("s8r14", "s8r16", "s8r18", "8")
    min_seconds: float = 60.0
    max_seconds: float = 900.0
    max_increment: float = 60.0
    min_extension: float = 0.0
    same_clocks: bool = True
    rated: str = "any"
    max_games: int = 2
    opponents: frozenset[str] = frozenset()
    blocked: frozenset[str] = frozenset()
    resume_stored: bool = True

    def decide(self, request: gp.Request, me: str, games: int) -> tuple[bool, str]:
        """Whether to accept `request`, made to `me` playing `games` games, and why not."""
        other = request.other(me)
        if other in self.blocked:
            return False, f"{other} blocked"
        if self.opponents and other not in self.opponents:
            return False, f"{other} not an allowed opponent"
        if games + request.type.games > self.max_games:
            return False, f"{games} games already"
        if request.rating_mode == "S":
            return (self.resume_stored, "stored matches not resumed")
        if request.type.text.lower() not in {t.lower() for t in self.types}:
            return False, f"type {request.type.text}"
        if self.rated == "rated" and not request.rated:
            return False, "unrated"
        if self.rated == "unrated" and request.rated:
            return False, "rated"
        mine = request.clock_for(me)
        if self.same_clocks and request.opponent_clock not in (None, request.clock):
            return False, "different clocks"
        if not self.min_seconds <= mine.ini <= self.max_seconds:
            return False, f"clock {mine}"
        if mine.inc > self.max_increment or mine.ext < self.min_extension:
            return False, f"clock {mine}"
        if mine.ini_moves or mine.ext_moves or not mine.ini_loss:
            # moves-per-period clocks: our time management plans a whole game
            return False, f"clock {mine}"
        return True, ""


# --- a game, as the server last showed it ---------------------------------------------------


@dataclass
class GameState:
    """A game we're playing: its start and moves, the board and clocks now (the server's)."""

    id: str
    type: gp.MatchType
    start: tuple[int, int, str] | None = None
    moves: list[int | None] = field(default_factory=list)
    number: int = 0
    black: int = 0
    white: int = 0
    to_move: str = "b"
    my_color: str | None = None
    komi: float | None = None
    players: tuple[gp.PlayerLine, ...] = ()
    finished: bool = False
    result: float | None = None
    # when our turn began, by our clock, and the move number it's for
    turn_since: float = 0.0
    turn_number: int | None = None
    # the move number we've asked the player for, and sent a move for
    asked: int | None = None
    sent: int | None = None
    # our soft timeout: playing on the extension
    extended: bool = False
    # the boards that didn't match our replay of the moves
    desyncs: int = 0

    @property
    def match_id(self) -> str:
        return gp.match_id_of(self.id)

    @property
    def our_turn(self) -> bool:
        if self.finished:
            return False
        if self.my_color is None:
            # a komi game's first move: both players send one before colours are set
            return self.type.komi and self.komi is None and self.number == 0
        return self.to_move == self.my_color

    def my_clock(self) -> gp.ClockState | None:
        for p in self.players:
            if self.my_color and p.color == ("*" if self.my_color == "b" else "O"):
                return p.clock if isinstance(p.clock, gp.ClockState) else None
        # before colours are set, both clocks are alike
        for p in self.players:
            if isinstance(p.clock, gp.ClockState):
                return p.clock
        return None


@dataclass
class MatchState:
    id: str
    type: gp.MatchType
    players: tuple[str, str]
    rated: bool
    started: float
    # since when it's been unconfirmed: we logged in again and the server hasn't shown it
    unconfirmed: float | None = None


@dataclass
class Job:
    """A decision asked of the player: a game's move `number`, from `view`."""

    game_id: str
    number: int
    view: GameView
    # the searches under way with it, for the cores it gets
    sharing: int = 1


# --- the client's logic ---------------------------------------------------------------------


class Session:
    """The client's state and decisions, network aside. Feed it the server's messages
    (`receive`), the player's decisions (`decided`) and the time (`tick`); it sends commands
    through `send`, and asks the player for moves through `ask_move`."""

    def __init__(
        self,
        login: str,
        password: str,
        policy: Policy,
        send: Callable[[str], None],
        ask_move: Callable[[Job], None],
        forget_game: Callable[[str], None] = lambda game_id: None,
        clock: Callable[[], float] = time.monotonic,
        aform: str | None = None,
        keepalive: float = 60.0,
        asks: tuple[str, ...] = (),
        max_matches: int = 0,
    ):
        self.login = login
        self.password = password
        self.policy = policy
        self._send = send
        self.ask_move = ask_move
        self.forget_game = forget_game
        self.clock = clock
        self.aform = aform
        self.keepalive = keepalive
        self.asks = asks
        self.max_matches = max_matches
        self.games: dict[str, GameState] = {}
        self.matches: dict[str, MatchState] = {}
        self.requests: dict[str, gp.Request] = {}
        self.logged_in = False
        self.fatal: str | None = None
        self.last_keepalive = -1e18
        self.last_ask = -1e18
        self.resyncs: dict[str, float] = {}
        # finished matches: (id, opponent, type, our result in discs or None, reason)
        self.results: list[tuple[str, str, str, float | None, str]] = []
        self.errors: list[str] = []
        self.desyncs = 0
        self.draining = False
        # decisions made while the connection was down, sent once the game's board is back
        self.pending: dict[str, tuple[Job, Decision]] = {}

    # -- sending

    def send(self, command: str) -> None:
        self._send(command)

    def open_value(self) -> int:
        return 0 if self.draining or self.done else self.policy.max_games

    @property
    def done(self) -> bool:
        return bool(self.max_matches) and len(self.results) >= self.max_matches

    def configure(self) -> None:
        """Quiet, plain output, then the Othello service's settings."""
        for command in (
            "vt100 -",
            "bell -t -tc -tg -n -nc -ng -ni -nn",
            "verbose -news -faq -help -ack",
            # no channels: no chat
            "chann %",
        ):
            self.send(command)
        self.configure_service()

    def configure_service(self) -> None:
        s = gp.SERVICE
        rated = {"rated": "+", "unrated": "-"}.get(self.policy.rated, "+")
        for command in (
            f"tell {s} client -",
            f"tell {s} trust +",
            f"tell {s} rated {rated}",
            # requests sent to us only, not everyone's
            f"tell {s} request -",
            f"tell {s} bell -r -p -w -n -ns -nn -nt -ni -nr -nw -ta -to -tp",
        ):
            self.send(command)
        if self.aform is not None:
            self.send(f"tell {s} aform {self.aform}")
        self.send(f"tell {s} open {self.open_value()}")

    # -- the server's messages

    def receive(self, event: gp.Event) -> None:
        if self.fatal is not None:
            return
        now = self.clock()
        if isinstance(event, gp.Prompt):
            if event.kind == "login":
                self.send(self.login)
            else:
                self.send(self.password)
                self.logged_in = True
                self.configure()
                # a tournament's matches come back (their boards sent again as we log in);
                # any other match ended when the connection dropped: a rated one was stored,
                # and `continue` asks to resume it (as a new match, which we accept)
                for match in self.matches.values():
                    match.unconfirmed = now
                if self.policy.resume_stored:
                    self.send(f"tell {gp.SERVICE} continue")
                self.last_keepalive = now
        elif isinstance(event, gp.ServerError):
            log.warning("server: ERR %s", event.text)
            self.errors.append(event.text)
            if "assword" in event.text or "Login" in event.text:
                self.fatal = event.text
        elif isinstance(event, gp.ServiceStatus):
            if event.login == gp.SERVICE and event.online and self.logged_in:
                log.info("%s is back", gp.SERVICE)
                self.configure_service()
        elif isinstance(event, gp.Request):
            self.on_request(event)
        elif isinstance(event, gp.MatchStart):
            self.on_match_start(event, now)
        elif isinstance(event, gp.MatchEnd):
            self.on_match_end(event)
        elif isinstance(event, gp.BoardMessage):
            self.on_board(event, now)
        elif isinstance(event, gp.GameEnd):
            self.on_game_end(event)
        elif isinstance(event, gp.Timeout):
            game = self.games.get(event.game_id)
            log.warning(
                "%s: %s ran out of time%s", event.game_id, event.name,
                " (past the extension)" if event.fatal else "",
            )  # fmt: skip
            if game is not None and event.name == self.login:
                game.extended = True
                self.errors.append(f"timeout {event.game_id}")
        elif isinstance(event, gp.Offer):
            # undos, aborts and scores declined: they need both players
            log.info(
                "%s: %s asks for %s; not agreeing",
                event.game_id,
                event.name,
                event.kind,
            )
        elif isinstance(event, gp.ServiceError):
            self.on_service_error(event, now)
        elif isinstance(event, gp.ServiceNotice):
            log.debug("%s: %s", gp.SERVICE, event.text)

    def on_request(self, request: gp.Request) -> None:
        if not request.added:
            self.requests.pop(request.id, None)
            return
        if request.asker == self.login:
            log.info(
                "asked: %s %s %s", request.type.text, request.clock, request.opponent
            )
            self.requests[request.id] = request
            return
        if request.opponent != self.login:
            return
        accept, why = self.policy.decide(request, self.login, self.games_playing())
        if self.draining or self.done:
            accept, why = False, "not taking matches"
        verb = "accept" if accept else "decline"
        log.info(
            "%s %s from %s: %s %s %s%s", verb, request.id, request.asker, request.type.text,
            request.clock, "rated" if request.rated else "unrated", f" ({why})" if why else "",
        )  # fmt: skip
        self.send(f"tell {gp.SERVICE} {verb} {request.id}")

    def games_playing(self) -> int:
        return sum(m.type.games for m in self.matches.values())

    def on_match_start(self, event: gp.MatchStart, now: float) -> None:
        if self.login not in event.players:
            return
        self.matches[event.id] = MatchState(
            event.id, event.type, event.players, event.rated, now
        )
        log.info(
            "match %s: %s vs %s, %s %s", event.id, *event.players, event.type.text,
            "rated" if event.rated else "unrated",
        )  # fmt: skip

    def on_match_end(self, event: gp.MatchEnd) -> None:
        if self.login not in event.players:
            return
        self.matches.pop(event.id, None)
        for gid in [g for g in self.games if gp.match_id_of(g) == event.id]:
            self.games.pop(gid)
            self.forget_game(gid)
        ours = event.result
        if ours is not None and event.players[0] != self.login:
            ours = -ours
        opponent = (
            event.players[1] if event.players[0] == self.login else event.players[0]
        )
        self.results.append((event.id, opponent, event.type.text, ours, event.reason))
        log.info(
            "match %s over: %s %s", event.id,
            f"{ours:+.2f} for us" if ours is not None else "", event.reason,
        )  # fmt: skip
        if self.done:
            self.send(f"tell {gp.SERVICE} open 0")

    def on_game_end(self, event: gp.GameEnd) -> None:
        game = self.games.get(event.game_id)
        if game is not None:
            game.finished = True
            result = event.result
            if result is not None and event.players[0] != self.login:
                result = -result
            game.result = result
            self.forget_game(event.game_id)
        log.info("game %s over: %s %s", event.game_id, event.result, event.reason)

    def on_board(self, msg: gp.BoardMessage, now: float) -> None:
        names = [p.name for p in msg.players]
        if self.login not in names:
            return  # a game we watch
        game = self.games.get(msg.game_id)
        if game is None:
            game = self.games[msg.game_id] = GameState(msg.game_id, msg.type)
            if not msg.join:
                # an update for a game we don't know: get all of it
                self.resync(msg.match_id, now)
        if msg.match_id in self.matches:
            self.matches[msg.match_id].unconfirmed = None
        if msg.join:
            if msg.number == 0:
                game.start = self.position(msg.board, msg.to_move)
                game.moves = []
            else:
                game.start = self.position(msg.start_board, msg.start_to_move)
                game.moves = [m.square for m in msg.moves]
            self.resyncs.pop(msg.match_id, None)
        elif msg.number == game.number + 1 and msg.last_move is not None:
            game.moves.append(msg.last_move.square)
        elif msg.number != game.number:
            # a move missed, or more than one taken back
            log.warning(
                "%s: move %d after %d; resyncing", msg.game_id, msg.number, game.number
            )
            game.start = None
            self.resync(msg.match_id, now)
        game.number = msg.number
        game.black, game.white = gp.bitboards(msg.board)
        game.to_move = gp.color_letter(msg.to_move)
        game.komi = msg.komi
        game.players = msg.players
        mine = msg.color_of(self.login)
        game.my_color = gp.color_letter(mine) if mine else None
        if game.start is not None:
            board = replay(game.start, game.moves)
            if board.discs != [game.black, game.white]:
                game.desyncs += 1
                self.desyncs += 1
                log.warning(
                    "%s: board differs from the moves' replay; resyncing", msg.game_id
                )
                game.start = None
                self.resync(msg.match_id, now)
        if game.our_turn and game.turn_number != game.number:
            game.turn_since, game.turn_number = now, game.number
        held = self.pending.pop(game.id, None)
        if held is not None:
            self.decided(*held)
        self.maybe_move(game)

    @staticmethod
    def position(board: str, to_move: str) -> tuple[int, int, str]:
        black, white = gp.bitboards(board)
        return black, white, gp.color_letter(to_move)

    def resync(self, match_id: str, now: float) -> None:
        """Asks for the match's games whole (`moves`), at most every few seconds."""
        if now - self.resyncs.get(match_id, -1e18) < 5:
            return
        self.resyncs[match_id] = now
        self.send(f"tell {gp.SERVICE} moves {match_id}")

    def view(self, game: GameState) -> GameView:
        clock = game.my_clock()
        left = clock.left if clock else 0.0
        extension = clock.ext_left if clock else 0.0
        if game.extended:
            # past the soft timeout, what's left is the extension
            extension = left
        start = game.start or (game.black, game.white, game.to_move)
        moves = tuple(game.moves) if game.start else ()
        return GameView(
            game_id=game.id,
            start=start,
            moves=moves,
            black=game.black,
            white=game.white,
            color=game.to_move,
            seconds_left=left,
            extension=extension,
            extended=game.extended,
            increment=clock.inc if clock else 0.0,
            komi=game.type.komi and game.komi is None,
        )

    def maybe_move(self, game: GameState) -> None:
        if not game.our_turn or game.asked == game.number or game.sent == game.number:
            return
        game.asked = game.number
        sharing = sum(1 for g in self.games.values() if g.our_turn)
        self.ask_move(Job(game.id, game.number, self.view(game), max(sharing, 1)))

    # -- the player's decisions

    def decided(self, job: Job, decision: Decision) -> None:
        if not self.logged_in:
            # sent once the server shows the game again, if it's still this move
            self.pending[job.game_id] = (job, decision)
            return
        game = self.games.get(job.game_id)
        if game is None or game.number != job.number or not game.our_turn:
            log.info("%s: decision for move %d dropped", job.game_id, job.number)
            if game is not None and game.asked == job.number:
                game.asked = None
                self.maybe_move(game)
            return
        # charged from when the turn reached us (the server trusts what we say if it's no more
        # than 4 seconds under its own count); the search's own time if that's all we know
        waited = (
            self.clock() - game.turn_since if game.turn_number == job.number else 0.0
        )
        seconds = max(waited, decision.seconds)
        self.send(gp.move_command(game.id, decision.square, decision.eval, seconds))
        game.sent = game.number
        log.info(
            "%s move %d: %s/%.2f in %.2f s %s", game.id, game.number + 1,
            "pass" if decision.square is None else gp.square(decision.square),
            decision.eval, seconds, decision.info,
        )  # fmt: skip

    def on_service_error(self, event: gp.ServiceError, now: float) -> None:
        if "not registered" in event.text:
            log.warning(
                "%s: this login isn't registered, so plays unrated games only",
                gp.SERVICE,
            )
            return
        log.warning("%s: ERR %s", gp.SERVICE, event.text)
        self.errors.append(event.text)
        # an error about one of our games (an illegal move, not our turn): get it whole again
        words = event.text.split()
        target = words[0] if words and words[0].startswith(".") else None
        for game in self.games.values():
            ours = target is None or target in (game.id, game.match_id)
            if ours and game.sent == game.number:
                game.sent = game.asked = None
                self.resync(game.match_id, now)

    # -- time

    def tick(self) -> None:
        """Keeps the connection alive, and asks for matches if it's to."""
        now = self.clock()
        if not self.logged_in:
            return
        for match in list(self.matches.values()):
            if match.unconfirmed is not None and now - match.unconfirmed > 15:
                # the server didn't show it again: adjourned when the connection dropped
                self.on_match_end(
                    gp.MatchEnd(match.id, match.players, match.type, match.rated, None,
                                "adjourned: our connection dropped")
                )  # fmt: skip
        if now - self.last_keepalive >= self.keepalive:
            self.last_keepalive = now
            self.send(f"tell {gp.SERVICE} open {self.open_value()}")
            if self.policy.resume_stored and not self.draining:
                # asks to resume our stored matches whose opponents are here
                self.send(f"tell {gp.SERVICE} continue")
        idle = not self.matches and not any(
            r.asker == self.login for r in self.requests.values()
        )
        if (
            self.asks
            and idle
            and not self.draining
            and not self.done
            and now - self.last_ask > 10
        ):
            self.last_ask = now
            self.send(f"tell {gp.SERVICE} ask {random.choice(self.asks)}")

    def drain(self) -> None:
        """Takes no more matches, finishing those under way."""
        self.draining = True
        self.send(f"tell {gp.SERVICE} open 0")

    def disconnected(self) -> None:
        """The connection dropped: games go on at the server; their boards come again."""
        self.logged_in = False
        for game in self.games.values():
            # a move sent but not shown may not have arrived; the board will say
            if game.sent == game.number:
                game.sent = game.asked = None


# --- the network ----------------------------------------------------------------------------


class Transcript:
    """Everything sent and received, with times, to a file (the password left out)."""

    def __init__(self, path: Path | None, password: str):
        self.file = open(path, "a", buffering=1) if path else None  # noqa: SIM115
        self.password = password
        self.start = time.monotonic()

    def write(self, direction: str, text: str) -> None:
        if self.file is None:
            return
        if direction == ">" and text == self.password:
            text = "********"
        stamp = time.monotonic() - self.start
        for line in text.splitlines() or [""]:
            self.file.write(f"{stamp:10.3f} {direction} {line}\n")


class Scheduler:
    """Runs the player's decisions on threads, a thread a game, handing each its share of the
    cores; results come back on the client's queue."""

    def __init__(self, bridge: Bridge, cores: int, events: "queue.Queue[tuple]"):
        self.bridge = bridge
        self.cores = cores
        self.events = events
        self.lock = threading.Lock()
        self.busy = 0

    def submit(self, job: Job) -> None:
        def work() -> None:
            with self.lock:
                self.busy += 1
                running = self.busy
            try:
                cores = max(1, self.cores // max(job.sharing, running))
                decision = self.bridge.decide(job.view, cores)
                self.events.put(("decision", job, decision))
            except Exception as e:
                log.exception("%s: the player failed", job.game_id)
                self.events.put(("failed", job, e))
            finally:
                with self.lock:
                    self.busy -= 1

        threading.Thread(target=work, name=f"move{job.game_id}", daemon=True).start()


# the connection now, for `drop_connection`
CONNECTION: dict[str, socket.socket] = {}


def drop_connection(*_) -> None:
    """Shuts the connection (SIGUSR1): it reconnects as after any other drop. For testing."""
    sock = CONNECTION.get("socket")
    if sock is not None:
        log.warning("dropping the connection")
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def run(
    session: Session,
    host: str,
    port: int,
    events: "queue.Queue[tuple]",
    stop: threading.Event,
    transcript: Transcript,
    backoff: tuple[float, float] = (1.0, 60.0),
    dead_after: float = 200.0,
) -> None:
    """Connects `session` to the server until `stop` is set or it's done, reconnecting with
    exponential backoff (from `backoff[0]` to `backoff[1]` seconds, jittered) when the
    connection drops or goes silent for `dead_after` seconds."""
    delay = backoff[0]
    while not stop.is_set() and session.fatal is None:
        try:
            sock = socket.create_connection((host, port), timeout=15)
        except OSError as e:
            log.warning(
                "can't connect to %s:%d: %s; again in %.0f s", host, port, e, delay
            )
            stop.wait(delay * random.uniform(0.8, 1.2))
            delay = min(delay * 2, backoff[1])
            continue
        log.info("connected to %s:%d", host, port)
        CONNECTION["socket"] = sock
        sock.settimeout(None)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        lock = threading.Lock()
        alive = threading.Event()
        alive.set()

        def send(
            command: str, sock: socket.socket = sock, lock: threading.Lock = lock
        ) -> None:
            transcript.write(">", command)
            try:
                with lock:
                    sock.sendall((command + "\n").encode("latin-1"))
            except OSError as e:
                log.warning("send failed: %s", e)

        def read(sock: socket.socket = sock, alive: threading.Event = alive) -> None:
            while alive.is_set():
                try:
                    data = sock.recv(65536)
                except OSError:
                    data = b""
                if not data:
                    events.put(("closed", sock))
                    return
                events.put(("data", data.decode("latin-1")))

        session._send = send
        threading.Thread(target=read, name="reader", daemon=True).start()
        assembler = gp.Assembler()
        last_data = time.monotonic()
        connected_at = last_data
        try:
            while not stop.is_set() and session.fatal is None:
                if session.done and not session.matches:
                    session.send("quit")
                    stop.set()
                    break
                try:
                    item = events.get(timeout=0.25)
                except queue.Empty:
                    item = None
                if item is None:
                    for lines in assembler.flush():
                        dispatch(session, lines)
                    session.tick()
                    if time.monotonic() - last_data > dead_after:
                        log.warning(
                            "nothing from the server for %.0f s; reconnecting",
                            dead_after,
                        )
                        break
                    continue
                kind = item[0]
                if kind == "data":
                    last_data = time.monotonic()
                    transcript.write("<", item[1].replace("\r", ""))
                    for lines in assembler.feed(item[1]):
                        dispatch(session, lines)
                elif kind == "closed":
                    if item[1] is sock:
                        log.warning("connection closed")
                        break
                elif kind == "decision":
                    session.decided(item[1], item[2])
                elif kind == "failed":
                    game = session.games.get(item[1].game_id)
                    if game is not None and game.asked == item[1].number:
                        game.asked = None
                        session.maybe_move(game)
                session.tick()
        finally:
            alive.clear()
            try:
                sock.close()
            except OSError:
                pass
            session.disconnected()
        if stop.is_set() or session.fatal is not None:
            break
        # a connection that lasted resets the backoff
        delay = (
            backoff[0]
            if time.monotonic() - connected_at > 60
            else min(delay * 2, backoff[1])
        )
        log.info("reconnecting in %.0f s", delay)
        stop.wait(delay * random.uniform(0.8, 1.2))
    if session.fatal:
        log.error("stopped: %s", session.fatal)


def dispatch(session: Session, lines: list[str]) -> None:
    try:
        event = gp.parse(lines)
    except gp.ProtocolError as e:
        log.error("%s", e)
        session.errors.append(str(e))
        return
    session.receive(event)


# --- the command line -----------------------------------------------------------------------


def parse_options(text: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in text.split(",") if item)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Play Othello on a GGS server with our CPU player.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--login", required=True)
    ap.add_argument("--password-file", type=Path, required=True,
                    help="a file holding the password (mode 600)")  # fmt: skip
    ap.add_argument("--player", required=True,
                    help="the net player's options, as clock_match.py's net: "
                    "(net=NAME,batch=8,solve=20,...)")  # fmt: skip
    ap.add_argument("--cores", type=int, default=0,
                    help="cores the player uses (default: all it may run on)")  # fmt: skip
    ap.add_argument("--types", default="s8r14,s8r16,s8r18,8",
                    help="match types to accept, comma-separated")  # fmt: skip
    ap.add_argument("--min-seconds", type=float, default=60.0)
    ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("--max-increment", type=float, default=60.0)
    ap.add_argument("--min-extension", type=float, default=0.0)
    ap.add_argument("--handicap-clocks", action="store_true",
                    help="accept matches whose players' clocks differ")  # fmt: skip
    ap.add_argument("--rated", choices=("any", "rated", "unrated"), default="any")
    ap.add_argument("--max-games", type=int, default=2,
                    help="games at once (a synchro match is 2)")  # fmt: skip
    ap.add_argument("--opponents", default="", help="only these, comma-separated")
    ap.add_argument("--blocked", default="", help="never these, comma-separated")
    ap.add_argument("--aform", default=None,
                    help="a server-side auto-accept formula (see the service's 'formula' help)")  # fmt: skip
    ap.add_argument("--ask", action="append", default=[],
                    help='when idle, ask for a match: "TYPE CLOCK OPPONENT" (repeatable)')  # fmt: skip
    ap.add_argument("--matches", type=int, default=0,
                    help="quit after this many matches (0: never)")  # fmt: skip
    ap.add_argument("--keepalive", type=float, default=60.0)
    ap.add_argument("--transcript", type=Path, default=None,
                    help="append everything sent and received to this file")  # fmt: skip
    ap.add_argument("--results", type=Path, default=None,
                    help="write the matches' results here (JSON) on exit")  # fmt: skip
    ap.add_argument("--fake-player", action="store_true",
                    help="play the first legal move instantly (protocol testing, no network)")  # fmt: skip
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    password = args.password_file.read_text().strip()
    policy = Policy(
        types=tuple(t for t in args.types.split(",") if t),
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        max_increment=args.max_increment,
        min_extension=args.min_extension,
        same_clocks=not args.handicap_clocks,
        rated=args.rated,
        max_games=args.max_games,
        opponents=frozenset(o for o in args.opponents.split(",") if o),
        blocked=frozenset(o for o in args.blocked.split(",") if o),
    )
    cores = args.cores or len(os.sched_getaffinity(0))
    if args.fake_player:
        from ggs_player import FirstLegal

        bridge: Bridge = FirstLegal()
    else:
        from ggs_player import NetBridge

        bridge = NetBridge(parse_options(args.player), cores)
    events: queue.Queue[tuple] = queue.Queue()
    scheduler = Scheduler(bridge, cores, events)
    session = Session(
        args.login,
        password,
        policy,
        send=lambda command: None,
        ask_move=scheduler.submit,
        forget_game=bridge.forget,
        aform=args.aform,
        keepalive=args.keepalive,
        asks=tuple(args.ask),
        max_matches=args.matches,
    )
    stop = threading.Event()
    signals = {"count": 0}

    def on_signal(signum, frame) -> None:
        signals["count"] += 1
        if signals["count"] == 1 and session.matches:
            log.info("finishing the matches under way (signal again to quit now)")
            session.drain()
            session.max_matches = max(len(session.results) + len(session.matches), 1)
        else:
            stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, drop_connection)
    transcript = Transcript(args.transcript, password)
    run(session, args.host, args.port, events, stop, transcript)
    bridge.close()
    if args.results:
        import json

        args.results.write_text(
            json.dumps(
                {
                    "results": session.results,
                    "errors": session.errors,
                    "desyncs": session.desyncs,
                },
                indent=1,
            )
        )
    sys.exit(1 if session.fatal else 0)


if __name__ == "__main__":
    main()
