"""What GGS, Michael Buro's Generic Game Server (https://skatgame.net/mburo/ggs/), says to an
Othello client, parsed; and the commands a client sends it. Nothing here touches a socket.

The server speaks lines of text (CRLF). Most of what matters comes from its Othello service,
`/os`, as `/os: <message>`, a message's further lines each starting with "|"; the server's own
prompt, `READY` (or `ALERT` when it isn't ready), follows what each command or tell produced.
The formats here are those of the server's source (GGS-Nov-09-2026: Service/GameLib/src/Game.C
and Match.C, Service/Othello/msg/format), checked against what a local server sends; the
transcripts in tests/fixtures/ggs are recorded from one.

A client in "client -" mode, as this one is, gets each game's board, clocks and last move in
text with every move (`join` and `update`):

    /os: update .1.0 s8r14 K?
    |  2: E2/0.50/0.30                       move number: move/eval/seconds
    |edax     (1720.0 *) 00:59,1:0//02:00,1:0  black's player, rating, colour, clock
    |alice    (1720.0 O) 00:59,1:0//02:00,1:0  then white's
    |
    |   A B C D E F G H
    | 1 - - - - - - - - 1
    ...
    | 8 - - - - - - - - 8
    |   A B C D E F G H
    |
    |* to move

A clock in a board message is `left,[N]moves:m1/[inc,[N]moves:m2]/ext,[N]moves:m3`, its times
in whole seconds; `left` is what the player has before the extension.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# the Othello service's login
SERVICE = "/os"
# a square's letters, a1 first: "*" black, "O" white, "-" empty
BLACK, WHITE, EMPTY = "*", "O", "-"


# --- times and clocks -----------------------------------------------------------------------


def parse_hhmmss(text: str) -> float:
    """Seconds of the server's `[days.]HH:MM:SS[.usec]`, any leading field left out (so "59",
    "01:00", "1:02:03")."""
    text = text.strip()
    sign = -1.0 if text.startswith("-") else 1.0
    text = text.lstrip("+-")
    days = 0.0
    if "." in text.split(":")[0] and text.count(":"):
        day, _, text = text.partition(".")
        days = float(day)
    seconds = 0.0
    for part in text.split(":"):
        seconds = 60 * seconds + float(part or 0)
    return sign * (days * 86400 + seconds)


def format_hhmmss(seconds: float) -> str:
    """`MM:SS`, or `HH:MM:SS` from an hour, for a clock the server reads."""
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class ClockSpec:
    """A clock as set for a match (`msg/clock`): `ini` seconds for the game, `inc` added each
    `inc_moves` moves, `ext` more once `ini` has run out. `ini_moves`/`ext_moves` 0 means the
    whole game; `ini_loss` False ("N") means running out of `ini` isn't a loss."""

    ini: float
    inc: float = 0.0
    ext: float = 0.0
    ini_moves: int = 0
    inc_moves: int = 1
    ext_moves: int = 0
    ini_loss: bool = True
    inc_add: bool = True
    ext_add: bool = True

    @classmethod
    def parse(cls, text: str) -> "ClockSpec":
        """`[ini[,[N]m1]]/[inc[,[N]m2]]/[ext[,[N]m3]]`, e.g. "01:00//02:00", "05:00,N0//02:00"."""
        parts = text.split("/")
        if len(parts) != 3:
            raise ValueError(f"a clock is ini/inc/ext: {text!r}")
        fields = []
        for part in parts:
            time, _, count = part.partition(",")
            flag = count.startswith("N")
            count = count.lstrip("N")
            fields.append(
                (
                    parse_hhmmss(time) if time else 0.0,
                    int(count) if count else None,
                    flag,
                )
            )
        (ini, m1, n1), (inc, m2, n2), (ext, m3, n3) = fields
        return cls(
            ini=ini,
            inc=inc,
            ext=ext,
            ini_moves=m1 or 0,
            inc_moves=m2 if m2 is not None else 1,
            ext_moves=m3 or 0,
            ini_loss=not n1,
            inc_add=not n2,
            ext_add=not n3,
        )

    def __str__(self) -> str:
        def part(seconds: float, moves: int, default: int, add: bool) -> str:
            text = format_hhmmss(seconds) if seconds else ""
            if moves != default or not add:
                text += (
                    ","
                    + ("" if add else "N")
                    + (str(moves) if moves != default else "")
                )
            return text

        return "/".join(
            (
                part(self.ini, self.ini_moves, 0, self.ini_loss),
                part(self.inc, self.inc_moves, 1, self.inc_add),
                part(self.ext, self.ext_moves, 0, self.ext_add),
            )
        )


@dataclass(frozen=True)
class ClockState:
    """A player's clock in a board message: `left` seconds before the extension (whole seconds,
    as the server prints them), `moves` made, `inc` and `ext` as set, `ext_left` the extension
    not yet added."""

    left: float
    moves: int
    inc: float
    ext_left: float

    @classmethod
    def parse(cls, text: str) -> "ClockState":
        """E.g. "00:59,1:0//02:00,1:0", "01:00,0:0/00:05,0:1/02:00,0:0"."""
        parts = text.split("/")
        if len(parts) != 3:
            raise ValueError(f"not a clock: {text!r}")

        def split(part: str) -> tuple[float, int]:
            time, _, rest = part.partition(",")
            moves = rest.lstrip("N").split(":")[0]
            return (parse_hhmmss(time) if time else 0.0), int(moves or 0)

        left, moves = split(parts[0])
        inc = split(parts[1])[0] if parts[1] else 0.0
        ext_left = split(parts[2])[0] if parts[2] else 0.0
        return cls(left, moves, inc, ext_left)


# --- match types ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchType:
    """A match type (`msg/types`): e.g. "s8r14" (synchro, 8x8, random start of 14 discs),
    "8", "8k" (komi), "8b" (the asker plays black). `text` is as the server writes it."""

    text: str
    size: int
    synchro: bool = False
    komi: bool = False
    random_discs: int = 0
    anti: bool = False
    color: str = ""  # "b" or "w": the asker's preference

    @classmethod
    def parse(cls, text: str) -> "MatchType":
        found = re.fullmatch(r"(s?)(88|\d+)(.*)", text.strip().lower())
        if not found:
            raise ValueError(f"not a match type: {text!r}")
        synchro, size, rest = found.groups()
        discs = re.search(r"r(\d+)", rest)
        return cls(
            text=text.strip(),
            size=int(size),
            synchro=bool(synchro),
            komi="k" in rest,
            random_discs=int(discs.group(1)) if discs else 0,
            anti="a" in rest,
            color="b" if "b" in rest else "w" if "w" in rest else "",
        )

    @property
    def games(self) -> int:
        """Games played at once."""
        return 2 if self.synchro else 1


# --- squares and moves ----------------------------------------------------------------------


def square(index: int) -> str:
    """A square's name, "A1" for 0 to "H8" for 63 (bit i is column i % 8, row i // 8)."""
    return "ABCDEFGH"[index % 8] + str(index // 8 + 1)


def square_index(name: str) -> int | None:
    """A square's index from its name ("d3" or "D3"); None for a pass ("PA", "pass")."""
    name = name.strip().upper()
    if name in ("PA", "PASS", "PS"):
        return None
    if len(name) != 2 or name[0] not in "ABCDEFGH" or name[1] not in "12345678":
        raise ValueError(f"not a square: {name!r}")
    return "ABCDEFGH".index(name[0]) + 8 * (int(name[1]) - 1)


@dataclass(frozen=True)
class MoveInfo:
    """A move as the server lists it: `square` (None for a pass), the eval its player sent
    (None if none), and its seconds."""

    square: int | None
    eval: float | None = None
    seconds: float | None = None

    @classmethod
    def parse(cls, text: str) -> "MoveInfo":
        """E.g. "F7/0.50/0.30", "c3/19.00/0.21", "PASS", "pass/0.00/0.01", "D3//1.20"."""
        move, *rest = text.split("/")
        values: list[float | None] = []
        for part in rest[:2]:
            values.append(float(part) if part else None)
        values += [None] * (2 - len(values))
        return cls(square_index(move), values[0], values[1])


def move_command(
    match_id: str, square_: int | None, eval_: float, seconds: float
) -> str:
    """`tell /os play <id> <move>/<eval>/<seconds>`: the eval in discs for the player moving
    (in a komi game's first move, the result it expects), the seconds it thought."""
    move = "pass" if square_ is None else square(square_)
    return f"tell {SERVICE} play {match_id} {move}/{eval_:.2f}/{max(seconds, 0.0):.2f}"


# --- what the server says -------------------------------------------------------------------


@dataclass(frozen=True)
class Prompt:
    """The server asking for the login ("login") or the password ("password")."""

    kind: str


@dataclass(frozen=True)
class Ready:
    """`READY`, after what a command produced; `alert` when it was `ALERT`."""

    alert: bool = False


@dataclass(frozen=True)
class ServerError:
    """An error from the server itself (`: ERR ...`), e.g. a wrong password."""

    text: str


@dataclass(frozen=True)
class ServiceStatus:
    """A login arriving or leaving that we're told of (`: + /os 1`)."""

    login: str
    online: bool


@dataclass(frozen=True)
class Request:
    """A match request, made (`added`) or gone (accepted, declined, cancelled):
    `/os: +   .1 1720.0 edax     01:00//02:00    s8r14 U 1720.0 alice`. The first player
    asked; the second is asked, `opponent_clock` theirs if it differs. A stored (adjourned)
    match being resumed is `S`, its stored id last: `... s8r14 S 1720.0 alice .1`."""

    id: str
    added: bool
    asker: str
    asker_rating: float
    clock: ClockSpec
    type: MatchType
    rating_mode: str  # "R" rated, "U" unrated, "S" a stored match resumed
    opponent: str
    opponent_rating: float
    opponent_clock: ClockSpec | None = None
    stored_id: str = ""

    @property
    def rated(self) -> bool:
        return self.rating_mode in ("R", "S")

    def other(self, me: str) -> str:
        """The player in it who isn't `me`."""
        return self.opponent if self.asker == me else self.asker

    def clock_for(self, me: str) -> ClockSpec:
        """`me`'s clock."""
        if me == self.opponent and self.opponent_clock is not None:
            return self.opponent_clock
        return self.clock


@dataclass(frozen=True)
class MatchStart:
    """`/os: + match .1 1720 edax 1720 alice s8r14 U`"""

    id: str
    players: tuple[str, str]
    ratings: tuple[float, float]
    type: MatchType
    rated: bool


@dataclass(frozen=True)
class MatchEnd:
    """`/os: - match .1 1720 edax 1720 alice s8r14 U +64.00`: `result` the first player's (in
    discs, the mean of a synchro match's games), None if it didn't finish: `reason` then says why
    ("aborted", "<player> left")."""

    id: str
    players: tuple[str, str]
    type: MatchType
    rated: bool
    result: float | None
    reason: str = ""


@dataclass(frozen=True)
class PlayerLine:
    """A player's line in a board message: name, rating, colour ("*", "O", "?" before a komi
    game's colours are set) and clock."""

    name: str
    rating: float
    color: str
    clock: ClockState | ClockSpec


@dataclass
class BoardMessage:
    """A game's state: `join` (sent on starting or rejoining a game, and on `moves`) or
    `update` (after each move). `board` is 64 squares, a1 first; `to_move` "*" or "O".

    A join after moves also has the game's start (`start_board`, `start_to_move`, the clocks as
    set) and every move (`moves`); an update has only the last (`last_move`, None before the
    first). `number` is the moves made so far, passes included."""

    join: bool
    game_id: str
    type: MatchType
    komi: float | None
    number: int
    last_move: MoveInfo | None
    players: tuple[PlayerLine, PlayerLine]
    board: str
    to_move: str
    moves: list[MoveInfo] = field(default_factory=list)
    start_board: str = ""
    start_to_move: str = ""

    @property
    def match_id(self) -> str:
        return match_id_of(self.game_id)

    def player(self, color: str) -> PlayerLine | None:
        for p in self.players:
            if p.color == color:
                return p
        return None

    def color_of(self, name: str) -> str | None:
        """`name`'s colour, None if it isn't playing (or colours aren't set yet)."""
        colors = [p.color for p in self.players if p.name == name]
        if len(colors) == 1 and colors[0] in (BLACK, WHITE):
            return colors[0]
        return None


@dataclass(frozen=True)
class GameEnd:
    """`/os: end .1.1 ( edax vs. alice ) +64.00`: one game of a synchro match over, `result` the
    first player's, or `reason` ("aborted", "player X left") if it didn't finish."""

    game_id: str
    players: tuple[str, str]
    result: float | None
    reason: str = ""


@dataclass(frozen=True)
class Timeout:
    """`/os: timeout .1.1 alice (*)`: `name` ran out of its time (`fatal` past its extension
    too, a hard timeout)."""

    game_id: str
    name: str
    color: str
    fatal: bool


@dataclass(frozen=True)
class Offer:
    """An opponent asking: `undo`, `abort`, or `score` (a result agreed)."""

    kind: str
    game_id: str
    name: str


@dataclass(frozen=True)
class ServiceError:
    """`/os: ERR ...` (e.g. an illegal move, a request it refused)."""

    text: str


@dataclass(frozen=True)
class ServiceNotice:
    """Anything else `/os` says (finger, help, trust-violation, comment, ...)."""

    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class Other:
    """Anything else: tells from people and channels, help text, acknowledgements."""

    lines: tuple[str, ...]


Event = (
    Prompt
    | Ready
    | ServerError
    | ServiceStatus
    | Request
    | MatchStart
    | MatchEnd
    | BoardMessage
    | GameEnd
    | Timeout
    | Offer
    | ServiceError
    | ServiceNotice
    | Other
)


def match_id_of(game_id: str) -> str:
    """A game's match: ".12" for ".12.0" and ".12.1" (a synchro match's games), ".12" for ".12"."""
    parts = game_id.split(".")
    return "." + parts[1] if len(parts) > 2 else game_id


class Assembler:
    """Joins the server's lines into messages: a line and the "|" lines after it. A message is
    complete once the next line that isn't one of its own arrives (the server's prompt follows
    everything it sends), or on `flush`."""

    def __init__(self) -> None:
        self.partial = ""
        self.pending: list[str] = []

    def feed(self, data: str) -> list[list[str]]:
        self.partial += data
        *lines, self.partial = self.partial.split("\n")
        messages: list[list[str]] = []
        for line in lines:
            line = line.rstrip("\r")
            if line.startswith("|") and self.pending:
                self.pending.append(line)
                continue
            if self.pending:
                messages.append(self.pending)
                self.pending = []
            if line.strip():
                self.pending = [line]
                # the prompts are complete as they come
                if line in ("READY", "ALERT") or line.startswith(": Enter "):
                    messages.append(self.pending)
                    self.pending = []
        return messages

    def flush(self) -> list[list[str]]:
        """What's pending, whether or not more might follow: after a pause in the stream, and
        the login prompt, which ends without a newline."""
        messages = [self.pending] if self.pending else []
        self.pending = []
        if self.partial.strip():
            messages.append([self.partial.rstrip("\r")])
        self.partial = ""
        return messages


def parse(lines: list[str]) -> Event:
    """What a message (`Assembler`'s) is."""
    head = lines[0]
    if head == "READY" or head == "ALERT":
        return Ready(alert=head == "ALERT")
    if head.startswith(": Enter login"):
        return Prompt("login")
    if head.startswith(": Enter your password"):
        return Prompt("password")
    if head.startswith(": ERR"):
        return ServerError(head[len(": ERR") :].strip())
    status = re.fullmatch(r": ([+-]) (\S+) \d+", head)
    if status:
        return ServiceStatus(status.group(2), status.group(1) == "+")
    prefix = SERVICE + ":"
    if head.startswith(prefix):
        body = head[len(prefix) :].strip()
        try:
            event = parse_service(body, [line[1:] for line in lines[1:]])
        except (ValueError, IndexError) as e:
            raise ProtocolError(f"can't read {lines!r}: {e}") from e
        if event is not None:
            return event
        return ServiceNotice(tuple([body] + [line[1:] for line in lines[1:]]))
    return Other(tuple(lines))


class ProtocolError(ValueError):
    """A message of a kind we know that we couldn't read."""


def parse_service(body: str, rest: list[str]) -> Event | None:
    """One of `/os`'s messages, from its first line (less `/os: `) and the rest (less "|")."""
    words = body.split()
    if not words:
        return None
    kind = words[0]
    if kind == "ERR":
        return ServiceError(body[3:].strip())
    if kind in ("+", "-") and len(words) > 1 and words[1] == "match":
        return parse_match(words)
    if kind in ("+", "-") and len(words) > 1 and words[1].startswith("."):
        return parse_request(words)
    if kind in ("join", "update"):
        return parse_board(kind == "join", words, rest)
    if kind == "end":
        return parse_game_end(body)
    if kind in ("timeout", "fatal-timeout"):
        # timeout .1.1 alice (*)
        return Timeout(
            words[1], words[2], words[3].strip("()"), kind == "fatal-timeout"
        )
    if (
        kind in ("undo", "abort", "score")
        and len(words) >= 3
        and words[1].startswith(".")
    ):
        return Offer(kind, words[1], words[2])
    return None


def parse_request(words: list[str]) -> Request:
    # + .1 1720.0 edax 01:00//02:00 s8r14 U 1720.0 alice [01:00//02:00] [.stored]
    sign, rid, r1, n1, clock, mtype, mode, r2, n2, *more = words
    stored = [w for w in more if w.startswith(".")]
    clocks = [w for w in more if not w.startswith(".")]
    return Request(
        id=rid,
        added=sign == "+",
        asker=n1,
        asker_rating=float(r1),
        clock=ClockSpec.parse(clock),
        type=MatchType.parse(mtype),
        rating_mode=mode,
        opponent=n2,
        opponent_rating=float(r2),
        opponent_clock=ClockSpec.parse(clocks[0]) if clocks else None,
        stored_id=stored[0] if stored else "",
    )


def parse_match(words: list[str]) -> MatchStart | MatchEnd:
    # + match .1 1720 edax 1720 alice s8r14 U
    # - match .1 1720 edax 1720 alice s8r14 U +64.00 [counter] | aborted | edax left
    sign, _, mid, r1, n1, r2, n2, mtype, mode, *rest = words
    players = (n1, n2)
    if sign == "+":
        return MatchStart(
            mid, players, (float(r1), float(r2)), MatchType.parse(mtype), mode == "R"
        )
    result: float | None = None
    reason = ""
    if rest and re.fullmatch(r"[+-]\d+(\.\d+)?", rest[0]):
        result = float(rest[0])
    else:
        reason = " ".join(rest)
    return MatchEnd(
        mid, players, MatchType.parse(mtype), mode in ("R", "T"), result, reason
    )


def parse_game_end(body: str) -> GameEnd:
    found = re.fullmatch(r"end (\S+) \( (\S+) vs\. (\S+) \)\s*(.*)", body.strip())
    if not found:
        raise ValueError(f"not a game end: {body!r}")
    gid, a, b, tail = found.groups()
    tail = tail.strip()
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", tail):
        return GameEnd(gid, (a, b), float(tail))
    return GameEnd(gid, (a, b), None, tail)


PLAYER_LINE = re.compile(r"^(\S+)\s+\(\s*(-?[\d.]+) ([*O?])\)\s+(\S+)\s*$")
MOVE_LINE = re.compile(r"^\s*(\d+): (\S+)\s*$")
ROW_LINE = re.compile(r"^\s*([1-8])((?: [*O\-.]){8}) \1\s*$")
TO_MOVE_LINE = re.compile(r"^([*O]) to move\s*$")


def parse_board(join: bool, words: list[str], rest: list[str]) -> BoardMessage:
    # join .1.0 s8r14 K?   /  update .1.0 s8r14 K-3.5
    gid, mtype = words[1], MatchType.parse(words[2])
    komi = None
    if len(words) > 3 and words[3].startswith("K") and words[3][1:] not in ("", "?"):
        komi = float(words[3][1:])
    players: list[PlayerLine] = []
    grids: list[str] = []
    to_moves: list[str] = []
    moves: list[tuple[int, MoveInfo | None]] = []
    rows: dict[int, str] = {}
    for line in rest:
        if found := PLAYER_LINE.match(line):
            name, rating, color, clock = found.groups()
            # a clock now has "left,moves:m1" first; as set (a join's start), "ini[,N]"
            now = re.search(r",N?\d+:\d+", clock.split("/")[0])
            parsed = ClockState.parse(clock) if now else ClockSpec.parse(clock)
            players.append(PlayerLine(name, float(rating), color, parsed))
        elif found := ROW_LINE.match(line):
            rows[int(found.group(1))] = (
                found.group(2).replace(" ", "").replace(".", "-")
            )
            if len(rows) == 8:
                grids.append("".join(rows[r] for r in range(1, 9)))
                rows = {}
        elif found := TO_MOVE_LINE.match(line):
            to_moves.append(found.group(1))
        elif found := MOVE_LINE.match(line):
            number, text = int(found.group(1)), found.group(2)
            moves.append((number, None if number == 0 else MoveInfo.parse(text)))
    if not grids or not to_moves or len(players) < 2 or not moves:
        raise ValueError("incomplete board message")
    number, last = moves[-1]
    message = BoardMessage(
        join=join,
        game_id=gid,
        type=mtype,
        komi=komi,
        number=number,
        last_move=last,
        players=(players[-2], players[-1]),
        board=grids[-1],
        to_move=to_moves[-1],
    )
    if join:
        message.moves = [m for _, m in moves if m is not None]
        message.start_board = grids[0]
        message.start_to_move = to_moves[0]
    return message


def bitboards(board: str) -> tuple[int, int]:
    """(black's discs, white's) as bits, a1 bit 0, of a board message's 64 squares."""
    black = sum(1 << i for i, c in enumerate(board) if c == BLACK)
    white = sum(1 << i for i, c in enumerate(board) if c == WHITE)
    return black, white


def color_letter(color: str) -> str:
    """ "b" or "w" for the server's "*" or "O"."""
    return {"*": "b", "O": "w"}[color]


def parse_stream(text: str) -> Iterable[Event]:
    """Every message of a recorded transcript, in order."""
    assembler = Assembler()
    for lines in assembler.feed(text) + assembler.flush():
        yield parse(lines)
