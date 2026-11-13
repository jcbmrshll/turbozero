"""Othello engines to test against, each driven as a subprocess over GTP (the Go Text
Protocol, which both speak for Othello). One process plays one game.

- Edax (https://github.com/okuhara/edax-reversi-AVX): build it with `setup_edax.sh`
- Egaroucid (https://github.com/Nyanyan/Egaroucid): build it with `setup_egaroucid.sh`

Both play at a "level" L, with their opening books off: up to level 10, they search L moves
ahead, and solve the game exactly once 2L or fewer squares are empty. Or on a clock, books on,
as `ClockedEngine`s (see `clock_match.py`).

An engine started with `analyse` also says, after each move it makes, what its search made
of the position (`GTPEngine.analysis`), read from the logs it writes as it searches; it plays
the same moves either way. Edax's search scores every legal move, each with the line it
expects after it; Egaroucid's scores only the move it plays. To score the moves of the
positions it didn't move from, Edax has an analyst (`EdaxAnalyst`), a process of its own, so
that searching them can't change the moves the one playing makes.
"""

import hashlib
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from game import PASS, action, square

CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "turbozero"
    / "othello"
)


DIRECTIONS = (1, -1, 8, -8, 9, -9, 7, -7)


def flips(mine: int, theirs: int, move: int) -> int:
    """The discs that `move` turns, as a bitboard (bit 0 is a1, bit 7 h1, bit 63 h8)."""
    result = 0
    for step in DIRECTIONS:
        run, at = 0, move
        while True:
            column = at % 8
            at += step
            # off the board, or wrapped around from one edge to the other
            if not 0 <= at < 64 or abs(at % 8 - column) > 1:
                run = 0
                break
            if theirs >> at & 1:
                run |= 1 << at
            else:
                if not mine >> at & 1:
                    run = 0
                break
        result |= run
    return result


def board_string(
    black: int,
    white: int,
    black_char: str = "*",
    white_char: str = "O",
    empty: str = "-",
) -> str:
    """The 64 squares a1, b1, ..., h1, a2, ..., h8 of bitboards, a character each."""
    return "".join(
        black_char if black >> i & 1 else white_char if white >> i & 1 else empty
        for i in range(64)
    )


# how a row of a printed board starts
BOARD_ROWS = tuple(f"{row} " for row in range(1, 9))


def read_board(
    text: str, black_char: str, white_char: str, black_to_move: str
) -> tuple[int, int, str]:
    """(black's discs, white's, who's to move) of a board an engine printed: rows "1 ..." to
    "8 ...", a square a token after the row's number; black to move if `black_to_move` is in
    `text`."""
    black = white = 0
    rows = [
        line.split()
        for line in text.splitlines()
        if line[:2] in BOARD_ROWS and len(line.split()) >= 9
    ]
    assert len(rows) == 8, f"no board in {text!r}"
    for row in rows:
        for column, cell in enumerate(row[1:9]):
            bit = 1 << ((int(row[0]) - 1) * 8 + column)
            black |= bit if cell == black_char else 0
            white |= bit if cell == white_char else 0
    return black, white, "b" if black_to_move in text else "w"


class Board:
    """The rules, enough to referee: the discs, who's to move, and passes."""

    def __init__(self) -> None:
        # bits for black's and white's discs (d5 and e4 are black, d4 and e5 white)
        self.discs = [1 << 28 | 1 << 35, 1 << 27 | 1 << 36]
        self.black_to_move = True

    @classmethod
    def at(cls, black: int, white: int, color: str) -> "Board":
        """A position given whole: the discs as bitboards, and who's to move, "b" or "w"."""
        board = cls()
        board.discs = [black, white]
        board.black_to_move = color == "b"
        return board

    @property
    def color(self) -> str:
        return "b" if self.black_to_move else "w"

    def text(self) -> str:
        """The squares as `board_string`'s, then who's to move, "*" or "O"."""
        return board_string(self.discs[0], self.discs[1]) + (
            "*" if self.black_to_move else "O"
        )

    def legal(self) -> list[int]:
        """The moves of the player to move."""
        mine = self.discs[not self.black_to_move]
        theirs = self.discs[self.black_to_move]
        taken = mine | theirs
        return [m for m in range(64) if not taken >> m & 1 and flips(mine, theirs, m)]

    def play(self, move: int) -> None:
        """Plays a legal `move`, then passes for the next player if they have no move."""
        side = not self.black_to_move
        turned = flips(self.discs[side], self.discs[not side], move)
        if not turned:
            raise ValueError(f"illegal move {square(move)}")
        self.discs[side] |= turned | 1 << move
        self.discs[not side] &= ~turned
        self.black_to_move = not self.black_to_move
        if not self.legal():
            self.black_to_move = not self.black_to_move

    def margin(self) -> int:
        """Black's discs less white's."""
        return self.discs[0].bit_count() - self.discs[1].bit_count()


@dataclass(frozen=True)
class Analysis:
    """What an engine's search made of one move of a position.

    Attributes:
        line: the moves it expects, the move first; just the move, from an engine that
            doesn't say more
        score: the disc margin it expects at the end, for the player making the move
        depth: how many moves ahead it searched
        bound: 0 if `score` is exact; -1 if the move is worth at most `score`, +1 at least
            (a search stops on a move once it knows the move is no better than another)
    """

    line: list[int]
    score: int
    depth: int
    bound: int = 0


# under CACHE_DIR, where the setup scripts build the engines; a new build goes in a new
# directory, since a rating belongs to a build
ENGINE_BUILDS = "engines-2026-10"
# what Edax's `time_left` sets its level to: the clock decides how deep it goes
CLOCKED_LEVEL = 60


@cache
def _file_sha256(path: Path, size: int, modified: int) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """A file's hash, worked out once a process unless the file changes."""
    stat = path.stat()
    return _file_sha256(path, stat.st_size, stat.st_mtime_ns)


class GTPEngine:
    """An engine process. Subclasses say how to start it and how to read its board, and
    its logs."""

    name = ""
    setup_script = ""
    # roughly how much memory one process takes, to decide how many to run at once
    memory_per_process = 0

    def __init__(
        self,
        engine_dir: Path,
        level: int,
        analyse: bool = False,
        args: Sequence[str] = (),
        threads: int = 1,
    ):
        self.analyse = analyse
        self.threads = threads
        # what its search made of each move of the position it last moved from, best
        # first; None if it didn't say
        self.analysis: list[Analysis] | None = None
        # how many nodes the search of its last move took, if it said
        self.nodes: int | None = None
        # what it wrote before its last reply: when analysing, its stderr (its logs)
        # comes in line with its replies
        self.log: list[str] = []
        self.engine_dir = engine_dir
        self.level = level
        self.process = subprocess.Popen(
            [*self.command_line(engine_dir, level), *args],
            cwd=self.working_dir(engine_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if analyse else subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.command("boardsize 8")
        self.command("clear_board")

    @classmethod
    def default_dir(cls) -> Path:
        return CACHE_DIR / ENGINE_BUILDS / cls.name

    @classmethod
    def installed(cls, engine_dir: Path) -> bool:
        raise NotImplementedError

    def command_line(self, engine_dir: Path, level: int) -> list[str]:
        raise NotImplementedError

    def working_dir(self, engine_dir: Path) -> Path:
        return engine_dir

    def command(self, command: str) -> str:
        """Sends a command and returns the response (without its leading "= ")."""
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        # a response is one or more lines, the first starting "=" (or "?" for an error),
        # then a blank line; what comes before it goes in `log`
        self.log, lines = [], []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(f"{self.name} exited after {command!r}")
            if not lines and not line.startswith(("=", "?")):
                if line.strip():
                    self.log.append(line.rstrip())
            elif line.strip():
                lines.append(line.rstrip())
            elif lines:
                break
        if lines[0].startswith("?"):
            raise RuntimeError(f"{self.name} rejected {command!r}: {lines[0]}")
        return "\n".join([lines[0].lstrip("=").strip(), *lines[1:]])

    def play(self, color: str, move: int) -> None:
        """Tells the engine about a move by `color` ("b" or "w")."""
        # both engines pass by themselves: a command for one color passes for the other
        # if it has no move, and an explicit pass would then be one too many
        if move != PASS:
            self.command(f"play {color} {square(move)}")

    def set_position(self, black: int, white: int, color: str) -> None:
        """Starts the game from a position given whole, one moves can't reach (see `Board.at`)."""
        raise NotImplementedError(f"{self.name}'s GTP can't take a board")

    def genmove(self, color: str) -> int:
        """The engine's move for `color`, which it also plays."""
        move = action(self.command(f"genmove {color}"))
        if self.analyse:
            self.analysis = self.read_analysis(move)
        return move

    def read_analysis(self, move: int) -> list[Analysis] | None:
        """What the search that chose `move` made of the position, from its logs."""
        raise NotImplementedError

    def weights(self) -> list[Path]:
        """The files it evaluates positions with."""
        raise NotImplementedError

    def book_and_hash(self, arguments: list[str]) -> tuple[bool, int]:
        """Whether a process started with `arguments` plays from its book, and its hash table's
        size (bits or level, the engine's own unit)."""
        raise NotImplementedError

    def settings(self) -> dict[str, Any]:
        """Everything that decides how it plays: what it was started with, what it is."""
        # a log's name changes with each process, and has no part in how it plays
        arguments = [str(a) for a in self.process.args]  # pyright: ignore[reportGeneralTypeIssues]
        if "-search-log-file" in arguments:
            at = arguments.index("-search-log-file")
            del arguments[at : at + 2]
        book, hash_size = self.book_and_hash(arguments)
        return {
            "engine": self.name,
            "level": self.level,
            "threads": self.threads,
            "book": book,
            "hash": hash_size,
            "command": arguments[1:],
            "binary_sha256": file_sha256(Path(arguments[0])),
            "weights_sha256": {p.name: file_sha256(p) for p in self.weights()},
        }

    def discs(self) -> tuple[int, int]:
        """(black, white) discs on the engine's board."""
        raise NotImplementedError

    def close(self) -> None:
        self.process.kill()
        self.process.wait()


# a move in Edax's logs: a square, "f5" or "F5" (the case is the color), or a pass, "pa"
EDAX_MOVE = r"[a-hA-H][1-8]|[pP][aA]"
# Edax searches every move with a full window up to this depth (its MULTIPV_DEPTH); deeper,
# it stops on a move once it knows it's no better than the best so far
EDAX_FULL_WINDOW_DEPTH = 10
EDAX_NODES = re.compile(r"; nodes = (\d+) N;")
# the size of its hash table unless it's told otherwise (its `options.hash_table_size`)
EDAX_HASH_BITS = 21


def _last_value(arguments: list[str], option: str, default: str) -> str:
    """The value given to `option`, the last if there's more than one, as engines read them."""
    found = [i for i, a in enumerate(arguments[:-1]) if a == option]
    return arguments[found[-1] + 1] if found else default


def edax_bound(score: int, after: tuple[int, int] | None, depth: int) -> int:
    """Whether Edax's `score` of a move is exact (0), at most (-1) or at least (+1) the
    move's value, from the bounds Edax has on the value of the position after it, for the
    other player, if it has them, else from how deep it searched."""
    if after is None:
        return 0 if depth <= EDAX_FULL_WINDOW_DEPTH else -1
    # for the player making the move
    lower, upper = -after[1], -after[0]
    return 0 if lower == upper else -1 if score >= upper else 1


class Edax(GTPEngine):
    name = "edax"
    setup_script = "setup_edax.sh"
    # mostly its hash table (2^22 entries, about 130 MB); measured at about 165 MB
    memory_per_process = 170 << 20

    def __init__(
        self,
        engine_dir: Path,
        level: int,
        analyse: bool = False,
        args: Sequence[str] = (),
        threads: int = 1,
    ):
        self.search_log = None
        if analyse:
            # it logs its searches in detail to a file
            self.log_dir = tempfile.TemporaryDirectory(prefix="edax-")
            path = Path(self.log_dir.name) / "search.log"
            # there to be read from the start, before Edax opens it to write
            path.touch()
            self.search_log = open(path)  # noqa: SIM115 - read all its life, closed by close()
        super().__init__(engine_dir, level, analyse, args, threads)

    @classmethod
    def installed(cls, engine_dir: Path) -> bool:
        return (engine_dir / "bin" / "edax").exists()

    def command_line(self, engine_dir: Path, level: int) -> list[str]:
        # It finds its weights in data/eval.dat, under its working directory
        command = [str(engine_dir / "bin" / "edax"), "-gtp", "-q", "-l", str(level),
                   "-n", str(self.threads), "-book-usage", "off"]  # fmt: skip
        if self.search_log is not None:
            command += ["-search-log-file", self.search_log.name]
        return command

    def read_analysis(self, move: int) -> list[Analysis] | None:
        # its search log has a line each time its search finishes with a move of the
        # position: "1> current move:  --  [-64, +64], D6 => +02; pv = D6 :04@100%[+02,+02];
        # c6:03@100%[-02,-02]; ...", the move's score for the player making it, then the
        # line it expects, the moves after the first each with what Edax knows of the
        # position it leads to: the depth searched and bounds on its value, for the player
        # to move there. Each step of its search, a move deeper, ends with "id = 1 ; level =
        #  8@100% ; ab = [-64, +64]:", at the depth searched
        assert self.search_log is not None
        log = self.search_log.read()
        # each move's analysis in the latest step that searched it
        latest: dict[int, Analysis] = {}
        # the moves of the step being read: (line, score, bounds after the first move)
        step = []
        for text in log.splitlines():
            if text.startswith("id = "):
                depth = int(text.split("level =")[1].split("@")[0])
                for line, score, after in step:
                    latest[line[0]] = Analysis(
                        line, score, depth, edax_bound(score, after, depth)
                    )
                step = []
                continue
            found = re.search(
                r"current move: .*?, (\S+) => ([+-]\d+); pv = (.*)$", text
            )
            if not found:
                continue
            pv = found[3]
            # the line, up to where Edax knows no move ("--")
            tokens = re.findall(rf"(?:^|\s)({EDAX_MOVE}|--)(?=[\s:;]|$)", pv)
            if "--" in tokens:
                tokens = tokens[: tokens.index("--")]
            line = [action(t) for t in tokens] or [action(found[1])]
            after = re.search(
                rf"(?:{EDAX_MOVE}|--):\d+@\d+%\[([+-]\d+),([+-]\d+)\]", pv
            )
            step.append((line, int(found[2]), after and (int(after[1]), int(after[2]))))
        if not latest:
            # a move it played without searching
            return None
        # each step's total so far: "+00 < score = +00 < +00; time = 0:00.001; nodes = 6632 N; ..."
        self.nodes = int(nodes[-1]) if (nodes := EDAX_NODES.findall(log)) else None
        # best first: exact scores before bounds of the same score
        return sorted(latest.values(), key=lambda a: (-a.score, a.bound != 0))

    def weights(self) -> list[Path]:
        return [self.engine_dir / "data" / "eval.dat"]

    def book_and_hash(self, arguments: list[str]) -> tuple[bool, int]:
        return (
            _last_value(arguments, "-book-usage", "off") == "on",
            int(_last_value(arguments, "-h", str(EDAX_HASH_BITS))),
        )

    def discs(self) -> tuple[int, int]:
        counts = {}
        for line in self.command("showboard").splitlines():
            if "discs =" in line:
                counts[line.split(":")[0].split()[-1]] = int(
                    line.split("discs =")[1].split()[0]
                )
        return counts["*"], counts["O"]

    def set_position(self, black: int, white: int, color: str) -> None:
        # its GTP has no command to set a board, but loads a game from a file, a GGF one
        # starting from its board, BO (no GM, which it reads only in lower case)
        side = "*" if color == "b" else "O"
        with tempfile.TemporaryDirectory(prefix="edax-") as directory:
            path = Path(directory) / "start.ggf"
            path.write_text(f"(;BO[8 {board_string(black, white)} {side}];)\n")
            self.command(f"loadsgf {path}")

    def position(self) -> tuple[int, int, str]:
        """(black's discs, white's, who's to move) on its board, read from `showboard`."""
        return read_board(self.command("showboard"), "*", "O", "* to move")

    def close(self) -> None:
        super().close()
        if self.search_log is not None:
            self.search_log.close()
            self.log_dir.cleanup()


class EdaxAnalyst:
    """An Edax process at a level that scores every legal move of a position it's given,
    each with the line it expects after it, over the NBoard protocol, whose `hint` searches
    each move with a full window. NBoard's reply cuts each line to 4 moves, so Edax also logs
    each search, line and all, to a file."""

    def __init__(self, engine_dir: Path, level: int):
        self.log_dir = tempfile.TemporaryDirectory(prefix="edax-")
        path = Path(self.log_dir.name) / "nboard.log"
        # there to be read from the start, before Edax opens it to write
        path.touch()
        self.log = open(path)  # noqa: SIM115 - read all its life, closed by close()
        # at verbosity 1 it logs each search's result, its line as long as fits the width
        command = [str(engine_dir / "bin" / "edax"), "-nboard", "-verbose", "1",
                   "-width", "250", "-ui-log-file", str(path), "-l", str(level),
                   "-n", "1", "-book-usage", "off"]  # fmt: skip
        # its errors, on stderr, come in line with its replies
        self.process = subprocess.Popen(
            command,
            cwd=engine_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.pings = 0
        self.send("nboard 1")

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def sync(self, after: str) -> list[str]:
        """Waits for Edax to have handled everything sent so far, and returns what it wrote
        meanwhile; an error it reported (a position it couldn't read, say) is raised."""
        assert self.process.stdout is not None
        self.pings += 1
        self.send(f"ping {self.pings}")
        lines = []
        while (
            line := self.process.stdout.readline().rstrip("\n")
        ) != f"pong {self.pings}":
            if line == "" and self.process.poll() is not None:
                raise RuntimeError(f"edax exited after {after!r}")
            if line.startswith("Error"):
                raise RuntimeError(f"edax rejected {after!r}: {line}")
            lines.append(line)
        return lines

    def analyse(self, board: np.ndarray, color: str) -> list[Analysis]:
        """Each legal move of the position, best first.

        Args:
            board: (64) the squares, a1, b1, ..., h8: 0 empty, 1 black, 2 white
            color: who's to move, "b" or "w", with a legal move
        """
        cells = "".join("-*O"[c] for c in board)
        side = "*" if color == "b" else "O"
        self.send(f"game (;GM[Othello]PC[NBoard]SZ[8]BO[8 {cells} {side}];)")
        self.send("hint 64")
        # a reply per move, "search <line> <score> 0 <depth>", best first
        replies = [
            line.split() for line in self.sync("hint") if line.startswith("search")
        ]
        # and a log line per move, in the same order, ending in the move's line, "D6 c3
        # D3 c4 F4 c5", after the depth, score, time, nodes and speed
        logged = [
            line for line in self.log.read().splitlines() if line.startswith("edax>")
        ]
        assert len(logged) == len(replies), (
            f"{len(replies)} moves but {len(logged)} logged"
        )
        return [
            Analysis(
                [action(t) for t in re.findall(rf"(?<!\S)(?:{EDAX_MOVE})(?!\S)", log)],
                int(score),
                int(depth),
            )
            for (_, _, score, _, depth), log in zip(replies, logged)
        ]

    def close(self) -> None:
        self.process.kill()
        self.process.wait()
        self.log.close()
        self.log_dir.cleanup()


class Egaroucid(GTPEngine):
    name = "egaroucid"
    setup_script = "setup_egaroucid.sh"
    # mostly its evaluation tables, whatever the hash size
    memory_per_process = 1300 << 20

    @classmethod
    def installed(cls, engine_dir: Path) -> bool:
        return (engine_dir / "bin" / "Egaroucid_for_Console.out").exists()

    def command_line(self, engine_dir: Path, level: int) -> list[str]:
        command = [str(engine_dir / "bin" / "Egaroucid_for_Console.out"), "-gtp",
                   "-quiet", "-nobook", "-level", str(level),
                   "-threads", str(self.threads)]  # fmt: skip
        # it logs its searches to stderr
        return command + (["-noise"] if self.analyse else [])

    def working_dir(self, engine_dir: Path) -> Path:
        # it finds its weights in resources/, under its working directory
        return engine_dir / "bin"

    def read_analysis(self, move: int) -> list[Analysis] | None:
        # its log has a line per step of its search, the last "main mid depth 6@100% value
        # 1 (raw 1) policy f6 ...": it scores only the move it plays
        found = [
            m
            for line in self.log
            if (m := re.search(r"depth (\d+)@\S+ value (-?\d+)", line))
        ]
        if not found:
            return None
        nodes = re.search(r"n_nodes (\d+)", found[-1].string)
        self.nodes = int(nodes[1]) if nodes else None
        return [Analysis([move], int(found[-1][2]), int(found[-1][1]))]

    def weights(self) -> list[Path]:
        resources = self.engine_dir / "bin" / "resources"
        return [resources / "eval.egev2", resources / "eval_move_ordering_end.egev"]

    def book_and_hash(self, arguments: list[str]) -> tuple[bool, int]:
        # its `DEFAULT_HASH_LEVEL`
        return "-nobook" not in arguments, int(_last_value(arguments, "-hash", "25"))

    def discs(self) -> tuple[int, int]:
        # its final_score counts for the side to move, not black; count the board
        # instead, where X is black and O is white
        rows = [
            line.split()[1:9]
            for line in self.command("showboard").splitlines()
            if line.split() and line.split()[0].isdigit()
        ]
        cells = [c for row in rows for c in row]
        assert len(cells) == 64, "unexpected showboard output"
        return cells.count("X"), cells.count("O")


class ClockedEngine:
    """An engine playing one game on its own clock, the way its author's time management
    expects: it's told what's left of the clock and plans the rest of the game itself, at
    no fixed level, with its shipped opening book on. Only `move` takes its time."""

    def set_position(self, black: int, white: int, color: str) -> None:
        """Starts the game from a position given whole (see `Board.at`), before any move."""
        raise NotImplementedError

    def play(self, color: str, move: int) -> None:
        """Tells the engine about a move by `color` ("b" or "w")."""
        raise NotImplementedError

    def move(self, color: str, seconds_left: float) -> int:
        """The engine's move for `color`, which it also plays."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class ClockedEdax(ClockedEngine):
    """Edax over GTP, whose `time_left` is to the millisecond."""

    def __init__(
        self,
        engine_dir: Path,
        seconds: float,
        threads: int,
        args: Sequence[str] = (),
    ):
        # the author's benchmarks use hash size 25, 2^25 entries; later options win
        self.engine = Edax(
            engine_dir,
            CLOCKED_LEVEL,
            args=["-h", "25", "-book-usage", "on", *args],
            threads=threads,
        )
        # whole seconds; it plans from each move's `time_left`
        self.engine.command(f"time_settings {max(round(seconds), 1)} 0 0")

    def set_position(self, black: int, white: int, color: str) -> None:
        self.engine.set_position(black, white, color)

    def play(self, color: str, move: int) -> None:
        self.engine.play(color, move)

    def move(self, color: str, seconds_left: float) -> int:
        self.engine.command(f"time_left {color} {max(seconds_left, 0):.3f} 0")
        return self.engine.genmove(color)

    def close(self) -> None:
        self.engine.close()


# Egaroucid's console's move: alone on a line, quiet; else in its search's table, a column
EGAROUCID_MOVE = r"([a-h][1-8])|\|[^|]*\|[^|]*\|\s*([a-h][1-8])\|.*"


class ClockedEgaroucid(ClockedEngine):
    """Egaroucid over its own console protocol: its GTP has no clock, and no way to set a
    board. It keeps the clock itself, in whole seconds to start with, and charges each search
    as it ends. It's told the clock (`settime`, in whole seconds) only when time has been added
    to it, an increment or an extension, that its own clock doesn't know of. Passes are its
    own: it plays them for whoever has no move.

    With `level`, it plays at that level, its book off, ignoring the clock. With `show_board`,
    it prints its board after each command, for `position` to read."""

    def __init__(
        self,
        engine_dir: Path,
        seconds: float,
        threads: int,
        args: Sequence[str] = (),
        level: int | None = None,
        show_board: bool = False,
    ):
        self.level = level
        clock = ["-time", str(max(round(seconds), 1))]
        if level is not None:
            clock = ["-level", str(level), "-nobook"]
        # quiet, it says only its move; showing its board, it's not quiet, so as to print it
        command = [str(engine_dir / "bin" / "Egaroucid_for_Console.out"),
                   *([] if show_board else ["-quiet", "-noboard"]),
                   "-threads", str(threads), *clock, *args]  # fmt: skip
        self.process = subprocess.Popen(
            command,
            cwd=engine_dir / "bin",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # what its own clock should have left, by our clock: each player's
        self.expected = {"b": float(round(seconds)), "w": float(round(seconds))}
        # replies once it has loaded its weights, so that no move is charged for that
        self.version()

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def reply(self) -> str:
        """A line of its output, without the prompts in front of it."""
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if line == "":
            raise RuntimeError("egaroucid exited")
        return line.replace(">", "").strip()

    def version(self) -> None:
        """Asks its version, and reads through the reply's last line."""
        self.send("version")
        while not self.reply().startswith("@website"):
            pass

    def set_position(self, black: int, white: int, color: str) -> None:
        side = "X" if color == "b" else "O"
        self.send(f"setboard {board_string(black, white, 'X', 'O', '-')}{side}")

    def position(self) -> tuple[int, int, str]:
        """(black's discs, white's, who's to move) on its board, as it prints it after its
        version; with `show_board` only."""
        self.version()
        lines = []
        while sum(line[:2] in BOARD_ROWS for line in lines) < 8:
            lines.append(self.reply())
        return read_board("\n".join(lines), "X", "O", "BLACK to move")

    def play(self, color: str, move: int) -> None:
        if move != PASS:
            self.send(f"play {square(move)}")

    def move(self, color: str, seconds_left: float) -> int:
        if self.level is None and seconds_left > self.expected[color] + 0.5:
            self.send(f"settime {'B' if color == 'b' else 'W'} {int(seconds_left)}")
            self.expected[color] = float(int(seconds_left))
        start = time.perf_counter()
        self.send("go")
        # its move alone on a line, or, not quiet, in its search's table: "|  3|  3@100%|  e3|"
        while not (found := re.fullmatch(EGAROUCID_MOVE, self.reply())):
            pass
        self.expected[color] = max(
            self.expected[color] - (time.perf_counter() - start), 0.0
        )
        return action(found[1] or found[2])

    def close(self) -> None:
        self.process.kill()
        self.process.wait()


# what makes an engine's `ClockedEngine`: (engine_dir, seconds, threads, args)
CLOCKED: dict[str, Callable[[Path, float, int, Sequence[str]], ClockedEngine]] = {
    Edax.name: ClockedEdax,
    Egaroucid.name: ClockedEgaroucid,
}
ENGINES: dict[str, type[GTPEngine]] = {e.name: e for e in (Edax, Egaroucid)}
# the engines with an analyst, by name
ANALYSTS: dict[str, type[EdaxAnalyst]] = {Edax.name: EdaxAnalyst}
