"""Othello engines to test against, each driven as a subprocess over GTP (the Go Text
Protocol, which both speak for Othello). One process plays one game.

- Edax (https://github.com/abulmo/edax-reversi): build it with `setup_edax.sh`
- Egaroucid (https://github.com/Nyanyan/Egaroucid): build it with `setup_egaroucid.sh`

Both play at a "level" L, with their opening books off: up to level 10, they search L moves
ahead, and solve the game exactly once 2L or fewer squares are empty.

An engine started with `analyse` also says, after each move it makes, what its search made
of the position (`GTPEngine.analysis`), read from the logs it writes as it searches; it plays
the same moves either way. Edax's search scores every legal move, each with the line it
expects after it; Egaroucid's scores only the move it plays. To score the moves of the
positions it didn't move from, Edax has an analyst (`EdaxAnalyst`), a process of its own, so
that searching them can't change the moves the one playing makes.
"""

import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from game import PASS, action, square

CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "turbozero"
    / "othello"
)


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
    ):
        self.analyse = analyse
        # what its search made of each move of the position it last moved from, best
        # first; None if it didn't say
        self.analysis: list[Analysis] | None = None
        # what it wrote before its last reply: when analysing, its stderr (its logs)
        # comes in line with its replies
        self.log: list[str] = []
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
        return CACHE_DIR / cls.name

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

    def genmove(self, color: str) -> int:
        """The engine's move for `color`, which it also plays."""
        move = action(self.command(f"genmove {color}"))
        if self.analyse:
            self.analysis = self.read_analysis(move)
        return move

    def read_analysis(self, move: int) -> list[Analysis] | None:
        """What the search that chose `move` made of the position, from its logs."""
        raise NotImplementedError

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
    ):
        self.search_log = None
        if analyse:
            # it logs its searches in detail to a file
            self.log_dir = tempfile.TemporaryDirectory(prefix="edax-")
            path = Path(self.log_dir.name) / "search.log"
            # there to be read from the start, before Edax opens it to write
            path.touch()
            self.search_log = open(path)  # noqa: SIM115 - read all its life, closed by close()
        super().__init__(engine_dir, level, analyse, args)

    @classmethod
    def installed(cls, engine_dir: Path) -> bool:
        return (engine_dir / "bin" / "edax").exists()

    def command_line(self, engine_dir: Path, level: int) -> list[str]:
        # one thread each: games run in parallel instead. It finds its weights in
        # data/eval.dat, under its working directory
        command = [str(engine_dir / "bin" / "edax"), "-gtp", "-q", "-l", str(level),
                   "-n", "1", "-book-usage", "off"]  # fmt: skip
        if self.search_log is not None:
            command += ["-search-log-file", self.search_log.name]
        return command

    def read_analysis(self, move: int) -> list[Analysis] | None:
        # its search log has a line each time its search finishes with a move of the
        # position: "1> current move:  --  [-64, +64], D6 => +02; pv = D6 :04@100%[+02,+02];
        # c6:03@100%[-02,-02]; ...", the move's score for the player making it, then the
        # line it expects, the moves after the first each with what Edax knows of the
        # position it leads to: the depth searched and bounds on its value, for the player
        # to move there. Each step of its search, a move deeper, ends with "=> BEST MOVE
        # FOUND! level = 8@100% ...", at the depth searched
        assert self.search_log is not None
        # each move's analysis in the latest step that searched it
        latest: dict[int, Analysis] = {}
        # the moves of the step being read: (line, score, bounds after the first move)
        step = []
        for text in self.search_log.read().splitlines():
            if "BEST MOVE FOUND" in text:
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
        # best first: exact scores before bounds of the same score
        return sorted(latest.values(), key=lambda a: (-a.score, a.bound != 0))

    def discs(self) -> tuple[int, int]:
        counts = {}
        for line in self.command("showboard").splitlines():
            if "discs =" in line:
                counts[line.split(":")[0].split()[-1]] = int(
                    line.split("discs =")[1].split()[0]
                )
        return counts["*"], counts["O"]

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
                   "-quiet", "-nobook", "-level", str(level), "-threads", "1"]  # fmt: skip
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
        return [Analysis([move], int(found[-1][2]), int(found[-1][1]))]

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


ENGINES: dict[str, type[GTPEngine]] = {e.name: e for e in (Edax, Egaroucid)}
# the engines with an analyst, by name
ANALYSTS: dict[str, type[EdaxAnalyst]] = {Edax.name: EdaxAnalyst}
