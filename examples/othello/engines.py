"""Othello engines to test against, each driven as a subprocess over GTP (the Go Text
Protocol, which both speak for Othello). One process plays one game.

- Edax (https://github.com/abulmo/edax-reversi): build it with `setup_edax.sh`
- Egaroucid (https://github.com/Nyanyan/Egaroucid): build it with `setup_egaroucid.sh`

Both search to a fixed depth set by their "level", with their opening books off.
"""

import os
import subprocess
from pathlib import Path

from game import PASS, action, square

CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "turbozero"
    / "othello"
)


class GTPEngine:
    """An engine process. Subclasses say how to start it and how to read its board."""

    name = ""
    setup_script = ""
    # roughly how much memory one process takes, to decide how many to run at once
    memory_per_process = 0

    def __init__(self, engine_dir: Path, level: int):
        self.process = subprocess.Popen(
            self.command_line(engine_dir, level),
            cwd=self.working_dir(engine_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
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
        # a response is one or more lines, then a blank line
        lines = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(f"{self.name} exited after {command!r}")
            if line.strip():
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
        return action(self.command(f"genmove {color}"))

    def discs(self) -> tuple[int, int]:
        """(black, white) discs on the engine's board."""
        raise NotImplementedError

    def close(self) -> None:
        self.process.kill()
        self.process.wait()


class Edax(GTPEngine):
    name = "edax"
    setup_script = "setup_edax.sh"
    # mostly its hash table (2^22 entries, about 130 MB); measured at about 165 MB
    memory_per_process = 170 << 20

    @classmethod
    def installed(cls, engine_dir: Path) -> bool:
        return (engine_dir / "bin" / "edax").exists()

    def command_line(self, engine_dir: Path, level: int) -> list[str]:
        # one thread each: games run in parallel instead. It finds its weights in
        # data/eval.dat, under its working directory
        return [str(engine_dir / "bin" / "edax"), "-gtp", "-q", "-l", str(level),
                "-n", "1", "-book-usage", "off"]  # fmt: skip

    def discs(self) -> tuple[int, int]:
        counts = {}
        for line in self.command("showboard").splitlines():
            if "discs =" in line:
                counts[line.split(":")[0].split()[-1]] = int(
                    line.split("discs =")[1].split()[0]
                )
        return counts["*"], counts["O"]


class Egaroucid(GTPEngine):
    name = "egaroucid"
    setup_script = "setup_egaroucid.sh"
    # mostly its evaluation tables, whatever the hash size
    memory_per_process = 1300 << 20

    @classmethod
    def installed(cls, engine_dir: Path) -> bool:
        return (engine_dir / "bin" / "Egaroucid_for_Console.out").exists()

    def command_line(self, engine_dir: Path, level: int) -> list[str]:
        return [str(engine_dir / "bin" / "Egaroucid_for_Console.out"), "-gtp", "-quiet",
                "-nobook", "-level", str(level), "-threads", "1"]  # fmt: skip

    def working_dir(self, engine_dir: Path) -> Path:
        # it finds its weights in resources/, under its working directory
        return engine_dir / "bin"

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
