"""MoHex 2.0 (Benzene, https://github.com/alcalyn/benzene-vanilla-cmake), a strong Hex engine,
as a subprocess over GTP (the Go Text Protocol). One process plays one game. Build it with
`setup_mohex.sh`.

It plays at a "level" L: L Monte Carlo playouts a move, on one thread, however long they take.
Its search also proves wins and losses, by virtual connections and by inferior-cell analysis, so
it often answers without any playouts, and it may have a position solved early in a game on a
small board.

Benzene and pgx agree on the board: "c4" is column c, row 4, the neighbours are the same, and the
first colour (black) connects the top and bottom. They differ on the swap: Benzene's leaves the
stone where it is and exchanges the players' colours, where pgx's reflects the stone and
keeps the colours. Those are the same position, reflected. So that our board and the engine's
are always the same, a swap by either side sets the engine's board up as pgx's is, with the
swap rule off: the reflected stone, as colour 1's, on an otherwise empty board.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from game import HexGame

CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "turbozero" / "hex"
)


@dataclass(frozen=True)
class Analysis:
    """What an engine's search made of one move of a position.

    Attributes:
        line: the moves it expects, the move first
        score: the value it expects for the player making the move, from -1 to 1
        proven: whether that's proven, not an estimate
        visits: the playouts through the move
    """

    line: list[int]
    score: float
    proven: bool = False
    visits: int = 0


# a move of `mohex-values`' reply: "c4 .512@1234" or "c4 .512@12k", or proven with "W@1234" for a
# win and "L@1234" for a loss of the player to move, or "0" if nothing went through it
VALUE = re.compile(r"(\S+) (?:\.(\d+)@|([WL])@)?(\d+(?:\.\d+)?[km]?|0)?")


def parse_values(reply: str, game: HexGame) -> list[Analysis]:
    """Each move of a `mohex-values` reply that a playout went through, best first."""
    moves = []
    for name, mean, proven, count in VALUE.findall(reply.strip()):
        if mean:
            score = 2 * int(mean) / 10 ** len(mean) - 1
        elif proven:
            score = 1.0 if proven == "W" else -1.0
        else:
            continue
        moves.append(
            Analysis(
                [game.action(name)],
                score,
                bool(proven),
                parse_count(count),
            )
        )
    return sorted(moves, key=lambda a: (-a.score, -a.visits))


def parse_count(text: str) -> int:
    """A count as MoHex writes it: "12", "12k" or "1.20m"."""
    if text.endswith("k"):
        return int(text[:-1]) * 1000
    if text.endswith("m"):
        return int(float(text[:-1]) * 1_000_000)
    return int(text) if text else 0


class MoHex:
    """A MoHex process playing one game. Moves are the game's actions.

    Attributes:
        analysis: with `analyse`, what its search made of each move of the position it last
            moved from, best first; None if it answered without searching
    """

    name = "mohex"
    setup_script = "setup_mohex.sh"
    # the patterns it loads and a search tree of `MAX_NODES`, at most
    memory_per_process = 500 << 20
    # nodes a search keeps: enough for millions of playouts; a node takes 88 bytes, in 2 trees
    MAX_NODES = 1_000_000

    def __init__(
        self,
        engine_dir: Path,
        game: HexGame,
        level: int,
        analyse: bool = False,
        seed: int = 1,
    ):
        self.game = game
        self.analyse = analyse
        self.analysis: list[Analysis] | None = None
        # it writes no log file with --use-logfile 0
        self.process = subprocess.Popen(
            [
                str(self.binary(engine_dir)),
                "--quiet",
                "--use-logfile",
                "0",
                "--seed",
                str(seed),
                "--boardsize",
                str(game.size),
            ],
            cwd=engine_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.command(f"boardsize {game.size} {game.size}")
        for setting in (
            f"max_games {level}",
            # the playouts alone limit a search, so that a busy machine doesn't weaken it
            "max_time 1000000",
            "use_time_management 0",
            "num_threads 1",
            f"max_nodes {self.MAX_NODES}",
        ):
            self.command(f"param_mohex {setting}")
        # the solvers' tables, which take most of its memory; it doesn't use them to play
        self.command("param_dfpn tt_size 1024")
        self.command("param_dfs tt_size 1024")
        self.command("param_game allow_swap 1")

    @classmethod
    def default_dir(cls) -> Path:
        return CACHE_DIR / "mohex" / "benzene"

    @staticmethod
    def binary(engine_dir: Path) -> Path:
        return engine_dir / "build" / "src" / "mohex" / "mohex"

    @classmethod
    def installed(cls, engine_dir: Path) -> bool:
        return cls.binary(engine_dir).exists()

    def command(self, command: str) -> str:
        """Sends a command and returns the response (without its leading "= ")."""
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        # a response is one or more lines, the first starting "=" (or "?" for an error), then
        # a blank line
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

    def swapped(self, stone: int) -> None:
        """Sets the board up as pgx's is after a swap of the first stone, at `stone`."""
        self.command("clear_board")
        self.command("param_game allow_swap 0")
        reflected = self.game.cell_name(self.game.reflected(stone))
        self.command(f"play w {reflected}")

    def play(self, color: str, move: int, stone: int | None = None) -> None:
        """Tells the engine about a move by `color` ("b" or "w"); `stone` is the first stone,
        if the move is a swap."""
        if move == self.game.swap:
            assert stone is not None
            self.swapped(stone)
        else:
            self.command(f"play {color} {self.game.cell_name(move)}")

    def genmove(self, color: str, stone: int | None = None) -> int:
        """The engine's move for `color`, which it also plays; `stone` is the first stone, if
        it's the second move."""
        reply = self.command(f"reg_genmove {color}").strip().lower()
        if reply == "swap-pieces":
            assert stone is not None
            self.analysis = None
            self.swapped(stone)
            return self.game.swap
        if self.analyse:
            self.analysis = self.read_analysis(reply)
        move = self.game.action(reply)
        self.command(f"play {color} {reply}")
        return move

    def read_analysis(self, move: str) -> list[Analysis] | None:
        """What the search that chose `move` made of the position: nothing if it needed none."""
        try:
            analysis = parse_values(self.command("mohex-values"), self.game)
        except RuntimeError:
            return None
        action = self.game.action(move)
        return (
            analysis
            if analysis and any(a.line[0] == action for a in analysis)
            else None
        )

    def winner(self) -> str:
        """Who has connected their sides on the engine's board: "b", "w", or "" for nobody."""
        reply = self.command("final_score").strip().lower()
        return reply[0] if reply[:1] in ("b", "w") else ""

    def close(self) -> None:
        self.process.kill()
        self.process.wait()


ENGINES: dict[str, type[MoHex]] = {MoHex.name: MoHex}
