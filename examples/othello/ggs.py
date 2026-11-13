"""The conditions of program-vs-program Othello on GGS, Michael Buro's Generic Game Server
(https://skatgame.net/mburo/ggs/), for a match driver or a client to use: its random start
positions, its clock and its scoring. Nothing here plays a game (see `clock_match.py`).

Most of its games are synchro matches, `s8r14`: two games from the same random position of 14
discs, the colors swapped, each on its own clock, played at once; the match's result is the
mean of the two games' disc results.

    uv run examples/othello/ggs.py extract Othello.2010-11-2026.ggf.bz2 starts.txt
    uv run examples/othello/ggs.py compare starts.txt --discs 14 --count 1000

`extract` writes the archive's start positions of one match type, each once; `compare` sets
them beside this module's own random starts (`random_start`): discs, colors, who's to move, and
how lopsided Edax judges them.

Random starts (`Board::init` in the server's `Service/Othello/src/OthelloImpl.C`): with N of 10
or more discs, half of them are `random_setup(N)`, N discs dropped near the center, the corners
and their neighbours avoided, and half `random_setup_2(N)` (`NICOLET_RANDOM`), 5 random discs
then N - 5 random legal moves, again away from the corners, and lopsided results rejected.
The server's random numbers are its C library's; here they're numpy's, so the positions are
drawn the same way but aren't the same ones.

The clock is `ini/inc/ext` (the server's `msg/clock`): `ini` for the whole game, `inc` added
after each move, and, once `ini` has run out (a "soft" timeout), `ext` more to finish the game
in. The game counts as lost by the least margin, 2 discs, at best, for whoever ran out first;
running out of `ext` too (a "hard" timeout) is the greatest loss, 64, and ends the game.
"""

import argparse
import bz2
import json
import math
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from engines import Board, board_string, flips

# the smallest difference of results, in discs (the server's RESULT_INCREMENT), so the least loss
MINIMAL_LOSS = 2
# the greatest loss, every square (the server's max_result)
MAXIMAL_LOSS = 64
# squares the random setups don't fill or move to: the corners and their neighbours
CORNERS = {0, 7, 56, 63}
AVOID = frozenset(
    s
    for s in range(64)
    if any(abs(s % 8 - c % 8) <= 1 and abs(s // 8 - c // 8) <= 1 for c in CORNERS)
)


@dataclass(frozen=True)
class Start:
    """A position a game starts from, given whole: black's and white's discs as bitboards (bit
    0 is a1, bit 7 h1, bit 63 h8), and who's to move, "b" or "w"."""

    black: int
    white: int
    color: str

    @classmethod
    def standard(cls) -> "Start":
        board = Board()
        return cls(board.discs[0], board.discs[1], board.color)

    @classmethod
    def parse(cls, text: str) -> "Start":
        """From `text()`'s 65 characters, or a GGF board, "8 <8 rows of 8> <*|O>", spaces
        anywhere: "*" (or "X", "B") is black, "O" (or "W") white, "-" (or ".") empty."""
        cells = re.sub(r"\s", "", text)
        if len(cells) == 66 and cells[0] == "8":
            cells = cells[1:]
        if len(cells) != 65:
            raise ValueError(f"not a board: {text!r}")
        black = sum(1 << i for i, c in enumerate(cells[:64]) if c in "*XxBb")
        white = sum(1 << i for i, c in enumerate(cells[:64]) if c in "OoWw")
        if cells[64] not in "*XxBbOoWw":
            raise ValueError(f"no player to move: {text!r}")
        return cls(black, white, "b" if cells[64] in "*XxBb" else "w")

    def text(self) -> str:
        """The squares a1, b1, ..., h8 ("*" black, "O" white, "-" empty), then who's to move."""
        return board_string(self.black, self.white) + (
            "*" if self.color == "b" else "O"
        )

    def ggf(self) -> str:
        """The position as a GGF board, `BO[...]`'s value."""
        squares = board_string(self.black, self.white)
        rows = " ".join(squares[i : i + 8] for i in range(0, 64, 8))
        return f"8 {rows} {'*' if self.color == 'b' else 'O'}"

    def board(self) -> Board:
        return Board.at(self.black, self.white, self.color)

    @property
    def discs(self) -> int:
        return (self.black | self.white).bit_count()

    def playable(self) -> "Start":
        """The same position with a player to move who has a move: the other player, if the one
        to move has none (the server has them pass); unchanged if neither has one."""
        board = self.board()
        if board.legal():
            return self
        board.black_to_move = not board.black_to_move
        return self if not board.legal() else Start(self.black, self.white, board.color)


def _shuffle(items: list, rng: np.random.Generator, end: int | None = None) -> None:
    """The server's `shuffle_with_rand` (libstdc++'s old `random_shuffle`) of `items[:end]`."""
    end = len(items) if end is None else end
    for i in range(1, end):
        j = int(rng.integers(i + 1))
        items[i], items[j] = items[j], items[i]


def random_setup(n: int, rng: np.random.Generator) -> Start:
    """The server's `random_setup(n)`: `n` discs on squares drawn ring by ring from the center
    out (the 4 central squares, then the 12 around them, ...), the corners and their neighbours
    avoided; half of them white (one more half the time with `n` odd), give or take a third;
    black to move with `n` even, as after `n - 4` moves."""
    n = max(n, 4)
    rings: list[list[int]] = [[] for _ in range(4)]
    for square in range(64):
        if square not in AVOID:
            x, y = square % 8, square // 8
            rings[(max(abs(2 * x - 7), abs(2 * y - 7)) - 1) // 2].append(square)
    order: list[int] = []
    for ring in rings:
        _shuffle(ring, rng)
        order += ring
    n = min(n, len(order))
    _shuffle(order, rng, n)
    whites = n // 2
    if n % 2 and rng.integers(2):
        whites += 1
    imbalance = whites // 3
    whites += int(rng.integers(2 * imbalance + 1)) - imbalance
    colors = [1] * whites + [0] * (n - whites)
    _shuffle(colors, rng)
    black = sum(1 << s for s, c in zip(order[:n], colors) if c == 0)
    white = sum(1 << s for s, c in zip(order[:n], colors) if c == 1)
    return Start(black, white, "w" if n % 2 else "b")


def random_setup_2(n: int, rng: np.random.Generator, tries: int = 5) -> Start:
    """The server's `random_setup_2(n)`, Stéphane Nicolet's: 5 discs by `random_setup(5)`,
    white to move, then `n - 5` legal moves drawn at random, none on a corner or next to one.
    It starts again if a player has no such move, or if either color ends with no more than
    `max(n // 4, 3)` discs; after `tries` failures it's `random_setup(n)`."""
    least = max(n // 4, 3)
    for _ in range(tries):
        start = random_setup(5, rng)
        discs = [start.black, start.white]
        side = start.color == "w"
        for _ in range(n - 5):
            mine, theirs = discs[side], discs[not side]
            taken = mine | theirs
            moves = [
                m
                for m in range(64)
                if m not in AVOID and not taken >> m & 1 and flips(mine, theirs, m)
            ]
            if not moves:
                break
            move = moves[int(rng.integers(len(moves)))]
            turned = flips(mine, theirs, move)
            discs[side] |= turned | 1 << move
            discs[not side] &= ~turned
            # no passes: a move always hands the turn over
            side = not side
        else:
            if min(discs[0].bit_count(), discs[1].bit_count()) > least:
                return Start(discs[0], discs[1], "w" if side else "b")
    return random_setup(n, rng)


def random_start(n: int, rng: np.random.Generator) -> Start:
    """A random start of `n` discs as the server makes them (its `Board::init`), as it is: the
    player to move may have no move (see `Start.playable`)."""
    if n <= 3:
        return Start.standard()
    if n < 10 or rng.integers(2):
        return random_setup(n, rng)
    return random_setup_2(n, rng)


def archive_games(path: str | Path) -> Iterator[dict[str, str]]:
    """The games of a GGS archive (a .ggf file, bz2-compressed or not): each game's tags, its
    moves left out. A synchro pair's two games come one after the other, a line between them."""
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            for game in re.findall(r"\(;(.*?);\)", line):
                tags = dict(re.findall(r"([A-Z]{2})\[([^\]]*)\]", game))
                yield tags


def archive_starts(
    path: str | Path, match_type: str = "s8r14", since: str = ""
) -> list[Start]:
    """The start positions of an archive's games of `match_type`, each once, in the order first
    played; only games dated `since` (e.g. "2024") or later."""
    seen: dict[Start, None] = {}
    for tags in archive_games(path):
        if tags.get("TY") == match_type and tags.get("DT", "") >= since:
            seen.setdefault(Start.parse(tags["BO"]), None)
    return list(seen)


def load_starts(
    path: str | Path, match_type: str = "s8r14", since: str = ""
) -> list[Start]:
    """Starts from a file: a GGS archive (.ggf, .ggf.bz2), its starts of `match_type`; else
    a start a line, as `Start.parse` reads them."""
    if re.search(r"\.ggf(\.bz2)?$", str(path)):
        return archive_starts(path, match_type, since)
    return [Start.parse(line) for line in Path(path).read_text().splitlines() if line]


@dataclass(frozen=True)
class Clock:
    """A game's clock for each player, in seconds: `ini` for the whole game, `inc` added after
    each move, and once `ini` has run out (a soft timeout, a loss), `ext` more to finish in.
    With `ext` infinite, a contest's clock: the game's played out after a timeout, with no
    time."""

    ini: float
    inc: float = 0.0
    ext: float = 0.0

    @classmethod
    def contest(cls, seconds: float) -> "Clock":
        return cls(seconds, 0.0, math.inf)

    @classmethod
    def parse(cls, text: str) -> "Clock":
        """GGS's `ini/inc/ext`, each `[[HH:]MM:]SS`, e.g. "01:00//02:00" or "60/0/120"; the
        default move counts only (",0" or nothing after a time)."""
        parts = text.split("/")
        if len(parts) != 3:
            raise ValueError(f"a clock is ini/inc/ext: {text!r}")
        seconds = []
        for part in parts:
            time, _, moves = part.partition(",")
            if moves not in ("", "0"):
                raise ValueError(f"only whole-game clocks: {text!r}")
            value = 0.0
            for field in time.split(":") if time else []:
                value = 60 * value + float(field)
            seconds.append(value)
        return cls(*seconds)

    @property
    def is_contest(self) -> bool:
        return math.isinf(self.ext)

    def __str__(self) -> str:
        if self.is_contest:
            return f"{self.ini:g}s"

        def mmss(seconds: float) -> str:
            if not seconds:
                return ""
            whole = (
                f"{seconds % 60:02.0f}"
                if seconds == int(seconds)
                else f"{seconds % 60:06.3f}"
            )
            return f"{int(seconds // 60):02d}:{whole}"

        return f"{mmss(self.ini)}/{mmss(self.inc)}/{mmss(self.ext)}"


class PlayerClock:
    """One player's clock through a game, as the server keeps it (its `GAME_Clock::update`):
    charged after each move, a timeout found then. `grace` is time past zero that doesn't count,
    for the pipes and the operating system's share of a move."""

    def __init__(self, clock: Clock, grace: float = 0.0):
        self.clock = clock
        self.grace = grace
        self.left = clock.ini
        self.used = 0.0
        self.extended = False
        self.soft = False
        self.hard = False

    def charge(self, seconds: float) -> None:
        """A move that took `seconds`."""
        self.used += seconds
        self.left -= seconds
        if self.left < -self.grace:
            if self.extended or self.clock.ext == 0:
                self.hard = self.soft = True
            else:
                self.soft = True
                self.extended = True
                self.left += self.clock.ext
        self.left += self.clock.inc

    def seconds_left(self) -> float:
        """What to tell the player is left: none, on a contest's clock after a timeout."""
        if self.extended and self.clock.is_contest:
            return 0.0
        return max(self.left, 0.0)


def disc_result(board: Board) -> int:
    """Black's result of a finished game: discs less white's, the empty squares the winner's."""
    margin = board.margin()
    empties = 64 - (board.discs[0] | board.discs[1]).bit_count()
    return margin + (empties if margin > 0 else -empties if margin < 0 else 0)


def game_result(board_result: int, soft: str | None, hard: str | None) -> int:
    """Black's result after timeouts (the server's `Game::blacks_result`): the greatest loss for
    a hard timeout; for whoever's soft timeout came first, at best the least loss."""
    if hard is not None:
        return -MAXIMAL_LOSS if hard == "b" else MAXIMAL_LOSS
    if soft == "b":
        return min(board_result, -MINIMAL_LOSS)
    if soft == "w":
        return max(board_result, MINIMAL_LOSS)
    return board_result


def start_stats(starts: Sequence[Start], values: Sequence[int] | None = None) -> dict:
    """What a set of starts is like: discs, colors, who's to move; with `values` (each start's
    value for the player to move, by an engine), how lopsided."""
    leads = np.array([s.black.bit_count() - s.white.bit_count() for s in starts])
    playable = [s.board().legal() != [] for s in starts]
    stats = {
        "starts": len(starts),
        "discs": dict(sorted(Counter(s.discs for s in starts).items())),
        "black_to_move": round(float(np.mean([s.color == "b" for s in starts])), 4),
        "mover_has_no_move": round(1 - float(np.mean(playable)), 4),
        "black_lead_mean": round(float(leads.mean()), 3),
        "black_lead_sd": round(float(leads.std()), 3),
        "black_lead": dict(sorted(Counter(leads.tolist()).items())),
    }
    if values is not None:
        magnitude = np.abs(np.asarray(values))
        stats["value_sd"] = round(float(np.std(values)), 2)
        stats["value_abs_mean"] = round(float(magnitude.mean()), 2)
        for at in (4, 8, 12, 16, 24):
            stats[f"lopsided_{at}+"] = round(float(np.mean(magnitude >= at)), 4)
    return stats


def edax_values(starts: Sequence[Start], level: int) -> list[int]:
    """Each start's value for its player to move (made playable first), by Edax at `level`."""
    from engines import ENGINES, EdaxAnalyst

    analyst = EdaxAnalyst(ENGINES["edax"].default_dir(), level)
    values = []
    try:
        for start in starts:
            start = start.playable()
            if not start.board().legal():
                board = start.board()
                values.append(board.margin() * (1 if start.color == "b" else -1))
                continue
            squares = np.array(
                [
                    1 if start.black >> i & 1 else 2 if start.white >> i & 1 else 0
                    for i in range(64)
                ]
            )
            values.append(analyst.analyse(squares, start.color)[0].score)
    finally:
        analyst.close()
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="GGS random starts: extract, compare.")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="an archive's starts, each once")
    extract.add_argument("archive", type=Path)
    extract.add_argument("out", type=Path)
    extract.add_argument("--type", default="s8r14")
    extract.add_argument("--since", default="", help='e.g. "2024"')
    compare = commands.add_parser("compare", help="archive starts vs random_start's")
    compare.add_argument("starts", type=Path, help="an archive, or extract's file")
    compare.add_argument("--type", default="s8r14")
    compare.add_argument("--since", default="")
    compare.add_argument("--discs", type=int, default=14)
    compare.add_argument("--count", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--edax-level", type=int, default=0, help="0: no values")
    args = parser.parse_args()

    if args.command == "extract":
        starts = archive_starts(args.archive, args.type, args.since)
        args.out.write_text("".join(s.text() + "\n" for s in starts))
        print(f"{len(starts)} starts")
        return
    rng = np.random.default_rng(args.seed)
    archive = load_starts(args.starts, args.type, args.since)
    picked = [archive[i] for i in rng.permutation(len(archive))[: args.count]]
    drawn = {
        "archive": picked,
        "ggs": [random_start(args.discs, rng) for _ in range(args.count)],
        "random_setup": [random_setup(args.discs, rng) for _ in range(args.count)],
        "random_setup_2": [random_setup_2(args.discs, rng) for _ in range(args.count)],
    }
    for name, starts in drawn.items():
        values = edax_values(starts, args.edax_level) if args.edax_level else None
        print(json.dumps({"source": name, **start_stats(starts, values)}), flush=True)


if __name__ == "__main__":
    main()
