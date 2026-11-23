"""engines.py's Melody, Neural Reversi, ymatioun and Edax-FM: with their pipes scripted, and
with the real engines where they're installed."""

import sys
import threading
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from engines import (
    Board,
    ClockedMelody,
    ClockedNeuralReversi,
    ClockedYmatioun,
    EdaxFM,
    ExactSolver,
    Melody,
    NeuralReversi,
    build_digest,
    default_dir,
    edax_level,
)
from game import action

START = "---------------------------OX------XO--------------------------- X"


class Script:
    """Stands in for a process's pipes."""

    def __init__(self, replies):
        self.sent = []
        self.replies = list(replies)
        self.stdin = self
        self.stdout = self

    def write(self, text):
        self.sent.append(text.strip())

    def flush(self):
        pass

    def readline(self):
        return self.replies.pop(0) if self.replies else ""


def scripted(level, replies):
    engine: Any = Melody.__new__(Melody)
    engine.board = Board()
    engine.level = level
    engine.analyse = True
    engine.process = Script(replies)
    return engine


def test_a_position_is_written_a1_to_h8_then_who_is_to_move():
    engine = scripted(1, [])
    assert engine.position() == START
    engine.play("b", action("f5"))
    text = engine.position()
    assert text[action("f5")] == text[action("e5")] == "X" and text.endswith(" O")


def test_levels_are_edaxs():
    # up to 10: L deep without ProbCut, exact from 2L empties
    assert edax_level(4, 40) == (4, 100) and edax_level(4, 8) == (8, 100)
    # above: L deep at 73% in the middle game, then its table's endgame searches
    assert edax_level(16, 40) == (16, 73)
    assert edax_level(16, 26) == (26, 87)
    assert edax_level(16, 23) == (23, 98)
    assert edax_level(16, 21) == (21, 100)
    assert edax_level(40, 34) == (34, 100) and edax_level(40, 49) == (49, 73)
    assert edax_level(40, 50) == (40, 73)
    assert edax_level(60, 50) == (50, 100)


def test_a_level_search_is_sent_with_the_position_and_its_move_played():
    engine = scripted(4, ["noise\n", "move F5 score 2\n"])
    assert engine.genmove("b") == action("f5")
    assert engine.process.sent == [f"IOStd search-fixed-depth {START} 4 100"]
    assert engine.analysis[0].line == [action("f5")] and engine.analysis[0].score == 2
    # it plays its own move on the board it keeps
    assert engine.board.color == "w"


def test_an_error_or_exit_is_raised():
    engine = scripted(4, ["SYNTAX ERROR :\n"])
    with pytest.raises(RuntimeError, match="rejected"):
        engine.genmove("b")
    with pytest.raises(RuntimeError, match="exited"):
        scripted(4, []).genmove("b")


def test_the_clock_is_sent_in_milliseconds_with_the_position_after_the_moves_told():
    clocked: Any = ClockedMelody.__new__(ClockedMelody)
    clocked.engine = scripted(60, ["move F6 score 0\n"])
    clocked.play("b", action("f5"))
    assert clocked.move("w", 1.2345) == action("f6")
    sent = clocked.engine.process.sent[0].split()
    assert sent[:2] == ["IOStd", "search-limited-time"]
    assert sent[3:] == ["O", "1234"]


def test_a_position_set_is_the_one_searched():
    board = Board()
    for name in ("f5", "f6", "e6", "f4"):
        board.play(action(name))
    engine = scripted(1, ["move E3 score 0\n"])
    engine.set_position(board.discs[0], board.discs[1], board.color)
    position = engine.position()
    engine.genmove("b")
    assert engine.process.sent[0].split()[2:4] == position.split()
    assert position.endswith(" X") and position[action("f4")] == "O"
    assert position[:64].count("X") == board.discs[0].bit_count()


@pytest.mark.skipif(
    not Melody.installed(Melody.default_dir()), reason="Melody isn't installed"
)
def test_the_real_engine_moves_from_a_position_set():
    board = Board()
    for name in ("f5", "d6", "c3", "d3", "c4"):
        board.play(action(name))
    engine = Melody(Melody.default_dir(), level=2)
    try:
        engine.set_position(board.discs[0], board.discs[1], board.color)
        assert engine.genmove(board.color) in board.legal()
    finally:
        engine.close()


def opened(moves):
    board = Board()
    for name in moves:
        board.play(action(name))
    return board


def test_neural_reversi_takes_a_board_with_setboard_and_whole_seconds_left():
    clocked: Any = ClockedNeuralReversi.__new__(ClockedNeuralReversi)
    engine: Any = NeuralReversi.__new__(NeuralReversi)
    engine.analyse = False
    engine.process = Script(["= \n", "\n", "= \n", "\n", "= f6\n", "\n"])
    clocked.engine = engine
    board = opened(["f5"])
    clocked.set_position(board.discs[0], board.discs[1], board.color)
    assert clocked.move("w", 7.9) == action("f6")
    setboard, time_left, genmove = engine.process.sent
    assert setboard.split()[0] == "setboard" and setboard.split()[2] == "W"
    assert setboard.split()[1][action("f5")] == "X"
    assert (time_left, genmove) == ("time_left w 7 0", "genmove w")


def test_neural_reversi_says_what_its_search_made_of_its_move():
    clocked: Any = ClockedNeuralReversi.__new__(ClockedNeuralReversi)
    engine: Any = NeuralReversi.__new__(NeuralReversi)
    engine.analyse = True
    search = "search score -2.37 depth 18 probability 86 endgame false nodes 123456\n"
    random = "search score 0.00 depth 0 probability 100 endgame false nodes 0\n"
    engine.process = Script(
        ["= \n", "\n", search, "= f5\n", "\n", "= \n", "\n", random, "= d3\n", "\n"]
    )
    clocked.engine = engine
    assert clocked.move("b", 9.0) == action("f5")
    assert clocked.info() == {
        "score": -2.37,
        "depth": 18,
        "probability": 86,
        "endgame": False,
        "nodes": 123456,
    }
    # its first move from the start is a random one, not searched
    assert clocked.move("w", 9.0) == action("d3")
    assert clocked.info() is None


def test_neural_reversi_is_its_upstream_build_whatever_patch_logs_its_searches(
    tmp_path,
):
    release = tmp_path / "target" / "release"
    release.mkdir(parents=True)
    (release / "cli").write_bytes(b"patched")
    (release / "upstream.sha256").write_text("ab" * 32 + "\n")
    first = build_digest(NeuralReversi.name, tmp_path)
    (release / "cli").write_bytes(b"patched again")
    assert build_digest(NeuralReversi.name, tmp_path) == first
    (release / "upstream.sha256").write_text("cd" * 32 + "\n")
    assert build_digest(NeuralReversi.name, tmp_path) != first


def test_neural_reversi_reads_its_board():
    engine: Any = NeuralReversi.__new__(NeuralReversi)
    rows = ["= ", "   a b c d e f g h", "  +-+-+-+-+-+-+-+-+"]
    for row in range(1, 9):
        cells = {4: "| | | |O|X| | | |", 5: "| | | |X|X|X| | |"}.get(
            row, "| | | | | | | | |"
        )
        rows += [f"{row} {cells} White(O)'s turn" if row == 1 else f"{row} {cells}"]
    engine.process = Script([line + "\n" for line in rows] + ["\n"])
    black, white, color = engine.position()
    board = opened(["f5"])
    assert (black, white, color) == (board.discs[0], board.discs[1], "w")


def test_ymatioun_is_sent_the_position_and_milliseconds_and_reads_its_square():
    clocked: Any = ClockedYmatioun.__new__(ClockedYmatioun)
    clocked.board = Board()
    clocked.process = Script(["info depth 12\n", "bestmove 37 score 0 time 9\n"])
    assert clocked.move("b", 2.5) == action("f5")
    position, go = clocked.process.sent
    assert position == "position fen " + START.replace("X", "*")
    assert go == "go time 2500"
    assert clocked.board.color == "w"


def test_engines_that_play_only_on_a_clock_are_found_too():
    assert default_dir("ymatioun") == ClockedYmatioun.default_dir()
    assert default_dir("edax-fm") == EdaxFM.default_dir() != default_dir("edax")


@pytest.mark.skipif(
    not NeuralReversi.installed(NeuralReversi.default_dir()),
    reason="Neural Reversi isn't installed",
)
def test_the_real_neural_reversi_takes_a_board():
    board = opened(["f5", "d6", "c3", "d3", "c4"])
    engine = NeuralReversi(NeuralReversi.default_dir(), level=2)
    try:
        engine.set_position(board.discs[0], board.discs[1], board.color)
        assert engine.position() == (board.discs[0], board.discs[1], board.color)
        assert engine.genmove(board.color) in board.legal()
    finally:
        engine.close()


@pytest.mark.skipif(
    not (ClockedYmatioun.default_dir() / "bin" / "ymatioun-t1").exists(),
    reason="ymatioun isn't installed",
)
def test_the_real_ymatioun_moves_from_a_position_set():
    board = opened(["f5", "d6", "c3", "d3", "c4"])
    engine = ClockedYmatioun(ClockedYmatioun.default_dir(), 10, threads=1)
    try:
        engine.set_position(board.discs[0], board.discs[1], board.color)
        assert engine.move(board.color, 0.5) in board.legal()
    finally:
        engine.close()


@pytest.mark.skipif(
    not EdaxFM.installed(EdaxFM.default_dir()), reason="Edax-FM isn't installed"
)
def test_the_real_edax_fm_takes_a_board():
    board = opened(["f5", "d6", "c3", "d3", "c4"])
    engine = EdaxFM(EdaxFM.default_dir(), level=2)
    try:
        engine.set_position(board.discs[0], board.discs[1], board.color)
        assert engine.position() == (board.discs[0], board.discs[1], board.color)
        assert engine.genmove(board.color) in board.legal()
    finally:
        engine.close()


# stands in for Neural Reversi's `cli solve FILE`: notes its arguments, then reads FILE a
# position at a time and answers each with a row of its table, the score the mover's discs less
# the opponent's; a position with none of the mover's discs ends it, as a crash would
FAKE_SOLVE = """#!{python}
import sys
from pathlib import Path

Path(sys.argv[0]).with_name("calls").open("a").write(" ".join(sys.argv[1:]) + "\\n")
print("Neural Reversi (fake)", flush=True)
print("|   #   | Depth  | Score |    Time     |", flush=True)
with open(sys.argv[2]) as positions:
    for n, line in enumerate(positions, 1):
        board = line.split()[0]
        if "X" not in board:
            print("no discs", file=sys.stderr)
            sys.exit(1)
        print(f"|  {{n:4d}} |   20   |  {{board.count('X') - board.count('O'):+03d}}  |    0:00.001 |", flush=True)
"""


@pytest.fixture
def fake_solver_dir(tmp_path):
    cli = tmp_path / "target" / "release" / "cli"
    cli.parent.mkdir(parents=True)
    cli.write_text(FAKE_SOLVE.format(python=sys.executable))
    cli.chmod(0o755)
    return tmp_path


def test_one_solver_process_answers_every_solve(fake_solver_dir):
    with ExactSolver(3, fake_solver_dir) as solve:
        assert solve(0b111, 1 << 63) == 2
        assert solve(0b1, 0b1111 << 60) == -3
        assert solve(0b1, 0) == 1
    calls = (fake_solver_dir / "target" / "release" / "calls").read_text().splitlines()
    assert calls == ["solve /dev/stdin --exact --threads 3 --hash-size 1024"]


def test_solves_from_many_threads_each_get_their_own_answer(fake_solver_dir):
    got: dict[int, int] = {}
    with ExactSolver(1, fake_solver_dir) as solve:

        def solving(k: int) -> None:
            got[k] = solve((1 << (k + 1)) - 1, 1 << 63)

        threads = [threading.Thread(target=solving, args=(k,)) for k in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert got == {k: k for k in range(20)}


def test_a_solver_that_dies_is_raised_with_what_it_said(fake_solver_dir):
    with ExactSolver(1, fake_solver_dir) as solve:
        assert solve(0b11, 0) == 2
        with pytest.raises(RuntimeError, match="position 2.*no discs"):
            solve(0, 0b11)
