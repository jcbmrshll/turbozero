"""clock_match.py's referee and clock, with fake players in place of the engines."""

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from clock_match import GRACE, Board, Game, flips, play_game, points, summarize
from engines import ClockedEdax, ClockedEgaroucid, Edax, Egaroucid
from game import action


def moves(line):
    return [action(line[i : i + 2]) for i in range(0, len(line), 2)]


class FirstLegal:
    """Plays the first legal move, taking `seconds` of a fake clock, and remembers what it
    was told."""

    def __init__(self, clock, seconds):
        self.clock = clock
        self.seconds = seconds
        self.board = Board()
        self.told = []
        self.seconds_left = []

    def play(self, color, move):
        self.told.append((color, move))
        self.board.play(move)

    def move(self, color, seconds_left):
        assert color == self.board.color
        self.seconds_left.append(seconds_left)
        self.clock[0] += self.seconds
        move = self.board.legal()[0]
        self.board.play(move)
        return move

    def close(self):
        pass


def play_with(clock, seconds):
    white = FirstLegal(clock, 1)
    return play_game(FirstLegal(clock, 0), white, (), seconds), white


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("clock_match.time.perf_counter", lambda: now[0])
    return now


def test_the_board_starts_with_the_standard_four_moves():
    assert sorted(Board().legal()) == sorted(moves("d3c4f5e6"))


def test_the_board_turns_discs_along_every_line():
    board = Board()
    board.play(action("f5"))
    assert board.color == "w"
    assert board.margin() == 3
    assert sorted(board.legal()) == sorted(moves("f4f6d6"))
    with pytest.raises(ValueError):
        board.play(action("a1"))


def test_the_board_does_not_wrap_around_edges():
    # h4 and a5 are neighbouring bits, but not neighbouring squares
    assert (
        flips(1 << action("c5"), 1 << action("a5") | 1 << action("b5"), action("h4"))
        == 0
    )


def test_the_board_passes_for_a_player_with_no_move():
    board = Board()
    # with b1 gone, white has no disc left to move from, and black has c3
    board.discs = [
        1 << action("a1") | 1 << action("a3"),
        1 << action("b1") | 1 << action("b3"),
    ]
    board.play(action("c1"))
    assert board.color == "b"
    assert board.legal() == [action("c3")]


def test_each_side_is_charged_only_its_own_moves_and_told_its_clock(clock):
    black, white = FirstLegal(clock, 0.5), FirstLegal(clock, 0.25)
    game = play_game(black, white, moves("f5d6"), seconds=100)
    assert game.flagged is None
    assert game.seconds == [
        0.5 * len(black.seconds_left),
        0.25 * len(white.seconds_left),
    ]
    assert black.seconds_left[:2] == [100, 99.5]
    assert white.seconds_left[:2] == [100, 99.75]
    # both hear the opening and each other's moves, but not their own
    assert (
        black.told[:2] == white.told[:2] == [("b", action("f5")), ("w", action("d6"))]
    )
    assert black.board.discs == white.board.discs


def test_who_runs_out_of_time_first_is_flagged_and_the_game_is_played_out(clock):
    game = play_game(FirstLegal(clock, 1), FirstLegal(clock, 3), (), seconds=4)
    assert game.flagged == "w"
    assert 64 - game.move_seconds[-1][0] >= 50


def test_passing_the_clock_by_less_than_the_grace_is_no_loss(clock):
    # a full game takes white some whole number of its one-second moves
    moves_white = len(play_with(clock, seconds=1000)[1].seconds_left)
    assert play_with(clock, seconds=moves_white - GRACE / 2)[0].flagged is None
    assert play_with(clock, seconds=moves_white - 2 * GRACE)[0].flagged == "w"


@pytest.mark.parametrize(
    "margin, flagged, a_black, expected",
    [
        (10, None, True, 1),
        (10, None, False, 0),
        (0, None, True, 0.5),
        (10, "b", True, 0),
        (10, "b", False, 1),
        (-10, "w", True, 1),
    ],
)
def test_points_are_for_player_a_and_a_flag_beats_the_board(
    margin, flagged, a_black, expected
):
    assert points(margin, flagged, a_black) == expected


def test_summary_is_for_player_a_whichever_color():
    games = [
        Game(margin=4, seconds=[1, 3], move_seconds=[(40, True, 1), (39, False, 3)]),
        Game(
            margin=4,
            seconds=[2, 4],
            move_seconds=[(20, True, 2), (19, False, 4)],
            flagged="b",
        ),
    ]
    summary = summarize(games, [True, False], seconds=5)
    assert summary["score"] == 1
    assert summary["board_score"] == 0.5
    assert summary["flags"] == {"a": 0, "b": 1}
    assert summary["a"]["seconds_a_game"] == 2.5
    assert summary["b"]["seconds_a_game"] == 2.5
    assert summary["a"]["phases"] == {"mid_34+": 0.5, "late_24-33": 0, "end_0-23": 2}


def test_threads_reach_each_engine_only_once():
    for engine in (Edax, Egaroucid):
        fake: Any = engine.__new__(engine)
        fake.threads = 8
        fake.search_log = None
        fake.analyse = False
        command = fake.command_line(Path("/x"), 12)
        flag = "-n" if engine is Edax else "-threads"
        assert command.count(flag) == 1
        assert command[command.index(flag) + 1] == "8"


def test_edax_analysis_is_read_from_the_end_of_each_step_of_its_search():
    engine: Any = Edax.__new__(Edax)
    engine.search_log = io.StringIO(
        "current move:  --  [-64, +64], F6 => -01; pv = F6 e6:05@100%[+01,+01]; F4:04@100%\n"
        "current move:  --  [-64, +64], D6 => +00; pv = D6 c3:05@100%[+00,+00];\n"
        "id = 1 ; level =  6@100% ; ab = [-64, +64]:\n"
        "current move:  --  [-64, +64], D6 => +02; pv = D6 c3:06@100%[-02,-02];\n"
        "id = 1 ; level =  8@100% ; ab = [-64, +64]:\n"
    )
    analysis = engine.read_analysis(action("d6"))
    assert [(a.line[0], a.score, a.depth) for a in analysis] == [
        (action("d6"), 2, 8),
        (action("f6"), -1, 6),
    ]


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
        return self.replies.pop(0)

    def kill(self):
        pass

    def wait(self):
        pass


def test_clocked_edax_says_what_is_left_of_the_clock_before_each_move():
    engine = ClockedEdax.__new__(ClockedEdax)
    inner: Any = Edax.__new__(Edax)
    inner.analyse = False
    script = Script(["= \n", "\n", "= f5\n", "\n"])
    inner.process = script
    engine.engine = inner
    assert engine.move("b", 1.234) == action("f5")
    assert script.sent == ["time_left b 1.234 0", "genmove b"]


def test_clocked_egaroucid_reads_its_move_past_prompts_and_noise():
    engine: Any = ClockedEgaroucid.__new__(ClockedEgaroucid)
    script = Script(["> > \n", "> f5\n"])
    engine.process = script
    engine.play("b", action("c4"))
    assert engine.move("w", 3.0) == action("f5")
    assert script.sent == ["play c4", "go"]


def test_a_summary_counts_results_and_writes_as_json():
    games = [Game(margin=4), Game(margin=-2), Game(margin=0), Game(margin=6)]
    for game in games:
        game.move_seconds.append((40, True, 0.1))
        game.move_seconds.append((39, False, 0.1))
    summary = summarize(games, [True, True, False, False], 10)
    assert (summary["board_wins"], summary["board_draws"]) == (1, 1)
    assert summary["board_losses"] == 2
    assert summary["margins"] == [4, -2, 0, -6]
    json.dumps(summary)


def test_a_game_records_its_moves_from_the_opening(clock):
    game, _ = play_with(clock, 1000)
    board = Board()
    for move in game.moves:
        board.play(move)
    assert not board.legal()
    assert board.margin() == game.margin
    assert len(game.moves) == (board.discs[0] | board.discs[1]).bit_count() - 4
