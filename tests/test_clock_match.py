"""clock_match.py's referee and clock, with fake players in place of the engines."""

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from clock_match import (
    GRACE,
    Game,
    pair_summary,
    play_game,
    play_match,
    points,
    summarize,
)
from engines import Board, ClockedEdax, ClockedEgaroucid, Edax, Egaroucid, flips
from game import action
from ggs import Clock, Start, disc_result


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

    def set_position(self, black, white, color):
        self.board = Board.at(black, white, color)

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
    game = play_game(black, white, moves("f5d6"), 100)
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
    game = play_game(FirstLegal(clock, 1), FirstLegal(clock, 3), (), 4)
    assert game.flagged == "w"
    assert 64 - game.move_seconds[-1][0] >= 50


def test_passing_the_clock_by_less_than_the_grace_is_no_loss(clock):
    # a full game takes white some whole number of its one-second moves
    moves_white = len(play_with(clock, seconds=1000)[1].seconds_left)
    assert play_with(clock, seconds=moves_white - GRACE / 2)[0].flagged is None
    assert play_with(clock, seconds=moves_white - 2 * GRACE)[0].flagged == "w"


@pytest.mark.parametrize(
    "result, a_black, expected",
    [(10, True, 1), (10, False, 0), (0, True, 0.5), (-2, True, 0), (-2, False, 1)],
)
def test_points_are_for_player_a(result, a_black, expected):
    assert points(result, a_black) == expected


def test_summary_is_for_player_a_whichever_color():
    games = [
        Game(margin=4, seconds=[1, 3], move_seconds=[(40, True, 1), (39, False, 3)]),
        Game(
            margin=4,
            result=-2,
            seconds=[2, 4],
            move_seconds=[(20, True, 2), (19, False, 4)],
            flagged="b",
        ),
    ]
    games[0].result = 4
    summary = summarize(games, [True, False], 5)
    assert summary["score"] == 1
    assert summary["board_score"] == 0.5
    assert summary["flags"] == {"a": 0, "b": 1}
    assert summary["clock_changed"] == 1
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
    engine.level = None
    engine.expected = {"b": 3.0, "w": 3.0}
    engine.play("b", action("c4"))
    assert engine.move("w", 3.0) == action("f5")
    assert script.sent == ["play c4", "go"]


def test_a_summary_counts_results_and_writes_as_json():
    games = [Game(margin=m, result=m) for m in (4, -2, 0, 6)]
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


# a 14-disc start of GGS's archive, black to move
GGS_START = Start.parse(
    "8 -------- -------- --*O-*-- ---***-- --**OO-- --O**O-- -------- -------- *"
)


def test_a_game_from_a_start_tells_both_players_its_board(clock):
    black, white = FirstLegal(clock, 0), FirstLegal(clock, 0)
    game = play_game(black, white, (), 100, GGS_START)
    assert game.start == GGS_START.text()
    board = GGS_START.board()
    for move in game.moves:
        board.play(move)
    assert not board.legal()
    assert black.board.discs == white.board.discs == board.discs
    assert len(game.moves) == (board.discs[0] | board.discs[1]).bit_count() - 14


def test_a_soft_timeout_on_a_ggs_clock_is_the_least_loss_and_the_game_goes_on(clock):
    # black takes a second a move, past its 3 s, but well within its extension
    game = play_game(FirstLegal(clock, 1), FirstLegal(clock, 0), (), Clock(3, 0, 100))
    assert game.flagged == "b" and game.hard is None
    board = Board()
    for move in game.moves:
        board.play(move)
    assert not board.legal()
    assert game.result == min(disc_result(board), -2)


def test_a_hard_timeout_ends_the_game_as_the_greatest_loss(clock):
    game = play_game(FirstLegal(clock, 0), FirstLegal(clock, 1), (), Clock(3, 0, 2))
    assert (game.flagged, game.hard) == ("w", "w")
    assert game.result == 64
    # white's 3 s and 2 s of extension last 5 of its moves; it ran out on its 6th
    assert sum(1 for _, black, _ in game.move_seconds if not black) == 6


def test_players_are_told_their_extension_after_a_soft_timeout(clock):
    black = FirstLegal(clock, 2)
    play_game(black, FirstLegal(clock, 0), (), Clock(3, 0, 10))
    assert black.seconds_left[:4] == [3, 1, 9, 7]


def test_pairs_are_scored_by_points_and_by_summed_discs():
    # a pair a wins 1-1 on points but by discs; one it wins 2-0; one it loses 0-2; one even
    results = [(10, -2), (4, 6), (-2, -8), (0, 0)]
    games, a_black = [], []
    for first, second in results:
        # a is black in a pair's first game, white in its second (results are black's)
        games += [Game(result=first), Game(result=-second)]
        a_black += [True, False]
    pairs = pair_summary(games, a_black)
    assert (pairs["split"], pairs["won"], pairs["lost"]) == (2, 1, 1)
    assert (pairs["ggs_won"], pairs["ggs_drawn"], pairs["ggs_lost"]) == (2, 1, 1)
    assert pairs["disc_sums"] == [8, 10, -10, 0]
    assert pairs["disc_sum"] == 2
    assert pairs["disc_sum_ci95"][0] < 2 < pairs["disc_sum_ci95"][1]
    assert pairs["decisive_elo"] == 0
    assert pairs["game_elo"] == 0


def test_a_match_plays_each_start_with_both_colors_also_in_parallel():
    def player():
        return FirstLegal([0.0], 0)

    openings = [((), GGS_START), (moves("f5d6"), None)]
    for parallel in (None, (player, player)):
        games, a_black = play_match(player, player, openings, 100, parallel=parallel)
        assert a_black == [True, False, True, False]
        assert [g.start for g in games] == [GGS_START.text()] * 2 + [None] * 2
        # first-legal players play the same game whatever the colors
        assert games[0].moves == games[1].moves


@pytest.mark.skipif(not Edax.installed(Edax.default_dir()), reason="Edax isn't built")
def test_edax_takes_a_board_and_plays_from_it():
    for start in (GGS_START, Start(GGS_START.black, GGS_START.white, "w").playable()):
        engine = ClockedEdax(Edax.default_dir(), 10, 1)
        try:
            engine.set_position(start.black, start.white, start.color)
            assert engine.engine.position() == (start.black, start.white, start.color)
            assert engine.move(start.color, 1.0) in start.board().legal()
        finally:
            engine.close()


@pytest.mark.skipif(
    not Egaroucid.installed(Egaroucid.default_dir()), reason="Egaroucid isn't built"
)
def test_egaroucid_takes_a_board_and_plays_from_it():
    start = GGS_START
    engine = ClockedEgaroucid(Egaroucid.default_dir(), 10, 1, show_board=True)
    try:
        engine.set_position(start.black, start.white, start.color)
        assert engine.position() == (start.black, start.white, start.color)
        assert engine.move(start.color, 10.0) in start.board().legal()
    finally:
        engine.close()


def test_egaroucid_is_told_the_clock_only_when_time_is_added():
    engine: Any = ClockedEgaroucid.__new__(ClockedEgaroucid)
    script = Script(["> f5\n", "> e6\n", "> c4\n"])
    engine.process = script
    engine.level = None
    engine.expected = {"b": 60.0, "w": 60.0}
    engine.set_position(GGS_START.black, GGS_START.white, "b")
    engine.move("b", 59.5)
    engine.move("b", 125.4)
    assert script.sent[0] == f"setboard {GGS_START.text().replace('*', 'X')[:-1]}X"
    assert script.sent[1:] == ["go", "settime B 125", "go"]


def test_a_synchro_pair_moves_in_lockstep_both_games_movers_thinking_at_once():
    import threading

    from clock_match import GameRun, play_synchro

    meet = threading.Barrier(2, timeout=10)
    log = []

    class Meeting(FirstLegal):
        def move(self, color, seconds_left):
            if not self.seconds_left:
                # each game's first mover waits here for the other's: they must think at once
                meet.wait()
            log.append(self)
            return super().move(color, seconds_left)

    players = [Meeting([0.0], 0) for _ in range(4)]
    runs = [
        GameRun(players[0], players[1], (), 100, GGS_START),
        GameRun(players[3], players[2], (), 100, GGS_START),
    ]
    games = play_synchro(runs)
    assert games[0].moves == games[1].moves
    assert {id(p) for p in log[:2]} == {id(players[0]), id(players[3])}


def test_a_pass_is_a_step_of_its_own_and_asks_no_player():
    from clock_match import GameRun

    # black's d1 takes c1, leaving white's a2 with no move: white passes, then black takes a3
    start = Start(
        1 << action("a1") | 1 << action("b1"),
        1 << action("c1") | 1 << action("a2"),
        "b",
    )
    black, white = FirstLegal([0.0], 0), FirstLegal([0.0], 0)
    run = GameRun(black, white, (), 100, start)
    run.step()
    assert run.game.moves == [action("d1")] and run.passing
    run.step()
    assert not run.passing and run.game.moves == [action("d1")]
    assert not white.seconds_left
    run.step()
    assert run.game.moves == [action("d1"), action("a3")]


def test_a_single_pair_has_no_intervals_and_writes_as_strict_json():
    pairs = pair_summary([Game(result=4), Game(result=4)], [True, False])
    assert pairs["disc_sum_ci95"] is None and pairs["game_elo_ci95"] is None
    assert pairs["decisive_elo"] is None and pairs["decisive_elo_ci95"] is None
    json.dumps(pairs, allow_nan=False)
