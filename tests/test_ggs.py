"""ggs.py: GGS's random starts, its archive, its clock and its scoring."""

import bz2
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from engines import Board
from ggs import (
    AVOID,
    Clock,
    PlayerClock,
    Start,
    archive_starts,
    disc_result,
    game_result,
    load_starts,
    random_setup,
    random_setup_2,
    random_start,
)

BOARD = "8 -------- -------- --*O-*-- ---***-- --**OO-- --O**O-- -------- -------- *"


def test_a_start_reads_and_writes_ggf_boards():
    start = Start.parse(BOARD)
    assert start.discs == 14 and start.color == "b"
    assert start.black >> (2 * 8 + 2) & 1  # c3
    assert start.white >> (2 * 8 + 3) & 1  # d3
    assert Start.parse(start.ggf()) == Start.parse(start.text()) == start
    assert start.board().text() == start.text()
    assert Start.standard().board().discs == Board().discs


def test_a_start_whose_mover_has_no_move_hands_the_move_over():
    # black's lone disc on a1, white's on b1 and c1: black can't move, white can't either
    stuck = Start(1, 0b110, "b")
    assert stuck.playable() == stuck
    # white's b1 between black's a1 and c1: black has no move, white has none to flip either,
    # but with a black disc on b2 white can take a3
    start = Start(1 | 1 << 2 | 1 << 9, 1 << 1 | 1 << 17, "b")
    if not start.board().legal():
        assert start.playable().color == "w"


@pytest.mark.parametrize("n", [5, 14, 16, 20])
def test_random_setup_drops_its_discs_near_the_center(n):
    rng = np.random.default_rng(0)
    for _ in range(200):
        start = random_setup(n, rng)
        taken = start.black | start.white
        assert start.discs == n and not start.black & start.white
        assert not any(taken >> s & 1 for s in AVOID)
        # the 4 central squares first
        assert all(taken >> s & 1 for s in (27, 28, 35, 36))
        whites = start.white.bit_count()
        assert abs(whites - n / 2) <= (n // 2 + 1) // 3 + 1
        assert start.color == ("w" if n % 2 else "b")


def test_random_setup_2_plays_legal_moves_to_a_balanced_position():
    rng = np.random.default_rng(1)
    for _ in range(200):
        start = random_setup_2(14, rng)
        assert start.discs == 14
        assert not any((start.black | start.white) >> s & 1 for s in AVOID)
        assert min(start.black.bit_count(), start.white.bit_count()) > 3


def test_random_starts_mix_both_setups_and_repeat_by_seed():
    def draw(seed):
        rng = np.random.default_rng(seed)
        return [random_start(14, rng) for _ in range(400)]

    assert draw(3) == draw(3)
    starts = draw(3)
    # random_setup fills only the 4x4 center (c3-f6); random_setup_2 rarely stays in it
    box = sum(1 << (y * 8 + x) for x in range(2, 6) for y in range(2, 6))
    inside = np.mean([((s.black | s.white) & ~box) == 0 for s in starts])
    assert 0.4 < inside < 0.65


def test_an_archive_gives_its_starts_of_a_type_once(tmp_path):
    game = "(;GM[Othello]DT[{date}]TY[{type}]BO[{board}]B[d2];)"
    other = BOARD.replace("--*O-*--", "--*O----").replace("--O**O--", "--O**OO-")
    lines = [
        "2 " + 2 * game.format(date="2025.01.01", type="s8r14", board=BOARD),
        "2 " + 2 * game.format(date="2025.01.02", type="s8r14", board=BOARD),
        "2 " + 2 * game.format(date="2023.01.01", type="s8r14", board=other),
        "1 " + game.format(date="2025.01.01", type="8", board=other),
    ]
    path = tmp_path / "games.ggf.bz2"
    path.write_bytes(bz2.compress("\n".join(lines).encode()))
    assert archive_starts(path) == [Start.parse(BOARD), Start.parse(other)]
    assert archive_starts(path, since="2024") == [Start.parse(BOARD)]
    assert load_starts(path, "8") == [Start.parse(other)]
    text = tmp_path / "starts.txt"
    text.write_text(Start.parse(BOARD).text() + "\n")
    assert load_starts(text) == [Start.parse(BOARD)]


@pytest.mark.parametrize(
    "text, clock",
    [
        ("01:00//02:00", Clock(60, 0, 120)),
        ("05:00//02:00", Clock(300, 0, 120)),
        ("03:00//00:30", Clock(180, 0, 30)),
        ("60/2/120", Clock(60, 2, 120)),
        ("1:00:00,0//2:00,0", Clock(3600, 0, 120)),
    ],
)
def test_clocks_read_ggs_notation(text, clock):
    assert Clock.parse(text) == clock
    assert Clock.parse(str(clock)) == clock


def test_clocks_with_move_counts_are_refused():
    with pytest.raises(ValueError):
        Clock.parse("30:00,N0//5:00,N10")


def test_a_clock_runs_into_its_extension_then_out():
    clock = PlayerClock(Clock(10, 0, 5))
    clock.charge(9)
    assert clock.seconds_left() == 1 and not clock.soft
    clock.charge(2)
    assert clock.soft and not clock.hard and clock.seconds_left() == 4
    clock.charge(4)
    assert not clock.hard and clock.seconds_left() == 0
    clock.charge(0.1)
    assert clock.hard


def test_a_clock_adds_its_increment_after_each_move_and_forgives_its_grace():
    clock = PlayerClock(Clock(10, 2, 0), grace=0.05)
    clock.charge(5)
    assert clock.seconds_left() == 7
    clock.charge(7.01)
    assert not clock.soft and clock.seconds_left() == pytest.approx(1.99)
    clock.charge(3)
    # no extension: out of time is out of the game
    assert clock.soft and clock.hard


def test_a_contest_clock_is_played_out_with_no_time():
    clock = PlayerClock(Clock.contest(10))
    clock.charge(11)
    assert clock.soft and not clock.hard
    assert clock.seconds_left() == 0
    clock.charge(100)
    assert not clock.hard
    assert math.isinf(Clock.contest(1).ext)


def test_results_give_the_winner_the_empty_squares_and_timeouts_their_losses():
    board = Board.at(0b111, 0b1000, "b")
    assert disc_result(board) == 3 - 1 + 60
    assert game_result(10, "b", None) == -2
    assert game_result(-10, "b", None) == -10
    assert game_result(-10, "w", None) == 2
    assert game_result(0, "w", None) == 2
    assert game_result(10, "b", "b") == -64
    assert game_result(-10, "w", "w") == 64
