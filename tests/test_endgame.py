"""endgame.py's solver: its rules against clock_match's referee, hand positions, plain minimax, and
Edax's own solver where Edax is built; and a network's player solving its endgame."""

import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")

from clock_match import Board, flips
from xot import XOT_PATH, load_xot


def square(name: str) -> int:
    return "abcdefgh".index(name[0]) + 8 * (int(name[1]) - 1)


def board(*rows: str) -> tuple[int, int]:
    """(me, opp) from rows 8 down to 1, as printed: X mine, O the opponent's, . empty."""
    me = opp = 0
    for r, row in enumerate(rows):
        for c, cell in enumerate(row.replace(" ", "")):
            bit = 1 << (8 * (7 - r) + c)
            if cell == "X":
                me |= bit
            elif cell == "O":
                opp |= bit
    return me, opp


def random_positions(
    empties: list[int], count: int, seed: int
) -> list[tuple[int, int]]:
    """Positions of the side to move with a move to make, at each of `empties` in turn, from XOT
    openings played on at random."""
    pool = load_xot(XOT_PATH)
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        target = empties[len(out) % len(empties)]
        b = Board()
        for m in pool[rng.randrange(len(pool))]:
            b.play(int(m))
        while b.legal() and 64 - (b.discs[0] | b.discs[1]).bit_count() > target:
            b.play(rng.choice(b.legal()))
        if b.legal() and 64 - (b.discs[0] | b.discs[1]).bit_count() == target:
            side = not b.black_to_move
            out.append((b.discs[side], b.discs[not side]))
    return out


def solve_all(positions, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    from endgame import solve_batch

    me = np.array([p[0] for p in positions], np.uint64)
    opp = np.array([p[1] for p in positions], np.uint64)
    return solve_batch(me, opp, **kwargs)


def test_rules_match_the_referee():
    from endgame import LIB

    rng = random.Random(0)
    for _ in range(40):
        b = Board()
        while b.legal():
            side = not b.black_to_move
            me, opp = b.discs[side], b.discs[not side]
            legal = sum(1 << m for m in b.legal())
            assert LIB.endgame_moves(me, opp) == legal
            assert LIB.endgame_moves_scalar(me, opp) == legal
            for m in range(64):
                if not (me | opp) >> m & 1:
                    expected = flips(me, opp, m)
                    assert LIB.endgame_flips(m, me, opp) == expected
                    assert LIB.endgame_flips_scalar(m, me, opp) == expected
            b.play(rng.choice(b.legal()))


def test_stable_discs_are_those_no_move_can_turn():
    from endgame import LIB

    me, opp = board(
        "X X O . . . . .",
        "X . . . . . . .",
        ". . . X . . . .",
        ". . . . . . . .",
        ". . . . . . . .",
        ". . . . . . . .",
        ". . . . . . . .",
        "X X X X X X X X",
    )
    stable = LIB.endgame_stable(me, opp)
    # the corners, the edge run from a8, and the full first row; not the lone disc in the middle
    expected = {"a8", "b8", "a7"} | {f"{c}1" for c in "abcdefgh"}
    assert stable == sum(1 << square(s) for s in expected)
    # a full board's discs are all stable
    full = (1 << 64) - 1
    assert (
        LIB.endgame_stable(0x5555_5555_5555_5555, full ^ 0x5555_5555_5555_5555)
        == 0x5555_5555_5555_5555
    )


def test_finished_games_score_their_discs_empties_to_the_winner():
    from endgame import NO_MOVE, Solver

    solver = Solver()
    full = (1 << 64) - 1
    me = (1 << 40) - 1
    solved = solver.solve(me, full ^ me)
    assert solved is not None and solved.move == NO_MOVE and solved.margin == 40 - 24
    # neither side has a move with squares empty: the winner takes them
    me, opp = board(
        "X X X . . . . .",
        "X X X . . . . .",
        ". . . . . . . .",
        ". . . . . . . .",
        ". . . . . . . .",
        ". . . . . . . .",
        ". . . . . . . .",
        ". . . . . . . .",
    )
    for exact in (True, False):
        solved = solver.solve(me, opp, exact=exact)
        assert solved is not None and solved.move == NO_MOVE and solved.margin == 64
        solved = solver.solve(opp, me, exact=exact)
        assert solved is not None and solved.margin == -64


def test_the_last_empty_square():
    from endgame import PASS, Solver

    solver = Solver()
    full = (1 << 64) - 1
    a1, b1 = 1 << square("a1"), 1 << square("b1")
    # the mover takes it, turning b1
    solved = solver.solve(full ^ a1 ^ b1, b1)
    assert solved is not None and (solved.move, solved.margin) == (square("a1"), 64)
    # the mover can't, and passes; the opponent takes it, turning b1
    solved = solver.solve(b1, full ^ a1 ^ b1)
    assert solved is not None and (solved.move, solved.margin) == (PASS, -64)
    # neither can: the row, column and diagonal from a1 are all the mover's, and it goes to the
    # opponent, who has more discs
    lines = sum(1 << s for s in range(1, 8)) | sum(1 << (8 * r) for r in range(1, 8))
    me = lines | sum(1 << (9 * i) for i in range(1, 8))
    solved = solver.solve(me, full ^ a1 ^ me)
    assert solved is not None and solved.margin == (21 - 42) - 1


def test_solves_match_plain_minimax():
    from endgame import LIB, move_margins

    positions = random_positions([6, 8, 9, 10], 60, seed=1)
    margins, moves = solve_all(positions)
    wld, wld_moves = solve_all(positions, exact=False)
    for (me, opp), margin, move, sign, sign_move in zip(
        positions, margins, moves, wld, wld_moves
    ):
        expected = LIB.endgame_minimax(me, opp)
        assert margin == expected
        assert np.sign(sign) == np.sign(expected)
        # a move worth the margin, and the margin is the best move's
        after = move_margins(me, opp)
        assert after.max() == expected
        assert after[move] == expected
        if expected >= 0:
            assert np.sign(after[sign_move]) == np.sign(expected)


def test_a_pass_is_solved_through():
    from endgame import LIB, PASS, move_margins

    # positions where the mover must pass and the opponent can move
    rng = random.Random(2)
    found = []
    while len(found) < 5:
        b = Board()
        while True:
            legal = b.legal()
            if not legal:
                break
            before = b.black_to_move
            b.play(rng.choice(legal))
            empties = 64 - (b.discs[0] | b.discs[1]).bit_count()
            if b.black_to_move == before and b.legal() and empties <= 12:
                # the side that just moved moves again: the other, the passer, is to move
                again = int(not b.black_to_move)
                found.append((b.discs[1 - again], b.discs[again]))
                break
    margins, moves = solve_all(found)
    for (me, opp), margin, move in zip(found, margins, moves):
        assert move == PASS
        assert margin == LIB.endgame_minimax(me, opp)
        assert move_margins(me, opp)[PASS] == margin


def test_threads_agree_and_a_solve_gives_up_at_its_deadline():
    from endgame import Solver

    positions = random_positions([14, 16], 8, seed=3)
    margins, _ = solve_all(positions)
    solver = Solver(threads=3)
    for (me, opp), margin in zip(positions, margins):
        solved = solver.solve(me, opp)
        assert solved is not None and solved.margin == margin
    # from the start of a game, nothing is solved in a millisecond
    start = time.perf_counter()
    assert (
        Solver(threads=2).solve(
            *board(
                *([". " * 8] * 3),
                ". . . O X . . .",
                ". . . X O . . .",
                *([". " * 8] * 3),
            ),
            seconds=1e-3,
        )
        is None
    )
    assert time.perf_counter() - start < 0.5
    # nor in one thread
    assert Solver().solve(*random_positions([40], 1, seed=4)[0], seconds=1e-3) is None


def test_a_batch_checks_its_positions():
    from endgame import solve_batch

    with pytest.raises(ValueError):
        solve_batch(np.array([3], np.uint64), np.array([1], np.uint64))
    with pytest.raises(ValueError):
        solve_batch(np.array([3], np.uint64), np.array([4, 8], np.uint64))
    margins, moves = solve_batch(np.zeros(0, np.uint64), np.zeros(0, np.uint64))
    assert len(margins) == len(moves) == 0


def edax_dir() -> Path:
    from engines import Edax

    return Edax.default_dir()


@pytest.mark.skipif(
    not (edax_dir() / "bin" / "edax").exists(), reason="needs Edax built"
)
def test_solves_match_edax():
    positions = random_positions(list(range(10, 21)), 220, seed=5)
    with tempfile.TemporaryDirectory() as tmp:
        problems = Path(tmp) / "positions.obf"
        with open(problems, "w") as f:
            for me, opp in positions:
                # Edax's boards: a1 first, X to move
                cells = "".join(
                    "X" if me >> i & 1 else "O" if opp >> i & 1 else "-"
                    for i in range(64)
                )
                f.write(f"{cells} X;\n")
        out = subprocess.run(
            [str(edax_dir() / "bin" / "edax"), "-solve", str(problems), "-l", "60"]
            + ["-n", str(min(4, len(os.sched_getaffinity(0)))), "-book-usage", "off"],
            cwd=edax_dir(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    edax = [
        int(m) for m in re.findall(r"^\s*\d+\|\s*\d+\S*\s+([+-]\d+)", out, re.MULTILINE)
    ]
    assert len(edax) == len(positions)
    margins, _ = solve_all(positions, threads=min(4, len(os.sched_getaffinity(0))))
    assert margins.tolist() == edax
    wld, _ = solve_all(positions, exact=False)
    assert np.sign(wld).tolist() == np.sign(edax).tolist()


def fake_eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A network of fixed numbers: logits by a hash of the board, the value by the lead in discs."""
    weights = np.sin(np.arange(128) * 1.3).astype(np.float32)
    flat = x.reshape(len(x), 128)
    mix = (flat * weights).sum(1, keepdims=True)
    logits = np.sin(mix * np.arange(1, 66, dtype=np.float32)).astype(np.float32)
    values = (flat[:, :64].sum(1) - flat[:, 64:].sum(1)) / 64
    return logits, values.astype(np.float32)


@pytest.mark.parametrize("exact", [True, False])
def test_a_net_player_plays_its_endgame_solved(exact):
    from clock_match import play_game
    from cpu_match import NetPlayer, Solving
    from cpu_search import SearchSettings

    n = 14
    opening = load_xot(XOT_PATH)[3]
    black = NetPlayer(
        SearchSettings(batch=2),
        fake_eval,
        solving=Solving(empties=n, exact=exact, threads=2),
    )
    white = NetPlayer(SearchSettings(batch=2), fake_eval, seed=1)
    game = play_game(black, white, opening, 1.0)
    assert game.flagged is None
    assert black.solved > 0 and black.gave_up == 0
    # black did no worse than the solution where it first solved: white can only give more away
    b = Board()
    for move in game.moves:
        if b.black_to_move and 64 - (b.discs[0] | b.discs[1]).bit_count() <= n:
            break
        b.play(move)
    (margin,), _ = solve_all([(b.discs[0], b.discs[1])])
    # the game's end scored as the solver scores it, empties to the winner
    end = Board()
    for move in game.moves:
        end.play(move)
    empties = 64 - (end.discs[0] | end.discs[1]).bit_count()
    final = game.margin + int(np.sign(game.margin)) * empties
    if exact:
        assert final >= margin
    else:
        assert np.sign(final) >= np.sign(margin)


def test_a_net_player_searches_when_a_solve_gives_up():
    from cpu_match import NetPlayer, Solving
    from cpu_search import SearchSettings

    player = NetPlayer(
        SearchSettings(), fake_eval, solving=Solving(empties=60, share=1e-4)
    )
    move = player.move("b", 0.2)
    assert player.gave_up == 1 and player.solved == 0
    assert move in Board().legal()
