"""teacher_dataset.py: games put back together from logged moves, labelled exactly, and read back
as distill.py trains on them."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from example_modules import load_example

from core.memory.move_log import ENDED, GOES_ON, META_FILE, STOPPED, moves_path

teacher_dataset, exact_labels, distill = load_example(
    "othello", "teacher_dataset", "exact_labels", "distill"
)


def board(empties: int, mine: int = 10) -> np.ndarray:
    """An (8, 8, 2) observation: the player to move has the first `mine` squares, the opponent
    the rest up to `empties` empty ones."""
    obs = np.zeros((64, 2), bool)
    obs[:mine, 0] = True
    obs[mine : 64 - empties, 1] = True
    return obs.reshape(8, 8, 2)


def write_moves(directory, epochs):
    """Writes a --log-moves directory: `epochs` maps an epoch to its rows, each (env, game, ply,
    empties, player, end, result)."""
    directory.mkdir()
    (directory / META_FILE).write_text(
        json.dumps({"num_envs": 2, "packed": {"observation": [8, 8, 2], "legal": [65]}})
    )
    for epoch, moves in epochs.items():
        rows = np.zeros(
            len(moves),
            [
                ("observation", np.uint8, (16,)),
                ("policy", np.float16, (65,)),
                ("legal", np.uint8, (9,)),
                ("value", np.float32),
                ("action", np.int32),
                ("player", np.int32),
                ("full", bool),
                ("ply", np.int32),
                ("game", np.int32),
                ("end", np.int8),
                ("result", np.float32),
                ("env", np.int32),
                ("epoch", np.int32),
            ],
        )
        for i, (env, game, ply, empties, player, end, result) in enumerate(moves):
            obs = board(empties).reshape(-1)
            rows[i]["observation"] = np.packbits(obs, bitorder="little")
            rows[i]["policy"][0] = 1.0
            rows[i]["legal"] = np.packbits(np.eye(65, dtype=bool)[0], bitorder="little")
            rows[i]["value"] = 0.5
            rows[i]["full"] = ply % 2 == 0
            rows[i]["player"], rows[i]["end"], rows[i]["result"] = player, end, result
            rows[i]["env"], rows[i]["game"], rows[i]["ply"] = env, game, ply
            rows[i]["action"] = -1 if end == STOPPED else 0
        rows["epoch"] = epoch
        np.save(moves_path(directory, epoch), rows)


@pytest.fixture
def moves(tmp_path):
    """Env 0: a game that reaches its solve point (20 empties) and stops one move past it, and an
    unfinished one; env 1: a game across two epochs that ends before any solve, its last mover
    (player 1) losing."""
    directory = tmp_path / "moves"
    write_moves(
        directory,
        {
            0: [
                (0, 0, 8, 30, 0, GOES_ON, 0),
                (0, 0, 9, 21, 1, GOES_ON, 0),
                (1, 0, 8, 40, 0, GOES_ON, 0),
            ],
            1: [
                (0, 0, 10, 20, 0, GOES_ON, 0),
                (0, 0, 11, 19, 1, STOPPED, 0),
                (0, 1, 8, 50, 0, GOES_ON, 0),
                (1, 0, 9, 39, 1, ENDED, -1),
            ],
        },
    )
    return directory


@pytest.fixture
def fake_solver(monkeypatch):
    """Solves a position with 20 empties as a win for its player to move, any other as a loss."""

    def solve(me, opp, threads):
        empties = 64 - np.bitwise_count(me | opp)
        return np.where(empties == 20, 1, -1).astype(np.int8)

    monkeypatch.setattr(teacher_dataset, "solve_wld", solve)


def column(directory, name):
    return np.load(directory / f"{name}.npy")


def test_games_are_put_back_together_and_labelled_exactly(moves, tmp_path, fake_solver):
    out = tmp_path / "dataset"
    meta = teacher_dataset.build([str(moves)], str(out), 20, 0.0, 0, threads=1)
    assert meta["games"] == 2 and meta["dropped_games"] == 1 and meta["positions"] == 6
    np.testing.assert_array_equal(column(out, "game"), [0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(column(out, "ply"), [8, 9, 10, 11, 8, 9])
    np.testing.assert_array_equal(column(out, "empties"), [30, 21, 20, 19, 40, 39])
    # before the solve point, its result (a win for player 0) for each player; from it on, each
    # position's own; the game that ended early, its result (player 1 lost)
    np.testing.assert_array_equal(column(out, "outcome"), [1, -1, 1, -1, 1, -1])
    np.testing.assert_array_equal(
        column(out, "solved"), [False, False, True, True, False, False]
    )
    np.testing.assert_array_equal(column(out, "action"), [0, 0, 0, -1, 0, 0])
    games = np.load(out / "games.npz")
    np.testing.assert_array_equal(games["start"], [0, 4])
    np.testing.assert_array_equal(games["length"], [4, 2])
    np.testing.assert_array_equal(games["env"], [0, 1])


def test_held_out_games_are_whole(moves, tmp_path, fake_solver):
    out = tmp_path / "dataset"
    meta = teacher_dataset.build([str(moves)], str(out), 20, 1.0, 0, threads=1)
    assert meta["heldout_games"] == 2 and column(out, "heldout").all()


def test_a_game_stopped_before_its_solve_point_is_refused(tmp_path, fake_solver):
    directory = tmp_path / "moves"
    write_moves(directory, {0: [(0, 0, 8, 30, 0, STOPPED, 0)]})
    with pytest.raises(ValueError, match="solve point"):
        teacher_dataset.build([str(directory)], str(tmp_path / "d"), 20, 0.0, 0, 1)


def test_rows_load_as_the_positions_self_play_stored(moves, tmp_path, fake_solver):
    out = tmp_path / "dataset"
    teacher_dataset.build([str(moves)], str(out), 20, 0.0, 0, threads=1)
    rows = teacher_dataset.select(out, heldout=False, full_only=True)
    np.testing.assert_array_equal(rows, [0, 2, 4])
    positions = teacher_dataset.load_rows(out, rows)
    np.testing.assert_array_equal(
        positions.observation_nn, [board(30), board(20), board(40)]
    )
    np.testing.assert_array_equal(positions.cur_player_id, [0, 0, 0])
    np.testing.assert_array_equal(positions.reward, [[1, -1], [1, -1], [1, -1]])
    assert positions.policy_mask[:, 0].all() and positions.policy_mask.sum() == 3
    narrowed = teacher_dataset.select(out, False, full_only=False, max_empties=21)
    np.testing.assert_array_equal(narrowed, [1, 2, 3])


def test_observations_undo_bitboards():
    obs = np.random.default_rng(0).random((5, 8, 8, 2)) < 0.3
    obs[..., 1] &= ~obs[..., 0]
    me, opp = exact_labels.bitboards(obs)
    np.testing.assert_array_equal(teacher_dataset.observations(me, opp), obs)


def test_distill_reads_a_datasets_training_and_held_out_games(
    moves, tmp_path, fake_solver
):
    out = tmp_path / "dataset"
    teacher_dataset.build([str(moves)], str(out), 20, 0.0, 0, threads=1)
    args = SimpleNamespace(all_positions=True, min_empties=0, max_empties=64)
    positions, rows = distill.load_data(str(out), False, args)
    assert len(rows) == 6 and positions.observation_nn.shape == (6, 8, 8, 2)
    _, held_rows = distill.load_data(str(out), True, args)
    assert len(held_rows) == 0
