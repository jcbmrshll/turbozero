import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

import engine_set
import label_positions
from engine_set import ILLEGAL, EngineSet, selectable
from engines import Board, MoveScores


def board_obs(n_discs: int, rng: np.random.Generator) -> np.ndarray:
    """An (8, 8, 2) observation with `n_discs` discs, at random, of either player."""
    obs = np.zeros((64, 2), bool)
    squares = rng.choice(64, n_discs, replace=False)
    obs[squares, rng.integers(0, 2, n_discs)] = True
    return obs.reshape(8, 8, 2)


def legal_mask(moves) -> np.ndarray:
    mask = np.zeros(65, bool)
    mask[list(moves)] = True
    return mask


def tiny_set(rng: np.random.Generator, n: int = 4) -> EngineSet:
    obs = np.stack([board_obs(20 + i, rng) for i in range(n)])
    legal = np.stack([legal_mask([i, i + 1, i + 7]) for i in range(n)])
    scores = np.where(legal, rng.integers(-20, 20, (n, 65)), ILLEGAL).astype(np.int16)
    empties = (64 - obs.sum(axis=(1, 2, 3))).astype(np.int8)
    return EngineSet(
        obs=obs,
        legal=legal,
        scores=scores,
        depth=np.minimum(empties, 26).astype(np.int8),
        exact=empties <= 26,
        labeller=np.zeros(n, np.int8),
        labeller_names=["neural-reversi level 26"],
        source=np.array([0, 0, 1, 1], np.int8)[:n],
        source_names=["vs-engines", "head_to_head"],
        source_descriptions=["evals.db games: a", "evals.db games: b"],
        game=np.array([7, 7, 9, 12], np.int64)[:n],
        empties=empties,
    )


def test_a_set_saves_and_loads_unchanged(tmp_path):
    s = tiny_set(np.random.default_rng(0))

    s.save(tmp_path / "sets" / "set.npz")
    loaded = EngineSet.load(tmp_path / "sets" / "set.npz")

    for field in ("obs", "legal", "scores", "depth", "exact", "labeller", "source"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(s, field))
    np.testing.assert_array_equal(loaded.game, s.game)
    np.testing.assert_array_equal(loaded.empties, s.empties)
    for names in ("labeller_names", "source_names", "source_descriptions"):
        assert getattr(loaded, names) == getattr(s, names)
    assert loaded.describe() == {
        "labellers": {"neural-reversi level 26": 4},
        "exact": int(s.exact.sum()),
        "sources": {
            "vs-engines": {"positions": 2, "description": "evals.db games: a"},
            "head_to_head": {"positions": 2, "description": "evals.db games: b"},
        },
    }
    assert not list((tmp_path / "sets").glob("*.tmp*"))


def test_a_subset_keeps_its_positions_with_every_source():
    s = tiny_set(np.random.default_rng(0))

    sub = s.subset(np.array([False, True, False, True]))

    np.testing.assert_array_equal(sub.game, [7, 12])
    np.testing.assert_array_equal(sub.obs, s.obs[[1, 3]])
    assert sub.source_names == s.source_names and len(sub) == 2


def test_a_set_reports_its_phases_and_on_request_its_sources():
    s = tiny_set(np.random.default_rng(0))
    s = EngineSet(**{**s.__dict__, "empties": np.array([45, 30, 10, 51], np.int8)})

    groups = s.groups()
    by_source = s.groups(by_source=True)

    # metrics by game phase; by source only when asked (for checking a set)
    assert list(groups) == ["opening", "midgame", "endgame"]
    assert list(by_source) == [
        "vs-engines",
        "head_to_head",
        "opening",
        "midgame",
        "endgame",
    ]
    np.testing.assert_array_equal(by_source["head_to_head"], [False, False, True, True])
    np.testing.assert_array_equal(groups["opening"], [True, False, False, True])
    np.testing.assert_array_equal(groups["endgame"], [False, False, True, False])
    positions = s.scored_positions()
    assert positions.scores.dtype == np.float32
    assert positions.observations.shape == (4, 8, 8, 2)
    assert list(positions.groups) == ["opening", "midgame", "endgame"]
    assert list(s.scored_positions(by_source=True).groups)[:2] == [
        "vs-engines",
        "head_to_head",
    ]


def test_filtered_positions_are_those_the_network_decides_in_halves():
    s = tiny_set(np.random.default_rng(0))
    # positions with 20 empty squares or fewer are the solver's
    s = EngineSet(**{**s.__dict__, "empties": np.array([45, 30, 20, 21], np.int8)})

    filtered = s.scored_positions(filtered=True)

    assert filtered.scores.dtype == np.float32
    np.testing.assert_array_equal(filtered.observations, s.obs[[0, 1, 3]])
    np.testing.assert_array_equal(filtered.scores, s.scores[[0, 1, 3]])
    np.testing.assert_array_equal(filtered.legal, s.legal[[0, 1, 3]])
    assert {k: v.tolist() for k, v in filtered.groups.items()} == {
        "opening": [True, False, False],
        "midgame": [False, True, True],
    }
    assert list(s.scored_positions(by_source=True, filtered=True).groups) == [
        "vs-engines",
        "head_to_head",
        "opening",
        "midgame",
    ]


@pytest.mark.parametrize(
    "change, error",
    [
        ({"source_names": ["a b", "c"]}, "letters, digits"),
        ({"scores": np.zeros((4, 65), np.int16)}, "every legal action needs a score"),
        ({"source_descriptions": ["a"]}, "every source needs its description"),
    ],
)
def test_a_set_refuses_bad_names_and_missing_scores(change, error):
    s = tiny_set(np.random.default_rng(0))
    with pytest.raises(ValueError, match=error):
        EngineSet(**{**s.__dict__, **change})


def test_only_positions_past_the_opening_with_a_choice_are_selectable_once():
    rng = np.random.default_rng(0)
    opening = board_obs(12, rng)  # 52 empty: an XOT opening's
    midgame = board_obs(30, rng)
    other = board_obs(40, rng)
    obs = np.stack([opening, midgame, midgame, other, other])
    legal = np.stack(
        [
            legal_mask([1, 2]),
            legal_mask([1, 2]),
            legal_mask([1, 2]),
            legal_mask([3, 64]),
            legal_mask([3, 4]),
        ]
    )

    # the repeat of the midgame position isn't selectable, nor is a single move (and a pass);
    # the second `other`, with two moves, is a different position only in its moves, so it's
    # a repeat too
    np.testing.assert_array_equal(
        selectable(obs, legal), [False, True, False, False, False]
    )


def random_game(rng: random.Random, plies: int = 60) -> list[int]:
    """A game of random legal moves from the standard start, as the evals store keeps one."""
    board, moves = Board(), []
    while len(moves) < plies and board.legal():
        move = rng.choice(board.legal())
        moves.append(move)
        board.play(move)
    return moves


def test_game_positions_put_the_player_to_move_first():
    # the standard start, black (d5, e4) to move, then white's turn after d3
    positions = label_positions.game_positions([19, 18])

    obs, legal = positions[0]
    assert obs[4, 3, 0] and obs[3, 4, 0] and obs[3, 3, 1] and obs[4, 4, 1]
    np.testing.assert_array_equal(np.nonzero(legal)[0], [19, 26, 37, 44])
    # white to move: its discs (now just e5) first
    obs, legal = positions[1]
    assert obs[4, 4, 0] and obs.sum() == 5 and obs[:, :, 0].sum() == 1
    assert legal[18]


def test_sample_spreads_over_bands_and_caps_each_game():
    rng = random.Random(0)
    games = [(gid, random_game(rng)) for gid in range(60)]
    taken: set[bytes] = set()

    obs, legal, game = label_positions.sample(
        games, 100, random.Random(1), taken, per_game=3
    )

    empties = 64 - obs.sum(axis=(1, 2, 3))
    assert len(obs) == 100 and (empties >= 21).all() and (empties <= 51).all()
    assert max(np.unique(game, return_counts=True)[1]) <= 3
    assert len({o.tobytes() for o in obs}) == 100 and len(taken) == 100
    assert (legal[:, :64].sum(axis=1) >= 2).all()
    # about the default shares of 21-27, 28-35, 36-43, 44-51 empties
    counts = [
        ((empties >= lo) & (empties <= hi)).sum() for lo, hi in label_positions.BANDS
    ]
    assert counts == [20, 35, 30, 15]
    # none twice across sources, and too few games is an error
    again, _, _ = label_positions.sample(games, 20, random.Random(2), taken, per_game=3)
    assert not {o.tobytes() for o in again} & {o.tobytes() for o in obs}
    with pytest.raises(ValueError, match="not 400"):
        label_positions.sample(games[:10], 400, random.Random(3), set(), per_game=3)


def write_store(path: Path, games: dict[tuple[str, str], list[list[int]]]) -> None:
    """An evals store with just what `source_games` reads: players and their games."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE players (id TEXT, label TEXT, kind TEXT, config TEXT)")
    db.execute(
        "CREATE TABLE games (id INTEGER, black_id TEXT, white_id TEXT, moves TEXT)"
    )
    clock = {"budget": {"clock": {"ext": None, "inc": 0.0, "ini": 30.0}}}
    players = [
        ("n1", "ray11x64-r1 30s", "net", {}),
        ("n2", "ray9x64n-r2", "net", {}),
        ("e1", "edax 30s", "engine", clock),
        ("e2", "egaroucid 30s", "engine", clock),
        ("e3", "edax level 4", "engine", {"budget": {"level": 4}}),
    ]
    for pid, label, kind, config in players:
        db.execute(
            "INSERT INTO players VALUES (?, ?, ?, ?)",
            (pid, label, kind, json.dumps(config)),
        )
    gid = 0
    for (black, white), lists in games.items():
        for moves in lists:
            gid += 1
            db.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?)",
                (gid, black, white, json.dumps(moves)),
            )
    db.commit()
    db.close()


def test_sources_take_their_kind_of_games_from_the_store(tmp_path):
    write_store(
        tmp_path / "evals.db",
        {
            ("n1", "e1"): [[19]],
            ("e2", "n1"): [[19, 18]],
            ("n1", "e3"): [[37]],  # an engine at a level, not on the clock
            ("e1", "e2"): [[26]],
            ("n2", "n1"): [[44]],
        },
    )
    sources = [
        label_positions.parse_source("vs=5:net-engine:ray11x64"),
        label_positions.parse_source("engines=5:engine-engine"),
        label_positions.parse_source("h2h=5:net-net:ray11x64/ray9x64n"),
    ]

    games = label_positions.source_games(tmp_path / "evals.db", sources, clock=30.0)

    assert games == {
        "vs": [(1, [19]), (2, [19, 18])],
        "engines": [(4, [26])],
        "h2h": [(5, [44])],
    }
    assert "30 s" in sources[0].description(30.0)
    with pytest.raises(argparse.ArgumentTypeError):
        label_positions.parse_source("vs=5:self-play")


class FakeAnalyst:
    """Scores each legal move by its square, exact from 26 empty squares down; counts the
    positions it saw and whether it was closed."""

    made: ClassVar[list["FakeAnalyst"]] = []

    def __init__(self) -> None:
        self.seen = 0
        self.closed = False
        FakeAnalyst.made.append(self)

    def analyse(self, me: int, opp: int) -> MoveScores:
        board = Board.at(me, opp, "b")
        self.seen += 1
        empties = 64 - (me | opp).bit_count()
        return MoveScores(
            {m: m - 30 for m in board.legal()}, min(empties, 26), empties <= 26
        )

    def close(self) -> None:
        self.closed = True


def test_labels_score_every_legal_move_with_several_analysts():
    rng = random.Random(0)
    games = [(gid, random_game(rng)) for gid in range(30)]
    obs, legal, _ = label_positions.sample(
        games, 40, random.Random(1), set(), per_game=3
    )
    FakeAnalyst.made = []

    scores, depth, exact = label_positions.label(obs, workers=3, analyst=FakeAnalyst)

    assert len(FakeAnalyst.made) == 3 and all(a.closed for a in FakeAnalyst.made)
    assert sum(a.seen for a in FakeAnalyst.made) == 40
    np.testing.assert_array_equal(scores != ILLEGAL, legal)
    np.testing.assert_array_equal(scores[legal], (np.nonzero(legal)[1] - 30))
    empties = 64 - obs.sum(axis=(1, 2, 3))
    np.testing.assert_array_equal(exact, empties <= 26)
    np.testing.assert_array_equal(depth, np.minimum(empties, 26))


def test_assemble_needs_every_position_labelled_once(tmp_path):
    rng = random.Random(0)
    games = [(gid, random_game(rng)) for gid in range(20)]
    obs, legal, game = label_positions.sample(
        games, 10, random.Random(1), set(), per_game=3
    )
    path = tmp_path / "positions.npz"
    label_positions.save_positions(
        path, obs, legal, game, np.zeros(10, np.int8), ["random"], ["random games"]
    )
    positions = label_positions.load_positions(path)
    scores, depth, exact = label_positions.label(obs, workers=1, analyst=FakeAnalyst)
    for name, index in (
        ("labels-0.npz", np.arange(0, 6)),
        ("labels-1.npz", np.arange(6, 10)),
    ):
        np.savez(
            tmp_path / name,
            index=index,
            scores=scores[index],
            depth=depth[index],
            exact=exact[index],
        )

    built = label_positions.assemble(
        positions, [tmp_path / "labels-0.npz", tmp_path / "labels-1.npz"], 26
    )

    np.testing.assert_array_equal(built.obs, obs)
    np.testing.assert_array_equal(built.scores, scores)
    np.testing.assert_array_equal(built.game, game)
    assert built.labeller_names == ["neural-reversi level 26"] and len(built) == 10
    with pytest.raises(ValueError, match="no labels"):
        label_positions.assemble(positions, [tmp_path / "labels-0.npz"], 26)
    some = label_positions.assemble(
        positions, [tmp_path / "labels-0.npz"], 26, partial=True
    )
    np.testing.assert_array_equal(some.obs, obs[:6])
    with pytest.raises(ValueError, match="already labelled"):
        label_positions.assemble(positions, [tmp_path / "labels-0.npz"] * 2, 26)


def test_training_skips_a_missing_default_set_but_not_a_missing_given_one(
    tmp_path, monkeypatch
):
    import train

    parser = argparse.ArgumentParser()
    monkeypatch.setattr(train, "DEFAULT_ENGINE_SET", tmp_path / "none.npz")
    args = argparse.Namespace(engine_set=None, engine_eval_every=1)
    with pytest.warns(UserWarning, match="no engine-labelled set"):
        assert train.load_engine_set(parser, args) is None

    with pytest.raises(SystemExit):
        train.load_engine_set(
            parser,
            argparse.Namespace(engine_set=str(tmp_path / "x.npz"), engine_eval_every=1),
        )

    tiny_set(np.random.default_rng(0)).save(tmp_path / "given.npz")
    given = train.load_engine_set(
        parser,
        argparse.Namespace(engine_set=str(tmp_path / "given.npz"), engine_eval_every=1),
    )
    assert given is not None and len(given) == 4
    off = argparse.Namespace(engine_set=str(tmp_path / "x.npz"), engine_eval_every=0)
    assert train.load_engine_set(parser, off) is None
    assert engine_set.DEFAULT_PATH.name.endswith(".npz")
