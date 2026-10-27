import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

import engine_set
import label_positions
from engine_set import ILLEGAL, EngineSet, SourceState, boards, selectable
from engines import Analysis


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
    return EngineSet(
        obs=obs,
        legal=legal,
        scores=scores,
        depth=np.where(legal, 24, 0).astype(np.int16),
        source=np.array([0, 0, 1, 1], np.int8)[:n],
        source_names=["held-out", "self_play"],
        source_states=[
            SourceState("/runs/a/state-150.npz", (896, 1024)),
            SourceState("/runs/b/state-450.npz"),
        ],
        empties=(64 - obs.sum(axis=(1, 2, 3))).astype(np.int8),
        engine="edax",
        level=24,
    )


def test_a_set_saves_and_loads_unchanged(tmp_path):
    s = tiny_set(np.random.default_rng(0))

    s.save(tmp_path / "sets" / "set.npz")
    loaded = EngineSet.load(tmp_path / "sets" / "set.npz")

    for field in ("obs", "legal", "scores", "depth", "source", "empties"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(s, field))
    assert (loaded.source_names, loaded.engine, loaded.level) == (
        ["held-out", "self_play"],
        "edax",
        24,
    )
    # where each source's positions came from, for training data to leave their games out
    assert loaded.source_states == s.source_states
    assert loaded.describe()["sources"]["held-out"] == {
        "positions": 2,
        "state": "/runs/a/state-150.npz",
        "envs": (896, 1024),
    }
    assert not list((tmp_path / "sets").glob("*.tmp*"))


def test_a_set_reports_its_phases_and_on_request_its_sources():
    s = tiny_set(np.random.default_rng(0))
    s = EngineSet(**{**s.__dict__, "empties": np.array([45, 30, 10, 51], np.int8)})

    groups = s.groups()
    by_source = s.groups(by_source=True)

    # metrics by game phase; by source only when asked (for checking a set)
    assert list(groups) == ["opening", "midgame", "endgame"]
    assert list(by_source) == ["held-out", "self_play", "opening", "midgame", "endgame"]
    np.testing.assert_array_equal(by_source["self_play"], [False, False, True, True])
    np.testing.assert_array_equal(groups["opening"], [True, False, False, True])
    np.testing.assert_array_equal(groups["endgame"], [False, False, True, False])
    positions = s.scored_positions()
    assert positions.scores.dtype == np.float32
    assert positions.observations.shape == (4, 8, 8, 2)
    assert list(positions.groups) == ["opening", "midgame", "endgame"]
    assert list(s.scored_positions(by_source=True).groups)[:2] == [
        "held-out",
        "self_play",
    ]


@pytest.mark.parametrize(
    "change, error",
    [
        ({"source_names": ["a b", "c"]}, "letters, digits"),
        ({"scores": np.zeros((4, 65), np.int16)}, "every legal action needs a score"),
        ({"source_states": [SourceState("/a.npz")]}, "every source needs its state"),
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


def test_boards_put_the_player_to_move_first():
    obs = np.zeros((1, 8, 8, 2), bool)
    obs[0, 0, 0, 0] = True  # a1: the player to move
    obs[0, 7, 7, 1] = True  # h8: the opponent

    board = boards(obs)[0]

    assert board[0] == 1 and board[63] == 2 and (board[1:63] == 0).all()


def fake_analyst(level):
    """Scores each legal move of a board by its square plus the level: an analyst the tests can
    check. The legal moves are the squares the player to move doesn't hold that a disc of theirs
    is next to... any rule works, so: the empty squares among the first 8."""

    def analyse(board):
        return [
            Analysis([sq], int(sq) + level, level) for sq in range(8) if board[sq] == 0
        ]

    return analyse


def test_build_samples_each_source_and_scores_every_legal_move():
    rng = np.random.default_rng(0)
    # boards whose empty squares among the first 8 are their legal moves (as fake_analyst sees them)
    obs, legal = [], []
    for _ in range(12):
        o = board_obs(30, rng).reshape(64, 2)
        o[:8] = False
        o[3, 0] = True
        obs.append(o.reshape(8, 8, 2))
        legal.append(legal_mask([0, 1, 2, 4, 5, 6, 7]))
    obs, legal = np.stack(obs), np.stack(legal)
    states = [SourceState("/a.npz", (0, 6)), SourceState("/b.npz")]
    sources = [
        ("a", states[0], obs[:6], legal[:6]),
        ("b", states[1], obs[6:], legal[6:]),
    ]

    built = label_positions.build(sources, 4, level=10, workers=0, factory=fake_analyst)

    assert len(built) == 8 and built.source_names == ["a", "b"]
    assert built.source_states == states
    np.testing.assert_array_equal(built.source, [0] * 4 + [1] * 4)
    expected = np.where(legal_mask([0, 1, 2, 4, 5, 6, 7]), np.arange(65) + 10, ILLEGAL)
    for scores in built.scores:
        np.testing.assert_array_equal(scores, expected)
    assert (built.depth[built.legal] == 10).all()
    # sampled without repeats, within and across sources
    assert len({o.tobytes() for o in built.obs}) == 8

    with pytest.raises(ValueError, match="fewer than 7"):
        label_positions.build(sources, 7, level=10, workers=0, factory=fake_analyst)


def write_state(path, observation, mask, has_reward, next_idx, transforms=1):
    """A saved state with just what `buffer_positions` reads (see `Trainer.save_state`)."""
    leaves = [
        (".buffer_state.next_idx", next_idx),
        (".buffer_state.populated", np.ones_like(has_reward)),
        (".buffer_state.has_reward", has_reward),
        (".buffer_state.buffer.observation_nn", observation),
        (".buffer_state.buffer.policy_mask", mask),
    ]
    meta = {
        "config": {"data_transforms": transforms},
        "leaves": {
            "collection_state": [
                [name, list(x.shape), str(x.dtype)] for name, x in leaves
            ]
        },
    }
    np.savez(
        path,
        meta=np.array(json.dumps(meta)),
        **{f"collection_state.{i}": x for i, (_, x) in enumerate(leaves)},
    )


def test_buffer_positions_are_the_finished_games_originals(tmp_path):
    # 3 environments, 4 entries each: 2 moves of a position and its copy
    observation = np.arange(3 * 4 * 128).reshape(3, 4, 8, 8, 2) % 2 == 0
    observation[:, :, 0, 0, 0] = np.arange(12).reshape(3, 4) % 3 == 0
    mask = np.zeros((3, 4, 65), bool)
    mask[..., 1] = True
    has_reward = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], bool)
    path = tmp_path / "state-1.npz"
    write_state(path, observation, mask, has_reward, next_idx=np.array([0, 2, 2]))

    obs, legal = label_positions.buffer_positions(path)
    some_obs, _ = label_positions.buffer_positions(path, envs=(1, 3))

    np.testing.assert_array_equal(
        obs, np.stack([observation[0, 0], observation[0, 2], observation[1, 0]])
    )
    assert legal.shape == (3, 65) and legal[:, 1].all()
    np.testing.assert_array_equal(some_obs, observation[1, :1])

    write_state(path, observation, mask, has_reward, next_idx=np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="groups of 2"):
        label_positions.buffer_positions(path)


@pytest.mark.parametrize(
    "text, parsed",
    [
        ("az=runs/a/state-150.npz", ("az", "runs/a/state-150.npz", None)),
        (
            "az=runs/a/state-150.npz:896-1024",
            ("az", "runs/a/state-150.npz", (896, 1024)),
        ),
    ],
)
def test_sources_name_a_state_and_optionally_its_environments(text, parsed):
    assert label_positions.parse_source(text) == parsed
    with pytest.raises(argparse.ArgumentTypeError):
        label_positions.parse_source("runs/a/state-150.npz")


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
