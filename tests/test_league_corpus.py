import json
import sqlite3
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np
import pytest
from example_modules import load_example

league_corpus, game, xot, distill = load_example(
    "othello", "league_corpus", "game", "xot", "distill"
)

NET = "resnet:1x4"
OPENINGS = [3, 50, 7000, 10783]


def random_games(openings: list[int], seed: int = 0) -> list[list[int]]:
    """Games from the standard start through each XOT opening, then random legal moves to the
    end, passes included."""
    rng = np.random.default_rng(seed)
    xot_openings = xot.load_xot()
    step = jax.jit(game.env.step)
    games = []
    for opening in openings:
        state = game.env.init(jax.random.PRNGKey(0))
        moves = [int(a) for a in xot_openings[opening]]
        for a in moves:
            state = step(state, a)
        while not bool(state.terminated):
            a = int(rng.choice(np.flatnonzero(np.asarray(state.legal_action_mask))))
            moves.append(a)
            state = step(state, a)
        games.append(moves)
    return games


def first_player_result(moves: list[int]) -> int:
    state = game.env.init(jax.random.PRNGKey(0))
    first = int(state.current_player)
    for a in moves:
        state = game.env.step(state, a)
    return int(np.sign(state.rewards[first]))


@pytest.fixture(scope="module")
def league(tmp_path_factory):
    """A league store's games table: a game per opening, then one whose recorded result is
    wrong (dropped) and one with no moves (from elsewhere, left out)."""
    root = tmp_path_factory.mktemp("league")
    games = random_games(OPENINGS + [OPENINGS[0]])
    db = sqlite3.connect(root / "league.db")
    db.execute(
        "CREATE TABLE games (id INTEGER PRIMARY KEY, opening INTEGER, result INTEGER, "
        "moves BLOB)"
    )
    for i, (opening, moves) in enumerate(
        zip(OPENINGS + [OPENINGS[0]], games, strict=True)
    ):
        result = first_player_result(moves)
        if i == len(OPENINGS):
            result = 1 if result != 1 else -1
        db.execute(
            "INSERT INTO games VALUES (?, ?, ?, ?)",
            (i + 1, opening, result, np.asarray(moves, np.uint16).tobytes()),
        )
    db.execute("INSERT INTO games VALUES (99, 0, 1, ?)", (b"",))
    db.commit()
    db.close()
    return root, games[: len(OPENINGS)]


@pytest.fixture(scope="module")
def corpus(league, tmp_path_factory):
    root, _ = league
    out = tmp_path_factory.mktemp("corpus")
    summary = league_corpus.build(root / "league.db", out, fraction=0.5, seed=1)
    return out, summary


def test_build_keeps_every_position_after_the_opening(league, corpus):
    _, games = league
    out, summary = corpus
    loaded = league_corpus.Corpus(out)
    expected = sum(len(moves) - 8 for moves in games)
    assert summary["games"] == len(games) and summary["games_dropped"] == 1
    assert summary["games_with_other_results"] == 1
    assert len(loaded) == summary["positions"] == expected
    game_ids = np.load(out / "game.npy")
    ply = np.load(out / "ply.npy")
    for g, moves in enumerate(games):
        # each position as the game reached it, with its game's opening and outcome
        rows = np.flatnonzero(game_ids == g + 1)
        np.testing.assert_array_equal(ply[rows], np.arange(8, len(moves)))
        assert (loaded.opening[rows] == OPENINGS[g]).all()
        state = game.env.init(jax.random.PRNGKey(0))
        players = []
        for i, a in enumerate(moves):
            if i >= 8:
                row = rows[i - 8]
                np.testing.assert_array_equal(
                    loaded.observations(np.array([row]))[0], state.observation
                )
                np.testing.assert_array_equal(
                    loaded.legal_actions(np.array([row]))[0], state.legal_action_mask
                )
                players.append(int(state.current_player))
            state = game.env.step(state, a)
        np.testing.assert_array_equal(
            loaded.z[rows], np.sign(np.asarray(state.rewards)[players])
        )


def test_build_splits_by_opening(corpus):
    out, summary = corpus
    loaded = league_corpus.Corpus(out)
    assert len(loaded.heldout) == summary["heldout_openings"] == 10784 // 2
    train, heldout = loaded.split("train"), loaded.split("heldout")
    assert len(train) + len(heldout) == len(loaded)
    assert np.isin(loaded.opening[heldout], loaded.heldout).all()
    assert not np.isin(loaded.opening[train], loaded.heldout).any()
    assert summary["heldout_positions"] == len(heldout)
    # the held-out openings as an opening list, for vs_engine.py
    np.testing.assert_array_equal(
        xot.load_xot(out / "heldout-openings.txt"), xot.load_xot()[loaded.heldout]
    )


def label_corpus(out, tmp_path) -> None:
    network, nn_state = game.make_network(NET)
    teacher = tmp_path / "teacher.eqx"
    eqx.tree_serialise_leaves(
        teacher, (eqx.filter(network, eqx.is_inexact_array), nn_state)
    )
    league_corpus.label(
        league_corpus.Corpus(out), str(teacher), NET, log=lambda line: None
    )


def test_label_gives_policies_over_legal_actions(corpus, tmp_path):
    out, _ = corpus
    with pytest.raises(ValueError, match="isn't labelled"):
        league_corpus.Corpus(out).experiences(np.arange(2))
    label_corpus(out, tmp_path)
    loaded = league_corpus.Corpus(out)
    assert json.loads((out / "teacher.json").read_text())["teacher_net"] == NET
    legal = loaded.legal_actions(np.arange(len(loaded)))
    policy = np.asarray(loaded.policy, np.float32)
    assert (policy[~legal] == 0).all()
    np.testing.assert_allclose(policy.sum(axis=1), 1, atol=1e-2)
    batch = loaded.experiences(np.array([5, 1]))
    # in the corpus's order, the player to move player 0
    np.testing.assert_array_equal(batch.reward[:, 0], loaded.z[[1, 5]])
    np.testing.assert_array_equal(batch.reward[:, 1], -loaded.z[[1, 5]])
    np.testing.assert_array_equal(batch.search_value, loaded.value[[1, 5]])
    assert (batch.cur_player_id == 0).all()


def test_distill_trains_on_a_corpus(corpus, tmp_path):
    out, _ = corpus
    if not league_corpus.Corpus(out).labelled:
        label_corpus(out, tmp_path)
    args = {
        "data": None,
        "heldout": None,
        "corpus": str(out),
        "heldout_size": 5,
        "teacher_values": None,
        "value_source": "teacher",
        "value_target_q": 1.0,
        "net": NET,
        "train_dtype": "float32",
        "seed": 0,
        "monitor": None,
        "name": None,
        "notes": None,
        "passes": 1,
        "batch": 4,
        "lr": 1e-3,
        "lr_final": 1e-4,
        "weight_decay": 0.5,
        "eval_every": 1,
        "ckpt_dir": str(tmp_path / "student"),
    }
    with pytest.raises(SystemExit, match="no search values"):
        distill.train_command(SimpleNamespace(**{**args, "value_source": "root"}))
    distill.train_command(SimpleNamespace(**args))
    game.load_checkpoint(str(tmp_path / "student" / "1.eqx"), NET)


def test_seed_positions_fill_every_environment_from_the_training_split(
    corpus, tmp_path, monkeypatch
):
    out, _ = corpus
    if not league_corpus.Corpus(out).labelled:
        label_corpus(out, tmp_path)
    (train,) = load_example("othello", "train")
    monkeypatch.setattr(train, "NUM_ENVS", 4)
    loaded = league_corpus.Corpus(out)
    positions = train.seed_positions(str(out), 5, seed=0)
    assert positions.reward.shape == (20, 2)
    # distinct positions of the training split, each with its own labels
    train_values = set(np.asarray(loaded.value)[loaded.split("train")].tolist())
    assert set(positions.search_value.tolist()) <= train_values
