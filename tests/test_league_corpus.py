import json
import sqlite3
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np
import pytest
from example_modules import load_example

from core.league.store import (
    EngineSearch,
    NetSearch,
    Search,
    TopMove,
    encode_searches,
)

league_corpus, game, xot, distill = load_example(
    "othello", "league_corpus", "game", "xot", "distill"
)

FIRST, SECOND = 10, 20

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
        "moves BLOB, first_id INTEGER, second_id INTEGER)"
    )
    db.execute(
        "CREATE TABLE game_searches (game_id INTEGER PRIMARY KEY, searches BLOB)"
    )
    for i, (opening, moves) in enumerate(
        zip(OPENINGS + [OPENINGS[0]], games, strict=True)
    ):
        result = first_player_result(moves)
        if i == len(OPENINGS):
            result = 1 if result != 1 else -1
        db.execute(
            "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?)",
            (
                i + 1,
                opening,
                result,
                np.asarray(moves, np.uint16).tobytes(),
                FIRST,
                SECOND,
            ),
        )
    # every game's searches: one at every ply, its move given 3 visits of 4 (in the first game an
    # engine's at ply 9; and one in each opening, which the corpus doesn't keep)
    for g, moves in enumerate(games[: len(OPENINGS)]):
        searches: dict[int, Search] = {
            p: NetSearch(
                value=p / 100,
                net_value=0.0,
                visits=4,
                evals=4,
                tree=4,
                top=(
                    TopMove(moves[p], 3, 0.0),
                    TopMove(64 if moves[p] != 64 else 0, 1, 0.0),
                ),
            )
            for p in range(len(moves))
        }
        if g == 0:
            searches[9] = EngineSearch(score=2, depth=10)
        db.execute(
            "INSERT INTO game_searches VALUES (?, ?)",
            (g + 1, encode_searches(searches)),
        )
    db.execute("INSERT INTO games VALUES (99, 0, 1, ?, 1, 2)", (b"",))
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
        "init_weights": None,
        "policy_source": "teacher",
        "searchers": None,
    }
    with pytest.raises(SystemExit, match="searches"):
        distill.train_command(SimpleNamespace(**{**args, "value_source": "root"}))
    distill.train_command(SimpleNamespace(**args))
    game.load_checkpoint(str(tmp_path / "student" / "1.eqx"), NET)
    # and on from that network, with steps too small to change it leaving it as it was
    first = tmp_path / "student" / "1.eqx"
    distill.train_command(
        SimpleNamespace(
            **{
                **args,
                "init_weights": str(first),
                "lr": 1e-20,
                "lr_final": 1e-20,
                "weight_decay": 0.0,
                "ckpt_dir": str(tmp_path / "continued"),
            }
        )
    )
    before = game.load_checkpoint(str(first), NET)[1][0]
    after = game.load_checkpoint(str(tmp_path / "continued" / "1.eqx"), NET)[1][0]
    for x, y in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        np.testing.assert_array_equal(x, y)


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


def test_searches_attach_each_position_its_movers_search(league, tmp_path):
    root, games = league
    out = tmp_path / "corpus"
    league_corpus.build(root / "league.db", out, fraction=0.5, seed=1)
    summary = league_corpus.attach_searches(
        out, root / "league.db", log=lambda line: None
    )
    loaded = league_corpus.Corpus(out)
    game_ids, ply = np.load(out / "game.npy"), np.load(out / "ply.npy")
    rows = np.flatnonzero(game_ids == 1)
    # every kept position but the engine's move at ply 9 of game 1 has a search
    assert summary["positions_with_searches"] == len(loaded) - 1
    state = game.env.init(jax.random.PRNGKey(0))
    first = int(state.current_player)
    for p, a in enumerate(games[0]):
        if p >= 8:
            row = rows[ply[rows] == p][0]
            if p == 9:
                assert loaded.searcher[row] == -1
            else:
                mover = FIRST if int(state.current_player) == first else SECOND
                assert loaded.searcher[row] == mover
                assert loaded.search_value[row] == pytest.approx(p / 100, abs=1e-3)
                assert float(loaded.search_policy[row][a]) == pytest.approx(
                    0.75, abs=1e-3
                )
        state = game.env.step(state, a)
    first_moves = (
        loaded.split("train", np.array([FIRST])).tolist()
        + loaded.split("heldout", np.array([FIRST])).tolist()
    )
    assert (
        sorted(first_moves)
        == np.flatnonzero(np.asarray(loaded.searcher) == FIRST).tolist()
    )
    label_corpus(out, tmp_path)
    loaded = league_corpus.Corpus(out)
    searched = np.flatnonzero(np.asarray(loaded.searcher) >= 0)
    batch = loaded.experiences(searched, policy="search", value="search")
    np.testing.assert_allclose(
        batch.search_value, np.asarray(loaded.search_value)[searched], atol=1e-6
    )
    with pytest.raises(ValueError, match="searches"):
        loaded.experiences(np.arange(len(loaded)), policy="search")
    # a student of the searches of one player: its positions only, on their visits and root values
    searchers = tmp_path / "searchers.npy"
    np.save(searchers, np.array([FIRST]))
    distill.train_command(
        SimpleNamespace(
            data=None,
            heldout=None,
            corpus=str(out),
            heldout_size=5,
            teacher_values=None,
            value_source="root",
            value_target_q=1.0,
            policy_source="search",
            searchers=str(searchers),
            net=NET,
            train_dtype="float32",
            seed=0,
            monitor=None,
            name=None,
            notes=None,
            passes=1,
            batch=2,
            lr=1e-3,
            lr_final=1e-4,
            weight_decay=0.5,
            eval_every=1,
            init_weights=None,
            ckpt_dir=str(tmp_path / "searched"),
        )
    )
    game.load_checkpoint(str(tmp_path / "searched" / "1.eqx"), NET)
