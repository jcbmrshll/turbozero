"""The league's commands for asking for a series."""

import pytest

from core.league.cli import main
from core.league.game import LeagueGame
from core.league.store import BatchSettings, GameRecord, Store


class FakeGame(LeagueGame):
    name = "fake"
    env = None

    def can_play(self, player):
        return player.name != "paper"

    def metadata(self, state):
        raise NotImplementedError

    @property
    def step_fn(self):
        raise NotImplementedError

    def load_network(self, path, net, dtype):
        raise NotImplementedError

    def make_evaluator(self, network, sims, tree_factor=2, prior_temperature=1.0):
        raise NotImplementedError

    def num_openings(self):
        return 100

    def openings(self, indices):
        raise NotImplementedError


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "league"
    store = Store(root)
    store.add_external_player("x", "edax")
    store.add_external_player("y", "edax")
    store.add_external_player("paper", "a paper")
    store.add_external_player("old", "edax")
    store.set_retired("old")
    store.close()
    return root


def run(root, *argv):
    main(FakeGame, root, ["--root", str(root), *argv])


def test_a_match_is_queued_listed_and_cancelled(root, capsys):
    run(root, "match", "x", "y", "--games", "40")
    assert capsys.readouterr().out.strip() == "#1"
    run(root, "matches")
    assert capsys.readouterr().out.split() == ["#1", "x", "vs", "y", "0/40"]
    run(root, "cancel-match", "1")
    capsys.readouterr()
    run(root, "matches")
    assert capsys.readouterr().out.strip() == ""


def test_more_players_of_a_network_can_have_a_bigger_tree(root, tmp_path):
    checkpoint = tmp_path / "n.ckpt"
    checkpoint.write_text("n")
    store = Store(root)
    store.add_network("n", checkpoint, "tiny", "r")
    store.close()
    run(root, "add-player", "n", "--sims", "128", "1000")
    run(root, "add-player", "n", "--sims", "1000", "--tree-factor", "4")
    store = Store(root)
    factors = {p.name: p.tree_factor for p in store.players() if p.network == "n"}
    assert factors == {"n@128": 2, "n@1000": 2, "n@1000-t4": 4}


def test_more_players_of_a_network_can_have_a_flatter_prior(root, tmp_path):
    checkpoint = tmp_path / "n.ckpt"
    checkpoint.write_text("n")
    store = Store(root)
    store.add_network("n", checkpoint, "tiny", "r")
    store.close()
    run(root, "add-player", "n", "--sims", "128", "--prior-temperature", "1.6")
    run(root, "add-player", "n", "--sims", "128", "--tree-factor", "4",
        "--prior-temperature", "2")  # fmt: skip
    store = Store(root)
    players = {p.name: p for p in store.players() if p.network == "n"}
    assert {name: p.settings for name, p in players.items()} == {
        "n@128-T1.6": {"prior_temperature": 1.6},
        "n@128-t4-T2": {"tree_factor": 4, "prior_temperature": 2.0},
    }
    assert players["n@128-T1.6"].prior_temperature == 1.6


def test_matches_says_when_a_player_has_retired_since(root, capsys):
    run(root, "match", "x", "y", "--games", "2")
    run(root, "retire", "y")
    capsys.readouterr()
    run(root, "matches")
    assert "retired" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("a", "b", "games", "says"),
    [
        ("x", "nobody", "2", "no player nobody"),
        ("old", "y", "2", "old is retired"),
        ("x", "paper", "2", "paper can't play here"),
        ("x", "x", "2", "itself"),
        ("x", "y", "3", "even"),
        ("x", "y", "0", "even"),
    ],
)
def test_a_match_that_cant_be_played_is_refused(root, capsys, a, b, games, says):
    with pytest.raises(SystemExit):
        run(root, "match", a, b, "--games", games)
    assert says in capsys.readouterr().err
    store = Store(root)
    assert store.pending_matches() == []
    store.close()


def test_cancelling_what_isnt_pending_is_refused(root, capsys):
    with pytest.raises(SystemExit):
        run(root, "cancel-match", "7")
    assert "no pending match #7" in capsys.readouterr().err


def test_settings_changes_says_who_and_what(root, capsys):
    store = Store(root)
    x, y = (p.id for p in store.players()[:2])

    def played(c_init):
        store.record_batch(
            x,
            y,
            1.0,
            False,
            "abc",
            "f",
            [GameRecord(x, y, 0, 1, [0])],
            settings=BatchSettings({"c_init": c_init}, {"level": 8}, 1, "cpu"),
        )

    played(1.25)
    played(1.25)
    run(root, "settings-changes")
    assert capsys.readouterr().out.strip() == ""
    played(2.0)
    store.close()
    run(root, "settings-changes")
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "x"
    assert "2 batches" in out
    assert "c_init: 1.25 -> 2.0" in out


def test_a_refused_player_can_be_accepted_or_made_anew(root, capsys):
    store = Store(root)
    x, y = store.players()[:2]
    store.record_batch(
        x.id,
        y.id,
        1.0,
        False,
        "abc",
        "f",
        [GameRecord(x.id, y.id, 0, 1, [0])],
        settings=BatchSettings({"c_init": 1.25}, {"level": 8}, 1, "cpu"),
    )
    store.refuse_settings(x.id, {"c_init": 2.0})
    store.refuse_settings(y.id, {"level": 9})
    store.close()
    run(root, "settings-changes")
    out = capsys.readouterr().out
    assert "x refused  c_init: 1.25 -> 2.0" in out
    assert "y refused  level: 8 -> 9" in out

    run(root, "accept-settings", "y")
    run(root, "variant", "c2")
    assert capsys.readouterr().out.strip() == "x-c2"
    run(root, "variant", "b2", "y")
    assert capsys.readouterr().out.strip() == "y-b2"

    store = Store(root)
    by_name = {p.name: p for p in store.players()}
    assert not by_name["y"].retired
    assert by_name["y"].played_settings == {"level": 9}
    assert by_name["x"].retired
    assert not by_name["x-c2"].retired
    store.close()
