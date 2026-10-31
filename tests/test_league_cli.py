"""The league's commands for asking for a series."""

import pytest

from core.league.cli import main
from core.league.game import LeagueGame
from core.league.store import Store


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

    def make_evaluator(self, network, sims):
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
