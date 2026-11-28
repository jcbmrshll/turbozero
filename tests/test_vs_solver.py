"""vs_solver.py's games against the solver: what a match logs, with a fake match, and one real
game with a tiny network."""

import fnmatch
import json
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import numpy as np
from example_modules import load_example
from seaglass import Arrays
from test_connect_four_solver import OneMoveSolver

game, vs_solver, solver_module = load_example(
    "connect_four", "game", "vs_solver", "solver"
)
SolverMatch = vs_solver.SolverMatch
LevelResult = vs_solver.LevelResult


class FakeMatch(SolverMatch):
    """Plays no games: the network `params` (a number) gets `fake_results` at `level`.
    Records each play."""

    def __init__(self, levels=(0, 42), sims=256, openings=3):  # pylint: disable=super-init-not-called
        self.solver = OneMoveSolver()
        self.levels = list(levels)
        self.sims = sims
        self.num_openings = openings
        self.num_games = 2 * openings
        self.seed = 0
        self.plays = []

    def play(self, params, level, record=False):
        self.plays.append((params, level, record))
        games = np.arange(self.num_games)
        return LevelResult(
            level,
            results=fake_results(params, level, self.num_games),
            we_first=games % 2 == 0,
            seconds=1.0,
            episode=Arrays("game", x=[params]) if record else None,
        )


def fake_results(params, level, num_games):
    """Wins in the first few games, as many as `params` and `level` say, then a draw and
    losses."""
    games = np.arange(num_games)
    wins = (params + level) % num_games
    return np.where(games < wins, 1, np.where(games == wins, 0, -1))


def points(results):
    return (1 + np.sign(results)) / 2


def test_every_logged_metric_is_declared():
    declared = json.loads(
        (Path(__file__).parent.parent / "core/monitoring/declared.json").read_text()
    )
    match = FakeMatch()
    names = {
        name
        for level in match.levels
        for name in match.play(1, level, record=True).metrics(match.name(level))
    }
    assert names and all(
        any(fnmatch.fnmatchcase(name, pattern) for pattern in declared)
        for name in names
    )


def test_describe_scores_declares_them_better_higher():
    calls = []
    monitor = SimpleNamespace(
        describe=lambda name, **kwargs: calls.append((name, kwargs))
    )
    vs_solver.describe_scores(monitor, ["a"])
    assert calls == [("a", {"better": "higher", "headline": True})]


def test_a_real_match_with_a_tiny_network():
    network, state = game.make_network("resnet:1x4")
    params = (eqx.filter(network, eqx.is_inexact_array), state)
    solver = OneMoveSolver()
    match = SolverMatch(network, solver, [0, 1], sims=4, openings=2)
    result = match.play(params, 1, record=True)
    assert result.results.shape == (4,)
    assert result.we_first.tolist() == [True, False, True, False]
    # a level 1 opponent takes any win it can
    assert set(result.results.tolist()) <= {-1, 0, 1}
    episode = result.episode
    assert episode.type == "turbozero/connect_four-episode"
    assert episode.attrs == {"opponent": "solver level 1"}
    frames = episode.arrays["board"].shape[0]
    assert int(episode.arrays["start"]) == 6
    # both players' records of every position from the opening on, the solver's of the opening too
    assert episode.arrays["search_player"].tolist().count(1) == frames - 1
    assert episode.arrays["search_player"].tolist().count(0) == frames - 1 - 6
    # the same games again, for any network
    again = match.play(params, 1)
    np.testing.assert_array_equal(again.results, result.results)
    assert match.name(42) == "solver_level42_sims4"


def test_perfect_lines_follow_the_best_moves():
    solver = OneMoveSolver()
    # the player to move has three in a column, and takes the win
    position = "121212"
    lines = vs_solver.perfect_lines(solver, position, max_plies=3)
    by_move = {line[0]: (line, value) for line, value, _ in lines}
    assert by_move[0] == ([0], 1.0)
    assert solver_module.COLUMNS == len(lines)
    # a line stops where its game ends: the opponent then wins in the second column
    assert by_move[2] == ([2, 1], 0.0)
    # or at the plies asked for
    lines = vs_solver.perfect_lines(solver, "", max_plies=3)
    assert [len(line) for line, _, _ in lines] == [3] * 7
