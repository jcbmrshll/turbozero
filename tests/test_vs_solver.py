"""vs_solver.py's games against the solver: as train.py plays them, between epochs
(`SolverMatchTester`), with a fake match, and one real game with a tiny network."""

import fnmatch
import json
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np
import pytest
from example_modules import load_example
from seaglass import Arrays
from test_connect_four_solver import OneMoveSolver

game, vs_solver, solver_module = load_example(
    "connect_four", "game", "vs_solver", "solver"
)
SolverMatch = vs_solver.SolverMatch
SolverMatchTester = vs_solver.SolverMatchTester
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


def run(tester, epoch, state, params, logged=None):
    return tester.run(
        key=jax.random.PRNGKey(0),
        epoch_num=epoch,
        max_steps=0,
        env_step_fn=None,
        env_init_fn=None,
        evaluator=None,
        state=state,
        params=params,
        log_fn=None if logged is None else logged.append,
    )


def test_plays_at_the_epochs_it_is_set_to():
    tester = SolverMatchTester(FakeMatch(), num_epochs=10, every=4, last=2)
    assert [e for e in range(10) if tester.plays_at(e)] == [3, 7, 8, 9]
    with pytest.raises(ValueError, match="between 0 and 10"):
        SolverMatchTester(FakeMatch(), num_epochs=10, last=11)


def test_logs_each_level_and_the_mean_of_the_last_networks(tmp_path):
    match = FakeMatch()
    results = tmp_path / "vs_solver.jsonl"
    tester = SolverMatchTester(
        match, num_epochs=3, last=2, record=True, results=results, config={"dtype": "x"}
    )
    state = tester.init()
    logged: list[dict] = []
    state, metrics, episode = run(tester, 0, state, 1, logged)
    assert logged == [] and metrics == {} and episode is None and match.plays == []
    for epoch, params in ((1, 2), (2, 3)):
        state, _, _ = run(tester, epoch, state, params, logged)
    assert match.plays == [(2, 0, True), (2, 42, True), (3, 0, True), (3, 42, True)]
    merged = {k: v for d in logged for k, v in d.items()}
    assert merged["solver_level42_sims256"] == points(fake_results(3, 42, 6)).mean()
    assert merged["solver_level42_sims256_game"].arrays["x"] == [3]
    expected = (points(fake_results(2, 42, 6)) + points(fake_results(3, 42, 6))) / 2
    assert merged["solver_level42_sims256_last2"] == pytest.approx(expected.mean())
    assert "solver_seconds" in merged
    lines = [json.loads(line) for line in results.read_text().splitlines()]
    assert [(r["epoch"], r["level"]) for r in lines] == [
        (1, 0),
        (1, 42),
        (2, 0),
        (2, 42),
    ]
    assert lines[0]["engine"] == "solver" and lines[0]["dtype"] == "x"


def test_every_logged_metric_is_declared():
    declared = json.loads(
        (Path(__file__).parent.parent / "core/monitoring/declared.json").read_text()
    )
    match = FakeMatch()
    tester = SolverMatchTester(match, num_epochs=1, last=1, record=True)
    logged: list[dict] = []
    run(tester, 0, tester.init(), 1, logged)
    names = {k for d in logged for k in d}
    assert names and all(
        any(fnmatch.fnmatchcase(name, pattern) for pattern in declared)
        for name in names
    )
    # the score alone is headed for the tiles, by describe_scores
    assert "solver_seconds" in declared


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
