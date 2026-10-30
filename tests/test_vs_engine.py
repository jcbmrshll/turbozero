"""vs_engine.py's games against an engine as train.py plays them, between epochs
(`EngineMatchTester`), with a fake match in place of the engine and our agent's search."""

import fnmatch
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from seaglass import Arrays

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from engines import Edax
from vs_engine import (
    EngineMatch,
    EngineMatchTester,
    LevelResult,
    describe_scores,
)

from core.monitoring import Monitor


class FakeMatch(EngineMatch):
    """Plays no games: the network `params` (a number) gets `fake_results` at `level`.
    Records each play."""

    def __init__(self, levels=(4, 8), sims=1000, openings=3):  # pylint: disable=super-init-not-called
        self.engine = Edax
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
            we_black=games % 2 == 0,
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
    """Runs `tester` at `epoch`, as the training loop does, collecting what it logs in
    `logged` (one dict a call)."""
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
        activity_fn=lambda text: None,
    )


@pytest.mark.parametrize(
    "every, last, epochs",
    [
        (0, 0, []),
        (4, 0, [3, 7]),
        (5, 0, [4, 9]),
        (0, 3, [7, 8, 9]),
        # the last epoch of every, and of the last ones, plays once
        (4, 3, [3, 7, 8, 9]),
        (5, 1, [4, 9]),
    ],
)
def test_it_plays_after_every_n_epochs_and_each_of_the_last(every, last, epochs):
    tester = EngineMatchTester(FakeMatch(), num_epochs=10, every=every, last=last)
    assert [e for e in range(10) if tester.plays_at(e)] == epochs


def test_it_logs_each_level_at_the_epoch_of_the_network_that_played():
    match = FakeMatch()
    tester = EngineMatchTester(match, num_epochs=10, every=4, record=True)
    state = tester.init()
    logged = []

    # an epoch it doesn't play at
    new_state, metrics, episode = run(tester, 2, state, params=7, logged=logged)
    assert (new_state, metrics, episode, logged, match.plays) == (
        state,
        {},
        None,
        [],
        [],
    )

    _, metrics, episode = run(tester, 3, state, params=7, logged=logged)

    # it returns nothing for the loop to log itself: it's logged a level at a time
    assert (metrics, episode) == ({}, None)
    assert match.plays == [(7, 4, True), (7, 8, True)]
    four, eight, timing = logged
    results = fake_results(7, 4, 6)
    assert four["edax_level4_sims1000"] == pytest.approx(points(results).mean())
    assert four["edax_level4_sims1000_se"] == pytest.approx(
        points(results).std() / np.sqrt(6)
    )
    # 7 + 4 = 11 = 5 wins (mod 6), a draw and no losses
    assert (
        four["edax_level4_sims1000_win"],
        four["edax_level4_sims1000_draw"],
        four["edax_level4_sims1000_loss"],
    ) == pytest.approx((5 / 6, 1 / 6, 0))
    assert isinstance(four["edax_level4_sims1000_game"], Arrays)
    assert "edax_level8_sims1000" in eight
    # the time the games took, to tell training time from it
    assert list(timing) == ["edax_seconds"] and timing["edax_seconds"] >= 0


def test_it_records_games_only_when_asked():
    match = FakeMatch(levels=[8])
    tester = EngineMatchTester(match, num_epochs=1, last=1)
    logged = []
    run(tester, 0, tester.init(), params=0, logged=logged)
    assert match.plays == [(0, 8, False)]
    assert "edax_level8_sims1000_game" not in logged[0]


def test_it_appends_each_result_to_the_results_file(tmp_path):
    results = tmp_path / "vs_engine.jsonl"
    tester = EngineMatchTester(
        FakeMatch(), num_epochs=4, every=2, results=results, config={"dtype": "bf16"}
    )
    state = tester.init()
    for epoch in range(4):
        state, _, _ = run(tester, epoch, state, params=epoch)

    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert [(r["epoch"], r["level"]) for r in records] == [
        (1, 4),
        (1, 8),
        (3, 4),
        (3, 8),
    ]
    first = records[0]
    assert (first["engine"], first["sims"], first["dtype"], first["games"]) == (
        "edax",
        1000,
        "bf16",
        6,
    )
    assert first["score"] == pytest.approx(points(fake_results(1, 4, 6)).mean())


def last_mean(params, level, num_games):
    """The mean the networks `params` score at `level`, and its standard error: each game's
    points averaged over the networks first."""
    per_game = np.mean(
        [points(fake_results(p, level, num_games)) for p in params], axis=0
    )
    return per_game.mean(), per_game.std() / np.sqrt(num_games)


def test_it_logs_the_mean_of_the_last_networks_at_the_last_epoch():
    tester = EngineMatchTester(FakeMatch(), num_epochs=5, last=3)
    state = tester.init()
    logged = {}
    for epoch in range(5):
        calls = []
        state, _, _ = run(tester, epoch, state, params=epoch, logged=calls)
        logged[epoch] = {k: v for call in calls for k, v in call.items()}

    assert [e for e in logged if logged[e]] == [2, 3, 4]
    assert not any("_last3" in k for e in (2, 3) for k in logged[e])
    for level in (4, 8):
        mean, se = last_mean([2, 3, 4], level, 6)
        assert logged[4][f"edax_level{level}_sims1000_last3"] == pytest.approx(mean)
        assert logged[4][f"edax_level{level}_sims1000_last3_se"] == pytest.approx(se)


def test_a_resumed_run_takes_the_last_networks_results_from_its_saved_state():
    straight = EngineMatchTester(FakeMatch(), num_epochs=5, last=3)
    state = straight.init()
    logged = []
    for epoch in range(5):
        state, _, _ = run(straight, epoch, state, params=epoch, logged=logged)

    # stopped after epoch 3, with its state saved (to the host and back)
    stopped = EngineMatchTester(FakeMatch(), num_epochs=5, last=3)
    state = stopped.init()
    for epoch in range(4):
        state, _, _ = run(stopped, epoch, state, params=epoch)
    saved = jax.tree.map(np.asarray, state)
    resumed = EngineMatchTester(FakeMatch(), num_epochs=5, last=3)
    resumed_logged = []
    run(
        resumed,
        4,
        jax.tree.map(jax.numpy.asarray, saved),
        params=4,
        logged=resumed_logged,
    )

    # the last log: the time the games took, and the means
    def means(metrics):
        return {k: v for k, v in metrics.items() if "_last3" in k}

    assert means(resumed_logged[-1]) == means(logged[-1]) != {}


def test_last_is_at_most_the_run():
    with pytest.raises(ValueError, match="last"):
        EngineMatchTester(FakeMatch(), num_epochs=2, last=3)


def declared(declarations, name, field):
    """A field of what's declared about `name`, as seaglass reads it: from the first
    pattern, in order, that matches it and has the field."""
    return next(
        (
            d[field]
            for p, d in declarations.items()
            if fnmatch.fnmatchcase(name, p) and field in d
        ),
        None,
    )


def test_scores_are_declared_better_higher_and_headlines_but_not_whats_beside_them():
    monitor = Monitor("http://127.0.0.1:9")
    tester = EngineMatchTester(FakeMatch(levels=[8]), num_epochs=5, last=3)

    describe_scores(monitor, tester.score_names())

    decls = monitor._declared  # pylint: disable=protected-access
    for name in ("edax_level8_sims1000", "edax_level8_sims1000_last3"):
        assert declared(decls, name, "better") == "higher"
        assert declared(decls, name, "headline") is True
        assert "score" in (declared(decls, name, "description") or "")
    for name in (
        "edax_level8_sims1000_se",
        "edax_level8_sims1000_draw",
        "edax_level8_sims1000_last3_se",
    ):
        assert declared(decls, name, "better") is None
        assert declared(decls, name, "headline") is None
    assert declared(decls, "edax_level8_sims1000_loss", "better") == "lower"
    assert declared(decls, "edax_seconds", "unit") == "s"
