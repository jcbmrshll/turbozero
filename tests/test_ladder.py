"""The ladder tester: climbing rungs of two-player baselines. Its bookkeeping is tested with rungs
whose results are made up (see `stub_rungs`); one test plays real games."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from seaglass import Arrays

from core.monitoring.episodes import episode_type
from core.testing.ladder import LadderTester, LadderTestState, Rung

MAX_STEPS = 10


def run(tester, ttt, scripted, state, epoch_num=0, **kwargs):
    return tester.run(
        key=jax.random.PRNGKey(0),
        epoch_num=epoch_num,
        max_steps=MAX_STEPS,
        env_step_fn=ttt.step_fn,
        env_init_fn=ttt.init_fn,
        evaluator=scripted.first_legal,
        state=state,
        params={"w": jnp.ones(3)},
        **kwargs,
    )


def stub_rungs(monkeypatch, scripted, results, **kwargs):
    """A ladder of rungs named after `results`' keys, in order, whose games aren't played: the
    agent's (win rate, loss rate) against each is the rung's value."""
    tester = LadderTester(
        num_episodes=8,
        rungs=[Rung(name, scripted.first_legal) for name in results],
        **kwargs,
    )
    for baseline, (win_rate, loss_rate) in zip(
        tester.baselines, results.values(), strict=True
    ):

        def test(*args, name=baseline.name, win=win_rate, loss=loss_rate):  # pylint: disable=unused-argument
            rates = {f"{name}_win_rate": win, f"{name}_loss_rate": loss}
            # (the ladder keeps no state of its rungs')
            return (
                None,
                {k: jnp.array([v]) for k, v in rates.items()},
                None,
                None,
            )

        monkeypatch.setattr(baseline, "test", test)
    return tester


# beaten, beaten, not beaten, beaten
RESULTS = {
    "easy": (1.0, 0.0),
    "easy2": (1.0, 0.0),
    "hard": (0.0, 1.0),
    "easy3": (1.0, 0.0),
}


def test_ladder_climbs_past_beaten_rungs_and_stops_at_the_first_it_cannot_beat(
    ttt, scripted
):
    # real games: the agent (first_legal) always beats `resign`, and scores about half against
    # itself, moving first in exactly half the games
    tester = LadderTester(
        num_episodes=8,
        rungs=[
            Rung("easy", scripted.resign),
            Rung("mirror", scripted.first_legal),
            Rung("easy2", scripted.resign),
        ],
    )

    state, metrics, episode = run(tester, ttt, scripted, tester.init())

    assert isinstance(state, LadderTestState)
    np.testing.assert_array_equal(state.rung, 1)
    assert metrics["ladder_rung"] == 1
    assert metrics["ladder_easy_score"] == 1.0
    assert metrics["ladder_mirror_score"] < 0.55
    # the rung above the one it failed isn't played
    assert "ladder_easy2_score" not in metrics
    assert episode is None


@pytest.mark.parametrize(
    "win_rate, loss_rate, promoted",
    [
        # a draw counts as half a win: (1 + 0.3 - 0.1) / 2 = 0.6
        (0.3, 0.1, True),
        # all draws: 0.5
        (0.0, 0.0, False),
    ],
)
def test_ladder_counts_draws_as_half_a_win(
    ttt, scripted, monkeypatch, win_rate, loss_rate, promoted
):
    tester = stub_rungs(monkeypatch, scripted, {"rival": (win_rate, loss_rate)})

    state, metrics, _ = run(tester, ttt, scripted, tester.init())

    assert metrics["ladder_rival_score"] == pytest.approx(
        (1 + win_rate - loss_rate) / 2
    )
    np.testing.assert_array_equal(state.rung, int(promoted))


def test_ladder_resumes_from_its_rung_and_never_revisits_beaten_ones(
    ttt, scripted, monkeypatch
):
    tester = stub_rungs(monkeypatch, scripted, RESULTS)
    state = LadderTestState(rung=jnp.array(3, dtype=jnp.int32))

    state, metrics, _ = run(tester, ttt, scripted, state)

    np.testing.assert_array_equal(state.rung, 4)
    assert set(metrics) == {
        "ladder_easy3_score",
        "ladder_easy3_win_rate",
        "ladder_easy3_loss_rate",
        "ladder_easy3_seconds",
        "ladder_rung",
    }

    # at the top there is nothing left to play
    state, metrics, _ = run(tester, ttt, scripted, state)
    np.testing.assert_array_equal(state.rung, 4)
    assert metrics == {"ladder_rung": 4}


def test_ladder_without_climb_moves_up_at_most_one_rung_per_test(
    ttt, scripted, monkeypatch
):
    tester = stub_rungs(monkeypatch, scripted, RESULTS, climb=False)

    state, metrics, _ = run(tester, ttt, scripted, tester.init())

    np.testing.assert_array_equal(state.rung, 1)
    assert "ladder_easy2_score" not in metrics


def test_ladder_episode_is_named_for_the_last_rung_played(ttt, scripted, monkeypatch):
    # it beats the first two rungs and stops at "hard"
    tester = stub_rungs(
        monkeypatch,
        scripted,
        RESULTS,
        episode_fn=lambda frames, p_ids, env_step_fn: Arrays(
            episode_type("tic_tac_toe"), attrs={"seed": 0}
        ),
    )

    _, _, episode = run(tester, ttt, scripted, tester.init())

    assert episode.attrs == {"seed": 0, "opponent": "hard"}


def test_ladder_skips_epochs_between_tests(ttt, scripted, monkeypatch):
    tester = stub_rungs(monkeypatch, scripted, RESULTS, epochs_per_test=2)
    state = tester.init()

    new_state, metrics, episode = run(tester, ttt, scripted, state, epoch_num=1)

    assert new_state is state
    assert metrics == {}
    assert episode is None


def test_ladder_reports_each_rung_as_it_goes(ttt, scripted, monkeypatch):
    tester = stub_rungs(monkeypatch, scripted, RESULTS)
    events = []

    _, metrics, _ = run(
        tester,
        ttt,
        scripted,
        tester.init(),
        log_fn=lambda m: events.append(("log", sorted(m))),
        activity_fn=lambda text: events.append(("activity", text)),
    )

    # each rung is announced before it's played and logged right after
    expected = []
    for rung, name in enumerate(["easy", "easy2", "hard"], start=1):
        expected += [
            ("activity", f"ladder: playing {name} (rung {rung}/4, 8 games)"),
            (
                "log",
                [
                    f"ladder_{name}_{k}"
                    for k in ("loss_rate", "score", "seconds", "win_rate")
                ],
            ),
        ]
    assert events == expected
    # what was logged isn't returned again
    assert metrics == {"ladder_rung": 2}
