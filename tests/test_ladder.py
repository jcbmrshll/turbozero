import jax
import jax.numpy as jnp
import numpy as np

from core.testing.ladder import LadderTester, LadderTestState, Rung

MAX_STEPS = 10


def make_ladder(scripted, **kwargs):
    # the agent (first_legal) always beats `resign`, and scores about half against itself
    return LadderTester(
        num_episodes=8,
        rungs=[
            Rung("easy", scripted.resign),
            Rung("easy2", scripted.resign),
            Rung("mirror", scripted.first_legal),
            Rung("easy3", scripted.resign),
        ],
        **kwargs,
    )


def run(tester, ttt, scripted, state, epoch_num=0, seed=0, **kwargs):
    return tester.run(
        key=jax.random.PRNGKey(seed),
        epoch_num=epoch_num,
        max_steps=MAX_STEPS,
        env_step_fn=ttt.step_fn,
        env_init_fn=ttt.init_fn,
        evaluator=scripted.first_legal,
        state=state,
        params={"w": jnp.ones(3)},
        **kwargs,
    )


def test_ladder_climbs_past_beaten_rungs_and_stops_at_the_first_it_cannot_beat(
    ttt, scripted
):
    tester = make_ladder(scripted)
    state = tester.init()

    state, metrics, episode = run(tester, ttt, scripted, state)

    assert isinstance(state, LadderTestState)
    np.testing.assert_array_equal(state.rung, 2)
    assert metrics["ladder_rung"] == 2
    assert metrics["ladder_easy_score"] == metrics["ladder_easy2_score"] == 1.0
    assert metrics["ladder_mirror_score"] < 0.55
    # the rung above the one it failed isn't played
    assert "ladder_easy3_score" not in metrics
    assert episode is None


def test_ladder_resumes_from_its_rung_and_never_revisits_beaten_ones(ttt, scripted):
    tester = make_ladder(scripted)
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


def test_ladder_without_climb_moves_up_at_most_one_rung_per_test(ttt, scripted):
    tester = make_ladder(scripted, climb=False)
    state = tester.init()

    state, metrics, _ = run(tester, ttt, scripted, state)

    np.testing.assert_array_equal(state.rung, 1)
    assert "ladder_easy2_score" not in metrics


def test_ladder_skips_epochs_between_tests(ttt, scripted):
    tester = make_ladder(scripted, epochs_per_test=2)
    state = tester.init()

    new_state, metrics, episode = run(tester, ttt, scripted, state, epoch_num=1)

    assert new_state is state
    assert metrics == {}
    assert episode is None


def test_ladder_reports_each_rung_as_it_goes(ttt, scripted):
    tester = make_ladder(scripted)
    state = tester.init()
    events = []

    state, metrics, _ = run(
        tester,
        ttt,
        scripted,
        state,
        log_fn=lambda m: events.append(("log", sorted(m))),
        activity_fn=lambda text: events.append(("activity", text)),
    )

    # each rung is announced before it's played and logged right after
    assert events == [
        ("activity", "ladder: playing easy (rung 1/4, 8 games)"),
        (
            "log",
            [f"ladder_easy_{k}" for k in ("loss_rate", "score", "seconds", "win_rate")],
        ),
        ("activity", "ladder: playing easy2 (rung 2/4, 8 games)"),
        (
            "log",
            [
                f"ladder_easy2_{k}"
                for k in ("loss_rate", "score", "seconds", "win_rate")
            ],
        ),
        ("activity", "ladder: playing mirror (rung 3/4, 8 games)"),
        (
            "log",
            [
                f"ladder_mirror_{k}"
                for k in ("loss_rate", "score", "seconds", "win_rate")
            ],
        ),
    ]
    # what was logged isn't returned again
    assert metrics == {"ladder_rung": 2}
