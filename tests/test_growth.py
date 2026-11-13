"""Tests of `Growth`'s rule for handing self-play to the next network (pure, no training)."""

import pytest

from core.training.growth import Growth, GrowthProgress, Learner


class FakeNet:
    """Stands in for a network: the rule never looks at it."""


def learner(name):
    return Learner(name=name, nn=FakeNet(), evaluator=None, evaluator_test=None)  # pyright: ignore[reportArgumentType]


def growth(names=("b", "c"), **kwargs):
    return Growth(first="a", learners=[learner(n) for n in names], **kwargs)


def run(rule, values_by_epoch, rule_done=()):
    """Applies `rule` epoch by epoch, `values_by_epoch[e]` the metric of each learner that
    trains, by index (learners that don't train are left out); returns the checks."""
    progress = rule.start()
    checks = []
    for epoch, values in enumerate(values_by_epoch):
        training = rule.training(progress)
        check = rule.after_epoch(
            progress,
            epoch + 1,
            {i: v for i, v in values.items() if i in training},
            rule_done=epoch in rule_done,
        )
        checks.append(check)
        progress = check.progress
    return checks


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"names": ()}, "at least one"),
        ({"names": ("a",)}, "distinct"),
        ({"names": ("b", "b")}, "distinct"),
        ({"names": ("b/c",)}, "can't hold"),
        ({"window": 0}, "positive"),
        ({"switch_at": (5,)}, "one increasing"),
        ({"switch_at": (5, 5)}, "one increasing"),
        ({"switch_at": (0, 5)}, "one increasing"),
        ({"switch_at": (2, 5), "manual": True}, "pass one"),
        ({"margin": 1.0}, "fraction"),
        ({"margin": -0.1}, "fraction"),
        ({"metric": "loss", "guard": "loss"}, "another metric"),
    ],
)
def test_bad_settings_are_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        growth(**kwargs)


def test_two_networks_train_at_a_time():
    rule = growth()
    assert rule.names == ("a", "b", "c")
    assert rule.training(GrowthProgress(0, (0, 0, 0), ((), (), ()))) == (0, 1)
    assert rule.training(GrowthProgress(1, (0, 0, 0), ((), (), ()))) == (1, 2)
    # the last plays alone
    assert rule.training(GrowthProgress(2, (0, 0, 0), ((), (), ()))) == (2,)


def test_the_next_takes_over_once_its_mean_over_the_window_is_no_worse():
    rule = growth(window=3, min_epochs=2)
    # b: 2, 2, 2, then 0.5: its 3-epoch mean reaches a's 1.0 after the 5th epoch (2, 0.5, 0.5)
    values = [{0: 1.0, 1: 2.0}] * 3 + [{0: 1.0, 1: 0.5}] * 2
    checks = run(rule, values)
    assert [c.switched for c in checks] == [False] * 4 + [True]
    # no gap until both have a full window; then b's mean minus a's
    assert [c.gap for c in checks[:2]] == [None, None]
    assert checks[2].gap == pytest.approx(1.0)
    assert checks[3].gap == pytest.approx(0.5)
    assert checks[4].gap == pytest.approx(0.0)
    after = checks[-1].progress
    assert after.playing == 1
    assert after.trained == (5, 5, 0)
    assert after.values[1] == (2.0, 0.5, 0.5)


def test_the_next_waits_its_minimum_epochs():
    rule = growth(window=1, min_epochs=3)
    checks = run(rule, [{0: 1.0, 1: 0.0}] * 4)
    assert [c.switched for c in checks] == [False, False, True, False]


def test_after_a_switch_the_one_after_starts_and_the_last_plays_alone():
    rule = growth(window=1, min_epochs=1)
    checks = run(rule, [{0: 1.0, 1: 0.0}, {1: 1.0, 2: 0.0}, {2: 0.0}, {2: 0.0}])
    assert [c.progress.playing for c in checks] == [1, 2, 2, 2]
    # c started with b's switch: it has trained the epochs since
    assert checks[-1].progress.trained == (1, 2, 3)
    assert checks[-1].gap is None


def test_fixed_epochs_switch_whatever_the_metric():
    rule = growth(switch_at=(2, 5))
    checks = run(rule, [{}] * 6)
    assert [c.switched for c in checks] == [False, True, False, False, True, False]


def test_the_playing_networks_rule_being_done_hands_over():
    rule = growth(names=("b",), window=1, min_epochs=5)
    # b is worse throughout; a's learning rate rule ends after epoch 2
    checks = run(rule, [{0: 0.0, 1: 1.0}] * 4, rule_done={2})
    assert [c.switched for c in checks] == [False, False, True, False]
    # once the last plays, its rule being done hands over to nothing (the run stops)
    assert not run(rule, [{0: 0.0, 1: 1.0}, {1: 1.0}], rule_done={0, 1})[1].switched


def test_a_manual_rule_reports_the_catch_up_and_never_switches():
    rule = growth(window=1, min_epochs=2, manual=True)
    # b is better from the first epoch, and a's learning rate rule is done after the third
    checks = run(rule, [{0: 1.0, 1: 0.0}] * 4, rule_done={2})
    assert [c.switched for c in checks] == [False] * 4
    # caught up once b has trained its minimum epochs
    assert [c.caught_up for c in checks] == [False, True, True, True]
    assert all(c.progress.playing == 0 for c in checks)


def test_caught_up_needs_a_full_window_and_no_worse_mean():
    rule = growth(window=2, min_epochs=1)
    assert not rule.caught_up(GrowthProgress(0, (2, 2, 0), ((1.0,), (0.0,), ())))
    assert rule.caught_up(GrowthProgress(0, (2, 2, 0), ((1.0, 1.0), (1.0, 1.0), ())))
    assert not rule.caught_up(
        GrowthProgress(0, (2, 2, 0), ((1.0, 1.0), (1.0, 1.5), ()))
    )
    # the last plays: nothing to catch up
    assert rule.gap(GrowthProgress(2, (2, 2, 2), ((1.0,), (1.0,), (1.0, 1.0)))) is None


def test_take_over_hands_self_play_to_the_next_until_the_last():
    rule = growth()
    progress = GrowthProgress(0, (3, 3, 0), ((1.0,), (0.5,), ()))
    taken = rule.take_over(progress)
    assert taken == GrowthProgress(1, (3, 3, 0), ((1.0,), (0.5,), ()))
    assert rule.training(taken) == (1, 2)
    with pytest.raises(ValueError, match="c is the last network"):
        rule.take_over(rule.take_over(taken))


def test_with_a_margin_the_next_must_pull_away():
    rule = growth(window=1, min_epochs=1, margin=0.01)
    # level, then 0.5% lower, then 1% lower: only a lead of 1% of the playing one's switches
    checks = run(rule, [{0: 1.0, 1: 1.0}, {0: 1.0, 1: 0.995}, {0: 1.0, 1: 0.99}])
    assert [c.caught_up for c in checks] == [False, False, True]
    assert [c.switched for c in checks] == [False, False, True]


def test_with_a_guard_the_next_must_also_be_no_worse_on_it():
    rule = growth(window=1, min_epochs=1, metric="loss", guard="engine_value_mse")

    def epochs(guards):
        progress, checks = rule.start(), []
        for guard in guards:
            check = rule.after_epoch(
                progress, len(checks) + 1, {0: 1.0, 1: 0.5}, guard_values=guard
            )
            checks.append(check)
            progress = check.progress
        return checks

    # ahead on the loss, but worse on the guard: no switch, until it's level on the guard too
    checks = epochs([{0: 0.4, 1: 0.5}, {0: 0.4, 1: 0.4}])
    assert [c.switched for c in checks] == [False, True]
    assert [c.guard_gap for c in checks] == [pytest.approx(0.1), pytest.approx(0.0)]
    assert checks[0].progress.guard_values[:2] == ((0.4,), (0.5,))
