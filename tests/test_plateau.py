"""Tests of `Plateau`, the trigger: on made-up signals, every epoch's or every few epochs'."""

import pytest

from core.training.plateau import Plateau, PlateauProgress, improvement


def test_improvement_is_the_fitted_lines_rise_over_the_window():
    assert improvement(range(40), [1.0 - 0.001 * i for i in range(40)], 40) == (
        pytest.approx(-0.04)
    )
    assert improvement(range(10), [0.5] * 10, 10) == pytest.approx(0.0)
    # the line through noise, not its ends
    assert improvement(range(5), [1.0, 0.0, 1.0, 0.0, 1.0], 5) == pytest.approx(0.0)
    # by the epochs measured at, over the whole window
    assert improvement([10, 20, 30, 40], [0.0, 10.0, 20.0, 30.0], 40) == (
        pytest.approx(40.0)
    )


def fires(plateau, values):
    """The epochs (counting from 0) at which `plateau` fires, following `values`, one an epoch,
    None where unmeasured."""
    progress, fired = PlateauProgress(), []
    for epoch, value in enumerate(values):
        step = plateau.after_epoch(progress, value)
        progress = step.progress
        if step.plateaued:
            fired.append(epoch)
    return fired


def test_it_fires_when_the_signal_stops_falling_and_starts_again():
    # falling 0.1 an epoch for 6 epochs (0.4 over 4), then flat
    values = [1.0 - 0.1 * min(e, 6) for e in range(20)]

    # the first window whose line falls less than 0.1 is epochs 6-9 (epochs 5-8's falls 0.12);
    # then the window starts again, and fires as soon as it's full
    assert fires(Plateau(window=4, threshold=0.1, min_epochs=4), values) == [9, 13, 17]


def test_it_waits_for_min_epochs():
    # flat from the start, its window full at epoch 1: it waits for epoch 4, the fifth
    assert fires(Plateau(window=2, threshold=0.1, min_epochs=5), [1.0] * 6) == [4]


def test_the_window_only_holds_epochs_since_it_fired():
    # flat, then falling fast right after it fires
    values = [1.0, 1.0, 1.0, 0.5, 0.0, -0.5, -0.5, -0.5]

    # epochs 3-5 fall, so it next fires at the first flat window, epochs 5-7
    assert fires(Plateau(window=3, threshold=0.1, min_epochs=1), values) == [2, 7]


def test_a_rising_signal_improves_with_maximize():
    # an Elo rising 10 an epoch for 8 epochs, then flat
    values = [10.0 * min(e, 8) for e in range(16)]

    # rising is getting worse, unless rising is better
    assert fires(Plateau(window=4, threshold=5.0, min_epochs=1), values) == [
        3,
        7,
        11,
        15,
    ]
    # epochs 7-10 still rise 12 over 4 epochs; epochs 8-11 don't
    assert fires(
        Plateau(maximize=True, window=4, threshold=5.0, min_epochs=1), values
    ) == [11, 15]


def test_a_signal_measured_every_few_epochs():
    """As games against anchors are: a value every third epoch, None in between."""
    plateau = Plateau(maximize=True, window=6, threshold=1.0, min_epochs=6)
    # rising 3 a measure (1 an epoch) for 4 measures, then flat
    values = [3.0 * min(e // 3, 4) if e % 3 == 2 else None for e in range(30)]
    progress, fitted = PlateauProgress(), []
    for value in values:
        step = plateau.after_epoch(progress, value)
        progress = step.progress
        fitted.append(step.improvement)

    # nothing until the window's 6 epochs hold two measures, at epochs 2 and 5
    assert fitted[:5] == [None] * 5
    assert fitted[5] == pytest.approx(6.0)
    # the window drops each measure 6 epochs after it's taken
    assert fires(plateau, values) == [17, 23, 29]


def test_unmeasured_epochs_still_count():
    plateau = Plateau(window=4, threshold=0.1, min_epochs=1)
    # one measure in the window is no line
    assert fires(plateau, [None, None, None, 1.0, None, None, None]) == []
    assert fires(plateau, [1.0, None, None, 1.0]) == [3]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window": 1}, "window must be at least 2"),
        ({"min_epochs": 0}, "min_epochs must be positive"),
    ],
)
def test_bad_settings_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Plateau(**kwargs)
