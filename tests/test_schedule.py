import pytest

from core.evaluators.random_evaluator import RandomEvaluator
from core.training.schedule import EvaluatorSchedule, Schedule, parse_schedule


def test_schedule_picks_the_latest_stage_started():
    a, b, c = RandomEvaluator(), RandomEvaluator(), RandomEvaluator()
    schedule = EvaluatorSchedule([(0, a), (5, b), (8, c)])

    assert [schedule.at(epoch) for epoch in (0, 4, 5, 7, 8, 100)] == [a, a, b, b, c, c]


def test_value_schedule():
    schedule = Schedule(parse_schedule("0:1000,20:3000,35:6000"))

    assert [schedule.at(epoch) for epoch in (0, 19, 20, 34, 35, 1000)] == [
        1000,
        1000,
        3000,
        3000,
        6000,
        6000,
    ]
    assert schedule.get_config() == [
        {"epoch": 0, "value": 1000},
        {"epoch": 20, "value": 3000},
        {"epoch": 35, "value": 6000},
    ]


@pytest.mark.parametrize(
    "epochs, match",
    [([], "start at epoch 0"), ([1, 5], "start at epoch 0"), ([0, 5, 5], "increase")],
)
def test_schedule_checks_its_epochs(epochs, match):
    with pytest.raises(ValueError, match=match):
        EvaluatorSchedule([(epoch, RandomEvaluator()) for epoch in epochs])


def test_parse_schedule():
    assert parse_schedule("0:32,50:64,150:128") == [(0, 32), (50, 64), (150, 128)]
    assert parse_schedule("0:64") == [(0, 64)]
    # OLIVAW's
    assert parse_schedule("0:100,4:200,11:400") == [(0, 100), (4, 200), (11, 400)]


@pytest.mark.parametrize(
    "text, match",
    [
        ("0:32,50", "epoch:value pairs"),
        ("0:32;50:64", "epoch:value pairs"),
        ("0:3.5", "epoch:value pairs"),
        ("10:32", "start at epoch 0"),
        ("0:32,50:64,40:128", "increase"),
    ],
)
def test_parse_schedule_rejects_bad_schedules(text, match):
    with pytest.raises(ValueError, match=match):
        parse_schedule(text)
