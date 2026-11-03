"""train.py's command line, up to the trainer it builds, with a fake trainer."""

import sys

import pytest
from example_modules import load_example

train = load_example("othello", "train")[0]


class BuiltTrainer(Exception):
    """Stops main at its Trainer, holding the arguments."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


def fake_trainer(**kwargs):
    raise BuiltTrainer(**kwargs)


def run_main(monkeypatch, *args):
    monkeypatch.setattr(train, "Trainer", fake_trainer)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--net", "resnet:1x4", "--engine-eval-every", "0", *args],
    )
    train.main()


def test_root_noise_options_reach_self_play_only(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--dirichlet-alpha", "1.0", "--dirichlet-epsilon", "0.1")
    evaluator = built.value.kwargs["evaluator"].at(0)
    assert evaluator.dirichlet_alpha == 1.0
    assert evaluator.dirichlet_epsilon == 0.1
    assert built.value.kwargs["evaluator_test"].dirichlet_alpha == 0.3


@pytest.mark.parametrize(
    "option", [("--dirichlet-alpha", "0"), ("--dirichlet-epsilon", "1.5")]
)
def test_root_noise_options_out_of_range_are_refused(monkeypatch, option):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *option)


@pytest.mark.parametrize(
    "flags, first_epoch",
    [
        ((), None),
        (("--save-positions-from", "0"), 0),
        (("--save-positions-from", "30"), 30),
        (("--save-positions-from", "0.75", "--epochs", "10"), 8),
    ],
)
def test_keeping_positions_is_off_unless_asked(monkeypatch, flags, first_epoch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, *flags)
    assert built.value.kwargs["save_positions_from"] == first_epoch


@pytest.mark.parametrize("value", ["-1", "1.5", "soon"])
def test_keeping_positions_from_nonsense_is_refused(monkeypatch, value):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--save-positions-from", value)
