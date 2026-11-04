"""train.py's command line, up to the trainer it builds, with a fake trainer."""

import sys

import jax.numpy as jnp
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


def test_lr_plateau_steps_from_lr_to_its_rates(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--lr",
            "3e-3",
            "--lr-plateau",
            "--lr-plateau-window",
            "20",
            "--lr-plateau-stop",
        )
    rule = built.value.kwargs["lr_plateau"]
    assert rule.rates == (3e-3, 1e-4, 1e-5) and rule.stop
    plateau = rule.plateau
    assert (plateau.signal, plateau.maximize) == ("loss", False)
    assert (plateau.window, plateau.threshold, plateau.min_epochs) == (20, 0.004, 40)
    state = built.value.kwargs["optimizer"].init({"w": jnp.zeros(2)})
    assert float(state.hyperparams["learning_rate"]) == pytest.approx(3e-3)


def test_without_lr_plateau_the_rule_is_off(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch)
    assert built.value.kwargs["lr_plateau"] is None


class FakeAnchorTester:
    def __init__(self, game, network, anchors, **kwargs):
        self.game, self.anchors, self.kwargs = game, anchors, kwargs


@pytest.fixture
def anchor_args(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "AnchorTester", FakeAnchorTester)
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.eqx"
        path.write_bytes(b"")
        paths.append(str(path))
    return [
        "--anchor", "e10", paths[0], "resnet:1x4", "-50",
        "--anchor", "e90", paths[1], "resnet:2x8,heads=pooled", "310.5",
    ]  # fmt: skip


def test_anchors_play_every_n_epochs(monkeypatch, anchor_args):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch, *anchor_args, "--anchor-every", "5", "--anchor-sims", "64"
        )
    (tester,) = built.value.kwargs["testers"]
    assert [(a.name, a.net, a.elo) for a in tester.anchors] == [
        ("e10", "resnet:1x4", -50.0),
        ("e90", "resnet:2x8,heads=pooled", 310.5),
    ]
    assert tester.game.name == "Othello"
    assert (tester.kwargs["every"], tester.kwargs["games"]) == (5, 400)
    assert (tester.kwargs["sims"], tester.kwargs["dtype"]) == (64, "bfloat16")


def test_lr_plateau_can_follow_the_anchors_elo(monkeypatch, anchor_args):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            *anchor_args,
            "--lr-plateau",
            "--lr-plateau-signal",
            "anchor-elo",
        )
    plateau = built.value.kwargs["lr_plateau"].plateau
    assert (plateau.signal, plateau.maximize) == ("anchor_elo", True)
    assert (plateau.window, plateau.threshold, plateau.min_epochs) == (40, 20.0, 40)


def test_grad_noise_reaches_the_trainer(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--grad-noise")
    assert built.value.kwargs["grad_noise"]
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch)
    assert not built.value.kwargs["grad_noise"]


@pytest.mark.parametrize(
    "options",
    [
        (
            "--lr-plateau",
            "--lr-plateau-signal",
            "anchor-elo",
            "--lr-plateau-window",
            "15",
        ),
        ("--anchor", "x", "a.eqx", "resnet:1x4", "high"),
        ("--anchor", "x", "missing.eqx", "resnet:1x4", "0"),
    ],
)
def test_bad_anchor_options_are_refused(monkeypatch, anchor_args, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *anchor_args, *options)


@pytest.mark.parametrize(
    "options",
    [
        ("--lr-plateau", "--lr-plateau-signal", "anchor-elo"),
        # run_main scores no engine-labelled set
        ("--lr-plateau", "--lr-plateau-signal", "engine-value"),
    ],
)
def test_a_signal_needs_its_measure(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)


@pytest.mark.parametrize(
    "options",
    [
        ("--lr-plateau", "1e-4", "--lr-drops", "0.5:1e-4"),
        ("--lr-plateau", "--lr-final", "1e-4"),
        ("--lr", "1e-4", "--lr-plateau", "1e-4,1e-5"),
        ("--lr-plateau", "1e-5,1e-4"),
        ("--lr-plateau", "--lr-plateau-window", "1"),
    ],
)
def test_bad_lr_plateau_options_are_refused(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)
