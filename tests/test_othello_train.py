"""train.py's command line, up to the trainer it builds, with a fake trainer."""

import sys

import jax.numpy as jnp
import numpy as np
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
    assert rule.rates == (3e-3, 1e-4, 1e-5)
    assert (rule.window, rule.threshold, rule.min_epochs, rule.stop) == (
        20,
        0.004,
        40,
        True,
    )
    state = built.value.kwargs["optimizer"].init({"w": jnp.zeros(2)})
    assert float(state.hyperparams["learning_rate"]) == pytest.approx(3e-3)


def test_without_lr_plateau_the_rule_is_off(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch)
    assert built.value.kwargs["lr_plateau"] is None


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


def test_lr_plateau_can_watch_the_engine_sets_value_error(monkeypatch, tmp_path):
    from test_engine_set import tiny_set

    tiny_set(np.random.default_rng(0)).save(tmp_path / "set.npz")
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--lr-plateau",
            "--lr-plateau-metric",
            "engine_value_mse",
            "--engine-eval-every",
            "1",
            "--engine-set",
            str(tmp_path / "set.npz"),
        )
    assert built.value.kwargs["lr_plateau_metric"] == "engine_value_mse"


@pytest.mark.parametrize(
    "options",
    [
        # without the rule
        ("--lr-plateau-metric", "engine_value_mse", "--engine-eval-every", "1"),
        # without the set scored every epoch
        (
            "--lr-plateau",
            "--lr-plateau-metric",
            "engine_value_mse",
            "--engine-eval-every",
            "2",
        ),
        ("--lr-plateau", "--lr-plateau-metric", "engine_value_mse"),
    ],
)
def test_lr_plateau_metric_needs_the_rule_and_the_set_every_epoch(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)


def test_grow_builds_a_network_to_grow_into_with_its_own_search_and_tests(
    monkeypatch, tmp_path
):
    from test_engine_set import tiny_set

    tiny_set(np.random.default_rng(0)).save(tmp_path / "set.npz")
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--grow",
            "resnet:2x8",
            "--grow-window",
            "5",
            "--sims",
            "16",
            "--engine-eval-every",
            "1",
            "--engine-set",
            str(tmp_path / "set.npz"),
            "--ckpt-dir",
            str(tmp_path / "run"),
        )
    growth = built.value.kwargs["growth"]
    assert growth.names == ("1x4", "2x8")
    assert (growth.metric, growth.window, growth.min_epochs, growth.switch_at) == (
        "engine_value_mse",
        5,
        10,
        (),
    )
    (grown,) = growth.learners
    assert grown.nn is not built.value.kwargs["nn"]
    assert grown.schedule.at(0).num_iterations == 16
    assert grown.evaluator_test is not built.value.kwargs["evaluator_test"]
    # each network has its own engine-set tester, of its own network
    (own,), (theirs,) = built.value.kwargs["testers"], grown.testers
    assert own is not theirs and theirs.name == own.name == "engine"


def test_grow_can_switch_at_fixed_epochs_without_the_engine_set(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--grow", "resnet:2x8", "--grow-switch-at", "80")
    assert built.value.kwargs["growth"].switch_at == (80,)
    assert built.value.kwargs["growth"].learners[0].testers == []


@pytest.mark.parametrize(
    "options",
    [
        # growth's settings without --grow
        ("--grow-window", "5"),
        ("--grow-switch-at", "80"),
        # the rule without the engine set every epoch
        ("--grow", "resnet:2x8"),
        # one switch epoch per network
        ("--grow", "resnet:2x8", "--grow-switch-at", "80,90"),
        # the same size twice
        ("--grow", "resnet:1x4,heads=pooled", "--grow-switch-at", "80"),
    ],
)
def test_bad_grow_options_are_refused(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)
