"""train.py's command line, up to the trainer it builds, with a fake trainer."""

import sys

import equinox as eqx
import jax
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


# and a network without state: no BatchNorm, nothing but its parameters
@pytest.mark.parametrize("net", ["resnet:1x4", "resnet:1x4,norm=channel,layout=pre"])
def test_init_weights_start_from_a_checkpoint(monkeypatch, tmp_path, net):
    game = load_example("othello", "game")[0]
    network, nn_state = game.make_network(net, seed=3)
    path = tmp_path / "pretrained.eqx"
    eqx.tree_serialise_leaves(
        path, (eqx.filter(network, eqx.is_inexact_array), nn_state)
    )
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--net", net, "--init-weights", str(path))
    for a, b in zip(
        jax.tree.leaves(eqx.filter(built.value.kwargs["nn"], eqx.is_inexact_array)),
        jax.tree.leaves(eqx.filter(network, eqx.is_inexact_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


def test_seed_buffer_is_for_runs_that_start_empty(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--seed-buffer", "corpus", "--resume", "--ckpt-dir", "x")


def test_self_play_keeps_off_excluded_openings(monkeypatch, tmp_path):
    xot = load_example("othello", "xot")[0]
    openings = xot.load_xot()
    excluded = tmp_path / "excluded.npy"
    # all but opening 5
    np.save(excluded, np.delete(np.arange(len(openings)), 5))
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--exclude-openings", str(excluded))
    init_fn = built.value.kwargs["env_init_fn"]
    expected = xot.make_xot_init_fn(openings[5:6], 0.0)(jax.random.PRNGKey(0))[0]
    for seed in range(3):
        state, _ = init_fn(jax.random.PRNGKey(seed))
        np.testing.assert_array_equal(state.observation, expected.observation)


def test_lr_warmup_starts_the_learning_rate_at_zero(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--epochs", "4", "--lr-warmup", "1", "--lr-final", "1e-3")
    optimizer = built.value.kwargs["optimizer"]
    params = {"w": jax.numpy.zeros(2)}
    state = optimizer.init(params)
    rates = []
    for _ in range(257):
        _, state = optimizer.update({"w": jax.numpy.ones(2)}, state, params)
        rates.append(float(state.hyperparams["learning_rate"]))
    # one epoch of 256 steps from 0 to 1e-3
    assert rates[0] == 0
    assert rates[128] == pytest.approx(5e-4, rel=1e-6)
    assert rates[256] == pytest.approx(1e-3, rel=1e-6)
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--epochs", "4", "--lr-warmup", "4")


def test_anchor_blends_a_frozen_networks_outputs_into_the_targets(
    monkeypatch, tmp_path
):
    game = load_example("othello", "game")[0]
    network, nn_state = game.make_network("resnet:1x4", seed=5)
    path = tmp_path / "anchor.eqx"
    eqx.tree_serialise_leaves(
        path, (eqx.filter(network, eqx.is_inexact_array), nn_state)
    )
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--anchor", str(path), "--anchor-policy", "0.25")
    loss_fn = built.value.kwargs["loss_fn"]
    assert loss_fn.func.__name__ == "anchored_loss_fn"
    assert loss_fn.keywords["policy_weight"] == 0.25
    assert loss_fn.keywords["value_weight"] == 0.5
    reference = loss_fn.keywords["reference"]
    assert str(path) in str(reference)
    # the reference's policy is over the legal actions only
    obs = jax.numpy.zeros((3, 8, 8, 2), bool)
    legal = jax.numpy.zeros((3, 65), bool).at[:, :2].set(True)
    policy, value = reference(obs, legal)
    np.testing.assert_allclose(policy.sum(-1), 1, rtol=1e-6)
    assert (policy[:, 2:] == 0).all() and value.shape == (3,)
    for bad in (["--anchor-value", "1.5"], ["--value-target-q", "0.5"]):
        with pytest.raises(SystemExit):
            run_main(monkeypatch, "--anchor", str(path), *bad)


def test_freeze_batchnorm_trains_with_batchnorm_in_inference_mode(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--freeze-batchnorm")
    assert built.value.kwargs["nn"].stem_norm.inference
    for bad in (["--train-dtype", "float32"], ["--remat"]):
        with pytest.raises(SystemExit):
            run_main(monkeypatch, "--freeze-batchnorm", *bad)


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
