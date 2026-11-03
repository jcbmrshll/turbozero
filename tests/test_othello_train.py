"""train.py's command line, up to the trainer it builds, with a fake trainer."""

import sys

import equinox as eqx
import jax
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


def test_init_weights_start_from_a_checkpoint(monkeypatch, tmp_path):
    game = load_example("othello", "game")[0]
    network, nn_state = game.make_network("resnet:1x4", seed=3)
    path = tmp_path / "pretrained.eqx"
    eqx.tree_serialise_leaves(
        path, (eqx.filter(network, eqx.is_inexact_array), nn_state)
    )
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--init-weights", str(path))
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
