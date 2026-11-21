"""train.py's command line, up to the trainer it builds, with a fake trainer."""

import math
import sys
from types import SimpleNamespace

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


def fake_solve_fn(solver, threads):
    return lambda positions: np.zeros((len(positions.current_player), 2), np.float32)


def run_main(monkeypatch, *args):
    monkeypatch.setattr(train, "Trainer", fake_trainer)
    monkeypatch.setattr(train, "make_solve_fn", fake_solve_fn)
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
    "options, resignation",
    [
        ((), None),
        (("--solve-empties", "off", "--resign"), (0.95, 0.1)),
        (
            (
                "--solve-empties",
                "off",
                "--resign",
                "0.9",
                "--resign-calibration",
                "0.2",
            ),
            (0.9, 0.2),
        ),
    ],
)
def test_resignation_is_off_unless_asked_for(monkeypatch, options, resignation):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, *options)
    got = built.value.kwargs["resignation"]
    if resignation is None:
        assert got is None
    else:
        assert (got.threshold, got.calibration_fraction) == resignation


@pytest.mark.parametrize(
    "option",
    [
        ("--solve-empties", "off", "--resign", "0"),
        ("--solve-empties", "off", "--resign", "1.5"),
        ("--resign-calibration", "2"),
        ("--resign",),
    ],
)
def test_resignation_options_out_of_range_are_refused(monkeypatch, option):
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


def test_playout_cap_and_selfplay_cycles_reach_self_play(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--sims",
            "63",
            "--playout-cap",
            "0.25:21",
            "--selfplay-cycles",
            str(128 * 65),
        )
    evaluator = built.value.kwargs["evaluator"].at(0)
    assert evaluator.playout_cap == train.PlayoutCap(0.25, 21)
    assert evaluator.num_iterations == 63
    # 8320 cycles: 130 full searches' worth
    assert built.value.kwargs["collection_steps_per_epoch"] == 130
    # searches of varying lengths: played every few cycles rather than every full search's
    assert built.value.kwargs["advance_every"] == 5
    assert built.value.kwargs["extra_config"]["args"]["playout_cap"] == {
        "full_probability": 0.25,
        "fast_iterations": 21,
        "noisy": "full",
    }


def test_playout_cap_noise_picks_the_noisy_searches(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--sims",
            "63",
            "--playout-cap",
            "0.25:21",
            "--playout-cap-noise",
            "fast",
        )
    evaluator = built.value.kwargs["evaluator"].at(0)
    assert evaluator.playout_cap == train.PlayoutCap(0.25, 21, "fast")


def test_playout_cap_noise_goes_with_a_playout_cap(monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--sims", "63", "--playout-cap-noise", "fast")


def test_without_playout_cap_every_search_is_full(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--sims", "64")
    assert built.value.kwargs["evaluator"].at(0).playout_cap is None
    assert built.value.kwargs["collection_steps_per_epoch"] == 128
    assert built.value.kwargs["advance_every"] is None


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
    # each network has its own engine-set testers, of its own network: the positions it
    # every position, and the positions it decides
    own, theirs = built.value.kwargs["testers"], grown.testers
    assert (
        [t.name for t in own]
        == [t.name for t in theirs]
        == ["engine", "filtered_engine"]
    )
    assert all(a is not b for a, b in zip(own, theirs, strict=True))


def test_grow_lr_scales_the_grown_networks_rate(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--grow",
            "resnet:2x8",
            "--grow-switch-at",
            "80",
            "--lr",
            "1e-5",
            "--grow-lr",
            "5e-5",
        )
    (grown,) = built.value.kwargs["growth"].learners
    assert grown.lr_scale == pytest.approx(5.0)


def test_grow_can_switch_at_fixed_epochs_without_the_engine_set(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--grow", "resnet:2x8", "--grow-switch-at", "80")
    assert built.value.kwargs["growth"].switch_at == (80,)
    assert built.value.kwargs["growth"].learners[0].testers == []


@pytest.mark.parametrize(
    "options",
    [
        ("--playout-cap", "0.25"),
        ("--playout-cap", "0:21"),
        ("--playout-cap", "1.5:21"),
        ("--sims", "32", "--playout-cap", "0.25:32"),
        ("--sims-schedule", "0:32,10:64", "--playout-cap", "0.25:8"),
        ("--sims", "63", "--selfplay-cycles", "100"),
        ("--sims-schedule", "0:32,10:64", "--selfplay-cycles", "330"),
    ],
)
def test_bad_playout_cap_options_are_refused(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)


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
        # a manual rule without --grow, or with fixed epochs
        ("--grow-manual",),
        # a margin or guard without --grow; a margin out of range; a guard without the set
        ("--grow-margin", "0.01"),
        ("--grow-guard", "engine_value_mse"),
        ("--grow", "resnet:2x8", "--grow-metric", "loss", "--grow-margin", "1.5"),
        (
            "--grow",
            "resnet:2x8",
            "--grow-metric",
            "loss",
            "--grow-guard",
            "engine_value_mse",
            "--engine-set",
            "no-such-set.npz",
        ),
        ("--grow", "resnet:2x8", "--grow-manual", "--grow-switch-at", "80"),
        # a learning rate for the grown networks: with --grow, positive
        ("--grow-lr", "5e-5"),
        ("--grow", "resnet:2x8", "--grow-switch-at", "80", "--grow-lr", "0"),
        # taking over needs a branch of a run with growth
        ("--grow", "resnet:2x8", "--grow-switch-at", "80", "--take-over"),
        ("--branch-from", "runs/trunk", "--take-over"),
        # a branch continues a run: it isn't also a fork or a resumed run
        ("--branch-from", "runs/trunk", "--resume", "--ckpt-dir", "runs/b"),
        ("--branch-from", "runs/trunk", "--init-from", "runs/trunk"),
        # a warm start: of a network of a new run with growth, as SIZE=PATH
        ("--warm-start", "1x4=runs/a/state-1.npz"),
        ("--grow", "resnet:2x8", "--grow-switch-at", "80", "--warm-start", "2x8"),
        (
            "--grow",
            "resnet:2x8",
            "--grow-switch-at",
            "80",
            "--warm-start",
            "2x8=runs/a/state-1.npz",
            "--resume",
        ),
    ],
)
def test_bad_grow_options_are_refused(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)


def test_growth_can_follow_a_weight_averages_metric(monkeypatch, tmp_path):
    from test_engine_set import tiny_set

    tiny_set(np.random.default_rng(0)).save(tmp_path / "set.npz")
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--grow",
            "resnet:2x8",
            "--grow-metric",
            "avg4/engine_value_mse",
            "--grow-guard",
            "avg4/engine_exp_regret",
            "--average-epochs",
            "4",
            "--engine-eval-every",
            "1",
            "--engine-set",
            str(tmp_path / "set.npz"),
        )
    growth = built.value.kwargs["growth"]
    assert (growth.metric, growth.guard) == (
        "avg4/engine_value_mse",
        "avg4/engine_exp_regret",
    )


@pytest.mark.parametrize(
    "metric",
    ["avg16/engine_value_mse", "avg4/loss", "engine_top1", "avg/engine_value_mse"],
)
def test_growth_metrics_name_a_kept_average_and_an_error(monkeypatch, metric):
    # (avg16 isn't one of --average-epochs; the rest aren't metrics the rule can follow)
    with pytest.raises(SystemExit):
        run_main(
            monkeypatch,
            "--grow",
            "resnet:2x8",
            "--grow-switch-at",
            "80",
            "--average-epochs",
            "4",
            "--grow-metric",
            metric,
        )


def test_a_buffer_window_reaches_the_replay_buffer(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--buffer", "9000", "--buffer-window", "1125")
    buffer = built.value.kwargs["memory_buffer"]
    assert (buffer.capacity, buffer.window) == (9000, 1125)
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--buffer", "9000")
    assert built.value.kwargs["memory_buffer"].window is None


@pytest.mark.parametrize("window", ["0", "9001"])
def test_a_buffer_window_is_within_the_buffer(monkeypatch, window):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--buffer", "9000", "--buffer-window", window)


def test_warm_starts_reach_the_training_loop(monkeypatch):
    built, loops = [], []

    class Trainer:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self.ckpt_dirs = ("a", "b")

        def train_loop(self, **kwargs):
            loops.append(kwargs)

    monkeypatch.setattr(train, "Trainer", Trainer)
    monkeypatch.setattr(train, "make_solve_fn", fake_solve_fn)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--net",
            "resnet:1x4",
            "--engine-eval-every",
            "0",
            "--grow",
            "resnet:2x8",
            "--grow-switch-at",
            "80",
            "--warm-start",
            "2x8=runs/a/state-1.npz",
            "--warm-start",
            "1x4=runs/b",
        ],
    )
    train.main()
    assert loops[0]["warm_start"] == [("2x8", "runs/a/state-1.npz"), ("1x4", "runs/b")]
    # the trainer records them itself, with the states they came from
    assert "warm_start" not in built[0]["extra_config"]["args"]


def test_a_manual_rule_reaches_growth(monkeypatch, tmp_path):
    from test_engine_set import tiny_set

    tiny_set(np.random.default_rng(0)).save(tmp_path / "set.npz")
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--grow",
            "resnet:2x8",
            "--grow-manual",
            "--engine-eval-every",
            "1",
            "--engine-set",
            str(tmp_path / "set.npz"),
        )
    assert built.value.kwargs["growth"].manual


def test_growth_on_the_training_loss_needs_no_engine_set(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--grow", "resnet:2x8", "--grow-metric", "loss")
    assert built.value.kwargs["growth"].metric == "loss"


def test_a_margin_and_guard_reach_growth(monkeypatch, tmp_path):
    from test_engine_set import tiny_set

    tiny_set(np.random.default_rng(0)).save(tmp_path / "set.npz")
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--grow",
            "resnet:2x8",
            "--grow-metric",
            "loss",
            "--grow-margin",
            "0.01",
            "--grow-guard",
            "engine_value_mse",
            "--engine-eval-every",
            "1",
            "--engine-set",
            str(tmp_path / "set.npz"),
        )
    growth = built.value.kwargs["growth"]
    assert (growth.metric, growth.margin, growth.guard) == (
        "loss",
        0.01,
        "engine_value_mse",
    )


def test_weight_averages_reach_the_trainer(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--average-epochs",
            "2",
            "8",
            "--selfplay-average",
            "8",
            "--average-warmup",
            "5",
        )
    assert built.value.kwargs["averages"] == [2.0, 8.0]
    assert built.value.kwargs["selfplay_average"] == 8.0
    assert built.value.kwargs["average_warmup"] == 5.0


def test_sgd_takes_its_momentum_and_its_own_weight_decay(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--optimizer", "sgd", "--lr", "0.2", "--momentum", "0.8")
    state = built.value.kwargs["optimizer"].init({"w": jnp.zeros((2, 2))})
    hyperparams = {k: float(v) for k, v in state.hyperparams.items()}
    assert hyperparams == pytest.approx(
        {"learning_rate": 0.2, "momentum": 0.8, "weight_decay": train.SGD_WEIGHT_DECAY}
    )


def test_adamw_keeps_its_weight_decay(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch)
    state = built.value.kwargs["optimizer"].init({"w": jnp.zeros((2, 2))})
    assert float(state.hyperparams["weight_decay"]) == train.WEIGHT_DECAY
    assert built.value.kwargs["lr_drop"] is None
    # measurements that cost compute are off unless asked for
    assert built.value.kwargs["noise_measure"] == 0
    assert built.value.kwargs["fresh_positions"] == 0


def test_lr_drop_and_the_noise_metrics_reach_the_trainer(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--lr-drop",
            "150:0.1",
            "--grad-accum",
            "4",
            "--noise-measure",
            "64",
            "--fresh-positions",
            "8192",
        )
    kwargs = built.value.kwargs
    assert (kwargs["lr_drop"].after_epochs, kwargs["lr_drop"].factor) == (150, 0.1)
    assert (kwargs["noise_measure"], kwargs["fresh_positions"]) == (64, 8192)


@pytest.mark.parametrize(
    "options",
    [
        ("--average-epochs", "0.5"),
        ("--average-epochs", "2", "2"),
        ("--average-epochs", "2", "--selfplay-average", "4"),
        ("--selfplay-average", "2"),
        ("--average-epochs", "2", "--average-warmup", "0"),
    ],
)
def test_bad_average_options_are_refused(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)


def test_the_waiting_networks_training_steps_reach_the_trainer(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--grow",
            "resnet:2x8",
            "--grow-switch-at",
            "80",
            "--grow-train-steps",
            "3",
        )
    assert built.value.kwargs["waiting_train_steps"] == 3


@pytest.mark.parametrize(
    "options",
    [
        ("--grow-train-steps", "3"),
        ("--grow", "resnet:2x8", "--grow-switch-at", "80", "--grow-train-steps", "0"),
    ],
)
def test_bad_waiting_train_steps_are_refused(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)


def test_retired_networks_training_reaches_the_trainer(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--grow",
            "resnet:2x8",
            "--grow-switch-at",
            "80",
            "--grow-train-retired",
        )
    assert built.value.kwargs["train_retired"] is True


def test_retired_networks_training_needs_grow(monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--grow-train-retired")


def test_a_frozen_teacher_reaches_the_trainer(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--grow",
            "resnet:2x8",
            "--grow-switch-at",
            "80",
            "--grow-train-retired",
            "--grow-frozen",
        )
    assert built.value.kwargs["train_playing"] is False
    assert built.value.kwargs["train_retired"] is True


def test_a_rack_of_named_students_with_their_own_rates(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--net",
            "teacher=resnet:1x4",
            "--grow",
            "wide=resnet:1x8",
            "--grow",
            "deep=resnet:2x4",
            "--grow",
            "resnet:1x16",
            "--grow-switch-at",
            "80,90,100",
            "--grow-frozen",
            "--grow-train-waiting",
            "--lr",
            "1e-4",
            "--lr-of",
            "deep=5e-5",
            "--weight-decay-of",
            "deep=0.05",
        )
    kwargs = built.value.kwargs
    growth = kwargs["growth"]
    assert growth.names == ("teacher", "wide", "deep", "1x16")
    wide, deep, plain = growth.learners
    assert (wide.lr_scale, wide.hyperparams) == (1.0, {})
    assert deep.lr_scale == pytest.approx(0.5)
    assert deep.hyperparams == {"weight_decay": 0.05}
    assert plain.lr_scale == 1.0
    # the specs, not their names, make the networks
    assert deep.nn is not wide.nn
    assert (kwargs["train_waiting"], kwargs["train_playing"]) == (True, False)


@pytest.mark.parametrize(
    "options",
    [
        ("--grow-train-waiting",),
        ("--grow", "a=resnet:2x8", "--grow-switch-at", "80", "--lr-of", "b=1e-4"),
        ("--grow", "a=resnet:2x8", "--grow-switch-at", "80", "--lr-of", "a=0"),
        ("--grow", "a=resnet:2x8", "--grow-switch-at", "80", "--lr-of", "a"),
        (
            "--grow",
            "a=resnet:2x8",
            "--grow-switch-at",
            "80",
            "--weight-decay-of",
            "a=-1",
        ),
        (
            "--grow",
            "a=resnet:2x8",
            "--grow-switch-at",
            "80",
            "--lr-of",
            "a=1e-4",
            "--lr-of",
            "a=2e-4",
        ),
    ],
)
def test_bad_rack_options_are_refused(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)


def test_named_specs():
    assert train.named_spec("r7=resnet:7x64,expand=2") == ("r7", "resnet:7x64,expand=2")
    assert train.named_spec("resnet:7x64,expand=2") == (None, "resnet:7x64,expand=2")
    assert train.net_name("r7=resnet:7x64") == "r7"
    assert train.net_name("resnet:7x64,expand=2") == "7x64"


def test_a_frozen_teacher_needs_grow(monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--grow-frozen")


@pytest.mark.parametrize(
    "options",
    [
        ("--decoupled-decay",),
        ("--lr-drop", "150"),
        ("--lr-drop", "150:2"),
        ("--lr-drop", "-1:0.1"),
        ("--grad-accum", "4", "--fresh-positions", "1000"),
        ("--noise-measure", "-1"),
    ],
)
def test_bad_optimizer_drop_and_noise_options_are_refused(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)


def test_a_fixed_network_can_play_and_keep_its_moves(monkeypatch, tmp_path):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch, "--train-steps", "0", "--log-moves", str(tmp_path / "moves")
        )
    assert built.value.kwargs["train_steps_per_epoch"] == 0
    assert built.value.kwargs["log_moves"] == str(tmp_path / "moves")


def test_every_position_at_20_empties_or_fewer_is_solved_by_default(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch)
    solve = built.value.kwargs["endgame_solve"]
    assert solve.play_past == math.inf
    assert solve.share == 1.0
    assert solve.get_config()["play_past"] == "end"
    # the cut is at 20 empty squares: 44 discs on the board
    for discs, stops in ((43, False), (44, True)):
        observation = jnp.zeros(128).at[:discs].set(1).reshape(8, 8, 2)
        assert bool(solve.stop_fn(SimpleNamespace(observation=observation))) is stops


def test_solve_empties_off_plays_games_out(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--solve-empties", "off")
    assert built.value.kwargs["endgame_solve"] is None


@pytest.mark.parametrize("value", ["-1", "60", "none", "2.5"])
def test_bad_solve_empties_is_refused(monkeypatch, value):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--solve-empties", value)


@pytest.mark.parametrize("value", ["end", "inf"])
def test_play_past_end_plays_every_game_to_its_end_with_room_for_its_solves(
    monkeypatch, value
):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--solve-empties", "20", "--solve-play-past", value)
    solve = built.value.kwargs["endgame_solve"]
    assert solve.play_past == math.inf
    with pytest.raises(BuiltTrainer) as mean:
        run_main(monkeypatch, "--solve-empties", "20", "--solve-play-past", "4")
    assert solve.slots > mean.value.kwargs["endgame_solve"].slots


@pytest.mark.parametrize("value", ["-1", "nan", "forever"])
def test_bad_play_past_is_refused(monkeypatch, value):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--solve-empties", "20", "--solve-play-past", value)


def test_solve_slots_hold_whole_games_and_at_most_a_solve_a_move():
    # 64 moves an epoch
    cycles, every = 64 * 129, 129
    # past a solve point at 20 a game can store more positions than it has moves before it,
    # so the slots are two epochs' moves
    assert train.solve_slots(cycles, every, 20, math.inf) == 2 * 64 + 2
    # at 2, a game stores at most 5 positions past it: 2 games an epoch, each 6 solves
    assert train.solve_slots(cycles, every, 2, math.inf) == 2 * 2 * 6 + 2
    for empties in (2, 10, 20):
        whole = train.solve_slots(cycles, every, empties, math.inf)
        for mean in (0.0, 1.0, 4.0, 50.0):
            assert train.solve_slots(cycles, every, empties, mean) <= whole
    # a mean's room is twice the mean, as before
    assert train.solve_slots(cycles, every, 20, 4.0) == 2 * 2 * 9 + 2
