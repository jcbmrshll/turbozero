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
        ["train.py", "--net", "resnet:1x4", "--validate-every", "0", *args],
    )
    train.main()


def test_root_noise_options_reach_self_play(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--dirichlet-alpha", "1.0", "--dirichlet-epsilon", "0.1")
    evaluator = built.value.kwargs["evaluator"]
    assert evaluator.dirichlet_alpha == 1.0
    assert evaluator.dirichlet_epsilon == 0.1


@pytest.mark.parametrize("options, envs", [((), 1024), (("--envs", "2048"), 2048)])
def test_envs_sets_the_self_play_environments(monkeypatch, options, envs):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, *options)
    assert built.value.kwargs["batch_size"] == envs


@pytest.mark.parametrize(
    "option", [("--dirichlet-alpha", "0"), ("--dirichlet-epsilon", "1.5")]
)
def test_root_noise_options_out_of_range_are_refused(monkeypatch, option):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *option)


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
    evaluator = built.value.kwargs["evaluator"]
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


def test_without_playout_cap_every_search_is_full(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--sims", "64")
    evaluator = built.value.kwargs["evaluator"]
    assert evaluator.playout_cap is None
    assert evaluator.max_nodes == 4 * 64
    assert built.value.kwargs["collection_steps_per_epoch"] == 128
    assert built.value.kwargs["advance_every"] is None


@pytest.mark.parametrize(
    "options",
    [
        ("--playout-cap", "0.25"),
        ("--playout-cap", "0:21"),
        ("--playout-cap", "1.5:21"),
        ("--sims", "32", "--playout-cap", "0.25:32"),
        ("--sims", "63", "--selfplay-cycles", "100"),
    ],
)
def test_bad_playout_cap_options_are_refused(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)


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


def test_self_play_validation_and_checkpoints_use_a_4_epoch_average(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch)
    kwargs = built.value.kwargs
    assert kwargs["average_epochs"] == 4.0
    assert kwargs["average_norm_batches"] == 32
    assert kwargs["save_weights_every"] == 1
    assert kwargs["save_state_every"] is None and kwargs["save_state_at"] == []
    # memorization is measured only if asked for
    assert kwargs["holdout_envs"] == 0 and kwargs["holdout_positions"] == 0


def test_average_and_save_options_reach_the_trainer(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--average-epochs",
            "8",
            "--average-norm-batches",
            "0",
            "--save-weights-every",
            "10",
            "--save-raw-every",
            "5",
            "--save-state-every",
            "20",
            "--save-state-at",
            "50,150",
        )
    kwargs = built.value.kwargs
    assert kwargs["average_epochs"] == 8.0
    assert kwargs["average_norm_batches"] == 0
    assert kwargs["save_weights_every"] == 10
    assert kwargs["save_raw_every"] == 5
    assert kwargs["save_state_every"] == 20
    assert kwargs["save_state_at"] == [50, 150]


@pytest.mark.parametrize(
    "options",
    [
        ("--average-epochs", "0.5"),
        ("--save-weights-every", "0"),
        ("--save-raw-every", "0"),
        ("--holdout-envs", "-1"),
        ("--decoupled-decay",),
        ("--init-without-buffers",),
        ("--resume",),
    ],
)
def test_bad_options_are_refused(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *options)


def test_held_out_positions_are_whole_micro_batches(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--holdout-envs",
            "32",
            "--holdout-positions",
            "5000",
            "--train-batch",
            "4096",
            "--grad-accum",
            "2",
        )
    assert built.value.kwargs["holdout_envs"] == 32
    assert built.value.kwargs["holdout_positions"] == 3 * 2048


def test_adamw_holds_its_learning_rate_and_weight_decay(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--lr", "3e-4")
    state = built.value.kwargs["optimizer"].init({"w": jnp.zeros((2, 2))})
    hyperparams = {k: float(v) for k, v in state.hyperparams.items()}
    assert hyperparams["learning_rate"] == pytest.approx(3e-4)
    assert hyperparams["weight_decay"] == train.WEIGHT_DECAY


def test_sgd_takes_its_momentum_and_its_own_weight_decay(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--optimizer", "sgd", "--lr", "0.2", "--momentum", "0.8")
    state = built.value.kwargs["optimizer"].init({"w": jnp.zeros((2, 2))})
    hyperparams = {k: float(v) for k, v in state.hyperparams.items()}
    assert hyperparams == pytest.approx(
        {"learning_rate": 0.2, "momentum": 0.8, "weight_decay": train.SGD_WEIGHT_DECAY}
    )


def test_the_engine_set_validates_the_average(monkeypatch, tmp_path):
    from test_engine_set import tiny_set

    tiny_set(np.random.default_rng(0)).save(tmp_path / "set.npz")
    made = []
    make_network = train.make_network

    def recording_make_network(*args, **kwargs):
        network = make_network(*args, **kwargs)
        made.append((kwargs.get("inference_dtype"), network[0]))
        return network

    monkeypatch.setattr(train, "make_network", recording_make_network)
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--validate-every",
            "2",
            "--engine-set",
            str(tmp_path / "set.npz"),
            "--headroom-every",
            "0",
        )
    [validator] = built.value.kwargs["validators"]
    assert validator.name == "engine"
    assert validator.prior_temperature == train.PLAY_PRIOR_TEMPERATURE
    assert validator.get_config()["path"] == str(tmp_path / "set.npz")
    assert built.value.kwargs["validate_every"] == 2
    # self-play's network computes in bfloat16, the validator's in float32
    assert [dtype for dtype, _ in made] == ["bfloat16", "float32"]


def test_the_headroom_probe_searches_the_engine_set_as_self_play_does(
    monkeypatch, tmp_path
):
    from test_engine_set import tiny_set

    tiny_set(np.random.default_rng(0)).save(tmp_path / "set.npz")
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--validate-every",
            "1",
            "--engine-set",
            str(tmp_path / "set.npz"),
            "--sims",
            "16",
            "--playout-cap",
            "0.25:4",
            "--dirichlet-alpha",
            "0.5",
            "--headroom-every",
            "7",
            "--headroom-positions",
            "3",
        )
    engine, headroom = built.value.kwargs["validators"]
    assert engine.name == "engine" and headroom.name == "search"
    assert headroom.every == 7
    assert headroom.num_positions == 3
    # self-play's budget and twice it, full searches from fresh trees, with its root noise
    assert list(headroom.evaluators) == ["16", "32"]
    for sims, evaluator in zip((16, 32), headroom.evaluators.values(), strict=True):
        assert evaluator.num_iterations == sims
        assert evaluator.max_nodes == sims + 1
        assert evaluator.playout_cap is None
        assert evaluator.dirichlet_alpha == 0.5
    # positions keep their phases, as the engine validator's
    assert headroom.group_names == engine.group_names


def test_headroom_sims_choose_the_budgets(monkeypatch, tmp_path):
    from test_engine_set import tiny_set

    tiny_set(np.random.default_rng(0)).save(tmp_path / "set.npz")
    with pytest.raises(BuiltTrainer) as built:
        run_main(
            monkeypatch,
            "--validate-every",
            "1",
            "--engine-set",
            str(tmp_path / "set.npz"),
            "--headroom-sims",
            "64",
            "256",
            "512",
        )
    _, headroom = built.value.kwargs["validators"]
    assert list(headroom.evaluators) == ["64", "256", "512"]


def test_self_play_data_is_scored_by_phase_and_side_to_move(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch, "--data-positions", "2048")
    assert built.value.kwargs["data_positions"] == 2048
    assert list(built.value.kwargs["position_groups"]) == [
        "opening",
        "midgame",
        "late",
        "endgame",
        "black",
        "white",
    ]


def test_without_an_engine_set_nothing_is_validated(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch)
    assert built.value.kwargs["validators"] == []


class FakeTrainer:
    """Records the Trainer's arguments, and how the run starts."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeTrainer.made = self

    def resume(self):
        self.calls.append("resume")
        return "resumed"

    def branch(self, path):
        self.calls.append(("branch", path))
        return "branched"

    def train_loop(self, **kwargs):
        self.calls.append(("train_loop", kwargs))


def run_fake(monkeypatch, *args):
    monkeypatch.setattr(train, "Trainer", FakeTrainer)
    monkeypatch.setattr(train, "make_solve_fn", fake_solve_fn)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--net", "resnet:1x4", "--validate-every", "0", *args],
    )
    train.main()
    return FakeTrainer.made


def test_a_new_run_starts_from_scratch(monkeypatch):
    trainer = run_fake(monkeypatch, "--epochs", "7", "--seed", "3")
    [(called, loop)] = trainer.calls
    assert called == "train_loop"
    assert loop == {
        "seed": 3,
        "num_epochs": 7,
        "initial_state": None,
        "fork_from": None,
        "fork_replay_buffer": True,
    }


def test_a_fork_starts_from_a_saved_state(monkeypatch):
    trainer = run_fake(monkeypatch, "--init-from", "runs/a", "--init-without-buffers")
    [(called, loop)] = trainer.calls
    assert called == "train_loop"
    assert loop["fork_from"] == "runs/a" and loop["fork_replay_buffer"] is False
    # how the run starts isn't part of its config: the trainer records its parent
    assert "init_from" not in trainer.kwargs["extra_config"]["args"]


def test_resume_and_branch_continue_a_saved_run(monkeypatch):
    trainer = run_fake(monkeypatch, "--resume", "--ckpt-dir", "runs/a")
    assert trainer.calls[0] == "resume"
    assert trainer.calls[1][1]["initial_state"] == "resumed"
    trainer = run_fake(
        monkeypatch, "--branch-from", "runs/a/state-10.npz", "--lr", "1e-4"
    )
    assert trainer.calls[0] == ("branch", "runs/a/state-10.npz")
    assert trainer.calls[1][1]["initial_state"] == "branched"
    assert "branch_from" not in trainer.kwargs["extra_config"]["args"]


@pytest.mark.parametrize(
    "options",
    [
        ("--resume", "--init-from", "runs/a"),
        ("--branch-from", "runs/a", "--init-from", "runs/b"),
    ],
)
def test_a_run_starts_one_way(monkeypatch, options):
    with pytest.raises(SystemExit):
        run_fake(monkeypatch, "--ckpt-dir", "runs/c", *options)


def test_every_position_at_20_empties_or_fewer_is_solved_by_default(monkeypatch):
    with pytest.raises(BuiltTrainer) as built:
        run_main(monkeypatch)
    solve = built.value.kwargs["endgame_solve"]
    assert solve.play_past == math.inf
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
