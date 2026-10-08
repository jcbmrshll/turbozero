"""End-to-end Trainer runs on tiny tic-tac-toe configurations.

Every Trainer instance compiles its own self-play, training and testing functions, so the tests
share one trainer per device count and only change settings that don't affect compilation."""

import os
import shutil
from dataclasses import replace
from functools import partial

import equinox as eqx
import jax
import numpy as np
import optax
import pytest

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.monitor import Monitor
from core.monitor.renderers import pgx_two_player_episode
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.testing.two_player_tester import TwoPlayerTester
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer, checkpoint_epochs, extract_params

# warmup is longer than a tic-tac-toe game, so the buffer holds finished episodes before training starts
STEPS_PER_EPOCH = 10


def make_trainer(ttt, num_devices, ckpt_dir):
    config = AZResnetConfig(
        policy_head_out_size=ttt.num_actions, num_blocks=1, num_channels=4
    )
    net, nn_state = eqx.nn.make_with_state(AZResnet)(
        config, ttt.env.observation_shape, key=jax.random.PRNGKey(0)
    )
    make_evaluator = partial(
        AlphaZero(MCTS),
        eval_fn=make_nn_eval_fn(net, ttt.state_to_nn_input),
        num_iterations=4,
        max_nodes=8,
        branching_factor=ttt.num_actions,
        action_selector=PUCTSelector(),
    )
    return Trainer(
        batch_size=2 * num_devices,
        train_batch_size=4 * num_devices,
        warmup_steps=STEPS_PER_EPOCH,
        collection_steps_per_epoch=STEPS_PER_EPOCH,
        train_steps_per_epoch=1,
        nn=net,
        nn_state=nn_state,
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=1e-4),
        optimizer=optax.adam(1e-3),
        evaluator=make_evaluator(temperature=1.0),
        evaluator_test=make_evaluator(temperature=0.0),
        memory_buffer=EpisodeReplayBuffer(capacity=32),
        max_episode_steps=10,
        env_step_fn=ttt.step_fn,
        env_init_fn=ttt.init_fn,
        state_to_nn_input_fn=ttt.state_to_nn_input,
        testers=[TwoPlayerTester(num_episodes=2)],
        ckpt_dir=str(ckpt_dir),
        num_devices=num_devices,
    )


@pytest.fixture(scope="module")
def trainers(ttt, tmp_path_factory):
    return {
        n: make_trainer(ttt, n, tmp_path_factory.mktemp(f"ckpt_{n}_devices"))
        for n in (1, 2)
    }


@pytest.fixture(scope="module")
def trained(trainers):
    """Output of a 2-epoch run on each device count."""
    return {
        n: trainer.train_loop(seed=0, num_epochs=2) for n, trainer in trainers.items()
    }


def leaves_equal(a, b):
    return all(
        np.array_equal(x, y)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


@pytest.mark.parametrize("num_devices", [1, 2])
def test_train_loop_runs(trainers, trained, num_devices):
    out = trained[num_devices]

    assert out.cur_epoch == 2
    np.testing.assert_array_equal(out.train_state.step, [2] * num_devices)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(out.train_state.params))
    # gradients are averaged across devices, so every device holds the same params
    for x in jax.tree.leaves(extract_params(out.train_state)):
        for device in range(1, num_devices):
            np.testing.assert_array_equal(x[device], x[0])
    buffer_state = out.collection_state.buffer_state
    assert (buffer_state.populated & buffer_state.has_reward).any()
    assert sorted(os.listdir(trainers[num_devices].ckpt_dir)) == ["0.eqx", "1.eqx"]


def test_checkpoint_round_trip(trainers, trained):
    trainer, out = trainers[2], trained[2]

    restored = trainer.load_train_state_from_checkpoint(trainer.ckpt_dir, 1)

    assert leaves_equal(extract_params(restored), extract_params(out.train_state))
    assert leaves_equal(restored.opt_state, out.train_state.opt_state)


def test_checkpoint_loads_on_other_device_count(trainers, trained):
    restored = trainers[1].load_train_state_from_checkpoint(trainers[2].ckpt_dir, 1)

    for x, y in zip(
        jax.tree.leaves(restored), jax.tree.leaves(trained[2].train_state), strict=True
    ):
        np.testing.assert_array_equal(x, y[:1])


def test_save_checkpoint_keeps_max_checkpoints(ttt, trainers, trained, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    shutil.copytree(trainers[2].ckpt_dir, ckpt_dir)
    trainer = make_trainer(ttt, 2, ckpt_dir)

    trainer.save_checkpoint(trained[2].train_state, 2)

    assert checkpoint_epochs(str(ckpt_dir)) == [1, 2]


def test_new_trainer_keeps_existing_checkpoints(ttt, trainers, trained, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    shutil.copytree(trainers[2].ckpt_dir, ckpt_dir)

    make_trainer(ttt, 2, ckpt_dir)

    assert sorted(os.listdir(ckpt_dir)) == ["0.eqx", "1.eqx"]


def test_save_checkpoint_refuses_existing_epoch(trainers, trained):
    trainer, out = trainers[2], trained[2]

    # e.g. a fresh run started in a directory that already holds checkpoints
    with pytest.raises(ValueError, match="already has a checkpoint"):
        trainer.save_checkpoint(out.train_state, 1)

    assert sorted(os.listdir(trainer.ckpt_dir)) == ["0.eqx", "1.eqx"]
    restored = trainer.load_train_state_from_checkpoint(trainer.ckpt_dir, 1)
    assert leaves_equal(extract_params(restored), extract_params(out.train_state))


def test_self_play_uses_latest_params(trainers, monkeypatch):
    trainer = trainers[1]
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    # record the params each self-play collection uses, and the params after each epoch's training
    collected, trained_params = [], []
    collect_steps, train_steps = trainer.collect_steps, trainer.train_steps

    def spy_collect_steps(key, state, params, num_steps):
        collected.append(params)
        return collect_steps(key, state, params, num_steps)

    def spy_train_steps(*args, **kwargs):
        collection_state, train_state, metrics = train_steps(*args, **kwargs)
        trained_params.append(extract_params(train_state))
        return collection_state, train_state, metrics

    monkeypatch.setattr(trainer, "collect_steps", spy_collect_steps)
    monkeypatch.setattr(trainer, "train_steps", spy_train_steps)

    trainer.train_loop(seed=0, num_epochs=3, eval_every=2)

    # warmup, then one collection per epoch
    assert len(collected) == 4
    for epoch in range(1, 3):
        assert leaves_equal(collected[epoch + 1], trained_params[epoch - 1]), (
            f"epoch {epoch} self-play did not use the params trained in epoch {epoch - 1}"
        )


def test_train_loop_with_tester_skipping_epochs(trainers, monkeypatch):
    trainer = trainers[1]
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer.testers[0], "epochs_per_test", 2)

    out = trainer.train_loop(seed=0, num_epochs=2)

    assert out.cur_epoch == 2


def test_train_loop_logs_to_monitor(trainers, monitor_server, monkeypatch):
    trainer = trainers[1]
    monitor = Monitor(monitor_server.url, project="tests")
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "monitor", monitor)
    monkeypatch.setattr(trainer, "extra_config", {"note": "hello"})
    monkeypatch.setattr(trainer.testers[0], "episode_fn", pgx_two_player_episode())

    trainer.train_loop(seed=0, num_epochs=2)

    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert (meta["status"], meta["step"], meta["seed"], meta["num_epochs"]) == (
        "finished",
        1,
        0,
        2,
    )
    assert meta["config"]["batch_size"] == trainer.batch_size
    assert meta["config"]["note"] == "hello"
    rows = monitor_server.get(f"/api/runs/{monitor.run_id}/metrics")["rows"]
    for epoch in range(2):
        logged = {k for r in rows if r["step"] == epoch for k in r}
        assert {
            "loss",
            "policy_loss",
            "value_loss",
            "TwoPlayerTester_avg_outcome",
            "selfplay_episodes",
            "buffer_distinct_positions",
            "buffer_distinct_fraction",
        } <= logged
    # each test's first game went to the server as raw arrays, for it to render
    episodes = sorted(
        p.name for p in (monitor_server.dir / monitor.run_id / "episodes").iterdir()
    )
    assert episodes == ["TwoPlayerTester_game-0.npz", "TwoPlayerTester_game-1.npz"]


def test_crashed_train_loop_marks_run_crashed(trainers, monitor_server, monkeypatch):
    trainer = trainers[1]
    monitor = Monitor(monitor_server.url, project="tests")
    monkeypatch.setattr(trainer, "monitor", monitor)

    def crash(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trainer, "train_steps", crash)

    with pytest.raises(RuntimeError, match="boom"):
        trainer.train_loop(seed=0, num_epochs=1)

    assert monitor_server.get(f"/api/runs/{monitor.run_id}")["status"] == "crashed"


@pytest.mark.parametrize(
    "rewards, max_episode_steps, episodes, draws",
    [
        ((1.0, -1.0), 3, 2, 0),
        ((0.0, 0.0), 3, 2, 2),
        # truncated episodes aren't counted
        ((0.0, 0.0), 2, 0, 0),
    ],
)
def test_collect_counts_terminated_episodes_and_draws(
    fixed_length_env,
    scripted,
    make_collector,
    tmp_path,
    rewards,
    max_episode_steps,
    episodes,
    draws,
):
    env = fixed_length_env(length=3, rewards=rewards)
    collect = make_collector(env, scripted.first_legal, max_episode_steps, tmp_path)

    state = collect(num_steps=7)

    assert (int(state.episodes[0]), int(state.draws[0])) == (episodes, draws)


def test_selfplay_metrics_cover_the_epochs_episodes(trainers, trained):
    trainer, after = trainers[2], trained[2].collection_state
    before = replace(
        after, episodes=after.episodes - 1, draws=after.draws - (after.draws > 0)
    )

    metrics = trainer.selfplay_metrics(before, after)

    num_envs = after.episodes.size
    assert metrics["selfplay_episodes"] == num_envs
    assert metrics["selfplay_draw_fraction"] == (after.draws > 0).sum() / num_envs
    buffer_state = after.buffer_state
    sampleable = (buffer_state.populated & buffer_state.has_reward).sum()
    assert 0 < metrics["buffer_distinct_positions"] <= sampleable
    assert metrics["buffer_distinct_fraction"] == pytest.approx(
        metrics["buffer_distinct_positions"] / sampleable
    )
