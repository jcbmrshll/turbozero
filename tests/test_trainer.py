"""End-to-end Trainer runs on tiny tic-tac-toe configurations.

Every Trainer instance compiles its own self-play, training and testing functions, so the tests
share one trainer and only change settings that don't affect compilation."""

import os
import shutil
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
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
from core.training.schedule import EvaluatorSchedule, Schedule
from core.training.train import Trainer, checkpoint_epochs, extract_params

# warmup is longer than a tic-tac-toe game, so the buffer holds finished episodes before training starts
STEPS_PER_EPOCH = 10


class RootVisitCounter(AlphaZero(MCTS)):
    """AlphaZero that appends to `root_visits_added` how many visits each search adds to the
    root's children: its number of iterations, given a tree with room for every node the
    search adds."""

    def __init__(self, root_visits_added: list, **kwargs):
        super().__init__(**kwargs)
        self.root_visits_added = root_visits_added

    def evaluate(
        self, key, eval_state, env_state, root_metadata, params, env_step_fn, **kwargs
    ):
        def root_visits(tree):
            return tree.get_child_data("n", tree.ROOT_INDEX).sum()

        output = super().evaluate(
            key, eval_state, env_state, root_metadata, params, env_step_fn, **kwargs
        )
        jax.debug.callback(
            lambda n: self.root_visits_added.extend(np.ravel(n).tolist()),
            root_visits(output.eval_state) - root_visits(eval_state),
        )
        return output


def make_trainer(ttt, ckpt_dir, schedule=None, root_visits_added=None, **kwargs):
    """A Trainer for tic-tac-toe. With `schedule`, (first epoch, num_iterations, max_nodes)
    triples, self-play follows a schedule of `RootVisitCounter`s that record to
    `root_visits_added`."""
    config = AZResnetConfig(
        policy_head_out_size=ttt.num_actions, num_blocks=1, num_channels=4
    )
    net, nn_state = eqx.nn.make_with_state(AZResnet)(
        config, ttt.env.observation_shape, key=jax.random.PRNGKey(0)
    )
    evaluator_kwargs = {
        "eval_fn": make_nn_eval_fn(net, ttt.state_to_nn_input),
        "branching_factor": ttt.num_actions,
        "action_selector": PUCTSelector(),
    }
    make_evaluator = partial(
        AlphaZero(MCTS), num_iterations=4, max_nodes=8, **evaluator_kwargs
    )
    evaluator = (
        make_evaluator(temperature=1.0)
        if schedule is None
        else EvaluatorSchedule(
            [
                (
                    epoch,
                    RootVisitCounter(
                        root_visits_added if root_visits_added is not None else [],
                        num_iterations=n,
                        max_nodes=m,
                        **evaluator_kwargs,
                    ),
                )
                for epoch, n, m in schedule
            ]
        )
    )
    kwargs = {"evaluator_test": make_evaluator(temperature=0.0), **kwargs}
    return Trainer(
        batch_size=2,
        train_batch_size=4,
        warmup_steps=STEPS_PER_EPOCH,
        collection_steps_per_epoch=STEPS_PER_EPOCH,
        train_steps_per_epoch=1,
        nn=net,
        nn_state=nn_state,
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=1e-4),
        optimizer=optax.adam(1e-3),
        evaluator=evaluator,
        memory_buffer=EpisodeReplayBuffer(capacity=32),
        max_episode_steps=10,
        env_step_fn=ttt.step_fn,
        env_init_fn=ttt.init_fn,
        state_to_nn_input_fn=ttt.state_to_nn_input,
        testers=[TwoPlayerTester(num_episodes=2)],
        ckpt_dir=str(ckpt_dir),
        **kwargs,
    )


@pytest.fixture(scope="module")
def trainer(ttt, tmp_path_factory):
    return make_trainer(ttt, tmp_path_factory.mktemp("ckpt"))


@pytest.fixture(scope="module")
def trained(trainer):
    """Output of a 2-epoch run."""
    return trainer.train_loop(seed=0, num_epochs=2)


def leaves_equal(a, b):
    return all(
        np.array_equal(x, y)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


def test_train_loop_runs(trainer, trained):
    assert trained.cur_epoch == 2
    assert trained.train_state.step == 2
    assert all(
        np.isfinite(x).all() for x in jax.tree.leaves(trained.train_state.params)
    )
    buffer_state = trained.collection_state.buffer_state
    assert (buffer_state.populated & buffer_state.has_reward).any()
    assert sorted(os.listdir(trainer.ckpt_dir)) == ["0.eqx", "1.eqx"]


def test_checkpoint_round_trip(trainer, trained):
    restored = trainer.load_train_state_from_checkpoint(trainer.ckpt_dir, 1)

    assert leaves_equal(extract_params(restored), extract_params(trained.train_state))
    assert leaves_equal(restored.opt_state, trained.train_state.opt_state)


def test_save_checkpoint_keeps_max_checkpoints(ttt, trainer, trained, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    shutil.copytree(trainer.ckpt_dir, ckpt_dir)

    make_trainer(ttt, ckpt_dir).save_checkpoint(trained.train_state, 2)

    assert checkpoint_epochs(str(ckpt_dir)) == [1, 2]


def test_save_checkpoint_keeps_every(ttt, trained, tmp_path):
    trainer = make_trainer(ttt, tmp_path / "ckpt", keep_every=3)

    for epoch in range(8):
        trainer.save_checkpoint(trained.train_state, epoch)

    # multiples of 3 stay, besides the 2 newest of the rest
    assert checkpoint_epochs(trainer.ckpt_dir) == [0, 3, 5, 6, 7]


def test_new_trainer_keeps_existing_checkpoints(ttt, trainer, trained, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    shutil.copytree(trainer.ckpt_dir, ckpt_dir)

    make_trainer(ttt, ckpt_dir)

    assert sorted(os.listdir(ckpt_dir)) == ["0.eqx", "1.eqx"]


def test_save_checkpoint_refuses_existing_epoch(trainer, trained):
    # e.g. a fresh run started in a directory that already holds checkpoints
    with pytest.raises(ValueError, match="already has a checkpoint"):
        trainer.save_checkpoint(trained.train_state, 1)

    assert sorted(os.listdir(trainer.ckpt_dir)) == ["0.eqx", "1.eqx"]
    restored = trainer.load_train_state_from_checkpoint(trainer.ckpt_dir, 1)
    assert leaves_equal(extract_params(restored), extract_params(trained.train_state))


def test_self_play_uses_latest_params(trainer, monkeypatch):
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    # record the params each self-play collection uses, and the params after each epoch's training
    collected, trained_params = [], []
    collect_steps, train_steps = trainer.collect_steps, trainer.train_steps

    def spy_collect_steps(key, state, params, num_steps, **kwargs):
        collected.append(params)
        return collect_steps(key, state, params, num_steps, **kwargs)

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


def test_train_loop_with_tester_skipping_epochs(trainer, monkeypatch):
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer.testers[0], "epochs_per_test", 2)

    out = trainer.train_loop(seed=0, num_epochs=2)

    assert out.cur_epoch == 2


def test_train_loop_logs_to_monitor(trainer, monitor_server, monkeypatch):
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
            "selfplay_iterations",
            "replay_window",
            "buffer_samples",
            "buffer_distinct_positions",
            "buffer_distinct_fraction",
        } <= logged
    # each test's first game went to the server as raw arrays, for it to render
    episodes = sorted(
        p.name for p in (monitor_server.dir / monitor.run_id / "episodes").iterdir()
    )
    assert episodes == ["TwoPlayerTester_game-0.npz", "TwoPlayerTester_game-1.npz"]


def test_test_games_start_with_test_env_init_fn(trainer, ttt, monkeypatch):
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    used = []

    def tagged(tag):
        def init_fn(key):
            # runs when traced, once per compiled use
            used.append(tag)
            return ttt.init_fn(key)

        return init_fn

    monkeypatch.setattr(trainer, "env_init_fn", tagged("self-play"))
    monkeypatch.setattr(trainer, "test_env_init_fn", tagged("test"))

    trainer.train_loop(seed=0, num_epochs=1)

    assert "self-play" in used and "test" in used
    # the tester got the test init fn, and only it
    assert used[-1] == "test"


def test_train_loop_tells_the_monitor_what_it_is_doing(
    trainer, monitor_server, monkeypatch
):
    monitor = Monitor(monitor_server.url, project="tests")
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "monitor", monitor)
    activities = []
    monkeypatch.setattr(monitor, "activity", activities.append)

    trainer.train_loop(seed=0, num_epochs=2, eval_every=2)

    assert trainer.warmup_steps > 0
    assert activities == [
        f"warmup self-play ({trainer.warmup_steps} steps)",
        "epoch 0: self-play",
        "epoch 0: training",
        "epoch 0: testing TwoPlayerTester",
        "epoch 0: saving checkpoint",
        "epoch 1: self-play",
        "epoch 1: training",
        "epoch 1: saving checkpoint",
    ]
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["status"] == "finished"
    assert meta["activity"] is None


def test_crashed_train_loop_marks_run_crashed(trainer, monitor_server, monkeypatch):
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


def test_selfplay_metrics_cover_the_epochs_episodes(trainer, trained):
    after = trained.collection_state
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


def test_train_steps_without_finished_episodes_raises(trainer):
    collection_state = trainer.init_collection_state(
        jax.random.PRNGKey(0), trainer.batch_size
    )

    with pytest.raises(ValueError, match="no episodes have finished"):
        trainer.train_steps(
            jax.random.PRNGKey(0), collection_state, trainer.init_train_state(), 1
        )


@pytest.fixture
def scheduled(ttt, tmp_path, monkeypatch):
    """`scheduled(schedule)`: a Trainer whose self-play follows `schedule` (see `make_trainer`),
    and the visits each `collect_steps` call's searches added to the root, one set per call."""

    def make(schedule):
        root_visits_added, calls = [], []
        trainer = make_trainer(
            ttt,
            tmp_path / "ckpt",
            schedule=schedule,
            root_visits_added=root_visits_added,
        )
        collect_steps = trainer.collect_steps

        def spy_collect_steps(*args, **kwargs):
            state = collect_steps(*args, **kwargs)
            # wait for the searches' callbacks
            jax.effects_barrier()
            calls.append(set(root_visits_added))
            root_visits_added.clear()
            return state

        monkeypatch.setattr(trainer, "collect_steps", spy_collect_steps)
        return trainer, calls

    return make


def capacity(collection_state):
    return collection_state.eval_state.parents.shape[-1]


def test_schedule_switch_changes_the_search(scheduled):
    # both stages' trees are the same size, so the switch changes no array shapes: if
    # self-play kept running the first stage's compiled search, only the root's visit counts
    # would show it
    trainer, calls = scheduled([(0, 2, 64), (1, 6, 64)])

    trainer.train_loop(seed=0, num_epochs=3)

    # warmup, then epochs 0, 1, 2
    assert calls == [{2}, {2}, {6}, {6}]


def test_schedule_switch_reinitializes_search_trees(scheduled):
    trainer, calls = scheduled([(0, 2, 32), (1, 6, 64)])

    out = trainer.train_loop(seed=0, num_epochs=2)

    assert calls == [{2}, {2}, {6}]
    assert capacity(out.collection_state) == 64


def test_continuing_across_a_schedule_switch(scheduled):
    trainer, calls = scheduled([(0, 2, 32), (1, 6, 64)])

    first = trainer.train_loop(seed=0, num_epochs=1)
    assert capacity(first.collection_state) == 32
    calls.clear()
    # continues into the second stage, then within it
    second = trainer.train_loop(seed=0, num_epochs=2, initial_state=first)
    assert capacity(second.collection_state) == 64
    third = trainer.train_loop(seed=0, num_epochs=3, initial_state=second)
    assert capacity(third.collection_state) == 64
    # each continuation starts with a warmup
    assert calls == [{6}, {6}, {6}, {6}]
    assert checkpoint_epochs(trainer.ckpt_dir) == [1, 2]


def test_schedule_logs_iterations_and_config(scheduled, monkeypatch):
    trainer, _ = scheduled([(0, 2, 32), (1, 6, 64)])
    logged = {}
    monkeypatch.setattr(
        trainer, "log_metrics", lambda metrics, epoch: logged.setdefault(epoch, metrics)
    )
    activities = []
    monkeypatch.setattr(
        trainer, "set_activity", lambda text, echo=False: activities.append(text)
    )

    trainer.train_loop(seed=0, num_epochs=2)

    assert [int(logged[epoch]["selfplay_iterations"]) for epoch in (0, 1)] == [2, 6]
    assert (
        "epoch 1: self-play switches to RootVisitCounter, 6 iterations a move "
        "(compiling)" in activities
    )
    config = trainer.get_config()
    assert config["evaluator_train_config"]["num_iterations"] == 2
    assert [
        (
            stage["epoch"],
            stage["config"]["num_iterations"],
            stage["config"]["max_nodes"],
        )
        for stage in config["selfplay_schedule"]
    ] == [(0, 2, 32), (1, 6, 64)]


def test_schedule_needs_a_test_evaluator(ttt, tmp_path):
    with pytest.raises(ValueError, match="evaluator_test"):
        make_trainer(
            ttt, tmp_path, schedule=[(0, 2, 4), (1, 4, 8)], evaluator_test=None
        )


# per environment: the buffer holds 32 entries, and each epoch adds 10 (warmup adds 10 too), so
# the window binds in the first two epochs and holds the whole buffer in the third
WINDOWS = [(0, 16), (1, 24), (2, 32)]


def window_mask(buffer_state, window):
    """The entries training can sample from the `window` newest of each environment's,
    worked out entry by entry."""
    populated = np.asarray(buffer_state.populated)
    has_reward = np.asarray(buffer_state.has_reward)
    next_idx = np.asarray(buffer_state.next_idx)
    num_envs, capacity = populated.shape
    mask = np.zeros_like(populated)
    for env in range(num_envs):
        for age in range(min(window, capacity)):
            i = (next_idx[env] - 1 - age) % capacity
            mask[env, i] = populated[env, i] and has_reward[env, i]
    return mask


@pytest.fixture(scope="module")
def windowed(ttt, tmp_path_factory):
    """A 3-epoch run whose replay window follows `WINDOWS`, recording what it logged, the buffer
    each epoch trained on, which entries training could sample and which it sampled, and how many
    times it traced a training step."""
    trainer = make_trainer(
        ttt, tmp_path_factory.mktemp("ckpt"), replay_window=Schedule(WINDOWS)
    )
    run = SimpleNamespace(trainer=trainer, logged={}, buffers=[], sampled=[], traces=0)

    train_step = trainer.train_step

    def counting_train_step(ts, batch):
        # runs once each time training is traced
        run.traces += 1
        return train_step(ts, batch)

    buffer = trainer.memory_buffer
    sample_indices = buffer.sample_indices

    def spy_sample_indices(key, mask, sample_size):
        indices = sample_indices(key, mask, sample_size)
        jax.debug.callback(
            lambda m, i: run.sampled.append((np.asarray(m), np.asarray(i))),
            mask,
            indices,
        )
        return indices

    train_steps = trainer.train_steps

    def spy_train_steps(key, collection_state, *args, **kwargs):
        run.buffers.append(collection_state.buffer_state)
        return train_steps(key, collection_state, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(trainer, "train_step", counting_train_step)
        mp.setattr(buffer, "sample_indices", spy_sample_indices)
        mp.setattr(trainer, "train_steps", spy_train_steps)
        mp.setattr(
            trainer,
            "log_metrics",
            lambda metrics, epoch: run.logged.setdefault(epoch, metrics),
        )
        run.out = trainer.train_loop(seed=0, num_epochs=3)
        jax.effects_barrier()
        yield run


def test_replay_window_follows_its_schedule(windowed):
    assert [int(windowed.logged[e]["replay_window"]) for e in range(3)] == [16, 24, 32]
    # the buffer's metrics count the window's samples, the ones training samples from
    for epoch, (buffer_state, (_, window)) in enumerate(
        zip(windowed.buffers, WINDOWS, strict=True)
    ):
        assert (
            windowed.logged[epoch]["buffer_samples"]
            == window_mask(buffer_state, window).sum()
        )
    assert windowed.trainer.get_config()["replay_window"] == [
        {"epoch": epoch, "value": window} for epoch, window in WINDOWS
    ]


def test_training_samples_only_from_the_replay_window(windowed):
    # one training step per epoch
    assert len(windowed.sampled) == len(windowed.buffers) == 3
    for (mask, indices), buffer_state, (_, window) in zip(
        windowed.sampled, windowed.buffers, WINDOWS, strict=True
    ):
        expected = window_mask(buffer_state, window)
        np.testing.assert_array_equal(mask, expected)
        assert expected.reshape(-1)[indices].all()
    # the window left out finished episodes in the first epoch
    first = windowed.buffers[0]
    assert window_mask(first, 16).sum() < (first.populated & first.has_reward).sum()


def test_changing_the_replay_window_does_not_recompile(windowed):
    trainer = windowed.trainer
    # three windows, one trace of training, and one of the buffer's metrics
    assert windowed.traces == 1
    assert trainer.count_distinct_observations._cache_size() == 1

    def train(num_steps, window):
        trainer.train_epoch(
            jax.random.PRNGKey(0),
            windowed.out.collection_state.buffer_state,
            windowed.out.train_state,
            num_steps,
            jnp.array(window, dtype=jnp.int32),
        )

    train(1, 20)
    assert windowed.traces == 1
    # what does recompile is counted: a different number of steps
    train(2, 20)
    assert windowed.traces == 2


@pytest.mark.parametrize(
    "replay_window", [0, 33, Schedule([(0, 8), (5, 64)]), Schedule([(0, 0)])]
)
def test_replay_window_must_fit_the_buffer(ttt, tmp_path, replay_window):
    with pytest.raises(ValueError, match="replay windows must be between 1 and"):
        make_trainer(ttt, tmp_path, replay_window=replay_window)


def test_train_steps_with_too_few_samples_in_the_window_raises(ttt, tmp_path, trained):
    # each environment's newest entry, at most: fewer than a minibatch of 4
    trainer = make_trainer(ttt, tmp_path, replay_window=1)

    with pytest.raises(ValueError, match="can be sampled"):
        trainer.train_steps(
            jax.random.PRNGKey(0),
            trained.collection_state,
            trainer.init_train_state(),
            1,
        )
