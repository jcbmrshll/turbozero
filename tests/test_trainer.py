"""Tests of the Trainer.

Most tests call one Trainer method on small hand-built states, or run the training loop with
self-play, training and testing replaced by fakes that record their calls (see `fake_loop`), so
that they compile nothing. A few compile the real thing on the smallest configuration that shows
what they test: self-play following the evaluator it's given, tree positions, and (on one shared
trainer, see `real`) a whole run, its exact resumption, and the donation of the replay buffers."""

import fnmatch
import gc
import itertools
import json
import os
import warnings
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from seaglass import Monitor
from seaglass.bundle import read_bundle

import core.training.train as train_module
from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import BaseExperience, EpisodeReplayBuffer
from core.monitoring import VIEWS
from core.networks.azresnet import batch_norm
from core.networks.utils import apply_nn
from core.testing.tester import BaseTester
from core.testing.two_player_tester import TwoPlayerTester
from core.training.loss_fns import az_default_loss_fn
from core.training.schedule import EvaluatorSchedule, describe
from core.training.train import (
    CollectionStats,
    Trainer,
    TrainLoopOutput,
    checkpoint_epochs,
    extract_params,
    read_state_meta,
    state_epochs,
    state_path,
)
from core.training.tree_positions import TreePositions, select_nodes

# self-play steps per epoch, and of warmup: more than a game of the test environment's 3 moves, so
# warmup finishes episodes for the first epoch's training
STEPS = 4

TREE_POSITIONS = TreePositions(per_move=2, min_visits=2, capacity=16)


@pytest.fixture(scope="module")
def env(fixed_length_env):
    """A two-player environment whose games last 3 moves and pay (1, -1), with 2 actions."""
    return fixed_length_env(length=3)


class MLP(eqx.Module):
    """A small network without BatchNorm: its outputs for a batch don't depend on the batch's
    other samples."""

    hidden: eqx.nn.Linear
    policy: eqx.nn.Linear
    value: eqx.nn.Linear

    def __init__(self, in_size: int, num_actions: int, *, key: jax.Array):
        keys = jax.random.split(key, 3)
        self.hidden = eqx.nn.Linear(in_size, 8, key=keys[0])
        self.policy = eqx.nn.Linear(8, num_actions, key=keys[1])
        self.value = eqx.nn.Linear(8, 1, key=keys[2])

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jax.nn.relu(self.hidden(x.reshape(-1).astype(jnp.float32)))
        return self.policy(h), jnp.tanh(self.value(h))


class NormMLP(eqx.Module):
    """A small network with BatchNorm, whose statistics are state that training updates."""

    hidden: eqx.nn.Linear
    norm: eqx.nn.BatchNorm
    policy: eqx.nn.Linear
    value: eqx.nn.Linear

    def __init__(self, in_size: int, num_actions: int, *, key: jax.Array):
        keys = jax.random.split(key, 3)
        self.hidden = eqx.nn.Linear(in_size, 8, key=keys[0])
        self.norm = batch_norm(8)
        self.policy = eqx.nn.Linear(8, num_actions, key=keys[1])
        self.value = eqx.nn.Linear(8, 1, key=keys[2])

    def __call__(self, x: jax.Array, state: eqx.nn.State):
        h, state = self.norm(self.hidden(x.reshape(-1).astype(jnp.float32)), state)
        h = jax.nn.relu(h)
        return (self.policy(h), jnp.tanh(self.value(h))), state


class TaggedEvaluator(Evaluator):
    """Plays the first legal move. Its value is `tag`, so the experiences it collects show which
    evaluator collected them; its state is `size` zeros, so evaluators can differ in their
    states' shapes too."""

    def __init__(self, tag: float, size: int = 1):
        super().__init__(discount=-1.0)
        self.tag = tag
        self.size = size

    def init(self, *args, **kwargs):
        return jnp.zeros((self.size,))

    def reset(self, state):
        return state

    def evaluate(self, key, eval_state, env_state, root_metadata, **kwargs):
        action = jnp.argmax(root_metadata.action_mask)
        return EvalOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=jax.nn.one_hot(action, root_metadata.action_mask.shape[-1]),
        )

    def get_value(self, state):
        return jnp.array(self.tag, dtype=jnp.float32)


def tiny_mcts(env, num_iterations=2, max_nodes=4, eval_fn=None, **kwargs):
    """An AlphaZero MCTS evaluator for `env`, by default with a uniform policy and a value of 0."""

    def uniform(state, params, key):  # pylint: disable=unused-argument
        return jnp.zeros((env.num_actions,)), jnp.array(0.0)

    return AlphaZero(MCTS)(
        eval_fn=eval_fn if eval_fn is not None else uniform,
        action_selector=PUCTSelector(),
        branching_factor=env.num_actions,
        max_nodes=max_nodes,
        num_iterations=num_iterations,
        **kwargs,
    )


def make_trainer(env, ckpt_dir, evaluator=None, network=None, **kwargs):
    """A Trainer for `env` that is cheap to compile: by default its self-play plays the first legal
    move (a `TaggedEvaluator`), it trains a small MLP, and it has no testers. `network` is
    (network, state)."""
    net, nn_state = (
        network
        if network is not None
        else (MLP(2, env.num_actions, key=jax.random.PRNGKey(0)), None)
    )
    kwargs = {
        "batch_size": 2,
        "train_batch_size": 4,
        "warmup_steps": STEPS,
        "collection_steps_per_epoch": STEPS,
        "train_steps_per_epoch": 1,
        "memory_buffer": EpisodeReplayBuffer(capacity=8),
        "optimizer": optax.adam(1e-3),
        "max_episode_steps": 8,
        "testers": [],
        **kwargs,
    }
    return Trainer(
        nn=net,
        nn_state=nn_state,
        loss_fn=az_default_loss_fn,
        evaluator=evaluator if evaluator is not None else TaggedEvaluator(0.0),
        env_step_fn=env.step_fn,
        env_init_fn=env.init_fn,
        state_to_nn_input_fn=env.state_to_nn_input,
        ckpt_dir=str(ckpt_dir),
        **kwargs,
    )


# the optimizer of trainers whose states are forked: a learning rate schedule, whose step count a
# fork restarts
def schedule_optimizer():
    return optax.adam(optax.linear_schedule(1e-3, 1e-4, 100))


def optimizer_states(train_state):
    """The states of `schedule_optimizer`'s Adam and learning rate schedule."""
    adam, schedule = cast(tuple, train_state.opt_state)
    return adam, schedule


def tree_trainer(env, ckpt_dir, **kwargs):
    """A trainer that stores tree positions, with `schedule_optimizer`. Its MCTS self-play doesn't
    search unless a test collects with it."""
    return make_trainer(
        env,
        ckpt_dir,
        **{
            "evaluator": tiny_mcts(env),
            "tree_positions": TREE_POSITIONS,
            "optimizer": schedule_optimizer(),
            **kwargs,
        },
    )


def swap_actions(experience):
    """A data transform for the test environment: its two actions swapped."""
    return replace(
        experience,
        policy_mask=experience.policy_mask[::-1],
        policy_weights=experience.policy_weights[::-1],
    )


def leaves_equal(a, b):
    return all(
        np.array_equal(x, y)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


def snapshot(tree):
    """A host copy of `tree`'s arrays, which outlives their donation to self-play (and doesn't
    keep them alive: on the CPU, `np.asarray` would be a view of a jax array's memory)."""
    return jax.tree.map(np.array, tree)


def trained_state(trainer, steps=2):
    """`trainer`'s initial train state after `steps` optimizer updates (with gradients of ones),
    so that its params and optimizer state are no longer the initial ones."""
    ts = trainer.init_train_state()
    for _ in range(steps):
        grads = jax.tree.map(jnp.ones_like, ts.params)
        updates, opt_state = trainer.optimizer.update(grads, ts.opt_state, ts.params)
        ts = replace(
            ts,
            params=optax.apply_updates(ts.params, updates),
            opt_state=opt_state,
            step=ts.step + 1,
        )
    return ts


def written(buffer_state, num_written, in_progress=0):
    """`buffer_state`, empty, after every environment wrote `num_written` entries, the last
    `in_progress` of them of a game still in progress. The i-th entry an environment wrote has
    search value i + 100 * the environment's index, so that entries can be told apart."""
    batch_size, capacity = buffer_state.populated.shape
    populated = np.zeros((batch_size, capacity), dtype=bool)
    has_reward = np.ones((batch_size, capacity), dtype=bool)
    search_value = np.zeros((batch_size, capacity), dtype=np.float32)
    for i in range(num_written):
        place = i % capacity
        populated[:, place] = True
        has_reward[:, place] = i < num_written - in_progress
        search_value[:, place] = i + 100 * np.arange(batch_size)

    def index(i):
        return jnp.full((batch_size,), i % capacity, dtype=jnp.int32)

    return replace(
        buffer_state,
        next_idx=index(num_written),
        episode_start_idx=index(num_written - in_progress),
        buffer=replace(buffer_state.buffer, search_value=jnp.asarray(search_value)),
        populated=jnp.asarray(populated),
        has_reward=jnp.asarray(has_reward),
    )


def save_a_state(trainer):
    """Saves a state of `trainer`'s run after 2 epochs, as `train_loop` would: a replay buffer
    that has wrapped around with a game in progress, a tree position buffer that has wrapped
    around (if the trainer stores tree positions), the train state after 2 updates, and an rng.

    Returns:
        the state, in host arrays, and its path
    """
    state = trainer.init_collection_state(jax.random.PRNGKey(0), trainer.batch_size)
    state = replace(
        state,
        buffer_state=written(state.buffer_state, 11, in_progress=2),
        episodes=jnp.full_like(state.episodes, 3),
        moves=jnp.full_like(state.moves, 11),
    )
    if trainer.tree_buffer is not None:
        tree = written(state.tree_buffer_state, 20)
        state = replace(
            state,
            tree_buffer_state=tree,
            # written one a move, say
            tree_written_at=(tree.buffer.search_value % 100).astype(jnp.int32),
            tree_positions=jnp.full_like(state.tree_positions, 20),
            tree_visits=jnp.full_like(state.tree_visits, 50),
        )
    train_state = trained_state(trainer)
    output = TrainLoopOutput(
        collection_state=state,
        train_state=train_state,
        test_states=[
            t.init(params=extract_params(train_state)) for t in trainer.testers
        ],
        cur_epoch=2,
        key=jax.random.PRNGKey(5),
    )
    return snapshot(output), trainer.save_state(output)


class FakeMonitor(Monitor):
    """Records what the training loop tells its monitor, in host arrays. Starting a run names it
    (a new name each time) unless it has one already."""

    names = itertools.count()

    def __init__(self, run_id: str | None = None):  # pylint: disable=super-init-not-called
        self.run_id = run_id
        self.started, self.totals, self.logged = [], [], []
        self.activities, self.finished = [], []

    def start(self, config=None, total=None):
        if self.run_id is None:
            self.run_id = f"run-{next(self.names)}"
        self.started.append(config)
        self.totals.append(total)
        return self

    def log(self, step, data):
        self.logged.append(
            (
                step,
                {
                    k: np.array(v) if isinstance(v, jax.Array) else v
                    for k, v in data.items()
                },
            )
        )

    def activity(self, text):
        self.activities.append(text)

    def finish(self, status="finished"):
        self.finished.append(status)

    def by_epoch(self):
        """Everything logged, merged by epoch."""
        logged = {}
        for epoch, data in self.logged:
            logged.setdefault(epoch, {}).update(data)
        return logged


class FakeTester(BaseTester):
    """Plays no games: records what each test is given, and reports an outcome of 1. With an
    `episode_fn`, its episode is whatever that returns."""

    def __init__(self, **kwargs):
        super().__init__(num_keys=1, **kwargs)
        self.calls = []

    def test(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, max_steps, env_step_fn, env_init_fn, evaluator, keys, state, params
    ):
        self.calls.append(
            SimpleNamespace(env_init_fn=env_init_fn, evaluator=evaluator, params=params)
        )
        return state, {"outcome": jnp.ones(1)}, None, None


@pytest.fixture
def fake_loop(monkeypatch):
    """`fake_loop(trainer)` replaces `trainer`'s self-play and training with fakes that return at
    once, so its training loop compiles nothing. Self-play counts an episode for every environment
    and the moves it was asked for; training counts a step and adds 1 to every param, and reports
    a loss of 0.

    Returns the calls, in order, as namespaces with the arguments they got: self-play's are
    `kind="collect"`, with `key`, `state`, `params`, `num_steps` and `evaluator`; training's
    `kind="train"`, with `key`, `state` (the collection state), `epoch` and `result`, the train
    state it returned.
    """

    def install(trainer):
        calls = []

        def collect_steps(key, state, params, num_steps, *, evaluator=None):
            calls.append(
                SimpleNamespace(
                    kind="collect",
                    key=key,
                    state=state,
                    params=params,
                    num_steps=num_steps,
                    evaluator=evaluator,
                )
            )
            return replace(
                state, episodes=state.episodes + 1, moves=state.moves + num_steps
            )

        def train_steps(key, collection_state, train_state, num_steps, epoch=0):  # pylint: disable=unused-argument
            result = replace(
                train_state,
                params=jax.tree.map(lambda x: x + 1, train_state.params),
                step=train_state.step + 1,
            )
            calls.append(
                SimpleNamespace(
                    kind="train",
                    key=key,
                    state=collection_state,
                    epoch=epoch,
                    result=result,
                )
            )
            return collection_state, result, {"loss": jnp.array(0.0)}

        monkeypatch.setattr(trainer, "collect_steps", collect_steps)
        monkeypatch.setattr(trainer, "train_steps", train_steps)
        monkeypatch.setattr(
            trainer,
            "collection_metrics",
            lambda before, after: {
                "episodes": (after.episodes - before.episodes).sum()
            },
        )
        return calls

    return install


def of_kind(calls, kind):
    return [c for c in calls if c.kind == kind]


# construction and configuration


@pytest.mark.parametrize("grad_accum_steps", [0, 3])
def test_grad_accum_must_divide_the_minibatch(env, tmp_path, grad_accum_steps):
    with pytest.raises(ValueError, match="must be positive and divide"):
        make_trainer(
            env, tmp_path, train_batch_size=4, grad_accum_steps=grad_accum_steps
        )


def test_a_schedule_needs_a_test_evaluator(env, tmp_path):
    schedule = EvaluatorSchedule([(0, TaggedEvaluator(0.0)), (1, TaggedEvaluator(1.0))])

    with pytest.raises(ValueError, match="evaluator_test"):
        make_trainer(env, tmp_path, evaluator=schedule)


def test_tree_positions_need_mcts_self_play(env, tmp_path):
    with pytest.raises(ValueError, match="needs MCTS"):
        make_trainer(env, tmp_path, tree_positions=TREE_POSITIONS)


def test_config_lists_the_schedule_and_grad_accum_only_when_used(env, tmp_path):
    schedule = EvaluatorSchedule([(0, tiny_mcts(env, 2, 4)), (1, tiny_mcts(env, 6, 8))])
    config = make_trainer(
        env,
        tmp_path,
        evaluator=schedule,
        evaluator_test=tiny_mcts(env),
        grad_accum_steps=2,
    ).get_config()

    assert config["grad_accum_steps"] == 2
    assert config["evaluator_train_config"]["num_iterations"] == 2
    assert [
        (
            stage["epoch"],
            stage["config"]["num_iterations"],
            stage["config"]["max_nodes"],
        )
        for stage in config["collection_schedule"]
    ] == [(0, 2, 4), (1, 6, 8)]
    plain = make_trainer(env, tmp_path).get_config()
    assert "grad_accum_steps" not in plain and "collection_schedule" not in plain


def test_config_records_the_loss_testers_and_injected_hyperparameters(env, tmp_path):
    config = make_trainer(
        env,
        tmp_path,
        optimizer=optax.inject_hyperparams(optax.adam)(
            learning_rate=optax.linear_schedule(1e-3, 0.0, 10)
        ),
        testers=[TwoPlayerTester(num_episodes=2)],
    ).get_config()

    # float32 values, recorded as the decimals they were given as
    assert config["optimizer"] == {
        "learning_rate": 1e-3,
        "b1": 0.9,
        "b2": 0.999,
        "eps": 1e-8,
        "eps_root": 0.0,
    }
    assert config["loss_fn"] == {
        "name": "az_default_loss_fn",
        "value_target_q": 0.0,
    }
    assert config["testers"] == [
        {
            "name": "TwoPlayerTester",
            "class": "TwoPlayerTester",
            "epochs_per_test": 1,
            "num_episodes": 2,
        }
    ]
    assert (config["nn"], config["max_episode_steps"]) == ("MLP", 8)
    # nothing to record of an optimizer whose hyperparameters aren't injected
    assert make_trainer(env, tmp_path).get_config()["optimizer"] is None


# checkpoints


def test_checkpoint_round_trip(env, tmp_path):
    trainer = make_trainer(env, tmp_path)
    ts = trained_state(trainer)

    trainer.save_checkpoint(ts, 0)

    assert leaves_equal(trainer.load_train_state_from_checkpoint(str(tmp_path), 0), ts)


@pytest.mark.parametrize(
    "kwargs, epochs, kept",
    [
        # the newest max_checkpoints
        ({}, 4, [2, 3]),
        # multiples of keep_every stay, besides the 2 newest of the rest
        ({"keep_every": 3}, 8, [0, 3, 5, 6, 7]),
    ],
)
def test_old_checkpoints_are_deleted(env, tmp_path, kwargs, epochs, kept):
    trainer = make_trainer(env, tmp_path, **kwargs)
    ts = trainer.init_train_state()

    for epoch in range(epochs):
        trainer.save_checkpoint(ts, epoch)

    assert checkpoint_epochs(str(tmp_path)) == kept


def test_existing_checkpoints_are_kept_and_never_overwritten(env, tmp_path):
    first = make_trainer(env, tmp_path)
    ts = trained_state(first)
    first.save_checkpoint(ts, 0)
    first.save_checkpoint(ts, 1)

    # e.g. a fresh run started in a directory that already holds checkpoints
    second = make_trainer(env, tmp_path)
    assert checkpoint_epochs(str(tmp_path)) == [0, 1]
    with pytest.raises(ValueError, match="already has a checkpoint"):
        second.save_checkpoint(second.init_train_state(), 1)

    assert checkpoint_epochs(str(tmp_path)) == [0, 1]
    assert leaves_equal(second.load_train_state_from_checkpoint(str(tmp_path), 1), ts)


# the training loop, with fake self-play, training and testing


def test_each_epoch_collects_trains_tests_and_saves_a_checkpoint(
    env, tmp_path, fake_loop
):
    monitor = FakeMonitor()
    tester = FakeTester()
    trainer = make_trainer(
        env, tmp_path, warmup_steps=3, testers=[tester], monitor=monitor
    )
    calls = fake_loop(trainer)

    trainer.train_loop(seed=0, num_epochs=2, eval_every=2)

    assert [
        (c.kind, c.num_steps if c.kind == "collect" else c.epoch) for c in calls
    ] == [
        ("collect", 3),
        ("collect", STEPS),
        ("train", 0),
        ("collect", STEPS),
        ("train", 1),
    ]
    # tested every other epoch
    assert len(tester.calls) == 1
    assert monitor.activities == [
        "warmup self-play (3 steps)",
        "epoch 0: self-play",
        "epoch 0: training",
        "epoch 0: testing FakeTester",
        "epoch 0: saving checkpoint",
        "epoch 1: self-play",
        "epoch 1: training",
        "epoch 1: saving checkpoint",
    ]
    assert checkpoint_epochs(str(tmp_path)) == [0, 1]
    assert monitor.finished == ["finished"]


def test_self_play_and_testers_use_the_latest_params(env, tmp_path, fake_loop):
    tester = FakeTester()
    trainer = make_trainer(env, tmp_path, testers=[tester])
    calls = fake_loop(trainer)

    trainer.train_loop(seed=0, num_epochs=3)

    collects, trains = of_kind(calls, "collect"), of_kind(calls, "train")
    initial = trainer.init_train_state().params
    # warmup and epoch 0's self-play play with the initial params
    for c in collects[:2]:
        assert leaves_equal(c.params[0], initial)
    for epoch in range(1, 3):
        assert collects[epoch + 1].params[0] is trains[epoch - 1].result.params, (
            f"epoch {epoch}'s self-play did not use the params trained in epoch {epoch - 1}"
        )
    # each epoch's test, the params it trained
    for epoch in range(3):
        assert tester.calls[epoch].params[0] is trains[epoch].result.params


def test_testers_play_from_the_test_env_init_fn(env, tmp_path, fake_loop):
    tester = FakeTester()
    evaluator_test = TaggedEvaluator(1.0)

    def test_init_fn(key):
        return env.init_fn(key)

    trainer = make_trainer(
        env,
        tmp_path,
        testers=[tester],
        evaluator_test=evaluator_test,
        test_env_init_fn=test_init_fn,
    )
    fake_loop(trainer)

    trainer.train_loop(seed=0, num_epochs=1)

    (call,) = tester.calls
    assert call.env_init_fn is test_init_fn and call.evaluator is evaluator_test
    # by default, self-play's
    assert make_trainer(env, tmp_path).test_env_init_fn is env.init_fn


def test_train_loop_reports_to_the_monitor(env, tmp_path, fake_loop):
    monitor = FakeMonitor()
    tester = FakeTester(episode_fn=lambda frames, p_ids, env_step_fn: "episode")
    trainer = make_trainer(
        env,
        tmp_path,
        testers=[tester],
        monitor=monitor,
        extra_config={"note": "hello"},
    )
    fake_loop(trainer)

    trainer.train_loop(seed=3, num_epochs=2)

    (config,) = monitor.started
    assert config["run"] == {"seed": 3, "num_epochs": 2, "eval_every": 1}
    # it ends at epoch 1
    assert monitor.totals == [1]
    assert config["batch_size"] == 2 and config["note"] == "hello"
    assert "parent" not in config
    logged = monitor.by_epoch()
    for epoch in range(2):
        assert logged[epoch]["loss"] == 0 and logged[epoch]["outcome"] == 1
        # each test's first game, for the monitor server to render
        assert logged[epoch]["FakeTester_game"] == "episode"
    assert monitor.finished == ["finished"]


@pytest.mark.parametrize(
    "error, status",
    [(RuntimeError("boom"), "crashed"), (KeyboardInterrupt(), "stopped")],
)
def test_an_interrupted_train_loop_tells_the_monitor(
    env, tmp_path, fake_loop, monkeypatch, error, status
):
    monitor = FakeMonitor()
    trainer = make_trainer(env, tmp_path, monitor=monitor)
    fake_loop(trainer)

    def interrupt(*args, **kwargs):
        raise error

    monkeypatch.setattr(trainer, "train_steps", interrupt)

    with pytest.raises(type(error)):
        trainer.train_loop(seed=0, num_epochs=1)

    assert monitor.finished == [status]


def capacity(collection_state):
    """The size of a collection state's search trees."""
    return collection_state.eval_state.parents.shape[-1]


def test_self_play_switches_evaluators_on_schedule(env, tmp_path, fake_loop):
    small, big = tiny_mcts(env, 2, 4), tiny_mcts(env, 6, 8)
    monitor = FakeMonitor()
    trainer = make_trainer(
        env,
        tmp_path,
        evaluator=EvaluatorSchedule([(0, small), (1, big)]),
        evaluator_test=small,
        monitor=monitor,
    )
    calls = fake_loop(trainer)

    trainer.train_loop(seed=0, num_epochs=3)

    # warmup, then epochs 0, 1 and 2
    collects = of_kind(calls, "collect")
    assert [c.evaluator for c in collects] == [small, small, big, big]
    # the switch starts the new evaluator's search trees, of its size
    assert [capacity(c.state) for c in collects] == [4, 4, 8, 8]
    assert [monitor.by_epoch()[e]["search_iterations"] for e in range(3)] == [2, 6, 6]
    assert f"epoch 1: self-play switches to {describe(big)} (compiling)" in (
        monitor.activities
    )


@pytest.mark.parametrize("with_key", [True, False])
def test_a_continued_run_follows_the_schedule_from_where_it_stopped(
    env, tmp_path, fake_loop, with_key
):
    small, big = tiny_mcts(env, 2, 4), tiny_mcts(env, 6, 8)
    trainer = make_trainer(
        env,
        tmp_path,
        evaluator=EvaluatorSchedule([(0, small), (1, big)]),
        evaluator_test=small,
    )
    calls = fake_loop(trainer)
    first = trainer.train_loop(seed=0, num_epochs=1)
    calls.clear()
    if not with_key:
        first = replace(first, key=None)

    output = trainer.train_loop(seed=0, num_epochs=3, initial_state=first)

    collects = of_kind(calls, "collect")
    # epochs 1 and 2, with the evaluator scheduled for them; without the rng to continue
    # exactly, a warmup first
    assert [c.evaluator for c in collects] == [big] * (2 if with_key else 3)
    assert capacity(collects[0].state) == 8
    assert [t.epoch for t in of_kind(calls, "train")] == [1, 2]
    assert output.cur_epoch == 3
    assert checkpoint_epochs(str(tmp_path)) == [1, 2]


def test_a_continued_run_gets_the_keys_it_would_have_without_stopping(
    env, tmp_path, fake_loop
):
    straight = make_trainer(env, tmp_path / "straight")
    straight_calls = fake_loop(straight)
    straight.train_loop(seed=0, num_epochs=3)
    stopped = make_trainer(env, tmp_path / "stopped")
    calls = fake_loop(stopped)
    first = stopped.train_loop(seed=0, num_epochs=1)
    calls.clear()

    # the seed is the stopped run's rng's, whatever is passed
    stopped.train_loop(seed=1, num_epochs=3, initial_state=first)

    # after warmup and epoch 0's self-play and training
    expected = straight_calls[3:]
    assert [c.kind for c in calls] == [c.kind for c in expected]
    for call, straight_call in zip(calls, expected, strict=True):
        np.testing.assert_array_equal(call.key, straight_call.key)


def test_states_are_saved_at_and_every(env, tmp_path, fake_loop):
    trainer = make_trainer(env, tmp_path, save_state_at=[3], save_state_every=2)
    fake_loop(trainer)
    assert [e for e in range(1, 9) if trainer.should_save_state(e)] == [2, 3, 4, 6, 8]

    trainer.train_loop(seed=0, num_epochs=6)

    # the newest, and those at save_state_at
    assert state_epochs(str(tmp_path)) == [3, 6]
    meta = read_state_meta(state_path(str(tmp_path), 6))
    assert (meta["epoch"], meta["num_envs"], meta["buffer_capacity"]) == (6, 2, 8)
    assert meta["config"]["batch_size"] == 2 and meta["parent"] is None


# resuming a run from a saved state, and forking new runs from it


def test_resume_loads_the_newest_state_and_drops_the_checkpoints_after_it(
    env, tmp_path
):
    trainer = tree_trainer(
        env, tmp_path, max_checkpoints=4, monitor=FakeMonitor(run_id="the-run")
    )
    trainer.parent = {"path": "elsewhere"}
    for epoch in range(4):
        trainer.save_checkpoint(trainer.init_train_state(), epoch)
    saved, _ = save_a_state(trainer)
    # as in a new process
    resumed = tree_trainer(env, tmp_path, max_checkpoints=4, monitor=FakeMonitor())

    state = resumed.resume()

    # train state, replay buffers, games, search trees, testers' states, rng
    assert jax.tree.all(jax.tree.map(np.array_equal, snapshot(state), saved))
    # the continued run saves its checkpoints from the state on again
    assert checkpoint_epochs(str(tmp_path)) == [0, 1]
    # it goes on logging to the run's monitor run, and records the same parent
    assert resumed.monitor is not None and resumed.monitor.run_id == "the-run"
    assert resumed.parent == {"path": "elsewhere"}


def by_age(buffer_state, *entry_data):
    """Which of a replay buffer's entries can be sampled, and the entries (with `entry_data`'s),
    each environment's newest first."""
    next_idx = np.asarray(buffer_state.next_idx)
    capacity = buffer_state.populated.shape[1]
    index = (next_idx[:, None] - 1 - np.arange(capacity)) % capacity

    def take(x):
        x = np.asarray(x)
        return np.take_along_axis(
            x, index.reshape(index.shape + (1,) * (x.ndim - 2)), 1
        )

    sampleable = take(
        np.asarray(buffer_state.populated) & np.asarray(buffer_state.has_reward)
    )
    return sampleable, jax.tree.map(take, (buffer_state.buffer, *entry_data))


def test_fork_starts_from_the_saved_network_optimizer_and_buffers(env, tmp_path):
    saved, path = save_a_state(tree_trainer(env, tmp_path / "trunk"))
    fork = tree_trainer(env, tmp_path / "fork")
    key = jax.random.PRNGKey(7)

    collection_state, train_state = fork.fork_state(path, key)

    assert leaves_equal(extract_params(train_state), extract_params(saved.train_state))
    # Adam's moments (and its bias correction's step count) carry on; the learning rate
    # schedule starts over
    adam, schedule = optimizer_states(train_state)
    saved_adam, saved_schedule = optimizer_states(saved.train_state)
    assert leaves_equal(adam, saved_adam) and int(adam.count) == 2
    assert int(saved_schedule.count) == 2 and int(schedule.count) == 0
    assert int(train_state.step) == 0
    # the replay buffer as saved, without the entries of the games that were in progress
    buffer, saved_buffer = (
        collection_state.buffer_state,
        saved.collection_state.buffer_state,
    )
    assert not saved_buffer.has_reward.all()
    assert leaves_equal(buffer.buffer, saved_buffer.buffer)
    np.testing.assert_array_equal(
        buffer.populated, saved_buffer.populated & saved_buffer.has_reward
    )
    np.testing.assert_array_equal(buffer.next_idx, saved_buffer.episode_start_idx)
    assert np.asarray(buffer.has_reward).all()
    # the tree positions as saved, and the move counts their ages are measured by
    assert leaves_equal(
        collection_state.tree_buffer_state, saved.collection_state.tree_buffer_state
    )
    for name in ("tree_written_at", "moves"):
        np.testing.assert_array_equal(
            getattr(collection_state, name), getattr(saved.collection_state, name)
        )
    # new games, and search trees for the fork's evaluator
    env_state, metadata, eval_state = fork.init_games(
        key, fork.batch_size, fork.collection_schedule.at(0)
    )
    assert leaves_equal(
        (
            collection_state.env_state,
            collection_state.metadata,
            collection_state.eval_state,
        ),
        (env_state, metadata, eval_state),
    )


@pytest.mark.parametrize("capacity", [4, 16])
def test_fork_resizes_the_replay_buffers(env, tmp_path, capacity):
    saved, path = save_a_state(tree_trainer(env, tmp_path / "trunk"))
    fork = tree_trainer(
        env,
        tmp_path / "fork",
        memory_buffer=EpisodeReplayBuffer(capacity=capacity),
        tree_positions=replace(TREE_POSITIONS, capacity=2 * capacity),
    )

    loaded, _ = fork.fork_state(path, jax.random.PRNGKey(0))

    saved = saved.collection_state
    pairs = [
        # without the games in progress
        (
            by_age(
                replace(
                    saved.buffer_state, next_idx=saved.buffer_state.episode_start_idx
                )
            ),
            by_age(loaded.buffer_state),
            min(8, capacity),
        ),
        (
            by_age(saved.tree_buffer_state, saved.tree_written_at),
            by_age(loaded.tree_buffer_state, loaded.tree_written_at),
            min(16, 2 * capacity),
        ),
    ]
    for (saved_ok, saved_data), (ok, data), kept in pairs:
        # each environment's newest entries, as many as fit, at the same ages
        np.testing.assert_array_equal(ok[:, :kept], saved_ok[:, :kept])
        assert not ok[:, kept:].any()
        for x, y in zip(
            jax.tree.leaves(data), jax.tree.leaves(saved_data), strict=True
        ):
            np.testing.assert_array_equal(
                x[:, :kept][ok[:, :kept]], y[:, :kept][ok[:, :kept]]
            )
        if kept < saved_ok.shape[1]:
            assert ok.sum() < saved_ok.sum()
        else:
            assert ok.sum() == saved_ok.sum()
    assert np.asarray(loaded.buffer_state.next_idx).tolist() == [0, 0]


def test_fork_without_replay_buffers(env, tmp_path):
    saved, path = save_a_state(tree_trainer(env, tmp_path / "trunk"))
    # the number of environments needn't match without them
    fork = tree_trainer(env, tmp_path / "fork", batch_size=3)

    collection_state, train_state = fork.fork_state(
        path, jax.random.PRNGKey(0), replay_buffers=False
    )

    assert leaves_equal(extract_params(train_state), extract_params(saved.train_state))
    assert int(optimizer_states(train_state)[1].count) == 0
    assert collection_state.episodes.shape == (3,)
    assert not np.asarray(collection_state.buffer_state.populated).any()


def test_fork_from_a_run_with_or_without_tree_positions(env, tmp_path):
    _, with_trees = save_a_state(tree_trainer(env, tmp_path / "with"))
    _, without_trees = save_a_state(make_trainer(env, tmp_path / "without"))

    # a fork without tree positions leaves them out
    fork = make_trainer(env, tmp_path / "fork-without", optimizer=schedule_optimizer())
    collection_state, _ = fork.fork_state(with_trees, jax.random.PRNGKey(0))
    assert collection_state.tree_buffer_state is None
    assert collection_state.tree_written_at is None
    # a fork with them, from a run without, starts with none stored
    fork = tree_trainer(env, tmp_path / "fork-with", optimizer=optax.adam(1e-3))
    collection_state, _ = fork.fork_state(without_trees, jax.random.PRNGKey(0))
    assert collection_state.tree_buffer_state is not None
    assert not np.asarray(collection_state.tree_buffer_state.populated).any()
    assert np.asarray(collection_state.buffer_state.populated).any()


def test_a_fork_records_its_parent(env, tmp_path, fake_loop):
    saved, path = save_a_state(tree_trainer(env, tmp_path / "trunk"))
    monitor = FakeMonitor()
    fork = tree_trainer(env, tmp_path / "fork", monitor=monitor, save_state_at=[1])
    calls = fake_loop(fork)

    # a directory: its newest state
    fork.train_loop(seed=1, num_epochs=1, fork_from=str(tmp_path / "trunk"))

    parent = {
        "path": os.path.abspath(path),
        "epoch": 2,
        "replay_buffers": True,
        "monitor_run": None,
        "parent": None,
    }
    assert monitor.started[0]["parent"] == parent
    # it trains on the saved replay buffer, without the games that were in progress
    saved_buffer = saved.collection_state.buffer_state
    np.testing.assert_array_equal(
        of_kind(calls, "train")[0].state.buffer_state.populated,
        saved_buffer.populated & saved_buffer.has_reward,
    )
    # its own saved states record it, for forks of the fork
    meta = read_state_meta(state_path(fork.ckpt_dir, 1))
    assert (meta["parent"], meta["monitor_run"]) == (parent, monitor.run_id)
    # a resumed fork goes on logging to its monitor run, and records the same parent
    resumed = tree_trainer(env, tmp_path / "fork", monitor=FakeMonitor())
    fake_loop(resumed)
    resumed.train_loop(seed=1, num_epochs=2, initial_state=resumed.resume())
    assert resumed.monitor is not None and resumed.monitor.run_id == monitor.run_id
    assert resumed.parent == parent


@pytest.mark.parametrize(
    "method, kwargs, match",
    [
        (
            "resume",
            {"batch_size": 3},
            "saved with 2 self-play environments, this run has 3",
        ),
        (
            "resume",
            {"memory_buffer": EpisodeReplayBuffer(capacity=16)},
            "replay buffer capacity of 8, this run's is 16",
        ),
        (
            "resume",
            {"tree_positions": None, "evaluator": TaggedEvaluator(0.0)},
            "tree position buffer capacity of 16, this run's is None",
        ),
        # search trees of a different size
        (
            "resume",
            {"max_nodes": 8},
            "the saved collection state doesn't match this run's",
        ),
        ("resume", {"optimizer": optax.sgd(1e-3)}, "the saved train state has"),
        (
            "fork",
            {"batch_size": 3},
            "saved with 2 self-play environments, this run has 3",
        ),
        ("fork", {"optimizer": optax.sgd(1e-3)}, "the saved train state has"),
    ],
)
def test_saved_states_must_fit_the_trainer(env, tmp_path, method, kwargs, match):
    _, path = save_a_state(tree_trainer(env, tmp_path / "trunk"))
    if "max_nodes" in kwargs:
        kwargs = {"evaluator": tiny_mcts(env, max_nodes=kwargs["max_nodes"])}
    trainer = tree_trainer(env, tmp_path / "other", **kwargs)

    with pytest.raises(ValueError, match=match):
        if method == "resume":
            trainer.resume(path)
        else:
            trainer.fork_state(path, jax.random.PRNGKey(0))


# self-play data collection


@pytest.mark.parametrize(
    "rewards, max_episode_steps, episodes, draws, first_player",
    [
        ((1.0, -1.0), 3, 2, 0.0, 2.0),
        ((0.0, 0.0), 3, 2, 2.0, 0.0),
        # truncated episodes aren't counted
        ((0.0, 0.0), 2, 0, 0.0, 0.0),
    ],
)
def test_collect_counts_terminated_episodes_and_sums_their_metrics(
    fixed_length_env,
    scripted,
    make_collector,
    tmp_path,
    rewards,
    max_episode_steps,
    episodes,
    draws,
    first_player,
):
    env = fixed_length_env(length=3, rewards=rewards)
    collect = make_collector(
        env,
        scripted.first_legal,
        max_episode_steps,
        tmp_path,
        episode_metrics_fn=lambda r: {"draw": (r == 0).all(), "first": r[0]},
    )

    state = collect(num_steps=7)

    assert int(state.episodes[0]) == episodes
    assert float(state.episode_metrics["draw"][0]) == draws
    assert float(state.episode_metrics["first"][0]) == first_player


def test_without_episode_metrics_fn_nothing_is_tracked(
    fixed_length_env, scripted, make_collector, tmp_path
):
    env = fixed_length_env(length=3, rewards=(0.0, 0.0))
    collect = make_collector(env, scripted.first_legal, 3, tmp_path)

    assert collect(num_steps=7).episode_metrics == {}


class CountingEvaluator(Evaluator):
    """Plays the first legal action; its state counts its evaluations this episode, and is its value."""

    def __init__(self):
        super().__init__(discount=-1.0)

    def init(self, *args, **kwargs):
        return jnp.array(0, dtype=jnp.int32)

    def reset(self, state):
        return jnp.zeros_like(state)

    def evaluate(self, key, eval_state, env_state, root_metadata, **kwargs):
        action = jnp.argmax(root_metadata.action_mask)
        return EvalOutput(
            eval_state=eval_state + 1,
            action=action,
            policy_weights=jax.nn.one_hot(action, root_metadata.action_mask.shape[-1]),
        )

    def get_value(self, state):
        return state.astype(jnp.float32)


def test_collect_stores_the_searchs_root_value(
    fixed_length_env, make_collector, tmp_path
):
    env = fixed_length_env(length=3)
    collect = make_collector(env, CountingEvaluator(), 10, tmp_path)

    buffer_state = collect(num_steps=4).buffer_state

    # the value as the evaluation left it: not after the evaluator stepped, nor after it was reset
    # at the end of the episode
    np.testing.assert_array_equal(
        buffer_state.buffer.search_value[0, :4], [1.0, 2.0, 3.0, 1.0]
    )


def test_collect_steps_runs_the_evaluator_it_is_given(env, tmp_path):
    # evaluators whose states have the same shapes: if self-play kept running the first one's
    # compiled code, only the values it stores would show it
    first, second = TaggedEvaluator(1.0), TaggedEvaluator(2.0)
    trainer = make_trainer(
        env,
        tmp_path,
        evaluator=EvaluatorSchedule([(0, first), (1, second)]),
        evaluator_test=first,
    )
    state = trainer.init_collection_state(jax.random.PRNGKey(0), trainer.batch_size)
    params = extract_params(trainer.init_train_state())
    keys = jax.random.split(jax.random.PRNGKey(1), trainer.batch_size)

    for evaluator in (first, second):
        state = trainer.collect_steps(keys, state, params, 1, evaluator=evaluator)

    np.testing.assert_array_equal(
        state.buffer_state.buffer.search_value[:, :2], [[1.0, 2.0], [1.0, 2.0]]
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BudgetState:
    """`BudgetEvaluator`'s state: the cycles its search takes, and how many it has run."""

    budget: jax.Array
    searched: jax.Array


class BudgetEvaluator(Evaluator):
    """Plays the first legal move after a search of as many cycles as its state's budget (at most
    3), each of which only counts itself. Its value is the cycles the search ran, so the experiences
    it collects show how long each move searched."""

    def __init__(self):
        super().__init__(discount=-1.0)

    def init(self, *args, **kwargs):
        return BudgetState(budget=jnp.array(3), searched=jnp.array(0))

    def reset(self, state):
        return replace(state, searched=jnp.zeros_like(state.searched))

    def step(self, state, action):
        return self.reset(state)

    def search_cycles(self):
        return 3

    def search_done(self, state, cycles):
        return cycles >= state.budget

    def search_cycle(self, key, state, cycle, *args, **kwargs):
        return replace(state, searched=state.searched + 1)

    def evaluate(self, key, eval_state, env_state, root_metadata, **kwargs):
        action = jnp.argmax(root_metadata.action_mask)
        return EvalOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=jax.nn.one_hot(action, root_metadata.action_mask.shape[-1]),
        )

    def get_value(self, state):
        return state.searched.astype(jnp.float32)


def budget_collection(env, tmp_path, budgets, **kwargs):
    """A trainer with a `BudgetEvaluator`, and a collection state whose environments' searches
    take `budgets` cycles."""
    trainer = make_trainer(env, tmp_path, evaluator=BudgetEvaluator(), **kwargs)
    state = trainer.init_collection_state(jax.random.PRNGKey(0), trainer.batch_size)
    eval_state = replace(state.eval_state, budget=jnp.array(budgets))
    return trainer, replace(state, eval_state=eval_state)


@pytest.mark.parametrize(
    "advance_every, moves, searches, in_progress",
    [
        # moves after 1 cycle and after 3, as soon as their searches are done
        (1, [6, 2], [[1] * 6, [3] * 2], [0, 0]),
        # by default, every whole search's 3 cycles: in step, the shorter search going on
        (None, [2, 2], [[3] * 2, [3] * 2], [0, 0]),
        # every 2 cycles: a 1-cycle search goes on for a cycle until its move is played, a
        # 3-cycle one for one too, and the second's next search has run 2 cycles when the 6 are up
        (2, [3, 1], [[2] * 3, [4]], [0, 2]),
    ],
)
def test_each_environment_plays_its_move_when_its_search_is_done(
    env, tmp_path, advance_every, moves, searches, in_progress
):
    trainer, state = budget_collection(
        env, tmp_path, [1, 3], advance_every=advance_every
    )
    keys = jax.random.split(jax.random.PRNGKey(1), trainer.batch_size)

    # 2 moves of the longest search: 6 cycles
    state = trainer.collect_steps(keys, state, None, 2)

    np.testing.assert_array_equal(state.moves, moves)
    np.testing.assert_array_equal(state.search_cycles, in_progress)
    # a move each 3-move game, and nothing else written
    np.testing.assert_array_equal(state.buffer_state.next_idx, moves)
    np.testing.assert_array_equal(state.episodes, np.array(moves) // 3)
    for i, searched in enumerate(searches):
        np.testing.assert_array_equal(
            state.buffer_state.buffer.search_value[i, : len(searched)], searched
        )


def test_advance_leaves_an_environment_whose_search_is_not_done_as_it_is(env, tmp_path):
    trainer, state = budget_collection(env, tmp_path, [2, 2])
    state = replace(state, search_cycles=jnp.array([1, 1]))

    advanced = jax.vmap(lambda s: trainer.advance(s, None))(state)

    jax.tree.map(np.testing.assert_array_equal, advanced, state)


def test_each_move_starting_in_a_call_takes_the_calls_next_key(env, tmp_path):
    trainer, state = budget_collection(env, tmp_path, [1, 3], advance_every=1)
    keys = jax.random.split(jax.random.PRNGKey(1), trainer.batch_size)

    state = trainer.collect_steps(keys, state, None, 2)

    # of the 6 cycles' keys, the first's 6th move took the 6th, and the second's 2nd the 2nd
    np.testing.assert_array_equal(state.move_key[0], jax.random.split(keys[0], 6)[5])
    np.testing.assert_array_equal(state.move_key[1], jax.random.split(keys[1], 6)[1])


def test_collect_stores_each_position_once(env, tmp_path):
    trainer = tree_trainer(
        env,
        tmp_path,
        evaluator=tiny_mcts(env, num_iterations=4, max_nodes=8),
        data_transform_fns=[swap_actions],
    )
    state = trainer.init_collection_state(jax.random.PRNGKey(0), trainer.batch_size)
    keys = jax.random.split(jax.random.PRNGKey(1), trainer.batch_size)

    state = trainer.collect_steps(keys, state, None, 6)

    # the 6 moves' played positions, of two finished games of 3 moves, without transformed copies
    np.testing.assert_array_equal(state.moves, [6, 6])
    played = np.asarray(trainer.memory_buffer.sample_mask(state.buffer_state))
    np.testing.assert_array_equal(played.sum(axis=-1), [6, 6])
    tree_buffer, tree_state = trainer.tree_buffer, state.tree_buffer_state
    assert tree_buffer is not None and tree_state is not None
    sampleable = np.asarray(tree_buffer.sample_mask(tree_state))
    assert int(state.tree_positions.sum()) > 0
    assert sampleable.sum() == int(state.tree_positions.sum())
    # each stored after one of the 6 moves
    written_at = np.asarray(state.tree_written_at)[sampleable]
    assert ((written_at >= 0) & (written_at < 6)).all()
    samples = jax.tree.map(lambda x: np.asarray(x)[sampleable], tree_state.buffer)
    np.testing.assert_allclose(samples.policy_weights.sum(-1), 1.0, rtol=1e-6)
    assert not samples.policy_weights[~samples.policy_mask].any()
    # the targets of the player to move, and of the other
    rows = np.arange(len(samples.reward))
    np.testing.assert_allclose(
        samples.reward[rows, samples.cur_player_id], samples.search_value
    )
    np.testing.assert_allclose(
        samples.reward[rows, 1 - samples.cur_player_id], -samples.search_value
    )


@pytest.mark.parametrize(
    "discarded_only, persist_tree, excluded",
    [(False, True, False), (True, True, True), (True, False, False)],
)
def test_discarded_only_leaves_out_the_subtree_the_next_search_reuses(
    env, tmp_path, monkeypatch, discarded_only, persist_tree, excluded
):
    exclusions = []

    def spy_select_nodes(*args):
        # (key, tree, per_move, min_visits, most_visited, exclude_subtree)
        exclusions.append(args[5] is not None)
        return select_nodes(*args)

    monkeypatch.setattr(train_module, "select_nodes", spy_select_nodes)
    trainer = tree_trainer(
        env,
        tmp_path,
        evaluator=tiny_mcts(env, persist_tree=persist_tree),
        tree_positions=replace(TREE_POSITIONS, discarded_only=discarded_only),
    )
    state = trainer.init_collection_state(jax.random.PRNGKey(0), trainer.batch_size)
    keys = jax.random.split(jax.random.PRNGKey(1), trainer.batch_size)

    # tracing is enough to see what self-play passes
    jax.eval_shape(jax.vmap(lambda key, s: trainer.collect(key, s, None)), keys, state)

    assert exclusions == [excluded]


def test_collection_metrics(env, tmp_path):
    trainer = tree_trainer(
        env, tmp_path, episode_metrics_fn=lambda r: {"draw": (r == 0).all()}
    )
    state = trainer.init_collection_state(jax.random.PRNGKey(0), trainer.batch_size)
    before = CollectionStats.of(state)
    # 3 finished entries in each environment, with observations (0, 0), (0, 0) and (1, 0) in the
    # first, and (0, 0) and (2, 0) in the second, whose third is of a game in progress: 5 samples,
    # 3 of them distinct
    buffer_state = written(state.buffer_state, 3)
    observations = jnp.zeros_like(buffer_state.buffer.observation_nn)
    observations = observations.at[0, 2, 0].set(1.0).at[1, 1, 0].set(2.0)
    buffer_state = replace(
        buffer_state,
        buffer=replace(buffer_state.buffer, observation_nn=observations),
        has_reward=buffer_state.has_reward.at[1, 2].set(False),
    )
    after = replace(
        state,
        buffer_state=buffer_state,
        episodes=state.episodes + jnp.array([1, 3]),
        moves=state.moves + jnp.array([3, 5]),
        episode_metrics={"draw": jnp.array([1.0, 0.0])},
        tree_positions=state.tree_positions + jnp.array([2, 4]),
        tree_visits=state.tree_visits + jnp.array([10, 20]),
    )

    metrics = trainer.collection_metrics(before, after)

    assert int(metrics["moves"]) == 8
    assert int(metrics["episodes"]) == 4
    assert float(metrics["episode_draw"]) == pytest.approx(0.25)
    assert int(metrics["buffer_samples"]) == 5
    assert int(metrics["buffer_distinct_positions"]) == 3
    assert float(metrics["buffer_distinct_fraction"]) == pytest.approx(0.6)
    assert int(metrics["tree_positions"]) == 6
    # over the epoch's moves, in every environment
    assert float(metrics["tree_positions_per_move"]) == pytest.approx(6 / 8)
    assert float(metrics["tree_mean_visits"]) == pytest.approx(5.0)
    # without episodes, no episode means
    no_episodes = replace(after, episodes=state.episodes)
    assert "episode_draw" not in trainer.collection_metrics(before, no_episodes)


# training


@pytest.mark.parametrize(
    "num_written, match", [(0, "no episodes have finished"), (3, "can be sampled")]
)
def test_train_steps_needs_a_minibatch_of_finished_entries(
    env, tmp_path, num_written, match
):
    # two environments' 3 entries at most: fewer than a minibatch of 8
    trainer = make_trainer(env, tmp_path, train_batch_size=8)
    state = trainer.init_collection_state(jax.random.PRNGKey(0), trainer.batch_size)
    state = replace(state, buffer_state=written(state.buffer_state, num_written))

    with pytest.raises(ValueError, match=match):
        trainer.train_steps(jax.random.PRNGKey(0), state, trainer.init_train_state(), 1)


@pytest.mark.parametrize("transform_fns, swapped", [([], 0.0), ([swap_actions], 0.5)])
def test_training_maps_each_sample_through_a_random_transform(
    env, tmp_path, monkeypatch, transform_fns, swapped
):
    def spying_loss_fn(nn, nn_state, experience):
        loss, (metrics, nn_state) = az_default_loss_fn(nn, nn_state, experience)
        # the samples whose policy target was swapped
        return loss, (
            {**metrics, "swapped": experience.policy_weights[:, 1].mean()},
            nn_state,
        )

    trainer = make_trainer(env, tmp_path, data_transform_fns=transform_fns)
    monkeypatch.setattr(trainer, "loss_fn", spying_loss_fn)
    state = trainer.init_collection_state(jax.random.PRNGKey(0), trainer.batch_size)
    # stored with the first action's policy target
    buffer_state = written(state.buffer_state, 8)
    policy = jnp.zeros_like(buffer_state.buffer.policy_weights).at[..., 0].set(1.0)
    buffer_state = replace(
        buffer_state, buffer=replace(buffer_state.buffer, policy_weights=policy)
    )

    _, _, metrics = trainer.train_steps(
        jax.random.PRNGKey(0),
        replace(state, buffer_state=buffer_state),
        trainer.init_train_state(),
        50,
    )

    # 200 samples, each swapped with probability 1/2 (or never)
    assert float(metrics["swapped"]) == pytest.approx(swapped, abs=0.12)


def test_tree_position_sampler_fills_the_first_places(env, tmp_path):
    trainer = tree_trainer(env, tmp_path)
    template = trainer.make_template_experience()
    batch = jax.tree.map(lambda x: jnp.stack([x] * trainer.train_batch_size), template)
    tree_buffer = trainer.tree_buffer
    assert tree_buffer is not None
    tree_state = tree_buffer.init(2, template)
    # one tree position, told apart by its search value
    tree_state = replace(
        tree_state,
        buffer=replace(
            tree_state.buffer,
            search_value=tree_state.buffer.search_value.at[1, 5].set(7.0),
        ),
        populated=tree_state.populated.at[1, 5].set(True),
    )

    for num_tree, expected in [(3, [7, 7, 7, 0]), (0, [0, 0, 0, 0])]:
        add = trainer.tree_position_sampler(
            tree_state, tree_buffer.sample_mask(tree_state), jnp.array(num_tree)
        )
        mixed = add(jax.random.PRNGKey(0), batch)
        np.testing.assert_array_equal(mixed.search_value, expected)

    # with no tree positions to sample (and so none asked for), the batch is left as it was
    empty = tree_buffer.init(2, template)
    mixed = trainer.tree_position_sampler(
        empty, tree_buffer.sample_mask(empty), jnp.array(0)
    )(jax.random.PRNGKey(0), batch)
    assert jax.tree.all(jax.tree.map(np.array_equal, mixed, batch))


def test_tree_positions_follow_the_replay_buffers_span_of_collection(
    env, tmp_path, monkeypatch
):
    trainer = tree_trainer(env, tmp_path)
    state = trainer.init_collection_state(jax.random.PRNGKey(0), trainer.batch_size)
    tree_state, written_at = state.tree_buffer_state, state.tree_written_at
    assert tree_state is not None and written_at is not None
    # 20 moves made; six tree positions, stored 17, 16, 10, 5, 1 and 20 moves ago
    state = replace(
        state,
        moves=jnp.full_like(state.moves, 20),
        tree_buffer_state=replace(
            tree_state, populated=tree_state.populated.at[:, :6].set(True)
        ),
        tree_written_at=written_at.at[:, :6].set(jnp.array([3, 4, 10, 15, 19, 0])),
    )
    # a buffer of 16 played positions spans 16 moves, one of 4 spans 4
    for capacity, expected in [(16, [1, 2, 3, 4]), (4, [4])]:
        monkeypatch.setattr(
            trainer, "memory_buffer", EpisodeReplayBuffer(capacity=capacity)
        )
        mask = np.asarray(trainer.tree_sample_mask(state))
        for env_index in range(trainer.batch_size):
            assert np.flatnonzero(mask[env_index]).tolist() == expected


def test_tree_batch_count_decays_and_is_zero_without_tree_positions(
    env, tmp_path, monkeypatch
):
    trainer = tree_trainer(env, tmp_path)
    collection_state = trainer.init_collection_state(
        jax.random.PRNGKey(0), trainer.batch_size
    )
    # a minibatch's worth of finished entries
    collection_state = replace(
        collection_state, buffer_state=written(collection_state.buffer_state, 2)
    )
    calls = []
    monkeypatch.setattr(
        trainer,
        "train_epoch",
        lambda key, buffer_state, train_state, num_steps, *, num_tree, **kwargs: (
            calls.append(int(num_tree)) or (train_state, {})
        ),
    )
    train_state = trainer.init_train_state()
    key = jax.random.PRNGKey(0)

    # no tree positions stored yet
    _, _, metrics = trainer.train_steps(key, collection_state, train_state, 1)
    assert calls[-1] == 0 and metrics["tree_batch_fraction"] == 0

    tree_state, tree_positions = (
        collection_state.tree_buffer_state,
        trainer.tree_positions,
    )
    assert tree_state is not None and tree_positions is not None
    collection_state = replace(
        collection_state,
        tree_buffer_state=replace(
            tree_state, populated=tree_state.populated.at[:, 0].set(True)
        ),
    )
    monkeypatch.setattr(
        trainer, "tree_positions", replace(tree_positions, half_life=1.0)
    )
    for epoch, num_tree in [(0, 2), (1, 1), (10, 0)]:
        trainer.train_steps(key, collection_state, train_state, 1, epoch=epoch)
        assert calls[-1] == num_tree


def random_batch(env, key, size=8):
    """A minibatch of random experiences of the test environment."""
    keys = jax.random.split(key, 5)
    num_actions = env.num_actions
    # the first action is always legal, and illegal ones have no weight
    mask = jax.random.bernoulli(keys[2], 0.8, (size, num_actions)).at[:, 0].set(True)
    logits = jax.random.normal(keys[1], (size, num_actions))
    return BaseExperience(
        reward=jax.random.uniform(keys[0], (size, 2), minval=-1, maxval=1),
        policy_weights=jax.nn.softmax(jnp.where(mask, logits, -jnp.inf)),
        policy_mask=mask,
        observation_nn=jax.random.normal(keys[3], (size, 2)),
        cur_player_id=jax.random.randint(keys[4], (size,), 0, 2),
        search_value=jnp.zeros(size),
    )


def batches(env, count=3):
    return [random_batch(env, jax.random.PRNGKey(i)) for i in range(count)]


def train_steps_on(trainer, batches):
    """The train state and metrics after a training step on each of `batches`, from the start."""
    train_step = jax.jit(trainer.train_step)
    ts = trainer.init_train_state()
    metrics = []
    for batch in batches:
        ts, step_metrics = train_step(ts, batch)
        metrics.append(step_metrics)
    return ts, metrics


def assert_trees_close(a, b, rtol=1e-5, atol=1e-6):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


def test_grad_accum_matches_one_minibatch_without_batch_norm(env, tmp_path):
    def make(**kwargs):
        return make_trainer(
            env, tmp_path, train_batch_size=8, optimizer=optax.sgd(0.1), **kwargs
        )

    whole, whole_metrics = train_steps_on(make(), batches(env))
    accumulated, accumulated_metrics = train_steps_on(
        make(grad_accum_steps=2), batches(env)
    )

    assert not leaves_equal(whole.params, make().init_train_state().params)
    assert_trees_close(accumulated, whole)
    assert_trees_close(accumulated_metrics, whole_metrics)


def test_train_step_reports_the_scheduled_hyperparameters_it_used(env, tmp_path):
    trainer = make_trainer(
        env,
        tmp_path,
        train_batch_size=8,
        optimizer=optax.inject_hyperparams(optax.adam)(
            learning_rate=optax.linear_schedule(1e-3, 0.0, 10)
        ),
    )

    _, metrics = train_steps_on(trainer, batches(env))

    assert [float(m["learning_rate"]) for m in metrics] == [
        pytest.approx(1e-3),
        pytest.approx(9e-4),
        pytest.approx(8e-4),
    ]
    # constant ones are in the config, not the metrics
    assert "b1" not in metrics[0]


def test_grad_accum_updates_batch_norm_statistics_once_per_step(env, tmp_path):
    trainer = make_trainer(
        env,
        tmp_path,
        network=eqx.nn.make_with_state(NormMLP)(
            2, env.num_actions, key=jax.random.PRNGKey(0)
        ),
        train_batch_size=8,
        grad_accum_steps=2,
    )
    ts = trainer.init_train_state()
    batch = random_batch(env, jax.random.PRNGKey(0))

    new_ts, _ = jax.jit(trainer.train_step)(ts, batch)

    # micro-batch i is samples i, i + 2, ..., each from the step's starting statistics
    nn = eqx.combine(ts.params, trainer.nn_static)
    states = [
        apply_nn(nn, ts.nn_state, batch.observation_nn[i::2])[1] for i in range(2)
    ]
    expected = jax.tree.map(
        lambda a, b, old: (a + b) / 2 if jnp.issubdtype(old.dtype, jnp.inexact) else a,
        *states,
        ts.nn_state,
    )
    assert_trees_close(new_ts.nn_state, expected)
    # BatchNorm counts one update
    counters = [
        (x, old)
        for x, old in zip(
            jax.tree.leaves(new_ts.nn_state), jax.tree.leaves(ts.nn_state)
        )
        if not jnp.issubdtype(x.dtype, jnp.inexact)
    ]
    assert counters and all(np.array_equal(x, old + 1) for x, old in counters)


# the real thing: one trainer with every piece real, on the smallest configuration, compiled once

# replay buffer capacities no other test's arrays have, so the real trainer's replay buffers are
# the live arrays of their shapes
REAL_CAPACITY, REAL_TREE_CAPACITY = 37, 41


@pytest.fixture(scope="module")
def real(env, tmp_path_factory):
    """A Trainer with AlphaZero self-play whose search grows from 2 to 3 iterations at epoch 2, a
    network with BatchNorm, gradient accumulation, tree positions with a data transform, a
    TwoPlayerTester, and a (fake) monitor. It saves its state after 2 epochs."""
    network = eqx.nn.make_with_state(NormMLP)(
        2, env.num_actions, key=jax.random.PRNGKey(0)
    )
    eval_fn = make_nn_eval_fn(network[0], env.state_to_nn_input)
    small, big = (tiny_mcts(env, num_iterations=n, eval_fn=eval_fn) for n in (2, 3))
    return make_trainer(
        env,
        tmp_path_factory.mktemp("real"),
        network=network,
        evaluator=EvaluatorSchedule([(0, small), (2, big)]),
        evaluator_test=small,
        testers=[TwoPlayerTester(num_episodes=2)],
        monitor=FakeMonitor(),
        memory_buffer=EpisodeReplayBuffer(capacity=REAL_CAPACITY),
        tree_positions=replace(TREE_POSITIONS, capacity=REAL_TREE_CAPACITY),
        data_transform_fns=[swap_actions],
        grad_accum_steps=2,
        save_state_at=[2],
    )


@pytest.fixture(scope="module")
def real_run(real):
    """A 4-epoch run of the real trainer: a host copy of its output, what it logged by epoch, its
    checkpoints and saved states, and the path of the state it saved after epoch 2."""
    real.monitor = FakeMonitor()
    output = snapshot(real.train_loop(seed=0, num_epochs=4))
    return SimpleNamespace(
        output=output,
        logged=real.monitor.by_epoch(),
        checkpoints=checkpoint_epochs(real.ckpt_dir),
        states=state_epochs(real.ckpt_dir),
        path=state_path(real.ckpt_dir, 2),
    )


def test_a_real_run_trains_and_reports_its_metrics(real_run):
    output = real_run.output
    assert output.cur_epoch == 4 and int(output.train_state.step) == 4
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(output.train_state.params))
    for epoch in range(4):
        assert {
            "loss",
            "policy_loss",
            "policy_target_entropy",
            "policy_kl",
            "value_loss",
            "weight_norm",
            "update_ratio",
            "TwoPlayerTester_avg_outcome",
            "episodes",
            "search_iterations",
            "buffer_samples",
            "buffer_distinct_positions",
            "buffer_distinct_fraction",
            "tree_positions",
            "tree_positions_per_move",
            "tree_mean_visits",
            "tree_batch_fraction",
            "tree_buffer_samples",
        } <= real_run.logged[epoch].keys()
    assert [int(real_run.logged[e]["search_iterations"]) for e in range(4)] == [
        2,
        2,
        3,
        3,
    ]
    assert real_run.checkpoints == [2, 3] and real_run.states == [2]


def test_the_views_describe_every_metric_a_real_run_logs(real_run):
    layout = json.loads(read_bundle(VIEWS)["layout.json"])
    logged = {name for metrics in real_run.logged.values() for name in metrics}
    undescribed = {
        name
        for name in logged
        if not any(fnmatch.fnmatchcase(name, p) for p in layout["descriptions"])
    }
    assert not undescribed


def test_a_run_resumed_from_its_saved_state_is_bit_identical(
    real, real_run, tmp_path, monkeypatch
):
    monitor = FakeMonitor()
    monkeypatch.setattr(real, "monitor", monitor)
    monkeypatch.setattr(real, "ckpt_dir", str(tmp_path))

    # the seed is the saved run's rng's, whatever is passed
    output = real.train_loop(
        seed=123, num_epochs=4, initial_state=real.resume(real_run.path)
    )

    # train state, replay buffers, games, search trees, testers' states, rng
    assert jax.tree.all(jax.tree.map(np.array_equal, snapshot(output), real_run.output))
    logged = monitor.by_epoch()
    assert logged.keys() == {2, 3}
    for epoch in (2, 3):
        assert logged[epoch].keys() == real_run.logged[epoch].keys()
        for name, value in logged[epoch].items():
            np.testing.assert_array_equal(
                value, real_run.logged[epoch][name], err_msg=f"epoch {epoch}: {name}"
            )
    assert leaves_equal(
        real.load_train_state_from_checkpoint(str(tmp_path), 3),
        real_run.output.train_state,
    )


def test_collection_updates_the_collection_state_in_place(real):
    state = real.init_collection_state(jax.random.PRNGKey(0), real.batch_size)
    keys = jax.random.split(jax.random.PRNGKey(1), real.batch_size)
    params = extract_params(real.init_train_state())
    evaluator = real.collection_schedule.at(0)
    state_bytes = sum(x.nbytes for x in jax.tree.leaves(state))

    for num_steps in (0, STEPS):
        analysis = (
            Trainer.collect_steps.lower(
                real, keys, state, params, num_steps, evaluator=evaluator
            )
            .compile()
            .memory_analysis()
        )
        # every array of the returned state reuses the memory of the given one's
        assert analysis is not None
        assert analysis.alias_size_in_bytes == state_bytes

    stats = CollectionStats.of(state)
    new_state = real.collect_steps(keys, state, params, STEPS, evaluator=evaluator)

    assert all(x.is_deleted() for x in jax.tree.leaves(state))
    assert not any(x.is_deleted() for x in jax.tree.leaves(new_state))
    # the copied stats outlive the donated state
    assert not any(x.is_deleted() for x in jax.tree.leaves(vars(stats)))


def test_train_loop_keeps_one_copy_of_the_replay_buffers(real, tmp_path, monkeypatch):
    monkeypatch.setattr(real, "ckpt_dir", str(tmp_path))
    shapes = [(real.batch_size, REAL_CAPACITY), (real.batch_size, REAL_TREE_CAPACITY)]

    def buffer_bytes(arrays):
        return sum(x.nbytes for x in arrays if x.shape[:2] in shapes)

    # (bytes in one collection state's replay buffers, bytes in all live ones), as each epoch's
    # training starts: after its self-play metrics
    measured = []
    train_steps = real.train_steps

    def spy_train_steps(key, collection_state, *args, **kwargs):
        gc.collect()
        measured.append(
            (
                buffer_bytes(jax.tree.leaves(collection_state)),
                buffer_bytes(jax.live_arrays()),
            )
        )
        return train_steps(key, collection_state, *args, **kwargs)

    monkeypatch.setattr(real, "train_steps", spy_train_steps)

    with warnings.catch_warnings():
        # every donated array is reused
        warnings.filterwarnings("error", message="Some donated buffers were not usable")
        real.train_loop(seed=0, num_epochs=2)

    assert len(measured) == 2
    for one, live in measured:
        assert one > 0
        assert live == one


def test_loaded_states_are_donated_to_collection(real, real_run, tmp_path, monkeypatch):
    monkeypatch.setattr(real, "ckpt_dir", str(tmp_path))
    monkeypatch.setattr(real, "save_checkpoint", lambda *args, **kwargs: None)
    given = []
    collect_steps = real.collect_steps

    def spy_collect_steps(key, state, *args, **kwargs):
        given.append(state)
        return collect_steps(key, state, *args, **kwargs)

    monkeypatch.setattr(real, "collect_steps", spy_collect_steps)

    with warnings.catch_warnings():
        # every array loaded is new, and reused by self-play in place
        warnings.filterwarnings("error", message="Some donated buffers were not usable")
        resumed = real.resume(real_run.path)
        real.train_loop(seed=0, num_epochs=3, initial_state=resumed)
        first_forked = len(given)
        real.train_loop(seed=0, num_epochs=1, fork_from=real_run.path)

    # the loaded states were donated to the first self-play, and nothing read them after; but
    # for the search trees and their searches' cycles, which the schedule's switch at epoch 2
    # replaced
    loaded = replace(resumed.collection_state, eval_state=None, search_cycles=None)
    assert all(x.is_deleted() for x in jax.tree.leaves(loaded))
    assert given[0].buffer_state is resumed.collection_state.buffer_state
    assert all(x.is_deleted() for x in jax.tree.leaves(given[first_forked]))
