"""Tests of the Trainer.

Most tests call one Trainer method on small hand-built states, or run the training loop with
self-play and training replaced by fakes that record their calls (see `fake_loop`), so that they
compile nothing. A few compile the real thing on the smallest configuration that shows what they
test: self-play's cycles, and (on one shared trainer, see `real`) a whole run, its exact
resumption, and the donation of the replay buffer."""

import fnmatch
import gc
import itertools
import os
import re
import warnings
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import collect_cycle
from seaglass import Monitor
from test_mcts import outer_loop_bodies

import core.training.train as train_module
from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS, PlayoutCap
from core.memory.replay_memory import BaseExperience, EpisodeReplayBuffer
from core.monitoring import DECLARED
from core.networks.azresnet import batch_norm
from core.networks.utils import apply_nn
from core.training.endgame_solve import EndgameSolve, Solved
from core.training.loss_fns import az_default_loss_fn
from core.training.train import (
    CollectionStats,
    Trainer,
    TrainLoopOutput,
    _move_average,
    average_share,
    checkpoint_epochs,
    checkpoint_path,
    injected_hyperparams,
    read_state_meta,
    state_epochs,
    state_path,
)

# self-play steps per epoch: more than a game of the test environment's 3 moves, so that the
# first epoch finishes games to train on
STEPS = 4


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
    evaluator collected them."""

    def __init__(self, tag: float):
        super().__init__(discount=-1.0)
        self.tag = tag

    def init(self, *args, **kwargs):
        return jnp.zeros((1,))

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
    move (a `TaggedEvaluator`) and it trains a small MLP. `network` is (network, state)."""
    net, nn_state = (
        network
        if network is not None
        else (MLP(2, env.num_actions, key=jax.random.PRNGKey(0)), None)
    )
    kwargs = {
        "batch_size": 2,
        "train_batch_size": 4,
        "collection_steps_per_epoch": STEPS,
        "train_steps_per_epoch": 1,
        "memory_buffer": EpisodeReplayBuffer(capacity=8),
        "optimizer": optax.adam(1e-3),
        "max_episode_steps": 8,
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


def norm_trainer(env, ckpt_dir, **kwargs):
    """A trainer of `NormMLP`, a network with BatchNorm statistics."""
    return make_trainer(
        env,
        ckpt_dir,
        network=eqx.nn.make_with_state(NormMLP)(
            2, env.num_actions, key=jax.random.PRNGKey(0)
        ),
        **kwargs,
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


def a_saved_run(trainer, **changes):
    """A state of `trainer`'s run after 2 epochs, as `train_loop` would leave it: a replay
    buffer that has wrapped around with a game in progress, the train state after 2 updates, a
    weight average that isn't its weights, with 3 epochs in it, and an rng."""
    state = trainer.init_collection_state(jax.random.PRNGKey(0))
    state = replace(
        state,
        buffer_state=written(state.buffer_state, 11, in_progress=2),
        episodes=jnp.full_like(state.episodes, 3),
        moves=jnp.full_like(state.moves, 11),
    )
    train_state = trained_state(trainer)
    output = TrainLoopOutput(
        collection_state=state,
        train_state=train_state,
        average=jax.tree.map(lambda x: x + 0.5, train_state.params),
        key=jax.random.PRNGKey(5),
        averaged=3,
        cur_epoch=2,
    )
    return replace(output, **changes)


def save_a_state(trainer, **changes):
    """Saves `a_saved_run`. Returns its host copy and its path."""
    output = a_saved_run(trainer, **changes)
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


def logged(monitor, name):
    """The values `name` was logged with, by epoch."""
    return {
        epoch: float(metrics[name])
        for epoch, metrics in monitor.by_epoch().items()
        if name in metrics
    }


class FakeValidator:
    """Records the weights it scores, and scores them `top1` 1."""

    def __init__(self, name="positions"):
        self.name = name
        self.calls = []

    def __call__(self, params):
        self.calls.append(params)
        return {"top1": jnp.array(1.0)}

    def get_config(self):
        return {"name": self.name}


@pytest.fixture
def fake_loop(monkeypatch):
    """`fake_loop(trainer)` replaces `trainer`'s self-play and training with fakes that return at
    once, so its training loop compiles nothing. Self-play counts an episode for every environment
    and the moves it was asked for; training adds 1 to every param, counts a step, and reports a
    loss of 0. By default the buffer can always train (`can_train`); with `can_train=None`, the
    trainer's own check stands.

    Returns the calls, in order, as namespaces: self-play's `kind="collect"`, with `key`, `state`,
    `params` and `num_steps`; training's `kind="train"`, with `key`, `buffer_state`,
    `num_steps`, `train_state`, the train state it got, and `result`, the one it returned.
    """

    def install(trainer, can_train=True):
        calls = []

        def collect_steps(key, state, params, num_steps):
            calls.append(
                SimpleNamespace(
                    kind="collect",
                    key=key,
                    state=state,
                    params=params,
                    num_steps=num_steps,
                )
            )
            return replace(
                state, episodes=state.episodes + 1, moves=state.moves + num_steps
            )

        def train_epoch(key, buffer_state, train_state, num_steps):
            result = replace(
                train_state,
                params=jax.tree.map(lambda x: x + 1, train_state.params),
                step=train_state.step + 1,
            )
            calls.append(
                SimpleNamespace(
                    kind="train",
                    key=key,
                    buffer_state=buffer_state,
                    num_steps=num_steps,
                    train_state=train_state,
                    result=result,
                )
            )
            return result, {"loss": jnp.array(0.0)}

        monkeypatch.setattr(trainer, "collect_steps", collect_steps)
        monkeypatch.setattr(trainer, "train_epoch", train_epoch)
        monkeypatch.setattr(
            trainer,
            "collection_metrics",
            lambda before, after: {
                "episodes": (after.episodes - before.episodes).sum()
            },
        )
        if can_train is not None:
            monkeypatch.setattr(trainer, "can_train", lambda buffer_state: can_train)
        return calls

    return install


def of_kind(calls, kind):
    return [c for c in calls if c.kind == kind]


def offset_from(params, initial):
    """The one amount every weight of `params` is above `initial`'s (as `fake_loop` trains),
    over the 1-dimensional ones: an average rescales the others to its network's norms (see
    `_move_average`)."""
    gaps = np.concatenate(
        [
            (np.asarray(p, np.float64) - np.asarray(q, np.float64)).ravel()
            for p, q in zip(
                jax.tree.leaves(params), jax.tree.leaves(initial), strict=True
            )
            if np.ndim(p) < 2
        ]
    )
    np.testing.assert_allclose(gaps, gaps[0], atol=1e-5)
    return float(gaps[0])


# construction and configuration


@pytest.mark.parametrize("grad_accum_steps", [0, 3])
def test_grad_accum_must_divide_the_batch(env, tmp_path, grad_accum_steps):
    with pytest.raises(ValueError, match="must be positive and divide"):
        make_trainer(env, tmp_path, grad_accum_steps=grad_accum_steps)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"average_epochs": 0.5}, "average_epochs"),
        ({"average_warmup": 0}, "average_warmup"),
        ({"average_norm_batches": -1}, "average_norm_batches"),
        ({"validate_every": 0}, "validate_every"),
        ({"save_weights_every": 0}, "save intervals"),
        ({"save_state_every": 0}, "save intervals"),
        ({"advance_every": 0}, "advance_every"),
        ({"moves_at_once": 0}, "moves_at_once"),
        ({"holdout_envs": 2, "holdout_positions": 4}, "holdout_envs"),
        ({"holdout_envs": 1, "holdout_positions": 3}, "holdout_positions"),
        ({"holdout_envs": 1, "holdout_positions": 0}, "holdout_positions"),
    ],
)
def test_bad_settings_are_refused(env, tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_trainer(env, tmp_path, **kwargs)


def test_config_records_the_loss_validators_and_injected_hyperparameters(env, tmp_path):
    config = make_trainer(
        env,
        tmp_path,
        optimizer=optax.inject_hyperparams(optax.adam)(learning_rate=1e-3),
        validators=[FakeValidator()],
        average_epochs=4,
        grad_accum_steps=2,
    ).get_config()

    # float32 values, recorded as the decimals they were given as
    assert config["optimizer"] == {
        "learning_rate": 1e-3,
        "b1": 0.9,
        "b2": 0.999,
        "eps": 1e-8,
        "eps_root": 0.0,
    }
    assert config["loss_fn"] == {"name": "az_default_loss_fn"}
    assert config["validators"] == [{"name": "positions"}]
    assert (config["average_epochs"], config["grad_accum_steps"]) == (4.0, 2)
    assert (config["nn"], config["max_episode_steps"]) == ("MLP", 8)
    # nothing to record of an optimizer whose hyperparameters aren't injected
    assert make_trainer(env, tmp_path).get_config()["optimizer"] is None


# the training loop, with fake self-play and training


def test_each_epoch_averages_plays_trains_validates_and_saves(env, tmp_path, fake_loop):
    monitor = FakeMonitor()
    validator = FakeValidator()
    trainer = make_trainer(
        env,
        tmp_path,
        validators=[validator],
        validate_every=2,
        save_weights_every=2,
        monitor=monitor,
    )
    calls = fake_loop(trainer)

    trainer.train_loop(seed=0, num_epochs=3)

    assert [c.kind for c in calls] == ["collect", "train"] * 3
    assert [c.num_steps for c in calls] == [STEPS, 1] * 3
    assert monitor.activities == [
        "epoch 0: self-play",
        "epoch 0: training",
        "epoch 0: validation",
        "epoch 0: saving weights",
        "epoch 1: self-play",
        "epoch 1: training",
        "epoch 2: self-play",
        "epoch 2: training",
        "epoch 2: validation",
        "epoch 2: saving weights",
    ]
    # validated every other epoch, logged under the average's name
    assert logged(monitor, "avg1/top1") == {0: 1.0, 2: 1.0}
    assert len(validator.calls) == 2
    assert checkpoint_epochs(trainer.weights_dir) == [0, 2]
    assert trainer.weights_dir == os.path.join(str(tmp_path), "avg1")
    assert monitor.finished == ["finished"]


@pytest.mark.parametrize(
    "average_epochs, offsets",
    # with a warmup of 1, a 2-epoch average is the mean of the last 2 epochs' weights (training
    # adds 1 an epoch); an average over 1 epoch is the network's weights
    [(2, [0.0, 0.5, 1.25, 2.125]), (1, [0.0, 1.0, 2.0, 3.0])],
)
def test_self_play_and_validation_use_the_average(
    env, tmp_path, fake_loop, average_epochs, offsets
):
    validator = FakeValidator()
    trainer = make_trainer(
        env,
        tmp_path,
        average_epochs=average_epochs,
        average_warmup=1,
        validators=[validator],
    )
    calls = fake_loop(trainer)

    output = trainer.train_loop(seed=0, num_epochs=4)

    initial = trainer.init_train_state().params
    collects = of_kind(calls, "collect")
    assert [offset_from(c.params[0], initial) for c in collects] == pytest.approx(
        offsets
    )
    # each epoch's validation scores the weights its self-play played
    for c, params in zip(collects, validator.calls, strict=True):
        assert params is c.params
    if average_epochs == 1:
        for c, t in zip(collects[1:], of_kind(calls, "train"), strict=False):
            assert leaves_equal(c.params[0], t.result.params)
    # the network's own weights trained on, every epoch
    assert offset_from(output.train_state.params, initial) == pytest.approx(4.0)
    assert offset_from(output.average, initial) == pytest.approx(offsets[-1])
    assert output.averaged == 4


def test_an_epoch_whose_buffer_cant_sample_a_batch_doesnt_train(
    env, tmp_path, fake_loop, capsys
):
    trainer = make_trainer(env, tmp_path)
    calls = fake_loop(trainer, can_train=False)

    output = trainer.train_loop(seed=0, num_epochs=2)

    assert [c.kind for c in calls] == ["collect", "collect"]
    assert leaves_equal(output.train_state, trainer.init_train_state())
    assert "epoch 0: the replay buffer can't sample a batch yet" in (
        capsys.readouterr().out
    )


def test_train_loop_reports_to_the_monitor(env, tmp_path, fake_loop):
    monitor = FakeMonitor()
    trainer = make_trainer(
        env,
        tmp_path,
        validators=[FakeValidator()],
        monitor=monitor,
        extra_config={"note": "hello"},
    )
    fake_loop(trainer)

    trainer.train_loop(seed=3, num_epochs=2)

    (config,) = monitor.started
    assert config["run"] == {"seed": 3, "num_epochs": 2}
    # it ends at epoch 1
    assert monitor.totals == [1]
    assert config["batch_size"] == 2 and config["note"] == "hello"
    assert "parent" not in config
    for epoch in range(2):
        metrics = monitor.by_epoch()[epoch]
        assert metrics["loss"] == 0 and metrics["avg1/top1"] == 1
        assert metrics["episodes"] == 2
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

    monkeypatch.setattr(trainer, "train_epoch", interrupt)

    with pytest.raises(type(error)):
        trainer.train_loop(seed=0, num_epochs=1)

    assert monitor.finished == [status]


def test_a_continued_run_gets_the_keys_it_would_have_without_stopping(
    env, tmp_path, fake_loop
):
    straight = make_trainer(env, tmp_path / "straight", average_epochs=2)
    straight_calls = fake_loop(straight)
    expected = straight.train_loop(seed=0, num_epochs=3)
    stopped = make_trainer(env, tmp_path / "stopped", average_epochs=2)
    calls = fake_loop(stopped)
    first = stopped.train_loop(seed=0, num_epochs=1)
    calls.clear()

    # the seed is the stopped run's rng's, whatever is passed
    output = stopped.train_loop(seed=1, num_epochs=3, initial_state=first)

    # after epoch 0's self-play and training
    assert [c.kind for c in calls] == [c.kind for c in straight_calls[2:]]
    for call, straight_call in zip(calls, straight_calls[2:], strict=True):
        np.testing.assert_array_equal(call.key, straight_call.key)
    assert output.cur_epoch == 3 and output.averaged == expected.averaged == 3
    assert leaves_equal(output.average, expected.average)


def test_states_are_saved_at_and_every(env, tmp_path, fake_loop):
    trainer = make_trainer(env, tmp_path, save_state_at=[3], save_state_every=2)
    fake_loop(trainer)
    assert [e for e in range(1, 9) if trainer.should_save_state(e)] == [2, 3, 4, 6, 8]

    trainer.train_loop(seed=0, num_epochs=6)

    # the newest, and those at save_state_at
    assert state_epochs(str(tmp_path)) == [3, 6]
    meta = read_state_meta(state_path(str(tmp_path), 6))
    assert (meta["epoch"], meta["num_envs"], meta["buffer_capacity"]) == (6, 2, 8)
    assert meta["averaged"] == 6
    assert meta["config"]["batch_size"] == 2 and meta["parent"] is None
    # no weights unless asked for
    assert checkpoint_epochs(trainer.weights_dir) == []


def test_saved_weights_are_the_averages_with_its_own_statistics(
    env, tmp_path, fake_loop, monkeypatch
):
    # its own network state (here a marked one) wherever the average stands in for the
    # network: self-play, validation and its saved weights
    validator = FakeValidator()
    trainer = norm_trainer(
        env,
        tmp_path,
        average_epochs=2,
        validators=[validator],
        save_weights_every=1,
    )
    calls = fake_loop(trainer)

    def marked(key, buffer_state, average):
        return jax.tree.map(
            lambda x: x + 7 if jnp.issubdtype(x.dtype, jnp.inexact) else x,
            trainer.nn_state,
        )

    monkeypatch.setattr(trainer, "average_norm_state", marked)

    trainer.train_loop(seed=0, num_epochs=3)

    template = (trainer.init_train_state().params, trainer.nn_state)
    assert checkpoint_epochs(str(tmp_path / "avg2")) == [0, 1, 2]
    for epoch, c in enumerate(of_kind(calls, "collect")):
        with open(checkpoint_path(str(tmp_path / "avg2"), epoch), "rb") as f:
            saved = eqx.tree_deserialise_leaves(f, template)
        assert leaves_equal(saved, c.params)
        assert validator.calls[epoch] is c.params
        assert leaves_equal(c.params[1], marked(None, None, None))


def test_resume_loads_the_newest_state_and_drops_the_weights_after_it(env, tmp_path):
    trainer = make_trainer(env, tmp_path, monitor=FakeMonitor(run_id="the-run"))
    trainer.parent = {"path": "elsewhere"}
    for epoch in range(4):
        trainer.save_weights(
            (trainer.init_train_state().params, trainer.nn_state), epoch
        )
    saved, _ = save_a_state(trainer)
    # as in a new process
    resumed = make_trainer(env, tmp_path, monitor=FakeMonitor())

    state = resumed.resume()

    # train state, weight average, replay buffer, games, search states, rng
    assert jax.tree.all(jax.tree.map(np.array_equal, snapshot(state), saved))
    # the continued run saves its weights from the state on again
    assert checkpoint_epochs(resumed.weights_dir) == [0, 1]
    # it goes on logging to the run's monitor run, and records the same parent
    assert resumed.monitor is not None and resumed.monitor.run_id == "the-run"
    assert resumed.parent == {"path": "elsewhere"}


@pytest.mark.parametrize(
    "method, kwargs, match",
    [
        ("resume", {"batch_size": 3}, "saved with 2 self-play environments"),
        # an exact continuation needs search states of the same size
        (
            "resume",
            {"max_nodes": 8},
            "the saved collection state doesn't match this run's",
        ),
        ("resume", {"optimizer": optax.sgd(1e-3)}, "the saved train state doesn't"),
        ("branch", {"batch_size": 3}, "saved with 2 self-play environments"),
        ("fork", {"batch_size": 3}, "saved with 2 self-play environments"),
        ("fork", {"optimizer": optax.sgd(1e-3)}, "the saved train state doesn't"),
    ],
)
def test_saved_states_must_fit_the_trainer(env, tmp_path, method, kwargs, match):
    _, path = save_a_state(
        make_trainer(env, tmp_path / "trunk", evaluator=tiny_mcts(env))
    )
    if "max_nodes" in kwargs:
        kwargs = {"evaluator": tiny_mcts(env, max_nodes=kwargs["max_nodes"])}
    trainer = make_trainer(
        env, tmp_path / "other", **{"evaluator": tiny_mcts(env), **kwargs}
    )

    with pytest.raises(ValueError, match=match):
        if method == "resume":
            trainer.resume(path)
        elif method == "branch":
            trainer.branch(path)
        else:
            trainer.fork(path, jax.random.PRNGKey(0))


def test_a_state_of_another_format_is_refused(env, tmp_path, monkeypatch):
    trainer = make_trainer(env, tmp_path)
    monkeypatch.setattr(train_module, "STATE_FORMAT", 2)
    _, path = save_a_state(trainer)
    monkeypatch.setattr(train_module, "STATE_FORMAT", 3)

    with pytest.raises(ValueError, match="format 2, this version reads format 3"):
        trainer.resume(path)


# forks and branches of a saved run


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


def test_a_fork_starts_from_the_saved_network_average_and_buffer(env, tmp_path):
    saved, path = save_a_state(make_trainer(env, tmp_path / "trunk"))
    fork = make_trainer(env, tmp_path / "fork")
    key = jax.random.PRNGKey(7)

    start = fork.fork(path, key)

    # a new run: epoch 0, with its own rng
    assert start.cur_epoch == 0
    np.testing.assert_array_equal(start.key, key)
    # the saved network and optimizer state, and its average with the epochs in it
    assert leaves_equal(start.train_state, saved.train_state)
    assert leaves_equal(start.average, saved.average) and start.averaged == 3
    # the collection state as saved: its replay buffer with the game in progress, and the games
    # and their search states, so self-play goes on as the saved run's would
    assert not saved.collection_state.buffer_state.has_reward.all()
    assert leaves_equal(start.collection_state, saved.collection_state)


def test_a_fork_keeps_the_games_waiting_for_solves(env, tmp_path):
    # a fork labels the saved run's finished games waiting for their solves, as the saved run
    # would have, rather than leaving their places in the buffer empty
    trunk = solving_trainer(env, tmp_path / "trunk")
    output = a_saved_run(trunk)
    pending = output.collection_state.pending
    assert pending is not None
    pending = replace(
        pending,
        added=jnp.array([2, 1], dtype=jnp.int32),
        entry_solve=pending.entry_solve.at[:, :3].set(0),
    )
    saved = snapshot(pending)
    path = trunk.save_state(
        replace(
            output, collection_state=replace(output.collection_state, pending=pending)
        )
    )

    forked = solving_trainer(env, tmp_path / "fork").fork(path, jax.random.PRNGKey(0))

    assert leaves_equal(forked.collection_state.pending, saved)
    # and so does a resumed run
    resumed = solving_trainer(env, tmp_path / "trunk").load_state(path)
    assert leaves_equal(resumed.collection_state.pending, saved)


@pytest.mark.parametrize("capacity", [4, 16])
def test_a_fork_resizes_the_replay_buffer(env, tmp_path, capacity, capsys):
    saved, path = save_a_state(make_trainer(env, tmp_path / "trunk"))
    fork = make_trainer(
        env, tmp_path / "fork", memory_buffer=EpisodeReplayBuffer(capacity=capacity)
    )

    loaded = fork.fork(path, jax.random.PRNGKey(0)).collection_state

    saved_ok, saved_data = by_age(saved.collection_state.buffer_state)
    ok, data = by_age(loaded.buffer_state)
    kept = min(8, capacity)
    # each environment's newest entries, as many as fit, at the same ages, with the games in
    # progress
    np.testing.assert_array_equal(ok[:, :kept], saved_ok[:, :kept])
    assert not ok[:, kept:].any()
    for x, y in zip(jax.tree.leaves(data), jax.tree.leaves(saved_data), strict=True):
        np.testing.assert_array_equal(
            x[:, :kept][ok[:, :kept]], y[:, :kept][ok[:, :kept]]
        )
    assert f"resized from 8 to {capacity}" in capsys.readouterr().out


def test_a_fork_without_the_replay_buffer(env, tmp_path):
    saved, path = save_a_state(make_trainer(env, tmp_path / "trunk"))
    # the number of environments needn't match without it
    fork = make_trainer(env, tmp_path / "fork", batch_size=3)

    start = fork.fork(path, jax.random.PRNGKey(0), replay_buffer=False)

    assert leaves_equal(start.train_state, saved.train_state)
    assert leaves_equal(start.average, saved.average)
    assert start.collection_state.episodes.shape == (3,)
    assert not np.asarray(start.collection_state.buffer_state.populated).any()


def test_a_fork_records_its_parent(env, tmp_path, fake_loop):
    saved, path = save_a_state(make_trainer(env, tmp_path / "trunk"))
    monitor = FakeMonitor()
    fork = make_trainer(env, tmp_path / "fork", monitor=monitor, save_state_at=[1])
    calls = fake_loop(fork)

    # a directory: its newest state
    fork.train_loop(seed=1, num_epochs=1, fork_from=str(tmp_path / "trunk"))

    parent = {
        "path": os.path.abspath(path),
        "epoch": 2,
        "replay_buffer": True,
        "monitor_run": None,
        "parent": None,
    }
    assert monitor.started[0]["parent"] == parent
    # it trains on the saved replay buffer, with the games that were in progress
    np.testing.assert_array_equal(
        of_kind(calls, "train")[0].buffer_state.populated,
        saved.collection_state.buffer_state.populated,
    )
    # its own saved states record it, for forks of the fork
    meta = read_state_meta(state_path(fork.ckpt_dir, 1))
    assert (meta["parent"], meta["monitor_run"]) == (parent, monitor.run_id)
    # a resumed fork goes on logging to its monitor run, and records the same parent
    resumed = make_trainer(env, tmp_path / "fork", monitor=FakeMonitor())
    fake_loop(resumed)
    resumed.train_loop(seed=1, num_epochs=2, initial_state=resumed.resume())
    assert resumed.monitor is not None and resumed.monitor.run_id == monitor.run_id
    assert resumed.parent == parent


def test_a_branch_with_the_runs_own_settings_carries_on_exactly(
    env, tmp_path, fake_loop
):
    straight = make_trainer(env, tmp_path / "straight", average_epochs=2)
    straight_calls = fake_loop(straight)
    expected = straight.train_loop(seed=0, num_epochs=4)
    trunk = make_trainer(
        env,
        tmp_path / "trunk",
        average_epochs=2,
        save_state_at=[2],
        save_weights_every=1,
    )
    fake_loop(trunk)
    trunk.train_loop(seed=0, num_epochs=2)
    monitor = FakeMonitor()
    branch = make_trainer(
        env,
        tmp_path / "branch",
        average_epochs=2,
        monitor=monitor,
        save_weights_every=1,
    )
    calls = fake_loop(branch)

    # the seed is the run's rng's, whatever is passed
    output = branch.train_loop(
        seed=1, num_epochs=4, initial_state=branch.branch(str(tmp_path / "trunk"))
    )

    # the run goes on as if it hadn't stopped: the same weights, average and keys
    assert leaves_equal(output.train_state, expected.train_state)
    assert leaves_equal(output.average, expected.average)
    assert len(calls) == 4
    for call, straight_call in zip(calls, straight_calls[-4:], strict=True):
        assert call.kind == straight_call.kind
        np.testing.assert_array_equal(call.key, straight_call.key)
    # in its own directory and monitor run, which records where it branched from
    assert checkpoint_epochs(branch.weights_dir) == [2, 3]
    assert checkpoint_epochs(trunk.weights_dir) == [0, 1]
    assert monitor.started[0]["parent"] == {
        "path": os.path.abspath(state_path(str(tmp_path / "trunk"), 2)),
        "epoch": 2,
        "branch": True,
        "monitor_run": None,
        "parent": None,
    }


def saved_mid_move(trainer):
    """Saves a state of `trainer`'s run after 2 epochs, with 6 entries in its replay buffer and
    every environment's move one search cycle in. Returns its path."""
    output = a_saved_run(trainer)
    state = replace(
        output.collection_state,
        buffer_state=written(output.collection_state.buffer_state, 6),
        moves=jnp.full_like(output.collection_state.moves, 6),
        search_cycles=jnp.ones_like(output.collection_state.search_cycles),
    )
    return trainer.save_state(replace(output, collection_state=state))


def capacity(collection_state):
    """The size of a collection state's search trees."""
    return collection_state.eval_state.parents.shape[-1]


@pytest.mark.parametrize("bigger", [False, True])
def test_a_branch_keeps_the_search_states_only_if_they_fit_its_search(
    env, tmp_path, bigger, capsys
):
    small, big = tiny_mcts(env, 2, 4), tiny_mcts(env, 6, 8)
    path = saved_mid_move(make_trainer(env, tmp_path / "trunk", evaluator=small))
    branch = make_trainer(env, tmp_path / "branch", evaluator=big if bigger else small)

    start = branch.branch(path)

    # a bigger search starts new trees, its moves in progress searched again from the start
    assert capacity(start.collection_state) == (8 if bigger else 4)
    np.testing.assert_array_equal(
        start.collection_state.search_cycles, np.full(2, 0 if bigger else 1)
    )
    assert ("new search states" in capsys.readouterr().out) == bigger
    # the rest of the run as it was saved
    assert start.cur_epoch == 2 and start.averaged == 3
    assert int(start.collection_state.moves[0]) == 6
    np.testing.assert_array_equal(
        start.collection_state.buffer_state.populated[:, :6], True
    )
    assert leaves_equal(start.train_state, trained_state(branch))


def test_a_branch_takes_its_trainers_buffer_capacity(env, tmp_path, capsys):
    saved, path = save_a_state(make_trainer(env, tmp_path / "trunk"))
    branch = make_trainer(
        env, tmp_path / "branch", memory_buffer=EpisodeReplayBuffer(capacity=4)
    )

    start = branch.branch(path)

    assert start.collection_state.buffer_state.populated.shape == (2, 4)
    assert "resized from 8 to 4" in capsys.readouterr().out
    # the newest entries
    np.testing.assert_array_equal(
        by_age(start.collection_state.buffer_state)[0],
        by_age(saved.collection_state.buffer_state)[0][:, :4],
    )


def sgd_optimizer(learning_rate, weight_decay, momentum):
    return optax.inject_hyperparams(
        train_module.sgdw, static_args=("mask", "decoupled")
    )(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        mask=train_module.weight_decay_mask,
    )


@pytest.mark.parametrize("start", ["resume", "branch", "fork"])
def test_a_run_from_a_saved_state_takes_its_own_hyperparameters(
    env, tmp_path, fake_loop, capsys, start
):
    # the saved run's optimizer injects learning rate 0.1, weight decay 0.5 and momentum 0.9,
    # which its optimizer state holds: e.g. a branch at a lower learning rate
    _, path = save_a_state(
        make_trainer(env, tmp_path / "saved", optimizer=sgd_optimizer(0.1, 0.5, 0.9))
    )
    # a resumed run is in the saved run's directory
    ckpt_dir = tmp_path / ("saved" if start == "resume" else "new")
    trainer = make_trainer(env, ckpt_dir, optimizer=sgd_optimizer(0.05, 0.1, 0.5))
    fake_loop(trainer)
    kwargs: dict[str, Any] = {
        "resume": lambda: {"initial_state": trainer.resume(), "num_epochs": 2},
        "branch": lambda: {"initial_state": trainer.branch(path), "num_epochs": 2},
        "fork": lambda: {"fork_from": path, "num_epochs": 0},
    }[start]()

    output = trainer.train_loop(seed=0, **kwargs)

    # the new run's values, not the saved state's, which the next update uses (see
    # `with_hyperparams`), and it says so
    hyperparams = injected_hyperparams(output.train_state.opt_state)
    assert float(hyperparams["learning_rate"]) == pytest.approx(0.05)
    assert float(hyperparams["weight_decay"]) == pytest.approx(0.1)
    assert float(hyperparams["momentum"]) == pytest.approx(0.5)
    assert (
        "learning_rate 0.1 -> 0.05, momentum 0.9 -> 0.5, weight_decay 0.5 -> 0.1"
        in capsys.readouterr().out
    )


# resizing the replay buffer


def entries_by_age(buffer_state):
    """A replay buffer's entries, each environment's newest first: whether populated, whether
    with a reward, and the search values."""
    _, (buffer, *data) = by_age(
        buffer_state, buffer_state.populated, buffer_state.has_reward
    )
    return np.stack([np.asarray(buffer.search_value), *map(np.asarray, data)])


@pytest.mark.parametrize("capacity", [4, 16])
def test_resize_keeps_the_newest_entries_and_the_games_in_progress(
    env, tmp_path, capacity
):
    trainer = make_trainer(env, tmp_path)
    # a replay buffer of 8 that has wrapped around, the last 2 entries of a game in progress
    state = a_saved_run(trainer).collection_state
    buffer = trainer.memory_buffer

    resized = trainer.resize_replay_buffer(state, capacity)

    # each environment's newest entries, at the same ages
    assert resized.buffer_state.populated.shape == (2, capacity)
    kept = min(8, capacity)
    old, new = entries_by_age(state.buffer_state), entries_by_age(resized.buffer_state)
    np.testing.assert_array_equal(new[:, :, :kept], old[:, :, :kept])
    assert not new[1][:, kept:].any()
    # the games in progress carry on: when they end, their entries get their rewards...
    ended = jax.vmap(buffer.assign_rewards)(resized.buffer_state, jnp.ones((2, 2)))
    assert int(buffer.sample_mask(ended).sum()) == 2 * kept
    # ... or truncating discards just them
    truncated = jax.vmap(buffer.truncate)(resized.buffer_state)
    assert int(truncated.populated.sum()) == 2 * (kept - 2)
    with pytest.raises(ValueError, match="must be positive"):
        trainer.resize_replay_buffer(resized, 0)


def test_a_resized_state_saves_and_resumes_at_its_capacity(env, tmp_path):
    trainer = make_trainer(env, tmp_path)
    output = a_saved_run(trainer)
    resized = replace(
        output,
        collection_state=trainer.resize_replay_buffer(output.collection_state, 12),
    )
    expected = snapshot(resized)

    path = trainer.save_state(resized)

    assert read_state_meta(path)["buffer_capacity"] == 12
    # as in a new process, configured with the run's first capacity
    state = make_trainer(env, tmp_path).resume(path)
    assert jax.tree.all(jax.tree.map(np.array_equal, snapshot(state), expected))


def test_self_play_and_training_carry_on_in_a_resized_buffer(env, tmp_path):
    # games of 3 moves, a buffer of 8 positions per environment
    trainer = make_trainer(env, tmp_path)
    buffer = trainer.memory_buffer
    state = trainer.init_collection_state(jax.random.PRNGKey(0))
    train_state = trainer.init_train_state()

    def collect(state, num_steps, seed):
        keys = jax.random.split(jax.random.PRNGKey(seed), trainer.batch_size)
        return trainer.collect_steps(keys, state, None, num_steps)

    def sampleable(state):
        return np.asarray(buffer.sample_mask(state.buffer_state)).sum(-1).tolist()

    # 4 moves: a game, and one in progress
    state = collect(state, 4, 1)
    # down to the newest 2: the first game's last move, and the move in progress
    state = trainer.resize_replay_buffer(state, 2)
    assert sampleable(state) == [1, 1]
    # 2 more moves end the second game, which keeps its newest 2 entries
    state = collect(state, 2, 2)
    assert sampleable(state) == [2, 2]
    trainer.train_epoch(jax.random.PRNGKey(3), state.buffer_state, train_state, 1)
    # up to 12: room for 4 more games' moves besides
    state = trainer.resize_replay_buffer(state, 12)
    state = collect(state, 7, 4)
    assert sampleable(state) == [8, 8]
    np.testing.assert_array_equal(state.moves, [13, 13])
    trainer.train_epoch(jax.random.PRNGKey(5), state.buffer_state, train_state, 1)


# self-play


@pytest.mark.parametrize(
    "rewards, max_episode_steps, episodes, draws, first_player",
    [
        ((1.0, -1.0), 3, 2, 0.0, 2.0),
        ((0.0, 0.0), 3, 2, 2.0, 0.0),
        # truncated episodes aren't counted
        ((0.0, 0.0), 2, 0, 0.0, 0.0),
    ],
)
def test_self_play_counts_terminated_episodes_and_sums_their_metrics(
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
    # without episode_metrics_fn nothing is tracked
    assert (
        make_collector(env, scripted.first_legal, 3, tmp_path)(1).episode_metrics == {}
    )


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


def test_self_play_stores_the_searchs_root_value(
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


def test_self_play_stores_no_uninitialized_rewards(
    fixed_length_env, scripted, make_collector, tmp_path
):
    # games cut off after 2 of their 3 moves: their entries keep the reward they were stored with
    env = fixed_length_env(length=3)
    collect = make_collector(env, scripted.first_legal, 2, tmp_path)
    state = collect.trainer.init_collection_state(jax.random.PRNGKey(0))

    # on GPU, `empty` is uninitialized memory; on CPU it's zeros, so only the trace shows it
    jaxpr = jax.make_jaxpr(collect.step)(
        jax.random.split(jax.random.PRNGKey(0), 1), state
    )
    assert not re.search(r"\bempty\[", str(jaxpr))
    # two games, both cut off: nothing to sample, but their entries' rewards stay in the buffer
    buffer_state = collect(num_steps=4).buffer_state
    assert not buffer_state.populated.any()
    np.testing.assert_array_equal(buffer_state.buffer.reward, 0.0)


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
    state = trainer.init_collection_state(jax.random.PRNGKey(0))
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


def test_playing_moves_leaves_an_environment_whose_search_is_not_done_as_it_is(
    env, tmp_path
):
    trainer, state = budget_collection(env, tmp_path, [2, 2])
    state = replace(state, search_cycles=jnp.array([1, 1]))

    def advance(s):
        s, (done, action, ended) = trainer.play_moves(s, None)
        return replace(
            s,
            eval_state=trainer.evaluator.next_state(
                s.eval_state, action, ended, active=done
            ),
        )

    advanced = jax.vmap(advance)(state)

    jax.tree.map(np.testing.assert_array_equal, advanced, state)


def test_each_move_starting_in_a_call_takes_the_calls_next_key(env, tmp_path):
    trainer, state = budget_collection(env, tmp_path, [1, 3], advance_every=1)
    keys = jax.random.split(jax.random.PRNGKey(1), trainer.batch_size)

    state = trainer.collect_steps(keys, state, None, 2)

    # of the 6 cycles' keys, the first's 6th move took the 6th, and the second's 2nd the 2nd
    np.testing.assert_array_equal(state.move_key[0], jax.random.split(keys[0], 6)[5])
    np.testing.assert_array_equal(state.move_key[1], jax.random.split(keys[1], 6)[1])


def uneven_search_collection(env, tmp_path, advance_every=1, **kwargs):
    """A trainer whose MCTS self-play searches take 2 or 4 cycles (a playout cap), in trees of
    10 nodes, and plays the moves of those done every `advance_every` cycles; and its
    collection state of 5 environments."""
    evaluator = tiny_mcts(
        env, num_iterations=3, max_nodes=10, playout_cap=PlayoutCap(0.5, 1)
    )
    trainer = make_trainer(
        env,
        tmp_path,
        evaluator=evaluator,
        batch_size=5,
        advance_every=advance_every,
        **kwargs,
    )
    return trainer, trainer.init_collection_state(jax.random.PRNGKey(0))


@pytest.mark.parametrize("moves_at_once", [1, 2, 5])
def test_self_play_moves_on_only_the_trees_of_the_moves_played(
    env, tmp_path, moves_at_once
):
    # `collect_steps` moves the trees on outside the vmap, those of the environments that played
    # `moves_at_once` at a time: as every environment's own cycle does (see `collect_cycle`)
    trainer, state = uneven_search_collection(
        env, tmp_path, moves_at_once=moves_at_once
    )
    keys = jax.random.split(jax.random.PRNGKey(1), 5)

    # `collect_steps` donates the state it's given
    collected = trainer.collect_steps(keys, jax.tree.map(jnp.copy, state), None, 3)

    cycles = 3 * trainer.evaluator.search_cycles()
    move_keys = jax.vmap(lambda k: jax.random.split(k, cycles))(keys)
    collect = jax.jit(jax.vmap(lambda k, s: collect_cycle(trainer, k, s)))
    started = np.zeros(5, dtype=int)
    for _ in range(cycles):
        starting = np.asarray(state.search_cycles == 0)
        state = collect(move_keys[np.arange(5), started], state)
        started += starting
    assert 0 < int(collected.moves.sum()) and int(collected.episodes.sum()) > 0
    jax.tree.map(np.testing.assert_array_equal, collected, state)


def test_playing_moves_reads_and_writes_only_the_trees_of_the_moves_played(
    env, tmp_path
):
    # in the loop of search cycles and plays, a pass over every tree (e.g. moving them all on,
    # under vmap) shows as an instruction writing a whole array of the trees' nodes; outside the
    # loops within it (the cycles', and moving on the played trees a few at a time). (Every 2
    # cycles: a loop of 1 cycle would be inlined.)
    trainer, state = uneven_search_collection(env, tmp_path, advance_every=2)
    keys = jax.random.split(jax.random.PRNGKey(1), 5)

    hlo = trainer.collect_steps.lower(trainer, keys, state, None, 3).compile().as_text()

    assert hlo is not None
    # the loop of cycles and plays: the outer loop with loops within it
    (body,) = [b for b in outer_loop_bodies(hlo) if " while(" in b]
    # [environments, nodes, ...]: the edge map, priors, embeddings
    whole = re.compile(r"= \w+\[5,10,\S* ([\w-]+)\(")
    writes = [m.group(1) for m in whole.finditer(body)]
    assert writes and set(writes) <= {"get-tuple-element", "tuple", "bitcast"}


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PlyState:
    """`EveryOtherMove`'s state: the moves played so far in the game."""

    ply: jax.Array


class EveryOtherMove(Evaluator):
    """Plays the first legal move, and trains on every other one (see `Evaluator.trains_on`),
    the game's first, third, ...: as playout cap randomization's fast searches are played
    without being stored. Its value is the game's ply, so the experiences show which moves were
    stored."""

    def __init__(self):
        super().__init__(discount=-1.0)

    def init(self, *args, **kwargs):
        return PlyState(ply=jnp.array(0))

    def reset(self, state):
        return PlyState(ply=jnp.zeros_like(state.ply))

    def step(self, state, action):
        return PlyState(ply=state.ply + 1)

    def trains_on(self, state):
        return state.ply % 2 == 0

    def evaluate(self, key, eval_state, env_state, root_metadata, **kwargs):
        action = jnp.argmax(root_metadata.action_mask)
        return EvalOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=jax.nn.one_hot(action, root_metadata.action_mask.shape[-1]),
        )

    def get_value(self, state):
        return state.ply.astype(jnp.float32)


def test_self_play_stores_only_the_moves_the_evaluator_trains_on(env, tmp_path):
    # and each once, untransformed: training transforms samples as it draws them
    trainer = make_trainer(
        env, tmp_path, evaluator=EveryOtherMove(), data_transform_fns=[swap_actions]
    )
    state = trainer.init_collection_state(jax.random.PRNGKey(0))
    keys = jax.random.split(jax.random.PRNGKey(1), trainer.batch_size)

    # two whole games of 3 moves, and a move of a third
    state = trainer.collect_steps(keys, state, None, 7)

    np.testing.assert_array_equal(state.moves, [7, 7])
    np.testing.assert_array_equal(state.episodes, [2, 2])
    buffer_state = state.buffer_state
    # each game's first and third moves, then the third game's first, still in progress
    np.testing.assert_array_equal(buffer_state.next_idx, [5, 5])
    np.testing.assert_array_equal(buffer_state.episode_start_idx, [4, 4])
    np.testing.assert_array_equal(
        buffer_state.buffer.search_value[:, :5], [[0, 2, 0, 2, 0]] * 2
    )
    # the finished games' stored moves have their outcomes: (1, -1), whoever moved
    np.testing.assert_array_equal(buffer_state.has_reward[:, :5], [[1, 1, 1, 1, 0]] * 2)
    np.testing.assert_array_equal(
        buffer_state.buffer.reward[:, :4], [[[1, -1]] * 4] * 2
    )
    # the first legal action's target, as played
    np.testing.assert_array_equal(buffer_state.buffer.policy_weights[:, :5, 0], 1.0)


def test_collection_metrics(env, tmp_path):
    trainer = make_trainer(
        env, tmp_path, episode_metrics_fn=lambda r: {"draw": (r == 0).all()}
    )
    state = trainer.init_collection_state(jax.random.PRNGKey(0))
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
    )

    metrics = trainer.collection_metrics(before, after)

    assert {k: float(v) for k, v in metrics.items()} == pytest.approx(
        {
            "moves": 8,
            "episodes": 4,
            "episode_draw": 0.25,
            "buffer_samples": 5,
            "buffer_distinct_positions": 3,
            "buffer_distinct_fraction": 0.6,
        }
    )
    # without episodes, no episode means
    no_episodes = replace(after, episodes=state.episodes)
    assert "episode_draw" not in trainer.collection_metrics(before, no_episodes)


# training


@pytest.mark.parametrize("num_written, can", [(0, False), (3, False), (4, True)])
def test_training_needs_a_batch_of_finished_entries(env, tmp_path, num_written, can):
    # two environments' entries: a batch of 8 needs 4 each
    trainer = make_trainer(env, tmp_path, train_batch_size=8)
    state = trainer.init_collection_state(jax.random.PRNGKey(0))

    assert trainer.can_train(written(state.buffer_state, num_written)) == can


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
    state = trainer.init_collection_state(jax.random.PRNGKey(0))
    # stored with the first action's policy target
    buffer_state = written(state.buffer_state, 8)
    policy = jnp.zeros_like(buffer_state.buffer.policy_weights).at[..., 0].set(1.0)
    buffer_state = replace(
        buffer_state, buffer=replace(buffer_state.buffer, policy_weights=policy)
    )

    _, metrics = trainer.train_epoch(
        jax.random.PRNGKey(0), buffer_state, trainer.init_train_state(), 50
    )

    # 200 samples, each swapped with probability 1/2 (or never)
    assert float(metrics["swapped"]) == pytest.approx(swapped, abs=0.12)


def random_batch(env, key, size=8):
    """A batch of random experiences of the test environment."""
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


def train_steps_on(trainer, env, count=3):
    """The train state and metrics after a training step on each of `count` random batches,
    from the start."""
    train_step = jax.jit(trainer.train_step)
    ts = trainer.init_train_state()
    metrics = []
    for i in range(count):
        ts, step_metrics = train_step(ts, random_batch(env, jax.random.PRNGKey(i)))
        metrics.append(step_metrics)
    return ts, metrics


def assert_trees_close(a, b, rtol=1e-5, atol=1e-6):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


def test_grad_accum_matches_one_batch_without_batch_norm(env, tmp_path):
    def make(**kwargs):
        return make_trainer(
            env, tmp_path, train_batch_size=8, optimizer=optax.sgd(0.1), **kwargs
        )

    whole, whole_metrics = train_steps_on(make(), env)
    accumulated, accumulated_metrics = train_steps_on(make(grad_accum_steps=2), env)

    assert not leaves_equal(whole.params, make().init_train_state().params)
    assert_trees_close(accumulated, whole)
    for a, w in zip(accumulated_metrics, whole_metrics, strict=True):
        assert (
            set(a)
            == set(w)
            == {
                "loss",
                "policy_loss",
                "value_loss",
                "policy_target_entropy",
                "policy_kl",
                "grad_norm",
                "weight_norm",
                "update_ratio",
            }
        )
        assert_trees_close(a, w)


def test_grad_accum_updates_batch_norm_statistics_once_per_step(env, tmp_path):
    trainer = norm_trainer(env, tmp_path, train_batch_size=8, grad_accum_steps=2)
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
            jax.tree.leaves(new_ts.nn_state), jax.tree.leaves(ts.nn_state), strict=True
        )
        if not jnp.issubdtype(x.dtype, jnp.inexact)
    ]
    assert counters and all(np.array_equal(x, old + 1) for x, old in counters)


# the weight average


def test_an_averages_horizon_grows_by_an_epoch_every_warmup_epochs_up_to_its_own():
    # from the newest weights alone: a horizon of 1, 1.5 after 5 epochs, 4 after 30
    assert [average_share(n, 4, 10) for n in (0, 5, 10, 20, 30, 100)] == pytest.approx(
        [1, 1 / 1.5, 1 / 2, 1 / 3, 1 / 4, 1 / 4]
    )
    assert [average_share(n, 1, 10) for n in range(3)] == [1, 1, 1]
    # with a warmup of 1, the mean until its horizon
    assert [average_share(n, 4, 1) for n in range(6)] == pytest.approx(
        [1, 1 / 2, 1 / 3, 1 / 4, 1 / 4, 1 / 4]
    )


def test_an_average_follows_the_weights_direction_at_their_norm():
    # weight decay shrinking the weights: the average's matrices take the weights' norm, in the
    # moving average's direction; vectors (biases, normalization scales) are the plain average
    average = {"w": jnp.array([[3.0, 0.0], [0.0, 4.0]]), "b": jnp.array([2.0, 2.0])}
    params = {"w": jnp.array([[0.0, 1.0], [1.0, 0.0]]), "b": jnp.array([0.0, 4.0])}
    moved = _move_average(average, params, jnp.float32(0.5))

    np.testing.assert_allclose(moved["b"], [1.0, 3.0])
    plain = 0.5 * (average["w"] + params["w"])
    np.testing.assert_allclose(
        moved["w"],
        plain * np.linalg.norm(params["w"]) / np.linalg.norm(plain),
        rtol=1e-6,
    )
    # a share of 1 is exactly the weights
    exact = _move_average(average, params, jnp.float32(1.0))
    assert all(np.array_equal(exact[k], params[k]) for k in params)


def test_an_averages_own_statistics_follow_its_weights(env, tmp_path):
    # through the average's weights: scaling its hidden layer scales the statistics BatchNorm
    # keeps of it, and the average plays the same; with the network's statistics it doesn't
    trainer = norm_trainer(env, tmp_path, average_norm_batches=3)
    state = trainer.init_collection_state(jax.random.PRNGKey(0))
    buffer_state = written(state.buffer_state, 8)
    buffer_state = replace(
        buffer_state,
        buffer=replace(
            buffer_state.buffer,
            observation_nn=jax.random.normal(
                jax.random.PRNGKey(3), buffer_state.buffer.observation_nn.shape
            ),
        ),
    )
    average = trainer.init_train_state().params
    scaled = eqx.tree_at(
        lambda p: (p.hidden.weight, p.hidden.bias),
        average,
        (average.hidden.weight * 3, average.hidden.bias * 3),
    )
    key = jax.random.PRNGKey(1)

    own = trainer.average_norm_state(key, buffer_state, average)
    own_scaled = trainer.average_norm_state(key, buffer_state, scaled)

    assert own is not None and own_scaled is not None
    counter, (mean, var) = jax.tree.leaves(own)[:1], jax.tree.leaves(own)[1:]
    _, (mean3, var3) = jax.tree.leaves(own_scaled)[:1], jax.tree.leaves(own_scaled)[1:]
    # from the network's initial state, one update a batch
    assert int(counter[0]) == 3
    np.testing.assert_allclose(mean3, 3 * mean, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(var3, 9 * var, rtol=1e-4, atol=1e-6)
    x = random_batch(env, jax.random.PRNGKey(2)).observation_nn

    def play(params, state):
        nn = eqx.nn.inference_mode(eqx.combine(params, trainer.nn_static))
        return apply_nn(nn, state, x)[0]

    assert_trees_close(
        play(scaled, own_scaled), play(average, own), rtol=1e-3, atol=1e-4
    )
    assert not np.allclose(
        play(scaled, trainer.nn_state)[0], play(average, trainer.nn_state)[0]
    )


def test_an_averages_statistics_come_from_micro_batches_of_enough_positions(
    env, tmp_path
):
    # micro-batches of 4: BatchNorm's batches in training, and sampled without replacement
    trainer = norm_trainer(
        env, tmp_path, average_norm_batches=2, train_batch_size=8, grad_accum_steps=2
    )
    params = trainer.init_train_state().params
    empty = trainer.init_collection_state(jax.random.PRNGKey(0)).buffer_state
    sizes = []
    loss_fn = trainer.loss_fn

    def recording(nn, state, batch):
        sizes.append(batch.reward.shape[0])
        return loss_fn(nn, state, batch)

    trainer.loss_fn = recording
    key = jax.random.PRNGKey(0)

    # 2 positions (one per environment), then 4
    assert trainer.average_norm_state(key, written(empty, 1), params) is None
    assert trainer.average_norm_state(key, written(empty, 2), params) is not None
    assert sizes == [4]
    # none without the option, or for a network without state
    trainer.average_norm_batches = 0
    assert trainer.average_norm_state(key, written(empty, 2), params) is None
    plain = make_trainer(env, tmp_path)
    assert (
        plain.average_norm_state(
            key, written(empty, 2), plain.init_train_state().params
        )
        is None
    )


def test_a_forks_average_gets_its_own_statistics_once_self_play_stores_a_micro_batch(
    env, tmp_path, fake_loop, monkeypatch
):
    # a network that plays its average, forked without the saved replay buffer: it has nothing
    # to estimate its average's statistics on until its first self-play
    trunk = norm_trainer(env, tmp_path / "trunk", average_epochs=2, save_state_at=[2])
    fake_loop(trunk)
    trunk.train_loop(seed=0, num_epochs=2)
    fork = norm_trainer(
        env,
        tmp_path / "fork",
        average_epochs=2,
        average_norm_batches=2,
        train_steps_per_epoch=0,
    )
    calls = fake_loop(fork)
    fake_collect = fork.collect_steps

    def collect_steps(key, state, params, num_steps):
        state = fake_collect(key, state, params, num_steps)
        return replace(state, buffer_state=written(state.buffer_state, 4))

    monkeypatch.setattr(fork, "collect_steps", collect_steps)

    output = fork.train_loop(
        seed=0,
        num_epochs=3,
        fork_from=state_path(str(tmp_path / "trunk"), 2),
        fork_replay_buffer=False,
    )

    network_state = output.train_state.nn_state
    played = [c.params[1] for c in of_kind(calls, "collect")]
    # epoch 0 plays the average with the network's statistics, then with its own
    assert len(played) == 3
    assert leaves_equal(played[0], network_state)
    for state in played[1:]:
        assert not leaves_equal(state, network_state)


# exact endgame labels


def solving_trainer(env, ckpt_dir, **kwargs):
    """A trainer with exact endgame labels: games stop at their 2nd move, and draw."""
    return make_trainer(
        env,
        ckpt_dir,
        endgame_solve=EndgameSolve(
            stop_fn=lambda state: state.step >= 2,
            solve_fn=lambda positions: np.zeros((len(positions.step), 2)),
            slots=2,
        ),
        **kwargs,
    )


def assert_solved(solved, expected):
    """The pending games each solve was given were `expected`'s; the last solve, submitted as
    the loop ends, may be cancelled before it starts, its results unused."""
    if len(solved) == len(expected) - 1:
        expected = expected[:-1]
    np.testing.assert_array_equal(solved, expected)


def test_games_waiting_for_solves_are_solved_during_self_play_and_labelled_after(
    env, tmp_path, fake_loop, monkeypatch
):
    monitor = FakeMonitor()
    # a batch of the 2 positions each self-play leaves to train on
    trainer = solving_trainer(env, tmp_path, monitor=monitor, train_batch_size=2)
    calls = fake_loop(trainer, can_train=None)
    fake_collect = trainer.collect_steps
    solved = []

    def collect_steps(key, state, params, num_steps):
        # every call stops a game, and finishes one to train on
        state = fake_collect(key, state, params, num_steps)
        return replace(
            state,
            pending=replace(state.pending, added=state.pending.added + 1),
            buffer_state=written(state.buffer_state, 1),
        )

    def solve_waiting(pending):
        # on the solver's thread
        solved.append(np.array(pending.added))
        return Solved(
            np.ones((2, 2, 2), np.float32),
            np.array([1, 0], np.int32),
            np.ones((1, 2), np.float32),
            np.array([True]),
        )

    def label_solved(state, results, counts):
        calls.append(SimpleNamespace(kind="label", counts=np.asarray(counts)))
        return state

    monkeypatch.setattr(trainer, "collect_steps", collect_steps)
    monkeypatch.setattr(trainer, "solve_waiting", solve_waiting)
    monkeypatch.setattr(trainer, "label_solved", label_solved)

    trainer.train_loop(seed=0, num_epochs=2)

    # those waiting at the start solve through epoch 0's self-play; each epoch's from the end of
    # its self-play, through training, to the end of the next's...
    assert_solved(solved, [[0, 0], [1, 1], [2, 2]])
    # ...and are labelled after it, before training
    assert [c.kind for c in calls] == ["collect", "label", "train"] * 2
    for c in of_kind(calls, "label"):
        np.testing.assert_array_equal(c.counts, [1, 0])
    assert logged(monitor, "solve_games") == {0: 1.0, 1: 1.0}
    assert logged(monitor, "solve_positions") == {0: 1.0, 1: 1.0}
    assert logged(monitor, "solve_draw") == {0: 0.0, 1: 0.0}


def labelling_fakes(trainer, monkeypatch, written_by_label, first_written=0):
    """Fakes for `trainer` whose self-play stops a game in every environment (and whose first
    finishes `first_written` positions to train on), whose solves solve every game waiting, and
    whose labels leave `written_by_label` positions to train on. Returns the pending games'
    counts each solve was given."""
    fake_collect = trainer.collect_steps
    solved = []

    def collect_steps(key, state, params, num_steps):
        state = fake_collect(key, state, params, num_steps)
        buffer_state = state.buffer_state
        if first_written and not np.asarray(buffer_state.populated).any():
            buffer_state = written(buffer_state, first_written)
        return replace(
            state,
            pending=replace(state.pending, added=state.pending.added + 1),
            buffer_state=buffer_state,
        )

    def solve_waiting(pending):
        solved.append(np.array(pending.added))
        counts = np.asarray(pending.added - pending.labelled)
        games = np.zeros((counts.sum(), 2), np.float32)
        return Solved(
            np.zeros((2, 2, 2), np.float32), counts, games, np.ones(len(games), bool)
        )

    def label_solved(state, results, counts):
        return replace(
            state,
            pending=replace(state.pending, labelled=state.pending.labelled + counts),
            buffer_state=written(state.buffer_state, written_by_label),
        )

    monkeypatch.setattr(trainer, "collect_steps", collect_steps)
    monkeypatch.setattr(trainer, "solve_waiting", solve_waiting)
    monkeypatch.setattr(trainer, "label_solved", label_solved)
    return solved


@pytest.mark.parametrize(
    "train_batch_size, first_written, written_by_label",
    [
        # nothing at all to train on
        (2, 0, 1),
        # a few games that finished before stopping, fewer positions than a batch
        (8, 1, 4),
    ],
)
def test_without_a_batch_to_train_on_the_games_just_stopped_are_solved_at_once(
    env,
    tmp_path,
    fake_loop,
    monkeypatch,
    train_batch_size,
    first_written,
    written_by_label,
):
    monitor = FakeMonitor()
    trainer = solving_trainer(
        env, tmp_path, monitor=monitor, train_batch_size=train_batch_size
    )
    calls = fake_loop(trainer, can_train=None)
    solved = labelling_fakes(trainer, monkeypatch, written_by_label, first_written)

    trainer.train_loop(seed=0, num_epochs=2)

    # epoch 0: none waiting during its self-play, then the games it stopped, at once; epoch 1:
    # none left over, then its own, for epoch 2
    assert_solved(solved, [[0, 0], [1, 1], [1, 1], [2, 2]])
    assert logged(monitor, "solve_games") == {0: 2.0, 1: 0.0}
    # both epochs train, once the games are labelled
    assert [c.kind for c in calls] == ["collect", "train"] * 2


# memorization


def held_out_trainer(env, tmp_path, **kwargs):
    """A trainer of 2 environments that holds out the second one's games, scoring 2
    micro-batches of 4 positions of each side."""
    return make_trainer(
        env, tmp_path, holdout_envs=1, holdout_positions=8, train_batch_size=4, **kwargs
    )


def buffer_of(trainer, experience, populated, has_reward=None):
    """A replay buffer of 2 environments holding `experience` (one position, without batch
    dimensions) in every place `populated` (2, capacity) marks, written up to place 0 (the newest
    is the last place); finished games unless `has_reward` says otherwise."""
    state = trainer.memory_buffer.init(2, experience)
    capacity = state.populated.shape[1]
    return replace(
        state,
        next_idx=jnp.zeros(2, jnp.int32),
        buffer=jax.tree.map(
            lambda x, b: jnp.broadcast_to(x, b.shape).astype(b.dtype),
            experience,
            state.buffer,
        ),
        populated=jnp.asarray(populated).reshape(2, capacity),
        has_reward=jnp.ones((2, capacity), bool)
        if has_reward is None
        else jnp.asarray(has_reward),
    )


def one_position(env, key=1):
    return jax.tree.map(
        lambda x: x[0], random_batch(env, jax.random.PRNGKey(key), size=1)
    )


def test_training_never_samples_the_held_out_games(env, tmp_path):
    trainer = held_out_trainer(env, tmp_path)
    full = np.ones((2, 8), bool)
    state = buffer_of(trainer, one_position(env), full)

    np.testing.assert_array_equal(
        trainer.training_mask(state), [[True] * 8, [False] * 8]
    )
    # only the held-out environment has finished games: nothing to train on
    unfinished = buffer_of(
        trainer, one_position(env), full, has_reward=[[False] * 8, [True] * 8]
    )
    assert not trainer.can_train(unfinished)


def test_a_loop_starts_the_held_out_buffers_with_untrained_positions_only(
    env, tmp_path
):
    trainer = held_out_trainer(env, tmp_path)
    unfinished = np.zeros((2, 8), bool)
    unfinished[:, :2] = True
    state = buffer_of(
        trainer, one_position(env), np.ones((2, 8), bool), has_reward=~unfinished
    )

    cleared = trainer.without_holdout_positions(state)

    # the trained-on environment keeps everything; the held-out one only its game in progress
    np.testing.assert_array_equal(cleared.populated[0], [True] * 8)
    np.testing.assert_array_equal(cleared.populated[1], [True] * 2 + [False] * 6)


def test_held_out_positions_are_compared_with_trained_on_ones_of_their_ages(
    env, tmp_path
):
    trainer = held_out_trainer(env, tmp_path)
    new, old = one_position(env, key=1), one_position(env, key=2)
    # the newest place is the last (next_idx 0): the held-out environment has only its 3 newest
    # positions, as after a loop's start; the trained-on one has older positions of another kind
    populated = np.ones((2, 8), bool)
    populated[1, :5] = False
    state = buffer_of(trainer, new, populated)
    is_old = jnp.arange(8) < 5
    state = replace(
        state,
        buffer=jax.tree.map(
            lambda b, o: jnp.where(
                (is_old & jnp.array([True, False])[:, None]).reshape(
                    (2, 8) + (1,) * (b.ndim - 2)
                ),
                o.astype(b.dtype),
                b,
            ),
            state.buffer,
            old,
        ),
    )
    params = eqx.filter(trainer.nn, eqx.is_inexact_array)

    metrics = trainer.holdout_metrics(
        jax.random.PRNGKey(0), state, {"raw": (params, None)}
    )

    # both sides sample the newest positions only, which are alike: nothing memorized
    for metric in ("policy_kl", "value_loss"):
        assert metrics[f"heldout_{metric}_raw"] == pytest.approx(
            metrics[f"seen_{metric}_raw"]
        )
        assert metrics[f"memorized_{metric}_raw"] == pytest.approx(0.0, abs=1e-6)
        assert metrics[f"memorized_{metric}_raw_se"] == pytest.approx(0.0, abs=1e-6)
    assert int(metrics["heldout_stored"]) == 3
    # held-out positions newer than every trained-on one score nothing
    populated = np.zeros((2, 8), bool)
    populated[0, :4] = True
    populated[1, 7] = True
    newer = buffer_of(trainer, new, populated)
    assert (
        trainer.holdout_metrics(jax.random.PRNGKey(0), newer, {"raw": (params, None)})
        == {}
    )


def test_the_network_and_its_average_are_scored_on_held_out_games(env, tmp_path):
    monitor = FakeMonitor()
    trainer = held_out_trainer(env, tmp_path, average_epochs=2, monitor=monitor)

    trainer.train_loop(seed=0, num_epochs=3)

    for name in ("policy_kl", "value_loss"):
        for weights in ("raw", "avg"):
            held = logged(monitor, f"heldout_{name}_{weights}")
            seen = logged(monitor, f"seen_{name}_{weights}")
            memorized = logged(monitor, f"memorized_{name}_{weights}")
            assert held and set(held) == set(seen) == set(memorized)
            for epoch in held:
                assert memorized[epoch] == pytest.approx(
                    held[epoch] - seen[epoch], abs=1e-6
                )
    # epoch 0's 3 trained-on positions are fewer than a batch: it trains from epoch 1
    assert set(logged(monitor, "loss")) == {1, 2}


# the real thing: one trainer with every piece real, on the smallest configuration, compiled once

# a replay buffer capacity no other test's arrays have, so the real trainer's replay buffer is
# the live arrays of its shape
REAL_CAPACITY = 37


class SumValidator:
    """Scores weights by the sum of their first array, so a resumed run's validation can be
    compared exactly."""

    def __call__(self, params):
        return {"engine_top1": jnp.sum(jax.tree.leaves(params)[0])}

    def get_config(self):
        return {"name": "sum"}


@pytest.fixture(scope="module")
def real(env, tmp_path_factory):
    """A Trainer with AlphaZero self-play, a network with BatchNorm, gradient accumulation, a
    data transform, a 2-epoch weight average with its own statistics, a validator, and a (fake)
    monitor. It saves its weights every epoch, and its state after 2 epochs."""
    network = eqx.nn.make_with_state(NormMLP)(
        2, env.num_actions, key=jax.random.PRNGKey(0)
    )
    eval_fn = make_nn_eval_fn(network[0], env.state_to_nn_input)
    return make_trainer(
        env,
        tmp_path_factory.mktemp("real"),
        network=network,
        evaluator=tiny_mcts(env, num_iterations=2, eval_fn=eval_fn),
        monitor=FakeMonitor(),
        memory_buffer=EpisodeReplayBuffer(capacity=REAL_CAPACITY),
        data_transform_fns=[swap_actions],
        grad_accum_steps=2,
        average_epochs=2,
        average_norm_batches=2,
        validators=[SumValidator()],
        save_weights_every=1,
        save_state_at=[2],
    )


@pytest.fixture(scope="module")
def real_run(real):
    """A 4-epoch run of the real trainer: a host copy of its output, what it logged by epoch, its
    weights and saved states, and the path of the state it saved after epoch 2."""
    real.monitor = FakeMonitor()
    output = snapshot(real.train_loop(seed=0, num_epochs=4))
    return SimpleNamespace(
        output=output,
        logged=real.monitor.by_epoch(),
        weights=checkpoint_epochs(real.weights_dir),
        states=state_epochs(real.ckpt_dir),
        path=state_path(real.ckpt_dir, 2),
    )


def test_a_real_run_trains_and_reports_its_metrics(real_run):
    output = real_run.output
    assert output.cur_epoch == 4 and int(output.train_state.step) == 4
    assert output.averaged == 4
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(output.train_state.params))
    for epoch in range(4):
        assert real_run.logged[epoch].keys() == {
            "moves",
            "episodes",
            "buffer_samples",
            "buffer_distinct_positions",
            "buffer_distinct_fraction",
            "loss",
            "policy_loss",
            "policy_target_entropy",
            "policy_kl",
            "value_loss",
            "weight_norm",
            "update_ratio",
            "grad_norm",
            "avg2/engine_top1",
        }
    assert real_run.weights == [0, 1, 2, 3] and real_run.states == [2]


def test_runs_describe_every_metric_a_real_run_logs(real_run):
    logged = {name for metrics in real_run.logged.values() for name in metrics}
    undescribed = {
        name
        for name in logged
        if not any(
            fnmatch.fnmatchcase(name, p) and "description" in d
            for p, d in DECLARED.items()
        )
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

    # train state, weight average, replay buffer, games, search trees, rng
    assert jax.tree.all(jax.tree.map(np.array_equal, snapshot(output), real_run.output))
    logged = monitor.by_epoch()
    assert logged.keys() == {2, 3}
    for epoch in (2, 3):
        assert logged[epoch].keys() == real_run.logged[epoch].keys()
        for name, value in logged[epoch].items():
            np.testing.assert_array_equal(
                value, real_run.logged[epoch][name], err_msg=f"epoch {epoch}: {name}"
            )
    # and the weights it saves
    for epoch in (2, 3):
        with (
            open(checkpoint_path(real.weights_dir, epoch), "rb") as f,
            open(
                checkpoint_path(
                    os.path.join(os.path.dirname(real_run.path), "avg2"), epoch
                ),
                "rb",
            ) as g,
        ):
            assert f.read() == g.read()


def test_self_play_updates_the_collection_state_in_place(real):
    state = real.init_collection_state(jax.random.PRNGKey(0))
    keys = jax.random.split(jax.random.PRNGKey(1), real.batch_size)
    ts = real.init_train_state()
    params = (ts.params, ts.nn_state)
    state_bytes = sum(x.nbytes for x in jax.tree.leaves(state))

    for num_steps in (0, STEPS):
        analysis = (
            Trainer.collect_steps.lower(real, keys, state, params, num_steps)
            .compile()
            .memory_analysis()
        )
        # every array of the returned state reuses the memory of the given one's
        assert analysis is not None
        assert analysis.alias_size_in_bytes == state_bytes

    stats = CollectionStats.of(state)
    new_state = real.collect_steps(keys, state, params, STEPS)

    assert all(x.is_deleted() for x in jax.tree.leaves(state))
    assert not any(x.is_deleted() for x in jax.tree.leaves(new_state))
    # the copied stats outlive the donated state
    assert not any(x.is_deleted() for x in jax.tree.leaves(vars(stats)))


def test_train_loop_keeps_one_copy_of_the_replay_buffer(real, tmp_path, monkeypatch):
    monkeypatch.setattr(real, "ckpt_dir", str(tmp_path))
    shape = (real.batch_size, REAL_CAPACITY)

    def buffer_bytes(arrays):
        return sum(x.nbytes for x in arrays if x.shape[:2] == shape)

    # (bytes in one replay buffer, bytes in all live ones), as each epoch's training starts:
    # after its self-play metrics
    measured = []
    train_epoch = real.train_epoch

    def spy_train_epoch(key, buffer_state, *args, **kwargs):
        gc.collect()
        measured.append(
            (
                buffer_bytes(jax.tree.leaves(buffer_state)),
                buffer_bytes(jax.live_arrays()),
            )
        )
        return train_epoch(key, buffer_state, *args, **kwargs)

    monkeypatch.setattr(real, "train_epoch", spy_train_epoch)

    with warnings.catch_warnings():
        # every donated array is reused
        warnings.filterwarnings("error", message="Some donated buffers were not usable")
        real.train_loop(seed=0, num_epochs=2)

    assert len(measured) == 2
    for one, live in measured:
        assert one > 0
        assert live == one


def test_loaded_states_are_donated_to_self_play(real, real_run, tmp_path, monkeypatch):
    monkeypatch.setattr(real, "ckpt_dir", str(tmp_path))
    monkeypatch.setattr(real, "save_weights", lambda *args, **kwargs: None)
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

    # the loaded states were donated to the first self-play, and nothing read them after
    assert all(x.is_deleted() for x in jax.tree.leaves(resumed.collection_state))
    assert given[0].buffer_state is resumed.collection_state.buffer_state
    assert all(x.is_deleted() for x in jax.tree.leaves(given[first_forked]))
