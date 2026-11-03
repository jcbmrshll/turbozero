import json
import sys
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from distill import (
    load_positions,
    measure,
    outcomes,
    teacher_values_path,
    train,
    train_command,
    values_command,
    with_value_source,
)
from game import load_checkpoint, make_network

NET = "resnet:1x4"
ENVS, CAPACITY = 2, 8


def save_state(path: Path) -> np.ndarray:
    """A training state as `Trainer.save_state` keeps its replay buffer. A game's in progress at
    the last place of each environment, which can't be sampled, and the first place of the first
    environment is empty. Returns which places can be sampled."""
    rng = np.random.default_rng(0)
    shape = (ENVS, CAPACITY)
    policy = np.zeros((*shape, 65), np.float32)
    policy[..., 19] = 1.0
    mask = np.zeros((*shape, 65), bool)
    mask[..., [19, 26]] = True
    populated = np.ones(shape, bool)
    populated[0, 0] = False
    has_reward = np.ones(shape, bool)
    has_reward[:, -1] = False
    player = rng.integers(0, 2, shape).astype(np.int32)
    # the player to move always wins
    reward = np.where(player[..., None] == 0, [1.0, -1.0], [-1.0, 1.0])
    leaves = {
        ".buffer_state.buffer.reward": reward.astype(np.float32),
        ".buffer_state.buffer.policy_weights": policy,
        ".buffer_state.buffer.policy_mask": mask,
        ".buffer_state.buffer.observation_nn": rng.random((*shape, 8, 8, 2)) < 0.3,
        ".buffer_state.buffer.cur_player_id": player,
        ".buffer_state.buffer.search_value": np.arange(
            ENVS * CAPACITY, dtype=np.float32
        ).reshape(shape),
        ".buffer_state.populated": populated,
        ".buffer_state.has_reward": has_reward,
    }
    meta = {"leaves": {"collection_state": [[p, 0, 0] for p in leaves]}}
    np.savez(
        path,
        meta=json.dumps(meta),
        **{f"collection_state.{i}": leaf for i, leaf in enumerate(leaves.values())},
    )
    return populated & has_reward


def test_load_positions_keeps_only_those_that_can_be_sampled(tmp_path):
    path = tmp_path / "state.npz"
    valid = save_state(path)
    positions = load_positions(str(path))
    assert len(positions.reward) == valid.sum()
    expected = np.arange(ENVS * CAPACITY).reshape(valid.shape)[valid]
    assert (positions.search_value == expected).all()
    assert (outcomes(positions) == 1).all()


def test_with_value_source(tmp_path):
    path = tmp_path / "state.npz"
    save_state(path)
    positions = load_positions(str(path))
    assert with_value_source(positions, "root", None) is positions
    teacher = np.full(len(positions.reward), 0.25, np.float32)
    assert (with_value_source(positions, "teacher", teacher).search_value == 0.25).all()
    with pytest.raises(ValueError, match="teacher values"):
        with_value_source(positions, "teacher", None)


def test_train_learns_and_saves_loadable_checkpoints(tmp_path):
    path = tmp_path / "state.npz"
    save_state(path)
    positions = load_positions(str(path))
    network, nn_state = make_network(NET)
    untrained, _ = measure(
        network, (eqx.filter(network, eqx.is_inexact_array), nn_state), positions
    )
    logged = {}
    train(
        network,
        nn_state,
        positions,
        positions,
        None,
        passes=100,
        batch_size=8,
        lr=1e-2,
        lr_final=1e-3,
        weight_decay=0.0,
        value_target_q=0.0,
        seed=0,
        eval_every=100,
        train_probe_size=4,
        ckpt_dir=tmp_path / "ckpts",
        log=lambda finished, metrics: logged.update({finished: metrics}),
    )
    # the one measurement, after the last pass
    assert list(logged) == [100]
    trained = logged[100]
    assert trained["heldout_policy_kl"] < untrained["policy_kl"] / 2
    assert trained["probe_value_mse_z"] < untrained["value_mse_z"] / 2
    load_checkpoint(str(tmp_path / "ckpts" / "100.eqx"), NET)


def test_values_then_train_with_teacher_targets(tmp_path):
    data, heldout = tmp_path / "a" / "state.npz", tmp_path / "b" / "state.npz"
    data.parent.mkdir()
    heldout.parent.mkdir()
    save_state(data)
    save_state(heldout)
    network, nn_state = make_network(NET)

    teacher = tmp_path / "teacher.eqx"
    eqx.tree_serialise_leaves(
        teacher, (eqx.filter(network, eqx.is_inexact_array), nn_state)
    )
    out = tmp_path / "values"
    values_command(
        SimpleNamespace(
            teacher=str(teacher),
            teacher_net=NET,
            data=[str(data), str(heldout)],
            out=str(out),
        )
    )
    # one value per position, in `load_positions`' order
    values = np.load(teacher_values_path(out, str(data)))
    assert values.shape == (len(load_positions(str(data)).reward),)
    train_command(
        SimpleNamespace(
            data=str(data),
            corpus=None,
            init_weights=None,
            policy_source="teacher",
            searchers=None,
            heldout=str(heldout),
            heldout_size=5,
            teacher_values=str(out),
            value_source="teacher",
            value_target_q=1.0,
            net=NET,
            train_dtype="float32",
            seed=0,
            monitor=None,
            name=None,
            notes=None,
            passes=1,
            batch=4,
            lr=1e-3,
            lr_final=1e-4,
            weight_decay=0.5,
            eval_every=1,
            ckpt_dir=str(tmp_path / "student"),
        )
    )
    load_checkpoint(str(tmp_path / "student" / "1.eqx"), NET)
