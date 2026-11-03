import json
import sys
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

import distill
from distill import (
    Dataset,
    Source,
    already_kept,
    load_dataset,
    load_positions,
    measure,
    outcomes,
    teacher_values_path,
    train,
    train_command,
    values_command,
)
from game import load_checkpoint, make_network

from core.memory.position_archive import ARCHIVE_META, positions_path, to_rows

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


def keep_positions(
    directory: Path, epochs: range, first_value: float = 0.0, parent: dict | None = None
) -> None:
    """A run's kept positions: 5 in each of `epochs`, numbered in `search_value` from `first_value`
    along the epochs."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ARCHIVE_META).write_text(json.dumps({"parent": parent}))
    template = load_template(directory)
    for i, epoch in enumerate(epochs):
        rows = to_rows(template, epoch)
        rows["search_value"] = first_value + 5 * i + np.arange(5)
        np.save(positions_path(directory, epoch), rows)


def load_template(tmp: Path):
    state = tmp / "template.npz"
    save_state(state)
    positions = load_positions(str(state))
    state.unlink()
    return jax.tree.map(lambda x: x[:5], positions)


def dataset_args(data, **more):
    return SimpleNamespace(
        **{
            "data": [str(p) for p in data],
            "from_epoch": None,
            "to_epoch": None,
            "value_source": "root",
            "teacher_values": None,
            "chunk_positions": 1000,
            **more,
        }
    )


def values_of(dataset: Dataset) -> list[float]:
    return sorted(dataset.take(np.arange(len(dataset))).search_value.tolist())


def test_a_fork_reads_its_ancestors_up_to_where_it_forked(tmp_path):
    trunk, fork = tmp_path / "trunk", tmp_path / "fork"
    keep_positions(trunk, range(4))
    parent = {"path": str(trunk / "state-2.npz"), "epoch": 2, "parent": None}
    keep_positions(fork, range(2), first_value=100, parent=parent)
    # the trunk's last 2 epochs aren't the fork's history
    assert values_of(load_dataset(dataset_args([fork]))) == [
        *range(10),
        *range(100, 110),
    ]
    # counted along the lineage: the fork's first epoch is 2
    assert values_of(load_dataset(dataset_args([fork], from_epoch=1, to_epoch=3))) == [
        *range(5, 10),
        *range(100, 105),
    ]
    # two forks share their ancestors' positions, once
    other = tmp_path / "other"
    keep_positions(other, range(1), first_value=200, parent=parent)
    assert len(load_dataset(dataset_args([fork, other]))) == 10 + 10 + 5


def test_a_final_buffer_counts_once_with_kept_positions(tmp_path):
    run = tmp_path / "run"
    keep_positions(run, range(3))
    state = tmp_path / "state.npz"
    save_state(state)
    rows = to_rows(load_positions(str(state)), 0)
    assert not already_kept(rows, [])[0]
    # the state's first positions were kept (by another epoch's file, whose column differs), the
    # rest weren't
    kept = to_rows(load_positions(str(state)), 7)[:4]
    np.save(positions_path(run, 3), kept)
    dataset = load_dataset(dataset_args([state, run]))
    assert len(dataset) == 3 * 5 + 4 + len(rows) - 4


def test_teacher_values_of_kept_positions_are_read_with_them(tmp_path):
    run = tmp_path / "run"
    keep_positions(run, range(2))
    network, nn_state = make_network(NET)
    teacher = tmp_path / "teacher.eqx"
    eqx.tree_serialise_leaves(
        teacher, (eqx.filter(network, eqx.is_inexact_array), nn_state)
    )
    out = tmp_path / "values"
    args = SimpleNamespace(
        teacher=str(teacher),
        teacher_net=NET,
        data=[str(run)],
        out=str(out),
        from_epoch=None,
        to_epoch=None,
    )
    values_command(args)
    computed = teacher_values_path(out, positions_path(run, 1))
    assert np.load(computed).shape == (5,)
    # a second run only computes what's missing
    computed.write_bytes(b"kept")
    values_command(args)
    assert computed.read_bytes() == b"kept"
    np.save(computed, np.full(5, 0.25, np.float32))
    np.save(
        teacher_values_path(out, positions_path(run, 0)), np.full(5, 0.5, np.float32)
    )
    dataset = load_dataset(
        dataset_args([run], value_source="teacher", teacher_values=str(out))
    )
    assert values_of(dataset) == [0.25] * 5 + [0.5] * 5


def test_batches_in_chunks_see_every_position_each_pass(tmp_path, monkeypatch):
    run = tmp_path / "run"
    keep_positions(run, range(40))
    monkeypatch.setattr(distill, "BLOCK", 2)
    dataset = load_dataset(dataset_args([run], chunk_positions=32))
    assert len(dataset) == 200
    batches = dataset.batches(8, np.random.default_rng(0))
    # whole chunks of 4 batches, as near a pass as they come, then nearly 3 passes in all
    seen, on_device = [], None
    for _ in range(18):
        for _ in range(4):
            on_device, indices = next(batches)
            assert len(on_device.reward) == 32
            seen += np.asarray(on_device.search_value[indices]).tolist()
        if len(seen) == 192:
            assert len(set(seen)) == 192  # no position twice in a pass
    values, counts = np.unique(seen, return_counts=True)
    assert len(values) == 200 and set(counts) <= {2, 3}
    # blocks from across the dataset: a chunk isn't a run of epochs
    assert on_device is not None and np.ptp(on_device.search_value) > 32


def test_all_positions_are_held_if_they_fit(tmp_path):
    run = tmp_path / "run"
    keep_positions(run, range(2))
    dataset = load_dataset(dataset_args([run]))
    batches = dataset.batches(4, np.random.default_rng(0))
    on_device, _ = next(batches)
    assert len(on_device.reward) == 10
    assert next(batches)[0] is on_device


def test_take_returns_positions_in_the_order_asked(tmp_path):
    a, b = np.arange(3), np.arange(3, 7)
    template = load_template(tmp_path)
    sources = []
    for values in (a, b):
        rows = to_rows(template, 0)[: len(values)]
        rows["search_value"] = values
        sources.append(Source(rows))
    taken = Dataset(sources).take(np.array([5, 0, 3, 2]))
    assert taken.search_value.tolist() == [5, 0, 3, 2]


def test_training_on_kept_positions_too_many_for_a_chunk(tmp_path):
    run, heldout = tmp_path / "run", tmp_path / "held" / "state.npz"
    keep_positions(run, range(10))
    heldout.parent.mkdir()
    save_state(heldout)
    train_command(
        SimpleNamespace(
            data=[str(run)],
            from_epoch=2,
            to_epoch=None,
            chunk_positions=16,
            heldout=str(heldout),
            heldout_size=5,
            teacher_values=None,
            value_source="root",
            value_target_q=0.0,
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
        Dataset.of(positions),
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
            from_epoch=None,
            to_epoch=None,
        )
    )
    # one value per position, in `load_positions`' order
    values = np.load(teacher_values_path(out, str(data)))
    assert values.shape == (len(load_positions(str(data)).reward),)
    train_command(
        SimpleNamespace(
            data=[str(data)],
            from_epoch=None,
            to_epoch=None,
            chunk_positions=None,
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
