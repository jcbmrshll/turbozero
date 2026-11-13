"""The move log (core/memory/move_log.py); the trainer's `log_moves` is tested in test_trainer.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.memory.move_log import (
    ENDED,
    GOES_ON,
    LoggedMove,
    MoveLogWriter,
    emptied,
    init_move_log,
    log_move,
    to_rows,
    unpack,
)


@pytest.fixture(scope="module")
def env(fixed_length_env):
    """Games of 3 moves that pay (1, -1), with 2 actions."""
    return fixed_length_env(length=3)


def a_move(seed: float, end: int = GOES_ON) -> LoggedMove:
    """A move of a log with (2, 2) boolean observations and 3 actions, its fields from `seed`."""
    return LoggedMove(
        observation=jnp.array([[True, False], [False, seed > 0]]),
        policy=jnp.array([seed, 1 - seed, 0.0]),
        legal=jnp.array([True, True, False]),
        value=jnp.array(seed),
        action=jnp.array(1),
        player=jnp.array(0),
        full=jnp.array(True),
        ply=jnp.array(int(seed * 10)),
        game=jnp.array(99),
        end=jnp.array(end, jnp.int8),
        result=jnp.array(0.0),
    )


def one_env_log(length: int):
    log = init_move_log(1, length, jnp.zeros((2, 2), bool), 3)
    return jax.tree.map(lambda x: x[0], log)


def test_a_log_keeps_the_moves_played_and_counts_games():
    log = one_env_log(4)
    log = log_move(log, a_move(0.1), jnp.array(True))
    log = log_move(log, a_move(0.9), jnp.array(False))
    log = log_move(log, a_move(0.2, ENDED), jnp.array(True))
    log = log_move(log, a_move(0.3), jnp.array(True))
    assert int(log.count) == 3
    np.testing.assert_allclose(log.rows.value[:3], [0.1, 0.2, 0.3])
    # the game comes from the log, and the next game starts after a move that ends one
    np.testing.assert_array_equal(log.rows.game[:3], [0, 0, 1])
    assert int(log.games) == 1
    log = emptied(log)
    assert int(log.count) == 0 and int(log.games) == 1


def test_files_pack_booleans_and_unpack_them_as_logged():
    log = init_move_log(2, 3, jnp.zeros((2, 2), bool), 3)
    for env, seeds in enumerate([(0.25, 0.5), (0.75,)]):
        one = jax.tree.map(lambda x, e=env: x[e], log)
        for seed in seeds:
            one = log_move(one, a_move(seed), jnp.array(True))
        log = jax.tree.map(lambda full, x, e=env: full.at[e].set(x), log, one)
    rows = to_rows(jax.device_get(log.rows), np.asarray(log.count), epoch=7)
    np.testing.assert_array_equal(rows["env"], [0, 0, 1])
    np.testing.assert_array_equal(rows["epoch"], [7, 7, 7])
    assert rows["observation"].dtype == np.uint8 and rows["policy"].dtype == np.float16
    meta = {"packed": {"observation": [2, 2], "legal": [3]}}
    expected = np.array([a_move(s).observation for s in (0.25, 0.5, 0.75)])
    np.testing.assert_array_equal(unpack(rows, "observation", meta), expected)
    np.testing.assert_array_equal(
        unpack(rows, "legal", meta), np.tile([True, True, False], (3, 1))
    )
    np.testing.assert_allclose(rows["policy"][:, 0], [0.25, 0.5, 0.75])


def test_the_writer_refuses_a_directory_with_moves(tmp_path):
    writer = MoveLogWriter(str(tmp_path), num_envs=1)
    log = init_move_log(1, 2, jnp.zeros((2, 2), bool), 3)
    writer.write(0, log)
    writer.flush()
    writer.close()
    with pytest.raises(ValueError, match="holds moves already"):
        MoveLogWriter(str(tmp_path), num_envs=1)


def test_the_writer_refuses_a_log_that_dropped_moves(tmp_path):
    writer = MoveLogWriter(str(tmp_path), num_envs=1)
    log = init_move_log(1, 1, jnp.zeros((2, 2), bool), 3)
    one = jax.tree.map(lambda x: x[0], log)
    for _ in range(2):
        one = log_move(one, a_move(0.5), jnp.array(True))
    with pytest.raises(ValueError, match="weren't logged"):
        writer.write(0, jax.tree.map(lambda x: x[None], one))
    writer.close()
