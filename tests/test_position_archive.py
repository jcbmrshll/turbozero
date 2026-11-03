"""Which positions a run keeps, across the replay buffer's stages, with real replay buffers filled
by hand. How they get there from a real training loop is in test_trainer.py."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.memory.position_archive import (
    ARCHIVE_META,
    PositionArchive,
    drop_positions_from,
    lineage,
    position_epochs,
    positions_path,
    select_files,
)
from core.memory.replay_memory import BaseExperience, EpisodeReplayBuffer

ENVS = 2


def experience(value: float) -> BaseExperience:
    """An experience whose `search_value` tells it apart."""
    return BaseExperience(
        reward=jnp.zeros(2),
        policy_weights=jnp.array([1.0, 0.0]),
        policy_mask=jnp.array([True, False]),
        observation_nn=jnp.zeros((2, 2), dtype=bool),
        cur_player_id=jnp.array(0),
        search_value=jnp.array(value),
    )


class Selfplay:
    """Replay buffers played by hand, numbering each environment's positions in order: the tens
    say which environment."""

    def __init__(self, tmp_path, capacity: int, from_epoch: int = 0):
        self.buffer = EpisodeReplayBuffer(capacity=capacity)
        self.state = self.buffer.init(ENVS, experience(0.0))
        self.moves = jnp.zeros(ENVS, dtype=jnp.int32)
        self.played = [0] * ENVS
        self.dir = tmp_path
        self.archive = PositionArchive(str(tmp_path), capacity, from_epoch, None)
        self.archive.start(self.state, self.moves)

    def _map(self, env: int, fn):
        one = jax.tree.map(lambda x: x[env], self.state)
        self.state = jax.tree.map(
            lambda whole, part: whole.at[env].set(part), self.state, fn(one)
        )

    def play(self, env: int, moves: int) -> None:
        for _ in range(moves):
            value = float(100 * env + self.played[env])
            self.played[env] += 1
            self._map(
                env, lambda s, v=value: self.buffer.add_experience(s, experience(v))
            )
        self.moves = self.moves.at[env].add(moves)

    def finish(self, env: int) -> None:
        reward = jnp.array([1.0, -1.0])
        self._map(env, lambda s: self.buffer.assign_rewards(s, reward))

    def truncate(self, env: int) -> None:
        self._map(env, self.buffer.truncate)

    def save(self, epoch: int) -> None:
        self.archive.save(epoch, self.state, self.moves)
        self.archive.flush()

    def kept(self, epoch: int) -> list[int]:
        rows = np.load(positions_path(self.dir, epoch))
        assert (rows["epoch"] == epoch).all()
        # finished games only: every one has its outcome
        assert (rows["reward"][:, 0] == 1.0).all()
        return rows["search_value"].astype(int).tolist()


def test_positions_are_saved_once_as_their_games_finish(tmp_path):
    play = Selfplay(tmp_path, capacity=16)
    # the buffer's first stage, partly filled: a finished game, and one in progress
    play.play(0, 3)
    play.finish(0)
    play.play(0, 2)
    play.play(1, 2)
    play.save(0)
    assert play.kept(0) == [0, 1, 2]
    # that game finishes, with another that starts and doesn't
    play.play(0, 1)
    play.finish(0)
    play.play(0, 1)
    play.play(1, 1)
    play.finish(1)
    play.save(1)
    assert play.kept(1) == [3, 4, 5, 100, 101, 102]
    # nothing finished
    play.save(2)
    assert play.kept(2) == []


def test_positions_are_saved_across_the_buffer_wrapping_around(tmp_path):
    play = Selfplay(tmp_path, capacity=8)
    kept = []
    for epoch in range(10):
        for _ in range(2):
            play.play(0, 3)
            play.finish(0)
        play.save(epoch)
        kept += play.kept(epoch)
    # all of 60 positions through a buffer of 8, in order
    assert kept == list(range(60))


def test_a_truncated_game_is_never_saved(tmp_path):
    play = Selfplay(tmp_path, capacity=16)
    play.play(0, 2)
    play.finish(0)
    play.play(0, 3)
    play.save(0)
    play.play(0, 1)
    play.truncate(0)
    play.play(0, 1)
    play.finish(0)
    play.save(1)
    # positions 2-5 were truncated; 6 took the place of 2
    assert play.kept(0) == [0, 1]
    assert play.kept(1) == [6]


def test_saving_starts_at_the_first_epoch_asked_for(tmp_path):
    play = Selfplay(tmp_path, capacity=16, from_epoch=1)
    play.play(0, 2)
    play.finish(0)
    play.play(0, 1)
    play.save(0)
    assert position_epochs(tmp_path) == []
    # the game in progress finishes in the first epoch kept, so all of it is kept
    play.play(0, 1)
    play.finish(0)
    play.save(1)
    assert play.kept(1) == [2, 3]


def test_a_buffer_too_small_to_keep_everything_is_an_error(tmp_path):
    play = Selfplay(tmp_path, capacity=4)
    play.play(0, 2)
    play.finish(0)
    play.save(0)
    play.play(0, 4)
    with pytest.raises(ValueError, match="use a larger one"):
        play.save(1)


def test_a_failed_write_stops_the_run(tmp_path, monkeypatch):
    play = Selfplay(tmp_path, capacity=8)
    monkeypatch.setattr(np, "save", lambda *_: (_ for _ in ()).throw(OSError("full")))
    play.play(0, 2)
    play.finish(0)
    with pytest.raises(OSError, match="full"):
        play.save(0)


def test_a_partial_write_is_not_a_file_of_positions(tmp_path):
    (tmp_path / "positions-3.npy").touch()
    (tmp_path / "positions-4.npy.tmp").touch()
    (tmp_path / "positions-5.npy.part").touch()
    (tmp_path / "state-9.npz").touch()
    assert position_epochs(tmp_path) == [3]


def test_going_back_drops_the_files_from_there(tmp_path):
    for epoch in (1, 2, 3):
        (tmp_path / f"positions-{epoch}.npy").touch()
    drop_positions_from(tmp_path, 2)
    assert position_epochs(tmp_path) == [1]


def run(directory, epochs, parent=None):
    directory.mkdir()
    (directory / ARCHIVE_META).write_text(json.dumps({"parent": parent}))
    for epoch in epochs:
        (directory / f"positions-{epoch}.npy").touch()


def forked_from(directory, epoch, parent=None):
    return {"path": str(directory / f"state-{epoch}.npz"), "epoch": epoch, "parent": parent}  # fmt: skip


def test_lineage_counts_epochs_across_forks_of_forks(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    run(a, range(5))
    run(b, range(5), forked_from(a, 3))
    run(c, range(2), forked_from(b, 2, forked_from(a, 3)))
    assert [(f.path, f.epoch) for f in lineage(c)] == [
        (str(a / "positions-0.npy"), 0),
        (str(a / "positions-1.npy"), 1),
        (str(a / "positions-2.npy"), 2),
        (str(b / "positions-0.npy"), 3),
        (str(b / "positions-1.npy"), 4),
        (str(c / "positions-0.npy"), 5),
        (str(c / "positions-1.npy"), 6),
    ]
    assert [f.epoch for f in select_files([c], from_epoch=2, to_epoch=5)] == [2, 3, 4]


def test_an_ancestor_that_saved_nothing_is_reported_and_skipped(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    run(b, range(1), forked_from(a, 3))
    assert [f.epoch for f in lineage(b)] == [3]
    assert "no positions were saved" in capsys.readouterr().out
