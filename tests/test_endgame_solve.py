"""Exact endgame labels in self-play (core.training.endgame_solve), through a `Trainer`'s
collection and labelling.

Uses `make_fixed_length_env` (see conftest), whose games end after exactly `length` moves, and a
fake solver: the player to move at the stopped position wins."""

import jax
import numpy as np
import pytest

from core.training.endgame_solve import EndgameSolve
from core.training.train import CollectionStats

LENGTH = 6
# games stop before their 4th move: 4 positions stored, the stopped one included
STOP = 3


def mover_wins(positions):
    """The fake solver: the player to move at each stopped position wins."""
    player = np.asarray(positions.current_player)
    results = np.full((len(player), 2), -1.0, dtype=np.float32)
    results[np.arange(len(player)), player] = 1.0
    return results


def endgame_solve(slots=4, share=1.0):
    return EndgameSolve(
        stop_fn=lambda state: state.step >= STOP,
        solve_fn=mover_wins,
        slots=slots,
        share=share,
    )


def collector(make_collector, env, scripted, tmp_path, **kwargs):
    return make_collector(
        env,
        scripted.first_legal,
        max_episode_steps=16,
        ckpt_dir=tmp_path,
        endgame_solve=endgame_solve(**kwargs),
    )


def sampleable(state):
    buffer = state.buffer_state
    return np.flatnonzero(np.asarray(buffer.populated & buffer.has_reward)[0])


def solve_and_label(trainer, state):
    """Solves the games waiting in `state`, then labels them, as the training loop does (with no
    self-play in between)."""
    solved = trainer.endgame_solve.solve(jax.tree.map(np.asarray, state.pending))
    return trainer.label_solved_games(state, solved)


def test_a_game_stops_before_its_move_and_waits_for_its_result(
    fixed_length_env, make_collector, scripted, tmp_path
):
    env = fixed_length_env(length=LENGTH)
    state = collector(make_collector, env, scripted, tmp_path)(STOP + 1)

    pending = state.pending
    # the stopped position is kept for the solver, and a new game started
    np.testing.assert_array_equal(pending.added, [1])
    np.testing.assert_array_equal(pending.labelled, [0])
    np.testing.assert_array_equal(pending.positions.step[0, 0], STOP)
    np.testing.assert_array_equal(state.metadata.step, [0])
    # its positions, the stopped one included, wait in the buffer, unsampleable, and it hasn't
    # counted as an episode yet
    assert len(sampleable(state)) == 0
    np.testing.assert_array_equal(pending.entry_game[0, : STOP + 1], 0)
    np.testing.assert_array_equal(pending.entry_game[0, STOP + 1 :], -1)
    np.testing.assert_array_equal(state.episodes, [0])


def test_labelling_gives_every_position_the_solved_result(
    fixed_length_env, make_collector, scripted, tmp_path
):
    env = fixed_length_env(length=LENGTH, rewards=(1.0, -1.0))
    collect = collector(make_collector, env, scripted, tmp_path)

    state = solve_and_label(collect.trainer, collect(STOP + 1))

    # player 1 moves at the stopped position, so wins, though the game played out would be
    # player 0's
    np.testing.assert_array_equal(sampleable(state), np.arange(STOP + 1))
    np.testing.assert_array_equal(
        state.buffer_state.buffer.reward[0, : STOP + 1], [[-1.0, 1.0]] * (STOP + 1)
    )
    np.testing.assert_array_equal(state.pending.labelled, [1])
    np.testing.assert_array_equal(state.pending.entry_game[0], -1)
    np.testing.assert_array_equal(state.episodes, [1])


def test_only_the_games_solved_are_labelled(
    fixed_length_env, make_collector, scripted, tmp_path
):
    env = fixed_length_env(length=LENGTH)
    collect = collector(make_collector, env, scripted, tmp_path)
    trainer = collect.trainer
    # one game stopped when the solve started, another while it ran
    solved = trainer.endgame_solve.solve(
        jax.tree.map(np.asarray, collect(STOP + 1).pending)
    )
    state = collect(2 * (STOP + 1))

    state = trainer.label_solved_games(state, solved)

    np.testing.assert_array_equal(sampleable(state), np.arange(STOP + 1))
    np.testing.assert_array_equal(state.pending.labelled, [1])
    np.testing.assert_array_equal(state.pending.added, [2])
    # the second still waits, and is labelled next time
    state = solve_and_label(trainer, state)
    np.testing.assert_array_equal(sampleable(state), np.arange(2 * (STOP + 1)))


def test_a_solve_reports_each_games_result_from_the_oldest(
    fixed_length_env, make_collector, scripted, tmp_path
):
    env = fixed_length_env(length=LENGTH)
    collect = collector(make_collector, env, scripted, tmp_path, slots=2)
    trainer = collect.trainer
    state = solve_and_label(trainer, collect(STOP + 1))
    # the 2nd and 3rd games stop, the 3rd in slot 0 again
    state = collect.step(jax.random.split(jax.random.PRNGKey(9), 1), state)
    for i in range(2 * (STOP + 1) - 1):
        state = collect.step(jax.random.split(jax.random.PRNGKey(10 + i), 1), state)

    solved = trainer.endgame_solve.solve(jax.tree.map(np.asarray, state.pending))

    np.testing.assert_array_equal(solved.counts, [2])
    np.testing.assert_array_equal(solved.games, [[-1.0, 1.0]] * 2)
    # by slot: the 2nd game's in slot 1, the 3rd's in slot 0
    np.testing.assert_array_equal(solved.results[0], [[-1.0, 1.0]] * 2)


def test_a_game_finding_every_slot_full_plays_on(
    fixed_length_env, make_collector, scripted, tmp_path
):
    env = fixed_length_env(length=LENGTH, rewards=(1.0, -1.0))
    collect = collector(make_collector, env, scripted, tmp_path, slots=1)

    # the first game stops; the second finds its slot taken and is played to the end
    state = collect(STOP + 1 + LENGTH)

    np.testing.assert_array_equal(state.pending.added, [1])
    np.testing.assert_array_equal(state.pending.missed, [1])
    np.testing.assert_array_equal(state.episodes, [1])
    np.testing.assert_array_equal(sampleable(state), STOP + 1 + np.arange(LENGTH))
    np.testing.assert_array_equal(
        state.buffer_state.buffer.reward[0, STOP + 1 : STOP + 1 + LENGTH],
        np.tile(env.rewards, (LENGTH, 1)),
    )


def test_with_a_share_of_zero_every_game_plays_out(
    fixed_length_env, make_collector, scripted, tmp_path
):
    env = fixed_length_env(length=LENGTH)
    state = collector(make_collector, env, scripted, tmp_path, share=0.0)(LENGTH)

    np.testing.assert_array_equal(state.pending.added, [0])
    np.testing.assert_array_equal(state.pending.missed, [0])
    np.testing.assert_array_equal(sampleable(state), np.arange(LENGTH))


def test_a_waiting_game_written_over_is_not_labelled(
    fixed_length_env, make_collector, scripted, tmp_path
):
    env = fixed_length_env(length=LENGTH)
    collect = collector(make_collector, env, scripted, tmp_path, slots=8)
    trainer = collect.trainer
    solved = trainer.endgame_solve.solve(
        jax.tree.map(np.asarray, collect(STOP + 1).pending)
    )
    # 5 games of 4 positions in a buffer of 16: the 5th writes over the 1st
    state = collect(5 * (STOP + 1))

    state = trainer.label_solved_games(state, solved)

    assert len(sampleable(state)) == 0
    np.testing.assert_array_equal(state.pending.entry_game[0, : STOP + 1], 4)


def test_a_resized_buffer_keeps_its_waiting_games(
    fixed_length_env, make_collector, scripted, tmp_path
):
    env = fixed_length_env(length=LENGTH)
    collect = collector(make_collector, env, scripted, tmp_path)
    trainer = collect.trainer
    # 2 games waiting, then 2 positions of a game in progress
    state = collect(2 * (STOP + 1) + 2)

    state = trainer.resize_replay_buffer(state, 6)

    # the newest 6 entries, oldest first: the 2nd game, then the game in progress (the 1st is
    # dropped)
    np.testing.assert_array_equal(state.pending.entry_game[0], [1] * 4 + [-1] * 2)
    state = solve_and_label(trainer, state)
    np.testing.assert_array_equal(sampleable(state), np.arange(4))
    np.testing.assert_array_equal(state.episodes, [2])


def test_collection_metrics_count_missed_games(
    fixed_length_env, make_collector, scripted, tmp_path
):
    env = fixed_length_env(length=LENGTH)
    collect = collector(make_collector, env, scripted, tmp_path, slots=1)

    metrics = collect.trainer.collection_metrics(
        CollectionStats.of(collect(0)), collect(STOP + 1 + LENGTH)
    )

    assert int(metrics["solve_missed"]) == 1
    # the played-out game; the stopped one counts when it's labelled
    assert int(metrics["episodes"]) == 1


@pytest.mark.parametrize("kwargs", [{"slots": 0}, {"share": 1.5}, {"share": -0.1}])
def test_settings_out_of_range_are_rejected(kwargs):
    with pytest.raises(ValueError):
        endgame_solve(**kwargs)
