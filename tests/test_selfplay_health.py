"""selfplay_health.py's diversity measures of stored positions, on hand-made boards."""

import numpy as np
from example_modules import load_example

health = load_example("othello", "selfplay_health")[0]


def boards(contents):
    """Observations from boards of 0 (empty), 1 (player to move) and 2 (opponent)."""
    contents = np.asarray(contents)
    return np.stack([contents == 1, contents == 2], -1)


def random_boards(n, seed=0):
    return boards(np.random.default_rng(seed).integers(0, 3, (n, 8, 8)))


def brute_force_distance(a, b):
    """Squares that differ between `a` and the closest of `b`'s 8 symmetries."""
    best = 64
    for transpose in (False, True):
        other = np.swapaxes(b, 0, 1) if transpose else b
        for turns in range(4):
            turned = np.rot90(other, turns)
            best = min(best, int((~(a == turned).all(-1)).sum()))
    return best


def test_a_positions_symmetries_are_no_distance_apart():
    obs = random_boards(1)
    variants = np.concatenate(
        [obs, np.rot90(obs, 1, axes=(1, 2)), np.swapaxes(obs, 1, 2)]
    )

    rows, distances = health.pair_distances(
        variants, np.zeros(3, int), np.random.default_rng(0)
    )

    np.testing.assert_array_equal(distances, 0.0)
    assert sorted(rows.tolist()) == [0, 1, 2]


def test_pair_distances_are_each_positions_mean_distance_to_the_others():
    obs = random_boards(12)
    empties = np.array([30] * 7 + [20] * 5)

    rows, distances = health.pair_distances(obs, empties, np.random.default_rng(0))

    by_row = dict(zip(rows.tolist(), distances))
    for i in range(12):
        same = [j for j in range(12) if j != i and empties[j] == empties[i]]
        expected = np.mean([brute_force_distance(obs[i], obs[j]) for j in same])
        assert by_row[i] == expected


def test_pair_distances_draw_a_sample_of_a_crowded_count(monkeypatch):
    monkeypatch.setattr(health, "PAIR_SAMPLE", 5)

    rows, distances = health.pair_distances(
        random_boards(20), np.zeros(20, int), np.random.default_rng(0)
    )

    assert len(rows) == len(set(rows.tolist())) == 5
    assert (distances > 0).all()


def test_pair_distances_within_groups_compare_only_a_groups_positions():
    obs = random_boards(2)
    # two copies of each board, the copies of a board in a group of their own
    four = obs[[0, 0, 1, 1]]
    groups = np.array([7, 7, 3, 3], np.uint64)

    rows, distances = health.pair_distances(
        four, np.zeros(4, int), np.random.default_rng(0), groups
    )

    assert sorted(rows.tolist()) == [0, 1, 2, 3]
    np.testing.assert_array_equal(distances, 0.0)
