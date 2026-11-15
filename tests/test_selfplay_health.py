"""selfplay_health.py's diversity measures of stored positions, on hand-made boards."""

import equinox as eqx
import numpy as np
import pytest
from example_modules import load_example

health = load_example("othello", "selfplay_health")[0]


def boards(contents):
    """Observations from boards of 0 (empty), 1 (player to move) and 2 (opponent)."""
    contents = np.asarray(contents)
    return np.stack([contents == 1, contents == 2], -1)


def random_boards(n, seed=0):
    return boards(np.random.default_rng(seed).integers(0, 3, (n, 8, 8)))


def pair_distances(*args, **kwargs):
    """The rows drawn and their mean disc distances."""
    rows, stats = health.pair_statistics(*args, **kwargs)
    return rows, stats["pair_distance"]


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

    rows, distances = pair_distances(
        variants, np.zeros(3, int), np.random.default_rng(0)
    )

    np.testing.assert_array_equal(distances, 0.0)
    assert sorted(rows.tolist()) == [0, 1, 2]


def test_pair_distances_are_each_positions_mean_distance_to_the_others():
    obs = random_boards(12)
    empties = np.array([30] * 7 + [20] * 5)

    rows, distances = pair_distances(obs, empties, np.random.default_rng(0))

    by_row = dict(zip(rows.tolist(), distances))
    for i in range(12):
        same = [j for j in range(12) if j != i and empties[j] == empties[i]]
        expected = np.mean([brute_force_distance(obs[i], obs[j]) for j in same])
        assert by_row[i] == expected


def test_pair_distances_draw_a_sample_of_a_crowded_count(monkeypatch):
    monkeypatch.setattr(health, "PAIR_SAMPLE", 5)

    rows, distances = pair_distances(
        random_boards(20), np.zeros(20, int), np.random.default_rng(0)
    )

    assert len(rows) == len(set(rows.tolist())) == 5
    assert (distances > 0).all()


def test_pair_distances_within_groups_compare_only_a_groups_positions():
    obs = random_boards(2)
    # two copies of each board, the copies of a board in a group of their own
    four = obs[[0, 0, 1, 1]]
    groups = np.array([7, 7, 3, 3], np.uint64)

    rows, distances = pair_distances(
        four, np.zeros(4, int), np.random.default_rng(0), groups
    )

    assert sorted(rows.tolist()) == [0, 1, 2, 3]
    np.testing.assert_array_equal(distances, 0.0)


def symmetries(board):
    """A board's 8 symmetries."""
    out = []
    for transpose in (False, True):
        other = np.swapaxes(board, 0, 1) if transpose else board
        out += [np.rot90(other, turns) for turns in range(4)]
    return out


def brute_force_occupancy(a, b):
    """Squares filled in `a` and empty in the closest of `b`'s symmetries, or the reverse."""
    return min(int((a.any(-1) != other.any(-1)).sum()) for other in symmetries(b))


def test_occupancy_distance_ignores_the_discs_colours():
    obs = random_boards(1)
    swapped = obs[..., ::-1]

    _, stats = health.pair_statistics(
        np.concatenate([obs, swapped]), np.zeros(2, int), np.random.default_rng(0)
    )

    np.testing.assert_array_equal(stats["occupancy_distance"], 0.0)
    assert (stats["pair_distance"] > 0).all()


def test_soft_collisions_are_the_mean_kernel_over_pairs():
    obs = random_boards(6, seed=3)

    rows, stats = health.pair_statistics(
        obs, np.zeros(6, int), np.random.default_rng(0)
    )

    by_row = {name: dict(zip(rows.tolist(), v)) for name, v in stats.items()}
    for i in range(6):
        others = [j for j in range(6) if j != i]
        discs = [brute_force_distance(obs[i], obs[j]) for j in others]
        shapes = [brute_force_occupancy(obs[i], obs[j]) for j in others]
        assert by_row["occupancy_distance"][i] == np.mean(shapes)
        for scale in health.SOFT_SCALES["disc"]:
            expected = np.mean(np.exp(-np.array(discs) / scale))
            assert by_row[f"soft_collision_disc_{scale:g}"][i] == pytest.approx(
                expected
            )


def test_copies_of_a_position_collide_fully():
    obs = random_boards(1)

    _, stats = health.pair_statistics(
        np.concatenate([obs, obs, np.rot90(obs, 2, axes=(1, 2))]),
        np.zeros(3, int),
        np.random.default_rng(0),
    )

    for name in ("disc", "occupancy"):
        for scale in health.SOFT_SCALES[name]:
            np.testing.assert_allclose(stats[f"soft_collision_{name}_{scale:g}"], 1.0)


def occupancy_features(observations):
    """A stand-in reference network: each symmetry's filled squares, of unit length."""
    sym = (
        health.symmetric_observations(observations)
        .any(-1)
        .reshape(8, len(observations), 64)
    )
    sym = sym.astype(float) + 1.0
    return sym / np.linalg.norm(sym, axis=-1, keepdims=True)


def test_feature_distance_is_the_closest_symmetrys_cosine_distance():
    obs = random_boards(4, seed=5)
    variants = np.concatenate([obs, np.rot90(obs[:1], 1, axes=(1, 2))])

    rows, stats = health.pair_statistics(
        variants,
        np.zeros(5, int),
        np.random.default_rng(0),
        features=occupancy_features,
    )

    f = occupancy_features(variants)
    by_row = dict(zip(rows.tolist(), stats["feature_distance"]))
    for i in range(5):
        expected = np.mean(
            [min(1 - f[0, i] @ f[s, j] for s in range(8)) for j in range(5) if j != i]
        )
        assert by_row[i] == pytest.approx(expected, abs=1e-12)
    assert all(
        f"soft_collision_feature_{s:g}" in stats for s in health.SOFT_SCALES["feature"]
    )


def test_reference_features_are_unit_length_for_every_symmetry():
    game = load_example("othello", "game")[0]
    network, state = game.make_network("resnet:1x4", seed=0)
    params = (eqx.filter(network, eqx.is_inexact_array), state)

    features = health.reference_features(network, params)(random_boards(3))

    assert features.shape == (8, 3, 4 * 64)
    np.testing.assert_allclose(np.linalg.norm(features, axis=-1), 1.0, rtol=1e-5)
