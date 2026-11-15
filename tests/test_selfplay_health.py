"""selfplay_health.py's diversity measures of stored positions, on hand-made boards."""

from types import SimpleNamespace

import equinox as eqx
import numpy as np
import pytest
from example_modules import load_example

health, engines = load_example("othello", "selfplay_health", "engines")
Analysis = engines.Analysis


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


def occupancy_maps(observations):
    """A stand-in reference network's maps, equivariant: each position's filled squares."""
    return observations.any(-1)[:, None].astype(float) + 1.0


def test_feature_distance_is_the_closest_symmetrys_cosine_distance():
    obs = random_boards(4, seed=5)
    variants = np.concatenate([obs, np.rot90(obs[:1], 1, axes=(1, 2))])

    rows, stats = health.pair_statistics(
        variants,
        np.zeros(5, int),
        np.random.default_rng(0),
        features=occupancy_maps(variants),
    )

    f = health.symmetric_maps(occupancy_maps(variants))
    by_row = dict(zip(rows.tolist(), stats["feature_distance"]))
    for i in range(5):
        expected = np.mean(
            [
                max(min(1 - f[0, i] @ f[s, j] for s in range(8)), 0.0)
                for j in range(5)
                if j != i
            ]
        )
        assert by_row[i] == pytest.approx(expected, abs=1e-6)
    # a position and its rotation are no distance apart
    _, pair = health.pair_statistics(
        variants[[0, 4]],
        np.zeros(2, int),
        np.random.default_rng(0),
        features=occupancy_maps(variants[[0, 4]]),
    )
    np.testing.assert_allclose(pair["feature_distance"], 0.0, atol=1e-6)
    assert all(
        f"soft_collision_feature_{s:g}" in stats for s in health.SOFT_SCALES["feature"]
    )


def test_reference_feature_maps_are_equivariant():
    game = load_example("othello", "game")[0]
    network, state = game.make_network("resnet:1x4", seed=0)
    params = (eqx.filter(network, eqx.is_inexact_array), state)
    maps = health.reference_features(network, params, batch=8)
    obs = random_boards(3)

    plain = maps(obs)
    turned = maps(np.rot90(np.swapaxes(obs, 1, 2), 3, axes=(1, 2)))

    assert plain.shape == (3, 4, 8, 8)
    np.testing.assert_allclose(
        turned, np.rot90(np.swapaxes(plain, -2, -1), 3, axes=(-2, -1)), atol=1e-5
    )


def test_label_metrics_by_hand():
    # three legal moves scoring +4, 0 and -2 discs for the player to move; the rest illegal
    scores = np.full((1, 65), health.ILLEGAL, np.int16)
    scores[0, [3, 10, 20]] = [4, 0, -2]
    policy = np.zeros((1, 65))
    policy[0, [3, 10, 20]] = [0.5, 0.25, 0.25]
    prior = np.zeros((1, 65))
    prior[0, [3, 10, 20]] = [0.2, 0.2, 0.6]

    metrics = health.label_metrics(scores, policy, prior)

    # 0.25 * 4 + 0.25 * 6, and 0.2 * 4 + 0.6 * 6
    assert metrics["edax_target_regret"][0] == pytest.approx(2.5)
    assert metrics["edax_prior_regret"][0] == pytest.approx(4.4)
    assert metrics["edax_target_top1"][0]


class CountingAnalyst:
    """A fake Edax: every legal-looking move (the empty squares) scores the square's index, and
    the positions it was asked about are counted."""

    calls = 0

    def __init__(self, level):
        self.level = level

    def __call__(self, board):
        CountingAnalyst.calls += 1
        return [
            Analysis(line=[int(i)], score=int(i), depth=self.level)
            for i in np.flatnonzero(board == 0)[:3]
        ]


def test_label_cache_scores_each_position_once_and_keeps_them(tmp_path):
    CountingAnalyst.calls = 0
    obs = random_boards(3, seed=7)
    cache = health.LabelCache(tmp_path, level=8, workers=0, factory=CountingAnalyst)

    first = cache(obs[[0, 1, 0]])
    again = health.LabelCache(tmp_path, level=8, workers=0, factory=CountingAnalyst)(
        obs[[1, 2]]
    )

    # positions 0 and 1, then only 2: the reloaded cache had the others
    assert CountingAnalyst.calls == 3
    np.testing.assert_array_equal(first[0], first[2])
    np.testing.assert_array_equal(again[0], first[1])
    assert (first[0] != health.ILLEGAL).sum() == 3


def test_after_offline_marks_a_games_moves_after_its_first_off_line_one():
    # two games of four moves (ply 8-11); the top move is 0. Game 1 goes off line at ply 9 (before
    # the start, ignored) and ply 10; game 2 stays on line
    policy = np.tile(np.eye(65)[0], (8, 1))
    moves = {
        "env": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        "ply": np.array([8, 9, 10, 11, 8, 9, 10, 11]),
        "action": np.array([0, 3, 3, 0, 0, 0, 0, 0]),
        "policy": policy,
        "legal": np.ones((8, 65), bool),
    }

    marked = health.after_offline(moves, start=10)

    np.testing.assert_array_equal(marked, [0, 0, 0, 1, 0, 0, 0, 0])


def test_the_probe_only_band_adds_endgame_positions_without_changing_the_others(
    monkeypatch,
):
    fake = SimpleNamespace(empties=np.random.default_rng(0).integers(5, 53, 2000))

    with_endgame = health.probe_set(fake, per_phase=50)
    monkeypatch.setattr(health, "PROBE_ONLY", {})
    without = health.probe_set(fake, per_phase=50)

    np.testing.assert_array_equal(with_endgame[: len(without)], without)
    extra = fake.empties[with_endgame[len(without) :]]
    assert len(extra) == 50 and ((extra >= 13) & (extra <= 20)).all()
