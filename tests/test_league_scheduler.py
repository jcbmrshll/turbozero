"""How the league picks its next batch."""

from collections import Counter
from typing import Any

import numpy as np
import pytest

from core.league.ratings import K, Ratings, expected_score, fit_ratings
from core.league.scheduler import (
    SchedulerConfig,
    batch_size,
    choose_openings,
    gains,
    network_size,
    next_pairing,
    opponent_weights,
    seconds_per_game,
)
from core.league.store import PairRecord, Player


def nets(count, sims=128, run=None):
    return [
        Player(
            id=i,
            name=f"p{i}",
            sims=sims,
            network=f"n{i}",
            net="resnet:8x128",
            run=run or f"r{i}",
        )
        for i in range(count)
    ]


def engine(id_):
    return Player(id=id_, name=f"e{id_}", external="edax", run="edax")


def known(elo, sd: Any = 30.0):
    """Ratings at `elo`, each independently uncertain by `sd` (a number or one per player)."""
    n = len(elo)
    return Ratings(
        list(range(n)),
        np.array(elo, dtype=float),
        np.diag(np.broadcast_to(np.asarray(sd, dtype=float) ** 2, (n,))),
        np.zeros(n, dtype=int),
        np.full(n, 100),
    )


def draws(config, ratings, players, costs=None, count=400, seed=0):
    rng = np.random.default_rng(seed)
    n = len(players)
    costs = np.full((n, n), 0.2) if costs is None else costs
    picks = [next_pairing(config, ratings, players, costs, rng) for _ in range(count)]
    return [p for p in picks if p is not None]


NEAR_ONLY = SchedulerConfig(far_fraction=0.0, external_fraction=0.0)


def test_gain_is_the_exact_drop_in_total_variance():
    fit = fit_ratings(
        range(3), [PairRecord(0, 1, 20, 0, 20), PairRecord(1, 2, 8, 0, 4)]
    )
    games = np.full((3, 3), 64.0)
    gain = gains(fit, games)
    p = expected_score(fit.elo[0] - fit.elo[2])
    d = np.array([1.0, 0.0, -1.0])
    precision = np.linalg.pinv(fit.cov) + 64 * K**2 * p * (1 - p) * np.outer(d, d)
    after = np.linalg.pinv(precision)
    assert gain[0, 2] == pytest.approx(np.trace(fit.cov) - np.trace(after), rel=1e-4)
    assert gain[0, 0] == 0


def test_opponents_near_in_rating_are_drawn_most():
    players = nets(5)
    ratings = known([0, 50, 200, 400, 800])
    weights = opponent_weights(NEAR_ONLY, ratings, players, np.full((5, 5), 0.2))
    assert weights[0, 1] > weights[0, 2] > weights[0, 3] > weights[0, 4]
    assert weights.sum(axis=1) == pytest.approx(np.ones(5))
    assert (np.diag(weights) == 0).all()


def test_uncertain_estimates_widen_the_draw():
    players = nets(3)
    costs = np.full((3, 3), 0.2)
    sure = opponent_weights(NEAR_ONLY, known([0, 100, 600], 5), players, costs)
    unsure = opponent_weights(NEAR_ONLY, known([0, 100, 600], 1000), players, costs)
    assert unsure[0, 2] > 5 * sure[0, 2]
    assert unsure[0, 1] == pytest.approx(unsure[0, 2], rel=0.2)


def test_the_most_uncertain_player_is_focal():
    players = nets(4)
    ratings = known([0, 10, 20, 30], [10, 10, 200, 10])
    picks = draws(NEAR_ONLY, ratings, players, count=50)
    assert all(p.a == 2 for p in picks)


def test_cost_shifts_the_focal_player_and_the_opponent():
    players = nets(3)
    ratings = known([0, 0, 0], [100, 100, 100])
    costs = np.array([[1.0, 0.1, 1.0], [0.1, 1.0, 1.0], [1.0, 1.0, 1.0]])
    picks = draws(NEAR_ONLY, ratings, players, costs)
    assert Counter(p.a for p in picks)[2] == 0
    assert Counter(p.b for p in picks)[1] > 3 * Counter(p.b for p in picks)[2]


def test_cross_run_opponents_are_preferred():
    players = [
        Player(id=0, name="a", sims=128, network="a", run="r1"),
        Player(id=1, name="b", sims=128, network="b", run="r1"),
        Player(id=2, name="c", sims=128, network="c", run="r2"),
    ]
    weights = opponent_weights(
        NEAR_ONLY, known([0, 0, 0]), players, np.full((3, 3), 0.2)
    )
    assert weights[0, 2] == pytest.approx(2 * weights[0, 1])


def test_a_network_at_another_budget_isnt_penalised_as_the_same_run():
    players = [
        Player(id=0, name="a@128", sims=128, network="a", run="r1"),
        Player(id=1, name="a@1000", sims=1000, network="a", run="r1"),
        Player(id=2, name="c", sims=128, network="c", run="r2"),
    ]
    weights = opponent_weights(
        NEAR_ONLY, known([0, 0, 0]), players, np.full((3, 3), 0.2)
    )
    assert weights[0, 1] == pytest.approx(weights[0, 2])


def test_some_batches_are_far_and_some_against_engines():
    players = [*nets(4), engine(4)]
    ratings = known([0, 50, 100, 900, 100])
    config = SchedulerConfig(far_fraction=0.2, external_fraction=0.1)
    modes = Counter(p.mode for p in draws(config, ratings, players, count=1000))
    assert modes["far"] == pytest.approx(200, abs=50)
    assert modes["external"] == pytest.approx(100, abs=40)
    assert modes["near"] == pytest.approx(700, abs=60)


def test_far_opponents_are_far():
    players = nets(4)
    ratings = known([0, 50, 100, 900])
    config = SchedulerConfig(far_fraction=1.0, external_fraction=0.0)
    for p in draws(config, ratings, players, count=50):
        assert p.mode == "far"
        assert abs(ratings.elo[p.a] - ratings.elo[p.b]) > 2 * config.width


def test_engines_only_play_networks():
    players = [*nets(2), engine(2), engine(3)]
    config = SchedulerConfig(external_fraction=0.5)
    for p in draws(config, known([0, 0, 0, 0]), players, count=100):
        assert p.a < 2 or p.b < 2


def test_retired_players_get_no_games():
    players = nets(3)
    players[1] = Player(id=1, name="p1", sims=128, network="n1", run="r1", retired=True)
    for p in draws(NEAR_ONLY, known([0, 0, 0]), players, count=50):
        assert 1 not in (p.a, p.b)


def test_initial_games_connect_everyone_at_random():
    players = nets(6)
    fit = fit_ratings(range(6), [])
    picks = draws(NEAR_ONLY, fit, players, count=200)
    assert {p.mode for p in picks} == {"connect"}
    assert {p.a for p in picks} == set(range(6))
    assert {p.b for p in picks} == set(range(6))
    assert {p.games for p in picks} == {min(NEAR_ONLY.batch_sizes)}


def test_a_separate_part_gets_joined_first():
    players = nets(5)
    fit = fit_ratings(
        range(5),
        [
            PairRecord(0, 1, 5, 0, 5),
            PairRecord(1, 2, 5, 0, 5),
            PairRecord(3, 4, 5, 0, 5),
        ],
    )
    for p in draws(NEAR_ONLY, fit, players, count=30):
        assert p.mode == "connect"
        assert {p.a < 3, p.b < 3} == {True, False}


def test_a_retired_player_with_no_gain_isnt_focal():
    players = nets(3)
    players[0] = Player(id=0, name="p0", sims=128, network="n0", run="r0", retired=True)
    # no uncertainty anywhere: every batch is worth nothing, and someone still plays
    for p in draws(NEAR_ONLY, known([0, 0, 0], 0.0), players, count=20):
        assert {p.a, p.b} == {1, 2}


def test_nobody_to_play():
    assert (
        next_pairing(
            NEAR_ONLY,
            fit_ratings([0], []),
            nets(1),
            np.ones((1, 1)),
            np.random.default_rng(0),
        )
        is None
    )


def test_batch_size_fits_the_time_and_memory():
    config = SchedulerConfig(
        batch_seconds=180, batch_sizes=(64, 128, 256), max_tree_nodes=300_000
    )
    low, high = nets(1, sims=128)[0], nets(1, sims=1000)[0]
    assert batch_size(config, low, low, 0.1) == 256
    assert batch_size(config, low, low, 1.0) == 128
    assert batch_size(config, low, low, 100.0) == 64
    # 128 games at a time, of 2 * (1000 + 1000) nodes each, is too many
    assert batch_size(config, high, high, 0.1) == 128


def test_openings_arent_repeated_until_theyre_all_played():
    rng = np.random.default_rng(0)
    chosen = choose_openings(10, {0, 1, 2}, 4, rng)
    assert len(set(chosen.tolist())) == 4
    assert not set(chosen.tolist()) & {0, 1, 2}
    last = choose_openings(10, set(range(8)), 5, rng)
    assert {8, 9} <= set(last.tolist())
    assert len(last) == 5


def test_network_size():
    assert network_size("resnet:8x128,pool_every=2") == pytest.approx(1)
    assert network_size("resnet:4x64") == pytest.approx(1 / 8)
    assert network_size("transformer") == 1
    assert network_size(None) == 1


def test_costs_are_corrected_by_measured_batches():
    players = nets(3, sims=1000)
    before = seconds_per_game(players, {}, [])
    assert before[0, 1] == pytest.approx(2 * 0.266 + 0.044, rel=0.01)
    # players 0 and 1 measure three times slower; player 2 stays as modelled
    after = seconds_per_game(players, {}, [(0, 1, 100, 300 * before[0, 1])])
    assert after[0, 1] > 2.5 * before[0, 1]
    assert after[2, 2] == pytest.approx(before[2, 2])


def test_external_costs_come_from_the_game():
    players = [nets(1)[0], engine(1)]
    costs = seconds_per_game(players, {1: 2.0}, [])
    assert costs[0, 1] > 2.0
