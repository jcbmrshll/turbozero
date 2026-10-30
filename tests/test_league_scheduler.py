"""How the league picks its next batch."""

from collections import Counter
from typing import Any

import numpy as np
import pytest

from core.league.ratings import Ratings, fit_ratings
from core.league.scheduler import (
    SchedulerConfig,
    batch_size,
    choose_openings,
    network_size,
    next_pairing,
    opponent_weights,
    seconds_per_game,
)
from core.league.store import Player


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


NEAR_ONLY = SchedulerConfig(uniform_fraction=0.0)


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


def test_a_player_without_games_comes_before_any_uncertainty():
    players = nets(3)
    ratings = known([0, 0, 0], [500, 5, 500])
    ratings.games[1] = 0
    assert all(p.a == 1 for p in draws(NEAR_ONLY, ratings, players, count=20))


def test_a_retired_player_isnt_focal():
    players = nets(3)
    players[0] = Player(id=0, name="p0", sims=128, network="n0", run="r0", retired=True)
    ratings = known([0, 0, 0], [500, 10, 20])
    assert all(
        p.a == 2 and p.b == 1 for p in draws(NEAR_ONLY, ratings, players, count=20)
    )


def test_cost_shifts_the_opponent():
    players = nets(3)
    ratings = known([0, 0, 0], [200, 100, 100])
    costs = np.array([[1.0, 0.1, 1.0], [0.1, 1.0, 1.0], [1.0, 1.0, 1.0]])
    picks = draws(NEAR_ONLY, ratings, players, costs)
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


def test_the_uniform_part_draws_far_opponents():
    players = nets(4)
    ratings = known([0, 50, 100, 900], 5.0)
    costs = np.full((4, 4), 0.2)
    config = SchedulerConfig(uniform_fraction=0.3)
    weights = opponent_weights(config, ratings, players, costs)
    assert weights.sum(axis=1) == pytest.approx(np.ones(4))
    # next to nothing from the normal: its share of the uniform part
    assert weights[0, 3] == pytest.approx(0.1, abs=0.001)
    assert opponent_weights(NEAR_ONLY, ratings, players, costs)[0, 3] < 1e-6


def test_an_uncertain_engine_is_focal_like_anyone_else():
    players = [*nets(3), engine(3)]
    ratings = known([0, 0, 0, 0], [10, 10, 10, 200])
    assert all(p.a == 3 for p in draws(NEAR_ONLY, ratings, players, count=20))


def test_engines_can_be_left_out():
    players = [*nets(3), engine(3)]
    ratings = known([0, 0, 0, 0], [10, 10, 10, 200])
    config = SchedulerConfig(engines=False)
    for p in draws(config, ratings, players, count=100):
        assert 3 not in (p.a, p.b)


def test_engines_play_each_other():
    players = [*nets(2), engine(2), engine(3)]
    picks = draws(SchedulerConfig(), known([0, 0, 0, 0], [10, 10, 200, 10]), players)
    assert {2, 3} in [{p.a, p.b} for p in picks]


def test_retired_players_get_no_games():
    players = nets(3)
    players[1] = Player(id=1, name="p1", sims=128, network="n1", run="r1", retired=True)
    for p in draws(NEAR_ONLY, known([0, 0, 0]), players, count=50):
        assert 1 not in (p.a, p.b)


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
    assert before[0, 1] == pytest.approx(0.405, rel=0.01)
    # players 0 and 1 measure three times slower; player 2 stays as modelled
    after = seconds_per_game(players, {}, [(0, 1, 100, 300 * before[0, 1])])
    assert after[0, 1] > 2.5 * before[0, 1]
    assert after[2, 2] == pytest.approx(before[2, 2])


def test_external_costs_come_from_the_game():
    players = [nets(1)[0], engine(1)]
    costs = seconds_per_game(players, {1: 2.0}, [])
    assert costs[0, 1] > 2.0
