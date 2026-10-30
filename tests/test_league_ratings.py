"""The league's fit of ratings to games."""

import numpy as np
import pytest

from core.league.ratings import connected_components, expected_score, fit_ratings
from core.league.store import PairRecord


def even_games(a, b, games):
    return PairRecord(a, b, games // 2, 0, games // 2)


def simulated(elo, games, draws=0.07, seed=0):
    """Every pair's games among players of true ratings `elo`."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(len(elo)):
        for j in range(i + 1, len(elo)):
            p = float(expected_score(elo[i] - elo[j]))
            wins, drawn, losses = rng.multinomial(
                games, [p - draws / 2, draws, 1 - p - draws / 2]
            )
            records.append(PairRecord(i, j, int(wins), int(drawn), int(losses)))
    return records


def test_recovers_known_ratings():
    elo = np.array([0.0, 100.0, -80.0, 250.0])
    fit = fit_ratings(range(4), simulated(elo, 4000))
    assert np.abs((fit.elo - fit.elo.mean()) - (elo - elo.mean())).max() < 20


def test_many_even_games_pin_a_difference_to_about_fifteen_elo():
    fit = fit_ratings([0, 1], [even_games(0, 1, 512)])
    assert fit.elo[0] == pytest.approx(fit.elo[1], abs=1e-6)
    assert fit.pair_sd(0, 1) == pytest.approx(15.4, abs=0.5)


def test_a_draw_counts_as_half_a_win():
    drawn = fit_ratings([0, 1], [PairRecord(0, 1, 30, 40, 10)])
    decided = fit_ratings([0, 1], [PairRecord(0, 1, 50, 0, 30)])
    assert drawn.elo == pytest.approx(decided.elo, abs=1e-6)
    assert drawn.elo[0] > drawn.elo[1]


def test_a_perfect_score_stays_finite():
    fit = fit_ratings([0, 1], [PairRecord(0, 1, 512, 0, 0)])
    difference = fit.elo[0] - fit.elo[1]
    assert 400 < difference < 3000
    assert np.isfinite(fit.sd).all()


def test_the_order_of_the_games_doesnt_matter():
    records = simulated(np.array([0.0, 50.0, 90.0]), 100)
    forward = fit_ratings(range(3), records)
    backward = fit_ratings(range(3), records[::-1])
    assert forward.elo == pytest.approx(backward.elo)


def test_a_chain_is_less_certain_than_a_direct_game():
    chain = fit_ratings(range(3), [even_games(0, 1, 512), even_games(1, 2, 512)])
    assert chain.pair_sd(0, 2) > chain.pair_sd(0, 1) * 1.3


def test_uncertainty_shrinks_with_games():
    few = fit_ratings([0, 1], [even_games(0, 1, 64)])
    many = fit_ratings([0, 1], [even_games(0, 1, 640)])
    assert many.pair_sd(0, 1) < few.pair_sd(0, 1) / 2.5


def test_finds_the_parts_of_the_game_graph():
    fit = fit_ratings(range(5), [even_games(0, 1, 10), even_games(2, 3, 10)])
    assert not fit.connected
    assert fit.components.tolist() == [0, 0, 1, 1, 2]
    assert fit_ratings(range(2), [even_games(0, 1, 10)]).connected
    assert connected_components(3, [(0, 2)]).tolist() == [0, 1, 0]


def test_a_new_player_with_no_games_is_very_uncertain():
    fit = fit_ratings(range(3), [even_games(0, 1, 512)])
    assert fit.sd[2] > 10 * fit.sd[0] or fit.sd[2] > 500


def test_the_anchor_is_at_zero_with_uncertainty_relative_to_the_mean():
    records = [PairRecord(0, 1, 90, 0, 10), PairRecord(1, 2, 3, 0, 1)]
    plain = fit_ratings(range(3), records)
    anchored = fit_ratings(range(3), records, anchor=1)
    assert anchored.elo[1] == pytest.approx(0)
    assert anchored.elo[0] - anchored.elo[2] == pytest.approx(
        plain.elo[0] - plain.elo[2]
    )
    assert anchored.sd == pytest.approx(plain.sd)


def test_an_anchor_outside_the_main_part_doesnt_set_the_zero():
    fit = fit_ratings(range(3), [PairRecord(0, 1, 90, 0, 10)], anchor=2)
    assert fit.elo.mean() == pytest.approx(0)


def test_games_of_unknown_players_are_left_out():
    fit = fit_ratings([0, 1], [even_games(0, 1, 10), even_games(0, 9, 10)])
    assert fit.games.tolist() == [10, 10]


def test_no_players_and_no_games():
    assert fit_ratings([], []).elo.size == 0
    fit = fit_ratings([3, 4], [])
    assert fit.elo.tolist() == [0.0, 0.0]
    assert not fit.connected
