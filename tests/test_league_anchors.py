"""Games against fixed anchors as a network trains, on tic-tac-toe (see test_league_match.py)."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_league_match import TicTacToe

from core.league.anchors import Anchor, AnchorTester
from core.league.ratings import rate_against
from core.testing import tester as testers


def checkpoint(tmp_path, seed):
    """A fake checkpoint: TicTacToe's networks are logits drawn from a seed."""
    path = tmp_path / f"{seed}.ckpt"
    path.write_text(str(seed))
    return str(path)


def params_of(seed):
    """The parameters `checkpoint(tmp_path, seed)` loads."""
    logits = jax.random.normal(jax.random.PRNGKey(seed), (9,))
    return {"logits": logits, "value": jnp.array(0.0)}


@pytest.fixture
def anchors(tmp_path):
    return [
        Anchor("weak", checkpoint(tmp_path, 1), "tiny", -100.0),
        Anchor("strong", checkpoint(tmp_path, 2), "tiny", 250.0),
    ]


def make_tester(anchors, **kwargs):
    return AnchorTester(
        TicTacToe(), None, anchors, **{"sims": 4, "games": 8, "every": 3, **kwargs}
    )


def run(tester, epoch, params, logged):
    return tester.run(
        key=jax.random.PRNGKey(epoch),
        epoch_num=epoch,
        max_steps=9,
        env_step_fn=None,
        env_init_fn=None,
        evaluator=None,
        state=testers.TestState(),
        params=params,
        log_fn=logged.append,
    )


def test_it_plays_after_every_n_epochs_and_logs_its_ratings(anchors, tmp_path):
    results = tmp_path / "anchors.jsonl"
    tester = make_tester(anchors, results=results)
    logged = []

    for epoch in range(6):
        _, metrics, episode = run(tester, epoch, params_of(3), logged)
        assert metrics == {} and episode is None

    # after epochs 2 and 5
    assert len(logged) == 2
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert [(r["epoch"], r["anchor"]) for r in records] == [
        (2, "weak"),
        (2, "strong"),
        (5, "weak"),
        (5, "strong"),
    ]
    for r in records:
        assert r["wins"] + r["draws"] + r["losses"] == 8
    metrics = logged[-1]
    assert metrics.keys() == {
        "anchor_weak_score",
        "anchor_weak_elo",
        "anchor_weak_elo_se",
        "anchor_strong_score",
        "anchor_strong_elo",
        "anchor_strong_elo_se",
        "anchor_elo",
        "anchor_elo_se",
        "anchor_seconds",
    }
    last = records[2:]
    points = [r["wins"] + r["draws"] / 2 for r in last]
    assert metrics["anchor_weak_score"] == points[0] / 8
    assert metrics["anchor_weak_elo"] == pytest.approx(
        rate_against([-100.0], [points[0]], [8]).elo
    )
    assert metrics["anchor_elo"] == pytest.approx(
        rate_against([-100.0, 250.0], points, [8, 8]).elo
    )


def test_every_evaluation_plays_the_same_games(anchors):
    tester = make_tester(anchors)
    logged = []

    run(tester, 2, params_of(3), logged)
    run(tester, 5, params_of(3), logged)

    # deterministic searches from the same openings: only the network's moves could differ
    assert {k: v for k, v in logged[0].items() if k != "anchor_seconds"} == {
        k: v for k, v in logged[1].items() if k != "anchor_seconds"
    }


def test_a_network_playing_itself_scores_half_from_each_opening_with_both_colours(
    anchors,
):
    tester = make_tester(anchors[1:])

    results = tester.play(anchors[1], params_of(2))

    # 4 openings with it first, then the same 4 with the anchor first: mirrored games
    assert results.shape == (8,)
    np.testing.assert_array_equal(results[:4], -results[4:])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"games": 7}, "even"),
        ({"every": 0}, "every must be positive"),
    ],
)
def test_bad_settings_are_refused(anchors, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_tester(anchors, **kwargs)


def test_anchors_are_needed_and_named_apart(anchors):
    with pytest.raises(ValueError, match="no anchors"):
        make_tester([])
    with pytest.raises(ValueError, match="distinct"):
        make_tester([anchors[0], anchors[0]])


def test_a_missing_anchor_fails_before_training(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_tester([Anchor("gone", str(tmp_path / "gone.ckpt"), "tiny", 0.0)])


def test_the_trained_network_plays_with_its_own_parameters(anchors):
    """Its program takes whatever parameters it's given, as training's change every epoch."""
    tester = make_tester(anchors[:1])
    zero = {"logits": jnp.zeros(9), "value": jnp.array(0.0)}
    a = tester.play(anchors[0], zero)
    b = tester.play(anchors[0], params_of(1))
    assert a.shape == b.shape == (8,)
