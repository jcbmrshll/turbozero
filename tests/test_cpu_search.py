"""cpu_search.py's rules and search, against pgx's Othello and core's MCTS, with fake networks."""

import shutil
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")


def fake_eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A network of fixed numbers, different for each position: logits by a hash of the board,
    the value by the lead in discs."""
    weights = np.sin(np.arange(128) * 1.3).astype(np.float32)
    flat = x.reshape(len(x), 128)
    mix = (flat * weights).sum(1, keepdims=True)
    logits = np.sin(mix * np.arange(1, 66, dtype=np.float32)).astype(np.float32)
    values = (flat[:, :64].sum(1) - flat[:, 64:].sum(1)) / 64
    return logits, values.astype(np.float32)


def pgx_me_opp(state) -> tuple[int, int]:
    obs = np.asarray(state.observation)
    bits = 1 << np.arange(64, dtype=np.uint64)
    return (
        int(bits[obs[..., 0].reshape(-1)].sum()),
        int(bits[obs[..., 1].reshape(-1)].sum()),
    )


def test_rules_match_pgx_over_random_games():
    from cpu_search import Search
    from game import env

    step = jax.jit(env.step)
    rng = np.random.default_rng(0)
    for seed in range(6):
        state = env.init(jax.random.PRNGKey(seed))
        search = Search()
        batch = np.empty((1, 2, 8, 8), np.float32)
        while True:
            me, opp, _, terminated = search.position()
            assert (me, opp) == pgx_me_opp(state)
            assert terminated == bool(state.terminated)
            if terminated:
                mover = int(state.current_player)
                assert search.reward() == float(state.rewards[mover])
                break
            # the observation a search evaluates, of the root: its first leaf
            search.begin(1)
            assert search.collect(batch) == 1
            assert np.array_equal(
                batch[0].astype(bool),
                np.moveaxis(np.asarray(state.observation), -1, 0),
            )
            logits, values = fake_eval(batch)
            search.apply(logits, values)
            legal = search.legal()
            assert np.array_equal(legal, np.asarray(state.legal_action_mask))
            action = int(rng.choice(np.flatnonzero(legal)))
            search.play(action)
            state = step(state, jnp.int32(action))


def core_visits(sims: int, moves: list[int]) -> np.ndarray:
    """The root visits of `core`'s MCTS, after `moves`, with `fake_eval` as its network."""
    from game import env, make_test_evaluator, metadata, step_fn

    from core.evaluators.mcts.policy_targets import VisitCounts

    def eval_callback(state, params, key):
        return jax.pure_callback(
            lambda o: tuple(
                x[0]
                for x in fake_eval(
                    np.moveaxis(np.asarray(o), -1, 0)[None].astype(np.float32)
                )
            ),
            (jnp.zeros(65, jnp.float32), jnp.zeros((), jnp.float32)),
            state.observation,
        )

    evaluator = make_test_evaluator(eval_callback, sims, noise=False)
    key = jax.random.PRNGKey(0)
    state = env.init(key)
    for action in moves:
        state = env.step(state, jnp.int32(action))
    tree = evaluator.init(template_embedding=state)
    out = evaluator.evaluate(
        key=key,
        eval_state=tree,
        env_state=state,
        root_metadata=metadata(state),
        params=None,
        env_step_fn=step_fn,
    )
    visits = np.asarray(VisitCounts()(out.eval_state, evaluator.discount))
    return np.round(visits * sims).astype(int)


def test_a_batch_of_one_visits_as_core_does():
    from cpu_search import Search

    moves = [19, 18, 26]
    search = Search()
    for action in moves:
        search.play(action)
    sims = 48
    # a visit for the root's own evaluation, which core counts apart
    search.run(sims + 1, fake_eval)
    assert np.array_equal(search.visits(), core_visits(sims, moves))


@pytest.mark.parametrize("mode", ["loss", "draw", "visit"])
def test_batches_keep_the_tree_consistent(mode):
    from cpu_search import Search, SearchSettings

    search = Search(SearchSettings(batch=8, virtual_loss=mode))
    search.run(200, fake_eval)
    assert search.visits().sum() == 199
    assert search.total_visits() == 200
    # nothing's left in flight: a batch of one now sees only real visits
    action = search.best_action()
    assert search.legal()[action]
    search.play(action)
    search.run(40, fake_eval)
    assert search.visits().sum() >= 39


def test_a_batch_steers_apart():
    from cpu_search import Search, SearchSettings

    # the root, then 8 traversals that can't all go down one path
    search = Search(SearchSettings(batch=8))
    batch = np.empty((8, 2, 8, 8), np.float32)
    search.begin(9)
    assert search.collect(batch) == 1
    search.apply(*fake_eval(batch[:1]))
    count = search.collect(batch)
    assert count > 1
    assert len({row.tobytes() for row in batch[:count]}) == count


def test_a_short_batch_still_finishes():
    from cpu_search import Search, SearchSettings

    search = Search(SearchSettings(batch=64, max_collisions=0))
    search.run(300, fake_eval)
    assert search.total_visits() == 300
    assert search.collisions() > 0


def test_export_matches_the_network():
    import equinox as eqx
    from cpu_search import openvino_eval, ort_eval, ort_session
    from export_onnx import to_onnx
    from game import make_network

    from core.networks.utils import apply_nn

    network, state = make_network(
        "resnet:2x8,pool_every=2,heads=pooled,pool_channels=4,head_channels=4"
    )
    model = eqx.nn.inference_mode(network)
    x = np.random.default_rng(0).random((5, 8, 8, 2)) < 0.3
    (logits, values), _ = apply_nn(model, state, jnp.asarray(x, jnp.float32))
    graph = to_onnx(model, state).SerializeToString()
    channels_first = np.moveaxis(x, -1, 1).astype(np.float32)
    for evaluate in (
        ort_eval(ort_session(graph, 1)),
        openvino_eval(graph, 1, "f32"),
        openvino_eval(graph, 2, "f32"),
    ):
        got_logits, got_values = evaluate(channels_first)
        np.testing.assert_allclose(got_logits, np.asarray(logits), atol=1e-5)
        np.testing.assert_allclose(got_values, np.asarray(values)[:, 0], atol=1e-5)


def test_move_budget_shares_the_clock():
    from cpu_match import move_budget

    assert move_budget(10, 40) == pytest.approx(0.95 * 10 / 20)
    assert move_budget(10, 1) == pytest.approx(9.5)


def test_net_player_plays_a_whole_game_on_a_clock():
    from cpu_match import NetPlayer
    from cpu_search import SearchSettings

    players = (
        NetPlayer(SearchSettings(batch=4), fake_eval),
        NetPlayer(SearchSettings(batch=1), fake_eval),
    )
    turn = 0
    while not players[0].search.terminated():
        action = players[turn].move("b", 0.05)
        players[1 - turn].play("w", action)
        turn = 1 - turn
    assert players[0].search.position() == players[1].search.position()
