"""cpu_search.py's rules and search, against pgx's Othello and core's MCTS, with fake networks."""

import shutil
import sys
import tempfile
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


def test_reused_evaluations_search_as_core_does():
    from cpu_search import Search, SearchSettings

    # deep enough that the searches below reach positions by more than one order of moves
    moves = [37, 45, 53, 38, 46, 54, 55, 20, 21, 47]
    sims = 400
    searches = {}
    for reuse in (True, False):
        search = Search(SearchSettings(reuse=reuse))
        for action in moves:
            search.play(action)
        search.run(sims + 1, fake_eval)
        searches[reuse] = search
    assert searches[True].reused() > 0
    assert searches[False].reused() == 0
    assert np.array_equal(searches[True].visits(), core_visits(sims, moves))
    assert np.array_equal(searches[False].visits(), searches[True].visits())


def test_a_game_searched_a_leaf_at_a_time_evaluates_no_position_twice():
    from cpu_search import Search

    seen = set()

    def recording(x):
        for row in x:
            key = row.tobytes()
            assert key not in seen
            seen.add(key)
        return fake_eval(x)

    search = Search()
    while not search.terminated() and search.total_visits() < 6000:
        search.run(300, recording)
        search.play(search.best_action())
    assert search.reused() > 0
    assert len(seen) + search.reused() <= search.total_visits()


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


def symmetry_tables() -> dict[int, np.ndarray]:
    """For each of the search's symmetries, (65) where each action goes, worked out by hand from
    what it does: 1 swaps rows and columns, 2 reverses the rows, 4 the columns."""
    tables = {}
    for symmetry in range(8):
        table = np.arange(65)
        for action in range(64):
            row, column = action // 8, action % 8
            if symmetry & 1:
                row, column = column, row
            if symmetry & 2:
                row = 7 - row
            if symmetry & 4:
                column = 7 - column
            table[action] = row * 8 + column
        tables[symmetry] = table
    return tables


def test_each_symmetry_maps_squares_and_undoes_with_its_inverse():
    from cpu_search import LIB, PASS

    tables = symmetry_tables()
    rng = np.random.default_rng(0)
    boards = [int(x) for x in rng.integers(0, 2**63, 50, dtype=np.uint64)]
    for symmetry, table in tables.items():
        for action in range(65):
            assert LIB.search_symmetric_action(action, symmetry) == table[action]
        assert LIB.search_symmetric_action(PASS, symmetry) == PASS
        # discs go where their squares do
        for action in range(64):
            assert LIB.search_symmetric_bits(1 << action, symmetry) == (
                1 << int(table[action])
            )
        # some symmetry undoes it, on whole boards too
        inverses = [
            other
            for other, back in tables.items()
            if np.array_equal(back[table], np.arange(65))
        ]
        assert len(inverses) == 1
        for board in boards:
            shown = LIB.search_symmetric_bits(board, symmetry)
            assert LIB.search_symmetric_bits(shown, inverses[0]) == board


def test_the_symmetries_are_those_training_uses():
    from game import make_symmetry_transform_fn

    from core.memory.replay_memory import BaseExperience

    actions = jnp.arange(65)
    experience = BaseExperience(
        reward=jnp.zeros(2),
        policy_weights=actions,
        policy_mask=actions,
        observation_nn=jnp.zeros((8, 8, 2)),
        cur_player_id=jnp.zeros(1),
        search_value=jnp.zeros(1),
    )
    training = set()
    for transpose in (False, True):
        for quarter_turns in range(4):
            transform = make_symmetry_transform_fn(quarter_turns, transpose)
            training.add(np.asarray(transform(experience).policy_weights).tobytes())
    assert training == {
        table.astype(np.asarray(actions).dtype).tobytes()
        for table in symmetry_tables().values()
    }


def board_planes(me: int, opp: int) -> np.ndarray:
    bits = 1 << np.arange(64, dtype=np.uint64)
    planes = [(int(x) & bits) > 0 for x in (me, opp)]
    return np.stack(planes).reshape(2, 8, 8).astype(np.float32)


def shown_symmetry(view: np.ndarray, me: int, opp: int) -> int:
    """Which symmetry an observation is of a position with no symmetry of its own."""
    from cpu_search import LIB

    found = [
        symmetry
        for symmetry in range(8)
        if np.array_equal(
            view,
            board_planes(
                LIB.search_symmetric_bits(me, symmetry),
                LIB.search_symmetric_bits(opp, symmetry),
            ),
        )
    ]
    assert len(found) == 1
    return found[0]


# a position with no symmetry of its own and several legal moves
UNSYMMETRIC = [19, 18, 26, 34]


def unsymmetric_search(seed: int, batch: int = 1):
    from cpu_search import Search, SearchSettings

    search = Search(SearchSettings(batch=batch, random_symmetry=True), seed)
    for action in UNSYMMETRIC:
        search.play(action)
    return search


def test_the_policy_reaches_the_edges_of_the_real_squares():
    tables = symmetry_tables()
    probe = unsymmetric_search(0)
    me, opp, _, _ = probe.position()
    targets = np.flatnonzero(probe.legal())
    assert len(targets) > 1
    seen = set()

    def favouring(target: int):
        def favour_target(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # a network that sees through the symmetry: a high logit where the target is shown
            symmetry = shown_symmetry(x[0], me, opp)
            seen.add(symmetry)
            logits = np.zeros((1, 65), np.float32)
            logits[0, tables[symmetry][target]] = 10
            return logits, np.zeros(1, np.float32)

        return favour_target

    for seed, target in enumerate(np.tile(targets, 16)):
        search = unsymmetric_search(seed)
        # the root's evaluation, then one traversal, down the edge of highest prior
        search.run(1, favouring(int(target)))
        search.run(1, fake_eval)
        visits = search.visits()
        assert visits[target] == 1 and visits.sum() == 1
    assert seen == set(range(8))


def test_a_position_with_only_a_pass_searches_under_any_symmetry():
    from cpu_search import PASS, Search, SearchSettings

    # a random game up to a position with only a pass to play
    rng = np.random.default_rng(1)
    moves: list[int] = []
    for _ in range(200):
        replay = Search()
        moves = []
        while not replay.terminated() and not replay.legal()[PASS]:
            action = int(rng.choice(np.flatnonzero(replay.legal())))
            moves.append(action)
            replay.play(action)
        if not replay.terminated():
            break
    for seed in range(16):
        search = Search(SearchSettings(random_symmetry=True), seed)
        for action in moves:
            search.play(action)
        assert search.legal()[PASS]
        search.run(2, fake_eval)
        assert search.visits()[PASS] == 1


def equivariant_eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A network no symmetry changes: logits depend on each square's distance from the board's
    centre and the discs around it, the value on the lead in discs."""
    me, opp = x[:, 0], x[:, 1]
    padded = np.pad(me - 0.7 * opp, ((0, 0), (1, 1), (1, 1)))
    around = sum(
        padded[:, 1 + dr : (9 + dr), 1 + dc : (9 + dc)]
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    )
    centre = np.abs(np.arange(8) - 3.5)
    distance = centre[None, :] ** 2 + centre[:, None] ** 2
    logits = 0.7 * around + 0.13 * distance[None] * (1 + me.sum((1, 2), keepdims=True))
    logits = np.concatenate([logits.reshape(len(x), 64), np.zeros((len(x), 1))], 1)
    values = np.tanh((me.sum((1, 2)) - opp.sum((1, 2))) * 0.1 + 0.02)
    return logits.astype(np.float32), values.astype(np.float32)


@pytest.mark.parametrize("batch", [1, 8])
def test_a_network_no_symmetry_changes_searches_the_same_tree(batch):
    from cpu_search import Search, SearchSettings

    plain = Search(SearchSettings(batch=batch))
    shown = Search(SearchSettings(batch=batch, random_symmetry=True), seed=3)
    for search in (plain, shown):
        for action in UNSYMMETRIC:
            search.play(action)
        search.run(400, equivariant_eval)
    assert np.array_equal(plain.visits(), shown.visits())
    assert plain.root_q() == pytest.approx(shown.root_q(), abs=1e-5)


def test_a_seed_replays_a_search_and_another_changes_it():
    def visits(seed: int) -> np.ndarray:
        search = unsymmetric_search(seed, batch=4)
        search.run(300, fake_eval)
        return search.visits()

    assert np.array_equal(visits(5), visits(5))
    assert any(not np.array_equal(visits(5), visits(seed)) for seed in (6, 7, 8))


def test_symmetries_are_drawn_evenly():
    from collections import Counter

    probe = unsymmetric_search(0)
    me, opp, _, _ = probe.position()
    counts: Counter[int] = Counter()
    batch = np.empty((1, 2, 8, 8), np.float32)
    for seed in range(800):
        search = unsymmetric_search(seed)
        search.begin(1)
        assert search.collect(batch) == 1
        counts[shown_symmetry(batch[0], me, opp)] += 1
    assert set(counts) == set(range(8))
    assert all(60 <= count <= 140 for count in counts.values())


def test_a_player_parses_random_symmetry():
    from cpu_match import Player

    def settings(spec: str):
        return Player.parse(spec, "net", "bf16").settings

    assert not settings("visits=10").random_symmetry
    assert settings("visits=10,random_symmetry=1").random_symmetry


@pytest.mark.parametrize(
    "norm",
    [
        "batch",
        "layer",
        "group",
        "rms",
        "channel",
        "fixed",
        "ws",
        *(f"{norm},layout=pre" for norm in ("batch", "layer", "channel", "rms")),
    ],
)
def test_export_matches_the_network(norm):
    import equinox as eqx
    from cpu_search import openvino_eval, ort_eval, ort_session
    from export_onnx import to_onnx
    from game import make_network

    from core.networks.utils import apply_nn

    network, state = make_network(
        "resnet:2x8,pool_every=2,heads=pooled,pool_channels=4,head_channels=4,"
        f"norm={norm},norm_groups=2"
    )
    # random scales and offsets in the normalizations, and every weight nonzero (the
    # pre-activation branches start at zero), so that they're all checked
    rng = np.random.default_rng(1)
    params, static = eqx.partition(network, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda x: (
            x
            + (0.3 if x.ndim == 1 else 0.05) * rng.normal(size=x.shape).astype(x.dtype)
        ),
        params,
    )
    model = eqx.nn.inference_mode(eqx.combine(params, static))
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


def test_an_arm_names_a_quantized_model_and_a_precision():
    from cpu_match import model_and_precision

    path, precision = model_and_precision("net", "int8-a_b")
    assert isinstance(path, str) and path.endswith("net-a_b.xml")
    assert precision == "f32"
    assert model_and_precision("net", "int8-a_b-bf16")[1] == "bf16"


@pytest.mark.parametrize("norm", ["batch", "layer"])
def test_float_heads_stay_unquantized_and_follow_the_network(norm):
    import equinox as eqx
    import openvino as ov
    from cpu_search import openvino_eval
    from export_onnx import to_onnx
    from game import make_network
    from quantize import BEST, quantize_graph

    network, state = make_network(
        "resnet:2x8,pool_every=2,heads=pooled,pool_channels=4,head_channels=4,"
        f"norm={norm}"
    )
    graph = to_onnx(eqx.nn.inference_mode(network), state).SerializeToString()
    x = (np.random.default_rng(0).random((64, 2, 8, 8)) < 0.3).astype(np.float32)
    quantized = quantize_graph(graph, BEST, x)

    def fake_quantizes(model) -> int:
        return sum(op.get_type_name() == "FakeQuantize" for op in model.get_ops())

    trunk_only = fake_quantizes(quantized)
    assert (
        0
        < trunk_only
        < fake_quantizes(quantize_graph(graph, "performance_leaves_minmax", x))
    )
    with tempfile.TemporaryDirectory() as d:
        ov.save_model(quantized, f"{d}/q.xml", compress_to_fp16=False)
        logits, values = openvino_eval(f"{d}/q.xml", 1, "f32")(x)
    reference_logits, reference_values = openvino_eval(graph, 1, "f32")(x)
    assert logits.shape == reference_logits.shape
    np.testing.assert_allclose(values, reference_values, atol=0.2)


def test_legal_squares_match_the_search():
    from cpu_search import Search
    from fidelity import legal_squares

    rng = np.random.default_rng(0)
    search = Search()
    batch = np.empty((1, 2, 8, 8), np.float32)
    while not search.terminated():
        search.begin(1)
        search.collect(batch)
        search.apply(*fake_eval(batch))
        legal = search.legal()
        if not legal[64]:
            assert np.array_equal(legal_squares(batch)[0], legal[:64])
        search.play(int(rng.choice(np.flatnonzero(legal))))


def test_move_budget_shares_the_clock():
    from cpu_match import move_budget

    assert move_budget(10, 40) == pytest.approx(0.95 * 10 / 20)
    assert move_budget(10, 1) == pytest.approx(9.5)
    # a taper moves time to the moves before it
    assert move_budget(10, 40, taper=24) > move_budget(10, 40)
    assert move_budget(10, 10, taper=24) == pytest.approx(
        0.95 * 10 * 100 / (100 + 64 + 36 + 16 + 4)
    )


def test_a_planned_solve_takes_its_moves_and_reserve_out_of_the_budget():
    from cpu_match import Solving, move_budget

    # 40 empties to a solve from 24: 8 moves share the clock less the reserve
    assert move_budget(10, 40, solve_from=24, reserve=2) == pytest.approx(0.95 * 8 / 8)
    # at or past the solve (it gave up): the moves to the end share all that's left
    assert move_budget(10, 24, solve_from=24, reserve=2) == move_budget(10, 24)
    solving = Solving(empties=24, threads=8, cost_at_24=0.2)
    assert solving.cost(22) == pytest.approx(0.05)
    assert Solving(empties=24, threads=4, cost_at_24=0.2).cost(24) == pytest.approx(0.4)
    # the most empties whose solve fits a quarter of the game's clock, no more than asked for
    assert solving.depth(10) == 24
    assert solving.depth(0.4) == 23
    assert solving.depth(1e-9) == 0


def test_net_players_play_a_game_through_the_clock_driver():
    from clock_match import play_game
    from cpu_match import NetPlayer
    from cpu_search import SearchSettings
    from xot import XOT_PATH, load_xot

    black = NetPlayer(SearchSettings(batch=4, random_symmetry=True), fake_eval, seed=1)
    white = NetPlayer(SearchSettings(batch=1), fake_eval)
    game = play_game(black, white, load_xot(XOT_PATH)[0], 1.0)
    assert game.flagged is None
    assert max(game.seconds) < 1.0


def test_a_net_player_passes_when_the_game_does():
    from cpu_match import NetPlayer
    from cpu_search import PASS, Search, SearchSettings

    # a random game up to its first pass that doesn't end it
    rng = np.random.default_rng(0)
    moves = []
    passed = False
    for _ in range(200):
        replay = Search()
        moves = []
        while not replay.terminated():
            action = int(rng.choice(np.flatnonzero(replay.legal())))
            moves.append(action)
            replay.play(action)
            if action == PASS:
                passed = not replay.terminated()
                break
        if passed:
            break
    assert passed
    player = NetPlayer(SearchSettings(), fake_eval)
    color = "b"
    for action in moves:
        if action != PASS:
            # the driver says nothing of a pass: the next mover's color shows it
            player.play(color, action)
        color = "w" if color == "b" else "b"
    # the side to move after the pass is the one that moved last
    assert player.to_move != color
    player.move(color, 0.01)
    me, opp, _, _ = player.search.position()
    assert (me | opp).bit_count() == 4 + sum(m != PASS for m in moves) + 1


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A small random network's ONNX graph, on disk, as an `Engine` reads it."""
    import equinox as eqx
    from export_onnx import to_onnx
    from game import make_network

    network, state = make_network(
        "resnet:2x8,pool_every=2,heads=pooled,pool_channels=4,head_channels=4"
    )
    params, static = eqx.partition(network, eqx.is_inexact_array)
    rng = np.random.default_rng(2)
    params = jax.tree.map(
        lambda x: x + 0.3 * rng.normal(size=x.shape).astype(x.dtype), params
    )
    model = eqx.nn.inference_mode(eqx.combine(params, static))
    path = tmp_path_factory.mktemp("engine") / "tiny.onnx"
    path.write_bytes(to_onnx(model, state).SerializeToString())
    return str(path)


def test_an_engine_of_one_worker_searches_as_a_batch_of_one(tiny_model):
    from cpu_search import Engine, Search, openvino_eval

    reference, native = Search(), Search()
    for search in (reference, native):
        for action in (19, 18, 26):
            search.play(action)
    reference.run(300, openvino_eval(tiny_model, 1, "f32"))
    Engine.openvino(tiny_model, 1, 1).search(native, visits=300)
    assert np.array_equal(native.visits(), reference.visits())
    assert native.total_visits() == reference.total_visits() == 300


@pytest.mark.parametrize("mode", ["loss", "visit"])
def test_workers_visit_exactly_and_leave_nothing_in_flight(tiny_model, mode):
    from cpu_search import Engine, Search, SearchSettings

    engine = Engine.openvino(tiny_model, 3, 4)
    search = Search(SearchSettings(batch=4, virtual_loss=mode))
    leaves = engine.search(search, visits=500)
    assert search.total_visits() == 500
    assert search.visits().sum() == 499
    assert 0 < leaves <= 500
    # the tree carries on as a sequential search would see it
    search.play(search.best_action())
    before = search.total_visits()
    engine.search(search, visits=100)
    assert search.total_visits() == before + 100
    stats = engine.stats()
    assert stats["calls"] > 0 and stats["leaves"] >= leaves


def test_a_timed_search_stops_at_its_time(tiny_model):
    import time

    from cpu_search import Engine, Search, SearchSettings

    engine = Engine.openvino(tiny_model, 2, 2)
    search = Search(SearchSettings(batch=2, virtual_loss="visit"))
    start = time.perf_counter()
    engine.search(search, seconds=0.05)
    assert 0.05 <= time.perf_counter() - start < 0.5
    assert search.total_visits() > 10


def test_a_solved_tree_stops_its_search_early(tiny_model):
    from cpu_search import Engine, Search, SearchSettings

    # a random game to a few empty squares: the whole tree below is soon all games' ends
    rng = np.random.default_rng(3)
    search = Search(SearchSettings(batch=2, virtual_loss="visit"))
    while True:
        me, opp, _, _ = search.position()
        if 64 - (me | opp).bit_count() <= 4 or search.terminated():
            break
        search.play(int(rng.choice(np.flatnonzero(search.legal()))))
    assert not search.terminated()
    Engine.openvino(tiny_model, 2, 2).search(search, seconds=5.0, stop_ended=64)
    # far fewer than 5 s of visits
    assert search.total_visits() < 100_000


def test_an_engine_player_plays_a_game_through_the_clock_driver(tiny_model):
    from clock_match import play_game
    from cpu_match import NetPlayer
    from cpu_search import Engine, SearchSettings
    from xot import XOT_PATH, load_xot

    engine = Engine.openvino(tiny_model, 2, 2)
    black = NetPlayer(SearchSettings(batch=2, virtual_loss="visit"), engine, seed=1)
    white = NetPlayer(SearchSettings(batch=1), fake_eval)
    game = play_game(black, white, load_xot(XOT_PATH)[0], 1.0)
    assert game.flagged is None
    assert max(game.seconds) < 1.1


def native_supported() -> bool:
    from cpu_search import LIB

    return bool(LIB.net_supported())


def native_levels() -> list[str]:
    """The levels with a native network that this CPU runs."""
    import native_target

    return [level for level in native_target.levels() if level not in ("v2", "arm")]


_LIBRARIES = {}


def library(level: str, monkeypatch):
    """The compiled search for `level` (`native_target.LEVELS`), whatever this CPU's own."""
    from cpu_search import load_library

    if level not in native_levels():
        pytest.skip(f"this CPU doesn't run level {level}")
    if level not in _LIBRARIES:
        monkeypatch.setenv("TURBOZERO_CPU_LEVEL", level)
        _LIBRARIES[level] = load_library()
    return _LIBRARIES[level]


def random_network(spec: str, seed: int = 1):
    """A network of `spec` in inference mode, every weight nonzero and normalizations not the
    identity, with its state."""
    import equinox as eqx
    from game import make_network

    network, state = make_network(spec)
    rng = np.random.default_rng(seed)
    params, static = eqx.partition(network, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda x: (
            x
            + (0.3 if x.ndim == 1 else 0.05) * rng.normal(size=x.shape).astype(x.dtype)
        ),
        params,
    )
    return eqx.nn.inference_mode(eqx.combine(params, static)), state


NATIVE_SPECS = [
    "resnet:2x32,pool_every=2,heads=pooled,pool_channels=16,head_channels=16",
    "resnet:3x64,pool_every=2,heads=pooled,pool_channels=16,norm=channel,layout=pre",
    # pooled channels leaving a number of others that is no multiple of 16
    "resnet:2x48,pool_every=1,heads=pooled,pool_channels=8,head_channels=16,layout=pre",
    "resnet:2x48,pool_every=2,heads=pooled,pool_channels=24,head_channels=16",
]


@pytest.mark.parametrize("spec", NATIVE_SPECS)
def test_the_native_networks_layers_compute_the_network(spec):
    import cpu_net
    from cpu_search import ort_eval, ort_session
    from export_onnx import to_onnx

    # against the exported graph, which `test_export_matches_the_network` checks against the network
    model, state = random_network(spec)
    x = (np.random.default_rng(0).random((16, 2, 8, 8)) < 0.3).astype(np.float32)
    logits, values = ort_eval(
        ort_session(to_onnx(model, state).SerializeToString(), 1)
    )(x)
    got_logits, got_values = cpu_net.float_forward(cpu_net.extract(model, state), x)
    np.testing.assert_allclose(got_logits, logits, atol=1e-5)
    np.testing.assert_allclose(got_values, values, atol=1e-5)


# each convolution kernel: AVX-512 VNNI's, AVX2's Winograd with and without AVX-VNNI, and NEON's with
# i8mm and with the dot product alone
@pytest.mark.parametrize("level", ["v4", "v3-vnni", "v3", "arm-i8mm", "arm-dotprod"])
@pytest.mark.parametrize("spec", NATIVE_SPECS)
def test_the_native_network_computes_its_quantized_layers(
    spec, level, tmp_path, monkeypatch
):
    import cpu_net

    lib = library(level, monkeypatch)
    program = cpu_net.extract(*random_network(spec))
    x = (np.random.default_rng(0).random((48, 2, 8, 8)) < 0.3).astype(np.float32)
    quantized = cpu_net.quantize(program, x)
    # saved and loaded as play loads it
    cpu_net.save(tmp_path / "net.npz", program, quantized)
    program, quantized = cpu_net.load(tmp_path / "net.npz")
    want_logits, want_values = cpu_net.int8_forward(program, quantized, x)
    network = cpu_net.NativeNet(program, quantized, lib)
    for threads in (1, 3):
        logits, values = network.forward(x, threads)
        # float sums in another order can round an activation to the next step, rarely
        np.testing.assert_allclose(logits, want_logits, atol=2e-2)
        np.testing.assert_allclose(values, want_values, atol=2e-2)
        assert np.abs(logits - want_logits).mean() < 1e-3
    # and quantized, it stays close to the float network
    float_logits, _ = cpu_net.float_forward(program, x)
    assert np.abs(logits - float_logits).mean() < 0.05


@pytest.mark.parametrize("spec", NATIVE_SPECS)
def test_avx_vnni_sums_as_avx2_does(spec, monkeypatch):
    import cpu_net

    # the same integers, multiplied and added in one instruction or two, and the same float code
    libs = [library(level, monkeypatch) for level in ("v3-vnni", "v3")]
    program = cpu_net.extract(*random_network(spec))
    x = (np.random.default_rng(0).random((48, 2, 8, 8)) < 0.3).astype(np.float32)
    quantized = cpu_net.quantize(program, x)
    (logits, values), (want_logits, want_values) = [
        cpu_net.NativeNet(program, quantized, lib).forward(x) for lib in libs
    ]
    assert np.array_equal(logits, want_logits) and np.array_equal(values, want_values)


def test_an_engine_of_the_native_network_searches_as_a_batch_of_one():
    import cpu_net
    from cpu_search import Engine, Search

    if not native_supported():
        pytest.skip("needs AVX2 or NEON's dot product")
    program = cpu_net.extract(*random_network(NATIVE_SPECS[1]))
    x = (np.random.default_rng(0).random((48, 2, 8, 8)) < 0.3).astype(np.float32)
    network = cpu_net.NativeNet(program, cpu_net.quantize(program, x))
    reference, native = Search(), Search()
    for search in (reference, native):
        for action in (19, 18, 26):
            search.play(action)
    reference.run(300, network.forward)
    Engine.native(network, 1, 1).search(native, visits=300)
    assert np.array_equal(native.visits(), reference.visits())
    # and with workers in parallel, every visit is made
    parallel = Search()
    Engine.native(network, 3, 2).search(parallel, visits=400)
    assert parallel.total_visits() == 400


def test_the_search_is_built_for_a_standard_level(monkeypatch):
    import native_target

    monkeypatch.delenv("TURBOZERO_CPU_LEVEL", raising=False)
    avx512 = {"avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512cd", "avx512_vnni"}
    for architecture, flags, level in [
        ("x86", avx512 | {"avx2", "bmi2", "fma", "movbe", "avx_vnni"}, "v4"),
        ("x86", {"avx2", "bmi2", "fma", "movbe", "avx_vnni"}, "v3-vnni"),
        ("x86", {"avx2", "bmi2", "fma", "movbe", "avx512f"}, "v3"),
        ("x86", set(), "v2"),
        ("arm", {"fp", "asimd", "asimddp", "i8mm"}, "arm-i8mm"),
        ("arm", {"fp", "asimd", "asimddp"}, "arm-dotprod"),
        ("arm", {"fp", "asimd"}, "arm"),
    ]:
        monkeypatch.setattr(native_target, "architecture", lambda a=architecture: a)
        monkeypatch.setattr(native_target, "cpu_flags", lambda flags=flags: flags)
        target = native_target.target_flags()
        assert target == native_target.LEVELS[level]
        assert target[0].startswith("-march=")
        assert not any("native" in flag for flag in target)


def test_a_net_player_takes_a_board_and_plays_from_it():
    from clock_match import play_game
    from cpu_match import NetPlayer
    from cpu_search import SearchSettings
    from ggs import Start

    start = Start.parse(
        "8 -------- -------- --*O-*-- ---***-- --**OO-- --O**O-- -------- -------- O"
    ).playable()
    for color in ("b", "w"):
        player = NetPlayer(SearchSettings(batch=2), fake_eval)
        player.set_position(start.black, start.white, color)
        me, opp = (
            (start.black, start.white) if color == "b" else (start.white, start.black)
        )
        assert player.search.position() == (me, opp, False, False)
    black = NetPlayer(SearchSettings(batch=2), fake_eval, seed=1)
    white = NetPlayer(SearchSettings(batch=1), fake_eval)
    game = play_game(black, white, (), 1.0, start)
    assert game.flagged is None and game.start == start.text()
    # every move was legal from the start (the referee would have raised), to the end
    board = start.board()
    for move in game.moves:
        board.play(move)
    assert not board.legal()
