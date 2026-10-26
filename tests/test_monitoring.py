"""Logging to a seaglass server: runs, with turbozero's views, and test episodes packed for
them."""

import functools
import io
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from seaglass import Arrays
from seaglass.bundle import problems, read_bundle

from core.common import GameFrame
from core.evaluators.mcts.state import SearchRecord
from core.monitoring import VIEWS, Monitor
from core.monitoring.episodes import (
    BOARDS,
    episode_type,
    line_record,
    pgx_two_player_episode,
    with_opening,
)

# one compilation of each environment's init and step, whichever test steps it
jit = functools.cache(jax.jit)


def test_run_is_logged_with_its_config_metrics_and_views(seaglass_server):
    monitor = Monitor(seaglass_server.url, project="tests", name="round-trip")
    monitor.start(
        config={"batch_size": 4, "run": {"seed": 3, "num_epochs": 2}, "fn": np.mean},
        total=1,
    )
    monitor.log(0, {"loss": np.float32(1.5), "policy_loss": jnp.asarray(2)})
    monitor.log(1, {"loss": 0.5, "value_loss": float("nan")})
    monitor.finish()

    meta = seaglass_server.get(f"/api/runs/{monitor.run_id}")
    assert (meta["project"], meta["name"], meta["status"]) == (
        "tests",
        "round-trip",
        "finished",
    )
    assert (meta["step"], meta["total"]) == (1, 1)
    # functions in the config are recorded by name
    assert meta["config"]["fn"] == "mean"
    assert meta["config"]["run"] == {"seed": 3, "num_epochs": 2}
    rows = seaglass_server.get(f"/api/runs/{monitor.run_id}/metrics")["rows"]
    assert [(r["step"], r["loss"]) for r in rows] == [(0, 1.5), (1, 0.5)]
    assert rows[0]["policy_loss"] == 2 and rows[1]["value_loss"] is None
    # the run's page is drawn with turbozero's views, which the server now has
    assert meta["bundle"] == monitor.bundle_hash
    files = seaglass_server.get(f"/api/bundles/{meta['bundle']}")["files"]
    assert {f["path"] for f in files} == set(read_bundle(VIEWS))


def test_views_draw_every_board_games_episodes():
    bundle = read_bundle(VIEWS)
    # every module it names is there, and imports only the bundle's own files
    assert problems(bundle) == []
    layout = json.loads(bundle["layout.json"])
    assert layout["contract"] == 1
    assert {episode_type(env) for env in BOARDS} <= set(layout["renderers"])


def stacked(states):
    """States stacked along a leading time axis."""
    return jax.tree.map(lambda *xs: np.stack(xs), *states)


def tic_tac_toe_frames(ttt, moves, num_frames, key, p1_search=None, p2_search=None):
    """The frames `two_player_game` records of tic-tac-toe played with `moves` from the initial
    state from `key`: the initial state and the state after each move, then the last one again
    up to `num_frames`, as after a game ends. The first evaluator's value estimates are 0,
    0.1, 0.2, ..., the second's their negatives."""
    states = [jit(ttt.env.init)(key)]
    for action in moves:
        states.append(jit(ttt.env.step)(states[-1], action))
    states += [states[-1]] * (num_frames - len(states))
    completed = np.array([bool(state.terminated) for state in states])
    values = np.arange(num_frames) / 10
    return GameFrame(
        env_state=stacked(states),
        p1_value_estimate=jnp.asarray(values),
        p2_value_estimate=jnp.asarray(-values),
        completed=jnp.asarray(completed),
        # each player's reward from the step that ends the game on
        outcomes=jnp.asarray(
            np.where(completed[:, None], np.asarray(states[-1].rewards), 0.0)
        ),
        p1_search=p1_search,
        p2_search=p2_search,
    )


def test_pgx_episode_is_logged_for_the_views(seaglass_server, ttt):
    # playing the first free square, X wins on the diagonal with the 7th move; the trained
    # agent (the first evaluator) moved second, so played O, and lost
    frames = tic_tac_toe_frames(ttt, range(7), num_frames=10, key=jax.random.PRNGKey(0))
    agent = 1 - int(frames.env_state.current_player[0])
    monitor = Monitor(seaglass_server.url)
    monitor.start()

    episode = pgx_two_player_episode()(
        frames, np.array([agent, 1 - agent]), ttt.step_fn
    )
    monitor.log(2, {"game": episode})
    monitor.finish()

    [entry] = seaglass_server.get(f"/api/runs/{monitor.run_id}/media")["rows"]
    assert (entry["key"], entry["step"], entry["type"]) == (
        "game",
        2,
        "turbozero/tic_tac_toe-episode",
    )
    assert "attrs" not in entry
    body = seaglass_server.file(f"/api/runs/{monitor.run_id}/files/{entry['file']}")
    with np.load(io.BytesIO(body)) as arrays:
        # the initial board and 7 moves, without the frames after the game ended
        board = arrays["board"]
        assert board.shape == (1 + 7, 3, 3)
        assert (board[0] == 0).all()
        assert board[-1].tolist() == [[1, 2, 1], [2, 1, 2], [1, 0, 0]]
        assert arrays["to_move"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
        np.testing.assert_allclose(arrays["values"][:, 0], np.arange(8) / 10)
        np.testing.assert_allclose(arrays["values"][:, 1], -np.arange(8) / 10)
        assert (arrays["agent"].item(), arrays["completed"].item()) == (1, True)
        assert arrays["outcome"].item() == -1
        # the game didn't record its searches
        assert "search_offset" not in arrays


def test_episode_names_its_opponent(seaglass_server):
    # as a ladder names its rung, or watch.py its engine
    monitor = Monitor(seaglass_server.url)
    monitor.start()
    episode = Arrays(episode_type("othello"), attrs={"opponent": "edax level 8"}, x=[1])

    monitor.log(0, {"edax_level8_game": episode})
    monitor.finish()

    [entry] = seaglass_server.get(f"/api/runs/{monitor.run_id}/media")["rows"]
    assert entry["attrs"] == {"opponent": "edax level 8"}


def search_record(*nodes, capacity=4):
    """A `SearchRecord` of tic-tac-toe of a hand-built tree: `nodes` are (parent, action, q),
    the root's first, with parent and action -1; the rest of the tree's capacity is unused. The
    node at index i has i + 1 visits, a prior of 1 / (i + 1), and isn't terminal."""
    unused = [(-1, -1, 0.0)] * (capacity - len(nodes))
    parents, actions, q = np.array(list(nodes) + unused).T
    visits = np.arange(1, capacity + 1)
    return SearchRecord(
        parents=jnp.asarray(parents, dtype=jnp.int32),
        actions=jnp.asarray(actions, dtype=jnp.int32),
        n=jnp.asarray(visits, dtype=jnp.int32),
        q=jnp.asarray(q, dtype=jnp.float32),
        prior=jnp.asarray(1 / visits, dtype=jnp.float32),
        terminated=jnp.asarray(np.zeros(capacity, dtype=bool)),
        policy=jnp.asarray(np.full(9, 1 / 9), dtype=jnp.float32),
        num_nodes=jnp.asarray(len(nodes), dtype=jnp.int32),
    )


def test_pgx_episode_carries_both_players_searches(ttt):
    # the trained agent moves first (X) to the centre, then its opponent to the corner. Frame
    # t + 1 holds what each player made of the position at t: the agent searched both
    # positions (keeping its tree of the second), its opponent only the second, so had no
    # tree of the first
    no_tree = search_record()
    agent_searches = [
        no_tree,
        # its first node took an evicted node's slot, before its parent
        search_record((-1, -1, 0.1), (2, 0, 0.3), (0, 4, -0.2)),
        search_record((-1, -1, 0.2), (0, 0, 0.4)),
    ]
    opponent_searches = [
        no_tree,
        no_tree,
        search_record((-1, -1, -0.2), (0, 8, 0.6)),
    ]
    frames = tic_tac_toe_frames(
        ttt,
        [4, 0],
        num_frames=3,
        key=jax.random.PRNGKey(0),
        p1_search=stacked(agent_searches),
        p2_search=stacked(opponent_searches),
    )
    agent = int(frames.env_state.current_player[0])

    episode = pgx_two_player_episode()(
        frames, np.array([agent, 1 - agent]), ttt.step_fn
    )

    arrays = episode.arrays
    assert arrays["search_frame"].tolist() == [0, 1, 1]
    assert arrays["search_player"].tolist() == [0, 0, 1]
    assert arrays["search_offset"].tolist() == [0, 3, 5, 7]
    assert arrays["node_parent"].tolist() == [-1, 2, 0, -1, 0, -1, 0]
    assert arrays["node_action"].tolist() == [-1, 0, 4, -1, 0, -1, 8]
    # each node's board, replayed from the position searched: 1 for X, 2 for O
    empty, centre = [0] * 9, [0, 0, 0, 0, 1, 0, 0, 0, 0]
    corner, other_corner = [2, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 2]
    expected = [empty, corner, centre, centre, corner, centre, other_corner]
    assert arrays["node_board"].dtype == np.uint8
    assert arrays["node_board"].reshape(7, 9).tolist() == expected
    assert arrays["node_to_move"].tolist() == [0, 0, 1, 1, 0, 1, 0]
    np.testing.assert_array_equal(
        arrays["node_legal"], arrays["node_board"].reshape(7, 9) == 0
    )
    # each node's value for the player who chose its move: the root's own, a child's its
    # parent's, the opposite of its q (players alternate in tic-tac-toe)
    np.testing.assert_allclose(
        arrays["node_value"], [0.1, -0.3, 0.2, 0.2, -0.4, -0.2, -0.6], rtol=1e-6
    )
    # the rest of each node's statistics as they are
    assert arrays["node_visits"].tolist() == [1, 2, 3, 1, 2, 1, 2]
    np.testing.assert_allclose(
        arrays["node_prior"], 1 / np.array([1, 2, 3, 1, 2, 1, 2]), rtol=1e-6
    )
    assert not arrays["node_terminal"].any()
    assert arrays["search_policy"].shape == (3, 9)


def test_pgx_episode_carries_an_engines_lines(ttt):
    # the trained agent moves first (X) to the centre, then an engine that only scores moves
    # (O) to the corner. It scored two: the corner, expecting the game to go 0, 8, 2, worth 0.5
    # to it, and the edge, 1, worth at most -0.25, from a search 3 moves deep. It scored
    # nothing of the agent's position
    lines = functools.partial(line_record, num_actions=9, capacity=6)
    no_tree = search_record()
    frames = tic_tac_toe_frames(
        ttt,
        [4, 0],
        num_frames=3,
        key=jax.random.PRNGKey(0),
        p1_search=stacked([no_tree, search_record((-1, -1, 0.1)), no_tree]),
        p2_search=stacked(
            [
                lines(None),
                lines(None),
                lines([([0, 8, 2], 0.5, 0), ([1], -0.25, -1)], 3),
            ]
        ),
    )
    agent = int(frames.env_state.current_player[0])

    episode = pgx_two_player_episode()(
        frames, np.array([agent, 1 - agent]), ttt.step_fn
    )

    arrays = episode.arrays
    assert arrays["search_frame"].tolist() == [0, 1]
    assert arrays["search_player"].tolist() == [0, 1]
    # a tree's search has no depth
    assert arrays["search_depth"].tolist() == [-1, 3]
    assert arrays["search_offset"].tolist() == [0, 1, 6]
    # a line per move scored, a node per move of it, each the last one's child
    assert arrays["node_parent"][1:].tolist() == [-1, 0, 1, 2, 0]
    assert arrays["node_action"][1:].tolist() == [-1, 0, 8, 2, 1]
    assert arrays["node_board"][4].reshape(9).tolist() == [2, 0, 2, 0, 1, 0, 0, 0, 1]
    assert arrays["node_board"][5].reshape(9).tolist() == [0, 2, 0, 0, 1, 0, 0, 0, 0]
    # each node's value for the player who chose it: the root's the best line's, a line's
    # own moves its value, the agent's the opposite
    np.testing.assert_allclose(arrays["node_value"][1:], [0.5, 0.5, -0.5, 0.5, -0.25])
    # the edge's value is at most that; the corner's line's, and the agent's tree's, exact
    assert arrays["node_bound"].tolist() == [0, 0, 0, 0, 0, -1]
    # no visits or priors
    assert not arrays["node_visits"][1:].any() and not arrays["node_prior"][1:].any()


def test_pgx_episode_keeps_the_opening_played_for_its_players(ttt):
    # X's first move to the centre was played for the players, who started from the position
    # after it: put back before their game, the opening's position is the episode's first,
    # and the views start at theirs
    whole = tic_tac_toe_frames(ttt, [4, 0, 8], num_frames=4, key=jax.random.PRNGKey(0))
    played = jax.tree.map(lambda x: x[1:], whole)
    opening = [jax.tree.map(lambda x: x[0], whole.env_state)]
    agent = int(played.env_state.current_player[0])

    episode = pgx_two_player_episode()(
        with_opening(played, opening),
        np.array([agent, 1 - agent]),
        ttt.step_fn,
        start=1,
    )

    expected = pgx_two_player_episode()(
        whole, np.array([agent, 1 - agent]), ttt.step_fn
    )
    np.testing.assert_array_equal(episode.arrays["board"], expected.arrays["board"])
    np.testing.assert_array_equal(episode.arrays["to_move"], expected.arrays["to_move"])
    assert int(episode.arrays["agent"]) == int(expected.arrays["agent"])
    # nobody had a value of the opening's position
    assert episode.arrays["values"][0].tolist() == [0, 0]
    assert (int(episode.arrays["start"]), int(expected.arrays["start"])) == (1, 0)


@pytest.mark.parametrize("env_id", ["othello", "connect_four"])
def test_pgx_boards_are_drawn_from_the_first_movers_side(env_id):
    import pgx

    env = pgx.make(env_id)
    states = [jit(env.init)(jax.random.PRNGKey(0))]
    for _ in range(3):
        action = int(np.argmax(states[-1].legal_action_mask))
        states.append(jit(env.step)(states[-1], action))

    board, to_move = BOARDS[env_id](stacked(states)._x)

    assert to_move.tolist() == [0, 1, 0, 1]
    # pieces of the colour that moves first are 1, whichever pgx player that is and
    # whoever is to move
    if env_id == "othello":
        assert board[0][3].tolist()[3:5] == [2, 1]
        # black's first move, at the first legal square (d3), turns d4 black
        assert (board[1][2][3], board[1][3][3]) == (1, 1)
    else:
        # the first two discs drop into the bottom of the leftmost column
        assert board[2][5][0] == 1 and board[2][4][0] == 2
