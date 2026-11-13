"""The Othello side of the league: its engines' colors, results and openings."""

import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from elo_league import EnginePlayer, Othello
from engines import Analysis, Edax, Egaroucid
from game import action, env
from xot import XOT_PATH, load_xot

from core.league.store import EngineSearch, Player


class FakeEngine:
    """Records what it's told, and replies from a list of moves."""

    name = "fake"
    instances: ClassVar[list] = []

    def __init__(self, engine_dir, level, analyse=False, args=()):
        self.level, self.args = level, args
        self.analyse = analyse
        self.analysis = None
        self.nodes = None
        self.played = []
        self.asked = []
        self.replies = [44, 19]
        FakeEngine.instances.append(self)

    @classmethod
    def default_dir(cls):
        return Path("/nowhere")

    def play(self, color, move):
        self.played.append((color, move))

    def genmove(self, color):
        self.asked.append(color)
        return self.replies.pop(0)

    def discs(self):
        return 40, 24

    def settings(self):
        return {"level": self.level, "args": self.args}

    def close(self):
        pass


@pytest.fixture
def fake():
    FakeEngine.instances = []
    return FakeEngine


def states_of(mask, to_move):
    class States:
        legal_action_mask = np.array(mask)
        current_player = np.array(to_move)

    return States


def test_engine_gets_the_colors_right_as_white(fake):
    player = EnginePlayer(fake, 8, ["-x"])
    legal = [[True] * 64 + [False]]
    player.start(states_of(legal, [0]), np.array([1]))
    assert fake.instances[0].level == 8
    assert fake.instances[0].args == ["-x"]
    everyone = np.array([True])
    # the opening's two moves: black's then white's
    player.observe(np.array([37]), everyone)
    player.observe(np.array([29]), everyone)
    assert fake.instances[0].played == [("b", 37), ("w", 29)]
    # black to move, and the first player (1) isn't the engine's concern: it's asked by id
    moves = player.move(states_of(legal, [1]), everyone)
    assert moves.tolist() == [44]
    assert fake.instances[0].asked == ["b"]
    player.observe(np.array([20]), everyone)
    assert fake.instances[0].played[-1] == ("w", 20)
    moves = player.move(states_of(legal, [1]), everyone)
    assert fake.instances[0].asked == ["b", "b"]
    assert moves.tolist() == [19]
    player.close()


def test_engine_passes_without_being_asked(fake):
    player = EnginePlayer(fake, 8, [])
    only_pass = [[False] * 64 + [True]]
    player.start(states_of(only_pass, [0]), np.array([0]))
    moves = player.move(states_of(only_pass, [0]), np.array([True]))
    assert moves.tolist() == [64]
    assert fake.instances[0].asked == []
    player.close()


def test_an_engine_is_started_to_say_what_its_searches_made_of_its_moves(fake):
    player = EnginePlayer(fake, 8, ["-x"])
    legal = [[True] * 64 + [False], [False] * 64 + [True]]
    player.start(states_of(legal, [0, 0]), np.array([0, 0]))
    assert fake.instances[0].analyse
    assert player.settings() == {"level": 8, "args": ["-x"]}
    fake.instances[0].analysis = [Analysis([19], 2, 10), Analysis([44], 6, 10)]
    fake.instances[0].nodes = 1234
    moves = player.move(states_of(legal, [0, 0]), np.array([True, True]))
    assert moves.tolist() == [44, 64]
    # the move it played's, and none for the game it passed in without being asked
    assert player.searches(np.array([True, True])) == {0: EngineSearch(6, 10, 1234)}
    player.close()


def test_an_engine_that_says_nothing_of_a_search_has_no_record_of_it(fake):
    player = EnginePlayer(fake, 8, [])
    legal = [[True] * 64 + [False]]
    player.start(states_of(legal, [0]), np.array([0]))
    player.move(states_of(legal, [0]), np.array([True]))
    assert player.searches(np.array([True])) == {}
    player.close()


EDAX_LOG = """initial bound = [-64, +64]
direct root_PVS [-64, 64]:
current move:  --  [-64, +64], D6 => -01; pv = D6 :06@73%[+00,+00]; c3:07@73%[+01,+01];
id = 1 ; level =  8@73% ; ab = [-64, +64]:
stability bounds = [-64, +64]:
-01 < score = -01 < -01; time =  0:00.001; nodes = 4729 N; speed =   4729000 Nps
pv = D6 c3
direct root_PVS [-64, 64]:
current move:  --  [-64, +64], D6 => -01; pv = D6 :08@73%[-01,-01]; c3:07@73%[+01,+01];
id = 1 ; level =  8@73% ; ab = [-64, +64]:
-01 < score = -01 < -01; time =  0:00.001; nodes = 6632 N; speed =   6632000 Nps
"""


def edax_reading(log) -> Any:
    engine: Any = Edax.__new__(Edax)
    engine.search_log = io.StringIO(log)
    engine.nodes = None
    return engine


def test_edax_says_how_many_nodes_its_whole_search_took():
    engine = edax_reading(EDAX_LOG)
    analysis = engine.read_analysis(action("d6"))
    assert analysis[0].line[0] == action("d6")
    assert (analysis[0].score, analysis[0].depth) == (-1, 8)
    assert engine.nodes == 6632


def test_egaroucid_says_how_many_nodes_its_whole_search_took():
    engine: Any = Egaroucid.__new__(Egaroucid)
    engine.nodes = None
    engine.log = [
        "level status 8 1 discs depth 8@100%",
        "pre mid depth 7@100% value 0 (raw 0) policy d6 n_worker 0 n_nodes 1442 time 0",
        "main mid depth 8@100% value 1 (raw 2) policy f6 n_worker 0 n_nodes 5293 time 2",
    ]
    analysis = engine.read_analysis(action("f6"))
    assert [(a.score, a.depth) for a in analysis] == [(1, 8)]
    assert engine.nodes == 5293


@pytest.mark.parametrize(
    ("kind", "arguments", "book", "hash_size"),
    [
        (Edax, ["edax", "-book-usage", "off"], False, 21),
        (
            Edax,
            ["edax", "-book-usage", "off", "-h", "25", "-book-usage", "on"],
            True,
            25,
        ),
        (Egaroucid, ["egaroucid", "-nobook"], False, 25),
        (Egaroucid, ["egaroucid", "-hash", "20"], True, 20),
    ],
)
def test_an_engines_book_and_hash_are_read_from_how_its_started(
    kind, arguments, book, hash_size
):
    assert kind.__new__(kind).book_and_hash(arguments) == (book, hash_size)


def test_an_engines_settings_say_what_it_is_and_not_where_it_logs(tmp_path):
    binary, weights = tmp_path / "bin" / "edax", tmp_path / "data" / "eval.dat"
    binary.parent.mkdir()
    weights.parent.mkdir()
    binary.write_bytes(b"binary")
    weights.write_bytes(b"weights")
    engine: Any = Edax.__new__(Edax)
    engine.engine_dir, engine.level, engine.threads = tmp_path, 8, 1
    engine.process = SimpleNamespace(
        args=[
            str(binary),
            "-l",
            "8",
            "-book-usage",
            "off",
            "-search-log-file",
            "/tmp/a",
        ]
    )
    assert engine.settings() == {
        "engine": "edax",
        "level": 8,
        "threads": 1,
        "book": False,
        "hash": 21,
        "command": ["-l", "8", "-book-usage", "off"],
        "binary_sha256": hashlib.sha256(b"binary").hexdigest(),
        "weights_sha256": {"eval.dat": hashlib.sha256(b"weights").hexdigest()},
    }


def test_engine_checks_the_result_against_its_board(fake):
    player = EnginePlayer(fake, 8, [])
    player.start(states_of([[True] * 65], [0]), np.array([0]))
    player.finish(None, np.array([1]))
    with pytest.raises(AssertionError, match="disagrees"):
        player.finish(None, np.array([-1]))
    player.close()


def test_scores_are_the_first_players_then_the_seconds():
    game = Othello()
    state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    scores = game.scores(state, np.asarray(state.current_player))
    assert scores.tolist() == [[2.0, 2.0], [2.0, 2.0]]
    # after black's first move: black has flipped one
    state = jax.vmap(env.step)(state, np.array([37, 37]))
    scores = game.scores(state, np.asarray(state.current_player))
    assert scores.tolist() == [[4.0, 1.0], [4.0, 1.0]]


def test_openings_are_xots():
    game = Othello()
    assert game.num_openings() == len(load_xot(XOT_PATH))
    assert game.openings(np.array([0, 5])).shape == (2, 8)


def test_the_external_settings_reach_the_engine(monkeypatch, fake):
    import elo_league

    monkeypatch.setitem(elo_league.ENGINES, "edax", fake)
    player = Player(
        id=1, name="e", external="edax", settings={"level": 6, "args": ["-a"]}
    )
    side = Othello().external_player(player)
    assert isinstance(side, EnginePlayer)
    assert (side.level, side.args) == (6, ["-a"])
    side.close()


def test_a_players_tree_is_its_factor_times_its_sims():
    othello = Othello()
    assert othello.make_evaluator(None, 100).max_nodes == 200
    assert othello.make_evaluator(None, 100, 4).max_nodes == 400
