"""The Othello side of the league: its engines' colors, and the dashboard's board rules."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from elo_league import EnginePlayer, Othello
from game import env
from xot import XOT_PATH, load_xot

from core.league.store import Player


class FakeEngine:
    """Records what it's told, and replies from a list of moves."""

    name = "fake"
    instances: ClassVar[list] = []

    def __init__(self, engine_dir, level, args=()):
        self.level, self.args = level, args
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


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_boards_rules_match_the_environment(tmp_path):
    """Random games through pgx and through the dashboard's board: the same boards."""
    key = jax.random.PRNGKey(0)
    games = []
    step = jax.jit(env.step)
    for g in range(3):
        key, init = jax.random.split(key)
        state = env.init(init)
        black = int(state.current_player)
        moves, boards = [], []
        while not bool(state.terminated):
            key, pick = jax.random.split(key)
            move = int(
                jax.random.choice(
                    pick, 65, p=state.legal_action_mask / state.legal_action_mask.sum()
                )
            )
            state = step(state, move)
            moves.append(move)
            # +1 for the player to move: black's discs are 1, white's 2
            flip = 1 if int(state.current_player) == black else -1
            board = np.asarray(state._x.board) * flip
            boards.append(np.select([board > 0, board < 0], [1, 2], 0).tolist())
        games.append({"moves": moves, "boards": boards})
    module = tmp_path / "board.mjs"
    module.write_text(
        Path(__file__)
        .parent.parent.joinpath("examples/othello/league_board.js")
        .read_text()
    )
    script = f"""
import board from {json.dumps(str(module))};
import {{ readFileSync }} from "node:fs";
const games = JSON.parse(readFileSync(0, "utf8"));
for (const game of games) {{
  let position = board.start();
  game.moves.forEach((move, i) => {{
    position = board.play(position, move);
    if (JSON.stringify(position.squares) !== JSON.stringify(game.boards[i]))
      throw new Error(`differs at ply ${{i}}`);
  }});
}}
console.log("same");
"""
    out = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(games),
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.stdout.strip() == "same", out.stderr
