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

from elo_league import EnginePlayer, Othello, player_facts
from game import env, export_last
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
    """Random games through pgx and through the dashboard's board: the same boards, moves, ends,
    results and network inputs."""
    key = jax.random.PRNGKey(0)
    games = []
    step = jax.jit(env.step)
    for g in range(3):
        key, init = jax.random.split(key)
        state = env.init(init)
        black = int(state.current_player)
        moves, boards, legal, over, seen, rewards = [], [], [], [], [], []
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
            legal.append(np.flatnonzero(state.legal_action_mask).tolist())
            over.append(bool(state.terminated))
            # (8, 8, 2) to the network's channels first
            seen.append(
                np.moveaxis(np.asarray(state.observation), -1, 0)
                .astype(int)
                .ravel()
                .tolist()
            )
            to_move = int(state.current_player)
            rewards.append(float(state.rewards[to_move]))
        games.append(
            {
                "moves": moves,
                "boards": boards,
                "legal": legal,
                "over": over,
                "seen": seen,
                "rewards": rewards,
            }
        )
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
    if (position.terminated !== game.over[i]) throw new Error(`end differs at ply ${{i}}`);
    if (position.terminated && position.reward !== game.rewards[i]) throw new Error("result differs");
    if (!position.terminated) {{
      if (JSON.stringify(board.legal(position)) !== JSON.stringify(game.legal[i]))
        throw new Error(`moves differ at ply ${{i}}`);
      const seen = new Float32Array(128);
      board.observe(position, seen);
      if (JSON.stringify([...seen]) !== JSON.stringify(game.seen[i]))
        throw new Error(`observation differs at ply ${{i}}`);
    }}
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


NET = "resnet:8x128,pool_every=2,heads=pooled,pool_channels=32"


def net_player(network, sims=1000, net=NET):
    return Player(id=1, name=f"{network}@{sims}", network=network, net=net, sims=sims)


def test_a_self_play_networks_facts_come_from_its_name():
    assert player_facts(net_player("8x128-128s-8p-w1125-274m")) == {
        "kind": "net", "net": "8x128", "blocks": 8, "channels": 128, "play sims": 1000,
        "train sims": 128, "passes": 8, "window": 1125, "minutes": 274,
        "vtq": False, "planned": False,
    }  # fmt: skip


def test_a_names_optional_parts_are_facts_too():
    facts = player_facts(net_player("8x128-128s-8p-w375-vtq-68m"))
    assert (facts["vtq"], facts["planned"], facts["minutes"]) == (True, False, 68)
    assert player_facts(net_player("8x128-128s-8p-w1125-68m-planned"))["planned"]


def test_a_students_facts_come_from_its_name():
    facts = player_facts(
        net_player("4x64-distill-teacherv-274m-32p", 128, "resnet:4x64")
    )
    assert (facts["net"], facts["distill target"], facts["teacher minutes"]) == (
        "4x64",
        "teacherv",
        274,
    )
    assert facts["passes"] == 32 and "minutes" not in facts


def test_a_name_that_doesnt_fit_still_has_its_nets_facts():
    assert player_facts(net_player("something-else")) == {
        "kind": "net", "net": "8x128", "blocks": 8, "channels": 128, "play sims": 1000,
    }  # fmt: skip


def test_an_engines_facts_come_from_its_settings():
    engine = Player(id=1, name="e", external="edax-okuhara", settings={"level": 20})
    assert player_facts(engine) == {
        "kind": "engine",
        "engine": "edax-okuhara",
        "level": 20,
    }
    paper = Player(id=2, name="OLIVAW@400", external="olivaw", settings={"sims": 400})
    assert player_facts(paper)["play sims"] == 400


def test_a_failed_export_at_the_end_of_a_run_only_says_so(tmp_path, capsys):
    (tmp_path / "7.eqx").write_bytes(b"not a network")
    export_last(str(tmp_path), "resnet:2x8,heads=pooled,pool_channels=2")
    assert "no ONNX export" in capsys.readouterr().err
    assert (tmp_path / "7.eqx").read_bytes() == b"not a network"
    assert not list(tmp_path.glob("*.onnx*"))


def test_a_run_with_no_checkpoints_has_nothing_to_export(tmp_path, capsys):
    export_last(str(tmp_path), "resnet:2x8")
    assert "no ONNX export" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_browser_search_counts_and_signs_its_values():
    """A fake network and game: the search visits what it's told to, each level up the tree negates
    a value, and it stops when told to."""
    root = Path(__file__).parent.parent
    script = f"""
import {{ Search, snapshot, policy }} from {json.dumps(str(root / "core/league/web/search.js"))};
const p = policy(new Float32Array([0, Math.log(3), 99]), 0, [0, 1], 3);
if (Math.abs(p[0] - 0.25) > 1e-6 || Math.abs(p[1] - 0.75) > 1e-6 || p[2] !== 0) throw new Error("policy");
// a game of two moves: each of two actions leads on, and the second move ends it with a win
const board = {{
  numActions: 2,
  legal: (position) => (position.terminated ? [] : [0, 1]),
  play: (position, action) => ({{ depth: position.depth + 1, terminated: position.depth === 1, reward: -1 }}),
}};
const start = {{ depth: 0, terminated: false, reward: 0 }};
const search = new Search(board, async () => ({{ logits: new Float32Array([1, 0]), value: -0.5 }}));
const root = await search.run(start, 8);
if (root.n !== 9) throw new Error(`root visits ${{root.n}}`);
const counts = [...root.children.values()].reduce((sum, child) => sum + child.n, 0);
if (counts !== 8) throw new Error(`child visits ${{counts}}`);
const tree = snapshot(root);
if (tree.parent[0] !== -1 || tree.n[0] !== 9) throw new Error("snapshot");
const small = snapshot(root, 2);
if (small.n.length !== 2 || small.parent[1] !== 0) throw new Error("limit");
let stopped = 0;
const partial = await new Search(board, async () => ({{ logits: new Float32Array([1, 0]), value: 0 }})).run(start, 100, async () => (++stopped < 2), 4);
if (partial.n !== 9) throw new Error(`stopping: ${{partial.n}}`);
// by hand: the root and first child evaluate to -0.5 for whoever moves, so the root's q averages
// -0.5 with +0.5; then the second move ends the game at -1 for the player to move at the root
const two = await new Search(board, async () => ({{ logits: new Float32Array([1, 0]), value: -0.5 }})).run(start, 1);
if (two.q !== 0 || two.children.get(0).q !== -0.5) throw new Error(`one: ${{two.q}}`);
const three = await new Search(board, async () => ({{ logits: new Float32Array([1, 0]), value: -0.5 }})).run(start, 2);
const child = three.children.get(0);
if (child.q !== 0.25 || Math.abs(three.q + 1 / 3) > 1e-12) throw new Error(`two: ${{three.q}} ${{child.q}}`);
console.log("ok");
"""
    out = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.stdout.strip() == "ok", out.stderr
