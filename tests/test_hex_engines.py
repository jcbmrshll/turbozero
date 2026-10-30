"""Hex's games against MoHex, with fake engines in place of the processes, and its labelled set."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from example_modules import load_example

engines, engine_set, game_module, label_positions, openings, vs_engine = load_example(
    "hex", "engines", "engine_set", "game", "label_positions", "openings", "vs_engine"
)
GAME = game_module.HexGame(5)
STEP = jax.jit(GAME.env.step)
ADVANCE = jax.jit(
    jax.vmap(lambda tree, state, a: (tree, GAME.env.step(state, a)), in_axes=(0, 0, 0))
)


class FakeEngine(engines.MoHex):
    """Plays the lowest legal cell, or swaps if told to, on a pgx board of its own, which it
    builds from what it's told as MoHex would: a swap sets its board up as pgx's is."""

    def __init__(self, game, swaps=False):  # pylint: disable=super-init-not-called
        self.game = game
        self.analysis = None
        self.swaps = swaps
        self.state = GAME.env.init(jax.random.PRNGKey(0))
        self.commands = []

    def advance(self, move):
        self.state = STEP(self.state, move)

    def play(self, color, move, stone=None):
        self.commands.append((color, move))
        assert ("b" if int(self.state._x.color) == 0 else "w") == color
        self.advance(move)

    def genmove(self, color, stone=None):
        assert ("b" if int(self.state._x.color) == 0 else "w") == color
        if self.swaps and bool(self.state.legal_action_mask[self.game.swap]):
            assert stone == np.flatnonzero(np.asarray(self.state._x.board))[0]
            move = self.game.swap
        else:
            move = int(np.flatnonzero(np.asarray(self.state.legal_action_mask))[0])
        self.advance(move)
        return move

    def winner(self):
        if not bool(self.state.terminated):
            return ""
        # the player to move is the loser, as pgx has it
        return "bw"[1 - int(self.state._x.color)]


class Out:
    def __init__(self, action, eval_state):
        self.action = action
        self.eval_state = eval_state


class FakeEvaluator:
    def init(self, template_embedding):
        return jnp.zeros(())


def lowest_legal_search(keys, trees, states):
    return Out(jnp.argmax(states.legal_action_mask, axis=1), trees)


def play(num_games, opening_list, we_first, swaps=False):
    fakes = [FakeEngine(GAME, swaps) for _ in range(num_games)]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(2) as pool:
        results = vs_engine.play_games(
            fakes,
            np.array(opening_list),
            np.array(we_first),
            lowest_legal_search,
            ADVANCE,
            FakeEvaluator(),
            pool,
            jax.random.split(jax.random.PRNGKey(0), num_games),
            GAME,
        )
    return results, fakes


def test_games_are_played_to_the_end_and_agree_with_the_engine():
    results, fakes = play(
        4,
        [[12, openings.NO_MOVE], [12, openings.NO_MOVE], [0, 6], [0, 6]],
        [True, False, True, False],
    )
    assert set(np.abs(results).tolist()) == {1}
    for fake, result, we_first in zip(fakes, results, [True, False, True, False]):
        assert bool(fake.state.terminated)
        ours = "b" if we_first else "w"
        assert fake.winner() == (ours if result > 0 else "bw"[ours == "b"])


def test_an_engine_that_swaps_keeps_the_boards_the_same():
    # the opening's single stone, then the engine's swap as colour 1
    results, fakes = play(2, [[7, openings.NO_MOVE]] * 2, [True, True], swaps=True)
    assert set(np.abs(results).tolist()) == {1}
    for fake in fakes:
        assert int(fake.state._x.step_count) >= 2
        # the stone it swapped, on c2, is on b3 now, and colour 1's
        board = np.asarray(fake.state._x.board)
        colour = int(fake.state._x.color)
        assert (board[11] > 0) == (colour == 1)
        assert board[11] != 0


class Scripted(engines.MoHex):
    """MoHex's commands as sent, with scripted replies."""

    def __init__(self, game, replies=()):  # pylint: disable=super-init-not-called
        self.game = game
        self.analyse = False
        self.analysis = None
        self.sent = []
        self.replies = list(replies)

    def command(self, command):
        self.sent.append(command)
        if command.startswith("reg_genmove"):
            return self.replies.pop(0)
        return ""


def test_a_swap_sets_the_engine_up_as_pgx_is():
    engine = Scripted(GAME)
    engine.play("w", GAME.swap, stone=7)
    # c2's reflection is b3, as colour 1's, with the swap rule off so that black moves next
    assert engine.sent == ["clear_board", "param_game allow_swap 0", "play w b3"]


def test_the_engines_swap_comes_back_as_the_swap_action():
    engine = Scripted(GAME, ["swap-pieces"])
    assert engine.genmove("w", stone=7) == GAME.swap
    assert engine.sent[1:] == ["clear_board", "param_game allow_swap 0", "play w b3"]


def test_its_move_is_played_on_its_board_and_returned():
    engine = Scripted(GAME, ["C2"])
    assert engine.genmove("b") == 7
    assert engine.sent == ["reg_genmove b", "play b c2"]


def test_parse_values():
    game = game_module.HexGame(7)
    values = engines.parse_values(
        " a1 .250@12 b1 W@3k c1 L@1.20m d1 0 e1 .750@2k", game
    )
    assert [game.cell_name(v.line[0]) for v in values] == ["b1", "e1", "a1", "c1"]
    assert [v.score for v in values] == pytest.approx([1, 0.5, -0.5, -1])
    assert [v.proven for v in values] == [True, False, False, True]
    assert [v.visits for v in values] == [3000, 2000, 12, 1_200_000]


def test_observation_of_a_position_matches_pgxs():
    moves = [7, 3, 12, 20]
    state = GAME.env.init(jax.random.PRNGKey(0))
    for move in moves:
        state = STEP(state, move)
    obs, legal = label_positions.observation(GAME, moves)
    np.testing.assert_array_equal(obs, state.observation)
    np.testing.assert_array_equal(legal, state.legal_action_mask)
    # and of the position the swap is on
    state = STEP(GAME.env.init(jax.random.PRNGKey(0)), 7)
    obs, legal = label_positions.observation(GAME, [7])
    np.testing.assert_array_equal(obs, state.observation)
    np.testing.assert_array_equal(legal, state.legal_action_mask)


def test_build_takes_distinct_positions_and_scores_each(tmp_path):
    def games(i):
        return [[0, 6], [0, 6, 12], [i % 3 + 1, 8]]

    def score(moves):
        _, legal = label_positions.observation(GAME, moves)
        return np.where(legal, 0.5, engine_set.ILLEGAL).astype(np.float32)

    obs, legal, scores, stones = label_positions.build(
        GAME, 4, score, games, 2, 3, np.random.default_rng(0)
    )
    assert len(obs) == 4
    assert len({o.tobytes() for o in obs}) == 4
    labelled = engine_set.EngineSet(obs, legal, scores, stones, "mohex", 0)
    path = tmp_path / "set.npz"
    labelled.save(path)
    loaded = engine_set.EngineSet.load(path)
    np.testing.assert_array_equal(loaded.scores, scores)
    np.testing.assert_array_equal(loaded.obs, obs)
    assert loaded.describe() == {"engine": "mohex", "level": 0, "positions": 4}
    assert set(loaded.scored_positions().groups) == {"opening", "midgame", "endgame"}


def test_a_set_must_score_exactly_the_legal_actions():
    legal = np.array([[True, False]])
    with pytest.raises(ValueError, match="every legal action"):
        engine_set.EngineSet(
            np.zeros((1, 1, 1, 4), bool),
            legal,
            np.array([[0.5, 0.5]], np.float32),
            np.zeros(1),
            "mohex",
            0,
        )
