"""Games between two players, on tic-tac-toe: whether each sees only the moves, and one tiny
run through the whole league."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pgx
import pytest

from core.evaluators.alphazero import AlphaZero
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.league.game import ExternalPlayer, LeagueGame
from core.league.match import (
    NetworkSide,
    Programs,
    Stopped,
    play_batch,
    play_games,
)
from core.league.ratings import fit_ratings
from core.league.scheduler import SchedulerConfig
from core.league.store import Player, Store
from core.league.worker import play
from core.types import StepMetadata

env = pgx.make("tic_tac_toe")


def step_fn(state, action):
    state = env.step(state, action)
    return state, metadata(state)


def metadata(state):
    return StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count,
    )


def eval_fn(state, params, *args):
    """A network that's only its parameters: fixed logits a square, and a value."""
    return params["logits"], params["value"]


class TicTacToe(LeagueGame):
    name = "tic-tac-toe"
    env = env

    def metadata(self, state):
        return metadata(state)

    @property
    def step_fn(self):
        return step_fn

    def load_network(self, path, net, dtype):
        seed = int(Path(path).read_text())
        logits = jax.random.normal(jax.random.PRNGKey(seed), (9,))
        return None, {"logits": logits, "value": jnp.array(0.0)}

    def make_evaluator(self, network, sims):
        return AlphaZero(MCTS)(
            eval_fn=eval_fn,
            num_iterations=sims,
            max_nodes=2 * sims,
            branching_factor=9,
            action_selector=PUCTSelector(),
            temperature=0.0,
            dirichlet_epsilon=0.0,
        )

    def num_openings(self):
        return 9

    def openings(self, indices):
        return np.asarray(indices)[:, None]


class Scripted(ExternalPlayer):
    """Plays the moves it's given, game by game, in order."""

    def __init__(self, games):
        self.games = [list(moves) for moves in games]
        self.seen = []

    def start(self, states, first):
        self.started = True

    def move(self, states, movers):
        return np.array(
            [g.pop(0) if m else 0 for g, m in zip(self.games, movers)], dtype=np.int32
        )

    def observe(self, actions, movers):
        self.seen.append((actions.copy(), movers.copy()))

    def finish(self, states, results):
        self.results = results

    def close(self):
        self.closed = True


def net_player(id_, sims, seed, tmp_path):
    path = tmp_path / f"{seed}.ckpt"
    path.write_text(str(seed))
    return Player(
        id=id_,
        name=f"{seed}@{sims}",
        sims=sims,
        network=str(seed),
        path=str(path),
        net="tiny",
        run=str(seed),
    )


@pytest.fixture
def game():
    return TicTacToe()


def side_of(game, programs, player, games, key=0):
    return NetworkSide(
        programs.program(player, games),
        programs.network(player)[1],
        jax.random.PRNGKey(key),
    )


OPENINGS = np.array([[0], [4], [8], [2]])


def test_a_players_moves_depend_only_on_the_moves_it_sees(game, tmp_path):
    programs = Programs(game, "float32")
    a = net_player(1, 8, seed=1, tmp_path=tmp_path)
    key = jax.random.PRNGKey(5)

    def against(opponent_seed):
        opponent = net_player(2, 8, seed=opponent_seed, tmp_path=tmp_path)
        first = side_of(game, programs, a, len(OPENINGS), key=1)
        second = side_of(game, programs, opponent, len(OPENINGS), key=2)
        _, moves, *_ = play_games(game, first, second, OPENINGS, key)
        return moves

    # a plays first, so moves 2, 4, 6, 8 of the game are its opponent's after the opening
    moves = against(opponent_seed=2)
    theirs = [g[1::2] for g in moves]

    # the same opponent's moves, from an engine with nothing else in common with it
    scripted = Scripted(theirs)
    first = side_of(game, programs, a, len(OPENINGS), key=1)
    _, replayed, *_ = play_games(game, first, scripted, OPENINGS, key)
    assert replayed == moves


def test_each_network_keeps_its_own_tree_and_parameters(game, tmp_path):
    programs = Programs(game, "float32")
    a = net_player(1, 8, seed=1, tmp_path=tmp_path)
    b = net_player(2, 8, seed=2, tmp_path=tmp_path)
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    first_id = np.asarray(states.current_player)
    one, two = side_of(game, programs, a, 2), side_of(game, programs, b, 2)
    assert programs.program(a, 2) is programs.program(b, 2)  # compiled once
    assert one.params is not two.params
    assert not (
        np.asarray(one.params["logits"]) == np.asarray(two.params["logits"])
    ).all()
    one.start(states, first_id)
    two.start(states, first_id)
    assert one.trees is not two.trees
    before = jax.tree.map(np.asarray, two.trees)
    one.move(states, np.ones(2, dtype=bool))
    # one searching changes nothing of the other's
    jax.tree.map(
        np.testing.assert_array_equal, before, jax.tree.map(np.asarray, two.trees)
    )


def test_openings_are_played_with_each_moving_first(game, tmp_path):
    programs = Programs(game, "float32")
    a = net_player(1, 2, seed=1, tmp_path=tmp_path)
    b = net_player(2, 2, seed=2, tmp_path=tmp_path)
    result = play_batch(game, programs, a, b, np.array([0, 4, 8, 2]), seed=3)
    assert len(result.games) == 8
    assert result.compiled
    # each side moves in each game, beyond the moves of the openings
    played = sum(
        len(g.moves) - game.openings(np.array([g.opening])).shape[1]
        for g in result.games
    )
    assert result.a_clock.moves + result.b_clock.moves <= played
    assert result.a_clock.moves > 0 and result.a_clock.seconds > 0
    assert result.b_clock.moves > 0 and result.b_clock.seconds > 0
    assert sorted((g.first, g.opening) for g in result.games) == sorted(
        [(p, o) for p in (1, 2) for o in (0, 4, 8, 2)]
    )
    for g in result.games:
        assert g.moves[0] == g.opening
        assert g.result in (-1, 0, 1)
    # the same programs a second time: nothing to compile
    assert not play_batch(game, programs, a, b, np.array([1, 3, 5, 7]), seed=4).compiled


def test_results_are_the_first_players(game):
    # the top row, for the first player, with the opening its first move
    first = Scripted([[1, 2]])
    second = Scripted([[3, 4]])
    results, moves, *_ = play_games(
        game, first, second, np.array([[0]]), jax.random.PRNGKey(0)
    )
    assert results.tolist() == [1]
    assert moves == [[0, 3, 1, 4, 2]]
    assert first.results.tolist() == [1]


def test_external_players_are_told_only_what_they_didnt_play(game):
    first, second = Scripted([[1, 2]]), Scripted([[3, 4]])
    play_games(game, first, second, np.array([[0]]), jax.random.PRNGKey(0))
    # the opening's move, and the two moves of the other's before the game ended
    assert [int(a[0]) for a, _ in first.seen] == [0, 3, 4]
    assert [int(a[0]) for a, _ in second.seen] == [0, 1, 2]


def test_should_stop_abandons_the_games(game):
    first, second = Scripted([[1, 2]]), Scripted([[3, 4]])
    calls = iter([False, False, True])
    with pytest.raises(Stopped):
        play_games(
            game,
            first,
            second,
            np.array([[0]]),
            jax.random.PRNGKey(0),
            lambda: next(calls),
        )


def test_an_illegal_move_is_refused(game):
    first, second = Scripted([[0]]), Scripted([[1]])
    with pytest.raises(AssertionError, match="illegal"):
        play_games(game, first, second, np.array([[0]]), jax.random.PRNGKey(0))


class NoGate:
    def busy(self):
        return None


def test_a_tiny_league_plays_and_rates(game, tmp_path):
    """The one run through it all: networks of a few sims, on tic-tac-toe."""
    store = Store(tmp_path / "league")
    for seed in (1, 2):
        path = tmp_path / f"{seed}.ckpt"
        path.write_text(str(seed))
        store.add_network(f"n{seed}", path, "tiny", f"run{seed}")
        store.add_net_player(f"n{seed}", 2)
    store.add_net_player("n1", 4)
    first, second = store.players()[:2]
    store.request_match(first.id, second.id, 4)
    config = SchedulerConfig(batch_sizes=(4,))
    reason = play(
        store,
        game,
        NoGate(),
        config=config,
        seed=0,
        max_batches=3,
        dtype="float32",
        version="v",
    )
    assert reason == "played its batches"
    assert store.num_games() == 12
    assert store.pending_matches() == []
    ids = [p.id for p in store.players()]
    fit = fit_ratings(ids, store.pair_records())
    assert np.isfinite(fit.elo).all()
    assert (
        store.db.execute("SELECT DISTINCT code_version FROM batches").fetchall()[0][0]
        == "v"
    )
    store.close()
