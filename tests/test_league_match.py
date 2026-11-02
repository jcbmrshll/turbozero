"""Games between two players, on tic-tac-toe: whether each sees only the moves, and one tiny
run through the whole league."""

import inspect
import json
from dataclasses import replace
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
    SettingsRefused,
    Stopped,
    play_batch,
    play_games,
    select,
)
from core.league.ratings import fit_ratings
from core.league.scheduler import SchedulerConfig
from core.league.store import NetSearch, Player, Store
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

    def make_evaluator(self, network, sims, tree_factor=2):
        return AlphaZero(MCTS)(
            eval_fn=eval_fn,
            num_iterations=sims,
            max_nodes=tree_factor * sims,
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


class UnrecordedSide(NetworkSide):
    """A network's side as it played before searches were recorded."""

    def move(self, states, movers):
        keys = jax.vmap(jax.random.fold_in)(self.keys, states._step_count)
        out = self.program.search(keys, self.trees, states, self.params)
        self.trees = select(movers, out.eval_state, self.trees)
        return np.asarray(out.action)

    def searches(self, movers):
        return {}


def test_recording_a_search_doesnt_change_the_moves_or_the_trees(game, tmp_path):
    programs = Programs(game, "float32")
    a = net_player(1, 16, seed=1, tmp_path=tmp_path)
    b = net_player(2, 16, seed=2, tmp_path=tmp_path)

    def played(kind):
        first, second = (
            kind(
                programs.program(p, len(OPENINGS)),
                programs.network(p)[1],
                jax.random.PRNGKey(key),
            )
            for p, key in ((a, 1), (b, 2))
        )
        results, moves, *_ = play_games(
            game, first, second, OPENINGS, jax.random.PRNGKey(5)
        )
        return results, moves, first.trees, second.trees

    (recorded, moves, *trees), (unrecorded, same_moves, *same_trees) = (
        played(NetworkSide),
        played(UnrecordedSide),
    )
    assert moves == same_moves
    np.testing.assert_array_equal(recorded, unrecorded)
    jax.tree.map(np.testing.assert_array_equal, trees, same_trees)


def test_what_a_search_made_of_a_move_is_recorded(game, tmp_path):
    programs = Programs(game, "float32")
    a = net_player(1, 16, seed=1, tmp_path=tmp_path)
    b = net_player(2, 16, seed=2, tmp_path=tmp_path)
    first = side_of(game, programs, a, len(OPENINGS), key=1)
    second = side_of(game, programs, b, len(OPENINGS), key=2)
    _, moves, _, _, _, searches, steps = play_games(
        game, first, second, OPENINGS, jax.random.PRNGKey(5)
    )
    for game_moves, by_ply in zip(moves, searches):
        # each move past the opening's, and nothing else
        assert sorted(by_ply) == list(range(1, len(game_moves)))
        for ply, search in by_ply.items():
            assert isinstance(search, NetSearch)
            # the fake network's value is 0
            assert search.net_value == 0.0
            assert search.evals == 17
            assert search.tree <= 32
            visits = [top.visits for top in search.top]
            assert visits == sorted(visits, reverse=True)
            assert visits[0] > 0
            assert game_moves[ply] in {
                top.action for top in search.top if top.visits == visits[0]
            }
            assert all(-1.0 <= top.value <= 1.0 for top in search.top)
    # a search's first has nothing to reuse; the root and its iterations. A later one keeps the
    # visits of the moves played since, where it had made them
    nets = [{p: s for p, s in g.items() if isinstance(s, NetSearch)} for g in searches]
    assert all(g[1].visits == 17 for g in nets)
    later = [s.visits for g in nets for ply, s in g.items() if ply > 2]
    assert min(later) >= 17 and max(later) > 17
    assert [s[0] for s in steps] == list(range(1, len(steps) + 1))
    assert steps[0][2] == len(OPENINGS)
    assert all(s[1] > 0 for s in steps)


def test_a_moves_values_are_for_the_player_making_it(game, tmp_path):
    """The root's value is the average of its first evaluation and its children's values, for
    whoever moves, so their signs have to be the root's."""
    programs = Programs(game, "float32")
    a = net_player(1, 32, seed=1, tmp_path=tmp_path)
    first = side_of(game, programs, a, 1, key=1)
    # nothing to reuse, and few enough moves that they all get kept
    plan = np.array([[0, 3, 1, 4]])
    second = Scripted([[5, 6, 7, 8]])
    _, _, _, _, _, searches, _ = play_games(
        game, first, second, plan, jax.random.PRNGKey(0)
    )
    search = searches[0][4]
    assert isinstance(search, NetSearch)
    assert sum(top.visits for top in search.top) == search.visits - 1
    total = search.net_value + sum(top.visits * top.value for top in search.top)
    assert search.value == pytest.approx(total / search.visits, abs=1e-6)
    assert search.value != 0.0


def test_a_batch_records_everything_each_player_was_given(game, tmp_path):
    programs = Programs(game, "float32")
    a = net_player(1, 8, seed=1, tmp_path=tmp_path)
    b = net_player(2, 16, seed=2, tmp_path=tmp_path)
    settings = play_batch(game, programs, a, b, np.array([0, 4]), seed=3).settings
    assert settings is not None
    assert settings.games_at_once == 2
    assert settings.device
    assert [len(half) > 0 for half in settings.steps] == [True, True]
    for played, sims in ((settings.a, 8), (settings.b, 16)):
        json.dumps(played)  # it's stored as text
        assert played["num_iterations"] == sims
        assert played["max_nodes"] == 2 * sims
        assert played["temperature"] == 0.0
        assert played["dirichlet_epsilon"] == 0.0
        assert played["action_selection_config"]["c_init"] == 1.25
        assert played["dtype"] == "float32"
        assert played["net"] == "tiny"
    assert settings.a["checkpoint_sha256"] == "1"
    assert settings.b["checkpoint_sha256"] == "2"


def search_of(kind):
    return kind(
        eval_fn=eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=9,
        max_nodes=4,
        num_iterations=2,
    )


@pytest.mark.parametrize(
    ("made", "kind"),
    [
        (search_of(MCTS), MCTS),
        (search_of(AlphaZero(MCTS)), AlphaZero(MCTS)),
        (PUCTSelector(), PUCTSelector),
    ],
    ids=["mcts", "alphazero", "puct"],
)
def test_every_setting_of_a_search_is_in_its_config(made, kind):
    """A setting left out of `get_config` would go unrecorded, and a change to it unnoticed."""
    aliases = {
        "action_selection_config": "action_selector",
        "policy_target_config": "policy_target",
    }
    configured = {aliases.get(name, name) for name in made.get_config()}
    for cls in kind.__mro__:
        if "__init__" in vars(cls) and cls is not object:
            given = set(inspect.signature(cls.__init__).parameters) - {
                "self",
                "args",
                "kwargs",
            }
            assert given <= configured, cls.__name__


def test_a_player_with_other_settings_than_its_own_is_refused_before_a_move(
    game, tmp_path
):
    programs = Programs(game, "float32")
    a = net_player(1, 8, seed=1, tmp_path=tmp_path)
    b = net_player(2, 8, seed=2, tmp_path=tmp_path)
    own = programs.settings(a, 2)
    # its own settings play on, whatever a traversal's checks
    pinned = replace(a, played_settings={**own, "traversal_steps": 99})
    assert play_batch(game, programs, pinned, b, np.array([0, 4]), seed=3).games
    stale = replace(a, played_settings={**own, "dtype": "bfloat16"})
    moved = []
    with pytest.raises(SettingsRefused) as refused:
        play_batch(
            game,
            programs,
            stale,
            b,
            np.array([0, 4]),
            seed=3,
            should_stop=lambda: moved.append(True) or False,
        )
    assert refused.value.player is stale
    assert refused.value.changed == ["dtype"]
    assert refused.value.settings == own
    assert not moved


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


def test_a_tree_factor_is_a_program_of_its_own(game, tmp_path):
    path = tmp_path / "1.ckpt"
    path.write_text("1")
    plain = Player(id=1, name="n@8", network="n", net="tiny", sims=8, path=str(path))
    wide = replace(plain, id=2, name="n@8-t4", settings={"tree_factor": 4})
    programs = Programs(game, "float32")
    assert programs.program(plain, 2).config["max_nodes"] == 16
    assert programs.program(wide, 2).config["max_nodes"] == 32


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
    # what the games were played with, and what their searches made of them
    pinned = next(p for p in store.players() if p.id == first.id).played_settings
    assert pinned is not None
    assert pinned["num_iterations"] == 2
    assert store.settings_changes() == {}
    assert store.db.execute(
        "SELECT a_settings LIKE '%\"num_iterations\":2%' FROM batches LIMIT 1"
    ).fetchone()[0]
    assert store.game_searches(1)
    store.close()
