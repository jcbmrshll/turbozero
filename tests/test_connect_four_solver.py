"""The solver's protocol and what's built on its scores, with a fake process and a small search
standing in for it (a test that needs the real solver is skipped where it isn't installed)."""

import stat
from pathlib import Path

import jax
import numpy as np
import pytest
from example_modules import load_example

game, solver_module, solver_set_module, label_positions, build_openings, openings = (
    load_example(
        "connect_four",
        "game",
        "solver",
        "solver_set",
        "label_positions",
        "build_openings",
        "openings",
    )
)
Solver = solver_module.Solver
ILLEGAL = solver_module.ILLEGAL
SolverSet = solver_set_module.SolverSet


def winning_lines(discs: set, column: int, row: int) -> bool:
    """Whether a disc at (column, row) makes four in a row with `discs`."""
    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
        run = 1
        for sign in (1, -1):
            c, r = column + sign * dc, row + sign * dr
            while (c, r) in discs:
                run += 1
                c, r = c + sign * dc, r + sign * dr
        if run >= 4:
            return True
    return False


def board_of(position: str) -> tuple[set, set, list]:
    """Each side's discs, the player to move's first, and the columns' heights."""
    discs: tuple[set, set] = (set(), set())
    heights = [0] * 7
    for ply, c in enumerate(solver_module.actions(position)):
        discs[(len(position) - ply) % 2 == 1].add((c, heights[c]))
        heights[c] += 1
    return discs[0], discs[1], heights


def exact_scores(position: str) -> np.ndarray:
    """The solver's scores, by exhaustive search: only for positions near the end of a game."""
    mine, _, heights = board_of(position)
    ply = len(position)
    scores = np.full(7, ILLEGAL, dtype=np.int16)
    for c in range(7):
        if heights[c] == 6:
            continue
        if winning_lines(mine, c, heights[c]):
            scores[c] = (43 - ply) // 2
        elif ply + 1 == 42:
            scores[c] = 0
        else:
            scores[c] = -exact_scores(position + str(c + 1)).max()
    return scores


class OneMoveSolver:
    """Stands in for the solver: knows only the wins it can take at once, scoring the rest 0."""

    name = "solver"

    def __init__(self):
        self.calls = 0

    def analyse(self, positions):
        self.calls += len(positions)
        rows = []
        for position in positions:
            mine, _, heights = board_of(position)
            row = np.full(7, ILLEGAL, dtype=np.int16)
            for c in range(7):
                if heights[c] < 6:
                    row[c] = (
                        (43 - len(position)) // 2
                        if winning_lines(mine, c, heights[c])
                        else 0
                    )
            rows.append(row)
        return np.stack(rows).reshape(len(positions), 7)


def random_position(ply: int, seed: int) -> str:
    """A position of `ply` moves, no game ended, from a random game."""
    rng = np.random.default_rng(seed)
    solver = OneMoveSolver()
    while True:
        position = ""
        while len(position) < ply:
            row = solver.analyse([position])[0]
            column = rng.choice(np.flatnonzero(row != ILLEGAL))
            if row[column] > 0:
                break
            position += str(column + 1)
        if len(position) == ply:
            return position


def test_plies_to_end_is_how_long_the_game_lasts_under_perfect_play():
    def length(position):
        """The game's remaining moves, with the winner hurrying and the loser delaying."""
        scores = exact_scores(position)
        best = scores.max()
        column = int(np.argmax(scores))
        if solver_module.wins_now(int(best), len(position)):
            return 1
        if len(position) + 1 == 42:
            return 1
        return 1 + length(position + str(column + 1))

    checked = 0
    for seed in range(30):
        position = random_position(33 + seed % 3, seed)
        scores = exact_scores(position)
        ply = len(position)
        for column in np.flatnonzero(scores != ILLEGAL):
            after = position + str(column + 1)
            ends_now = solver_module.wins_now(int(scores[column]), ply) or ply == 41
            expected = 1 if ends_now else 1 + length(after)
            assert solver_module.plies_to_end(scores[column], ply) == expected
            checked += 1
    assert checked > 50


# a position's scores, 10 moves in, with the first column full
SCORES = np.array([ILLEGAL, 16, -16, 1, 0, -1, 1], dtype=np.int16)
PLY = 10


def test_levels_see_as_far_as_they_look():
    rng = np.random.default_rng(0)
    depths = solver_module.plies_to_end(SCORES, PLY)
    assert depths[1] == 43 - 2 * 16 - PLY == 1  # the win, at once
    assert depths[2] == 44 - 2 * 16 - PLY == 2  # the opponent wins straight after

    def moves(level):
        return {solver_module.choose_move(SCORES, PLY, level, rng) for _ in range(200)}

    # level 0 sees nothing: any legal move
    assert moves(0) == {1, 2, 3, 4, 5, 6}
    # level 1 takes the win
    assert moves(1) == {1}
    # with nothing to take, level 2 won't let the opponent win, and level 1 doesn't care
    no_win = np.where(SCORES == 16, 0, SCORES)
    level1 = {solver_module.choose_move(no_win, PLY, 1, rng) for _ in range(200)}
    level2 = {solver_module.choose_move(no_win, PLY, 2, rng) for _ in range(200)}
    assert 2 in level1 and 2 not in level2
    # perfect: the best score, at random between equal ones
    assert moves(solver_module.PERFECT_LEVEL) == {1}
    equal = np.where(SCORES == 16, 1, SCORES)
    assert {
        solver_module.choose_move(equal, PLY, solver_module.PERFECT_LEVEL, rng)
        for _ in range(100)
    } == {1, 3, 6}


def test_canonical_counts_mirror_images_and_move_orders_once():
    assert solver_module.canonical("1234") == solver_module.canonical("7654")
    assert solver_module.canonical("1234") == solver_module.canonical("3214")
    assert solver_module.canonical("1234") != solver_module.canonical("2134")


def fake_solver_dir(tmp_path: Path) -> Path:
    """A directory with a stand-in c4solver: says each position's length in every column, and
    writes each position it's asked to a log."""
    script = tmp_path / "c4solver"
    script.write_text(
        "#!/bin/sh\n"
        "while IFS= read -r line; do\n"
        f'  echo "$line" >> {tmp_path}/asked\n'
        "  n=${#line}\n"
        '  echo "$line $n $n $n $n $n $n -1000"\n'
        "done\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "7x6.book").write_text("")
    return tmp_path


def test_solver_needs_installing(tmp_path):
    with pytest.raises(ValueError, match="setup_solver.sh"):
        Solver(tmp_path)


def test_solver_answers_each_position_once_in_order(tmp_path):
    solver = Solver(fake_solver_dir(tmp_path), workers=2)
    scores = solver.analyse(["", "12", "", "345"])
    np.testing.assert_array_equal(scores, [[n] * 6 + [ILLEGAL] for n in (0, 2, 0, 3)])
    solver.analyse(["12"])
    solver.close()
    assert sorted((tmp_path / "asked").read_text().split("\n")) == ["", "", "12", "345"]
    assert solver.analyse([]).shape == (0, 7)


def test_solver_says_when_it_stops(tmp_path):
    script = tmp_path / "c4solver"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "7x6.book").write_text("")
    solver = Solver(tmp_path, workers=1)
    with pytest.raises(RuntimeError, match="stopped on position"):
        solver.analyse(["1"])


@pytest.mark.skipif(
    not Solver.installed(Solver.default_dir()), reason="the solver isn't installed"
)
def test_the_real_solver_agrees_with_the_environment():
    solver = Solver(workers=1)
    # the known values of the first move
    np.testing.assert_array_equal(solver.analyse([""])[0], [-2, -1, 0, 1, 0, -1, -2])
    step = jax.jit(game.env.step)
    rng = np.random.default_rng(0)
    for seed in range(5):
        state = game.env.init(jax.random.PRNGKey(seed))
        position = ""
        while not state.terminated:
            scores = solver.analyse([position])[0]
            np.testing.assert_array_equal(scores != ILLEGAL, state.legal_action_mask)
            # the left-right mirror, exactly
            mirrored = "".join(str(8 - int(c)) for c in position)
            np.testing.assert_array_equal(solver.analyse([mirrored])[0], scores[::-1])
            np.testing.assert_array_equal(
                solver_set_module.observation(position), state.observation
            )
            move = int(rng.choice(np.flatnonzero(state.legal_action_mask)))
            won = solver_module.wins_now(int(scores[move]), len(position))
            state = step(state, move)
            position += str(move + 1)
            assert bool(state.terminated) == (won or len(position) == 42)
    solver.close()


def test_observation_is_the_environments():
    step = jax.jit(game.env.step)
    rng = np.random.default_rng(1)
    state = game.env.init(jax.random.PRNGKey(0))
    position = ""
    while not state.terminated:
        np.testing.assert_array_equal(
            solver_set_module.observation(position), state.observation
        )
        move = int(rng.choice(np.flatnonzero(state.legal_action_mask)))
        state = step(state, move)
        position += str(move + 1)


def test_the_opening_list_is_playable():
    moves = openings.load_openings()
    assert moves.shape == (3407, 6)
    step = jax.jit(jax.vmap(game.env.step))
    states = jax.vmap(game.env.init)(jax.random.split(jax.random.PRNGKey(0), 3407))
    for ply in range(6):
        assert np.asarray(states.legal_action_mask)[
            np.arange(3407), moves[:, ply]
        ].all()
        states = step(states, moves[:, ply])
    assert not np.asarray(states.terminated).any()


def test_opening_init_fn_plays_openings_or_leaves_the_empty_board():
    moves = openings.load_openings()[:4]
    key = jax.random.PRNGKey(0)
    state, metadata = openings.make_opening_init_fn(moves, 0.0)(key)
    assert int(state._step_count) == 6 and int(metadata.step) == 6
    assert int(state.observation.sum()) == 6
    state, _ = openings.make_opening_init_fn(moves, 1.0)(key)
    assert int(state._step_count) == 0


def test_build_openings_takes_close_positions_once_each():
    class Fake(OneMoveSolver):
        def analyse(self, positions):
            scores = super().analyse(positions)
            # any position with a disc in the last column is lost by far
            return np.where(
                [["7" in p] * 7 for p in positions],
                np.where(scores == ILLEGAL, ILLEGAL, -9),
                scores,
            )

    built = build_openings.build_openings(Fake(), plies=2, max_score=2)
    assert built == sorted(built)
    assert all(len(p) == 2 for p in built)
    assert len({solver_module.canonical(p) for p in built}) == len(built)
    assert "44" in built and all("7" not in p for p in built)


def make_set() -> SolverSet:
    scores = np.array(
        [
            # the outcome depends on the move, early on
            [1, 2, -1, ILLEGAL, 0, 3, -2],
            # lost whatever the move, midway
            [-3, -1, -2, -3, -1, -4, -2],
            # won whatever the move, near the end
            [5, 6, 7, 8, 9, 1, 2],
        ],
        dtype=np.int16,
    )
    return SolverSet(
        positions=np.array(
            [
                "4",
                "1" * 3 + "2" * 3 + "3" * 3 + "5" * 5,
                "1" * 6 + "2" * 6 + "3" * 6 + "5" * 6 + "6" * 6 + "7",
            ]
        ),
        scores=scores,
        source=np.array([0, 0, 1], dtype=np.int8),
        source_names=["noise10", "noise100"],
    )


def test_solver_set_survives_saving(tmp_path):
    path = tmp_path / "set.npz"
    saved = make_set()
    saved.save(path)
    loaded = SolverSet.load(path)
    np.testing.assert_array_equal(loaded.scores, saved.scores)
    np.testing.assert_array_equal(loaded.positions, saved.positions)
    np.testing.assert_array_equal(loaded.source, saved.source)
    assert loaded.source_names == saved.source_names
    assert not list(tmp_path.glob("*.tmp*"))


def test_solver_set_groups_by_phase_and_what_the_move_decides():
    solver_set = make_set()
    groups = solver_set.groups()
    assert groups["opening"].tolist() == [True, False, False]
    assert groups["midgame"].tolist() == [False, True, False]
    assert groups["endgame"].tolist() == [False, False, True]
    assert groups["critical"].tolist() == [True, False, False]
    assert set(solver_set.groups(by_source=True)) > {"noise10", "noise100"}


def test_solver_set_scores_by_the_solver_or_by_outcomes():
    solver_set = make_set()
    by_score = solver_set.scored_positions()
    np.testing.assert_array_equal(by_score.scores[0, [0, 5]], [1, 3])
    by_outcome = solver_set.scored_positions(by_outcome=True)
    np.testing.assert_array_equal(by_outcome.scores[0], [1, 1, -1, ILLEGAL, 0, 1, -1])
    assert by_outcome.observations.shape == (3, 6, 7, 2)
    np.testing.assert_array_equal(
        by_outcome.legal[0], [True, True, True, False, True, True, True]
    )
    solver_set.describe()


def test_label_positions_builds_distinct_positions_with_a_choice():
    solver = OneMoveSolver()
    built = label_positions.build(solver, [0.5, 1.0], per_source=20, seed=0)
    assert len(built) == 40
    assert built.source_names == ["noise50", "noise100"]
    assert (built.source == 0).sum() == 20
    assert (built.legal.sum(axis=1) >= 2).all()
    keys = {solver_module.canonical(str(p)) for p in built.positions}
    assert len(keys) == 40
    # the same seed, the same set
    again = label_positions.build(OneMoveSolver(), [0.5, 1.0], per_source=20, seed=0)
    np.testing.assert_array_equal(again.positions, built.positions)


def test_label_positions_says_when_a_source_runs_dry():
    with pytest.raises(ValueError, match="positions to be found"):
        # with no noise and no ties, every game is the same one
        class AlwaysFirst(OneMoveSolver):
            def analyse(self, positions):
                rows = super().analyse(positions)
                return np.where(rows == ILLEGAL, ILLEGAL, np.arange(7)[None] * -1)

        label_positions.build(AlwaysFirst(), [0.0], per_source=1000, seed=0)
