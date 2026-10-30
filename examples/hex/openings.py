"""Openings for test games and matches: the first moves, played for both players, so that
deterministic players give varied games, as XOT openings do in Othello.

Hex has no draws and the first player wins with perfect play, so a match has to be fair by its
construction: each opening is played twice, with the colours swapped. For games to say anything,
the opening shouldn't already decide them either. A list has two kinds of opening:

- every single first stone, after which the player to move has the swap, as in a real game: it
  chooses the better side, so these are as even as that player's judgement
- two stones, one of each colour, whose position MoHex judges close to even (see
  `make_openings.py`)

The lists, `openings-<size>.txt`, are one opening a line in cell names, e.g. "d4" or "d4b2".
"""

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from game import HexGame, metadata

from core.monitoring.episodes import with_opening

NO_MOVE = -1


def openings_path(size: int) -> Path:
    return Path(__file__).parent / f"openings-{size}.txt"


def parse_opening(line: str, game: HexGame) -> list[int]:
    """An opening in cell names as actions, e.g. "d4b2" -> [d4's, b2's]."""
    names = []
    for char in line.strip():
        if char.isalpha():
            names.append(char)
        else:
            names[-1] += char
    return [game.action(name) for name in names]


def load_openings(game: HexGame, path: str | Path | None = None) -> np.ndarray:
    """The openings as actions, one row per opening, padded with NO_MOVE.

    Args:
        game: the game, at the openings' size
        path: (optional) an opening list; by default the one for the board size
    """
    lines = Path(path or openings_path(game.size)).read_text().split()
    openings = [parse_opening(line, game) for line in lines]
    longest = max(map(len, openings))
    return np.array(
        [o + [NO_MOVE] * (longest - len(o)) for o in openings], dtype=np.int32
    )


def make_opening_init_fn(
    game: HexGame, openings: np.ndarray, standard_start_fraction: float
):
    """An env init fn (see HexGame.init_fn) that starts `standard_start_fraction` of the episodes
    from the empty board, and the rest from a random one of `openings`, already played."""
    moves = jnp.asarray(openings)

    def init_fn(key):
        init_key, pick_key, standard_key = jax.random.split(key, 3)
        state = game.env.init(init_key)
        opening = moves[jax.random.randint(pick_key, (), 0, len(moves))]
        from_opening = jax.random.uniform(standard_key) >= standard_start_fraction

        def play(ply, state):
            # play the opening's moves, or leave the empty board as it is
            return jax.tree.map(
                lambda new, old: jnp.where(
                    from_opening & (opening[ply] != NO_MOVE), new, old
                ),
                game.env.step(state, opening[ply]),
                state,
            )

        state = jax.lax.fori_loop(0, moves.shape[1], play, state)
        return state, metadata(state)

    return init_fn


def opening_episode_fn(game: HexGame, episode_fn):
    """An `episode_fn` (see `BaseTester`) for games started from an opening by an init fn from
    `make_opening_init_fn`, already played: packs the game with the opening's moves before it, by
    `episode_fn` (`core.monitoring.episodes.pgx_two_player_episode`'s), and the run's page showing
    it from the position the players started from.

    The game's first position gives the opening's stones: none have been captured or reflected, as
    the openings end before a swap can have happened. A game from the empty board is packed as
    it is."""
    step = jax.jit(game.env.step)

    def packed(frames, p_ids, env_step_fn):
        first = jax.tree.map(lambda x: np.asarray(x[0]), frames.env_state)
        plies = int(first._step_count)
        if plies == 0:
            return episode_fn(frames, p_ids, env_step_fn)
        # the first stone is the opponent's on a first position after one move, and the player
        # to move's after two
        board = first._x.board
        stones = np.flatnonzero(board < 0 if plies == 1 else board > 0)
        moves = list(stones)
        if plies == 2:
            moves += list(np.flatnonzero(board < 0))
        # replayed from the empty board with the game's player ids: colour 0 moves first
        state: Any = game.env.init(jax.random.PRNGKey(0))
        colour0 = (
            int(first.current_player)
            if plies % 2 == 0
            else 1 - int(first.current_player)
        )
        state = dataclasses.replace(
            state,
            current_player=jnp.int32(colour0),
            _player_order=jnp.int32([colour0, 1 - colour0]),
        )
        states = []
        for move in moves:
            states.append(jax.device_get(state))
            state = step(state, int(move))
        return episode_fn(
            with_opening(frames, states), p_ids, env_step_fn, start=len(states)
        )

    return packed
