"""Packing test episodes for a seaglass server, whose run pages draw them in the browser.

A training run packs an episode's raw arrays into seaglass `Arrays` of type
`episode_type(env)` (with a packer here, passed to a tester as its `episode_fn`); the server
stores them as they are, and the run's page plays them with turbozero's views (`views/`, whose
`episode.js` renders the type), so neither the training run nor the server draws anything.

When the test recorded its searches (`two_player_game`'s `record_search`), the episode also
carries what each player's search explored at each position, for the views' search explorer:
every node's statistics and, replayed through the environment here, its board. A player that
only scores moves, each with the line it expects after it (an engine's analysis), records them
as a `LineRecord`, a tree of those lines.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from seaglass import Arrays

from core.common import GameFrame
from core.evaluators.mcts.state import SearchRecord
from core.trees.tree import Tree
from core.types import EnvStepFn


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LineRecord(SearchRecord):
    """A `SearchRecord` of a search that only says what it expects after each move it scored
    (an engine's analysis): a line per move, its nodes each the child of the one before,
    without visits or priors.

    Attributes:
        depth: () how many moves ahead the search looked
        bound: (N) for the player to move at each node, as q: 0 if the node's value is
            exact, -1 if its value is at most q, +1 at least (a search can stop on a move
            once it knows the move is no better than another)
    """

    depth: jax.Array
    bound: jax.Array


def line_record(
    lines: Sequence[tuple[Sequence[int], float, int]] | None,
    depth: int = 0,
    *,
    num_actions: int,
    capacity: int,
) -> LineRecord:
    """The `LineRecord` of a search that scored the moves its `lines` start with, or of no
    search with `lines` None. Players alternate along a line, each move (a pass too) handing
    the turn to the other, so every node of a line has its value for the player to move at
    the root; the root's is the best line's.

    Args:
        lines: (line, value, bound) for each move: the moves it expects, that one first,
            their value in [-1, 1] for the player to move at the root, and whether that's
            exact (0), at most (-1) or at least (+1) their value
        depth: how many moves ahead the search looked
        num_actions: the environment's number of actions
        capacity: the record's nodes, at least the lines' moves plus the root
    """
    num_nodes = 0 if lines is None else 1 + sum(len(line) for line, _, _ in lines)
    assert num_nodes <= capacity, f"{num_nodes} nodes in a record of {capacity}"
    parents = np.full(capacity, Tree.NULL_INDEX, dtype=np.int32)
    actions = np.full(capacity, Tree.NULL_INDEX, dtype=np.int32)
    q = np.zeros(capacity, dtype=np.float32)
    bounds = np.zeros(capacity, dtype=np.int32)
    if lines:
        # the best line, an exact value before a bound on the same value
        _, q[0], bounds[0] = max(lines, key=lambda x: (x[1], x[2] == 0))
        node = 1
        for line, value, bound in lines:
            for k, a in enumerate(line):
                parents[node] = 0 if k == 0 else node - 1
                actions[node] = a
                # for the player to move at each node: the root's every other move
                sign = (-1) ** (k + 1)
                q[node], bounds[node] = value * sign, bound * sign
                node += 1
    return LineRecord(
        parents=jnp.asarray(parents),
        actions=jnp.asarray(actions),
        n=jnp.zeros(capacity, dtype=jnp.int32),
        q=jnp.asarray(q),
        prior=jnp.zeros(capacity, dtype=jnp.float32),
        terminated=jnp.zeros(capacity, dtype=bool),
        policy=jnp.zeros(num_actions, dtype=jnp.float32),
        num_nodes=jnp.asarray(num_nodes, dtype=jnp.int32),
        depth=jnp.asarray(depth, dtype=jnp.int32),
        bound=jnp.asarray(bounds),
    )


def _othello(x: Any) -> tuple[np.ndarray, np.ndarray]:
    # the board is +1 for the player to move and -1 for the other; turn 0 is black's
    to_move = np.asarray(x.turn)
    board = np.where(to_move[:, None] == 0, x.board, -x.board)
    return np.select([board > 0, board < 0], [1, 2], 0).reshape(-1, 8, 8), to_move


def _tic_tac_toe(x: Any) -> tuple[np.ndarray, np.ndarray]:
    # -1 for empty, else the colour (0 is X)
    return (np.asarray(x.board) + 1).reshape(-1, 3, 3), np.asarray(x.color)


def _connect_four(x: Any) -> tuple[np.ndarray, np.ndarray]:
    # rows from the top, -1 for empty, else the colour (0 moves first)
    return (np.asarray(x.board) + 1).reshape(-1, 6, 7), np.asarray(x.color)


def _hex(x: Any) -> tuple[np.ndarray, np.ndarray]:
    # a cell of the board is > 0 for the player to move's stones and < 0 for the other's; colour 0
    # moves first and connects the top and bottom, colour 1 the left and right
    to_move = np.asarray(x.step_count) % 2
    board = np.where(to_move[:, None] == 0, x.board, -x.board)
    size = round(board.shape[1] ** 0.5)
    return np.select([board > 0, board < 0], [1, 2], 0).reshape(-1, size, size), to_move


def episode_type(env_id: str) -> str:
    """The seaglass type of a pgx env's episodes, which picks the views' renderer for them."""
    return f"turbozero/{env_id}-episode"


# pgx env id -> the boards of a game's states, (frames, rows, cols) of 0 for empty, 1
# for a piece of the colour that moves first in the game's usual opening (black, X)
# and 2 for the other's, and the colour to move in each, 0 or 1
BOARDS: dict[str, Callable[[Any], tuple[np.ndarray, np.ndarray]]] = {
    "othello": _othello,
    "tic_tac_toe": _tic_tac_toe,
    "connect_four": _connect_four,
    "hex": _hex,
}


@partial(jax.jit, static_argnums=0)
def _replay(
    env_step_fn: EnvStepFn, roots: Any, paths: jax.Array, depths: jax.Array
) -> Any:
    """The states `depths` moves along `paths` (one action per move) from `roots`."""

    def replay(root, path, depth):
        def move(state, d):
            stepped, _ = env_step_fn(state, path[d])
            return jax.tree.map(
                lambda a, b: jnp.where(d < depth, a, b), stepped, state
            ), None

        return jax.lax.scan(move, root, jnp.arange(path.shape[0]))[0]

    return jax.vmap(replay)(roots, paths, depths)


def _bucket(n: int) -> int:
    """`n` rounded up to a power of two, so replays of similar size share a compilation."""
    return 1 << max(0, n - 1).bit_length()


def _paths(parents: np.ndarray, actions: np.ndarray) -> list[list[int]]:
    """Each node's actions from the root. A node can come before its parent in the tree (when it
    took an evicted node's slot), so each path is built from its parent's once that's done."""
    paths: list[list[int] | None] = [None] * len(parents)
    paths[Tree.ROOT_INDEX] = []

    def path(i: int) -> list[int]:
        if paths[i] is None:
            paths[i] = path(int(parents[i])) + [int(actions[i])]
        return paths[i]  # type: ignore[return-value]

    return [path(i) for i in range(len(parents))]


def _searches(
    env_id: str,
    env_state: Any,
    searches: tuple[Any, Any],
    num_moves: int,
    env_step_fn: EnvStepFn,
) -> dict[str, np.ndarray]:
    """Each player's record of each position played from (see `GameFrame.p1_search`), as the
    views' search explorer reads them: one search per (position, player) with a tree, its
    nodes in a flat list, each with its board replayed through `env_step_fn`.

    Args:
        env_id: the pgx env, which names its board (`BOARDS`)
        env_state: the episode's states, stacked along a leading time axis
        searches: the agent's and the opponent's records, stacked along the same axis
        num_moves: the moves played
        env_step_fn: the environment step function, which the searches used
    """
    found, roots, paths, offsets = [], [], [], [0]
    for t in range(num_moves):
        # the frame after a move holds what each player made of the position it was played from
        for player, records in enumerate(searches):
            if records is None:
                continue
            record = jax.tree.map(lambda x, t=t: np.asarray(x[t + 1]), records)
            k = int(record.num_nodes)
            if k == 0:
                continue
            found.append((t, player, record))
            paths += _paths(record.parents[:k], record.actions[:k])
            roots += [t] * k
            offsets.append(offsets[-1] + k)
    if not found:
        return {}
    num_nodes = offsets[-1]
    depth = max(map(len, paths))
    # padded to sizes shared with other episodes, so most tests don't compile a new replay
    size, depth = _bucket(num_nodes), _bucket(max(depth, 1))
    padded = np.zeros((size, depth), dtype=np.int32)
    depths = np.zeros(size, dtype=np.int32)
    for i, path in enumerate(paths):
        padded[i, : len(path)] = path
        depths[i] = len(path)
    roots = np.pad(np.array(roots), (0, size - num_nodes))
    states = _replay(
        env_step_fn, jax.tree.map(lambda x: x[roots], env_state), padded, depths
    )
    states = jax.device_get(jax.tree.map(lambda x: x[:num_nodes], states))
    board, to_move = BOARDS[env_id](states._x)
    player = np.asarray(states.current_player)

    sizes = np.diff(offsets)
    stats = {
        name: np.concatenate(
            [getattr(record, name)[:k] for (_, _, record), k in zip(found, sizes)]
        )
        for name in ("parents", "actions", "n", "q", "prior", "terminated")
    }
    # each node's value for the player who chose it, its parent's player to move (the root's,
    # for the player to move at the root): q is for the player to move at the node
    parent = stats["parents"] + np.repeat(offsets[:-1], sizes)
    chooser = np.where(stats["parents"] < 0, player, player[np.maximum(parent, 0)])
    value = np.where(chooser == player, stats["q"], -stats["q"])
    # a `LineRecord`'s bounds on those values, the same way round; a tree's are exact
    bound = np.concatenate(
        [
            getattr(record, "bound", np.zeros(k, dtype=np.int32))[:k]
            for (_, _, record), k in zip(found, sizes)
        ]
    )
    bound = np.where(chooser == player, bound, -bound)
    return {
        "search_frame": np.array([t for t, _, _ in found]),
        # 0 for the trained agent's, 1 for its opponent's
        "search_player": np.array([p for _, p, _ in found]),
        "search_offset": np.array(offsets),
        # the root's policy, over every action
        "search_policy": np.stack([record.policy for _, _, record in found]),
        # how many moves ahead a `LineRecord`'s search looked; -1 for a tree's
        "search_depth": np.array(
            [int(getattr(record, "depth", -1)) for _, _, record in found]
        ),
        # local to the node's search: -1 for the root
        "node_parent": stats["parents"],
        "node_action": stats["actions"],
        "node_visits": stats["n"],
        "node_value": value,
        # 0 if node_value is exact, -1 if it's at most that, +1 at least
        "node_bound": bound,
        "node_prior": stats["prior"],
        "node_terminal": stats["terminated"],
        "node_board": board.astype(np.uint8),
        "node_to_move": to_move,
        "node_legal": np.asarray(states.legal_action_mask),
    }


def with_opening(frames: GameFrame, states: Sequence[Any]) -> GameFrame:
    """`frames` (`GameFrame`s stacked along a leading time axis) after the positions of an
    opening that led to their first: `states`, each the one before the next, the last one
    move before `frames`' first. Nobody searched the opening's positions.
    """
    num = len(states)
    opening = jax.tree.map(lambda *xs: np.stack(xs), *states)

    def before(x, opening_x=None):
        x = np.asarray(x)
        head = (
            np.zeros((num, *x.shape[1:]), x.dtype) if opening_x is None else opening_x
        )
        return np.concatenate([head, x])

    return GameFrame(
        env_state=jax.tree.map(before, frames.env_state, opening),
        p1_value_estimate=jnp.asarray(before(frames.p1_value_estimate)),
        p2_value_estimate=jnp.asarray(before(frames.p2_value_estimate)),
        completed=jnp.asarray(before(frames.completed)),
        outcomes=jnp.asarray(before(frames.outcomes)),
        # frame t + 1 holds the record of position t; no records hold zeros
        p1_search=None
        if frames.p1_search is None
        else jax.tree.map(before, frames.p1_search),
        p2_search=None
        if frames.p2_search is None
        else jax.tree.map(before, frames.p2_search),
    )


def pgx_two_player_episode() -> Callable[..., Arrays]:
    """An `episode_fn` for testers of two-player pgx board games (those in `BOARDS`):
    packs the first test episode for the views to play, as `Arrays` of the game's
    `episode_type`. Whoever packs it can name the trained agent's opponent in its attrs,
    `{"opponent": "edax level 8"}`.

    Returns:
        Callable: `episode_fn(frames, p_ids, env_step_fn, start=0) -> Arrays`, see
        `BaseTester`; `start` is the frame the players started playing from, after an
        opening played for them (see `with_opening`), where the views show the game from
    """

    def episode_fn(frames, p_ids, env_step_fn, start: int = 0) -> Arrays:
        env_state = frames.env_state
        env_id = str(env_state.env_id)
        if env_id not in BOARDS:
            known = ", ".join(sorted(BOARDS))
            raise ValueError(f"no board for pgx env {env_id!r} (known: {known})")
        board, to_move = BOARDS[env_id](env_state._x)
        completed = np.asarray(frames.completed, dtype=bool)
        # frames after the game ended repeat its last state
        num_frames = (
            int(np.argmax(completed)) + 1 if completed.any() else len(completed)
        )
        agent, opponent = (int(p) for p in p_ids)
        # the trained agent's colour: the one to move first, if it moves first
        first = int(to_move[0])
        moves_first = int(np.asarray(env_state.current_player)[0]) == agent
        outcomes = np.asarray(frames.outcomes)[num_frames - 1]
        searches: dict[str, Any] = _searches(
            env_id,
            env_state,
            (frames.p1_search, frames.p2_search),
            num_frames - 1,
            env_step_fn,
        )
        return Arrays(
            episode_type(env_id),
            board=board[:num_frames],
            to_move=to_move[:num_frames],
            agent=first if moves_first else 1 - first,
            # each one's value estimate of the position, from its own side
            values=np.stack(
                [frames.p1_value_estimate, frames.p2_value_estimate], axis=-1
            )[:num_frames],
            completed=completed.any(),
            # the trained agent's result once the game is over (or cut off): 1 for a
            # win, -1 for a loss, 0 for a draw
            outcome=np.sign(outcomes[agent] - outcomes[opponent]),
            start=np.int32(start),
            **searches,
        )

    return episode_fn
