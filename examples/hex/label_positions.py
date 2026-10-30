"""Builds an engine-labelled set (see engine_set.py): positions from MoHex's games against itself
from the openings, with every legal move scored by MoHex.

    examples/hex/setup_mohex.sh                       # once
    taskset -c 16-23 uv run examples/hex/label_positions.py --size 7 --solve --workers 8
    taskset -c 16-23 uv run examples/hex/label_positions.py --size 11 --level 100000 --workers 8

The games are at --game-level playouts a move, with a --random-moves share of the moves at random,
so that they reach positions of all kinds, not just good ones; each gives up to --per-game
positions, with at least two legal moves.

With --solve, MoHex's solver finds the moves that win, in seconds a position at 7 x 7: those
score 1, and the rest -1. A move it doesn't consider, inferior or dead, also scores -1, though it
may win where a better one does. With --level, its search gives each move its value after that
many playouts, at most that of the move the search would play, whose value rests on the most
playouts; a move no playout went through gets the lowest value of the others.
"""

import argparse
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from engine_set import ILLEGAL, EngineSet, default_path
from engines import MoHex, parse_values
from game import HexGame
from openings import NO_MOVE, load_openings


def observation(game: HexGame, moves: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """A position's observation and legal actions, after `moves` (none a swap)."""
    n = game.size
    board = np.zeros((2, n * n), bool)
    for ply, move in enumerate(moves):
        board[ply % 2, move] = True
    colour = len(moves) % 2
    legal = np.append(~board.any(axis=0), len(moves) == 1)
    obs = np.zeros((n, n, 4), bool)
    obs[..., 0] = board[colour].reshape(n, n)
    obs[..., 1] = board[1 - colour].reshape(n, n)
    obs[..., 2] = colour == 1
    obs[..., 3] = len(moves) == 1
    return obs, legal


def self_play_game(
    game: HexGame,
    level: int,
    opening: np.ndarray,
    random_moves: float,
    rng: np.random.Generator,
    seed: int,
) -> list[list[int]]:
    """The positions of a game of MoHex against itself from `opening`, each as its moves: every one
    after the opening and with at least two legal moves, the swap rule off. Not those with a single
    stone, where the swap is legal, which MoHex can't score here."""
    engine = MoHex(MoHex.default_dir(), game, level, seed=seed)
    try:
        engine.command("param_game allow_swap 0")
        moves = [int(m) for m in opening if m != NO_MOVE]
        for ply, move in enumerate(moves):
            engine.play("bw"[ply % 2], move)
        positions = []
        while not engine.winner():
            positions.append(list(moves))
            colour = "bw"[len(moves) % 2]
            if rng.random() < random_moves:
                empty = np.setdiff1d(np.arange(game.swap), moves)
                move = int(rng.choice(empty))
                engine.play(colour, move)
            else:
                move = engine.genmove(colour)
            moves.append(move)
    finally:
        engine.close()
    return [p for p in positions if len(p) >= 2 and game.swap - len(p) >= 2]


def position_engine(game: HexGame, level: int, moves: list[int]) -> MoHex:
    """MoHex at `level` on the position after `moves`, with its solvers' tables as they start."""
    engine = MoHex(MoHex.default_dir(), game, level)
    engine.command("param_game allow_swap 0")
    for ply, move in enumerate(moves):
        engine.play("bw"[ply % 2], move)
    return engine


def solved_scores(game: HexGame, moves: list[int]) -> np.ndarray:
    """The legal moves of a position, each 1 if MoHex's solver finds it wins, else -1."""
    engine = position_engine(game, 1, moves)
    try:
        engine.command("param_dfpn tt_size 2097152")
        reply = engine.command(f"dfpn-solver-find-winning {'bw'[len(moves) % 2]}")
    finally:
        engine.close()
    _, legal = observation(game, moves)
    scores = np.where(legal, -1.0, ILLEGAL).astype(np.float32)
    for name in reply.split():
        scores[game.action(name)] = 1.0
    return scores


def searched_scores(game: HexGame, level: int, moves: list[int]) -> np.ndarray:
    """The legal moves of a position, each with its value after `level` playouts of MoHex's
    search: see the module's docstring."""
    engine = position_engine(game, level, moves)
    try:
        colour = "bw"[len(moves) % 2]
        played = game.action(engine.command(f"reg_genmove {colour}").strip().lower())
        try:
            values = parse_values(engine.command("mohex-values"), game)
        except RuntimeError:
            # it answered without searching: the position is decided
            values = []
    finally:
        engine.close()
    _, legal = observation(game, moves)
    if not any(v.line[0] == played for v in values):
        # it knows the move it plays wins, or that all of them lose. Scores of a decided position
        # say only that: it might equally have played another
        scores = np.where(legal, -1.0, ILLEGAL).astype(np.float32)
        scores[played] = 1.0
        return scores
    top = next(v.score for v in values if v.line[0] == played)
    lowest = min(v.score for v in values)
    scores = np.where(legal, lowest, ILLEGAL).astype(np.float32)
    for v in values:
        scores[v.line[0]] = min(v.score, top)
    return scores


def build(
    game: HexGame,
    count: int,
    score: Callable[[list[int]], np.ndarray],
    games: Callable[[int], list[list[int]]],
    workers: int,
    per_game: int,
    rng: np.random.Generator,
    progress: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`count` distinct positions from `games(i)`, each scored by `score`.

    Returns:
        tuple: obs, legal, scores, stones, as `EngineSet` holds them
    """
    positions: dict[bytes, list[int]] = {}
    with ThreadPoolExecutor(workers) as pool:
        start = 0
        while len(positions) < count:
            for played in pool.map(games, range(start, start + workers)):
                picks = rng.permutation(len(played))[:per_game]
                for i in picks:
                    obs, _ = observation(game, played[i])
                    positions.setdefault(obs.tobytes(), played[i])
            start += workers
        chosen = list(positions.values())[:count]
        obs, legal = zip(*(observation(game, moves) for moves in chosen))
        scores = []
        for done, s in enumerate(pool.map(score, chosen)):
            scores.append(s)
            if progress is not None:
                progress(done + 1)
    stones = np.array([len(moves) for moves in chosen])
    return np.stack(obs), np.stack(legal), np.stack(scores), stones


def main():
    parser = argparse.ArgumentParser(
        description="Builds an engine-labelled set of Hex positions."
    )
    parser.add_argument("--size", type=int, required=True)
    how = parser.add_mutually_exclusive_group(required=True)
    how.add_argument(
        "--solve", action="store_true", help="score moves with MoHex's solver"
    )
    how.add_argument(
        "--level",
        type=int,
        help="score moves with MoHex's search of this many playouts",
    )
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--per-game", type=int, default=4)
    parser.add_argument("--game-level", type=int, default=200)
    parser.add_argument("--random-moves", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8, help="MoHex processes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    game = HexGame(args.size)
    openings = load_openings(game)
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()

    def progress(done: int) -> None:
        if done % 100 == 0 or done == args.positions:
            elapsed = time.perf_counter() - start
            print(f"{done}/{args.positions} scored in {elapsed:.0f} s", flush=True)

    def games(i: int) -> list[list[int]]:
        game_rng = np.random.default_rng([args.seed, i])
        opening = openings[game_rng.integers(len(openings))]
        return self_play_game(
            game, args.game_level, opening, args.random_moves, game_rng, i + 1
        )

    score = (
        (lambda moves: solved_scores(game, moves))
        if args.solve
        else (lambda moves: searched_scores(game, args.level, moves))
    )
    obs, legal, scores, stones = build(
        game, args.positions, score, games, args.workers, args.per_game, rng, progress
    )
    engine_set = EngineSet(obs, legal, scores, stones, "mohex", args.level or 0)
    out = args.out or default_path(args.size)
    engine_set.save(out)
    print(f"{out}: {len(engine_set)} positions", flush=True)


if __name__ == "__main__":
    main()
