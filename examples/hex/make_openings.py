"""Builds an opening list (see openings.py): every single first stone, and two-stone positions
MoHex judges close to even.

    examples/hex/setup_mohex.sh                  # once
    taskset -c 16-23 uv run examples/hex/make_openings.py --size 7 --workers 8

Each two-stone position, drawn at random, gets a search of --games playouts. It's kept if the
player to move's best move is worth from --low to --high to MoHex (win rates, 0.5 is even),
and none of its moves has been proven to win or lose, which it often has early on a small board.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from engines import MoHex, parse_values
from game import HexGame
from openings import openings_path


def even(game: HexGame, position: tuple[int, int], games: int, low: float, high: float):
    """Whether MoHex judges the position of black's stone and white's stone close to even."""
    engine = MoHex(MoHex.default_dir(), game, games)
    try:
        engine.command("param_game allow_swap 0")
        engine.command(f"play b {game.cell_name(position[0])}")
        engine.command(f"play w {game.cell_name(position[1])}")
        move = engine.command("reg_genmove b").strip().lower()
        try:
            values = parse_values(engine.command("mohex-values"), game)
        except RuntimeError:
            # it answered without searching: it knows who wins
            return False
        played = [v for v in values if game.cell_name(v.line[0]) == move]
        return (
            bool(played)
            and not any(v.proven for v in values)
            and low <= (1 + played[0].score) / 2 <= high
        )
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description="Builds an opening list for Hex.")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument(
        "--count", type=int, default=500, help="two-stone openings to find"
    )
    parser.add_argument("--games", type=int, default=20000, help="playouts a position")
    parser.add_argument("--low", type=float, default=0.35)
    parser.add_argument("--high", type=float, default=0.65)
    parser.add_argument("--workers", type=int, default=8, help="MoHex processes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    game = HexGame(args.size)
    rng = np.random.default_rng(args.seed)
    cells = args.size * args.size
    # up to the board's half-turn, which leaves a position's value as it is
    half_turn = lambda cell: cells - 1 - cell
    candidates = [
        (b, w)
        for b in range(cells)
        for w in range(cells)
        if b != w and (b, w) <= (half_turn(b), half_turn(w))
    ]
    rng.shuffle(candidates)
    found = []
    with ThreadPoolExecutor(args.workers) as pool:
        for start in range(0, len(candidates), args.workers * 8):
            batch = candidates[start : start + args.workers * 8]
            kept = pool.map(
                lambda p: even(game, p, args.games, args.low, args.high), batch
            )
            found += [p for p, keep in zip(batch, kept) if keep]
            print(f"{start + len(batch)} tried, {len(found)} even", flush=True)
            if len(found) >= args.count:
                break
    found = found[: args.count]
    lines = [game.cell_name(cell) for cell in range(cells)] + [
        game.cell_name(b) + game.cell_name(w) for b, w in found
    ]
    openings_path(args.size).write_text("\n".join(lines) + "\n")
    print(f"{openings_path(args.size)}: {len(lines)} openings")


if __name__ == "__main__":
    main()
