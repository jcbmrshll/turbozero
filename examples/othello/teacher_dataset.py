"""A dataset of whole self-play games, from the moves runs kept with train.py's --log-moves (see
core/memory/move_log.py), labelled with exact endgame results, for students and architecture
experiments to train on from disk (distill.py's --data takes its directory).

    uv run examples/othello/teacher_dataset.py build --moves RUN/moves [RUN2/moves ...] \\
        --out DATASET --solve-empties 20 --threads 8

`build` puts each game back together from its moves, drops those the runs ended in the middle of
(or that hit the step limit), and labels every position with the game's result for its player to
move, exactly: every position with --solve-empties or fewer empty squares is solved (win, draw or
loss) and has its own result, and every position before a game's first such position, its solve
point, has the solve point's. A game that ended before any has the result it ended with.

The dataset's directory holds one .npy per column, a row per position, game by game, each game's
in the order played (see `COLUMNS`), and:
- games.npz: per game, its first row (`start`), its rows (`length`), and where it came from
  (`run`, the index of its --moves directory; `env`; `env_game`, the environment's game number)
- meta.json: what it was built from and how, and counts

About a few percent of games (--heldout) are held out, whole games chosen at random: their rows
have `heldout` set. The same XOT opening starts many games, so early positions of held-out games
also appear in the rest, as they do between a real run's games.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from exact_labels import bitboards

from core.memory.move_log import (
    ENDED,
    GOES_ON,
    STOPPED,
    TRUNCATED,
    read_meta,
    read_moves,
    unpack,
)
from core.memory.replay_memory import BaseExperience

# the dataset's columns, a row per position: what each is (see `build`)
COLUMNS = {
    "me": "uint64, the player to move's discs (bit i is square i, a1 = 0 to h8 = 63)",
    "opp": "uint64, the opponent's discs",
    "legal": "uint8 (9,), the legal actions, 65 bits packed little-endian (see `unpack_legal`)",
    "policy": "float16 (65,), the search's policy target (visit distribution) as self-play "
    "stored it, root noise and the reused tree's visits included",
    "value": "float32, the search's root value, for the player to move",
    "action": "int8, the move played, -1 if the game stopped before it",
    "player": "int8, the player to move",
    "full": "bool, whether it was a full search (a training position), not a fast one",
    "ply": "uint8, the moves played since the standard start (XOT openings start at 8)",
    "empties": "uint8, empty squares",
    "game": "int32, the game's index (see games.npz)",
    "outcome": "int8, the game's exact result for the player to move: 1 win, 0 draw, -1 loss",
    "solved": "bool, whether `outcome` is the position's own solve (empties <= the solve "
    "empties), rather than its game's solve point's",
    "heldout": "bool, whether its game is held out",
}


def unpack_legal(legal: np.ndarray) -> np.ndarray:
    """(n, 65) bool, from the `legal` column."""
    return np.unpackbits(legal, axis=1, count=65, bitorder="little").astype(bool)


def observations(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """(n, 8, 8, 2) bool, the network's input (pgx's observation), from the `me` and `opp`
    columns."""

    def squares(bits: np.ndarray) -> np.ndarray:
        as_bytes = np.ascontiguousarray(bits, dtype="<u8").view(np.uint8)
        return np.unpackbits(as_bytes.reshape(-1, 8), axis=1, bitorder="little")

    return (
        np.stack([squares(me), squares(opp)], axis=-1).astype(bool).reshape(-1, 8, 8, 2)
    )


def column(directory: str | Path, name: str) -> np.ndarray:
    """A column of a dataset, mapped rather than read."""
    return np.load(Path(directory) / f"{name}.npy", mmap_mode="r")


def select(
    directory: str | Path,
    heldout: bool,
    full_only: bool = True,
    min_empties: int = 0,
    max_empties: int = 64,
) -> np.ndarray:
    """The rows of a dataset's training games (or held-out ones), of full searches only (or all),
    with from `min_empties` to `max_empties` empty squares."""
    keep = np.asarray(column(directory, "heldout")) == heldout
    if full_only:
        keep &= np.asarray(column(directory, "full"))
    empties = np.asarray(column(directory, "empties"))
    keep &= (empties >= min_empties) & (empties <= max_empties)
    return np.nonzero(keep)[0]


def load_rows(directory: str | Path, rows: np.ndarray) -> BaseExperience:
    """A dataset's rows as positions to train on, as a replay buffer holds them: the reward is
    the exact outcome (see `COLUMNS`), the search value the root's."""

    def take(name: str) -> np.ndarray:
        return np.asarray(column(directory, name)[rows])

    player = take("player").astype(np.int32)
    outcome = take("outcome").astype(np.float32)
    reward = np.empty((len(rows), 2), np.float32)
    reward[np.arange(len(rows)), player] = outcome
    reward[np.arange(len(rows)), 1 - player] = -outcome
    fields: dict[str, Any] = {
        "observation_nn": observations(take("me"), take("opp")),
        "policy_weights": take("policy").astype(np.float32),
        "policy_mask": unpack_legal(take("legal")),
        "reward": reward,
        "cur_player_id": player,
        "search_value": take("value").astype(np.float32),
    }
    return BaseExperience(**fields)


def solve_wld(me: np.ndarray, opp: np.ndarray, threads: int) -> np.ndarray:
    """Each position's exact result for the player to move, -1, 0 or 1 (int8)."""
    import endgame

    margins, _ = endgame.solve_batch(me, opp, threads=threads, exact=False)
    return np.sign(margins).astype(np.int8)


def label_games(
    starts: np.ndarray,
    lengths: np.ndarray,
    player: np.ndarray,
    end: np.ndarray,
    result: np.ndarray,
    empties: np.ndarray,
    solves: np.ndarray,
    solve_empties: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Each row's outcome for its player to move, and whether it's its own solve.

    Args:
        starts, lengths: each game's rows, which are contiguous and in order
        player, end, result, empties: the rows' (see `core.memory.move_log.LoggedMove`)
        solves: each row's own exact result, wherever empties <= `solve_empties` (else unused)
        solve_empties: the empties at which solves start

    Returns:
        (outcome int8, solved bool) per row
    """
    solved = empties <= solve_empties
    outcome = np.where(solved, solves, 0).astype(np.int8)
    game_of_row = np.repeat(np.arange(len(starts)), lengths)
    # each game's reference row: its solve point, or its last row if it never reached one
    last = starts + lengths - 1
    first_solved = np.full(len(starts), -1)
    rows = np.nonzero(solved)[0]
    # the first solved row of each game (rows are game by game, in order)
    games_solved, first = np.unique(game_of_row[rows], return_index=True)
    first_solved[games_solved] = rows[first]
    reference = np.where(first_solved >= 0, first_solved, last)
    reached = first_solved >= 0
    if (~reached & (end[last] != ENDED)).any():
        raise ValueError(
            "a game stopped before its solve point: is --solve-empties right?"
        )
    ref_result = np.where(reached, solves[reference], result[last]).astype(np.int8)
    ref_player = player[reference]
    # the game's result for each row's player: the reference's, negated for the other player
    sign = np.where(player == ref_player[game_of_row], 1, -1).astype(np.int8)
    from_game = ref_result[game_of_row] * sign
    return np.where(solved, outcome, from_game).astype(np.int8), solved


def build(
    moves_dirs: list[str],
    out: str,
    solve_empties: int,
    heldout: float,
    seed: int,
    threads: int,
) -> dict:
    """Builds the dataset (see the module's docstring) in `out`, a new directory."""
    out_path = Path(out)
    if out_path.exists() and any(out_path.iterdir()):
        raise ValueError(f"{out} isn't empty")
    out_path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    parts = []
    for run, directory in enumerate(moves_dirs):
        rows = read_moves(directory)
        meta = read_meta(directory)
        parts.append((run, rows, meta))
        print(f"{directory}: {len(rows)} moves", flush=True)
    run = np.concatenate([np.full(len(r), i, np.int32) for i, r, _ in parts])
    rows = np.concatenate([r for _, r, _ in parts])
    observation = np.concatenate([unpack(r, "observation", m) for _, r, m in parts])
    del parts

    # games: rows by run, environment, game, ply
    order = np.lexsort((rows["ply"], rows["game"], rows["env"], run))
    rows, run, observation = rows[order], run[order], observation[order]
    key = np.stack([run, rows["env"], rows["game"]], axis=1)
    new_game = np.r_[True, (key[1:] != key[:-1]).any(axis=1)]
    starts = np.nonzero(new_game)[0]
    lengths = np.diff(np.r_[starts, len(rows)])
    last = starts + lengths - 1
    if (np.diff(rows["ply"]) != 1)[~new_game[1:]].any():
        raise ValueError("a game's moves aren't consecutive: are moves missing?")
    finished = (rows["end"][last] == ENDED) | (rows["end"][last] == STOPPED)
    truncated = int((rows["end"][last] == TRUNCATED).sum())
    if (rows["end"][np.setdiff1d(np.arange(len(rows)), last)] != GOES_ON).any():
        raise ValueError("a game goes on after its end")
    keep = np.repeat(finished, lengths)
    dropped_rows = int((~keep).sum())
    rows, run, observation = rows[keep], run[keep], observation[keep]
    starts = np.r_[0, np.cumsum(lengths[finished])[:-1]]
    game_runs = run[starts]
    lengths = lengths[finished]
    print(
        f"{len(starts)} finished games, {len(rows)} positions; dropped "
        f"{int((~finished).sum())} games ({dropped_rows} moves) unfinished or truncated",
        flush=True,
    )

    me, opp = bitboards(observation)
    del observation
    empties = (64 - np.bitwise_count(me | opp)).astype(np.uint8)
    to_solve = np.nonzero(empties <= solve_empties)[0]
    print(f"solving {len(to_solve)} positions on {threads} threads", flush=True)
    solve_started = time.time()
    solves = np.zeros(len(rows), np.int8)
    solves[to_solve] = solve_wld(me[to_solve], opp[to_solve], threads)
    solve_seconds = time.time() - solve_started
    outcome, solved = label_games(
        starts,
        lengths,
        rows["player"],
        rows["end"],
        rows["result"],
        empties,
        solves,
        solve_empties,
    )
    held = np.random.default_rng(seed).random(len(starts)) < heldout
    game = np.repeat(np.arange(len(starts), dtype=np.int32), lengths)
    columns = {
        "me": me,
        "opp": opp,
        "legal": rows["legal"],
        "policy": rows["policy"].astype(np.float16),
        "value": rows["value"].astype(np.float32),
        "action": rows["action"].astype(np.int8),
        "player": rows["player"].astype(np.int8),
        "full": rows["full"],
        "ply": rows["ply"].astype(np.uint8),
        "empties": empties,
        "game": game,
        "outcome": outcome,
        "solved": solved,
        "heldout": held[game],
    }
    assert set(columns) == set(COLUMNS)
    for name, x in columns.items():
        np.save(out_path / f"{name}.npy", np.ascontiguousarray(x))
    np.savez(
        out_path / "games.npz",
        start=starts,
        length=lengths,
        run=game_runs,
        env=rows["env"][starts],
        env_game=rows["game"][starts],
    )
    full = rows["full"]
    meta = {
        "moves": [os.path.abspath(d) for d in moves_dirs],
        "solve_empties": solve_empties,
        "heldout_fraction": heldout,
        "seed": seed,
        "games": len(starts),
        "heldout_games": int(held.sum()),
        "positions": len(rows),
        "full_positions": int(full.sum()),
        "heldout_full_positions": int((full & held[game]).sum()),
        "solved_positions": int(solved.sum()),
        "dropped_games": int((~finished).sum()),
        "truncated_games": truncated,
        "draw_fraction": float((outcome[starts] == 0).mean()),
        "solve_seconds": solve_seconds,
        "build_seconds": time.time() - started,
        "columns": COLUMNS,
    }
    (out_path / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser(
        "build", help="a dataset from runs' --log-moves directories"
    )
    b.add_argument("--moves", nargs="+", required=True, help="--log-moves directories")
    b.add_argument("--out", required=True, help="the dataset's directory, new")
    b.add_argument(
        "--solve-empties",
        type=int,
        default=20,
        help="positions with this many empty squares or fewer are solved exactly (the runs' "
        "--solve-empties)",
    )
    b.add_argument(
        "--heldout", type=float, default=0.03, help="the share of games held out"
    )
    b.add_argument("--seed", type=int, default=0, help="the held-out games' seed")
    b.add_argument("--threads", type=int, default=8, help="the solver's threads")
    args = parser.parse_args()
    if args.command == "build":
        meta = build(
            args.moves,
            args.out,
            args.solve_empties,
            args.heldout,
            args.seed,
            args.threads,
        )
        print(json.dumps({k: v for k, v in meta.items() if k != "columns"}, indent=1))


if __name__ == "__main__":
    main()
