"""The positions of the Elo league's games (see core.league) as a corpus to train networks on
without self-play: every position after its game's XOT opening, with the game, its opening, the
position's ply and the game's outcome, split by opening into positions to train on and held-out
ones, from openings no network trained on them has seen, to evaluate on. A teacher network then
labels every position with its policy and value, once.

    uv run examples/othello/league_corpus.py build --root ~/.cache/turbozero/othello/league \\
        --out corpus
    uv run --extra cuda examples/othello/league_corpus.py label --corpus corpus \\
        --teacher 82.eqx --teacher-net resnet:12x144,pool_every=2,heads=pooled,pool_channels=32

`distill.py train --corpus corpus` trains a network on it, and train.py's --seed-buffer starts
self-play's replay memory with positions of it (with --exclude-openings
corpus/heldout_openings.npy, self-play keeps off the held-out openings too).

A corpus is a directory of .npy files, a row per position: `obs` (packed bits of the player to
move's observation, see game.state_to_nn_input), `legal` (packed bits of the legal actions),
`game` (the league's game id), `opening` (its XOT opening, a row of xot-openings.txt), `ply`
(moves from the standard start, passes and the opening's included) and `z` (the game's outcome
for the player to move); `heldout_openings` and `heldout-openings.txt`, the held-out openings as
indices and as an opening list (e.g. for vs_engine.py's --xot); and once labelled, `policy` (the
teacher's policy over legal actions, float16) and `value` (its value).
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from game import env, load_checkpoint
from xot import XOT_PATH, load_xot

from core.memory.replay_memory import BaseExperience
from core.networks.utils import apply_nn

# games replayed at once in building a corpus, and positions a teacher labels at once
REPLAY_BATCH = 4096
LABEL_BATCH = 8192
OBS_SHAPE = (8, 8, 2)


def read_games(db_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """The league store's games that have moves (results added from elsewhere have none), read
    without writing to the store.

    Returns:
        (ids, openings, results, moves): the game ids, their openings, the first player's
        results (+1, 0, -1), and each game's moves from the standard start
    """
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    try:
        rows = db.execute(
            "SELECT id, opening, result, moves FROM games WHERE length(moves) > 0 "
            "ORDER BY id"
        ).fetchall()
    finally:
        db.close()
    ids, openings, results, moves = zip(*rows) if rows else ((), (), (), ())
    return (
        np.array(ids, np.int64),
        np.array(openings, np.int64),
        np.array(results, np.int64),
        [np.frombuffer(m, dtype=np.uint16).astype(np.int32) for m in moves],
    )


@jax.jit
def _replay(moves: jax.Array):
    """Plays each row of `moves` (games, plies), -1 past a game's end, from the standard start.

    Returns:
        (rewards, terminated, (obs, legal, player, live, legal_move)): the end of each game, and at
        each ply (leading dimension) the position before the move and whether a move was played
        there and was legal
    """
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), len(moves)))
    games = jnp.arange(len(moves))

    def step(states, action):
        live = (action >= 0) & ~states.terminated
        played = jnp.maximum(action, 0)
        legal_move = states.legal_action_mask[games, played]
        new = jax.vmap(env.step)(states, played)
        row = (
            states.observation,
            states.legal_action_mask,
            states.current_player,
            live,
            legal_move | ~live,
        )
        states = jax.tree.map(
            lambda n, o: jnp.where(live.reshape(-1, *([1] * (n.ndim - 1))), n, o),
            new,
            states,
        )
        return states, row

    states, rows = jax.lax.scan(step, states, moves.T)
    return states.rewards, states.terminated, rows


def replay(
    moves: list[np.ndarray], first_ply: int
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """The positions of games, from `first_ply` on, until each game's end.

    Args:
        moves: each game's moves from the standard start
        first_ply: the first ply whose position is kept (e.g. an opening's length)

    Returns:
        (positions, results, ok): the positions' fields (`obs`, `legal`, `index` of their game in
        `moves`, `ply`, `z`), each game's result for its first player as the replay found it,
        and whether the game replayed legally to a finished game (only those games' positions
        are kept)
    """
    length = max(len(m) for m in moves)
    padded = np.full((len(moves), length), -1, np.int32)
    for i, m in enumerate(moves):
        padded[i, : len(m)] = m
    rewards, terminated, (obs, legal, player, live, legal_move) = jax.tree.map(
        np.asarray, _replay(jnp.asarray(padded))
    )
    games = np.arange(len(moves))
    ok = (
        terminated
        & legal_move.all(axis=0)
        & (live.sum(axis=0) == [len(m) for m in moves])
    )
    results = np.sign(rewards[games, player[0]]).astype(np.int64)
    # (plies, games) to rows of positions
    keep = live & ok[None]
    keep[:first_ply] = False
    ply, index = np.nonzero(keep)
    return (
        {
            "obs": np.packbits(obs[ply, index].reshape(len(ply), -1), axis=1),
            "legal": np.packbits(legal[ply, index], axis=1),
            "index": index,
            "ply": ply.astype(np.int16),
            "z": np.sign(rewards[index, player[ply, index]]).astype(np.int8),
        },
        results,
        ok,
    )


def heldout_openings(num_openings: int, fraction: float, seed: int) -> np.ndarray:
    """A fixed random `fraction` of the openings, sorted."""
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(num_openings, round(fraction * num_openings), replace=False)
    )


def build(
    db_path: str | Path,
    out: Path,
    fraction: float = 0.1,
    seed: int = 0,
    xot: str | Path = XOT_PATH,
    log=print,
) -> dict:
    """Builds a corpus (see the module's docstring) in `out` from a league store's games.

    Returns:
        what it holds: counts of games and positions, in all and by split
    """
    openings_list = load_xot(xot)
    ids, openings, results, moves = read_games(db_path)
    opening_length = openings_list.shape[1]
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("obs", "legal", "game", "opening", "ply", "z")
    }
    dropped = mismatched = 0
    started = time.time()
    for start in range(0, len(moves), REPLAY_BATCH):
        chunk = slice(start, start + REPLAY_BATCH)
        positions, replayed, ok = replay(moves[chunk], opening_length)
        # a game must start with its opening and end as recorded
        ok &= np.array(
            [
                len(m) > opening_length
                and (m[:opening_length] == openings_list[o]).all()
                for m, o in zip(moves[chunk], openings[chunk], strict=True)
            ]
        )
        mismatched += int((ok & (replayed != results[chunk])).sum())
        ok &= replayed == results[chunk]
        dropped += int((~ok).sum())
        kept = ok[positions["index"]]
        index = positions.pop("index")[kept]
        for k, v in positions.items():
            parts[k].append(v[kept])
        parts["game"].append(ids[chunk][index])
        parts["opening"].append(openings[chunk][index].astype(np.int16))
        log(
            f"{min(start + REPLAY_BATCH, len(moves))}/{len(moves)} games replayed "
            f"({time.time() - started:.0f} s)"
        )
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v)
    heldout = heldout_openings(len(openings_list), fraction, seed)
    np.save(out / "heldout_openings.npy", heldout)
    lines = Path(xot).read_text().split()
    (out / "heldout-openings.txt").write_text("".join(lines[i] + "\n" for i in heldout))
    is_heldout = np.isin(arrays["opening"], heldout)
    distinct = len(np.unique(arrays["obs"].view(f"V{arrays['obs'].shape[1]}")))
    summary = {
        "source": str(Path(db_path).resolve()),
        "games": len(np.unique(arrays["game"])),
        "games_dropped": dropped,
        "games_with_other_results": mismatched,
        "positions": len(is_heldout),
        "distinct_positions": distinct,
        "heldout_fraction": fraction,
        "seed": seed,
        "heldout_openings": len(heldout),
        "train_positions": int((~is_heldout).sum()),
        "heldout_positions": int(is_heldout.sum()),
        "heldout_games": len(np.unique(arrays["game"][is_heldout])),
    }
    (out / "corpus.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


class Corpus:
    """A corpus that `build` made, its arrays memory-mapped."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.info = json.loads((self.directory / "corpus.json").read_text())

        def load(name: str) -> np.ndarray:
            return np.load(self.directory / f"{name}.npy", mmap_mode="r")

        self.obs, self.legal, self.opening, self.z = (
            load(k) for k in ("obs", "legal", "opening", "z")
        )
        self.heldout = np.load(self.directory / "heldout_openings.npy")
        self.labelled = (self.directory / "value.npy").exists()
        if self.labelled:
            self.policy, self.value = load("policy"), load("value")

    def __len__(self) -> int:
        return len(self.z)

    def split(self, name: str) -> np.ndarray:
        """The indices of the "train" or "heldout" positions."""
        heldout = np.isin(self.opening, self.heldout)
        return np.flatnonzero(heldout if name == "heldout" else ~heldout)

    def observations(self, indices: np.ndarray) -> np.ndarray:
        """(len(indices), 8, 8, 2) bool, the positions' observations."""
        bits = np.unpackbits(self.obs[indices], axis=1, count=int(np.prod(OBS_SHAPE)))
        return bits.astype(bool).reshape(len(indices), *OBS_SHAPE)

    def legal_actions(self, indices: np.ndarray) -> np.ndarray:
        """(len(indices), 65) bool, the positions' legal actions."""
        bits = np.unpackbits(self.legal[indices], axis=1, count=env.num_actions)
        return bits.astype(bool)

    def experiences(self, indices: np.ndarray) -> BaseExperience:
        """Positions as training samples, as self-play would store them: policy targets the
        teacher's policies (float16, to keep them small), the player to move player 0, its
        outcome the game's, and `search_value` the teacher's value. Needs a labelled corpus."""
        if not self.labelled:
            raise ValueError(f"{self.directory} isn't labelled: run `label` first")
        indices = np.sort(indices)
        z = np.asarray(self.z[indices], np.float32)
        fields: dict[str, Any] = {
            "reward": np.stack([z, -z], axis=1),
            "policy_weights": np.asarray(self.policy[indices]),
            "policy_mask": self.legal_actions(indices),
            "observation_nn": self.observations(indices),
            "cur_player_id": np.zeros(len(indices), np.int32),
            "search_value": np.asarray(self.value[indices], np.float32),
        }
        return BaseExperience(**fields)


@eqx.filter_jit
def _label(net, nn_state, obs, legal):
    (logits, value), _ = apply_nn(net, nn_state, obs)
    logits = jnp.where(legal, logits.astype(jnp.float32), -jnp.inf)
    return jax.nn.softmax(logits, axis=-1), value.reshape(-1)


def label(corpus: Corpus, teacher: str, teacher_net: str, log=print) -> None:
    """Labels every position of `corpus` with a teacher's policy and value, from one forward pass
    of the network in inference mode (BatchNorm's stored statistics), computing in float32."""
    network, (params, nn_state) = load_checkpoint(teacher, teacher_net)
    net = eqx.nn.inference_mode(eqx.combine(params, network))
    policy = np.lib.format.open_memmap(
        corpus.directory / "policy.npy.part",
        mode="w+",
        dtype=np.float16,
        shape=(len(corpus), env.num_actions),
    )
    value = np.empty(len(corpus), np.float32)
    started = time.time()
    for start in range(0, len(corpus), LABEL_BATCH):
        indices = np.arange(start, min(start + LABEL_BATCH, len(corpus)))
        p, v = _label(
            net,
            nn_state,
            jnp.asarray(corpus.observations(indices)),
            jnp.asarray(corpus.legal_actions(indices)),
        )
        policy[indices] = np.asarray(p, np.float16)
        value[indices] = np.asarray(v)
        if (start // LABEL_BATCH) % 200 == 0:
            log(
                f"{indices[-1] + 1}/{len(corpus)} labelled ({time.time() - started:.0f} s)"
            )
    policy.flush()
    del policy
    (corpus.directory / "policy.npy.part").rename(corpus.directory / "policy.npy")
    # value.npy last: its presence marks a labelled corpus
    np.save(corpus.directory / "value.npy", value)
    (corpus.directory / "teacher.json").write_text(
        json.dumps(
            {
                "teacher": str(Path(teacher).resolve()),
                "teacher_net": teacher_net,
                "seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Builds a corpus of the league's positions, and labels it."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="replay the league's games")
    build_parser.add_argument("--root", required=True, help="the league's store")
    build_parser.add_argument("--out", required=True, help="the corpus's directory")
    build_parser.add_argument(
        "--heldout-fraction",
        type=float,
        default=0.1,
        help="of the XOT openings, held out with their games' positions",
    )
    build_parser.add_argument(
        "--seed", type=int, default=0, help="draws the held-out openings"
    )
    label_parser = commands.add_parser("label", help="label it with a teacher")
    label_parser.add_argument("--corpus", required=True)
    label_parser.add_argument("--teacher", required=True, help="a checkpoint")
    label_parser.add_argument("--teacher-net", required=True, help="its --net")
    args = parser.parse_args()
    if args.command == "build":
        summary = build(
            Path(args.root) / "league.db",
            Path(args.out),
            args.heldout_fraction,
            args.seed,
            log=lambda line: print(line, flush=True),
        )
        print(json.dumps(summary, indent=2))
    else:
        label(
            Corpus(args.corpus),
            args.teacher,
            args.teacher_net,
            log=lambda line: print(line, flush=True),
        )


if __name__ == "__main__":
    main()
