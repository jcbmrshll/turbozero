"""The league's permanent record: its networks, players and every game, in one SQLite file.

A player is a network at a search budget, or an external engine with its settings. The same
checkpoint at 128 and 1000 sims is two players; a game's code version is recorded with it, but
isn't part of who played. A network's checkpoint is copied into the store when it's registered,
so it can't be deleted from under the league.

A batch of games is written in one transaction, so stopping at any moment, however abruptly,
leaves whole batches. Ratings aren't stored: they're fit to the games whenever needed.
"""

import hashlib
import json
import shutil
import sqlite3
import time
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS networks (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    net TEXT NOT NULL,
    run TEXT NOT NULL,
    url TEXT,
    source TEXT NOT NULL,
    added REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    network_id INTEGER REFERENCES networks(id),
    sims INTEGER,
    external TEXT,
    settings TEXT NOT NULL DEFAULT '{}',
    retired INTEGER NOT NULL DEFAULT 0,
    added REAL NOT NULL,
    CHECK ((network_id IS NULL) != (external IS NULL))
);
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY,
    a_id INTEGER NOT NULL REFERENCES players(id),
    b_id INTEGER NOT NULL REFERENCES players(id),
    seconds REAL NOT NULL,
    compiled INTEGER NOT NULL,
    code_version TEXT NOT NULL,
    dtype TEXT NOT NULL,
    finished REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    first_id INTEGER NOT NULL REFERENCES players(id),
    second_id INTEGER NOT NULL REFERENCES players(id),
    opening INTEGER NOT NULL,
    result INTEGER NOT NULL CHECK (result IN (-1, 0, 1)),
    first_score REAL,
    second_score REAL,
    moves BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS games_by_pair ON games (first_id, second_id);
CREATE INDEX IF NOT EXISTS games_by_second ON games (second_id);
CREATE INDEX IF NOT EXISTS games_by_batch ON games (batch_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Player:
    """A player of the league.

    Attributes:
        id: its id in the store
        name: e.g. "r2-A-drop8x@1000" or "edax-L8"
        run: where its network was trained, or its engine's name: games within a run say less
        sims: a network's MCTS iterations a move
        network: its network's name
        path: its network's checkpoint, in the store
        net: its network's spec (see `core.networks.spec`)
        url: where to read about it, e.g. its training run's page (an engine's is its settings')
        external: its engine's name, if it's an engine (see `game.LeagueGame.external_player`)
        settings: its engine's settings, e.g. its level
        retired: whether it no longer plays new games (it stays rated)
    """

    id: int
    name: str
    run: str = ""
    sims: int | None = None
    network: str | None = None
    path: str | None = None
    net: str | None = None
    url: str | None = None
    external: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    retired: bool = False

    @property
    def is_external(self) -> bool:
        return self.external is not None


@dataclass(frozen=True)
class GameRecord:
    """A game to record.

    Attributes:
        first: the id of the player who moved first in the game (from the standard start)
        second: the id of the other
        opening: the index of the opening it was played from
        result: +1 if the first player won, -1 if the second did, 0 for a draw
        moves: every action from the standard start, the opening's first
        first_score: the first player's score at the end, if the game has one (e.g. discs)
        second_score: the second player's
    """

    first: int
    second: int
    opening: int
    result: int
    moves: Sequence[int]
    first_score: float | None = None
    second_score: float | None = None


@dataclass(frozen=True)
class PairRecord:
    """All the games between two players: `wins` of the first, `draws` and `losses`."""

    a: int
    b: int
    wins: int
    draws: int
    losses: int


def encode_moves(moves: Sequence[int]) -> bytes:
    return np.asarray(moves, dtype=np.uint16).tobytes()


def decode_moves(moves: bytes) -> list[int]:
    return np.frombuffer(moves, dtype=np.uint16).tolist()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class Store:
    """The league's file, and the checkpoints beside it, under `root`. Any number of
    processes can have it open: one writes at a time, and readers don't wait for it."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        (self.root / "networks").mkdir(parents=True, exist_ok=True)
        # the dashboard reads it from its server's threads, one at a time
        self.db = sqlite3.connect(
            self.root / "league.db", timeout=30, check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        # survives the process being killed, and the machine losing power
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection]:
        """All of what's written inside it, or, if it raises, none of it."""
        with self.db:
            yield self.db

    def add_network(self, name: str, checkpoint: Path | str, net: str, run: str) -> int:
        """Registers a checkpoint under `name`, copying it into the store. Registering the
        same one again changes nothing.

        Raises:
            ValueError: if `name` already holds a different checkpoint or `net`
        """
        checkpoint = Path(checkpoint)
        digest = sha256(checkpoint)
        row = self.db.execute(
            "SELECT id, sha256, net FROM networks WHERE name = ?", (name,)
        ).fetchone()
        if row is not None:
            if (row["sha256"], row["net"]) != (digest, net):
                raise ValueError(f"{name} is already a different network")
            return row["id"]
        path = self.root / "networks" / f"{digest}.eqx"
        if not path.exists():
            # whole or not at all: a rename can't be interrupted halfway
            partial = path.with_suffix(".partial")
            shutil.copyfile(checkpoint, partial)
            partial.chmod(0o444)
            partial.rename(path)
        with self.transaction() as db:
            cursor = db.execute(
                "INSERT INTO networks (name, path, sha256, net, run, source, added) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, str(path), digest, net, run, str(checkpoint), time.time()),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def set_network_url(self, name: str, url: str) -> None:
        """Links `name` to `url`.

        Raises:
            ValueError: if there's no such network
        """
        with self.transaction() as db:
            cursor = db.execute(
                "UPDATE networks SET url = ? WHERE name = ?", (url, name)
            )
        if cursor.rowcount == 0:
            raise ValueError(f"no network {name} in the store")

    def add_net_player(self, network: str, sims: int) -> int:
        """The player that is `network` searching `sims` iterations a move."""
        row = self.db.execute(
            "SELECT id FROM networks WHERE name = ?", (network,)
        ).fetchone()
        if row is None:
            raise ValueError(f"no network {network} in the store")
        return self._add_player(
            f"{network}@{sims}",
            "INSERT INTO players (name, network_id, sims, added) VALUES (?, ?, ?, ?)",
            (row["id"], sims),
        )

    def add_external_player(
        self, name: str, external: str, settings: dict[str, Any] | None = None
    ) -> int:
        """The player that is the engine `external` with `settings`."""
        return self._add_player(
            name,
            "INSERT INTO players (name, external, settings, added) VALUES (?, ?, ?, ?)",
            (external, json.dumps(settings or {}, sort_keys=True)),
        )

    def _add_player(self, name: str, insert: str, values: tuple) -> int:
        row = self.db.execute(
            "SELECT id FROM players WHERE name = ?", (name,)
        ).fetchone()
        if row is not None:
            return row["id"]
        with self.transaction() as db:
            cursor = db.execute(insert, (name, *values, time.time()))
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def set_retired(self, name: str, retired: bool = True) -> None:
        with self.transaction() as db:
            changed = db.execute(
                "UPDATE players SET retired = ? WHERE name = ?", (int(retired), name)
            ).rowcount
        if not changed:
            raise ValueError(f"no player {name} in the store")

    def players(self) -> list[Player]:
        """Every player, in the order they were added."""
        rows = self.db.execute(
            "SELECT p.id, p.name, p.sims, p.external, p.settings, p.retired, "
            "n.name AS network, n.path, n.net, n.run, n.url "
            "FROM players p LEFT JOIN networks n ON n.id = p.network_id ORDER BY p.id"
        )
        players = []
        for row in rows:
            settings = json.loads(row["settings"])
            players.append(
                Player(
                    id=row["id"],
                    name=row["name"],
                    run=row["run"] or row["external"],
                    sims=row["sims"],
                    network=row["network"],
                    path=row["path"],
                    net=row["net"],
                    url=row["url"] or settings.get("url"),
                    external=row["external"],
                    settings=settings,
                    retired=bool(row["retired"]),
                )
            )
        return players

    def record_batch(
        self,
        a: int,
        b: int,
        seconds: float,
        compiled: bool,
        code_version: str,
        dtype: str,
        games: Iterable[GameRecord],
    ) -> int:
        """Records a batch of games between players `a` and `b`, all or none of it, and how
        it was played: how long it took, whether that includes compiling, the code that played it and the dtype of its networks."""
        with self.transaction() as db:
            cursor = db.execute(
                "INSERT INTO batches (a_id, b_id, seconds, compiled, code_version, "
                "dtype, finished) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (a, b, seconds, int(compiled), code_version, dtype, time.time()),
            )
            batch = cursor.lastrowid
            db.executemany(
                "INSERT INTO games (batch_id, first_id, second_id, opening, result, "
                "first_score, second_score, moves) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (batch, g.first, g.second, g.opening, g.result, g.first_score)
                    + (g.second_score, encode_moves(g.moves))
                    for g in games
                ],
            )
        assert batch is not None
        return batch

    def pair_records(self) -> list[PairRecord]:
        """Each pair's games so far, the pair in order of id."""
        rows = self.db.execute(
            "SELECT first_id, second_id, result, COUNT(*) FROM games "
            "GROUP BY first_id, second_id, result"
        )
        totals: dict[tuple[int, int], list[int]] = {}
        for first, second, result, games in rows:
            a, b = sorted((first, second))
            # the result for `a`: wins, draws, losses
            result = result if a == first else -result
            totals.setdefault((a, b), [0, 0, 0])[1 - result] += games
        return [PairRecord(a, b, *counts) for (a, b), counts in totals.items()]

    def played_openings(self, a: int, b: int) -> set[int]:
        """The openings the two players have already played each other from."""
        rows = self.db.execute(
            "SELECT DISTINCT opening FROM games "
            "WHERE (first_id = ? AND second_id = ?) OR (first_id = ? AND second_id = ?)",
            (a, b, b, a),
        )
        return {row[0] for row in rows}

    def batch_costs(self) -> list[tuple[int, int, int, float]]:
        """(a, b, games, seconds) of each batch, leaving out those that compiled something
        and results recorded from elsewhere, which took no time here."""
        rows = self.db.execute(
            "SELECT b.a_id, b.b_id, COUNT(g.id), b.seconds FROM batches b "
            "JOIN games g ON g.batch_id = b.id WHERE b.compiled = 0 AND b.seconds > 0 "
            "GROUP BY b.id"
        )
        return [(row[0], row[1], row[2], row[3]) for row in rows]

    def num_games(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM games").fetchone()[0]

    def set_meta(self, key: str, value: Any) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])
