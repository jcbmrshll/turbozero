"""Every evaluation game, wherever it was played, in one SQLite file.

A player is everything that decides how it plays, as a config: a network's checkpoint (by its
digest), how it searches and on what, and its budget (visits a move, or a clock); an engine's build
(by its files' digest), its options, and its level or clock. Its id is the config's digest
(`player_id`), so the same config is the same player wherever and whenever it plays, and any change
to it is a new player without anyone deciding so. What the game was played on isn't part of who
played: that's each game's host.

A match is two players and how their games are set up: where they start, whether they're stopped
and solved at so many empties. Its games are numbered, each start's pair of games two numbers
in a row, so a game recorded twice (a job run again, a file imported twice) is kept once.

A game keeps its moves from the standard start (passes aren't moves), the seconds each move after
the opening took where it was timed alone, how it ended, and the host it ran on. Hosts are kept
once each, by what they are (`hardware.describe`), not by when they were seen.

Only one process writes at a time (WAL mode lets readers carry on). Remote jobs write lines of
`GameRecord.to_json` to files, which `import_records` brings in.
"""

import hashlib
import json
import sqlite3
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    kind TEXT NOT NULL,
    config TEXT NOT NULL,
    added REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS hosts (
    id TEXT PRIMARY KEY,
    info TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY,
    a_id TEXT NOT NULL REFERENCES players(id),
    b_id TEXT NOT NULL REFERENCES players(id),
    config TEXT NOT NULL,
    placement TEXT NOT NULL,
    code_version TEXT NOT NULL,
    planned INTEGER NOT NULL,
    started REAL NOT NULL,
    finished REAL,
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(id),
    number INTEGER NOT NULL,
    black_id TEXT NOT NULL REFERENCES players(id),
    white_id TEXT NOT NULL REFERENCES players(id),
    start TEXT NOT NULL,
    moves TEXT NOT NULL,
    opening_moves INTEGER NOT NULL,
    move_seconds TEXT NOT NULL,
    ending TEXT NOT NULL CHECK (ending IN ('end', 'solved', 'timeout')),
    solved_at INTEGER,
    solve_exact INTEGER,
    margin INTEGER NOT NULL,
    result INTEGER NOT NULL,
    flagged TEXT,
    hard TEXT,
    black_seconds REAL,
    white_seconds REAL,
    host_id TEXT REFERENCES hosts(id),
    load REAL,
    finished REAL NOT NULL,
    UNIQUE (match_id, number)
);
CREATE INDEX IF NOT EXISTS games_by_black ON games (black_id);
CREATE INDEX IF NOT EXISTS games_by_white ON games (white_id);
CREATE TABLE IF NOT EXISTS match_runs (
    id INTEGER PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(id),
    host_id TEXT REFERENCES hosts(id),
    info TEXT NOT NULL,
    at REAL NOT NULL
);
"""


def canonical(value: Any) -> str:
    """`value` as JSON that's the same for equal values: keys sorted, no spaces."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:length]


def player_id(config: dict[str, Any]) -> str:
    """A player's id: its config's digest."""
    return "p-" + digest(config)


def host_id(info: dict[str, Any]) -> str:
    return "h-" + digest(info)


@dataclass
class GameRecord:
    """A finished game, as a job writes it and the store keeps it.

    Attributes:
        match: the match's id
        number: the game's number in its match: 2 * its start's index, plus 1 if the match's
            player b is black
        black: the id of the player who moved first from `start`
        white: the other's
        start: where it started: "xot:INDEX" (an opening of the XOT list, whose moves are the
            first of `moves`), or "board:TEXT" (a position given whole, see `ggs.Start.text`)
        moves: every move from the start, the opening's first, as squares (0 is a1); no passes
        opening_moves: how many of `moves` are the opening's
        move_seconds: each move's seconds after the opening, None where moves weren't timed
            alone (games played in batches)
        ending: "end", played out; "solved", stopped at `solved_at` empties and solved; "timeout",
            ended by a hard timeout (a GGS clock's extension spent)
        solved_at: the empties of the position solved, if it was
        solve_exact: whether the solve was for the exact margin; else only its sign is right
        margin: black's discs less white's at the end (or the solved margin), whatever the clock
        result: black's result counting the clock: the disc result, with the empty squares the
            winner's, as GGS counts it, a timeout a loss (`ggs.game_result`)
        flagged: who ran out of time first, "b" or "w", if anyone did
        hard: who ran out of a GGS clock's extension too, ending the game
        black_seconds: black's seconds on the game, if timed alone
        white_seconds: white's
        host: what it was played on (`hardware.describe`)
        load: the host's load average over the last minute, as it finished
        finished: when it finished, in seconds since the epoch
    """

    match: str
    number: int
    black: str
    white: str
    start: str
    moves: list[int]
    opening_moves: int
    move_seconds: list[float | None]
    ending: str
    margin: int
    result: int
    solved_at: int | None = None
    solve_exact: bool | None = None
    flagged: str | None = None
    hard: str | None = None
    black_seconds: float | None = None
    white_seconds: float | None = None
    host: dict[str, Any] = field(default_factory=dict)
    load: float | None = None
    finished: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> "GameRecord":
        return cls(**json.loads(line))


@dataclass(frozen=True)
class PlayerRow:
    id: str
    label: str
    kind: str
    config: dict[str, Any]
    added: float


@dataclass(frozen=True)
class MatchRow:
    id: str
    a: str
    b: str
    config: dict[str, Any]
    placement: str
    code_version: str
    planned: int
    started: float
    finished: float | None
    note: str
    played: int


def _player_row(r: tuple) -> PlayerRow:
    return PlayerRow(r[0], r[1], r[2], json.loads(r[3]), r[4])


class Store:
    """The evaluations' store, a SQLite file at `path`, made if it isn't there."""

    def __init__(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection]:
        with self.db:
            yield self.db

    def add_player(self, config: dict[str, Any], label: str) -> str:
        """The player of `config`, added if it's new; its id."""
        id_ = player_id(config)
        with self.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO players (id, label, kind, config, added) "
                "VALUES (?, ?, ?, ?, ?)",
                (id_, label, config["kind"], canonical(config), time.time()),
            )
        return id_

    def player(self, id_: str) -> PlayerRow | None:
        row = self.db.execute(
            "SELECT id, label, kind, config, added FROM players WHERE id = ?", (id_,)
        ).fetchone()
        return None if row is None else _player_row(row)

    def players(self) -> list[PlayerRow]:
        rows = self.db.execute(
            "SELECT id, label, kind, config, added FROM players ORDER BY added"
        ).fetchall()
        return [_player_row(r) for r in rows]

    def add_host(self, info: dict[str, Any]) -> str:
        id_ = host_id(info)
        with self.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO hosts (id, info) VALUES (?, ?)",
                (id_, canonical(info)),
            )
        return id_

    def host(self, id_: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT info FROM hosts WHERE id = ?", (id_,)).fetchone()
        return None if row is None else json.loads(row[0])

    def start_match(
        self,
        match: str,
        a: str,
        b: str,
        config: dict[str, Any],
        placement: str,
        code_version: str,
        planned: int,
        note: str = "",
    ) -> None:
        """Records a match about to be played; one already here is left as it was, as a job run
        again continues its match."""
        with self.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO matches (id, a_id, b_id, config, placement, "
                "code_version, planned, started, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    match,
                    a,
                    b,
                    canonical(config),
                    placement,
                    code_version,
                    planned,
                    time.time(),
                    note,
                ),
            )

    def finish_match(self, match: str) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE matches SET finished = ? WHERE id = ?", (time.time(), match)
            )

    def add_run(self, match: str, info: dict[str, Any], host: dict[str, Any]) -> None:
        """What a run of a match's games said of itself as a whole: e.g. a batch's step times."""
        host_ = self.add_host(host) if host else None
        with self.transaction() as db:
            db.execute(
                "INSERT INTO match_runs (match_id, host_id, info, at) VALUES (?, ?, ?, ?)",
                (match, host_, canonical(info), time.time()),
            )

    def match(self, id_: str) -> MatchRow | None:
        found = [m for m in self.matches() if m.id == id_]
        return found[0] if found else None

    def matches(self) -> list[MatchRow]:
        rows = self.db.execute(
            "SELECT m.id, m.a_id, m.b_id, m.config, m.placement, m.code_version, m.planned, "
            "m.started, m.finished, m.note, "
            "(SELECT COUNT(*) FROM games g WHERE g.match_id = m.id) "
            "FROM matches m ORDER BY m.started"
        ).fetchall()
        return [
            MatchRow(
                id=r[0],
                a=r[1],
                b=r[2],
                config=json.loads(r[3]),
                placement=r[4],
                code_version=r[5],
                planned=r[6],
                started=r[7],
                finished=r[8],
                note=r[9],
                played=r[10],
            )
            for r in rows
        ]

    def record_games(self, records: Iterable[GameRecord]) -> int:
        """Keeps `records` in one transaction, skipping any its match already has; how many
        were new."""
        added = 0
        with self.transaction() as db:
            for r in records:
                host = None
                if r.host:
                    host = host_id(r.host)
                    db.execute(
                        "INSERT OR IGNORE INTO hosts (id, info) VALUES (?, ?)",
                        (host, canonical(r.host)),
                    )
                cursor = db.execute(
                    "INSERT OR IGNORE INTO games (match_id, number, black_id, white_id, start, "
                    "moves, opening_moves, move_seconds, ending, solved_at, solve_exact, margin, "
                    "result, flagged, hard, black_seconds, white_seconds, host_id, load, "
                    "finished) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        r.match,
                        r.number,
                        r.black,
                        r.white,
                        r.start,
                        json.dumps(r.moves),
                        r.opening_moves,
                        json.dumps(r.move_seconds),
                        r.ending,
                        r.solved_at,
                        None if r.solve_exact is None else int(r.solve_exact),
                        r.margin,
                        r.result,
                        r.flagged,
                        r.hard,
                        r.black_seconds,
                        r.white_seconds,
                        host,
                        r.load,
                        r.finished,
                    ),
                )
                added += cursor.rowcount
        return added

    def import_records(self, lines: Iterable[str]) -> int:
        """Keeps the games of `lines` of `GameRecord.to_json`, blank lines skipped."""
        return self.record_games(
            GameRecord.from_json(line) for line in lines if line.strip()
        )

    def numbers(self, match: str) -> set[int]:
        """The numbers of the games `match` has."""
        rows = self.db.execute(
            "SELECT number FROM games WHERE match_id = ?", (match,)
        ).fetchall()
        return {r[0] for r in rows}

    def games(self, match: str) -> list[GameRecord]:
        """`match`'s games, by number."""
        rows = self.db.execute(
            "SELECT g.match_id, g.number, g.black_id, g.white_id, g.start, g.moves, "
            "g.opening_moves, g.move_seconds, g.ending, g.margin, g.result, g.solved_at, "
            "g.solve_exact, g.flagged, g.hard, g.black_seconds, g.white_seconds, h.info, "
            "g.load, g.finished FROM games g LEFT JOIN hosts h ON h.id = g.host_id "
            "WHERE g.match_id = ? ORDER BY g.number",
            (match,),
        ).fetchall()
        return [
            GameRecord(
                match=r[0],
                number=r[1],
                black=r[2],
                white=r[3],
                start=r[4],
                moves=json.loads(r[5]),
                opening_moves=r[6],
                move_seconds=json.loads(r[7]),
                ending=r[8],
                margin=r[9],
                result=r[10],
                solved_at=r[11],
                solve_exact=None if r[12] is None else bool(r[12]),
                flagged=r[13],
                hard=r[14],
                black_seconds=r[15],
                white_seconds=r[16],
                host=json.loads(r[17]) if r[17] else {},
                load=r[18],
                finished=r[19],
            )
            for r in rows
        ]
