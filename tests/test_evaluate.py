"""evaluate.py's players, matches, games and store, with fake players and engine builds."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

import eval_players
import evaluate
from clock_match import UNTIMED, GameRun
from engines import Board
from eval_games import (
    MatchSpec,
    Opening,
    numbers_of,
    play_parallel,
    play_sequential,
    replay,
)
from eval_players import PlayerSpec, resolve
from ggs import Clock

from core.evals import hardware
from core.evals.store import GameRecord, Store, player_id


class FirstLegal:
    """Plays the first legal move, taking `seconds` of a fake clock each."""

    def __init__(self, now: list[float], seconds: float = 0.0):
        self.now = now
        self.seconds = seconds
        self.board = Board()
        self.closed = False

    def set_position(self, black, white, color):
        self.board = Board.at(black, white, color)

    def play(self, color, move):
        self.board.play(move)

    def move(self, color, seconds_left):
        self.now[0] += self.seconds
        move = self.board.legal()[0]
        self.board.play(move)
        return move

    def close(self):
        self.closed = True


@pytest.fixture
def now(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("clock_match.time.perf_counter", lambda: now[0])
    return now


@pytest.fixture
def engine_dir(tmp_path):
    """A fake Edax build: its runtime files, which are all its digest is of."""
    root = tmp_path / "edax"
    (root / "bin").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "bin" / "lEdax-x64").write_bytes(b"binary")
    (root / "data" / "eval.dat").write_bytes(b"weights")
    return root


def fake(label: str, budget: dict, kind: str = "engine", **config) -> PlayerSpec:
    return PlayerSpec(
        {"kind": kind, "budget": budget, "name": label, **config}, {}, label, ()
    )


def record(match: str, number: int, **changes) -> GameRecord:
    from dataclasses import replace

    game = GameRecord(
        match=match,
        number=number,
        black="p-a" if number % 2 == 0 else "p-b",
        white="p-b" if number % 2 == 0 else "p-a",
        start="xot:1",
        moves=[37, 43],
        opening_moves=1,
        move_seconds=[0.5],
        ending="end",
        margin=10,
        result=10,
        black_seconds=0.5,
        white_seconds=0.0,
        host={"cpu": "fake"},
    )
    return replace(game, **changes)


# --- players ---------------------------------------------------------------------------------


def test_an_engine_is_the_same_player_with_its_defaults_said_or_not(engine_dir):
    plain = resolve(f"edax:level=8,dir={engine_dir}", cores=8)
    said = resolve(f"edax:level=8,threads=8,dir={engine_dir}", cores=4)
    assert plain.id == said.id
    assert plain.config["threads"] == 8 and plain.label == "edax L8 t8"


def test_any_change_to_how_a_player_plays_is_a_new_player(engine_dir):
    base = resolve(f"edax:level=8,dir={engine_dir}", cores=8)
    others = [
        resolve(f"edax:level=9,dir={engine_dir}", cores=8),
        resolve(f"edax:level=8,dir={engine_dir}", cores=4),
        resolve(f"edax:clock=10,dir={engine_dir}", cores=8),
        resolve(f"edax:level=8,args=-h 22,dir={engine_dir}", cores=8),
    ]
    assert len({base.id, *(o.id for o in others)}) == 5
    # a rebuild that changes a file
    (engine_dir / "data" / "eval.dat").write_bytes(b"other weights")
    assert resolve(f"edax:level=8,dir={engine_dir}", cores=8).id != base.id


def test_where_an_engine_is_built_is_not_who_it_is(engine_dir, tmp_path):
    copy = tmp_path / "copy"
    import shutil

    shutil.copytree(engine_dir, copy)
    assert (
        resolve(f"edax:level=8,dir={engine_dir}", cores=8).id
        == resolve(f"edax:level=8,dir={copy}", cores=8).id
    )


def test_a_player_has_exactly_one_budget(engine_dir):
    with pytest.raises(ValueError, match="one budget"):
        resolve(f"edax:dir={engine_dir}", cores=8)
    with pytest.raises(ValueError, match="one budget"):
        resolve(f"edax:level=8,clock=10,dir={engine_dir}", cores=8)


def test_unknown_options_are_refused(engine_dir, tmp_path):
    with pytest.raises(ValueError, match="unknown options"):
        resolve(f"edax:level=8,depth=3,dir={engine_dir}", cores=8)
    checkpoint = tmp_path / "1.eqx"
    checkpoint.write_bytes(b"params")
    with pytest.raises(ValueError, match="unknown options"):
        resolve(f"net:checkpoint={checkpoint},spec=resnet:2x8,visits=10,sims=3", 8)


def test_a_network_is_its_checkpoint_by_content_and_its_search(tmp_path):
    one, two = tmp_path / "a" / "1.eqx", tmp_path / "b" / "1.eqx"
    for path in (one, two):
        path.parent.mkdir()
        path.write_bytes(b"params")
    spec = "spec=resnet:2x8,visits=100"
    a = resolve(f"net:checkpoint={one},{spec}", cores=8)
    b = resolve(f"net:checkpoint={two},{spec}", cores=8)
    assert a.id == b.id
    assert a.config["search"]["batch"] == 1 and a.config["search"]["threads"] == 8
    assert resolve(f"net:checkpoint={one},{spec},batch=1", cores=8).id == a.id
    assert resolve(f"net:checkpoint={one},{spec},batch=8", cores=8).id != a.id
    assert resolve(f"net:checkpoint={one},{spec},solve=20", cores=8).id != a.id
    two.write_bytes(b"other params")
    assert resolve(f"net:checkpoint={two},{spec}", cores=8).id != a.id


def test_a_clock_is_a_contests_or_ggss(tmp_path):
    checkpoint = tmp_path / "1.eqx"
    checkpoint.write_bytes(b"params")
    contest = resolve(f"net:checkpoint={checkpoint},spec=s,clock=10", 8)
    ggs = resolve(f"net:checkpoint={checkpoint},spec=s,clock=01:00//02:00", 8)
    assert contest.clock == Clock.contest(10)
    assert ggs.clock == Clock(60, 0, 120)
    assert contest.config["budget"]["clock"]["ext"] is None


def test_batched_networks_play_at_fixed_visits_only(tmp_path):
    checkpoint = tmp_path / "1.eqx"
    checkpoint.write_bytes(b"params")
    base = f"net:checkpoint={checkpoint},spec=s,device=gpu-batched"
    assert resolve(f"{base},visits=100", 8).config["search"]["tree_factor"] == 4
    with pytest.raises(ValueError, match="fixed visits"):
        resolve(f"{base},clock=10", 8)


def test_engines_in_batched_matches_default_to_a_thread(engine_dir):
    assert (
        resolve(f"edax:level=8,dir={engine_dir}", 8, batched=True).config["threads"]
        == 1
    )


def test_a_league_network_is_found_by_name(tmp_path):
    import sqlite3

    store = tmp_path / "league.db"
    checkpoint = tmp_path / "abc.eqx"
    checkpoint.write_bytes(b"params")
    db = sqlite3.connect(store)
    db.execute("CREATE TABLE networks (name TEXT, path TEXT, net TEXT, sha256 TEXT)")
    db.execute(
        "INSERT INTO networks VALUES (?, ?, ?, ?)",
        ("4x64-x", str(checkpoint), "resnet:4x64", "abc"),
    )
    db.commit()
    db.close()
    player = resolve("net:net=4x64-x,visits=10", 8, store=store)
    assert player.config["network"] == {"sha256": "abc", "net": "resnet:4x64"}
    assert player.runtime["network"]["name"] == "4x64-x"
    with pytest.raises(ValueError, match="no network"):
        resolve("net:net=nope,visits=10", 8, store=store)


def test_a_table_is_a_player_as_its_string_is(engine_dir):
    table = {"kind": "edax", "level": 8, "dir": str(engine_dir)}
    assert resolve(table, 8).id == resolve(f"edax:level=8,dir={engine_dir}", 8).id


def test_a_network_runs_where_its_runtime_says(monkeypatch, tmp_path):
    made = {}

    def player_maker(name, options, clock):
        made.update(name=name, options=options, clock=clock)
        return lambda: None

    import clock_match

    monkeypatch.setattr(clock_match, "player_maker", player_maker)
    monkeypatch.setattr(eval_players, "CACHE_DIR", tmp_path)
    from bench_cpu_backends import NETWORKS

    runtime = {
        "player": "net",
        "network": {"name": "n", "checkpoint": "league/x.eqx", "net": "resnet:2x8"},
        "options": {"visits": 10, "batch": 1, "device": "cpu"},
    }
    eval_players.maker(runtime, UNTIMED)
    assert NETWORKS["n"] == (str(tmp_path / "league/x.eqx"), "resnet:2x8")
    assert made["options"] == {
        "visits": "10",
        "batch": "1",
        "device": "cpu",
        "net": "n",
    }


# --- matches ---------------------------------------------------------------------------------


def test_a_matchs_id_is_its_setup_and_where_its_played():
    a, b = fake("a", {"level": 1}, threads=1), fake("b", {"level": 2}, threads=1)
    spec = MatchSpec(a, b, openings=4)
    assert spec.id("local") == MatchSpec(a, b, openings=4).id("local")
    assert spec.id("local") != spec.id("gcp:n4d-highcpu-16")
    assert spec.id("local") != MatchSpec(a, b, openings=4, tag="again").id("local")
    assert spec.id("local") != MatchSpec(a, b, openings=4, solve_at=24).id("local")
    assert MatchSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_batched_matches_are_at_fixed_budgets_and_an_engine_a_game():
    net = fake(
        "n", {"visits": 10}, kind="net", device="gpu-batched", search={"dtype": "x"}
    )
    with pytest.raises(ValueError, match="a thread"):
        MatchSpec(net, fake("e", {"level": 1}, threads=8), openings=1)
    with pytest.raises(ValueError, match="fixed budgets"):
        MatchSpec(net, fake("e", {"clock": Clock.contest(1).__dict__}), openings=1)
    cpu_net = fake("c", {"visits": 10}, kind="net", device="cpu")
    with pytest.raises(ValueError, match="gpu-batched too"):
        MatchSpec(net, cpu_net, openings=1)


def test_jobs_share_a_matchs_starts_each_with_both_colors():
    shares = [numbers_of(5, (k, 3)) for k in range(3)]
    assert sorted(n for share in shares for n in share) == list(range(10))
    assert shares[1] == [2, 3, 8, 9]


def test_the_same_seed_draws_the_same_openings():
    a, b = fake("a", {"level": 1}), fake("b", {"level": 2})
    first = MatchSpec(a, b, openings=3, seed=7).choose_openings()
    assert first == MatchSpec(a, b, openings=3, seed=7).choose_openings()
    assert first != MatchSpec(a, b, openings=3, seed=8).choose_openings()
    assert all(o.label.startswith("xot:") and len(o.moves) == 8 for o in first)
    randoms = MatchSpec(a, b, openings=2, starts="random:14").choose_openings()
    assert all(o.label.startswith("board:") and o.start is not None for o in randoms)


# --- games -----------------------------------------------------------------------------------


def test_each_side_is_on_its_own_clock(now):
    timed, fixed = FirstLegal(now, 1.0), FirstLegal(now, 5.0)
    run = GameRun(timed, fixed, (), (Clock.contest(3), UNTIMED))
    while not run.over:
        run.step()
    game = run.finish()
    # black ran out on its 4th move; white, at a fixed budget, took 5 s a move and never did
    assert game.flagged == "b"
    assert game.seconds[1] > 100


def test_a_game_stops_at_its_solve_depth_for_the_solvers_result(now):
    solved = []

    def solve(me, opp):
        solved.append(64 - (me | opp).bit_count())
        return 6

    run = GameRun(FirstLegal(now), FirstLegal(now), (), 100, solve_at=40, solve=solve)
    while not run.over:
        run.step()
    game = run.finish()
    assert solved == [40] and game.solved_at == 40
    # 24 discs on the board after 20 moves from the standard start; the mover's +6 is black's
    # when black is to move
    assert len(game.moves) == 20
    board = Board()
    for move in game.moves:
        board.play(move)
    assert game.margin == game.result == (6 if board.black_to_move else -6)


def test_a_batched_game_cut_where_it_would_have_stopped_is_the_same_game(now):
    def solve(me, opp):
        return (me.bit_count() - opp.bit_count()) * 2

    run = GameRun(FirstLegal(now), FirstLegal(now), (), 100, solve_at=30, solve=solve)
    while not run.over:
        run.step()
    stopped = run.finish()
    whole = GameRun(FirstLegal(now), FirstLegal(now), (), 100)
    while not whole.over:
        whole.step()
    cut = replay(whole.finish().moves, 0, 30, solve)
    assert cut.moves == stopped.moves
    assert (cut.margin, cut.solved_at) == (stopped.margin, stopped.solved_at)


def test_a_match_plays_its_numbers_each_start_both_ways_by_fresh_players(now):
    a, b = (
        fake("a", {"level": 1}),
        fake("b", {"clock": {"ini": 50.0, "inc": 0.0, "ext": None}}),
    )
    spec = MatchSpec(a, b, openings=3, seed=1)
    made = []

    def make(player, clock):
        assert clock == (Clock.contest(50) if player is b else UNTIMED)

        def build():
            made.append(FirstLegal(now, 0.25))
            return made[-1]

        return build

    records: list[GameRecord] = []
    play_sequential(spec, "m-x", [0, 1, 5], records.append, make=make)
    assert [r.number for r in records] == [0, 1, 5]
    assert [r.black for r in records] == [a.id, b.id, b.id]
    assert len(made) == 6 and all(p.closed for p in made)
    openings = spec.choose_openings()
    for r in records:
        assert r.start == openings[r.number // 2].label
        assert r.moves[: r.opening_moves] == list(openings[r.number // 2].moves)
        assert len(r.move_seconds) == len(r.moves) - r.opening_moves
        assert r.move_seconds == [0.25] * len(r.move_seconds)
        assert r.host["cpu"] == hardware.describe()["cpu"]


def test_a_synchro_match_plays_a_starts_two_games_at_once(now):
    a, b = fake("a", {"level": 1}), fake("b", {"level": 2})
    spec = MatchSpec(a, b, openings=1, synchro=True, starts="random:14")
    makes = []

    def make(player, clock):
        makes.append(player.label)
        return lambda: FirstLegal(now)

    records: list[GameRecord] = []
    play_sequential(spec, "m-x", [0, 1], records.append, make=make)
    # a maker each for the pair's second game, as a network can't search two trees at once
    assert makes == ["a", "b", "a", "b"]
    assert [r.number for r in records] == [0, 1]
    assert records[0].moves == records[1].moves


# --- the store -------------------------------------------------------------------------------


def test_the_store_keeps_a_player_once_by_its_config(tmp_path):
    store = Store(tmp_path / "e.db")
    config = {"kind": "engine", "engine": "edax", "budget": {"level": 8}}
    assert store.add_player(config, "x") == store.add_player(dict(config), "y")
    assert store.add_player(config, "x") == player_id(config)
    assert [p.label for p in store.players()] == ["x"]


def test_a_game_is_kept_once_however_often_its_recorded(tmp_path):
    store = Store(tmp_path / "e.db")
    for p in ("p-a", "p-b"):
        store.add_player({"kind": "engine", "id": p}, p)
    store.db.execute("UPDATE players SET id = label")
    store.db.commit()
    store.start_match("m-1", "p-a", "p-b", {}, "local", "abc", 4)
    assert store.record_games([record("m-1", 0), record("m-1", 1)]) == 2
    assert (
        store.import_records(
            [record("m-1", 1).to_json(), "", record("m-1", 2).to_json()]
        )
        == 1
    )
    assert store.numbers("m-1") == {0, 1, 2}
    back = store.games("m-1")[0]
    assert back == record("m-1", 0, finished=back.finished, load=None)
    # a host is kept once
    assert store.db.execute("SELECT COUNT(*) FROM hosts").fetchone()[0] == 1
    # a match started again is left as it was
    store.start_match("m-1", "p-a", "p-b", {"other": 1}, "gcp", "def", 9)
    match = store.match("m-1")
    assert match is not None and (match.planned, match.played) == (4, 3)


def test_a_hosts_description_is_the_same_each_time():
    assert hardware.describe() == hardware.describe()
    assert {"cpu", "isa", "affinity", "physical_cores", "placement"} <= set(
        hardware.describe()
    )


# --- the CLI ---------------------------------------------------------------------------------


def test_a_summary_scores_player_a_whichever_color(tmp_path):
    store = Store(tmp_path / "e.db")
    for p in ("p-a", "p-b"):
        store.add_player({"kind": "engine", "id": p}, p)
    store.db.execute("UPDATE players SET id = label")
    store.db.commit()
    store.start_match("m-1", "p-a", "p-b", {}, "local", "abc", 4)
    # a wins as black by 10, loses as white by 4 (black's +4), draws, and wins on b's timeout
    store.record_games(
        [
            record("m-1", 0, margin=10, result=10),
            record("m-1", 1, margin=4, result=4),
            record("m-1", 2, margin=0, result=0),
            record("m-1", 3, margin=6, result=-1, flagged="b"),
        ]
    )
    summary = evaluate.summary(store, "m-1")
    assert (summary["wins"], summary["draws"], summary["losses"]) == (2, 1, 1)
    assert summary["board_score"] == 0.375
    assert summary["timeouts"] == {"a": 0, "b": 1}
    assert summary["margins"] == [10, -4, 0, -6]
    assert summary["pairs"]["pairs"] == 2
    assert summary["a"]["seconds_a_game"] == 0.25


def test_cloud_jobs_play_shares_of_the_match_and_send_their_games(engine_dir):
    a = resolve(f"edax:level=1,dir={engine_dir}", 8)
    spec = MatchSpec(a, a, openings=4)
    jobs = evaluate.cloud_jobs(spec, "m-1", "gcp:x", 2, {3})
    assert [j["name"] for j in jobs] == ["m-1-0", "m-1-1"]
    cmd = jobs[1]["cmd"]
    assert (
        cmd[cmd.index("--slice") + 1] == "1/2" and cmd[cmd.index("--skip") + 1] == "3"
    )
    assert jobs[1]["outputs"] == [cmd[cmd.index("--out") + 1]]
    match = json.loads(cmd[cmd.index("--match") + 1])
    assert MatchSpec.from_dict(match["spec"]) == spec


def test_a_hosts_lines_come_into_the_store(tmp_path):
    store = Store(tmp_path / "e.db")
    for p in ("p-a", "p-b"):
        store.add_player({"kind": "engine", "id": p}, p)
    store.db.execute("UPDATE players SET id = label")
    store.db.commit()
    store.start_match("m-1", "p-a", "p-b", {}, "gcp", "abc", 4)
    lines = [
        record("m-1", 0).to_json(),
        json.dumps({"match": "m-1", "run": {"seconds": 3}, "host": {"cpu": "x"}}),
        "not json",
        record("m-1", 1).to_json(),
    ]
    assert evaluate.sync_lines(store, lines) == 2
    assert evaluate.sync_lines(store, lines) == 0
    assert store.db.execute("SELECT COUNT(*) FROM match_runs").fetchone()[0] == 2


def test_a_host_plays_its_share_and_picks_up_where_it_stopped(monkeypatch, tmp_path):
    a, b = fake("a", {"level": 1}), fake("b", {"level": 2})
    spec = MatchSpec(a, b, openings=3)
    played = []

    def play(spec_, match_id, numbers, emit, record_run, parallel):
        played.append(list(numbers))
        for n in numbers:
            emit(record(match_id, n))

    monkeypatch.setattr(evaluate, "play", play)
    out = tmp_path / "out.jsonl"
    args = Namespace(
        match=json.dumps({"id": "m-1", "spec": spec.to_dict()}),
        out=str(out),
        skip="1",
        slice="0/1",
        parallel=1,
    )
    evaluate.host_play(args)
    evaluate.host_play(args)
    assert played == [[0, 2, 3, 4, 5], []]
    assert [json.loads(line)["number"] for line in out.read_text().splitlines()] == [
        0,
        2,
        3,
        4,
        5,
    ]


def test_replay_leaves_a_whole_game_whole():
    moves = [37, 43, 34]
    game = replay(moves, 1, None, None)
    assert game.moves == moves and game.solved_at is None
    assert len(game.move_seconds) == 2


def test_an_opening_is_a_label_moves_and_maybe_a_start():
    assert Opening("xot:1", (1, 2), None).label == "xot:1"


def test_jax_stays_off_the_gpu_unless_a_player_is_on_it(tmp_path):
    config = tmp_path / "match.toml"
    config.write_text('a = "net:net=x,visits=10"\nb = "edax:level=8"\n')
    assert not evaluate.wants_gpu(["run", "--config", str(config)])
    assert not evaluate.wants_gpu(["run", "--a", "net:net=x,device=cpu,visits=1"])
    assert evaluate.wants_gpu(["run", "--a", "net:net=x,device=gpu-batched,visits=1"])
    config.write_text('[a]\nkind = "net"\ndevice = "gpu"\n')
    assert evaluate.wants_gpu(["run", "--config", str(config)])
    match = tmp_path / "match.json"
    match.write_text(json.dumps({"spec": {"a": {"config": {"device": "gpu"}}}}))
    assert evaluate.wants_gpu(["play", "--match", f"@{match}"])
    assert evaluate.wants_gpu(["play", "--match", match.read_text()])
    assert evaluate.wants_gpu(["run", "--a", "pgx:visits=16", "--b", "edax:level=1"])
    config.write_text('a = "pgx:visits=16"\nb = "edax:level=1"\n')
    assert evaluate.wants_gpu(["run", "--config", str(config)])
    match.write_text(json.dumps({"spec": {"a": {"config": {"kind": "pgx"}}}}))
    assert evaluate.wants_gpu(["play", "--match", f"@{match}"])
    # an engine's options that merely mention pgx aren't a pgx player
    assert not evaluate.wants_gpu(["run", "--a", "edax:level=1,args=--note=nopgx:x"])


def test_games_in_parallel_are_the_games_one_at_a_time(now):
    a, b = fake("a", {"level": 1}), fake("b", {"level": 2})
    spec = MatchSpec(a, b, openings=3, seed=2)

    def make(player, clock):
        return lambda: FirstLegal([0.0])

    alone: list[GameRecord] = []
    together: list[GameRecord] = []
    play_sequential(spec, "m-x", numbers_of(3), alone.append, make=make)
    play_parallel(spec, "m-x", numbers_of(3), together.append, 4, make=make)
    by_number = {r.number: r.moves for r in together}
    assert by_number == {r.number: r.moves for r in alone}


def test_networks_on_the_cpu_play_one_game_at_a_time():
    net = fake("n", {"visits": 10}, kind="net", device="cpu")
    spec = MatchSpec(net, fake("e", {"level": 1}), openings=1)
    with pytest.raises(ValueError, match="one game at a time"):
        play_parallel(spec, "m-x", [0, 1], lambda r: None, 2)


def test_pgx_plays_only_in_batches_at_fixed_visits(engine_dir):
    pgx = resolve("pgx:visits=64", 8)
    assert pgx.device == "gpu-batched" and pgx.config["model"] == "othello_v0"
    assert resolve("pgx:visits=64,tree_factor=4", 8).id == pgx.id
    with pytest.raises(ValueError, match="fixed visits"):
        resolve("pgx:clock=10", 8)
    edax = resolve(f"edax:level=8,dir={engine_dir}", 8, batched=True)
    assert MatchSpec(pgx, edax, openings=1).batched
    cpu_edax = fake("e", {"level": 1}, threads=1)
    with pytest.raises(ValueError, match="only in batched"):
        MatchSpec(
            fake("p", {"visits": 1}, kind="pgx", device="cpu"), cpu_edax, openings=1
        )


def test_timings_are_each_fixed_budget_players_side_on_one_host(tmp_path):
    store = Store(tmp_path / "e.db")
    engine = {"kind": "engine", "engine": "edax", "threads": 8, "budget": {"level": 20}}
    clocked = {
        "kind": "engine",
        "engine": "edax",
        "threads": 8,
        "budget": {"clock": {}},
    }
    me, other = store.add_player(engine, "edax L20"), store.add_player(clocked, "c")
    store.start_match("m-self", me, me, {}, "local", "abc", 4)
    store.start_match("m-vs", me, other, {}, "local", "abc", 2)
    here, there = {"cpu": "here"}, {"cpu": "there"}
    # moves: an opening move, then black's, white's, black's
    timed = {"moves": [37, 43, 34, 29], "opening_moves": 1, "black": me, "white": me}
    store.record_games(
        [
            record("m-self", 0, **timed, move_seconds=[1.0, 2.0, 3.0], host=here,
                   black_seconds=4.0, white_seconds=2.0),
            record("m-self", 1, **timed, move_seconds=[1.0, 1.0, 1.0], host=here,
                   black_seconds=2.0, white_seconds=1.0),
            # elsewhere, solved, or played in a batch: not timings
            record("m-self", 2, **timed, move_seconds=[9.0, 9.0, 9.0], host=there,
                   black_seconds=18.0, white_seconds=9.0),
            record("m-self", 3, **timed, move_seconds=[9.0] * 3, host=here,
                   black_seconds=18.0, white_seconds=9.0, ending="solved"),
            record("m-vs", 0, moves=[37, 43, 34, 29], opening_moves=1, black=me,
                   white=other, move_seconds=[5.0] * 3, host=here,
                   black_seconds=10.0, white_seconds=5.0),
        ]
    )  # fmt: skip
    from core.evals.store import host_id

    (config,) = evaluate.timing_configs(store, host_id(here))
    assert (config["engine"], config["level"], config["games"]) == ("edax", 20, 2)
    assert config["seconds_per_game"] == 2.25
    assert config["seconds_per_move"] == 1.5
    assert config["moves_per_game"] == 1.5
    with_any = evaluate.timing_configs(store, host_id(here), any_opponent=True)
    assert [c["games"] for c in with_any] == [3]


def test_games_in_parallel_are_emitted_in_the_callers_thread(now):
    import threading

    a, b = fake("a", {"level": 1}), fake("b", {"level": 2})
    spec = MatchSpec(a, b, openings=2)
    threads = []

    def emit(record):
        threads.append(threading.current_thread())

    play_parallel(
        spec, "m-x", numbers_of(2), emit, 3, make=lambda p, c: lambda: FirstLegal(now)
    )
    assert threads == [threading.current_thread()] * 4


def test_pgx_makes_a_match_batched_its_engines_a_thread_each(engine_dir):
    spec = evaluate.build_match(
        {"a": "pgx:visits=16", "b": f"edax:level=1,dir={engine_dir}"}, cores=8
    )
    assert spec.batched and spec.b.config["threads"] == 1
