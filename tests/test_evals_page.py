"""The evaluation store's web page: what its server answers, from a store of a few games."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from core.evals.page import Evals, make_handler
from core.evals.store import GameRecord, Store

# a contest's clock: games between players on it are competition games
CLOCK = {"clock": {"ini": 30.0, "inc": 0.0, "ext": None}}


def engine(store, name, budget=CLOCK):
    return store.add_player({"kind": "engine", "n": name, "budget": budget}, name)


def game(match, number, black, white, result, finished, start="xot:0"):
    return GameRecord(
        match=match,
        number=number,
        black=black,
        white=white,
        start=start,
        moves=[19],
        opening_moves=1,
        move_seconds=[],
        ending="solved",
        solved_at=20,
        margin=result,
        result=result,
        finished=finished,
    )


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "evals.db"
    store = Store(path)
    x, y, z = (engine(store, n) for n in "xyz")
    engine(store, "idle")
    for match, a, b in (("m-xy", x, y), ("m-xz", x, z)):
        store.start_match(match, a, b, {}, "local", "abc", 2)
    store.record_games(
        [
            game("m-xy", 0, x, y, 4, 1.0),
            game("m-xy", 1, y, x, 0, 2.0),
            game("m-xz", 0, z, x, 2, 3.0),
            game("m-xz", 1, x, z, 10, 4.0),
        ]
    )
    store.close()
    return path


def ids(path):
    store = Store(path)
    found = {p.label: p.id for p in store.players()}
    store.close()
    return found


def test_the_summary_has_each_players_record_and_the_latest_games_first(path):
    summary = Evals(path).summary()
    assert summary["games"] == 4
    records = {
        p["name"]: (p["wins"], p["draws"], p["losses"]) for p in summary["players"]
    }
    assert records == {
        "x": (2, 1, 1),
        "y": (0, 1, 1),
        "z": (1, 0, 1),
        "idle": (0, 0, 0),
    }
    elos = [p["elo"] for p in summary["players"]]
    assert elos[:3] == sorted(elos[:3], reverse=True) and elos[3] is None
    assert [p["pool"] for p in summary["players"]] == [1, 1, 1, None]
    assert [g["finished"] for g in summary["recent"]] == [4.0, 3.0, 2.0, 1.0]
    last = summary["recent"][-1]
    assert (last["black_name"], last["white_name"], last["result"]) == ("x", "y", 4)
    assert (last["ending"], last["solved_at"]) == ("solved", 20)
    # solved for the result alone: its margin's size means nothing
    assert last["margin"] is None


def test_a_players_page_has_its_record_against_each_opponent(path):
    evals = Evals(path)
    page = evals.player(ids(path)["x"])
    assert page is not None
    assert (page["player"]["wins"], page["player"]["games"]) == (2, 4)
    against = {
        o["name"]: (o["wins"], o["draws"], o["losses"]) for o in page["opponents"]
    }
    assert against == {"y": (1, 1, 0), "z": (1, 0, 1)}
    assert all(o["games"] == 2 and o["diff_sd"] > 0 for o in page["opponents"])
    assert evals.player("p-nobody") is None


def test_players_who_never_met_even_through_others_are_rated_apart(path):
    store = Store(path)
    v, w = (engine(store, n) for n in "vw")
    store.start_match("m-vw", v, w, {}, "local", "abc", 2)
    store.record_games([game("m-vw", i, v, w, 2, 5.0 + i) for i in range(4)])
    store.close()
    players = {p["name"]: p for p in Evals(path).summary()["players"]}
    assert {players[n]["pool"] for n in "xyz"} == {1} and players["v"]["pool"] == 2
    assert players["v"]["elo"] > 0
    assert players["v"]["elo"] == pytest.approx(-players["w"]["elo"])
    assert players["v"]["sd"] < 200


def test_new_games_are_seen_on_the_next_read(path):
    evals = Evals(path)
    assert evals.summary()["games"] == 4
    found = ids(path)
    store = Store(path)
    store.start_match("m-yz", found["y"], found["z"], {}, "local", "abc", 2)
    store.record_games([game("m-yz", 0, found["y"], found["z"], -6, 5.0)])
    store.close()
    summary = evals.summary()
    assert summary["games"] == 5 and summary["recent"][0]["white_name"] == "z"


@pytest.fixture
def url(path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Evals(path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def status(url, path):
    try:
        with urllib.request.urlopen(url + path) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_a_game_is_there_move_by_move_with_what_its_searches_said(path):
    store = Store(path)
    found = {p.label: p.id for p in store.players()}
    store.start_match("m-yz", found["y"], found["z"], {}, "local", "abc", 2)
    said = [{"q": 0.25, "visits": 100, "added": 60}, None]
    record = game("m-yz", 0, found["y"], found["z"], -6, 5.0)
    record.moves, record.move_seconds, record.move_info = (
        [19, 18, 17],
        [0.5, 0.25],
        said,
    )
    store.record_games([record, game("m-yz", 1, found["z"], found["y"], 2, 6.0)])
    # kept as it came, and only for the game that had it
    assert [g.move_info for g in store.games("m-yz")] == [said, []]
    ids_ = [
        r[0] for r in store.db.execute("SELECT id FROM games WHERE match_id = 'm-yz'")
    ]
    store.close()
    evals = Evals(path)
    shown = evals.game(ids_[0])
    assert shown is not None
    g = shown["game"]
    assert (g["black_name"], g["white_name"], g["moves"]) == ("y", "z", [19, 18, 17])
    assert (g["move_seconds"], g["move_info"]) == ([0.5, 0.25], said)
    assert evals.game(ids_[1])["game"]["move_info"] == []  # pyright: ignore[reportOptionalSubscript]
    assert evals.game(10**9) is None
    # and on its players' pages, latest first
    games = evals.player(found["y"])["games"]  # pyright: ignore[reportOptionalSubscript]
    assert [x["finished"] for x in games] == [6.0, 5.0, 2.0, 1.0]


def test_the_server_answers_the_page_and_its_questions(url, path):
    code, page = status(url, "/")
    assert code == 200 and b"<html" in page
    code, body = status(url, "/api/summary")
    assert code == 200 and len(json.loads(body)["players"]) == 4
    assert status(url, f"/api/player?id={ids(path)['x']}")[0] == 200
    assert status(url, "/api/player?id=p-nobody")[0] == 404
    assert status(url, "/api/player")[0] == 404
    assert status(url, "/api/game?id=1")[0] == 200
    assert status(url, "/api/game?id=999")[0] == 404
    assert status(url, "/api/game?id=x")[0] == 404
    code, body = status(url, "/api/openings")
    assert code == 200 and json.loads(body)["openings"][0]["start"] == "xot:0"
    assert status(url, "/api/opening?start=xot%3A0")[0] == 200
    assert status(url, "/api/opening?start=xot%3A9")[0] == 404
    assert json.loads(status(url, "/api/summary?games=all")[1])["games"] == 4
    assert status(url, "/nothing")[0] == 404


def test_competition_games_alone_count_unless_every_game_is_asked_for(path):
    store = Store(path)
    x = ids(path)["x"]
    fast = engine(store, "fast", {"visits": 128})
    short = engine(store, "short", {"clock": {"ini": 5.0, "inc": 0.0, "ext": None}})
    for match, other in (("m-fast", fast), ("m-short", short)):
        store.start_match(match, x, other, {}, "local", "abc", 2)
        store.record_games([game(match, 0, x, other, 2, 5.0)])
    store.close()
    evals = Evals(path)
    shown = {p["name"]: p["games"] for p in evals.summary()["players"]}
    assert shown["x"] == 4 and shown["fast"] == shown["short"] == 0
    every = evals.summary("all")
    assert every["games"] == 6
    assert {p["name"]: p["games"] for p in every["players"]}["fast"] == 1
    assert len(evals.player(x)["opponents"]) == 2  # pyright: ignore[reportOptionalSubscript]
    assert len(evals.player(x, "all")["opponents"]) == 4  # pyright: ignore[reportOptionalSubscript]
    page = evals.player(fast)
    assert page is not None and page["player"]["elo"] is None and page["games"] == []


def test_an_opening_favours_a_color_when_black_scores_far_from_half(path):
    store = Store(path)
    found = ids(path)
    x, y = found["x"], found["y"]
    store.start_match("m-open", x, y, {}, "local", "abc", 20)
    # black wins all 8 of xot:1's games, each player black in half, solved for their margin;
    # xot:2 is split
    favoured = [
        game("m-open", i, *((x, y) if i % 2 else (y, x)), 6, 10.0 + i, "xot:1")
        for i in range(8)
    ]
    for record in favoured:
        record.solve_exact = True
    store.record_games(
        favoured
        + [
            game("m-open", 8 + i, x, y, 4 if i % 2 else -4, 20.0 + i, "xot:2")
            for i in range(8)
        ]
    )
    store.close()
    evals = Evals(path)
    openings = {o["start"]: o for o in evals.openings()["openings"]}
    one, two = openings["xot:1"], openings["xot:2"]
    assert (one["games"], one["black"], one["score"], one["favours"]) == (
        8,
        8,
        1.0,
        "black",
    )
    assert one["sd"] == pytest.approx(0.5 / 8**0.5) and one["margin"] == 6
    assert (two["score"], two["favours"]) == (0.5, None)
    # the most played first
    assert next(iter(openings)) in ("xot:1", "xot:2")

    shown = evals.opening("xot:1")
    assert shown is not None
    assert shown["moves"] == [19] and len(shown["games"]) == 8
    players = {p["name"]: (p["black"], p["white"]) for p in shown["players"]}
    assert players == {"x": ([4, 0, 0], [0, 0, 4]), "y": ([4, 0, 0], [0, 0, 4])}
    assert evals.opening("xot:404") is None

    first = Store(path)
    game_id = first.db.execute(
        "SELECT id FROM games WHERE start = 'xot:1' LIMIT 1"
    ).fetchone()[0]
    first.close()
    assert evals.game(game_id)["game"]["opening"]["favours"] == "black"  # pyright: ignore[reportOptionalSubscript]
