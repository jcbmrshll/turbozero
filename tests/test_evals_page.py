"""The evaluation store's web page: what its server answers, from a store of a few games."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from core.evals.page import Evals, make_handler
from core.evals.store import GameRecord, Store


def game(match, number, black, white, result, finished):
    return GameRecord(
        match=match,
        number=number,
        black=black,
        white=white,
        start="xot:0",
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
    x, y, z = (store.add_player({"kind": "engine", "n": n}, n) for n in "xyz")
    store.add_player({"kind": "engine", "n": "idle"}, "idle")
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
    v, w = (store.add_player({"kind": "engine", "n": n}, n) for n in "vw")
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


def test_the_server_answers_the_page_and_its_questions(url, path):
    code, page = status(url, "/")
    assert code == 200 and b"<html" in page
    code, body = status(url, "/api/summary")
    assert code == 200 and len(json.loads(body)["players"]) == 4
    assert status(url, f"/api/player?id={ids(path)['x']}")[0] == 200
    assert status(url, "/api/player?id=p-nobody")[0] == 404
    assert status(url, "/api/player")[0] == 404
    assert status(url, "/nothing")[0] == 404
