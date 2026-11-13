"""The league's web page: what its server answers, from a store of a few games."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from core.league.dashboard import League, make_handler
from core.league.store import GameRecord, Store


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "league")
    x = store.add_external_player("x", "edax", {"level": 8})
    y = store.add_external_player("y", "edax", {"level": 4})
    z = store.add_external_player("z", "edax", {"level": 1})
    store.add_external_player("idle", "edax", {"level": 2})
    store.record_batch(
        x, y, 12.0, False, "abc", "float32",
        [GameRecord(x, y, 3, 1, [1], 40.0, 24.0), GameRecord(y, x, 3, 0, [2])],
    )  # fmt: skip
    store.record_batch(
        x, z, 12.0, False, "abc", "float32",
        [GameRecord(z, x, 4, 1, [3]), GameRecord(x, z, 4, 1, [4])],
    )  # fmt: skip
    yield store
    store.close()


def test_the_summary_has_each_players_record_and_the_latest_games_first(store):
    summary = League(store, "Fake").summary()
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
    assert elos == sorted(elos, reverse=True)
    assert [g["id"] for g in summary["recent"]] == [4, 3, 2, 1]
    last = summary["recent"][-1]
    assert (last["first_name"], last["second_name"], last["result"]) == ("x", "y", 1)
    assert (last["first_score"], last["second_score"]) == (40.0, 24.0)


def test_a_players_page_has_its_record_against_each_opponent(store):
    league = League(store, "Fake")
    x = next(p.id for p in store.players() if p.name == "x")
    page = league.player(x)
    assert page is not None
    assert (page["player"]["wins"], page["player"]["games"]) == (2, 4)
    against = {
        o["name"]: (o["wins"], o["draws"], o["losses"]) for o in page["opponents"]
    }
    assert against == {"y": (1, 1, 0), "z": (1, 0, 1)}
    assert all(o["games"] == 2 and o["diff_sd"] > 0 for o in page["opponents"])
    assert league.player(999) is None


def test_new_games_are_seen_on_the_next_read(store):
    league = League(store, "Fake")
    assert league.summary()["games"] == 4
    y, z = (p.id for p in store.players() if p.name in ("y", "z"))
    store.record_batch(
        y, z, 1.0, False, "abc", "float32", [GameRecord(y, z, 5, 1, [5])]
    )
    summary = league.summary()
    assert summary["games"] == 5 and summary["recent"][0]["first_name"] == "y"


@pytest.fixture
def url(store):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(League(store, "Fake")))
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


def test_the_server_answers_the_page_and_its_questions(url):
    code, page = status(url, "/")
    assert code == 200 and b"<html" in page
    code, body = status(url, "/api/summary")
    assert code == 200 and json.loads(body)["game"] == "Fake"
    assert status(url, "/api/player?id=1")[0] == 200
    assert status(url, "/api/player?id=999")[0] == 404
    assert status(url, "/api/player")[0] == 400
    assert status(url, "/api/player?id=x")[0] == 400
    assert status(url, "/nothing")[0] == 404
