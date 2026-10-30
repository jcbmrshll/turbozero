"""The league's web page: what its server answers, from a store of a few games."""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from core.league.dashboard import League, make_handler
from core.league.game import LeagueGame
from core.league.store import GameRecord, Store


class FakeGame(LeagueGame):
    name = "Fake"
    env = None

    def metadata(self, state):
        raise NotImplementedError

    @property
    def step_fn(self):
        raise NotImplementedError

    def load_network(self, path, net, dtype):
        raise NotImplementedError

    def make_evaluator(self, network, sims):
        raise NotImplementedError

    def num_openings(self):
        return 9

    def openings(self, indices):
        return np.zeros((len(indices), 2), dtype=int)


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "league")
    x = store.add_external_player("x", "edax", {"level": 8})
    y = store.add_external_player("y", "edax", {"level": 4})
    store.record_batch(
        x, y, 12.0, False, "abc", "bfloat16",
        [GameRecord(x, y, 3, 1, [1, 2, 64], 40.0, 24.0), GameRecord(y, x, 3, -1, [5, 6])],
    )  # fmt: skip
    yield store
    store.close()


@pytest.fixture
def url(store):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(League(store, FakeGame()))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url, path):
    with urllib.request.urlopen(url + path) as response:
        return json.load(response)


def test_summary_rates_the_players(url):
    summary = get(url, "/api/summary")
    assert summary["games"] == 2
    assert summary["seconds"] == 12.0
    by_name = {p["name"]: p for p in summary["players"]}
    assert by_name["x"]["elo"] > by_name["y"]["elo"]
    assert by_name["x"]["games"] == 2
    assert by_name["x"]["external"] == "edax"
    assert summary["parts"] == 1
    assert summary["recent"][0]["a"] == "x"


def test_summary_says_what_the_worker_is_doing(url, store):
    assert get(url, "/api/summary")["worker"]["state"] == "stopped"
    store.set_meta(
        "supervisor", {"state": "waiting", "reason": "busy", "time": time.time()}
    )
    worker = get(url, "/api/summary")["worker"]
    assert (worker["state"], worker["reason"]) == ("waiting", "busy")
    # one that's gone quiet isn't running, whatever it last said
    store.set_meta("supervisor", {"state": "playing", "time": time.time() - 10_000})
    assert get(url, "/api/summary")["worker"]["state"] == "stopped"


def test_ratings_follow_new_games(url, store):
    before = get(url, "/api/summary")["players"][0]["elo"]
    store.record_batch(1, 2, 1.0, False, "abc", "f", [GameRecord(1, 2, 4, 1, [1])] * 5)
    assert get(url, "/api/summary")["players"][0]["elo"] > before


def test_games_are_filtered_and_paged(url):
    assert get(url, "/api/games")["total"] == 2
    page = get(url, "/api/games?player=1&opponent=2&limit=1")
    assert page["total"] == 2
    assert len(page["games"]) == 1
    assert page["games"][0]["plies"] == 2
    assert get(url, "/api/games?player=99")["total"] == 0


def test_a_game_comes_with_its_moves(url):
    game = get(url, "/api/game?id=1")
    assert game["moves"] == [1, 2, 64]
    assert game["first"] == "x"
    assert game["opening_plies"] == 2
    assert game["first_score"] == 40.0


def test_a_missing_game_is_a_404(url):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(url, "/api/game?id=99")
    assert e.value.code == 404


def test_the_pair_view_has_uncertainty(url):
    pair = get(url, "/api/pair?a=1&b=2")
    assert pair["difference"] > 0
    assert pair["sd"] > 0
    assert 0.5 < pair["expected"] < 1


def test_the_page_is_served(url):
    with urllib.request.urlopen(url + "/") as response:
        assert b"Elo league" in response.read()
