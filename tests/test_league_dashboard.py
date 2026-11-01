"""The league's web page: what its server answers, from a store of a few games."""

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from core.league.dashboard import League, make_handler, pareto
from core.league.game import LeagueGame
from core.league.ratings import fit_ratings
from core.league.runtime import FILES, runtime_dir
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


def test_a_network_links_to_its_url(store, url, tmp_path):
    checkpoint = tmp_path / "net.eqx"
    checkpoint.write_bytes(b"weights")
    store.add_network("net", checkpoint, "resnet:4x64", "run")
    store.add_net_player("net", 128)
    store.set_network_url("net", "https://example.com/run/1")
    players = get(url, "/api/summary")["players"]
    assert {p["name"]: p["url"] for p in players} == {
        "x": None,
        "y": None,
        "net@128": "https://example.com/run/1",
    }


def post(url, path, body):
    request = urllib.request.Request(
        url + path,
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def test_a_player_can_be_paused_and_resumed(url):
    x = get(url, "/api/summary")["players"][0]
    assert not x["retired"] and x["can_play"]
    post(url, "/api/paused", {"player": x["id"], "paused": True})
    assert get(url, "/api/summary")["players"][0]["retired"]
    post(url, "/api/paused", {"player": x["id"], "paused": False})
    assert not get(url, "/api/summary")["players"][0]["retired"]


def test_pausing_nobody_is_a_404(url):
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(url, "/api/paused", {"player": 999, "paused": True})
    assert caught.value.code == 404


def test_a_player_that_cant_play_here_cant_be_resumed(store):
    class OnlySomePlay(FakeGame):
        def can_play(self, player):
            return player.name != "x"

    league = League(store, OnlySomePlay())
    x, y = store.players()
    assert not league.set_paused(x.id, False)
    assert league.set_paused(y.id, True)
    assert store.players()[1].retired


def test_linking_an_unknown_network_fails(store):
    with pytest.raises(ValueError):
        store.set_network_url("nobody", "https://example.com")


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


def test_summary_says_which_batch_is_playing(url, store):
    assert get(url, "/api/summary")["playing"] is None
    playing = {"state": "playing", "a": "x", "b": "y", "games": 64, "started": 1.0}
    store.set_meta("player", {**playing, "pid": os.getpid()})
    assert get(url, "/api/summary")["playing"]["a"] == "x"
    # the player died without saying so
    gone = subprocess.Popen(["true"])
    gone.wait()
    store.set_meta("player", {**playing, "pid": gone.pid})
    assert get(url, "/api/summary")["playing"] is None
    store.set_meta("player", {"state": "idle", "reason": "stopped"})
    assert get(url, "/api/summary")["playing"] is None


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


def test_summary_lists_the_pending_series(url, store):
    assert get(url, "/api/summary")["requests"] == []
    store.request_match(1, 2, 40)
    pending = get(url, "/api/summary")["requests"]
    assert [(r["a"], r["b"], r["played"], r["games"]) for r in pending] == [
        (1, 2, 0, 40)
    ]


def refused(url, path, body):
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(url, path, body)
    return caught.value.code


def test_writing_needs_a_json_body(url):
    request = urllib.request.Request(
        url + "/api/paused", method="POST", data=b"player=1&paused=1"
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request)
    assert caught.value.code == 415
    assert not get(url, "/api/summary")["players"][0]["retired"]


def test_notes_are_edited_and_shown(url):
    post(url, "/api/notes", {"player": 1, "description": "deep", "tags": ["t"]})
    x = get(url, "/api/summary")["players"][0]
    assert (x["description"], x["tags"]) == ("deep", ["t"])


def test_a_series_is_added_moved_edited_and_cancelled(url):
    first = post(url, "/api/match/add", {"a": 1, "b": 2, "games": 4})["id"]
    second = post(url, "/api/match/add", {"a": 2, "b": 1, "games": 4})["id"]
    post(url, "/api/match/move", {"id": second, "index": 0})
    post(url, "/api/match/games", {"id": first, "games": 8})
    requests = get(url, "/api/summary")["requests"]
    assert [(r["id"], r["games"]) for r in requests] == [(second, 4), (first, 8)]
    post(url, "/api/match/cancel", {"id": second})
    assert [r["id"] for r in get(url, "/api/summary")["requests"]] == [first]


@pytest.mark.parametrize(
    "body",
    [
        {"a": 1, "b": 2, "games": 3},
        {"a": 1, "b": 2, "games": 0},
        {"a": 1, "b": 1, "games": 2},
        {"a": 1, "b": 99, "games": 2},
    ],
)
def test_a_series_that_cant_be_played_is_refused(url, body):
    assert refused(url, "/api/match/add", body) == 400


def test_a_paused_player_cant_be_asked_to_play(url):
    post(url, "/api/paused", {"player": 1, "paused": True})
    assert refused(url, "/api/match/add", {"a": 1, "b": 2, "games": 2}) == 400


def test_a_series_cant_be_given_an_odd_number_of_games(url, store):
    request = store.request_match(1, 2, 4)
    assert refused(url, "/api/match/games", {"id": request, "games": 3}) == 400
    assert refused(url, "/api/match/move", {"id": 99, "index": 0}) == 400


def test_the_queue_says_whats_happening_to_each_series(url, store):
    playing = store.request_match(1, 2, 4)
    next_up = store.request_match(1, 2, 4)
    waiting = store.request_match(1, 2, 4)
    statuses = lambda: {
        r["id"]: r["status"] for r in get(url, "/api/summary")["requests"]
    }
    assert statuses() == {playing: "next", next_up: "", waiting: ""}
    store.set_meta(
        "player",
        {"state": "playing", "a": "x", "b": "y", "games": 4, "started": 0.0,
         "pid": os.getpid(), "request": next_up},
    )  # fmt: skip
    assert statuses() == {playing: "next", next_up: "playing", waiting: ""}
    store.move_match(next_up, 0)
    assert statuses() == {next_up: "playing", playing: "", waiting: ""}
    store.set_retired("x")
    assert statuses() == {next_up: "playing", playing: "paused", waiting: "paused"}


def test_a_series_a_player_cant_play_is_marked(store):
    class OnlyYPlays(FakeGame):
        def can_play(self, player):
            return player.name == "y"

    request = store.request_match(1, 2, 2)
    league = League(store, OnlyYPlays())
    assert league.summary()["requests"][0] == {
        "id": request, "a": 1, "b": 2, "games": 2, "played": 0, "status": "can't play",
    }  # fmt: skip


def status_of(url, path):
    try:
        with urllib.request.urlopen(url + path) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, b""


def test_the_pages_scripts_are_served(url):
    status, headers, body = status_of(url, "/web/app.js")
    assert status == 200 and headers["Content-Type"] == "text/javascript" and body


@pytest.mark.parametrize(
    "path",
    [
        "/web/../store.py",
        "/web/%2e%2e/store.py",
        "/onnx/../league.db",
        "/onnx/nothex.onnx",
        "/onnx/" + "a" * 64 + ".onnx",
        "/ort/../../league.db",
        "/ort/missing.wasm",
    ],
)
def test_only_the_files_for_the_page_are_served(url, path):
    assert status_of(url, path)[0] == 404


def test_a_networks_export_and_the_runtime_are_served_once_they_exist(
    url, store, tmp_path
):
    checkpoint = tmp_path / "net.eqx"
    checkpoint.write_bytes(b"weights")
    store.add_network("net", checkpoint, "resnet:4x64", "run")
    store.add_net_player("net", 128)
    summary = get(url, "/api/summary")
    assert not summary["runtime"] and summary["players"][-1]["onnx"] is None
    player = store.players()[-1]
    exported = store.onnx_path(player)
    exported.write_bytes(b"graph")
    name = get(url, "/api/summary")["players"][-1]["onnx"]
    assert name == exported.name
    status, headers, body = status_of(url, f"/onnx/{name}")
    assert (status, body) == (200, b"graph")
    assert "immutable" in headers["Cache-Control"]
    out = runtime_dir(store.root)
    out.mkdir(parents=True)
    for file in FILES:
        (out / file).write_bytes(b"x")
    assert get(url, "/api/summary")["runtime"]
    assert (
        status_of(url, "/ort/ort-wasm-simd-threaded.wasm")[1]["Content-Type"]
        == "application/wasm"
    )


def test_a_game_says_who_played_it(url):
    assert get(url, "/api/game?id=1")["first_id"] == 1


NET = "resnet:4x64"


@pytest.fixture
def timed(tmp_path):
    """Four players, of whom a is weakest and fastest, c strongest, and d beaten by both of them."""
    store = Store(tmp_path / "timed")
    checkpoint = tmp_path / "net.eqx"
    checkpoint.write_bytes(b"weights")
    store.add_network("net", checkpoint, NET, "run")
    a, b = store.add_net_player("net", 128), store.add_net_player("net", 1000)
    c = store.add_external_player("c", "edax", {"level": 20})
    d = store.add_external_player("d", "edax", {"level": 4})
    games = []
    for winner, loser, count in ((b, a, 30), (c, b, 30), (a, d, 30)):
        games += [GameRecord(winner, loser, 0, 1, [])] * count + [
            GameRecord(loser, winner, 0, -1, [])
        ] * 10
        games += [GameRecord(loser, winner, 0, 1, [])] * 5
    store.record_batch(a, b, 1.0, False, "abc", "", games)
    yield store, (a, b, c, d)
    store.close()


def run_of(*configs, **fields):
    return {
        "name": "r",
        "hardware": {"cpu": "fake"},
        "configs": list(configs),
        **fields,
    }


def measured(**fields):
    return {"source": "measured", "plot": True, "seconds_per_game": 1.0, **fields}


def by_name(store, timings):
    players = store.players()
    ratings = fit_ratings([p.id for p in players], store.pair_records())
    return [
        {p["name"]: p for p in run["points"]}
        for run in pareto(timings, players, ratings)
    ]


def test_timings_join_networks_by_architecture_and_sims_and_engines_by_level(timed):
    store, _ = timed
    [points] = by_name(store, {"runs": [run_of(
        measured(net=NET, sims=1000, seconds_per_game=2.0),
        measured(engine="edax", level=20, seconds_per_game=9.0),
        measured(net="other", sims=128),
        measured(engine="other", level=4),
    )]})  # fmt: skip
    assert {name: p["seconds"] for name, p in points.items()} == {
        "net@1000": 2.0,
        "c": 9.0,
    }


def test_only_whats_measured_and_marked_for_plotting_is_plotted(timed):
    store, _ = timed
    [points] = by_name(store, {"runs": [run_of(
        {**measured(net=NET, sims=128), "source": "fit"},
        {**measured(net=NET, sims=1000), "plot": False},
    )]})  # fmt: skip
    assert points == {}


def test_runs_stay_apart(timed):
    store, _ = timed
    runs = by_name(store, {"runs": [
        run_of(measured(net=NET, sims=128)),
        run_of(measured(net=NET, sims=1000)),
    ]})  # fmt: skip
    assert [set(points) for points in runs] == [{"net@128"}, {"net@1000"}]


def test_the_frontier_is_who_nothing_is_both_faster_and_stronger_than(timed):
    store, _ = timed
    [points] = by_name(store, {"runs": [run_of(
        measured(net=NET, sims=128, seconds_per_game=1.0),
        measured(net=NET, sims=1000, seconds_per_game=4.0),
        measured(engine="edax", level=20, seconds_per_game=8.0),
        measured(engine="edax", level=4, seconds_per_game=2.0),
    )]})  # fmt: skip
    assert {name for name, p in points.items() if p["frontier"]} == {
        "net@128",
        "net@1000",
        "c",
    }
    assert "d" in points


def test_an_unverified_batch_is_shown_but_isnt_the_frontier(timed):
    store, _ = timed
    [points] = by_name(store, {"runs": [run_of(
        measured(net=NET, sims=128, seconds_per_game=4.0),
        measured(net=NET, sims=1000, seconds_per_game=1.0, batch_strength="unmeasured"),
    )]})  # fmt: skip
    assert not points["net@1000"]["solid"] and not points["net@1000"]["frontier"]
    assert points["net@128"]["frontier"]


def test_there_are_no_points_without_timings(url):
    assert get(url, "/api/pareto") == []


def test_a_timings_file_is_read_again_when_it_changes(url, store):
    path = store.root / "timings.json"
    path.write_text(json.dumps({"runs": [run_of(measured(engine="edax", level=8))]}))
    assert [p["name"] for p in get(url, "/api/pareto")[0]["points"]] == ["x"]
    path.write_text("{ half written")
    assert get(url, "/api/pareto") == []
    path.write_text(json.dumps({"runs": []}))
    os.utime(path, (1, 1))
    assert get(url, "/api/pareto") == []
