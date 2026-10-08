"""The run monitor: the client logging to a live server, and the server storing, rendering and serving runs."""

import importlib
import io
import time
import urllib.error
import urllib.request
from functools import partial

import jax
import numpy as np
import pytest
from PIL import Image
from PIL.GifImagePlugin import GifImageFile

from core.common import two_player_game
from core.monitor import Episode, Monitor, Video, client
from core.monitor.renderers import pgx_two_player_episode


def cairo_available():
    try:
        importlib.import_module("cairosvg")
    except OSError:  # the cairo system library is missing
        return False
    return True


def wait_for_media(server, run_id, count, timeout=60):
    """Media entries of a run once there are `count` of them (episodes render in the background)."""
    deadline = time.time() + timeout
    while True:
        rows = server.get(f"/api/runs/{run_id}/media")["rows"]
        if len(rows) >= count or time.time() > deadline:
            return rows
        time.sleep(0.05)


def test_run_round_trip(monitor_server):
    monitor = Monitor(monitor_server.url, project="tests", name="round-trip")
    monitor.start(
        config={"batch_size": 4, "run": {"seed": 3, "num_epochs": 2}, "fn": np.mean}
    )
    monitor.log(
        0, {"loss": np.float32(1.5), "policy_loss": 2, "ignored": "not a metric"}
    )
    monitor.log(1, {"loss": 0.5})
    monitor.finish()

    [run] = monitor_server.get("/api/runs")
    assert (run["project"], run["name"], run["status"]) == (
        "tests",
        "round-trip",
        "finished",
    )
    assert (run["step"], run["seed"], run["num_epochs"]) == (1, 3, 2)
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    # functions in the config are recorded by name
    assert meta["config"]["fn"] == "mean"
    rows = monitor_server.get(f"/api/runs/{monitor.run_id}/metrics")["rows"]
    assert [(r["step"], r["loss"]) for r in rows] == [(0, 1.5), (1, 0.5)]
    assert rows[0]["policy_loss"] == 2 and "ignored" not in rows[0]


def test_log_does_not_wait_on_the_server(monitor_server, monkeypatch):
    monitor = Monitor(monitor_server.url)
    monitor.start()
    send = monitor._request

    def slow_request(*args, **kwargs):
        time.sleep(0.5)
        return send(*args, **kwargs)

    monkeypatch.setattr(monitor, "_request", slow_request)

    start = time.time()
    monitor.log(0, {"loss": 1.0})
    assert time.time() - start < 0.25
    monitor.flush()

    assert len(monitor_server.get(f"/api/runs/{monitor.run_id}/metrics")["rows"]) == 1


def test_metrics_since(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()
    for step in range(3):
        monitor.log(step, {"loss": step})
    monitor.flush()

    page = monitor_server.get(f"/api/runs/{monitor.run_id}/metrics?since=2")

    assert [r["step"] for r in page["rows"]] == [2]
    assert page["next"] == 3


def test_non_finite_metrics_are_stored_as_null(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()

    monitor.log(0, {"loss": float("nan"), "value_loss": np.inf, "policy_loss": 1.0})
    monitor.flush()

    [row] = monitor_server.get(f"/api/runs/{monitor.run_id}/metrics")["rows"]
    assert (
        row["loss"] is None and row["value_loss"] is None and row["policy_loss"] == 1.0
    )


def test_media(monitor_server):
    frames = [Image.new("RGB", (8, 8), c) for c in ("red", "blue")]
    monitor = Monitor(monitor_server.url)
    monitor.start()

    monitor.log(4, {"video": Video(frames), "board": frames[0]})
    monitor.flush()

    entries = monitor_server.get(f"/api/runs/{monitor.run_id}/media")["rows"]
    assert {(e["key"], e["step"], e["file"]) for e in entries} == {
        ("video", 4, "video-4.gif"),
        ("board", 4, "board-4.png"),
    }
    with urllib.request.urlopen(
        f"{monitor_server.url}/media/{monitor.run_id}/video-4.gif"
    ) as resp:
        assert resp.headers["Content-Type"] == "image/gif"
        gif = Image.open(io.BytesIO(resp.read()))
    assert isinstance(gif, GifImageFile)
    assert gif.n_frames == 2


@pytest.mark.skipif(
    not cairo_available(), reason="rendering needs the cairo system library"
)
def test_pgx_episode_is_rendered_by_the_server(monitor_server, ttt, scripted):
    game = jax.jit(
        partial(
            two_player_game,
            evaluator_1=scripted.first_legal,
            evaluator_2=scripted.first_legal,
            params_1=None,
            params_2=None,
            env_step_fn=ttt.step_fn,
            env_init_fn=ttt.init_fn,
            max_steps=12,
        )
    )
    _, frames, p_ids = jax.device_get(game(jax.random.PRNGKey(0)))
    monitor = Monitor(monitor_server.url)
    monitor.start()

    monitor.log(
        2, {"game": pgx_two_player_episode(p1_label="X", p2_label="O")(frames, p_ids)}
    )
    monitor.flush()

    [entry] = wait_for_media(monitor_server, monitor.run_id, 1)
    assert (entry["key"], entry["step"], entry.get("error")) == ("game", 2, None)
    with urllib.request.urlopen(
        f"{monitor_server.url}/media/{monitor.run_id}/{entry['file']}"
    ) as resp:
        gif = Image.open(io.BytesIO(resp.read()))
    # playing the first legal move, X wins down the left column on the 7th move: the
    # initial board and 7 moves, each shown for 900ms, the final board held 3 times as long
    assert isinstance(gif, GifImageFile)
    assert gif.n_frames == 1 + 7
    assert gif.info["duration"] == 900
    gif.seek(7)
    assert gif.info["duration"] == 3 * 900


def test_render_error_is_recorded(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()

    # missing every array the renderer needs
    monitor.log(0, {"game": Episode("pgx_two_player", env_id="tic_tac_toe")})
    monitor.flush()

    [entry] = wait_for_media(monitor_server, monitor.run_id, 1)
    assert entry["key"] == "game" and entry["error"].startswith("KeyError")


def test_unknown_renderer_is_rejected(monitor_server, capsys):
    monitor = Monitor(monitor_server.url)
    monitor.start()

    monitor.log(0, {"game": Episode("nope", x=[1, 2])})
    monitor.flush()

    assert "no renderer for env 'nope'" in capsys.readouterr().err
    assert monitor_server.get(f"/api/runs/{monitor.run_id}/media")["rows"] == []


def test_continued_run_is_running_again(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()
    monitor.finish()
    run_id = monitor.run_id

    # e.g. a second train_loop continuing from the first one's output
    monitor.start()
    monitor.log(1, {"loss": 1.0})
    monitor.flush()

    assert monitor.run_id == run_id
    assert monitor_server.get(f"/api/runs/{run_id}")["status"] == "running"


def test_activity_is_shown_until_the_run_finishes(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()
    monitor.activity("epoch 0: self-play")
    monitor.flush()
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["activity"] == "epoch 0: self-play"
    since = meta["activity_since"]

    # resending the same activity (a heartbeat) keeps when it started
    monitor.activity("epoch 0: self-play")
    monitor.flush()
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["activity_since"] == since
    assert meta["updated"] >= since

    monitor.activity("epoch 0: training")
    monitor.flush()
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["activity"] == "epoch 0: training"
    assert meta["activity_since"] >= since

    monitor.finish()
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["status"] == "finished"
    assert meta["activity"] is None


def test_heartbeat_keeps_a_busy_run_fresh(monitor_server, monkeypatch):
    monkeypatch.setattr(client, "HEARTBEAT_S", 0.05)
    monitor = Monitor(monitor_server.url)
    monitor.start()
    monitor.activity("testing: a slow opponent")
    monitor.flush()
    first = monitor_server.get(f"/api/runs/{monitor.run_id}")["updated"]

    # nothing is logged, but the heartbeat still reaches the server
    deadline = time.time() + 5
    while monitor_server.get(f"/api/runs/{monitor.run_id}")["updated"] == first:
        assert time.time() < deadline, "no heartbeat arrived"
        time.sleep(0.05)
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["activity"] == "testing: a slow opponent"

    # and stops once the run finishes
    monitor.finish()
    finished = monitor_server.get(f"/api/runs/{monitor.run_id}")["updated"]
    time.sleep(0.3)
    assert monitor_server.get(f"/api/runs/{monitor.run_id}")["updated"] == finished


def test_unreachable_server_never_raises(capsys):
    # nothing listens on port 9 (discard) on the test machines
    monitor = Monitor("http://127.0.0.1:9")

    monitor.start(config={"batch_size": 4})
    monitor.log(0, {"loss": 1.0})
    monitor.finish()

    assert monitor.run_id is None
    assert "can't reach" in capsys.readouterr().err


def test_unknown_run_is_not_found(monitor_server):
    with pytest.raises(urllib.error.HTTPError) as e:
        monitor_server.get("/api/runs/nope/metrics")
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        monitor_server.get("/media/..%2F..%2Fetc/passwd")
    assert e.value.code == 404


def test_dashboard_is_served(monitor_server):
    for path in ("/", "/run/anything", "/static/app.js", "/static/style.css"):
        with urllib.request.urlopen(monitor_server.url + path) as resp:
            assert resp.status == 200
