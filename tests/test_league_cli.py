"""The league's commands for asking for a series."""

import base64
import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from core.league.cli import main
from core.league.game import LeagueGame
from core.league.runtime import FILES, fetch_runtime, has_runtime, runtime_dir
from core.league.store import Store


class FakeGame(LeagueGame):
    name = "fake"
    env = None

    def can_play(self, player):
        return player.name != "paper"

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
        return 100

    def openings(self, indices):
        raise NotImplementedError


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "league"
    store = Store(root)
    store.add_external_player("x", "edax")
    store.add_external_player("y", "edax")
    store.add_external_player("paper", "a paper")
    store.add_external_player("old", "edax")
    store.set_retired("old")
    store.close()
    return root


def run(root, *argv):
    main(FakeGame, root, ["--root", str(root), *argv])


def test_a_match_is_queued_listed_and_cancelled(root, capsys):
    run(root, "match", "x", "y", "--games", "40")
    assert capsys.readouterr().out.strip() == "#1"
    run(root, "matches")
    assert capsys.readouterr().out.split() == ["#1", "x", "vs", "y", "0/40"]
    run(root, "cancel-match", "1")
    capsys.readouterr()
    run(root, "matches")
    assert capsys.readouterr().out.strip() == ""


def test_matches_says_when_a_player_has_retired_since(root, capsys):
    run(root, "match", "x", "y", "--games", "2")
    run(root, "retire", "y")
    capsys.readouterr()
    run(root, "matches")
    assert "retired" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("a", "b", "games", "says"),
    [
        ("x", "nobody", "2", "no player nobody"),
        ("old", "y", "2", "old is retired"),
        ("x", "paper", "2", "paper can't play here"),
        ("x", "x", "2", "itself"),
        ("x", "y", "3", "even"),
        ("x", "y", "0", "even"),
    ],
)
def test_a_match_that_cant_be_played_is_refused(root, capsys, a, b, games, says):
    with pytest.raises(SystemExit):
        run(root, "match", a, b, "--games", games)
    assert says in capsys.readouterr().err
    store = Store(root)
    assert store.pending_matches() == []
    store.close()


def test_cancelling_what_isnt_pending_is_refused(root, capsys):
    with pytest.raises(SystemExit):
        run(root, "cancel-match", "7")
    assert "no pending match #7" in capsys.readouterr().err


class ExportingGame(FakeGame):
    """Exports each checkpoint as its own bytes, unless it says to fail."""

    exported = 0

    def export_network(self, checkpoint, net, out):
        contents = Path(checkpoint).read_bytes()
        if contents == b"bad":
            raise TypeError("only pooled heads export")
        type(self).exported += 1
        out.write_bytes(contents)


def onnx_of(store, player):
    path = store.onnx_path(player)
    assert path is not None
    return path


def add_network(tmp_path, name, contents):
    checkpoint = tmp_path / f"{name}.eqx"
    checkpoint.write_bytes(contents)
    root = tmp_path / "league"
    main(ExportingGame, root, ["--root", str(root), "add-network", name, str(checkpoint),
                               "--net", "resnet:2x8", "--run", "r", "--sims", "8", "16"])  # fmt: skip
    return root


def test_a_networks_export_comes_with_it(tmp_path):
    root = add_network(tmp_path, "good", b"weights")
    store = Store(root)
    assert onnx_of(store, store.players()[0]).read_bytes() == b"weights"
    store.close()


def test_a_failed_export_doesnt_lose_the_network(tmp_path, capsys):
    root = add_network(tmp_path, "broken", b"bad")
    assert "no ONNX export" in capsys.readouterr().err
    store = Store(root)
    player = store.players()[0]
    assert player.network == "broken" and not onnx_of(store, player).exists()
    store.close()


def test_backfilling_exports_only_whats_missing_and_says_if_any_failed(tmp_path):
    root = add_network(tmp_path, "good", b"weights")
    add_network(tmp_path, "broken", b"bad")
    ExportingGame.exported = 0
    with pytest.raises(SystemExit) as caught:
        main(ExportingGame, root, ["--root", str(root), "export-onnx"])
    assert caught.value.code == 1
    assert ExportingGame.exported == 0
    store = Store(root)
    added = store.players()
    onnx_of(store, added[0]).unlink()
    store.close()
    with pytest.raises(SystemExit):
        main(ExportingGame, root, ["--root", str(root), "export-onnx"])
    assert ExportingGame.exported == 1


def tarball(tmp_path, files):
    path = tmp_path / "package.tgz"
    with tarfile.open(path, "w:gz") as tar:
        for name, contents in files.items():
            info = tarfile.TarInfo(f"package/dist/{name}")
            info.size = len(contents)
            tar.addfile(info, io.BytesIO(contents))
    return path, base64.b64encode(hashlib.sha512(path.read_bytes()).digest()).decode()


def test_the_runtime_is_fetched_once_and_checked(tmp_path):
    path, sha512 = tarball(
        tmp_path, {name: name.encode() for name in FILES} | {"evil.js": b"x"}
    )
    root = tmp_path / "league"
    with pytest.raises(ValueError):
        fetch_runtime(root, path.as_uri(), "wrong")
    assert not has_runtime(root)
    fetch_runtime(root, path.as_uri(), sha512)
    assert has_runtime(root)
    assert sorted(p.name for p in runtime_dir(root).iterdir()) == sorted(FILES)
    path.unlink()
    fetch_runtime(root, path.as_uri(), sha512)
