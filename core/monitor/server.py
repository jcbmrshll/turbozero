"""A small HTTP server that training runs push metrics and media to, and that serves
the dashboard for browsing them.

    uv run turbozero-monitor --port 8008 --dir runs

Everything lives on disk under the run directory, one folder per run:

    <dir>/<run id>/meta.json      project, name, config, timestamps, status
    <dir>/<run id>/metrics.jsonl  one {"step": ..., "time": ..., <metrics>} per line
    <dir>/<run id>/media.jsonl    one {"key": ..., "step": ..., "file": ...} per line
    <dir>/<run id>/media/         the media files themselves
    <dir>/<run id>/episodes/      raw episode data (.npz), rendered into media/

Runs send episodes as raw arrays; the server renders them (core.monitor.renderers)
on a background thread, so neither the training run nor the dashboard draws them.
"""

import argparse
import json
import math
import mimetypes
import os
import re
import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from core.monitor.media import encode_media
from core.monitor.renderers import RENDERERS

STATIC_DIR = Path(__file__).parent / "static"

# content types the server accepts as media, and the extension each is stored under
MEDIA_TYPES = {
    "image/png": "png",
    "image/gif": "gif",
    "image/jpeg": "jpg",
    "image/svg+xml": "svg",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "application/json": "json",
}


# runs get a memorable name unless the training script gives one
ADJECTIVES = [
    "amber",
    "brisk",
    "calm",
    "deft",
    "eager",
    "fallow",
    "gilded",
    "hardy",
    "idle",
    "jolly",
    "keen",
    "lucid",
    "mellow",
    "nimble",
    "olive",
    "patient",
    "quiet",
    "rustic",
    "sly",
    "tawny",
    "umber",
    "vivid",
    "wary",
    "young",
    "zesty",
]
NOUNS = [
    "badger",
    "crane",
    "dune",
    "ember",
    "finch",
    "grove",
    "heron",
    "ibis",
    "juniper",
    "kestrel",
    "lark",
    "maple",
    "newt",
    "otter",
    "pine",
    "quail",
    "reed",
    "sparrow",
    "thistle",
    "vole",
    "wren",
    "yarrow",
]


def _safe(name: str) -> str:
    """Make a metric/media key safe to use as a file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "media"


class RunStore:
    """Runs on disk. One lock guards every write; reads go straight to the files."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise KeyError(run_id)
        path = self.root / run_id
        if not path.is_dir():
            raise KeyError(run_id)
        return path

    def _read_meta(self, run_dir: Path) -> dict[str, Any]:
        return json.loads((run_dir / "meta.json").read_text())

    def _write_meta(self, run_dir: Path, meta: dict[str, Any]) -> None:
        tmp = run_dir / "meta.json.tmp"
        tmp.write_text(json.dumps(meta))
        tmp.replace(run_dir / "meta.json")

    def create(
        self, project: str, name: str | None, config: dict[str, Any]
    ) -> dict[str, Any]:
        now = time.time()
        run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        run_id += "-" + secrets.token_hex(2)
        if name is None:
            num = sum(1 for m in self.list() if m["project"] == project) + 1
            name = f"{secrets.choice(ADJECTIVES)}-{secrets.choice(NOUNS)}-{num}"
        run = config.get("run", {})
        meta = {
            "id": run_id,
            "project": project,
            "name": name,
            "config": config,
            "created": now,
            "updated": now,
            "status": "running",
            "step": None,
            "seed": run.get("seed"),
            "num_epochs": run.get("num_epochs"),
        }
        with self.lock:
            run_dir = self.root / run_id
            (run_dir / "media").mkdir(parents=True)
            (run_dir / "episodes").mkdir()
            self._write_meta(run_dir, meta)
        return meta

    def log(self, run_id: str, step: int, metrics: dict[str, Any]) -> None:
        run_dir = self._dir(run_id)
        now = time.time()
        # JSON has no NaN or inf (the dashboard couldn't parse them), so a diverged
        # metric is stored as null
        metrics = {
            k: v if isinstance(v, (int, float)) and math.isfinite(v) else None
            for k, v in metrics.items()
        }
        with self.lock:
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": now, **metrics}) + "\n")
            meta = self._read_meta(run_dir)
            meta["updated"] = now
            meta["step"] = step if meta["step"] is None else max(meta["step"], step)
            # a run that logs is running, even one that finished and was continued
            meta["status"] = "running"
            self._write_meta(run_dir, meta)

    def add_media(
        self, run_id: str, key: str, step: int, content_type: str, data: bytes
    ) -> dict[str, Any]:
        run_dir = self._dir(run_id)
        ext = MEDIA_TYPES[content_type]
        file = f"{_safe(key)}-{step}.{ext}"
        entry = {"key": key, "step": step, "file": file, "time": time.time()}
        with self.lock:
            (run_dir / "media" / file).write_bytes(data)
            with open(run_dir / "media.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            meta = self._read_meta(run_dir)
            meta["updated"] = entry["time"]
            self._write_meta(run_dir, meta)
        return entry

    def add_media_error(self, run_id: str, key: str, step: int, error: str) -> None:
        """Record media that failed to render, so the dashboard can say why."""
        run_dir = self._dir(run_id)
        entry = {"key": key, "step": step, "error": error, "time": time.time()}
        with self.lock, open(run_dir / "media.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_episode(self, run_id: str, key: str, step: int, data: bytes) -> Path:
        path = self._dir(run_id) / "episodes" / f"{_safe(key)}-{step}.npz"
        path.write_bytes(data)
        return path

    def finish(self, run_id: str, status: str) -> None:
        run_dir = self._dir(run_id)
        with self.lock:
            meta = self._read_meta(run_dir)
            meta["status"] = status
            meta["updated"] = time.time()
            self._write_meta(run_dir, meta)

    def list(self) -> list[dict[str, Any]]:
        runs = []
        for run_dir in self.root.iterdir():
            try:
                meta = self._read_meta(run_dir)
            except (OSError, ValueError):
                continue
            meta.pop("config", None)
            runs.append(meta)
        return sorted(runs, key=lambda m: m["created"], reverse=True)

    def get(self, run_id: str) -> dict[str, Any]:
        return self._read_meta(self._dir(run_id))

    def _read_jsonl(self, path: Path, since: int) -> dict[str, Any]:
        """Lines from `since` on, plus the offset to ask for next time."""
        if not path.exists():
            return {"rows": [], "next": since}
        with open(path) as f:
            lines = f.readlines()
        # a line still being written has no newline yet; leave it for the next poll
        if lines and not lines[-1].endswith("\n"):
            lines = lines[:-1]
        return {
            "rows": [json.loads(line) for line in lines[since:]],
            "next": len(lines),
        }

    def metrics(self, run_id: str, since: int = 0) -> dict[str, Any]:
        return self._read_jsonl(self._dir(run_id) / "metrics.jsonl", since)

    def media(self, run_id: str, since: int = 0) -> dict[str, Any]:
        return self._read_jsonl(self._dir(run_id) / "media.jsonl", since)

    def media_path(self, run_id: str, file: str) -> Path:
        path = self._dir(run_id) / "media" / _safe(file)
        if not path.is_file():
            raise KeyError(file)
        return path


def render_episode(
    store: RunStore, run_id: str, key: str, step: int, env: str, path: Path
) -> None:
    """Render a saved episode into media; on failure, record the error instead."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            data = {name: npz[name] for name in npz.files}
        media = encode_media(RENDERERS[env](data))
        if media is None:
            raise TypeError(f"the {env!r} renderer returned nothing displayable")
        store.add_media(run_id, key, step, *media)
    except Exception as e:  # noqa: BLE001 - a broken render is recorded, not raised
        traceback.print_exc()
        store.add_media_error(run_id, key, step, f"{type(e).__name__}: {e}")


class Handler(BaseHTTPRequestHandler):
    store: RunStore
    renderer: ThreadPoolExecutor

    def log_message(self, format, *args):
        # one line per request is noise next to a training run
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]
        try:
            since = int(query.get("since", ["0"])[0])
            match parts:
                case ["api", "runs"]:
                    self._json(self.store.list())
                case ["api", "runs", run_id]:
                    self._json(self.store.get(run_id))
                case ["api", "runs", run_id, "metrics"]:
                    self._json(self.store.metrics(run_id, since))
                case ["api", "runs", run_id, "media"]:
                    self._json(self.store.media(run_id, since))
                case ["media", run_id, file]:
                    self._file(self.store.media_path(run_id, file))
                case [] | ["run", _]:
                    # the dashboard routes client-side, so every page is index.html
                    self._file(STATIC_DIR / "index.html")
                case ["static", file] if (STATIC_DIR / _safe(file)).is_file():
                    self._file(STATIC_DIR / _safe(file))
                case _:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "no such run")
        except ValueError as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))

    def do_POST(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]
        try:
            match parts:
                case ["api", "runs"]:
                    body = json.loads(self._body())
                    meta = self.store.create(
                        project=body.get("project") or "default",
                        name=body.get("name"),
                        config=body.get("config") or {},
                    )
                    self._json(meta, HTTPStatus.CREATED)
                case ["api", "runs", run_id, "log"]:
                    body = json.loads(self._body())
                    self.store.log(run_id, int(body["step"]), body["metrics"])
                    self._json({"ok": True})
                case ["api", "runs", run_id, "media"]:
                    content_type = self.headers.get("Content-Type", "")
                    if content_type not in MEDIA_TYPES:
                        self._error(
                            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                            f"unsupported media type {content_type!r}",
                        )
                        return
                    entry = self.store.add_media(
                        run_id,
                        key=query["key"][0],
                        step=int(query["step"][0]),
                        content_type=content_type,
                        data=self._body(),
                    )
                    self._json(entry, HTTPStatus.CREATED)
                case ["api", "runs", run_id, "episodes"]:
                    env = query["env"][0]
                    if env not in RENDERERS:
                        known = ", ".join(sorted(RENDERERS))
                        self._error(
                            HTTPStatus.BAD_REQUEST,
                            f"no renderer for env {env!r} (known: {known})",
                        )
                        return
                    key, step = query["key"][0], int(query["step"][0])
                    path = self.store.save_episode(run_id, key, step, self._body())
                    self.renderer.submit(
                        render_episode, self.store, run_id, key, step, env, path
                    )
                    self._json({"queued": True}, HTTPStatus.ACCEPTED)
                case ["api", "runs", run_id, "finish"]:
                    body = json.loads(self._body() or b"{}")
                    self.store.finish(run_id, body.get("status", "finished"))
                    self._json({"ok": True})
                case _:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "no such run")
        except (ValueError, TypeError) as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))


def make_server(host: str, port: int, run_dir: str) -> ThreadingHTTPServer:
    """The monitor's HTTP server, storing runs in `run_dir` (port 0 picks a free one)."""
    handler = type(
        "Handler",
        (Handler,),
        {
            "store": RunStore(Path(run_dir)),
            # one thread, so renders queue up rather than competing with requests
            "renderer": ThreadPoolExecutor(max_workers=1),
        },
    )
    return ThreadingHTTPServer((host, port), handler)


def main():
    parser = argparse.ArgumentParser(description="Serve the turbozero run monitor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--dir", default="runs", help="where runs are stored")
    args = parser.parse_args()
    # renderers use jax (to rebuild pgx states); keep it off the accelerator, which
    # belongs to the training runs. Must be set before jax is first imported
    os.environ["JAX_PLATFORMS"] = "cpu"
    server = make_server(args.host, args.port, args.dir)
    print(
        f"turbozero monitor on http://{args.host}:{args.port}  "
        f"(runs in {os.path.abspath(args.dir)})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
