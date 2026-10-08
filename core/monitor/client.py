"""Push metrics and media from a training run to the monitor server.

    monitor = Monitor("http://localhost:8008", project="othello")
    monitor.start(config=trainer.get_config())
    monitor.log(epoch, {"loss": 0.41, "greedy_game": Episode("pgx_two_player", ...)})
    monitor.finish()

`Trainer` does all of this when it's given a monitor. Everything but `start` happens
on a background thread: `log` only queues its values, so the training loop never
waits on encoding, copies off the device or the network. Logging never raises: if
the server is down the run keeps training and the failure is reported once.
"""

import io
import json
import queue
import sys
import threading
import urllib.error
import urllib.request
from numbers import Number
from typing import Any
from urllib.parse import quote

import numpy as np

from core.monitor.media import encode_media

DEFAULT_URL = "http://localhost:8008"


class Episode:
    """The raw data of one episode, e.g. Episode("pgx_two_player", **arrays). The
    server renders it with its renderer for `env` (see core.monitor.renderers), so
    the training run never draws anything."""

    def __init__(self, env: str, **data: Any):
        self.env = env
        self.data = data

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        arrays = {k: np.asarray(v) for k, v in self.data.items()}
        np.savez_compressed(buf, **arrays)  # pyright: ignore[reportArgumentType]
        return buf.getvalue()


def to_number(value: Any) -> float | None:
    """A plain float for scalars (python, numpy or jax), None for anything else."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Number):
        return float(value)  # type: ignore[arg-type]
    if getattr(value, "shape", None) == () and hasattr(value, "item"):
        return float(value.item())
    return None


class Monitor:
    def __init__(
        self,
        url: str = DEFAULT_URL,
        project: str = "default",
        name: str | None = None,
    ):
        self.url = url.rstrip("/")
        self.project = project
        self.name = name
        self.run_id: str | None = None
        self._failing = False
        self._rejected: set = set()
        # (function, args) to call in order on the sender thread
        self._queue: queue.Queue = queue.Queue()
        self._sender: threading.Thread | None = None

    def _request(
        self,
        path: str,
        data: bytes,
        content_type: str = "application/json",
        timeout: float = 10,
    ) -> dict[str, Any] | None:
        req = urllib.request.Request(
            self.url + path,
            data=data,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # the server is up but refused this request; say why, once per reason
            reason = e.read().decode(errors="replace")
            if reason not in self._rejected:
                self._rejected.add(reason)
                print(
                    f"monitor: {path.split('?')[0]} rejected: {reason}", file=sys.stderr
                )
            return None
        except (urllib.error.URLError, OSError, ValueError) as e:
            if not self._failing:
                print(f"monitor: can't reach {self.url} ({e})", file=sys.stderr)
            self._failing = True
            return None
        if self._failing:
            print(f"monitor: reconnected to {self.url}", file=sys.stderr)
            self._failing = False
        return result

    def _post_json(self, path: str, obj: Any) -> dict[str, Any] | None:
        # config can hold functions and classes; name them rather than repr
        body = json.dumps(obj, default=lambda o: getattr(o, "__name__", None) or str(o))
        return self._request(path, body.encode())

    def _send(self) -> None:
        """The sender thread: runs queued calls in order, forever."""
        while True:
            fn, args = self._queue.get()
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001 - logging must never stop training
                # e.g. a value that failed to encode; lose it, not the run
                print(
                    f"monitor: dropped a log ({type(e).__name__}: {e})", file=sys.stderr
                )
            finally:
                self._queue.task_done()

    def _enqueue(self, fn, *args) -> None:
        if self._sender is None:
            self._sender = threading.Thread(target=self._send, daemon=True)
            self._sender.start()
        self._queue.put((fn, args))

    def start(self, config: dict[str, Any] | None = None) -> None:
        """Create the run on the server. Does nothing if the run already exists, so
        continuing training keeps logging to the same run."""
        if self.run_id is not None:
            return
        meta = self._post_json(
            "/api/runs",
            {"project": self.project, "name": self.name, "config": config or {}},
        )
        if meta is None:
            print("monitor: logging is off for this run", file=sys.stderr)
            return
        self.run_id = meta["id"]
        print(f"monitor: {self.url}/run/{self.run_id}")

    def log(self, step: int, data: dict[str, Any]) -> None:
        """Queue scalars, media and episodes for one step; they're sent in the
        background. Values that are none of these are dropped."""
        if self.run_id is None:
            return
        self._enqueue(self._log, self.run_id, step, dict(data))

    def _log(self, run_id: str, step: int, data: dict[str, Any]) -> None:
        metrics = {}
        for key, value in data.items():
            number = to_number(value)
            if number is not None:
                metrics[key] = number
                continue
            if isinstance(value, Episode):
                self._request(
                    f"/api/runs/{run_id}/episodes"
                    f"?key={quote(key)}&step={step}&env={quote(value.env)}",
                    value.to_bytes(),
                    content_type="application/x-npz",
                    timeout=60,
                )
                continue
            media = encode_media(value)
            if media is not None:
                content_type, body = media
                self._request(
                    f"/api/runs/{run_id}/media?key={quote(key)}&step={step}",
                    body,
                    content_type=content_type,
                    timeout=60,
                )
        if metrics:
            self._post_json(
                f"/api/runs/{run_id}/log", {"step": step, "metrics": metrics}
            )

    def flush(self) -> None:
        """Wait until everything logged so far has been sent."""
        self._queue.join()

    def finish(self, status: str = "finished") -> None:
        """Mark the run finished (or e.g. "crashed"), once everything logged has been
        sent."""
        if self.run_id is None:
            return
        self._enqueue(
            self._post_json, f"/api/runs/{self.run_id}/finish", {"status": status}
        )
        self.flush()
