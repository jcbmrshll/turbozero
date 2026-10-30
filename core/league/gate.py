"""Whether anyone else wants the GPU: what the league's worker checks before it uses it, and
every few seconds while it does.

Two things count, since neither knows about the other: slick's run queue, which only knows
its own entries, and the GPU's own list of processes, which shows work started outside it.
Anything that can't be told counts as busy: the league's games are never the urgent thing.
"""

import json
import os
import subprocess
import time
import urllib.request
from typing import Protocol

# slick's server, on this machine's tailnet
SLICK_URL = os.environ.get("SLICK_URL", "http://slick")


class Gate(Protocol):
    def busy(self) -> str | None:
        """Why the GPU isn't free for the league, or None if it is."""
        ...


def gpu_processes() -> set[int] | None:
    """The ids of the processes computing on the GPU, or None if that can't be told."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {int(line) for line in out.split() if line.strip().isdigit()}


def queue_board(url: str = SLICK_URL) -> dict | None:
    """The run queue's board, or None if it can't be read."""
    try:
        with urllib.request.urlopen(f"{url}/jobs", timeout=5) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def lane(board: dict, resource: str) -> dict:
    """The queue's lane of `resource` on `board`."""
    for found in board.get("resources", []):
        if found.get("name") == resource:
            return found
    return {}


class SlickGate:
    """The GPU of slick's `resource`, from the point of view of the process it's in, and of
    its entry in the queue, if it has one (`SLICK_RUN` says so, in an entry's environment).

    Args:
        resource: the queue's name for the GPU
        url: slick's server
        own_job: the queue entry that's allowed to be running
        own_pid: the process that's allowed to be on the GPU
    """

    def __init__(
        self,
        resource: str = "5080",
        url: str = SLICK_URL,
        own_job: int | None = None,
        own_pid: int | None = None,
    ):
        self.resource = resource
        self.url = url
        self.own_job = own_job
        self.own_pid = own_pid

    def busy(self) -> str | None:
        processes = gpu_processes()
        if processes is None:
            return "can't read the GPU's processes"
        others = processes - {self.own_pid}
        if others:
            return f"on the GPU: {' '.join(str(p) for p in sorted(others))}"
        board = queue_board(self.url)
        if board is None:
            return "can't read the queue"
        found = lane(board, self.resource)
        running = found.get("running")
        if running and running.get("id") != self.own_job:
            return f"queue running #{running.get('id')}: {running.get('about', '')}"
        for job in found.get("queued") or []:
            if job.get("id") != self.own_job:
                return f"queue waiting #{job.get('id')}: {job.get('about', '')}"
        return None


class ThrottledGate:
    """A gate looked at no more often than every `every` seconds, to check before each move."""

    def __init__(self, gate: Gate, every: float = 5.0):
        self.gate = gate
        self.every = every
        self.checked = -float("inf")
        self.reason: str | None = None

    def busy(self) -> str | None:
        now = time.monotonic()
        if now - self.checked >= self.every:
            self.checked = now
            self.reason = self.gate.busy()
        return self.reason
