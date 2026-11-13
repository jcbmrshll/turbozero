"""A CPU job's VM side (see gcp_cpu_jobs.py): its slot's commands in order, each one's result to
the bucket as it ends, then the VM deletes itself. The files a job names as `outputs` go to the
bucket while it runs too, every `STREAM_SECONDS`, for `gcp_cpu_jobs.py follow`. A replacement
for a preempted VM runs this again, and skips what already has a result.

    python examples/othello/gcp_cpu_runner.py BUCKET PREFIX SLOT
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from modal_store import GcsStore, Store, get_json, put_json

METADATA = "http://metadata.google.internal/computeMetadata/v1/"
HERE = Path(__file__).resolve().parent
# what a log keeps of a command's output, from the end
LOG_TAIL = 200_000
# how often a running job's outputs are sent, if they grew
STREAM_SECONDS = 20.0
# before the first job: how idle the CPUs must be, over a window of seconds, and how long to wait
# at most (a fresh VM's own startup work skewed a first timed job by 11%)
IDLE_SHARE = 0.97
IDLE_WINDOW = 5.0
SETTLE_LIMIT = 180.0


def result_key(prefix: str, slot: int, name: str) -> str:
    return f"{prefix}/results/{slot}/{name}.json"


def output_key(prefix: str, slot: int, name: str, path: str) -> str:
    return f"{prefix}/outputs/{slot}/{name}/{Path(path).name}"


def cpu_model() -> dict[str, Any]:
    """What this VM's CPU is, for the record: results only compare on the same model."""
    text = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith("model name")
        ),
        None,
    )
    return {"model": model, "cpus": len(os.sched_getaffinity(0))}


def cpu_times() -> tuple[int, int]:
    """All CPUs' idle and total time so far, in clock ticks."""
    fields = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
    return fields[3] + fields[4], sum(fields)


def settle(
    times: Callable[[], tuple[int, int]] = cpu_times,
    sleep: Callable[[float], None] = time.sleep,
    window: float = IDLE_WINDOW,
    limit: float = SETTLE_LIMIT,
) -> float:
    """Waits for the CPUs to be idle over a window (or `limit` seconds); the idle share found."""
    share = 0.0
    for _ in range(max(1, int(limit / window))):
        idle, total = times()
        sleep(window)
        idle2, total2 = times()
        share = (idle2 - idle) / max(total2 - total, 1)
        if share >= IDLE_SHARE:
            break
    return share


class Outputs:
    """Sends a job's output files to the bucket when they've changed."""

    def __init__(self, store: Store, prefix: str, slot: int, job: dict[str, Any]):
        self.store = store
        self.keys = {
            HERE / path: output_key(prefix, slot, job["name"], path)
            for path in job.get("outputs", [])
        }
        self.sent: dict[Path, tuple[int, int]] = {}

    def send(self) -> None:
        for path, key in self.keys.items():
            if not path.is_file():
                continue
            stat = path.stat()
            sig = (stat.st_size, stat.st_mtime_ns)
            if self.sent.get(path) != sig:
                self.store.put_file(key, str(path))
                self.sent[path] = sig


def run_job(
    job: dict[str, Any], outputs: Outputs, every: float = STREAM_SECONDS
) -> tuple[dict[str, Any], str]:
    """Runs `job["cmd"]` in examples/othello, sending its outputs as they grow: its last line of
    JSON on stdout is its result."""
    begun = time.time()
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        process = subprocess.Popen(
            job["cmd"], cwd=HERE, stdout=out, stderr=err, text=True
        )
        while True:
            try:
                process.wait(timeout=every)
                break
            except subprocess.TimeoutExpired:
                outputs.send()
        outputs.send()
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    record: dict[str, Any] = {
        "returncode": process.returncode,
        "seconds": round(time.time() - begun, 1),
        "result": json.loads(lines[-1]) if lines else None,
    }
    log = f"$ {' '.join(job['cmd'])}\n--- stdout\n{stdout}\n--- stderr\n{stderr}"
    return record, log[-LOG_TAIL:]


def run_slot(
    store: Store,
    prefix: str,
    slot: int,
    machine: dict[str, Any],
    every: float = STREAM_SECONDS,
) -> None:
    """Every job of `slot` without a result yet, in order."""
    spec = get_json(store, f"{prefix}/spec.json")
    for job in spec["slots"][str(slot)]:
        key = result_key(prefix, slot, job["name"])
        if store.sizes(key):
            continue
        put_json(
            store,
            f"{prefix}/status/{slot}.json",
            {"doing": job["name"], "at": time.time()},
        )
        # outputs are sent before the result that says the job is done
        record, log = run_job(job, Outputs(store, prefix, slot, job), every)
        store.put_bytes(f"{prefix}/logs/{slot}/{job['name']}.log", log.encode())
        put_json(store, key, {"name": job["name"], "slot": slot, **machine, **record})
        print(
            f"{job['name']}: returncode {record['returncode']}, {record['seconds']} s",
            flush=True,
        )
    put_json(store, f"{prefix}/status/{slot}.json", {"doing": None, "at": time.time()})


def metadata(path: str) -> str:
    request = urllib.request.Request(
        METADATA + path, headers={"Metadata-Flavor": "Google"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode()


def delete_self() -> None:
    """Deletes this VM, disk and all, by the Compute API as its own service account."""
    token = json.loads(metadata("instance/service-accounts/default/token"))[
        "access_token"
    ]
    project = metadata("project/project-id")
    zone = metadata("instance/zone").rsplit("/", 1)[-1]
    name = metadata("instance/name")
    url = f"https://compute.googleapis.com/compute/v1/projects/{project}/zones/{zone}/instances/{name}"
    request = urllib.request.Request(
        url, method="DELETE", headers={"Authorization": f"Bearer {token}"}
    )
    urllib.request.urlopen(request, timeout=30).close()


def main() -> None:
    """On a VM: `BUCKET PREFIX SLOT`, the bucket by the VM's own account, deleting itself when
    done. In a Modal container (`modal_hosts.py`): `--modal BUCKET PREFIX SLOT`, the bucket by
    the key in its environment, the container ending by itself."""
    modal = sys.argv[1] == "--modal"
    bucket, prefix, slot = (
        sys.argv[1 + modal],
        sys.argv[2 + modal],
        int(sys.argv[3 + modal]),
    )
    idle = settle()
    if modal:
        host = {"container": os.environ.get("MODAL_TASK_ID", "modal")}
        store = GcsStore(bucket, os.environ["GCS_KEY_JSON"])
    else:
        host = {"vm": metadata("instance/name")}
        store = GcsStore(bucket, None)
    run_slot(store, prefix, slot, {**host, **cpu_model(), "idle_at_start": idle})
    if not modal:
        delete_self()


if __name__ == "__main__":
    main()
