"""Modal containers as `gcp_cpu_jobs`' hosts: each slot of a job runs in a container of its own,
as it would on a Spot VM, from the same bucket, by the same runner (`gcp_cpu_runner.run_slot`).

The image holds the environment, built from `uv.lock`, and Modal keeps it until the lock changes; a
container takes the job's code and inputs from the bucket, as a VM does. On a GPU, JAX runs on
it. Modal's CPUs are a mix of models that changes from day to day (some without AVX-512, which the
engines' default builds need), and seconds from different models don't compare: with a `cpu_id`,
a container on another model leaves at once, and another is started in its place, up to
`REROLLS` times a host. A container that ends without its slot done is replaced by
`gcp_cpu_jobs.watch`, the replacement skipping what's done.
"""

import os
import subprocess
import tarfile
from pathlib import Path
from typing import Any

HOME = "/root"
# what a container on another CPU model returns, before its model
WRONG_CPU = "wrong cpu: "
# how many containers a host may start to land on the CPU model asked for
REROLLS = 20


def _slot(
    bucket: str,
    prefix: str,
    slot: int,
    inputs: list[str],
    gpu: bool,
    cpu_id: str | None = None,
) -> str | None:
    """A container's work: the job's code and inputs from the bucket, then its slot's jobs; on a
    CPU that isn't `cpu_id` (see `hardware.cpu_id`), nothing but saying which it is."""
    import io
    import sys

    from modal_store import GcsStore

    store = GcsStore(bucket, os.environ["GCS_KEY_JSON"])
    code = store.get_bytes(f"{prefix}/code.tar.gz")
    assert code is not None, f"no code for {prefix}"
    with tarfile.open(fileobj=io.BytesIO(code), mode="r:gz") as tar:
        tar.extractall("/src", filter="data")
    sys.path.insert(0, "/src")
    from core.evals.hardware import cpu_id as this_cpu

    found = this_cpu()
    if cpu_id is not None and found != cpu_id:
        return f"{WRONG_CPU}{found}"
    cache = Path(HOME) / ".cache" / "turbozero" / "othello"
    cache.mkdir(parents=True, exist_ok=True)
    for key in inputs:
        data = store.get_bytes(key)
        assert data is not None, f"no input {key}"
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(cache, filter="data")
    env = {
        **os.environ,
        "HOME": HOME,
        "PATH": f"/src/.venv/bin:{os.environ['PATH']}",
        "PYTHONPATH": "/src",
    }
    if not gpu:
        env["JAX_PLATFORMS"] = "cpu"
    subprocess.run(
        [
            "/src/.venv/bin/python",
            "examples/othello/gcp_cpu_runner.py",
            "--modal",
            bucket,
            prefix,
            str(slot),
        ],
        cwd="/src",
        env=env,
        check=True,
    )
    return None


class ModalHosts:
    """`gcp_cpu_jobs.Cloud` on Modal: a container a slot, `gpu` ("CPU" for none), `cpu` cores and
    `memory` GiB, on CPU model `cpu_id` if one is given. It knows the containers it started, in
    this process: a launcher run again starts new ones for what's left."""

    def __init__(self, gpu: str, cpu: int, memory: int, cpu_id: str | None = None):
        self.gpu = gpu
        self.cpu = cpu
        self.memory = memory
        self.cpu_id = cpu_id
        # each container's: (slot, job, its call)
        self.calls: dict[str, tuple[str, str, Any]] = {}
        self.apps: dict[str, Any] = {}
        # each host's: what `create` was given, and containers started on other CPU models
        self.made: dict[str, tuple[str, dict[str, str], dict[str, str], float]] = {}
        self.rerolls: dict[str, int] = {}

    def image(self) -> Any:
        import modal

        here = Path(__file__).resolve().parent
        repo = here.parents[1]
        image = (
            # Ubuntu's glibc, which the engines' builds (made on Ubuntu 24.04) need
            modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
            # g++ builds the search's and the solver's libraries on first use
            .apt_install("g++", "git")
            .pip_install("uv", "google-cloud-storage")
            .add_local_file(repo / "pyproject.toml", "/src/pyproject.toml", copy=True)
            .add_local_file(repo / "uv.lock", "/src/uv.lock", copy=True)
            .add_local_file(repo / ".python-version", "/src/.python-version", copy=True)
            .run_commands(
                "cd /src && uv sync --frozen --no-install-project --extra cuda"
            )
        )
        # the container's function is this module's, and what it imports
        for module in ("modal_hosts", "modal_store"):
            image = image.add_local_file(here / f"{module}.py", f"/root/{module}.py")
        return image

    def create(
        self,
        name: str,
        zone: str,
        labels: dict[str, str],
        metadata: dict[str, str],
        hours: float,
    ) -> str | None:
        import modal
        from modal_jobs import SECRET

        app = modal.App(name)
        function = app.function(
            image=self.image(),
            gpu=None if self.gpu == "CPU" else self.gpu,
            cpu=float(self.cpu),
            memory=self.memory * 1024,
            secrets=[modal.Secret.from_name(SECRET)],
            timeout=int(hours * 3600),
            # a container that dies is replaced by the launcher, which knows what's left
            retries=0,
            scaledown_window=2,
            serialized=True,
        )(_slot)
        try:
            with app.run(detach=True):
                call = function.spawn(
                    metadata["tz-bucket"],
                    metadata["tz-prefix"],
                    int(metadata["tz-slot"]),
                    metadata["tz-inputs"].split(),
                    self.gpu != "CPU",
                    self.cpu_id,
                )
        except Exception as error:  # noqa: BLE001 - Modal's errors, to try again
            return str(error)[-500:]
        self.calls[name] = (
            labels.get("tz-cpu-slot", ""),
            labels.get("tz-cpu-job", ""),
            call,
        )
        self.apps[name] = app.app_id
        self.made[name] = (zone, labels, metadata, hours)
        return None

    def reroll(self, name: str, said: Any) -> bool:
        """Whether the container `name` left for being on another CPU model and another was
        started in its place."""
        if not (isinstance(said, str) and said.startswith(WRONG_CPU)):
            return False
        if self.rerolls.get(name, 0) >= REROLLS:
            print(f"{name}: {said}, {REROLLS} times: giving up on it", flush=True)
            return False
        self.rerolls[name] = self.rerolls.get(name, 0) + 1
        print(f"{name}: {said}: another ({self.rerolls[name]})", flush=True)
        self._stop_app(name)
        zone, labels, metadata, hours = self.made[name]
        return self.create(name, zone, labels, metadata, hours) is None

    def instances(self, job: str | None) -> list[dict[str, str]]:
        out = []
        for name, (slot, of, call) in list(self.calls.items()):
            if job is not None and of != job:
                continue
            try:
                said = call.get(timeout=0)
                running = self.reroll(name, said)
            except TimeoutError:
                running = True
            except Exception:  # noqa: BLE001 - it ended, and how is in the bucket
                running = False
            if running:
                out.append(
                    {
                        "name": name,
                        "zone": "modal",
                        "status": "RUNNING",
                        "job": of,
                        "slot": slot,
                    }
                )
            else:
                self.calls.pop(name, None)
        return out

    def delete(self, names: list[tuple[str, str]]) -> None:
        for name, _ in names:
            found = self.calls.pop(name, None)
            if found is not None:
                found[2].cancel()
            self._stop_app(name)

    def _stop_app(self, name: str) -> None:
        app = self.apps.pop(name, None)
        if app is not None:
            subprocess.run(
                ["modal", "app", "stop", app, "--yes"],
                check=False,
                capture_output=True,
                timeout=60,
            )
