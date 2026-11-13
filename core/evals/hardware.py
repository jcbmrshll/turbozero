"""What a game was played on, for the record: games only compare in time on the same hardware.

`describe` is what the host is, not how busy it was: the same machine, with the same cores, gives
the same description every time, so it's kept once (see `store.Store`). How busy it was goes with
each game, as its load average.
"""

import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

# the instruction set extensions that decide how fast engines and networks run
ISA_FLAGS = (
    "avx2",
    "fma",
    "avx512f",
    "avx512bw",
    "avx512_vnni",
    "avx_vnni",
    "amx_tile",
)
# set by the cloud launchers: where the host is, e.g. "gcp:n4d-highcpu-16" or "modal:L40S"
PLACEMENT_ENV = "TZ_EVAL_PLACEMENT"
# set by the cloud launchers: the commit their files are from
CODE_ENV = "TZ_EVAL_CODE"


def _cpuinfo() -> tuple[str | None, set[str]]:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return platform.processor() or None, set()
    model, flags = None, set()
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "model name" and model is None:
            model = value.strip()
        elif key in ("flags", "Features") and not flags:
            flags = set(value.split())
    return model, flags


def physical_cores(cpus: list[int]) -> int:
    """How many physical cores `cpus` (logical CPUs) are on."""
    cores = set()
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            return len(cpus)
        cores.add((package, core))
    return len(cores)


def gpu() -> list[str]:
    """The GPUs' names and driver, if there's an NVIDIA GPU."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def describe(with_gpu: bool = False) -> dict[str, Any]:
    """This host as a game runs on it: its CPU and the cores this process may use, its memory,
    and with `with_gpu`, its GPUs."""
    model, flags = _cpuinfo()
    cpus = sorted(os.sched_getaffinity(0))
    memory = None
    try:
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        memory = round(pages / 2**30)
    except (ValueError, OSError):
        pass
    info: dict[str, Any] = {
        "hostname": socket.gethostname()
        if os.environ.get(PLACEMENT_ENV) is None
        else None,
        "placement": os.environ.get(PLACEMENT_ENV, "local"),
        "cpu": model,
        "machine": platform.machine(),
        "isa": [f for f in ISA_FLAGS if f in flags],
        "cpus_online": os.cpu_count(),
        "affinity": cpus,
        "physical_cores": physical_cores(cpus),
        "memory_gib": memory,
        "kernel": platform.release(),
    }
    if with_gpu:
        info["gpu"] = gpu()
    return info


def load() -> float:
    return os.getloadavg()[0]


def code_version() -> str:
    """The commit of the code that's running, marked if it's been changed since; on a cloud host,
    which has the files and not the repository, what the launcher said (`CODE_ENV`)."""
    if os.environ.get(CODE_ENV):
        return os.environ[CODE_ENV]
    root = os.path.dirname(__file__)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=root, check=False
        ).returncode
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")
