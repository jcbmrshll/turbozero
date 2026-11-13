"""Commands on Google Cloud CPUs of one known model: Spot VMs of 8 physical cores each (AMD Turin,
AVX-512 VNNI, one thread a core), for timed games whose seconds must mean the same from VM to VM,
which Modal's mixed CPUs don't give.

    uv run examples/othello/gcp_cpu_jobs.py setup --project PROJECT     (once)
    uv run examples/othello/gcp_cpu_jobs.py run --name t1 --jobs jobs.json --out results.jsonl \\
        --vms 2 --files quantized/4x64-...-native.npz
    uv run examples/othello/gcp_cpu_jobs.py follow t1                 (outputs' new lines, as written)
    uv run examples/othello/gcp_cpu_jobs.py list
    uv run examples/othello/gcp_cpu_jobs.py cleanup                   (every job's VMs)

jobs.json is a list of {"name": ..., "cmd": [...], "outputs": [...]}: each command runs in
examples/othello on the VM, with the venv's python first on PATH, and the last line of JSON it
prints is its result. Files it writes and names in "outputs" (optional, paths from
examples/othello) go to the bucket every 20 s while it runs, for `follow`, and come back to
OUT-files/SLOT/NAME/ beside --out when it ends. The jobs are split over the VMs in order
(--each: every VM runs every job, to compare the VMs). A VM waits for its CPUs to go idle before
its first job. Each result goes to the bucket as it ends and to --out here; a VM deletes itself
when its jobs are done, and one preempted is replaced (up to --attempts times a slot), the
replacement skipping what's done. Run it again with the same --name to pick up a job that was cut
short.

What a VM has: this checkout's files as they are now (uncommitted and new ones too, not ignored),
the engines' builds from ~/.cache/turbozero/othello (built here for Zen 5 at x86-64-v4 or v3,
which Turin runs), and each --files path, under the same ~/.cache/turbozero/othello. The bucket and its key
are modal_jobs.py's. Every VM has --hours as its limit, after which Google deletes it whatever
it's doing, and `run` deletes its VMs when it ends, however it ends.
"""

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from engines import RUNTIME, default_dir
from gcp_cpu_runner import result_key
from modal_jobs import open_store
from modal_store import Store, get_json, put_json

CONFIG = Path.home() / ".config" / "turbozero" / "gcp-cpu-jobs.json"
CACHE = Path.home() / ".cache" / "turbozero" / "othello"
REPO = Path(__file__).resolve().parents[2]
PREFIX = "cpu-jobs"
# 16 vCPUs at one thread a core: 8 physical cores of AMD Turin (EPYC 9B45), AVX-512 VNNI
MACHINE = "n4d-highcpu-16"
ZONES = ("us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f")
# the engines' runtime files, what evaluate.py needs of them
ENGINES = tuple(
    f"{default_dir(name).relative_to(CACHE)}/{part}"
    for name, parts in RUNTIME.items()
    for part in parts
)
LABEL = "tz-cpu-job"
POLL_SECONDS = 20.0
NAME = re.compile(r"[a-z][a-z0-9-]{0,30}")

# The VM's startup script. It runs as root on each boot; a Spot VM is deleted when preempted, so
# it boots once. On any failure it leaves its log in the bucket and deletes the VM.
BOOT = r"""#!/bin/bash
set -uo pipefail
export HOME=/root
md() { curl -sf -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/$1"; }
BUCKET=$(md instance/attributes/tz-bucket); PREFIX=$(md instance/attributes/tz-prefix)
SLOT=$(md instance/attributes/tz-slot); NAME=$(md instance/name)
token() { md instance/service-accounts/default/token | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'; }
quote() { python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }
get() { curl -sfL -H "Authorization: Bearer $(token)" "https://storage.googleapis.com/storage/v1/b/$BUCKET/o/$(quote "$1")?alt=media" -o "$2"; }
# -T streams the file (--data-binary reads it all into memory, too much for the environment)
put() { curl -sSf -X POST -T "$2" -H "Authorization: Bearer $(token)" "https://storage.googleapis.com/upload/storage/v1/b/$BUCKET/o?uploadType=media&name=$(quote "$1")" >/dev/null; }
delete_self() {
  curl -sf -X DELETE -H "Authorization: Bearer $(token)" \
    "https://compute.googleapis.com/compute/v1/projects/$(md project/project-id)/zones/$(md instance/zone | sed 's|.*/||')/instances/$NAME" >/dev/null
}
LOG=/var/log/tz-boot.log
fail() { echo "FAILED: $1" >>$LOG; put "$PREFIX/boot/$SLOT-$NAME.log" $LOG; put "$PREFIX/failed/$SLOT.log" $LOG; delete_self; exit 1; }
{
  # no tracing: the commands carry access tokens, and the log goes to the bucket
  step() { echo "$(date +%T) $1"; }
  # Ubuntu's own updates would take CPU from timed jobs, and hold apt's lock
  systemctl stop apt-daily.timer apt-daily-upgrade.timer apt-daily.service apt-daily-upgrade.service \
    unattended-upgrades 2>/dev/null
  step "code and inputs"
  mkdir -p /src $HOME/.cache/turbozero/othello && cd /src
  get "$PREFIX/code.tar.gz" code.tar.gz && tar xzf code.tar.gz && rm code.tar.gz || exit 1
  for key in $(md instance/attributes/tz-inputs); do
    get "$key" in.tar.gz && tar xzf in.tar.gz -C $HOME/.cache/turbozero/othello && rm in.tar.gz || exit 1
  done
  # The environment (g++'s packages, uv, the venv) from the bucket when an earlier VM left one
  # for this uv.lock: under a minute, where Ubuntu's mirror and PyPI took 3-13
  ENV=$(md instance/attributes/tz-env)
  if get "$ENV" env.tar; then
    step "environment from the bucket"; tar xf env.tar -C / && rm env.tar; cached=1
    dpkg -i /var/cache/tz-debs/*.deb >/dev/null || cached=0
  else
    rm -f env.tar; cached=0
  fi
  if [ $cached = 0 ]; then
    step "environment from Ubuntu and PyPI"
    apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
      -o APT::Keep-Downloaded-Packages=true g++ || exit 1
    mkdir -p /var/cache/tz-debs && cp /var/cache/apt/archives/*.deb /var/cache/tz-debs/
    [ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh || exit 1
  fi
  $HOME/.local/bin/uv sync --frozen -q || exit 1
  if [ $cached = 0 ]; then
    tar cf env.tar -C / src/.venv root/.local/share/uv root/.local/bin var/cache/tz-debs 2>/dev/null
    put "$ENV" env.tar && step "environment sent to the bucket"; rm env.tar
  fi
  step "ready"
} >$LOG 2>&1 || fail boot
put "$PREFIX/boot/$SLOT-$NAME.log" $LOG
cd /src && PATH=/src/.venv/bin:$PATH JAX_PLATFORMS=cpu PYTHONPATH=/src \
  python examples/othello/gcp_cpu_runner.py "$BUCKET" "$PREFIX" "$SLOT" >>$LOG 2>&1 || fail runner
"""


def gcloud_binary() -> str:
    found = shutil.which("gcloud")
    fallback = Path.home() / "google-cloud-sdk" / "bin" / "gcloud"
    if found is None and not fallback.exists():
        raise SystemExit("no gcloud: install the Google Cloud CLI")
    return found or str(fallback)


# machine families of ARM cores (Axion, Ampere), and those whose boot disk can't be a hyperdisk;
# other families (N4D, C4D, C3, C4A...) are timed on one as they are
ARM_FAMILIES = ("c4a", "t2a")
PD_FAMILIES = ("c2d", "c2", "n2d", "n2", "n1", "t2d", "e2")


def machine_options(machine: str) -> list[str]:
    """What creating a VM of `machine` needs: its image's architecture, its boot disk's type,
    and one thread a core (ARM's cores run one already)."""
    family = machine.split("-")[0]
    arm = family in ARM_FAMILIES
    return [
        f"--image-family=ubuntu-2404-lts-{'arm64' if arm else 'amd64'}",
        f"--boot-disk-type={'pd-balanced' if family in PD_FAMILIES else 'hyperdisk-balanced'}",
        *([] if arm else ["--threads-per-core=1"]),
    ]


class Cloud(Protocol):
    def create(
        self,
        name: str,
        zone: str,
        labels: dict[str, str],
        metadata: dict[str, str],
        hours: float,
    ) -> str | None: ...
    def instances(self, job: str | None) -> list[dict[str, str]]: ...
    def delete(self, names: list[tuple[str, str]]) -> None: ...


class Gcloud:
    """Compute Engine by the gcloud CLI, in one project."""

    def __init__(self, project: str, machine: str = MACHINE):
        self.project = project
        self.machine = machine
        self.binary = gcloud_binary()

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.binary, *args, f"--project={self.project}", "--quiet"],
            check=False,
            capture_output=True,
            text=True,
        )

    def create(
        self,
        name: str,
        zone: str,
        labels: dict[str, str],
        metadata: dict[str, str],
        hours: float,
    ) -> str | None:
        """A Spot VM, deleted by Google after `hours` or when preempted: None if it's running,
        else why not (e.g. no Spot capacity in `zone`)."""
        values = {**metadata, "startup-script": BOOT}
        with tempfile.TemporaryDirectory() as directory:
            # --metadata-from-file takes each value from a file, which spares quoting
            for key, value in values.items():
                (Path(directory) / key).write_text(value)
            files = ",".join(f"{key}={Path(directory) / key}" for key in values)
            done = self._run(
                "compute", "instances", "create", name,
                f"--zone={zone}",
                f"--machine-type={self.machine}",
                *machine_options(self.machine),
                "--provisioning-model=SPOT",
                "--instance-termination-action=DELETE",
                f"--max-run-duration={max(1, round(hours * 60))}m",
                "--image-project=ubuntu-os-cloud",
                "--boot-disk-size=20GB",
                "--scopes=cloud-platform",
                "--labels=" + ",".join(f"{k}={v}" for k, v in labels.items()),
                f"--metadata-from-file={files}",
            )  # fmt: skip
        return None if done.returncode == 0 else done.stderr.strip()[-500:]

    def instances(self, job: str | None) -> list[dict[str, str]]:
        """Our VMs, of `job` or of every job: name, zone, status, labels."""
        where = f"labels.{LABEL}={job}" if job else f"labels.{LABEL}:*"
        done = self._run(
            "compute", "instances", "list", f"--filter={where}", "--format=json"
        )
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip()[-500:])
        return [
            {
                "name": vm["name"],
                "zone": vm["zone"].rsplit("/", 1)[-1],
                "status": vm["status"],
                "job": vm.get("labels", {}).get(LABEL, ""),
                "slot": vm.get("labels", {}).get("tz-cpu-slot", ""),
            }
            for vm in json.loads(done.stdout or "[]")
        ]

    def delete(self, names: list[tuple[str, str]]) -> None:
        for zone in {zone for _, zone in names}:
            here = [name for name, z in names if z == zone]
            self._run("compute", "instances", "delete", *here, f"--zone={zone}")


def split_jobs(
    jobs: list[dict[str, Any]], vms: int, each: bool
) -> dict[str, list[dict[str, Any]]]:
    """Each slot's jobs: all of them for every slot with `each`, else dealt out in turn."""
    names = [job["name"] for job in jobs]
    if len(set(names)) != len(names):
        raise SystemExit("job names must be unique")
    if each:
        return {str(slot): list(jobs) for slot in range(vms)}
    slots = {str(slot): [] for slot in range(min(vms, len(jobs)))}
    for index, job in enumerate(jobs):
        slots[str(index % len(slots))].append(job)
    return slots


def tar_gz(add: Callable[[tarfile.TarFile], None]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        add(tar)
    return buffer.getvalue()


def code_archive(repo: Path = REPO) -> bytes:
    """The checkout's files as they are on disk, new ones too, but not what git ignores."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout.decode()
    files = [f for f in listed.split("\0") if f and (repo / f).is_file()]

    def add(tar: tarfile.TarFile) -> None:
        for f in files:
            tar.add(repo / f, arcname=f)

    return tar_gz(add)


def env_key(repo: Path = REPO, machine: str = MACHINE) -> str:
    """Where a VM leaves its built environment for the next VMs, by what decides it."""
    # what the boot script puts in an environment: change it with the script; and the CPU's
    # architecture, whose packages and venv it holds
    digest = hashlib.sha256(b"debs+uv")
    if machine.split("-")[0] in ARM_FAMILIES:
        digest.update(b"arm64")
    for name in ("uv.lock", "pyproject.toml", ".python-version"):
        digest.update((repo / name).read_bytes())
    return f"{PREFIX}/env/{digest.hexdigest()[:24]}.tar"


def input_key(path: str, cache: Path = CACHE) -> str:
    """Where a path under `cache` goes in the bucket: by what's in it (names, sizes, times), so
    unchanged inputs are uploaded once."""
    root = cache / path
    if not root.exists():
        raise SystemExit(f"{root} isn't here")
    files = sorted(
        p for p in ([root] if root.is_file() else root.rglob("*")) if p.is_file()
    )
    digest = hashlib.sha256()
    for p in files:
        stat = p.stat()
        digest.update(
            f"{p.relative_to(cache)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return f"{PREFIX}/inputs/{digest.hexdigest()[:24]}.tar.gz"


def upload_inputs(store: Store, paths: list[str], cache: Path = CACHE) -> list[str]:
    keys = []
    for path in paths:
        key = input_key(path, cache)
        if not store.sizes(key):
            print(f"uploading {path}", flush=True)
            store.put_bytes(
                key, tar_gz(lambda tar, p=path: tar.add(cache / p, arcname=p))
            )
        keys.append(key)
    return keys


def slot_done(store: Store, prefix: str, slot: str, jobs: list[dict[str, Any]]) -> bool:
    finished = store.sizes(f"{prefix}/results/{slot}/")
    return all(result_key(prefix, int(slot), job["name"]) in finished for job in jobs)


def watch(
    store: Store,
    cloud: Cloud,
    name: str,
    spec: dict[str, Any],
    out: Path,
    attempts: int,
    clock: Callable[[], float] = time.time,
    pause: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """Keeps a VM on each slot with jobs left until they're done, a slot has failed, or time is up,
    writing results to `out` as they come, and the files jobs name as `outputs` beside it. Each
    slot's outcome: done, failed or out of time."""
    prefix = f"{PREFIX}/{name}"
    deadline = clock() + spec["hours"] * 3600 + 600
    launched = {slot: 0 for slot in spec["slots"]}
    zone_at = 0
    seen = (
        {json.loads(line)["key"] for line in out.read_text().splitlines()}
        if out.exists()
        else set()
    )
    # each job's `outputs`, beside `out`, by slot and job
    files = out.with_name(out.stem + "-files")
    outcome: dict[str, str] = {}
    while True:
        # VMs first: one that has gone wrote its results before it went, so the look at the
        # bucket after this sees them
        running = {
            vm["slot"]: vm
            for vm in cloud.instances(name)
            if vm["status"] not in ("STOPPING", "TERMINATED")
        }
        failed = store.sizes(f"{prefix}/failed/")
        for key in sorted(store.sizes(f"{prefix}/results/")):
            if key not in seen:
                record = {"key": key, **get_json(store, key)}
                # its outputs first: a line in `out` says they're here
                where = f"{prefix}/outputs/{record['slot']}/{record['name']}/"
                for output in store.sizes(where):
                    target = files / str(record["slot"]) / record["name"]
                    store.get_file(output, str(target / output[len(where) :]))
                with out.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                seen.add(key)
                print(
                    f"{record['name']} (slot {record['slot']}): {record['seconds']} s, returncode {record['returncode']}",
                    flush=True,
                )
        for slot, jobs in spec["slots"].items():
            if slot in outcome:
                continue
            if slot_done(store, prefix, slot, jobs):
                outcome[slot] = "done"
            elif f"{prefix}/failed/{slot}.log" in failed:
                outcome[slot] = "failed"
            elif slot not in running:
                if launched[slot] >= attempts:
                    outcome[slot] = "failed"
                    continue
                vm = f"tz-{name}-{slot}-{launched[slot]}"
                for _ in ZONES:
                    zone = ZONES[zone_at % len(ZONES)]
                    error = cloud.create(
                        vm, zone,
                        {LABEL: name, "tz-cpu-slot": slot},
                        {"tz-bucket": spec["bucket"], "tz-prefix": prefix, "tz-slot": slot, "tz-inputs": " ".join(spec["inputs"]), "tz-env": spec["env"]},
                        spec["hours"],
                    )  # fmt: skip
                    if error is None:
                        print(f"slot {slot}: {vm} in {zone}", flush=True)
                        break
                    print(
                        f"slot {slot}: not in {zone}: {error.splitlines()[-1]}",
                        flush=True,
                    )
                    zone_at += 1
                # a try whether it started or not: no capacity anywhere doesn't loop forever
                launched[slot] += 1
        if len(outcome) == len(spec["slots"]):
            return outcome
        if clock() > deadline:
            return {slot: outcome.get(slot, "out of time") for slot in spec["slots"]}
        pause(POLL_SECONDS)


def new_lines(store: Store, prefix: str, offsets: dict[str, int]) -> list[str]:
    """The complete lines added to a job's outputs since `offsets` (key: bytes read), each as
    "SLOT/JOB: line"; a file that shrank, rewritten, is read again from its start."""
    out = []
    where = f"{prefix}/outputs/"
    for key, size in sorted(store.sizes(where).items()):
        start = offsets.get(key, 0)
        if size < start:
            start = 0
        if size == start:
            continue
        data = store.get_bytes(key, start) or b""
        complete = data[: data.rfind(b"\n") + 1]
        offsets[key] = start + len(complete)
        slot, job = key[len(where) :].split("/")[:2]
        out += [f"{slot}/{job}: {line}" for line in complete.decode().splitlines()]
    return out


def follow(args: argparse.Namespace) -> None:
    """Prints what jobs add to their outputs, as the VMs send it, until every slot is done."""
    store, _ = open_store()
    prefix = f"{PREFIX}/{args.name}"
    spec = get_json(store, f"{prefix}/spec.json")
    if spec is None:
        raise SystemExit(f"no job {args.name}")
    offsets: dict[str, int] = {}
    while True:
        done = all(slot_done(store, prefix, s, j) for s, j in spec["slots"].items())
        for line in new_lines(store, prefix, offsets):
            print(line, flush=True)
        if done:
            return
        time.sleep(POLL_SECONDS)


def project() -> str:
    if not CONFIG.exists():
        raise SystemExit(f"no {CONFIG}: run `{Path(__file__).name} setup` first")
    return json.loads(CONFIG.read_text())["project"]


def setup(args: argparse.Namespace) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"project": args.project}))
    print(
        f"project {args.project}; VMs found: {len(Gcloud(args.project).instances(None))}"
    )


def launch(
    store: Store,
    cloud: "Cloud",
    name: str,
    jobs: list[dict[str, Any]] | None,
    out: Path,
    vms: int = 1,
    hours: float = 2.0,
    attempts: int = 3,
    files: list[str] | None = None,
    machine: str = MACHINE,
    each: bool = False,
    bucket: str = "",
    with_engines: bool = True,
) -> dict[str, str]:
    """Runs `jobs` on `cloud`'s hosts as the job `name`, or continues `name` as it was, until
    they're done; each slot's outcome. Hosts get `files` from ~/.cache/turbozero/othello, and with
    `with_engines` every engine's build. Every host of it is deleted when it ends, however it
    ends."""
    if not NAME.fullmatch(name):
        raise SystemExit(f"{name}: lowercase letters, digits and -, up to 31")
    prefix = f"{PREFIX}/{name}"
    spec = get_json(store, f"{prefix}/spec.json")
    if spec is None:
        if jobs is None:
            raise SystemExit(f"{name} is new: give its jobs")
        spec = {
            "slots": split_jobs(jobs, vms, each),
            "bucket": bucket,
            "hours": hours,
            "inputs": upload_inputs(
                store, [*(ENGINES if with_engines else ()), *(files or [])]
            ),
            "env": env_key(machine=machine),
        }
        store.put_bytes(f"{prefix}/code.tar.gz", code_archive())
        put_json(store, f"{prefix}/spec.json", spec)
    else:
        print(f"{name} exists: continuing it, its jobs, files and code as they were")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        outcome = watch(store, cloud, name, spec, out, attempts)
    finally:
        left = cloud.instances(name)
        if left:
            cloud.delete([(vm["name"], vm["zone"]) for vm in left])
            print(f"deleted {', '.join(vm['name'] for vm in left)}")
    print(json.dumps(outcome))
    for slot, how in outcome.items():
        log = (
            store.get_bytes(f"{prefix}/failed/{slot}.log") if how == "failed" else None
        )
        if log:
            print(f"slot {slot}'s log:\n{log.decode()[-3000:]}")
    return outcome


def run(args: argparse.Namespace) -> None:
    store, config = open_store()
    launch(
        store,
        Gcloud(project(), args.machine),
        args.name,
        json.loads(Path(args.jobs).read_text()) if args.jobs else None,
        Path(args.out),
        vms=args.vms,
        hours=args.hours,
        attempts=args.attempts,
        files=args.files,
        machine=args.machine,
        each=args.each,
        bucket=config["bucket"],
    )


def list_vms(_args: argparse.Namespace) -> None:
    for vm in Gcloud(project()).instances(None):
        print(
            f"{vm['name']}  {vm['zone']}  {vm['status']}  job {vm['job']}, slot {vm['slot']}"
        )


def cleanup(args: argparse.Namespace) -> None:
    cloud = Gcloud(project())
    left = cloud.instances(args.name)
    cloud.delete([(vm["name"], vm["zone"]) for vm in left])
    print(f"deleted {len(left)} VMs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Commands on Google Cloud CPUs.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("setup", help="the project to run in; once")
    p.add_argument("--project", required=True)
    p.set_defaults(fn=setup)

    p = commands.add_parser("run", help="run jobs; again with a --name to continue it")
    p.add_argument("--name", required=True, help="the job's name, unique")
    p.add_argument("--jobs", help="a JSON list of {name, cmd}")
    p.add_argument("--out", required=True, help="results, a JSON line each")
    p.add_argument("--vms", type=int, default=1)
    p.add_argument("--each", action="store_true", help="every VM runs every job")
    p.add_argument("--hours", type=float, default=2.0, help="each VM's limit")
    p.add_argument("--attempts", type=int, default=3, help="VMs a slot may start")
    p.add_argument("--machine", default=MACHINE)
    p.add_argument(
        "--files", nargs="*", default=[], help="paths under ~/.cache/turbozero/othello"
    )
    p.set_defaults(fn=run)

    p = commands.add_parser(
        "follow", help="lines jobs add to their outputs, as written"
    )
    p.add_argument("name")
    p.set_defaults(fn=follow)

    p = commands.add_parser("list", help="our VMs")
    p.set_defaults(fn=list_vms)

    p = commands.add_parser("cleanup", help="delete our VMs")
    p.add_argument("--name", default=None, help="only this job's")
    p.set_defaults(fn=cleanup)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
