"""CPU jobs on Google Cloud, with a directory as the bucket and a stand-in for Compute Engine whose
VMs run their slot's jobs (gcp_cpu_runner.run_slot) or get preempted, so a job's whole trip runs
in a second."""

import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from example_modules import load_example

gcp_cpu_jobs, gcp_cpu_runner, modal_store = load_example(
    "othello", "gcp_cpu_jobs", "gcp_cpu_runner", "modal_store"
)


def job(name: str, value: int, tmp: Path | None = None) -> dict[str, Any]:
    """A command printing its result; given `tmp`, writing a file there too."""
    script = f"import json; print('working'); print(json.dumps({{'value': {value}}}))"
    if tmp is None:
        return {"name": name, "cmd": [sys.executable, "-c", script]}
    path = tmp / f"{name}-games.jsonl"
    script += f"; open({str(path)!r}, 'w').write('game {value}')"
    return {"name": name, "cmd": [sys.executable, "-c", script], "outputs": [str(path)]}


def test_jobs_are_dealt_out_in_turn_or_given_to_every_vm():
    jobs = [job(f"j{i}", i) for i in range(5)]
    split = gcp_cpu_jobs.split_jobs(jobs, 2, each=False)
    assert {s: [j["name"] for j in js] for s, js in split.items()} == {
        "0": ["j0", "j2", "j4"],
        "1": ["j1", "j3"],
    }
    # no idle VMs for want of jobs
    assert list(gcp_cpu_jobs.split_jobs(jobs[:1], 3, each=False)) == ["0"]
    each = gcp_cpu_jobs.split_jobs(jobs, 2, each=True)
    assert [len(js) for js in each.values()] == [5, 5]
    with pytest.raises(SystemExit):
        gcp_cpu_jobs.split_jobs([job("a", 1), job("a", 2)], 2, each=False)


def test_a_slot_runs_what_has_no_result_and_keeps_each_result(tmp_path):
    store = modal_store.DirStore(tmp_path)
    prefix = "cpu-jobs/t"
    slots = {"0": [job("a", 1), job("b", 2)]}
    modal_store.put_json(store, f"{prefix}/spec.json", {"slots": slots})
    # a's result is there from a VM that was preempted after it
    done = gcp_cpu_runner.result_key(prefix, 0, "a")
    modal_store.put_json(store, done, {"name": "a", "result": {"value": "old"}})
    gcp_cpu_runner.run_slot(store, prefix, 0, {"vm": "vm-0", "model": "AMD"})
    assert modal_store.get_json(store, done)["result"] == {"value": "old"}
    b = modal_store.get_json(store, gcp_cpu_runner.result_key(prefix, 0, "b"))
    assert b["result"] == {"value": 2}
    assert (b["returncode"], b["vm"], b["model"], b["slot"]) == (0, "vm-0", "AMD", 0)
    log = store.get_bytes(f"{prefix}/logs/0/b.log").decode()
    assert "working" in log


class FakeCloud:
    """Compute Engine whose VMs, on the next look, have run their slot's jobs and deleted
    themselves, or vanished half-way (preempted) when listed in `preempt`."""

    def __init__(self, store, prefix: str, preempt=(), no_capacity=()):
        self.store = store
        self.prefix = prefix
        self.preempt = set(preempt)
        self.no_capacity = set(no_capacity)
        self.vms: dict[str, dict[str, str]] = {}
        self.created: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    def create(self, name, zone, labels, metadata, hours):
        if zone in self.no_capacity:
            return "ZONE_RESOURCE_POOL_EXHAUSTED: no Spot capacity"
        assert metadata["tz-prefix"] == self.prefix and hours == 1.0
        self.created.append((name, zone))
        self.vms[name] = {
            "name": name,
            "zone": zone,
            "status": "RUNNING",
            "slot": labels["tz-cpu-slot"],
        }
        return None

    def instances(self, job):
        for name, vm in list(self.vms.items()):
            if vm.get("seen"):
                if name not in self.preempt:
                    gcp_cpu_runner.run_slot(
                        self.store, self.prefix, int(vm["slot"]), {"vm": name}
                    )
                del self.vms[name]
            else:
                vm["seen"] = "1"
        return [dict(vm) for vm in self.vms.values()]

    def delete(self, names):
        self.deleted += [name for name, _ in names]


def watch(store, cloud, spec, out, attempts=3):
    return gcp_cpu_jobs.watch(
        store, cloud, "t", spec, out, attempts, clock=lambda: 0.0, pause=lambda s: None
    )


def spec(slots):
    return {
        "slots": slots,
        "bucket": "b",
        "hours": 1.0,
        "inputs": ["k1", "k2"],
        "env": "cpu-jobs/env/e.tar",
    }


def test_a_preempted_slot_is_replaced_and_every_result_reaches_out(tmp_path):
    store = modal_store.DirStore(tmp_path / "bucket")
    prefix = "cpu-jobs/t"
    jobs = [job(f"j{i}", i, tmp_path) for i in range(3)]
    slots = gcp_cpu_jobs.split_jobs(jobs, 2, each=False)
    modal_store.put_json(store, f"{prefix}/spec.json", {"slots": slots})
    cloud = FakeCloud(
        store, prefix, preempt={"tz-t-1-0"}, no_capacity={"us-central1-a"}
    )
    out = tmp_path / "results.jsonl"
    assert watch(store, cloud, spec(slots), out) == {"0": "done", "1": "done"}
    names = [vm for vm, _ in cloud.created]
    assert sorted(names) == ["tz-t-0-0", "tz-t-1-0", "tz-t-1-1"]
    assert {zone for _, zone in cloud.created} == {"us-central1-b"}
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted((r["name"], r["result"]["value"]) for r in records) == [
        ("j0", 0),
        ("j1", 1),
        ("j2", 2),
    ]
    games = tmp_path / "results-files" / "1" / "j1" / "j1-games.jsonl"
    assert games.read_text() == "game 1"
    # run again, nothing is written twice and nothing starts
    assert watch(store, cloud, spec(slots), out) == {"0": "done", "1": "done"}
    assert len(out.read_text().splitlines()) == 3 and len(cloud.created) == 3


def test_a_slot_fails_on_a_boot_failure_or_when_out_of_attempts(tmp_path):
    store = modal_store.DirStore(tmp_path / "bucket")
    prefix = "cpu-jobs/t"
    slots = {"0": [job("a", 1)], "1": [job("b", 2)]}
    modal_store.put_json(store, f"{prefix}/spec.json", {"slots": slots})
    store.put_bytes(f"{prefix}/failed/0.log", b"FAILED: boot")
    cloud = FakeCloud(store, prefix, preempt={"tz-t-1-0", "tz-t-1-1"})
    outcome = watch(store, cloud, spec(slots), tmp_path / "out.jsonl", attempts=2)
    assert outcome == {"0": "failed", "1": "failed"}
    assert [vm for vm, _ in cloud.created] == ["tz-t-1-0", "tz-t-1-1"]


def test_inputs_are_uploaded_once_until_they_change(tmp_path):
    cache = tmp_path / "cache"
    (cache / "engines" / "bin").mkdir(parents=True)
    (cache / "engines" / "bin" / "edax").write_text("binary")
    store = modal_store.DirStore(tmp_path / "bucket")
    [key] = gcp_cpu_jobs.upload_inputs(store, ["engines/bin"], cache)
    assert gcp_cpu_jobs.upload_inputs(store, ["engines/bin"], cache) == [key]
    (cache / "engines" / "bin" / "edax").write_text("rebuilt binary")
    [changed] = gcp_cpu_jobs.upload_inputs(store, ["engines/bin"], cache)
    assert changed != key and len(store.sizes("cpu-jobs/inputs/")) == 2
    with pytest.raises(SystemExit):
        gcp_cpu_jobs.input_key("missing", cache)


def test_the_code_is_the_checkouts_files_as_on_disk_but_not_ignored_ones(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "kept.py").write_text("v1")
    (repo / "new.py").write_text("x")
    (repo / ".gitignore").write_text("ignored.npz\n")
    (repo / "ignored.npz").write_text("big")
    subprocess.run(["git", "add", "kept.py"], cwd=repo, check=True)
    (repo / "kept.py").write_text("v2, uncommitted")
    archive = tmp_path / "code.tar.gz"
    archive.write_bytes(gcp_cpu_jobs.code_archive(repo))
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == [".gitignore", "kept.py", "new.py"]
        assert tar.extractfile("kept.py").read() == b"v2, uncommitted"  # type: ignore[union-attr]


def test_the_boot_script_is_valid_bash(tmp_path):
    path = tmp_path / "boot.sh"
    path.write_text(gcp_cpu_jobs.BOOT)
    subprocess.run(["bash", "-n", str(path)], check=True)


class RecordingStore:
    """A DirStore that keeps what each put_file sent, to see outputs go up mid-job."""

    def __init__(self, root):
        self.inner = modal_store.DirStore(root)
        self.sent: list[tuple[str, bytes]] = []

    def put_file(self, key, path):
        self.sent.append((key, Path(path).read_bytes()))
        self.inner.put_file(key, path)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_a_running_jobs_outputs_go_up_as_they_grow(tmp_path):
    store = RecordingStore(tmp_path / "bucket")
    games = tmp_path / "games.jsonl"
    script = (
        "import time\n"
        f"f = open({str(games)!r}, 'a')\n"
        "for i in range(3):\n"
        "    f.write(f'game {i}\\n'); f.flush(); time.sleep(0.3)\n"
        "print('{}')\n"
    )
    job = {"name": "g", "cmd": [sys.executable, "-c", script], "outputs": [str(games)]}
    modal_store.put_json(store, "p/spec.json", {"slots": {"0": [job]}})
    gcp_cpu_runner.run_slot(store, "p", 0, {}, every=0.1)
    sent = [data for key, data in store.sent if key == "p/outputs/0/g/games.jsonl"]
    # sent while it ran, each time it grew, and whole at the end
    assert len(sent) >= 3 and sent[-1] == b"game 0\ngame 1\ngame 2\n"
    assert len(set(sent)) == len(sent)


def test_follow_prints_only_complete_new_lines(tmp_path):
    store = modal_store.DirStore(tmp_path)
    key = "cpu-jobs/t/outputs/1/m/games.jsonl"
    offsets: dict[str, int] = {}
    store.put_bytes(key, b"a\nb\npart")
    assert gcp_cpu_jobs.new_lines(store, "cpu-jobs/t", offsets) == ["1/m: a", "1/m: b"]
    store.put_bytes(key, b"a\nb\npartial\nc\n")
    assert gcp_cpu_jobs.new_lines(store, "cpu-jobs/t", offsets) == [
        "1/m: partial",
        "1/m: c",
    ]
    assert gcp_cpu_jobs.new_lines(store, "cpu-jobs/t", offsets) == []
    # rewritten shorter: read again from the start
    store.put_bytes(key, b"x\n")
    assert gcp_cpu_jobs.new_lines(store, "cpu-jobs/t", offsets) == ["1/m: x"]


def test_a_vm_waits_for_its_cpus_to_go_idle_but_not_forever():
    # idle ticks out of total, cumulative: busy for two windows, then idle
    readings = iter(
        [(0, 100), (50, 200), (50, 200), (100, 300), (100, 300), (199, 400)]
    )
    share = gcp_cpu_runner.settle(lambda: next(readings), lambda s: None, 5.0, 100.0)
    assert share == 0.99
    always_busy = gcp_cpu_runner.settle(lambda: (0, 100), lambda s: None, 5.0, 20.0)
    assert always_busy == 0.0


def test_each_machine_gets_its_image_disk_and_threads():
    assert gcp_cpu_jobs.machine_options("n4d-highcpu-16") == [
        "--image-family=ubuntu-2404-lts-amd64",
        "--boot-disk-type=hyperdisk-balanced",
        "--threads-per-core=1",
    ]
    # Milan's C2D boots only from persistent disks; Axion is ARM, a thread a core already
    assert "--boot-disk-type=pd-balanced" in gcp_cpu_jobs.machine_options(
        "c2d-standard-16"
    )
    axion = gcp_cpu_jobs.machine_options("c4a-standard-8")
    assert "--image-family=ubuntu-2404-lts-arm64" in axion
    assert not any(o.startswith("--threads-per-core") for o in axion)
    # an environment holds one architecture's packages and venv
    assert gcp_cpu_jobs.env_key(machine="c4a-standard-8") != gcp_cpu_jobs.env_key()
    assert gcp_cpu_jobs.env_key(machine="c2d-standard-16") == gcp_cpu_jobs.env_key()
