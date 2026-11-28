"""Modal jobs, with a directory as the bucket and a stand-in for train.py that writes what it
writes (checkpoints, states, a real seaglass run), so a job's whole trip runs in seconds."""

import dataclasses
import json
import socket
import textwrap
import threading
import time
from pathlib import Path

import pytest
from example_modules import load_example

modal_jobs, modal_run, modal_spec, modal_store, modal_sync = load_example(
    "othello", "modal_jobs", "modal_run", "modal_spec", "modal_store", "modal_sync"
)

# as train.py: rolling checkpoints, keep-every, states, --resume, a run that continues
FAKE_TRAIN = """
import argparse, json, os, re, signal, sys, time
from pathlib import Path
from seaglass import Monitor

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int)
p.add_argument("--ckpt-dir")
p.add_argument("--keep-every", type=int)
p.add_argument("--save-state-at", default="")
p.add_argument("--save-state-every", type=int)
# as train.py's: a run starts one of these ways
start = p.add_mutually_exclusive_group()
start.add_argument("--init-from")
start.add_argument("--resume", action="store_true")
start.add_argument("--branch-from")
p.add_argument("--init-without-buffers", action="store_true")
p.add_argument("--fork-into")
p.add_argument("--warm-start", action="append", default=[])
p.add_argument("--take-over", action="store_true")
p.add_argument("--monitor", nargs="?", const="")
p.add_argument("--name")
p.add_argument("--notes")
p.add_argument("--die-at", type=int)
p.add_argument("--fail-at", type=int)
p.add_argument("--hang-at", type=int)
p.add_argument("--grow-into")
args = p.parse_args()
ckpt = Path(args.ckpt_dir)
start, run_id = 0, None
if args.resume:
    states = sorted(int(re.fullmatch(r"state-(\\d+).npz", f.name).group(1)) for f in ckpt.glob("state-*.npz"))
    start = states[-1]
    run_id = json.loads((ckpt / f"state-{start}.npz").read_text())["run"]
    print(f"resuming from {start}", flush=True)
if args.init_from:
    assert Path(args.init_from).exists()
if args.branch_from:
    assert Path(args.branch_from).exists()
    print(f"branching from {args.branch_from}", flush=True)
if args.take_over and not args.branch_from:
    sys.exit("--take-over goes with --branch-from")
if args.fork_into and not args.init_from:
    sys.exit("--fork-into needs --init-from")
if args.warm_start and args.resume:
    sys.exit("--warm-start is for a new run")
for spec in args.warm_start:
    assert Path(spec.partition("=")[2]).exists()
at = {int(e) for e in args.save_state_at.split(",") if e}
monitor = None
if args.monitor is not None:
    monitor = Monitor(args.monitor, project="othello", name=args.name, notes=args.notes)
    monitor.run_id = run_id
    monitor.start(config={"args": sys.argv[1:]}, total=args.epochs - 1)
try:
    for epoch in range(start, args.epochs):
        if epoch == args.die_at and not os.path.exists(str(ckpt) + ".died"):
            Path(str(ckpt) + ".died").touch()
            if monitor:
                monitor._queue.join()
            time.sleep(0.5)  # a mirror tick gets what's there
            os.kill(os.getpid(), signal.SIGKILL)
        if epoch == args.fail_at:
            sys.exit(3)
        if epoch == args.hang_at:
            time.sleep(60)
        (ckpt / f"{epoch}.eqx").write_text("net")
        if args.grow_into:  # as a growth run: a network's checkpoints in a directory of its own
            (ckpt / args.grow_into).mkdir(exist_ok=True)
            (ckpt / args.grow_into / f"{epoch}.eqx").write_text("grown")
        kept = [
            e
            for e in sorted(int(f.stem) for f in ckpt.glob("*.eqx"))
            if not (args.keep_every and e % args.keep_every == 0)
        ]
        for old in kept[:-2]:
            (ckpt / f"{old}.eqx").unlink()
        print(f"Epoch {epoch}: {{'loss': 1.0}}", flush=True)
        if monitor:
            monitor.log(epoch, {"loss": 1.0 / (epoch + 1)})
        done = epoch + 1
        if done in at or (args.save_state_every and done % args.save_state_every == 0):
            path = ckpt / f"state-{done}.npz"
            path.with_suffix(".tmp").write_text(json.dumps({"run": monitor.run_id if monitor else None}))
            os.replace(path.with_suffix(".tmp"), path)
            if monitor:
                monitor._queue.join()
            if args.save_state_every:
                for f in ckpt.glob("state-*.npz"):
                    e = int(f.name[6:-4])
                    if e < done and e % args.save_state_every == 0 and e not in at:
                        f.unlink()
except KeyboardInterrupt:
    if monitor:
        monitor.finish("stopped")
    raise
if monitor:
    monitor.finish()
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Scene:
    """A home here, a bucket, a pretend container's repo, and slick's seaglass root."""

    def __init__(self, tmp: Path):
        self.home = tmp / "home"
        self.root = self.home / ".cache" / "turbozero"
        self.store = modal_store.DirStore(tmp / "bucket")
        self.repo = tmp / "repo"
        script = self.repo / "examples" / "othello" / "train.py"
        script.parent.mkdir(parents=True)
        script.write_text(textwrap.dedent(FAKE_TRAIN))
        self.slick = tmp / "slick"
        self.slick.mkdir()
        self.tmp = tmp
        self.store.put_bytes(
            ".cache/x", b""
        )  # an unrelated object: nothing should trip on it
        self.store.put_bytes(
            "files/.cache/turbozero/othello/engine-sets/nr26-evals-29k.npz", b"set"
        )

    def job(self, *steps: str, hours: float = 1.0, name: str = "j", **more):
        job = modal_spec.Job(
            name=name,
            commit="abc",
            gpu="L40S",
            hours=hours,
            home=str(self.home),
            steps=[modal_spec.parse_step(step, str(self.home)) for step in steps],
            tags="slick:personal/turbozero/t,modal",
            **more,
        )
        modal_store.put_json(
            self.store, modal_spec.job_key(name, "job.json"), job.to_dict()
        )
        return job

    def place(self):
        import sys

        return modal_run.Place(
            repo=str(self.repo),
            python=sys.executable,
            seaglass_root=str(self.tmp / "container-seaglass"),
            seaglass_port=free_port(),
            mirror_seconds=0.1,
            poll_seconds=0.02,
        )

    def wipe_container(self):
        """What a preemption does to the container's disk: only the bucket survives."""
        import shutil

        shutil.rmtree(self.home, ignore_errors=True)
        shutil.rmtree(self.tmp / "container-seaglass", ignore_errors=True)

    def runner(self, name: str = "j"):
        return modal_run.Runner(
            self.store, name, self.place(), describe_gpu=lambda: {"chip": "fake"}
        )

    def sync(self):
        return modal_sync.Sync(
            self.store,
            str(self.home),
            self.slick,
            self.tmp / "sync.json",
            seaglass_url=None,
        )


@pytest.fixture
def scene(tmp_path):
    return Scene(tmp_path)


def step(name: str, flags: str = "--epochs 5", home: str = "~") -> str:
    return (
        f"examples/othello/train.py {flags} --monitor --name {name} "
        f"--ckpt-dir {home}/.cache/turbozero/c/{name} > {home}/.cache/turbozero/l/{name}.log 2>&1"
    )


# --- steps


def test_parse_step_takes_a_local_command():
    parsed = modal_spec.parse_step(
        "PYTHONUNBUFFERED=1 taskset -c 0-7 uv run --extra cuda examples/othello/train.py --epochs 2 "
        "--notes 'a test' --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l.log 2>&1 < /dev/null",
        "/h",
    )
    assert parsed.log == "/h/.cache/turbozero/l.log"
    assert parsed.ckpt_dir == "/h/.cache/turbozero/c"
    assert parsed.argv == [
        "--epochs",
        "2",
        "--notes",
        "a test",
        "--ckpt-dir",
        "/h/.cache/turbozero/c",
    ]


@pytest.mark.parametrize(
    "text, says",
    [
        ("python other.py --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l", "train.py"),
        ("examples/othello/train.py --ckpt-dir ~/.cache/turbozero/c", "> LOG"),
        ("examples/othello/train.py > ~/.cache/turbozero/l", "--ckpt-dir"),
        ("examples/othello/train.py --ckpt-dir /tmp/c > ~/.cache/turbozero/l", "wouldn't come back"),
        ("examples/othello/train.py --resume --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l", "--resume"),
        ("examples/othello/train.py --log-moves ~/.cache/turbozero/m --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l", "local runs"),
        (
            "examples/othello/train.py --init-from ~/.cache/turbozero/dir --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l",
            "state-<EPOCH>",
        ),
        (
            "examples/othello/train.py --branch-from ~/.cache/turbozero/dir --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l",
            "--branch-from names a state-<EPOCH>",
        ),
        (
            "examples/othello/train.py --branch-from /tmp/state-1.npz --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l",
            "wouldn't come back",
        ),
    ],
)  # fmt: skip
def test_parse_step_refuses_what_wouldnt_work(text, says):
    with pytest.raises(ValueError, match=says):
        modal_spec.parse_step(text, "/h")


def test_with_monitor_replaces_the_server():
    assert modal_spec.with_monitor(["--monitor", "--x"], "u") == [
        "--monitor",
        "u",
        "--x",
    ]
    assert modal_spec.with_monitor(["--monitor", "http://slick"], "u") == [
        "--monitor",
        "u",
    ]
    assert modal_spec.with_monitor(["--monitor=a"], "u") == ["--monitor=u"]
    assert modal_spec.with_monitor(["--x"], "u") == ["--x"]  # fmt: skip


def test_cost_per_hour_is_the_list_prices():
    job = modal_spec.Job("j", "c", "L40S", 1, "/h", [], cpu=4, memory_gb=16)
    assert job.dollars_per_hour() == pytest.approx(1.951 + 0.189 + 0.128, abs=0.01)


# --- a whole job


def test_a_sequence_ends_up_here_like_a_local_run(scene):
    home = scene.home
    scene.job(
        step("trunk", "--epochs 25 --keep-every 10 --save-state-at 20 --notes why"),
        step(
            "fork",
            "--epochs 3 --keep-every 1 "
            f"--init-from {home}/.cache/turbozero/c/trunk/state-20.npz",
        ),
    )
    assert scene.runner().run() == "done"
    container = sorted(p.name for p in (scene.root / "c" / "trunk").iterdir())
    scene.wipe_container()
    lines = scene.sync().run()
    assert any("done, all here" in line for line in lines)

    trunk = scene.root / "c" / "trunk"
    names = {p.name for p in trunk.iterdir()}
    # as a local run keeps them: multiples of 10, the 2 newest, the state asked for
    assert names == {"0.eqx", "10.eqx", "20.eqx", "23.eqx", "24.eqx", "state-20.npz", "modal-job.json"}  # fmt: skip
    assert "state-10.npz" not in names  # only saved to continue from
    assert "state-20.npz" in container
    assert {p.name for p in (scene.root / "c" / "fork").glob("*.eqx")} == {"0.eqx", "1.eqx", "2.eqx"}  # fmt: skip

    log = (scene.root / "l" / "trunk.log").read_text()
    assert "Epoch 24:" in log

    record = json.loads((trunk / "modal-job.json").read_text())
    assert record["commit"] == "abc"
    assert record["attempts"][0]["chip"] == "fake"
    assert record["outcome"] == "done"
    assert record["seconds"] > 0

    metas = [json.loads(p.read_text()) for p in scene.slick.glob("*/meta.json")]
    assert {m["name"] for m in metas} == {"trunk", "fork"}
    meta = next(m for m in metas if m["name"] == "trunk")
    assert meta["status"] == "finished"
    assert meta["notes"] == "why"
    assert "slick:personal/turbozero/t" in meta["tags"]
    assert meta["provenance"]["modal"]["chip"] == "fake"
    rows = (scene.slick / meta["id"] / "metrics.jsonl").read_text().splitlines()
    assert len(rows) == 25


def test_a_growth_runs_networks_directories_end_up_here(scene):
    scene.job(step("grow", "--epochs 3 --keep-every 1 --grow-into 4x64"))
    assert scene.runner().run() == "done"
    scene.wipe_container()
    scene.sync().run()
    grown = scene.root / "c" / "grow" / "4x64"
    assert sorted(p.name for p in grown.iterdir()) == ["0.eqx", "1.eqx", "2.eqx"]
    assert (grown / "2.eqx").read_text() == "grown"


def test_a_preempted_job_goes_on_from_its_state_on_a_new_machine(scene):
    scene.job(
        step("done-first", "--epochs 2"),
        step("t", "--epochs 12 --keep-every 5 --save-state-at 12 --die-at 8"),
        name="p",
        resume_every=5,
    )
    with pytest.raises(RuntimeError, match="lost"):
        scene.runner("p").run()
    died = scene.root / "c" / "t.died"  # the stand-in's own memory of having died once
    assert died.exists()
    keep = died.read_text()
    scene.wipe_container()
    died.parent.mkdir(parents=True)
    died.write_text(keep)

    assert scene.runner("p").run() == "done"
    status = modal_store.get_json(scene.store, modal_spec.job_key("p", "status.json"))
    assert len(status["attempts"]) == 2

    log = (scene.root / "l" / "t.log").read_text()
    assert "resuming from 5" in log
    # the step finished on the first attempt isn't run again
    assert not (scene.root / "c" / "done-first").exists()

    scene.wipe_container()
    scene.sync().run()
    names = {p.name for p in (scene.root / "c" / "t").iterdir()}
    assert names == {"0.eqx", "5.eqx", "9.eqx", "10.eqx", "11.eqx", "state-12.npz", "modal-job.json"}  # fmt: skip
    metas = [json.loads(p.read_text()) for p in scene.slick.glob("*/meta.json")]
    # one run of the step, across both attempts, finished
    assert sorted(m["name"] for m in metas) == ["done-first", "t"]
    t = next(m for m in metas if m["name"] == "t")
    assert t["status"] == "finished"
    assert t["step"] == 11


def test_steps_finished_before_a_preemption_keep_their_files_in_the_manifest(scene):
    # two segments done, the third branched from the second's end and preempted
    c = f"{scene.home}/.cache/turbozero/c"
    scene.job(
        step("a", "--epochs 2 --save-state-at 2"),
        step("b", f"--epochs 4 --save-state-at 4 --branch-from {c}/a/state-2.npz"),
        step("t", f"--epochs 12 --die-at 8 --branch-from {c}/b/state-4.npz"),
        name="p",
        resume_every=5,
    )
    with pytest.raises(RuntimeError, match="lost"):
        scene.runner("p").run()
    manifest_key = modal_spec.job_key("p", "manifest.json")
    before = modal_store.get_json(scene.store, manifest_key)
    earlier = sorted(k for k in before if "/c/a/" in k or "/c/b/" in k)
    assert any(k.endswith("/c/a/1.eqx") for k in earlier)
    assert any(k.endswith("/c/b/modal-job.json") for k in earlier)
    died = scene.root / "c" / "t.died"
    keep = died.read_text()
    scene.wipe_container()
    died.parent.mkdir(parents=True)
    died.write_text(keep)

    # the new machine runs only the third; the first two's files aren't on its disk (but
    # b's final state, which the third branches from), and aren't gone from the run
    assert scene.runner("p").run() == "done"
    assert not (scene.root / "c" / "a").exists()
    after = modal_store.get_json(scene.store, manifest_key)
    assert {k: after.get(k) for k in earlier} == {k: before[k] for k in earlier}
    scene.wipe_container()
    scene.sync().run()
    assert {p.name for p in (scene.root / "c" / "a").iterdir()} == {"0.eqx", "1.eqx", "state-2.npz", "modal-job.json"}  # fmt: skip
    assert {p.name for p in (scene.root / "c" / "b").iterdir()} == {"2.eqx", "3.eqx", "state-4.npz", "modal-job.json"}  # fmt: skip


def test_a_forked_step_takes_its_states_and_resumes_without_them(scene):
    # states of runs here, which the launch uploads
    trunk = f"{scene.home}/.cache/turbozero/c/trunk/state-20.npz"
    other = f"{scene.home}/.cache/turbozero/c/other/state-5.npz"
    for path in (trunk, other):
        scene.store.put_bytes(path.replace(f"{scene.home}/", "files/"), b"{}")
    job = scene.job(
        step(
            "t",
            f"--epochs 12 --keep-every 5 --die-at 8 --init-from {trunk} --fork-into 8x128 "
            "--warm-start 4x64=~/.cache/turbozero/c/other/state-5.npz",
        ),
        name="p",
        resume_every=5,
    )
    assert modal_jobs.inputs_to_upload(job)[1:] == [trunk, other]
    with pytest.raises(RuntimeError, match="lost"):
        scene.runner("p").run()
    died = scene.root / "c" / "t.died"
    keep = died.read_text()
    scene.wipe_container()
    died.parent.mkdir(parents=True)
    died.write_text(keep)

    # the new machine goes on from the step's state, without how the step started
    assert scene.runner("p").run() == "done"
    assert "resuming from 5" in (scene.root / "l" / "t.log").read_text()


def test_a_branched_step_takes_its_state_and_resumes_without_it(scene):
    # a growth run's state here, which the launch uploads: e.g. a step of an LR staircase
    trunk = f"{scene.home}/.cache/turbozero/c/trunk/state-20.npz"
    scene.store.put_bytes(trunk.replace(f"{scene.home}/", "files/"), b"{}")
    job = scene.job(
        step(
            "t",
            f"--epochs 32 --keep-every 5 --die-at 28 --branch-from {trunk} --take-over",
        ),
        name="p",
        resume_every=5,
    )
    assert modal_jobs.inputs_to_upload(job)[1:] == [trunk]
    with pytest.raises(RuntimeError, match="lost"):
        scene.runner("p").run()
    assert "branching from" in (scene.root / "l" / "t.log").read_text()
    died = scene.root / "c" / "t.died"
    keep = died.read_text()
    scene.wipe_container()
    died.parent.mkdir(parents=True)
    died.write_text(keep)

    # the new machine goes on from the branch's own state: --branch-from and --take-over were
    # how the branch started (train.py refuses either with --resume)
    assert scene.runner("p").run() == "done"
    log = (scene.root / "l" / "t.log").read_text()
    # (the log goes on from the first machine's)
    assert "resuming from 25" in log and log.count("branching from") == 1


def test_a_branch_of_an_earlier_steps_state_isnt_uploaded(scene):
    first = f"{scene.home}/.cache/turbozero/c/a"
    job = scene.job(
        step("a", "--epochs 4 --save-state-at 4"),
        step("b", f"--epochs 6 --branch-from {first}/state-4.npz"),
        name="p",
    )
    assert modal_jobs.inputs_to_upload(job)[1:] == []
    assert scene.runner("p").run() == "done"
    assert (scene.root / "c" / "b" / "5.eqx").exists()


@pytest.mark.parametrize(
    "warm_start", ["4x64=~/.cache/turbozero/c/x", "4x64=/tmp/state-1.npz"]
)
def test_parse_step_refuses_a_warm_start_it_cant_bring(warm_start):
    with pytest.raises(ValueError, match="state-<EPOCH>|wouldn't come back"):
        modal_spec.parse_step(
            f"examples/othello/train.py --warm-start {warm_start} "
            "--ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l",
            "/h",
        )


def test_a_step_interrupted_before_its_first_state_starts_over(scene):
    scene.job(step("t", "--epochs 4 --die-at 1"), name="p", resume_every=10)
    with pytest.raises(RuntimeError):
        scene.runner("p").run()
    died = (scene.root / "c" / "t.died").read_text()
    scene.wipe_container()
    (scene.root / "c").mkdir(parents=True)
    (scene.root / "c" / "t.died").write_text(died)
    assert scene.runner("p").run() == "done"
    scene.wipe_container()
    scene.sync().run()
    metas = [json.loads(p.read_text()) for p in scene.slick.glob("*/meta.json")]
    assert sorted(m["status"] for m in metas) == ["crashed", "finished"]


def test_a_failing_step_ends_the_job_without_retrying(scene):
    scene.job(step("t", "--epochs 4 --fail-at 1"), step("u"), name="f")
    assert scene.runner("f").run() == "failed"
    status = modal_store.get_json(scene.store, modal_spec.job_key("f", "status.json"))
    assert status["state"] == "failed"
    assert not (scene.root / "c" / "u").exists()
    assert "Epoch 0:" in (scene.store.get_bytes("files/.cache/turbozero/l/t.log") or b"").decode()  # fmt: skip


def test_the_time_limit_stops_a_job_across_attempts(scene):
    # long enough for its first epoch to log, however slowly the machine starts it
    scene.job(step("t", "--epochs 4 --hang-at 1"), name="l", hours=5 / 3600)
    assert scene.runner("l").run() == "timeout"
    scene.wipe_container()
    scene.sync().run()
    meta = json.loads(next(scene.slick.glob("*/meta.json")).read_text())
    assert meta["status"] == "stopped"


def test_a_stop_in_the_bucket_stops_a_job(scene):
    scene.job(step("t", "--epochs 4"), name="s")
    modal_store.put_json(
        scene.store, modal_spec.job_key("s", "stop.json"), {"stop": True}
    )
    assert scene.runner("s").run() == "stopped"


def test_a_bucket_outage_doesnt_stop_training(scene):
    scene.job(step("t", "--epochs 3"), name="o")

    class Flaky(modal_store.DirStore):
        failures = 3

        def put_bytes(self, key, data):
            if Flaky.failures > 0 and "manifest" in key:
                Flaky.failures -= 1
                raise OSError("bucket away")
            super().put_bytes(key, data)

    flaky = Flaky(scene.store.root)
    runner = modal_run.Runner(flaky, "o", scene.place(), describe_gpu=dict)
    assert runner.run() == "done"
    assert Flaky.failures == 0
    assert modal_store.get_json(scene.store, modal_spec.job_key("o", "manifest.json"))


def test_a_heartbeat_in_flight_at_the_end_doesnt_undo_the_final_status(scene):
    """A slow bucket holds the mirror thread's heartbeat until the job's final status is written
    (or a quarter second): written after it, its "running" would be the job's last word."""
    scene.job(step("t", "--epochs 2"), name="h")

    class Slow(modal_store.DirStore):
        # held while a heartbeat of the mirror thread's is written
        writing = threading.Lock()
        in_flight = threading.Event()
        final = threading.Event()

        def put_bytes(self, key, data):
            if not key.endswith("status.json"):
                return super().put_bytes(key, data)
            if threading.current_thread() is not threading.main_thread():
                with Slow.writing:
                    Slow.in_flight.set()
                    Slow.final.wait(0.25)
                    super().put_bytes(key, data)
                    Slow.in_flight.clear()
            elif json.loads(data)["state"] == "running":
                super().put_bytes(key, data)
            else:
                Slow.in_flight.wait(0.25)
                super().put_bytes(key, data)
                Slow.final.set()

    # ticking often enough that one is under way however soon the step ends
    place = dataclasses.replace(scene.place(), mirror_seconds=0.01)
    runner = modal_run.Runner(Slow(scene.store.root), "h", place, dict)
    assert runner.run() == "done"
    with Slow.writing:  # what was still being written has landed
        status = modal_store.get_json(
            scene.store, modal_spec.job_key("h", "status.json")
        )
    assert status["state"] == "done"


class Recording(modal_store.DirStore):
    """A bucket that notes each key written to it."""

    def __init__(self, root):
        super().__init__(root)
        self.puts: list[str] = []

    def put_file(self, key, path):
        self.puts.append(key)
        super().put_file(key, path)

    def put_bytes(self, key, data):
        self.puts.append(key)
        super().put_bytes(key, data)


def test_a_new_attempt_doesnt_send_back_the_state_it_branches_from(scene):
    # a job's step saved a state and the job ended; a step added later branches from it
    c = f"{scene.home}/.cache/turbozero/c"
    scene.job(step("a", "--epochs 3 --save-state-at 3"), name="p")
    assert scene.runner("p").run() == "done"
    manifest_key = modal_spec.job_key("p", "manifest.json")
    state_key = "files/.cache/turbozero/c/a/state-3.npz"
    before = modal_store.get_json(scene.store, manifest_key)[state_key]
    branch = step("b", f"--epochs 2 --branch-from {c}/a/state-3.npz")
    modal_jobs.extend_job(
        scene.store, "p", [modal_spec.parse_step(branch, str(scene.home))]
    )
    scene.wipe_container()
    store = Recording(scene.store.root)
    runner = modal_run.Runner(store, "p", scene.place(), describe_gpu=dict)
    assert runner.run() == "done"
    assert "branching from" in (scene.root / "l" / "b.log").read_text()
    assert state_key not in store.puts
    # nor does the manifest say it changed: the sync would bring it again
    assert modal_store.get_json(scene.store, manifest_key)[state_key] == before


def test_a_new_attempt_doesnt_send_back_what_it_resumes_from(scene):
    scene.job(
        step("t", "--epochs 12 --keep-every 5 --die-at 8"), name="p", resume_every=5
    )
    with pytest.raises(RuntimeError, match="lost"):
        scene.runner("p").run()
    died = scene.root / "c" / "t.died"
    keep = died.read_text()
    scene.wipe_container()
    died.parent.mkdir(parents=True)
    died.write_text(keep)

    store = Recording(scene.store.root)
    runner = modal_run.Runner(store, "p", scene.place(), describe_gpu=dict)
    assert runner.run() == "done"
    assert "resuming from 5" in (scene.root / "l" / "t.log").read_text()
    resume_key = modal_spec.job_key("p", "resume", "0", "state.npz")
    # the state it went on from was in the bucket: only the next one, of epoch 10, is sent
    assert store.puts.count(resume_key) == 1
    assert modal_store.get_json(scene.store, resume_key + ".json") == {"epoch": 10}
    # a checkpoint it brought back and didn't write again
    assert "files/.cache/turbozero/c/t/0.eqx" not in store.puts


def test_a_state_on_its_way_doesnt_hold_up_the_rest(scene):
    """A big state takes minutes to send: meanwhile the checkpoints, the log, the seaglass
    rows and the manifest go on arriving."""
    scene.job(step("t", "--epochs 6 --save-state-at 2"), name="g")
    manifest_key = modal_spec.job_key("g", "manifest.json")

    class Gate(modal_store.DirStore):
        release = threading.Event()

        def put_bytes(self, key, data):
            if key.endswith(".npz"):
                assert Gate.release.wait(30)
            super().put_bytes(key, data)

    seen = {}

    def watch():
        deadline = time.time() + 20
        while time.time() < deadline:
            manifest = modal_store.get_json(scene.store, manifest_key) or {}
            rows = [
                key
                for key, size in scene.store.sizes("jobs/g/seaglass/").items()
                if key.endswith("metrics.jsonl") and size
            ]
            if "files/.cache/turbozero/c/t/5.eqx" in manifest and rows:
                rows_there = scene.store.get_bytes(rows[0]).decode().splitlines()
                if len(rows_there) == 6:
                    seen["manifest"] = manifest
                    break
            time.sleep(0.05)
        Gate.release.set()

    thread = threading.Thread(target=watch)
    thread.start()
    runner = modal_run.Runner(Gate(scene.store.root), "g", scene.place(), dict)
    assert runner.run() == "done"
    thread.join()
    assert "manifest" in seen, "the rest waited for the state"
    state_key = "files/.cache/turbozero/c/t/state-2.npz"
    assert state_key not in seen["manifest"]  # listed once it's there, not before
    assert state_key in modal_store.get_json(scene.store, manifest_key)
    assert scene.store.sizes(state_key)


# --- sync


def make_finished_job(scene, name="j", flags="--epochs 3 --notes why"):
    scene.job(step(name, flags), name=name)
    assert scene.runner(name).run() == "done"
    scene.wipe_container()


def test_syncing_again_keeps_what_slick_edited(scene):
    make_finished_job(scene)
    sync = scene.sync()
    # not over yet, as the sync sees it: so that it reads the run twice
    status_key = modal_spec.job_key("j", "status.json")
    status = modal_store.get_json(scene.store, status_key)
    modal_store.put_json(scene.store, status_key, {**status, "state": "running"})
    sync.run()
    meta_path = next(scene.slick.glob("*/meta.json"))
    meta = json.loads(meta_path.read_text())
    meta["notes"] = "edited in slick"
    meta["conclusions"] = "it showed"
    meta_path.write_text(json.dumps(meta))
    # the job goes on
    run = meta["id"]
    key = modal_spec.job_key("j", "seaglass", run, "meta.json")
    there = json.loads(scene.store.get_bytes(key))
    scene.store.put_bytes(key, json.dumps({**there, "step": 99}).encode())
    sync.run()
    meta = json.loads(meta_path.read_text())
    assert meta["notes"] == "edited in slick"
    assert meta["conclusions"] == "it showed"
    assert meta["step"] == 99


def test_sync_appends_only_whole_rows(scene):
    make_finished_job(scene)
    status_key = modal_spec.job_key("j", "status.json")
    status = modal_store.get_json(scene.store, status_key)
    modal_store.put_json(scene.store, status_key, {**status, "state": "running"})
    sync = scene.sync()
    sync.run()
    run = next(scene.slick.glob("*/meta.json")).parent
    before = (run / "metrics.jsonl").read_bytes()
    key = modal_spec.job_key("j", "seaglass", run.name, "metrics.jsonl")
    there = scene.store.get_bytes(key)
    scene.store.put_bytes(key, there + b'{"step":7}\n{"step":8,"lo')
    sync.run()
    assert (run / "metrics.jsonl").read_bytes() == before + b'{"step":7}\n'
    scene.store.put_bytes(key, there + b'{"step":7}\n{"step":8,"loss":1}\n')
    sync.run()
    assert (run / "metrics.jsonl").read_bytes() == before + b'{"step":7}\n{"step":8,"loss":1}\n'  # fmt: skip


def test_sync_leaves_alone_a_file_it_didnt_bring(scene):
    make_finished_job(scene)
    mine = scene.root / "c" / "j" / "2.eqx"
    mine.parent.mkdir(parents=True)
    mine.write_text("a local run's")
    lines = scene.sync().run()
    assert mine.read_text() == "a local run's"
    assert any("left alone" in line for line in lines)


def test_sync_drops_only_checkpoints_the_run_rolled_off(scene):
    make_finished_job(scene)
    manifest_key = modal_spec.job_key("j", "manifest.json")
    status_key = modal_spec.job_key("j", "status.json")
    status = modal_store.get_json(scene.store, status_key)
    modal_store.put_json(scene.store, status_key, {**status, "state": "running"})
    sync = scene.sync()
    sync.run()
    manifest = modal_store.get_json(scene.store, manifest_key)
    gone = next(key for key in manifest if key.endswith("1.eqx"))
    other = scene.root / "c" / "j" / "not-brought-here"
    other.write_text("mine")
    del manifest[gone]
    modal_store.put_json(scene.store, manifest_key, manifest)
    sync.run()
    assert not (scene.root / "c" / "j" / "1.eqx").exists()
    assert (scene.root / "c" / "j" / "2.eqx").exists()
    assert other.exists()
    assert scene.store.get_bytes(gone) is not None  # nothing leaves the bucket


def test_a_finished_job_is_left_alone_after_its_last_sync(scene):
    make_finished_job(scene)
    sync = scene.sync()
    assert sync.run()
    assert sync.run() == []


def test_one_jobs_failure_doesnt_stop_the_others(scene):
    make_finished_job(scene, name="a")
    make_finished_job(scene, name="b")

    class Away(modal_store.DirStore):
        def get_bytes(self, key, start=0):
            if key == modal_spec.job_key("a", "manifest.json"):
                raise TimeoutError("timed out")
            return super().get_bytes(key, start)

    sync = modal_sync.Sync(
        Away(scene.store.root),
        str(scene.home),
        scene.slick,
        scene.tmp / "sync.json",
        None,
    )
    lines = sync.run()
    assert any(line.startswith("a: sync failed") for line in lines)
    assert sync.failed == ["a"]
    assert (scene.root / "l" / "b.log").exists()
    assert any("b: done, all here" in line for line in lines)
    # tried again next time, and done then
    lines = scene.sync().run()
    assert any("a: done, all here" in line for line in lines)


def test_each_bundle_is_asked_about_once(scene):
    make_finished_job(scene)
    status_key = modal_spec.job_key("j", "status.json")
    status = modal_store.get_json(scene.store, status_key)
    modal_store.put_json(scene.store, status_key, {**status, "state": "running"})
    for rel in ("have/a.json", "have/b.json", "new/a.json", "new/b.json"):
        key = modal_spec.job_key("j", "seaglass", "bundles", rel)
        scene.store.put_bytes(key, b"{}")

    asked: list[str] = []
    sent: dict[str, list[str]] = {}

    class Slick(modal_sync.Sync):
        def _has_bundle(self, bundle):
            asked.append(bundle)
            return bundle == "have"

        def _put_bundle(self, bundle, files):
            sent[bundle] = sorted(files)

    sync = Slick(
        scene.store, str(scene.home), scene.slick, scene.tmp / "sync.json", "http://x"
    )
    sync.run()
    sync.run()
    assert sorted(asked) == ["have", "new"]
    assert sent == {"new": ["a.json", "b.json"]}


def test_a_job_run_again_after_its_end_is_synced_again(scene):
    make_finished_job(scene)
    sync = scene.sync()
    sync.run()
    assert sync.run() == []
    # a status rewritten without a new attempt (a last heartbeat): still over
    status_key = modal_spec.job_key("j", "status.json")
    status = modal_store.get_json(scene.store, status_key)
    modal_store.put_json(scene.store, status_key, {**status, "heartbeat": 12345678.9})
    assert sync.run() == []
    # `resume`: a new attempt, running
    attempts = [*status["attempts"], {"start": 1.0, "seconds": 0.0}]
    modal_store.put_json(scene.store, status_key, {**status, "state": "running", "attempts": attempts})  # fmt: skip
    log_key = "files/.cache/turbozero/l/j.log"
    scene.store.put_bytes(log_key, scene.store.get_bytes(log_key) + b"more\n")
    manifest_key = modal_spec.job_key("j", "manifest.json")
    manifest = modal_store.get_json(scene.store, manifest_key)
    manifest[log_key] = [manifest[log_key][0] + 5, 1]
    modal_store.put_json(scene.store, manifest_key, manifest)
    lines = scene.sync().run()  # a sync started since, from what the last one saved
    assert "j: running again" in lines
    assert (scene.root / "l" / "j.log").read_text().endswith("more\n")


def test_jobs_in_finds_jobs(scene):
    scene.job(step("a"), name="a")
    scene.job(step("b"), name="b")
    assert modal_sync.jobs_in(scene.store) == ["a", "b"]


# --- jobs that grow: a driver appends segments as they run


def test_a_following_job_runs_steps_added_while_it_waits(scene):
    home = scene.home
    first = f"{home}/.cache/turbozero/c/a"
    scene.job(
        step("a", "--epochs 3 --save-state-at 3"),
        name="p",
        follow_minutes=0.05,
        states_in_bucket=True,
    )
    runner = scene.runner("p")
    added = threading.Event()

    def driver():
        # once the first segment is done, as a driver would
        while not (Path(first) / "state-3.npz").exists():
            threading.Event().wait(0.05)
        threading.Event().wait(0.3)
        modal_jobs.extend_job(
            scene.store,
            "p",
            [
                modal_spec.parse_step(
                    step("b", f"--epochs 2 --branch-from {first}/state-3.npz"),
                    str(home),
                )
            ],
        )
        added.set()

    thread = threading.Thread(target=driver)
    thread.start()
    assert runner.run() == "done"
    thread.join()
    assert added.is_set()
    assert "branching from" in (scene.root / "l" / "b.log").read_text()
    # its states stay in the bucket: the sync leaves them there
    scene.wipe_container()
    scene.sync().run()
    assert (scene.root / "c" / "a" / "2.eqx").exists()
    assert not (scene.root / "c" / "a" / "state-3.npz").exists()
    assert scene.store.sizes("files/.cache/turbozero/c/a/state-3.npz")


def test_a_following_job_ends_when_nothing_more_comes(scene):
    scene.job(step("a", "--epochs 2"), name="p", follow_minutes=0.01)
    assert scene.runner("p").run() == "done"


def test_a_job_of_segments_keeps_only_the_newest_states_on_its_disk(scene):
    c = f"{scene.home}/.cache/turbozero/c"
    scene.job(
        step("s0", "--epochs 2 --save-state-at 2"),
        step("s1", f"--epochs 2 --save-state-at 2 --branch-from {c}/s0/state-2.npz"),
        step("s2", f"--epochs 2 --save-state-at 2 --branch-from {c}/s1/state-2.npz"),
        name="p",
        states_in_bucket=True,
    )
    assert scene.runner("p").run() == "done"
    # s2 needed only s1's: s0's left the disk, and the bucket keeps every one
    assert not (scene.root / "c" / "s0" / "state-2.npz").exists()
    assert (scene.root / "c" / "s1" / "state-2.npz").exists()
    for name in ("s0", "s1", "s2"):
        assert scene.store.sizes(f"files/.cache/turbozero/c/{name}/state-2.npz")


def test_extending_a_job_refuses_a_step_it_already_has_or_a_taken_place(scene):
    scene.job(step("a"), name="p")
    home = str(scene.home)
    with pytest.raises(SystemExit, match="already a step's"):
        modal_jobs.extend_job(
            scene.store, "p", [modal_spec.parse_step(step("a"), home)]
        )
    taken = scene.root / "c" / "b"
    taken.mkdir(parents=True)
    (taken / "x").write_text("a local run's")
    with pytest.raises(SystemExit, match="already has files"):
        modal_jobs.extend_job(
            scene.store, "p", [modal_spec.parse_step(step("b"), home)]
        )
    job = modal_jobs.extend_job(
        scene.store, "p", [modal_spec.parse_step(step("c"), home)], hours=3
    )
    assert [Path(s.ckpt_dir).name for s in job.steps] == ["a", "c"] and job.hours == 3


def test_a_job_stopped_while_it_waits_for_more_steps_ends_as_stopped(scene):
    scene.job(step("a", "--epochs 2"), name="p", follow_minutes=0.2)
    done = scene.root / "c" / "a" / "1.eqx"

    def stop_once_done():
        while not done.exists():
            threading.Event().wait(0.05)
        threading.Event().wait(0.3)
        modal_store.put_json(
            scene.store, modal_spec.job_key("p", "stop.json"), {"stop": True}
        )

    thread = threading.Thread(target=stop_once_done)
    thread.start()
    assert scene.runner("p").run() == "stopped"
    thread.join()


def test_a_step_whose_start_was_never_sent_runs_the_step_that_makes_it_again(scene):
    c = f"{scene.home}/.cache/turbozero/c"
    scene.job(
        step("a", "--epochs 3 --save-state-at 3"),
        step("b", f"--epochs 4 --die-at 1 --branch-from {c}/a/state-3.npz"),
        name="p",
        resume_every=10,
    )
    with pytest.raises(RuntimeError, match="lost"):
        scene.runner("p").run()
    # the machine was lost before a's final state reached the bucket
    (scene.tmp / "bucket" / "files/.cache/turbozero/c/a/state-3.npz").unlink()
    died = (scene.root / "c" / "b.died").read_text()
    scene.wipe_container()
    (scene.root / "c").mkdir(parents=True)
    (scene.root / "c" / "b.died").write_text(died)

    assert scene.runner("p").run() == "done"
    assert "attempt 2, from the start" in (scene.root / "l" / "a.log").read_text()
    assert scene.store.sizes("files/.cache/turbozero/c/a/state-3.npz")
