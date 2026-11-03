"""Modal jobs, with a directory as the bucket and a stand-in for train.py that writes what it
writes (checkpoints, states, a real seaglass run), so a job's whole trip runs in seconds."""

import json
import socket
import textwrap
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
p.add_argument("--init-from")
p.add_argument("--init-weights")
p.add_argument("--exclude-openings")
p.add_argument("--init-without-buffers", action="store_true")
p.add_argument("--resume", action="store_true")
p.add_argument("--monitor", nargs="?", const="")
p.add_argument("--name")
p.add_argument("--notes")
p.add_argument("--die-at", type=int)
p.add_argument("--fail-at", type=int)
p.add_argument("--hang-at", type=int)
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
if args.init_weights and args.resume:
    sys.exit("--init-weights and --resume")
for path in (args.init_weights, args.exclude_openings):
    if path:
        assert Path(path).exists(), path
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
            "files/.cache/turbozero/othello/engine-sets/edax-l24.npz", b"set"
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
        ("examples/othello/train.py --vs-engine-last 3 --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l", "Edax"),
        (
            "examples/othello/train.py --init-from ~/.cache/turbozero/dir --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l",
            "state-<EPOCH>",
        ),
        ("examples/othello/train.py --init-weights /tmp/w.eqx --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l", "wouldn't come back"),
        ("examples/othello/train.py --seed-buffer ~/.cache/turbozero/corpus --ckpt-dir ~/.cache/turbozero/c > ~/.cache/turbozero/l", "corpus"),
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


def test_a_preempted_step_from_pretrained_weights_resumes_without_them(scene):
    weights = ".cache/turbozero/w/pretrained.eqx"
    openings = ".cache/turbozero/w/heldout.npy"
    for key in (weights, openings):
        scene.store.put_bytes(f"files/{key}", b"x")  # as `launch` uploads them
    job = scene.job(
        step(
            "t",
            f"--epochs 8 --die-at 6 --init-weights ~/{weights} --exclude-openings ~/{openings}",
        ),
        name="p",
        resume_every=5,
    )
    assert modal_jobs.inputs_to_upload(job)[1:] == [
        str(scene.home / weights),
        str(scene.home / openings),
    ]
    with pytest.raises(RuntimeError, match="lost"):
        scene.runner("p").run()
    died = (scene.root / "c" / "t.died").read_text()
    scene.wipe_container()
    (scene.root / "c").mkdir(parents=True)
    (scene.root / "c" / "t.died").write_text(died)
    assert scene.runner("p").run() == "done"
    assert "resuming from 5" in (scene.root / "l" / "t.log").read_text()


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


def test_jobs_in_finds_jobs(scene):
    scene.job(step("a"), name="a")
    scene.job(step("b"), name="b")
    assert modal_sync.jobs_in(scene.store) == ["a", "b"]
