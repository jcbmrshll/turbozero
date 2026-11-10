"""Train on Modal's GPUs, with the results ending up here as if the runs had been local.

    uv run examples/othello/modal_jobs.py launch --name r2-G --gpu L40S --hours 6 --commit 7b52872 \\
        --step 'examples/othello/train.py --epochs 93 ... --monitor --ckpt-dir ~/.cache/turbozero/... > ~/.cache/turbozero/....log' \\
        --step 'examples/othello/train.py --init-from ... > ...'
    uv run examples/othello/modal_jobs.py list
    uv run examples/othello/modal_jobs.py stop r2-G    (or --all)
    uv run examples/othello/modal_jobs.py sync         (once; --watch 60 keeps going)

A job is steps in order, each a train.py command as written here, on one GPU, at one commit that
is on GitHub. It runs without this machine. Its checkpoints, states and logs go to the bucket as
they're written, and `sync` brings them to the same paths here; `launch` keeps a sync running in
tmux session modal-sync. Each --monitor run appears on slick's seaglass while it runs, tagged
to the task that launched it ($SEAGLASS_TAGS). If Modal preempts a job it continues from its
newest state (every --resume-every epochs): see modal_run.py. --hours limits its whole time.

Setup, once, with a Modal login (`uvx modal setup`) and a service account's key for the bucket:

    uv run examples/othello/modal_jobs.py setup --bucket BUCKET --key ~/.config/turbozero/gcs-modal.json

See README.md's "Training on Modal" for what a job costs and which chip to pick.
"""

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from modal_run import engine_set_path
from modal_spec import GPU_PER_SECOND, Job, flag, job_key, parse_step
from modal_store import GcsStore, Store, get_json, put_json
from modal_sync import TERMINAL, Sync, jobs_in

REPO_URL = "https://github.com/jcbmrshll/turbozero"
SECRET = "turbozero-gcs"
SYNC_SESSION = "modal-sync"
CONFIG = Path.home() / ".config" / "turbozero" / "modal-jobs.json"
SYNC_STATE = Path.home() / ".local" / "state" / "turbozero-modal" / "sync.json"
SYNC_LOCK = Path.home() / ".local" / "state" / "turbozero-modal" / "sync.lock"
SYNC_LOG = Path.home() / ".local" / "state" / "turbozero-modal" / "sync.log"
SEAGLASS_ROOT = Path.home() / ".local" / "state" / "slick" / "seaglass"
# Modal's limit on one attempt
MAX_ATTEMPT_HOURS = 24


def open_store() -> tuple[Store, dict[str, str]]:
    if not CONFIG.exists():
        raise SystemExit(f"no {CONFIG}: run `{Path(__file__).name} setup` first")
    config = json.loads(CONFIG.read_text())
    return GcsStore(config["bucket"], Path(config["key"]).read_text()), config


def setup(args: argparse.Namespace) -> None:
    import modal

    key = Path(args.key).expanduser()
    GcsStore(args.bucket, key.read_text()).sizes(
        "jobs/"
    )  # the key can reach the bucket
    modal.Secret.objects.create(
        SECRET,
        {"GCS_KEY_JSON": key.read_text(), "TZ_MODAL_BUCKET": args.bucket},
        allow_existing=True,
    )
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"bucket": args.bucket, "key": str(key)}))
    print(f"bucket {args.bucket} reachable, Modal secret {SECRET} set")


def resolve_commit(commit: str) -> str:
    """The commit's full hash. It has to be on GitHub, where the image gets it."""
    sha = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "branch", "-r", "--contains", sha],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if not remote.strip():
        raise SystemExit(f"{sha} isn't on GitHub: push it first")
    return sha


def inputs_to_upload(job: Job) -> list[str]:
    """Files of this machine a job reads: those no earlier step of it makes. The engine set
    always (train.py's default), and whichever states and sets its flags name."""
    out = [engine_set_path(job.home)]
    made = {step.ckpt_dir for step in job.steps}
    for step in job.steps:
        for name in ("--init-from", "--branch-from", "--engine-set"):
            path = flag(step.argv, name)
            if path is not None and os.path.dirname(path) not in made:
                out.append(path)
    return out


def check_claims(store: Store, job: Job) -> None:
    """A job's files would land on these paths here: they must be free for it."""
    for step in job.steps:
        ckpt = Path(step.ckpt_dir)
        if ckpt.is_dir() and any(ckpt.iterdir()):
            raise SystemExit(
                f"{ckpt} already has files here: a job would mix with them"
            )
        if store.sizes(job.key_of(step.ckpt_dir) + "/"):
            raise SystemExit(f"{ckpt} already has files in the bucket")
        if Path(step.log).exists():
            raise SystemExit(f"{step.log} is already here")


def build_job(args: argparse.Namespace) -> Job:
    if args.gpu not in GPU_PER_SECOND:
        raise SystemExit(f"--gpu is one of {', '.join(GPU_PER_SECOND)}")
    home = str(Path.home())
    steps = []
    for text in args.step:
        try:
            steps.append(parse_step(text, home))
        except ValueError as e:
            raise SystemExit(str(e)) from None
    tags = args.tags or os.environ.get("SEAGLASS_TAGS", "")
    if not tags and any("--monitor" in step.argv for step in steps):
        raise SystemExit(
            "no $SEAGLASS_TAGS: a run needs its task's tag (or pass --tags)"
        )
    return Job(
        name=args.name,
        commit=resolve_commit(args.commit),
        gpu=args.gpu,
        hours=args.hours,
        home=home,
        steps=steps,
        cpu=args.cpu,
        memory_gb=args.memory,
        resume_every=args.resume_every,
        tags=",".join(t for t in (tags, "modal") if t),
        created=time.time(),
        launched_from={"cwd": os.getcwd(), "task": os.environ.get("SLICK_TASK")},
    )


def _entry(name: str) -> str:
    """The job, in its container."""
    from modal_run import main as run_job

    return run_job(name)


def spawn(job: Job, store: Store) -> None:
    """Starts the job's own Modal app, which keeps going without this process."""
    import modal

    here = Path(__file__).parent
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install("uv", "google-cloud-storage")
        .run_commands(
            f"git clone {REPO_URL} /src",
            f"git -C /src checkout {job.commit}",
            "cd /src && uv sync --frozen --extra cuda",
        )
    )
    for module in ("modal_run", "modal_spec", "modal_store"):
        image = image.add_local_file(here / f"{module}.py", f"/root/{module}.py")
    app = modal.App(f"tz-{job.name}")
    function = app.function(
        image=image,
        gpu=None if job.gpu == "CPU" else job.gpu,
        cpu=job.cpu,
        memory=job.memory_gb * 1024,
        secrets=[modal.Secret.from_name(SECRET)],
        # each attempt's limit; the job's own, across attempts, is modal_run's
        timeout=int(min(job.hours + 0.25, MAX_ATTEMPT_HOURS) * 3600),
        retries=modal.Retries(max_retries=2, initial_delay=30.0),
        # by default a finished job's GPU stays up, billed, for a minute in case of more work
        scaledown_window=2,
        serialized=True,
    )(_entry)
    with app.run(detach=True):
        call = function.spawn(job.name)
    put_json(
        store,
        job_key(job.name, "app.json"),
        {"app_id": app.app_id, "call_id": call.object_id},
    )


def sync_running() -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", f"={SYNC_SESSION}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def ensure_sync() -> None:
    if sync_running():
        return
    me = shlex.quote(str(Path(__file__).resolve()))
    SYNC_LOG.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            SYNC_SESSION,
            f"uv run {me} sync --watch 60 2>&1 | tee -a {shlex.quote(str(SYNC_LOG))}",
        ],
        check=True,
    )
    print(f"started the sync, in tmux session {SYNC_SESSION}")


def launch(args: argparse.Namespace) -> None:
    job = build_job(args)
    store, _ = open_store()
    if get_json(store, job_key(job.name, "job.json")) is not None:
        raise SystemExit(
            f"there's already a job {job.name}: `resume` it or use another name"
        )
    check_claims(store, job)
    for path in inputs_to_upload(job):
        if not store.sizes(job.key_of(path)):
            if not os.path.exists(path):
                raise SystemExit(f"{path} isn't here, nor in the bucket")
            print(f"uploading {path}")
            store.put_file(job.key_of(path), path)
    put_json(store, job_key(job.name, "job.json"), job.to_dict())
    spawn(job, store)
    print(
        f"launched {job.name}: {len(job.steps)} steps on {job.gpu} at {job.commit[:8]}, "
        f"at most {job.hours} h, about ${job.dollars_per_hour():.2f}/h "
        f"(at most ${job.dollars_per_hour() * job.hours:.2f})"
    )
    ensure_sync()


def resume(args: argparse.Namespace) -> None:
    """Runs a job again after it ended short, for more time if asked."""
    store, _ = open_store()
    data = get_json(store, job_key(args.name, "job.json"))
    if data is None:
        raise SystemExit(f"no job {args.name}")
    job = Job.from_dict(data)
    if args.hours is not None:
        job = Job.from_dict({**job.to_dict(), "hours": args.hours})
        put_json(store, job_key(job.name, "job.json"), job.to_dict())
    put_json(store, job_key(job.name, "stop.json"), {"stop": False})
    spawn(job, store)
    ensure_sync()
    print(f"resumed {job.name}")


def describe(store: Store, name: str, now: float) -> dict[str, Any]:
    """A job's line in `list`, and what it has cost so far at list prices."""
    job = Job.from_dict(get_json(store, job_key(name, "job.json")))
    status = get_json(store, job_key(name, "status.json")) or {}
    attempts = status.get("attempts", [])
    seconds = sum(attempt["seconds"] for attempt in attempts)
    state = status.get("state", "queued")
    heartbeat = status.get("heartbeat")
    if state == "running" and heartbeat and now - heartbeat > 300:
        state = f"silent {int(now - heartbeat) // 60} min"
    return {
        "job": name,
        "state": state,
        "step": f"{status.get('step', 0) + 1}/{len(job.steps)}",
        "gpu": job.gpu,
        "chip": attempts[-1].get("chip") if attempts else None,
        "attempts": len(attempts),
        "minutes": round(seconds / 60, 1),
        "limit_hours": job.hours,
        "dollars": round(seconds / 3600 * job.dollars_per_hour(), 2),
    }


def billed_dollars(period: str) -> dict[str, float]:
    """What Modal has billed per app in `period`: its report lags the job by a little."""
    out = subprocess.run(
        ["modal", "billing", "report", "--for", period, "--json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    billed: dict[str, float] = {}
    for row in json.loads(out):
        name = row["description"]
        billed[name] = billed.get(name, 0.0) + float(row["cost"])
    return billed


def list_jobs(args: argparse.Namespace) -> None:
    store, _config = open_store()
    now, total = time.time(), 0.0
    billed = billed_dollars(args.billed) if args.billed else {}
    for name in jobs_in(store):
        line = describe(store, name, now)
        total += line["dollars"]
        if args.billed:
            line["billed"] = round(billed.get(f"tz-{name}", 0.0), 2)
        print(json.dumps(line))
    print(f"about ${total:.2f} in all at list prices, without start-up time")
    if args.billed:
        print(
            f"${sum(billed.values()):.2f} billed by Modal for {args.billed}, all apps"
        )


def stop(args: argparse.Namespace) -> None:
    store, _ = open_store()
    names = jobs_in(store) if args.all else args.names
    if not names:
        raise SystemExit("name a job, or --all")
    for name in names:
        status = get_json(store, job_key(name, "status.json")) or {}
        if status.get("state") in TERMINAL:
            continue
        # the runner sees this within a minute, and its run ends as stopped
        put_json(store, job_key(name, "stop.json"), {"stop": True})
        app = get_json(store, job_key(name, "app.json"))
        if args.now and app:
            subprocess.run(
                ["modal", "app", "stop", "--yes", app["app_id"]], check=False
            )
            if not status.get(
                "attempts"
            ):  # it never started, so nothing else will say so
                put_json(store, job_key(name, "status.json"), {"state": "stopped"})
        print(f"stopping {name}")


def sync(args: argparse.Namespace) -> None:
    store, _ = open_store()
    SYNC_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNC_LOCK, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another sync is running") from None
        syncer = Sync(store, str(Path.home()), SEAGLASS_ROOT, SYNC_STATE)
        while True:
            try:
                for line in syncer.run():
                    print(line, flush=True)
            except Exception as e:
                # the bucket or slick is away: all this is still there for the next time
                print(f"sync failed, will retry: {e!r}", file=sys.stderr, flush=True)
                if args.watch is None:
                    raise
            if args.watch is None:
                return
            time.sleep(args.watch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on Modal's GPUs.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("setup", help="point this at the bucket; once")
    p.add_argument("--bucket", required=True)
    p.add_argument(
        "--key", required=True, help="a service account's key, as a JSON file"
    )
    p.set_defaults(fn=setup)

    p = commands.add_parser("launch", help="start a job")
    p.add_argument("--name", required=True, help="the job's name, unique")
    p.add_argument(
        "--gpu", required=True, help=f"Modal's GPU: {', '.join(GPU_PER_SECOND)}"
    )
    p.add_argument(
        "--hours", type=float, required=True, help="the job's limit, over all attempts"
    )
    p.add_argument(
        "--step",
        action="append",
        required=True,
        help="a train.py command, with `> LOG`; repeat for steps to run in order",
    )
    p.add_argument("--commit", default="HEAD", help="what to run, on GitHub (HEAD)")
    p.add_argument("--cpu", type=float, default=Job.cpu, help="physical cores")
    p.add_argument("--memory", type=int, default=Job.memory_gb, help="GiB")
    p.add_argument(
        "--resume-every",
        type=int,
        default=Job.resume_every,
        help="epochs between saved states to continue from",
    )
    p.add_argument("--tags", default=None, help="else $SEAGLASS_TAGS")
    p.set_defaults(fn=launch)

    p = commands.add_parser("resume", help="run a job that ended short again")
    p.add_argument("name")
    p.add_argument("--hours", type=float, default=None, help="a new limit")
    p.set_defaults(fn=resume)

    p = commands.add_parser("list", help="every job, what it has cost")
    p.add_argument(
        "--billed",
        metavar="PERIOD",
        help="also what Modal billed, for its report's --for: today, this month...",
    )
    p.set_defaults(fn=list_jobs)

    p = commands.add_parser("stop", help="stop jobs")
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument(
        "--now",
        action="store_true",
        help="stop the app too, without waiting for the run to end as stopped",
    )
    p.set_defaults(fn=stop)

    p = commands.add_parser("sync", help="bring results here")
    p.add_argument("--watch", type=float, default=None, metavar="SECONDS")
    p.set_defaults(fn=sync)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
