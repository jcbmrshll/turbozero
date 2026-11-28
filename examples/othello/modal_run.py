"""Runs a Modal job in its container (see modal_jobs.py): each step's train.py in turn, a local
seaglass server for it to log to, and a thread copying what they write to the bucket.

A job continues where it was if it runs again, as Modal does after a preemption: finished steps
are skipped, and a step goes on from its newest saved state with train.py's --resume. Standard
library only, beside the bucket's client: this runs outside the pinned commit's environment.
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modal_spec import (
    STATE as STATE_PATTERN,
)
from modal_spec import (
    Job,
    Step,
    flag,
    job_key,
    start_states,
    state_epochs_requested,
    with_monitor,
    without_flag,
)
from modal_store import GcsStore, Store, get_json, put_json

STATE = re.compile(STATE_PATTERN)
SEAGLASS_PORT = 8008
# how long a step may wait for what's uploading to finish, once it's over or interrupted
FLUSH_RETRIES = 5
MIRROR_SECONDS = 30.0
POLL_SECONDS = 2.0


@dataclass(frozen=True)
class Place:
    """Where the job runs: paths and programs of the container, in tests a temporary directory."""

    repo: str = "/src"
    python: str = "/src/.venv/bin/python"
    seaglass_root: str = "/tmp/seaglass"
    seaglass_port: int = SEAGLASS_PORT
    mirror_seconds: float = MIRROR_SECONDS
    poll_seconds: float = POLL_SECONDS


def gpu_description() -> dict[str, str]:
    """The chip the job got, and its driver."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    name, driver, memory = (out.splitlines()[0].split(", ") + ["", "", ""])[:3]
    return {"chip": name, "driver": driver, "gpu_memory": memory}


class Mirror:
    """Copies files to the bucket when they change: a step's checkpoint directory and log under
    their own keys, the seaglass runs and the newest state to continue from under the job's.
    The manifest lists what's live, as a local run's directories would hold it, for sync.

    Saved states (GBs each with PCR trees, minutes to send) go on a lane of their own, so the
    rest (logs, checkpoints, seaglass, the manifest) keeps arriving while one is on its way.
    """

    def __init__(self, store: Store, job: Job, name: str, place: Place):
        self.store, self.job, self.name, self.place = store, job, name, place
        # bucket key -> (size, mtime) of the file here that's in the bucket under it
        self.sent: dict[str, tuple[int, int]] = {}
        # those brought here from the bucket: their manifest entries stay as they were
        self.restored: dict[str, tuple[int, int]] = {}
        # bucket key -> [size, mtime]: what's live and already in the bucket
        self.manifest: dict[str, list[int]] = (
            get_json(store, job_key(name, "manifest.json")) or {}
        )
        self.lock = threading.Lock()  # one tick of the rest at a time
        self.states_lock = threading.Lock()  # one state upload at a time
        self.manifest_lock = threading.Lock()
        self.manifest_sent: dict[str, list[int]] | None = None
        # the steps that ran on this container: only their directories here are as train.py
        # left them (a step finished on an earlier attempt's container isn't on this disk)
        self.ran: set[int] = set()

    def fetch(self, key: str, path: str) -> bool:
        """Brings `key` to `path`, as get_file does, noting that the bucket has it: a new
        attempt doesn't send back what it restored."""
        if not self.store.get_file(key, path):
            return False
        stat = os.stat(path)
        self.sent[key] = self.restored[key] = (stat.st_size, stat.st_mtime_ns)
        return True

    def _send(self, key: str, path: str, sig: tuple[int, int]) -> bool:
        """Sends `path` if it changed; whether it now is in the bucket as `sig` says."""
        if self.sent.get(key) == sig:
            return True
        try:
            self._put(key, path)
        except FileNotFoundError:
            return False  # a checkpoint replaced as it was being sent
        self.sent[key] = sig
        return True

    def _put(self, key: str, path: str) -> None:
        # a log and the seaglass files grow as they're read: send a copy of one moment
        if os.path.getsize(path) < 32 << 20:
            self.store.put_bytes(key, Path(path).read_bytes())
        else:
            self.store.put_file(key, path)

    def _implicit_state(self, step: Step, epoch: int) -> bool:
        """Whether a state is only there to continue from: not what the step asked to keep."""
        return flag(
            step.argv, "--save-state-every"
        ) is None and epoch not in state_epochs_requested(step.argv)

    def _files(self, step: Step) -> list[tuple[str, str, bool]]:
        """A step's (path, key, whether it's a state only to continue from), smallest first: the
        files in its --ckpt-dir and the directories in it (a growth run's networks' checkpoints)."""
        out = []
        for path in Path(step.ckpt_dir).rglob("*"):
            if path.name.endswith((".tmp", ".part")) or not path.is_file():
                continue
            match = STATE.fullmatch(path.name)
            implicit = bool(match) and self._implicit_state(step, int(match.group(1)))
            out.append((str(path), self.job.key_of(str(path)), implicit))
        if os.path.exists(step.log):
            out.append((step.log, self.job.key_of(step.log), False))
        return sorted(out, key=lambda file: os.path.getsize(file[0]))

    def _send_seaglass(self) -> None:
        root = Path(self.place.seaglass_root)
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.name.endswith(".tmp") or path.name.startswith(".upload-"):
                continue
            stat = path.stat()
            key = job_key(self.name, "seaglass", str(path.relative_to(root)))
            self._send(key, str(path), (stat.st_size, stat.st_mtime_ns))

    def _send_file(self, index: int, path: str, key: str, implicit: bool) -> None:
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            return  # rolled off since it was listed
        sig = (stat.st_size, stat.st_mtime_ns)
        if implicit:
            self._send_resume_state(index, path, sig)
        elif self.restored.get(key) == sig:
            # as the bucket has it: a manifest entry of it stays as it was, or the sync would
            # bring the same file again for its new mtime
            with self.manifest_lock:
                self.manifest.setdefault(key, [stat.st_size, stat.st_mtime_ns])
        elif self._send(key, path, sig):
            with self.manifest_lock:
                self.manifest[key] = [stat.st_size, stat.st_mtime_ns]

    def _send_step(self, index: int, step: Step) -> None:
        """A step's files but its saved states, and the manifest's entries of those it deleted
        (if it ran here: what isn't on this disk of a step that didn't, it didn't delete)."""
        files = self._files(step)
        for path, key, implicit in files:
            if not STATE.fullmatch(os.path.basename(path)):
                self._send_file(index, path, key, implicit)
        if index not in self.ran:
            return
        # what train.py deleted is gone from a local run too; a saved state it asked for stays,
        # sent on and needed on no disk here after a restart
        live = {key for _, key, _ in files}
        prefix = self.job.key_of(step.ckpt_dir) + "/"
        with self.manifest_lock:
            for key in list(self.manifest):
                name = key.rsplit("/", 1)[-1]
                if (
                    key.startswith(prefix)
                    and key not in live
                    and not STATE.fullmatch(name)
                ):
                    del self.manifest[key]

    def _send_resume_state(self, index: int, path: str, sig: tuple[int, int]) -> None:
        key = job_key(self.name, "resume", str(index), "state.npz")
        if self.sent.get(key) == sig:
            return
        epoch = int(STATE.fullmatch(os.path.basename(path)).group(1))  # pyright: ignore[reportOptionalMemberAccess]
        known = get_json(self.store, key + ".json")
        if known and known["epoch"] > epoch:
            return
        self._put(key, path)
        self.sent[key] = sig
        put_json(self.store, key + ".json", {"epoch": epoch})

    def _put_manifest(self) -> None:
        with self.manifest_lock:
            if self.manifest != self.manifest_sent:
                put_json(self.store, job_key(self.name, "manifest.json"), self.manifest)
                self.manifest_sent = dict(self.manifest)

    def _steps(self, step: tuple[int, Step] | None) -> list[tuple[int, Step]]:
        """The running step's and earlier ones' (a step's last files go on uploading while the
        next one runs)."""
        return [] if step is None else list(enumerate(self.job.steps[: step[0] + 1]))

    def tick_rest(self, step: tuple[int, Step] | None) -> None:
        """Sends what changed but saved states, then the manifest. One at a time, so a later
        tick picks up what this one missed."""
        with self.lock:
            self._send_seaglass()
            for index, earlier in self._steps(step):
                self._send_step(index, earlier)
            self._put_manifest()

    def tick_states(self, step: tuple[int, Step] | None) -> None:
        """Sends the saved states not yet sent, oldest step's and smallest first, the manifest
        after each."""
        with self.states_lock:
            for index, earlier in self._steps(step):
                for path, key, implicit in self._files(earlier):
                    if STATE.fullmatch(os.path.basename(path)):
                        self._send_file(index, path, key, implicit)
                        self._put_manifest()

    def tick(self, step: tuple[int, Step] | None) -> None:
        """Sends all that changed."""
        self.tick_rest(step)
        self.tick_states(step)


def restore_seaglass(mirror: Mirror, root: str) -> None:
    """The job's seaglass runs as they are in the bucket, for a run to go on with."""
    prefix = job_key(mirror.name, "seaglass") + "/"
    for key in mirror.store.sizes(prefix):
        path = os.path.join(root, key.removeprefix(prefix))
        mirror.fetch(key, path)
    # a file sent mid-line ends in part of a row: what's next would be appended to it
    for path in Path(root).rglob("*.jsonl"):
        data = path.read_bytes()
        path.write_bytes(data[: data.rfind(b"\n") + 1])


def restore_step(mirror: Mirror, index: int, step: Step) -> bool:
    """A step's files as the bucket has them, its newest state among them. Whether there's a
    state to continue from: otherwise the step starts over, on what it sent before."""
    store, job, name = mirror.store, mirror.job, mirror.name
    prefix = job.key_of(step.ckpt_dir) + "/"
    states: dict[int, str] = {}
    for key in store.sizes(prefix):
        filename = key.removeprefix(prefix)
        match = STATE.fullmatch(filename)
        if match:
            states[int(match.group(1))] = key
    marker = get_json(store, job_key(name, "resume", str(index), "state.npz.json"))
    implicit_key = job_key(name, "resume", str(index), "state.npz")
    if marker:
        states[marker["epoch"]] = implicit_key
    if not states:
        return False
    epoch = max(states)
    mirror.fetch(states[epoch], os.path.join(step.ckpt_dir, f"state-{epoch}.npz"))
    manifest = get_json(store, job_key(name, "manifest.json")) or {}
    for key in manifest:
        filename = key.removeprefix(prefix)
        if key.startswith(prefix) and not STATE.fullmatch(filename):
            mirror.fetch(key, os.path.join(step.ckpt_dir, filename))
    mirror.fetch(job.key_of(step.log), step.log)
    return True


def ensure_input(mirror: Mirror, path: str | None) -> None:
    """A file the step reads, from the bucket if an earlier attempt or job made it."""
    if path is None or os.path.exists(path):
        return
    if not mirror.fetch(mirror.job.key_of(path), path):
        raise FileNotFoundError(f"{path} isn't here or in the bucket")


def engine_set_path(home: str) -> str:
    """train.py's default engine-labelled set."""
    return os.path.join(home, ".cache/turbozero/othello/engine-sets/nr26-evals-29k.npz")


class Runner:
    """One attempt at a job."""

    def __init__(
        self,
        store: Store,
        name: str,
        place: Place,
        describe_gpu: Callable[[], dict[str, str]] = gpu_description,
        status_store: Store | None = None,
    ):
        self.store, self.name, self.place = store, name, place
        # the heartbeat's own client: it writes while the mirror uploads
        self.status_store = status_store or store
        job = Job.from_dict(get_json(store, job_key(name, "job.json")))
        self.job = job
        self.status: dict[str, Any] = get_json(store, job_key(name, "status.json")) or {
            "state": "queued",
            "attempts": [],
            "steps": {},
        }
        self.status["state"] = "running"
        self.status["attempts"].append(
            {"start": time.time(), "seconds": 0.0, **describe_gpu()}
        )
        self.step_now: tuple[int, Step] | None = None
        self.mirror = Mirror(store, job, name, place)
        self.stop_thread = threading.Event()

    def _spent(self) -> float:
        return sum(attempt["seconds"] for attempt in self.status["attempts"])

    def _heartbeat(self) -> None:
        attempt = self.status["attempts"][-1]
        attempt["seconds"] = time.time() - attempt["start"]
        self.status["heartbeat"] = time.time()
        put_json(
            self.status_store,
            job_key(self.name, "status.json"),
            json.loads(json.dumps(self.status)),
        )

    def _mirror_loop(self) -> None:
        while not self.stop_thread.wait(self.place.mirror_seconds):
            self._safe_tick(states=False)

    def _states_loop(self) -> None:
        """Saved states on their own: one takes minutes to send."""
        while not self.stop_thread.wait(self.place.mirror_seconds):
            try:
                self.mirror.tick_states(self.step_now)
            except Exception as e:  # noqa: BLE001
                print(f"modal-jobs: state upload failed, will retry: {e!r}", flush=True)

    def _heartbeat_loop(self) -> None:
        """The heartbeat on its own: an upload of a big state mustn't make the job look silent."""
        while not self.stop_thread.wait(self.place.mirror_seconds):
            try:
                self._heartbeat()
            except Exception as e:  # noqa: BLE001
                print(f"modal-jobs: heartbeat failed, will retry: {e!r}", flush=True)

    def _safe_tick(self, states: bool = True) -> bool:
        """A trouble with the bucket mustn't stop training: the next tick tries again. Without
        `states`, all but the saved states, which may be uploading for minutes on their own
        thread."""
        try:
            if states:
                self.mirror.tick(self.step_now)
            else:
                self.mirror.tick_rest(self.step_now)
            self._heartbeat()
            return True
        except Exception as e:  # noqa: BLE001
            print(f"modal-jobs: upload failed, will retry: {e!r}", flush=True)
            return False

    def _flush(self) -> None:
        for attempt in range(FLUSH_RETRIES):
            if self._safe_tick():
                return
            time.sleep(2**attempt)
        print("modal-jobs: giving up on the last upload", flush=True)

    def _start_seaglass(self) -> subprocess.Popen[bytes]:
        os.makedirs(self.place.seaglass_root, exist_ok=True)
        restore_seaglass(self.mirror, self.place.seaglass_root)
        server = subprocess.Popen(
            [
                self.place.python,
                "-m",
                "seaglass",
                "serve",
                "--dir",
                self.place.seaglass_root,
                "--port",
                str(self.place.seaglass_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"{self.seaglass_url}/api/runs"
        for _ in range(100):
            try:
                urllib.request.urlopen(url, timeout=2).close()
                return server
            except (urllib.error.URLError, OSError):
                time.sleep(0.2)
        raise RuntimeError("the local seaglass server didn't start")

    @property
    def seaglass_url(self) -> str:
        return f"http://127.0.0.1:{self.place.seaglass_port}"

    def _runs(self) -> set[str]:
        root = Path(self.place.seaglass_root)
        return {path.parent.name for path in root.glob("*/meta.json")}

    def _crash_runs(self, runs: list[str]) -> None:
        """A step that starts over leaves its first run's page behind, as crashed."""
        for run in runs:
            request = urllib.request.Request(
                f"{self.seaglass_url}/api/runs/{run}/finish",
                data=json.dumps({"status": "crashed"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(request, timeout=10).close()
            except (urllib.error.URLError, OSError) as e:
                print(f"modal-jobs: can't close run {run}: {e!r}", flush=True)

    def _command(self, step: Step, resume: bool) -> list[str]:
        argv = with_monitor(step.argv, self.seaglass_url)
        if resume:
            # how the run started: its state has all that now
            argv = without_flag(argv, "--init-from", takes_value=True)
            argv = without_flag(argv, "--init-without-buffers", takes_value=False)
            argv = without_flag(argv, "--fork-into", takes_value=True)
            argv = without_flag(argv, "--warm-start", takes_value=True)
            argv = without_flag(argv, "--branch-from", takes_value=True)
            argv = without_flag(argv, "--take-over", takes_value=False)
            argv = [*argv, "--resume"]
        if flag(step.argv, "--save-state-every") is None:
            argv = [*argv, "--save-state-every", str(self.job.resume_every)]
        return [self.place.python, "examples/othello/train.py", *argv]

    def _write_record(self, index: int, step: Step, **more: Any) -> None:
        """What a comparison with a local run needs, beside the step's checkpoints."""
        record = {
            "job": self.name,
            "step": index,
            "commit": self.job.commit,
            "gpu": self.job.gpu,
            "flags": step.argv,
            "attempts": self.status["attempts"],
            **more,
        }
        Path(step.ckpt_dir, "modal-job.json").write_text(json.dumps(record, indent=1))

    def _run_step(self, index: int, step: Step) -> str:
        """Runs a step to its end; "done", "failed", "stopped", "timeout" or "lost"."""
        job, done = self.job, self.status["steps"].setdefault(str(index), {})
        if done.get("seconds") is not None:
            return "done"
        os.makedirs(step.ckpt_dir, exist_ok=True)
        os.makedirs(os.path.dirname(step.log), exist_ok=True)
        ensure_input(self.mirror, flag(step.argv, "--engine-set"))
        for _, path in start_states(step.argv):
            ensure_input(self.mirror, path)
        resume = restore_step(self.mirror, index, step)
        self.mirror.ran.add(index)
        if not resume:
            self._crash_runs(done.get("runs", []))
            done["runs"] = []
        before = self._runs() - set(done.get("runs", []))
        env = {
            **os.environ,
            "HOME": job.home,
            "PYTHONUNBUFFERED": "1",
            "SEAGLASS_TAGS": job.tags,
            "SEAGLASS_URL": self.seaglass_url,
            # the image's jax is the CUDA build, whose plugin fails to start without a GPU
            **({"JAX_PLATFORMS": "cpu"} if job.gpu == "CPU" else {}),
        }
        self.step_now = (index, step)
        self._write_record(index, step)
        started = time.time()
        with open(step.log, "ab") as log:
            log.write(
                f"modal-jobs: step {index} of {self.name}, attempt "
                f"{len(self.status['attempts'])}, {'resuming' if resume else 'from the start'}\n".encode()
            )
            log.flush()
            child = subprocess.Popen(
                self._command(step, resume),
                cwd=self.place.repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            outcome = self._wait(child)
        new = sorted(self._runs() - before)
        done["runs"] = sorted({*done.get("runs", []), *new})
        if outcome == "done":
            done["seconds"] = done.get("seconds_before", 0.0) + time.time() - started
        else:
            done["seconds_before"] = (
                done.get("seconds_before", 0.0) + time.time() - started
            )
        done["outcome"] = outcome
        self._write_record(index, step, seconds=done.get("seconds"), outcome=outcome)
        if outcome != "done":
            self._flush()
        # else the next step starts now, and the mirror goes on sending this one's last files
        # (its final state can take minutes); the job's end flushes all
        return outcome

    def _wait(self, child: subprocess.Popen[bytes]) -> str:
        """Waits for train.py. A stop or the time limit interrupts it, which it reports to its
        run as stopped."""
        limit = self.job.hours * 3600
        last_check = 0.0
        try:
            while True:
                code = child.poll()
                if code is not None:
                    # killed by a signal: the machine's trouble, so Modal should try again
                    return "done" if code == 0 else "lost" if code < 0 else "failed"
                now = time.time()
                self.status["attempts"][-1]["seconds"] = (
                    now - self.status["attempts"][-1]["start"]
                )
                if self._spent() > limit:
                    self._interrupt(child)
                    return "timeout"
                if now - last_check > self.place.mirror_seconds:
                    last_check = now
                    if self._stop_asked():
                        self._interrupt(child)
                        return "stopped"
                time.sleep(self.place.poll_seconds)
        except BaseException:
            # interrupted, probably preempted: leave the run as it is, to go on next attempt
            os.killpg(child.pid, signal.SIGKILL)
            raise

    def _reload(self) -> Job:
        """The job as the bucket has it now: steps appended since, a new time limit."""
        data = get_json(self.store, job_key(self.name, "job.json"))
        if data is not None:
            self.job = self.mirror.job = Job.from_dict(data)
        return self.job

    def _more_steps(self, index: int) -> bool:
        """After the last step: whether one more comes within `follow_minutes`, before a stop
        or the time limit."""
        deadline = time.time() + self.job.follow_minutes * 60
        while time.time() < deadline and self._spent() < self.job.hours * 3600:
            if self._stop_asked():
                return False
            if index < len(self._reload().steps):
                return True
            self._safe_tick(states=False)
            time.sleep(self.place.mirror_seconds)
        return False

    def _prune_states(self, index: int) -> None:
        """With `states_in_bucket`, the saved states of steps before the last one leave this
        disk (the bucket has those asked for): a long run of segments would fill it."""
        if not self.job.states_in_bucket:
            return
        for step in self.job.steps[: max(index - 1, 0)]:
            for path in Path(step.ckpt_dir).glob("state-*.npz"):
                stat = path.stat()
                key = self.job.key_of(str(path))
                if self.mirror.sent.get(key) == (stat.st_size, stat.st_mtime_ns):
                    path.unlink()

    def _unmade_input(self, index: int) -> int | None:
        """An earlier step of this job whose final state step `index` starts from, when that
        state is neither here nor in the bucket (a machine lost before it was sent): that step
        runs again first."""
        step = self.job.steps[index]
        made = {s.ckpt_dir: i for i, s in enumerate(self.job.steps[:index])}
        for _, path in start_states(step.argv):
            source = made.get(os.path.dirname(path))
            if (
                source is not None
                and not os.path.exists(path)
                and not self.store.sizes(self.job.key_of(path))
            ):
                return source
        return None

    def _stop_asked(self) -> bool:
        stop = get_json(self.store, job_key(self.name, "stop.json"))
        return bool(stop and stop["stop"])

    @staticmethod
    def _interrupt(child: subprocess.Popen[bytes]) -> None:
        os.killpg(child.pid, signal.SIGINT)
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)

    def run(self) -> str:
        """The job's outcome: "done" if all its steps are. Raises what interrupted it, and
        RuntimeError if the machine lost a step, for Modal to run it again."""
        job = self.job
        Path(engine_set_path(job.home)).parent.mkdir(parents=True, exist_ok=True)
        ensure_input(self.mirror, engine_set_path(job.home))
        server = self._start_seaglass()
        thread = threading.Thread(target=self._mirror_loop, daemon=True)
        thread.start()
        beat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        beat.start()
        states = threading.Thread(target=self._states_loop, daemon=True)
        states.start()
        outcome, index = "done", 0
        try:
            self._heartbeat()
            # steps may be added while the job runs (see `Job.follow_minutes`)
            while index < len(self._reload().steps) or self._more_steps(index):
                step = self.job.steps[index]
                self.status["step"] = index
                if self._stop_asked():
                    outcome = "stopped"
                    break
                again = self._unmade_input(index)
                if again is not None:
                    print(
                        f"modal-jobs: step {index} starts from step {again}'s final state, "
                        f"which was never sent: step {again} runs again",
                        flush=True,
                    )
                    self.status["steps"].pop(str(again), None)
                    index = again
                    continue
                self._prune_states(index)
                outcome = self._run_step(index, step)
                if outcome != "done":
                    break
                index += 1
            if outcome == "done" and self._stop_asked():
                outcome = "stopped"  # while it waited for more steps
            self.status["state"] = "lost" if outcome == "lost" else outcome
        except BaseException:
            self.status["state"] = "interrupted"
            raise
        finally:
            # the final status is the last word: a heartbeat still on its way, of a status from
            # before the job ended, would otherwise land after it
            self.stop_thread.set()
            thread.join()
            beat.join()
            # the rest as the job ends, before waiting on a state that may take minutes: no
            # heartbeat, whose final status would tell the sync it's all there before it is
            try:
                self.mirror.tick_rest(self.step_now)
            except Exception as e:  # noqa: BLE001
                print(f"modal-jobs: upload failed, will retry: {e!r}", flush=True)
            states.join()
            self._flush()
            server.terminate()
        print(f"modal-jobs: {self.name}: {outcome}", flush=True)
        if outcome == "lost":
            raise RuntimeError(f"step {self.status['step']} lost: try again")
        return outcome


def main(name: str) -> str:
    """The container's entry point: the bucket's in the environment (see modal_jobs.py)."""

    def open_store() -> GcsStore:
        return GcsStore(os.environ["TZ_MODAL_BUCKET"], os.environ["GCS_KEY_JSON"])

    return Runner(open_store(), name, Place(), status_store=open_store()).run()


if __name__ == "__main__":
    print(main(sys.argv[1]))
