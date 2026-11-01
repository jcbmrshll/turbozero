"""Brings Modal jobs' results here (see modal_jobs.py): their files to the paths a local run
would write, and their seaglass runs into slick's root, as the runs slick already reads.

Slick reads run directories on every request, so a new run is written aside and renamed into
place, and rows are appended in one write. Slick's own edits (notes, conclusions) win over what
the job wrote at its start. Only complete lines are taken from a file the job is still writing.
"""

import base64
import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from modal_spec import job_key, path_of
from modal_store import Store, get_json

SLICK_SEAGLASS = "http://slick/seaglass"
# edited in slick after the fact: keep those edits
ABOUT = ("notes", "conclusions", "tags")
TERMINAL = ("done", "failed", "stopped", "timeout")
APPEND_ONLY = (".log", ".jsonl")


def write_atomic(path: Path, data: bytes) -> None:
    tmp = Path(str(path) + ".sync-tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def jobs_in(store: Store) -> list[str]:
    return sorted(
        key.split("/")[1]
        for key in store.sizes("jobs/")
        if key.count("/") == 2 and key.endswith("/job.json")
    )


class Sync:
    def __init__(
        self,
        store: Store,
        home: str,
        seaglass_root: str | Path,
        state_path: str | Path,
        seaglass_url: str | None = SLICK_SEAGLASS,
    ):
        self.store, self.home = store, home
        self.seaglass_root = Path(seaglass_root)
        self.state_path = Path(state_path)
        self.seaglass_url = seaglass_url
        self.state: dict[str, Any] = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        )

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(self.state_path, json.dumps(self.state).encode())

    def run(self) -> list[str]:
        """Brings what's new of every job not yet done; what it did, a line each."""
        out = []
        for name in jobs_in(self.store):
            record = self.state.setdefault(name, {"files": {}, "runs": {}})
            if record.get("complete"):
                continue
            out += self._job(name, record)
            self._save()
        return out

    def _job(self, name: str, record: dict[str, Any]) -> list[str]:
        # read before anything else: if it says the job is over, what follows is all there is
        status = get_json(self.store, job_key(name, "status.json")) or {}
        out = self._files(name, record)
        out += self._seaglass(name, record, status)
        if status.get("state") in TERMINAL:
            record["complete"] = True
            out.append(f"{name}: {status['state']}, all here")
        return out

    def _files(self, name: str, record: dict[str, Any]) -> list[str]:
        manifest = get_json(self.store, job_key(name, "manifest.json")) or {}
        files, out = record["files"], []
        for key, sig in manifest.items():
            if files.get(key) == sig:
                continue
            path = path_of(key, self.home)
            if key not in files and os.path.exists(path):
                out.append(f"{name}: {path} is already here: left alone")
                files[key] = sig
                continue
            self._pull(key, path, key in files)
            files[key] = sig
            out.append(f"{name}: {path}")
        # train.py deleted it there, as it would here: only ever a file brought here by this
        for key in [key for key in files if key not in manifest]:
            Path(path_of(key, self.home)).unlink(missing_ok=True)
            del files[key]
            out.append(f"{name}: {path_of(key, self.home)} dropped, as the run did")
        return out

    def _pull(self, key: str, path: str, had_it: bool) -> None:
        """A file the job may have added to since the last time: only the new part of it."""
        if had_it and path.endswith(APPEND_ONLY) and os.path.exists(path):
            more = self.store.get_bytes(key, os.path.getsize(path))
            if more:
                with open(path, "ab") as f:
                    f.write(more)
            return
        self.store.get_file(key, path)

    def _seaglass(
        self, name: str, record: dict[str, Any], status: dict[str, Any]
    ) -> list[str]:
        prefix = job_key(name, "seaglass") + "/"
        sizes = {
            key.removeprefix(prefix): size
            for key, size in self.store.sizes(prefix).items()
        }
        out: list[str] = []
        by_run: dict[str, dict[str, int]] = {}
        for rel, size in sizes.items():
            top, _, rest = rel.partition("/")
            if top == "bundles":
                self._bundle(prefix, rest.partition("/")[0], sizes)
            else:
                by_run.setdefault(top, {})[rest] = size
        for run, files in sorted(by_run.items()):
            if "meta.json" in files:
                out += self._run(prefix, name, run, files, record, status)
        return out

    def _bundle(self, prefix: str, bundle: str, sizes: dict[str, int]) -> None:
        """A run's views, over slick's API, which checks them."""
        if self.seaglass_url is None or self._has_bundle(bundle):
            return
        at = f"bundles/{bundle}/"
        files = {
            rel.removeprefix(at): base64.b64encode(
                self.store.get_bytes(prefix + rel) or b""
            ).decode()
            for rel in sizes
            if rel.startswith(at)
        }
        request = urllib.request.Request(
            f"{self.seaglass_url}/api/bundles/{bundle}",
            data=json.dumps({"files": files}).encode(),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        urllib.request.urlopen(request, timeout=60).close()

    def _has_bundle(self, bundle: str) -> bool:
        assert self.seaglass_url is not None
        try:
            url = f"{self.seaglass_url}/api/bundles/{bundle}"
            urllib.request.urlopen(url, timeout=30).close()
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def _run(
        self,
        prefix: str,
        name: str,
        run: str,
        files: dict[str, int],
        record: dict[str, Any],
        status: dict[str, Any],
    ) -> list[str]:
        meta_bytes = self.store.get_bytes(f"{prefix}{run}/meta.json")
        if meta_bytes is None:
            return []
        meta = json.loads(meta_bytes)
        attempts = status.get("attempts") or [{}]
        modal = {
            "job": name,
            "attempts": len(attempts),
            "chip": attempts[-1].get("chip"),
        }
        meta["provenance"] = {**meta.get("provenance", {}), "modal": modal}
        seen = record["runs"].get(run)
        target = self.seaglass_root / run
        if seen is None:
            if target.exists():
                raise RuntimeError(
                    f"{target} exists, and isn't this job's: not touching it"
                )
            # whole before slick can see it: slick lists names that look like run ids
            tmp = self.seaglass_root / f".modal-{run}"
            shutil.rmtree(tmp, ignore_errors=True)  # what an interrupted sync left
            tmp.mkdir(parents=True)
            seen = {"offsets": {}, "files": [], "about": {}}
            self._rows_and_files(prefix, run, files, tmp, seen)
            write_atomic(tmp / "meta.json", json.dumps(meta).encode())
            seen["about"] = {k: meta.get(k) for k in ABOUT}
            os.rename(tmp, target)
            record["runs"][run] = seen
            return [f"{name}: new seaglass run {run} ({meta.get('name')})"]
        self._rows_and_files(prefix, run, files, target, seen)
        current = json.loads((target / "meta.json").read_text())
        for k in ABOUT:
            # the job's own change comes through, unless slick's copy was edited since
            if current.get(k) != seen["about"].get(k):
                meta[k] = current.get(k)
        seen["about"] = {k: meta.get(k) for k in ABOUT}
        if meta != current:
            write_atomic(target / "meta.json", json.dumps(meta).encode())
        return []

    def _rows_and_files(
        self,
        prefix: str,
        run: str,
        files: dict[str, int],
        target: Path,
        seen: dict[str, Any],
    ) -> None:
        for rel, size in sorted(files.items()):
            if rel == "meta.json":
                continue
            path = target / rel
            key = f"{prefix}{run}/{rel}"
            if rel.endswith(".jsonl"):
                offset = seen["offsets"].get(rel, 0)
                if size <= offset:
                    continue
                data = self.store.get_bytes(key, offset) or b""
                data = data[: data.rfind(b"\n") + 1]
                if data:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(
                        path, "ab"
                    ) as f:  # one write: slick indexes what's appended
                        f.write(data)
                    seen["offsets"][rel] = offset + len(data)
            elif rel not in seen["files"]:
                self.store.get_file(key, str(path))
                seen["files"].append(rel)
