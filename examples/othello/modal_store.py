"""The bucket Modal jobs write to and this machine reads from (see modal_jobs.py). An object
appears whole or not at all, so what a dead job wrote is always usable. Nothing here deletes."""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Protocol


class Store(Protocol):
    def put_file(self, key: str, path: str) -> None: ...
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str, start: int = 0) -> bytes | None: ...
    def get_file(self, key: str, path: str) -> bool: ...
    def sizes(self, prefix: str) -> dict[str, int]: ...


def put_json(store: Store, key: str, value: Any) -> None:
    store.put_bytes(key, json.dumps(value).encode())


def get_json(store: Store, key: str) -> Any:
    data = store.get_bytes(key)
    return None if data is None else json.loads(data)


def _download_aside(download: Any, path: str) -> None:
    """Runs `download(tmp)`, then moves it to `path`: a reader never sees part of a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    download(tmp)
    os.replace(tmp, path)


class GcsStore:
    """A Google Cloud Storage bucket, with a service account's key as JSON, or with no key the
    machine's own account (a Compute Engine VM's)."""

    def __init__(self, bucket: str, key_json: str | None):
        from google.cloud import storage

        if key_json is None:
            self.client = storage.Client()
        else:
            info = json.loads(key_json)
            self.client = storage.Client.from_service_account_info(info)
        self.bucket = self.client.bucket(bucket)

    def put_file(self, key: str, path: str) -> None:
        self.bucket.blob(key, chunk_size=64 << 20).upload_from_filename(path)

    def put_bytes(self, key: str, data: bytes) -> None:
        self.bucket.blob(key).upload_from_string(data)

    def get_bytes(self, key: str, start: int = 0) -> bytes | None:
        from google.api_core.exceptions import NotFound, RequestRangeNotSatisfiable

        try:
            return self.bucket.blob(key).download_as_bytes(start=start or None)
        except NotFound:
            return None
        except RequestRangeNotSatisfiable:
            return b""

    def get_file(self, key: str, path: str) -> bool:
        from google.api_core.exceptions import NotFound

        try:
            _download_aside(self.bucket.blob(key).download_to_filename, path)
        except NotFound:
            Path(path + ".part").unlink(missing_ok=True)
            return False
        return True

    def sizes(self, prefix: str) -> dict[str, int]:
        return {
            blob.name: blob.size
            for blob in self.client.list_blobs(self.bucket, prefix=prefix)
        }


class DirStore:
    """A directory as a bucket, for tests."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put_file(self, key: str, path: str) -> None:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, str(target) + ".up")
        os.replace(str(target) + ".up", target)

    def put_bytes(self, key: str, data: bytes) -> None:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        Path(str(target) + ".up").write_bytes(data)
        os.replace(str(target) + ".up", target)

    def get_bytes(self, key: str, start: int = 0) -> bytes | None:
        target = self.root / key
        return target.read_bytes()[start:] if target.is_file() else None

    def get_file(self, key: str, path: str) -> bool:
        target = self.root / key
        if not target.is_file():
            return False
        _download_aside(lambda tmp: shutil.copyfile(target, tmp), path)
        return True

    def sizes(self, prefix: str) -> dict[str, int]:
        if not self.root.exists():
            return {}
        files = (p for p in self.root.rglob("*") if p.is_file())
        keys = {str(p.relative_to(self.root)): p for p in files}
        return {
            key: path.stat().st_size
            for key, path in keys.items()
            if key.startswith(prefix) and not key.endswith(".up")
        }
