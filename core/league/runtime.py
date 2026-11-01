"""What the dashboard runs networks with in the browser: onnxruntime-web, downloaded once into
the store and served from there, so the page needs nothing from outside the tailnet."""

import base64
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

VERSION = "1.31.1"
URL = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz"
# npm's `dist.integrity` of that release
SHA512 = "5GeH+bWQ+HxM10z7erGlsWqryrwMbSvm/1tkIsKTT48g0fI19VXVVC435t4t7VsPdSQeS7NmuZPXrQ73k82URw=="
FILES = (
    "ort.wasm.min.mjs",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
)


def runtime_dir(root: Path) -> Path:
    return root / "ort" / VERSION


def has_runtime(root: Path) -> bool:
    return all((runtime_dir(root) / name).exists() for name in FILES)


def fetch_runtime(root: Path, url: str = URL, sha512: str = SHA512) -> None:
    """Puts the runtime's files under `root`, unless they're there already.

    Raises:
        ValueError: if what was downloaded isn't the release expected
    """
    if has_runtime(root):
        return
    with urllib.request.urlopen(url) as response:
        package = response.read()
    if base64.b64encode(hashlib.sha512(package).digest()).decode() != sha512:
        raise ValueError(f"{url} isn't the expected release")
    out = runtime_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(package)) as tar:
        for name in FILES:
            member = tar.extractfile(f"package/dist/{name}")
            assert member is not None
            partial = out / f"{name}.partial"
            partial.write_bytes(member.read())
            partial.rename(out / name)
