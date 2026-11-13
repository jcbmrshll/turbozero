"""Builds a private copy of GGS, Michael Buro's Generic Game Server, for testing the client.

Downloads the server's source (pinned by hash) into a new directory under
~/.cache/turbozero/ggs (or $GGS_DIR), and compiles Central and the Othello service as
native 64-bit binaries, the way the source's own tests/security_options_build.py does:
its makefiles build 32-bit, which needs multilib. Needs g++, and gdbm's header (from
libgdbm-dev, or `--gdbm-include DIR`); bzip2 and zlib come with most systems.

    python -I examples/othello/ggs/setup_ggs.py
    python -I examples/othello/ggs/setup_ggs.py --gdbm-include /tmp/gdbm/usr/include

The binaries land in <dir>/bin. local_server.py runs them, bound to loopback.
"""

import argparse
import concurrent.futures
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://skatgame.net/mburo/ggs/GGS-Nov-09-2026.tar.gz"
SHA256 = "66b5170f88df39cf68e61985c300b1e637a7c2752b07f98da00cfe6757346bb4"
DEFAULT_DIR = Path(os.environ.get("GGS_DIR") or Path.home() / ".cache/turbozero/ggs")

# (component, preprocessor define); GameLib's objects are linked into the Othello service
COMPONENTS = [
    ("Server/Central", "SERVER"),
    ("Service/GameLib", "SERVICE"),
    ("Service/Othello", "SERVICE"),
]
LIBS = ["-Wl,-l:libgdbm.so.6,-l:libbz2.so.1,-l:libz.so.1"]


def fetch(dest: Path) -> Path:
    """The source tree, downloaded and checked against SHA256, unpacked in dest/src."""
    src = dest / "src"
    if src.is_dir():
        return src
    dest.mkdir(parents=True, exist_ok=True)
    tarball = dest / "ggs.tar.gz"
    if not tarball.exists():
        with urllib.request.urlopen(URL, timeout=60) as r, open(tarball, "wb") as f:
            shutil.copyfileobj(r, f)
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(
            f"setup_ggs.py: {tarball} has sha256 {digest}, expected {SHA256}"
        )
    staging = Path(tempfile.mkdtemp(dir=dest))
    with tarfile.open(tarball) as t:
        t.extractall(staging, filter="data")
    staging.rename(src)
    link_shared_sources(src)
    return src


def link_shared_sources(src: Path) -> None:
    """What the source's bin/apply-links does, without running it: Multiplexor and
    GameLib share many of Central's files through symlinks."""
    for line in (src / "bin/apply-links").read_text().splitlines():
        parts = line.replace("(", " ").replace(")", " ").replace(";", " ").split()
        # ( cd ./Service/GameLib/src ; link ../../../Server/Central/src/IO.C IO.C )
        if len(parts) == 5 and parts[0] == "cd" and parts[2] == "link":
            where, target, name = src / parts[1], parts[3], parts[4]
            if not where.resolve().is_relative_to(src.resolve()) or "/" in name:
                raise SystemExit(f"setup_ggs.py: unexpected link line: {line}")
            path = where / name
            if path.is_symlink():
                path.unlink()
            if not path.exists():
                path.symlink_to(target)


def build(src: Path, out: Path, gdbm_include: Path | None, jobs: int) -> None:
    includes = [f"-I{src / 'Service/GameLib/src'}"]
    if gdbm_include:
        includes.append(f"-I{gdbm_include}")
    for component, define in COMPONENTS:
        csrc = src / component / "src"
        objdir = out / "obj" / Path(component).name
        objdir.mkdir(parents=True, exist_ok=True)

        def compile_one(p: Path, csrc=csrc, objdir=objdir, define=define):
            obj = objdir / (p.stem + ".o")
            if obj.exists() and obj.stat().st_mtime >= p.stat().st_mtime:
                return p.name, 0, ""
            cmd = [
                "g++",
                "-std=c++11",
                "-O1",
                "-g",
                "-w",
                "-DLOG=2",
                "-DLOG_ERR",
                f"-D{define}",
                f"-I{csrc}",
                *includes,
                "-c",
                str(p),
                "-o",
                str(obj),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return p.name, r.returncode, r.stderr

        sources = sorted(csrc.glob("*.C"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(compile_one, sources))
        errors = [f"{n}: {e}" for n, code, e in results if code]
        if errors:
            raise SystemExit("setup_ggs.py: compile failed\n" + "\n".join(errors))
        name = Path(component).name
        if name == "GameLib":
            continue
        objects = list(objdir.glob("*.o"))
        if name == "Othello":
            objects += list((out / "obj/GameLib").glob("*.o"))
        (out / "bin").mkdir(exist_ok=True)
        subprocess.run(
            ["g++", *map(str, objects), *LIBS, "-o", str(out / "bin" / name)],
            check=True,
        )
        print(f"built {out / 'bin' / name}", flush=True)
    # the Othello service's help files, which it reads from its directory at run time
    shutil.copytree(
        src / "Service/Othello/msg", out / "msg/Othello", dirs_exist_ok=True
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--gdbm-include", type=Path, help="directory holding gdbm.h")
    ap.add_argument("--jobs", type=int, default=4, help="compiles at once")
    args = ap.parse_args()
    if shutil.which("g++") is None:
        raise SystemExit("setup_ggs.py: needs g++")
    src = fetch(args.dir)
    build(src, args.dir, args.gdbm_include, args.jobs)
    print(f"GGS is in {args.dir} (binaries in bin/)")


if __name__ == "__main__":
    main()
