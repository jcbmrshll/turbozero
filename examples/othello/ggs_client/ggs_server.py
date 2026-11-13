"""A private GGS server on this machine, for testing the client: Central and the Othello
service (built by setup_ggs.py), bound to 127.0.0.1 only, in a runtime directory of their own.

    uv run examples/othello/ggs_client/ggs_server.py start --run /tmp/ggs-run --port 15000
    uv run examples/othello/ggs_client/ggs_server.py register --run /tmp/ggs-run tzbot PASSWORD
    uv run examples/othello/ggs_client/ggs_server.py stop --run /tmp/ggs-run

`start` makes the runtime if it's new: the databases, a random root password (kept in
<run>/credentials, mode 600), and the `/os` service's account; then starts Central and the
service, each in a process group of its own, recorded in <run>/pids.json, which `stop` ends.
Logins need no account: a new login takes any password. `register` gives one a name and an
email (as an admin does on the public server), which rated games need; it's online meanwhile.
"""

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

DEFAULT_GGS = Path(os.environ.get("GGS_DIR") or Path.home() / ".cache/turbozero/ggs")
HOST = "127.0.0.1"


class Telnet:
    """A line-at-a-time connection to the server, for its admin and for tests."""

    def __init__(self, port: int, timeout: float = 10.0):
        self.sock = socket.create_connection((HOST, port), timeout=timeout)
        self.buffer = ""
        self.syncs = 0

    def read_until(self, text: str, timeout: float = 10.0) -> str:
        end = time.monotonic() + timeout
        while text not in self.buffer:
            if time.monotonic() > end:
                raise TimeoutError(f"no {text!r} in {self.buffer[-500:]!r}")
            self.sock.settimeout(max(end - time.monotonic(), 0.01))
            data = self.sock.recv(65536)
            if not data:
                raise ConnectionError(f"closed before {text!r}: {self.buffer[-500:]!r}")
            self.buffer += data.decode("latin-1").replace("\r", "")
        found, _, self.buffer = self.buffer.partition(text)
        return found + text

    def send(self, line: str) -> None:
        self.sock.sendall((line + "\n").encode("latin-1"))

    def sync(self, timeout: float = 10.0) -> str:
        """Everything up to now, read: up to an echo of a marker of our own."""
        self.syncs += 1
        marker = f"tz-sync-{self.syncs}"
        self.send(f"echo {marker}")
        out = self.read_until(marker, timeout)
        self.read_until("READY\n", timeout)
        return out

    def command(self, line: str, timeout: float = 10.0) -> str:
        """`line`'s output (and whatever else came meanwhile)."""
        self.send(line)
        out = self.sync(timeout)
        return out.rsplit(f"tz-sync-{self.syncs}", 1)[0]

    def login(self, login: str, password: str) -> None:
        self.read_until("Enter login")
        self.send(login)
        self.read_until("Enter your password.")
        self.send(password)
        out = self.sync()
        if "Password is incorrect" in out or "Enter login" in out:
            raise PermissionError(f"{login}: {out.strip()[-200:]}")
        self.command("verbose -news -faq -help -ack")

    def close(self) -> None:
        try:
            self.send("quit")
        except OSError:
            pass
        self.sock.close()


class LocalGGS:
    """A GGS server's runtime directory, `run`, and its processes."""

    def __init__(self, run: Path, port: int = 15000, ggs: Path = DEFAULT_GGS):
        self.run = Path(run).resolve()
        self.port = port
        self.ggs = Path(ggs)
        self.central_dir = self.run / "Server/Central"
        self.service_dir = self.run / "Service/Othello"
        self.credentials = self.run / "credentials"

    @property
    def root_password(self) -> str:
        return (self.credentials / "root").read_text()

    def binary(self, name: str) -> Path:
        path = self.ggs / "bin" / name
        if not path.exists():
            raise FileNotFoundError(f"{path}: build it with setup_ggs.py")
        return path

    def _env(self) -> dict[str, str]:
        return dict(os.environ, GGS_RUN=str(self.run))

    def _spawn(self, args: list[str], cwd: Path, name: str) -> int:
        """Starts a server process in a process group of its own; returns its pid (= pgid)."""
        (cwd / "log").mkdir(exist_ok=True)
        with open(cwd / "log" / "out", "ab") as out:
            process = subprocess.Popen(
                args,
                cwd=cwd,
                env=self._env(),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pids = self._pids()
        pids[name] = process.pid
        (self.run / "pids.json").write_text(json.dumps(pids))
        return process.pid

    def _pids(self) -> dict[str, int]:
        path = self.run / "pids.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _wait_listening(self, timeout: float = 10.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                socket.create_connection((HOST, self.port), timeout=1).close()
                return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError(f"nothing listening on {HOST}:{self.port}")

    def _stop(self, name: str) -> None:
        pids = self._pids()
        pid = pids.pop(name, None)
        if pid is None:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
            for _ in range(50):
                os.killpg(pid, 0)
                time.sleep(0.1)
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        (self.run / "pids.json").write_text(json.dumps(pids))

    def create(self) -> None:
        """A new runtime: databases, a root password, the service's account."""
        if (self.credentials / "os").exists():
            return
        if self.run.exists() and any(self.run.iterdir()):
            raise FileExistsError(f"{self.run} isn't empty, nor a runtime of ours")
        for component, src in (
            (self.central_dir, self.ggs / "src/Server/Central/msg"),
            (self.service_dir, self.ggs / "src/Service/Othello/msg"),
        ):
            for sub in ("db", "log", "bak"):
                (component / sub).mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, component / "msg", dirs_exist_ok=True)
        self.credentials.mkdir(mode=0o700, parents=True, exist_ok=True)
        for binary, cwd in (
            ("Central", self.central_dir),
            ("Othello", self.service_dir),
        ):
            subprocess.run(
                [str(self.binary(binary)), "-db_init"], cwd=cwd, env=self._env(),
                check=True, capture_output=True,
            )  # fmt: skip
        # setup mode: loopback only, root's password still "root"
        self._spawn(
            [str(self.binary("Central")), str(self.port), "-setup"],
            self.central_dir,
            "central",
        )
        try:
            self._wait_listening()
            admin = Telnet(self.port)
            admin.login("root", "root")
            password = secrets.token_hex(16)
            admin.command(f"passw {password} root")
            out = admin.command("service create /os")
            admin.close()
            service = out.split("password", 1)[1].split()[0]
        finally:
            self._stop("central")
        for name, secret in (("root", password), ("os", service)):
            path = self.credentials / name
            path.write_text(secret)
            path.chmod(0o600)

    def start(self) -> None:
        self.create()
        self._spawn(
            [str(self.binary("Central")), str(self.port), f"-serverbind={HOST}"],
            self.central_dir, "central",
        )  # fmt: skip
        self._wait_listening()
        self._spawn(
            [str(self.binary("Othello")), "-s", HOST, "-p", str(self.port),
             "-pwfile", str(self.credentials / "os")],
            self.service_dir, "othello",
        )  # fmt: skip
        self.wait_service()

    def wait_service(self, timeout: float = 15.0) -> None:
        """Until /os is logged in."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            admin = Telnet(self.port)
            admin.login("root", self.root_password)
            out = admin.command("who")
            admin.close()
            if "/os" in out:
                return
            time.sleep(0.3)
        raise TimeoutError("the Othello service didn't log in")

    def stop_service(self) -> None:
        self._stop("othello")

    def start_service(self) -> None:
        self._spawn(
            [str(self.binary("Othello")), "-s", HOST, "-p", str(self.port),
             "-pwfile", str(self.credentials / "os")],
            self.service_dir, "othello",
        )  # fmt: skip
        self.wait_service()

    def stop(self) -> None:
        self._stop("othello")
        self._stop("central")

    def register(self, login: str, password: str) -> None:
        """Makes `login` a registered account (its password `password`), as rated games need."""
        player = Telnet(self.port)
        player.login(login, password)
        admin = Telnet(self.port)
        admin.login("root", self.root_password)
        for command in (f"name {login} {login}", f"email {login} {login}@localhost"):
            out = admin.command(command)
            if "ERR" in out:
                raise RuntimeError(f"{command}: {out.strip()}")
        admin.close()
        player.close()

    def admin(self, command: str) -> str:
        """One command as root; its output."""
        admin = Telnet(self.port)
        admin.login("root", self.root_password)
        try:
            return admin.command(command)
        finally:
            admin.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="A private GGS server on 127.0.0.1.")
    ap.add_argument("command", choices=("start", "stop", "register", "admin"))
    ap.add_argument("args", nargs="*", help="register: LOGIN PASSWORD; admin: COMMAND")
    ap.add_argument("--run", type=Path, required=True, help="its runtime directory")
    ap.add_argument("--port", type=int, default=15000)
    ap.add_argument(
        "--ggs", type=Path, default=DEFAULT_GGS, help="setup_ggs.py's directory"
    )
    args = ap.parse_args()
    server = LocalGGS(args.run, args.port, args.ggs)
    if args.command == "start":
        server.start()
        print(f"GGS on {HOST}:{args.port}, runtime {server.run}")
    elif args.command == "stop":
        server.stop()
    elif args.command == "register":
        server.register(*args.args)
    else:
        print(server.admin(" ".join(args.args)))


if __name__ == "__main__":
    main()
