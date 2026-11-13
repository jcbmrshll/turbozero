"""The client end to end, on a private GGS server on 127.0.0.1 (see ggs_server.py): our player
against Edax's GGS bot, as on the public server. Not part of the test suite: a match takes
minutes. Run it on idle cores, ours and Edax's apart, with nothing else of yours on them:

    JAX_PLATFORMS=cpu uv run examples/othello/ggs_client/ggs_e2e.py \
        --net 4x64-tandem-s13t-511s-solve20past3.5-e339-avg4 \
        --client-cores 16-19 --edax-cores 20-21

Edax asks our login for each match; our client accepts by its policy. The matches:

1. `--matches` (3) synchro matches, `s8r14` at `01:00//02:00`, the second rated;
2. one where our client's connection is dropped mid-game (SIGUSR1: it reconnects, the server
   having kept the games, and carries on, its trees kept);
3. one where our client is killed mid-game and started again (a new process: it learns the
   games from what the server sends on login);
4. a komi game, `8k` at `01:00//02:00` (each side sends its first move with the result it
   expects; the server sets the komi and colours from them).

It passes if every match finished with a result, and our client saw no protocol errors, no
timeouts of ours and no boards that disagreed with its replay of the moves. `--fake-player`
plays the first legal move instead of the network, to test the protocol alone in seconds.
Logs, transcripts and the server's runtime are left in the run directory it prints.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ggs_protocol as gp
from ggs_server import LocalGGS

HERE = Path(__file__).resolve().parent
EDAX_DIR = Path.home() / ".cache/turbozero/othello/engines-2026-10/edax"
LOGIN = "tzbot"
EDAX = "edax"


def free_port() -> int:
    """A loopback port nothing listens on (Central takes only this one, uncompressed)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Process:
    """A process in a group of its own, ended by its group."""

    def __init__(self, args: list[str], log: Path, cwd: Path | None = None, **kwargs):
        self.log = open(log, "ab")  # noqa: SIM115 - closed by stop()
        self.popen = subprocess.Popen(
            args, cwd=cwd, stdout=self.log, stderr=subprocess.STDOUT,
            start_new_session=True, **kwargs,
        )  # fmt: skip

    def signal(self, signum: int) -> None:
        try:
            os.killpg(self.popen.pid, signum)
        except ProcessLookupError:
            pass

    def stop(self, timeout: float = 30.0) -> None:
        if self.popen.poll() is None:
            self.signal(signal.SIGTERM)
            try:
                self.popen.wait(timeout)
            except subprocess.TimeoutExpired:
                self.signal(signal.SIGKILL)
                self.popen.wait()
        self.log.close()


class EdaxBot(Process):
    """Edax logged in to GGS (its `-ggs` mode), its stdin open: what it reads there it sends
    to the server, so it asks for our matches. It sets `open` to 1 itself, too few for a
    synchro match, so each ask raises it first."""

    def __init__(self, port: int, password: str, cores: str, threads: int, log: Path):
        command = [
            "taskset", "-c", cores, str(EDAX_DIR / "bin/edax"), "-ggs",
            "-ggs-host", "127.0.0.1", "-ggs-port", str(port), "-ggs-login", EDAX,
            "-ggs-password", password, "-ggs-open", "on", "-n", str(threads),
            "-book-usage", "off",
            # it plays on the game's clock (any time here does: it takes the server's)
            "-game-time", "1:00",
        ]  # fmt: skip
        super().__init__(command, log, cwd=EDAX_DIR, stdin=subprocess.PIPE, text=True)

    def ask(self, match: str, rated: bool) -> None:
        stdin = self.popen.stdin
        assert stdin is not None
        stdin.write(f"tell /os rated {'+' if rated else '-'}\n")
        stdin.write(f"tell /os open 2\ntell /os ask {match} {LOGIN}\n")
        stdin.flush()


class Client(Process):
    def __init__(self, args: list[str], cores: str, log: Path):
        env = dict(os.environ)
        if "device=gpu" not in " ".join(args):
            # JAX would otherwise take the GPU
            env["JAX_PLATFORMS"] = "cpu"
        command = [
            "taskset",
            "-c",
            cores,
            sys.executable,
            str(HERE / "ggs_client.py"),
            *args,
        ]
        super().__init__(command, log, env=env)


def ended_matches(transcript: Path) -> list[gp.MatchEnd]:
    """The matches of ours that ended, by the client's transcript."""
    if not transcript.exists():
        return []
    received = "".join(
        line.split(" < ", 1)[1] + "\n"
        for line in transcript.read_text().splitlines()
        if " < " in line
    )
    return [
        e
        for e in gp.parse_stream(received)
        if isinstance(e, gp.MatchEnd) and LOGIN in e.players
    ]


def our_moves(transcript: Path, offset: int) -> int:
    """Moves we've sent since `offset` in the transcript."""
    with open(transcript) as f:
        f.seek(offset)
        return f.read().count("> tell /os play ")


def check_transcript(transcript: Path) -> dict:
    """What the server said to us that a clean run shouldn't have: its errors, our timeouts."""
    received = "".join(
        line.split(" < ", 1)[1] + "\n"
        for line in transcript.read_text().splitlines()
        if " < " in line
    )
    problems = {"service_errors": [], "timeouts": [], "unreadable": []}
    assembler = gp.Assembler()
    for lines in assembler.feed(received) + assembler.flush():
        try:
            event = gp.parse(lines)
        except gp.ProtocolError as e:
            problems["unreadable"].append(str(e))
            continue
        if isinstance(event, gp.ServiceError) and "not registered" not in event.text:
            problems["service_errors"].append(event.text)
        if isinstance(event, gp.Timeout) and event.name == LOGIN:
            problems["timeouts"].append(f"{event.game_id} fatal={event.fatal}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--net", default="4x64-tandem-s13t-511s-solve20past3.5-e339-avg4")
    ap.add_argument("--player", default="batch=8,solve=20",
                    help="the net player's other options (see clock_match.py's net:)")  # fmt: skip
    ap.add_argument("--fake-player", action="store_true")
    ap.add_argument("--client-cores", default="16-19")
    ap.add_argument("--edax-cores", default="20-21")
    ap.add_argument("--edax-threads", type=int, default=2)
    ap.add_argument("--matches", type=int, default=3, help="plain synchro matches")
    ap.add_argument("--type", default="s8r14")
    ap.add_argument("--clock", default="01:00//02:00")
    ap.add_argument("--komi-type", default="8k")
    ap.add_argument("--skip", default="", help="phases to skip: drop,restart,komi")
    ap.add_argument("--run", type=Path, default=None, help="a new directory for it all")
    ap.add_argument("--ggs", type=Path, default=None, help="setup_ggs.py's directory")
    args = ap.parse_args()

    load = os.getloadavg()[0]
    print(f"load {load:.1f} on {os.cpu_count()} cores", flush=True)
    run = args.run or Path(tempfile.mkdtemp(prefix="ggs-e2e-"))
    run.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server = LocalGGS(run / "server", port, *([args.ggs] if args.ggs else []))
    processes: list[Process] = []
    skip = set(args.skip.split(","))
    failures: list[str] = []
    try:
        server.start()
        print(f"GGS on 127.0.0.1:{port}; everything in {run}", flush=True)
        passwords = {LOGIN: "tzbot-pw", EDAX: "edax-pw"}
        for login, password in passwords.items():
            server.register(login, password)
        (run / "tzbot.pw").write_text(passwords[LOGIN])
        (run / "tzbot.pw").chmod(0o600)

        transcript = run / "tzbot.transcript"
        client_args = [
            "--port", str(port), "--login", LOGIN, "--password-file", str(run / "tzbot.pw"),
            "--types", f"{args.type},{args.komi_type}", "--transcript", str(transcript),
            "--player", f"net={args.net},{args.player}", "--keepalive", "30",
        ]  # fmt: skip
        if args.fake_player:
            client_args.append("--fake-player")
        runs = 0

        def start_client() -> Client:
            nonlocal runs
            runs += 1
            client = Client(
                client_args + ["--results", str(run / f"tzbot.results.{runs}.json")],
                args.client_cores,
                run / f"tzbot.{runs}.log",
            )
            processes.append(client)
            return client

        client = start_client()
        edax = EdaxBot(
            port, passwords[EDAX], args.edax_cores, args.edax_threads, run / "edax.log"
        )
        processes.append(edax)
        time.sleep(5)

        # (match type, what to do to our client during it)
        plan = [(args.type, None)] * args.matches
        if "drop" not in skip:
            plan.append((args.type, "drop"))
        if "restart" not in skip:
            plan.append((args.type, "restart"))
        if "komi" not in skip:
            plan.append((args.komi_type, None))
        timeout = 600.0
        for i, (match_type, action) in enumerate(plan):
            before = len(ended_matches(transcript))
            print(f"match {i + 1}/{len(plan)}: {match_type} {args.clock}"
                  + (f", {action}" if action else ""), flush=True)  # fmt: skip
            offset = transcript.stat().st_size if transcript.exists() else 0
            # a dropped connection ends any match outside a tournament; a rated one is stored,
            # and resumed as a new match
            edax.ask(f"{match_type} {args.clock}", rated=i == 1 or action is not None)
            start = time.monotonic()
            acted = False
            while not [
                m for m in ended_matches(transcript)[before:] if m.result is not None
            ]:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"match {i + 1} didn't end in {timeout:.0f} s")
                if action and not acted and our_moves(transcript, offset) >= 8:
                    acted = True
                    if action == "drop":
                        print("  dropping the client's connection", flush=True)
                        client.signal(signal.SIGUSR1)
                    else:
                        print("  killing the client and starting it again", flush=True)
                        client.signal(signal.SIGKILL)
                        client.popen.wait()
                        client = start_client()
                for name, process in (("client", client), ("edax", edax)):
                    if process.popen.poll() is not None:
                        raise RuntimeError(f"{name} exited")
                time.sleep(1)
            for end in ended_matches(transcript)[before:]:
                ours = end.result if end.players[0] == LOGIN else -(end.result or 0)
                print(f"  {end.id} ended: {end.reason or f'{ours:+.2f} for us'}"
                      f" ({time.monotonic() - start:.0f} s)", flush=True)  # fmt: skip
                if end.result is None and not action:
                    failures.append(f"match {end.id} didn't finish: {end.reason}")
            time.sleep(3)
        client.stop(timeout=60)
        edax.stop()

        results = [
            json.loads(path.read_text())
            for path in sorted(run.glob("tzbot.results.*.json"))
        ]
        problems = check_transcript(transcript)
        for kind, items in problems.items():
            if items:
                failures.append(f"{kind}: {items}")
        for result in results:
            adjourned = [r for r in result["results"] if r[3] is None]
            if adjourned and len(results) == 1 and "drop" in skip:
                failures.append(f"adjourned: {adjourned}")
            errors = [e for e in result["errors"] if "not registered" not in e]
            if errors:
                failures.append(f"client errors: {errors}")
            if result["desyncs"]:
                failures.append(f"{result['desyncs']} desyncs")
        summary = {
            "matches": [
                (
                    m.id,
                    m.type.text,
                    m.result
                    if m.players[0] == LOGIN
                    else None
                    if m.result is None
                    else -m.result,
                    m.reason,
                )
                for m in ended_matches(transcript)
            ],
            "client_runs": results,
            "failures": failures,
            "run": str(run),
        }
        (run / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
    finally:
        for process in processes:
            process.stop()
        server.stop()
    print("PASS" if not failures else "FAIL", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
