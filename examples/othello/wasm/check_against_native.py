"""Whether the browser's Edax and Egaroucid answer as the native ones do, on positions from a
running dashboard's games: each position's best move and score, and how long each took. The browser's
go through the page's own worker, each search a fresh one, as a new native process is. Native Egaroucid
gets the same hash size and keys as the browser's, from a copy of it in a temporary directory, so that
only the code differs, and also its usual table if `--usual-hash`.

    uv run --no-project --with playwright python examples/othello/wasm/check_against_native.py \\
        --url http://127.0.0.1:8941 --levels 8 16 21 --positions 30

Needs Chromium from Playwright (`playwright install chromium`), or --chromium, and the native builds.
"""

import argparse
import array
import json
import random
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright  # pyright: ignore[reportMissingImports]

ENGINES = Path.home() / ".cache/turbozero/othello/engines-2026-10"
# Egaroucid's table in the browser: EGAROUCID_HASH_LEVEL in setup_wasm_engines.sh
BROWSER_HASH_LEVEL = 23


def get(url):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def native_edax(cells, side, level, limit):
    command = [str(ENGINES / "edax/bin/edax"), "-nboard", "-verbose", "1", "-l", str(level),
               "-n", "1", "-book-usage", "off"]  # fmt: skip
    process = subprocess.Popen(command, cwd=ENGINES / "edax", stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)  # fmt: skip
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write(
            f"nboard 1\ngame (;GM[Othello]PC[NBoard]SZ[8]BO[8 {cells} {side}];)\nping 1\n"
        )
        process.stdin.flush()
        while process.stdout.readline().strip() != "pong 1":
            pass
        start = time.time()
        process.stdin.write("hint 1\nping 2\n")
        process.stdin.flush()
        timer = _kill_after(process, limit)
        lines = []
        while (line := process.stdout.readline()) and line.strip() != "pong 2":
            lines.append(line.strip())
        timer.cancel()
        seconds = time.time() - start
        found = [line.split() for line in lines if line.startswith("search")]
        if not found:
            return None
        nodes = [int(line.split()[1]) for line in lines if line.startswith("nodestats")]
        return {"move": found[-1][1].lower()[:2], "score": int(found[-1][-3]), "depth": int(found[-1][-1]),
                "nodes": nodes[-1] if nodes else None, "seconds": seconds}  # fmt: skip
    finally:
        process.kill()
        process.wait()


def _kill_after(process, seconds):
    timer = threading.Timer(seconds, process.kill)
    timer.start()
    return timer


def native_egaroucid(directory, cells, side, level, limit, hash_level):
    board = cells.replace("*", "X")
    side = "X" if side == "*" else "O"
    command = ["./Egaroucid_for_Console.out", "-quiet", "-noboard", "-nobook", "-noise", "-level",
               str(level), "-threads", "1", "-hash", str(hash_level)]  # fmt: skip
    process = subprocess.Popen(command, cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)  # fmt: skip
    assert process.stdin is not None and process.stdout is not None
    try:
        # it replies once it has loaded its weights
        process.stdin.write("version\n")
        process.stdin.flush()
        while "egaroucid" not in (line := process.stdout.readline()).lower():
            if not line:
                raise RuntimeError("egaroucid exited")
        start = time.time()
        process.stdin.write(f"setboard {board} {side}\ngo\n")
        process.stdin.flush()
        timer = _kill_after(process, limit)
        last = None
        while line := process.stdout.readline():
            if found := re.search(
                r"depth (\d+)@(\d+)%.*? value (-?\d+).*? policy (\S+).*? n_nodes (\d+)",
                line,
            ):
                last = found
            if re.fullmatch(r"[a-h][1-8]", line.replace(">", "").strip()):
                break
        else:
            return None
        timer.cancel()
        assert last is not None
        return {"move": last[4], "score": int(last[3]), "depth": int(last[1]), "nodes": int(last[5]),
                "seconds": time.time() - start}  # fmt: skip
    finally:
        process.kill()
        process.wait()


def egaroucid_copy(directory, hash_level):
    """A copy of native Egaroucid that finds hash keys cut to `hash_level` bits, as the browser's."""
    source = ENGINES / "egaroucid/bin"
    directory = Path(directory)
    (directory / "resources/hash").mkdir(parents=True)
    (directory / "Egaroucid_for_Console.out").write_bytes(
        (source / "Egaroucid_for_Console.out").read_bytes()
    )
    (directory / "Egaroucid_for_Console.out").chmod(0o755)
    for name in ("eval.egev2", "eval_move_ordering_end.egev"):
        (directory / "resources" / name).symlink_to(source / "resources" / name)
    keys = (source / "resources/hash/hash25.eghs").read_bytes()
    (directory / "resources/hash/hash25.eghs").write_bytes(keys)
    numbers = array.array("I")
    numbers.frombytes(keys)
    cut = array.array("I", (n & ((1 << hash_level) - 1) for n in numbers))
    (directory / f"resources/hash/hash{hash_level}.eghs").write_bytes(cut.tobytes())
    return directory


# in the page: a game's positions, each with what native engines are told, and a search in a new worker
HELPERS = """
async () => {
  const board = (await import("/game.js")).default;
  const { BrowserEngine } = await import("/web/browser_engine.js");
  window.positionsOf = (moves) => {
    const states = [board.start()];
    for (const move of moves) states.push(board.play(states[states.length - 1], move));
    return states.filter((s) => !s.terminated).map((position) => {
      const passed = board.legal(position)[0] === 64;
      const asked = passed ? board.play(position, 64) : position;
      const cells = asked.squares.map((s) => "-*O"[s]).join("");
      return { position, cells, side: asked.turn === 1 ? "*" : "O", passed, empties: 64 - position.squares.filter((s) => s).length };
    });
  };
  window.browserSearch = (build, engine, position, level, limit) => new Promise((resolve) => {
    const start = performance.now();
    let timer = null;
    const runner = new BrowserEngine(build, (data) => {
      if (data.type === "progress") return;
      clearTimeout(timer);
      runner.close();
      resolve(data.type === "result" ? { move: board.moveName(data.move), score: data.score, depth: data.depth, percent: data.percent, nodes: data.nodes, seconds: data.ms / 1000, total: (performance.now() - start) / 1000 } : { error: data.message });
    });
    runner.search(engine, position, level);
    timer = setTimeout(() => { runner.close(); resolve({ timeout: true }); }, limit * 1000);
  });
}
"""


def sample_positions(page, url, count, low, high, rng):
    games = get(f"{url}/api/games?limit=200&offset=0")
    games = games["games"] if isinstance(games, dict) else games
    chosen = []
    passes = []
    rng.shuffle(games)
    for game in games:
        record = get(f"{url}/api/game?id={game['id']}")
        positions = page.evaluate("(moves) => positionsOf(moves)", record["moves"])
        passes += [p for p in positions if p["passed"] and low <= p["empties"] <= high]
        usable = [
            p for p in positions if low <= p["empties"] <= high and not p["passed"]
        ]
        if usable:
            chosen.append(rng.choice(usable))
        if len(chosen) >= count:
            break
    return chosen[:count] + passes[:2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8031")
    parser.add_argument("--engines", nargs="+", default=["edax", "egaroucid"])
    parser.add_argument("--levels", nargs="+", type=int, default=[8, 16, 21])
    parser.add_argument("--positions", type=int, default=30)
    parser.add_argument("--min-empties", type=int, default=1)
    parser.add_argument("--max-empties", type=int, default=56)
    parser.add_argument("--limit", type=float, default=120.0, help="seconds per search")
    parser.add_argument(
        "--no-native", action="store_true", help="only time the browser"
    )
    parser.add_argument(
        "--usual-hash", action="store_true", help="also native Egaroucid's usual table"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chromium", default=None)
    args = parser.parse_args()
    build = get(f"{args.url}/api/summary")["browser_engines"]
    rng = random.Random(args.seed)

    with sync_playwright() as p, tempfile.TemporaryDirectory() as temporary:
        browser = p.chromium.launch(executable_path=args.chromium, headless=True)
        page = browser.new_page()
        page.goto(args.url)
        page.evaluate(HELPERS)
        positions = sample_positions(
            page, args.url, args.positions, args.min_empties, args.max_empties, rng
        )
        print(f"{len(positions)} positions, {sum(q['passed'] for q in positions)} with only a pass, "
              f"empties {min(q['empties'] for q in positions)} to {max(q['empties'] for q in positions)}", flush=True)  # fmt: skip
        directory = egaroucid_copy(temporary, BROWSER_HASH_LEVEL)
        for engine in args.engines:
            for level in args.levels:
                same = usual_same = total = timeouts = 0
                exact = exact_same = 0
                browser_seconds = native_seconds = 0.0
                slowest = 0.0
                for q in positions:
                    got = page.evaluate(
                        "([b, e, p, l, t]) => browserSearch(b, e, p, l, t)",
                        [build, engine, q["position"], level, args.limit],
                    )
                    if got.get("timeout") or got.get("error"):
                        timeouts += 1
                        print(
                            f"  {engine} L{level} e{q['empties']}: browser {got}",
                            flush=True,
                        )
                        continue
                    slowest = max(slowest, got["seconds"])
                    if args.no_native:
                        browser_seconds += got["seconds"]
                        continue
                    if engine == "edax":
                        native = native_edax(q["cells"], q["side"], level, args.limit)
                    else:
                        native = native_egaroucid(
                            directory,
                            q["cells"],
                            q["side"],
                            level,
                            args.limit,
                            BROWSER_HASH_LEVEL,
                        )
                    if native is None:
                        timeouts += 1
                        print(
                            f"  {engine} L{level} e{q['empties']}: native over the limit",
                            flush=True,
                        )
                        continue
                    move, score = native["move"], native["score"]
                    if q["passed"]:
                        move, score = "pass", -score
                    agrees = got["move"] == move and got["score"] == score
                    total += 1
                    same += agrees
                    browser_seconds += got["seconds"]
                    native_seconds += native["seconds"]
                    if got["percent"] == 100 and got["depth"] == q["empties"]:
                        exact += 1
                        exact_same += got["score"] == score
                    if args.usual_hash and engine == "egaroucid":
                        usual = native_egaroucid(
                            directory, q["cells"], q["side"], level, args.limit, 25
                        )
                        assert usual is not None, "over the limit"
                        usual_move, usual_score = (
                            ("pass", -usual["score"])
                            if q["passed"]
                            else (usual["move"], usual["score"])
                        )
                        usual_same += (
                            got["move"] == usual_move and got["score"] == usual_score
                        )
                    if not agrees:
                        print(
                            f"  {engine} L{level} e{q['empties']} {q['cells']} {q['side']}: browser {got['move']} {got['score']:+d} d{got['depth']}@{got['percent']} n{got['nodes']:.0f}; native {move} {score:+d} d{native['depth']} n{native['nodes']}",
                            flush=True,
                        )
                line = (
                    f"{engine} L{level}: slowest {slowest:.1f} s, over limit {timeouts}"
                )
                if not args.no_native:
                    ratio = (
                        browser_seconds / native_seconds
                        if native_seconds
                        else float("nan")
                    )
                    line += f"; same {same}/{total}, exact {exact_same}/{exact}, {browser_seconds:.1f} s vs native {native_seconds:.1f} s = {ratio:.2f}x"
                    if args.usual_hash and engine == "egaroucid":
                        line += f"; same as usual table {usual_same}/{total}"
                else:
                    line += f"; total {browser_seconds:.1f} s"
                print(line, flush=True)
        browser.close()


if __name__ == "__main__":
    main()
