"""A continuously running Elo league of Othello players (see `core.league`): networks at a
search budget, and Edax and Egaroucid at a level. Games start from XOT openings (`xot.py`),
each played with both colors, and no pair repeats an opening until it's played them all.

    uv run examples/othello/elo_league.py add-network A-8x CHECKPOINT \\
        --net resnet:8x128,pool_every=2,heads=pooled,pool_channels=32 --run r2-A --sims 128 1000
    uv run examples/othello/elo_league.py add-external edax-L8 edax --settings '{"level": 8}'
    uv run examples/othello/elo_league.py anchor edax-L8
    uv run examples/othello/elo_league.py match A-8x@1000 edax-L24 --games 400
    uv run examples/othello/elo_league.py ratings
    uv run examples/othello/elo_league.py dashboard
    uv run examples/othello/elo_league.py export-onnx    # networks registered without an export
    uv run examples/othello/elo_league.py fetch-ort      # once, to run networks in the browser
    uv run examples/othello/elo_league.py build-browser-engines  # once, for Edax and Egaroucid there

The worker runs from a frozen copy of the code, from a slick task whose worktree it queues in
(SLICK_TASK names the task; a queue entry's end prompts that task's agent):

    git worktree add --detach ~/worktrees/turbozero/elo-frozen && cd ~/worktrees/turbozero/elo-frozen
    ./install.sh
    tmux new-session -d -s elo-league \
        uv run --extra cuda examples/othello/elo_league.py start --queue-cwd "$TASK_WORKTREE"
    uv run --extra cuda examples/othello/elo_league.py stop

The store is in ~/.cache/turbozero/othello/league (see --root).
"""

import os
import re
import subprocess
import sys

# only the player needs the GPU: everything else stays off it, where it would take memory
if "play" not in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from engines import CACHE_DIR, ENGINES, GTPEngine
from game import (
    PASS,
    env,
    load_checkpoint,
    make_test_evaluator,
    metadata,
    state_to_nn_input,
    step_fn,
)
from vs_engine import available_cpus, max_parallel_games
from xot import XOT_PATH, load_xot

from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.league.cli import main
from core.league.game import ExternalPlayer, LeagueGame
from core.league.store import EngineSearch, Player, Search
from core.monitoring.episodes import BOARDS

# a self-play network's name and a student's (see `distill.py`)
SELF_PLAY = re.compile(
    r"\d+x\d+-(?P<sims>\d+)s-(?P<passes>\d+)p-w(?P<window>\d+)(?P<vtq>-vtq)?"
    r"-(?P<minutes>\d+)m(?P<planned>-planned)?"
)
STUDENT = re.compile(
    r"\d+x\d+-distill-(?P<target>\w+)-(?P<minutes>\d+)m-(?P<passes>\d+)p"
)


def player_facts(player: Player) -> dict[str, str | int | bool]:
    """What's in a player's settings and name, as in 8x128-128s-8p-w1125-274m@1000: its net, its
    training sims, passes per position, window and GPU-minutes, and its sims in play."""
    if player.external is not None:
        facts: dict[str, str | int | bool] = {
            "kind": "engine",
            "engine": player.external,
        }
        for label, key in (("level", "level"), ("play sims", "sims")):
            if key in player.settings:
                facts[label] = player.settings[key]
        return facts
    facts = {"kind": "net"}
    assert player.net is not None and player.network is not None
    if shape := re.search(r"(\d+)x(\d+)", player.net):
        facts["net"] = shape[0]
        facts["blocks"], facts["channels"] = int(shape[1]), int(shape[2])
    if player.sims is not None:
        facts["play sims"] = player.sims
    if name := SELF_PLAY.fullmatch(player.network):
        facts |= {
            "train sims": int(name["sims"]),
            "passes": int(name["passes"]),
            "window": int(name["window"]),
            "minutes": int(name["minutes"]),
            "vtq": name["vtq"] is not None,
            "planned": name["planned"] is not None,
        }
    elif name := STUDENT.fullmatch(player.network):
        facts |= {
            "distill target": name["target"],
            "teacher minutes": int(name["minutes"]),
            "passes": int(name["passes"]),
        }
    return facts


def color(first: bool) -> str:
    return "b" if first else "w"


class EnginePlayer(ExternalPlayer):
    """An engine's side of a batch: a process per game (`engines.GTPEngine`), running on the
    cores this process may use, whose result is checked against each one's own board."""

    def __init__(self, engine: type[GTPEngine], level: int, args: list[str]):
        self.engine = engine
        self.level = level
        self.args = args
        self.pool = ThreadPoolExecutor(max_workers=available_cpus())
        self.processes: list[GTPEngine] = []
        self.first: np.ndarray = np.zeros(0, dtype=int)
        self.started: dict[str, Any] = {}
        self.asked: dict[int, int] = {}

    def start(self, states: Any, first: np.ndarray) -> None:
        self.first = first
        count = len(first)
        self.processes = list(
            self.pool.map(
                lambda _: self.engine(
                    self.engine.default_dir(), self.level, analyse=True, args=self.args
                ),
                range(count),
            )
        )
        self.started = self.processes[0].settings()
        self.plies = np.zeros(count, dtype=int)

    def move(self, states: Any, movers: np.ndarray) -> np.ndarray:
        mask = np.asarray(states.legal_action_mask)
        to_move = np.asarray(states.current_player)
        moves = np.full(len(movers), PASS, dtype=np.int32)
        # a side with no move passes without asking
        asking = [g for g in np.flatnonzero(movers) if not mask[g, PASS]]
        replies = self.pool.map(
            lambda g: self.processes[g].genmove(color(to_move[g] == self.first[g])),
            asking,
        )
        for g, move in zip(asking, replies):
            moves[g] = move
        self.asked = {int(g): int(moves[g]) for g in asking}
        self.plies[movers] += 1
        return moves

    def searches(self, movers: np.ndarray) -> dict[int, Search]:
        found: dict[int, Search] = {}
        for g, move in self.asked.items():
            process = self.processes[g]
            analysis = next(
                (a for a in process.analysis or [] if a.line[0] == move), None
            )
            if analysis is not None:
                found[g] = EngineSearch(analysis.score, analysis.depth, process.nodes)
        return found

    def settings(self) -> dict[str, Any]:
        return self.started

    def observe(self, actions: np.ndarray, movers: np.ndarray) -> None:
        # players alternate, a pass being a move: black moves at even plies
        games = np.flatnonzero(movers)
        list(
            self.pool.map(
                lambda g: self.processes[g].play(
                    color(self.plies[g] % 2 == 0), int(actions[g])
                ),
                games,
            )
        )
        self.plies[movers] += 1

    def finish(self, states: Any, results: np.ndarray) -> None:
        for g, (black, white) in enumerate(
            self.pool.map(lambda e: e.discs(), self.processes)
        ):
            assert np.sign(black - white) == results[g], (
                f"game {g}: {self.engine.name} disagrees on the result"
            )

    def close(self) -> None:
        for process in self.processes:
            process.close()
        self.processes = []
        self.pool.shutdown()


class Othello(LeagueGame):
    name = "Othello"
    env = env
    board_js = Path(__file__).with_name("league_board.js")
    # a new build needs a new name: browsers keep its files for good
    browser_engines = "edax-8a2b3ea-egaroucid-3b1c6ad-1"

    def __init__(self, xot: Path = XOT_PATH):
        self.xot = load_xot(xot)

    def facts(self, player: Player) -> dict[str, str | int | bool]:
        return player_facts(player)

    def export_network(self, checkpoint: str, net: str, out: Path) -> None:
        from export_onnx import save_onnx

        save_onnx(checkpoint, net, out)

    def build_browser_engines(self, out: Path) -> None:
        script = Path(__file__).with_name("setup_wasm_engines.sh")
        subprocess.run([str(script), str(out)], check=True)

    def metadata(self, state: Any):
        return metadata(state)

    @property
    def step_fn(self):
        return step_fn

    def load_network(self, path: str, net: str, dtype: str):
        return load_checkpoint(path, net, dtype)

    def make_evaluator(self, network: Any, sims: int):
        return make_test_evaluator(
            make_nn_eval_fn(network, state_to_nn_input), sims, noise=False
        )

    def num_openings(self) -> int:
        return len(self.xot)

    def openings(self, indices: np.ndarray) -> np.ndarray:
        return self.xot[indices]

    def can_play(self, player: Player) -> bool:
        return player.external is None or player.external in ENGINES

    def external_player(self, player: Player) -> ExternalPlayer:
        assert player.external is not None
        settings = player.settings
        return EnginePlayer(
            ENGINES[player.external], settings["level"], settings.get("args", [])
        )

    def external_seconds(self, player: Player) -> float:
        # Edax at level 8 takes about 0.1 s of wall clock a game with 8 cores in parallel;
        # each level about 1.5x the one below is a guess, which measured batches correct
        return 0.1 * 1.5 ** (player.settings.get("level", 8) - 8)

    def external_parallel(self, player: Player) -> int:
        assert player.external is not None
        return max_parallel_games(ENGINES[player.external])

    def scores(self, states: Any, first: np.ndarray) -> np.ndarray:
        board, _ = BOARDS["othello"](states._x)
        black = (board == 1).sum(axis=(1, 2))
        white = (board == 2).sum(axis=(1, 2))
        return np.stack([black, white], axis=1).astype(float)


if __name__ == "__main__":
    main(Othello, CACHE_DIR / "league")
