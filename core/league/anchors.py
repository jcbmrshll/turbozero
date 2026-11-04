"""Games against fixed anchors as a network trains, for its Elo on a fixed scale: a measure of
strength whose opponents don't move as it learns, unlike its training loss's targets.

Every few epochs the network plays each anchor, a frozen network of known rating (e.g. a league
player's checkpoint and rating), as the league's direct games are played (see `core.league.match`):
both searching the same iterations a move, with no noise, from the game's openings, each opening
played with both colours. The openings and the games' keys are drawn once, so every evaluation
plays the same games but for the network's moves. Its rating against each anchor, and against all of them at once, is
the most likely one given the anchors' ratings (see `core.league.ratings.rate_against`).
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from core.league.game import LeagueGame, check_match_games
from core.league.match import NetworkSide, Programs, compile_program, play_games
from core.league.ratings import rate_against
from core.league.store import DEFAULT_TREE_FACTOR, Player
from core.testing.tester import BaseTester, TestState


@dataclass(frozen=True)
class Anchor:
    """A frozen network to measure against: its name in the metrics, its checkpoint, the spec of
    its network (see `core.networks.spec`), and its rating, playing at the tester's search budget."""

    name: str
    path: str
    net: str
    elo: float


class AnchorTester(BaseTester):
    """Plays `games` games against each of `anchors` after every `every` epochs (epochs every - 1,
    2 * every - 1, ... counting from 0, as the run logs them), and logs, at the epoch of the network
    that played:

    - `anchor_<name>_score`: its score against an anchor, a draw counting half
    - `anchor_<name>_elo`, `anchor_<name>_elo_se`: its rating from those games, and its standard error
    - `anchor_elo`, `anchor_elo_se`: its rating from all of them, on the anchors' scale
    - `anchor_seconds`: the time the games took, to take out of the run's wall time

    It plays in the training process, on the same accelerator, between epochs: the anchors' networks
    stay loaded, and the searches take their memory from what the run has.
    """

    def __init__(
        self,
        game: LeagueGame,
        network: Any,
        anchors: Sequence[Anchor],
        sims: int = 128,
        games: int = 400,
        every: int = 10,
        dtype: str = "bfloat16",
        tree_factor: int = DEFAULT_TREE_FACTOR,
        seed: int = 0,
        results: Path | None = None,
    ):
        """
        Args:
            game: the game, as the league plays it
            network: the network that trains, computing in the dtype to play in (`dtype`, for the
                anchors)
            anchors: the anchors, at least one, their names distinct
            sims: MCTS iterations a move, each side's
            games: games against each anchor, half with each colour
            every: play after every this many epochs
            dtype: the dtype the anchors' networks compute in
            tree_factor: each search's tree holds this many nodes per iteration
            seed: draws the openings, and seeds the games
            results: (optional) also append each anchor's result to this file, as a line of JSON
        """
        super().__init__(num_keys=1, name="anchors")
        if not anchors:
            raise ValueError("no anchors to play")
        if len({a.name for a in anchors}) < len(anchors):
            raise ValueError("anchors' names must be distinct")
        if every < 1:
            raise ValueError(f"every must be positive, got {every}")
        check_match_games(games)
        self.game = game
        self.anchors = list(anchors)
        self.sims = sims
        self.games = games
        self.every = every
        self.dtype = dtype
        self.tree_factor = tree_factor
        self.seed = seed
        self.results = results
        rng = np.random.default_rng(seed)
        self.openings = {
            a.name: game.openings(
                rng.choice(game.num_openings(), games // 2, replace=False)
            )
            for a in self.anchors
        }
        self.program = compile_program(game, network, sims, tree_factor)
        self.programs = Programs(game, dtype, networks=len(self.anchors))
        self.players = {
            a.name: Player(
                id=-1 - i,
                name=a.name,
                sims=sims,
                path=a.path,
                net=a.net,
                settings={"tree_factor": tree_factor},
            )
            for i, a in enumerate(self.anchors)
        }
        # loaded now, so that a wrong path or spec fails before training starts
        for player in self.players.values():
            self.programs.network(player)

    def plays_at(self, epoch: int) -> bool:
        """Whether it plays the network after `epoch` (counting from 0)."""
        return (epoch + 1) % self.every == 0

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "anchors": [vars(a) for a in self.anchors],
            "sims": self.sims,
            "games": self.games,
            "every": self.every,
            "dtype": self.dtype,
            "tree_factor": self.tree_factor,
            "seed": self.seed,
        }

    def play(self, anchor: Anchor, params: Any) -> np.ndarray:
        """(games) the network's result in each game against `anchor` (+1, 0, -1): the openings
        with it moving first, then with the anchor."""
        key = jax.random.fold_in(
            jax.random.PRNGKey(self.seed), self.anchors.index(anchor)
        )
        player = self.players[anchor.name]
        openings = self.openings[anchor.name]
        results = []
        for half, net_first in enumerate((True, False)):
            env_key, net_key, anchor_key = jax.random.split(
                jax.random.fold_in(key, half), 3
            )
            net = NetworkSide(self.program, params, net_key)
            other = NetworkSide(
                self.programs.program(player, len(openings)),
                self.programs.network(player)[1],
                anchor_key,
            )
            first, second = (net, other) if net_first else (other, net)
            try:
                first_results, *_ = play_games(
                    self.game, first, second, openings, env_key
                )
            finally:
                first.close()
                second.close()
            results.append(first_results if net_first else -first_results)
        return np.concatenate(results)

    def run(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        key: jax.Array,
        epoch_num: int,
        max_steps: int,
        env_step_fn: Any,
        env_init_fn: Any,
        evaluator: Any,
        state: TestState,
        params: Any,
        *args,
        log_fn: Callable[[dict], None] | None = None,
        activity_fn: Callable[[str], None] | None = None,
    ) -> tuple[TestState, dict, None]:
        """Plays the anchors at the epochs it's set to (see `BaseTester.run`), logging the results
        with `log_fn`; returns no metrics or episode of its own. Its games take their keys from
        its seed, not `key`."""
        if not self.plays_at(epoch_num):
            return state, {}, None
        start = time.perf_counter()
        metrics: dict[str, float] = {}
        scores, counts = [], []
        for anchor in self.anchors:
            if activity_fn is not None:
                activity_fn(
                    f"{self.games} games against {anchor.name}, {self.sims} sims"
                )
            results = self.play(anchor, params)
            score = float(((1 + results) / 2).sum())
            rating = rate_against([anchor.elo], [score], [len(results)])
            scores.append(score)
            counts.append(len(results))
            metrics |= {
                f"anchor_{anchor.name}_score": score / len(results),
                f"anchor_{anchor.name}_elo": rating.elo,
                f"anchor_{anchor.name}_elo_se": rating.sd,
            }
            print(
                f"epoch {epoch_num}: {anchor.name} (rated {anchor.elo:.0f}): "
                f"+{(results > 0).sum()} ={(results == 0).sum()} -{(results < 0).sum()}, "
                f"rated {rating.elo:.0f} ± {rating.sd:.0f}",
                flush=True,
            )
            if self.results is not None:
                with open(self.results, "a") as f:
                    record = {
                        "epoch": epoch_num,
                        "anchor": anchor.name,
                        "anchor_elo": anchor.elo,
                        "wins": int((results > 0).sum()),
                        "draws": int((results == 0).sum()),
                        "losses": int((results < 0).sum()),
                        "elo": rating.elo,
                        "elo_se": rating.sd,
                        "sims": self.sims,
                        "dtype": self.dtype,
                    }
                    f.write(json.dumps(record) + "\n")
        combined = rate_against([a.elo for a in self.anchors], scores, counts)
        metrics |= {
            "anchor_elo": combined.elo,
            "anchor_elo_se": combined.sd,
            "anchor_seconds": time.perf_counter() - start,
        }
        print(
            f"epoch {epoch_num}: rated {combined.elo:.0f} ± {combined.sd:.0f} by the anchors",
            flush=True,
        )
        if log_fn is not None:
            log_fn(metrics)
        return state, {}, None
