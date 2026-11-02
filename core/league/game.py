"""What a game brings to the league (`core.league`): everything else is the same for all
two-player games."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from core.evaluators.evaluator import Evaluator
from core.league.store import DEFAULT_TREE_FACTOR, Player, Search
from core.types import EnvStepFn, StepMetadata


class ExternalPlayer(ABC):
    """An engine outside the league, playing a batch of games at once, in step with them."""

    @abstractmethod
    def start(self, states: Any, first: np.ndarray) -> None:
        """Starts a game per state, the games' standard starts; `first` is the id of the
        player who moves first in each."""

    @abstractmethod
    def move(self, states: Any, movers: np.ndarray) -> np.ndarray:
        """(games) its action in the games marked `movers`, each of which it has now
        played on its own board; the others' values don't matter."""

    @abstractmethod
    def observe(self, actions: np.ndarray, movers: np.ndarray) -> None:
        """Tells it about its opponent's `actions` in the games marked `movers`."""

    @abstractmethod
    def finish(self, states: Any, results: np.ndarray) -> None:
        """Called when the games are over, with the first player's `results` (+1, 0, -1),
        for it to check them against its own."""

    @abstractmethod
    def close(self) -> None:
        """Ends its games, however they went."""

    def searches(self, movers: np.ndarray) -> dict[int, Search]:
        """What its search made of the last `move` in the games marked `movers`, by game, where
        it says."""
        return {}

    def settings(self) -> dict[str, Any]:
        """Everything that decides how it plays, once it's `start`ed."""
        return {}


def check_match(game: "LeagueGame", a: Player, b: Player, games: int) -> None:
    """Checks that a series can be asked for: its games are split evenly between the colours.

    Raises:
        ValueError: if it can't, saying why
    """
    for player in (a, b):
        if player.retired:
            raise ValueError(f"{player.name} is retired")
        if not game.can_play(player):
            raise ValueError(f"{player.name} can't play here")
    if a.id == b.id:
        raise ValueError("a player can't play itself")
    check_match_games(games)


def check_match_games(games: int) -> None:
    """Raises ValueError unless `games` can be split evenly between the colours."""
    if games <= 0 or games % 2:
        raise ValueError("games must be positive and even: half with each colour")


class LeagueGame(ABC):
    """A two-player game's part in the league."""

    # the game's name, on the dashboard
    name: str = ""
    # the game's environment (pgx's interface: init, step and each state's fields)
    env: Any
    # a module of the dashboard's: draws the game's boards and steps through its moves. See
    # `examples/othello/league_board.js`
    board_js: Path | None = None
    # engines the dashboard can run in the browser, the build's name: see `build_browser_engines`,
    # and `board_js`'s `engines`
    browser_engines: str | None = None

    def facts(self, player: Player) -> dict[str, str | int | bool]:
        """What can be read from `player`'s name and settings, to filter by: label to value."""
        return {}

    def export_network(self, checkpoint: str, net: str, out: Path) -> None:
        """Writes the `net` network in `checkpoint` to `out` as ONNX, whole or not at all, for
        the dashboard to run in the browser. The dashboard's `board_js` says how to use it."""
        raise NotImplementedError(f"{self.name} has no ONNX export")

    def build_browser_engines(self, out: Path) -> None:
        """Fills the directory `out` with `browser_engines`' files, a file named `complete` last."""
        raise NotImplementedError(f"{self.name} has no browser engines")

    def can_play(self, player: Player) -> bool:
        """Whether `player` can play here, rather than having only results from elsewhere."""
        return True

    @abstractmethod
    def metadata(self, state: Any) -> StepMetadata:
        """The `StepMetadata` of a state."""

    @property
    @abstractmethod
    def step_fn(self) -> EnvStepFn:
        """The environment's step fn, as an evaluator's search takes it."""

    @abstractmethod
    def load_network(self, path: str, net: str, dtype: str) -> tuple[Any, Any]:
        """(network, params): a checkpoint's network and the parameters to evaluate it
        with, computing in `dtype`."""

    @abstractmethod
    def make_evaluator(
        self, network: Any, sims: int, tree_factor: int = DEFAULT_TREE_FACTOR
    ) -> Evaluator:
        """The evaluator of `network`, as it plays at full strength: `sims` iterations a
        move, temperature 0, no root noise, with a tree of `tree_factor` * `sims` nodes.
        Called once per player in a batch."""

    @abstractmethod
    def num_openings(self) -> int:
        """How many openings there are to start games from."""

    @abstractmethod
    def openings(self, indices: np.ndarray) -> np.ndarray:
        """(games, plies) the moves of the openings `indices`, all of the same length,
        which may be 0. They should end in positions that are about even."""

    def external_player(self, player: Player) -> ExternalPlayer:
        """The engine of an external `player`, which it's up to the game to support."""
        raise ValueError(f"{self.name} has no external player {player.external}")

    def external_seconds(self, player: Player) -> float:
        """Roughly how long an external `player` takes over a game, before it's measured."""
        return 0.05

    def external_parallel(self, player: Player) -> int:
        """How many games at once an external `player` can play here."""
        return 128

    def scores(self, states: Any, first: np.ndarray) -> np.ndarray | None:
        """(games, 2) the scores of the first and second players at the end of each game,
        for a game that has them; otherwise None."""
        return None
