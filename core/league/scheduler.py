"""Picks the next batch of games, in two steps:

- The focal player is the most uncertain one. A player with no games is more uncertain than
  anyone with some, whatever the prior says.
- The opponent is drawn from a normal distribution of ratings centred on the focal player's,
  mixed with a uniform one. The normal's variance includes both players' uncertainty, so it
  widens to nearly uniform where little is known. It's weighted by the inverse of the pair's
  cost, and down for pairs from the same training run, whose descendants beat their ancestors
  out of proportion to real strength (`same_run_weight`).

Far-apart games say little about the ratings under the model, and the normal's tails almost
never draw them. They're what shows the things the model can't see, such as ratings drifting
along chains of near neighbours and inflation against engines, so `uniform_fraction` of the
draw keeps them going, at the cost of that much of the GPU.

Engines are players like any other, except that `engines` can leave them out of new games
altogether.

Other trade-offs, in `SchedulerConfig`: `width` trades information per game (narrow) against
ties across the scale (wide).
"""

from dataclasses import dataclass

import numpy as np

from core.league.ratings import Ratings
from core.league.store import Player


@dataclass(frozen=True)
class SchedulerConfig:
    """How the next batch is picked.

    Attributes:
        width: the standard deviation of the ratings opponents are drawn from, in Elo
        same_run_weight: how much less likely an opponent from the same training run is
        uniform_fraction: how much of the opponent draw is uniform over all the players
        engines: whether external engines play new games
        batch_seconds: how long a batch should take at most
        batch_sizes: the sizes of batch to choose from, in games, each even: the search
            compiles for each, so the fewer the better
        max_tree_nodes: the most tree nodes (of both sides' searches) a batch may hold at
            once, for the memory they take
    """

    width: float = 150.0
    same_run_weight: float = 0.5
    uniform_fraction: float = 0.1
    engines: bool = True
    batch_seconds: float = 180.0
    batch_sizes: tuple[int, ...] = (64, 128, 256, 512, 1024)
    max_tree_nodes: int = 6_000_000


@dataclass(frozen=True)
class Pairing:
    """The next batch.

    Attributes:
        a: the focal player's id
        b: its opponent's id
        games: how many games to play, half with each moving first
    """

    a: int
    b: int
    games: int


def batch_size(
    config: SchedulerConfig, a: Player, b: Player, seconds_per_game: float
) -> int:
    """The largest batch of the pair's that takes at most `batch_seconds` and fits in
    memory, or the smallest there is. A game's search keeps 2 * sims nodes for each side."""
    nodes = 2 * ((a.sims or 0) + (b.sims or 0))
    fits = [
        size
        for size in sorted(config.batch_sizes)
        if size * seconds_per_game <= config.batch_seconds
        # half the games are played at a time
        and size // 2 * nodes <= config.max_tree_nodes
    ]
    return fits[-1] if fits else min(config.batch_sizes)


def opponent_weights(
    config: SchedulerConfig,
    ratings: Ratings,
    players: list[Player],
    costs: np.ndarray,
) -> np.ndarray:
    """(n, n) how likely each column is as the opponent of each row (each row sums to 1,
    or to 0 if it has no opponent): the normal distribution centred on the row's rating,
    weighted by the pair's cost and by whether it's within a training run, mixed with a
    uniform one."""
    sd = ratings.sd
    difference = ratings.elo[:, None] - ratings.elo[None, :]
    var = config.width**2 + sd[:, None] ** 2 + sd[None, :] ** 2
    weights = np.exp(-(difference**2) / (2 * var)) / np.sqrt(var)
    runs = np.array([p.run for p in players], dtype=object)
    same_run = runs[:, None] == runs[None, :]
    # the same network at another search budget is how those budgets connect
    networks = np.array([p.network or p.name for p in players], dtype=object)
    same_network = networks[:, None] == networks[None, :]
    weights = np.where(
        same_run & ~same_network, weights * config.same_run_weight, weights
    )
    allowed = _allowed(config, players)
    normal = _normalized(weights / costs * allowed)
    uniform = _normalized(allowed.astype(float))
    return (1 - config.uniform_fraction) * normal + config.uniform_fraction * uniform


def _allowed(config: SchedulerConfig, players: list[Player]) -> np.ndarray:
    """(n, n) which pairs may play: not a player itself, a retired one, or any engine while
    they're left out."""
    external = np.array([p.is_external for p in players])
    active = np.array([not p.retired for p in players])
    if not config.engines:
        active = active & ~external
    return active[:, None] & active[None, :] & ~np.eye(len(players), dtype=bool)


def _normalized(weights: np.ndarray) -> np.ndarray:
    totals = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)


def next_pairing(
    config: SchedulerConfig,
    ratings: Ratings,
    players: list[Player],
    seconds_per_game: np.ndarray,
    rng: np.random.Generator,
) -> Pairing | None:
    """The next batch to play.

    Args:
        config: how to pick it
        ratings: the fit, of `players`, in the same order
        players: every player
        seconds_per_game: (n, n) how long a game of each pair takes
        rng: for the draws

    Returns:
        the batch, or None if no two players can play
    """
    allowed = _allowed(config, players)
    if not allowed.any():
        return None
    uncertainty = np.where(ratings.games == 0, np.inf, ratings.sd)
    i = int(np.argmax(np.where(allowed.any(axis=1), uncertainty, -np.inf)))
    opponents = opponent_weights(config, ratings, players, seconds_per_game)
    j = int(rng.choice(len(players), p=opponents[i]))
    games = batch_size(config, players[i], players[j], seconds_per_game[i, j])
    return Pairing(players[i].id, players[j].id, games)


def choose_openings(
    num_openings: int, played: set[int], count: int, rng: np.random.Generator
) -> np.ndarray:
    """`count` openings a pair hasn't played yet, the played ones only when none are left,
    so that a pair of nearly deterministic players doesn't repeat its games."""
    all_openings = np.arange(num_openings)
    fresh = np.setdiff1d(
        all_openings, np.fromiter(played, dtype=int, count=len(played))
    )
    taken = rng.permutation(fresh)[:count]
    missing = count - len(taken)
    if missing > 0:
        # every opening has been played: repeats
        again = rng.choice(num_openings, size=missing, replace=missing > num_openings)
        taken = np.concatenate([taken, again])
    return taken


@dataclass(frozen=True)
class CostModel:
    """How long a game takes, in seconds, from 8x128 and 6x96 nets on an RTX 5080 in bfloat16
    (0.04 s at 128 sims a side, 0.4 s at 1000): a side's search costs `per_sim` per iteration, in
    proportion to its network's `size(net)`, on top of `per_game`; an external engine
    costs what the game says. Measured batches correct this, each player by the median of
    (seconds measured / seconds modelled) over its batches.

    Attributes:
        per_game: the seconds of a game that don't depend on either player
        per_sim: seconds of a game per MCTS iteration a move, of a network of size 1
    """

    per_game: float = 0.005
    per_sim: float = 2.0e-4


def network_size(net: str | None) -> float:
    """A network's cost relative to resnet:8x128, from a spec of its blocks x channels:
    blocks * channels^2. Unknown specs count as 1."""
    if net is None or ":" not in net:
        return 1.0
    shape = net.split(":")[1].split(",")[0]
    try:
        blocks, channels = (int(x) for x in shape.split("x"))
    except ValueError:
        return 1.0
    return blocks * channels**2 / (8 * 128**2)


def seconds_per_game(
    players: list[Player],
    external_seconds: dict[int, float],
    batches: list[tuple[int, int, int, float]],
    model: CostModel | None = None,
) -> np.ndarray:
    """(n, n) how long a game of each pair takes, at the model's costs corrected by the
    measured `batches` (`Store.batch_costs`).

    Args:
        players: every player
        external_seconds: seconds of a game for each external player, by id
        batches: (a, b, games, seconds) of the batches measured
        model: the costs before measuring

    Returns:
        the seconds, with each pair's sum of its two sides plus the fixed part
    """
    model = model or CostModel()
    index = {p.id: i for i, p in enumerate(players)}
    side = np.array(
        [
            external_seconds.get(p.id, 0.05)
            if p.is_external
            else (p.sims or 0) * model.per_sim * network_size(p.net)
            for p in players
        ]
    )
    predicted = side[:, None] + side[None, :] + model.per_game
    ratios: list[list[float]] = [[] for _ in players]
    for a, b, games, seconds in batches:
        if a in index and b in index and games:
            i, j = index[a], index[b]
            ratio = (seconds / games) / predicted[i, j]
            ratios[i].append(ratio)
            ratios[j].append(ratio)
    correction = np.array([float(np.median(r)) if r else 1.0 for r in ratios])
    # each side's correction goes with its share of the time
    corrected = side * correction
    return corrected[:, None] + corrected[None, :] + model.per_game
