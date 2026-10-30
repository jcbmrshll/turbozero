"""Picks the next batch of games: where they'd reduce the uncertainty of the ratings most,
per second of the GPU.

What a batch is worth is known exactly, given the fit (`ratings`): `n` games between two
players add `c = n K^2 p (1 - p)` to the precision of the difference `d` between their ratings
(the near-even games, with p near 1/2, add the most), which removes

    gain = c |Sigma d|^2 / (1 + c d' Sigma d)

from the sum of all the ratings' variances, the Sherman-Morrison update of the covariance
`Sigma`. It counts what the games tell about everyone else through these two as well.

Each batch is drawn by one of these modes:

- connect: while the game graph has separate parts, a small batch joining the smallest to
  another, chosen at random. Before anyone has played, that's the initial random draws.
- near (most batches): the focal player is the one with the highest expected gain per second,
  given how its opponent will be drawn; the opponent is drawn from a normal distribution of
  ratings centred on the focal player's. Its variance includes both players' uncertainty, so
  it widens to nearly uniform where little is known. It's weighted by the inverse of the
  pair's cost, and down for pairs from the same training run, whose descendants beat their
  ancestors out of proportion to real strength (`same_run_weight`).
- far: as near, with the opponent drawn uniformly from the players further than two widths
  away, so that ratings can't drift along chains of near neighbours.
- external: as near, with an external engine as the focal player, so that engines keep tying
  the scale to something outside the league, however few games they need by the gain alone.

Trade-offs, all in `SchedulerConfig`: `width` trades information per game (narrow) against
ties across the scale (wide); `far_fraction` and `external_fraction` take that many batches
from the best use of the GPU by the model, for what the model can't see (drift, inflation).
"""

from dataclasses import dataclass

import numpy as np

from core.league.ratings import K, Ratings, expected_score
from core.league.store import Player


@dataclass(frozen=True)
class SchedulerConfig:
    """How the next batch is picked.

    Attributes:
        width: the standard deviation of the ratings opponents are drawn from, in Elo
        same_run_weight: how much less likely an opponent from the same training run is
        far_fraction: the fraction of batches with a far opponent
        external_fraction: the fraction of batches with an external engine
        batch_seconds: how long a batch should take at most
        batch_sizes: the sizes of batch to choose from, in games, each even: the search
            compiles for each, so the fewer the better
        max_tree_nodes: the most tree nodes (of both sides' searches) a batch may hold at
            once, for the memory they take
    """

    width: float = 150.0
    same_run_weight: float = 0.5
    far_fraction: float = 0.1
    external_fraction: float = 0.1
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
        mode: how it was drawn: "connect", "near", "far" or "external"
    """

    a: int
    b: int
    games: int
    mode: str


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


def gains(ratings: Ratings, games: np.ndarray) -> np.ndarray:
    """(n, n) what `games[i, j]` games between each pair would remove from the sum of the
    ratings' variances."""
    cov = ratings.cov
    squared = cov @ cov
    diagonal = np.diag(cov)
    diagonal_squared = np.diag(squared)
    # d' Sigma d and |Sigma d|^2, for d the difference between each pair
    var = diagonal[:, None] + diagonal[None, :] - 2 * cov
    norm = diagonal_squared[:, None] + diagonal_squared[None, :] - 2 * squared
    p = expected_score(ratings.elo[:, None] - ratings.elo[None, :])
    c = games * K**2 * p * (1 - p)
    return c * norm / (1 + c * var)


def opponent_weights(
    config: SchedulerConfig,
    ratings: Ratings,
    players: list[Player],
    costs: np.ndarray,
) -> np.ndarray:
    """(n, n) how likely each column is as the opponent of each row (each row sums to 1,
    or to 0 if it has no opponent): the normal distribution centred on the row's rating,
    weighted by the pair's cost and by whether it's within a training run."""
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
    return _normalized(weights / costs * _allowed(players))


def _allowed(players: list[Player]) -> np.ndarray:
    """(n, n) which pairs may play: not a player itself, a retired one, or two engines,
    whose games are all CPU and say nothing about the networks."""
    external = np.array([p.is_external for p in players])
    active = np.array([not p.retired for p in players])
    return (
        active[:, None]
        & active[None, :]
        & ~np.eye(len(players), dtype=bool)
        & ~(external[:, None] & external[None, :])
    )


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
    n = len(players)
    allowed = _allowed(players)
    if not allowed.any():
        return None
    ids = [p.id for p in players]

    def pairing(i: int, j: int, mode: str) -> Pairing:
        games = batch_size(config, players[i], players[j], seconds_per_game[i, j])
        return Pairing(ids[i], ids[j], games, mode)

    connect = _connect(ratings, allowed, rng)
    if connect is not None:
        # a few games are enough to join them, and to tell where to go from there
        i, j = connect
        return Pairing(ids[i], ids[j], min(config.batch_sizes), "connect")

    games = np.array(
        [
            [batch_size(config, players[i], players[j], seconds_per_game[i, j])
             for j in range(n)]
            for i in range(n)
        ],
        dtype=float,
    )  # fmt: skip
    per_second = gains(ratings, games) / (games * seconds_per_game)
    weights = opponent_weights(config, ratings, players, seconds_per_game)
    is_far = np.abs(ratings.elo[:, None] - ratings.elo[None, :]) > 2 * config.width
    far_weights = _normalized((is_far & allowed).astype(float))

    external = np.array([p.is_external for p in players])
    draw = rng.random()
    if draw < config.external_fraction and (external & allowed.any(axis=1)).any():
        mode, focal, opponents = "external", external, weights
    elif draw < config.external_fraction + config.far_fraction and far_weights.any():
        mode, focal, opponents = "far", ~external, far_weights
    else:
        mode, focal, opponents = "near", ~external, weights

    # what the focal player's next batch is worth on average, given how it'd be drawn
    worth = (opponents * per_second).sum(axis=1)
    can_play = opponents.sum(axis=1) > 0
    eligible = can_play & focal if (can_play & focal).any() else can_play
    if not eligible.any():
        return None
    i = int(np.argmax(np.where(eligible, worth, -np.inf)))
    j = int(rng.choice(n, p=opponents[i]))
    return pairing(i, j, mode)


def _connect(
    ratings: Ratings, allowed: np.ndarray, rng: np.random.Generator
) -> tuple[int, int] | None:
    """A random allowed pair joining the smallest part of the game graph to another; None
    if that isn't possible (everyone playing is connected already)."""
    components = ratings.components
    joining = allowed & (components[:, None] != components[None, :])
    if not joining.any():
        return None
    sizes = np.bincount(components)[components]
    can_join = joining.any(axis=1)
    smallest = sizes[can_join].min()
    choices = np.flatnonzero(can_join & (sizes == smallest))
    i = int(rng.choice(choices))
    j = int(rng.choice(np.flatnonzero(joining[i])))
    return i, j


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
    """How long a game takes, in seconds: a side's search costs `per_sim` per iteration, in
    proportion to its network's `size(net)`, on top of `per_game`; an external engine
    costs what the game says. Measured batches correct this, each player by the median of
    (seconds measured / seconds modelled) over its batches.

    Attributes:
        per_game: the seconds of a game that don't depend on either player
        per_sim: seconds of a game per MCTS iteration a move, of a network of size 1
    """

    per_game: float = 0.044
    per_sim: float = 2.66e-4


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
