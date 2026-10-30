"""Fits Elo ratings to every game played, in one batch: the most likely ratings under the
logistic model, BayesElo-style, with a weak prior that keeps them finite.

A draw counts as half a win and half a loss. The prior is `prior_sd` around the mean for each
rating, and `pair_draws` virtual draws between each pair that has played, so that a pair with
a perfect score ends up far apart rather than infinitely so.

Ratings only mean anything relative to each other. Uncertainty is therefore reported relative
to the mean of all the players (`Ratings.sd`) or between two of them (`Ratings.pair_sd`), and
not relative to the player at zero, whose own few games would otherwise inflate everyone's.
The scale's zero is the anchor, an engine's fixed configuration, once the anchor is connected
to the rest: a value that means the same when more players join. Without one it's the mean.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from core.league.store import PairRecord

# d(log odds)/d(Elo)
K = np.log(10) / 400


@dataclass(frozen=True)
class Ratings:
    """A fit of ratings.

    Attributes:
        ids: (n) the players' ids
        elo: (n) their ratings
        cov: (n, n) their ratings' covariance, relative to the players' mean
        components: (n) which connected part of the game graph each is in
        games: (n) how many games each has played
    """

    ids: list[int]
    elo: np.ndarray
    cov: np.ndarray
    components: np.ndarray
    games: np.ndarray

    @property
    def sd(self) -> np.ndarray:
        """Each rating's standard deviation, relative to the players' mean."""
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))

    def index(self, player: int) -> int:
        return self.ids.index(player)

    def pair_sd(self, a: int, b: int) -> float:
        """The standard deviation of the difference between two players' ratings."""
        i, j = self.index(a), self.index(b)
        var = self.cov[i, i] + self.cov[j, j] - 2 * self.cov[i, j]
        return float(np.sqrt(max(var, 0.0)))

    @property
    def connected(self) -> bool:
        return len(set(self.components.tolist())) <= 1


def expected_score(difference):
    """The expected score of a player rated `difference` above their opponent."""
    return 1 / (1 + np.exp(-K * np.asarray(difference)))


def connected_components(n: int, edges: Iterable[tuple[int, int]]) -> np.ndarray:
    """(n) a label for each of `n` nodes, the same within each connected part, the labels
    numbered in order of each part's first node."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    roots: dict[int, int] = {}
    return np.array(
        [roots.setdefault(find(i), len(roots)) for i in range(n)], dtype=int
    )


def fit_ratings(
    ids: Sequence[int],
    records: Iterable[PairRecord],
    anchor: int | None = None,
    prior_sd: float = 1000.0,
    pair_draws: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> Ratings:
    """The most likely ratings of the players `ids` given their games, by Newton's method.

    Args:
        ids: the players
        records: the games between them, of any pairs (a pair with an unknown player is left
            out)
        anchor: (optional) the player to rate 0, if it's connected to everyone else's main
            part; otherwise the mean rating is 0
        prior_sd: the prior's standard deviation of a rating around the mean
        pair_draws: virtual draws between each pair that has played
        max_iterations: Newton steps to take at most
        tolerance: stop once no rating moves by more than this many Elo

    Returns:
        the fit
    """
    ids = list(ids)
    n = len(ids)
    index = {player: i for i, player in enumerate(ids)}
    pairs = [
        (index[r.a], index[r.b], r.wins + r.draws / 2, r.wins + r.draws + r.losses)
        for r in records
        if r.a in index and r.b in index
    ]
    games = np.zeros(n, dtype=int)
    if pairs:
        i, j, score, num = (np.array(x) for x in zip(*pairs))
        i, j = i.astype(int), j.astype(int)
        np.add.at(games, i, num.astype(int))
        np.add.at(games, j, num.astype(int))
        score = score + pair_draws / 2
        num = num + pair_draws
    else:
        i = j = np.zeros(0, dtype=int)
        score = num = np.zeros(0)
    edges = [(a, b) for a, b, _, _ in pairs]
    components = connected_components(n, edges)

    prior = 1 / prior_sd**2
    elo = np.zeros(n)

    def hessian(elo: np.ndarray) -> np.ndarray:
        p = expected_score(elo[i] - elo[j])
        weight = K**2 * num * p * (1 - p)
        h = prior * np.eye(n)
        np.add.at(h, (i, i), weight)
        np.add.at(h, (j, j), weight)
        np.subtract.at(h, (i, j), weight)
        np.subtract.at(h, (j, i), weight)
        return h

    for _ in range(max_iterations):
        p = expected_score(elo[i] - elo[j])
        residual = K * (score - num * p)
        gradient = -prior * elo
        np.add.at(gradient, i, residual)
        np.subtract.at(gradient, j, residual)
        step = np.linalg.solve(hessian(elo), gradient)
        # a step past the curve's bend would overshoot, far from the optimum
        largest = np.abs(step).max(initial=0.0)
        elo = elo + step * min(1.0, 400.0 / max(largest, 1e-12))
        if largest < tolerance:
            break

    cov = np.linalg.inv(hessian(elo))
    # relative to the mean: differences between ratings are all the games can tell
    centre = np.eye(n) - np.ones((n, n)) / max(n, 1)
    cov = centre @ cov @ centre
    elo = elo - elo.mean() if n else elo
    if anchor in index:
        at = index[anchor]
        counts = np.bincount(components)
        if counts[components[at]] == counts.max():
            elo = elo - elo[at]
    return Ratings(ids, elo, cov, components, games)
