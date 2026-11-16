"""How healthy a network's self-play data is under a self-play config: how varied its games are,
how sharp and how good its targets are, and what it costs. One result per (network, config), for
the dashboard (`serve`) to compare.

A config is train.py's self-play flags, e.g. `--sims 511 --playout-cap 0.25:127 --solve-empties
20`: the search, exploration, playout cap, resignation and solve stop are built by train.py's own
code, so anything train.py's self-play takes is measured as it plays. Two passes each:

- probe: fixed positions of the Edax-labelled set (engine_set.py), PHASES' bands of empties,
  each searched afresh as a full search, the same search without root noise, and (with a playout
  cap) a fast search, beside the raw network: their policies' entropy and Edax regret, their
  values against Edax's predicted result. The same positions for every network and config.
- self-play: train.py's self-play (`Trainer.collect_steps`) of --games games, one an environment,
  from --openings fixed XOT openings, every move kept. Games stopped at the solve point are solved
  exactly. Diversity among the games from the same opening, the stored targets as training would
  see them, outcomes, and moves per second.

    uv run --extra cuda examples/othello/selfplay_health.py run --ckpt CKPT.eqx --net NET \\
        --config "511/127=--sims 511 --playout-cap 0.25:127 --solve-empties 20" \\
        --config "255/63=--sims 255 --playout-cap 0.25:63 --solve-empties 20"
    uv run examples/othello/selfplay_health.py serve --port 8061

Results are JSON files in --results (default ~/.cache/turbozero/othello/selfplay-health), one
per run, named by model and config; a run with the same names replaces its file.
"""

import os
import sys

# the dashboard needs no device, and JAX on the GPU would take memory self-play needs
if sys.argv[1:2] == ["serve"]:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import datetime
import hashlib
import http.server
import itertools
import json
import re
import shlex
import socket
import subprocess
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from engine_set import DEFAULT_PATH as DEFAULT_ENGINE_SET
from engine_set import ILLEGAL, EngineSet, boards
from engines import CACHE_DIR
from exact_labels import default_threads, stops_at
from game import (
    SYMMETRY_TRANSFORM_FNS,
    load_checkpoint,
    make_optimizer,
    metadata,
    position_states,
    state_to_nn_input,
    step_fn,
)
from label_positions import AnalystFactory, edax_analyst, label
from train import check_selfplay_args, make_exploration, make_parser, make_selfplay
from xot import load_xot, make_xot_init_fn

from core.memory.move_log import ENDED, GOES_ON, emptied
from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.utils import BATCH_AXIS, apply_nn
from core.training.endgame_solve import EndgameSolve
from core.training.loss_fns import az_default_loss_fn
from core.training.resignation import Resignation
from core.training.train import Trainer

RESULTS_DIR = CACHE_DIR / "selfplay-health"
PAGE = Path(__file__).with_name("selfplay_health.html")
# bands of empty squares metrics are reported for, all above the usual solve point (20); the
# opening band starts at an XOT opening's 52
PHASES = {"opening": (40, 52), "midgame": (30, 39), "late": (21, 29)}
# bands of empties the probe also reports, outside "all": positions at or below the usual solve
# point (20), which a search from a few more empties reaches as leaves
PROBE_ONLY = {"endgame": (13, 20)}
# the train.py arguments that make up a self-play config, recorded with each result
SELFPLAY_ARGS = (
    "sims",
    "playout_cap",
    "tree_factor",
    "traversal_steps",
    "count_reused_visits",
    "no_persist_tree",
    "min_iterations",
    "advance_every",
    "sampling_moves",
    "move_temperature",
    "store_until_offline",
    "dirichlet_alpha",
    "dirichlet_epsilon",
    "resign",
    "resign_calibration",
    "solve_empties",
    "solve_share",
    "solve_play_past",
    "policy_target",
    "value_scale",
    "maxvisit_init",
    "standard_starts",
    "inference_dtype",
)
# a search's value counts as decided past this (an expected score of 95% or 5%)
DECIDED = 0.9
# moves' compute per self-play call between reads of the move log
CHUNK_STEPS = 4
# stored positions per count of empty squares that pair distances are measured among, at most
PAIR_SAMPLE = 384
# Edax's labels of stored positions (--label-stored), by level, kept for every run to reuse
LABELS_DIR = RESULTS_DIR / "labels"
# a labelled position is close if Edax's best move's margin is at most this many discs: there an
# outcome is hard to get right whatever the labels' noise, so label quality is compared within
# close positions and within clear ones (lopsided positions are easy to label)
EDAX_CLOSE = 6
# --coverage: stored positions per count of empty squares a reference position is compared with, at most
# (drawn at random), so that a config storing more positions isn't favoured
COVERAGE_SAMPLE = 400
# --coverage: buckets of disc distance to the nearest stored position that the network's error is
# reported by: [0, 2], [3, 4], [5, 6], [7, 8], 9 or more
COVERAGE_BUCKETS = (0, 3, 5, 7, 9, 65)
# the scales of the soft collisions, exp(-distance / scale), in each distance's units: squares,
# and the reference network's cosine distance
SOFT_SCALES = {
    "disc": (2.0, 4.0, 8.0),
    "occupancy": (1.0, 2.0, 4.0),
    "feature": (0.03, 0.1, 0.3),
}


def parse_config(text: str, net: str) -> tuple[str, list[str], argparse.Namespace]:
    """A --config, NAME=FLAGS: its name, flags and train.py's arguments from them."""
    name, sep, flags = text.partition("=")
    if not (name and sep):
        raise argparse.ArgumentTypeError(f"{text!r} isn't NAME=FLAGS")
    parser = make_parser()
    tokens = shlex.split(flags)
    args = parser.parse_args([*tokens, "--net", net])
    if args.sims_schedule is not None:
        parser.error("a config searches a fixed --sims, not --sims-schedule")
    check_selfplay_args(parser, args)
    return name, tokens, args


def config_record(args: argparse.Namespace) -> dict[str, Any]:
    """The self-play arguments of a config, as JSON."""
    record = {}
    for key in SELFPLAY_ARGS:
        value = getattr(args, key)
        record[key] = value.get_config() if hasattr(value, "get_config") else value
    return record


def variant(args: argparse.Namespace, **changes: Any) -> argparse.Namespace:
    """`args` with some arguments changed."""
    return argparse.Namespace(**{**vars(args), **changes})


def entropy(p: np.ndarray) -> np.ndarray:
    """Each row's entropy, in nats."""
    p = np.asarray(p, dtype=np.float64)
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0.0), axis=-1)


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Each row's KL(p || q), in nats."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    terms = np.where(
        p > 0, p * (np.log(np.where(p > 0, p, 1)) - np.log(np.maximum(q, 1e-12))), 0
    )
    return terms.sum(-1)


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation, None when either side is constant or empty."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def cluster_sums(
    ids: np.ndarray, columns: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Per-unit sums of `columns`, and the units' sizes (`n`) and ids, for `bootstrap`."""
    unique, inverse = np.unique(np.asarray(ids, np.uint64), return_inverse=True)
    sums = {
        name: np.bincount(inverse, weights=values, minlength=len(unique))
        for name, values in columns.items()
    }
    return {"ids": unique, "n": np.bincount(inverse, minlength=len(unique)), **sums}


def statistic(sums: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    """A metric from its units' sums, weighted by unit: a mean of `x`, or with `xy` the Pearson
    correlation of x and y. `weights` is (units,) or (resamples, units)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        n = weights @ sums["n"]
        mean_x = (weights @ sums["x"]) / n
        if "xy" not in sums:
            return mean_x
        mean_y = (weights @ sums["y"]) / n
        cov = (weights @ sums["xy"]) / n - mean_x * mean_y
        var_x = (weights @ sums["xx"]) / n - mean_x**2
        var_y = (weights @ sums["yy"]) / n - mean_y**2
        return cov / np.sqrt(var_x * var_y)


# bootstrap resamples behind each metric's standard error
BOOTSTRAP = 1000
_unit_weights: dict[int, np.ndarray] = {}


def unit_weights(ids: np.ndarray) -> np.ndarray:
    """(BOOTSTRAP, units) Poisson(1) bootstrap weights, drawn from each unit's id: a unit gets the
    same weights in every run, so the bootstrap of a difference between runs is paired."""
    for i in ids:
        i = int(i)
        if i not in _unit_weights:
            seed = [7, i >> 32, i & 0xFFFFFFFF]
            _unit_weights[i] = np.random.default_rng(seed).poisson(1.0, BOOTSTRAP)
    return np.stack([_unit_weights[int(i)] for i in ids], axis=1).astype(np.float64)


class Stats:
    """The per-unit inputs of a run's metrics, so that each metric gets a standard error, and two
    runs a paired difference (see `compare`). A unit is what the bootstrap resamples whole: a probe
    position, or an XOT opening with all its games. Its id means the same in every run (the
    position's index in the engine set, the opening position's key)."""

    def __init__(self) -> None:
        self.inputs: dict[str, dict[str, np.ndarray]] = {}

    def mean(self, path: str, ids: np.ndarray, x: np.ndarray) -> float | None:
        """The mean of `x`, one value per row, each row in unit `ids`, recorded as `path`."""
        x = np.asarray(x, np.float64)
        if not len(x):
            return None
        self.inputs[path] = cluster_sums(ids, {"x": x})
        return float(x.mean())

    def corr(
        self, path: str, ids: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> float | None:
        """The Pearson correlation of `x` and `y`, recorded as `path`."""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        r = pearson(x, y)
        if r is None:
            return None
        columns = {"x": x, "y": y, "xx": x * x, "yy": y * y, "xy": x * y}
        self.inputs[path] = cluster_sums(ids, columns)
        return r

    def value(
        self, prefix: str, ids: np.ndarray, value: np.ndarray, z: np.ndarray, name=""
    ) -> dict[str, Any]:
        """A value's agreement with results z (-1, 0, 1): MSE, Pearson r, and sign agreement
        where z isn't 0, recorded as `prefix`.<name>value_mse etc."""
        decided = z != 0
        key = f"{name}value"
        return {
            f"{key}_mse": self.mean(f"{prefix}.{key}_mse", ids, (value - z) ** 2),
            f"{key}_r": self.corr(f"{prefix}.{key}_r", ids, value, z),
            f"{key}_sign": self.mean(
                f"{prefix}.{key}_sign",
                ids[decided],
                np.sign(value[decided]) == z[decided],
            ),
        }

    def standard_errors(self) -> dict[str, float]:
        """Each metric's bootstrap standard error."""
        errors = {}
        for path, sums in self.inputs.items():
            draws = statistic(sums, unit_weights(sums["ids"]))
            errors[path] = float(np.nanstd(draws))
        return errors

    def save(self, path: str | Path) -> None:
        arrays: dict[str, Any] = {
            f"{p}|{k}": v for p, sums in self.inputs.items() for k, v in sums.items()
        }
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Stats":
        stats = cls()
        with np.load(path) as f:
            for name in f.files:
                metric, field = name.rsplit("|", 1)
                stats.inputs.setdefault(metric, {})[field] = f[name]
        return stats


def compare(base: Stats, other: Stats) -> dict[str, tuple[float, float]]:
    """Each metric both runs have: `other`'s minus `base`'s, and its paired bootstrap standard
    error (units resampled alike in both runs)."""
    out = {}
    for path in base.inputs.keys() & other.inputs.keys():
        a, b = base.inputs[path], other.inputs[path]
        diff = float(
            statistic(b, np.ones(len(b["ids"]))) - statistic(a, np.ones(len(a["ids"])))
        )
        draws = statistic(b, unit_weights(b["ids"])) - statistic(
            a, unit_weights(a["ids"])
        )
        if np.isfinite(diff):
            out[path] = (diff, float(np.nanstd(draws)))
    return out


def bands(empties: np.ndarray) -> dict[str, np.ndarray]:
    """Which positions are in each of PHASES, and in any of them ("all")."""
    groups = {
        phase: (empties >= low) & (empties <= high)
        for phase, (low, high) in PHASES.items()
    }
    groups["all"] = np.any(list(groups.values()), axis=0)
    return groups


def raw_outputs(
    network: Any, params: Any, observations: np.ndarray, batch: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """The network's raw logits and values for positions given as its input."""
    nn_params, nn_state = params
    static = eqx.filter(network, eqx.is_inexact_array, inverse=True)

    @jax.jit
    def forward(x):
        net = eqx.nn.inference_mode(eqx.combine(nn_params, static))
        logits, value = apply_nn(net, nn_state, x)[0]
        return logits.astype(jnp.float32), value.astype(jnp.float32).reshape(-1)

    logits, values = [], []
    for start in range(0, len(observations), batch):
        chunk = observations[start : start + batch]
        padded = np.concatenate([chunk, np.repeat(chunk[:1], batch - len(chunk), 0)])
        out_logits, out_value = forward(jnp.asarray(padded))
        logits.append(np.asarray(out_logits)[: len(chunk)])
        values.append(np.asarray(out_value)[: len(chunk)])
    return np.concatenate(logits), np.concatenate(values)


def masked_softmax(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """The policy over legal actions, from logits."""
    logits = np.where(legal, logits.astype(np.float64), -np.inf)
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(-1, keepdims=True)


def search_positions(
    evaluator: Any, params: Any, states: Any, key: jax.Array, batch: int
) -> tuple[np.ndarray, np.ndarray]:
    """One fresh search of each position by `evaluator`: its policy target and root value."""

    @jax.jit
    def run(states, keys):
        def one(state, k):
            tree = evaluator.init(template_embedding=state)
            out = evaluator.evaluate(k, tree, state, metadata(state), params, step_fn)
            return out.policy_weights, evaluator.get_value(out.eval_state)

        return jax.vmap(one)(states, keys)

    n = len(states.current_player)
    keys = jax.random.split(key, n)
    policies, values = [], []
    for start in range(0, n, batch):
        idx = np.arange(start, start + batch) % n
        chunk = jax.tree.map(lambda x, idx=idx: x[idx], states)
        policy, value = run(chunk, keys[idx])
        count = min(batch, n - start)
        policies.append(np.asarray(policy, np.float32)[:count])
        values.append(np.asarray(value, np.float32)[:count])
    return np.concatenate(policies), np.concatenate(values)


def probe_set(engine_set: EngineSet, per_phase: int) -> np.ndarray:
    """The probe positions: up to `per_phase` of each of PHASES and PROBE_ONLY, the same for
    every run (PHASES' drawn first, as before PROBE_ONLY was added)."""
    rng = np.random.default_rng(0)
    chosen = []
    for low, high in [*PHASES.values(), *PROBE_ONLY.values()]:
        idx = np.flatnonzero((engine_set.empties >= low) & (engine_set.empties <= high))
        chosen.append(np.sort(rng.permutation(idx)[:per_phase]))
    return np.concatenate(chosen)


def probe(
    network: Any,
    params: Any,
    args: argparse.Namespace,
    engine_set: EngineSet,
    per_phase: int,
    seed: int,
    batch: int,
) -> tuple[dict[str, Any], Stats]:
    """The probe pass (see the module's docstring): its metrics, and their inputs."""
    idx = probe_set(engine_set, per_phase)
    observations = engine_set.obs[idx].astype(bool)
    legal, scores = engine_set.legal[idx], engine_set.scores[idx].astype(np.float64)
    empties = engine_set.empties[idx].astype(int)
    best = np.max(np.where(legal, scores, -np.inf), axis=1)
    loss = np.where(legal, best[:, None] - scores, 0.0)
    z = np.sign(best)
    groups = bands(empties)
    for name, (low, high) in PROBE_ONLY.items():
        groups[name] = (empties >= low) & (empties <= high)

    logits, raw_value = raw_outputs(network, params, observations)
    outputs = {"raw": (masked_softmax(logits, legal), raw_value)}
    states = position_states(observations)
    searches = {
        "full": variant(args, playout_cap=None),
        "full_no_noise": variant(args, playout_cap=None, dirichlet_epsilon=0.0),
    }
    if args.playout_cap is not None:
        searches["fast"] = variant(
            args,
            sims=args.playout_cap.fast_iterations,
            playout_cap=None,
            dirichlet_epsilon=0.0,
            # a fast search ends at its own budget whatever the minimum (see MCTS.search_done)
            min_iterations=min(args.min_iterations, args.playout_cap.fast_iterations),
        )
    key = jax.random.PRNGKey(seed)
    for i, (name, search_args) in enumerate(searches.items()):
        evaluator = make_selfplay(network, search_args).at(0)
        outputs[name] = search_positions(
            evaluator, params, states, jax.random.fold_in(key, i), batch
        )

    stats = Stats()
    metrics: dict[str, Any] = {}
    for name, (policy, value) in outputs.items():
        top = np.argmax(np.where(legal, policy, -1), axis=1)
        per_position = {
            "entropy": entropy(policy),
            "exp_regret": np.sum(policy * loss, axis=1),
            "top1": np.take_along_axis(loss, top[:, None], 1)[:, 0] == 0,
        }
        metrics[name] = {}
        for phase, sel in groups.items():
            prefix = f"probe.metrics.{name}.{phase}"
            row = {
                k: stats.mean(f"{prefix}.{k}", idx[sel], v[sel])
                for k, v in per_position.items()
            }
            row["effective_moves"] = (
                float(np.exp(row["entropy"])) if row["entropy"] is not None else None
            )
            row.update(stats.value(prefix, idx[sel], value[sel], z[sel]))
            metrics[name][phase] = row
    full, quiet = outputs["full"][0], outputs["full_no_noise"][0]
    flips = np.argmax(full, 1) != np.argmax(quiet, 1)
    result = {
        "positions": {phase: int(sel.sum()) for phase, sel in groups.items()},
        "metrics": metrics,
        "noise_flip": {
            phase: stats.mean(f"probe.noise_flip.{phase}", idx[sel], flips[sel])
            for phase, sel in groups.items()
        },
    }
    return result, stats


def symmetry_keys(observations: np.ndarray) -> np.ndarray:
    """A 64-bit key of each position, the same for all 8 of its symmetries (and the player to
    move's discs told from the opponent's)."""
    obs = np.asarray(observations, dtype=bool)
    mix = np.uint64(0x9E3779B97F4A7C15)
    keys = None
    for transpose in (False, True):
        board = np.swapaxes(obs, 1, 2) if transpose else obs
        for turns in range(4):
            turned = np.rot90(board, turns, axes=(1, 2)).reshape(len(obs), 64, 2)
            me = np.packbits(turned[..., 0], axis=1, bitorder="little").view("<u8")[
                :, 0
            ]
            opp = np.packbits(turned[..., 1], axis=1, bitorder="little").view("<u8")[
                :, 0
            ]
            with np.errstate(over="ignore"):
                key = me ^ (opp * mix) ^ (opp >> np.uint64(29))
            keys = key if keys is None else np.minimum(keys, key)
    assert keys is not None
    return keys


def symmetric_bitboards(observations: np.ndarray) -> np.ndarray:
    """Each position's 8 symmetries as bitboards: (8, n, 2), the player to move's discs, then the
    opponent's."""
    obs = np.asarray(observations, dtype=bool)
    out = []
    for transpose in (False, True):
        board = np.swapaxes(obs, 1, 2) if transpose else obs
        for turns in range(4):
            turned = np.rot90(board, turns, axes=(1, 2)).reshape(len(obs), 64, 2)
            packed = np.packbits(turned, axis=1, bitorder="little")
            out.append(np.stack([packed[:, :8, 0], packed[:, :8, 1]], -1).copy())
    # (8, n, 8 bytes, 2) -> (8, n, 2) uint64
    return np.stack(out).transpose(0, 1, 3, 2).copy().view("<u8")[..., 0]


def symmetric_observations(observations: np.ndarray) -> np.ndarray:
    """Each position's 8 symmetries as the network's input: (8, n, 8, 8, 2), in the order of
    `symmetric_bitboards`."""
    obs = np.asarray(observations, dtype=bool)
    out = []
    for transpose in (False, True):
        board = np.swapaxes(obs, 1, 2) if transpose else obs
        for turns in range(4):
            out.append(np.rot90(board, turns, axes=(1, 2)))
    return np.stack(out)


# positions' feature maps, (n, channels, *board), equivariant: a symmetry of a position has the
# same symmetry of its map (see `reference_features`)
FeatureFn = Callable[[np.ndarray], np.ndarray]


def symmetric_maps(maps: np.ndarray) -> np.ndarray:
    """Feature maps (n, channels, *board) in each of the 8 symmetries, flattened and of unit
    length: (8, n, features), in the order of `symmetric_bitboards`."""
    out = []
    for transpose in (False, True):
        board = np.swapaxes(maps, -2, -1) if transpose else maps
        for turns in range(4):
            out.append(np.rot90(board, turns, axes=(-2, -1)).reshape(len(maps), -1))
    flat = np.stack(out).astype(np.float32)
    return flat / np.maximum(np.linalg.norm(flat, axis=-1, keepdims=True), 1e-12)


def pair_statistics(
    observations: np.ndarray,
    empties: np.ndarray,
    rng: np.random.Generator,
    groups: np.ndarray | None = None,
    features: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """How far apart positions are, where nearly all are distinct: each of up to PAIR_SAMPLE
    positions per count of empty squares (drawn at random), its mean distance to the others drawn
    with as many empties, by each of these distances, each in the closest of the other's 8
    symmetries, which training treats as one position:

    - disc (`pair_distance`): the squares whose contents differ (empty, the player to move's,
      the opponent's)
    - occupancy (`occupancy_distance`): the squares filled in one and empty in the other, the
      shape of the filled region whatever its colours, which change every move
    - feature (`feature_distance`, with `features`, the positions' equivariant feature maps, see
      `reference_features`): the cosine distance between a reference network's trunk features,
      positions as that network sees them

    And for each, soft collisions `soft_collision_<distance>_<scale>`: the mean of
    exp(-distance / scale) for the scales of SOFT_SCALES, the chance two positions are the same
    with near-misses counting in part (at a small scale, the exact collision). Means over pairs,
    so none depends on how many positions there are. With `groups` (e.g. each position's
    opening), only positions of the same group are compared.

    Returns:
        the rows drawn, and each statistic's value for each of them
    """
    rows: list[np.ndarray] = []
    values: dict[str, list[np.ndarray]] = {}
    keys = np.asarray(empties, np.int64)
    if groups is not None:
        _, group = np.unique(groups, return_inverse=True)
        keys = keys * (int(group.max()) + 1) + group
    for key in np.unique(keys):
        level = np.flatnonzero(keys == key)
        if len(level) < 2:
            continue
        if len(level) > PAIR_SAMPLE:
            level = np.sort(rng.choice(level, PAIR_SAMPLE, replace=False))
        boards = symmetric_bitboards(observations[level])
        me, opp = boards[0, :, 0], boards[0, :, 1]
        # (8, m, m): row i against each symmetry of row j
        sym_me, sym_opp = boards[:, None, :, 0], boards[:, None, :, 1]
        distances = {
            "disc": (me[None, :, None] ^ sym_me) | (opp[None, :, None] ^ sym_opp),
            "occupancy": (me | opp)[None, :, None] ^ (sym_me | sym_opp),
        }
        distances = {
            name: np.bitwise_count(bits).min(0).astype(float)
            for name, bits in distances.items()
        }
        if features is not None:
            f = symmetric_maps(features[level])
            distances["feature"] = np.maximum(
                1.0 - f[0][None] @ f.transpose(0, 2, 1), 0.0
            ).min(0)
        off = ~np.eye(len(level), dtype=bool)
        rows.append(level)
        for name, distance in distances.items():
            stat = {"disc": "pair_distance"}.get(name, f"{name}_distance")
            values.setdefault(stat, []).append(
                np.where(off, distance, 0.0).sum(1) / (len(level) - 1)
            )
            for scale in SOFT_SCALES[name]:
                kernel = np.where(off, np.exp(-distance / scale), 0.0)
                values.setdefault(f"soft_collision_{name}_{scale:g}", []).append(
                    kernel.sum(1) / (len(level) - 1)
                )
    if not rows:
        return np.zeros(0, int), {}
    return np.concatenate(rows), {k: np.concatenate(v) for k, v in values.items()}


def reference_features(network: Any, params: Any, batch: int = 4096) -> FeatureFn:
    """A reference network's trunk feature maps of positions (see `pair_statistics`), averaged
    over the 8 symmetries, each turned back: exactly equivariant, so a position's symmetries'
    maps are its own map's symmetries, whether or not the network is."""
    nn_params, nn_state = params
    static = eqx.filter(network, eqx.is_inexact_array, inverse=True)

    @jax.jit
    def forward(x):
        net = eqx.nn.inference_mode(eqx.combine(nn_params, static))
        trunk = jax.vmap(net.trunk, axis_name=BATCH_AXIS, in_axes=(0, None))
        return trunk(x, nn_state)[0].astype(jnp.float32)

    def maps(observations: np.ndarray) -> np.ndarray:
        out = []
        for start in range(0, len(observations), batch):
            chunk = observations[start : start + batch]
            padded = np.concatenate(
                [chunk, np.repeat(chunk[:1], batch - len(chunk), 0)]
            )
            total = None
            for k, sym in enumerate(symmetric_observations(padded)):
                turned = np.asarray(forward(jnp.asarray(sym)))[: len(chunk)]
                # undo the symmetry: turns back, then the transpose
                back = np.rot90(turned, -(k % 4), axes=(-2, -1))
                back = np.swapaxes(back, -2, -1) if k >= 4 else back
                total = back if total is None else total + back
            assert total is not None
            out.append(total / 8)
        return np.concatenate(out)

    return maps


def bitboards(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The player to move's and the opponent's discs, as bitboards."""
    flat = np.asarray(observations, dtype=bool).reshape(len(observations), 64, 2)
    me = np.packbits(flat[..., 0], axis=1, bitorder="little").view("<u8")[:, 0]
    opp = np.packbits(flat[..., 1], axis=1, bitorder="little").view("<u8")[:, 0]
    return me, opp


# Edax's scores of every legal move of positions given as the network's input, (n, 65) int16,
# ILLEGAL where illegal, from the player to move's side
LabelFn = Callable[[np.ndarray], np.ndarray]


class LabelCache:
    """Edax's scores of positions' moves (see `label_positions.label`), kept in `directory` as
    edax-L<level>.npz by position, so a position is scored once whichever run stores it (exactly,
    not up to symmetry: the scores are per move)."""

    def __init__(
        self,
        directory: str | Path,
        level: int,
        workers: int,
        factory: AnalystFactory = edax_analyst,
    ):
        self.path = Path(directory) / f"edax-L{level}.npz"
        self.level, self.workers, self.factory = level, workers, factory
        self.scores: dict[tuple[int, int], np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as saved:
                for me, opp, row in zip(saved["me"], saved["opp"], saved["scores"]):
                    self.scores[(int(me), int(opp))] = row

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        me, opp = bitboards(observations)
        keys = list(zip(me.tolist(), opp.tolist()))
        missing = sorted({k for k in keys if k not in self.scores})
        if missing:
            index = {k: i for i, k in enumerate(keys)}
            rows = observations[[index[k] for k in missing]]
            scores, _ = label(boards(rows), self.level, self.workers, self.factory)
            self.scores.update(zip(missing, scores))
            self.save()
        return np.stack([self.scores[k] for k in keys])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.scores), dtype=np.uint64).reshape(-1, 2)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(
            tmp,
            me=keys[:, 0],
            opp=keys[:, 1],
            scores=np.stack(list(self.scores.values())).astype(np.int16),
        )
        tmp.replace(self.path)


def label_metrics(
    scores: np.ndarray,
    policy: np.ndarray,
    prior: np.ndarray,
) -> dict[str, np.ndarray]:
    """How good stored targets are on the positions stored, by Edax's scores of their moves
    (`LabelFn`'s): each row's expected regret in discs (the best move's score less the move's)
    of the policy target and of the network's prior, and whether the target's top move is a best
    one."""
    legal = scores != ILLEGAL
    s = scores.astype(np.float64)
    best = np.max(np.where(legal, s, -np.inf), axis=1)
    loss = np.where(legal, best[:, None] - s, 0.0)
    top = np.argmax(np.where(legal, policy, -1.0), axis=1)
    return {
        "edax_target_regret": np.sum(policy * loss, axis=1),
        "edax_prior_regret": np.sum(prior * loss, axis=1),
        "edax_target_top1": np.take_along_axis(loss, top[:, None], 1)[:, 0] == 0,
    }


def nearest_stored(
    targets: np.ndarray,
    target_empties: np.ndarray,
    stored: np.ndarray,
    stored_empties: np.ndarray,
    rng: np.random.Generator,
    features: FeatureFn | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """How close a config's stored positions come to reference positions (e.g. from games against
    strong opponents): for each reference position, the distance to its nearest stored position with
    as many empties, in squares (the closest of the stored one's 8 symmetries, as `pair_statistics`)
    and, with `features`, in the reference network's cosine distance. Up to COVERAGE_SAMPLE stored
    positions per count of empties are compared, drawn at random, whatever the config stores.

    Returns:
        disc and feature distances, (n,) each; NaN where no position with as many empties is stored
    """
    disc = np.full(len(targets), np.nan)
    feature = np.full(len(targets), np.nan)
    for count in np.unique(target_empties):
        rows = np.flatnonzero(target_empties == count)
        pool = np.flatnonzero(stored_empties == count)
        if not len(pool):
            continue
        if len(pool) > COVERAGE_SAMPLE:
            pool = rng.choice(pool, COVERAGE_SAMPLE, replace=False)
        mine = symmetric_bitboards(targets[rows])[0]
        theirs = symmetric_bitboards(stored[pool])
        # (8, rows, pool): each reference position against each symmetry of each stored one
        differ = (mine[None, :, None, 0] ^ theirs[:, None, :, 0]) | (
            mine[None, :, None, 1] ^ theirs[:, None, :, 1]
        )
        disc[rows] = np.bitwise_count(differ).min(0).min(1)
        if features is not None:
            a = symmetric_maps(features(targets[rows]))[0]
            b = symmetric_maps(features(stored[pool]))
            cosine = a[None] @ b.transpose(0, 2, 1)
            feature[rows] = np.maximum(1.0 - cosine, 0.0).min(0).min(1)
    return disc, feature


def coverage_metrics(
    targets: dict[str, np.ndarray],
    stored: np.ndarray,
    stored_empties: np.ndarray,
    rng: np.random.Generator,
    raw: tuple[np.ndarray, np.ndarray],
    features: FeatureFn | None = None,
) -> dict[str, Any]:
    """`nearest_stored` for each reference set and phase (PHASES, by the reference positions'
    empties), with the network's raw accuracy on those positions by Edax's scores of their moves
    (value against the sign of the best score, the policy's expected regret and top-1), also within
    close and clear positions (EDAX_CLOSE) and by disc distance to the nearest stored position
    (COVERAGE_BUCKETS): where the data reaches the positions those games visit, and what being
    far from it costs the network.

    Args:
        targets: the reference positions: obs (n, 8, 8, 2), set (n,) names, empties (n,), scores
            (n, 65) Edax's scores of their moves (ILLEGAL where illegal)
        stored: the config's stored positions (m, 8, 8, 2), and their empties
        raw: the network's raw policy (n, 65) and value (n,) on the reference positions
    """
    obs, names, empties, scores = (
        targets["obs"],
        targets["set"],
        targets["empties"],
        targets["scores"],
    )
    disc, feature = nearest_stored(obs, empties, stored, stored_empties, rng, features)
    policy, value = raw
    legal = scores != ILLEGAL
    s = scores.astype(np.float64)
    best = np.max(np.where(legal, s, -np.inf), axis=1)
    loss = np.where(legal, best[:, None] - s, 0.0)
    top = np.argmax(np.where(legal, policy, -1.0), axis=1)
    per_row = {
        "value_error": (value - np.sign(best)) ** 2,
        "exp_regret": np.sum(policy * loss, axis=1),
        "top1": np.take_along_axis(loss, top[:, None], 1)[:, 0] == 0,
    }
    close = np.abs(best) <= EDAX_CLOSE

    def summary(sel: np.ndarray) -> dict[str, Any]:
        sel = sel & ~np.isnan(disc)
        if not sel.any():
            return {"positions": 0}
        out: dict[str, Any] = {
            "positions": int(sel.sum()),
            "nearest_disc": float(disc[sel].mean()),
            "within_2": float((disc[sel] <= 2).mean()),
            "within_4": float((disc[sel] <= 4).mean()),
            "close_share": float(close[sel].mean()),
            **{k: float(np.mean(v[sel])) for k, v in per_row.items()},
        }
        if features is not None:
            out["nearest_feature"] = float(np.nanmean(feature[sel]))
        for part, mask in (("close", close), ("clear", ~close)):
            both = sel & mask
            if both.any():
                out[f"nearest_disc_{part}"] = float(disc[both].mean())
                out[f"value_error_{part}"] = float(per_row["value_error"][both].mean())
        buckets = {}
        for low, high in itertools.pairwise(COVERAGE_BUCKETS):
            both = sel & (disc >= low) & (disc < high)
            buckets[f"{low}-{high - 1}"] = {
                "positions": int(both.sum()),
                "value_error": float(per_row["value_error"][both].mean())
                if both.any()
                else None,
                "exp_regret": float(per_row["exp_regret"][both].mean())
                if both.any()
                else None,
            }
        out["by_distance"] = buckets
        return out

    result: dict[str, Any] = {}
    for name in sorted(set(names.tolist())):
        mine = names == name
        result[name] = {"all": summary(mine)}
        for phase, (low, high) in PHASES.items():
            result[name][phase] = summary(mine & (empties >= low) & (empties <= high))
    return result


def play_games(
    network: Any,
    params: Any,
    args: argparse.Namespace,
    games: int,
    envs: int,
    openings: int,
    seed: int,
    max_calls: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Self-play as train.py's, `games` games from `openings` XOT openings, `envs` at a time (a
    batch plays until every environment has finished its first game): every move of those games
    (the move log's fields, plus `env`, the game's number), and the timings."""
    if games % envs:
        raise ValueError(f"games ({games}) must be a multiple of envs ({envs})")
    chosen = np.random.default_rng(0).choice(len(load_xot()), openings, replace=False)
    evaluator = make_selfplay(network, args).at(0)
    endgame_solve = None
    if args.solve_empties is not None:

        def unsolved(_: Any) -> np.ndarray:
            raise AssertionError("selfplay_health solves the games itself")

        endgame_solve = EndgameSolve(
            stop_fn=stops_at(args.solve_empties),
            solve_fn=unsolved,
            # solves are never labelled here, so a game's all stay pending: room for an
            # environment's first game's (its solve point and each position stored past it), as a
            # game whose solve finds every slot full would play on unsolved
            slots=args.solve_empties + 4,
            share=args.solve_share,
            play_past=args.solve_play_past,
        )
    trainer = Trainer(
        batch_size=envs,
        train_batch_size=1,
        warmup_steps=0,
        collection_steps_per_epoch=CHUNK_STEPS,
        train_steps_per_epoch=0,
        advance_every=args.advance_every,
        nn=network,
        nn_state=params[1],
        loss_fn=az_default_loss_fn,
        optimizer=make_optimizer(1),
        evaluator=evaluator,
        # room for every move of an environment's first game (and its next), none trained on
        memory_buffer=EpisodeReplayBuffer(capacity=192),
        max_episode_steps=80,
        env_step_fn=step_fn,
        env_init_fn=make_xot_init_fn(load_xot()[chosen], 0.0),
        state_to_nn_input_fn=state_to_nn_input,
        testers=[],
        data_transform_fns=SYMMETRY_TRANSFORM_FNS,
        exploration=make_exploration(args),
        resignation=Resignation(args.resign, args.resign_calibration)
        if args.resign is not None
        else None,
        endgame_solve=endgame_solve,
        ckpt_dir="/tmp/turbozero-selfplay-health",
    )
    pieces: list[dict[str, np.ndarray]] = []
    first_seconds, timed_seconds, timed_moves, calls = 0.0, 0.0, 0, 0
    for batch in range(games // envs):
        key = jax.random.fold_in(jax.random.PRNGKey(seed), batch)
        init_key, key = jax.random.split(key)
        state = trainer.init_collection_state(init_key, envs)
        state = replace(state, move_log=trainer.init_move_log())
        finished = np.zeros(envs, bool)
        batch_calls = 0
        while not finished.all():
            if batch_calls == max_calls:
                raise RuntimeError(
                    f"{int((~finished).sum())} of {envs} games unfinished after "
                    f"{batch_calls} calls"
                )
            start = time.perf_counter()
            keys = jax.random.split(jax.random.fold_in(key, batch_calls), envs)
            state = trainer.collect_steps(keys, state, params, CHUNK_STEPS)
            assert state.move_log is not None
            rows, count = jax.device_get((state.move_log.rows, state.move_log.count))
            seconds = time.perf_counter() - start
            # the first call compiles
            if calls == 0:
                first_seconds = seconds
            else:
                timed_seconds += seconds
                timed_moves += int(count.sum())
            calls += 1
            batch_calls += 1
            state = replace(state, move_log=emptied(state.move_log))
            kept = np.arange(rows.end.shape[1])[None, :] < count[:, None]
            env = np.broadcast_to(np.arange(envs)[:, None], kept.shape)[kept]
            piece = {
                f: np.asarray(getattr(rows, f))[kept] for f in rows.__dataclass_fields__
            }
            piece["env"] = env + batch * envs
            first = piece["game"] == 0
            pieces.append({k: v[first] for k, v in piece.items()})
            finished[env[first & (piece["end"] != GOES_ON)]] = True
        del state
    moves = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    # each environment's moves in the order played
    order = np.lexsort((np.arange(len(moves["env"])), moves["env"]))
    moves = {k: v[order] for k, v in moves.items()}
    timing = {
        "first_call_seconds": first_seconds,
        "timed_seconds": timed_seconds,
        "timed_moves": timed_moves,
        "calls": calls,
        "chunk_steps": CHUNK_STEPS,
    }
    return moves, timing


def game_outcomes(
    moves: dict[str, np.ndarray], args: argparse.Namespace, threads: int
) -> tuple[np.ndarray, float]:
    """Each row's exact result for its player to move (the game's solve point's, or its played
    result without --solve-empties), and the seconds spent solving. Rows of a game that ended
    otherwise (truncated) get NaN."""
    env, player = moves["env"], moves["player"]
    empties = 64 - moves["observation"].reshape(len(env), -1).sum(-1)
    starts = np.flatnonzero(np.r_[True, env[1:] != env[:-1]])
    ends = np.r_[starts[1:], len(env)]
    z = np.full(len(env), np.nan)
    point_rows = []
    for start, end in zip(starts, ends):
        if args.solve_empties is not None:
            at = np.flatnonzero(empties[start:end] <= args.solve_empties)
            if len(at):
                point_rows.append(start + at[0])
                continue
        last = end - 1
        if moves["end"][last] == ENDED:
            z[start:end] = moves["result"][last] * np.where(
                player[start:end] == player[last], 1, -1
            )
    seconds = 0.0
    if point_rows:
        import endgame

        rows = np.array(point_rows)
        me, opp = bitboards(moves["observation"][rows])
        start_time = time.perf_counter()
        margins, _ = endgame.solve_batch(me, opp, threads=threads, exact=False)
        seconds = time.perf_counter() - start_time
        for row, result in zip(rows, np.sign(margins)):
            game = np.searchsorted(starts, row, side="right") - 1
            s, e = starts[game], ends[game]
            z[s:e] = result * np.where(player[s:e] == player[row], 1, -1)
    return z, seconds


def novelty(
    stats: Stats,
    keys: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    opening: np.ndarray,
    sample: int,
    sampled_moves: int,
    max_offset: int = 48,
) -> dict[str, Any]:
    """Diversity among the games from the same opening: at each move since the opening, the
    distinct positions among `sample` games of an opening alive then, as a share of them (and its
    mean over the first `sampled_moves` moves); and how many moves two such games stay the same.
    Openings with fewer games are left out."""
    by_opening: dict[int, list[np.ndarray]] = {}
    for s, e, o in zip(starts, ends, opening):
        by_opening.setdefault(int(o), []).append(keys[s:e])
    groups = {o: g[:sample] for o, g in by_opening.items() if len(g) >= sample}
    ids = np.array(list(groups), np.uint64)
    shares = np.full((len(groups), max_offset), np.nan)
    prefixes, prefix_ids = [], []
    for i, (o, group) in enumerate(groups.items()):
        length = max(len(g) for g in group)
        grid = np.zeros((len(group), length), np.uint64)
        alive = np.zeros((len(group), length), bool)
        for j, g in enumerate(group):
            grid[j, : len(g)], alive[j, : len(g)] = g, True
        for t in range(1, min(length, max_offset)):
            live = grid[alive[:, t], t]
            if len(live) >= 2:
                shares[i, t] = len(np.unique(live)) / len(live)
        # moves two games agree for, from the opening on: rows equal up to the first difference
        same = (
            (grid[:, None, :] == grid[None, :, :])
            & alive[:, None, :]
            & alive[None, :, :]
        )
        agree = np.cumprod(same, axis=2).sum(axis=2)
        upper = np.triu_indices(len(group), 1)
        prefixes.append(agree[upper] - 1)
        prefix_ids.append(np.full(len(upper[0]), o, np.uint64))
    curve: list[float | None] = []
    for t in range(max_offset):
        known = ~np.isnan(shares[:, t])
        curve.append(
            stats.mean(
                f"selfplay.novelty.distinct_share.{t}", ids[known], shares[known, t]
            )
            if t and known.any()
            else None
        )
    window = shares[:, 1 : sampled_moves + 1]
    early = ~np.isnan(window).any(axis=1)
    return {
        "openings_used": len(groups),
        "games_per_opening": sample,
        "distinct_share": curve,
        "distinct_share_sampled": stats.mean(
            "selfplay.novelty.distinct_share_sampled",
            ids[early],
            window[early].mean(axis=1) if early.any() else np.zeros(0),
        ),
        "shared_moves": stats.mean(
            "selfplay.novelty.shared_moves",
            np.concatenate(prefix_ids) if prefix_ids else np.zeros(0, np.uint64),
            np.concatenate(prefixes) if prefixes else np.zeros(0),
        ),
    }


def after_offline(moves: dict[str, np.ndarray], start: int) -> np.ndarray:
    """Which logged moves come after their game went off line (see train.py's
    --store-until-offline): after a move at or past ply `start` that isn't the target's top one.
    Training stores none of those positions."""
    env, ply, action, policy = (
        moves["env"],
        moves["ply"],
        moves["action"],
        moves["policy"],
    )
    legal = moves["legal"].astype(bool)
    played = action >= 0
    weights = np.where(legal, policy, -np.inf)
    top = np.take_along_axis(weights, np.maximum(action, 0)[:, None], 1)[
        :, 0
    ] == weights.max(1)
    offline = played & (ply >= start) & ~top
    starts = np.r_[True, (env[1:] != env[:-1]) | (ply[1:] <= ply[:-1])]
    # an off-line move's game stores nothing after it: off-line moves so far, this one aside,
    # within each game
    before = np.cumsum(offline) - offline
    so_far = before - before[np.flatnonzero(starts)][np.cumsum(starts) - 1]
    return so_far > 0


def selfplay_metrics(
    moves: dict[str, np.ndarray],
    z: np.ndarray,
    prior: np.ndarray,
    args: argparse.Namespace,
    timing: dict[str, float],
    games_per_opening: int,
    features: FeatureFn | None = None,
    labels: LabelFn | None = None,
    label_per_phase: int = 0,
) -> tuple[dict[str, Any], Stats]:
    """The self-play pass's metrics (see the module's docstring), and their inputs. The
    bootstrap's unit is the opening: its games are resampled together. With `features`, the
    stored positions' distances include a reference network's (see `pair_statistics`). With
    `labels`, up to `label_per_phase` stored positions of each phase (with a choice of moves) are
    scored by Edax: the targets' regret where self-play actually went, and the stored values and
    outcomes against Edax's result (`edax_z_*`, `edax_q_*`), also within close positions (Edax's
    margin at most EDAX_CLOSE discs, `edax_z_close_*`) and clear ones (`edax_z_clear_*`)."""
    stats = Stats()
    obs, policy, value = moves["observation"], moves["policy"], moves["value"]
    full, env, action = moves["full"].astype(bool), moves["env"], moves["action"]
    n = len(env)
    if args.store_until_offline is not None:
        full = full & ~after_offline(moves, args.store_until_offline)
    empties = 64 - obs.reshape(n, -1).sum(-1)
    keys = symmetry_keys(obs)
    groups = bands(empties)
    target_top = np.argmax(policy, 1)
    prior_top = np.argmax(prior, 1)
    known = ~np.isnan(z)
    starts = np.flatnonzero(np.r_[True, env[1:] != env[:-1]])
    ends = np.r_[starts[1:], len(env)]
    # each game's opening, by its first position's key; and each row's
    game_opening = keys[starts]
    opening = np.repeat(game_opening, ends - starts)
    phases: dict[str, Any] = {}
    for phase, sel in groups.items():
        train = sel & full
        prefix = f"selfplay.phases.{phase}"
        ids = opening[train]
        # 1 - 1 / (copies of the row's position): its mean is 1 - distinct / rows
        _, inverse, counts = np.unique(
            keys[train], return_inverse=True, return_counts=True
        )
        stored = int(train.sum())
        per_row = {
            "duplicate_rate": 1 - 1 / counts[inverse],
            # the other stored positions that are this one: its mean is the chance that two
            # stored positions drawn at random are the same, whatever their number (Simpson's
            # index; its inverse, the effective number of distinct positions)
            "collision": (counts[inverse] - 1) / max(stored - 1, 1),
            "target_entropy": entropy(policy[train]),
            "prior_entropy": entropy(prior[train]),
            "kl_target_prior": kl(policy[train], prior[train]),
            "flip_rate": target_top[train] != prior_top[train],
            "decided_share": np.abs(value[train]) > DECIDED,
        }
        both = train & known
        maps = None if features is None else features(obs[train])
        drawn, pairs = pair_statistics(
            obs[train], empties[train], np.random.default_rng(args.seed), None, maps
        )
        same, same_pairs = pair_statistics(
            obs[train], empties[train], np.random.default_rng(args.seed), ids, maps
        )
        phases[phase] = {
            "training_positions": stored,
            **{k: stats.mean(f"{prefix}.{k}", ids, v) for k, v in per_row.items()},
            **{k: stats.mean(f"{prefix}.{k}", ids[drawn], v) for k, v in pairs.items()},
            # the same among the stored positions of one opening: how far its games spread
            **{
                f"{k}_same_opening": stats.mean(
                    f"{prefix}.{k}_same_opening", ids[same], v
                )
                for k, v in same_pairs.items()
            },
            **stats.value(prefix, opening[both], value[both], z[both], name="stored_"),
        }
        if labels is not None and label_per_phase > 0:
            choice = np.flatnonzero(both & (moves["legal"][:, :64].sum(1) >= 2))
            if len(choice) > label_per_phase:
                choice = np.sort(
                    np.random.default_rng(args.seed).choice(
                        choice, label_per_phase, replace=False
                    )
                )
            scored = labels(obs[choice])
            best = np.max(np.where(scored != ILLEGAL, scored, -9999), axis=1)
            edax_z = np.sign(best).astype(np.float64)
            phases[phase].update(
                {
                    k: stats.mean(f"{prefix}.{k}", opening[choice], v)
                    for k, v in label_metrics(
                        scored, policy[choice], prior[choice]
                    ).items()
                }
            )
            phases[phase]["edax_labelled"] = len(choice)
            # the stored outcome (the value target) and search value against Edax's result, over
            # all the labelled positions and within close and clear ones
            close = np.abs(best) <= EDAX_CLOSE
            phases[phase]["edax_close_share"] = stats.mean(
                f"{prefix}.edax_close_share", opening[choice], close
            )
            for part, sel in (("", slice(None)), ("close_", close), ("clear_", ~close)):
                part_ids = opening[choice][sel]
                phases[phase].update(
                    stats.value(
                        prefix,
                        part_ids,
                        z[choice][sel],
                        edax_z[sel],
                        name=f"edax_z_{part}",
                    )
                )
                phases[phase].update(
                    stats.value(
                        prefix,
                        part_ids,
                        value[choice][sel],
                        edax_z[sel],
                        name=f"edax_q_{part}",
                    )
                )
    # the moves sampled rather than played greedily: with a move temperature, all of them
    sample_until = 60 if args.move_temperature is not None else args.sampling_moves
    sampled = (action >= 0) & (moves["ply"] < sample_until)
    first_z = z[starts]
    solved = ~np.isnan(first_z)
    by_opening: dict[int, list[float]] = {}
    for o, result in zip(game_opening[solved], first_z[solved]):
        by_opening.setdefault(int(o), []).append(result)
    entropy_ids = [o for o, r in by_opening.items() if len(r) >= games_per_opening]
    result_entropy = [
        entropy(
            np.bincount(np.array(by_opening[o], int) + 1, minlength=3)[None]
            / len(by_opening[o])
        )[0]
        for o in entropy_ids
    ]
    all_train = full & groups["all"]
    seconds = timing["timed_seconds"]
    moves_per_second = timing["timed_moves"] / seconds if seconds else None
    train_share = float(full.mean())
    distinct_share = (
        len(np.unique(keys[all_train])) / all_train.sum() if all_train.any() else 0.0
    )
    result = {
        "games": len(starts),
        "moves": int(n),
        "phases": phases,
        "novelty": novelty(
            stats,
            keys,
            starts,
            ends,
            game_opening,
            games_per_opening,
            max(sample_until - 8, 1),
        ),
        "off_target_sampled": stats.mean(
            "selfplay.off_target_sampled",
            opening[sampled],
            action[sampled] != target_top[sampled],
        ),
        "outcomes": {
            "solved_games": int(solved.sum()),
            "draw_rate": stats.mean(
                "selfplay.outcomes.draw_rate",
                game_opening[solved],
                first_z[solved] == 0,
            ),
            "opening_mover_score": stats.mean(
                "selfplay.outcomes.opening_mover_score",
                game_opening[solved],
                (first_z[solved] + 1) / 2,
            ),
            "result_entropy_per_opening": stats.mean(
                "selfplay.outcomes.result_entropy_per_opening",
                np.array(entropy_ids, np.uint64),
                np.array(result_entropy),
            ),
            "moves_per_game": stats.mean(
                "selfplay.outcomes.moves_per_game", game_opening, ends - starts
            ),
        },
        "throughput": {
            "moves_per_second": moves_per_second,
            "training_positions_per_second": moves_per_second * train_share
            if moves_per_second
            else None,
            "distinct_training_positions_per_second": moves_per_second
            * train_share
            * distinct_share
            if moves_per_second
            else None,
            "training_share": train_share,
        },
    }
    return result, stats


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            check=False,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def run(args: argparse.Namespace) -> None:
    net = args.net
    configs = [parse_config(text, net) for text in args.config]
    names = [name for name, _, _ in configs]
    if len(set(names)) != len(names):
        raise SystemExit("--config names must differ")
    model = args.name or f"{Path(args.ckpt).parent.name}-{Path(args.ckpt).stem}"
    engine_set = EngineSet.load(args.engine_set)
    results = Path(args.results).expanduser()
    results.mkdir(parents=True, exist_ok=True)
    sha = file_sha256(args.ckpt)
    threads = args.solve_threads or default_threads()
    features = None
    if args.reference_ckpt is not None:
        if args.reference_net is None:
            raise SystemExit("--reference-ckpt needs its --reference-net")
        features = reference_features(
            *load_checkpoint(args.reference_ckpt, args.reference_net)
        )
    labels = (
        LabelCache(args.label_dir, args.label_level, args.label_workers)
        if args.label_stored > 0
        else None
    )
    targets = None
    if args.coverage is not None:
        if args.coverage_labels is None:
            raise SystemExit("--coverage needs its --coverage-labels")
        with np.load(args.coverage) as saved:
            targets = {k: saved[k] for k in ("obs", "set", "empties")}
        targets["obs"] = targets["obs"].astype(bool)
        targets["scores"] = np.load(args.coverage_labels)
        if args.coverage_sets:
            keep = np.isin(targets["set"], args.coverage_sets.split(","))
            targets = {k: v[keep] for k, v in targets.items()}
    for name, flags, config in configs:
        print(f"== {model} / {name}: {shlex.join(flags)}", flush=True)
        network, params = load_checkpoint(args.ckpt, net, config.inference_dtype)
        began = time.perf_counter()
        probed, stats = probe(
            network, params, config, engine_set, args.probe, args.seed, args.probe_batch
        )
        probe_seconds = time.perf_counter() - began
        print(f"   probe: {probe_seconds:.1f}s", flush=True)
        began = time.perf_counter()
        moves, timing = play_games(
            network,
            params,
            config,
            args.games,
            args.envs,
            args.openings,
            args.seed,
            args.max_calls,
        )
        selfplay_seconds = time.perf_counter() - began
        print(
            f"   self-play: {selfplay_seconds:.1f}s, {len(moves['env'])} moves, "
            f"{timing['calls']} calls",
            flush=True,
        )
        z, solve_seconds = game_outcomes(moves, config, threads)
        logits, _ = raw_outputs(network, params, moves["observation"].astype(bool))
        prior = masked_softmax(logits, moves["legal"].astype(bool))
        sample = args.games_per_opening or max(2, args.games // args.openings // 2)
        selfplay, selfplay_stats = selfplay_metrics(
            moves, z, prior, config, timing, sample, features, labels, args.label_stored
        )
        stats.inputs.update(selfplay_stats.inputs)
        if targets is not None:
            stored = moves["full"].astype(bool)
            if config.store_until_offline is not None:
                stored = stored & ~after_offline(moves, config.store_until_offline)
            stored_obs = moves["observation"][stored].astype(bool)
            stored_empties = 64 - stored_obs.reshape(len(stored_obs), -1).sum(-1)
            target_logits, target_value = raw_outputs(network, params, targets["obs"])
            target_policy = masked_softmax(target_logits, targets["scores"] != ILLEGAL)
            selfplay["coverage"] = coverage_metrics(
                targets,
                stored_obs,
                stored_empties,
                np.random.default_rng(args.seed),
                (target_policy, target_value),
                features,
            )
        result = {
            "id": f"{slug(model)}__{slug(name)}",
            "model": {
                "name": model,
                "checkpoint": str(args.ckpt),
                "net": net,
                "sha256": sha,
            },
            "config": {
                "name": name,
                "flags": shlex.join(flags),
                "args": config_record(config),
            },
            "settings": {
                "games": args.games,
                "envs": args.envs,
                "openings": args.openings,
                "probe_per_phase": args.probe,
                "engine_set": str(args.engine_set),
                "reference": None
                if args.reference_ckpt is None
                else {
                    "checkpoint": str(args.reference_ckpt),
                    "net": args.reference_net,
                    "sha256": file_sha256(args.reference_ckpt),
                },
                "seed": args.seed,
                "phases": PHASES,
                "edax_labels": {
                    "per_phase": args.label_stored,
                    "level": args.label_level,
                }
                if args.label_stored > 0
                else None,
            },
            "created": datetime.datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "device": str(jax.devices()[0].device_kind),
            "commit": git_commit(),
            "probe": probed,
            "selfplay": selfplay,
            # bootstrap standard errors, by metric path (see `Stats`)
            "se": stats.standard_errors(),
            "timing": {
                "probe_seconds": probe_seconds,
                "selfplay_seconds": selfplay_seconds,
                "solve_seconds": solve_seconds,
                **timing,
            },
        }
        path = results / f"{result['id']}.json"
        stats.save(results / f"{result['id']}.npz")
        path.write_text(json.dumps(result, indent=1))
        summary = selfplay["phases"]["all"]
        print(
            f"   solve: {solve_seconds:.1f}s; wrote {path}\n"
            f"   target entropy {summary['target_entropy']:.3f}, duplicates "
            f"{summary['duplicate_rate']:.3f}, shared moves {selfplay['novelty']['shared_moves']}",
            flush=True,
        )


def serve(args: argparse.Namespace) -> None:
    """The dashboard: the page, `results.json` (every run's result), and `compare?base=ID&ids=A,B`
    (each run's paired differences from the base run, see `compare`)."""
    results = Path(args.results).expanduser()
    loaded: dict[str, tuple[float, Stats]] = {}

    def stats_of(run_id: str) -> Stats | None:
        path = results / f"{run_id}.npz"
        if not run_id or run_id != Path(run_id).name or not path.exists():
            return None
        mtime = path.stat().st_mtime
        if run_id not in loaded or loaded[run_id][0] != mtime:
            loaded[run_id] = (mtime, Stats.load(path))
        return loaded[run_id][1]

    def comparisons(query: dict[str, list[str]]) -> dict[str, Any]:
        base = stats_of(query.get("base", [""])[0])
        out: dict[str, Any] = {}
        if base is None:
            return out
        for run_id in ",".join(query.get("ids", [])).split(","):
            other = stats_of(run_id) if run_id else None
            if other is not None:
                out[run_id] = compare(base, other)
        return out

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path == "/results.json":
                runs = []
                for path in sorted(results.glob("*.json")):
                    try:
                        runs.append(json.loads(path.read_text()))
                    except (OSError, json.JSONDecodeError):
                        continue
                body, kind = json.dumps(runs).encode(), "application/json"
            elif url.path == "/compare":
                query = urllib.parse.parse_qs(url.query)
                body, kind = json.dumps(comparisons(query)).encode(), "application/json"
            elif url.path in ("/", "/index.html"):
                body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {results} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser(
        "run", help="measure a checkpoint under self-play configs"
    )
    run_parser.add_argument(
        "--ckpt", required=True, help="the network's checkpoint (.eqx)"
    )
    run_parser.add_argument(
        "--net", required=True, help="its network, as train.py's --net"
    )
    run_parser.add_argument("--name", default=None, help="the model's name in results")
    run_parser.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="NAME=FLAGS",
        help="a self-play config: a name, and train.py's self-play flags (repeatable)",
    )
    run_parser.add_argument("--games", type=int, default=2048, help="self-play games")
    run_parser.add_argument(
        "--envs",
        type=int,
        default=1024,
        help="games played at once (train.py's 1024); --games is a multiple of it",
    )
    run_parser.add_argument(
        "--openings", type=int, default=64, help="XOT openings the games start from"
    )
    run_parser.add_argument(
        "--games-per-opening",
        type=int,
        default=None,
        help="games of each opening diversity is measured among (default half the mean)",
    )
    run_parser.add_argument(
        "--probe", type=int, default=1024, help="probe positions per phase, at most"
    )
    run_parser.add_argument(
        "--probe-batch", type=int, default=512, help="probe positions searched at once"
    )
    run_parser.add_argument("--engine-set", default=str(DEFAULT_ENGINE_SET))
    run_parser.add_argument(
        "--reference-ckpt",
        default=None,
        help="a fixed network whose trunk features measure how alike stored positions are "
        "(feature_distance, soft_collision_feature_*): the same one for every model compared",
    )
    run_parser.add_argument(
        "--reference-net", default=None, help="its network, as train.py's --net"
    )
    run_parser.add_argument(
        "--label-stored",
        type=int,
        default=0,
        metavar="N",
        help="score up to N stored positions of each phase with Edax (edax_* metrics: the "
        "targets' regret where self-play went, the value targets against Edax's result); "
        "labels are kept in --label-dir for every run",
    )
    run_parser.add_argument(
        "--label-level",
        type=int,
        default=8,
        help="with --label-stored, Edax's level: at 8 a position takes about 17 ms of a core, "
        "and its best move loses about 0.5 discs by level 24's scores",
    )
    run_parser.add_argument(
        "--label-workers",
        type=int,
        default=8,
        help="with --label-stored, Edax processes",
    )
    run_parser.add_argument("--label-dir", default=str(LABELS_DIR))
    run_parser.add_argument(
        "--coverage",
        default=None,
        metavar="NPZ",
        help="reference positions (obs, set, empties: e.g. from games against strong opponents) "
        "to measure how close each config's stored positions come, and the network's raw "
        "accuracy on them by that distance (selfplay.coverage)",
    )
    run_parser.add_argument(
        "--coverage-labels",
        default=None,
        metavar="NPY",
        help="with --coverage, Edax's scores of every move of its positions, (n, 65)",
    )
    run_parser.add_argument(
        "--coverage-sets",
        default=None,
        help="with --coverage, only these of its sets, comma-separated (default all)",
    )
    run_parser.add_argument(
        "--solve-threads", type=int, default=None, help="exact solver's threads"
    )
    run_parser.add_argument(
        "--max-calls", type=int, default=200, help="self-play calls before giving up"
    )
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--results", default=str(RESULTS_DIR))
    serve_parser = sub.add_parser("serve", help="the dashboard")
    serve_parser.add_argument("--port", type=int, default=8061)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--results", default=str(RESULTS_DIR))
    args = parser.parse_args()
    if args.command == "run":
        os.makedirs("/tmp/turbozero-selfplay-health", exist_ok=True)
        run(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
