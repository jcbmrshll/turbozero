import inspect
import json
import os
import re
import time
import zipfile
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.typing import ArrayLike
from seaglass.client import to_number

from core.common import step_env_and_evaluator, step_with_output
from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.mcts import MCTS
from core.memory.move_log import (
    ENDED,
    GOES_ON,
    STOPPED,
    TRUNCATED,
    LoggedMove,
    MoveLog,
    MoveLogWriter,
    emptied,
    init_move_log,
    log_move,
)
from core.memory.replay_memory import (
    BaseExperience,
    DataTransformFn,
    EpisodeReplayBuffer,
    ReplayBufferState,
    buffer_capacity,
    transform_randomly,
)
from core.pytree import map_where, where_tree
from core.testing.tester import BaseTester, TestState
from core.training.endgame_solve import EndgameSolve, PendingSolves, Solved
from core.training.exploration import Exploration
from core.training.growth import Growth, GrowthProgress, Learner
from core.training.lr_drop import LRDrop
from core.training.lr_plateau import LRPlateau, LRPlateauProgress, LRPlateauState
from core.training.noise_scale import noise_scale_estimates, noise_scale_metrics
from core.training.resignation import Resignation, ResignState
from core.training.schedule import EvaluatorSchedule, describe
from core.training.tree_positions import (
    TreePositions,
    select_nodes,
    tree_experiences,
)
from core.types import (
    EnvInitFn,
    EnvStepFn,
    ExtractModelParamsFn,
    LossFn,
    StateToNNInputFn,
    StepMetadata,
    TrainState,
)

if TYPE_CHECKING:
    # only a type: the trainer logs to whatever it's given
    from seaglass import Monitor


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CollectionState:
    """Stores state of self-play episode collection. Persists across generations.

    Attributes:
        eval_state: state of the evaluator
        env_state: state of the environment
        buffer_state: state of the replay buffer
        metadata: metadata of the current environment state
        episodes: number of episodes that have terminated so far (not counting truncated ones)
        tree_positions: number of search tree positions stored so far (see `Trainer`'s
            `tree_positions`)
        tree_visits: their visit counts, summed
        moves: number of self-play moves made so far
        episode_metrics: sums over the terminated episodes of each metric `Trainer`'s
            `episode_metrics_fn` returns (empty without it)
        move_key: rng of the move in progress, its search's and the move's (see
            `Trainer.start_searches`)
        search_cycles: cycles the move's search has run so far (see `Evaluator.search_cycle`)
        tree_buffer_state: state of the replay buffer of search tree positions, None without them
        tree_written_at: for each entry of the tree position replay buffer, the number of moves made
            before it was written (see `Trainer.tree_sample_mask`), None without tree positions
        resign_state: each environment's resignation state (see `Trainer`'s `resignation`), None
            without resignation
        pending: each environment's positions waiting for exact solves (see `Trainer`'s
            `endgame_solve`), None without
        move_log: each environment's moves since the end of the last epoch's self-play (see
            `Trainer`'s `log_moves`), None without. Not part of a saved state
    """

    eval_state: Any
    env_state: Any
    buffer_state: ReplayBufferState
    metadata: StepMetadata
    episodes: jax.Array
    tree_positions: jax.Array
    tree_visits: jax.Array
    moves: jax.Array
    episode_metrics: dict[str, jax.Array]
    move_key: jax.Array
    search_cycles: jax.Array
    tree_buffer_state: ReplayBufferState | None = None
    tree_written_at: jax.Array | None = None
    resign_state: ResignState | None = None
    pending: PendingSolves | None = None
    move_log: MoveLog | None = None


@dataclass(frozen=True)
class CollectionStats:
    """The stats of a collection state that `Trainer.collection_metrics` compares across an
    epoch's data collection (see `CollectionState`).

    Attributes:
        episodes: number of episodes that have terminated so far, per environment
        moves: number of self-play moves made so far, per environment
        tree_positions: number of search tree positions stored so far
        tree_visits: their visit counts, summed
        episode_metrics: sums over the terminated episodes of each episode metric
        resign_state: the resignation state's counters, None without resignation
        solve_missed: with exact endgame labels, the solves that found every slot full, per
            environment (see `PendingSolves.missed`); else None
    """

    episodes: jax.Array
    moves: jax.Array
    tree_positions: jax.Array
    tree_visits: jax.Array
    episode_metrics: dict[str, jax.Array]
    resign_state: ResignState | None = None
    solve_missed: jax.Array | None = None

    @classmethod
    def of(cls, state: CollectionState) -> "CollectionStats":
        """Copies the stats out of `state`, so that they outlive it when it is donated to
        `Trainer.collect_steps`."""
        return cls(
            episodes=jnp.copy(state.episodes),
            moves=jnp.copy(state.moves),
            tree_positions=jnp.copy(state.tree_positions),
            tree_visits=jnp.copy(state.tree_visits),
            episode_metrics={k: jnp.copy(v) for k, v in state.episode_metrics.items()},
            resign_state=jax.tree.map(jnp.copy, state.resign_state),
            solve_missed=jnp.copy(state.pending.missed)
            if state.pending is not None
            else None,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainLoopOutput:
    """Stores the state of the training loop.

    `collection_state` is included to access replay memory.

    Attributes:
        collection_state: state of self-play episode collection.
        train_states: each network's TrainState (model params and state, optimizer state): the
            Trainer's own, then, with `growth`, each it grows into
        test_states: each network's testers' states
        cur_epoch: current epoch num
        key: (optional) the training loop's rng, for the next epoch. With it, a run continued
            from this state carries on exactly as if it hadn't stopped (see `Trainer.train_loop`).
        lr_plateau: with the Trainer's `lr_plateau`, each network's progress with the rule at its
            current rate (its index is in the optimizer's state), None for a network the rule
            doesn't follow (yet); empty without the rule
        growth: (optional) with the Trainer's `growth`, which network plays and the switch rule's
            progress
        seconds: the wall time the run's training loop has taken so far, over every call
        averages: with the Trainer's `averages`, each network's weight averages, one params tree
            per horizon; empty without them, or for a state that has none to continue (the run
            starts them at the networks' weights)
        averaged: the number of epochs each network's averages hold
    """

    collection_state: CollectionState
    train_states: tuple[TrainState, ...]
    test_states: tuple[list[TestState], ...]
    cur_epoch: int
    key: jax.Array | None = None
    lr_plateau: tuple[LRPlateauProgress | None, ...] = field(
        default=(), metadata={"static": True}
    )
    growth: GrowthProgress | None = field(default=None, metadata={"static": True})
    seconds: float = field(default=0.0, metadata={"static": True})
    averages: tuple[tuple[Any, ...], ...] = ()
    averaged: tuple[int, ...] = field(default=(), metadata={"static": True})

    @property
    def playing(self) -> int:
        """The index of the network self-play plays with."""
        return self.growth.playing if self.growth is not None else 0

    @property
    def train_state(self) -> TrainState:
        """The TrainState of the network self-play plays with."""
        return self.train_states[self.playing]


def extract_params(state: TrainState) -> Any:
    """Extracts model parameters from TrainState.

    Args:
        state: TrainState containing model parameters

    Returns:
        Tuple[Any, Optional[eqx.nn.State]]: model parameters and state (nn_params, nn_state), as used by
            core.evaluators.evaluation_fns.make_nn_eval_fn
    """
    return state.params, state.nn_state


def average_name(horizon: float) -> str:
    """The name of a network's weight average over about `horizon` epochs (see `Trainer`'s
    `averages`): its checkpoints' directory and its metrics' prefix, e.g. avg4."""
    return f"avg{horizon:g}"


def average_share(averaged: int, horizon: float, warmup: float) -> float:
    """The share a network's newest weights take in its average over about `horizon` epochs,
    once `averaged` epochs are in it: an exponential moving average whose horizon starts at 1
    (the newest weights) and grows by an epoch every `warmup` epochs averaged, up to
    `horizon`. With a `warmup` of 1, the plain mean of the epochs so far until there are
    `horizon` of them.

    An average lags its network by about (1 - share) / share epochs. Early in training each
    epoch improves the network a lot, and a lag of more than about a tenth of the epochs
    trained costs more than averaging gains: in Othello, a 4-epoch average without a warmup
    lost 110-210 Elo to the current weights at epoch 10, and a 16-epoch one 110-500 Elo
    throughout the first 60 epochs, while 4 epochs gained 30-40 from epoch 30."""
    return 1.0 / min(horizon, 1.0 + averaged / warmup)


@jax.jit
def _move_average(average: Any, params: Any, share: jax.Array) -> Any:
    # with a share of 1, exactly the params
    return jax.tree.map(lambda a, p: (1 - share) * a + share * p, average, params)


def checkpoint_path(ckpt_dir: str, epoch: int) -> str:
    """Path of the checkpoint for `epoch` in `ckpt_dir`."""
    return os.path.join(ckpt_dir, f"{epoch}.eqx")


def checkpoint_epochs(ckpt_dir: str) -> list[int]:
    """Epochs with a checkpoint in `ckpt_dir`, in ascending order."""
    return _epochs_in(ckpt_dir, r"(\d+)\.eqx")


def state_path(ckpt_dir: str, epoch: int) -> str:
    """Path of the state saved after `epoch` epochs in `ckpt_dir` (see `Trainer.save_state`)."""
    return os.path.join(ckpt_dir, f"state-{epoch}.npz")


def state_epochs(ckpt_dir: str) -> list[int]:
    """Numbers of epochs done at which `ckpt_dir` holds a saved state, in ascending order."""
    return _epochs_in(ckpt_dir, r"state-(\d+)\.npz")


def _epochs_in(ckpt_dir: str, pattern: str) -> list[int]:
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        int(m.group(1)) for f in os.listdir(ckpt_dir) if (m := re.fullmatch(pattern, f))
    )


def latest_state_path(path: str) -> str:
    """`path` if it's a saved state, or the newest saved state in the directory `path`."""
    if not os.path.isdir(path):
        return path
    epochs = state_epochs(path)
    if not epochs:
        raise FileNotFoundError(
            f"no saved state (state-<epoch>.npz) in {path}: a run saves one at the epochs "
            "given by the Trainer's save_state_at and save_state_every"
        )
    return state_path(path, epochs[-1])


def saved_growth(saved: dict) -> GrowthProgress:
    """A growth run's progress, as `Trainer.save_state` saved it in a state's metadata."""
    return GrowthProgress(
        saved["playing"],
        tuple(saved["trained"]),
        tuple(tuple(v) for v in saved["values"]),
        tuple(tuple(v) for v in saved["guard_values"]),
        # a state saved before networks counted their epochs playing: none counted
        tuple(saved.get("played") or (0,) * len(saved["trained"])),
    )


def saved_network_names(meta: dict) -> list[str]:
    """The names of a saved growth run's networks, in order (its own first), from a state's
    metadata (see `Trainer.save_state`)."""
    growth = meta["config"]["growth"]
    return [growth["first"], *(learner["name"] for learner in growth["learners"])]


def read_state_meta(path: str) -> dict:
    """The metadata of a saved state (see `Trainer.save_state`): the epochs done, the run's
    configuration, its monitor run and parent, and what each array is."""
    with np.load(path) as saved:
        return json.loads(str(saved["meta"]))


# saved states' format, recorded in their metadata
STATE_FORMAT = 2


def _describe(tree: Any) -> list[tuple[str, tuple[int, ...], str]]:
    """Each leaf's path, shape and dtype, in the order `jax.tree.leaves` gives them (leaves may
    be `jax.ShapeDtypeStruct`s)."""
    described = []
    for path, x in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not hasattr(x, "dtype"):
            x = np.asarray(x)
        described.append((jax.tree_util.keystr(path), tuple(x.shape), str(x.dtype)))
    return described


def _flatten(config: dict, prefix: str = "") -> dict:
    """`config` with its nested dicts' entries in its own, named by their path, e.g. `args.lr`."""
    flat = {}
    for k, v in config.items():
        if isinstance(v, dict) and v:
            flat.update(_flatten(v, f"{prefix}{k}."))
        else:
            flat[f"{prefix}{k}"] = v
    return flat


def _to_config(x: Any) -> Any:
    """Array `x` as a JSON value, its floats the shortest decimals that round-trip at their precision
    (a float32 0.9 is 0.9, not 0.8999999761581421)."""
    x = np.asarray(x)
    if x.dtype.kind != "f":
        return x.tolist()
    shortest = [float(str(v)) for v in x.reshape(-1)]
    return np.array(shortest).reshape(x.shape).tolist()


def _to_host(x: Any) -> np.ndarray:
    """`x` as a numpy array, through a temporary array on the CPU device if there is one: for an
    accelerator's array, `np.asarray` would keep the host copy on it, for as long as it lives."""
    if not isinstance(x, jax.Array):
        return np.asarray(x)
    try:
        cpu = jax.devices("cpu")[0]
    except RuntimeError:
        return np.asarray(x)
    return np.asarray(jax.device_put(x, cpu))


def _to_device(x: Any) -> jax.Array:
    """A new device array holding `x`, sharing memory with nothing else, so that it can be
    donated (see `Trainer.collect_steps`)."""
    return jnp.array(x, copy=True)


def _is_injected(x: Any) -> bool:
    """Whether `x` is the state of an `optax.inject_hyperparams` optimizer."""
    return isinstance(
        x, optax.InjectHyperparamsState | optax.InjectStatefulHyperparamsState
    )


def weight_decay_mask(params: Any) -> Any:
    """Which of `params` weight decay applies to: the weight matrices and convolution kernels (arrays
    of 2 or more dimensions), not biases or normalization layers' scales and offsets. For optax's
    `mask` argument, e.g. `optax.adamw(learning_rate, weight_decay=0.5, mask=weight_decay_mask)`,
    and with `optax.inject_hyperparams`, which takes a callable argument for a schedule unless it's
    static: `optax.inject_hyperparams(optax.adamw, static_args=("mask",))`."""
    return jax.tree.map(lambda x: x.ndim >= 2, params)


def sgdw(
    learning_rate: optax.ScalarOrSchedule,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    decoupled: bool = False,
    mask: Any = None,
) -> optax.GradientTransformation:
    """SGD with (heavy-ball) momentum and weight decay, as AlphaZero and KataGo trained: each step
    the momentum buffer m <- momentum * m + gradient, and the weights w <- w - learning_rate * m.

    The decay adds `weight_decay * w` to what's applied: with `decoupled` False (L2, AlphaZero's
    and KataGo's), to the gradient before momentum, as the gradient of an L2 penalty
    `weight_decay / 2 * |w|^2` in the loss would; with `decoupled`, to the step after momentum
    (w <- w - learning_rate * (m + weight_decay * w)), as AdamW's decay is.

    For `optax.inject_hyperparams`, with `mask` and `decoupled` static, e.g.
    `optax.inject_hyperparams(sgdw, static_args=("mask", "decoupled"))(learning_rate=0.2,
    momentum=0.9, weight_decay=1e-4, mask=weight_decay_mask)`."""
    decay = optax.add_decayed_weights(weight_decay, mask)
    trace = optax.trace(decay=momentum)
    return optax.chain(
        *((trace, decay) if decoupled else (decay, trace)),
        optax.scale_by_learning_rate(learning_rate),
    )


def update_size_metrics(params: Any, updates: Any) -> dict:
    """How big an optimizer step is, relative to the weights it moves: over the weights weight
    decay applies to (see `weight_decay_mask`),
    - `weight_norm`: their norm, ||w||
    - `update_ratio`: the step's norm relative to theirs, ||update|| / ||w||

    Under BatchNorm a convolution's output doesn't depend on its weights' scale, so it's the
    relative step, not the learning rate, that says how fast such a layer moves; with AdamW's
    weight decay it settles at about sqrt(2 * learning rate * weight decay) per step. Comparable
    across learning rates, weight decays and schedules.
    """
    mask = weight_decay_mask(params)

    def squared_norm(tree: Any) -> jax.Array:
        leaves = [
            jnp.sum(jnp.square(x.astype(jnp.float32)))
            for x, m in zip(jax.tree.leaves(tree), jax.tree.leaves(mask), strict=True)
            if m
        ]
        return jnp.sum(jnp.stack(leaves)) if leaves else jnp.zeros(())

    weight_norm = jnp.sqrt(squared_norm(params))
    return {
        "weight_norm": weight_norm,
        "update_ratio": jnp.sqrt(squared_norm(updates)) / weight_norm,
    }


def reduce_step_metrics(metrics: dict) -> dict:
    """An epoch's metrics from its training steps' (each array one value per step): their means,
    besides what a mean hides, the spikes that come before training blows up:
    - `grad_norm_max`, `grad_norm_median`: the largest and the median of the steps' global
      gradient norms, and `grad_norm_spike`, their ratio
    - `loss_spike`: the largest step's loss over the median step's
    - `nonfinite_steps`: how many steps had a loss or gradient that wasn't finite (a count, not a mean)
    """
    out = {
        name: jnp.mean(values)
        for name, values in metrics.items()
        if name != "nonfinite_steps"
    }
    if "grad_norm" in metrics:
        grad_norm = metrics["grad_norm"]
        median = jnp.median(grad_norm)
        out["grad_norm_max"] = jnp.max(grad_norm)
        out["grad_norm_median"] = median
        out["grad_norm_spike"] = jnp.max(grad_norm) / jnp.maximum(median, 1e-12)
    if "loss" in metrics:
        out["loss_spike"] = jnp.max(metrics["loss"]) / jnp.maximum(
            jnp.median(metrics["loss"]), 1e-12
        )
    if "nonfinite_steps" in metrics:
        out["nonfinite_steps"] = jnp.sum(metrics["nonfinite_steps"]).astype(jnp.float32)
    return out


def restart_schedules(opt_state: Any) -> Any:
    """`opt_state` with its learning rate schedules' step counts back at 0, so that they start
    over: optax's `ScaleByScheduleState` (a schedule passed as an optimizer's learning rate), and
    `optax.inject_hyperparams`' own step count and its schedules' states, an `LRPlateau`'s back
    at its first rate.

    Everything else (e.g. Adam's moments, and the step count of its bias correction) is
    kept."""

    def is_schedule(x: Any) -> bool:
        return isinstance(x, optax.ScaleByScheduleState) or _is_injected(x)

    def has_count(x: Any) -> bool:
        return isinstance(x, tuple) and "count" in getattr(x, "_fields", ())

    def restart_count(x: Any) -> Any:
        return x._replace(count=jnp.zeros_like(x.count))

    def restart_schedule_state(x: Any) -> Any:
        if isinstance(x, LRPlateauState):
            return LRPlateauState(rate_index=jnp.zeros_like(x.rate_index))
        return restart_count(x) if has_count(x) else x

    def restart(x: Any) -> Any:
        if isinstance(x, optax.ScaleByScheduleState):
            return restart_count(x)
        x = restart_count(x)._replace(inner_state=restart_schedules(x.inner_state))
        if isinstance(x, optax.InjectStatefulHyperparamsState):
            # each schedule counts its own steps (optax's `WrappedScheduleState`)
            x = x._replace(
                hyperparams_states=jax.tree.map(
                    restart_schedule_state,
                    x.hyperparams_states,
                    is_leaf=lambda s: has_count(s) or isinstance(s, LRPlateauState),
                )
            )
        return x

    return jax.tree.map(
        lambda x: restart(x) if is_schedule(x) else x, opt_state, is_leaf=is_schedule
    )


def injected_hyperparams(
    opt_state: Any, scheduled_only: bool = False
) -> dict[str, jax.Array]:
    """The hyperparameters of the `optax.inject_hyperparams` optimizers in `opt_state`, by name:
    the values its last update used (or, before any, the first update will).

    Args:
        opt_state: an optimizer's state
        scheduled_only: only those that follow a schedule (e.g. a decaying learning rate)

    Returns:
        dict[str, jax.Array]: the hyperparameters' values, none if the optimizer doesn't inject any
    """
    hyperparams = {}
    for state in jax.tree.leaves(opt_state, is_leaf=_is_injected):
        if not _is_injected(state):
            continue
        scheduled = getattr(state, "hyperparams_states", {})
        hyperparams.update(
            (name, value)
            for name, value in state.hyperparams.items()
            if not scheduled_only or name in scheduled
        )
    return hyperparams


def with_hyperparams(opt_state: Any, hyperparams: Mapping[str, Any]) -> Any:
    """`opt_state` with its `optax.inject_hyperparams` optimizers' hyperparameters named in
    `hyperparams` set to those values, each in its array's dtype.

    An injected optimizer keeps its hyperparameters in its state, and computes only the scheduled
    ones (e.g. a decaying learning rate) again each update: a restored state would go on with
    its saved values of the others (e.g. weight decay, momentum), whatever the new run's
    optimizer was built with (see `Trainer`, which sets them to its own)."""

    def set_values(x: Any) -> Any:
        if not _is_injected(x):
            return x
        return x._replace(
            hyperparams={
                name: jnp.asarray(hyperparams[name], dtype=jnp.asarray(value).dtype)
                if name in hyperparams
                else value
                for name, value in x.hyperparams.items()
            }
        )

    return jax.tree.map(set_values, opt_state, is_leaf=_is_injected)


def _is_lr_plateau(x: Any) -> bool:
    return isinstance(x, LRPlateauState)


def lr_plateau_states(opt_state: Any) -> list[LRPlateauState]:
    """The states of the `LRPlateau` schedules in `opt_state`."""
    return [
        x
        for x in jax.tree.leaves(opt_state, is_leaf=_is_lr_plateau)
        if _is_lr_plateau(x)
    ]


def with_rate_index(opt_state: Any, rate_index: int) -> Any:
    """`opt_state` with its `LRPlateau` schedules at rate `rate_index`, in an array like the one
    it replaces, so that nothing compiles again."""
    return jax.tree.map(
        lambda x: (
            LRPlateauState(jnp.full_like(x.rate_index, rate_index))
            if _is_lr_plateau(x)
            else x
        ),
        opt_state,
        is_leaf=_is_lr_plateau,
    )


def solve_summary(solved: Sequence[Solved], waited: float) -> dict[str, jax.Array]:
    """An epoch's metrics of exact endgame labels (see `Trainer.collection_metrics`), from its
    solves and the seconds self-play waited for them."""
    whole = np.concatenate(
        [s.solves[s.whole] for s in solved if len(s.solves)] or [np.zeros((0, 0))]
    )
    return {
        "solve_games": jnp.asarray(len(whole)),
        "solve_positions": jnp.asarray(sum(len(s.solves) for s in solved)),
        "solve_seconds": jnp.asarray(sum(s.seconds for s in solved)),
        "solve_wait_seconds": jnp.asarray(waited),
        **(
            {"solve_draw": jnp.asarray((whole == 0).all(-1).mean())}
            if len(whole)
            else {}
        ),
    }


def describe_fn(fn: Callable) -> dict:
    """`fn`'s name and keyword arguments, with their defaults and any `functools.partial` sets.
    Used for logging."""
    keywords = {}
    while isinstance(fn, partial):
        keywords = {**fn.keywords, **keywords}
        fn = fn.func
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        params = []
    defaults = {
        p.name: p.default for p in params if p.default is not inspect.Parameter.empty
    }
    return {"name": getattr(fn, "__name__", str(fn)), **defaults, **keywords}


class Trainer:
    """Implements a training loop for AlphaZero.

    Maintains state across self-play game collection, training, and testing.
    """

    def __init__(
        self,
        batch_size: int,
        train_batch_size: int,
        warmup_steps: int,
        collection_steps_per_epoch: int,
        train_steps_per_epoch: int,
        nn: eqx.Module,
        loss_fn: LossFn,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator | EvaluatorSchedule,
        memory_buffer: EpisodeReplayBuffer,
        max_episode_steps: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        state_to_nn_input_fn: StateToNNInputFn,
        testers: Sequence[BaseTester],
        nn_state: eqx.nn.State | None = None,
        grad_accum_steps: int = 1,
        lr_drop: LRDrop | None = None,
        noise_measure: int = 0,
        fresh_positions: int = 0,
        evaluator_test: Evaluator | None = None,
        test_env_init_fn: EnvInitFn | None = None,
        exploration: Exploration | None = None,
        resignation: Resignation | None = None,
        endgame_solve: EndgameSolve | None = None,
        episode_metrics_fn: Callable[[jax.Array], dict[str, jax.Array]] | None = None,
        data_transform_fns: Sequence[DataTransformFn] = (),
        tree_positions: TreePositions | None = None,
        advance_every: int | None = None,
        moves_at_once: int = 32,
        extract_model_params_fn: ExtractModelParamsFn = extract_params,
        monitor: "Monitor | None" = None,
        ckpt_dir: str = "/tmp/turbozero_checkpoints",
        max_checkpoints: int = 2,
        keep_every: int | None = None,
        save_state_at: Collection[int] = (),
        save_state_every: int | None = None,
        lr_plateau: LRPlateau | None = None,
        lr_plateau_metric: str = "loss",
        growth: Growth | None = None,
        waiting_train_steps: int | None = None,
        train_retired: bool = False,
        train_playing: bool = True,
        averages: Sequence[float] = (),
        selfplay_average: float | None = None,
        average_warmup: float = 10.0,
        log_moves: str | None = None,
        extra_config: dict | None = None,
    ):
        """Initializes a Trainer.

        Args:
            batch_size: batch size for self-play games
            train_batch_size: minibatch size for training steps
            warmup_steps: # of steps (per batch) to collect via self-play prior to entering the training loop.
                - This is used to populate the replay memory with some initial samples
            collection_steps_per_epoch: # of steps (per batch) to collect via self-play in each epoch
            train_steps_per_epoch: # of training steps to take in each epoch. 0 trains nothing:
                the networks, their weight averages and their testers stay as they are, and
                self-play is all an epoch does (e.g. a fixed network's games, with `log_moves`)
            nn: neural network (an equinox module, see core.networks.utils.apply_nn), training starts from its parameters
            loss_fn: loss function for training (see core.training.loss_fns)
            optimizer: optax optimizer. Wrap it in `optax.inject_hyperparams` to have its hyperparameters
                recorded with the monitor's run (their initial values), and the ones that follow a schedule
                (e.g. a decaying learning rate) logged every epoch, their mean over its training steps:
                `optax.inject_hyperparams(optax.adam)(learning_rate=schedule)`
            evaluator: the `Evaluator` to use during self-play, or an `EvaluatorSchedule` of evaluators
                that take over from one another at given epochs (e.g. searching more MCTS iterations
                later in training)
            memory_buffer: replay memory buffer class, used to store self-play experiences
            max_episode_steps: maximum number of steps in an episode. Self-play episodes still running after this many
                steps are truncated and their experiences discarded; an episode that terminates on its last allowed step is kept.
            env_step_fn: environment step function (env_state, action) -> (new_env_state, metadata)
            env_init_fn: environment initialization function (key) -> (env_state, metadata)
            state_to_nn_input_fn: function to convert environment state to neural network input
            testers: list of testers to evaluate the agent against (see core.testing.tester)
            nn_state: (optional) initial state of `nn` for stateful networks (e.g. with BatchNorm), from `eqx.nn.make_with_state`
            grad_accum_steps: (optional) split each training minibatch into this many micro-batches, and
                apply one optimizer update with their gradients' mean (gradient accumulation): training
                holds one micro-batch's activations at a time, so takes less memory, for the same
                compute. It must divide `train_batch_size`. Micro-batch i takes samples i, i + n, i + 2n,
                ... of the minibatch (for n micro-batches), so that each holds its share of tree positions
                (see `tree_positions`). The update is the minibatch's, up to float rounding, except for
                layers that compute statistics across the batch: BatchNorm normalizes each micro-batch
                with its own statistics, and its running statistics are updated once per step, with the
                mean of the micro-batches' updates (for BatchNorm's running mean, that's the
                minibatch's; its running variance misses the variance between micro-batches' means).
                Default 1: no accumulation. With 2 or more, training logs the gradient noise scale
                from the micro-batches' gradients (see `core.training.noise_scale`), as `gns_*`.
            lr_drop: (optional) drop each network's learning rate once it has played self-play for
                a number of epochs (see `core.training.lr_drop.LRDrop`), by scaling its updates;
                a network waiting to take over trains at the full rate. It logs `lr_scale`.
            noise_measure: (optional) every epoch, measure each training network's gradient noise
                scale at its weight average (with `averages`; the self-play one's horizon, else
                the first) from this many micro-batches, logged as `gns_avg_*`: for watching,
                nothing follows it (see `noise_scale_at`)
            fresh_positions: (optional) every epoch, score the playing network on this many of the
                positions its self-play just stored (finished games only, sampled with
                replacement), before it trains on them: its raw weights' and, with `averages`, its
                self-play average's policy KL and value loss, logged as `fresh_*`, and the raw
                weights' excess over the average's, `fresh_*_gap`: how much the step's jitter
                costs on new data (see `fresh_position_metrics`). A multiple of the micro-batch
            evaluator_test: (optional) evaluator to use during testing. If not provided, `evaluator` is used
                (required with a schedule of several evaluators).
            test_env_init_fn: (optional) environment initialization function for test episodes, e.g. to test from
                the standard start while self-play starts from varied positions. If not provided, `env_init_fn` is used.
            exploration: (optional) chooses the move data collection plays from the evaluator's output,
                e.g. to play random moves some of the time (see core.training.exploration). The evaluator's
                policy weights stay the training target. If not provided, collection plays the evaluator's move.
            resignation: (optional) self-play's players resign lost games (see core.training.resignation),
                ending them early as losses. Two-player zero-sum games only. Resigned games count as
                terminated episodes. If not provided, every game is played to the end.
            endgame_solve: (optional) exact endgame labels (see core.training.endgame_solve): once a
                self-play game reaches a position an exact solver solves, on the host while self-play
                goes on, every position of the game before it gets its solved result as the value
                target; the game stops there, or plays on a few moves, each position stored past it
                solved on its own. Positions solved in an epoch are labelled after the next epoch's
                self-play, so training samples them from then on. Not with `resignation`.
            episode_metrics_fn: (optional) metrics of a finished episode to track, from its final rewards (one per
                player) to a dict of scalars, e.g. `lambda rewards: {"draw": (rewards == 0).all()}`. Each is logged
                every epoch as `episode_<name>`, its mean over the episodes that terminated that epoch (truncated
                ones aren't counted). Must be compatible with `jax.jit`.
            data_transform_fns: (optional) data transforms (e.g. a board's rotations and reflections, see
                `core.memory.replay_memory.DataTransformFn`) besides the identity. The replay buffers store
                each position once, and training maps each sample it draws through one of the identity and
                these, drawn uniformly at random. Together they must be a group (as a board's symmetries are),
                for `buffer_distinct_positions` (see `collection_metrics`) to count positions up to them.
            tree_positions: (optional) also train on positions from self-play's search trees
                (see core.training.tree_positions). They're kept in a replay buffer of their own, and make up
                a share of each training batch, sampled from those stored in the replay buffer's span of
                self-play (see `tree_sample_mask`). Needs MCTS self-play evaluators.
            advance_every: (optional) self-play plays the moves whose searches are done every this many
                cycles (see `collect_steps`). Playing a move is computed for every environment,
                whether its search is done or not (under vmap, a select), all but moving the
                search trees on, which is done for the moves played only (see `moves_at_once`):
                checking less often costs a little less; a search done in between goes on until
                the check, at most
                `advance_every - 1` cycles more. By default, every evaluator's `search_cycles`: with
                searches of a fixed length, a check as each is done, as if every environment searched
                and played in step. With searches of varying lengths (e.g. MCTS's
                `count_reused_visits`), a few cycles balance the two: about
                sqrt(2 * mean search cycles * cost of a check / cost of a cycle).
            moves_at_once: (optional) when self-play plays moves, the evaluator's states (e.g. the
                search trees) of the environments that played move on this many at a time (see
                `core.pytree.map_where`); the others' aren't touched. Each time costs a loop trip,
                and this many environments' work however few are left. Doesn't change the results.
            extract_model_params_fn: (optional) function to extract model parameters from TrainState
            monitor: (optional) `seaglass.Monitor` to log metrics and test episodes to (see a tester's `episode_fn`),
                e.g. `core.monitoring.Monitor`, whose runs are drawn with turbozero's views
                - it logs to $SEAGLASS_URL, else slick's; a run is created on the first `train_loop`,
                  and later calls (e.g. continuing from `initial_state`) keep logging to it
            ckpt_dir: directory to save checkpoints
            max_checkpoints: maximum number of checkpoints to keep
            keep_every: (optional) also keep every checkpoint whose epoch is a multiple of this, beyond `max_checkpoints`
            save_state_at: (optional) numbers of epochs done after which to save the whole training state to
                `ckpt_dir` too (see `save_state`), to continue the run from (see `resume`) or fork new runs
                from (see `train_loop`'s `fork_from`). Kept.
            save_state_every: (optional) also save the whole training state after every this many epochs,
                keeping only the newest of these, to `resume` the run from if it stops. A state takes
                as much disk space as the replay buffers' device memory.
            lr_plateau: (optional) the `optimizer`'s learning rate, if it's an `LRPlateau`: after
                every epoch, the rule is applied to the epoch's training loss (its mean `loss`),
                dropping to the next rate when the current one plateaus, and with its `stop`, ending
                training once the last does. It logs `lr_plateau_improvement`, the fitted
                improvement, every epoch the window is full; `lr_drop`, the new rate, at the epoch
                whose loss made it drop; and `lr_last_plateau`, the last rate, when that plateaus.
                A continued run carries on with the rule's progress (see `save_state`); a fork
                starts again from the first rate (see `restart_schedules`).
            lr_plateau_metric: with `lr_plateau`, what the rule watches, lower being better:
                "loss", the epoch's training loss, or the name of a metric a tester logs every
                epoch (e.g. a `ScoredPositionsTester`'s "engine_value_mse", on fixed positions
                whose targets don't move as the network improves). For a tester's metric the rule
                is applied after the epoch's tests, and an epoch that didn't log it is an error.
            growth: (optional) bigger networks to grow into (see `core.training.growth.Growth`):
                `nn` plays self-play first, and each of them takes over in turn, training alongside
                the one playing until it does, on the same replay buffer. Each network's
                checkpoints go to `<ckpt_dir>/<its name>`, and its metrics, its training's and its
                testers', are logged as `<its name>/<metric>`; those of the network playing are
                also logged as they are without growth. `testers` and `evaluator_test` are `nn`'s.
            waiting_train_steps: (optional) with `growth`, the training steps an epoch of the
                network that trains without playing (by default `train_steps_per_epoch`, as the
                one playing): fewer charge its training what the run could spend on it, e.g. the
                steps its own play would afford for the same compute
            train_retired: (optional) with `growth`, the networks that played before the one
                playing keep training on the replay buffer, `train_steps_per_epoch` steps an
                epoch each, without playing: what a smaller network makes of a bigger one's
                games (not compute-optimal). Not part of a saved state: a run or branch turns
                it on or off from any state, and a retired network takes up its training where
                it left it
            train_playing: (optional) with `growth`, whether the network playing self-play
                trains (default) or stays as it is, weights and weight averages: a fixed teacher
                whose games the networks before it learn from with `train_retired`. Not part of
                a saved state, as `train_retired`
            averages: (optional) horizons, in epochs, of averages of each network's weights
                (as KataGo plays and rates averaged weights while training at a constant learning
                rate): after each epoch a network trains, each of its averages moves towards its
                new weights by `average_share` (an exponential moving average whose horizon
                grows from 1 to the average's with `average_warmup`). Training itself is
                unchanged. Each average's checkpoints go to `avg<horizon>` in the network's
                checkpoint directory (see `average_name`), and testers that are `stateless`
                (e.g. a `ScoredPositionsTester`) also run on it, logging `avg<horizon>/<metric>`.
                A saved state keeps them (see `save_state`); a state saved without them, or with
                other horizons, starts them at its weights. BatchNorm statistics aren't
                averaged: an average plays with the network's current ones.
            selfplay_average: (optional) self-play plays with this average (one of
                `averages`) of the network playing, rather than its current weights
            average_warmup: the epochs an average takes in for each epoch its horizon grows
                by, from 1 to its own (see `average_share`); 1 makes it the plain mean of the
                epochs so far until its horizon
            log_moves: (optional) a directory to keep every move self-play plays in, whether
                its search trains or not, an epoch's moves to a file (see
                `core.memory.move_log`). It doesn't change training. A new directory, or one
                without moves: the log isn't saved with the training state, so a run that keeps
                its moves doesn't continue into the same directory
            extra_config: (optional) extra config to record with the monitor's run
        """
        # environment
        self.env_step_fn = env_step_fn
        self.env_init_fn = env_init_fn
        self.test_env_init_fn = (
            test_env_init_fn if test_env_init_fn is not None else env_init_fn
        )
        self.max_episode_steps = max_episode_steps
        self.template_env_state = self.make_template_env_state()
        # nn
        self.state_to_nn_input_fn = state_to_nn_input_fn
        self.nn = nn
        self.nn_static = eqx.filter(nn, eqx.is_inexact_array, inverse=True)
        self.nn_state = nn_state
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        init_opt_state = jax.jit(optimizer.init)(eqx.filter(nn, eqx.is_inexact_array))
        # the injected hyperparameters' initial values, for the config
        self.optimizer_config = {
            name: _to_config(value)
            for name, value in injected_hyperparams(init_opt_state).items()
        }
        # those no schedule computes: a saved optimizer state's would stay (see
        # `with_hyperparams`), so a run starting from one takes these
        scheduled = injected_hyperparams(init_opt_state, scheduled_only=True)
        self.fixed_hyperparams = {
            name: value
            for name, value in injected_hyperparams(init_opt_state).items()
            if name not in scheduled
        }
        if (lr_plateau is not None) != bool(lr_plateau_states(init_opt_state)):
            raise ValueError(
                "pass lr_plateau with an optimizer whose learning rate it is, and only then"
            )
        self.lr_plateau = lr_plateau
        self.lr_plateau_metric = lr_plateau_metric
        self.extract_model_params_fn = extract_model_params_fn
        # data collection
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.collection_steps_per_epoch = collection_steps_per_epoch
        self.memory_buffer = memory_buffer
        # a single evaluator is a schedule with one stage
        self.collection_schedule = (
            evaluator
            if isinstance(evaluator, EvaluatorSchedule)
            else EvaluatorSchedule([(0, evaluator)])
        )
        self.transform_fns = data_transform_fns
        if advance_every is not None and advance_every < 1:
            raise ValueError(f"advance_every must be positive, got {advance_every}")
        self.advance_every = advance_every
        if moves_at_once < 1:
            raise ValueError(f"moves_at_once must be positive, got {moves_at_once}")
        self.moves_at_once = moves_at_once
        self.exploration = exploration
        self.resignation = resignation
        if endgame_solve is not None and resignation is not None:
            raise ValueError("endgame_solve doesn't go with resignation")
        self.endgame_solve = endgame_solve
        self.episode_metrics_fn = episode_metrics_fn
        if tree_positions is not None and not all(
            isinstance(e, MCTS) for _, e in self.collection_schedule.stages
        ):
            raise ValueError("tree_positions needs MCTS self-play evaluators")
        self.tree_positions = tree_positions
        self.tree_buffer = (
            EpisodeReplayBuffer(capacity=tree_positions.capacity)
            if tree_positions is not None
            else None
        )
        self.count_distinct_observations = jax.jit(
            partial(
                self.memory_buffer.count_distinct_observations,
                transform_fns=tuple(self.transform_fns),
            )
        )
        # training
        self.train_steps_per_epoch = train_steps_per_epoch
        if train_steps_per_epoch == 0 and (
            lr_plateau is not None or growth is not None
        ):
            raise ValueError("lr_plateau and growth need training steps")
        if log_moves is not None and warmup_steps:
            raise ValueError("log_moves doesn't go with warmup_steps")
        self.log_moves = log_moves
        self.move_log_writer: MoveLogWriter | None = None
        self.train_batch_size = train_batch_size
        if grad_accum_steps < 1 or train_batch_size % grad_accum_steps:
            raise ValueError(
                f"grad_accum_steps must be positive and divide train_batch_size, "
                f"{train_batch_size}, got {grad_accum_steps}"
            )
        self.grad_accum_steps = grad_accum_steps
        self.micro_batch_size = train_batch_size // grad_accum_steps
        self.lr_drop = lr_drop
        if fresh_positions % self.micro_batch_size:
            raise ValueError(
                f"fresh_positions ({fresh_positions}) must be a multiple of the micro-batch "
                f"({self.micro_batch_size})"
            )
        self.fresh_positions = fresh_positions
        # micro-batches to measure the noise scale on at a weight average, every epoch
        self.noise_measure = noise_measure
        if lr_plateau is not None and train_steps_per_epoch < 1:
            raise ValueError("lr_plateau needs training steps, to measure the loss")
        if waiting_train_steps is not None and (
            growth is None or waiting_train_steps < 1
        ):
            raise ValueError(
                "waiting_train_steps is a positive number of steps, for a run with growth"
            )
        self.waiting_train_steps = (
            waiting_train_steps
            if waiting_train_steps is not None
            else train_steps_per_epoch
        )
        if train_retired and growth is None:
            raise ValueError("train_retired is for a run with growth")
        self.train_retired = train_retired
        if not train_playing and growth is None:
            raise ValueError(
                "a network that plays without training is for a run with growth"
            )
        if not train_playing and lr_plateau is not None:
            raise ValueError(
                "the learning rate rule follows the network playing, which doesn't train"
            )
        self.train_playing = train_playing
        # testing
        self.testers = testers
        if evaluator_test is None:
            if len(self.collection_schedule.stages) > 1:
                raise ValueError(
                    "pass an evaluator_test with a schedule of self-play evaluators"
                )
            evaluator_test = self.collection_schedule.at(0)
        self.evaluator_test = evaluator_test
        self.step_test = partial(
            step_env_and_evaluator,
            evaluator=self.evaluator_test,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.test_env_init_fn,
            max_steps=self.max_episode_steps,
        )
        # growth: the networks, the Trainer's own first
        self.growth = growth
        own = Learner(
            name=growth.first if growth is not None else "",
            nn=nn,
            evaluator=self.collection_schedule,
            evaluator_test=evaluator_test,
            testers=testers,
            nn_state=nn_state,
        )
        self.learners: tuple[Learner, ...] = (
            (own, *growth.learners) if growth is not None else (own,)
        )
        for learner in self.learners[1:]:
            if tree_positions is not None and not all(
                isinstance(e, MCTS) for _, e in learner.schedule.stages
            ):
                raise ValueError("tree_positions needs MCTS self-play evaluators")
            if lr_plateau is not None:
                state = jax.eval_shape(
                    optimizer.init, eqx.filter(learner.nn, eqx.is_inexact_array)
                )
                if not lr_plateau_states(state):
                    raise ValueError(
                        f"{learner.name}: the optimizer's learning rate must be lr_plateau"
                    )
        # weight averages
        self.averages = tuple(float(h) for h in averages)
        if any(h < 1 for h in self.averages) or len(set(self.averages)) != len(
            self.averages
        ):
            raise ValueError(
                f"averages are distinct horizons of at least 1 epoch, got {list(averages)}"
            )
        if selfplay_average is not None and selfplay_average not in self.averages:
            raise ValueError(
                f"selfplay_average must be one of averages {list(self.averages)}, got "
                f"{selfplay_average}"
            )
        self.selfplay_average = selfplay_average
        if average_warmup <= 0:
            raise ValueError(f"average_warmup must be positive, got {average_warmup}")
        self.average_warmup = float(average_warmup)
        # checkpoints
        self.ckpt_dir = ckpt_dir
        self.max_checkpoints = max_checkpoints
        self.keep_every = keep_every
        self.save_state_at = frozenset(save_state_at)
        self.save_state_every = save_state_every
        for d in self.ckpt_dirs:
            os.makedirs(d, exist_ok=True)
        # monitor
        self.monitor = monitor
        self.extra_config = extra_config if extra_config is not None else {}
        # the saved state this run was forked from (see `train_loop`), recorded in the monitor's
        # config and in states the run saves
        self.parent: dict | None = None

    @property
    def ckpt_dirs(self) -> tuple[str, ...]:
        """Each network's checkpoint directory: `ckpt_dir`, or with `growth`, a directory in it
        for each network, by its name."""
        if self.growth is None:
            return (self.ckpt_dir,)
        return tuple(os.path.join(self.ckpt_dir, name) for name in self.growth.names)

    def init_train_state(self, learner: int = 0) -> TrainState:
        """Initializes the training state (params, optimizer, etc.) from `nn` and `nn_state`.

        Args:
            learner: (optional) with `growth`, the network's index (`nn` is 0)

        Returns:
            TrainState: initialized training state
        """
        net = self.learners[learner]
        params = eqx.filter(net.nn, eqx.is_inexact_array)
        return TrainState(
            params=params,
            nn_state=net.nn_state,
            opt_state=self.optimizer.init(params),
            step=jnp.array(0, dtype=jnp.int32),
        )

    def with_fixed_hyperparams(self, train_state: TrainState, name: str) -> TrainState:
        """`train_state` with its optimizer state's unscheduled hyperparameters (e.g. weight
        decay, momentum) this trainer's own (see `with_hyperparams`), saying which of them
        change, e.g. from a saved state's. `name` names the network in that message."""
        before = injected_hyperparams(train_state.opt_state)
        opt_state = with_hyperparams(train_state.opt_state, self.fixed_hyperparams)
        changed = [
            f"{k} {_to_config(before[k])} -> {_to_config(v)}"
            for k, v in injected_hyperparams(opt_state).items()
            if k in self.fixed_hyperparams and _to_config(before[k]) != _to_config(v)
        ]
        if changed:
            print(f"{name}: the optimizer's {', '.join(changed)}", flush=True)
        return replace(train_state, opt_state=opt_state)

    def steps_of(self, learner: int, playing: int) -> int:
        """The training steps an epoch of a network that trains: the one waiting's
        `waiting_train_steps`; the one playing's, and with `train_retired` those that played
        before, `train_steps_per_epoch`."""
        return (
            self.waiting_train_steps
            if learner == playing + 1
            else self.train_steps_per_epoch
        )

    def average_dirs(self, learner: int = 0) -> tuple[str, ...]:
        """A network's weight averages' checkpoint directories, one per horizon of `averages`
        (see `average_name`)."""
        return tuple(
            os.path.join(self.ckpt_dirs[learner], average_name(h))
            for h in self.averages
        )

    def fresh_averages(self, train_state: TrainState) -> tuple[Any, ...]:
        """A network's weight averages before any epoch is in them: copies of its weights."""
        return tuple(jax.tree.map(jnp.copy, train_state.params) for _ in self.averages)

    def selfplay_params(self, train_state: TrainState, averages: Sequence[Any]) -> Any:
        """The model parameters self-play plays with: the network's, or with `selfplay_average`,
        that average's."""
        if self.selfplay_average is None:
            return self.extract_model_params_fn(train_state)
        average = averages[self.averages.index(self.selfplay_average)]
        return self.extract_model_params_fn(replace(train_state, params=average))

    def get_config(self):
        """Returns a dictionary of the configuration of the trainer. Used for logging to the monitor.

        With a schedule of several self-play evaluators, `evaluator_train` is the first, and
        `collection_schedule` lists them all.
        """
        evaluator_train = self.collection_schedule.at(0)
        schedule = self.collection_schedule
        return {
            "batch_size": self.batch_size,
            "train_batch_size": self.train_batch_size,
            "warmup_steps": self.warmup_steps,
            "collection_steps_per_epoch": self.collection_steps_per_epoch,
            "advance_every": self.advance_every,
            "train_steps_per_epoch": self.train_steps_per_epoch,
            **(
                {"waiting_train_steps": self.waiting_train_steps}
                if self.waiting_train_steps != self.train_steps_per_epoch
                else {}
            ),
            **({"train_retired": True} if self.train_retired else {}),
            **({"train_playing": False} if not self.train_playing else {}),
            "max_episode_steps": self.max_episode_steps,
            "nn": self.nn.__class__.__name__,
            "nn_params": sum(
                x.size
                for x in jax.tree.leaves(eqx.filter(self.nn, eqx.is_inexact_array))
            ),
            "loss_fn": describe_fn(self.loss_fn),
            "optimizer": self.optimizer_config or None,
            **(
                {
                    "lr_plateau": self.lr_plateau.get_config(),
                    "lr_plateau_metric": self.lr_plateau_metric,
                }
                if self.lr_plateau is not None
                else {}
            ),
            "data_transforms": len(self.transform_fns),
            **(
                {"grad_accum_steps": self.grad_accum_steps}
                if self.grad_accum_steps > 1
                else {}
            ),
            **(
                {"lr_drop": self.lr_drop.get_config()}
                if self.lr_drop is not None
                else {}
            ),
            "evaluator_train": evaluator_train.__class__.__name__,
            "evaluator_train_config": evaluator_train.get_config(),
            **(
                {"collection_schedule": schedule.get_config()}
                if len(schedule.stages) > 1
                else {}
            ),
            "evaluator_test": self.evaluator_test.__class__.__name__,
            "evaluator_test_config": self.evaluator_test.get_config(),
            "exploration_config": self.exploration.get_config()
            if self.exploration is not None
            else None,
            **(
                {"resignation": self.resignation.get_config()}
                if self.resignation is not None
                else {}
            ),
            **(
                {"endgame_solve": self.endgame_solve.get_config()}
                if self.endgame_solve is not None
                else {}
            ),
            "episode_metrics_fn": getattr(self.episode_metrics_fn, "__name__", None)
            if self.episode_metrics_fn is not None
            else None,
            "memory_buffer": self.memory_buffer.__class__.__name__,
            "memory_buffer_config": self.memory_buffer.get_config(),
            "tree_positions": self.tree_positions.get_config()
            if self.tree_positions is not None
            else None,
            "testers": [tester.get_config() for tester in self.testers],
            **({"growth": self.growth.get_config()} if self.growth is not None else {}),
            **(
                {
                    "averages": list(self.averages),
                    "selfplay_average": self.selfplay_average,
                    "average_warmup": self.average_warmup,
                }
                if self.averages
                else {}
            ),
        }

    def move_keys(
        self, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        """Splits a move's key (see `CollectionState.move_key`).

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array | None]: the search's key, the key for
                choosing and playing the move (and starting a new game after it), and the key
                for storing tree positions (None without them)
        """
        tree_key = None
        if self.tree_positions is not None:
            key, tree_key = jax.random.split(key)
        key, evaluate_key = jax.random.split(key)
        return evaluate_key, key, tree_key

    def start_searches(
        self,
        key: jax.Array,
        state: CollectionState,
        evaluator: Evaluator | None = None,
    ) -> CollectionState:
        """Starts the search for a move, if one starts: if the last one's `advance` played its
        move (or none was in progress). `key` becomes the move's key (see
        `CollectionState.move_key`), and the evaluator starts its search (see
        `Evaluator.start_search`). Otherwise, the state is left as it is.

        Args:
            key: rng, for a move starting
            state: current collection state
            evaluator: (optional) the self-play evaluator, whose states `state` holds. If not
                provided, the first in the schedule.

        Returns:
            CollectionState: updated collection state
        """
        if evaluator is None:
            evaluator = self.collection_schedule.at(0)
        starting = state.search_cycles == 0
        move_key = jnp.where(starting, key, state.move_key)
        evaluate_key, _, _ = self.move_keys(move_key)
        return replace(
            state,
            eval_state=evaluator.start_search(
                evaluate_key, state.eval_state, active=starting
            ),
            move_key=move_key,
        )

    def search(
        self,
        state: CollectionState,
        params: Any,
        evaluator: Evaluator | None = None,
    ) -> CollectionState:
        """Runs a cycle of the search for the move in progress (see `Evaluator.search_cycle`),
        which `start_searches` started. A search that is done goes on until `advance` plays its
        move: under vmap, its cycle is computed anyway.

        Args:
            state: current collection state
            params: model parameters
            evaluator: (optional) the self-play evaluator, whose states `state` holds. If not
                provided, the first in the schedule.

        Returns:
            CollectionState: updated collection state
        """
        if evaluator is None:
            evaluator = self.collection_schedule.at(0)
        evaluate_key, _, _ = self.move_keys(state.move_key)
        eval_state = evaluator.search_cycle(
            evaluate_key,
            state.eval_state,
            state.search_cycles,
            state.env_state,
            state.metadata,
            params,
            self.env_step_fn,
        )
        return replace(
            state, eval_state=eval_state, search_cycles=state.search_cycles + 1
        )

    def advance(
        self,
        state: CollectionState,
        params: Any,
        evaluator: Evaluator | None = None,
    ) -> CollectionState:
        """Plays the move whose search is done, if it is (see `Evaluator.search_done`): see
        `play_moves`, then the evaluator's state moves on to the next move's (see
        `Evaluator.next_state`).

        If the search isn't done, the state is left as it is. Under vmap, where that differs
        across environments, both are computed: `collect_steps` moves the evaluator's states of
        the environments that played on its own instead.

        Args:
            state: current collection state
            params: model parameters
            evaluator: (optional) the self-play evaluator, whose states `state` holds. If not
                provided, the first in the schedule.

        Returns:
            CollectionState: updated collection state
        """
        if evaluator is None:
            evaluator = self.collection_schedule.at(0)
        state, (done, action, ended) = self.play_moves(state, params, evaluator)
        return replace(
            state,
            eval_state=evaluator.next_state(
                state.eval_state, action, ended, active=done
            ),
        )

    def play_moves(
        self,
        state: CollectionState,
        params: Any,
        evaluator: Evaluator | None = None,
    ) -> tuple[CollectionState, tuple[jax.Array, jax.Array, jax.Array]]:
        """Plays the move whose search is done, if it is (see `Evaluator.search_done`), all but
        moving the evaluator's state on (see `advance`):

        - Stores experience in replay buffer, if the evaluator `trains_on` the search.
        - Resets the environment if the episode is terminated.

        If the search isn't done, the state is left as it is. Under vmap, where that differs
        across environments, both are computed, but the replay buffers are only written where a
        move is played, rather than selected between whole buffers.

        Args:
            state: current collection state
            params: model parameters
            evaluator: (optional) the self-play evaluator, whose states `state` holds. If not
                provided, the first in the schedule.

        Returns:
            tuple[CollectionState, tuple[jax.Array, jax.Array, jax.Array]]: updated collection
                state, its evaluator's state as the search left it; and for
                `Evaluator.next_state`, whether a move was played, the move, and whether the
                episode ended
        """
        if evaluator is None:
            evaluator = self.collection_schedule.at(0)
        done = evaluator.search_done(state.eval_state, state.search_cycles)
        # the moves whose positions are trained on (see `Evaluator.trains_on`); the others are
        # played, and count towards their game's length and outcome, without being stored
        stored = done & evaluator.trains_on(state.eval_state)
        evaluate_key, key, tree_key = self.move_keys(state.move_key)
        output = evaluator.search_output(
            evaluate_key,
            state.eval_state,
            state.env_state,
            state.metadata,
            params,
            self.env_step_fn,
        )
        search_value = evaluator.get_value(output.eval_state)
        resigned = (
            done & self.resignation.resigns(state.resign_state, search_value)
            if self.resignation is not None and state.resign_state is not None
            else jnp.array(False)
        )
        # exact endgame labels: a game ending for its solves ends before its move, as a resigned
        # one does
        solve_move = None
        stopped = jnp.array(False)
        if self.endgame_solve is not None and state.pending is not None:
            solve_move = self.endgame_solve.move(
                jax.random.fold_in(key, 3), state.pending, state.env_state, done, stored
            )
            stopped = solve_move.end
        # step environment and evaluator
        (
            eval_output,
            new_env_state,
            new_metadata,
            terminated,
            truncated,
            rewards,
            searched_eval_state,
        ) = step_with_output(
            key=key,
            output=output,
            env_state=state.env_state,
            env_state_metadata=state.metadata,
            evaluator=evaluator,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn,
            max_steps=self.max_episode_steps,
            choose_action=self.exploration.choose_action
            if self.exploration is not None
            else None,
            active=done,
            end=resigned | stopped,
            step_evaluator=False,
        )
        terminated, truncated = done & terminated, done & truncated
        resign_state = state.resign_state
        if self.resignation is not None and resign_state is not None:
            resign_state = self.resignation.update(
                jax.random.fold_in(key, 1),
                resign_state,
                moved=done,
                value=search_value,
                player=state.metadata.cur_player_id,
                resigned=resigned,
                terminated=terminated,
                ended=terminated | truncated | resigned,
                rewards=rewards,
            )
            # the player to move loses, the other wins
            rewards = jnp.where(
                resigned,
                jnp.where(
                    jnp.arange(rewards.shape[-1]) == state.metadata.cur_player_id,
                    -1.0,
                    1.0,
                ).astype(rewards.dtype),
                rewards,
            )
            terminated = terminated | resigned

        # store experience in replay buffer
        written = state.buffer_state.next_idx
        buffer_state = self.memory_buffer.add_experience(
            state=state.buffer_state,
            experience=BaseExperience(
                observation_nn=self.state_to_nn_input_fn(state.env_state),
                policy_mask=state.metadata.action_mask,
                policy_weights=eval_output.policy_weights,
                reward=jnp.empty_like(state.metadata.rewards),
                cur_player_id=state.metadata.cur_player_id,
                search_value=search_value,
            ),
            valid=stored,
        )
        pending = state.pending
        # the games that count as episodes now: with exact endgame labels, a game past its solve
        # point counted when that was labelled
        counted = terminated
        if (
            self.endgame_solve is not None
            and pending is not None
            and solve_move is not None
        ):
            counted = terminated & ~pending.past
            # an entry written over is no longer a waiting solve's
            pending = replace(
                pending,
                entry_solve=pending.entry_solve.at[
                    jnp.where(stored, written, buffer_capacity(buffer_state))
                ].set(-1, mode="drop"),
            )
            # a solved position's entries wait in the buffer for its result (before a game that
            # ends with this move assigns its own result to them)
            buffer_state, entries = self.memory_buffer.set_aside(
                buffer_state, solve_move.solve
            )
            buffer_state = where_tree(
                solve_move.drop, self.memory_buffer.truncate(buffer_state), buffer_state
            )
            pending = self.endgame_solve.add(
                pending, state.env_state, entries, solve_move
            )
            pending = self.endgame_solve.next_game(
                jax.random.fold_in(key, 2), pending, terminated | truncated | stopped
            )
        # assign rewards to buffer if episode is terminated
        buffer_state = where_tree(
            terminated,
            self.memory_buffer.assign_rewards(buffer_state, rewards),
            buffer_state,
        )
        # discard the episode's experiences if it hit the step limit without terminating
        # (`truncated` is never set together with `terminated`)
        buffer_state = where_tree(
            truncated, self.memory_buffer.truncate(buffer_state), buffer_state
        )
        if tree_key is not None:
            # the tree the next search reuses: the played move's subtree, unless the episode ended
            reused = jnp.where(
                terminated | truncated | stopped,
                searched_eval_state.NULL_INDEX,
                searched_eval_state.edge_map[
                    searched_eval_state.ROOT_INDEX, eval_output.action
                ],
            )
            state = self.store_tree_positions(
                tree_key, state, searched_eval_state, reused, evaluator, valid=stored
            )
        move_log = state.move_log
        if move_log is not None:
            move_log = log_move(
                move_log,
                LoggedMove(
                    observation=self.state_to_nn_input_fn(state.env_state),
                    policy=eval_output.policy_weights,
                    legal=state.metadata.action_mask,
                    value=search_value,
                    action=jnp.where(stopped, -1, eval_output.action),
                    player=state.metadata.cur_player_id,
                    full=evaluator.trains_on(state.eval_state),
                    ply=state.metadata.step,
                    game=jnp.zeros((), jnp.int32),
                    end=jnp.select(
                        [terminated, stopped, truncated],
                        [ENDED, STOPPED, TRUNCATED],
                        GOES_ON,
                    ),
                    result=jnp.where(
                        terminated, rewards[state.metadata.cur_player_id], 0.0
                    ),
                ),
                done,
            )
        # return new collection state
        state = replace(
            state,
            # as `search_output` left it, which is as the search did (an evaluator whose
            # `search_output` does the search, see `Evaluator.search_cycle`, is done every
            # cycle)
            eval_state=searched_eval_state,
            env_state=where_tree(done, new_env_state, state.env_state),
            buffer_state=buffer_state,
            metadata=where_tree(done, new_metadata, state.metadata),
            episodes=state.episodes + counted,
            moves=state.moves + done,
            episode_metrics=self.add_episode_metrics(
                state.episode_metrics, rewards, counted
            ),
            search_cycles=jnp.where(done, 0, state.search_cycles),
            resign_state=resign_state,
            pending=pending,
            move_log=move_log,
        )
        # the games that ended: `terminated` counts a resignation, `stopped` a game that
        # stopped for its solves
        return state, (done, eval_output.action, terminated | truncated | stopped)

    def collect(
        self,
        key: jax.Array,
        state: CollectionState,
        params: Any,
        evaluator: Evaluator | None = None,
    ) -> CollectionState:
        """A self-play cycle: `start_searches` (with `key`), `search`, then `advance`. With an
        evaluator that doesn't search a cycle at a time (see `Evaluator.search_cycle`), a move.

        Args: see `start_searches` and `search`.

        Returns:
            CollectionState: updated collection state
        """
        state = self.start_searches(key, state, evaluator)
        state = self.search(state, params, evaluator)
        return self.advance(state, params, evaluator)

    def store_tree_positions(
        self,
        key: jax.Array,
        state: CollectionState,
        tree: Any,
        reused: jax.Array,
        evaluator: Evaluator,
        valid: ArrayLike = True,
    ) -> CollectionState:
        """Stores positions from a self-play search tree in the tree position replay buffer
        (see `tree_positions`).

        Args:
            key: rng
            state: current collection state
            tree: the search tree, after the search
            reused: the root child whose subtree the next search reuses, NULL_INDEX for none
            evaluator: the self-play evaluator that searched the tree
            valid: (optional) whether to store them: if not, the state is left as it is

        Returns:
            CollectionState: updated collection state
        """
        assert self.tree_positions is not None and self.tree_buffer is not None
        assert state.tree_buffer_state is not None and state.tree_written_at is not None
        assert isinstance(evaluator, MCTS)
        indices, selected = select_nodes(
            key,
            tree,
            self.tree_positions.per_move,
            self.tree_positions.min_visits,
            self.tree_positions.most_visited,
            reused
            if self.tree_positions.discarded_only and evaluator.persist_tree
            else None,
        )
        valid = jnp.logical_and(valid, selected)
        experiences = tree_experiences(
            tree, indices, self.env_step_fn, self.state_to_nn_input_fn
        )
        written = self.tree_buffer.write_indices(state.tree_buffer_state, valid)
        return replace(
            state,
            tree_buffer_state=self.tree_buffer.add_experiences(
                state.tree_buffer_state, experiences, valid
            ),
            tree_written_at=state.tree_written_at.at[written].set(
                state.moves, mode="drop"
            ),
            tree_positions=state.tree_positions + valid.sum(),
            tree_visits=state.tree_visits
            + jnp.where(valid, tree.data.n[indices], 0).sum(),
        )

    # the evaluator is a static argument, rather than read from `self` while tracing: `self` is
    # static too, but hashes by identity, so a compiled function reading an attribute of `self`
    # would keep running for a new value of the attribute.
    # `state` is donated: it holds the replay buffers, so self-play updates them in place rather
    # than in a second copy
    @partial(
        jax.jit,
        static_argnums=(0, 4),
        static_argnames=("evaluator",),
        donate_argnames=("state",),
    )
    def collect_steps(
        self,
        key: jax.Array,
        state: CollectionState,
        params: Any,
        num_steps: int,
        *,
        evaluator: Evaluator | None = None,
    ) -> CollectionState:
        """Collects self-play data for the compute of `num_steps` moves in every environment: as
        many cycles as `num_steps` moves' searches take at most (see `Evaluator.search_cycles`).

        Each cycle, every environment runs a cycle of its move's search (`search`); every
        `advance_every` cycles (see `Trainer`), those whose search is done play their moves
        (`advance`, its evaluator's states moved on only for the moves played, `moves_at_once`
        at a time). With searches that all take the same cycles, from environments that start
        together, and the default `advance_every`, that's `num_steps` moves in each, as many
        cycles apart.

        Compiled once for each evaluator.

        Args:
            key: rng, one key per environment
            state: current collection state, donated: its arrays are deleted, and can't be used
                after the call
            params: model parameters
            num_steps: number of moves' compute to collect for
            evaluator: (optional) the self-play evaluator, whose states `state` holds. If not
                provided, the first in the schedule.

        Returns:
            CollectionState: updated collection state
        """
        if evaluator is None:
            evaluator = self.collection_schedule.at(0)
        num_cycles = num_steps * evaluator.search_cycles()
        if num_cycles == 0:
            return state
        advance_every = (
            evaluator.search_cycles()
            if self.advance_every is None
            else self.advance_every
        )

        batch = key.shape[0]
        # in each environment, the i-th move to start in this call gets the i-th key
        keys = jax.vmap(lambda k: jax.random.split(k, num_cycles))(key)

        def search_env(
            keys: jax.Array, state: CollectionState, started: jax.Array, num: int
        ) -> tuple[CollectionState, jax.Array]:
            """`num` cycles of search in an environment, without playing any move: they only
            write the trees, in place. Moves start only in the first (after an `advance`, or
            at the call's start), so their searches start before it, rather than in every
            cycle."""
            starting = state.search_cycles == 0
            state = self.start_searches(keys[started], state, evaluator)
            state = jax.lax.fori_loop(
                0, num, lambda _, s: self.search(s, params, evaluator), state
            )
            return state, started + starting

        def search(
            state: CollectionState, started: jax.Array, num: int
        ) -> tuple[CollectionState, jax.Array]:
            return jax.vmap(partial(search_env, num=num))(keys, state, started)

        def chunk(
            _: jax.Array, carry: tuple[CollectionState, jax.Array]
        ) -> tuple[CollectionState, jax.Array]:
            state, started = carry
            state, started = search(state, started, advance_every)
            state, (done, action, ended) = jax.vmap(
                lambda s: self.play_moves(s, params, evaluator)
            )(state)
            # everything read from the evaluator's states (to play the moves), then they move
            # on in place: the barrier makes the loop that writes them depend on every read.
            # XLA otherwise copies the arrays read to keep them for reads it doesn't order
            # first (see `Tree.set_child`)
            state, (done, action, ended) = jax.lax.optimization_barrier(
                (state, (done, action, ended))
            )
            # the evaluator's states (e.g. the search trees) move on only where a move was
            # played (see `advance`): under vmap, every one would be read and written
            eval_state = map_where(
                evaluator.next_state,
                done,
                state.eval_state,
                action,
                ended,
                size=self.moves_at_once,
            )
            return replace(state, eval_state=eval_state), started

        # every `advance_every` cycles, the moves whose searches are done are played: loops
        # within a loop rather than a lax.cond every cycle, through which XLA would copy the
        # whole state (the trees and the replay buffers)
        state, started = jax.lax.fori_loop(
            0,
            num_cycles // advance_every,
            chunk,
            (state, jnp.zeros(batch, dtype=jnp.int32)),
        )
        if num_cycles % advance_every:
            state, started = search(state, started, num_cycles % advance_every)
        return state

    @partial(jax.jit, static_argnums=(0,), donate_argnames=("state",))
    def label_solved(
        self, state: CollectionState, results: jax.Array, counts: jax.Array
    ) -> CollectionState:
        """Labels the games waiting for exact solves with their results (see `endgame_solve`):
        every stored position of each game gets its game's result, and training can sample it.
        They count as episodes that terminated, in `episode_metrics_fn`'s metrics too.

        Args:
            state: collection state, donated (see `collect_steps`)
            results: (environments, slots, players) the results, by slot, from
                `EndgameSolve.solve`
            counts: (environments,) how many of each environment's oldest waiting games to label

        Returns:
            CollectionState: updated collection state
        """
        assert self.endgame_solve is not None and state.pending is not None
        endgame_solve = self.endgame_solve

        def label(
            state: CollectionState, results: jax.Array, count: jax.Array
        ) -> CollectionState:
            assert state.pending is not None
            pending, reward, populated, games = endgame_solve.label(
                state.pending,
                state.buffer_state.buffer.reward,
                state.buffer_state.populated,
                results,
                count,
            )
            zeros = {name: jnp.zeros((), jnp.float32) for name in state.episode_metrics}
            added = jax.vmap(lambda r, t: self.add_episode_metrics(zeros, r, t))(
                results, games
            )
            return replace(
                state,
                pending=pending,
                buffer_state=replace(
                    state.buffer_state,
                    buffer=replace(state.buffer_state.buffer, reward=reward),
                    populated=populated,
                ),
                episodes=state.episodes + games.sum(),
                episode_metrics={
                    name: total + added[name].sum()
                    for name, total in state.episode_metrics.items()
                },
            )

        return jax.vmap(label)(state, results, counts)

    def label_solved_games(
        self, state: CollectionState, solved: Solved
    ) -> CollectionState:
        """`label_solved` with a solve's results, if it solved anything."""
        if not len(solved.solves):
            return state
        return self.label_solved(
            state, jnp.asarray(solved.results), jnp.asarray(solved.counts)
        )

    def solve_waiting(self, pending: PendingSolves) -> Solved:
        """On the host: solves the games waiting for exact solves, in `pending` copied to the
        host (see `EndgameSolve.solve`)."""
        assert self.endgame_solve is not None
        return self.endgame_solve.solve(pending)

    def train_step(
        self,
        ts: TrainState,
        batch: BaseExperience,
        learner: int = 0,
        lr_scale: jax.Array | None = None,
    ) -> tuple[TrainState, dict]:
        """Make a single training step, on micro-batches of `micro_batch_size` samples (see
        `grad_accum_steps`): as many as the minibatch holds.

        Args:
            ts: TrainState
            batch: minibatch of experiences, a multiple of `micro_batch_size` samples
            learner: (optional) with `growth`, the index of the network `ts` is (`nn` is 0)
            lr_scale: (optional) scales the optimizer's update, as a learning rate would (see
                `lr_drop`)

        Returns:
            Tuple[TrainState, dict]: updated TrainState and metrics; with 2 or more micro-batches,
                `gns_micro_sq` and `gns_batch_sq`, their gradients' mean squared norm and their
                mean's (see `core.training.noise_scale.noise_scale_metrics`)
        """
        # calculate loss, get gradients
        nn_static = eqx.filter(
            self.learners[learner].nn, eqx.is_inexact_array, inverse=True
        )
        nn = eqx.combine(ts.params, nn_static)
        grad_fn = eqx.filter_value_and_grad(self.loss_fn, has_aux=True)
        num_micro = jax.tree.leaves(batch)[0].shape[0] // self.micro_batch_size
        gns: dict[str, jax.Array] = {}
        if num_micro == 1:
            (loss, (metrics, nn_state)), grads = grad_fn(nn, ts.nn_state, batch)
        else:
            loss, metrics, nn_state, grads, micro_sq = self.accumulate_grads(
                grad_fn, nn, ts.params, ts.nn_state, batch, num_micro
            )
            gns["gns_micro_sq"] = micro_sq
        # apply gradients
        updates, opt_state = self.optimizer.update(grads, ts.opt_state, ts.params)
        if lr_scale is not None:
            updates = jax.tree.map(lambda u: u * lr_scale.astype(u.dtype), updates)
        update_size = update_size_metrics(ts.params, updates)
        grad_norm = optax.tree.norm(grads).astype(jnp.float32)
        if gns:
            gns["gns_batch_sq"] = jnp.square(grad_norm)
        ts = replace(
            ts,
            params=optax.apply_updates(ts.params, updates),
            nn_state=nn_state,
            opt_state=opt_state,
            step=ts.step + 1,
        )
        # return updated train state and metrics, with the hyperparameters this step used
        metrics = {
            **metrics,
            **update_size,
            "loss": loss,
            "grad_norm": grad_norm,
            "nonfinite_steps": ~(jnp.isfinite(loss) & jnp.isfinite(grad_norm)),
            **gns,
            **injected_hyperparams(opt_state, scheduled_only=True),
        }
        return ts, metrics

    def accumulate_grads(
        self,
        grad_fn: Callable,
        nn: eqx.Module,
        params: Any,
        nn_state: eqx.nn.State | None,
        batch: BaseExperience,
        n: int,
    ) -> tuple[jax.Array, dict, eqx.nn.State | None, Any, jax.Array]:
        """Computes the loss and gradients of a minibatch in `n` micro-batches, one after
        another in a `jax.lax.scan` (see `grad_accum_steps` in `__init__`).

        Args:
            grad_fn: `loss_fn`'s value and gradient function
            nn: the neural network
            params: its parameters, which its gradients have the structure of
            nn_state: state of the neural network, which each micro-batch starts from
            batch: minibatch of experiences
            n: the number of micro-batches, which must divide the minibatch's size

        Returns:
            Tuple[jax.Array, dict, Optional[eqx.nn.State], Any, jax.Array]: the micro-batches' mean
                loss, mean metrics, mean state (taking the last of non-float leaves, e.g. step
                counters), mean gradients, and their gradients' mean squared norm
        """
        # micro-batch i is samples i, i + n, i + 2n, ...
        micro_batches = jax.tree.map(
            lambda x: x.reshape(x.shape[0] // n, n, *x.shape[1:]).swapaxes(0, 1),
            batch,
        )

        def micro_step(grads: Any, micro_batch: BaseExperience) -> tuple[Any, tuple]:
            (loss, (metrics, new_state)), micro_grads = grad_fn(
                nn, nn_state, micro_batch
            )
            grads = jax.tree.map(jnp.add, grads, micro_grads)
            micro_sq = jnp.square(optax.tree.norm(micro_grads).astype(jnp.float32))
            return grads, (loss, metrics, new_state, micro_sq)

        grads, (losses, metrics, states, micro_sq) = jax.lax.scan(
            micro_step, jax.tree.map(jnp.zeros_like, params), micro_batches
        )
        nn_state = jax.tree.map(
            lambda x: x.mean(axis=0) if jnp.issubdtype(x.dtype, jnp.inexact) else x[-1],
            states,
        )
        return (
            losses.mean(),
            jax.tree.map(jnp.mean, metrics),
            nn_state,
            jax.tree.map(lambda g: g / n, grads),
            micro_sq.mean(),
        )

    @partial(jax.jit, static_argnums=(0, 4), static_argnames=("learner", "batch_size"))
    def train_epoch(
        self,
        key: jax.Array,
        buffer_state: ReplayBufferState,
        train_state: TrainState,
        num_steps: int,
        tree_buffer_state: ReplayBufferState | None = None,
        tree_mask: jax.Array | None = None,
        num_tree: jax.Array | None = None,
        learner: int = 0,
        batch_size: int | None = None,
        lr_scale: jax.Array | None = None,
    ) -> tuple[TrainState, dict]:
        """Performs `num_steps` training steps, compiled into a single `jax.lax.scan`.

        Each step samples a minibatch from the replay buffer, maps each sample through a random data
        transform (see `data_transform_fns`), and updates the parameters.

        Does not check that the replay buffer holds enough entries to sample, see `train_steps`.

        Args:
            key: rng
            buffer_state: replay buffer state
            train_state: current training state
            num_steps: number of training steps to perform
            tree_buffer_state: (optional) state of the search tree positions' replay buffer
                (see `tree_positions`), to sample the first `num_tree` samples of each minibatch from
            tree_mask: the tree positions that can be sampled (see `tree_sample_mask`)
            num_tree: number of tree positions in each minibatch, a traced value so that changing it
                doesn't recompile. With none to sample, it must be 0.
            learner: (optional) with `growth`, the index of the network `train_state` is
            batch_size: (optional) samples per step, a multiple of `micro_batch_size`
                (default `train_batch_size`)
            lr_scale: (optional) scales every step's update, as a learning rate would

        Returns:
            Tuple[TrainState, dict]: updated training state and metrics (mean across steps), with
                2 or more micro-batches a step, the noise scale's (see
                `core.training.noise_scale.noise_scale_metrics`)
        """
        if batch_size is None:
            batch_size = self.train_batch_size
        # the buffer doesn't change while training, so find which entries can be sampled once per epoch
        mask = self.memory_buffer.sample_mask(buffer_state)
        add_tree_positions = (
            self.tree_position_sampler(
                tree_buffer_state, tree_mask, num_tree, batch_size
            )
            if tree_buffer_state is not None
            and tree_mask is not None
            and num_tree is not None
            else None
        )

        def step(
            carry: tuple[jax.Array, TrainState], _
        ) -> tuple[tuple[jax.Array, TrainState], dict]:
            key, ts = carry
            step_key, key = jax.random.split(key)
            # sample from replay memory
            indices = jnp.unravel_index(
                self.memory_buffer.sample_indices(step_key, mask, batch_size),
                mask.shape,
            )
            batch = jax.tree.map(lambda x: x[indices], buffer_state.buffer)
            if add_tree_positions is not None:
                tree_key, key = jax.random.split(key)
                batch = add_tree_positions(tree_key, batch)
            transform_key, key = jax.random.split(key)
            batch = transform_randomly(transform_key, batch, self.transform_fns)
            # make training step
            ts, metrics = self.train_step(ts, batch, learner, lr_scale)
            return (key, ts), metrics

        (_, train_state), metrics = jax.lax.scan(
            step, (key, train_state), length=num_steps
        )
        metrics = reduce_step_metrics(metrics)
        if "gns_micro_sq" in metrics:
            metrics.update(
                noise_scale_metrics(
                    metrics.pop("gns_micro_sq"),
                    metrics.pop("gns_batch_sq"),
                    self.micro_batch_size,
                    batch_size,
                )
            )
        return train_state, metrics

    def resize_replay_buffer(
        self,
        state: CollectionState,
        capacity: int,
        tree_capacity: int | None = None,
    ) -> CollectionState:
        """Changes the capacity of a collection state's replay buffer between epochs. Training
        samples the whole buffer, so this sets the replay window: the newest `capacity` positions
        of each environment.

        Each environment keeps its newest entries, as many as fit (see
        `EpisodeReplayBuffer.resized`): shrinking drops the oldest, and growing leaves room that
        self-play fills, but can't bring back positions dropped before. Games in progress carry
        on, entries and all, and so do games waiting for exact solves (see `endgame_solve`).

        With tree positions (see `tree_positions`), their buffer is resized too, by default by the
        same factor (rounded up), so that it still holds the replay buffer's span of self-play,
        which training samples them from (see `tree_sample_mask`).

        The arrays change shape, so self-play and training compile again, once for each new
        capacity. `save_state` saves the capacities with the state, and `load_state` (so `resume`)
        takes them back; a fork takes its trainer's (see `fork_state`).

        The buffers are resized on the host and copied back to the device: that takes about the
        host memory of the old buffers and the new ones, and the device memory of both until
        `state` is dropped.

        Args:
            state: collection state, between epochs
            capacity: the replay buffer's new capacity, entries per environment
            tree_capacity: (optional) the tree position buffer's new capacity, entries per
                environment, only with tree positions. If not provided, the old one times
                `capacity` over the replay buffer's old capacity, rounded up.

        Returns:
            CollectionState: the collection state, its buffers resized
        """
        if capacity < 1 or (tree_capacity is not None and tree_capacity < 1):
            raise ValueError(
                f"capacities must be positive, got {capacity} and {tree_capacity}"
            )
        old_capacity = buffer_capacity(state.buffer_state)
        # with exact endgame labels, which waiting game each entry is: + 1, for the places left
        # empty to be none (-1)
        entry_games = (
            [_to_host(state.pending.entry_solve) + 1]
            if state.pending is not None
            else []
        )
        buffer_state, entry_games = EpisodeReplayBuffer(capacity).resized(
            jax.tree.map(_to_host, state.buffer_state), *entry_games
        )
        resized = replace(state, buffer_state=jax.tree.map(_to_device, buffer_state))
        if state.pending is not None:
            resized = replace(
                resized,
                pending=replace(
                    state.pending, entry_solve=_to_device(entry_games[0] - 1)
                ),
            )
        if state.tree_buffer_state is None or state.tree_written_at is None:
            if tree_capacity is not None:
                raise ValueError("tree_capacity needs tree positions")
            return resized
        if tree_capacity is None:
            old_tree = buffer_capacity(state.tree_buffer_state)
            tree_capacity = -(-old_tree * capacity // old_capacity)
        tree_buffer_state, (tree_written_at,) = EpisodeReplayBuffer(
            tree_capacity
        ).resized(
            jax.tree.map(_to_host, state.tree_buffer_state),
            _to_host(state.tree_written_at),
        )
        return replace(
            resized,
            tree_buffer_state=jax.tree.map(_to_device, tree_buffer_state),
            tree_written_at=_to_device(tree_written_at),
        )

    def tree_sample_mask(self, state: CollectionState) -> jax.Array:
        """Marks the tree positions (see `tree_positions`) training can sample: those stored within
        the replay buffer's span of self-play.

        The replay buffer holds each environment's newest C played positions (C its capacity, which
        `resize_replay_buffer` may have changed), its last C moves. Tree positions are sampled
        from the ones stored in as many moves, however many each move stored.

        Args:
            state: current collection state

        Returns:
            jax.Array: boolean mask, the shape of the tree position buffer's `populated`
        """
        assert self.tree_buffer is not None
        assert state.tree_buffer_state is not None and state.tree_written_at is not None
        moves = buffer_capacity(state.buffer_state)
        # positions the last move stored are 1 move old
        age = state.moves[..., None] - state.tree_written_at
        return self.tree_buffer.sample_mask(state.tree_buffer_state) & (age <= moves)

    def tree_position_sampler(
        self,
        tree_buffer_state: ReplayBufferState,
        mask: jax.Array,
        num_tree: jax.Array,
        batch_size: int,
    ) -> Callable[[jax.Array, BaseExperience], BaseExperience]:
        """Makes a function that puts tree positions (see `tree_positions`) into a minibatch: it
        replaces the first `num_tree` samples of a minibatch with ones sampled from the tree position
        replay buffer, uniformly with replacement (so however few there are).

        Args:
            tree_buffer_state: state of the tree position replay buffer, which doesn't change while
                the function is used
            mask: the tree positions to sample from (see `tree_sample_mask`)
            num_tree: number of tree positions per minibatch; with none in `mask`, it must be 0
            batch_size: samples per minibatch

        Returns:
            Callable[[jax.Array, BaseExperience], BaseExperience]: (rng, minibatch) -> minibatch
        """
        assert self.tree_buffer is not None
        tree_buffer = self.tree_buffer
        # with nothing to sample, sample anything: `num_tree` is 0, so none of it is used
        mask = mask | ~mask.any()
        is_tree = jnp.arange(batch_size) < num_tree

        def add_tree_positions(key: jax.Array, batch: BaseExperience) -> BaseExperience:
            indices = jnp.unravel_index(
                tree_buffer.sample_indices(key, mask, batch_size, replace=True),
                mask.shape,
            )
            return jax.tree.map(
                lambda x, b: jnp.where(
                    is_tree.reshape((-1,) + (1,) * (b.ndim - 1)), x[indices], b
                ),
                tree_buffer_state.buffer,
                batch,
            )

        return add_tree_positions

    def noise_scale_at(
        self,
        key: jax.Array,
        buffer_state: ReplayBufferState,
        train_state: TrainState,
        params: Any,
        num_micro: int,
        learner: int = 0,
    ) -> tuple[jax.Array, jax.Array]:
        """tr(Sigma) and |G|^2 at `params` (e.g. a weight average, where the jitter's pull back
        to the centre doesn't inflate |G|^2; see core.training.noise_scale), from the gradients
        of `num_micro` micro-batches sampled from the replay buffer as training samples them.

        In chunks of up to 16 micro-batches: each chunk's samples are gathered by one compiled
        function and their gradients summed by another, so neither holds more than a chunk."""
        mask = self.memory_buffer.sample_mask(buffer_state)
        chunk = min(16, num_micro)
        total, micro_sq = None, jnp.zeros((), jnp.float32)
        for start in range(0, num_micro, chunk):
            n = min(chunk, num_micro - start)
            gather_key, grad_key, key = jax.random.split(key, 3)
            batches = self._noise_gather(gather_key, buffer_state, mask, n)
            sums, sq = self._noise_grads(
                grad_key, train_state.nn_state, params, batches, learner=learner
            )
            total = sums if total is None else jax.tree.map(jnp.add, total, sums)
            micro_sq = micro_sq + sq
        assert total is not None
        batch_sq = jnp.square(
            optax.tree.norm(jax.tree.map(lambda t: t / num_micro, total))
        )
        return noise_scale_estimates(
            micro_sq / num_micro,
            batch_sq,
            self.micro_batch_size,
            num_micro * self.micro_batch_size,
        )

    @partial(jax.jit, static_argnums=(0, 4))
    def _noise_gather(
        self, key: jax.Array, buffer_state: ReplayBufferState, mask: jax.Array, n: int
    ) -> BaseExperience:
        """`n` micro-batches sampled from the replay buffer, stacked."""
        indices = jnp.unravel_index(
            self.memory_buffer.sample_indices(key, mask, n * self.micro_batch_size),
            mask.shape,
        )
        return jax.tree.map(
            lambda x: x[indices].reshape(n, self.micro_batch_size, *x.shape[2:]),
            buffer_state.buffer,
        )

    @partial(jax.jit, static_argnums=(0,), static_argnames=("learner",))
    def _noise_grads(
        self,
        key: jax.Array,
        nn_state: eqx.nn.State | None,
        params: Any,
        batches: BaseExperience,
        learner: int = 0,
    ) -> tuple[Any, jax.Array]:
        """The sum of the micro-batches' gradients at `params`, and of their squared norms."""
        nn_static = eqx.filter(
            self.learners[learner].nn, eqx.is_inexact_array, inverse=True
        )
        nn = eqx.combine(params, nn_static)
        grad_fn = eqx.filter_value_and_grad(self.loss_fn, has_aux=True)

        def one(carry: tuple[Any, jax.Array], xs: tuple) -> tuple[tuple, None]:
            batch, k = xs
            batch = transform_randomly(k, batch, self.transform_fns)
            _, grads = grad_fn(nn, nn_state, batch)
            grads = jax.tree.map(lambda g: g.astype(jnp.float32), grads)
            return (
                jax.tree.map(jnp.add, carry[0], grads),
                carry[1] + jnp.square(optax.tree.norm(grads)),
            ), None

        n = jax.tree.leaves(batches)[0].shape[0]
        zeros = jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
        (sums, sq), _ = jax.lax.scan(
            one,
            (zeros, jnp.zeros((), jnp.float32)),
            (batches, jax.random.split(key, n)),
        )
        return sums, sq

    def fresh_position_metrics(
        self,
        key: jax.Array,
        buffer_state: ReplayBufferState,
        start_idx: jax.Array,
        weights: dict[str, Any],
        nn_state: eqx.nn.State | None,
        learner: int = 0,
    ) -> dict[str, jax.Array]:
        """The policy KL and value loss of each of `weights` (by name) on `fresh_positions`
        positions sampled from those stored since the replay buffer's write indices were
        `start_idx` (of finished games), as `fresh_policy_kl_<name>` and `fresh_value_loss_<name>`;
        with both `raw` and `avg`, also the raw weights' excess over the average's,
        `fresh_policy_kl_gap` and `fresh_value_loss_gap`. Empty if none of them has finished."""
        mask = self._fresh_mask(buffer_state, start_idx)
        if not bool(mask.any()):
            return {}
        n = self.fresh_positions // self.micro_batch_size
        batches = self._gather_masked(key, buffer_state, mask, n)
        out: dict[str, jax.Array] = {}
        for name, params in weights.items():
            kl, value = self._fresh_losses(params, nn_state, batches, learner=learner)
            out[f"fresh_policy_kl_{name}"] = kl
            out[f"fresh_value_loss_{name}"] = value
        if "raw" in weights and "avg" in weights:
            for metric in ("policy_kl", "value_loss"):
                out[f"fresh_{metric}_gap"] = (
                    out[f"fresh_{metric}_raw"] - out[f"fresh_{metric}_avg"]
                )
        return out

    @partial(jax.jit, static_argnums=(0,))
    def _fresh_mask(
        self, buffer_state: ReplayBufferState, start_idx: jax.Array
    ) -> jax.Array:
        """The buffer entries written since its write indices were `start_idx` (at most one
        turnover of each environment's buffer), of finished games."""
        capacity = buffer_state.populated.shape[1]
        written = (buffer_state.next_idx - start_idx) % capacity
        since = (jnp.arange(capacity)[None, :] - start_idx[:, None]) % capacity
        return (
            (since < written[:, None])
            & buffer_state.populated
            & buffer_state.has_reward
        )

    @partial(jax.jit, static_argnums=(0, 4))
    def _gather_masked(
        self, key: jax.Array, buffer_state: ReplayBufferState, mask: jax.Array, n: int
    ) -> BaseExperience:
        """`n` micro-batches sampled with replacement from the entries `mask` marks, stacked."""
        indices = jnp.unravel_index(
            self.memory_buffer.sample_indices(
                key, mask, n * self.micro_batch_size, replace=True
            ),
            mask.shape,
        )
        return jax.tree.map(
            lambda x: x[indices].reshape(n, self.micro_batch_size, *x.shape[2:]),
            buffer_state.buffer,
        )

    @partial(jax.jit, static_argnums=(0,), static_argnames=("learner",))
    def _fresh_losses(
        self,
        params: Any,
        nn_state: eqx.nn.State | None,
        batches: BaseExperience,
        learner: int = 0,
    ) -> tuple[jax.Array, jax.Array]:
        """The mean policy KL and value loss of the network with `params` over the micro-batches."""
        nn_static = eqx.filter(
            self.learners[learner].nn, eqx.is_inexact_array, inverse=True
        )
        nn = eqx.combine(params, nn_static)

        def one(_, batch: BaseExperience) -> tuple[None, tuple[jax.Array, jax.Array]]:
            _, (metrics, _) = self.loss_fn(nn, nn_state, batch)
            return None, (metrics["policy_kl"], metrics["value_loss"])

        _, (kl, value) = jax.lax.scan(one, None, batches)
        return kl.mean(), value.mean()

    def train_steps(
        self,
        key: jax.Array,
        collection_state: CollectionState,
        train_state: TrainState,
        num_steps: int,
        epoch: int = 0,
        learner: int = 0,
        batch_size: int | None = None,
        lr_scale: float | None = None,
    ) -> tuple[CollectionState, TrainState, dict]:
        """Performs `num_steps` training steps.

        Each step consists of sampling a minibatch from the replay buffer and updating the parameters.
        The minibatch is sampled uniformly without replacement from the finished episodes in the buffer,
        and each sample mapped through a data transform drawn at random (see `data_transform_fns`).
        With `tree_positions`, a share of it is sampled (uniformly, with replacement) from the search tree
        positions stored in the same span of self-play instead (see `tree_sample_mask`).

        Args:
            key: rng
            collection_state: current collection state
            train_state: current training state
            num_steps: number of training steps to perform
            epoch: the current epoch, which sets the share of tree positions (see `TreePositions.ratio_at`)
            learner: (optional) with `growth`, the index of the network `train_state` is
            batch_size: (optional) samples per step, a multiple of `micro_batch_size` (default
                `train_batch_size`)
            lr_scale: (optional) scales every step's update, as a learning rate would

        Returns:
            Tuple[CollectionState, TrainState, dict]:
                - updated collection state
                - updated training state
                - metrics

        Raises:
            ValueError: if no episode has finished yet, or the buffer holds fewer samples than a minibatch
        """
        if num_steps == 0:
            return collection_state, train_state, {}
        if batch_size is None:
            batch_size = self.train_batch_size
        if batch_size % self.micro_batch_size:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of the micro-batch size "
                f"({self.micro_batch_size})"
            )
        scale = None if lr_scale is None else jnp.asarray(lr_scale, dtype=jnp.float32)
        # the buffer doesn't change while training, so checking once covers every step
        self.memory_buffer.check_can_sample(collection_state.buffer_state, batch_size)
        if self.tree_positions is None:
            train_state, metrics = self.train_epoch(
                key,
                collection_state.buffer_state,
                train_state,
                num_steps,
                learner=learner,
                batch_size=batch_size,
                lr_scale=scale,
            )
            return collection_state, train_state, metrics
        tree_mask = self.tree_sample_mask(collection_state)
        num_tree = (
            self.tree_positions.batch_count(epoch, batch_size) if tree_mask.any() else 0
        )
        train_state, metrics = self.train_epoch(
            key,
            collection_state.buffer_state,
            train_state,
            num_steps,
            tree_buffer_state=collection_state.tree_buffer_state,
            tree_mask=tree_mask,
            num_tree=jnp.array(num_tree, dtype=jnp.int32),
            learner=learner,
            batch_size=batch_size,
            lr_scale=scale,
        )
        metrics = {
            **metrics,
            "tree_batch_fraction": jnp.array(num_tree / batch_size),
            "tree_buffer_samples": tree_mask.sum(),
        }
        return collection_state, train_state, metrics

    def collection_metrics(
        self, before: CollectionStats, after: CollectionState
    ) -> dict:
        """Measures how varied the collected data is, to spot it collapsing into the same few games.

        Args:
            before: the collection state's stats before this epoch's data collection
            after: collection state after it

        Returns:
            dict: metrics
                - `moves`: self-play moves played in between, in all environments
                - `episodes`: episodes that terminated in between
                - `buffer_capacity`: the replay buffer's capacity, entries per environment (see
                  `resize_replay_buffer`)
                - with `episode_metrics_fn`: `episode_<name>`, each metric's mean over those
                  episodes, omitted if none terminated
                - `buffer_samples`: experiences in the replay buffer training can sample
                - `buffer_distinct_positions`: distinct observations among them, those a data
                  transform maps to one another counting as one (e.g. a board and its rotations)
                - `buffer_distinct_fraction`: that as a fraction of those experiences
                - with `resignation`: `resigned`, the fraction of those episodes that ended in a
                  resignation; `resign_calibration_games`, the calibration games that ended;
                  `resign_calibration_below`, the fraction of them in which a player's value fell to
                  the threshold; and `resign_false`, the fraction of those that player didn't lose
                - with `endgame_solve`: `solve_missed`, the solves that found every slot full
                  (the training loop adds `solve_games`, the games labelled at their solve points,
                  `solve_draw`, the fraction of them drawn, `solve_positions`, every position
                  solved, `solve_seconds`, the solve's time, and `solve_wait_seconds`, how long
                  self-play waited for it after it was done)
                - with `tree_positions`: `tree_positions` stored in between,
                  `tree_positions_per_move`, and their `tree_mean_visits`
        """
        episodes = (after.episodes - before.episodes).sum()
        moves = (after.moves - before.moves).sum()
        distinct, total = self.count_distinct_observations(after.buffer_state)
        metrics = {
            "moves": moves,
            "episodes": episodes,
            "buffer_capacity": jnp.asarray(buffer_capacity(after.buffer_state)),
            "buffer_samples": total,
            "buffer_distinct_positions": distinct,
            "buffer_distinct_fraction": distinct / jnp.maximum(total, 1),
        }
        if episodes > 0:
            for name, total_after in after.episode_metrics.items():
                change = (total_after - before.episode_metrics[name]).sum()
                metrics[f"episode_{name}"] = change / episodes
        if after.resign_state is not None and before.resign_state is not None:

            def resign_change(name: str) -> jax.Array:
                return (
                    getattr(after.resign_state, name)
                    - getattr(before.resign_state, name)
                ).sum()

            calibration_games = resign_change("calibration_games")
            calibration_below = resign_change("calibration_below")
            metrics["resigned"] = resign_change("resigned") / jnp.maximum(episodes, 1)
            metrics["resign_calibration_games"] = calibration_games
            metrics["resign_calibration_below"] = calibration_below / jnp.maximum(
                calibration_games, 1
            )
            metrics["resign_false"] = resign_change("calibration_false") / jnp.maximum(
                calibration_below, 1
            )
        if after.pending is not None and before.solve_missed is not None:
            metrics["solve_missed"] = (after.pending.missed - before.solve_missed).sum()
        if self.tree_positions is not None:
            positions = (after.tree_positions - before.tree_positions).sum()
            visits = (after.tree_visits - before.tree_visits).sum()
            metrics["tree_positions"] = positions
            metrics["tree_positions_per_move"] = positions / jnp.maximum(moves, 1)
            metrics["tree_mean_visits"] = visits / jnp.maximum(positions, 1)
        return metrics

    def log_metrics(self, metrics: dict, epoch: int):
        """Logs metrics to console and the monitor.

        Args:
            metrics: dictionary of metrics, and media for the monitor (e.g. a tester's
                games, logged with its `log_fn`), which the console skips
            epoch: current epoch
        """
        # log to console
        metrics_str = {
            k: f"{number:.4f}"
            for k, v in metrics.items()
            if (number := to_number(v)) is not None
        }
        print(f"Epoch {epoch}: {metrics_str}")
        # log to monitor
        if self.monitor is not None:
            self.monitor.log(epoch, metrics)

    def set_activity(self, text: str | None, echo: bool = False) -> None:
        """Tells the monitor what the training loop is doing now, so the run's page can show
        what a slow step is busy with.

        Args:
            text: what the loop is doing, None once it's done
            echo: also print it to the console
        """
        if echo and text is not None:
            print(text, flush=True)
        if self.monitor is not None:
            self.monitor.activity(text)

    def save_checkpoint(
        self,
        train_state: TrainState,
        epoch: int,
        learner: int = 0,
        average: float | None = None,
    ) -> None:
        """Saves a checkpoint of the training state to `ckpt_dir` (with `growth`, to the
        network's directory in it).

        Deletes the oldest checkpoints so that at most `max_checkpoints` remain, besides
        those kept by `keep_every`.

        Args:
            train_state: current training state
            epoch: current epoch
            learner: (optional) with `growth`, the index of the network `train_state` is
            average: (optional) the horizon of the weight average whose checkpoint it is (its
                params are the average's), saved to that average's directory (see
                `average_dirs`)
        """
        ckpt_dir = self.ckpt_dirs[learner]
        if average is not None:
            ckpt_dir = os.path.join(ckpt_dir, average_name(average))
            os.makedirs(ckpt_dir, exist_ok=True)
        epochs = checkpoint_epochs(ckpt_dir)
        if epochs and epochs[-1] >= epoch:
            raise ValueError(
                f"{ckpt_dir} already has a checkpoint at or after epoch {epoch}, "
                "resume from it or use a different ckpt_dir"
            )
        # write to a temporary file first so an interrupted save doesn't leave a partial checkpoint
        path = checkpoint_path(ckpt_dir, epoch)
        with open(path + ".tmp", "wb") as f:
            eqx.tree_serialise_leaves(f, train_state)
        os.replace(path + ".tmp", path)
        # delete old checkpoints
        if self.keep_every is not None:
            epochs = [e for e in epochs if e % self.keep_every != 0]
            if epoch % self.keep_every == 0:
                return
        for old_epoch in epochs[: max(0, len(epochs) + 1 - self.max_checkpoints)]:
            os.remove(checkpoint_path(ckpt_dir, old_epoch))

    def load_train_state_from_checkpoint(
        self, path_to_checkpoint: str, epoch: int, learner: int = 0
    ) -> TrainState:
        """Loads a training state from a checkpoint.

        Args:
            path_to_checkpoint: path to checkpoint
            epoch: epoch to load
            learner: (optional) with `growth`, the index of the network it's a checkpoint of

        Returns:
            TrainState: loaded training state
        """
        with open(checkpoint_path(path_to_checkpoint, epoch), "rb") as f:
            return eqx.tree_deserialise_leaves(f, self.init_train_state(learner))

    def should_save_state(self, epochs_done: int) -> bool:
        """Whether to save the whole training state after `epochs_done` epochs (see
        `save_state_at` and `save_state_every`)."""
        return epochs_done in self.save_state_at or (
            self.save_state_every is not None
            and epochs_done % self.save_state_every == 0
        )

    def save_state(self, state: TrainLoopOutput) -> str:
        """Saves the whole training state to `ckpt_dir`, as `state-<epochs done>.npz`: the train
        states (with `growth`, every network's), the collection state (with the replay buffers),
        the testers' states, the number of epochs done, the training loop's rng, the wall time it
        has taken, with `lr_plateau` the rule's progress, and with `growth` its progress.

        The run continues from it exactly (see `resume`), and new runs fork from it (see
        `train_loop`'s `fork_from`). It also records the run's configuration, its monitor run, and
        the state the run was forked from, if any (see `read_state_meta`).

        With `save_state_every`, deletes the states saved before it at multiples of it, besides
        those at `save_state_at`.

        Arrays are copied to the host one at a time, as they're written, so it takes about the
        host memory of the largest. Loading takes about that of the replay buffers.

        Args:
            state: the training loop's state, with its rng (`key`), as `train_loop` returns it

        Returns:
            str: the saved state's path
        """
        if state.key is None:
            raise ValueError("can't save a training state without its rng (`key`)")
        parts = {
            "train_states": state.train_states,
            # without the move log (see `log_moves`)
            "collection_state": replace(state.collection_state, move_log=None),
            "test_states": state.test_states,
            # with `averages`, the weight averages and how many epochs they hold
            **({"averages": state.averages} if state.averages else {}),
        }
        meta = {
            "format": STATE_FORMAT,
            "epoch": state.cur_epoch,
            "num_envs": int(state.collection_state.episodes.shape[0]),
            "buffer_capacity": buffer_capacity(state.collection_state.buffer_state),
            "tree_buffer_capacity": buffer_capacity(
                state.collection_state.tree_buffer_state
            )
            if state.collection_state.tree_buffer_state is not None
            else None,
            "monitor_run": self.monitor.run_id if self.monitor is not None else None,
            "parent": self.parent,
            "lr_plateau": [
                asdict(progress) if progress is not None else None
                for progress in state.lr_plateau
            ],
            "growth": asdict(state.growth) if state.growth is not None else None,
            # whether the next network has caught up by the rule (e.g. for a controller to
            # decide what a manual rule only reports), None when the last network plays
            "growth_caught_up": self.growth.caught_up(state.growth)
            if self.growth is not None
            and state.growth is not None
            and state.growth.playing < len(self.growth.learners)
            else None,
            "seconds": state.seconds,
            "averages": {
                "horizons": list(self.averages),
                "averaged": list(state.averaged),
            }
            if state.averages
            else None,
            "config": {**self.get_config(), **self.extra_config},
            "leaves": {part: _describe(tree) for part, tree in parts.items()},
        }
        arrays = {
            "meta": np.array(json.dumps(meta, default=str)),
            "key": state.key,
            **{
                f"{part}.{i}": x
                for part, tree in parts.items()
                for i, x in enumerate(jax.tree.leaves(tree))
            },
        }
        path = state_path(self.ckpt_dir, state.cur_epoch)
        # write to a temporary file first so an interrupted save doesn't leave a partial state.
        # As np.savez does, but copying one array at a time to the host (see `_to_host`)
        with zipfile.ZipFile(path + ".tmp", "w", allowZip64=True) as f:
            for name, x in arrays.items():
                with f.open(f"{name}.npy", "w", force_zip64=True) as entry:
                    np.lib.format.write_array(entry, _to_host(x), allow_pickle=False)
        os.replace(path + ".tmp", path)
        if self.save_state_every is not None:
            for epoch in state_epochs(self.ckpt_dir):
                if (
                    epoch < state.cur_epoch
                    and epoch % self.save_state_every == 0
                    and epoch not in self.save_state_at
                ):
                    os.remove(state_path(self.ckpt_dir, epoch))
        return path

    def load_state(self, path: str) -> TrainLoopOutput:
        """Loads a state saved by `save_state`, to continue its run exactly: pass it to
        `train_loop` as its `initial_state` (see `resume`, which also takes care of the run's
        checkpoints and monitor run).

        The trainer must be configured as the run's was: the number of environments, networks,
        optimizer, self-play evaluator (the one scheduled for the last epoch done, of the network
        playing), tree positions or none, and testers must match, so that every array does. The
        replay buffers keep the capacities they were saved with, which `resize_replay_buffer` may
        have changed.

        Args:
            path: the saved state

        Returns:
            TrainLoopOutput: the run's state, in new device arrays, with its rng

        Raises:
            ValueError: if the saved state doesn't fit this trainer
        """
        with np.load(path) as saved:
            meta = self._check_meta(path, json.loads(str(saved["meta"])))
            self._check_sizes(path, meta)
            if (meta["tree_buffer_capacity"] is None) != (self.tree_buffer is None):
                theirs, ours = (
                    ("with", "without")
                    if self.tree_buffer is None
                    else ("without", "with")
                )
                raise ValueError(
                    f"{path} was saved {theirs} tree positions, this run is {ours}: continuing "
                    "a run needs the same (a fork can differ)"
                )
            epoch = meta["epoch"]
            if (meta["growth"] is None) != (self.growth is None):
                raise ValueError(
                    f"{path} was saved by a run {'without' if meta['growth'] is None else 'with'} "
                    "growth: continuing it needs the same"
                )
            growth = (
                saved_growth(meta["growth"]) if meta["growth"] is not None else None
            )
            playing = growth.playing if growth is not None else 0
            # the evaluator whose states the collection state holds
            evaluator = self.learners[playing].schedule.at(max(epoch - 1, 0))
            train_templates = tuple(
                jax.eval_shape(partial(self.init_train_state, i))
                for i in range(len(self.learners))
            )
            train_states = self._load_part(saved, meta, "train_states", train_templates)
            collection_state = self._load_part(
                saved,
                meta,
                "collection_state",
                jax.eval_shape(
                    lambda: self.init_collection_state(
                        jax.random.PRNGKey(0),
                        self.batch_size,
                        evaluator,
                        meta["buffer_capacity"],
                        meta["tree_buffer_capacity"],
                    )
                ),
            )
            test_states = self._load_part(
                saved,
                meta,
                "test_states",
                self.test_state_templates(train_templates),
            )
            averages, averaged = self._load_averages(saved, meta, train_templates)
            key = _to_device(saved["key"])
        lr_plateau: tuple[LRPlateauProgress | None, ...] = ()
        if self.lr_plateau is not None:
            if not meta["lr_plateau"]:
                raise ValueError(
                    f"{path} was saved by a run without lr_plateau: fork a run from it instead"
                )
            lr_plateau = tuple(
                LRPlateauProgress(progress["epochs"], tuple(progress["losses"]))
                if progress is not None
                else None
                for progress in meta["lr_plateau"]
            )
        return TrainLoopOutput(
            collection_state=collection_state,
            train_states=train_states,
            test_states=test_states,
            cur_epoch=epoch,
            key=key,
            lr_plateau=lr_plateau,
            growth=growth,
            seconds=meta["seconds"],
            averages=averages,
            averaged=averaged,
        )

    def _load_averages(
        self, saved: Any, meta: dict, train_templates: Sequence[Any]
    ) -> tuple[tuple[tuple[Any, ...], ...], tuple[int, ...]]:
        """A saved state's weight averages and their epochs, if it was saved with this
        trainer's horizons; else none, and the run starts them at its networks' weights."""
        info = meta.get("averages")
        if not self.averages or info is None or info["horizons"] != list(self.averages):
            if self.averages:
                print(
                    f"the saved state has no averages over {list(self.averages)} epochs: they "
                    "start at its networks' weights",
                    flush=True,
                )
            return (), ()
        template = tuple(
            tuple(ts.params for _ in self.averages) for ts in train_templates
        )
        averages = self._load_part(saved, meta, "averages", template)
        return averages, tuple(info["averaged"])

    def test_state_templates(self, train_states: Sequence[Any]) -> tuple[list, ...]:
        """The shapes of the networks' testers' states, from their train states (or their
        shapes): the first networks', as many as `train_states` has."""
        return tuple(
            jax.eval_shape(
                lambda params, testers=learner.testers: [
                    t.init(params=params) for t in testers
                ],
                self.extract_model_params_fn(ts),
            )
            for learner, ts in zip(
                self.learners[: len(train_states)], train_states, strict=True
            )
        )

    def resume(self, path: str | None = None) -> TrainLoopOutput:
        """Loads the newest state saved in `ckpt_dir` (or `path`, a saved state or a directory
        of them) to continue its run exactly, with `train_loop`'s `initial_state`.

        With the run's configuration (see `load_state`), the continued run is the same as if it
        hadn't stopped (its rng is saved, so `train_loop`'s seed is unused), given deterministic computations (e.g. on the CPU; GPU convolutions'
        autotuning may differ). It logs to the run's monitor run (whose metrics then hold the
        epochs between the state and the stop twice), and deletes the checkpoints saved after
        the state, which it saves again.

        Args:
            path: (optional) the saved state, or a directory to take the newest from. If not
                provided, `ckpt_dir`.

        Returns:
            TrainLoopOutput: the run's state, for `train_loop`'s `initial_state`
        """
        path = latest_state_path(path if path is not None else self.ckpt_dir)
        state = self.load_state(path)
        meta = read_state_meta(path)
        current = json.loads(
            json.dumps({**self.get_config(), **self.extra_config}, default=str)
        )
        current, saved = _flatten(current), _flatten(meta["config"])
        changed = sorted(
            k for k in current.keys() | saved.keys() if current.get(k) != saved.get(k)
        )
        if changed:
            print(
                f"warning: resuming {path} with a different configuration: {changed}",
                flush=True,
            )
        self.parent = meta["parent"]
        if self.monitor is not None and self.monitor.run_id is None:
            self.monitor.run_id = meta["monitor_run"]
        average_dirs = [
            d for i in range(len(self.learners)) for d in self.average_dirs(i)
        ]
        for ckpt_dir in [*self.ckpt_dirs, *average_dirs]:
            for epoch in checkpoint_epochs(ckpt_dir):
                if epoch >= state.cur_epoch:
                    os.remove(checkpoint_path(ckpt_dir, epoch))
        print(f"resuming from {path}: {state.cur_epoch} epochs done", flush=True)
        return state

    def branch(self, path: str, take_over: bool = False) -> TrainLoopOutput:
        """Loads a saved state (or the newest in a directory) to continue its run in this
        trainer's `ckpt_dir`, with this trainer's self-play: a branch of the run, e.g. searching
        more, or with the next network taking over. Pass it to `train_loop` as its
        `initial_state`.

        The run goes on from where it was saved, as with `resume`: its epochs, every network
        and its optimizer state, the learning rate rule's and growth's progress, the replay
        buffers, the games in progress, the testers' states and the rng. Its self-play search's
        states (e.g. its trees) are kept if the same network plays and they fit this trainer's
        search, so a branch with the run's own settings continues it exactly; otherwise they
        start anew, and the moves in progress are searched again from the start (as when a
        schedule switches searches, or the next network takes over, see `follow_schedule`).

        Branches of one state are compared at equal compute, so each is the run going on,
        rather than a fork (see `train_loop`'s `fork_from`), which starts a new run. The number
        of environments and whether there are tree positions must be the run's. The replay
        buffers take this trainer's capacities, resized from the saved ones if they differ (see
        `resize_replay_buffer`). A branch logs to a monitor run of its own, whose config records
        the state as its parent, and leaves the saved run's directory as it is.

        With `growth`, the trainer's networks may go on past the saved run's (the run grows into
        a bigger one than it was set up with): those start from scratch, at their own weights,
        as a network that hasn't joined a run yet is. One can't take over in the branch.

        Args:
            path: the saved state (see `save_state`), or a directory to take the newest from
            take_over: (with `growth`) the next network plays from the branch's first epoch, and
                the one after it starts training

        Returns:
            TrainLoopOutput: the run's state, to continue with `train_loop`'s `initial_state`

        Raises:
            ValueError: if the saved state doesn't fit this trainer, or nothing can take over
        """
        path = latest_state_path(path)
        with np.load(path) as saved:
            meta = self._check_meta(path, json.loads(str(saved["meta"])))
            self._check_sizes(path, meta)
            epoch = meta["epoch"]
            if (meta["growth"] is None) != (self.growth is None):
                raise ValueError(
                    f"{path} was saved by a run {'without' if meta['growth'] is None else 'with'} "
                    "growth: a branch of it needs the same"
                )
            if (meta["tree_buffer_capacity"] is None) != (self.tree_buffer is None):
                raise ValueError(
                    f"{path} was saved {'without' if self.tree_buffer else 'with'} tree "
                    "positions: a branch of it needs the same"
                )
            growth = None
            # the networks the saved run has: the trainer's first ones (see the docstring)
            saved_networks = 1
            if self.growth is not None:
                growth = saved_growth(meta["growth"])
                saved_networks = len(growth.trained)
                if saved_networks > len(self.learners):
                    raise ValueError(
                        f"{path} was saved with {saved_networks} networks, this run has "
                        f"{len(self.learners)}: a branch can add networks, not drop them"
                    )
                if take_over:
                    if growth.playing + 1 == saved_networks < len(self.learners):
                        raise ValueError(
                            f"{self.learners[growth.playing].name} is the saved run's last "
                            "network: a network the branch adds can't take over from it"
                        )
                    growth = self.growth.take_over(growth)
            elif take_over:
                raise ValueError("only a run with growth has a network to take over")
            playing = growth.playing if growth is not None else 0
            # the search whose states the branch starts with (see `train_loop`'s initial_state)
            evaluator = self.learners[playing].schedule.at(max(epoch - 1, 0))
            train_templates = tuple(
                jax.eval_shape(partial(self.init_train_state, i))
                for i in range(saved_networks)
            )
            train_states = self._load_part(saved, meta, "train_states", train_templates)
            collection_state, kept_search = self._branch_collection_state(
                saved, meta, evaluator, keep_search=not take_over
            )
            test_states = self._load_part(
                saved,
                meta,
                "test_states",
                self.test_state_templates(train_templates),
            )
            averages, averaged = self._load_averages(saved, meta, train_templates)
            key = _to_device(saved["key"])
        added = range(saved_networks, len(self.learners))
        if added:
            assert growth is not None
            fresh = tuple(self.init_train_state(i) for i in added)
            train_states = (*train_states, *fresh)
            test_states = (
                *test_states,
                *(
                    [
                        tester.init(params=self.extract_model_params_fn(ts))
                        for tester in self.learners[i].testers
                    ]
                    for i, ts in zip(added, fresh, strict=True)
                ),
            )
            if averages:
                averages = (*averages, *(self.fresh_averages(ts) for ts in fresh))
                averaged = (*averaged, *(0 for _ in added))
            growth = replace(
                growth,
                trained=(*growth.trained, *(0 for _ in added)),
                values=(*growth.values, *(() for _ in added)),
                guard_values=(*growth.guard_values, *(() for _ in added))
                if growth.guard_values
                else (),
                played=(*growth.played, *(0 for _ in added)),
            )
        resized = ""
        if self.memory_buffer.capacity != meta["buffer_capacity"]:
            collection_state = self.resize_replay_buffer(
                collection_state,
                self.memory_buffer.capacity,
                self.tree_buffer.capacity if self.tree_buffer is not None else None,
            )
            resized = (
                f", replay buffer resized from {meta['buffer_capacity']} to "
                f"{self.memory_buffer.capacity}"
            )
        lr_plateau: tuple[LRPlateauProgress | None, ...] = ()
        if self.lr_plateau is not None:
            if not meta["lr_plateau"]:
                raise ValueError(
                    f"{path} was saved by a run without lr_plateau: fork a run from it instead"
                )
            progress = [
                LRPlateauProgress(p["epochs"], tuple(p["losses"]))
                if p is not None
                else None
                for p in meta["lr_plateau"]
            ]
            if take_over:
                # its own rule starts, at the rate it has trained at (as when the rule switches)
                progress[playing] = LRPlateauProgress()
            lr_plateau = (*progress, *(None for _ in added))
        self.parent = {
            "path": os.path.abspath(path),
            "epoch": epoch,
            "branch": True,
            "take_over": take_over,
            "monitor_run": meta["monitor_run"],
            "parent": meta["parent"],
        }
        whose = f", {self.learners[playing].name} playing" if take_over else ""
        search = "its search's states kept" if kept_search else "new search states"
        new = "".join(f", {self.learners[i].name} added" for i in added)
        print(
            f"branching from {path}: {epoch} epochs done{whose}, {search}{resized}{new}",
            flush=True,
        )
        return TrainLoopOutput(
            collection_state=collection_state,
            train_states=train_states,
            test_states=test_states,
            cur_epoch=epoch,
            key=key,
            lr_plateau=lr_plateau,
            growth=growth,
            seconds=meta["seconds"],
            averages=averages,
            averaged=averaged,
        )

    def _branch_collection_state(
        self, saved: Any, meta: dict, evaluator: Evaluator, keep_search: bool = True
    ) -> tuple[CollectionState, bool]:
        """A saved collection state, for `evaluator`, at its saved capacities: with
        `keep_search`, the saved search states if they have its states' shapes, else new ones,
        with the moves in progress searched again (see `branch`). Returns it, and whether the
        saved search states were kept. With `endgame_solve`, the saved pending solves if they
        have this run's shapes, else none waiting (any the saved state had are dropped)."""
        template = jax.eval_shape(
            lambda: self.init_collection_state(
                jax.random.PRNGKey(0),
                self.batch_size,
                evaluator,
                meta["buffer_capacity"],
                meta["tree_buffer_capacity"],
            )
        )
        expected = _describe(template)
        described = [
            (leaf_path, tuple(shape), dtype)
            for leaf_path, shape, dtype in meta["leaves"]["collection_state"]
        ]
        index = {leaf_path: i for i, (leaf_path, _, _) in enumerate(described)}

        def searching(leaf_path: str) -> bool:
            return leaf_path.startswith(".eval_state")

        def solving(leaf_path: str) -> bool:
            return leaf_path.startswith(".pending")

        keep = keep_search and [d for d in described if searching(d[0])] == [
            e for e in expected if searching(e[0])
        ]
        keep_pending = [d for d in described if solving(d[0])] == [
            e for e in expected if solving(e[0])
        ]
        mismatched = [
            f"  {leaf_path}: saved "
            + (
                f"{described[index[leaf_path]][1]} {described[index[leaf_path]][2]}"
                if leaf_path in index
                else "nothing"
            )
            + f", this run's {shape} {dtype}"
            for leaf_path, shape, dtype in expected
            if not searching(leaf_path)
            and not solving(leaf_path)
            and (
                leaf_path not in index
                or described[index[leaf_path]] != (leaf_path, shape, dtype)
            )
        ]
        if mismatched:
            raise ValueError(
                "the saved collection state doesn't match this run's:\n"
                + "\n".join(mismatched[:8])
            )
        fresh = (
            None
            if keep
            else iter(jax.tree.leaves(self.init_eval_state(evaluator, self.batch_size)))
        )
        leaves = [
            jnp.zeros(shape, dtype)  # replaced below
            if solving(leaf_path) and not keep_pending
            else _to_device(saved[f"collection_state.{index[leaf_path]}"])
            if keep or not searching(leaf_path)
            else next(fresh)  # pyright: ignore[reportArgumentType]
            for leaf_path, shape, dtype in expected
        ]
        state = jax.tree.unflatten(jax.tree.structure(template), leaves)
        if not keep_pending:
            if any(solving(d[0]) for d in described):
                print(
                    "branch: the saved games waiting for solves don't fit this run's, so "
                    "they're dropped",
                    flush=True,
                )
            state = replace(
                state,
                pending=self.init_pending(
                    jax.random.PRNGKey(meta["epoch"]),
                    self.batch_size,
                    state.env_state,
                    meta["buffer_capacity"],
                ),
            )
        if not keep:
            # the moves in progress start their searches again
            state = replace(state, search_cycles=jnp.zeros_like(state.search_cycles))
        return state, keep

    def fork_state(
        self,
        path: str,
        key: jax.Array,
        replay_buffers: bool = True,
        into: int = 0,
        name: str | None = None,
    ) -> tuple[CollectionState, TrainState]:
        """The states a run forked from a saved state (see `train_loop`'s `fork_from`) starts with.

        - the train state: the saved network and optimizer state, with the step count and the
          learning rate schedules' (see `restart_schedules`) back at 0
        - with `replay_buffers`, the collection state: the saved replay buffers (and tree
          positions', if both runs have them), with self-play's games and the evaluator's states
          (e.g. search trees) initialized anew, for the evaluator scheduled for epoch 0. The
          entries of the games in progress are discarded. A buffer of a different capacity
          keeps each environment's newest entries (see `EpisodeReplayBuffer.resized`).
        - without, a new collection state, as a run from scratch starts with

        The number of environments, network and optimizer must match the saved run's. A run
        with `growth` forks from a run without: the saved network is the Trainer's own, and
        those it grows into start from scratch. From a run with growth, the network forked is
        the saved run's named `name`, by default network `into`'s name (see `train_loop`'s
        `fork_into`).

        Args:
            path: the saved state (see `save_state`)
            key: rng, for the games
            replay_buffers: load the replay buffers (default), or start with empty ones
            into: the network (an index into the Trainer's, the Trainer's own 0) the saved one
                forks into; its self-play's games start the collection state
            name: (from a growth run's state) the saved network to fork, by name, if not
                network `into`'s

        Returns:
            Tuple[CollectionState, TrainState]: the collection state and train state

        Raises:
            ValueError: if the saved state doesn't fit this trainer
        """
        template = jax.eval_shape(partial(self.init_train_state, into))
        with np.load(path) as saved:
            meta = self._check_meta(path, json.loads(str(saved["meta"])))
            if meta["growth"] is not None:
                names = saved_network_names(meta)
                name = name or self.learners[into].name
                if name not in names:
                    raise ValueError(
                        f"{path} has no network {name} to fork (it has {', '.join(names)})"
                    )
                train_state = self._load_network(
                    saved, meta, "train_states", names.index(name), template
                )
            else:
                if into != 0:
                    raise ValueError(
                        f"{path} has one network: it forks into the Trainer's own"
                    )
                (train_state,) = self._load_part(
                    saved, meta, "train_states", (template,)
                )
            train_state = replace(
                train_state,
                opt_state=restart_schedules(train_state.opt_state),
                step=jnp.zeros_like(train_state.step),
            )
            evaluator = self.learners[into].schedule.at(0)
            if not replay_buffers:
                return (
                    self.init_collection_state(key, self.batch_size, evaluator),
                    train_state,
                )
            self._check_sizes(path, meta)
            index = {
                leaf_path: i
                for i, (leaf_path, _, _) in enumerate(
                    meta["leaves"]["collection_state"]
                )
            }

            def get(leaf_path: str) -> np.ndarray:
                if leaf_path not in index:
                    raise ValueError(f"{path} has no collection state{leaf_path}")
                return saved[f"collection_state.{index[leaf_path]}"]

            stats: dict[str, Any] = {
                name: _to_device(get(f".{name}"))
                for name in (
                    "episodes",
                    "tree_positions",
                    "tree_visits",
                    "moves",
                )
            }
            buffer_state, _ = self._load_buffer(
                get, ".buffer_state", self.memory_buffer
            )
            tree_buffer_state, tree_written_at = None, None
            if self.tree_buffer is not None:
                if meta["tree_buffer_capacity"] is not None:
                    tree_buffer_state, (tree_written_at,) = self._load_buffer(
                        get, ".tree_buffer_state", self.tree_buffer, ".tree_written_at"
                    )
                else:
                    print(
                        f"fork: {path} has no tree positions, the tree position buffer "
                        "starts empty",
                        flush=True,
                    )
                    tree_buffer_state = self.tree_buffer.init(
                        self.batch_size, self.make_template_experience()
                    )
                    tree_written_at = jnp.zeros(
                        (self.batch_size, self.tree_buffer.capacity), dtype=jnp.int32
                    )
        env_state, metadata, eval_state = self.init_games(
            key, self.batch_size, evaluator
        )
        collection_state = CollectionState(
            eval_state=eval_state,
            env_state=env_state,
            buffer_state=buffer_state,
            metadata=metadata,
            tree_buffer_state=tree_buffer_state,
            tree_written_at=tree_written_at,
            # metrics are compared within an epoch, so their sums can start again
            episode_metrics=self.init_episode_metrics(
                self.batch_size, metadata.rewards[0]
            ),
            **self.init_moves(key, self.batch_size),
            **stats,
            resign_state=self.init_resign_state(key, self.batch_size),
            pending=self.init_pending(
                key, self.batch_size, env_state, buffer_capacity(buffer_state)
            ),
        )
        return collection_state, train_state

    def warm_state(
        self, path: str, learner: int, name: str | None = None
    ) -> TrainState:
        """The train state a network of a new run starts from with a warm start (see
        `train_loop`'s `warm_start`): a saved state's network, its weights and optimizer state,
        with the step count and the learning rate schedules back at 0, as a fork's are (see
        `fork_state`). From a growth run's state, the network named `name` (by default network
        `learner`'s name); from a run without growth, its one network. Its shapes must be
        network `learner`'s.

        Raises:
            ValueError: if the saved state has no such network, or it doesn't fit
        """
        template = jax.eval_shape(partial(self.init_train_state, learner))
        with np.load(path) as saved:
            meta = self._check_meta(path, json.loads(str(saved["meta"])))
            if meta["growth"] is not None:
                names = saved_network_names(meta)
                name = name or self.learners[learner].name
                if name not in names:
                    raise ValueError(
                        f"{path} has no network {name} to start from (it has "
                        f"{', '.join(names)})"
                    )
                train_state = self._load_network(
                    saved, meta, "train_states", names.index(name), template
                )
            else:
                (train_state,) = self._load_part(
                    saved, meta, "train_states", (template,)
                )
        return replace(
            train_state,
            opt_state=restart_schedules(train_state.opt_state),
            step=jnp.zeros_like(train_state.step),
        )

    def fork_averages(
        self, path: str, into: int, name: str | None = None
    ) -> tuple[tuple[Any, ...], int] | None:
        """The weight averages of the network a fork takes from a saved state (see `fork_state`):
        a growth run's network `name` (by default network `into`'s name), or a run without
        growth's one network; and the epochs in them. None if the state has no averages over
        this trainer's horizons, and the fork starts them at the network's weights."""
        meta = read_state_meta(path)
        info = meta.get("averages")
        if not self.averages or info is None or info["horizons"] != list(self.averages):
            return None
        template = tuple(
            jax.eval_shape(partial(self.init_train_state, into)).params
            for _ in self.averages
        )
        with np.load(path) as saved:
            if meta["growth"] is None:
                (averages,) = self._load_part(saved, meta, "averages", (template,))
                return averages, int(info["averaged"][0])
            index = saved_network_names(meta).index(name or self.learners[into].name)
            averages = self._load_network(saved, meta, "averages", index, template)
        return averages, int(info["averaged"][index])

    def _check_meta(self, path: str, meta: dict) -> dict:
        if meta.get("format") != STATE_FORMAT:
            raise ValueError(
                f"{path} is a saved state of format {meta.get('format')}, this version reads "
                f"format {STATE_FORMAT}"
            )
        return meta

    def _check_sizes(self, path: str, meta: dict) -> None:
        """Checks that a saved state's number of environments is this trainer's."""
        if meta["num_envs"] != self.batch_size:
            raise ValueError(
                f"{path} was saved with {meta['num_envs']} self-play environments, this run "
                f"has {self.batch_size}: they must be the same"
            )

    def _load_network(
        self, saved: Any, meta: dict, part: str, index: int, template: Any
    ) -> Any:
        """One network's part of a saved state whose part holds every network's (e.g. a growth
        run's `train_states`): network `index`'s arrays, checked against `template`'s."""
        prefix = f"[{index}]"
        chosen = [
            (i, leaf_path[len(prefix) :], tuple(shape), dtype)
            for i, (leaf_path, shape, dtype) in enumerate(meta["leaves"][part])
            if leaf_path.startswith((prefix + ".", prefix + "["))
        ]
        expected = _describe(template)
        if [c[1:] for c in chosen] != expected:
            raise ValueError(
                f"the saved network {index}'s {part.replace('_', ' ')} don't match this run's"
            )
        return jax.tree.unflatten(
            jax.tree.structure(template),
            [_to_device(saved[f"{part}.{i}"]) for i, _, _, _ in chosen],
        )

    def _load_part(self, saved: Any, meta: dict, part: str, template: Any) -> Any:
        """Loads part of a saved state (see `save_state`) into new device arrays, checking each
        array's shape and dtype against `template`'s (e.g. from `jax.eval_shape`)."""
        expected = _describe(template)
        described = [
            (leaf_path, tuple(shape), dtype)
            for leaf_path, shape, dtype in meta["leaves"][part]
        ]
        name = part.replace("_", " ")
        if len(described) != len(expected):
            raise ValueError(
                f"saved {name}: {len(described)} arrays, this run's: {len(expected)}: was "
                "it saved with different settings (e.g. the networks, optimizer, self-play "
                "search, tree positions, testers or growth)?"
            )
        mismatched = [
            f"  {leaf_path}: saved {s_shape} {s_dtype}, this run's {shape} {dtype}"
            for (leaf_path, shape, dtype), (_, s_shape, s_dtype) in zip(
                expected, described, strict=True
            )
            if (s_shape, s_dtype) != (shape, dtype)
        ]
        if mismatched:
            raise ValueError(
                f"the saved {name} doesn't match this run's:\n"
                + "\n".join(mismatched[:8])
            )
        return jax.tree.unflatten(
            jax.tree.structure(template),
            [_to_device(saved[f"{part}.{i}"]) for i in range(len(expected))],
        )

    def _load_buffer(
        self,
        get: Callable[[str], np.ndarray],
        prefix: str,
        buffer: EpisodeReplayBuffer,
        *entry_paths: str,
    ) -> tuple[ReplayBufferState, list[jax.Array]]:
        """Loads a saved collection state's replay buffer (at `prefix`) into `buffer`'s capacity,
        without the entries of games in progress, with arrays at `entry_paths` that have an
        entry per place in it (see `EpisodeReplayBuffer.resized`)."""
        template = jax.eval_shape(
            lambda: buffer.init(self.batch_size, self.make_template_experience())
        )
        arrays = []
        for leaf_path, shape, dtype in _describe(template):
            x = get(prefix + leaf_path)
            # all but the capacity dimension must match
            if str(x.dtype) != dtype or (x.shape[:1] + x.shape[2:]) != (
                shape[:1] + shape[2:]
            ):
                raise ValueError(
                    f"the saved collection state{prefix}{leaf_path} is {x.shape} {x.dtype}, "
                    f"this run's {shape} {dtype}, which only differ in the buffer's capacity"
                )
            arrays.append(x)
        state = jax.tree.unflatten(jax.tree.structure(template), arrays)
        entry_data = [get(p) for p in entry_paths]
        # the games in progress don't carry on: their entries would never get rewards
        state = jax.tree.map(np.asarray, buffer.truncate(state))
        capacity = state.populated.shape[1]
        if capacity != buffer.capacity:
            name = "tree position" if buffer is self.tree_buffer else "replay"
            print(
                f"fork: resizing the saved {name} buffer from {capacity} to "
                f"{buffer.capacity} entries per environment",
                flush=True,
            )
            state, entry_data = buffer.resized(state, *entry_data)
        return jax.tree.map(_to_device, state), [_to_device(x) for x in entry_data]

    def make_template_env_state(self) -> Any:
        """Create a template environment state used for initializing data structures that hold environment states to the correct shape.

        Returns:
            pytree: template environment state
        """
        env_state, _ = self.env_init_fn(jax.random.PRNGKey(0))
        return env_state

    def move_log_length(self) -> int:
        """The rows an environment's move log needs (see `log_moves`): the most moves it can play
        in an epoch's self-play, one each time moves are played (see `collect_steps`), with any of
        the schedule's evaluators."""
        return max(
            self.collection_steps_per_epoch
            * evaluator.search_cycles()
            // (
                evaluator.search_cycles()
                if self.advance_every is None
                else self.advance_every
            )
            for _, evaluator in self.collection_schedule.stages
        )

    def init_move_log(self) -> MoveLog:
        """Empty move logs (see `log_moves`), one per environment."""
        template = self.make_template_experience()
        return init_move_log(
            self.batch_size,
            self.move_log_length(),
            template.observation_nn,
            template.policy_weights.shape[-1],
        )

    def make_template_experience(self) -> BaseExperience:
        """Create a template experience used for initializing data structures that hold experiences to the correct shape.

        Returns:
            BaseExperience: template experience
        """
        env_state, metadata = self.env_init_fn(jax.random.PRNGKey(0))
        return BaseExperience(
            observation_nn=self.state_to_nn_input_fn(env_state),
            policy_mask=metadata.action_mask,
            policy_weights=jnp.zeros_like(metadata.action_mask, dtype=jnp.float32),
            reward=jnp.zeros_like(metadata.rewards),
            cur_player_id=metadata.cur_player_id,
            search_value=jnp.zeros((), dtype=jnp.float32),
        )

    def init_eval_state(self, evaluator: Evaluator, batch_size: int) -> Any:
        """Initializes a self-play evaluator's states, one per environment.

        Args:
            evaluator: the self-play evaluator
            batch_size: number of parallel environments

        Returns:
            pytree: the evaluator's states
        """
        return evaluator.init_batched(
            batch_size, template_embedding=self.template_env_state
        )

    def init_episode_metrics(
        self, batch_size: int, rewards: jax.Array
    ) -> dict[str, jax.Array]:
        """Zeroed sums of each metric `episode_metrics_fn` returns, one per environment.

        Args:
            batch_size: number of parallel environments
            rewards: an environment's rewards (one per player), for their shape and dtype

        Returns:
            dict: name -> zeros of shape (batch_size,); empty without `episode_metrics_fn`
        """
        if self.episode_metrics_fn is None:
            return {}
        shapes = jax.eval_shape(self.episode_metrics_fn, rewards)
        return {name: jnp.zeros((batch_size,), dtype=jnp.float32) for name in shapes}

    def add_episode_metrics(
        self, sums: dict[str, jax.Array], rewards: jax.Array, terminated: jax.Array
    ) -> dict[str, jax.Array]:
        """Adds a step's episode metrics to `sums` if its episode terminated (one environment)."""
        if self.episode_metrics_fn is None:
            return sums
        metrics = self.episode_metrics_fn(rewards)
        return {
            name: sums[name]
            + jnp.where(terminated, metrics[name], 0).astype(jnp.float32)
            for name in sums
        }

    def init_collection_state(
        self,
        key: jax.Array,
        batch_size: int,
        evaluator: Evaluator | None = None,
        capacity: int | None = None,
        tree_capacity: int | None = None,
    ) -> CollectionState:
        """Initializes the collection state (see CollectionState).

        Args:
            key: rng
            batch_size: number of parallel environments
            evaluator: (optional) the self-play evaluator to initialize states for. If not
                provided, the first in the schedule.
            capacity: (optional) the replay buffer's capacity, if not `memory_buffer`'s (see
                `resize_replay_buffer`)
            tree_capacity: (optional) with tree positions, their buffer's capacity, if not
                `tree_positions`'

        Returns:
            CollectionState: initialized collection state
        """
        if evaluator is None:
            evaluator = self.collection_schedule.at(0)
        # make template experience
        template_experience = self.make_template_experience()
        # init buffer state
        memory_buffer = (
            self.memory_buffer if capacity is None else EpisodeReplayBuffer(capacity)
        )
        buffer_state = memory_buffer.init(batch_size, template_experience)
        tree_buffer = (
            self.tree_buffer
            if self.tree_buffer is None or tree_capacity is None
            else EpisodeReplayBuffer(tree_capacity)
        )
        tree_buffer_state = (
            tree_buffer.init(batch_size, template_experience)
            if tree_buffer is not None
            else None
        )
        env_state, metadata, eval_state = self.init_games(key, batch_size, evaluator)
        # return collection state
        return CollectionState(
            eval_state=eval_state,
            env_state=env_state,
            buffer_state=buffer_state,
            metadata=metadata,
            episodes=jnp.zeros((batch_size,), dtype=jnp.int32),
            tree_positions=jnp.zeros((batch_size,), dtype=jnp.int32),
            tree_visits=jnp.zeros((batch_size,), dtype=jnp.int32),
            moves=jnp.zeros((batch_size,), dtype=jnp.int32),
            episode_metrics=self.init_episode_metrics(batch_size, metadata.rewards[0]),
            **self.init_moves(key, batch_size),
            tree_buffer_state=tree_buffer_state,
            tree_written_at=jnp.zeros(
                (batch_size, tree_buffer.capacity), dtype=jnp.int32
            )
            if tree_buffer is not None
            else None,
            resign_state=self.init_resign_state(key, batch_size),
            pending=self.init_pending(
                key, batch_size, env_state, memory_buffer.capacity
            ),
        )

    def init_pending(
        self, key: jax.Array, batch_size: int, env_state: Any, capacity: int
    ) -> PendingSolves | None:
        """`CollectionState`'s pending solves, with nothing waiting, for environments whose games
        are `env_state` (one per environment) and a replay buffer of `capacity`; None without
        `endgame_solve`."""
        if self.endgame_solve is None:
            return None
        endgame_solve = self.endgame_solve
        keys = jax.random.split(jax.random.fold_in(key, 2), batch_size)
        return jax.vmap(lambda k, e: endgame_solve.init(k, e, capacity))(
            keys, env_state
        )

    def init_resign_state(self, key: jax.Array, batch_size: int) -> ResignState | None:
        """`CollectionState`'s resignation states, for `batch_size` environments whose games start
        now; None without resignation."""
        if self.resignation is None:
            return None
        return jax.vmap(self.resignation.init)(
            jax.random.split(jax.random.fold_in(key, 1), batch_size)
        )

    @staticmethod
    def init_moves(key: jax.Array, batch_size: int) -> dict[str, Any]:
        """`CollectionState`'s moves in progress, for `batch_size` environments: none, so each
        environment's first cycle starts one."""
        return {
            # its first cycle sets it
            "move_key": jnp.zeros_like(jax.random.split(key, batch_size)),
            "search_cycles": jnp.zeros((batch_size,), dtype=jnp.int32),
        }

    def init_games(
        self, key: jax.Array, batch_size: int, evaluator: Evaluator
    ) -> tuple[Any, StepMetadata, Any]:
        """Initializes self-play's games, one per environment, and the evaluator's states.

        Args:
            key: rng
            batch_size: number of parallel environments
            evaluator: the self-play evaluator

        Returns:
            Tuple[Any, StepMetadata, Any]: the environment states, their metadata, and the
                evaluator's states
        """
        env_init_key, _ = jax.random.split(key)
        env_keys = jax.random.split(env_init_key, batch_size)
        # compiled, so that no two of the state's arrays are one array, as an environment's eager
        # init can return (e.g. its current player, in both the state and the metadata):
        # `collect_steps` can't be donated the same array twice
        env_state, metadata = jax.jit(jax.vmap(self.env_init_fn))(env_keys)
        return env_state, metadata, self.init_eval_state(evaluator, batch_size)

    def follow_schedule(
        self,
        epoch: int,
        evaluator: Evaluator,
        collection_state: CollectionState,
        learner: int = 0,
    ) -> tuple[Evaluator, CollectionState]:
        """Switches self-play to the evaluator scheduled for `epoch`, if it isn't `evaluator`:
        with `growth`, the evaluator of network `learner`, the one playing (so also when a
        network takes over).

        The new evaluator's states replace the old one's, initialized from scratch: they may
        differ in shape (e.g. MCTS trees with a different `max_nodes`). Games in progress
        carry on, their moves in progress searched again from the start. Self-play is compiled again for the new evaluator, the first time it plays.

        Args:
            epoch: the epoch about to start
            evaluator: the evaluator whose states `collection_state` holds
            collection_state: current collection state
            learner: (optional) with `growth`, the index of the network playing

        Returns:
            Tuple[Evaluator, CollectionState]: the evaluator for `epoch`, and the collection
                state with its states
        """
        scheduled = self.learners[learner].schedule.at(epoch)
        if scheduled is evaluator:
            return evaluator, collection_state
        whose = f" ({self.learners[learner].name})" if self.growth is not None else ""
        self.set_activity(
            f"epoch {epoch}: self-play switches to {describe(scheduled)}{whose} (compiling)",
            echo=True,
        )
        num_envs = collection_state.episodes.shape[0]
        eval_state = self.init_eval_state(scheduled, num_envs)
        # the moves in progress start their searches again
        return scheduled, replace(
            collection_state,
            eval_state=eval_state,
            search_cycles=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    def follow_lr_plateau(
        self,
        epoch: int,
        train_state: TrainState,
        progress: LRPlateauProgress,
        value: jax.Array | float,
    ) -> tuple[TrainState, LRPlateauProgress, dict, bool]:
        """Applies `lr_plateau`'s rule after `epoch`, whose `lr_plateau_metric` was `value`.

        Returns:
            Tuple[TrainState, LRPlateauProgress, dict, bool]: the train state, at the rate to train
                the next epoch at; the rule's progress; its metrics (see `lr_plateau`); and whether
                to stop training
        """
        rule = self.lr_plateau
        assert rule is not None
        index = int(lr_plateau_states(train_state.opt_state)[0].rate_index)
        step = rule.after_epoch(progress, index, float(value))
        metrics = {}
        if step.improvement is not None:
            metrics["lr_plateau_improvement"] = step.improvement
        if not step.plateaued:
            return train_state, step.progress, metrics, False
        rate = rule.rates[index]
        watched = (
            "the training loss"
            if self.lr_plateau_metric == "loss"
            else self.lr_plateau_metric
        )
        plateau = (
            f"epoch {epoch}: {watched} improved {step.improvement:.4f} over the "
            f"last {rule.window} epochs, under {rule.threshold:g}"
        )
        if step.index == index:
            metrics["lr_last_plateau"] = rate
            print(
                f"{plateau}: the last learning rate, {rate:g}, has plateaued"
                + (", stopping" if rule.stop else ""),
                flush=True,
            )
            return train_state, step.progress, metrics, rule.stop
        metrics["lr_drop"] = rule.rates[step.index]
        print(
            f"{plateau}: the learning rate drops from {rate:g} to "
            f"{rule.rates[step.index]:g}",
            flush=True,
        )
        train_state = replace(
            train_state, opt_state=with_rate_index(train_state.opt_state, step.index)
        )
        return train_state, step.progress, metrics, False

    def train_loop(
        self,
        seed: int,
        num_epochs: int,
        eval_every: int = 1,
        initial_state: TrainLoopOutput | None = None,
        fork_from: str | None = None,
        fork_replay_buffers: bool = True,
        fork_into: str | None = None,
        warm_start: Sequence[tuple[str, str]] = (),
    ) -> TrainLoopOutput:
        """Runs the training loop for `num_epochs` epochs. Mostly configured by the Trainer's attributes.

        - Collects self-play episdoes across a batch of environments.
        - Trains the neural network on the collected experiences.
        - Tests the agent on a set of Testers, which evaluate the agent's performance.

        Saves a checkpoint of the train state every epoch, and the whole training state after the
        epochs set by `save_state_at` and `save_state_every` (see `save_state`).

        A network whose optimizer state comes from a saved state (`initial_state`, `fork_from`,
        `warm_start`) trains with this trainer's injected hyperparameters, e.g. its weight decay
        and momentum, not those it was saved with (see `with_hyperparams`).

        Args:
            seed: rng seed (int)
            num_epochs: number of epochs to run the training loop for, in all: a continued run
                stops after this many epochs, counting those done before (or sooner, when
                `lr_plateau` stops it)
            eval_every: number of epochs between evaluations
            initial_state: (optional) TrainLoopOutput, used to continue training from a previous state
                - its collection state must hold the states of the self-play evaluator scheduled for
                  the epoch before `initial_state.cur_epoch`, as one this trainer returned does
                - with its `key` (as `train_loop` and `resume` return it), the run continues
                  exactly as if it hadn't stopped, and `seed` is unused. Without, the rng is
                  `seed`'s, folded with the epoch, and self-play starts with `warmup_steps`.
                - its collection state is donated to self-play (see `collect_steps`), so it
                  can't be used afterwards
            fork_from: (optional) a saved state (see `save_state`), or a directory to take the
                newest from, to fork a new run from. It's a run from scratch at epoch 0 (with its
                own schedules, rng, games and testers), except that it starts with the saved
                network, optimizer state (its learning rate schedule starting over) and weight
                averages (see `fork_averages`), and its replay buffers (see `fork_state`). The
                monitor run's config records it as `parent`. Not with `initial_state`.
            fork_replay_buffers: with `fork_from`, start with its replay buffers (default), or with
                empty ones
            fork_into: (with `fork_from`) the name of the saved network to fork, from a growth
                run's state too. With `growth`, it forks into this
                run's network of that name: it plays from the run's start, those before it are
                retired (training with `train_retired`), the next waits, and the rest start
                from scratch. Without growth, it forks into the Trainer's own network (one
                network of a growth run trained on alone). Without `fork_into`, the saved
                network forks into the Trainer's own, which plays
            warm_start: (with growth, for a new run) (name, saved state) pairs, a state or a
                directory to take the newest from: the network of that name starts from the saved
                state's network of the same name (or its only one) instead of from scratch, with
                its optimizer state, its step count and schedules back at 0 (see `warm_state`),
                and its weight averages (see `fork_averages`), e.g. a
                student trained beside a forked teacher. Not the network a fork goes into

        Returns:
            TrainLoopOutput: contains train_state, collection_state, test_states, cur_epoch after
                training loop, and the rng to continue it with
        """
        if fork_into is not None and fork_from is None:
            raise ValueError("fork_into is for a run forked from a saved state")
        if warm_start and self.growth is None:
            raise ValueError("warm_start names networks of a run with growth")
        if warm_start and initial_state is not None:
            raise ValueError(
                "warm_start is for a new run, not one continued from a state"
            )
        names = [learner.name for learner in self.learners]
        warmed = [name for name, _ in warm_start]
        for name in warmed:
            if name not in names:
                raise ValueError(
                    f"warm_start: no network {name} (this run has {', '.join(names)})"
                )
        if len(set(warmed)) != len(warmed):
            raise ValueError("warm_start names a network more than once")
        if fork_from is not None and (fork_into or names[0]) in warmed:
            raise ValueError(
                f"warm_start: {fork_into or names[0]} is the network the run forks into"
            )
        warm_start = [(name, latest_state_path(path)) for name, path in warm_start]
        warm_records = []
        for name, path in warm_start:
            meta = read_state_meta(path)
            warm_records.append(
                {
                    "network": name,
                    "path": os.path.abspath(path),
                    "epoch": meta["epoch"],
                    "monitor_run": meta["monitor_run"],
                }
            )
        if fork_from is not None:
            if initial_state is not None:
                raise ValueError("pass initial_state or fork_from, not both")
            fork_from = latest_state_path(fork_from)
            meta = read_state_meta(fork_from)
            self.parent = {
                "path": os.path.abspath(fork_from),
                "epoch": meta["epoch"],
                "replay_buffers": fork_replay_buffers,
                "monitor_run": meta["monitor_run"],
                "parent": meta["parent"],
            }
        elif initial_state is None:
            self.parent = None
        if self.monitor is not None:
            self.monitor.start(
                config={
                    **self.get_config(),
                    "run": {
                        "seed": seed,
                        "num_epochs": num_epochs,
                        "eval_every": eval_every,
                    },
                    **({"parent": self.parent} if self.parent is not None else {}),
                    **({"warm_start": warm_records} if warm_records else {}),
                    **self.extra_config,
                },
                # the last epoch
                total=num_epochs - 1,
            )
        self.move_log_writer = (
            MoveLogWriter(self.log_moves, self.batch_size)
            if self.log_moves is not None
            else None
        )
        try:
            output = self._train_loop(
                seed,
                num_epochs,
                eval_every,
                initial_state,
                fork_from,
                fork_replay_buffers,
                fork_into,
                warm_start,
            )
        except BaseException as e:
            if self.monitor is not None:
                self.monitor.finish(
                    "stopped" if isinstance(e, KeyboardInterrupt) else "crashed"
                )
            raise
        finally:
            if self.move_log_writer is not None:
                self.move_log_writer.close()
        if self.monitor is not None:
            self.monitor.finish()
        return output

    def log_learner_metrics(
        self, metrics: dict, epoch: int, learner: int, playing: int
    ) -> None:
        """Logs a network's metrics (or media): without `growth`, as they are; with it, as
        `<its name>/<metric>`, and as they are too if it's the network playing."""
        if self.growth is None:
            self.log_metrics(metrics, epoch)
            return
        name = self.learners[learner].name
        named = {f"{name}/{k}": v for k, v in metrics.items()}
        if learner == playing:
            named = {**metrics, **named}
        self.log_metrics(named, epoch)

    def _train_loop(
        self,
        seed: int,
        num_epochs: int,
        eval_every: int,
        initial_state: TrainLoopOutput | None,
        fork_from: str | None = None,
        fork_replay_buffers: bool = True,
        fork_into: str | None = None,
        warm_start: Sequence[tuple[str, str]] = (),
    ) -> TrainLoopOutput:
        started = time.perf_counter()
        # init rng
        key = jax.random.PRNGKey(seed)
        # an exact continuation picks up where the run stopped, without a warmup
        exact = initial_state is not None and initial_state.key is not None
        num_learners = len(self.learners)
        # the network a fork goes into (see `train_loop`'s `fork_into`)
        into = 0
        # the networks a warm start takes from saved states: (index, name, path)
        warm: list[tuple[int, str, str]] = []

        # initialize states
        if initial_state:
            collection_state = initial_state.collection_state
            train_states = list(initial_state.train_states)
            tester_states = [list(states) for states in initial_state.test_states]
            cur_epoch = initial_state.cur_epoch
            lr_progress: list[LRPlateauProgress | None] = (
                list(initial_state.lr_plateau) or [None] * num_learners
            )
            growth = initial_state.growth
            seconds = initial_state.seconds
            if initial_state.key is not None:
                key = initial_state.key
            else:
                # don't replay the keys the original run used from epoch 0
                key = jax.random.fold_in(key, cur_epoch)
            # the evaluator whose states the collection state holds
            evaluator = self.learners[initial_state.playing].schedule.at(
                max(cur_epoch - 1, 0)
            )
        else:
            cur_epoch = 0
            lr_progress = [None] * num_learners
            growth = self.growth.start() if self.growth is not None else None
            # the network the run starts playing with: the Trainer's own, or `fork_into`
            if fork_into is not None and self.growth is not None:
                names = [learner.name for learner in self.learners]
                if fork_into not in names:
                    raise ValueError(
                        f"fork_into: no network {fork_into} (this run has {', '.join(names)})"
                    )
                into = names.index(fork_into)
                assert growth is not None
                growth = replace(growth, playing=into)
            seconds = 0.0
            evaluator = self.learners[into].schedule.at(0)
            init_key, key = jax.random.split(key)
            if fork_from is None:
                # initialize collection state
                collection_state = self.init_collection_state(
                    init_key, self.batch_size, evaluator
                )
                # initialize train state
                forked = self.init_train_state()
            else:
                self.set_activity(f"loading {fork_from}", echo=True)
                collection_state, forked = self.fork_state(
                    fork_from, init_key, fork_replay_buffers, into, fork_into
                )
            train_states = [
                forked if i == into else self.init_train_state(i)
                for i in range(num_learners)
            ]
            names = [learner.name for learner in self.learners]
            for name, path in warm_start:
                index = names.index(name)
                self.set_activity(f"loading {name} from {path}", echo=True)
                train_states[index] = self.warm_state(path, index, name)
                warm.append((index, name, path))
            # initialize tester states
            tester_states = [
                [
                    tester.init(params=self.extract_model_params_fn(ts))
                    for tester in learner.testers
                ]
                for learner, ts in zip(self.learners, train_states, strict=True)
            ]

        # a saved optimizer state (resumed, branched, forked or warm started) takes this
        # trainer's fixed hyperparameters, e.g. a new weight decay or momentum
        for i, ts in enumerate(train_states):
            train_states[i] = self.with_fixed_hyperparams(ts, self.learners[i].name)

        # weight averages: the saved state's, or started at the networks' weights (a fork's
        # network goes on with its saved ones, if any)
        if initial_state is not None and initial_state.averages:
            averages = list(initial_state.averages)
            averaged = list(initial_state.averaged)
        else:
            averages = [self.fresh_averages(ts) for ts in train_states]
            averaged = [0] * num_learners
            if fork_from is not None:
                saved_averages = self.fork_averages(fork_from, into, fork_into)
                if saved_averages is not None:
                    averages[into], averaged[into] = saved_averages
            for index, name, path in warm:
                saved_averages = self.fork_averages(path, index, name)
                if saved_averages is not None:
                    averages[index], averaged[index] = saved_averages

        def playing_now() -> int:
            return growth.playing if growth is not None else 0

        def training_now() -> tuple[int, ...]:
            if self.train_steps_per_epoch == 0:
                return ()
            if growth is None or self.growth is None:
                return (0,)
            # the growth rule's (the one playing, and the next), and those retired before it;
            # the one playing only if it trains
            retired = range(growth.playing) if self.train_retired else ()
            return tuple(
                i
                for i in (*retired, *self.growth.training(growth))
                if self.train_playing or i != growth.playing
            )

        def state_now() -> TrainLoopOutput:
            return TrainLoopOutput(
                collection_state=collection_state,
                train_states=tuple(train_states),
                test_states=tuple(tester_states),
                cur_epoch=cur_epoch,
                key=key,
                lr_plateau=tuple(lr_progress) if self.lr_plateau is not None else (),
                growth=growth,
                seconds=seconds + time.perf_counter() - started,
                averages=tuple(averages) if self.averages else (),
                averaged=tuple(averaged) if self.averages else (),
            )

        def named(learner: int, text: str) -> str:
            if self.growth is None:
                return text
            return f"{text} ({self.learners[learner].name})"

        if self.move_log_writer is not None:
            collection_state = replace(collection_state, move_log=self.init_move_log())
        evaluator, collection_state = self.follow_schedule(
            cur_epoch, evaluator, collection_state, playing_now()
        )
        if self.lr_plateau is not None and lr_progress[playing_now()] is None:
            lr_progress[playing_now()] = LRPlateauProgress()
        params = self.selfplay_params(
            train_states[playing_now()], averages[playing_now()]
        )
        # exact endgame labels' solver, on a thread of its own
        solver = ThreadPoolExecutor(1) if self.endgame_solve is not None else None
        # the solve in progress, if any: of the positions waiting at the end of the last
        # self-play, for after the next
        solving = None
        if not exact:
            # warmup
            # populate replay buffer with initial self-play games
            if self.warmup_steps > 0:
                self.set_activity(f"warmup self-play ({self.warmup_steps} steps)")
            collect_key, key = jax.random.split(key)
            collect_keys = jax.random.split(collect_key, self.batch_size)
            collection_state = self.collect_steps(
                collect_keys,
                collection_state,
                params,
                self.warmup_steps,
                evaluator=evaluator,
            )

        # training loop
        while cur_epoch < num_epochs:
            epoch_started = time.perf_counter()
            playing, training = playing_now(), training_now()
            # collect self-play games, with the network playing
            evaluator, collection_state = self.follow_schedule(
                cur_epoch, evaluator, collection_state, playing
            )
            params = self.selfplay_params(train_states[playing], averages[playing])
            collect_key, key = jax.random.split(key)
            collect_keys = jax.random.split(collect_key, self.batch_size)
            # only the counts outlive the state donated to self-play: keeping the whole state
            # would keep a second copy of the replay buffers
            before = CollectionStats.of(collection_state)
            fresh_from = (
                jnp.copy(collection_state.buffer_state.next_idx)
                if self.fresh_positions and playing in training
                else None
            )
            # the positions waiting for exact solves are solved on the host from the end of one
            # epoch's self-play to the end of the next's, through training and tests: those of a
            # new or resumed run's start, during its first self-play
            if solver is not None and solving is None:
                assert collection_state.pending is not None
                solving = solver.submit(
                    self.solve_waiting,
                    jax.tree.map(np.asarray, collection_state.pending),
                )
            self.set_activity(f"epoch {cur_epoch}: self-play")
            collection_state = self.collect_steps(
                collect_keys,
                collection_state,
                params,
                self.collection_steps_per_epoch,
                evaluator=evaluator,
            )
            solve_metrics: dict[str, jax.Array] = {}
            if solver is not None and solving is not None:
                jax.block_until_ready(collection_state)
                waited = time.perf_counter()
                self.set_activity(f"epoch {cur_epoch}: waiting for solves")
                solved = [solving.result()]
                waited = time.perf_counter() - waited
                collection_state = self.label_solved_games(collection_state, solved[0])
                # this epoch's positions start solving now
                assert collection_state.pending is not None
                solving = solver.submit(
                    self.solve_waiting,
                    jax.tree.map(np.asarray, collection_state.pending),
                )
                if not self.memory_buffer.sample_mask(
                    collection_state.buffer_state
                ).any():
                    # nothing to train on yet (e.g. a new run's first epoch, whose games all
                    # stopped): they're labelled once solved
                    self.set_activity(f"epoch {cur_epoch}: solving")
                    started_waiting = time.perf_counter()
                    solved.append(solving.result())
                    solving = None
                    waited += time.perf_counter() - started_waiting
                    collection_state = self.label_solved_games(
                        collection_state, solved[-1]
                    )
                solve_metrics = solve_summary(solved, waited)
            if self.move_log_writer is not None:
                assert collection_state.move_log is not None
                self.move_log_writer.write(cur_epoch, collection_state.move_log)
                collection_state = replace(
                    collection_state, move_log=emptied(collection_state.move_log)
                )
            collection_metrics = {
                **self.collection_metrics(before, collection_state),
                **solve_metrics,
            }
            # self-play's time includes measuring its data, so that training's is its own
            jax.block_until_ready((collection_state, collection_metrics))
            selfplay_seconds = time.perf_counter() - epoch_started
            if isinstance(evaluator, MCTS):
                collection_metrics["search_iterations"] = jnp.asarray(
                    evaluator.mean_iterations()
                )
            # the cycles a move took, its search's and any it waited to be played
            cycles = (
                self.collection_steps_per_epoch
                * evaluator.search_cycles()
                * self.batch_size
            )
            moves = (collection_state.moves - before.moves).sum()
            collection_metrics["cycles_per_move"] = cycles / jnp.maximum(moves, 1)
            # the playing network on the positions it just stored, before it trains on them
            fresh_metrics: dict[str, jax.Array] = {}
            if fresh_from is not None:
                fresh_key, key = jax.random.split(key)
                weights = {"raw": train_states[playing].params}
                if self.averages:
                    horizon = self.selfplay_average or self.averages[0]
                    weights["avg"] = averages[playing][self.averages.index(horizon)]
                fresh_metrics = self.fresh_position_metrics(
                    fresh_key,
                    collection_state.buffer_state,
                    fresh_from,
                    weights,
                    train_states[playing].nn_state,
                    learner=playing,
                )
            # train every network that trains, on the same replay buffer
            train_metrics: dict[int, dict] = {}
            train_seconds: dict[int, float] = {}
            for i in training:
                train_started = time.perf_counter()
                self.set_activity(named(i, f"epoch {cur_epoch}: training"))
                train_key, key = jax.random.split(key)
                # epochs it has played before this one (without growth, the run's)
                played = growth.played[i] if growth is not None else cur_epoch
                # its learning rate: its own multiple of the optimizer's, and the drop's
                drop = self.lr_drop.scale(played) if self.lr_drop is not None else None
                own = self.learners[i].lr_scale
                lr_scale = None if drop is None and own == 1.0 else (drop or 1.0) * own
                collection_state, train_states[i], train_metrics[i] = self.train_steps(
                    train_key,
                    collection_state,
                    train_states[i],
                    self.steps_of(i, playing),
                    epoch=cur_epoch,
                    learner=i,
                    lr_scale=lr_scale,
                )
                if lr_scale is not None:
                    train_metrics[i] = {
                        **train_metrics[i],
                        "lr_scale": jnp.array(lr_scale, dtype=jnp.float32),
                    }
                jax.block_until_ready(train_states[i])
                train_seconds[i] = time.perf_counter() - train_started
                # its weight averages take in the epoch's weights
                averages[i] = tuple(
                    _move_average(
                        average,
                        train_states[i].params,
                        jnp.float32(
                            average_share(averaged[i], horizon, self.average_warmup)
                        ),
                    )
                    for average, horizon in zip(averages[i], self.averages, strict=True)
                )
                averaged[i] += 1
                if self.noise_measure and self.averages:
                    horizon = self.selfplay_average or self.averages[0]
                    measure_key, key = jax.random.split(key)
                    tr_sigma, grad_sq = self.noise_scale_at(
                        measure_key,
                        collection_state.buffer_state,
                        train_states[i],
                        averages[i][self.averages.index(horizon)],
                        num_micro=self.noise_measure,
                        learner=i,
                    )
                    train_metrics[i] = {
                        **train_metrics[i],
                        "gns_avg_tr_sigma": tr_sigma,
                        "gns_avg_grad_sq": grad_sq,
                        "gns_avg_noise_scale": jnp.where(
                            grad_sq > 0, tr_sigma / grad_sq, jnp.inf
                        ),
                    }
            if fresh_metrics and playing in train_metrics:
                train_metrics[playing] = {**train_metrics[playing], **fresh_metrics}
            # the learning rate rule follows the network playing
            stop = False
            watch_loss = self.lr_plateau_metric == "loss"
            if self.lr_plateau is not None and watch_loss:
                progress = lr_progress[playing]
                assert progress is not None
                train_states[playing], lr_progress[playing], lr_metrics, stop = (
                    self.follow_lr_plateau(
                        cur_epoch,
                        train_states[playing],
                        progress,
                        train_metrics[playing]["loss"],
                    )
                )
                train_metrics[playing] = {**train_metrics[playing], **lr_metrics}
            # log metrics
            self.log_metrics(collection_metrics, cur_epoch)
            for i in training:
                self.log_learner_metrics(train_metrics[i], cur_epoch, i, playing)

            # test every network that trains
            tested: dict[int, dict] = {i: {} for i in training}
            test_seconds: dict[int, float] = {}
            if cur_epoch % eval_every == 0:
                for i in training:
                    test_started = time.perf_counter()
                    learner = self.learners[i]
                    learner_params = self.extract_model_params_fn(train_states[i])
                    for j, test_state in enumerate(tester_states[i]):
                        tester = learner.testers[j]
                        run_key, key = jax.random.split(key)
                        self.set_activity(
                            named(i, f"epoch {cur_epoch}: testing {tester.name}")
                        )
                        new_test_state, metrics, episode = tester.run(
                            key=run_key,
                            epoch_num=cur_epoch,
                            max_steps=self.max_episode_steps,
                            env_step_fn=self.env_step_fn,
                            env_init_fn=self.test_env_init_fn,
                            evaluator=learner.evaluator_test,
                            state=test_state,
                            params=learner_params,
                            log_fn=partial(
                                self.log_learner_metrics,
                                epoch=cur_epoch,
                                learner=i,
                                playing=playing,
                            ),
                            activity_fn=lambda text, epoch=cur_epoch, i=i: (
                                self.set_activity(
                                    named(i, f"epoch {epoch}: {text}"), echo=True
                                )
                            ),
                        )

                        if metrics:
                            metrics = {k: v.mean() for k, v in metrics.items()}
                            self.log_learner_metrics(metrics, cur_epoch, i, playing)
                            tested[i].update(metrics)
                        # the run's views draw the episode, so the training loop never does
                        if episode is not None and self.monitor is not None:
                            self.log_learner_metrics(
                                {f"{tester.name}_game": episode}, cur_epoch, i, playing
                            )
                        # the network's weight averages, with keys of their own, so that the
                        # run's rng is the same with averages as without
                        if tester.stateless:
                            for n, (average, horizon) in enumerate(
                                zip(averages[i], self.averages, strict=True)
                            ):
                                _, average_metrics, _ = tester.run(
                                    key=jax.random.fold_in(run_key, n + 1),
                                    epoch_num=cur_epoch,
                                    max_steps=self.max_episode_steps,
                                    env_step_fn=self.env_step_fn,
                                    env_init_fn=self.test_env_init_fn,
                                    evaluator=learner.evaluator_test,
                                    state=test_state,
                                    params=self.extract_model_params_fn(
                                        replace(train_states[i], params=average)
                                    ),
                                )
                                if average_metrics:
                                    self.log_learner_metrics(
                                        {
                                            f"{average_name(horizon)}/{k}": v.mean()
                                            for k, v in average_metrics.items()
                                        },
                                        cur_epoch,
                                        i,
                                        playing,
                                    )
                        tester_states[i][j] = new_test_state
                    test_seconds[i] = time.perf_counter() - test_started
            # a tester's metric: the rule follows it once the epoch's tests have run
            if self.lr_plateau is not None and not watch_loss:
                if self.lr_plateau_metric not in tested[playing]:
                    raise ValueError(
                        f"lr_plateau_metric {self.lr_plateau_metric!r} wasn't logged at epoch "
                        f"{cur_epoch}: a tester must log it every epoch"
                    )
                progress = lr_progress[playing]
                assert progress is not None
                train_states[playing], lr_progress[playing], lr_metrics, stop = (
                    self.follow_lr_plateau(
                        cur_epoch,
                        train_states[playing],
                        progress,
                        tested[playing][self.lr_plateau_metric],
                    )
                )
                if lr_metrics:
                    self.log_learner_metrics(lr_metrics, cur_epoch, playing, playing)
            # growth: does the next network take over?
            if growth is not None and self.growth is not None:
                # the rule's metrics: a tester's, or the network's training's (e.g. its loss)
                logged: dict[str, dict[int, float]] = {}
                for metric in filter(None, (self.growth.metric, self.growth.guard)):
                    logged[metric] = {
                        i: float(source[i][metric])
                        for i in training
                        for source in (tested, train_metrics)
                        if metric in source[i]
                    }
                    if not self.growth.switch_at and len(logged[metric]) < len(
                        training
                    ):
                        raise ValueError(
                            f"growth's metric {metric!r} wasn't logged at epoch {cur_epoch} "
                            "by every network that trains: their testers or training must "
                            "log it every epoch"
                        )
                metric = self.growth.metric
                check = self.growth.after_epoch(
                    growth,
                    cur_epoch + 1,
                    logged[metric],
                    rule_done=stop,
                    guard_values=logged.get(self.growth.guard or ""),
                )
                growth = check.progress
                growth_metrics: dict = {"growth_playing": playing}
                if check.gap is not None:
                    growth_metrics["growth_gap"] = check.gap
                    growth_metrics["growth_caught_up"] = int(check.caught_up)
                if check.guard_gap is not None:
                    growth_metrics["growth_guard_gap"] = check.guard_gap
                if check.switched:
                    growth_metrics["growth_switch"] = growth.playing
                    taking_over = self.learners[growth.playing].name
                    why = (
                        f"{self.learners[playing].name}'s learning rate rule is done"
                        if stop
                        else f"{metric} {check.gap:+.4f} over the last {self.growth.window} "
                        "epochs"
                        if check.gap is not None and not self.growth.switch_at
                        else f"after {cur_epoch + 1} epochs"
                    )
                    print(
                        f"epoch {cur_epoch}: {taking_over} takes over self-play from "
                        f"{self.learners[playing].name} ({why})",
                        flush=True,
                    )
                    # its own learning rate rule starts, at the rate it has trained at
                    if self.lr_plateau is not None:
                        lr_progress[growth.playing] = LRPlateauProgress()
                    stop = False
                self.log_metrics(growth_metrics, cur_epoch)
            # save checkpoints
            self.set_activity(f"epoch {cur_epoch}: saving checkpoint")
            for i in training:
                self.save_checkpoint(train_states[i], cur_epoch, i)
                for average, horizon in zip(averages[i], self.averages, strict=True):
                    self.save_checkpoint(
                        replace(train_states[i], params=average),
                        cur_epoch,
                        i,
                        average=horizon,
                    )
            # the epoch's compute: self-play, and each network's training and tests, in wall time
            # and in network evaluations (under vmap every environment pays every cycle)
            train_samples = {
                i: self.steps_of(i, playing) * self.train_batch_size for i in training
            }
            timing = {
                "selfplay_seconds": selfplay_seconds,
                "train_seconds": sum(train_seconds.values()),
                "test_seconds": sum(test_seconds.values()),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "run_seconds": seconds + time.perf_counter() - started,
                "selfplay_evals": cycles,
                "train_samples": sum(train_samples.values()),
            }
            if self.growth is not None:
                for i in training:
                    name = self.learners[i].name
                    timing[f"{name}/train_seconds"] = train_seconds[i]
                    timing[f"{name}/train_samples"] = train_samples[i]
                    if i in test_seconds:
                        timing[f"{name}/test_seconds"] = test_seconds[i]
                timing[f"{self.learners[playing].name}/selfplay_evals"] = cycles
            self.log_metrics(timing, cur_epoch)
            # next epoch
            cur_epoch += 1
            if self.should_save_state(cur_epoch):
                self.set_activity(f"epoch {cur_epoch - 1}: saving the training state")
                self.save_state(state_now())
            if stop:
                break

        if solver is not None:
            solver.shutdown()
        if self.move_log_writer is not None:
            self.move_log_writer.flush()
        # return state so that training can be continued!
        return state_now()
