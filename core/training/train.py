import inspect
import json
import os
import re
import zipfile
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from seaglass.client import to_number

from core.common import step_with_output
from core.evaluators.evaluator import Evaluator
from core.memory.replay_memory import (
    BaseExperience,
    DataTransformFn,
    EpisodeReplayBuffer,
    ReplayBufferState,
    buffer_capacity,
    transform_randomly,
)
from core.pytree import map_where, where_tree
from core.training.endgame_solve import EndgameSolve, PendingSolves, Solved
from core.training.exploration import Exploration
from core.types import (
    EnvInitFn,
    EnvStepFn,
    LossFn,
    StateToNNInputFn,
    StepMetadata,
    TrainState,
)

if TYPE_CHECKING:
    # only a type: the trainer logs to whatever it's given
    from seaglass import Monitor


class Validator(Protocol):
    """Scores a network's weights on data training never sees (e.g.
    `core.training.validation.ScoredPositionsValidator`): called with (parameters, network
    state), it returns metrics by name. `get_config` describes it for the run's config."""

    def __call__(self, params: Any) -> dict[str, jax.Array]: ...

    def get_config(self) -> dict: ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CollectionState:
    """Self-play's state, carried from epoch to epoch.

    Attributes:
        eval_state: the evaluator's states, one per environment
        env_state: the environments' states
        buffer_state: the replay buffer's state
        metadata: the environments' metadata
        episodes: episodes each environment has finished so far (not counting truncated ones)
        moves: moves each environment has played so far
        episode_metrics: sums over the finished episodes of each metric `Trainer`'s
            `episode_metrics_fn` returns (empty without it)
        move_key: rng of the move in progress, its search's and the move's (see
            `Trainer.start_searches`)
        search_cycles: cycles the move's search has run so far (see `Evaluator.search_cycle`)
        pending: each environment's positions waiting for exact solves (see `Trainer`'s
            `endgame_solve`), None without
    """

    eval_state: Any
    env_state: Any
    buffer_state: ReplayBufferState
    metadata: StepMetadata
    episodes: jax.Array
    moves: jax.Array
    episode_metrics: dict[str, jax.Array]
    move_key: jax.Array
    search_cycles: jax.Array
    pending: PendingSolves | None = None


@dataclass(frozen=True)
class CollectionStats:
    """The counters of a collection state that `Trainer.collection_metrics` compares across an
    epoch's self-play: copies, which outlive the state donated to `Trainer.collect_steps`."""

    episodes: jax.Array
    moves: jax.Array
    episode_metrics: dict[str, jax.Array]
    solve_missed: jax.Array | None = None

    @classmethod
    def of(cls, state: CollectionState) -> "CollectionStats":
        return cls(
            episodes=jnp.copy(state.episodes),
            moves=jnp.copy(state.moves),
            episode_metrics={k: jnp.copy(v) for k, v in state.episode_metrics.items()},
            solve_missed=jnp.copy(state.pending.missed)
            if state.pending is not None
            else None,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainLoopOutput:
    """The training loop's state, as `Trainer.train_loop` returns it and `Trainer.save_state`
    saves it.

    Attributes:
        collection_state: self-play's state, with the replay buffer
        train_state: the network's weights, network state and optimizer state
        average: the network's weight average (see `Trainer`'s `average_epochs`), the weights
            self-play played the last epoch with
        averaged: the epochs in the average
        cur_epoch: epochs done
        key: the loop's rng, for the next epoch: a run continued from this state goes on as if
            it hadn't stopped
    """

    collection_state: CollectionState
    train_state: TrainState
    average: Any
    key: jax.Array
    averaged: int = field(default=0, metadata={"static": True})
    cur_epoch: int = field(default=0, metadata={"static": True})


def average_name(horizon: float) -> str:
    """The name of a weight average over about `horizon` epochs: its checkpoints' directory and
    its validation metrics' prefix, e.g. avg4."""
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
    """`average` moved toward `params` by `share` (with a share of 1, exactly the params), then
    each weight tensor weight decay shrinks (2 or more dimensions, see `weight_decay_mask`)
    rescaled to the norm of the same tensor in `params`: the average follows the weights'
    direction, not their scale.

    While weight decay shrinks the weights, as it does for tens of epochs after a learning rate
    drop, an average lags them in scale too. In a pre-activation trunk, whose stem and residual
    branches nothing normalizes, that scale error reaches the outputs: an Othello ConvNeXt
    net's 4-epoch average, 2-4% larger than its weights after a drop to 1e-4, rose to 0.40
    engine value MSE against its weights' 0.34; rescaled to their norms it scored 0.342. A
    3-seed A/B (2026-11-27) found plain averages behind on every seed."""

    def move(a: jax.Array, p: jax.Array) -> jax.Array:
        moved = (1 - share) * a + share * p
        if moved.ndim < 2:
            return moved
        norm = jnp.linalg.norm(moved)
        return jnp.where(norm > 0, moved * (jnp.linalg.norm(p) / norm), moved)

    return jax.tree.map(move, average, params)


def checkpoint_path(ckpt_dir: str, epoch: int) -> str:
    """Path of the weights checkpoint for `epoch` in `ckpt_dir`."""
    return os.path.join(ckpt_dir, f"{epoch}.eqx")


def checkpoint_epochs(ckpt_dir: str) -> list[int]:
    """Epochs with a weights checkpoint in `ckpt_dir`, in ascending order."""
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


def read_state_meta(path: str) -> dict:
    """The metadata of a saved state (see `Trainer.save_state`): the epochs done, the run's
    configuration, its monitor run and parent, and what each array is."""
    with np.load(path) as saved:
        return json.loads(str(saved["meta"]))


# saved states' format, recorded in their metadata
STATE_FORMAT = 3

# folded into the run's key for the average's own network state (see
# `Trainer.average_norm_state`), so that the run's rng is the same with it as without
AVERAGE_NORM_SALT = 0x61766E73


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
    weight decay it settles at about sqrt(2 * learning rate * weight decay * (1 - b1) / (1 + b1))
    per step. Comparable across learning rates and weight decays.
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


def injected_hyperparams(opt_state: Any) -> dict[str, jax.Array]:
    """The hyperparameters of the `optax.inject_hyperparams` optimizers in `opt_state`, by
    name; none if the optimizer doesn't inject any."""
    hyperparams = {}
    for state in jax.tree.leaves(opt_state, is_leaf=_is_injected):
        if _is_injected(state):
            hyperparams.update(state.hyperparams)
    return hyperparams


def with_hyperparams(opt_state: Any, hyperparams: Mapping[str, Any]) -> Any:
    """`opt_state` with its `optax.inject_hyperparams` optimizers' hyperparameters named in
    `hyperparams` set to those values, each in its array's dtype.

    An injected optimizer keeps its hyperparameters in its state: a restored state would go on
    with its saved values (e.g. its learning rate and weight decay), whatever the new run's
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


def solve_summary(solved: Sequence[Solved]) -> dict[str, jax.Array]:
    """An epoch's metrics of exact endgame labels (see `Trainer.collection_metrics`), from its
    solves: `solve_games`, the games labelled at their solve points, `solve_draw`, the fraction
    of them drawn, and `solve_positions`, every position solved."""
    whole = np.concatenate(
        [s.solves[s.whole] for s in solved if len(s.solves)] or [np.zeros((0, 0))]
    )
    return {
        "solve_games": jnp.asarray(len(whole)),
        "solve_positions": jnp.asarray(sum(len(s.solves) for s in solved)),
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
    """AlphaZero's training loop: self-play with a weight average of the network, training on
    the replay buffer, and validation of the average.

    Each epoch (see `train_loop`):
    1. the weight average moves toward the network's weights (see `average_epochs`)
    2. self-play plays with the average, storing its positions in the replay buffer
    3. with `endgame_solve`, the last epoch's exact solves are waited for and labelled
    4. self-play's metrics: what the epoch played and what the buffer holds (see
       `collection_metrics`)
    5. training: `train_steps_per_epoch` steps on the buffer, and their metrics
    6. with `holdout_envs`, memorization metrics (see `holdout_metrics`)
    7. with `validators`, the average's validation metrics
    8. the average's weights and the whole training state, saved if configured
    """

    def __init__(
        self,
        batch_size: int,
        train_batch_size: int,
        collection_steps_per_epoch: int,
        train_steps_per_epoch: int,
        nn: eqx.Module,
        loss_fn: LossFn,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator,
        memory_buffer: EpisodeReplayBuffer,
        max_episode_steps: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        state_to_nn_input_fn: StateToNNInputFn,
        nn_state: eqx.nn.State | None = None,
        grad_accum_steps: int = 1,
        average_epochs: float = 1.0,
        average_warmup: float = 10.0,
        average_norm_batches: int = 32,
        holdout_envs: int = 0,
        holdout_positions: int = 0,
        validators: Sequence[Validator] = (),
        validate_every: int = 1,
        exploration: Exploration | None = None,
        endgame_solve: EndgameSolve | None = None,
        episode_metrics_fn: Callable[[jax.Array], dict[str, jax.Array]] | None = None,
        data_transform_fns: Sequence[DataTransformFn] = (),
        advance_every: int | None = None,
        moves_at_once: int = 32,
        monitor: "Monitor | None" = None,
        ckpt_dir: str = "/tmp/turbozero_checkpoints",
        save_weights_every: int | None = None,
        save_state_every: int | None = None,
        save_state_at: Collection[int] = (),
        extra_config: dict | None = None,
    ):
        """
        Args:
            batch_size: self-play environments
            train_batch_size: samples per training step
            collection_steps_per_epoch: self-play moves' compute per environment each epoch
                (see `collect_steps`)
            train_steps_per_epoch: training steps each epoch. An epoch whose buffer can't yet
                sample a batch doesn't train
            nn: the network (an equinox module, see core.networks.utils.apply_nn); training
                starts from its parameters
            loss_fn: training's loss (see core.training.loss_fns)
            optimizer: an optax optimizer. Wrapped in `optax.inject_hyperparams`, its
                hyperparameters are recorded with the run, and a run continued, branched or
                forked from a saved state trains with this trainer's, not the state's (see
                `with_hyperparams`): e.g. a branch at a lower learning rate
            evaluator: self-play's search
            memory_buffer: the replay buffer
            max_episode_steps: self-play games still running after this many moves are
                truncated and their positions discarded
            env_step_fn: (env_state, action) -> (new_env_state, metadata)
            env_init_fn: (key) -> (env_state, metadata)
            state_to_nn_input_fn: an environment state as the network takes it
            nn_state: (optional) `nn`'s state for stateful networks (e.g. with BatchNorm), from
                `eqx.nn.make_with_state`
            grad_accum_steps: split each training batch into this many micro-batches and apply
                one update with their gradients' mean: training holds one micro-batch's
                activations at a time. It must divide `train_batch_size`. Micro-batch i takes
                samples i, i + n, i + 2n, ... BatchNorm normalizes each micro-batch with its own
                statistics, and its running statistics take the micro-batches' mean
            average_epochs: horizon, in epochs, of the weight average self-play plays with and
                validation scores (as KataGo plays and rates averaged weights): each epoch it
                moves toward the network's weights by `average_share`. 1 is the network's
                weights. Training itself is unchanged
            average_warmup: the epochs the average takes in for each epoch its horizon grows
                by, from 1 to `average_epochs` (see `average_share`)
            average_norm_batches: for a network with state (e.g. BatchNorm's statistics), the
                average gets its own each epoch, rather than using the network's: from the
                network's initial state, those of this many training micro-batches of the replay
                buffer through the average's weights (see `average_norm_state`). With a
                normalization after every convolution of an Othello ConvNeXt net, its 4-epoch
                average lost 0.3-0.5 engine regret on the network's statistics, and with its
                own its engine value error varied about 8x less from epoch to epoch. 0 for none
            holdout_envs: hold out the games of this many self-play environments (the last
                ones): stored like the others but never trained on, so that each epoch the
                network is scored on `holdout_positions` of their positions and as many of the
                trained-on games' of the same ages (see `holdout_metrics`): memorization of the
                replay buffer. A train loop starts their buffers empty, since a saved state's may
                hold positions trained on
            holdout_positions: with `holdout_envs`, the positions sampled from each side, with
                replacement. A multiple of the micro-batch
            validators: score the average each `validate_every` epochs, logged as
                `avg<horizon>/<metric>` (see `average_name`)
            validate_every: epochs between validations
            exploration: (optional) chooses the move self-play plays from the search's output,
                e.g. sampling from its policy early in the game (see core.training.exploration).
                The search's policy stays the training target
            endgame_solve: (optional) exact endgame labels (see core.training.endgame_solve):
                once a self-play game reaches a position the solver solves, on the host while
                self-play goes on, every position of the game before it gets its solved result
                as the value target. Positions solved in an epoch are labelled after the next
                epoch's self-play
            episode_metrics_fn: (optional) metrics of a finished episode from its final rewards
                (one per player), e.g. `lambda rewards: {"draw": (rewards == 0).all()}`, logged
                each epoch as `episode_<name>`, their mean over the episodes finished. Must be
                compatible with `jax.jit`
            data_transform_fns: (optional) data transforms (e.g. a board's rotations and
                reflections, see `core.memory.replay_memory.DataTransformFn`) besides the
                identity. Training maps each sample through one drawn uniformly at random.
                Together they must be a group, for `buffer_distinct_positions` to count
                positions up to them
            advance_every: (optional) self-play plays the moves whose searches are done every
                this many cycles (see `collect_steps`); by default the evaluator's
                `search_cycles`. With searches of varying lengths, a few cycles balance the
                cost of a check against a search waiting for it: about sqrt(2 * mean search
                cycles * cost of a check / cost of a cycle)
            moves_at_once: when self-play plays moves, the evaluator's states (e.g. the search
                trees) of the environments that played move on this many at a time (see
                `core.pytree.map_where`). Doesn't change the results
            monitor: (optional) `seaglass.Monitor` to log metrics to, e.g.
                `core.monitoring.Monitor`; a run is created on the first `train_loop`, and later
                calls (e.g. continuing from `initial_state`) keep logging to it
            ckpt_dir: directory to save to
            save_weights_every: (optional) save the average's weights and network state, all
                a player needs, every this many epochs, to `<ckpt_dir>/avg<horizon>/<epoch>.eqx`
                (see `save_weights`). Kept
            save_state_every: (optional) save the whole training state after every this many
                epochs (see `save_state`), keeping only the newest of these, to `resume` the run
                from. A state takes as much disk as the replay buffer's device memory
            save_state_at: (optional) numbers of epochs done after which to save the whole
                training state too, to branch or fork from. Kept
            extra_config: (optional) extra config to record with the monitor's run
        """
        # environment
        self.env_step_fn = env_step_fn
        self.env_init_fn = env_init_fn
        self.max_episode_steps = max_episode_steps
        self.template_env_state, _ = env_init_fn(jax.random.PRNGKey(0))
        # network and training
        self.state_to_nn_input_fn = state_to_nn_input_fn
        self.nn = nn
        self.nn_static = eqx.filter(nn, eqx.is_inexact_array, inverse=True)
        self.nn_state = nn_state
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        init_opt_state = jax.jit(optimizer.init)(eqx.filter(nn, eqx.is_inexact_array))
        # the injected hyperparameters, which a run from a saved state takes (see
        # `with_fixed_hyperparams`), and the config records
        self.hyperparams = injected_hyperparams(init_opt_state)
        self.train_batch_size = train_batch_size
        self.train_steps_per_epoch = train_steps_per_epoch
        if grad_accum_steps < 1 or train_batch_size % grad_accum_steps:
            raise ValueError(
                f"grad_accum_steps must be positive and divide train_batch_size, "
                f"{train_batch_size}, got {grad_accum_steps}"
            )
        self.grad_accum_steps = grad_accum_steps
        self.micro_batch_size = train_batch_size // grad_accum_steps
        self.transform_fns = data_transform_fns
        # self-play
        self.batch_size = batch_size
        self.collection_steps_per_epoch = collection_steps_per_epoch
        self.memory_buffer = memory_buffer
        self.evaluator = evaluator
        if advance_every is not None and advance_every < 1:
            raise ValueError(f"advance_every must be positive, got {advance_every}")
        self.advance_every = advance_every
        if moves_at_once < 1:
            raise ValueError(f"moves_at_once must be positive, got {moves_at_once}")
        self.moves_at_once = moves_at_once
        self.exploration = exploration
        self.endgame_solve = endgame_solve
        self.episode_metrics_fn = episode_metrics_fn
        self.count_distinct_observations = jax.jit(
            partial(
                self.memory_buffer.count_distinct_observations,
                transform_fns=tuple(self.transform_fns),
            )
        )
        # weight average
        if average_epochs < 1:
            raise ValueError(f"average_epochs must be at least 1, got {average_epochs}")
        self.average_epochs = float(average_epochs)
        if average_warmup <= 0:
            raise ValueError(f"average_warmup must be positive, got {average_warmup}")
        self.average_warmup = float(average_warmup)
        if average_norm_batches < 0:
            raise ValueError(
                f"average_norm_batches can't be negative, got {average_norm_batches}"
            )
        self.average_norm_batches = average_norm_batches
        # memorization
        if not 0 <= holdout_envs < batch_size:
            raise ValueError(
                f"holdout_envs must hold out fewer than the {batch_size} environments, got "
                f"{holdout_envs}"
            )
        if holdout_envs and (
            holdout_positions <= 0 or holdout_positions % self.micro_batch_size
        ):
            raise ValueError(
                f"holdout_positions ({holdout_positions}) must be a positive multiple of the "
                f"micro-batch ({self.micro_batch_size})"
            )
        self.holdout_envs = holdout_envs
        self.holdout_positions = holdout_positions
        # the environments whose games training samples: all but the held-out ones
        self.trained_envs = jnp.arange(batch_size) < batch_size - holdout_envs
        # validation
        self.validators = tuple(validators)
        if validate_every < 1:
            raise ValueError(f"validate_every must be positive, got {validate_every}")
        self.validate_every = validate_every
        # checkpoints
        self.ckpt_dir = ckpt_dir
        for every in (save_weights_every, save_state_every):
            if every is not None and every < 1:
                raise ValueError(f"save intervals must be positive, got {every}")
        self.save_weights_every = save_weights_every
        self.save_state_every = save_state_every
        self.save_state_at = frozenset(save_state_at)
        os.makedirs(self.weights_dir, exist_ok=True)
        # monitor
        self.monitor = monitor
        self.extra_config = extra_config if extra_config is not None else {}
        # the saved state this run was forked or branched from, recorded in the monitor's
        # config and in states the run saves
        self.parent: dict | None = None

    @property
    def weights_dir(self) -> str:
        """Where the average's weights are saved (see `save_weights`)."""
        return os.path.join(self.ckpt_dir, average_name(self.average_epochs))

    def init_train_state(self) -> TrainState:
        """The training state from `nn` and `nn_state`, with a new optimizer state."""
        params = eqx.filter(self.nn, eqx.is_inexact_array)
        return TrainState(
            params=params,
            nn_state=self.nn_state,
            opt_state=self.optimizer.init(params),
            step=jnp.array(0, dtype=jnp.int32),
        )

    def with_fixed_hyperparams(self, train_state: TrainState) -> TrainState:
        """`train_state` with its optimizer's injected hyperparameters this trainer's (see
        `with_hyperparams`), printing those that change, e.g. from a saved state's."""
        before = injected_hyperparams(train_state.opt_state)
        opt_state = with_hyperparams(train_state.opt_state, self.hyperparams)
        changed = [
            f"{k} {_to_config(before[k])} -> {_to_config(v)}"
            for k, v in injected_hyperparams(opt_state).items()
            if _to_config(before[k]) != _to_config(v)
        ]
        if changed:
            print(f"the optimizer's {', '.join(changed)}", flush=True)
        return replace(train_state, opt_state=opt_state)

    def get_config(self) -> dict:
        """The trainer's configuration, recorded with the monitor's run and saved states."""
        return {
            "batch_size": self.batch_size,
            "train_batch_size": self.train_batch_size,
            "collection_steps_per_epoch": self.collection_steps_per_epoch,
            "advance_every": self.advance_every,
            "train_steps_per_epoch": self.train_steps_per_epoch,
            "grad_accum_steps": self.grad_accum_steps,
            "max_episode_steps": self.max_episode_steps,
            "nn": self.nn.__class__.__name__,
            "nn_params": sum(
                x.size
                for x in jax.tree.leaves(eqx.filter(self.nn, eqx.is_inexact_array))
            ),
            "loss_fn": describe_fn(self.loss_fn),
            "optimizer": {k: _to_config(v) for k, v in self.hyperparams.items()}
            or None,
            "data_transforms": len(self.transform_fns),
            "average_epochs": self.average_epochs,
            "average_warmup": self.average_warmup,
            "average_norm_batches": self.average_norm_batches,
            "holdout_envs": self.holdout_envs,
            "holdout_positions": self.holdout_positions,
            "evaluator": self.evaluator.__class__.__name__,
            "evaluator_config": self.evaluator.get_config(),
            "exploration": self.exploration.get_config()
            if self.exploration is not None
            else None,
            "endgame_solve": self.endgame_solve.get_config()
            if self.endgame_solve is not None
            else None,
            "memory_buffer": self.memory_buffer.get_config(),
            "validators": [v.get_config() for v in self.validators],
            "validate_every": self.validate_every,
        }

    # self-play

    def move_keys(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Splits a move's key (see `CollectionState.move_key`) into its search's key and the
        key for choosing and playing the move (and starting a new game after it)."""
        key, evaluate_key = jax.random.split(key)
        return evaluate_key, key

    def start_searches(self, key: jax.Array, state: CollectionState) -> CollectionState:
        """Starts the search for a move, if one starts: if the last one's move was played (or
        none was in progress). `key` becomes the move's key, and the evaluator starts its
        search (see `Evaluator.start_search`). Otherwise, the state is left as it is."""
        starting = state.search_cycles == 0
        move_key = jnp.where(starting, key, state.move_key)
        evaluate_key, _ = self.move_keys(move_key)
        return replace(
            state,
            eval_state=self.evaluator.start_search(
                evaluate_key, state.eval_state, active=starting
            ),
            move_key=move_key,
        )

    def search(self, state: CollectionState, params: Any) -> CollectionState:
        """Runs a cycle of the search for the move in progress (see `Evaluator.search_cycle`).
        A search that is done goes on until its move is played: under vmap, its cycle is
        computed anyway."""
        evaluate_key, _ = self.move_keys(state.move_key)
        eval_state = self.evaluator.search_cycle(
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

    def play_moves(
        self, state: CollectionState, params: Any
    ) -> tuple[CollectionState, tuple[jax.Array, jax.Array, jax.Array]]:
        """Plays the move whose search is done, if it is (see `Evaluator.search_done`), all but
        moving the evaluator's state on (see `collect_steps`): stores its position in the
        replay buffer if the evaluator `trains_on` the search, and starts a new game once one
        ends.

        If the search isn't done, the state is left as it is. Under vmap, where that differs
        across environments, both are computed, but the replay buffer is only written where a
        move is played, rather than selected between whole buffers.

        Returns:
            the updated collection state, its evaluator's states as the search left them; and
            for `Evaluator.next_state`, whether a move was played, the move, and whether the
            episode ended
        """
        evaluator = self.evaluator
        done = evaluator.search_done(state.eval_state, state.search_cycles)
        # the moves whose positions are trained on (see `Evaluator.trains_on`); the others are
        # played, and count towards their game's length and outcome, without being stored
        stored = done & evaluator.trains_on(state.eval_state)
        evaluate_key, key = self.move_keys(state.move_key)
        output = evaluator.search_output(
            evaluate_key,
            state.eval_state,
            state.env_state,
            state.metadata,
            params,
            self.env_step_fn,
        )
        search_value = evaluator.get_value(output.eval_state)
        # exact endgame labels: a game stopping for its solves stops before its move
        solve_move = None
        stopped = jnp.array(False)
        if self.endgame_solve is not None and state.pending is not None:
            solve_move = self.endgame_solve.move(
                jax.random.fold_in(key, 3), state.pending, state.env_state, done, stored
            )
            stopped = solve_move.end
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
            end=stopped,
            step_evaluator=False,
        )
        terminated, truncated = done & terminated, done & truncated

        written = state.buffer_state.next_idx
        buffer_state = self.memory_buffer.add_experience(
            state=state.buffer_state,
            experience=BaseExperience(
                observation_nn=self.state_to_nn_input_fn(state.env_state),
                policy_mask=state.metadata.action_mask,
                policy_weights=eval_output.policy_weights,
                # its game's result replaces it; zeros, not `empty_like`: on GPU that's
                # uninitialized memory, which stays in the entries of games set aside or cut off
                reward=jnp.zeros_like(state.metadata.rewards),
                cur_player_id=state.metadata.cur_player_id,
                search_value=search_value,
            ),
            valid=stored,
        )
        pending = state.pending
        # the games that count as episodes now: with exact endgame labels, a game past its solve
        # point counted when that was labelled
        counted = terminated
        if pending is not None and solve_move is not None:
            assert self.endgame_solve is not None
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
                pending, terminated | truncated | stopped
            )
        buffer_state = where_tree(
            terminated,
            self.memory_buffer.assign_rewards(buffer_state, rewards),
            buffer_state,
        )
        # discard the episode's positions if it hit the step limit without terminating
        # (`truncated` is never set together with `terminated`)
        buffer_state = where_tree(
            truncated, self.memory_buffer.truncate(buffer_state), buffer_state
        )
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
            pending=pending,
        )
        return state, (done, eval_output.action, terminated | truncated | stopped)

    # `self` is static, hashing by identity. `state` is donated: it holds the replay buffer, so
    # self-play updates it in place rather than in a second copy
    @partial(jax.jit, static_argnums=(0, 4), donate_argnames=("state",))
    def collect_steps(
        self,
        key: jax.Array,
        state: CollectionState,
        params: Any,
        num_steps: int,
    ) -> CollectionState:
        """Self-play for the compute of `num_steps` moves in every environment: as many cycles
        as `num_steps` moves' searches take at most (see `Evaluator.search_cycles`).

        Each cycle, every environment runs a cycle of its move's search (`search`); every
        `advance_every` cycles, those whose search is done play their moves (`play_moves`, the
        evaluator's states moved on only for the moves played, `moves_at_once` at a time). With
        searches that all take the same cycles, from environments that start together, and the
        default `advance_every`, that's `num_steps` moves in each.

        Args:
            key: rng, one key per environment
            state: collection state, donated: its arrays are deleted, and can't be used after
                the call
            params: the network's (parameters, network state) to play with
            num_steps: number of moves' compute
        """
        evaluator = self.evaluator
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
            write the trees, in place. Moves start only in the first (after moves are played,
            or at the call's start), so their searches start before it, rather than in every
            cycle."""
            starting = state.search_cycles == 0
            state = self.start_searches(keys[started], state)
            state = jax.lax.fori_loop(
                0, num, lambda _, s: self.search(s, params), state
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
                lambda s: self.play_moves(s, params)
            )(state)
            # everything read from the evaluator's states (to play the moves), then they move
            # on in place: the barrier makes the loop that writes them depend on every read.
            # XLA otherwise copies the arrays read to keep them for reads it doesn't order
            # first (see `Tree.set_child`)
            state, (done, action, ended) = jax.lax.optimization_barrier(
                (state, (done, action, ended))
            )
            # the evaluator's states (e.g. the search trees) move on only where a move was
            # played: under vmap, every one would be read and written
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
        # whole state (the trees and the replay buffer)
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

    def collection_metrics(
        self, before: CollectionStats, after: CollectionState
    ) -> dict:
        """What an epoch's self-play played and what the replay buffer holds after it:
        - `moves`, `episodes`: moves played and episodes finished in all environments
        - with `episode_metrics_fn`: `episode_<name>`, each metric's mean over those episodes,
          omitted if none finished
        - `buffer_samples`: positions in the replay buffer training can sample
        - `buffer_distinct_positions`: distinct observations among them, those a data transform
          maps to one another counting as one (e.g. a board and its rotations), and
          `buffer_distinct_fraction`, that as a fraction of `buffer_samples`: to spot self-play
          collapsing into the same few games
        - with `endgame_solve`: `solve_missed`, the solves that found every slot full (the
          training loop adds `solve_summary`'s)

        Args:
            before: the collection state's counters before the epoch's self-play
            after: the collection state after it
        """
        episodes = (after.episodes - before.episodes).sum()
        distinct, total = self.count_distinct_observations(after.buffer_state)
        metrics = {
            "moves": (after.moves - before.moves).sum(),
            "episodes": episodes,
            "buffer_samples": total,
            "buffer_distinct_positions": distinct,
            "buffer_distinct_fraction": distinct / jnp.maximum(total, 1),
        }
        if episodes > 0:
            for name, total_after in after.episode_metrics.items():
                change = (total_after - before.episode_metrics[name]).sum()
                metrics[f"episode_{name}"] = change / episodes
        if after.pending is not None and before.solve_missed is not None:
            metrics["solve_missed"] = (after.pending.missed - before.solve_missed).sum()
        return metrics

    # training

    def train_step(
        self, ts: TrainState, batch: BaseExperience
    ) -> tuple[TrainState, dict]:
        """One optimizer step on `batch`, in micro-batches of `micro_batch_size` samples (see
        `grad_accum_steps`). Returns the new state, and the loss function's metrics with
        `loss`, `grad_norm`, `weight_norm` and `update_ratio` (see `update_size_metrics`)."""
        nn = eqx.combine(ts.params, self.nn_static)
        grad_fn = eqx.filter_value_and_grad(self.loss_fn, has_aux=True)
        num_micro = jax.tree.leaves(batch)[0].shape[0] // self.micro_batch_size
        if num_micro == 1:
            (loss, (metrics, nn_state)), grads = grad_fn(nn, ts.nn_state, batch)
        else:
            loss, metrics, nn_state, grads = self.accumulate_grads(
                grad_fn, nn, ts.params, ts.nn_state, batch, num_micro
            )
        updates, opt_state = self.optimizer.update(grads, ts.opt_state, ts.params)
        metrics = {
            **metrics,
            **update_size_metrics(ts.params, updates),
            "loss": loss,
            "grad_norm": optax.tree.norm(grads).astype(jnp.float32),
        }
        ts = replace(
            ts,
            params=optax.apply_updates(ts.params, updates),
            nn_state=nn_state,
            opt_state=opt_state,
            step=ts.step + 1,
        )
        return ts, metrics

    def accumulate_grads(
        self,
        grad_fn: Callable,
        nn: eqx.Module,
        params: Any,
        nn_state: eqx.nn.State | None,
        batch: BaseExperience,
        n: int,
    ) -> tuple[jax.Array, dict, eqx.nn.State | None, Any]:
        """The loss and gradients of a batch in `n` micro-batches, one after another in a
        `jax.lax.scan` (see `grad_accum_steps`): their mean loss, mean metrics, mean network
        state (the last of non-float leaves, e.g. step counters) and mean gradients."""
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
            return grads, (loss, metrics, new_state)

        grads, (losses, metrics, states) = jax.lax.scan(
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
        )

    @partial(jax.jit, static_argnums=(0, 4))
    def train_epoch(
        self,
        key: jax.Array,
        buffer_state: ReplayBufferState,
        train_state: TrainState,
        num_steps: int,
    ) -> tuple[TrainState, dict]:
        """`num_steps` training steps in one `jax.lax.scan`, each on a batch sampled without
        replacement from the positions training can sample (see `training_mask`), each sample
        mapped through a random data transform (see `data_transform_fns`). Returns the new
        state and the steps' mean metrics."""
        # the buffer doesn't change while training, so find which entries can be sampled once
        mask = self.training_mask(buffer_state)

        def step(
            carry: tuple[jax.Array, TrainState], _
        ) -> tuple[tuple[jax.Array, TrainState], dict]:
            key, ts = carry
            step_key, transform_key, key = jax.random.split(key, 3)
            indices = jnp.unravel_index(
                self.memory_buffer.sample_indices(
                    step_key, mask, self.train_batch_size
                ),
                mask.shape,
            )
            batch = jax.tree.map(lambda x: x[indices], buffer_state.buffer)
            batch = transform_randomly(transform_key, batch, self.transform_fns)
            ts, metrics = self.train_step(ts, batch)
            return (key, ts), metrics

        (_, train_state), metrics = jax.lax.scan(
            step, (key, train_state), length=num_steps
        )
        return train_state, jax.tree.map(jnp.mean, metrics)

    def training_mask(self, buffer_state: ReplayBufferState) -> jax.Array:
        """The replay buffer entries training samples: those the buffer can sample (see
        `EpisodeReplayBuffer.sample_mask`), of the environments whose games aren't held out (see
        `holdout_envs`)."""
        return self.memory_buffer.sample_mask(buffer_state) & self.trained_envs[:, None]

    def can_train(self, buffer_state: ReplayBufferState) -> bool:
        """Whether training can sample a batch (see `training_mask`), on the host."""
        return int(self.training_mask(buffer_state).sum()) >= self.train_batch_size

    def resize_replay_buffer(
        self, state: CollectionState, capacity: int
    ) -> CollectionState:
        """A collection state's replay buffer at another capacity, between epochs: each
        environment keeps its newest entries, as many as fit (see
        `EpisodeReplayBuffer.resized`). Games in progress carry on, entries and all, and so do
        games waiting for exact solves. Done on the host: it takes about the host memory of the
        old buffer and the new one."""
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
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
        return resized

    # the weight average

    def average_norm_state(
        self, key: jax.Array, buffer_state: ReplayBufferState, average: Any
    ) -> Any:
        """The average's own network state (e.g. BatchNorm's statistics), with
        `average_norm_batches`: from the network's initial state, updated in training mode on
        that many micro-batches of the replay buffer through the average's weights. None
        without it, for a network without state, or while the buffer holds fewer positions
        training can sample than a micro-batch: the average then uses the network's state."""
        if not self.average_norm_batches or self.nn_state is None:
            return None
        mask = self.training_mask(buffer_state)
        if int(mask.sum()) < self.micro_batch_size:
            return None
        return self._norm_state(key, buffer_state, mask, average)

    @partial(jax.jit, static_argnums=(0,))
    def _norm_state(
        self,
        key: jax.Array,
        buffer_state: ReplayBufferState,
        mask: jax.Array,
        params: Any,
    ) -> Any:
        nn = eqx.combine(params, self.nn_static)

        def one(state: Any, k: jax.Array) -> tuple[Any, None]:
            sample_key, transform_key = jax.random.split(k)
            indices = jnp.unravel_index(
                self.memory_buffer.sample_indices(
                    sample_key, mask, self.micro_batch_size
                ),
                mask.shape,
            )
            batch = jax.tree.map(lambda x: x[indices], buffer_state.buffer)
            batch = transform_randomly(transform_key, batch, self.transform_fns)
            _, (_, state) = self.loss_fn(nn, state, batch)
            return state, None

        state, _ = jax.lax.scan(
            one, self.nn_state, jax.random.split(key, self.average_norm_batches)
        )
        return state

    # memorization

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

    def holdout_metrics(
        self,
        key: jax.Array,
        buffer_state: ReplayBufferState,
        weights: Mapping[str, Any],
    ) -> dict[str, jax.Array]:
        """Scores each of `weights` (by name: (parameters, network state)) on positions of the
        held-out games (see `holdout_envs`) and on as many of the trained-on games' positions of
        the same ages, `holdout_positions` each, sampled with replacement. In inference mode, so
        a BatchNorm network uses its running statistics. Empty until a held-out game has
        finished and a trained-on game's position is no older than its oldest.

        For each name it logs the policy KL and value loss on both, `heldout_*_<name>` and
        `seen_*_<name>`, and the excess on the held-out positions, `memorized_*_<name>`: what
        training has fit in the positions it trains on that doesn't carry over to new games of
        the same kind. Each excess has a standard error (`*_se`), from the spread of its
        micro-batches' means. Also `heldout_stored`, the held-out positions stored."""
        stored = self.memory_buffer.sample_mask(buffer_state)
        held = stored & ~self.trained_envs[:, None]
        if not bool(held.any()):
            return {}
        # the held-out buffers start empty at the train loop's start: the trained-on positions
        # compared with them are those no older than their oldest
        capacity = stored.shape[1]
        age = (buffer_state.next_idx[:, None] - 1 - jnp.arange(capacity)) % capacity
        oldest = jnp.max(jnp.where(held, age, -1))
        seen = stored & self.trained_envs[:, None] & (age <= oldest)
        if not bool(seen.any()):
            return {}
        n = self.holdout_positions // self.micro_batch_size
        held_key, seen_key = jax.random.split(key)
        held_batches = self._gather_masked(held_key, buffer_state, held, n)
        seen_batches = self._gather_masked(seen_key, buffer_state, seen, n)

        def se(x: jax.Array) -> jax.Array:
            return jnp.sqrt(jnp.var(x, ddof=1) / n) if n > 1 else jnp.array(jnp.nan)

        out: dict[str, jax.Array] = {"heldout_stored": held.sum()}
        for name, params in weights.items():
            held_losses = self._probe_losses(params, held_batches)
            seen_losses = self._probe_losses(params, seen_batches)
            for metric, h, t in zip(
                ("policy_kl", "value_loss"), held_losses, seen_losses, strict=True
            ):
                out[f"heldout_{metric}_{name}"] = h.mean()
                out[f"seen_{metric}_{name}"] = t.mean()
                out[f"memorized_{metric}_{name}"] = h.mean() - t.mean()
                out[f"memorized_{metric}_{name}_se"] = jnp.sqrt(
                    jnp.square(se(h)) + jnp.square(se(t))
                )
        return out

    @partial(jax.jit, static_argnums=(0,))
    def _probe_losses(
        self, params: Any, batches: BaseExperience
    ) -> tuple[jax.Array, jax.Array]:
        """Each micro-batch's mean policy KL and value loss, for the network with `params`
        ((parameters, network state)) in inference mode."""
        nn_params, nn_state = params
        nn = eqx.nn.inference_mode(eqx.combine(nn_params, self.nn_static))

        def one(_, batch: BaseExperience) -> tuple[None, tuple[jax.Array, jax.Array]]:
            _, (metrics, _) = self.loss_fn(nn, nn_state, batch)
            return None, (metrics["policy_kl"], metrics["value_loss"])

        _, (kl, value) = jax.lax.scan(one, None, batches)
        return kl, value

    def without_holdout_positions(
        self, buffer_state: ReplayBufferState
    ) -> ReplayBufferState:
        """`buffer_state` with the held-out environments' finished games' positions dropped (see
        `holdout_envs`): those of a saved state may have been trained on. Positions waiting for
        their game's result never were, so they stay."""
        held = ~self.trained_envs[:, None] & buffer_state.has_reward
        return replace(buffer_state, populated=buffer_state.populated & ~held)

    # logging

    def log_metrics(self, metrics: dict, epoch: int):
        """Logs metrics to the console and the monitor."""
        metrics_str = {
            k: f"{number:.4f}"
            for k, v in metrics.items()
            if (number := to_number(v)) is not None
        }
        print(f"Epoch {epoch}: {metrics_str}")
        if self.monitor is not None:
            self.monitor.log(epoch, metrics)

    def set_activity(self, text: str | None, echo: bool = False) -> None:
        """Tells the monitor what the training loop is doing now, so the run's page can show
        what a slow step is busy with; with `echo`, prints it too."""
        if echo and text is not None:
            print(text, flush=True)
        if self.monitor is not None:
            self.monitor.activity(text)

    # saving and loading

    def save_weights(self, params: Any, epoch: int) -> str:
        """Saves the average's (parameters, network state) for `epoch` to `weights_dir`: what a
        player loads (e.g. `eqx.tree_deserialise_leaves(f, (params, nn_state))`). Written to a
        temporary file first, so an interrupted save leaves no partial checkpoint."""
        os.makedirs(self.weights_dir, exist_ok=True)
        path = checkpoint_path(self.weights_dir, epoch)
        with open(path + ".tmp", "wb") as f:
            eqx.tree_serialise_leaves(f, params)
        os.replace(path + ".tmp", path)
        return path

    def should_save_state(self, epochs_done: int) -> bool:
        """Whether to save the whole training state after `epochs_done` epochs (see
        `save_state_at` and `save_state_every`)."""
        return epochs_done in self.save_state_at or (
            self.save_state_every is not None
            and epochs_done % self.save_state_every == 0
        )

    def save_state(self, state: TrainLoopOutput) -> str:
        """Saves the whole training state to `ckpt_dir`, as `state-<epochs done>.npz`: the
        train state, the weight average, the collection state (with the replay buffer), the
        epochs done and the loop's rng. The run continues from it exactly (see `resume`), and
        others branch or fork from it (see `branch` and `train_loop`'s `fork_from`). It also
        records the run's configuration, its monitor run, and its parent, if any (see
        `read_state_meta`).

        With `save_state_every`, deletes the states saved before it at multiples of it, besides
        those at `save_state_at`. Arrays are copied to the host one at a time, as they're
        written, so it takes about the host memory of the largest."""
        parts = {
            "train_state": state.train_state,
            "average": state.average,
            "collection_state": state.collection_state,
        }
        meta = {
            "format": STATE_FORMAT,
            "epoch": state.cur_epoch,
            "num_envs": int(state.collection_state.episodes.shape[0]),
            "buffer_capacity": buffer_capacity(state.collection_state.buffer_state),
            "averaged": state.averaged,
            "monitor_run": self.monitor.run_id if self.monitor is not None else None,
            "parent": self.parent,
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
        # as np.savez does, but copying one array at a time to the host (see `_to_host`)
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

    def _read_state(self, path: str, saved: Any) -> dict:
        meta = json.loads(str(saved["meta"]))
        if meta.get("format") != STATE_FORMAT:
            raise ValueError(
                f"{path} is a saved state of format {meta.get('format')}, this version reads "
                f"format {STATE_FORMAT}"
            )
        return meta

    def _load_part(self, saved: Any, meta: dict, part: str, template: Any) -> Any:
        """Loads part of a saved state into new device arrays, checking each array's shape and
        dtype against `template`'s (e.g. from `jax.eval_shape`)."""
        expected = _describe(template)
        described = [
            (leaf_path, tuple(shape), dtype)
            for leaf_path, shape, dtype in meta["leaves"][part]
        ]
        name = part.replace("_", " ")
        mismatched = [
            f"  {leaf_path}: saved {s_shape} {s_dtype}, this run's {shape} {dtype}"
            for (leaf_path, shape, dtype), (_, s_shape, s_dtype) in zip(
                expected, described, strict=False
            )
            if (s_shape, s_dtype) != (shape, dtype)
        ]
        if len(described) != len(expected) or mismatched:
            raise ValueError(
                f"the saved {name} doesn't match this run's ({len(described)} arrays saved, "
                f"{len(expected)} here):\n" + "\n".join(mismatched[:8])
            )
        return jax.tree.unflatten(
            jax.tree.structure(template),
            [_to_device(saved[f"{part}.{i}"]) for i in range(len(expected))],
        )

    def _load_collection_state(
        self, path: str, saved: Any, meta: dict, strict: bool = False
    ) -> tuple[CollectionState, bool]:
        """A saved collection state, at its saved capacity: its search states if they have this
        evaluator's shapes, else new ones, the moves in progress searched again; with
        `endgame_solve`, the saved pending solves if they have this run's shapes, else none
        waiting. With `strict`, every array must fit, as an exact continuation needs. Returns
        it, and whether the search states were kept."""
        if meta["num_envs"] != self.batch_size:
            raise ValueError(
                f"{path} was saved with {meta['num_envs']} self-play environments, this run "
                f"has {self.batch_size}: they must be the same"
            )
        template = jax.eval_shape(
            lambda: self.init_collection_state(
                jax.random.PRNGKey(0), meta["buffer_capacity"]
            )
        )
        expected = _describe(template)
        described = [
            (leaf_path, tuple(shape), dtype)
            for leaf_path, shape, dtype in meta["leaves"]["collection_state"]
        ]
        index = {leaf_path: i for i, (leaf_path, _, _) in enumerate(described)}
        # the parts that start anew when the saved ones don't fit; the rest must
        groups = [] if strict else [".eval_state", ".pending"]

        def group(leaf_path: str) -> str | None:
            return next((g for g in groups if leaf_path.startswith(g)), None)

        def fits(name: str) -> bool:
            return [d for d in described if group(d[0]) == name] == [
                e for e in expected if group(e[0]) == name
            ]

        kept = {name: fits(name) for name in groups}
        mismatched = [
            f"  {leaf_path}: saved "
            + (
                f"{described[index[leaf_path]][1]} {described[index[leaf_path]][2]}"
                if leaf_path in index
                else "nothing"
            )
            + f", this run's {shape} {dtype}"
            for leaf_path, shape, dtype in expected
            if group(leaf_path) is None
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
        leaves = [
            _to_device(saved[f"collection_state.{index[leaf_path]}"])
            if kept.get(group(leaf_path) or "", True)
            else jnp.zeros(shape, dtype)  # replaced below
            for leaf_path, shape, dtype in expected
        ]
        state = jax.tree.unflatten(jax.tree.structure(template), leaves)
        if not kept.get(".pending", True):
            if any(group(d[0]) == ".pending" for d in described):
                print(
                    "the saved games waiting for solves don't fit this run's, so they're "
                    "dropped",
                    flush=True,
                )
            state = replace(
                state,
                pending=self.init_pending(state.env_state, meta["buffer_capacity"]),
            )
        if not kept.get(".eval_state", True):
            # the moves in progress start their searches again
            state = replace(
                state,
                eval_state=self.init_eval_state(),
                search_cycles=jnp.zeros_like(state.search_cycles),
            )
        return state, kept.get(".eval_state", True)

    def _load_networks(self, saved: Any, meta: dict) -> tuple[TrainState, Any]:
        """A saved state's train state and weight average."""
        template = jax.eval_shape(self.init_train_state)
        train_state = self._load_part(saved, meta, "train_state", template)
        average = self._load_part(saved, meta, "average", template.params)
        return train_state, average

    def _fit_buffer(self, state: CollectionState, what: str) -> CollectionState:
        """`state`'s replay buffer at this trainer's capacity, resized if it differs."""
        capacity = buffer_capacity(state.buffer_state)
        if capacity == self.memory_buffer.capacity:
            return state
        print(
            f"{what}: the replay buffer is resized from {capacity} to "
            f"{self.memory_buffer.capacity} entries per environment",
            flush=True,
        )
        return self.resize_replay_buffer(state, self.memory_buffer.capacity)

    def load_state(self, path: str) -> TrainLoopOutput:
        """Loads a state saved by `save_state`, to continue its run exactly (see `resume`). The
        trainer must be configured as the run's was, so that every array fits (`branch`
        continues one with other settings); the replay buffer keeps the capacity it was saved
        with."""
        with np.load(path) as saved:
            meta = self._read_state(path, saved)
            train_state, average = self._load_networks(saved, meta)
            collection_state, _ = self._load_collection_state(
                path, saved, meta, strict=True
            )
            key = _to_device(saved["key"])
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            average=average,
            key=key,
            averaged=meta["averaged"],
            cur_epoch=meta["epoch"],
        )

    def _check_config(self, path: str, meta: dict) -> None:
        current = json.loads(
            json.dumps({**self.get_config(), **self.extra_config}, default=str)
        )
        current, saved = _flatten(current), _flatten(meta["config"])
        changed = sorted(
            k for k in current.keys() | saved.keys() if current.get(k) != saved.get(k)
        )
        if changed:
            print(f"{path} was saved with a different configuration: {changed}")

    def resume(self, path: str | None = None) -> TrainLoopOutput:
        """Loads the newest state saved in `ckpt_dir` (or `path`, a saved state or a directory
        of them) to continue its run exactly, as `train_loop`'s `initial_state`: given
        deterministic computations, the same as if it hadn't stopped. It logs to the run's
        monitor run, and deletes the weights saved after the state, which it saves again."""
        path = latest_state_path(path if path is not None else self.ckpt_dir)
        state = self.load_state(path)
        meta = read_state_meta(path)
        self._check_config(path, meta)
        self.parent = meta["parent"]
        if self.monitor is not None and self.monitor.run_id is None:
            self.monitor.run_id = meta["monitor_run"]
        for epoch in checkpoint_epochs(self.weights_dir):
            if epoch >= state.cur_epoch:
                os.remove(checkpoint_path(self.weights_dir, epoch))
        print(f"resuming from {path}: {state.cur_epoch} epochs done", flush=True)
        return state

    def branch(self, path: str) -> TrainLoopOutput:
        """Loads a saved state (or the newest in a directory) to continue its run with this
        trainer's settings, in this trainer's `ckpt_dir`, as `train_loop`'s `initial_state`:
        e.g. at a lower learning rate, or searching more. The run goes on from where it was
        saved: its epochs, network, optimizer state, weight average, replay buffer (resized to
        this trainer's capacity), games in progress and rng; its search states are kept if
        they fit this trainer's search, else the moves in progress are searched again. It logs
        to a monitor run of its own, whose config records the state as its parent."""
        path = latest_state_path(path)
        with np.load(path) as saved:
            meta = self._read_state(path, saved)
            train_state, average = self._load_networks(saved, meta)
            collection_state, kept_search = self._load_collection_state(
                path, saved, meta
            )
            key = _to_device(saved["key"])
        collection_state = self._fit_buffer(collection_state, "branch")
        self.parent = {
            "path": os.path.abspath(path),
            "epoch": meta["epoch"],
            "branch": True,
            "monitor_run": meta["monitor_run"],
            "parent": meta["parent"],
        }
        search = "its search's states kept" if kept_search else "new search states"
        print(
            f"branching from {path}: {meta['epoch']} epochs done, {search}", flush=True
        )
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            average=average,
            key=key,
            averaged=meta["averaged"],
            cur_epoch=meta["epoch"],
        )

    def fork(
        self, path: str, key: jax.Array, replay_buffer: bool = True
    ) -> TrainLoopOutput:
        """A new run's start from a saved state (see `train_loop`'s `fork_from`): at epoch 0,
        with its own rng, but the saved network, optimizer state and weight average, and with
        `replay_buffer` the saved collection state (resized to this trainer's capacity), so
        that its self-play goes on as the saved run's would: the replay buffer, the games in
        progress, those waiting for exact solves, and the search states if they fit. Starting
        new games instead would leave the fork's first epochs training on a smaller, staler
        buffer."""
        with np.load(path) as saved:
            meta = self._read_state(path, saved)
            train_state, average = self._load_networks(saved, meta)
            if replay_buffer:
                collection_state, kept_search = self._load_collection_state(
                    path, saved, meta
                )
                if not kept_search:
                    print(
                        "fork: the saved search states don't fit, the moves in progress "
                        "are searched again",
                        flush=True,
                    )
            else:
                init_key, key = jax.random.split(key)
                collection_state = self.init_collection_state(init_key)
        if replay_buffer:
            collection_state = self._fit_buffer(collection_state, "fork")
        self.parent = {
            "path": os.path.abspath(path),
            "epoch": meta["epoch"],
            "replay_buffer": replay_buffer,
            "monitor_run": meta["monitor_run"],
            "parent": meta["parent"],
        }
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            average=average,
            key=key,
            averaged=meta["averaged"],
        )

    # initial states

    def make_template_experience(self) -> BaseExperience:
        """An experience of the replay buffer's shapes."""
        env_state, metadata = self.env_init_fn(jax.random.PRNGKey(0))
        return BaseExperience(
            observation_nn=self.state_to_nn_input_fn(env_state),
            policy_mask=metadata.action_mask,
            policy_weights=jnp.zeros_like(metadata.action_mask, dtype=jnp.float32),
            reward=jnp.zeros_like(metadata.rewards),
            cur_player_id=metadata.cur_player_id,
            search_value=jnp.zeros((), dtype=jnp.float32),
        )

    def init_eval_state(self) -> Any:
        """The self-play evaluator's states, one per environment."""
        return self.evaluator.init_batched(
            self.batch_size, template_embedding=self.template_env_state
        )

    def init_episode_metrics(self, rewards: jax.Array) -> dict[str, jax.Array]:
        """Zeroed sums of each metric `episode_metrics_fn` returns, one per environment;
        `rewards` is an environment's (one per player), for their shape and dtype."""
        if self.episode_metrics_fn is None:
            return {}
        shapes = jax.eval_shape(self.episode_metrics_fn, rewards)
        return {
            name: jnp.zeros((self.batch_size,), dtype=jnp.float32) for name in shapes
        }

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
        self, key: jax.Array, capacity: int | None = None
    ) -> CollectionState:
        """A new collection state: new games, an empty replay buffer (of `capacity` entries per
        environment, by default `memory_buffer`'s), and nothing waiting for solves."""
        memory_buffer = (
            self.memory_buffer if capacity is None else EpisodeReplayBuffer(capacity)
        )
        buffer_state = memory_buffer.init(
            self.batch_size, self.make_template_experience()
        )
        env_init_key, _ = jax.random.split(key)
        # compiled, so that no two of the state's arrays are one array, as an environment's eager
        # init can return (e.g. its current player, in both the state and the metadata):
        # `collect_steps` can't be donated the same array twice
        env_state, metadata = jax.jit(jax.vmap(self.env_init_fn))(
            jax.random.split(env_init_key, self.batch_size)
        )
        return CollectionState(
            eval_state=self.init_eval_state(),
            env_state=env_state,
            buffer_state=buffer_state,
            metadata=metadata,
            episodes=jnp.zeros((self.batch_size,), dtype=jnp.int32),
            moves=jnp.zeros((self.batch_size,), dtype=jnp.int32),
            episode_metrics=self.init_episode_metrics(metadata.rewards[0]),
            # each environment's first cycle starts a move, and sets its key
            move_key=jnp.zeros_like(jax.random.split(key, self.batch_size)),
            search_cycles=jnp.zeros((self.batch_size,), dtype=jnp.int32),
            pending=self.init_pending(env_state, memory_buffer.capacity),
        )

    def init_pending(self, env_state: Any, capacity: int) -> PendingSolves | None:
        """`CollectionState`'s pending solves, with nothing waiting, for environments whose games
        are `env_state` and a replay buffer of `capacity`; None without `endgame_solve`."""
        if self.endgame_solve is None:
            return None
        endgame_solve = self.endgame_solve
        return jax.vmap(lambda e: endgame_solve.init(e, capacity))(env_state)

    # the loop

    def train_loop(
        self,
        seed: int,
        num_epochs: int,
        initial_state: TrainLoopOutput | None = None,
        fork_from: str | None = None,
        fork_replay_buffer: bool = True,
    ) -> TrainLoopOutput:
        """Runs the training loop until `num_epochs` epochs are done, counting those of a run
        continued from `initial_state`. Each epoch (see `Trainer`):

        1. the weight average moves toward the network's weights, and gets its own network
           state (see `average_norm_state`)
        2. self-play plays with it (see `collect_steps`)
        3. with `endgame_solve`, the solves of the positions waiting at the end of the last
           epoch's self-play, which ran on a host thread meanwhile, are waited for and
           labelled, and this epoch's start. While the buffer can't sample a batch (a new
           run's first epoch, whose games nearly all stopped for solves), those are waited
           for too
        4. self-play's metrics are logged (see `collection_metrics`, `solve_summary`)
        5. the network trains `train_steps_per_epoch` steps, if the buffer can sample a batch,
           and their metrics are logged
        6. with `holdout_envs`, the network's and the average's memorization metrics are
           logged (see `holdout_metrics`)
        7. every `validate_every` epochs, the average's validation metrics are logged
        8. the average's weights are saved (see `save_weights_every`), then the whole state
           (see `save_state_every`, `save_state_at`)

        A network whose optimizer state comes from a saved state trains with this trainer's
        injected hyperparameters (see `with_fixed_hyperparams`).

        Args:
            seed: rng seed of a new run (unused with `initial_state`, whose rng goes on)
            num_epochs: epochs to run to, counting those done before
            initial_state: (optional) a state to continue, from `resume` or `branch`. Its
                collection state is donated to self-play, so it can't be used afterwards
            fork_from: (optional) a saved state, or a directory to take the newest from, to
                fork a new run from (see `fork`). Not with `initial_state`
            fork_replay_buffer: with `fork_from`, start with its replay buffer (default), or an
                empty one

        Returns:
            TrainLoopOutput: the state, to continue the run from
        """
        if initial_state is not None and fork_from is not None:
            raise ValueError("pass initial_state or fork_from, not both")
        key = jax.random.PRNGKey(seed)
        if initial_state is not None:
            state = initial_state
        elif fork_from is not None:
            fork_from = latest_state_path(fork_from)
            self.set_activity(f"loading {fork_from}", echo=True)
            state = self.fork(fork_from, key, fork_replay_buffer)
        else:
            self.parent = None
            init_key, key = jax.random.split(key)
            train_state = self.init_train_state()
            state = TrainLoopOutput(
                collection_state=self.init_collection_state(init_key),
                train_state=train_state,
                average=jax.tree.map(jnp.copy, train_state.params),
                key=key,
            )
        if self.monitor is not None:
            self.monitor.start(
                config={
                    **self.get_config(),
                    "run": {"seed": seed, "num_epochs": num_epochs},
                    **({"parent": self.parent} if self.parent is not None else {}),
                    **self.extra_config,
                },
                # the last epoch
                total=num_epochs - 1,
            )
        try:
            output = self._train_loop(state, num_epochs)
        except BaseException as e:
            if self.monitor is not None:
                self.monitor.finish(
                    "stopped" if isinstance(e, KeyboardInterrupt) else "crashed"
                )
            raise
        if self.monitor is not None:
            self.monitor.finish()
        return output

    def _train_loop(self, state: TrainLoopOutput, num_epochs: int) -> TrainLoopOutput:
        collection_state = state.collection_state
        train_state = self.with_fixed_hyperparams(state.train_state)
        average, averaged = state.average, state.averaged
        key, cur_epoch = state.key, state.cur_epoch
        if self.holdout_envs:
            collection_state = replace(
                collection_state,
                buffer_state=self.without_holdout_positions(
                    collection_state.buffer_state
                ),
            )
        # exact endgame labels' solver, on a thread of its own, and the solve in progress: of
        # the positions waiting at the end of the last self-play, for after the next
        solver = ThreadPoolExecutor(1) if self.endgame_solve is not None else None
        solving = None
        if solver is not None:
            assert collection_state.pending is not None
            solving = solver.submit(
                self.solve_waiting, jax.tree.map(np.asarray, collection_state.pending)
            )
        try:
            while cur_epoch < num_epochs:
                # 1. the weight average, with its own network state
                share = average_share(
                    averaged, self.average_epochs, self.average_warmup
                )
                average = _move_average(average, train_state.params, jnp.float32(share))
                averaged += 1
                average_state = self.average_norm_state(
                    jax.random.fold_in(key, AVERAGE_NORM_SALT),
                    collection_state.buffer_state,
                    average,
                )
                params = (
                    average,
                    train_state.nn_state if average_state is None else average_state,
                )
                # 2. self-play, with the average
                self.set_activity(f"epoch {cur_epoch}: self-play")
                before = CollectionStats.of(collection_state)
                collect_key, key = jax.random.split(key)
                collection_state = self.collect_steps(
                    jax.random.split(collect_key, self.batch_size),
                    collection_state,
                    params,
                    self.collection_steps_per_epoch,
                )
                # 3. the last epoch's exact solves
                solve_metrics: dict[str, jax.Array] = {}
                if solver is not None and solving is not None:
                    self.set_activity(f"epoch {cur_epoch}: waiting for solves")
                    solved = [solving.result()]
                    collection_state = self.label_solved_games(
                        collection_state, solved[0]
                    )
                    assert collection_state.pending is not None
                    solving = solver.submit(
                        self.solve_waiting,
                        jax.tree.map(np.asarray, collection_state.pending),
                    )
                    if not self.can_train(collection_state.buffer_state):
                        self.set_activity(f"epoch {cur_epoch}: solving")
                        solved.append(solving.result())
                        collection_state = self.label_solved_games(
                            collection_state, solved[-1]
                        )
                        assert collection_state.pending is not None
                        solving = solver.submit(
                            self.solve_waiting,
                            jax.tree.map(np.asarray, collection_state.pending),
                        )
                    solve_metrics = solve_summary(solved)
                # 4. self-play's metrics
                self.log_metrics(
                    {
                        **self.collection_metrics(before, collection_state),
                        **solve_metrics,
                    },
                    cur_epoch,
                )
                # 5. training
                if self.train_steps_per_epoch and self.can_train(
                    collection_state.buffer_state
                ):
                    self.set_activity(f"epoch {cur_epoch}: training")
                    train_key, key = jax.random.split(key)
                    train_state, train_metrics = self.train_epoch(
                        train_key,
                        collection_state.buffer_state,
                        train_state,
                        self.train_steps_per_epoch,
                    )
                    self.log_metrics(train_metrics, cur_epoch)
                elif self.train_steps_per_epoch:
                    print(
                        f"epoch {cur_epoch}: the replay buffer can't sample a batch yet, "
                        "no training",
                        flush=True,
                    )
                # 6. memorization
                if self.holdout_envs:
                    holdout_key, key = jax.random.split(key)
                    holdout = self.holdout_metrics(
                        holdout_key,
                        collection_state.buffer_state,
                        {
                            "raw": (train_state.params, train_state.nn_state),
                            "avg": params,
                        },
                    )
                    if holdout:
                        self.log_metrics(holdout, cur_epoch)
                # 7. validation, of the average
                if self.validators and cur_epoch % self.validate_every == 0:
                    self.set_activity(f"epoch {cur_epoch}: validation")
                    prefix = average_name(self.average_epochs)
                    validation = {
                        f"{prefix}/{name}": value
                        for validator in self.validators
                        for name, value in validator(params).items()
                    }
                    self.log_metrics(validation, cur_epoch)
                # 8. saves
                if (
                    self.save_weights_every is not None
                    and cur_epoch % self.save_weights_every == 0
                ):
                    self.set_activity(f"epoch {cur_epoch}: saving weights")
                    self.save_weights(params, cur_epoch)
                cur_epoch += 1
                if self.should_save_state(cur_epoch):
                    self.set_activity(
                        f"epoch {cur_epoch - 1}: saving the training state"
                    )
                    self.save_state(
                        TrainLoopOutput(
                            collection_state=collection_state,
                            train_state=train_state,
                            average=average,
                            key=key,
                            averaged=averaged,
                            cur_epoch=cur_epoch,
                        )
                    )
        finally:
            if solver is not None:
                solver.shutdown(wait=False, cancel_futures=True)
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            average=average,
            key=key,
            averaged=averaged,
            cur_epoch=cur_epoch,
        )
