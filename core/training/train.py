import json
import os
import re
import zipfile
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from core.common import search_and_step, step_env_and_evaluator
from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import (
    BaseExperience,
    EpisodeReplayBuffer,
    ReplayBufferState,
)
from core.monitor import Monitor
from core.testing.tester import BaseTester, TestState
from core.training.exploration import SelfPlayExploration
from core.training.schedule import EvaluatorSchedule, Schedule, describe
from core.training.tree_positions import (
    TreePositions,
    select_nodes,
    tree_experiences,
)
from core.types import (
    DataTransformFn,
    EnvInitFn,
    EnvStepFn,
    ExtractModelParamsFn,
    LossFn,
    StateToNNInputFn,
    StepMetadata,
    TrainState,
)


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
        draws: how many of those ended with every player's reward 0
        tree_positions: number of search tree positions stored so far (not counting transformed copies,
            see `Trainer`'s `tree_positions`)
        tree_visits: their visit counts, summed
        moves: number of self-play moves made so far
        tree_buffer_state: state of the replay buffer of search tree positions, None without them
        tree_written_at: for each entry of the tree position replay buffer, the number of moves made
            before it was written (see `Trainer.tree_sample_mask`), None without tree positions
    """

    eval_state: Any
    env_state: Any
    buffer_state: ReplayBufferState
    metadata: StepMetadata
    episodes: jax.Array
    draws: jax.Array
    tree_positions: jax.Array
    tree_visits: jax.Array
    moves: jax.Array
    tree_buffer_state: ReplayBufferState | None = None
    tree_written_at: jax.Array | None = None


@dataclass(frozen=True)
class SelfplayCounters:
    """The counts of a collection state that `Trainer.selfplay_metrics` compares across an epoch's
    self-play (see `CollectionState`).

    Attributes:
        episodes: number of episodes that have terminated so far, per environment
        draws: how many of those were draws
        tree_positions: number of search tree positions stored so far
        tree_visits: their visit counts, summed
    """

    episodes: jax.Array
    draws: jax.Array
    tree_positions: jax.Array
    tree_visits: jax.Array

    @classmethod
    def of(cls, state: CollectionState) -> "SelfplayCounters":
        """Copies the counts out of `state`, so that they outlive it when it is donated to
        `Trainer.collect_steps`."""
        return cls(
            episodes=jnp.copy(state.episodes),
            draws=jnp.copy(state.draws),
            tree_positions=jnp.copy(state.tree_positions),
            tree_visits=jnp.copy(state.tree_visits),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainLoopOutput:
    """Stores the state of the training loop.

    `collection_state` is included to access replay memory.

    Attributes:
        collection_state: state of self-play episode collection.
        train_state: TrainState, holds model params and state, optimizer state
        test_states: states of testers
        cur_epoch: current epoch num: the number of epochs done
        key: (optional) the training loop's rng, for the next epoch. With it, a run continued
            from this state carries on exactly as if it hadn't stopped (see `Trainer.train_loop`).
    """

    collection_state: CollectionState
    train_state: TrainState
    test_states: list[TestState]
    cur_epoch: int
    key: jax.Array | None = None


def extract_params(state: TrainState) -> Any:
    """Extracts model parameters from TrainState.

    Args:
        state: TrainState containing model parameters

    Returns:
        Tuple[Any, Optional[eqx.nn.State]]: model parameters and state (nn_params, nn_state), as used by
            core.evaluators.evaluation_fns.make_nn_eval_fn
    """
    return state.params, state.nn_state


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


def read_state_meta(path: str) -> dict:
    """The metadata of a saved state (see `Trainer.save_state`): the epochs done, the run's
    configuration, its monitor run and parent, and what each array is."""
    with np.load(path) as saved:
        return json.loads(str(saved["meta"]))


# saved states' format, recorded in their metadata
STATE_FORMAT = 1


def _describe(tree: Any) -> list[tuple[str, tuple[int, ...], str]]:
    """Each leaf's path, shape and dtype, in the order `jax.tree.leaves` gives them (leaves may
    be `jax.ShapeDtypeStruct`s)."""
    described = []
    for path, x in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not hasattr(x, "dtype"):
            x = np.asarray(x)
        described.append((jax.tree_util.keystr(path), tuple(x.shape), str(x.dtype)))
    return described


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


def restart_schedules(opt_state: Any) -> Any:
    """`opt_state` with its learning rate schedules' step counts back at 0 (optax's
    `ScaleByScheduleState` and `InjectHyperparamsState`), so that they start over.

    Everything else (e.g. Adam's moments, and the step count of its bias correction) is
    kept."""

    def is_schedule(x: Any) -> bool:
        return isinstance(x, optax.ScaleByScheduleState | optax.InjectHyperparamsState)

    return jax.tree.map(
        lambda x: x._replace(count=jnp.zeros_like(x.count)) if is_schedule(x) else x,
        opt_state,
        is_leaf=is_schedule,
    )


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
        replay_window: int | Schedule[int] | None = None,
        evaluator_test: Evaluator | None = None,
        test_env_init_fn: EnvInitFn | None = None,
        selfplay_exploration: SelfPlayExploration | None = None,
        data_transform_fns: Sequence[DataTransformFn] = (),
        tree_positions: TreePositions | None = None,
        extract_model_params_fn: ExtractModelParamsFn = extract_params,
        monitor: Monitor | None = None,
        ckpt_dir: str = "/tmp/turbozero_checkpoints",
        max_checkpoints: int = 2,
        keep_every: int | None = None,
        save_state_at: Collection[int] = (),
        save_state_every: int | None = None,
        extra_config: dict | None = None,
    ):
        """Initializes a Trainer.

        Args:
            batch_size: batch size for self-play games
            train_batch_size: minibatch size for training steps
            warmup_steps: # of steps (per batch) to collect via self-play prior to entering the training loop.
                - This is used to populate the replay memory with some initial samples
            collection_steps_per_epoch: # of steps (per batch) to collect via self-play in each epoch
            train_steps_per_epoch: # of training steps to take in each epoch
            nn: neural network (an equinox module, see core.networks.utils.apply_nn), training starts from its parameters
            loss_fn: loss function for training (see core.training.loss_fns)
            optimizer: optax optimizer
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
            replay_window: (optional) how many of each environment's newest replay buffer entries training samples
                from, at most `memory_buffer.capacity`, or a `Schedule` of it by epoch, e.g. to sample only recent
                data early on, while the network changes quickly, and more later. Changing it doesn't recompile.
                If not provided, the whole buffer.
            evaluator_test: (optional) evaluator to use during testing. If not provided, `evaluator` is used
                (required with a schedule of several evaluators).
            test_env_init_fn: (optional) environment initialization function for test episodes, e.g. to test from
                the standard start while self-play starts from varied positions. If not provided, `env_init_fn` is used.
            selfplay_exploration: (optional) chooses the move self-play plays from the evaluator's output,
                e.g. to play random moves some of the time (see core.training.exploration). The evaluator's
                policy weights stay the training target. If not provided, self-play plays the evaluator's move.
            data_transform_fns: (optional) list of data transform functions to apply to self-play experiences (e.g. rotation, reflection, etc.)
            tree_positions: (optional) also train on positions from self-play's search trees, as OLIVAW did
                (see core.training.tree_positions). They're kept in a replay buffer of their own, and make up
                a share of each training batch, sampled from those stored in the replay window's span of
                self-play (see `tree_sample_mask`). Needs MCTS self-play evaluators.
            extract_model_params_fn: (optional) function to extract model parameters from TrainState
            monitor: (optional) `core.monitor.Monitor` to log metrics and test episodes to (see a tester's `episode_fn`)
                - start the server with `turbozero-monitor`; a run is created on the first `train_loop`,
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
        self.extract_model_params_fn = extract_model_params_fn
        # selfplay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.collection_steps_per_epoch = collection_steps_per_epoch
        self.memory_buffer = memory_buffer
        # a constant window is a schedule with one stage
        if replay_window is None:
            replay_window = memory_buffer.capacity
        self.replay_window = (
            replay_window
            if isinstance(replay_window, Schedule)
            else Schedule([(0, replay_window)])
        )
        if not all(
            0 < window <= memory_buffer.capacity
            for _, window in self.replay_window.stages
        ):
            raise ValueError(
                f"replay windows must be between 1 and the replay buffer's capacity, "
                f"{memory_buffer.capacity}, got {self.replay_window.get_config()}"
            )
        # a single evaluator is a schedule with one stage
        self.selfplay_schedule = (
            evaluator
            if isinstance(evaluator, EvaluatorSchedule)
            else EvaluatorSchedule([(0, evaluator)])
        )
        self.transform_fns = data_transform_fns
        self.selfplay_exploration = selfplay_exploration
        if tree_positions is not None and not all(
            isinstance(e, MCTS) for _, e in self.selfplay_schedule.stages
        ):
            raise ValueError("tree_positions needs MCTS self-play evaluators")
        self.tree_positions = tree_positions
        self.tree_buffer = (
            EpisodeReplayBuffer(capacity=tree_positions.capacity)
            if tree_positions is not None
            else None
        )
        self.count_distinct_observations = jax.jit(
            self.memory_buffer.count_distinct_observations
        )
        # training
        self.train_steps_per_epoch = train_steps_per_epoch
        self.train_batch_size = train_batch_size
        # testing
        self.testers = testers
        if evaluator_test is None:
            if len(self.selfplay_schedule.stages) > 1:
                raise ValueError(
                    "pass an evaluator_test with a schedule of self-play evaluators"
                )
            evaluator_test = self.selfplay_schedule.at(0)
        self.evaluator_test = evaluator_test
        self.step_test = partial(
            step_env_and_evaluator,
            evaluator=self.evaluator_test,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.test_env_init_fn,
            max_steps=self.max_episode_steps,
        )
        # checkpoints
        self.ckpt_dir = ckpt_dir
        self.max_checkpoints = max_checkpoints
        self.keep_every = keep_every
        self.save_state_at = frozenset(save_state_at)
        self.save_state_every = save_state_every
        os.makedirs(ckpt_dir, exist_ok=True)
        # monitor
        self.monitor = monitor
        self.extra_config = extra_config if extra_config is not None else {}
        # the saved state this run was forked from (see `train_loop`), recorded in the monitor's
        # config and in states the run saves
        self.parent: dict | None = None

    def init_train_state(self) -> TrainState:
        """Initializes the training state (params, optimizer, etc.) from `nn` and `nn_state`.

        Returns:
            TrainState: initialized training state
        """
        params = eqx.filter(self.nn, eqx.is_inexact_array)
        return TrainState(
            params=params,
            nn_state=self.nn_state,
            opt_state=self.optimizer.init(params),
            step=jnp.array(0, dtype=jnp.int32),
        )

    def get_config(self):
        """Returns a dictionary of the configuration of the trainer. Used for logging to the monitor.

        With a schedule of several self-play evaluators, `evaluator_train` is the first, and
        `selfplay_schedule` lists them all.
        """
        evaluator_train = self.selfplay_schedule.at(0)
        schedule = self.selfplay_schedule
        return {
            "batch_size": self.batch_size,
            "train_batch_size": self.train_batch_size,
            "warmup_steps": self.warmup_steps,
            "collection_steps_per_epoch": self.collection_steps_per_epoch,
            "train_steps_per_epoch": self.train_steps_per_epoch,
            "evaluator_train": evaluator_train.__class__.__name__,
            "evaluator_train_config": evaluator_train.get_config(),
            **(
                {"selfplay_schedule": schedule.get_config()}
                if len(schedule.stages) > 1
                else {}
            ),
            "evaluator_test": self.evaluator_test.__class__.__name__,
            "evaluator_test_config": self.evaluator_test.get_config(),
            "selfplay_exploration_config": self.selfplay_exploration.get_config()
            if self.selfplay_exploration is not None
            else None,
            "memory_buffer": self.memory_buffer.__class__.__name__,
            "memory_buffer_config": self.memory_buffer.get_config(),
            "replay_window": self.replay_window.get_config(),
            "tree_positions": self.tree_positions.get_config()
            if self.tree_positions is not None
            else None,
        }

    def collect(
        self,
        key: jax.Array,
        state: CollectionState,
        params: Any,
        evaluator: Evaluator | None = None,
    ) -> CollectionState:
        """Collects self-play data for a single step.

        - Stores experience in replay buffer.
        - Resets environment/evaluator if episode is terminated.

        Args:
            key: rng
            state: current collection state (environment, evaluator, replay buffer)
            params: model parameters
            evaluator: (optional) the self-play evaluator, whose states `state` holds. If not
                provided, the first in the schedule.

        Returns:
            CollectionState: updated collection state
        """
        if evaluator is None:
            evaluator = self.selfplay_schedule.at(0)
        tree_key = None
        if self.tree_positions is not None:
            key, tree_key = jax.random.split(key)
        # step environment and evaluator
        (
            eval_output,
            new_env_state,
            new_metadata,
            terminated,
            truncated,
            rewards,
            searched_eval_state,
        ) = search_and_step(
            key=key,
            env_state=state.env_state,
            env_state_metadata=state.metadata,
            eval_state=state.eval_state,
            params=params,
            evaluator=evaluator,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn,
            max_steps=self.max_episode_steps,
            choose_action=self.selfplay_exploration.choose_action
            if self.selfplay_exploration is not None
            else None,
        )
        search_value = evaluator.get_value(searched_eval_state)

        # store experience in replay buffer
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
        )
        # apply transforms
        for transform_fn in self.transform_fns:
            t_policy_mask, t_policy_weights, t_env_state = transform_fn(
                state.metadata.action_mask, eval_output.policy_weights, state.env_state
            )
            buffer_state = self.memory_buffer.add_experience(
                state=buffer_state,
                experience=BaseExperience(
                    observation_nn=self.state_to_nn_input_fn(t_env_state),
                    policy_mask=t_policy_mask,
                    policy_weights=t_policy_weights,
                    reward=jnp.empty_like(state.metadata.rewards),
                    cur_player_id=state.metadata.cur_player_id,
                    search_value=search_value,
                ),
            )
        # assign rewards to buffer if episode is terminated
        buffer_state = jax.lax.cond(
            terminated,
            lambda s: self.memory_buffer.assign_rewards(s, rewards),
            lambda s: s,
            buffer_state,
        )
        # discard the episode's experiences if it hit the step limit without terminating
        # (`truncated` is never set together with `terminated`)
        buffer_state = jax.lax.cond(
            truncated, self.memory_buffer.truncate, lambda s: s, buffer_state
        )
        if tree_key is not None:
            # the tree the next search reuses: the played move's subtree, unless the episode ended
            reused = jnp.where(
                terminated | truncated,
                searched_eval_state.NULL_INDEX,
                searched_eval_state.edge_map[
                    searched_eval_state.ROOT_INDEX, eval_output.action
                ],
            )
            state = self.store_tree_positions(
                tree_key, state, searched_eval_state, reused, evaluator
            )
        # return new collection state
        return replace(
            state,
            eval_state=eval_output.eval_state,
            env_state=new_env_state,
            buffer_state=buffer_state,
            metadata=new_metadata,
            episodes=state.episodes + terminated,
            draws=state.draws + (terminated & (rewards == 0).all()),
            moves=state.moves + 1,
        )

    def store_tree_positions(
        self,
        key: jax.Array,
        state: CollectionState,
        tree: Any,
        reused: jax.Array,
        evaluator: Evaluator,
    ) -> CollectionState:
        """Stores positions from a self-play search tree in the tree position replay buffer
        (see `tree_positions`), with their transformed copies.

        Args:
            key: rng
            state: current collection state
            tree: the search tree, after the search
            reused: the root child whose subtree the next search reuses, NULL_INDEX for none
            evaluator: the self-play evaluator that searched the tree

        Returns:
            CollectionState: updated collection state
        """
        assert self.tree_positions is not None and self.tree_buffer is not None
        assert state.tree_buffer_state is not None and state.tree_written_at is not None
        assert isinstance(evaluator, MCTS)
        indices, valid = select_nodes(
            key,
            tree,
            self.tree_positions.per_move,
            self.tree_positions.min_visits,
            self.tree_positions.most_visited,
            reused
            if self.tree_positions.discarded_only and evaluator.persist_tree
            else None,
        )
        experiences, valid_experiences = tree_experiences(
            tree,
            indices,
            valid,
            self.env_step_fn,
            self.state_to_nn_input_fn,
            self.transform_fns,
        )
        written = self.tree_buffer.write_indices(
            state.tree_buffer_state, valid_experiences
        )
        return replace(
            state,
            tree_buffer_state=self.tree_buffer.add_experiences(
                state.tree_buffer_state, experiences, valid_experiences
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
        """Collects self-play data for `num_steps` steps in every environment.

        Compiled once for each evaluator.

        Args:
            key: rng, one key per environment
            state: current collection state, donated: its arrays are deleted, and can't be used
                after the call
            params: model parameters
            num_steps: number of self-play steps to collect
            evaluator: (optional) the self-play evaluator, whose states `state` holds. If not
                provided, the first in the schedule.

        Returns:
            CollectionState: updated collection state
        """
        if num_steps > 0:

            def collect_env(key: jax.Array, state: CollectionState) -> CollectionState:
                keys = jax.random.split(key, num_steps)
                return jax.lax.fori_loop(
                    0,
                    num_steps,
                    lambda i, s: self.collect(keys[i], s, params, evaluator),
                    state,
                )

            return jax.vmap(collect_env)(key, state)
        return state

    def train_step(
        self, ts: TrainState, batch: BaseExperience
    ) -> tuple[TrainState, dict]:
        """Make a single training step.

        Args:
            ts: TrainState
            batch: minibatch of experiences

        Returns:
            Tuple[TrainState, dict]: updated TrainState and metrics
        """
        # calculate loss, get gradients
        nn = eqx.combine(ts.params, self.nn_static)
        grad_fn = eqx.filter_value_and_grad(self.loss_fn, has_aux=True)
        (loss, (metrics, nn_state)), grads = grad_fn(nn, ts.nn_state, batch)
        # apply gradients
        updates, opt_state = self.optimizer.update(grads, ts.opt_state, ts.params)
        ts = replace(
            ts,
            params=optax.apply_updates(ts.params, updates),
            nn_state=nn_state,
            opt_state=opt_state,
            step=ts.step + 1,
        )
        # return updated train state and metrics
        metrics = {**metrics, "loss": loss}
        return ts, metrics

    @partial(jax.jit, static_argnums=(0, 4))
    def train_epoch(
        self,
        key: jax.Array,
        buffer_state: ReplayBufferState,
        train_state: TrainState,
        num_steps: int,
        window: jax.Array,
        tree_buffer_state: ReplayBufferState | None = None,
        tree_mask: jax.Array | None = None,
        num_tree: jax.Array | None = None,
    ) -> tuple[TrainState, dict]:
        """Performs `num_steps` training steps, compiled into a single `jax.lax.scan`.

        Each step samples a minibatch from the replay buffer and updates the parameters.

        Does not check that the replay buffer holds enough entries to sample, see `train_steps`.

        Args:
            key: rng
            buffer_state: replay buffer state
            train_state: current training state
            num_steps: number of training steps to perform
            window: entries per environment, newest first, to sample from (see
                `EpisodeReplayBuffer.sample_mask`), a traced value so that changing it doesn't recompile
            tree_buffer_state: (optional) state of the search tree positions' replay buffer
                (see `tree_positions`), to sample the first `num_tree` samples of each minibatch from
            tree_mask: the tree positions that can be sampled (see `tree_sample_mask`)
            num_tree: number of tree positions in each minibatch, a traced value so that changing it
                doesn't recompile. With none to sample, it must be 0.

        Returns:
            Tuple[TrainState, dict]: updated training state and metrics (mean across steps)
        """
        # the buffer doesn't change while training, so find which entries can be sampled once per epoch
        mask = self.memory_buffer.sample_mask(buffer_state, window)
        add_tree_positions = (
            self.tree_position_sampler(tree_buffer_state, tree_mask, num_tree)
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
                self.memory_buffer.sample_indices(
                    step_key, mask, self.train_batch_size
                ),
                mask.shape,
            )
            batch = jax.tree.map(lambda x: x[indices], buffer_state.buffer)
            if add_tree_positions is not None:
                tree_key, key = jax.random.split(key)
                batch = add_tree_positions(tree_key, batch)
            # make training step
            ts, metrics = self.train_step(ts, batch)
            return (key, ts), metrics

        (_, train_state), metrics = jax.lax.scan(
            step, (key, train_state), length=num_steps
        )
        return train_state, jax.tree.map(jnp.mean, metrics)

    def tree_sample_mask(self, state: CollectionState, epoch: int) -> jax.Array:
        """Marks the tree positions (see `tree_positions`) training can sample in `epoch`: those stored
        within the replay window's span of self-play.

        The replay window (see `replay_window`) holds each environment's newest W played positions,
        with data transforms' copies: its last W / (1 + number of data transforms) moves. Tree
        positions are sampled from the ones stored in as many moves, however many each move stored.

        Args:
            state: current collection state
            epoch: the current epoch, which sets the replay window

        Returns:
            jax.Array: boolean mask, the shape of the tree position buffer's `populated`
        """
        assert self.tree_buffer is not None
        assert state.tree_buffer_state is not None and state.tree_written_at is not None
        moves = -(-self.replay_window.at(epoch) // (1 + len(self.transform_fns)))
        # positions the last move stored are 1 move old
        age = state.moves[..., None] - state.tree_written_at
        return self.tree_buffer.sample_mask(state.tree_buffer_state) & (age <= moves)

    def tree_position_sampler(
        self,
        tree_buffer_state: ReplayBufferState,
        mask: jax.Array,
        num_tree: jax.Array,
    ) -> Callable[[jax.Array, BaseExperience], BaseExperience]:
        """Makes a function that puts tree positions (see `tree_positions`) into a minibatch: it
        replaces the first `num_tree` samples of a minibatch with ones sampled from the tree position
        replay buffer, uniformly with replacement (so however few there are).

        Args:
            tree_buffer_state: state of the tree position replay buffer, which doesn't change while
                the function is used
            mask: the tree positions to sample from (see `tree_sample_mask`)
            num_tree: number of tree positions per minibatch; with none in `mask`, it must be 0

        Returns:
            Callable[[jax.Array, BaseExperience], BaseExperience]: (rng, minibatch) -> minibatch
        """
        assert self.tree_buffer is not None
        tree_buffer = self.tree_buffer
        # with nothing to sample, sample anything: `num_tree` is 0, so none of it is used
        mask = mask | ~mask.any()
        is_tree = jnp.arange(self.train_batch_size) < num_tree

        def add_tree_positions(key: jax.Array, batch: BaseExperience) -> BaseExperience:
            indices = jnp.unravel_index(
                tree_buffer.sample_indices(
                    key, mask, self.train_batch_size, replace=True
                ),
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

    def train_steps(
        self,
        key: jax.Array,
        collection_state: CollectionState,
        train_state: TrainState,
        num_steps: int,
        epoch: int = 0,
    ) -> tuple[CollectionState, TrainState, dict]:
        """Performs `num_steps` training steps.

        Each step consists of sampling a minibatch from the replay buffer and updating the parameters.
        The minibatch is sampled uniformly without replacement from the finished episodes in the buffer's
        replay window (see `replay_window`). With `tree_positions`, a share of it is sampled (uniformly,
        with replacement) from the search tree positions stored in the same span of self-play instead
        (see `tree_sample_mask`).

        Args:
            key: rng
            collection_state: current collection state
            train_state: current training state
            num_steps: number of training steps to perform
            epoch: the current epoch, which sets the replay window, and the share of tree positions
                (see `TreePositions.ratio_at`)

        Returns:
            Tuple[CollectionState, TrainState, dict]:
                - updated collection state
                - updated training state
                - metrics

        Raises:
            ValueError: if no episode has finished yet, or the replay window holds fewer samples than a
                minibatch
        """
        if num_steps == 0:
            return collection_state, train_state, {}
        window = self.replay_window.at(epoch)
        # the buffer doesn't change while training, so checking once covers every step
        self.memory_buffer.check_can_sample(
            collection_state.buffer_state, window, self.train_batch_size
        )
        if self.tree_positions is None:
            train_state, metrics = self.train_epoch(
                key,
                collection_state.buffer_state,
                train_state,
                num_steps,
                jnp.array(window, dtype=jnp.int32),
            )
            return collection_state, train_state, metrics
        tree_mask = self.tree_sample_mask(collection_state, epoch)
        num_tree = (
            self.tree_positions.batch_count(epoch, self.train_batch_size)
            if tree_mask.any()
            else 0
        )
        train_state, metrics = self.train_epoch(
            key,
            collection_state.buffer_state,
            train_state,
            num_steps,
            jnp.array(window, dtype=jnp.int32),
            tree_buffer_state=collection_state.tree_buffer_state,
            tree_mask=tree_mask,
            num_tree=jnp.array(num_tree, dtype=jnp.int32),
        )
        metrics = {
            **metrics,
            "tree_batch_fraction": jnp.array(num_tree / self.train_batch_size),
            "tree_buffer_samples": tree_mask.sum(),
        }
        return collection_state, train_state, metrics

    def selfplay_metrics(
        self, before: SelfplayCounters, after: CollectionState, epoch: int = 0
    ) -> dict:
        """Measures how varied self-play is, to spot it collapsing into the same few games.

        Args:
            before: the collection state's counts before this epoch's self-play
            after: collection state after it
            epoch: the epoch, which sets the replay window

        Returns:
            dict: metrics
                - `selfplay_episodes`: episodes that terminated in between
                - `selfplay_draw_fraction`: fraction of them that ended with every reward 0
                  (draws, in two-player zero-sum games), omitted if none terminated
                - `replay_window`: entries per environment training samples from (see `replay_window`)
                - `buffer_samples`: experiences in the replay window training can sample
                - `buffer_distinct_positions`: distinct observations among them (data
                  transforms' outputs count as observations too)
                - `buffer_distinct_fraction`: that as a fraction of those experiences
                - with `tree_positions`: `tree_positions` stored in between (not counting transformed
                  copies), `tree_positions_per_move`, and their `tree_mean_visits`
        """
        episodes = (after.episodes - before.episodes).sum()
        draws = (after.draws - before.draws).sum()
        window = self.replay_window.at(epoch)
        distinct, total = self.count_distinct_observations(
            after.buffer_state, jnp.array(window, dtype=jnp.int32)
        )
        metrics = {
            "selfplay_episodes": episodes,
            "replay_window": jnp.array(window),
            "buffer_samples": total,
            "buffer_distinct_positions": distinct,
            "buffer_distinct_fraction": distinct / jnp.maximum(total, 1),
        }
        if episodes > 0:
            metrics["selfplay_draw_fraction"] = draws / episodes
        if self.tree_positions is not None:
            positions = (after.tree_positions - before.tree_positions).sum()
            visits = (after.tree_visits - before.tree_visits).sum()
            moves = self.collection_steps_per_epoch * after.tree_positions.size
            metrics["tree_positions"] = positions
            metrics["tree_positions_per_move"] = positions / max(moves, 1)
            metrics["tree_mean_visits"] = visits / jnp.maximum(positions, 1)
        return metrics

    def log_metrics(self, metrics: dict, epoch: int):
        """Logs metrics to console and the monitor.

        Args:
            metrics: dictionary of metrics
            epoch: current epoch
        """
        # log to console
        metrics_str = {k: f"{v.item():.4f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {metrics_str}")
        # log to monitor
        if self.monitor is not None:
            self.monitor.log(epoch, metrics)

    def set_activity(self, text: str | None, echo: bool = False) -> None:
        """Tells the monitor what the training loop is doing now, so the dashboard can show
        what a slow step is busy with.

        Args:
            text: what the loop is doing, None once it's done
            echo: also print it to the console
        """
        if echo and text is not None:
            print(text, flush=True)
        if self.monitor is not None:
            self.monitor.activity(text)

    def save_checkpoint(self, train_state: TrainState, epoch: int) -> None:
        """Saves a checkpoint of the training state to `ckpt_dir`.

        Deletes the oldest checkpoints so that at most `max_checkpoints` remain, besides
        those kept by `keep_every`.

        Args:
            train_state: current training state
            epoch: current epoch
        """
        epochs = checkpoint_epochs(self.ckpt_dir)
        if epochs and epochs[-1] >= epoch:
            raise ValueError(
                f"{self.ckpt_dir} already has a checkpoint at or after epoch {epoch}, "
                "resume from it or use a different ckpt_dir"
            )
        # write to a temporary file first so an interrupted save doesn't leave a partial checkpoint
        path = checkpoint_path(self.ckpt_dir, epoch)
        with open(path + ".tmp", "wb") as f:
            eqx.tree_serialise_leaves(f, train_state)
        os.replace(path + ".tmp", path)
        # delete old checkpoints
        if self.keep_every is not None:
            epochs = [e for e in epochs if e % self.keep_every != 0]
            if epoch % self.keep_every == 0:
                return
        for old_epoch in epochs[: max(0, len(epochs) + 1 - self.max_checkpoints)]:
            os.remove(checkpoint_path(self.ckpt_dir, old_epoch))

    def load_train_state_from_checkpoint(
        self, path_to_checkpoint: str, epoch: int
    ) -> TrainState:
        """Loads a training state from a checkpoint.

        Args:
            path_to_checkpoint: path to checkpoint
            epoch: epoch to load

        Returns:
            TrainState: loaded training state
        """
        with open(checkpoint_path(path_to_checkpoint, epoch), "rb") as f:
            return eqx.tree_deserialise_leaves(f, self.init_train_state())

    def should_save_state(self, epochs_done: int) -> bool:
        """Whether to save the whole training state after `epochs_done` epochs (see
        `save_state_at` and `save_state_every`)."""
        return epochs_done in self.save_state_at or (
            self.save_state_every is not None
            and epochs_done % self.save_state_every == 0
        )

    def save_state(self, state: TrainLoopOutput) -> str:
        """Saves the whole training state to `ckpt_dir`, as `state-<epochs done>.npz`: the train
        state, the collection state (with the replay buffers), the testers' states, the number of
        epochs done and the training loop's rng.

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
            "train_state": state.train_state,
            "collection_state": state.collection_state,
            "test_states": state.test_states,
        }
        meta = {
            "format": STATE_FORMAT,
            "epoch": state.cur_epoch,
            "num_envs": int(state.collection_state.episodes.shape[0]),
            "buffer_capacity": self.memory_buffer.capacity,
            "tree_buffer_capacity": self.tree_buffer.capacity
            if self.tree_buffer is not None
            else None,
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

        The trainer must be configured as the run's was: the number of environments, replay
        buffer capacities, network, optimizer, self-play evaluator (the one scheduled for the last
        epoch done) and testers must match, so that every array does.

        Args:
            path: the saved state

        Returns:
            TrainLoopOutput: the run's state, in new device arrays, with its rng

        Raises:
            ValueError: if the saved state doesn't fit this trainer
        """
        with np.load(path) as saved:
            meta = self._check_meta(path, json.loads(str(saved["meta"])))
            self._check_sizes(path, meta, resizable=False)
            epoch = meta["epoch"]
            # the evaluator whose states the collection state holds
            evaluator = self.selfplay_schedule.at(max(epoch - 1, 0))
            train_template = jax.eval_shape(self.init_train_state)
            train_state = self._load_part(saved, meta, "train_state", train_template)
            collection_state = self._load_part(
                saved,
                meta,
                "collection_state",
                jax.eval_shape(
                    lambda: self.init_collection_state(
                        jax.random.PRNGKey(0), self.batch_size, evaluator
                    )
                ),
            )
            test_states = self._load_part(
                saved,
                meta,
                "test_states",
                jax.eval_shape(
                    lambda params: [t.init(params=params) for t in self.testers],
                    self.extract_model_params_fn(train_template),
                ),
            )
            key = _to_device(saved["key"])
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            test_states=test_states,
            cur_epoch=epoch,
            key=key,
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
        changed = sorted(
            k
            for k in current.keys() | meta["config"].keys()
            if current.get(k) != meta["config"].get(k)
        )
        if changed:
            print(
                f"warning: resuming {path} with a different configuration: {changed}",
                flush=True,
            )
        self.parent = meta["parent"]
        if self.monitor is not None and self.monitor.run_id is None:
            self.monitor.run_id = meta["monitor_run"]
        for epoch in checkpoint_epochs(self.ckpt_dir):
            if epoch >= state.cur_epoch:
                os.remove(checkpoint_path(self.ckpt_dir, epoch))
        print(f"resuming from {path}: {state.cur_epoch} epochs done", flush=True)
        return state

    def fork_state(
        self, path: str, key: jax.Array, replay_buffers: bool = True
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

        The number of environments, network and optimizer must match the saved run's.

        Args:
            path: the saved state (see `save_state`)
            key: rng, for the games
            replay_buffers: load the replay buffers (default), or start with empty ones

        Returns:
            Tuple[CollectionState, TrainState]: the collection state and train state

        Raises:
            ValueError: if the saved state doesn't fit this trainer
        """
        with np.load(path) as saved:
            meta = self._check_meta(path, json.loads(str(saved["meta"])))
            train_state = self._load_part(
                saved, meta, "train_state", jax.eval_shape(self.init_train_state)
            )
            train_state = replace(
                train_state,
                opt_state=restart_schedules(train_state.opt_state),
                step=jnp.zeros_like(train_state.step),
            )
            if not replay_buffers:
                return self.init_collection_state(key, self.batch_size), train_state
            self._check_sizes(path, meta, resizable=True)
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

            counters = {
                name: _to_device(get(f".{name}"))
                for name in (
                    "episodes",
                    "draws",
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
            key, self.batch_size, self.selfplay_schedule.at(0)
        )
        collection_state = CollectionState(
            eval_state=eval_state,
            env_state=env_state,
            buffer_state=buffer_state,
            metadata=metadata,
            tree_buffer_state=tree_buffer_state,
            tree_written_at=tree_written_at,
            **counters,
        )
        return collection_state, train_state

    def _check_meta(self, path: str, meta: dict) -> dict:
        if meta.get("format") != STATE_FORMAT:
            raise ValueError(
                f"{path} is a saved state of format {meta.get('format')}, this version reads "
                f"format {STATE_FORMAT}"
            )
        return meta

    def _check_sizes(self, path: str, meta: dict, resizable: bool) -> None:
        """Checks that a saved state's number of environments, and unless `resizable` its replay
        buffers' capacities, are this trainer's."""
        if meta["num_envs"] != self.batch_size:
            raise ValueError(
                f"{path} was saved with {meta['num_envs']} self-play environments, this run "
                f"has {self.batch_size}: they must be the same"
            )
        if resizable:
            return
        tree_capacity = (
            self.tree_buffer.capacity if self.tree_buffer is not None else None
        )
        for name, saved, ours in [
            ("replay buffer", meta["buffer_capacity"], self.memory_buffer.capacity),
            ("tree position buffer", meta["tree_buffer_capacity"], tree_capacity),
        ]:
            if saved != ours:
                raise ValueError(
                    f"{path} was saved with a {name} capacity of {saved}, this run's is "
                    f"{ours}: continuing a run needs the same (a fork can differ)"
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
                f"the saved {name} has {len(described)} arrays, this run's has "
                f"{len(expected)}: was it saved with different settings (e.g. the network, "
                "optimizer, self-play search, tree positions or testers)?"
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

    def init_collection_state(
        self, key: jax.Array, batch_size: int, evaluator: Evaluator | None = None
    ) -> CollectionState:
        """Initializes the collection state (see CollectionState).

        Args:
            key: rng
            batch_size: number of parallel environments
            evaluator: (optional) the self-play evaluator to initialize states for. If not
                provided, the first in the schedule.

        Returns:
            CollectionState: initialized collection state
        """
        if evaluator is None:
            evaluator = self.selfplay_schedule.at(0)
        # make template experience
        template_experience = self.make_template_experience()
        # init buffer state
        buffer_state = self.memory_buffer.init(batch_size, template_experience)
        tree_buffer_state = (
            self.tree_buffer.init(batch_size, template_experience)
            if self.tree_buffer is not None
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
            draws=jnp.zeros((batch_size,), dtype=jnp.int32),
            tree_positions=jnp.zeros((batch_size,), dtype=jnp.int32),
            tree_visits=jnp.zeros((batch_size,), dtype=jnp.int32),
            moves=jnp.zeros((batch_size,), dtype=jnp.int32),
            tree_buffer_state=tree_buffer_state,
            tree_written_at=jnp.zeros(
                (batch_size, self.tree_buffer.capacity), dtype=jnp.int32
            )
            if self.tree_buffer is not None
            else None,
        )

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
        self, epoch: int, evaluator: Evaluator, collection_state: CollectionState
    ) -> tuple[Evaluator, CollectionState]:
        """Switches self-play to the evaluator scheduled for `epoch`, if it isn't `evaluator`.

        The new evaluator's states replace the old one's, initialized from scratch: they may
        differ in shape (e.g. MCTS trees with a different `max_nodes`). Games in progress
        carry on. Self-play is compiled again for the new evaluator, the first time it plays.

        Args:
            epoch: the epoch about to start
            evaluator: the evaluator whose states `collection_state` holds
            collection_state: current collection state

        Returns:
            Tuple[Evaluator, CollectionState]: the evaluator for `epoch`, and the collection
                state with its states
        """
        scheduled = self.selfplay_schedule.at(epoch)
        if scheduled is evaluator:
            return evaluator, collection_state
        self.set_activity(
            f"epoch {epoch}: self-play switches to {describe(scheduled)} (compiling)",
            echo=True,
        )
        eval_state = self.init_eval_state(scheduled, collection_state.episodes.shape[0])
        return scheduled, replace(collection_state, eval_state=eval_state)

    def train_loop(
        self,
        seed: int,
        num_epochs: int,
        eval_every: int = 1,
        initial_state: TrainLoopOutput | None = None,
        fork_from: str | None = None,
        fork_replay_buffers: bool = True,
    ) -> TrainLoopOutput:
        """Runs the training loop for `num_epochs` epochs. Mostly configured by the Trainer's attributes.

        - Collects self-play episdoes across a batch of environments.
        - Trains the neural network on the collected experiences.
        - Tests the agent on a set of Testers, which evaluate the agent's performance.

        Saves a checkpoint of the train state every epoch, and the whole training state after the
        epochs set by `save_state_at` and `save_state_every` (see `save_state`).

        Args:
            seed: rng seed (int)
            num_epochs: number of epochs to run the training loop for, in all: a continued run
                stops after this many epochs, counting those done before
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
                network and optimizer state (its learning rate schedule starting over), and its
                replay buffers (see `fork_state`). The monitor run's config records it as
                `parent`. Not with `initial_state`.
            fork_replay_buffers: with `fork_from`, start with its replay buffers (default), or with
                empty ones

        Returns:
            TrainLoopOutput: contains train_state, collection_state, test_states, cur_epoch after
                training loop, and the rng to continue it with
        """
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
                    **self.extra_config,
                }
            )
        try:
            output = self._train_loop(
                seed,
                num_epochs,
                eval_every,
                initial_state,
                fork_from,
                fork_replay_buffers,
            )
        except BaseException as e:
            if self.monitor is not None:
                self.monitor.finish(
                    "stopped" if isinstance(e, KeyboardInterrupt) else "crashed"
                )
            raise
        if self.monitor is not None:
            self.monitor.finish()
        return output

    def _train_loop(
        self,
        seed: int,
        num_epochs: int,
        eval_every: int,
        initial_state: TrainLoopOutput | None,
        fork_from: str | None = None,
        fork_replay_buffers: bool = True,
    ) -> TrainLoopOutput:
        # init rng
        key = jax.random.PRNGKey(seed)
        # an exact continuation picks up where the run stopped, without a warmup
        exact = initial_state is not None and initial_state.key is not None

        # initialize states
        if initial_state:
            collection_state = initial_state.collection_state
            train_state = initial_state.train_state
            tester_states = initial_state.test_states
            cur_epoch = initial_state.cur_epoch
            if initial_state.key is not None:
                key = initial_state.key
            else:
                # don't replay the keys the original run used from epoch 0
                key = jax.random.fold_in(key, cur_epoch)
            # the evaluator whose states the collection state holds
            evaluator = self.selfplay_schedule.at(max(cur_epoch - 1, 0))
        else:
            cur_epoch = 0
            evaluator = self.selfplay_schedule.at(0)
            init_key, key = jax.random.split(key)
            if fork_from is None:
                # initialize collection state
                collection_state = self.init_collection_state(
                    init_key, self.batch_size, evaluator
                )
                # initialize train state
                train_state = self.init_train_state()
            else:
                self.set_activity(f"loading {fork_from}", echo=True)
                collection_state, train_state = self.fork_state(
                    fork_from, init_key, fork_replay_buffers
                )
            params = self.extract_model_params_fn(train_state)
            # initialize tester states
            tester_states = [tester.init(params=params) for tester in self.testers]

        evaluator, collection_state = self.follow_schedule(
            cur_epoch, evaluator, collection_state
        )
        params = self.extract_model_params_fn(train_state)
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
            # collect self-play games
            evaluator, collection_state = self.follow_schedule(
                cur_epoch, evaluator, collection_state
            )
            collect_key, key = jax.random.split(key)
            collect_keys = jax.random.split(collect_key, self.batch_size)
            # only the counts outlive the state donated to self-play: keeping the whole state
            # would keep a second copy of the replay buffers
            before = SelfplayCounters.of(collection_state)
            self.set_activity(f"epoch {cur_epoch}: self-play")
            collection_state = self.collect_steps(
                collect_keys,
                collection_state,
                params,
                self.collection_steps_per_epoch,
                evaluator=evaluator,
            )
            selfplay_metrics = self.selfplay_metrics(
                before, collection_state, cur_epoch
            )
            if isinstance(evaluator, MCTS):
                selfplay_metrics["selfplay_iterations"] = jnp.asarray(
                    evaluator.num_iterations
                )
            # train
            self.set_activity(f"epoch {cur_epoch}: training")
            train_key, key = jax.random.split(key)
            collection_state, train_state, metrics = self.train_steps(
                train_key,
                collection_state,
                train_state,
                self.train_steps_per_epoch,
                epoch=cur_epoch,
            )
            params = self.extract_model_params_fn(train_state)
            # log metrics
            self.log_metrics({**metrics, **selfplay_metrics}, cur_epoch)

            # test
            if cur_epoch % eval_every == 0:
                for i, test_state in enumerate(tester_states):
                    run_key, key = jax.random.split(key)
                    self.set_activity(
                        f"epoch {cur_epoch}: testing {self.testers[i].name}"
                    )
                    new_test_state, metrics, episode = self.testers[i].run(
                        key=run_key,
                        epoch_num=cur_epoch,
                        max_steps=self.max_episode_steps,
                        env_step_fn=self.env_step_fn,
                        env_init_fn=self.test_env_init_fn,
                        evaluator=self.evaluator_test,
                        state=test_state,
                        params=params,
                        log_fn=partial(self.log_metrics, epoch=cur_epoch),
                        activity_fn=lambda text, epoch=cur_epoch: self.set_activity(
                            f"epoch {epoch}: {text}", echo=True
                        ),
                    )

                    if metrics:
                        metrics = {k: v.mean() for k, v in metrics.items()}
                        self.log_metrics(metrics, cur_epoch)
                    # the monitor server renders the episode, off the training loop
                    if episode is not None and self.monitor is not None:
                        self.monitor.log(
                            cur_epoch, {f"{self.testers[i].name}_game": episode}
                        )
                    tester_states[i] = new_test_state
            # save checkpoint
            self.set_activity(f"epoch {cur_epoch}: saving checkpoint")
            self.save_checkpoint(train_state, cur_epoch)
            # next epoch
            cur_epoch += 1
            if self.should_save_state(cur_epoch):
                self.set_activity(f"epoch {cur_epoch - 1}: saving the training state")
                self.save_state(
                    TrainLoopOutput(
                        collection_state=collection_state,
                        train_state=train_state,
                        test_states=tester_states,
                        cur_epoch=cur_epoch,
                        key=key,
                    )
                )

        # return state so that training can be continued!
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            test_states=tester_states,
            cur_epoch=cur_epoch,
            key=key,
        )
