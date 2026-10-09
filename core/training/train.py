import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
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
        tree_buffer_state: state of the replay buffer of search tree positions, None without them
    """

    eval_state: Any
    env_state: Any
    buffer_state: ReplayBufferState
    metadata: StepMetadata
    episodes: jax.Array
    draws: jax.Array
    tree_positions: jax.Array
    tree_visits: jax.Array
    tree_buffer_state: ReplayBufferState | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainLoopOutput:
    """Stores the state of the training loop.

    `collection_state` is included to access replay memory.

    Attributes:
        collection_state: state of self-play episode collection.
        train_state: TrainState, holds model params and state, optimizer state
        test_states: states of testers
        cur_epoch: current epoch num
    """

    collection_state: CollectionState
    train_state: TrainState
    test_states: list[TestState]
    cur_epoch: int


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
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        int(m.group(1))
        for f in os.listdir(ckpt_dir)
        if (m := re.fullmatch(r"(\d+)\.eqx", f))
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
        evaluator: Evaluator,
        memory_buffer: EpisodeReplayBuffer,
        max_episode_steps: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        state_to_nn_input_fn: StateToNNInputFn,
        testers: Sequence[BaseTester],
        nn_state: eqx.nn.State | None = None,
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
            evaluator: the `Evaluator` to use during self-play
            memory_buffer: replay memory buffer class, used to store self-play experiences
            max_episode_steps: maximum number of steps in an episode. Self-play episodes still running after this many
                steps are truncated and their experiences discarded; an episode that terminates on its last allowed step is kept.
            env_step_fn: environment step function (env_state, action) -> (new_env_state, metadata)
            env_init_fn: environment initialization function (key) -> (env_state, metadata)
            state_to_nn_input_fn: function to convert environment state to neural network input
            testers: list of testers to evaluate the agent against (see core.testing.tester)
            nn_state: (optional) initial state of `nn` for stateful networks (e.g. with BatchNorm), from `eqx.nn.make_with_state`
            evaluator_test: (optional) evaluator to use during testing. If not provided, `evaluator` is used.
            test_env_init_fn: (optional) environment initialization function for test episodes, e.g. to test from
                the standard start while self-play starts from varied positions. If not provided, `env_init_fn` is used.
            selfplay_exploration: (optional) chooses the move self-play plays from the evaluator's output,
                e.g. to play random moves some of the time (see core.training.exploration). The evaluator's
                policy weights stay the training target. If not provided, self-play plays the evaluator's move.
            data_transform_fns: (optional) list of data transform functions to apply to self-play experiences (e.g. rotation, reflection, etc.)
            tree_positions: (optional) also train on positions from self-play's search trees, as OLIVAW did
                (see core.training.tree_positions). They're kept in a replay buffer of their own, and make up
                a share of each training batch. Needs an MCTS `evaluator`.
            extract_model_params_fn: (optional) function to extract model parameters from TrainState
            monitor: (optional) `core.monitor.Monitor` to log metrics and test episodes to (see a tester's `episode_fn`)
                - start the server with `turbozero-monitor`; a run is created on the first `train_loop`,
                  and later calls (e.g. continuing from `initial_state`) keep logging to it
            ckpt_dir: directory to save checkpoints
            max_checkpoints: maximum number of checkpoints to keep
            keep_every: (optional) also keep every checkpoint whose epoch is a multiple of this, beyond `max_checkpoints`
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
        self.evaluator_train = evaluator
        self.transform_fns = data_transform_fns
        self.selfplay_exploration = selfplay_exploration
        if tree_positions is not None and not isinstance(evaluator, MCTS):
            raise ValueError("tree_positions needs an MCTS evaluator")
        self.tree_positions = tree_positions
        self.tree_buffer = (
            EpisodeReplayBuffer(capacity=tree_positions.capacity)
            if tree_positions is not None
            else None
        )
        self.step_train = partial(
            search_and_step,
            evaluator=self.evaluator_train,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn,
            max_steps=self.max_episode_steps,
            choose_action=selfplay_exploration.choose_action
            if selfplay_exploration is not None
            else None,
        )
        self.count_distinct_observations = jax.jit(
            self.memory_buffer.count_distinct_observations
        )
        # training
        self.train_steps_per_epoch = train_steps_per_epoch
        self.train_batch_size = train_batch_size
        # testing
        self.testers = testers
        self.evaluator_test = (
            evaluator_test if evaluator_test is not None else evaluator
        )
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
        os.makedirs(ckpt_dir, exist_ok=True)
        # monitor
        self.monitor = monitor
        self.extra_config = extra_config if extra_config is not None else {}

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
        """Returns a dictionary of the configuration of the trainer. Used for logging to the monitor."""
        return {
            "batch_size": self.batch_size,
            "train_batch_size": self.train_batch_size,
            "warmup_steps": self.warmup_steps,
            "collection_steps_per_epoch": self.collection_steps_per_epoch,
            "train_steps_per_epoch": self.train_steps_per_epoch,
            "evaluator_train": self.evaluator_train.__class__.__name__,
            "evaluator_train_config": self.evaluator_train.get_config(),
            "evaluator_test": self.evaluator_test.__class__.__name__,
            "evaluator_test_config": self.evaluator_test.get_config(),
            "selfplay_exploration_config": self.selfplay_exploration.get_config()
            if self.selfplay_exploration is not None
            else None,
            "memory_buffer": self.memory_buffer.__class__.__name__,
            "memory_buffer_config": self.memory_buffer.get_config(),
            "tree_positions": self.tree_positions.get_config()
            if self.tree_positions is not None
            else None,
        }

    def collect(
        self, key: jax.Array, state: CollectionState, params: Any
    ) -> CollectionState:
        """Collects self-play data for a single step.

        - Stores experience in replay buffer.
        - Resets environment/evaluator if episode is terminated.

        Args:
            key: rng
            state: current collection state (environment, evaluator, replay buffer)
            params: model parameters

        Returns:
            CollectionState: updated collection state
        """
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
        ) = self.step_train(
            key=key,
            env_state=state.env_state,
            env_state_metadata=state.metadata,
            eval_state=state.eval_state,
            params=params,
        )
        search_value = self.evaluator_train.get_value(searched_eval_state)

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
                tree_key, state, searched_eval_state, reused
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
        )

    def store_tree_positions(
        self, key: jax.Array, state: CollectionState, tree: Any, reused: jax.Array
    ) -> CollectionState:
        """Stores positions from a self-play search tree in the tree position replay buffer
        (see `tree_positions`), with their transformed copies.

        Args:
            key: rng
            state: current collection state
            tree: the search tree, after the search
            reused: the root child whose subtree the next search reuses, NULL_INDEX for none

        Returns:
            CollectionState: updated collection state
        """
        assert self.tree_positions is not None and self.tree_buffer is not None
        assert state.tree_buffer_state is not None
        assert isinstance(self.evaluator_train, MCTS)
        indices, valid = select_nodes(
            key,
            tree,
            self.tree_positions.per_move,
            self.tree_positions.min_visits,
            self.tree_positions.most_visited,
            reused
            if self.tree_positions.discarded_only and self.evaluator_train.persist_tree
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
        return replace(
            state,
            tree_buffer_state=self.tree_buffer.add_experiences(
                state.tree_buffer_state, experiences, valid_experiences
            ),
            tree_positions=state.tree_positions + valid.sum(),
            tree_visits=state.tree_visits
            + jnp.where(valid, tree.data.n[indices], 0).sum(),
        )

    @partial(jax.jit, static_argnums=(0, 4))
    def collect_steps(
        self, key: jax.Array, state: CollectionState, params: Any, num_steps: int
    ) -> CollectionState:
        """Collects self-play data for `num_steps` steps in every environment.

        Args:
            key: rng, one key per environment
            state: current collection state
            params: model parameters
            num_steps: number of self-play steps to collect

        Returns:
            CollectionState: updated collection state
        """
        if num_steps > 0:

            def collect_env(key: jax.Array, state: CollectionState) -> CollectionState:
                keys = jax.random.split(key, num_steps)
                return jax.lax.fori_loop(
                    0, num_steps, lambda i, s: self.collect(keys[i], s, params), state
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
        tree_buffer_state: ReplayBufferState | None = None,
        num_tree: jax.Array | None = None,
    ) -> tuple[TrainState, dict]:
        """Performs `num_steps` training steps, compiled into a single `jax.lax.scan`.

        Each step samples a minibatch from the replay buffer and updates the parameters.

        Does not check that the replay buffer holds a finished episode to sample, see `train_steps`.

        Args:
            key: rng
            buffer_state: replay buffer state
            train_state: current training state
            num_steps: number of training steps to perform
            tree_buffer_state: (optional) state of the search tree positions' replay buffer
                (see `tree_positions`), to sample the first `num_tree` samples of each minibatch from
            num_tree: number of tree positions in each minibatch, a traced value so that changing it
                doesn't recompile. With none to sample, it must be 0.

        Returns:
            Tuple[TrainState, dict]: updated training state and metrics (mean across steps)
        """
        # the buffer doesn't change while training, so find which entries can be sampled once per epoch
        mask = self.memory_buffer.sample_mask(buffer_state)
        add_tree_positions = (
            self.tree_position_sampler(tree_buffer_state, num_tree)
            if tree_buffer_state is not None and num_tree is not None
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

    def tree_position_sampler(
        self, tree_buffer_state: ReplayBufferState, num_tree: jax.Array
    ) -> Callable[[jax.Array, BaseExperience], BaseExperience]:
        """Makes a function that puts tree positions (see `tree_positions`) into a minibatch: it
        replaces the first `num_tree` samples of a minibatch with ones sampled from the tree position
        replay buffer, uniformly with replacement (so however few there are).

        Args:
            tree_buffer_state: state of the tree position replay buffer, which doesn't change while
                the function is used
            num_tree: number of tree positions per minibatch; with none in the buffer, it must be 0

        Returns:
            Callable[[jax.Array, BaseExperience], BaseExperience]: (rng, minibatch) -> minibatch
        """
        assert self.tree_buffer is not None
        tree_buffer = self.tree_buffer
        mask = tree_buffer.sample_mask(tree_buffer_state)
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
        The minibatch is sampled uniformly without replacement from the finished episodes in the buffer.
        With `tree_positions`, a share of it is sampled (uniformly, with replacement) from the search tree
        positions instead.

        Args:
            key: rng
            collection_state: current collection state
            train_state: current training state
            num_steps: number of training steps to perform
            epoch: the current epoch, which sets the share of tree positions (see `TreePositions.ratio_at`)

        Returns:
            Tuple[CollectionState, TrainState, dict]:
                - updated collection state
                - updated training state
                - metrics

        Raises:
            ValueError: if no episode has finished yet, so there is nothing to sample
        """
        if num_steps == 0:
            return collection_state, train_state, {}
        # the buffer doesn't change while training, so checking once covers every step
        self.memory_buffer.check_can_sample(collection_state.buffer_state)
        if self.tree_positions is None:
            train_state, metrics = self.train_epoch(
                key, collection_state.buffer_state, train_state, num_steps
            )
            return collection_state, train_state, metrics
        assert self.tree_buffer is not None
        tree_buffer_state = collection_state.tree_buffer_state
        assert tree_buffer_state is not None
        num_tree = (
            self.tree_positions.batch_count(epoch, self.train_batch_size)
            if self.tree_buffer.sample_mask(tree_buffer_state).any()
            else 0
        )
        train_state, metrics = self.train_epoch(
            key,
            collection_state.buffer_state,
            train_state,
            num_steps,
            tree_buffer_state,
            jnp.array(num_tree, dtype=jnp.int32),
        )
        metrics = {
            **metrics,
            "tree_batch_fraction": jnp.array(num_tree / self.train_batch_size),
        }
        return collection_state, train_state, metrics

    def selfplay_metrics(self, before: CollectionState, after: CollectionState) -> dict:
        """Measures how varied self-play is, to spot it collapsing into the same few games.

        Args:
            before: collection state before this epoch's self-play
            after: collection state after it

        Returns:
            dict: metrics
                - `selfplay_episodes`: episodes that terminated in between
                - `selfplay_draw_fraction`: fraction of them that ended with every reward 0
                  (draws, in two-player zero-sum games), omitted if none terminated
                - `buffer_distinct_positions`: distinct observations among the experiences the
                  replay buffer can sample (data transforms' outputs count as observations too)
                - `buffer_distinct_fraction`: that as a fraction of those experiences
                - with `tree_positions`: `tree_positions` stored in between (not counting transformed
                  copies), `tree_positions_per_move`, and their `tree_mean_visits`
        """
        episodes = (after.episodes - before.episodes).sum()
        draws = (after.draws - before.draws).sum()
        distinct, total = self.count_distinct_observations(after.buffer_state)
        metrics = {
            "selfplay_episodes": episodes,
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

    def init_collection_state(self, key: jax.Array, batch_size: int) -> CollectionState:
        """Initializes the collection state (see CollectionState).

        Args:
            key: rng
            batch_size: number of parallel environments

        Returns:
            CollectionState: initialized collection state
        """
        # make template experience
        template_experience = self.make_template_experience()
        # init buffer state
        buffer_state = self.memory_buffer.init(batch_size, template_experience)
        tree_buffer_state = (
            self.tree_buffer.init(batch_size, template_experience)
            if self.tree_buffer is not None
            else None
        )
        # init env state
        env_init_key, key = jax.random.split(key)
        env_keys = jax.random.split(env_init_key, batch_size)
        env_state, metadata = jax.vmap(self.env_init_fn)(env_keys)
        # init evaluator state
        eval_state = self.evaluator_train.init_batched(
            batch_size, template_embedding=self.template_env_state
        )
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
            tree_buffer_state=tree_buffer_state,
        )

    def train_loop(
        self,
        seed: int,
        num_epochs: int,
        eval_every: int = 1,
        initial_state: TrainLoopOutput | None = None,
    ) -> TrainLoopOutput:
        """Runs the training loop for `num_epochs` epochs. Mostly configured by the Trainer's attributes.

        - Collects self-play episdoes across a batch of environments.
        - Trains the neural network on the collected experiences.
        - Tests the agent on a set of Testers, which evaluate the agent's performance.

        Args:
            seed: rng seed (int)
            num_epochs: number of epochs to run the training loop for
            eval_every: number of epochs between evaluations
            initial_state: (optional) TrainLoopOutput, used to continue training from a previous state

        Returns:
            TrainLoopOutput: contains train_state, collection_state, test_states, cur_epoch after training loop
        """
        if self.monitor is not None:
            self.monitor.start(
                config={
                    **self.get_config(),
                    "run": {
                        "seed": seed,
                        "num_epochs": num_epochs,
                        "eval_every": eval_every,
                    },
                    **self.extra_config,
                }
            )
        try:
            output = self._train_loop(seed, num_epochs, eval_every, initial_state)
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
    ) -> TrainLoopOutput:
        # init rng
        key = jax.random.PRNGKey(seed)

        # initialize states
        if initial_state:
            collection_state = initial_state.collection_state
            train_state = initial_state.train_state
            tester_states = initial_state.test_states
            cur_epoch = initial_state.cur_epoch
            # don't replay the keys the original run used from epoch 0
            key = jax.random.fold_in(key, cur_epoch)
        else:
            cur_epoch = 0
            # initialize collection state
            init_key, key = jax.random.split(key)
            collection_state = self.init_collection_state(init_key, self.batch_size)
            # initialize train state
            train_state = self.init_train_state()
            params = self.extract_model_params_fn(train_state)
            # initialize tester states
            tester_states = [tester.init(params=params) for tester in self.testers]

        # warmup
        # populate replay buffer with initial self-play games
        if self.warmup_steps > 0:
            self.set_activity(f"warmup self-play ({self.warmup_steps} steps)")
        params = self.extract_model_params_fn(train_state)
        collect_key, key = jax.random.split(key)
        collect_keys = jax.random.split(collect_key, self.batch_size)
        collection_state = self.collect_steps(
            collect_keys, collection_state, params, self.warmup_steps
        )

        # training loop
        while cur_epoch < num_epochs:
            # collect self-play games
            collect_key, key = jax.random.split(key)
            collect_keys = jax.random.split(collect_key, self.batch_size)
            prev_collection_state = collection_state
            self.set_activity(f"epoch {cur_epoch}: self-play")
            collection_state = self.collect_steps(
                collect_keys, collection_state, params, self.collection_steps_per_epoch
            )
            selfplay_metrics = self.selfplay_metrics(
                prev_collection_state, collection_state
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

        # return state so that training can be continued!
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            test_states=tester_states,
            cur_epoch=cur_epoch,
        )
