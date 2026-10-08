"""AlphaZero on Othello, tested on a ladder of opponents: a random player, a greedy
tile counter, then pgx's pretrained Othello model searching more and more.

Self-play games are collected in parallel across a batch of environments (and across
every available GPU), with Monte Carlo Tree Search run on each of them; the network
then trains on minibatches sampled from replay memory.

    uv run examples/othello.py
    uv run examples/othello.py --epochs 20 --monitor

Start the monitor first, in another shell, with `uv run turbozero-monitor`. It shows
the metrics, which rung of the ladder the agent has reached, and a game against the
last opponent it played, which it renders itself.

The first epoch is slow: nearly all of the training loop is JIT-compiled the first
time it runs. The hyperparameters here are only an example; tune them for your task
and hardware.
"""

import argparse
from functools import partial
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pgx

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import (
    make_nn_eval_fn,
    make_nn_eval_fn_no_params_callable,
)
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.random_evaluator import RandomEvaluator
from core.memory.replay_memory import EpisodeReplayBuffer
from core.monitor import DEFAULT_URL, Monitor
from core.monitor.renderers import pgx_two_player_episode
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.testing.ladder import LadderTester, Rung
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
from core.types import StepMetadata

# vectorized environments pair well with batched AlphaZero; pgx has many more:
# https://sotets.uk/pgx/othello/
env = pgx.make("othello")


# turbozero interfaces with an environment through a step fn (state, action) and an
# init fn (key), each returning the new state along with the StepMetadata it needs:
# rewards for each player, a mask of legal actions, whether the episode has
# terminated, the id of the player to move, and the step number
def step_fn(state, action):
    state = env.step(state, action)
    return state, metadata(state)


def init_fn(key):
    state = env.init(key)
    return state, metadata(state)


def metadata(state) -> StepMetadata:
    return StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count,
    )


def state_to_nn_input(state):
    """Converts an environment state to the network's input. pgx provides this as
    `state.observation`; other environments may need their own conversion."""
    return state.observation


def greedy_eval(obs):
    """Values a position by the active player's lead in tiles, with a uniform policy:
    a baseline that doesn't use a neural network at all."""
    value = (obs[..., 0].sum() - obs[..., 1].sum()) / 64
    return jnp.ones((1, env.num_actions)), jnp.array([value])


def make_symmetry_transform_fn(quarter_turns: int, transpose: bool):
    """A DataTransformFn that maps the board through one of its symmetries (an optional
    transpose, then `quarter_turns` quarter turns), to generate an extra training sample
    from each self-play step. Othello's rules don't change under any of the board's 8
    symmetries. The policy mask and weights are mapped to match: only the first 64
    actions are board squares, the 65th (pass) stays where it is."""

    def transform_fn(mask, policy, state):
        # we only use state.observation, no need to update the rest of the state fields
        new_obs = state.observation
        # idxs[r, c] is the square that ends up at (r, c)
        idxs = jnp.arange(64).reshape(8, 8)
        if transpose:
            new_obs = jnp.swapaxes(new_obs, -3, -2)
            idxs = idxs.T
        new_obs = jnp.rot90(new_obs, quarter_turns, axes=(-3, -2))
        idxs = jnp.rot90(idxs, quarter_turns, axes=(0, 1))
        action_ids = jnp.arange(65).at[:64].set(idxs.flatten())
        return (
            mask[..., action_ids],
            policy[..., action_ids],
            state.replace(observation=new_obs),
        )

    return transform_fn


# every symmetry but the identity: 7 extra samples per self-play step
SYMMETRY_TRANSFORM_FNS = [
    make_symmetry_transform_fn(quarter_turns, transpose)
    for transpose in (False, True)
    for quarter_turns in range(4)
    if quarter_turns or transpose
]


def make_test_evaluator(eval_fn, num_iterations: int = 64) -> MCTS:
    """Evaluator used in test games: temperature 0 to always play the most-visited
    action. Opponents share these settings, so that only the quality of their
    policy/value estimates and their search budget differ."""
    return AlphaZero(MCTS)(
        eval_fn=eval_fn,
        num_iterations=num_iterations,
        max_nodes=num_iterations + 16,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=0.0,
    )


def main():
    parser = argparse.ArgumentParser(description="AlphaZero on Othello.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--eval-every", type=int, default=5, help="epochs between test games"
    )
    parser.add_argument(
        "--monitor",
        nargs="?",
        const=DEFAULT_URL,
        default=None,
        metavar="URL",
        help=f"log to a turbozero monitor (default {DEFAULT_URL}); start it with `uv run turbozero-monitor`",
    )
    args = parser.parse_args()

    # the residual network from the AlphaZero paper; any equinox module works (see
    # core.networks.utils.apply_nn). It uses BatchNorm, so it's created along with its state
    resnet, resnet_state = eqx.nn.make_with_state(AZResnet)(
        AZResnetConfig(
            policy_head_out_size=env.num_actions,
            num_blocks=4,
            num_channels=32,
        ),
        # pgx types observation_shape as Tuple[int, ...]; for board games it's (height, width, channels)
        cast(tuple[int, int, int], env.observation_shape),
        key=jax.random.PRNGKey(args.seed),
    )

    # AlphaZero takes an arbitrary search backend, here classic MCTS. Temperature 1.0
    # samples moves in proportion to visit counts, for exploration during self-play
    evaluator = AlphaZero(MCTS)(
        eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
        num_iterations=32,
        max_nodes=40,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=1.0,
    )
    evaluator_test = make_test_evaluator(make_nn_eval_fn(resnet, state_to_nn_input))

    # opponents: the greedy tile counter and pgx's pretrained model (others are listed
    # at https://sotets.uk/pgx/api/#pgx.BaselineModelId)
    pretrained = make_nn_eval_fn_no_params_callable(
        pgx.make_baseline_model("othello_v0"), state_to_nn_input
    )
    greedy = make_nn_eval_fn_no_params_callable(greedy_eval, state_to_nn_input)

    # with a monitor, each test sends a game for the monitor server to render
    # (drawing it needs the cairo system library there, not here)
    episode_fn = (
        pgx_two_player_episode(p1_label="Black", p2_label="White")
        if args.monitor
        else None
    )

    # the ladder, easiest first: each test plays the lowest rung the agent hasn't
    # beaten, and moves up past every rung it scores at least 55% against. Our agent
    # searches 64 iterations a move throughout; pgx's model searches more and more
    rungs = [
        Rung("random", RandomEvaluator()),
        Rung("greedy", make_test_evaluator(greedy)),
    ] + [
        Rung(f"pgx{n}", make_test_evaluator(pretrained, n)) for n in (1, 4, 16, 64, 256)
    ]
    testers = [LadderTester(num_episodes=128, rungs=rungs, episode_fn=episode_fn)]

    # each epoch collects `collection_steps_per_epoch` self-play steps in each of
    # `batch_size` environments, then takes `train_steps_per_epoch` training steps
    trainer = Trainer(
        batch_size=1024,
        train_batch_size=4096,
        warmup_steps=0,
        collection_steps_per_epoch=256,
        train_steps_per_epoch=64,
        nn=resnet,
        nn_state=resnet_state,
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=0.0),
        optimizer=optax.adam(1e-3),
        evaluator=evaluator,
        # stores `capacity` samples for each of the `batch_size` environments
        memory_buffer=EpisodeReplayBuffer(capacity=1000),
        max_episode_steps=80,
        env_step_fn=step_fn,
        env_init_fn=init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=testers,
        evaluator_test=evaluator_test,
        # add each sample's 7 symmetric copies
        data_transform_fns=SYMMETRY_TRANSFORM_FNS,
        monitor=Monitor(args.monitor, project="othello") if args.monitor else None,
    )
    trainer.train_loop(
        seed=args.seed, num_epochs=args.epochs, eval_every=args.eval_every
    )


if __name__ == "__main__":
    main()
