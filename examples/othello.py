"""AlphaZero on Othello, tested against pgx's pretrained Othello model and a greedy
tile-counting baseline.

Self-play games are collected in parallel across a batch of environments (and across
every available GPU), with Monte Carlo Tree Search run on each of them; the network
then trains on minibatches sampled from replay memory.

    uv run examples/othello.py
    uv run examples/othello.py --epochs 20 --wandb turbozero-othello
    uv run examples/othello.py --render renders

The first epoch is slow: nearly all of the training loop is JIT-compiled the first
time it runs. The hyperparameters here are only an example; tune them for your task
and hardware.
"""

import argparse
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pgx

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn, make_nn_eval_fn_no_params_callable
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.testing.two_player_baseline import TwoPlayerBaseline
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


def make_rot_transform_fn(amnt: int):
    """A DataTransformFn that rotates the board by `amnt` quarter turns, to generate
    an extra training sample from each self-play step. The policy mask and weights
    are rotated to match: only the first 64 actions are board squares, the 65th
    (pass) stays where it is."""

    def rot_transform_fn(mask, policy, state):
        action_ids = jnp.arange(65)
        # we only use state.observation, no need to update the rest of the state fields
        new_obs = jnp.rot90(state.observation, amnt, axes=(-3, -2))
        # map action ids to new action ids
        idxs = jnp.arange(64).reshape(8, 8)
        new_idxs = jnp.rot90(idxs, amnt, axes=(0, 1)).flatten()
        action_ids = action_ids.at[:64].set(new_idxs)
        return mask[..., action_ids], policy[..., action_ids], state.replace(observation=new_obs)

    return rot_transform_fn


def make_test_evaluator(eval_fn) -> MCTS:
    """Evaluator used in test games: a larger search budget than self-play, and
    temperature 0 to always play the most-visited action. Baselines share these
    settings so that only the quality of the policy/value estimates differs."""
    return AlphaZero(MCTS)(
        eval_fn=eval_fn,
        num_iterations=64,
        max_nodes=80,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=0.0,
    )


def main():
    parser = argparse.ArgumentParser(description="AlphaZero on Othello.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5, help="epochs between test games")
    parser.add_argument("--wandb", metavar="PROJECT", default="", help="log to this wandb project")
    parser.add_argument(
        "--render",
        metavar="DIR",
        default=None,
        help="save a .gif of a game against each baseline to DIR (needs the cairo system library)",
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
        env.observation_shape,
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

    # baselines: pgx's pretrained model (others are listed at
    # https://sotets.uk/pgx/api/#pgx.BaselineModelId) and the greedy tile counter
    pretrained = make_nn_eval_fn_no_params_callable(pgx.make_baseline_model("othello_v0"), state_to_nn_input)
    greedy = make_nn_eval_fn_no_params_callable(greedy_eval, state_to_nn_input)

    render_fn = None
    if args.render is not None:
        # imported here since it needs cairo (on Ubuntu: apt-get install libcairo2-dev)
        from core.testing.utils import render_pgx_2p
        render_fn = partial(render_pgx_2p, p1_label="Black", p2_label="White", duration=900)

    testers = [
        TwoPlayerBaseline(
            num_episodes=128,
            baseline_evaluator=make_test_evaluator(eval_fn),
            render_fn=render_fn,
            render_dir=args.render,
            name=name,
        )
        for name, eval_fn in [("pretrained", pretrained), ("greedy", greedy)]
    ]

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
        # rotate each sample by 90, 180 and 270 degrees
        data_transform_fns=[make_rot_transform_fn(i) for i in range(1, 4)],
        wandb_project_name=args.wandb,
    )
    trainer.train_loop(seed=args.seed, num_epochs=args.epochs, eval_every=args.eval_every)


if __name__ == "__main__":
    main()
