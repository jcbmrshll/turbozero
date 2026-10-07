"""AlphaZero on Connect Four with weighted MCTS, tested against the best parameters
found so far.

Weighted MCTS (https://twitter.com/ptrschmdtnlsn/status/1748800529608888362) backs up
a softmax-weighted sum of child Q-values instead of a plain average, with its
temperature set by `--q-temperature`. Use `--search mcts` to train with classic MCTS
for comparison. See examples/othello.py for a walk-through of each component.

    uv run examples/connect_four.py
    uv run examples/connect_four.py --search mcts --wandb weighted-mcts-test

The hyperparameters here are only an example; tune them for your task and hardware.
"""

import argparse
from functools import partial

import optax
import pgx

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.weighted_mcts import WeightedMCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.testing.two_player_tester import TwoPlayerTester
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
from core.types import StepMetadata

env = pgx.make("connect_four")

# a game lasts at most one move per square
MAX_STEPS = 42


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
    return state.observation


def main():
    parser = argparse.ArgumentParser(description="AlphaZero on Connect Four with weighted MCTS.")
    parser.add_argument("--search", choices=["weighted", "mcts"], default="weighted")
    parser.add_argument(
        "--q-temperature",
        type=float,
        default=1.0,
        help="temperature applied to child Q-values when backing them up (weighted search only)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--wandb", metavar="PROJECT", default="", help="log to this wandb project")
    args = parser.parse_args()

    resnet = AZResnet(AZResnetConfig(
        policy_head_out_size=env.num_actions,
        num_blocks=4,
        num_channels=16,
    ))

    if args.search == "weighted":
        search, search_kwargs = WeightedMCTS, {"q_temperature": args.q_temperature}
    else:
        search, search_kwargs = MCTS, {}
    make_evaluator = partial(
        AlphaZero(search),
        eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
        num_iterations=100,
        max_nodes=200,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        **search_kwargs,
    )

    batch_size = 128
    train_batch_size = 512
    trainer = Trainer(
        batch_size=batch_size,
        train_batch_size=train_batch_size,
        warmup_steps=MAX_STEPS,
        collection_steps_per_epoch=MAX_STEPS,
        # roughly one pass over each epoch's new samples
        train_steps_per_epoch=(batch_size * MAX_STEPS) // train_batch_size,
        nn=resnet,
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=0.0001),
        optimizer=optax.adam(5e-3),
        # self-play samples moves in proportion to visit counts; test games play the most-visited
        evaluator=make_evaluator(temperature=1.0, dirichlet_alpha=0.6),
        evaluator_test=make_evaluator(temperature=0.0),
        memory_buffer=EpisodeReplayBuffer(capacity=300),
        max_episode_steps=MAX_STEPS,
        env_step_fn=step_fn,
        env_init_fn=init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=[TwoPlayerTester(num_episodes=64)],
        wandb_project_name=args.wandb,
    )
    trainer.train_loop(seed=args.seed, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
