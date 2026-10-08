"""AlphaZero on tic-tac-toe, tested against an opponent that plays random legal moves.

A sanity check that training works end to end: small and fast enough to train on a
CPU in a minute or two, and the trained agent should soon win most games against
the random player and lose almost none. Test games give the agent only
`--test-iterations` MCTS simulations (1 by default: it plays the move its network
likes best), so the result measures what the network learned rather than what
search finds on its own (in a game this small, search alone beats a random player).

    uv run examples/tic_tac_toe.py
    uv run examples/tic_tac_toe.py --monitor

Start the monitor first, in another shell, with `uv run turbozero-monitor`. See
examples/othello.py for a walk-through of each component.
"""

import argparse
import tempfile
from functools import partial
from typing import cast

import equinox as eqx
import jax
import optax
import pgx

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.random_evaluator import RandomEvaluator
from core.memory.replay_memory import EpisodeReplayBuffer
from core.monitor import DEFAULT_URL, Monitor
from core.monitor.renderers import pgx_two_player_episode
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
from core.types import StepMetadata

env = pgx.make("tic_tac_toe")

# a game lasts at most one move per square
MAX_STEPS = 9


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
    parser = argparse.ArgumentParser(
        description="AlphaZero on tic-tac-toe, tested against a random player."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--eval-every", type=int, default=5, help="epochs between test games"
    )
    parser.add_argument(
        "--test-iterations",
        type=int,
        default=1,
        help="MCTS simulations per move in test games; 1 plays the network's favourite move",
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

    resnet, resnet_state = eqx.nn.make_with_state(AZResnet)(
        AZResnetConfig(
            policy_head_out_size=env.num_actions,
            num_blocks=2,
            num_channels=16,
        ),
        # pgx types observation_shape as Tuple[int, ...]; for board games it's (height, width, channels)
        cast(tuple[int, int, int], env.observation_shape),
        key=jax.random.PRNGKey(args.seed),
    )
    eval_fn = make_nn_eval_fn(resnet, state_to_nn_input)

    # self-play samples moves in proportion to visit counts, to explore
    evaluator = AlphaZero(MCTS)(
        eval_fn=eval_fn,
        num_iterations=32,
        max_nodes=40,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=1.0,
    )
    # test games play the most-visited move, with (by default) next to no search, and
    # without the Dirichlet noise AlphaZero mixes into the root policy to explore: with
    # 1 simulation, the noise would pick the move instead of the network
    evaluator_test = AlphaZero(MCTS)(
        eval_fn=eval_fn,
        num_iterations=args.test_iterations,
        max_nodes=args.test_iterations + 1,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=0.0,
        dirichlet_epsilon=0.0,
    )

    trainer = Trainer(
        batch_size=64,
        train_batch_size=256,
        warmup_steps=MAX_STEPS,
        collection_steps_per_epoch=MAX_STEPS,
        train_steps_per_epoch=8,
        nn=resnet,
        nn_state=resnet_state,
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=1e-4),
        optimizer=optax.adam(3e-3),
        evaluator=evaluator,
        evaluator_test=evaluator_test,
        memory_buffer=EpisodeReplayBuffer(capacity=64),
        max_episode_steps=MAX_STEPS,
        env_step_fn=step_fn,
        env_init_fn=init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        # each player moves first in about half of the games
        testers=[
            TwoPlayerBaseline(
                num_episodes=512,
                baseline_evaluator=RandomEvaluator(),
                episode_fn=pgx_two_player_episode(p1_label="X", p2_label="O")
                if args.monitor
                else None,
                name="random",
            )
        ],
        monitor=Monitor(args.monitor, project="tic_tac_toe") if args.monitor else None,
        # a fresh directory each run: the trainer won't overwrite another run's checkpoints
        ckpt_dir=tempfile.mkdtemp(prefix="turbozero-tic-tac-toe-"),
    )
    trainer.train_loop(
        seed=args.seed, num_epochs=args.epochs, eval_every=args.eval_every
    )


if __name__ == "__main__":
    main()
