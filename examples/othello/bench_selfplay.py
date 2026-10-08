"""Times self-play: the trainer's `collect_steps` with the settings of `train.py`, so
changes to search or the network can be measured on their own.

    uv run examples/othello/bench_selfplay.py
    uv run examples/othello/bench_selfplay.py --envs 2048 --ckpt CKPT.eqx
    uv run examples/othello/bench_selfplay.py --trace /tmp/selfplay-trace

Starts from fresh environments with fixed keys, so with --save/--compare it also
checks that a change leaves self-play's results (every move's visit distribution, and
the positions it led to) the same. --trace writes a profile of one timed call; see
`profile_summary.py` to read it.
"""

import argparse
import time
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from game import (
    SYMMETRY_TRANSFORM_FNS,
    env,
    load_checkpoint,
    make_network,
    make_optimizer,
    state_to_nn_input,
    step_fn,
)
from xot import load_xot, make_xot_init_fn, split_xot

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer


def make_trainer(args) -> tuple[Trainer, tuple]:
    """A trainer set up like train.py's, and the parameters to self-play with."""
    xot_train, _ = split_xot(load_xot())
    if args.ckpt:
        resnet, params = load_checkpoint(
            args.ckpt, args.blocks, args.channels, args.dtype
        )
    else:
        resnet, resnet_state = make_network(
            args.blocks, args.channels, seed=args.seed, inference_dtype=args.dtype
        )
        params = (eqx.filter(resnet, eqx.is_inexact_array), resnet_state)
    evaluator = AlphaZero(MCTS)(
        eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
        num_iterations=args.sims,
        max_nodes=2 * args.sims,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=1.0,
    )
    trainer = Trainer(
        batch_size=args.envs,
        train_batch_size=4096,
        warmup_steps=0,
        collection_steps_per_epoch=args.steps,
        train_steps_per_epoch=0,
        nn=resnet,
        nn_state=params[1],
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=1e-4),
        optimizer=make_optimizer(1),
        evaluator=evaluator,
        # room for every step's sample and its symmetric copies, and no more
        memory_buffer=EpisodeReplayBuffer(
            capacity=(args.warmup + args.steps) * (1 + len(SYMMETRY_TRANSFORM_FNS))
        ),
        max_episode_steps=80,
        env_step_fn=step_fn,
        env_init_fn=make_xot_init_fn(xot_train, 0.15),
        state_to_nn_input_fn=state_to_nn_input,
        testers=[],
        data_transform_fns=SYMMETRY_TRANSFORM_FNS,
        ckpt_dir="/tmp/turbozero-bench-selfplay",
    )
    return trainer, params


def main():
    parser = argparse.ArgumentParser(description="Time Othello self-play.")
    parser.add_argument("--envs", type=int, default=1024, help="parallel games")
    parser.add_argument("--sims", type=int, default=64, help="MCTS iterations a move")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--ckpt", default=None, help="checkpoint to play with")
    parser.add_argument(
        "--dtype", default="float32", help="the network's dtype in self-play"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warmup", type=int, default=8, help="steps played before timing (untimed)"
    )
    parser.add_argument("--steps", type=int, default=16, help="steps per timed call")
    parser.add_argument("--repeats", type=int, default=3, help="timed calls")
    parser.add_argument("--trace", default=None, help="profile one more call to DIR")
    parser.add_argument(
        "--save", default=None, help="save the results of the warmup steps (.npz)"
    )
    parser.add_argument(
        "--compare", default=None, help="compare the warmup steps' results to a --save"
    )
    args = parser.parse_args()

    trainer, params = make_trainer(args)
    collect = trainer.collect_steps
    key = jax.random.PRNGKey(args.seed)
    init_key, key = jax.random.split(key)
    state = trainer.init_collection_state(init_key, args.envs)

    def keys(k):
        return jax.random.split(k, args.envs)

    # warmup: compiles, and plays the first moves (searches on an empty board are short)
    t = time.perf_counter()
    state = collect(keys(jax.random.fold_in(key, 0)), state, params, args.warmup)
    jax.block_until_ready(state)
    print(
        f"warmup ({args.warmup} steps, with compilation): {time.perf_counter() - t:.1f}s"
    )
    results: dict[str, Any] = {
        "policy_weights": np.asarray(state.buffer_state.buffer.policy_weights),
        "observation": np.asarray(state.buffer_state.buffer.observation_nn),
        "position": np.asarray(state.env_state.observation),
    }
    if args.save:
        np.savez_compressed(args.save, **results)
    if args.compare:
        compare(results, dict(np.load(args.compare)))

    # compile the timed call
    t = time.perf_counter()
    jax.block_until_ready(collect(keys(key), state, params, args.steps))
    print(f"compile + first call ({args.steps} steps): {time.perf_counter() - t:.1f}s")

    times = []
    for i in range(args.repeats):
        k = jax.random.fold_in(key, i + 1)
        t = time.perf_counter()
        state = collect(keys(k), state, params, args.steps)
        jax.block_until_ready(state)
        times.append(time.perf_counter() - t)
    per_step = np.array(times) / args.steps
    print(
        f"{args.envs} envs, {args.sims} sims, {args.blocks}x{args.channels}: "
        f"{1e3 * np.median(per_step):.1f} ms/step (min {1e3 * per_step.min():.1f}), "
        f"{args.envs / np.median(per_step):,.0f} moves/s"
    )
    stats = jax.devices()[0].memory_stats() or {}
    if "peak_bytes_in_use" in stats:
        print(f"peak device memory: {stats['peak_bytes_in_use'] / 2**30:.2f} GiB")

    if args.trace:
        k = jax.random.fold_in(key, args.repeats + 1)
        with jax.profiler.trace(args.trace, create_perfetto_trace=True):
            state = collect(keys(k), state, params, args.steps)
            jax.block_until_ready(state)
        print(f"trace of {args.steps} steps written to {args.trace}")


def compare(new: dict, old: dict) -> None:
    """Prints how far the warmup steps' results are from a saved run's."""
    for name, saved in old.items():
        a, b = jnp.asarray(new[name]), jnp.asarray(saved)
        if a.shape != b.shape:
            print(f"  {name}: shapes differ, {a.shape} vs {b.shape}")
            continue
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        different = (a != b).reshape(a.shape[0], -1).any(-1)
        print(
            f"  {name}: {int(different.sum())}/{different.size} games differ, "
            f"max abs diff {float(jnp.abs(a - b).max()):.3g}"
        )


if __name__ == "__main__":
    main()
