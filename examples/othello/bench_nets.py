"""Times networks on their own: a forward pass in inference mode, at self-play's batch, and a
training step (loss, gradients and AdamW's update), at training's batch, so architectures can be
compared by the time they cost rather than their FLOPs.

    uv run --extra cuda examples/othello/bench_nets.py resnet:10x128 transformer:8x256
    uv run --extra cuda examples/othello/bench_nets.py resnet:10x128 --json bench.jsonl

Networks are named as train.py's --net names them. Inputs are random boards; a network's
time doesn't depend on them. Each time is the median, over --rounds, of the mean time of
--reps calls in a row. Also reports each network's parameters and XLA's count of the forward
pass's FLOPs per position.
"""

import argparse
import json
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from game import env, make_network, make_optimizer

from core.memory.replay_memory import BaseExperience
from core.networks.utils import apply_nn
from core.training.loss_fns import az_default_loss_fn


def timed(fn, *args, reps: int, rounds: int) -> float:
    """Median over `rounds` of the mean seconds of `reps` calls of fn(*args) in a row, after a
    first call (compilation). `fn` returns its arguments for the next call, so it can donate
    them."""
    args = jax.block_until_ready(fn(*args))
    means = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            args = fn(*args)
        jax.block_until_ready(args)
        means.append((time.perf_counter() - start) / reps)
    return float(np.median(means))


def random_batch(rng: np.random.Generator, size: int) -> BaseExperience:
    """`size` random positions, each with every action legal and a random search policy."""
    squares = rng.integers(0, 3, size=(size, 8, 8))
    observation = np.stack([squares == 1, squares == 2], axis=-1)
    policy = rng.random((size, env.num_actions)).astype(np.float32)
    return BaseExperience(
        reward=jnp.asarray(rng.choice([-1.0, 0.0, 1.0], size=(size, 2)), jnp.float32),
        policy_weights=jnp.asarray(policy / policy.sum(axis=1, keepdims=True)),
        policy_mask=jnp.ones((size, env.num_actions), dtype=bool),
        observation_nn=jnp.asarray(observation),
        cur_player_id=jnp.asarray(rng.integers(0, 2, size=size), jnp.int32),
        search_value=jnp.zeros(size, jnp.float32),
    )


def forward_flops(network, network_state, x: jax.Array) -> float:
    """XLA's count of the FLOPs of an inference-mode forward pass on `x`, per position. Counted
    on the CPU: on the GPU, XLA doesn't count the convolutions it hands to cuDNN."""
    cpu = jax.devices("cpu")[0]
    network, network_state, x = jax.tree.map(
        lambda a: jax.device_put(a, cpu) if eqx.is_array(a) else a,
        (network, network_state, x),
    )
    with jax.default_device(cpu):
        compiled = (
            jax.jit(lambda x: apply_nn(network, network_state, x)[0]).lower(x).compile()
        )
    cost: Any = compiled.cost_analysis()
    if isinstance(cost, list):
        cost = cost[0]
    return float(cost["flops"]) / x.shape[0]


def bench(net: str, args) -> dict:
    network, network_state = make_network(
        net, inference_dtype=args.dtype, train_dtype=args.dtype, remat=args.remat
    )
    params = eqx.filter(network, eqx.is_inexact_array)
    static = eqx.filter(network, eqx.is_inexact_array, inverse=True)
    rng = np.random.default_rng(0)
    result: dict[str, Any] = {
        "net": net,
        "dtype": args.dtype,
        "params": int(sum(x.size for x in jax.tree.leaves(params))),
    }

    # forward: inference mode, as self-play evaluates its search's leaves
    inference = eqx.nn.inference_mode(network)
    x = random_batch(rng, args.batch).observation_nn
    result["gflop_per_position"] = forward_flops(inference, network_state, x[:64]) / 1e9

    @jax.jit
    def forward(x, carry):
        (policy, value), _ = apply_nn(inference, network_state, x)
        # depend on the outputs, so the calls queue one after another
        return x, carry + policy[0, 0] + value[0, 0]

    result["forward_ms"] = 1e3 * timed(
        forward, x, jnp.float32(0), reps=args.reps, rounds=args.rounds
    )
    result["forward_batch"] = args.batch

    # a training step, as the trainer's (see Trainer.train_step)
    optimizer = make_optimizer(10_000)
    batch = random_batch(rng, args.train_batch)

    def step(params, opt_state, network_state, batch):
        grad_fn = eqx.filter_value_and_grad(az_default_loss_fn, has_aux=True)
        (_, (_, network_state)), grads = grad_fn(
            eqx.combine(params, static), network_state, batch
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return eqx.apply_updates(params, updates), opt_state, network_state, batch

    result["train_step_ms"] = 1e3 * timed(
        jax.jit(step, donate_argnums=(0, 1, 2)),
        params,
        # copies, so that no two leaves share a buffer to donate
        jax.tree.map(jnp.copy, optimizer.init(params)),
        jax.tree.map(jnp.copy, network_state),
        batch,
        reps=max(args.reps // 4, 1),
        rounds=args.rounds,
    )
    result["train_batch"] = args.train_batch
    stats = jax.devices()[0].memory_stats()
    if stats is not None and "peak_bytes_in_use" in stats:
        result["peak_gb"] = stats["peak_bytes_in_use"] / 1e9
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Times networks' forward and training steps."
    )
    parser.add_argument(
        "nets", nargs="+", help="networks, as train.py's --net names them"
    )
    parser.add_argument("--dtype", default="bfloat16", help="inference and train dtype")
    parser.add_argument("--remat", action="store_true", help="as train.py's --remat")
    parser.add_argument("--batch", type=int, default=1024, help="forward batch")
    parser.add_argument("--train-batch", type=int, default=4096)
    parser.add_argument("--reps", type=int, default=40, help="calls per timed round")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json", default=None, help="append each result to this file")
    args = parser.parse_args()
    print(f"devices: {jax.devices()}", flush=True)
    for net in args.nets:
        result = bench(net, args)
        print(
            f"{net}: {result['params'] / 1e6:.2f}M params, "
            f"{result['gflop_per_position']:.3f} GFLOP/position, forward "
            f"{result['forward_ms']:.2f} ms @ {args.batch}, train step "
            f"{result['train_step_ms']:.1f} ms @ {args.train_batch}"
            + (f", peak {result['peak_gb']:.1f} GB" if "peak_gb" in result else ""),
            flush=True,
        )
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
