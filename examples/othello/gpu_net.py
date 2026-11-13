"""A network on the GPU for play: JAX, jitted once for each of a few batch sizes, evaluating
`cpu_search.py`'s observations, (batch, 2, 8, 8) float32, as `Eval` takes them.

    uv run --extra cuda examples/othello/gpu_net.py --net NAME --batch 1 64 256 1024
    uv run --extra cuda examples/othello/gpu_net.py --spec resnet:10x192,... --batch 256

`--spec` times a network of random weights, for sizes not trained yet. Calls are asynchronous:
`launch` returns at once, `GpuNet.wait` waits for its result, so a caller can prepare the next
batch while the GPU computes. Only for play (see `gpu_play.py`); self-play uses `core`.
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# the batch sizes compiled: a call is padded to the least that holds it
DEFAULT_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def bucket(count: int, sizes: tuple[int, ...]) -> int:
    """The least of `sizes` holding `count`; the largest, for more (the caller splits)."""
    for size in sizes:
        if size >= count:
            return size
    return sizes[-1]


@dataclass
class Pending:
    """A call in flight: its outputs on the GPU, and how many of its rows are real."""

    outputs: Any
    count: int


class GpuNet:
    """A network's inference on the default JAX device, in the dtype it was made to compute in
    (`make_network`'s `inference_dtype`: "bfloat16", or "float32", whose convolutions use TF32
    on the GPU, as JAX's default precision does)."""

    def __init__(self, network, state, sizes: tuple[int, ...] = DEFAULT_SIZES):
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        from core.networks.utils import apply_nn

        self.sizes = tuple(sorted(sizes))
        # the weights are constants of the compiled call, cast to `dtype` once by XLA
        model = jax.device_put(eqx.nn.inference_mode(network))
        state = jax.device_put(state)

        def forward(x):
            x = jnp.moveaxis(x, 1, -1)
            (policy, value), _ = apply_nn(model, state, x)
            return policy.astype(jnp.float32), value.reshape(-1).astype(jnp.float32)

        self.forward = jax.jit(forward)
        self.jax = jax

    @classmethod
    def load(cls, name: str, dtype: str = "bfloat16", **kwargs) -> "GpuNet":
        """The league store's network `name`."""
        import equinox as eqx
        from bench_cpu_backends import network_path
        from game import load_checkpoint

        path, net = network_path(name)
        network, (params, state) = load_checkpoint(path, net, inference_dtype=dtype)
        static = eqx.filter(network, eqx.is_inexact_array, inverse=True)
        return cls(eqx.combine(params, static), state, **kwargs)

    @classmethod
    def random(cls, spec: str, dtype: str = "bfloat16", **kwargs) -> "GpuNet":
        """A network of `spec` (as train.py's --net) with random weights: for timing."""
        from game import make_network

        network, state = make_network(spec, inference_dtype=dtype)
        return cls(network, state, **kwargs)

    def warm(self) -> None:
        """Compiles every size."""
        for size in self.sizes:
            self.wait(self.launch(np.zeros((size, 2, 8, 8), np.float32)))

    def launch(self, x: np.ndarray) -> Pending:
        """Starts evaluating `x`, at most the largest size, padded to a size compiled."""
        count = len(x)
        size = bucket(count, self.sizes)
        if count > size:
            raise ValueError(f"{count} positions, more than the largest size {size}")
        if count < size:
            padded = np.zeros((size, 2, 8, 8), np.float32)
            padded[:count] = x
            x = padded
        else:
            # JAX may read the host buffer after the call returns: a copy the caller can't reuse
            x = np.array(x, np.float32, copy=True)
        return Pending(self.forward(x), count)

    def wait(self, pending: Pending) -> tuple[np.ndarray, np.ndarray]:
        """A launched call's (count, 65) policy logits and (count) values."""
        policy, value = pending.outputs
        count = pending.count
        return np.asarray(policy)[:count], np.asarray(value)[:count]

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`Eval`: evaluated at once, split into calls of the largest size."""
        largest = self.sizes[-1]
        if len(x) <= largest:
            return self.wait(self.launch(x))
        calls = [self.launch(x[i : i + largest]) for i in range(0, len(x), largest)]
        parts = [self.wait(p) for p in calls]
        return np.concatenate([p[0] for p in parts]), np.concatenate(
            [p[1] for p in parts]
        )


def throughput(
    net: GpuNet, size: int, inflight: int, seconds: float = 1.5
) -> dict[str, float]:
    """Positions a second at batch `size` with `inflight` calls queued, and a call's latency
    alone. Inputs are random boards: the time doesn't depend on them."""
    rng = np.random.default_rng(0)
    squares = rng.integers(0, 3, size=(size, 8, 8))
    x = np.stack([squares == 1, squares == 2], axis=1).astype(np.float32)
    net.wait(net.launch(x))
    # latency: one call at a time
    start, calls = time.perf_counter(), 0
    while time.perf_counter() - start < seconds / 3:
        net.wait(net.launch(x))
        calls += 1
    latency = (time.perf_counter() - start) / calls
    # throughput: `inflight` calls queued at all times
    queue = [net.launch(x) for _ in range(inflight)]
    start, calls = time.perf_counter(), 0
    while time.perf_counter() - start < seconds:
        net.wait(queue.pop(0))
        queue.append(net.launch(x))
        calls += 1
    for p in queue:
        net.wait(p)
    elapsed = time.perf_counter() - start
    return {
        "batch": size,
        "inflight": inflight,
        "latency_ms": round(1e3 * latency, 3),
        "evals_per_s_alone": round(size / latency),
        "evals_per_s": round(size * calls / elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--net", help="a network of the league's store")
    which.add_argument("--spec", help="random weights of this --net spec")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--inflight", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--results", default=None)
    args = parser.parse_args()
    sizes = tuple(sorted(set(args.batch)))
    net = (
        GpuNet.load(args.net, args.dtype, sizes=sizes)
        if args.net
        else GpuNet.random(args.spec, args.dtype, sizes=sizes)
    )
    for size in sizes:
        record = {
            "net": args.net or args.spec,
            "dtype": args.dtype,
            **throughput(net, size, args.inflight, args.seconds),
        }
        print(json.dumps(record), flush=True)
        if args.results:
            with open(args.results, "a") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
