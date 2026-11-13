"""A network on the GPU for play: JAX, compiled once for each of a few batch sizes, evaluating
`cpu_search.py`'s positions packed, two 64-bit words each (or its observations, as `Eval`).

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
    on the GPU, as JAX's default precision does). It takes positions packed, as
    `Search.collect_bits` writes them, (n, 2) uint64: 16 bytes a position to the GPU, unpacked
    there, rather than 128 floats."""

    def __init__(self, network, state, sizes: tuple[int, ...] = DEFAULT_SIZES):
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        from core.networks.utils import apply_nn

        self.sizes = tuple(sorted(sizes))
        # the weights are constants of the compiled call, cast to `dtype` once by XLA
        model = jax.device_put(eqx.nn.inference_mode(network))
        state = jax.device_put(state)

        def forward(words):
            # (n, 4) uint32, each position's two words in halves, the low first: bit i of a
            # plane is bit i % 32 of its half i // 32
            shifts = jnp.arange(32, dtype=jnp.uint32)
            bits = (words.reshape(-1, 2, 2, 1) >> shifts) & 1
            x = bits.reshape(-1, 2, 8, 8).astype(jnp.float32)
            (policy, value), _ = apply_nn(model, state, jnp.moveaxis(x, 1, -1))
            return policy.astype(jnp.float32), value.reshape(-1).astype(jnp.float32)

        jitted = jax.jit(forward)
        # compiled ahead for each size: a call skips jit's dispatch of a traced function
        self.compiled = {
            size: jitted.lower(jax.ShapeDtypeStruct((size, 4), jnp.uint32)).compile()
            for size in self.sizes
        }

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
        """Runs every size once."""
        for size in self.sizes:
            self.wait(self.launch(np.zeros((size, 2), np.uint64), size))

    def launch(self, bits: np.ndarray, count: int | None = None) -> Pending:
        """Starts evaluating the first `count` (by default all) of `bits`, packed positions,
        (n, 2) uint64, at most the largest size. The call is the least size compiled holding
        them: rows of `bits` past `count` up to it are evaluated too, whatever they hold, and
        their outputs dropped, so a caller filling a buffer of the largest size passes it whole,
        with no copy. JAX may read `bits` after this returns: the caller mustn't reuse it."""
        count = len(bits) if count is None else count
        size = bucket(count, self.sizes)
        if count > size:
            raise ValueError(f"{count} positions, more than the largest size {size}")
        if len(bits) < size:
            padded = np.zeros((size, 2), np.uint64)
            padded[:count] = bits[:count]
            bits = padded
        words = np.ascontiguousarray(bits[:size]).view(np.uint32)
        return Pending(self.compiled[size](words), count)

    def wait(self, pending: Pending) -> tuple[np.ndarray, np.ndarray]:
        """A launched call's (count, 65) policy logits and (count) values."""
        policy, value = pending.outputs
        count = pending.count
        return np.asarray(policy)[:count], np.asarray(value)[:count]

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`Eval`, of (n, 2, 8, 8) observations: packed, evaluated at once in calls of the
        largest size."""
        from cpu_search import pack

        bits = pack(x)
        largest = self.sizes[-1]
        calls = [
            self.launch(bits[i : i + largest]) for i in range(0, len(bits), largest)
        ]
        parts = [self.wait(p) for p in calls]
        if not parts:
            return np.empty((0, 65), np.float32), np.empty((0,), np.float32)
        return np.concatenate([p[0] for p in parts]), np.concatenate(
            [p[1] for p in parts]
        )


def throughput(
    net: GpuNet, size: int, inflight: int, seconds: float = 1.5
) -> dict[str, float]:
    """Positions a second at batch `size` with `inflight` calls queued, and a call's latency
    alone, each call's input sent from the host as in play. Inputs are random boards: the time
    doesn't depend on them."""
    from cpu_search import pack

    rng = np.random.default_rng(0)
    squares = rng.integers(0, 3, size=(size, 8, 8))
    x = pack(np.stack([squares == 1, squares == 2], axis=1))
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
