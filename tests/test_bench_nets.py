import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from example_modules import load_example

(bench_nets, game) = load_example("othello", "bench_nets", "game")

# the controlled runs' kind of network, tiny: channel norm normalizes each sample on its own
NET = "resnet:2x8,pool_every=2,heads=pooled,pool_channels=4,norm=channel,layout=pre"


def test_accumulated_training_step_is_the_whole_batchs():
    # with a normalization of each sample on its own, micro-batches' mean gradient is the whole
    # batch's: the step the benchmark times with --grad-accum makes the same update
    network, state = game.make_network(NET)
    params, static = eqx.partition(network, eqx.is_inexact_array)
    optimizer = game.make_optimizer(10)
    batch = bench_nets.random_batch(np.random.default_rng(0), 8)
    updated = [
        jax.jit(bench_nets.training_step(static, optimizer, n))(
            params, optimizer.init(params), state, batch
        )[0]
        for n in (1, 2)
    ]
    for other in updated[1:]:
        for a, b in zip(
            jax.tree.leaves(updated[0]), jax.tree.leaves(other), strict=True
        ):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    # and it trains: the parameters moved
    moved = [
        bool(jnp.any(a != b))
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated[0]))
    ]
    assert any(moved)


def test_bench_reports_accumulation_and_writes_a_trace(tmp_path):
    args = argparse.Namespace(
        dtype="float32",
        remat=False,
        grad_accum=2,
        trace=str(tmp_path),
        batch=4,
        train_batch=8,
        reps=4,
        rounds=1,
    )
    result = bench_nets.bench(NET, args)
    assert result["grad_accum"] == 2
    assert result["train_step_ms"] > 0 and result["forward_ms"] > 0
    for part in ("forward", "train"):
        assert list((tmp_path / part).rglob("perfetto_trace.json.gz"))
