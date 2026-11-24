"""Othello's gated scan (examples/othello/gated_scan.py): its fast form against its definition,
and in a network."""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_azresnet import make_variant, observations

from core.networks.azresnet import Depthwise, PreActConvNeXt, conv_norm, norm_maker
from core.networks.utils import BATCH_AXIS, apply_nn

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "othello"))

from gated_scan import GatedScan, scan_mixer


def gated_scan(gate: float, channels: int = 2, length: int = 3):
    """A scan along rows and columns whose gates are all `gate` (a constant gate logit), with
    random weights, and its state."""
    scan, state = eqx.nn.make_with_state(GatedScan)(
        channels,
        ((0, 1), (1, 0), (0, -1), (-1, 0)),
        length,
        1,
        key=jax.random.PRNGKey(0),
        make_norm=norm_maker(),
        stats=(1.0, 1.0),
    )
    weight = np.random.default_rng(1).uniform(0.5, 1.5, scan.weight.shape)
    logit = float(np.log(gate / (1 - gate)))
    zero = jnp.zeros_like(scan.gate_conv.weight)
    scan = eqx.tree_at(
        lambda s: (s.weight, s.gate_conv.weight, s.gate_offset),
        scan,
        (
            jnp.asarray(weight, jnp.float32),
            zero,
            jnp.full_like(scan.gate_offset, logit),
        ),
    )
    return eqx.nn.inference_mode(scan), state


def line_filter(scan: GatedScan, gate: float) -> Depthwise:
    """The depthwise filter a scan whose gates are all `gate` computes: the tap at distance k
    along a line weighs gate^(k - 1)."""
    taps, weights = [(0, 0)], [scan.weight[:, 0]]
    for d, (dy, dx) in enumerate(scan.steps):
        for k in range(1, scan.length + 1):
            taps.append((k * dy, k * dx))
            weights.append(
                scan.weight[:, 1 + d * scan.length + k - 1] * gate ** (k - 1)
            )
    dw = Depthwise(scan.weight.shape[0], tuple(taps), key=jax.random.PRNGKey(0))
    return eqx.tree_at(lambda d: d.weight, dw, jnp.stack(weights, axis=1))


@pytest.mark.parametrize("gate", [0.999999, 0.5, 1e-6])
def test_gated_lines_with_equal_gates_are_a_line_filter_fading_by_the_gate(gate):
    # open gates: the plain filter along the lines; closed ones: only the nearest square each way
    scan, state = gated_scan(gate)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 5, 5)), jnp.float32)

    y, _ = scan(x, state, jnp.dtype(jnp.float32))

    expected = line_filter(scan, gate)(x, jnp.dtype(jnp.float32))
    np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-4)


def test_a_gated_line_stops_at_a_closed_square():
    # one square's gate closed (its own input decides it): what lies beyond it along a line
    # no longer reaches the square filtered
    scan, state = gated_scan(0.999999, channels=1, length=4)
    gate_conv = eqx.tree_at(
        lambda c: c.weight, scan.gate_conv, jnp.full_like(scan.gate_conv.weight, -100.0)
    )
    scan = eqx.tree_at(lambda s: s.gate_conv, scan, gate_conv)
    x = jnp.zeros((1, 1, 5)).at[0, 0, 2].set(1.0)  # the blocking square, at column 2
    far = x.at[0, 0, 4].set(1.0)  # and something beyond it

    near, _ = scan(x, state, jnp.dtype(jnp.float32))
    both, _ = scan(far, state, jnp.dtype(jnp.float32))

    # column 0 sees column 2 (one square between, open) but not column 4 (behind column 2)
    np.testing.assert_allclose(both[0, 0, 0], near[0, 0, 0], atol=1e-6)
    # column 3 sees column 4 directly
    assert abs(float(both[0, 0, 3] - near[0, 0, 3])) > 0.1


def unrolled_scan(scan: GatedScan, x, state, dtype):
    """GatedScan's definition computed step by step, the reference for its fast form: along
    each direction, shift and multiply by each square passed."""
    channels, height, width = x.shape
    g, state = scan.gates(x, state, dtype)
    g = jnp.repeat(g, channels // g.shape[0], axis=0)
    n = scan.length * max(abs(s) for step in scan.steps for s in step)
    pad = ((0, 0), (n, n), (n, n))
    xf = x.astype(jnp.float32)
    xp, gp = jnp.pad(xf, pad), jnp.pad(g, pad)

    def shifted(a, dy, dx):
        return a[:, n + dy : n + dy + height, n + dx : n + dx + width]

    y = scan.weight[:, 0, None, None] * xf
    for d, (dy, dx) in enumerate(scan.steps):
        through = jnp.ones_like(xf)
        for k in range(1, scan.length + 1):
            tap = scan.weight[:, 1 + d * scan.length + k - 1, None, None]
            y = y + tap * through * shifted(xp, k * dy, k * dx)
            if k < scan.length:
                through = through * shifted(gp, k * dy, k * dx)
    return y.astype(dtype), state


EIGHT_WAYS = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)


@pytest.mark.parametrize(
    "steps, length, board, offset",
    [
        (EIGHT_WAYS, 7, (8, 8), 2.0),
        # nearly closed gates, and so closed that their products underflow
        (EIGHT_WAYS, 7, (8, 8), -4.0),
        (EIGHT_WAYS, 7, (8, 8), -60.0),
        # a grid that isn't square, lines longer than it, one way only
        (((0, 1), (1, 1)), 6, (3, 5), 0.0),
        # steps that skip squares, and two that reach one square by different paths
        (((1, 2), (0, 1), (0, 2)), 4, (6, 7), 1.0),
        # no square between any two: nothing is gated
        (((0, 1), (-1, 0)), 1, (4, 4), 2.0),
    ],
)
def test_a_gated_scan_is_its_unrolled_definition_with_its_gradients(
    steps, length, board, offset
):
    rng = np.random.default_rng(0)
    scan, state = eqx.nn.make_with_state(GatedScan)(
        6,
        steps,
        length,
        3,
        key=jax.random.PRNGKey(0),
        make_norm=norm_maker(),
        stats=(1.0, 1.0),
    )
    scan = eqx.tree_at(lambda s: s.gate_offset, scan, jnp.full((3,), offset))
    params, static = eqx.partition(scan, eqx.is_inexact_array)
    x = jnp.asarray(rng.normal(size=(3, 6, *board)), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=x.shape), jnp.float32)

    def outputs_and_gradients(call):
        # batched and training, as the trainer runs it
        def batched(params, x):
            scan = eqx.combine(params, static)
            fresh = jax.tree.map(lambda a: a, state)  # a state is used once
            return jax.vmap(
                lambda x: call(scan, x, fresh, jnp.dtype(jnp.float32))[0],
                axis_name=BATCH_AXIS,
            )(x)

        @jax.jit
        def run(params, x):
            y, pull = jax.vjp(batched, params, x)
            return y, pull(cotangent), batched(params, x)  # also undifferentiated

        return jax.tree.leaves(run(params, x))

    actual = outputs_and_gradients(GatedScan.__call__)
    expected = outputs_and_gradients(unrolled_scan)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_a_gated_scans_matrix_products_ask_for_full_float32():
    # a GPU computes float32 matrix products with 10-bit mantissas unless asked not to, which
    # tests on the CPU can't see: the scan was 1e-3 off there
    scan, state = gated_scan(0.5)
    params, static = eqx.partition(scan, eqx.is_inexact_array)
    x = jnp.ones((2, 5, 5))

    def forward_and_backward(params, x):
        def apply(params, x):
            fresh = jax.tree.map(lambda a: a, state)
            scan = eqx.combine(params, static)
            return scan(x, fresh, jnp.dtype(jnp.float32))[0]

        y, pull = jax.vjp(apply, params, x)
        return pull(y)

    products = [
        eqn.params["precision"]
        for eqn in jax.make_jaxpr(forward_and_backward)(params, x).eqns
        if eqn.primitive.name == "dot_general"
    ]

    assert (
        len(products) >= 3
    )  # the paths' gates, their transpose, and the taps' weights
    highest = jax.lax.Precision.HIGHEST
    assert all(p is not None and set(p) == {highest} for p in products)


# a scan along rows and columns of a test network's 3 x 3 inputs, 2 gate channels of its 6
SCANNED = {"mixer": scan_mixer(((0, 1), (1, 0), (0, -1), (-1, 0)), length=2, gates=2)}


def test_a_scanned_network_infers_as_it_trained_on_its_batch():
    # the gates' BatchNorm too: after one training step, inference on the same batch gives the
    # outputs training did
    net, state = make_variant("normalized convnext", **SCANNED)
    assert all(
        isinstance(b, PreActConvNeXt) and isinstance(b.dw, GatedScan)
        for b in net.blocks
    )
    x = observations(np.random.default_rng(0), batch_size=8)

    @eqx.filter_jit
    def run(net, state, x):
        training, state = apply_nn(net, state, x)
        inference, _ = apply_nn(eqx.nn.inference_mode(net), state, x)
        return training, inference

    training, inference = run(net, state, x)

    for a, b in zip(inference, training, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)


def test_a_scanned_trunk_starts_as_the_stem():
    net, state = make_variant("normalized convnext", norm="channel", **SCANNED)
    x = jnp.moveaxis(observations(np.random.default_rng(0), 1)[0], -1, 0)
    dtype = jnp.dtype("float32")
    assert net.stem_norm is not None  # a ConvNeXt trunk's default

    trunk = stem = conv_norm(net.stem_conv, net.stem_norm, x, state, dtype)[0]
    for block in net.blocks:
        trunk, _ = block(trunk, state, dtype)

    np.testing.assert_allclose(trunk, stem, atol=1e-6)
