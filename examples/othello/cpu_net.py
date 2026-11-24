"""An Othello network's forward pass for CPU play, int8 in the trunk and float elsewhere, in C++
(`cpu_net.h`), built from a trained network: its layers as plain arrays (`extract`), quantized after
training on calibration positions (`quantize`), and saved, so that play needs neither JAX nor
OpenVINO.

    uv run examples/othello/cpu_net.py --net 8x128-128s-8p-w1125-274m --positions LEAVES.npy
    uv run examples/othello/cpu_net.py --checkpoint RUN/avg4/210.eqx --spec SPEC --positions LEAVES.npy

A checkpoint outside the league is saved under the name evaluate.py's checkpoint= player looks for
(its directory and file names, avg4-210), or --name, as its net= gives it.

Each convolution of the trunk takes 8-bit inputs, scaled for the whole tensor by the minimum and
maximum seen in calibration (with a zero point when the minimum is below 0), and 8-bit weights
scaled for each output channel, as `quantize.py`'s best OpenVINO model does; and so do the heads'
first convolutions. The rest of the heads stays in float. The stem's weights take 16 bits, as its
sums are looked up. Each convolution's biases are then corrected for the mean shift quantization
leaves in its outputs, and so is the value's last bias (`bias_corrected`).
A normalization of each square's channels runs in the pass that quantizes the next convolution's
input, where OpenVINO runs it as several ops of its own.

`float_forward` is the network in NumPy, which `quantize` calibrates on; `int8_forward` is what the
C++ computes, in NumPy, for tests.
"""

import argparse
import ctypes
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NORM_NONE, NORM_AFFINE, NORM_CHANNEL = 0, 1, 2


@dataclass
class Conv:
    w: np.ndarray  # (out, in, k, k)
    b: np.ndarray  # (out)


@dataclass
class Norm:
    type: int = NORM_NONE
    a: np.ndarray | None = None
    c: np.ndarray | None = None
    eps: float = 1e-5


@dataclass
class Linear:
    w: np.ndarray  # (out, in)
    b: np.ndarray


@dataclass
class Block:
    conv1: Conv
    norm1: Norm
    conv2: Conv
    norm2: Norm
    pool_channels: int = 0
    pool: Linear | None = None


# the gated scan's directions in the C++'s order (`SCAN_DY`, `SCAN_DX` in cpu_net.h)
SCAN_DIRECTIONS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


@dataclass
class Scan:
    """A gated scan in place of a ConvNeXt block's depthwise filter (see
    `core.networks.azresnet.GatedScan`), its BatchNorm folded in: y = dw_bias + w[:, 0] x + the
    taps along each direction of `SCAN_DIRECTIONS`, from 1 step to `length`, each weighted by the
    gates of the squares between; the gates sigmoid(gate_w x + gate_b), each shared by a run of
    channels / gates channels."""

    length: int
    w: np.ndarray  # (channels, 1 + 8 length)
    gate_w: np.ndarray  # (gates, channels)
    gate_b: np.ndarray  # (gates)


@dataclass
class ConvNeXtBlock:
    """ConvNeXt's block (see `core.networks.azresnet.PreActConvNeXt`): x + project(relu(expand(y) +
    pool)), y = dw(x) + dw_bias, a depthwise filter of the trunk with its BatchNorm folded in;
    `pool` biases the hidden channels from the last `pool_channels` of y. BatchNorms after expand
    and project fold into them. The depthwise filter stays in float."""

    taps: np.ndarray  # (taps, 2): each tap's (dy, dx), in order of dy, then dx
    dw: np.ndarray  # (channels, taps)
    dw_bias: np.ndarray  # (channels)
    expand: Conv
    project: Conv
    pool_channels: int = 0
    pool: Linear | None = None
    scan: Scan | None = None  # in place of the taps


@dataclass
class Program:
    """A network's layers as arrays, its normalizations folded into its convolutions where they
    can be. In the post-activation layout (`pre` False) a block's norms follow its convolutions; in
    the pre-activation layout they precede them."""

    channels: int
    pre: bool
    head_channels: int
    stem: Conv
    stem_norm: Norm
    blocks: list[Block | ConvNeXtBlock]
    final_norm: Norm
    # the heads: (channels, 3 * head_channels) of policy, its pooled part, and value
    head: Conv
    policy_norm: Norm
    pool_norm: Norm
    value_norm: Norm
    pool_linear: Linear
    pass_linear: Linear
    value_hidden: Linear
    value_out: Linear
    square_w: np.ndarray  # (head_channels)
    # a value head over the final scores (see `AZResnetConfig.outcome_scores`): each score's value,
    # its sign, the value their mean under value_out's softmax. None: value_out's one output, tanh'd
    value_weights: np.ndarray | None = None
    # how each convolution's input is quantized, by `quantize`: (scale, zero point), by name
    inputs: dict[str, tuple[float, int]] = field(default_factory=dict)


def f32(x) -> np.ndarray:
    return np.asarray(x, np.float32)


def extract(network, state) -> Program:
    """The layers of `network`, an `AZResnet` of basic blocks and pooled heads in inference mode, with
    its `state` (BatchNorm's)."""
    import equinox as eqx

    from core.networks.azresnet import (
        FixedScale,
        GatedScan,
        GroupNorm,
        PooledHeads,
        PreActBlock,
        PreActConvNeXt,
        ResidualBlock,
        inference_affine,
    )

    def unsupported(norm):
        return TypeError(f"no native normalization for {norm!r}")

    def conv_norm(conv, norm) -> tuple[Conv, Norm]:
        """A convolution and the normalization after it, folded into it where it can be."""
        w = f32(conv.weight)
        zeros = np.zeros(w.shape[0], np.float32)
        if isinstance(norm, eqx.nn.BatchNorm):
            scale, offset = inference_affine(norm, state)
            return Conv(w * f32(scale)[:, None, None, None], f32(offset)), Norm()
        if isinstance(norm, FixedScale):
            return Conv(f32(norm.conv_weight(conv.weight)), f32(norm.bias)), Norm()
        if isinstance(norm, GroupNorm) and not norm.over_board and norm.center:
            return Conv(w, zeros), Norm(
                NORM_CHANNEL, f32(norm.weight), f32(norm.bias), norm.eps
            )
        raise unsupported(norm)

    def before(norm) -> Norm:
        """A normalization before a convolution (the pre-activation layout's)."""
        if isinstance(norm, eqx.nn.BatchNorm):
            scale, offset = inference_affine(norm, state)
            return Norm(NORM_AFFINE, f32(scale), f32(offset))
        if isinstance(norm, GroupNorm) and not norm.over_board and norm.center:
            return Norm(NORM_CHANNEL, f32(norm.weight), f32(norm.bias), norm.eps)
        raise unsupported(norm)

    def linear(layer) -> Linear:
        return Linear(f32(layer.weight), f32(layer.bias))

    def folded(conv, norm) -> Conv:
        """A convolution with the normalization after it, if any, folded in: one that must fold."""
        if norm is None:
            return Conv(f32(conv.weight), np.zeros(conv.out_channels, np.float32))
        c, rest = conv_norm(conv, norm)
        if rest.type != NORM_NONE:
            raise unsupported(norm)
        return c

    def scan(dw, scale) -> Scan:
        """A gated scan, the BatchNorm after it of `scale`, its directions in the C++'s order."""
        if sorted(dw.steps) != sorted(SCAN_DIRECTIONS):
            raise TypeError(f"no native gated scan along {dw.steps}")
        n = dw.length
        w = f32(dw.weight)
        order = [dw.steps.index(d) for d in SCAN_DIRECTIONS]
        taps = [0] + [1 + d * n + k for d in order for k in range(n)]
        gate_conv, _ = conv_norm(dw.gate_conv, dw.gate_norm)
        return Scan(
            length=n,
            w=w[:, taps] * f32(scale)[:, None],
            gate_w=gate_conv.w[:, :, 0, 0],
            gate_b=gate_conv.b + f32(dw.gate_offset),
        )

    def convnext(block, pool: Linear | None) -> ConvNeXtBlock:
        gated = isinstance(block.dw, GatedScan)
        if not (
            block.plain
            or (gated and block.position is None and block.mlp_router is None)
        ):
            raise TypeError(
                "no native ConvNeXt block but the plain one (a depthwise filter) or a gated scan"
            )
        if not isinstance(block.norm, eqx.nn.BatchNorm):
            raise unsupported(block.norm)
        scale, offset = inference_affine(block.norm, state)
        if gated:
            return ConvNeXtBlock(
                taps=np.zeros((0, 2), np.int32),
                dw=np.zeros((len(offset), 0), np.float32),
                dw_bias=f32(offset),
                expand=folded(block.expand, block.expand_norm),
                project=folded(block.project, block.project_norm),
                pool_channels=block.pool_channels,
                pool=pool,
                scan=scan(block.dw, scale),
            )
        order = sorted(range(len(block.dw.taps)), key=lambda t: block.dw.taps[t])
        return ConvNeXtBlock(
            taps=np.array([block.dw.taps[t] for t in order], np.int32).reshape(-1, 2),
            dw=f32(block.dw.weight)[:, order] * f32(scale)[:, None],
            dw_bias=f32(offset),
            expand=folded(block.expand, block.expand_norm),
            project=folded(block.project, block.project_norm),
            pool_channels=block.pool_channels,
            pool=pool,
        )

    if network.input_features is not None or network.board_plane:
        raise TypeError("no native input planes but the discs")
    pre = network.final_norm is not None
    if pre:
        # the pre-activation trunk starts from the stem as it is, or normalized (folded in)
        stem, stem_norm = folded(network.stem_conv, network.stem_norm), Norm()
    else:
        stem, stem_norm = conv_norm(network.stem_conv, network.stem_norm)
    blocks: list[Block | ConvNeXtBlock] = []
    for block in network.blocks:
        pool = linear(block.pool_linear) if block.pool_linear is not None else None
        if isinstance(block, PreActConvNeXt):
            blocks.append(convnext(block, pool))
        elif isinstance(block, PreActBlock):
            zeros = np.zeros(block.conv1.out_channels, np.float32)
            blocks.append(
                Block(
                    Conv(f32(block.conv1.weight), zeros),
                    before(block.norm1),
                    Conv(f32(block.conv2.weight), zeros),
                    before(block.norm2),
                    block.pool_channels,
                    pool,
                )
            )
        elif isinstance(block, ResidualBlock):
            conv1, norm1 = conv_norm(block.conv1, block.norm1)
            conv2, norm2 = conv_norm(block.conv2, block.norm2)
            blocks.append(Block(conv1, norm1, conv2, norm2, block.pool_channels, pool))
        else:
            raise TypeError(f"no native block {type(block).__name__}")
    heads = network.heads
    if not isinstance(heads, PooledHeads):
        raise TypeError("only pooled heads are native")
    parts = [
        conv_norm(heads.policy_conv, heads.policy_norm),
        conv_norm(heads.policy_pool_conv, heads.policy_pool_norm),
        conv_norm(heads.value_conv, heads.value_norm),
    ]
    head = Conv(
        np.concatenate([p[0].w for p in parts]), np.concatenate([p[0].b for p in parts])
    )
    return Program(
        channels=network.stem_conv.out_channels,
        pre=pre,
        head_channels=heads.policy_conv.out_channels,
        stem=stem,
        stem_norm=stem_norm,
        blocks=blocks,
        final_norm=before(network.final_norm) if pre else Norm(),
        head=head,
        policy_norm=parts[0][1],
        pool_norm=parts[1][1],
        value_norm=parts[2][1],
        pool_linear=linear(heads.policy_pool_linear),
        pass_linear=linear(heads.pass_linear),
        value_hidden=linear(heads.value_hidden),
        value_out=linear(heads.value_linear),
        square_w=f32(heads.policy_square_conv.weight).reshape(-1),
        value_weights=None
        if network.outcome_scores is None
        else np.sign(f32(network.outcome_scores)),
    )


def extract_checkpoint(path: str, net: str) -> Program:
    import equinox as eqx
    from game import load_checkpoint

    network, (params, state) = load_checkpoint(path, net)
    static = eqx.filter(network, eqx.is_inexact_array, inverse=True)
    return extract(eqx.nn.inference_mode(eqx.combine(params, static)), state)


# NumPy: features are (n, 8, 8, channels)


def conv2d(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """A 'same' convolution of (n, 8, 8, in) features by (out, in, k, k) weights, in float32."""
    k = w.shape[-1]
    if k == 1:
        return np.einsum("nyxi,oi->nyxo", x, w[:, :, 0, 0])
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    out = np.zeros(x.shape[:3] + (w.shape[0],), np.float32)
    for dy in range(3):
        for dx in range(3):
            out += padded[:, dy : dy + 8, dx : dx + 8] @ w[:, :, dy, dx].T
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return f32(1 / (1 + np.exp(-x)))


def gated_scan(b: ConvNeXtBlock, x: np.ndarray) -> np.ndarray:
    """`b`'s gated scan of (n, 8, 8, channels) features (see `Scan`)."""
    assert b.scan is not None
    sc, n = b.scan, b.scan.length
    gates = sigmoid(x @ sc.gate_w.T + sc.gate_b)
    gates = np.repeat(gates, x.shape[-1] // gates.shape[-1], axis=-1)
    pad = ((0, 0), (n, n), (n, n), (0, 0))
    xp, gp = np.pad(x, pad), np.pad(gates, pad)

    def shifted(a, dy, dx):
        return a[:, n + dy : n + dy + 8, n + dx : n + dx + 8]

    out = b.dw_bias + sc.w[:, 0] * x
    for d, (dy, dx) in enumerate(SCAN_DIRECTIONS):
        through = np.ones_like(x)
        for k in range(1, n + 1):
            out = out + sc.w[:, 1 + d * n + k - 1] * through * shifted(
                xp, k * dy, k * dx
            )
            through = through * shifted(gp, k * dy, k * dx)
    return f32(out)


def depthwise(b: ConvNeXtBlock, x: np.ndarray) -> np.ndarray:
    """`b`'s depthwise filter of (n, 8, 8, channels) features, its bias added first, then each tap's
    product in order, taps off the board adding nothing (or its gated scan)."""
    if b.scan is not None:
        return gated_scan(b, x)
    out = np.broadcast_to(b.dw_bias, x.shape).astype(np.float32)
    padded = np.pad(x, ((0, 0), (7, 7), (7, 7), (0, 0)))
    for t, (dy, dx) in enumerate(b.taps):
        out = out + padded[:, 7 + dy : 15 + dy, 7 + dx : 15 + dx] * b.dw[:, t]
    return out


def normalize(norm: Norm, x: np.ndarray, relu: bool = False) -> np.ndarray:
    if norm.type == NORM_AFFINE:
        x = x * norm.a + norm.c
    elif norm.type == NORM_CHANNEL:
        mean = x.mean(-1, keepdims=True)
        var = np.square(x - mean).mean(-1, keepdims=True)
        x = (x - mean) / np.sqrt(var + np.float32(norm.eps)) * norm.a + norm.c
    return np.maximum(x, 0) if relu else x


def board_pool(x: np.ndarray) -> np.ndarray:
    """(n, 8, 8, c) to (n, 2c): each channel's mean and maximum over the board."""
    return np.concatenate([x.mean((1, 2)), x.max((1, 2))], -1)


def dense(layer: Linear, x: np.ndarray) -> np.ndarray:
    return x @ layer.w.T + layer.b


def forward(
    p: Program, observations: np.ndarray, conv, record=None
) -> tuple[np.ndarray, np.ndarray]:
    """The network on (n, 2, 8, 8) observations, its trunk's convolutions by `conv(name, conv, x)`;
    `record(name, x)` sees each one's input, and value_out's output ("value", before its tanh or
    softmax)."""
    record = record or (lambda name, x: None)
    x = np.moveaxis(f32(observations), 1, -1)

    def apply(name, c, x):
        record(name, x)
        return conv(name, c, x)

    x = apply("stem", p.stem, x)
    if not p.pre:
        x = normalize(p.stem_norm, x, relu=True)
    for i, b in enumerate(p.blocks):
        if isinstance(b, ConvNeXtBlock):
            y = depthwise(b, x)
            h = apply(f"{i}.expand", b.expand, y)
            if b.pool_channels:
                assert b.pool is not None
                pooled = board_pool(y[..., -b.pool_channels :])
                h = h + dense(b.pool, pooled)[:, None, None, :]
            x = x + apply(f"{i}.project", b.project, np.maximum(h, 0))
            continue
        kept = p.channels - b.pool_channels
        if p.pre:
            y = apply(f"{i}.conv1", b.conv1, normalize(b.norm1, x, relu=True))
            y = normalize(b.norm2, y, relu=True)
        else:
            y = normalize(b.norm1, apply(f"{i}.conv1", b.conv1, x))
        if b.pool_channels:
            assert b.pool is not None
            pooled = y[..., kept:] if p.pre else np.maximum(y[..., kept:], 0)
            y = y[..., :kept] + dense(b.pool, board_pool(pooled))[:, None, None, :]
        if not p.pre:
            y = np.maximum(y, 0)
        y = apply(f"{i}.conv2", b.conv2, y)
        x = x + y if p.pre else np.maximum(x + normalize(b.norm2, y), 0)
    if p.pre:
        x = normalize(p.final_norm, x, relu=True)
    h = p.head_channels
    heads = apply("head", p.head, x)
    policy = normalize(p.policy_norm, heads[..., :h])
    pooled = board_pool(normalize(p.pool_norm, heads[..., h : 2 * h], relu=True))
    policy = np.maximum(policy + dense(p.pool_linear, pooled)[:, None, None, :], 0)
    squares = (policy @ p.square_w).reshape(len(x), 64)
    logits = np.concatenate([squares, dense(p.pass_linear, pooled)], -1)
    value = board_pool(normalize(p.value_norm, heads[..., 2 * h :], relu=True))
    value = dense(p.value_out, np.maximum(dense(p.value_hidden, value), 0))
    record("value", value)
    return f32(logits), value_of(p, value)


def value_of(p: Program, out: np.ndarray) -> np.ndarray:
    """(n) values from value_out's (n, outputs): its one output through tanh, or the scores'
    values' mean under its softmax (see `Program.value_weights`)."""
    if p.value_weights is None:
        return f32(np.tanh(out[:, 0]))
    e = np.exp(out - out.max(-1, keepdims=True))
    return f32((e @ p.value_weights) / e.sum(-1))


def float_forward(p: Program, observations: np.ndarray, record=None):
    return forward(p, observations, lambda _, c, x: conv2d(x, c.w) + c.b, record)


def input_ranges(
    p: Program, positions: np.ndarray, chunk: int = 512
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each convolution's input's minimum and maximum, channel by channel, over `positions`, by
    name."""
    ranges: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def record(name, x):
        flat = x.reshape(-1, x.shape[-1])
        low, high = flat.min(0), flat.max(0)
        if name in ranges:
            low, high = (
                np.minimum(ranges[name][0], low),
                np.maximum(ranges[name][1], high),
            )
        ranges[name] = (low, high)

    for i in range(0, len(positions), chunk):
        float_forward(p, positions[i : i + chunk], record)
    return ranges


def weights_int(w: np.ndarray, levels: int = 127) -> tuple[np.ndarray, np.ndarray]:
    """(out, ...) float weights as signed integers from -`levels` to `levels`, symmetric (8 bits
    for 127, else 16), and each output channel's scale."""
    flat = np.abs(w.reshape(len(w), -1)).max(1)
    scale = np.where(flat > 0, flat / levels, 1).astype(np.float32)
    q = np.clip(np.rint(w / scale.reshape(-1, *([1] * (w.ndim - 1)))), -levels, levels)
    return q.astype(np.int8 if levels <= 127 else np.int16), scale


@dataclass
class QuantizedConv:
    """A convolution as the C++ takes it: int8 weights (the stem's int16), (cout, cin, k, k), cin a
    multiple of 4; what its sums are multiplied by and added to; and how its input is quantized:
    round(x * inverse) + zero_point, `inverse` (cin) a scale for each channel."""

    q: np.ndarray
    scale: np.ndarray
    bias: np.ndarray
    inverse: np.ndarray
    zero_point: int


# how far each input channel's scale follows its own range (SmoothQuant's alpha): its range ** ALPHA
# over its weights' ** (1 - ALPHA), folded into the weights. On 8x128 nets it cut the value's error
# to the float network 1.6-3.5x against one scale for the whole input (0.75 best of 0.5, 0.75, 1)
ALPHA = 0.75


def quantize_conv(
    c: Conv,
    low: np.ndarray,
    high: np.ndarray,
    alpha: float = ALPHA,
) -> QuantizedConv:
    """`c`, its input's channels ranging from `low` to `high`: each channel divided by a scale of
    its own (SmoothQuant's, `alpha`), folded into the weights, and then the whole input by one scale
    and zero point."""
    w = c.w
    reach = np.maximum(np.maximum(-low, high), 1e-3)
    weights = np.maximum(np.abs(w).max((0, 2, 3)), 1e-6)
    channel = reach**alpha / weights ** (1 - alpha)
    channel = f32(channel / channel.max())
    w = w * channel[None, :, None, None]
    low_all = min(float((low / channel).min()), 0.0)
    high_all = max(float((high / channel).max()), 0.0)
    in_scale = np.float32(max(high_all - low_all, 1e-6) / 255)
    zero_point = int(np.clip(np.rint(-low_all / in_scale), 0, 255))
    inverse = f32(1 / (channel * in_scale))
    if w.shape[1] % 4:
        pad = -w.shape[1] % 4
        w = np.pad(w, ((0, 0), (0, pad), (0, 0), (0, 0)))
        inverse = np.pad(inverse, (0, pad), constant_values=1)
    q, w_scale = weights_int(w)
    scale = f32(in_scale * w_scale)
    # the zero point's share of each sum, taken back out
    sums = q.reshape(len(q), -1).astype(np.int64).sum(1)
    bias = f32(c.b - scale * zero_point * sums)
    return QuantizedConv(q, scale, bias, inverse, zero_point)


# the stem's weights' largest magnitude: its input is the discs, 0 or 1, and its sums are looked up from
# a table of 16-bit integers (`stem_table` in cpu_net.h), up to 18 weights a sum, so they take this
# many steps a side where the other convolutions take 127. On one normalized ray 9x64, 8 bits' rounding
# in the stem alone shifted the value by about 0.012 on average
STEM_LEVELS = 32767 // 18


def quantize_stem(c: Conv) -> QuantizedConv:
    """The stem, `c`: its input 0 or 1, which the C++ takes as it is, and 16-bit weights."""
    w = np.pad(c.w, ((0, 0), (0, -c.w.shape[1] % 4), (0, 0), (0, 0)))
    q, scale = weights_int(w, STEM_LEVELS)
    return QuantizedConv(q, scale, c.b.copy(), np.ones(w.shape[1], np.float32), 0)


def pointwise(qc: QuantizedConv, x: np.ndarray) -> np.ndarray:
    """A quantized 1x1 convolution of (n, cin) inputs, as the C++ computes it."""
    if x.shape[-1] % 4:
        x = np.pad(x, ((0, 0), (0, -x.shape[-1] % 4)))
    u = np.clip(np.rint(x * qc.inverse) + qc.zero_point, 0, 255)
    return f32(u @ qc.q[:, :, 0, 0].T.astype(np.float32)) * qc.scale + qc.bias


# what `searched_conv` tries: SmoothQuant's alpha, and the share of each channel's calibration range
# that the quantization covers (beyond it, values saturate)
ALPHAS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
CLIPS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def searched_conv(
    c: Conv, low: np.ndarray, high: np.ndarray, x: np.ndarray
) -> QuantizedConv:
    """`c`, a 1x1 convolution, quantized with the alpha and clip (`ALPHAS`, `CLIPS`) whose outputs on
    `x`, (n, cin) samples of its input, are nearest the float convolution's (least squared error).
    A ConvNeXt block's expand takes the depthwise filter's output, signed, with channels' ranges
    many times apart and set by rare outliers, which no one setting suited: the value's error on
    two ray-filter nets was least at alpha 1.0 for one and 0.9 for the other."""
    want = x @ c.w[:, :, 0, 0].T + c.b
    best, best_error = None, np.inf
    for alpha in ALPHAS:
        for clip in CLIPS:
            qc = quantize_conv(c, low * clip, high * clip, alpha=alpha)
            error = float(np.square(pointwise(qc, x) - want).mean())
            if error < best_error:
                best, best_error = qc, error
    assert best is not None
    return best


def layer_inputs(
    p: Program, positions: np.ndarray, names: set[str], rows: int = 16384, seed: int = 0
) -> dict[str, np.ndarray]:
    """Up to `rows` squares' inputs, (rows, cin), to each convolution of `names`, from `positions`."""
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.permutation(len(positions) * 64)[:rows])
    seen: dict[str, np.ndarray] = {}

    def record(name, x):
        if name in names:
            seen[name] = x.reshape(-1, x.shape[-1])[pick]

    float_forward(p, positions, record)
    return seen


def quantize(
    p: Program, positions: np.ndarray
) -> tuple[Program, dict[str, QuantizedConv]]:
    """The program as the C++ should run it (`bias_corrected`) and its convolutions quantized, by
    name, their inputs' ranges from `positions`, (n, 2, 8, 8) observations (best the leaves of the
    network's own searches). A ConvNeXt block's 1x1s search their quantization's settings on those
    inputs (`searched_conv`)."""
    ranges = input_ranges(p, positions)
    convs = {"stem": p.stem, "head": p.head}
    for i, b in enumerate(p.blocks):
        if isinstance(b, ConvNeXtBlock):
            convs[f"{i}.expand"], convs[f"{i}.project"] = b.expand, b.project
        else:
            convs[f"{i}.conv1"], convs[f"{i}.conv2"] = b.conv1, b.conv2
    searched = {
        f"{i}.{part}"
        for i, b in enumerate(p.blocks)
        if isinstance(b, ConvNeXtBlock)
        for part in ("expand", "project")
    }
    samples = layer_inputs(p, positions[:256], searched) if searched else {}
    quantized = {
        name: searched_conv(c, *ranges[name], samples[name])
        if name in searched
        else quantize_conv(c, *ranges[name])
        for name, c in convs.items()
        if name != "stem"
    }
    quantized["stem"] = quantize_stem(p.stem)
    return bias_corrected(p, quantized, positions)


def int8_conv(qc: QuantizedConv, x: np.ndarray) -> np.ndarray:
    """A quantized convolution of (n, 8, 8, cin) features, as the C++ computes it (its integer sums
    exact in float64)."""
    if x.shape[-1] % 4:
        x = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, -x.shape[-1] % 4)))
    u = np.clip(np.rint(x * qc.inverse) + qc.zero_point, 0, 255)
    k = qc.q.shape[-1]
    edge = k // 2
    padded = np.pad(
        u.astype(np.float64),
        ((0, 0), (edge, edge), (edge, edge), (0, 0)),
        constant_values=qc.zero_point,
    )
    acc = np.zeros(x.shape[:3] + (len(qc.q),), np.float64)
    w = qc.q.astype(np.float64)
    for dy in range(k):
        for dx in range(k):
            acc += padded[:, dy : dy + 8, dx : dx + 8] @ w[:, :, dy, dx].T
    return f32(acc) * qc.scale + qc.bias


def bias_corrected(
    p: Program, q: dict[str, QuantizedConv], positions: np.ndarray
) -> tuple[Program, dict[str, QuantizedConv]]:
    """Moves each convolution's biases, in order through the network, by how far the mean of each
    of its outputs over `positions` lies from the float network's, with the convolutions before it
    already quantized and corrected; then the value's last biases by how far their means before
    tanh or softmax lie (the bias correction of Nagel et al. 2019, measured on calibration positions
    rather than derived from BatchNorm's statistics).

    Rounding leaves errors that don't average out, and they seem to add up through the trunk to a
    shift of the whole value: on three ray 9x64 nets' test leaves, mean errors of 0.018-0.028
    carried shifts of -0.023 to +0.004, which this left within 0.002 (0.004 on the engine-labelled
    set) and the mean errors 0.014-0.016."""
    means: dict[str, np.ndarray] = {}

    def float_conv(name, c, x):
        out = conv2d(x, c.w) + c.b
        means[name] = out.reshape(-1, out.shape[-1]).mean(0, dtype=np.float64)
        return out

    def record_float(name, x):
        if name == "value":
            means[name] = x.mean(0, dtype=np.float64)

    forward(p, positions, float_conv, record_float)
    corrected: dict[str, QuantizedConv] = {}
    shift = np.zeros(len(p.value_out.b))

    def conv(name, _c, x):
        qc = q[name]
        out = int8_conv(qc, x)
        delta = f32(
            out.reshape(-1, out.shape[-1]).mean(0, dtype=np.float64) - means[name]
        )
        corrected[name] = dataclasses.replace(qc, bias=f32(qc.bias - delta))
        return out - delta

    def record(name, x):
        if name == "value":
            shift[:] = x.mean(0, dtype=np.float64) - means[name]

    forward(p, positions, conv, record)
    value_out = Linear(p.value_out.w, f32(p.value_out.b - shift))
    return dataclasses.replace(p, value_out=value_out), corrected


def int8_forward(p: Program, q: dict[str, QuantizedConv], observations: np.ndarray):
    """What the C++ computes, in NumPy."""
    return forward(p, observations, lambda name, _c, x: int8_conv(q[name], x))


def save(path: Path, p: Program, q: dict[str, QuantizedConv]) -> None:
    """The program and its quantized convolutions, in one .npz."""
    arrays: dict[str, np.ndarray] = {
        "meta": np.array([p.channels, int(p.pre), p.head_channels, len(p.blocks)]),
        "square_w": p.square_w,
    }
    if p.value_weights is not None:
        arrays["value_weights"] = p.value_weights

    def put(name, value):
        if isinstance(value, Conv):
            arrays[f"{name}.w"], arrays[f"{name}.b"] = value.w, value.b
        elif isinstance(value, Norm):
            arrays[f"{name}.norm"] = np.array([value.type, value.eps], np.float64)
            if value.a is not None and value.c is not None:
                arrays[f"{name}.a"], arrays[f"{name}.c"] = value.a, value.c
        elif isinstance(value, Linear):
            arrays[f"{name}.w"], arrays[f"{name}.b"] = value.w, value.b

    for name in (
        "stem",
        "stem_norm",
        "final_norm",
        "head",
        "policy_norm",
        "pool_norm",
        "value_norm",
        "pool_linear",
        "pass_linear",
        "value_hidden",
        "value_out",
    ):
        put(name, getattr(p, name))
    for i, b in enumerate(p.blocks):
        arrays[f"b{i}.pool_channels"] = np.array(b.pool_channels)
        if isinstance(b, ConvNeXtBlock):
            arrays[f"b{i}.taps"], arrays[f"b{i}.dw"] = b.taps, b.dw
            arrays[f"b{i}.dw_bias"] = b.dw_bias
            if b.scan is not None:
                arrays[f"b{i}.scan_length"] = np.array(b.scan.length)
                arrays[f"b{i}.scan_w"] = b.scan.w
                arrays[f"b{i}.scan_gate_w"] = b.scan.gate_w
                arrays[f"b{i}.scan_gate_b"] = b.scan.gate_b
            for part in ("expand", "project", "pool"):
                if getattr(b, part) is not None:
                    put(f"b{i}.{part}", getattr(b, part))
            continue
        for part in ("conv1", "norm1", "conv2", "norm2", "pool"):
            if getattr(b, part) is not None:
                put(f"b{i}.{part}", getattr(b, part))
    for name, qc in q.items():
        arrays[f"q.{name}.q"] = qc.q
        arrays[f"q.{name}.scale"] = qc.scale
        arrays[f"q.{name}.bias"] = qc.bias
        arrays[f"q.{name}.inverse"] = qc.inverse
        arrays[f"q.{name}.zero_point"] = np.array(qc.zero_point)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial.npz")
    np.savez(partial, allow_pickle=False, **arrays)
    partial.replace(path)


def load(path: Path) -> tuple[Program, dict[str, QuantizedConv]]:
    a = np.load(path)
    channels, pre, head_channels, blocks = (int(v) for v in a["meta"])

    def conv(name):
        return Conv(a[f"{name}.w"], a[f"{name}.b"])

    def norm(name):
        kind, eps = a[f"{name}.norm"]
        if int(kind) == NORM_NONE:
            return Norm(eps=float(eps))
        return Norm(int(kind), a[f"{name}.a"], a[f"{name}.c"], float(eps))

    def linear(name):
        return Linear(a[f"{name}.w"], a[f"{name}.b"])

    def block(i) -> Block | ConvNeXtBlock:
        pool = linear(f"b{i}.pool") if f"b{i}.pool.w" in a else None
        pool_channels = int(a[f"b{i}.pool_channels"])
        if f"b{i}.taps" in a:
            return ConvNeXtBlock(
                a[f"b{i}.taps"],
                a[f"b{i}.dw"],
                a[f"b{i}.dw_bias"],
                conv(f"b{i}.expand"),
                conv(f"b{i}.project"),
                pool_channels,
                pool,
                Scan(
                    int(a[f"b{i}.scan_length"]),
                    a[f"b{i}.scan_w"],
                    a[f"b{i}.scan_gate_w"],
                    a[f"b{i}.scan_gate_b"],
                )
                if f"b{i}.scan_w" in a
                else None,
            )
        return Block(
            conv(f"b{i}.conv1"),
            norm(f"b{i}.norm1"),
            conv(f"b{i}.conv2"),
            norm(f"b{i}.norm2"),
            pool_channels,
            pool,
        )

    program = Program(
        channels=channels,
        pre=bool(pre),
        head_channels=head_channels,
        stem=conv("stem"),
        stem_norm=norm("stem_norm"),
        blocks=[block(i) for i in range(blocks)],
        final_norm=norm("final_norm"),
        head=conv("head"),
        policy_norm=norm("policy_norm"),
        pool_norm=norm("pool_norm"),
        value_norm=norm("value_norm"),
        pool_linear=linear("pool_linear"),
        pass_linear=linear("pass_linear"),
        value_hidden=linear("value_hidden"),
        value_out=linear("value_out"),
        square_w=a["square_w"],
        value_weights=a.get("value_weights"),
    )
    names = {k[2:].rsplit(".", 1)[0] for k in a.files if k.startswith("q.")}
    quantized = {
        name: QuantizedConv(
            a[f"q.{name}.q"],
            a[f"q.{name}.scale"],
            a[f"q.{name}.bias"],
            a[f"q.{name}.inverse"],
            int(a[f"q.{name}.zero_point"]),
        )
        for name in names
    }
    return program, quantized


class NativeNet:
    """A quantized program built in C++, for `cpu_search.Engine.native` or `forward` here."""

    def __init__(self, p: Program, q: dict[str, QuantizedConv], lib=None):
        """`lib`: the compiled search to build it in (`cpu_search.load_library`), by default the
        one play loads, for this CPU's level."""
        if lib is None:
            from cpu_search import LIB as lib

        if p.channels % 16 or (3 * p.head_channels) % 16:
            raise ValueError("the native network takes channels in multiples of 16")
        self.lib = lib
        self.handle = lib.net_new(p.channels, int(p.pre), p.head_channels)
        self.channels = p.channels
        # the C++ copies everything it's given
        floats = ctypes.POINTER(ctypes.c_float)
        ptr = lambda x: np.ascontiguousarray(x, np.float32).ctypes.data_as(floats)

        def conv(where: str, block: int, qc: QuantizedConv) -> None:
            cout, cin, k, _ = qc.q.shape
            err = lib.net_set_conv(
                self.handle,
                where.encode(),
                block,
                cin,
                cout,
                k,
                np.ascontiguousarray(qc.q).ctypes.data_as(
                    ctypes.POINTER(ctypes.c_int8)
                ),
                ptr(qc.scale),
                ptr(qc.bias),
                ptr(qc.inverse),
                qc.zero_point,
            )
            if err:
                raise ValueError(
                    f"the C++ can't take convolution {where} {block}: {qc.q.shape}"
                )

        def norm(where: str, block: int, n: Norm, channels: int) -> None:
            a = n.a if n.a is not None else np.zeros(channels, np.float32)
            c = n.c if n.c is not None else np.zeros(channels, np.float32)
            lib.net_set_norm(
                self.handle,
                where.encode(),
                block,
                n.type,
                channels,
                ptr(a),
                ptr(c),
                n.eps,
            )

        def linear(where: str, block: int, layer: Linear) -> None:
            out, n_in = layer.w.shape
            lib.net_set_linear(
                self.handle,
                where.encode(),
                block,
                n_in,
                out,
                ptr(layer.w),
                ptr(layer.b),
            )

        for i, b in enumerate(p.blocks):
            lib.net_add_block(self.handle, b.pool_channels)
        stem = q["stem"]
        if stem.q.shape[1:] != (4, 3, 3) or lib.net_set_stem(
            self.handle,
            len(stem.q),
            np.ascontiguousarray(stem.q, np.int16).ctypes.data_as(
                ctypes.POINTER(ctypes.c_int16)
            ),
            ptr(stem.scale),
            ptr(stem.bias),
        ):
            raise ValueError(f"the C++ can't take the stem: {stem.q.shape}")
        norm("stem", 0, p.stem_norm, p.channels)
        for i, b in enumerate(p.blocks):
            if isinstance(b, ConvNeXtBlock) and b.scan is not None:
                sc = b.scan
                if lib.net_set_scan(
                    self.handle,
                    i,
                    sc.length,
                    len(sc.gate_b),
                    ptr(sc.w),
                    ptr(b.dw_bias),
                    ptr(sc.gate_w),
                    ptr(sc.gate_b),
                ):
                    raise ValueError(f"the C++ can't take block {i}'s gated scan")
            elif isinstance(b, ConvNeXtBlock):
                taps = np.ascontiguousarray(b.taps, np.int32)
                if lib.net_set_depthwise(
                    self.handle,
                    i,
                    len(taps),
                    taps.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    ptr(b.dw.T),
                    ptr(b.dw_bias),
                ):
                    raise ValueError(f"the C++ can't take block {i}'s taps")
            if isinstance(b, ConvNeXtBlock):
                conv("expand", i, q[f"{i}.expand"])
                conv("project", i, q[f"{i}.project"])
                if b.pool is not None:
                    linear("pool", i, b.pool)
                continue
            conv("conv1", i, q[f"{i}.conv1"])
            conv("conv2", i, q[f"{i}.conv2"])
            norm("norm1", i, b.norm1, p.channels)
            norm("norm2", i, b.norm2, p.channels)
            if b.pool is not None:
                linear("pool", i, b.pool)
        norm("final", 0, p.final_norm, p.channels)
        h = p.head_channels
        norm("policy", 0, p.policy_norm, h)
        norm("pool", 0, p.pool_norm, h)
        norm("value", 0, p.value_norm, h)
        linear("pool", -1, p.pool_linear)
        linear("pass", -1, p.pass_linear)
        linear("value_hidden", -1, p.value_hidden)
        linear("value_out", -1, p.value_out)
        conv("head", 0, q["head"])
        lib.net_set_heads(self.handle, ptr(p.square_w))
        if p.value_weights is not None and lib.net_set_value_weights(
            self.handle, len(p.value_weights), ptr(p.value_weights)
        ):
            raise ValueError(
                f"the C++ can't take a value head over {len(p.value_weights)} scores"
            )

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            self.lib.net_free(self.handle)

    def forward(
        self, observations: np.ndarray, threads: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """(n, 2, 8, 8) observations to (n, 65) logits and (n) values, on `threads` threads."""
        x = np.ascontiguousarray(observations, np.float32)
        logits = np.empty((len(x), 65), np.float32)
        values = np.empty(len(x), np.float32)
        floats = ctypes.POINTER(ctypes.c_float)
        self.lib.net_forward(
            self.handle,
            len(x),
            x.ctypes.data_as(floats),
            logits.ctypes.data_as(floats),
            values.ctypes.data_as(floats),
            threads,
        )
        return logits, values


def native_path(name: str) -> Path:
    from quantize import QUANTIZED_DIR

    return QUANTIZED_DIR / f"{name}-native.npz"


def load_native(name: str) -> NativeNet:
    """The network `name` of the league's store, as `main` saved it."""
    return NativeNet(*load(native_path(name)))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--net", nargs="+", default=[], help="the league's networks")
    parser.add_argument("--checkpoint", type=Path, help="or a checkpoint, of --spec")
    parser.add_argument("--spec")
    parser.add_argument("--name", help="the checkpoint's name (see above)")
    parser.add_argument(
        "--positions", type=Path, required=True, help="calibration .npy"
    )
    parser.add_argument("--calibration", type=int, default=4000)
    args = parser.parse_args()
    from bench_cpu_backends import network_path

    nets = [(name, *network_path(name)) for name in args.net]
    if args.checkpoint is not None:
        if args.spec is None:
            parser.error("--checkpoint needs its network's --spec")
        checkpoint = args.checkpoint.expanduser()
        name = args.name or f"{checkpoint.parent.name}-{checkpoint.stem}"
        nets.append((name, str(checkpoint), args.spec))
    if not nets:
        parser.error("--net or --checkpoint")
    positions = np.load(args.positions)[: args.calibration]
    for name, path, net in nets:
        save(native_path(name), *quantize(extract_checkpoint(path, net), positions))
        print(native_path(name), flush=True)


if __name__ == "__main__":
    main()
