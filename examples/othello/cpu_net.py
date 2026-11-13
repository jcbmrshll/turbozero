"""An Othello network's forward pass for CPU play, int8 in the trunk and float elsewhere, in C++
(`cpu_net.h`), built from a trained network: its layers as plain arrays (`extract`), quantized after
training on calibration positions (`quantize`), and saved, so that play needs neither JAX nor
OpenVINO.

    uv run examples/othello/cpu_net.py --net 8x128-128s-8p-w1125-274m --positions LEAVES.npy

Each convolution of the trunk takes 8-bit inputs, scaled for the whole tensor by the minimum and
maximum seen in calibration (with a zero point when the minimum is below 0), and 8-bit weights
scaled for each output channel, as `quantize.py`'s best OpenVINO model does; and so do the heads'
first convolutions. The rest of the heads stays in float.
A normalization of each square's channels runs in the convolution's float epilogue, where OpenVINO
runs it as several ops of its own.

`float_forward` is the network in NumPy, which `quantize` calibrates on; `int8_forward` is what the
C++ computes, in NumPy, for tests.
"""

import argparse
import ctypes
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
    blocks: list[Block]
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
        GroupNorm,
        PooledHeads,
        PreActBlock,
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

    pre = network.stem_norm is None
    if pre:
        stem, stem_norm = (
            Conv(
                f32(network.stem_conv.weight),
                np.zeros(network.stem_conv.out_channels, np.float32),
            ),
            Norm(),
        )
    else:
        stem, stem_norm = conv_norm(network.stem_conv, network.stem_norm)
    blocks = []
    for block in network.blocks:
        pool = linear(block.pool_linear) if block.pool_linear is not None else None
        if isinstance(block, PreActBlock):
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
    `record(name, x)` sees each one's input."""
    record = record or (lambda name, x: None)
    x = np.moveaxis(f32(observations), 1, -1)

    def apply(name, c, x):
        record(name, x)
        return conv(name, c, x)

    x = apply("stem", p.stem, x)
    if not p.pre:
        x = normalize(p.stem_norm, x, relu=True)
    for i, b in enumerate(p.blocks):
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
    value = np.tanh(dense(p.value_out, np.maximum(dense(p.value_hidden, value), 0)))
    return f32(logits), f32(value[:, 0])


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


def weights_int8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(out, ...) float weights as signed 8-bit, symmetric, and each output channel's scale."""
    flat = np.abs(w.reshape(len(w), -1)).max(1)
    scale = np.where(flat > 0, flat / 127, 1).astype(np.float32)
    q = np.clip(np.rint(w / scale.reshape(-1, *([1] * (w.ndim - 1)))), -127, 127)
    return q.astype(np.int8), scale


@dataclass
class QuantizedConv:
    """A convolution as the C++ takes it: int8 weights, (cout, cin, k, k), cin a multiple of 4; what
    its sums are multiplied by and added to; and how its input is quantized: round(x * inverse) +
    zero_point, `inverse` (cin) a scale for each channel."""

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
    c: Conv, low: np.ndarray, high: np.ndarray, smooth: bool = True
) -> QuantizedConv:
    """`c`, its input's channels ranging from `low` to `high`: each channel divided by a scale of
    its own, folded into the weights, and then the whole input by one scale and zero point."""
    w = c.w
    if smooth:
        reach = np.maximum(np.maximum(-low, high), 1e-3)
        weights = np.maximum(np.abs(w).max((0, 2, 3)), 1e-6)
        channel = reach**ALPHA / weights ** (1 - ALPHA)
        channel = f32(channel / channel.max())
    else:
        channel = np.ones(w.shape[1], np.float32)
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
    q, w_scale = weights_int8(w)
    scale = f32(in_scale * w_scale)
    # the zero point's share of each sum, taken back out
    sums = q.reshape(len(q), -1).astype(np.int64).sum(1)
    bias = f32(c.b - scale * zero_point * sums)
    return QuantizedConv(q, scale, bias, inverse, zero_point)


def quantize(p: Program, positions: np.ndarray) -> dict[str, QuantizedConv]:
    """The convolutions quantized, by name, their inputs' ranges from `positions`, (n, 2, 8,
    8) observations (best the leaves of the network's own searches)."""
    ranges = input_ranges(p, positions)
    convs = {"stem": p.stem, "head": p.head}
    for i, b in enumerate(p.blocks):
        convs[f"{i}.conv1"] = b.conv1
        convs[f"{i}.conv2"] = b.conv2
    quantized = {
        name: quantize_conv(c, *ranges[name])
        for name, c in convs.items()
        if name != "stem"
    }
    # the stem's input is 0 or 1, which the C++ writes as it is
    ones = np.ones(2, np.float32)
    quantized["stem"] = quantize_conv(p.stem, 0 * ones, 255 * ones, smooth=False)
    return quantized


def int8_forward(p: Program, q: dict[str, QuantizedConv], observations: np.ndarray):
    """What the C++ computes, in NumPy."""

    def conv(name, _c, x):
        qc = q[name]
        if x.shape[-1] % 4:
            x = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, -x.shape[-1] % 4)))
        u = np.clip(np.rint(x * qc.inverse) + qc.zero_point, 0, 255)
        k = qc.q.shape[-1]
        edge = k // 2
        padded = np.pad(
            u,
            ((0, 0), (edge, edge), (edge, edge), (0, 0)),
            constant_values=qc.zero_point,
        )
        acc = np.zeros(x.shape[:3] + (len(qc.q),), np.int64)
        w = qc.q.astype(np.int64)
        for dy in range(k):
            for dx in range(k):
                acc += (
                    padded[:, dy : dy + 8, dx : dx + 8].astype(np.int64)
                    @ w[:, :, dy, dx].T
                )
        return f32(acc) * qc.scale + qc.bias

    return forward(p, observations, conv)


def save(path: Path, p: Program, q: dict[str, QuantizedConv]) -> None:
    """The program and its quantized convolutions, in one .npz."""
    arrays: dict[str, np.ndarray] = {
        "meta": np.array([p.channels, int(p.pre), p.head_channels, len(p.blocks)]),
        "square_w": p.square_w,
    }

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

    program = Program(
        channels=channels,
        pre=bool(pre),
        head_channels=head_channels,
        stem=conv("stem"),
        stem_norm=norm("stem_norm"),
        blocks=[
            Block(
                conv(f"b{i}.conv1"),
                norm(f"b{i}.norm1"),
                conv(f"b{i}.conv2"),
                norm(f"b{i}.norm2"),
                int(a[f"b{i}.pool_channels"]),
                linear(f"b{i}.pool") if f"b{i}.pool.w" in a else None,
            )
            for i in range(blocks)
        ],
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
        conv("stem", 0, q["stem"])
        norm("stem", 0, p.stem_norm, p.channels)
        for i, b in enumerate(p.blocks):
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
    parser.add_argument("--net", nargs="+", required=True)
    parser.add_argument(
        "--positions", type=Path, required=True, help="calibration .npy"
    )
    parser.add_argument("--calibration", type=int, default=4000)
    args = parser.parse_args()
    from bench_cpu_backends import network_path

    positions = np.load(args.positions)[: args.calibration]
    for name in args.net:
        path, net = network_path(name)
        program = extract_checkpoint(path, net)
        save(native_path(name), program, quantize(program, positions))
        print(native_path(name), flush=True)


if __name__ == "__main__":
    main()
