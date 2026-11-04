import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from core.networks.utils import BATCH_AXIS


@dataclass
class AZResnetConfig:
    """Configuration for AlphaZero ResNet model.

    Attributes:
        policy_head_out_size: output size of the policy head (number of actions)
        num_blocks: number of residual blocks
        num_channels: number of channels in each residual block
        block: "basic", AlphaZero's residual block: two 3x3 convolutions. Or "bottleneck",
            KataGo's nested bottleneck block: a 1x1 convolution down to `bottleneck_channels`,
            `bottleneck_inner` basic blocks at that width, and a 1x1 convolution back up. At half
            the width with one inner block it takes about 5.5 d^2 multiply-adds per square, against
            the basic block's 18 d^2 (d channels)
        bottleneck_channels: the bottleneck blocks' inner width, by default half of `num_channels`
        bottleneck_inner: basic blocks inside each bottleneck block
        pool_every: every `pool_every`-th basic block (counting those inside bottleneck blocks) has
            KataGo's global pooling bias: `pool_channels` of its first convolution's channels are
            pooled over the board (mean and max), and a linear map of them biases the others, which
            go on to the second convolution. 0 for none
        pool_channels: channels each global pooling bias pools
        heads: "alphazero", AlphaZero's heads: a 1x1 convolution to 2 channels and a linear map to
            the policy logits; a 1x1 convolution to 1 channel and a hidden layer to the value. Or
            "pooled", KataGo's: a policy logit per square from 1x1 convolutions, biased by features
            pooled over the board, and the pass logit from those; the value a hidden layer on
            features pooled over the board. Only for games whose actions are the board's squares,
            row by row, then a pass
        head_channels: the pooled heads' channels
        value_hidden_size: width of the value head's hidden layer (AlphaZero's: 256)
        inference_dtype: dtype the network computes in, in inference mode (see `eqx.nn.inference_mode`),
            e.g. "bfloat16" for faster self-play
        train_dtype: dtype the convolutions compute in, in training, e.g. "bfloat16" for mixed precision:
            the parameters, BatchNorm, the activations between residual blocks and the heads stay in
            float32, and the gradients reach the float32 parameters. bfloat16 has float32's exponent
            range, so the loss needs no scaling
        norm: what follows each convolution, in place of AlphaZero's BatchNorm:
            - "batch": BatchNorm. In training each channel is normalized by the batch's statistics,
              in inference by their running averages (see `bn_momentum`), folded into the
              convolution
            - "layer": each input normalized on its own, over its channels and the board (LayerNorm)
            - "group": the same over each of `norm_groups` groups of channels (GroupNorm)
            - "rms": as "layer", divided by the root mean square, the mean not subtracted (RMSNorm)
            - "channel": each square's channels normalized on their own (ConvNeXt's LayerNorm)
            - "fixed": no normalization, but a fixed scale that gives the convolution's output
              about unit variance at initialization (KataGo's fixed variance initialization)
            - "ws": no normalization, but the convolution's weights standardized, each output
              channel's to mean 0 and norm 1, and a fixed scale as "fixed"'s (NFNets' scaled weight
              standardization): the output's variance doesn't drift as the weights' scale does
            Each then has a learned scale and offset per channel. All but "batch" treat each input
            on its own: the network has no state, and training and inference compute the same.
            "fixed" and "ws" fold into the convolution as BatchNorm does in inference; "layer",
            "group", "rms" and "channel" cost a pass over the activations
        layout: where the residual blocks normalize. "post", AlphaZero's: each convolution followed
            by its normalization, then a ReLU; a block's output relu(x + branch). Or "pre", the
            pre-activation residual network's (as KataGo's, and pre-norm transformers'): each
            convolution preceded by a normalization and a ReLU, a block's output x + branch, the
            trunk itself never normalized, and a normalization and a ReLU at its end before the
            heads. Every branch then sees normalized input, however the trunk grows with depth.
            Not with "fixed" or "ws", which scale the convolution they follow. BatchNorm in "pre"
            doesn't fold into a convolution in inference
        norm_groups: "group"'s groups of channels: of a layer of C channels, gcd(norm_groups, C)
        bn_momentum: how slowly BatchNorm's running averages of the batch statistics, which inference
            uses, follow them: at about 1 / (1 - momentum) training steps. A long window lags the
            weights while they move fast: 0.9 had about a third of 0.99's jumps in value error
            between epochs at a learning rate of 1e-3 (4x64, one run), and played level
        remat: in training, keep only each residual block's input for the backward pass, and compute
            the activations inside the block again from it (gradient checkpointing, see `jax.checkpoint`):
            training takes much less memory, for about one more forward pass of compute. The
            results are the same up to float rounding. Inference is unchanged.
    """

    policy_head_out_size: int
    num_blocks: int
    num_channels: int
    block: Literal["basic", "bottleneck"] = "basic"
    bottleneck_channels: int | None = None
    bottleneck_inner: int = 1
    pool_every: int = 0
    pool_channels: int = 32
    heads: Literal["alphazero", "pooled"] = "alphazero"
    head_channels: int = 32
    value_hidden_size: int = 256
    inference_dtype: str = "float32"
    train_dtype: str = "float32"
    norm: Literal["batch", "layer", "group", "rms", "channel", "fixed", "ws"] = "batch"
    layout: Literal["post", "pre"] = "post"
    norm_groups: int = 8
    bn_momentum: float = 0.9
    remat: bool = False


def conv(
    in_channels: int, out_channels: int, kernel_size: int, key: jax.Array
) -> eqx.nn.Conv2d:
    return eqx.nn.Conv2d(
        in_channels, out_channels, kernel_size, padding="SAME", use_bias=False, key=key
    )


def batch_norm(channels: int, momentum: float = 0.99) -> eqx.nn.BatchNorm:
    """BatchNorm whose running averages of the batch statistics both start at 0.

    Equinox (0.13) starts the running variance at 1, but divides both running averages by
    1 - momentum^steps, as if they started at 0 (the zero-debiasing of Adam's moments): in inference
    the variance was too large by momentum^steps / (1 - momentum^steps), 99 times after one training
    step, a third too large after 128, and only negligibly so after about 1000.
    """
    bn = eqx.nn.BatchNorm(
        channels, axis_name=BATCH_AXIS, mode="batch", momentum=momentum
    )
    zeros = jnp.zeros((channels,))
    return eqx.tree_at(
        lambda bn: bn.batch_state_index, bn, eqx.nn.StateIndex((zeros, zeros))
    )


def inference_affine(
    bn: eqx.nn.BatchNorm, state: eqx.nn.State
) -> tuple[jax.Array, jax.Array]:
    """The per-channel affine map, x * scale + offset, BatchNorm (see `batch_norm`) applies in
    inference mode: normalization by the averages of the batch statistics it has seen in training,
    or before it has seen any, by mean 0 and variance 1 (where its running averages, still 0, would
    divide by the square root of its epsilon).

    Returns:
        tuple[jax.Array, jax.Array]: scale and offset, each (channels,)
    """
    assert bn.batch_counter is not None and bn.batch_state_index is not None
    steps = state.get(bn.batch_counter)
    running_mean, running_var = state.get(bn.batch_state_index)
    # the running averages start at 0, so they fall short by a factor of 1 - momentum^steps
    debias = jnp.where(steps == 0, 1, 1 - bn.momentum**steps)
    mean = running_mean / debias
    var = jnp.where(steps == 0, 1, running_var / debias)
    scale = jax.lax.rsqrt(var + bn.eps)
    if bn.weight is not None and bn.bias is not None:
        return scale * bn.weight, bn.bias - mean * scale * bn.weight
    return scale, -mean * scale


class GroupNorm(eqx.Module):
    """Normalizes each input on its own: each group of its channels, over them and the board, to mean
    0 and variance 1 (GroupNorm; LayerNorm with one group), or without `center` only divided by its
    root mean square (RMSNorm); or, not `over_board`, each square's channels on their own
    (ConvNeXt's LayerNorm). Then a learned scale and offset for each channel. Computes in
    float32."""

    weight: jax.Array
    bias: jax.Array
    inference: bool
    groups: int = eqx.field(static=True)
    center: bool = eqx.field(static=True)
    over_board: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        groups: int = 1,
        center: bool = True,
        over_board: bool = True,
        eps: float = 1e-5,
    ):
        self.weight = jnp.ones((channels,))
        self.bias = jnp.zeros((channels,))
        self.inference = False
        self.groups = groups
        self.center = center
        self.over_board = over_board
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        """(channels, height, width) features, normalized, in float32."""
        x = x.astype(jnp.float32)
        # (groups, values normalized together, squares normalized apart)
        grouped = (
            x.reshape(self.groups, -1, 1)
            if self.over_board
            else x.reshape(1, x.shape[0], -1)
        )
        if self.center:
            grouped = grouped - grouped.mean(axis=1, keepdims=True)
        square = jnp.square(grouped).mean(axis=1, keepdims=True)
        normalized = (grouped * jax.lax.rsqrt(square + self.eps)).reshape(x.shape)
        return normalized * self.weight.reshape(-1, 1, 1) + self.bias.reshape(-1, 1, 1)


class FixedScale(eqx.Module):
    """No normalization, but a fixed scale, `gain`, chosen to give the convolution it follows an
    output of about unit variance at initialization, then a learned scale and offset for each
    channel (KataGo's fixed variance initialization). With `standardize`, the convolution's weights
    are standardized first, each output channel's to mean 0 and norm 1, so that its output's
    variance is its input's however the weights' scale moves (NFNets' scaled weight
    standardization). Both fold into the convolution's weights and a bias."""

    weight: jax.Array
    bias: jax.Array
    inference: bool
    gain: float = eqx.field(static=True)
    standardize: bool = eqx.field(static=True)

    def __init__(self, channels: int, gain: float, standardize: bool = False):
        self.weight = jnp.ones((channels,))
        self.bias = jnp.zeros((channels,))
        self.inference = False
        self.gain = gain
        self.standardize = standardize

    def conv_weight(self, weight: jax.Array) -> jax.Array:
        """The weights, (out, in, height, width), of the convolution with this folded into it, in
        float32: standardized if `standardize`, and scaled. Its bias is `self.bias`."""
        weight = weight.astype(jnp.float32)
        if self.standardize:
            fan_in = math.prod(weight.shape[1:])
            mean = weight.mean(axis=(1, 2, 3), keepdims=True)
            var = weight.var(axis=(1, 2, 3), keepdims=True)
            weight = (weight - mean) * jax.lax.rsqrt(jnp.maximum(var * fan_in, 1e-4))
        return weight * (self.gain * self.weight).reshape(-1, 1, 1, 1)


Norm = eqx.nn.BatchNorm | GroupNorm | FixedScale

# the second moment, E[x^2], and the variance of the features a convolution takes: "fixed" and
# "ws" (see `AZResnetConfig.norm`) scale its output to unit variance at initialization by them
Stats = tuple[float, float]

# a normalization for a layer of so many channels, given the statistics of the input of the
# convolution it follows, and that convolution's kernel size (see `norm_maker`)
NormMaker = Callable[[int, Stats, int], Norm]

# the statistics taken for the network's input: boards of two 0/1 planes, about half full
INPUT_STATS: Stats = (0.25, 0.19)


@functools.cache
def residual_stats(blocks: int) -> tuple[Stats, ...]:
    """The statistics of the features of a stack of residual blocks at initialization, as
    normalization makes them: a ReLU of a unit-variance input with mean 0, then after each block
    relu(x + y), where the block's branch y has unit variance and mean 0. Estimated by sampling.

    Returns:
        tuple[Stats, ...]: the statistics of each block's input, and of the last block's output
    """
    branches = np.random.default_rng(0).standard_normal((blocks + 1, 1 << 16))
    x = np.maximum(branches[0], 0)
    stats = [(float(np.mean(x**2)), float(np.var(x)))]
    for y in branches[1:]:
        x = np.maximum(x + y, 0)
        stats.append((float(np.mean(x**2)), float(np.var(x))))
    return tuple(stats)


def norm_maker(
    norm: str = "batch",
    groups: int = 8,
    bn_momentum: float = 0.9,
    board: tuple[int, int] = (8, 8),
) -> NormMaker:
    """Makes the normalizations `AZResnetConfig.norm` names, with its `norm_groups` and
    `bn_momentum`, for convolutions over a board of `board` (height, width) squares."""

    def make(channels: int, stats: Stats, kernel: int) -> Norm:
        if norm == "batch":
            return batch_norm(channels, bn_momentum)
        if norm == "layer":
            return GroupNorm(channels)
        if norm == "group":
            return GroupNorm(channels, math.gcd(groups, channels))
        if norm == "rms":
            return GroupNorm(channels, center=False)
        if norm == "channel":
            return GroupNorm(channels, over_board=False)
        # zero padding: of a kernel's taps, on average this fraction falls on the board
        coverage = math.prod((n - (kernel - 1) / kernel) / n for n in board)
        moment, variance = (coverage * stat for stat in stats)
        # equinox initializes a convolution's weights uniformly in +-1/sqrt(fan in): its output's
        # variance is a third of its input's second moment
        if norm == "fixed":
            return FixedScale(channels, math.sqrt(3 / moment))
        # standardized weights have mean 0, so they cancel the input's mean: the output's variance
        # is the input's
        if norm == "ws":
            return FixedScale(channels, 1 / math.sqrt(variance), standardize=True)
        raise ValueError(f"no normalization {norm!r}")

    return make


def convolve(conv: eqx.nn.Conv2d, x: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """`conv` of `x`, computing in `dtype`."""
    cast = eqx.tree_at(lambda c: c.weight, conv, conv.weight.astype(dtype))
    return cast(x.astype(dtype))


def conv_norm(
    conv: eqx.nn.Conv2d,
    norm: Norm,
    x: jax.Array,
    state: eqx.nn.State,
    dtype: jnp.dtype,
) -> tuple[jax.Array, eqx.nn.State]:
    """A convolution followed by its normalization (see `AZResnetConfig.norm`). The convolution
    computes in `dtype`; the output is float32 in training, `dtype` in inference mode.

    In training, normalization computes in float32 (BatchNorm's batch statistics are too coarse in
    bfloat16). BatchNorm in inference mode is a fixed per-channel affine map (see
    `inference_affine`), and so are "fixed" and "ws" always: they're folded into the convolution's
    weights and a bias, one pass over the activations instead of two (or more).
    """

    def weighted(weight: jax.Array) -> jax.Array:
        return convolve(eqx.tree_at(lambda c: c.weight, conv, weight), x, dtype)

    if isinstance(norm, eqx.nn.BatchNorm):
        if not norm.inference:
            return norm(convolve(conv, x, dtype).astype(jnp.float32), state)
        scale, offset = inference_affine(norm, state)
        y = weighted(conv.weight * scale.reshape(-1, 1, 1, 1))
        return y + offset.reshape(-1, 1, 1).astype(dtype), state
    out_dtype = dtype if norm.inference else jnp.dtype(jnp.float32)
    if isinstance(norm, FixedScale):
        y = weighted(norm.conv_weight(conv.weight)).astype(out_dtype)
        return y + norm.bias.reshape(-1, 1, 1).astype(out_dtype), state
    return norm(convolve(conv, x, dtype)).astype(out_dtype), state


def norm_act(
    norm: Norm, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
) -> tuple[jax.Array, eqx.nn.State]:
    """A normalization of `x`, (channels, height, width), then a ReLU, as the "pre" layout puts
    before each convolution (see `AZResnetConfig.layout`): in float32 in training, `dtype` in
    inference mode."""
    if isinstance(norm, FixedScale):
        raise TypeError("fixed scales go after a convolution, not before")
    if isinstance(norm, eqx.nn.BatchNorm) and norm.inference:
        scale, offset = inference_affine(norm, state)
        y = x.astype(jnp.float32) * scale.reshape(-1, 1, 1) + offset.reshape(-1, 1, 1)
    elif isinstance(norm, eqx.nn.BatchNorm):
        y, state = norm(x.astype(jnp.float32), state)
    else:
        y = norm(x)
    y = jax.nn.relu(y)
    return (y.astype(dtype) if norm.inference else y), state


def board_pool(x: jax.Array) -> jax.Array:
    """KataGo's global pooling of (channels, height, width) features: each channel's mean and
    maximum over the board, (2 * channels,), in float32."""
    x = x.astype(jnp.float32)
    return jnp.concatenate([x.mean(axis=(1, 2)), x.max(axis=(1, 2))])


class ResidualBlock(eqx.Module):
    """Residual block for AlphaZero ResNet model, optionally with KataGo's global pooling bias (see
    `AZResnetConfig.pool_every`)."""

    conv1: eqx.nn.Conv2d
    norm1: Norm
    pool_linear: eqx.nn.Linear | None
    conv2: eqx.nn.Conv2d
    norm2: Norm
    pool_channels: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        pool_channels: int = 0,
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
    ):
        """
        Args:
            channels: number of channels
            pool_channels: channels of the first convolution's output pooled over the board to bias
                the others, 0 for no global pooling bias
            key: rng used to initialize parameters
            make_norm: makes the normalizations (see `norm_maker`)
            stats: the statistics of the block's input (see `Stats`)
        """
        key1, key2, key3 = jax.random.split(key, 3)
        self.conv1 = conv(channels, channels, 3, key1)
        self.norm1 = make_norm(channels, stats, 3)
        self.pool_linear = (
            eqx.nn.Linear(2 * pool_channels, channels - pool_channels, key=key3)
            if pool_channels
            else None
        )
        self.conv2 = conv(channels - pool_channels, channels, 3, key2)
        self.norm2 = make_norm(channels, residual_stats(0)[0], 3)
        self.pool_channels = pool_channels

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        y, state = conv_norm(self.conv1, self.norm1, x, state, dtype)
        if self.pool_linear is not None:
            y, pooled = y[: -self.pool_channels], y[-self.pool_channels :]
            bias = self.pool_linear(board_pool(jax.nn.relu(pooled)))
            y = y + bias.reshape(-1, 1, 1).astype(y.dtype)
        y = jax.nn.relu(y)
        y, state = conv_norm(self.conv2, self.norm2, y, state, dtype)
        return jax.nn.relu(x + y), state


class BottleneckBlock(eqx.Module):
    """KataGo's nested bottleneck residual block: a 1x1 convolution down to a narrower width, basic
    residual blocks at that width, and a 1x1 convolution back up."""

    down_conv: eqx.nn.Conv2d
    down_norm: Norm
    inner: list[ResidualBlock]
    up_conv: eqx.nn.Conv2d
    up_norm: Norm

    def __init__(
        self,
        channels: int,
        inner_channels: int,
        inner_pool_channels: list[int],
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
    ):
        """
        Args:
            channels: number of channels in and out
            inner_channels: number of channels of the inner blocks
            inner_pool_channels: each inner block's global pooling channels (see `ResidualBlock`),
                one per inner block
            key: rng used to initialize parameters
            make_norm: makes the normalizations (see `norm_maker`)
            stats: the statistics of the block's input (see `Stats`)
        """
        keys = jax.random.split(key, len(inner_pool_channels) + 2)
        self.down_conv = conv(channels, inner_channels, 1, keys[0])
        self.down_norm = make_norm(inner_channels, stats, 1)
        inner_stats = residual_stats(len(inner_pool_channels))
        self.inner = [
            ResidualBlock(
                inner_channels, pool, key=k, make_norm=make_norm, stats=block_stats
            )
            for pool, k, block_stats in zip(
                inner_pool_channels, keys[2:], inner_stats[:-1], strict=True
            )
        ]
        self.up_conv = conv(inner_channels, channels, 1, keys[1])
        self.up_norm = make_norm(channels, inner_stats[-1], 1)

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        y, state = conv_norm(self.down_conv, self.down_norm, x, state, dtype)
        y = jax.nn.relu(y)
        for block in self.inner:
            y, state = block(y, state, dtype)
        y, state = conv_norm(self.up_conv, self.up_norm, y, state, dtype)
        return jax.nn.relu(x + y), state


def zeroed(conv: eqx.nn.Conv2d) -> eqx.nn.Conv2d:
    """`conv` with its weights at zero: a residual branch ending in it starts adding nothing, so a
    pre-activation trunk starts as the stem's features (SkipInit, Fixup's zeroed last layer). With
    random branches, 16 of them drowned the stem's features (RMS 7 against 0.24 at 16x64), and the
    network learned 10-15 epochs late."""
    return eqx.tree_at(lambda c: c.weight, conv, jnp.zeros_like(conv.weight))


class PreActBlock(eqx.Module):
    """A pre-activation residual block (see `AZResnetConfig.layout`): x + conv2(relu(norm2(conv1(
    relu(norm1(x)))))), optionally with KataGo's global pooling bias, from the last
    `pool_channels` of conv1's normalized output. conv2 starts at zero (see `zeroed`)."""

    norm1: Norm
    conv1: eqx.nn.Conv2d
    norm2: Norm
    pool_linear: eqx.nn.Linear | None
    conv2: eqx.nn.Conv2d
    pool_channels: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        pool_channels: int = 0,
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
    ):
        """
        Args:
            channels: number of channels
            pool_channels: channels of the first convolution's output pooled over the board to bias
                the others, 0 for no global pooling bias
            key: rng used to initialize parameters
            make_norm: makes the normalizations (see `norm_maker`)
            stats: unused: the normalizations before the convolutions need none
        """
        key1, key2, key3 = jax.random.split(key, 3)
        self.norm1 = make_norm(channels, stats, 3)
        self.conv1 = conv(channels, channels, 3, key1)
        self.norm2 = make_norm(channels, stats, 3)
        self.pool_linear = (
            eqx.nn.Linear(2 * pool_channels, channels - pool_channels, key=key3)
            if pool_channels
            else None
        )
        self.conv2 = zeroed(conv(channels - pool_channels, channels, 3, key2))
        self.pool_channels = pool_channels

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        y, state = norm_act(self.norm1, x, state, dtype)
        y = convolve(self.conv1, y, dtype)
        y, state = norm_act(self.norm2, y, state, dtype)
        if self.pool_linear is not None:
            y, pooled = y[: -self.pool_channels], y[-self.pool_channels :]
            bias = self.pool_linear(board_pool(pooled))
            y = y + bias.reshape(-1, 1, 1).astype(y.dtype)
        y = convolve(self.conv2, y, dtype)
        return x + y.astype(x.dtype), state


class PreActBottleneck(eqx.Module):
    """KataGo's nested bottleneck block in the pre-activation layout: x + up(relu(norm(inner(down(
    relu(norm(x))))))), the inner blocks pre-activation basic blocks. up starts at zero (see
    `zeroed`)."""

    down_norm: Norm
    down_conv: eqx.nn.Conv2d
    inner: list[PreActBlock]
    up_norm: Norm
    up_conv: eqx.nn.Conv2d

    def __init__(
        self,
        channels: int,
        inner_channels: int,
        inner_pool_channels: list[int],
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
    ):
        keys = jax.random.split(key, len(inner_pool_channels) + 2)
        self.down_norm = make_norm(channels, stats, 1)
        self.down_conv = conv(channels, inner_channels, 1, keys[0])
        self.inner = [
            PreActBlock(inner_channels, pool, key=k, make_norm=make_norm, stats=stats)
            for pool, k in zip(inner_pool_channels, keys[2:], strict=True)
        ]
        self.up_norm = make_norm(inner_channels, stats, 1)
        self.up_conv = zeroed(conv(inner_channels, channels, 1, keys[1]))

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        y, state = norm_act(self.down_norm, x, state, dtype)
        y = convolve(self.down_conv, y, dtype)
        for block in self.inner:
            y, state = block(y, state, dtype)
        y, state = norm_act(self.up_norm, y, state, dtype)
        y = convolve(self.up_conv, y, dtype)
        return x + y.astype(x.dtype), state


Block = ResidualBlock | BottleneckBlock | PreActBlock | PreActBottleneck


def checkpointed(
    block: Block, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
) -> tuple[jax.Array, eqx.nn.State]:
    """Applies `block` under `jax.checkpoint`: differentiating it saves only its inputs, and computes
    its activations again in the backward pass."""
    params, static = eqx.partition(block, eqx.is_array)

    @jax.checkpoint
    def apply(
        params: Block, x: jax.Array, state: eqx.nn.State
    ) -> tuple[jax.Array, eqx.nn.State]:
        return eqx.combine(params, static)(x, state, dtype)

    return apply(params, x, state)


class AlphaZeroHeads(eqx.Module):
    """AlphaZero's policy and value heads (see `AZResnetConfig.heads`)."""

    policy_conv: eqx.nn.Conv2d
    policy_norm: Norm
    policy_linear: eqx.nn.Linear
    value_conv: eqx.nn.Conv2d
    value_norm: Norm
    value_hidden: eqx.nn.Linear
    value_linear: eqx.nn.Linear

    def __init__(
        self,
        channels: int,
        squares: int,
        num_actions: int,
        value_hidden_size: int,
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
    ):
        keys = jax.random.split(key, 5)
        self.policy_conv = conv(channels, 2, 1, keys[0])
        self.policy_norm = make_norm(2, stats, 1)
        self.policy_linear = eqx.nn.Linear(2 * squares, num_actions, key=keys[1])
        # as AlphaZero's: a 1x1 convolution to one channel, then a hidden layer
        self.value_conv = conv(channels, 1, 1, keys[2])
        self.value_norm = make_norm(1, stats, 1)
        self.value_hidden = eqx.nn.Linear(squares, value_hidden_size, key=keys[3])
        self.value_linear = eqx.nn.Linear(value_hidden_size, 1, key=keys[4])

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State]:
        policy, state = conv_norm(self.policy_conv, self.policy_norm, x, state, dtype)
        policy = jax.nn.relu(policy)
        policy = self.policy_linear(policy.reshape(-1).astype(jnp.float32))

        value, state = conv_norm(self.value_conv, self.value_norm, x, state, dtype)
        value = jax.nn.relu(value)
        value = jax.nn.relu(self.value_hidden(value.reshape(-1).astype(jnp.float32)))
        value = jnp.tanh(self.value_linear(value))
        return (policy, value), state


class PooledHeads(eqx.Module):
    """KataGo's policy and value heads, on features pooled over the board (see
    `AZResnetConfig.heads`)."""

    policy_conv: eqx.nn.Conv2d
    policy_norm: Norm
    policy_pool_conv: eqx.nn.Conv2d
    policy_pool_norm: Norm
    policy_pool_linear: eqx.nn.Linear
    policy_square_conv: eqx.nn.Conv2d
    pass_linear: eqx.nn.Linear
    value_conv: eqx.nn.Conv2d
    value_norm: Norm
    value_hidden: eqx.nn.Linear
    value_linear: eqx.nn.Linear

    def __init__(
        self,
        channels: int,
        head_channels: int,
        value_hidden_size: int,
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
    ):
        keys = jax.random.split(key, 8)
        self.policy_conv = conv(channels, head_channels, 1, keys[0])
        self.policy_norm = make_norm(head_channels, stats, 1)
        self.policy_pool_conv = conv(channels, head_channels, 1, keys[1])
        self.policy_pool_norm = make_norm(head_channels, stats, 1)
        self.policy_pool_linear = eqx.nn.Linear(
            2 * head_channels, head_channels, key=keys[2]
        )
        self.policy_square_conv = conv(head_channels, 1, 1, keys[3])
        self.pass_linear = eqx.nn.Linear(2 * head_channels, 1, key=keys[4])
        self.value_conv = conv(channels, head_channels, 1, keys[5])
        self.value_norm = make_norm(head_channels, stats, 1)
        self.value_hidden = eqx.nn.Linear(
            2 * head_channels, value_hidden_size, key=keys[6]
        )
        self.value_linear = eqx.nn.Linear(value_hidden_size, 1, key=keys[7])

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State]:
        policy, state = conv_norm(self.policy_conv, self.policy_norm, x, state, dtype)
        pooled, state = conv_norm(
            self.policy_pool_conv, self.policy_pool_norm, x, state, dtype
        )
        pooled = board_pool(jax.nn.relu(pooled))
        bias = self.policy_pool_linear(pooled)
        policy = jax.nn.relu(policy + bias.reshape(-1, 1, 1).astype(policy.dtype))
        squares = self.policy_square_conv(policy.astype(jnp.float32)).reshape(-1)
        policy = jnp.concatenate([squares, self.pass_linear(pooled)])

        value, state = conv_norm(self.value_conv, self.value_norm, x, state, dtype)
        value = board_pool(jax.nn.relu(value))
        value = jax.nn.relu(self.value_hidden(value))
        value = jnp.tanh(self.value_linear(value))
        return (policy, value), state


class AZResnet(eqx.Module):
    """Implements the AlphaZero ResNet model, and KataGo's variants of it (see `AZResnetConfig`).

    Create it with its state, BatchNorm's (empty with the other normalizations):
    `eqx.nn.make_with_state(AZResnet)(config, input_shape, key=key)`
    """

    stem_conv: eqx.nn.Conv2d
    stem_norm: Norm | None
    blocks: list[Block]
    final_norm: Norm | None
    heads: AlphaZeroHeads | PooledHeads
    inference: bool
    inference_dtype: str = eqx.field(static=True)
    train_dtype: str = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __init__(
        self,
        config: AZResnetConfig,
        input_shape: tuple[int, int, int],
        *,
        key: jax.Array,
    ):
        """
        Args:
            config: network configuration
            input_shape: shape of a single (unbatched) input, (height, width, channels)
            key: rng used to initialize parameters
        """
        height, width, in_channels = input_shape
        stem_key, heads_key, *block_keys = jax.random.split(key, config.num_blocks + 2)
        channels = config.num_channels
        make_norm = norm_maker(
            config.norm, config.norm_groups, config.bn_momentum, (height, width)
        )
        pre = config.layout == "pre"
        if pre and config.norm in ("fixed", "ws"):
            raise ValueError(
                f'norm={config.norm} scales the convolution it follows: not with layout="pre"'
            )
        # initial conv layer: in the pre-activation layout, the trunk starts from it unnormalized
        self.stem_conv = conv(in_channels, channels, 3, stem_key)
        self.stem_norm = None if pre else make_norm(channels, INPUT_STATS, 3)

        # residual blocks: the global pooling channels of each basic block, counting those inside
        # bottleneck blocks
        inner = 1 if config.block == "basic" else config.bottleneck_inner
        pool_channels = [
            config.pool_channels
            if config.pool_every and (i + 1) % config.pool_every == 0
            else 0
            for i in range(config.num_blocks * inner)
        ]
        stats = residual_stats(config.num_blocks)
        basic, bottleneck = (
            (PreActBlock, PreActBottleneck) if pre else (ResidualBlock, BottleneckBlock)
        )
        if config.block == "basic":
            self.blocks = [
                basic(channels, pool, key=k, make_norm=make_norm, stats=block_stats)
                for pool, k, block_stats in zip(
                    pool_channels, block_keys, stats[:-1], strict=True
                )
            ]
        else:
            self.blocks = [
                bottleneck(
                    channels,
                    config.bottleneck_channels or channels // 2,
                    pool_channels[i * inner : (i + 1) * inner],
                    key=k,
                    make_norm=make_norm,
                    stats=stats[i],
                )
                for i, k in enumerate(block_keys)
            ]

        # the pre-activation trunk's end: normalized, and a ReLU, before the heads
        self.final_norm = make_norm(channels, stats[-1], 1) if pre else None

        if config.heads == "alphazero":
            self.heads = AlphaZeroHeads(
                channels,
                height * width,
                config.policy_head_out_size,
                config.value_hidden_size,
                key=heads_key,
                make_norm=make_norm,
                stats=stats[-1],
            )
        else:
            if config.policy_head_out_size != height * width + 1:
                raise ValueError(
                    "the pooled heads output a logit per square and one to pass: "
                    f"{height * width + 1} actions, not {config.policy_head_out_size}"
                )
            self.heads = PooledHeads(
                channels,
                config.head_channels,
                config.value_hidden_size,
                key=heads_key,
                make_norm=make_norm,
                stats=stats[-1],
            )
        self.inference = False
        self.inference_dtype = config.inference_dtype
        self.train_dtype = config.train_dtype
        self.remat = config.remat

    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State]:
        # the convolutions compute in the inference dtype in inference mode (and so do the activations
        # between them), in the train dtype in training
        dtype = jnp.dtype(self.inference_dtype if self.inference else self.train_dtype)
        # inputs are channels-last (and may be e.g. boolean), equinox convolutions are channels-first
        x = jnp.moveaxis(x, -1, 0).astype(dtype)
        # initial conv layer
        if self.stem_norm is None:
            # the pre-activation trunk, in float32 in training as the post-activation one is
            x = convolve(self.stem_conv, x, dtype)
            x = x if self.inference else x.astype(jnp.float32)
        else:
            x, state = conv_norm(self.stem_conv, self.stem_norm, x, state, dtype)
            x = jax.nn.relu(x)

        # residual blocks
        for block in self.blocks:
            if self.remat and not self.inference:
                x, state = checkpointed(block, x, state, dtype)
            else:
                x, state = block(x, state, dtype)

        if self.final_norm is not None:
            x, state = norm_act(self.final_norm, x, state, dtype)
        return self.heads(x, state, dtype)
