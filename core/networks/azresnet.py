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
            the basic block's 18 d^2 (d channels). Or "convnext", ConvNeXt's block (see
            `PreActConvNeXt`), in the "pre" layout only: a depthwise filter of each channel on its
            own, the block's only step between squares, a normalization, then a per-square MLP of
            1x1 convolutions through `expand` times the width: 2 * expand * d^2 + taps * d
            multiply-adds per square
        bottleneck_channels: the bottleneck blocks' inner width, by default half of `num_channels`
        bottleneck_inner: basic blocks inside each bottleneck block
        expand: the ConvNeXt blocks' hidden width, in multiples of `num_channels`
        expand_norm: the ConvNeXt blocks also normalize their hidden channels, after the expand
            convolution, before the global pooling bias and the ReLU. By default (None), ConvNeXt
            blocks do. With BatchNorm it folds into
            the expand convolution in inference. The pooling bias comes after it: under the norm,
            whatever part of the bias hardly varies from input to input cost nothing in training,
            and grew to 2-6 standard deviations of the hidden channels in the ray net, which any
            change of its weights (e.g. their average) then moved by many
        project_norm: the ConvNeXt blocks also normalize their output, after the project
            convolution, its learned scale starting at zero (Goyal et al.'s zero gamma) in place
            of a zeroed project convolution. With BatchNorm it folds into the project convolution
            in inference. With `stem_norm` and `expand_norm` as well, every convolution of a
            BatchNorm ConvNeXt trunk is followed by a normalization, and the function the trunk
            computes no longer depends on the scale of any convolution's weights, which weight
            decay shrinks (the pooling biases' linear maps still do). By default (None), ConvNeXt
            blocks do
        dw_kernel: the ConvNeXt blocks' depthwise filters' size: every square within
            `dw_kernel // 2` rows and columns (odd)
        dw_taps: the depthwise filters' taps instead, as (row, column) offsets from the square
            filtered (see `Depthwise`): a sparse filter of long reach, e.g. along chosen lines.
            Not settable in a spec: an application names its own sets and passes them in
        dw_branches: the ConvNeXt blocks' depthwise filters as branches instead, each of its own
            taps and normalized on its own, summed (see `BranchedDepthwise`): in inference they
            fold into one filter on the union of the taps. Not settable in a spec
        dw_experts: the ConvNeXt blocks' depthwise filters each a mixture of this many, chosen for
            each input from its features pooled over the board (see `DynamicDepthwise`); 0 for one
        scan_steps: the ConvNeXt blocks filter along lines from each square instead, in these
            (row, column) steps, each line gated by the squares it passes (see `GatedScan`). Not
            settable in a spec
        scan_length: the lines' length in steps; 0 for the board's longer side less one
        scan_gates: the gated lines' gate channels, each shared by a group of channels
        attention_last: the last this many ConvNeXt blocks step between squares by attention (see
            `BoardAttention`) instead of a filter
        attention_heads: their attention heads
        position_bias: each ConvNeXt block adds a learned bias for each square and channel to its
            step between squares' output (absolute position embedding, as CoordConv's coordinates
            or a vision transformer's), starting at zero
        mlp_experts: each ConvNeXt block's 1x1 convolutions a mixture of this many, chosen for each
            input from its normalized features pooled over the board (CondConv); 0 for one
        board_plane: the input gets one more plane, all ones (KataGo's on-board plane): the
            stem's zero padding then marks where the board ends, for empty squares too
        input_features: planes computed from the input and added to it, as a function of one
            (height, width, channels) input to (height, width, more channels), e.g. features a
            game knows how to compute. Not settable in a spec
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
        stem_norm: in the "pre" layout, normalize the stem convolution's output too (no ReLU: the
            trunk starts from it). The "post" layout always does, with a ReLU. By default (None),
            a trunk of ConvNeXt blocks does, and one of other blocks doesn't
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
    block: Literal["basic", "bottleneck", "convnext"] = "basic"
    bottleneck_channels: int | None = None
    bottleneck_inner: int = 1
    expand: int = 4
    expand_norm: bool | None = None
    project_norm: bool | None = None
    dw_kernel: int = 7
    dw_taps: tuple[tuple[int, int], ...] | None = None
    dw_branches: tuple[tuple[tuple[int, int], ...], ...] | None = None
    dw_experts: int = 0
    scan_steps: tuple[tuple[int, int], ...] | None = None
    scan_length: int = 0
    scan_gates: int = 8
    attention_last: int = 0
    attention_heads: int = 4
    position_bias: bool = False
    mlp_experts: int = 0
    board_plane: bool = False
    input_features: Callable[[jax.Array], jax.Array] | None = None
    pool_every: int = 0
    pool_channels: int = 32
    heads: Literal["alphazero", "pooled"] = "alphazero"
    head_channels: int = 32
    value_hidden_size: int = 256
    inference_dtype: str = "float32"
    train_dtype: str = "float32"
    norm: Literal["batch", "layer", "group", "rms", "channel", "fixed", "ws"] = "batch"
    layout: Literal["post", "pre"] = "post"
    stem_norm: bool | None = None
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
    norm: Norm,
    x: jax.Array,
    state: eqx.nn.State,
    dtype: jnp.dtype,
    relu: bool = True,
) -> tuple[jax.Array, eqx.nn.State]:
    """A normalization of `x`, (channels, height, width), then a ReLU (unless not `relu`), as the
    "pre" layout puts before each convolution (see `AZResnetConfig.layout`): in float32 in
    training, `dtype` in inference mode."""
    if isinstance(norm, FixedScale):
        raise TypeError("fixed scales go after a convolution, not before")
    if isinstance(norm, eqx.nn.BatchNorm) and norm.inference:
        scale, offset = inference_affine(norm, state)
        y = x.astype(jnp.float32) * scale.reshape(-1, 1, 1) + offset.reshape(-1, 1, 1)
    elif isinstance(norm, eqx.nn.BatchNorm):
        y, state = norm(x.astype(jnp.float32), state)
    else:
        y = norm(x)
    if relu:
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


def zero_scale(norm: Norm) -> Norm:
    """`norm` with its learned scale at zero: a residual branch ending in it starts adding nothing,
    as one ending in a `zeroed` convolution does, but the convolution before it starts random (a
    zeroed one under a normalization would never move: its output, and so its gradient, is 0)."""
    if not isinstance(norm, eqx.nn.BatchNorm | GroupNorm) or norm.weight is None:
        raise TypeError(f"no learned scale to zero in {norm!r}")
    return eqx.tree_at(lambda n: n.weight, norm, jnp.zeros_like(norm.weight))


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


def square_taps(kernel: int) -> tuple[tuple[int, int], ...]:
    """Every (row, column) offset of a `kernel` x `kernel` filter (odd) from its centre."""
    if kernel % 2 == 0:
        raise ValueError(f"depthwise filters have an odd size, not {kernel}")
    r = range(-(kernel // 2), kernel // 2 + 1)
    return tuple((dy, dx) for dy in r for dx in r)


class Depthwise(eqx.Module):
    """A depthwise convolution of (channels, height, width) inputs: each channel filtered on its
    own, y[c, i, j] = sum over taps t of weight[c, t] * x[c, i + dy_t, j + dx_t], zero outside,
    by a filter with weights only at its `taps`' (dy, dx) offsets. Every offset of a k x k square
    makes the usual depthwise convolution; a sparse set reaches far for few weights. No bias."""

    weight: jax.Array
    taps: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(
        self, channels: int, taps: tuple[tuple[int, int], ...], *, key: jax.Array
    ):
        self.taps = tuple((int(dy), int(dx)) for dy, dx in taps)
        if len(set(self.taps)) != len(self.taps) or not self.taps:
            raise ValueError(f"taps are distinct offsets, at least one: {taps}")
        # as equinox initializes a convolution: uniform in +-1/sqrt(fan in), the fan in the taps
        bound = 1 / math.sqrt(len(self.taps))
        self.weight = jax.random.uniform(
            key, (channels, len(self.taps)), minval=-bound, maxval=bound
        )

    @property
    def kernel_size(self) -> int:
        """The side of the smallest square filter, centred, that holds every tap."""
        return 2 * max(max(abs(dy), abs(dx)) for dy, dx in self.taps) + 1

    def dense(self) -> jax.Array:
        """The filters as a grouped convolution's (channels, 1, k, k) kernel, zero off the taps."""
        r = self.kernel_size // 2
        rows, cols = (np.array(a) + r for a in zip(*self.taps, strict=True))
        k = self.kernel_size
        kernel = jnp.zeros((self.weight.shape[0], k, k), self.weight.dtype)
        return kernel.at[:, rows, cols].set(self.weight)[:, None]

    def __call__(self, x: jax.Array, dtype: jnp.dtype) -> jax.Array:
        """The filters applied to `x`, computing in `dtype`: each channel's filter as a matrix from
        the squares to the squares (see `tap_index`), all channels one batched matrix product. A
        grouped convolution with a kernel as large as the grid ran 8x slower on a GPU."""
        channels, height, width = x.shape
        index = tap_index(self.taps, height, width)
        # one more weight, zero, for the pairs of squares no tap joins
        weight = jnp.pad(self.weight, ((0, 0), (0, 1)))[:, index]
        y = jnp.einsum(
            "cpq,cq->cp",
            weight.astype(dtype),
            x.reshape(channels, height * width).astype(dtype),
        )
        return y.reshape(channels, height, width)


@functools.cache
def tap_index(taps: tuple[tuple[int, int], ...], height: int, width: int) -> np.ndarray:
    """(squares, squares), squares = height * width row by row: which of `taps` joins square p to
    square q (q = p + its offset), or len(taps) where none does."""
    index = np.full((height * width, height * width), len(taps))
    rows, cols = np.divmod(np.arange(height * width), width)
    for t, (dy, dx) in enumerate(taps):
        r, c = rows + dy, cols + dx
        inside = (0 <= r) & (r < height) & (0 <= c) & (c < width)
        index[np.flatnonzero(inside), (r * width + c)[inside]] = t
    return index


def zero_linear(linear: eqx.nn.Linear) -> eqx.nn.Linear:
    """`linear` with its weights and bias at zero: a router starting from it weighs its experts
    equally (see `DynamicDepthwise`)."""
    assert linear.bias is not None
    return eqx.tree_at(
        lambda lin: (lin.weight, lin.bias),
        linear,
        (jnp.zeros_like(linear.weight), jnp.zeros_like(linear.bias)),
    )


class BranchedDepthwise(eqx.Module):
    """Depthwise filters (see `Depthwise`) side by side, each normalized on its own, their
    outputs summed: sum over branches b of norm_b(depthwise_b(x)). In training each branch
    learns at its own scale (RepVGG's and MobileOne's train-time branches); in inference, with
    BatchNorm, each is a per-channel affine map of a linear filter, so together they fold into one
    depthwise filter on the union of their taps plus a bias (see `fold`), costing what that one
    filter does."""

    branches: list[Depthwise]
    norms: list[Norm]

    def __init__(
        self,
        channels: int,
        branches: tuple[tuple[tuple[int, int], ...], ...],
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
    ):
        keys = jax.random.split(key, len(branches))
        self.branches = [
            Depthwise(channels, taps, key=k) for taps, k in zip(branches, keys)
        ]
        self.norms = [
            make_norm(channels, stats, dw.kernel_size) for dw in self.branches
        ]

    @property
    def kernel_size(self) -> int:
        return max(dw.kernel_size for dw in self.branches)

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        total = None
        for dw, norm in zip(self.branches, self.norms, strict=True):
            y, state = norm_act(norm, dw(x, dtype), state, dtype, relu=False)
            total = y if total is None else total + y
        assert total is not None
        return total, state

    def fold(self, state: eqx.nn.State) -> tuple[Depthwise, jax.Array]:
        """The branches in inference as one filter on the union of their taps and a bias per
        channel: what `__call__` computes in inference mode, for BatchNorm."""
        taps = tuple(dict.fromkeys(t for dw in self.branches for t in dw.taps))
        weight = jnp.zeros((self.branches[0].weight.shape[0], len(taps)))
        bias = jnp.zeros(weight.shape[0])
        for dw, norm in zip(self.branches, self.norms, strict=True):
            if not isinstance(norm, eqx.nn.BatchNorm):
                raise TypeError("only BatchNorm branches fold into one filter")
            scale, offset = inference_affine(norm, state)
            columns = np.array([taps.index(t) for t in dw.taps])
            weight = weight.at[:, columns].add(dw.weight * scale[:, None])
            bias = bias + offset
        folded = Depthwise(weight.shape[0], taps, key=jax.random.PRNGKey(0))
        return eqx.tree_at(lambda d: d.weight, folded, weight), bias


class DynamicDepthwise(eqx.Module):
    """A depthwise filter (see `Depthwise`) chosen for each input: `experts` filters on the same
    taps, mixed by a softmax of a linear map of the input's features pooled over the board (mean and
    max, see `board_pool`). The conv form of dynamic convolution (CondConv, DynamicConv) and of Lc0's
    smolgen, which makes the mixing between squares depend on the position. The router starts at
    zero: the experts start equally weighted. In inference it costs one filter plus mixing the
    experts' weights."""

    weight: jax.Array
    router: eqx.nn.Linear
    taps: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        taps: tuple[tuple[int, int], ...],
        experts: int,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, experts + 1)
        self.weight = jnp.stack(
            [Depthwise(channels, taps, key=k).weight for k in keys[:experts]]
        )
        self.taps = Depthwise(channels, taps, key=keys[0]).taps
        self.router = zero_linear(eqx.nn.Linear(2 * channels, experts, key=keys[-1]))

    @property
    def kernel_size(self) -> int:
        return 2 * max(max(abs(dy), abs(dx)) for dy, dx in self.taps) + 1

    def mixing(self, x: jax.Array) -> jax.Array:
        """The experts' weights for input `x`, (experts,), summing to 1."""
        return jax.nn.softmax(self.router(board_pool(x)))

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        channels, height, width = x.shape
        index = tap_index(self.taps, height, width)
        # each expert's output, then mixed: a weight matrix per input would be the batch times
        # larger than these, which every input shares
        weight = jnp.pad(self.weight, ((0, 0), (0, 0), (0, 1)))[:, :, index]
        y = jnp.einsum(
            "kcpq,cq->kcp",
            weight.astype(dtype),
            x.reshape(channels, height * width).astype(dtype),
        )
        y = jnp.einsum("k,kcp->cp", self.mixing(x).astype(dtype), y)
        return y.reshape(channels, height, width), state


class GatedScan(eqx.Module):
    """A depthwise filter along lines that the board can block: from each square, `length` squares
    each way along each of `steps` (dy, dx) directions, as `Depthwise`'s taps along them would, but
    each tap's weight multiplied by the gates of the squares between it and the square filtered:
    y[c, p] = w[c, 0] x[c, p] + sum over directions d and distances k of
    w[c, d, k] * g[p + d] ... g[p + (k - 1) d] * x[c, p + k d].
    A gate is in (0, 1), a sigmoid of a 1x1 convolution of the input, normalized, so a line can
    stop at a square, e.g. an empty one (a selective scan along lines, as VMamba's along rows and
    columns). `gates` gate channels, each shared by a group of consecutive channels. Gates start
    near 1 (an offset of 2): the filter starts close to the plain filter along the lines."""

    weight: jax.Array
    gate_conv: eqx.nn.Conv2d
    gate_norm: Norm
    gate_offset: jax.Array
    steps: tuple[tuple[int, int], ...] = eqx.field(static=True)
    length: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        steps: tuple[tuple[int, int], ...],
        length: int,
        gates: int,
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
    ):
        if channels % gates:
            raise ValueError(f"{gates} gates don't divide {channels} channels")
        keys = jax.random.split(key, 2)
        self.steps = tuple((int(dy), int(dx)) for dy, dx in steps)
        self.length = length
        taps = 1 + len(steps) * length
        bound = 1 / math.sqrt(taps)
        self.weight = jax.random.uniform(
            keys[0], (channels, taps), minval=-bound, maxval=bound
        )
        self.gate_conv = conv(channels, gates, 1, keys[1])
        self.gate_norm = make_norm(gates, stats, 1)
        self.gate_offset = jnp.full((gates,), 2.0)

    @property
    def kernel_size(self) -> int:
        return 2 * self.length + 1

    def gates(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        """Each gate channel's gate at each square, (gates, height, width), float32."""
        g, state = conv_norm(self.gate_conv, self.gate_norm, x, state, dtype)
        offset = self.gate_offset.reshape(-1, 1, 1)
        return jax.nn.sigmoid(g.astype(jnp.float32) + offset), state

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        channels, height, width = x.shape
        g, state = self.gates(x, state, dtype)
        g = jnp.repeat(g, channels // g.shape[0], axis=0)
        n = self.length
        pad = ((0, 0), (n, n), (n, n))
        xf = x.astype(jnp.float32)
        xp, gp = jnp.pad(xf, pad), jnp.pad(g, pad)
        w = self.weight.astype(jnp.float32)

        def shifted(a: jax.Array, dy: int, dx: int) -> jax.Array:
            # a[c, i + dy, j + dx], zero off the board
            return a[:, n + dy : n + dy + height, n + dx : n + dx + width]

        y = w[:, 0, None, None] * xf
        for d, (dy, dx) in enumerate(self.steps):
            through = jnp.ones_like(xf)
            for k in range(1, n + 1):
                tap = w[:, 1 + d * n + k - 1, None, None]
                y = y + tap * through * shifted(xp, k * dy, k * dx)
                if k < n:
                    through = through * shifted(gp, k * dy, k * dx)
        return y.astype(dtype), state


@functools.cache
def offset_index(height: int, width: int) -> np.ndarray:
    """(squares, squares): the index of square q's offset from square p, among the
    (2 height - 1) (2 width - 1) offsets, row by row."""
    rows, cols = np.divmod(np.arange(height * width), width)
    dy = rows[None, :] - rows[:, None] + height - 1
    dx = cols[None, :] - cols[:, None] + width - 1
    return dy * (2 * width - 1) + dx


class BoardAttention(eqx.Module):
    """Multi-head self-attention between the squares, as a block's step between squares (in place
    of a depthwise filter: a MetaFormer block, attention then the block's MLP): the input
    normalized, a 1x1 convolution to queries, keys and values, and each head's attention logits
    biased by a learned bias for each offset between two squares (relative position bias, Swin's;
    zero at the start). The heads' outputs are concatenated, with no projection: the block's MLP
    follows."""

    norm: Norm
    qkv: eqx.nn.Conv2d
    bias: jax.Array
    heads: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        heads: int,
        board: tuple[int, int],
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
    ):
        if channels % heads:
            raise ValueError(f"{heads} heads don't divide {channels} channels")
        height, width = board
        self.norm = make_norm(channels, stats, 1)
        self.qkv = conv(channels, 3 * channels, 1, key)
        self.bias = jnp.zeros((heads, (2 * height - 1) * (2 * width - 1)))
        self.heads = heads

    @property
    def kernel_size(self) -> int:
        # the offsets span 2 height - 1 rows, as a filter that reaches the whole board would
        return math.isqrt(self.bias.shape[1])

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        channels, height, width = x.shape
        h, state = norm_act(self.norm, x, state, dtype, relu=False)
        qkv = convolve(self.qkv, h, dtype).astype(jnp.float32)
        q, k, v = qkv.reshape(3, self.heads, channels // self.heads, height * width)
        logits = jnp.einsum("hdp,hdq->hpq", q, k) / math.sqrt(q.shape[1])
        logits = logits + self.bias[:, offset_index(height, width)]
        attention = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("hpq,hdq->hdp", attention, v)
        return y.reshape(channels, height, width).astype(dtype), state


Mixer = Depthwise | BranchedDepthwise | DynamicDepthwise | GatedScan | BoardAttention


class PreActConvNeXt(eqx.Module):
    """ConvNeXt's block: x + project(relu(expand(norm(depthwise(x))))), a depthwise filter of the
    trunk itself (see `Depthwise`), one normalization, then a per-square MLP of 1x1 convolutions
    through `hidden` channels. With global pooling, the last `pool_channels` of the normalized
    features, pooled over the board, bias the hidden channels. project starts at zero (see
    `zeroed`). Optionally normalized after expand and after project too, `expand_norm` and
    `project_norm`: x + project_norm(project(relu(expand_norm(expand(y)) + pool bias))),
    project_norm's scale then starting at zero instead (see `zero_scale`).

    Options (see `AZResnetConfig`): another step between squares in place of the depthwise filter
    (`mixer`); a learned bias for each square and channel added to its output, before the
    normalization (`position`); and expand and project each a mixture of `mlp_experts`
    convolutions, mixed for each input by a softmax of a linear map of the normalized features
    pooled over the board (CondConv), the router starting at zero."""

    dw: Mixer
    norm: Norm
    expand: eqx.nn.Conv2d
    expand_norm: Norm | None
    pool_linear: eqx.nn.Linear | None
    project: eqx.nn.Conv2d
    project_norm: Norm | None
    position: jax.Array | None
    mlp_router: eqx.nn.Linear | None
    expand_experts: jax.Array | None
    project_experts: jax.Array | None
    pool_channels: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        hidden: int,
        taps: tuple[tuple[int, int], ...],
        pool_channels: int = 0,
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
        expand_norm: bool = False,
        project_norm: bool = False,
        mixer: Mixer | None = None,
        position: tuple[int, int] | None = None,
        mlp_experts: int = 0,
    ):
        """
        Args:
            channels: number of channels
            hidden: the MLP's hidden channels
            taps: the depthwise filters' offsets (see `Depthwise`)
            pool_channels: normalized channels pooled over the board to bias the hidden ones, 0
                for no global pooling bias
            key: rng used to initialize parameters
            make_norm: makes the normalizations (see `norm_maker`)
            stats: unused: the normalizations need none
            expand_norm: normalize the hidden channels before their ReLU
            project_norm: normalize the output, its scale starting at zero
            mixer: the step between squares, in place of a depthwise filter of `taps`
            position: the board's (height, width), for a learned bias per square, starting at 0
            mlp_experts: expand and project as mixtures of this many convolutions each (0 or 1:
                one each)
        """
        keys = jax.random.split(key, 4)
        self.dw = Depthwise(channels, taps, key=keys[0]) if mixer is None else mixer
        self.norm = make_norm(channels, stats, self.dw.kernel_size)
        self.expand = conv(channels, hidden, 1, keys[1])
        self.expand_norm = make_norm(hidden, stats, 1) if expand_norm else None
        self.pool_linear = (
            eqx.nn.Linear(2 * pool_channels, hidden, key=keys[2])
            if pool_channels
            else None
        )
        self.project = conv(hidden, channels, 1, keys[3])
        if project_norm:
            self.project_norm = zero_scale(make_norm(channels, stats, 1))
        else:
            self.project, self.project_norm = zeroed(self.project), None
        self.pool_channels = pool_channels
        self.position = None if position is None else jnp.zeros((channels, *position))
        self.mlp_router = self.expand_experts = self.project_experts = None
        if mlp_experts > 1:
            # more keys from the block's own, so that the other parameters start as without them
            more = jax.random.split(jax.random.fold_in(key, 1), 2 * mlp_experts - 1)
            self.mlp_router = zero_linear(
                eqx.nn.Linear(2 * channels, mlp_experts, key=more[0])
            )
            self.expand_experts = jnp.stack(
                [conv(channels, hidden, 1, k).weight for k in more[1:mlp_experts]]
            )
            project_experts = jnp.stack(
                [conv(hidden, channels, 1, k).weight for k in more[mlp_experts:]]
            )
            # a zeroed project's experts start at zero too: the branch starts adding nothing
            self.project_experts = (
                project_experts if project_norm else jnp.zeros_like(project_experts)
            )

    @property
    def plain(self) -> bool:
        """Whether the block is ConvNeXt's with a depthwise filter, without the options (as the
        native and ONNX exports need)."""
        return (
            isinstance(self.dw, Depthwise)
            and self.position is None
            and self.mlp_router is None
        )

    def experts(self, y: jax.Array) -> jax.Array | None:
        """With `mlp_experts`, the experts' weights for the input whose normalized features are
        `y`, (experts,), summing to 1."""
        if self.mlp_router is None:
            return None
        return jax.nn.softmax(self.mlp_router(board_pool(y)))

    @staticmethod
    def mixture(
        c: eqx.nn.Conv2d,
        experts: jax.Array | None,
        weights: jax.Array | None,
        x: jax.Array,
    ) -> tuple[eqx.nn.Conv2d, jax.Array]:
        """The convolution `c`, mixed with its `experts` by `weights`, and its input `x`: as one
        convolution of the experts side by side, of the input scaled by each expert's weight,
        sum_k w_k W_k x = [W_1 ... W_K] [w_1 x; ...; w_K x]. Mixing the weights instead makes
        a convolution per input, about 40x slower in batched training."""
        if experts is None or weights is None:
            return c, x
        stacked = jnp.concatenate([c.weight, *experts], axis=1)
        x = jnp.concatenate([w * x for w in weights.astype(x.dtype)])
        return eqx.tree_at(lambda c: c.weight, c, stacked), x

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        if isinstance(self.dw, Depthwise):
            y = self.dw(x, dtype)
        else:
            y, state = self.dw(x, state, dtype)
        if self.position is not None:
            y = y + self.position.astype(y.dtype)
        y, state = norm_act(self.norm, y, state, dtype, relu=False)
        weights = self.experts(y)
        expand, z = self.mixture(self.expand, self.expand_experts, weights, y)
        if self.expand_norm is None:
            h = convolve(expand, z, dtype)
        else:
            h, state = conv_norm(expand, self.expand_norm, z, state, dtype)
        if self.pool_linear is not None:
            bias = self.pool_linear(board_pool(y[-self.pool_channels :]))
            h = h + bias.reshape(-1, 1, 1).astype(h.dtype)
        h = jax.nn.relu(h)
        project, h = self.mixture(self.project, self.project_experts, weights, h)
        if self.project_norm is None:
            y = convolve(project, h, dtype)
        else:
            y, state = conv_norm(project, self.project_norm, h, state, dtype)
        return x + y.astype(x.dtype), state


Block = (
    ResidualBlock | BottleneckBlock | PreActBlock | PreActBottleneck | PreActConvNeXt
)


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


def mixer(
    config: AZResnetConfig,
    attention: bool,
    taps: tuple[tuple[int, int], ...],
    board: tuple[int, int],
    key: jax.Array,
    make_norm: NormMaker,
    stats: Stats,
) -> Mixer | None:
    """A ConvNeXt block's step between squares as `config` sets it (with `attention`, attention's),
    or None for the plain depthwise filter of `taps`."""
    channels = config.num_channels
    chosen = [
        name
        for name, on in (
            ("dw_branches", config.dw_branches),
            ("dw_experts", config.dw_experts),
            ("scan_steps", config.scan_steps),
        )
        if on
    ]
    if len(chosen) > 1:
        raise ValueError(f"one step between squares, not {' and '.join(chosen)}")
    if attention:
        return BoardAttention(
            channels,
            config.attention_heads,
            board,
            key=key,
            make_norm=make_norm,
            stats=stats,
        )
    if config.dw_branches:
        return BranchedDepthwise(
            channels, config.dw_branches, key=key, make_norm=make_norm, stats=stats
        )
    if config.dw_experts > 1:
        return DynamicDepthwise(channels, taps, config.dw_experts, key=key)
    if config.scan_steps:
        return GatedScan(
            channels,
            config.scan_steps,
            config.scan_length or max(board) - 1,
            config.scan_gates,
            key=key,
            make_norm=make_norm,
            stats=stats,
        )
    return None


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
    board_plane: bool = eqx.field(static=True)
    input_features: Callable[[jax.Array], jax.Array] | None = eqx.field(static=True)

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
        # by default, a ConvNeXt trunk is normalized after every convolution
        convnext = config.block == "convnext"
        stem_norm = convnext if config.stem_norm is None else config.stem_norm
        expand_norm = convnext if config.expand_norm is None else config.expand_norm
        project_norm = convnext if config.project_norm is None else config.project_norm
        if config.stem_norm and not pre:
            raise ValueError(
                'stem_norm is the "pre" layout\'s option: "post" always has it'
            )
        if (expand_norm or project_norm) and not convnext:
            raise ValueError(
                'expand_norm and project_norm are block="convnext"\'s options'
            )
        convnext_options = (
            config.dw_branches,
            config.dw_experts,
            config.scan_steps,
            config.attention_last,
            config.position_bias,
            config.mlp_experts,
        )
        if any(convnext_options) and not convnext:
            raise ValueError(
                "dw_branches, dw_experts, scan_steps, attention_last, position_bias and "
                'mlp_experts are block="convnext"\'s options'
            )
        more_inputs = 1 if config.board_plane else 0
        if config.input_features is not None:
            features = jax.eval_shape(
                config.input_features,
                jax.ShapeDtypeStruct(input_shape, jnp.float32),
            )
            more_inputs += features.shape[-1]
        self.stem_conv = conv(in_channels + more_inputs, channels, 3, stem_key)
        self.stem_norm = (
            make_norm(channels, INPUT_STATS, 3) if stem_norm or not pre else None
        )

        # residual blocks: the global pooling channels of each basic block, counting those inside
        # bottleneck blocks
        inner = config.bottleneck_inner if config.block == "bottleneck" else 1
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
        if config.block == "convnext":
            if not pre:
                raise ValueError('block="convnext" needs layout="pre"')
            taps = config.dw_taps or square_taps(config.dw_kernel)
            self.blocks = [
                PreActConvNeXt(
                    channels,
                    config.expand * channels,
                    taps,
                    pool,
                    key=k,
                    make_norm=make_norm,
                    stats=block_stats,
                    expand_norm=expand_norm,
                    project_norm=project_norm,
                    mixer=mixer(
                        config,
                        i >= config.num_blocks - config.attention_last,
                        taps,
                        (height, width),
                        jax.random.fold_in(k, 2),
                        make_norm,
                        block_stats,
                    ),
                    position=(height, width) if config.position_bias else None,
                    mlp_experts=config.mlp_experts,
                )
                for i, (pool, k, block_stats) in enumerate(
                    zip(pool_channels, block_keys, stats[:-1], strict=True)
                )
            ]
        elif config.block == "basic":
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
        self.board_plane = config.board_plane
        self.input_features = config.input_features

    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State]:
        # the convolutions compute in the inference dtype in inference mode (and so do the activations
        # between them), in the train dtype in training
        dtype = jnp.dtype(self.inference_dtype if self.inference else self.train_dtype)
        planes = [x]
        if self.input_features is not None:
            planes.append(self.input_features(x))
        if self.board_plane:
            planes.append(jnp.ones((*x.shape[:-1], 1)))
        if len(planes) > 1:
            x = jnp.concatenate([p.astype(jnp.float32) for p in planes], axis=-1)
        # inputs are channels-last (and may be e.g. boolean), equinox convolutions are channels-first
        x = jnp.moveaxis(x, -1, 0).astype(dtype)
        # initial conv layer
        if self.stem_norm is None:
            # the pre-activation trunk, in float32 in training as the post-activation one is
            x = convolve(self.stem_conv, x, dtype)
            x = x if self.inference else x.astype(jnp.float32)
        else:
            x, state = conv_norm(self.stem_conv, self.stem_norm, x, state, dtype)
            # the pre-activation trunk starts from the normalized stem as it is
            x = x if self.final_norm is not None else jax.nn.relu(x)

        # residual blocks
        for block in self.blocks:
            if self.remat and not self.inference:
                x, state = checkpointed(block, x, state, dtype)
            else:
                x, state = block(x, state, dtype)

        if self.final_norm is not None:
            x, state = norm_act(self.final_norm, x, state, dtype)
        return self.heads(x, state, dtype)
