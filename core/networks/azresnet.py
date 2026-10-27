from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

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
    remat: bool = False


def conv(
    in_channels: int, out_channels: int, kernel_size: int, key: jax.Array
) -> eqx.nn.Conv2d:
    return eqx.nn.Conv2d(
        in_channels, out_channels, kernel_size, padding="SAME", use_bias=False, key=key
    )


def batch_norm(channels: int) -> eqx.nn.BatchNorm:
    """BatchNorm whose running averages of the batch statistics both start at 0.

    Equinox (0.13) starts the running variance at 1, but divides both running averages by
    1 - momentum^steps, as if they started at 0 (the zero-debiasing of Adam's moments): in inference
    the variance was too large by momentum^steps / (1 - momentum^steps), 99 times after one training
    step, a third too large after 128, and only negligibly so after about 1000.
    """
    bn = eqx.nn.BatchNorm(channels, axis_name=BATCH_AXIS, mode="batch")
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


def conv_bn(
    conv: eqx.nn.Conv2d,
    bn: eqx.nn.BatchNorm,
    x: jax.Array,
    state: eqx.nn.State,
    dtype: jnp.dtype,
) -> tuple[jax.Array, eqx.nn.State]:
    """A convolution followed by BatchNorm. The convolution computes in `dtype`.

    In training, BatchNorm normalizes in float32 (its batch statistics are too coarse in bfloat16).
    In inference mode BatchNorm is a fixed per-channel affine map (see `inference_affine`), so it's
    folded into the convolution's weights and a bias: one pass over the activations instead of two (or
    more).
    """
    if not bn.inference:
        cast = eqx.tree_at(lambda c: c.weight, conv, conv.weight.astype(dtype))
        return bn(cast(x.astype(dtype)).astype(jnp.float32), state)
    scale, offset = inference_affine(bn, state)
    weight = conv.weight * scale.reshape(-1, 1, 1, 1)
    folded = eqx.tree_at(lambda c: c.weight, conv, weight.astype(dtype))
    return folded(x.astype(dtype)) + offset.reshape(-1, 1, 1).astype(dtype), state


def board_pool(x: jax.Array) -> jax.Array:
    """KataGo's global pooling of (channels, height, width) features: each channel's mean and
    maximum over the board, (2 * channels,), in float32."""
    x = x.astype(jnp.float32)
    return jnp.concatenate([x.mean(axis=(1, 2)), x.max(axis=(1, 2))])


class ResidualBlock(eqx.Module):
    """Residual block for AlphaZero ResNet model, optionally with KataGo's global pooling bias (see
    `AZResnetConfig.pool_every`)."""

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    pool_linear: eqx.nn.Linear | None
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    pool_channels: int = eqx.field(static=True)

    def __init__(self, channels: int, pool_channels: int = 0, *, key: jax.Array):
        """
        Args:
            channels: number of channels
            pool_channels: channels of the first convolution's output pooled over the board to bias
                the others, 0 for no global pooling bias
            key: rng used to initialize parameters
        """
        key1, key2, key3 = jax.random.split(key, 3)
        self.conv1 = conv(channels, channels, 3, key1)
        self.bn1 = batch_norm(channels)
        self.pool_linear = (
            eqx.nn.Linear(2 * pool_channels, channels - pool_channels, key=key3)
            if pool_channels
            else None
        )
        self.conv2 = conv(channels - pool_channels, channels, 3, key2)
        self.bn2 = batch_norm(channels)
        self.pool_channels = pool_channels

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        y, state = conv_bn(self.conv1, self.bn1, x, state, dtype)
        if self.pool_linear is not None:
            y, pooled = y[: -self.pool_channels], y[-self.pool_channels :]
            bias = self.pool_linear(board_pool(jax.nn.relu(pooled)))
            y = y + bias.reshape(-1, 1, 1).astype(y.dtype)
        y = jax.nn.relu(y)
        y, state = conv_bn(self.conv2, self.bn2, y, state, dtype)
        return jax.nn.relu(x + y), state


class BottleneckBlock(eqx.Module):
    """KataGo's nested bottleneck residual block: a 1x1 convolution down to a narrower width, basic
    residual blocks at that width, and a 1x1 convolution back up."""

    down_conv: eqx.nn.Conv2d
    down_bn: eqx.nn.BatchNorm
    inner: list[ResidualBlock]
    up_conv: eqx.nn.Conv2d
    up_bn: eqx.nn.BatchNorm

    def __init__(
        self,
        channels: int,
        inner_channels: int,
        inner_pool_channels: list[int],
        *,
        key: jax.Array,
    ):
        """
        Args:
            channels: number of channels in and out
            inner_channels: number of channels of the inner blocks
            inner_pool_channels: each inner block's global pooling channels (see `ResidualBlock`),
                one per inner block
            key: rng used to initialize parameters
        """
        keys = jax.random.split(key, len(inner_pool_channels) + 2)
        self.down_conv = conv(channels, inner_channels, 1, keys[0])
        self.down_bn = batch_norm(inner_channels)
        self.inner = [
            ResidualBlock(inner_channels, pool, key=k)
            for pool, k in zip(inner_pool_channels, keys[2:], strict=True)
        ]
        self.up_conv = conv(inner_channels, channels, 1, keys[1])
        self.up_bn = batch_norm(channels)

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        y, state = conv_bn(self.down_conv, self.down_bn, x, state, dtype)
        y = jax.nn.relu(y)
        for block in self.inner:
            y, state = block(y, state, dtype)
        y, state = conv_bn(self.up_conv, self.up_bn, y, state, dtype)
        return jax.nn.relu(x + y), state


Block = ResidualBlock | BottleneckBlock


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
    policy_bn: eqx.nn.BatchNorm
    policy_linear: eqx.nn.Linear
    value_conv: eqx.nn.Conv2d
    value_bn: eqx.nn.BatchNorm
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
    ):
        keys = jax.random.split(key, 5)
        self.policy_conv = conv(channels, 2, 1, keys[0])
        self.policy_bn = batch_norm(2)
        self.policy_linear = eqx.nn.Linear(2 * squares, num_actions, key=keys[1])
        # as AlphaZero's: a 1x1 convolution to one channel, then a hidden layer
        self.value_conv = conv(channels, 1, 1, keys[2])
        self.value_bn = batch_norm(1)
        self.value_hidden = eqx.nn.Linear(squares, value_hidden_size, key=keys[3])
        self.value_linear = eqx.nn.Linear(value_hidden_size, 1, key=keys[4])

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State]:
        policy, state = conv_bn(self.policy_conv, self.policy_bn, x, state, dtype)
        policy = jax.nn.relu(policy)
        policy = self.policy_linear(policy.reshape(-1).astype(jnp.float32))

        value, state = conv_bn(self.value_conv, self.value_bn, x, state, dtype)
        value = jax.nn.relu(value)
        value = jax.nn.relu(self.value_hidden(value.reshape(-1).astype(jnp.float32)))
        value = jnp.tanh(self.value_linear(value))
        return (policy, value), state


class PooledHeads(eqx.Module):
    """KataGo's policy and value heads, on features pooled over the board (see
    `AZResnetConfig.heads`)."""

    policy_conv: eqx.nn.Conv2d
    policy_bn: eqx.nn.BatchNorm
    policy_pool_conv: eqx.nn.Conv2d
    policy_pool_bn: eqx.nn.BatchNorm
    policy_pool_linear: eqx.nn.Linear
    policy_square_conv: eqx.nn.Conv2d
    pass_linear: eqx.nn.Linear
    value_conv: eqx.nn.Conv2d
    value_bn: eqx.nn.BatchNorm
    value_hidden: eqx.nn.Linear
    value_linear: eqx.nn.Linear

    def __init__(
        self,
        channels: int,
        head_channels: int,
        value_hidden_size: int,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 8)
        self.policy_conv = conv(channels, head_channels, 1, keys[0])
        self.policy_bn = batch_norm(head_channels)
        self.policy_pool_conv = conv(channels, head_channels, 1, keys[1])
        self.policy_pool_bn = batch_norm(head_channels)
        self.policy_pool_linear = eqx.nn.Linear(
            2 * head_channels, head_channels, key=keys[2]
        )
        self.policy_square_conv = conv(head_channels, 1, 1, keys[3])
        self.pass_linear = eqx.nn.Linear(2 * head_channels, 1, key=keys[4])
        self.value_conv = conv(channels, head_channels, 1, keys[5])
        self.value_bn = batch_norm(head_channels)
        self.value_hidden = eqx.nn.Linear(
            2 * head_channels, value_hidden_size, key=keys[6]
        )
        self.value_linear = eqx.nn.Linear(value_hidden_size, 1, key=keys[7])

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State]:
        policy, state = conv_bn(self.policy_conv, self.policy_bn, x, state, dtype)
        pooled, state = conv_bn(
            self.policy_pool_conv, self.policy_pool_bn, x, state, dtype
        )
        pooled = board_pool(jax.nn.relu(pooled))
        bias = self.policy_pool_linear(pooled)
        policy = jax.nn.relu(policy + bias.reshape(-1, 1, 1).astype(policy.dtype))
        squares = self.policy_square_conv(policy.astype(jnp.float32)).reshape(-1)
        policy = jnp.concatenate([squares, self.pass_linear(pooled)])

        value, state = conv_bn(self.value_conv, self.value_bn, x, state, dtype)
        value = board_pool(jax.nn.relu(value))
        value = jax.nn.relu(self.value_hidden(value))
        value = jnp.tanh(self.value_linear(value))
        return (policy, value), state


class AZResnet(eqx.Module):
    """Implements the AlphaZero ResNet model, and KataGo's variants of it (see `AZResnetConfig`).

    Uses BatchNorm, so create it with its state: `eqx.nn.make_with_state(AZResnet)(config, input_shape, key=key)`
    """

    stem_conv: eqx.nn.Conv2d
    stem_bn: eqx.nn.BatchNorm
    blocks: list[Block]
    heads: AlphaZeroHeads | PooledHeads
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
        # initial conv layer
        self.stem_conv = conv(in_channels, channels, 3, stem_key)
        self.stem_bn = batch_norm(channels)

        # residual blocks: the global pooling channels of each basic block, counting those inside
        # bottleneck blocks
        inner = 1 if config.block == "basic" else config.bottleneck_inner
        pool_channels = [
            config.pool_channels
            if config.pool_every and (i + 1) % config.pool_every == 0
            else 0
            for i in range(config.num_blocks * inner)
        ]
        if config.block == "basic":
            self.blocks = [
                ResidualBlock(channels, pool, key=k)
                for pool, k in zip(pool_channels, block_keys, strict=True)
            ]
        else:
            self.blocks = [
                BottleneckBlock(
                    channels,
                    config.bottleneck_channels or channels // 2,
                    pool_channels[i * inner : (i + 1) * inner],
                    key=k,
                )
                for i, k in enumerate(block_keys)
            ]

        if config.heads == "alphazero":
            self.heads = AlphaZeroHeads(
                channels,
                height * width,
                config.policy_head_out_size,
                config.value_hidden_size,
                key=heads_key,
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
            )
        self.inference_dtype = config.inference_dtype
        self.train_dtype = config.train_dtype
        self.remat = config.remat

    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[tuple[jax.Array, jax.Array], eqx.nn.State]:
        # the convolutions compute in the inference dtype in inference mode (and so do the activations
        # between them), in the train dtype in training
        dtype = jnp.dtype(
            self.inference_dtype if self.stem_bn.inference else self.train_dtype
        )
        # inputs are channels-last (and may be e.g. boolean), equinox convolutions are channels-first
        x = jnp.moveaxis(x, -1, 0).astype(dtype)
        # initial conv layer
        x, state = conv_bn(self.stem_conv, self.stem_bn, x, state, dtype)
        x = jax.nn.relu(x)

        # residual blocks
        for block in self.blocks:
            if self.remat and not self.stem_bn.inference:
                x, state = checkpointed(block, x, state, dtype)
            else:
                x, state = block(x, state, dtype)

        return self.heads(x, state, dtype)
