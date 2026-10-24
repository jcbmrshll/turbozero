from dataclasses import dataclass

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


class ResidualBlock(eqx.Module):
    """Residual block for AlphaZero ResNet model."""

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm

    def __init__(self, channels: int, *, key: jax.Array):
        """
        Args:
            channels: number of channels
            key: rng used to initialize parameters
        """
        key1, key2 = jax.random.split(key)
        self.conv1 = conv(channels, channels, 3, key1)
        self.bn1 = batch_norm(channels)
        self.conv2 = conv(channels, channels, 3, key2)
        self.bn2 = batch_norm(channels)

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        y, state = conv_bn(self.conv1, self.bn1, x, state, dtype)
        y = jax.nn.relu(y)
        y, state = conv_bn(self.conv2, self.bn2, y, state, dtype)
        return jax.nn.relu(x + y), state


def checkpointed(
    block: ResidualBlock, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
) -> tuple[jax.Array, eqx.nn.State]:
    """Applies `block` under `jax.checkpoint`: differentiating it saves only its inputs, and computes
    its activations again in the backward pass."""
    params, static = eqx.partition(block, eqx.is_array)

    @jax.checkpoint
    def apply(
        params: ResidualBlock, x: jax.Array, state: eqx.nn.State
    ) -> tuple[jax.Array, eqx.nn.State]:
        return eqx.combine(params, static)(x, state, dtype)

    return apply(params, x, state)


class AZResnet(eqx.Module):
    """Implements the AlphaZero ResNet model.

    Uses BatchNorm, so create it with its state: `eqx.nn.make_with_state(AZResnet)(config, input_shape, key=key)`
    """

    stem_conv: eqx.nn.Conv2d
    stem_bn: eqx.nn.BatchNorm
    blocks: list[ResidualBlock]
    policy_conv: eqx.nn.Conv2d
    policy_bn: eqx.nn.BatchNorm
    policy_linear: eqx.nn.Linear
    value_conv: eqx.nn.Conv2d
    value_bn: eqx.nn.BatchNorm
    value_hidden: eqx.nn.Linear
    value_linear: eqx.nn.Linear
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
        keys = jax.random.split(key, config.num_blocks + 6)
        # initial conv layer
        self.stem_conv = conv(in_channels, config.num_channels, 3, keys[0])
        self.stem_bn = batch_norm(config.num_channels)
        # residual blocks
        self.blocks = [ResidualBlock(config.num_channels, key=k) for k in keys[6:]]
        # policy head
        self.policy_conv = conv(config.num_channels, 2, 1, keys[1])
        self.policy_bn = batch_norm(2)
        self.policy_linear = eqx.nn.Linear(
            2 * height * width, config.policy_head_out_size, key=keys[2]
        )
        # value head, as AlphaZero's: a 1x1 convolution to one channel, then a hidden layer
        self.value_conv = conv(config.num_channels, 1, 1, keys[3])
        self.value_bn = batch_norm(1)
        self.value_hidden = eqx.nn.Linear(
            height * width, config.value_hidden_size, key=keys[4]
        )
        self.value_linear = eqx.nn.Linear(config.value_hidden_size, 1, key=keys[5])
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

        # policy head
        policy, state = conv_bn(self.policy_conv, self.policy_bn, x, state, dtype)
        policy = jax.nn.relu(policy)
        policy = self.policy_linear(policy.reshape(-1).astype(jnp.float32))

        # value head
        value, state = conv_bn(self.value_conv, self.value_bn, x, state, dtype)
        value = jax.nn.relu(value)
        value = jax.nn.relu(self.value_hidden(value.reshape(-1).astype(jnp.float32)))
        value = self.value_linear(value)
        value = jnp.tanh(value)

        return (policy, value), state
