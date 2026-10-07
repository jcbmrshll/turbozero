from dataclasses import dataclass
from typing import List, Tuple

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
    """
    policy_head_out_size: int
    num_blocks: int
    num_channels: int


def conv(in_channels: int, out_channels: int, kernel_size: int, key: jax.Array) -> eqx.nn.Conv2d:
    return eqx.nn.Conv2d(in_channels, out_channels, kernel_size, padding='SAME', use_bias=False, key=key)


def batch_norm(channels: int) -> eqx.nn.BatchNorm:
    return eqx.nn.BatchNorm(channels, axis_name=BATCH_AXIS, mode='batch')


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

    def __call__(self, x: jax.Array, state: eqx.nn.State) -> Tuple[jax.Array, eqx.nn.State]:
        y, state = self.bn1(self.conv1(x), state)
        y = jax.nn.relu(y)
        y, state = self.bn2(self.conv2(y), state)
        return jax.nn.relu(x + y), state


class AZResnet(eqx.Module):
    """Implements the AlphaZero ResNet model.

    Uses BatchNorm, so create it with its state: `eqx.nn.make_with_state(AZResnet)(config, input_shape, key=key)`
    """
    stem_conv: eqx.nn.Conv2d
    stem_bn: eqx.nn.BatchNorm
    blocks: List[ResidualBlock]
    policy_conv: eqx.nn.Conv2d
    policy_bn: eqx.nn.BatchNorm
    policy_linear: eqx.nn.Linear
    value_conv: eqx.nn.Conv2d
    value_bn: eqx.nn.BatchNorm
    value_linear: eqx.nn.Linear

    def __init__(self, config: AZResnetConfig, input_shape: Tuple[int, int, int], *, key: jax.Array):
        """
        Args:
            config: network configuration
            input_shape: shape of a single (unbatched) input, (height, width, channels)
            key: rng used to initialize parameters
        """
        height, width, in_channels = input_shape
        keys = jax.random.split(key, config.num_blocks + 5)
        # initial conv layer
        self.stem_conv = conv(in_channels, config.num_channels, 3, keys[0])
        self.stem_bn = batch_norm(config.num_channels)
        # residual blocks
        self.blocks = [ResidualBlock(config.num_channels, key=k) for k in keys[5:]]
        # policy head
        self.policy_conv = conv(config.num_channels, 2, 1, keys[1])
        self.policy_bn = batch_norm(2)
        self.policy_linear = eqx.nn.Linear(2 * height * width, config.policy_head_out_size, key=keys[2])
        # value head
        self.value_conv = conv(config.num_channels, 1, 1, keys[3])
        self.value_bn = batch_norm(1)
        self.value_linear = eqx.nn.Linear(height * width, 1, key=keys[4])

    def __call__(self, x: jax.Array, state: eqx.nn.State) -> Tuple[Tuple[jax.Array, jax.Array], eqx.nn.State]:
        # inputs are channels-last (and may be e.g. boolean), equinox convolutions are channels-first
        x = jnp.moveaxis(x, -1, 0).astype(self.stem_conv.weight.dtype)
        # initial conv layer
        x, state = self.stem_bn(self.stem_conv(x), state)
        x = jax.nn.relu(x)

        # residual blocks
        for block in self.blocks:
            x, state = block(x, state)

        # policy head
        policy, state = self.policy_bn(self.policy_conv(x), state)
        policy = jax.nn.relu(policy)
        policy = self.policy_linear(policy.reshape(-1))

        # value head
        value, state = self.value_bn(self.value_conv(x), state)
        value = jax.nn.relu(value)
        value = self.value_linear(value.reshape(-1))
        value = jnp.tanh(value)

        return (policy, value), state
