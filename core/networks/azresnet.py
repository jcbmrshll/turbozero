
from dataclasses import dataclass

import flax.linen as nn


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


class ResidualBlock(nn.Module):
    """Residual block for AlphaZero ResNet model.

    Attributes:
        channels: number of channels
    """
    channels: int

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Conv(features=self.channels, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(features=self.channels, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        return nn.relu(x + y)


class AZResnet(nn.Module):
    """Implements the AlphaZero ResNet model.

    Attributes:
        config: network configuration
    """
    config: AZResnetConfig

    @nn.compact
    def __call__(self, x, train: bool):
        # initial conv layer
        x = nn.Conv(features=self.config.num_channels, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # residual blocks
        for _ in range(self.config.num_blocks):
            x = ResidualBlock(channels=self.config.num_channels)(x, train=train)

        # policy head
        policy = nn.Conv(features=2, kernel_size=(1,1), strides=(1,1), padding='SAME', use_bias=False)(x)
        policy = nn.BatchNorm(use_running_average=not train)(policy)
        policy = nn.relu(policy)
        policy = policy.reshape((policy.shape[0], -1))
        policy = nn.Dense(features=self.config.policy_head_out_size)(policy)

        # value head
        value = nn.Conv(features=1, kernel_size=(1,1), strides=(1,1), padding='SAME', use_bias=False)(x)
        value = nn.BatchNorm(use_running_average=not train)(value)
        value = nn.relu(value)
        value = value.reshape((value.shape[0], -1))
        value = nn.Dense(features=1)(value)
        value = nn.tanh(value)

        return policy, value
