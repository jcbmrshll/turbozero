import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.networks.azresnet import AZResnet, AZResnetConfig
from core.networks.utils import apply_nn

INPUT_SHAPE = (4, 4, 2)


def make_net(inference_dtype="float32"):
    """A small AZResnet whose BatchNorm layers have seen a few training batches, so their
    running statistics aren't the identity."""
    config = AZResnetConfig(
        policy_head_out_size=17,
        num_blocks=2,
        num_channels=8,
        inference_dtype=inference_dtype,
    )
    net, state = eqx.nn.make_with_state(AZResnet)(
        config, INPUT_SHAPE, key=jax.random.PRNGKey(0)
    )
    for i in range(3):
        _, state = apply_nn(net, state, observations(jax.random.PRNGKey(i)))
    return net, state


def observations(key, batch_size=16):
    return jax.random.bernoulli(key, shape=(batch_size, *INPUT_SHAPE))


def unfolded(net: AZResnet, state, x):
    """Inference without folding: each convolution, then its BatchNorm."""

    def conv_bn(conv, bn, x):
        return bn(conv(x), state, inference=True)[0]

    x = jnp.moveaxis(x, -1, 0).astype(jnp.float32)
    x = jax.nn.relu(conv_bn(net.stem_conv, net.stem_bn, x))
    for block in net.blocks:
        y = jax.nn.relu(conv_bn(block.conv1, block.bn1, x))
        x = jax.nn.relu(x + conv_bn(block.conv2, block.bn2, y))
    policy = jax.nn.relu(conv_bn(net.policy_conv, net.policy_bn, x))
    value = jax.nn.relu(conv_bn(net.value_conv, net.value_bn, x))
    policy = net.policy_linear(policy.reshape(-1))
    value = jnp.tanh(net.value_linear(value.reshape(-1)))
    return policy, value


@pytest.mark.parametrize("dtype, tolerance", [("float32", 1e-5), ("bfloat16", 5e-2)])
def test_inference_folds_batch_norm_into_convolutions(dtype, tolerance):
    net, state = make_net(dtype)
    x = observations(jax.random.PRNGKey(10))
    (policy, value), _ = apply_nn(eqx.nn.inference_mode(net), state, x)
    expected_policy, expected_value = jax.vmap(lambda x: unfolded(net, state, x))(x)

    assert policy.dtype == value.dtype == jnp.float32
    np.testing.assert_allclose(policy, expected_policy, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(value, expected_value, atol=tolerance, rtol=tolerance)


def test_inference_dtype_does_not_change_training():
    net32, state = make_net("float32")
    net16, _ = make_net("bfloat16")
    x = observations(jax.random.PRNGKey(10))
    (policy32, value32), state32 = apply_nn(net32, state, x)
    (policy16, value16), state16 = apply_nn(net16, state, x)

    np.testing.assert_array_equal(policy32, policy16)
    np.testing.assert_array_equal(value32, value16)
    for a, b in zip(jax.tree.leaves(state32), jax.tree.leaves(state16)):
        np.testing.assert_array_equal(a, b)
