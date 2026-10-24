import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.networks.azresnet import (
    AZResnet,
    AZResnetConfig,
    batch_norm,
    conv,
    conv_bn,
    inference_affine,
)
from core.networks.utils import BATCH_AXIS, apply_nn

INPUT_SHAPE = (3, 3, 2)


def with_statistics(bn: eqx.nn.BatchNorm, state, mean, var, steps=3):
    """`state` with `bn`'s running averages as they are after `steps` training batches whose
    statistics averaged `mean` and `var`: zero-started, so short of them by 1 - momentum^steps."""
    assert bn.batch_counter is not None and bn.batch_state_index is not None
    debias = 1 - bn.momentum**steps
    counter = state.get(bn.batch_counter)
    state = state.set(bn.batch_counter, jnp.zeros_like(counter) + steps)
    return state.set(
        bn.batch_state_index,
        (
            jnp.asarray(mean * debias, dtype=jnp.float32),
            jnp.asarray(var * debias, dtype=jnp.float32),
        ),
    )


def random_statistics(rng: np.random.Generator, channels: int):
    """A random mean and variance for each channel, the variances from 0.5 to 2."""
    return rng.normal(size=channels), rng.uniform(0.5, 2.0, size=channels)


def with_affine(bn: eqx.nn.BatchNorm, rng: np.random.Generator):
    """`bn` with a random channelwise weight and bias, rather than 1 and 0."""
    assert bn.weight is not None
    weight, bias = rng.normal(size=(2, *bn.weight.shape)).astype(np.float32)
    return eqx.tree_at(lambda bn: (bn.weight, bn.bias), bn, (1 + weight, bias))


@pytest.mark.parametrize("dtype, tolerance", [("float32", 1e-5), ("bfloat16", 5e-2)])
def test_inference_folds_batch_norm_into_the_convolution(dtype, tolerance):
    rng = np.random.default_rng(0)
    convolution = conv(2, 3, 3, jax.random.PRNGKey(0))
    bn, state = eqx.nn.make_with_state(batch_norm)(3)
    bn = eqx.nn.inference_mode(with_affine(bn, rng))
    mean, var = random_statistics(rng, 3)
    state = with_statistics(bn, state, mean, var)
    x = jnp.asarray(rng.normal(size=(2, 4, 4)), dtype=jnp.float32)

    y, _ = conv_bn(convolution, bn, x, state, jnp.dtype(dtype))

    def channelwise(a):
        return np.asarray(a)[:, None, None]

    expected = (np.asarray(convolution(x)) - channelwise(mean)) / np.sqrt(
        channelwise(var) + bn.eps
    ) * channelwise(bn.weight) + channelwise(bn.bias)
    assert y.dtype == jnp.dtype(dtype)
    np.testing.assert_allclose(
        np.asarray(y, dtype=np.float32), expected, atol=tolerance, rtol=tolerance
    )


def make_net(inference_dtype="float32", remat=False, train_dtype="float32"):
    """A tiny AZResnet; the same parameters whatever its static options."""
    config = AZResnetConfig(
        policy_head_out_size=5,
        num_blocks=1,
        num_channels=4,
        inference_dtype=inference_dtype,
        train_dtype=train_dtype,
        remat=remat,
    )
    return eqx.nn.make_with_state(AZResnet)(
        config, INPUT_SHAPE, key=jax.random.PRNGKey(0)
    )


def batch_norms(net: AZResnet) -> list[eqx.nn.BatchNorm]:
    return [net.stem_bn, net.policy_bn, net.value_bn] + [
        bn for block in net.blocks for bn in (block.bn1, block.bn2)
    ]


def observations(rng: np.random.Generator, batch_size=4):
    return rng.random((batch_size, *INPUT_SHAPE)) < 0.5


def unfolded(net: AZResnet, state, x):
    """Inference without folding: each convolution, then equinox's BatchNorm."""

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


def test_inference_folds_each_batch_norm_into_its_convolution():
    net32, state = make_net("float32")
    net16, _ = make_net("bfloat16")
    rng = np.random.default_rng(0)
    # every BatchNorm with statistics of its own
    for bn in batch_norms(net32):
        assert bn.weight is not None
        state = with_statistics(bn, state, *random_statistics(rng, len(bn.weight)))
    x = observations(rng)

    @eqx.filter_jit
    def outputs(net32, net16, state, x):
        folded32, _ = apply_nn(eqx.nn.inference_mode(net32), state, x)
        folded16, _ = apply_nn(eqx.nn.inference_mode(net16), state, x)
        return folded32, folded16, jax.vmap(lambda x: unfolded(net32, state, x))(x)

    folded32, folded16, expected = outputs(net32, net16, state, x)

    for folded, tolerance in ((folded32, 1e-5), (folded16, 5e-2)):
        for a, b in zip(folded, expected, strict=True):
            # the heads are float32 whatever the inference dtype
            assert a.dtype == jnp.float32
            np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


def test_static_options_do_not_change_training():
    # inference_dtype only applies in inference; remat only changes what the backward pass
    # computes again, so gives the same gradients up to rounding, and leaves inference alone
    def loss(net, state, x):
        (policy, value), state = apply_nn(net, state, x)
        return (policy**2).mean() + (value**2).mean(), state

    @eqx.filter_jit
    def run(nets, state, x):
        grad_fn = eqx.filter_value_and_grad(loss, has_aux=True)
        return [
            (grad_fn(net, state, x), apply_nn(eqx.nn.inference_mode(net), state, x))
            for net in nets
        ]

    plain, state = make_net()
    nets = (plain, make_net("bfloat16")[0], make_net(remat=True)[0])
    results = run(nets, state, observations(np.random.default_rng(0)))
    [(trained, inference), (trained16, _), (trained_remat, inference_remat)] = [
        (jax.tree.leaves(training), jax.tree.leaves(inference))
        for training, inference in results
    ]

    for a, b in zip(trained, trained16, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(trained, trained_remat, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
    for a, b in zip(inference, inference_remat, strict=True):
        np.testing.assert_array_equal(a, b)


def test_mixed_precision_training_stays_close_to_float32():
    # bfloat16 convolutions: the loss, the gradients (float32, as the parameters are) and the
    # BatchNorm statistics (float32) within bfloat16's rounding of float32 training's
    def loss(net, state, x):
        (policy, value), state = apply_nn(net, state, x)
        return (policy**2).mean() + (value**2).mean(), state

    @eqx.filter_jit
    def run(net, state, x):
        (value, state), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            net, state, x
        )
        return value, grads, state

    net32, state = make_net()
    net16, _ = make_net(train_dtype="bfloat16")
    x = observations(np.random.default_rng(0), batch_size=16)
    loss32, grads32, state32 = run(net32, state, x)
    loss16, grads16, state16 = run(net16, state, x)

    np.testing.assert_allclose(loss16, loss32, rtol=2e-2)
    leaves16, leaves32 = jax.tree.leaves(grads16), jax.tree.leaves(grads32)
    assert all(a.dtype == b.dtype == jnp.float32 for a, b in zip(leaves16, leaves32))
    # bfloat16 rounding through the network: the whole gradient within about 10%, pointing the same way
    a, b = (
        np.concatenate([np.ravel(x) for x in leaves]) for leaves in (leaves16, leaves32)
    )
    assert np.linalg.norm(a - b) <= 0.15 * np.linalg.norm(b)
    assert a @ b >= 0.99 * np.linalg.norm(a) * np.linalg.norm(b)
    for a, b in zip(jax.tree.leaves(state16), jax.tree.leaves(state32), strict=True):
        assert a.dtype == b.dtype
        np.testing.assert_allclose(a, b, rtol=2e-2, atol=2e-2)


@eqx.filter_jit
def train_batch_norm(bn, state, x):
    """`bn` in training mode on the batch `x`: its outputs, normalized by the batch's statistics,
    and its state."""
    return jax.vmap(bn, axis_name=BATCH_AXIS, in_axes=(0, None), out_axes=(0, None))(
        x, state
    )


@pytest.mark.parametrize("steps", [1, 10])
def test_inference_matches_training_on_the_batch_it_trained_on(steps):
    # training on the same batch every step, the debiased running averages are that batch's
    # statistics from the first step on, so inference gives the outputs training mode does. Not
    # with equinox's own BatchNorm, whose running variance starts at 1 (see `batch_norm`)
    rng = np.random.default_rng(0)
    bn, state = eqx.nn.make_with_state(batch_norm)(3)
    bn = with_affine(bn, rng)
    # channels far from mean 0 and variance 1
    x = (3 + 0.1 * rng.normal(size=(8, 3, 2, 2))).astype(np.float32)
    training, state = train_batch_norm(bn, state, x)
    for _ in range(steps - 1):
        training, state = train_batch_norm(bn, state, x)

    scale, offset = inference_affine(bn, state)

    inference = x * scale[:, None, None] + offset[:, None, None]
    np.testing.assert_allclose(inference, training, rtol=1e-4, atol=1e-4)


def test_inference_before_any_training_leaves_batch_norm_inputs_as_they_are():
    # without statistics yet, inference normalizes by mean 0 and variance 1
    bn, state = eqx.nn.make_with_state(batch_norm)(3)

    scale, offset = inference_affine(bn, state)

    np.testing.assert_allclose(scale, np.full(3, 1 / np.sqrt(1 + bn.eps)), rtol=1e-6)
    np.testing.assert_array_equal(offset, np.zeros(3))
