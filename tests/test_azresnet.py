import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.networks.azresnet import (
    AlphaZeroHeads,
    AZResnet,
    AZResnetConfig,
    BottleneckBlock,
    FixedScale,
    GroupNorm,
    ResidualBlock,
    batch_norm,
    conv,
    conv_norm,
    inference_affine,
    norm_maker,
    residual_stats,
)
from core.networks.utils import BATCH_AXIS, apply_nn, freeze_batchnorm

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

    y, _ = conv_norm(convolution, bn, x, state, jnp.dtype(dtype))

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
    assert isinstance(net.heads, AlphaZeroHeads)
    norms = [net.stem_norm, net.heads.policy_norm, net.heads.value_norm] + [
        norm
        for block in net.blocks
        if isinstance(block, ResidualBlock)
        for norm in (block.norm1, block.norm2)
    ]
    return [norm for norm in norms if isinstance(norm, eqx.nn.BatchNorm)]


def observations(rng: np.random.Generator, batch_size=4):
    return rng.random((batch_size, *INPUT_SHAPE)) < 0.5


def unfolded(net: AZResnet, state, x):
    """Inference without folding: each convolution, then equinox's BatchNorm."""

    def conv_bn(conv, bn, x):
        return bn(conv(x), state, inference=True)[0]

    x = jnp.moveaxis(x, -1, 0).astype(jnp.float32)
    x = jax.nn.relu(conv_bn(net.stem_conv, net.stem_norm, x))
    for block in net.blocks:
        assert isinstance(block, ResidualBlock)
        y = jax.nn.relu(conv_bn(block.conv1, block.norm1, x))
        x = jax.nn.relu(x + conv_bn(block.conv2, block.norm2, y))
    heads = net.heads
    assert isinstance(heads, AlphaZeroHeads)
    policy = jax.nn.relu(conv_bn(heads.policy_conv, heads.policy_norm, x))
    value = jax.nn.relu(conv_bn(heads.value_conv, heads.value_norm, x))
    policy = heads.policy_linear(policy.reshape(-1))
    value = jax.nn.relu(heads.value_hidden(value.reshape(-1)))
    value = jnp.tanh(heads.value_linear(value))
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


VARIANTS = {
    "global pooling": {
        "num_blocks": 2,
        "num_channels": 6,
        "pool_every": 2,
        "pool_channels": 2,
    },
    "bottleneck": {
        "num_blocks": 2,
        "num_channels": 8,
        "block": "bottleneck",
        "bottleneck_inner": 2,
        "pool_every": 3,
        "pool_channels": 2,
    },
    "pooled heads": {
        "num_blocks": 1,
        "num_channels": 4,
        "heads": "pooled",
        "head_channels": 3,
    },
}


def make_variant(name: str, **options):
    config = AZResnetConfig(policy_head_out_size=10, **{**VARIANTS[name], **options})
    return eqx.nn.make_with_state(AZResnet)(
        config, INPUT_SHAPE, key=jax.random.PRNGKey(0)
    )


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("dtype, tolerance", [("float32", 1e-4), ("bfloat16", 5e-2)])
def test_inference_matches_training_on_the_batch_a_variant_trained_on(
    name, dtype, tolerance
):
    # after one training step, each BatchNorm's (debiased) running averages are the statistics of
    # its inputs on that batch, so inference on it gives the outputs training gave: whatever the
    # blocks and heads, their BatchNorms fold into the convolutions they follow
    net, state = make_variant(name, inference_dtype=dtype)
    x = observations(np.random.default_rng(0), batch_size=8)

    @eqx.filter_jit
    def run(net, state, x):
        training, state = apply_nn(net, state, x)
        inference, _ = apply_nn(eqx.nn.inference_mode(net), state, x)
        return training, inference

    (policy, value), inference = run(net, state, x)

    assert policy.shape == (8, 10) and value.shape == (8, 1)
    for a, b in zip(inference, (policy, value), strict=True):
        assert a.dtype == jnp.float32
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


def test_the_bottleneck_variant_has_the_blocks_it_is_configured_with():
    net, _ = make_variant("bottleneck")

    # four basic blocks inside two bottleneck blocks, of half the width; every third has a
    # global pooling bias
    inner = [
        block
        for outer in net.blocks
        if isinstance(outer, BottleneckBlock)
        for block in outer.inner
    ]
    assert [block.conv1.in_channels for block in inner] == [4, 4, 4, 4]
    assert [block.pool_linear is not None for block in inner] == [
        False,
        False,
        True,
        False,
    ]


@pytest.mark.parametrize("pool_channels, reaches", [(0, False), (2, True)])
def test_global_pooling_carries_information_across_the_board(pool_channels, reaches):
    # a basic block's two 3x3 convolutions reach two squares away; with global pooling a change
    # in one corner of a 7x7 board reaches the opposite corner
    block, state = eqx.nn.make_with_state(ResidualBlock)(
        6,
        pool_channels,
        key=jax.random.PRNGKey(0),
        make_norm=norm_maker(board=(7, 7)),
        stats=residual_stats(0)[0],
    )
    block = eqx.nn.inference_mode(block)
    x = np.abs(np.random.default_rng(0).normal(size=(6, 7, 7))).astype(np.float32)
    y = x.copy()
    y[:, 0, 0] += 1

    def far_corner(x):
        return block(jnp.asarray(x), state, jnp.dtype("float32"))[0][:, 6, 6]

    assert np.allclose(far_corner(x), far_corner(y)) != reaches


def test_frozen_batch_norm_trains_with_the_stored_statistics():
    net, state = make_net()
    x = jnp.asarray(observations(np.random.default_rng(1)))
    # some statistics to freeze: a training step's
    _, state = apply_nn(net, state, x)
    frozen = freeze_batchnorm(net)

    (policy, value), after = apply_nn(frozen, state, x)
    (inf_policy, inf_value), _ = apply_nn(eqx.nn.inference_mode(net), state, x)

    # the state stays as it was, and the outputs are inference mode's
    for a, b in zip(jax.tree.leaves(after), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(policy, inf_policy, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(value, inf_value, rtol=1e-5, atol=1e-6)
    # and it still trains: BatchNorm's scales get gradients
    grads = eqx.filter_grad(lambda n: (apply_nn(n, state, x)[0][1] ** 2).mean())(frozen)
    assert np.abs(np.asarray(grads.stem_norm.weight)).sum() > 0
BATCH_INDEPENDENT = [
    *(("post", norm) for norm in ("layer", "group", "rms", "channel", "fixed", "ws")),
    *(("pre", norm) for norm in ("layer", "group", "rms", "channel")),
]


@pytest.mark.parametrize("layout, norm", BATCH_INDEPENDENT)
@pytest.mark.parametrize("name", VARIANTS)
def test_batch_independent_norms_train_as_they_infer_one_input_at_a_time(
    name, layout, norm
):
    # no state, and each input's outputs its own: the same in training as in inference, and
    # whatever else is in the batch
    net, state = make_variant(name, norm=norm, norm_groups=2, layout=layout)
    # random scales and offsets, and every weight nonzero (the pre-activation branches start at
    # zero), so that they're all checked
    rng = np.random.default_rng(0)
    params, static = eqx.partition(net, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda x: (
            x + (1 if x.ndim == 1 else 0.1) * rng.normal(size=x.shape).astype(x.dtype)
        ),
        params,
    )
    net = eqx.combine(params, static)
    x = observations(rng, batch_size=8)

    @eqx.filter_jit
    def run(net, state, x):
        training, _ = apply_nn(net, state, x)
        inference, _ = apply_nn(eqx.nn.inference_mode(net), state, x)
        alone, _ = apply_nn(net, state, x[:1])
        return training, inference, alone

    training, inference, alone = run(net, state, x)

    assert jax.tree.leaves(state) == []
    for a, b, c in zip(training, inference, alone, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(c, a[:1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "groups, center, over_board",
    [(1, True, True), (2, True, True), (1, False, True), (1, True, False)],
)
def test_group_norm_normalizes_each_group_over_the_board_or_each_square(
    groups, center, over_board
):
    rng = np.random.default_rng(0)
    norm = GroupNorm(4, groups, center, over_board)
    weight, bias = rng.normal(size=(2, 4)).astype(np.float32)
    norm = eqx.tree_at(lambda n: (n.weight, n.bias), norm, (weight, bias))
    x = (3 + 2 * rng.normal(size=(4, 3, 3))).astype(np.float32)

    # (groups, normalized together, apart)
    grouped = x.reshape(groups, -1, 1) if over_board else x.reshape(1, 4, -1)
    if center:
        grouped = grouped - grouped.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(grouped**2, axis=1, keepdims=True) + norm.eps)
    expected = (grouped / rms).reshape(x.shape) * weight[:, None, None]
    np.testing.assert_allclose(
        norm(jnp.asarray(x)), expected + bias[:, None, None], atol=1e-5
    )


def test_weight_standardization_ignores_the_scale_and_mean_of_the_weights():
    # each output channel's weights to mean 0 and norm 1: scaling them, or shifting them, leaves
    # the convolution as it was
    norm = FixedScale(3, 0.5, standardize=True)
    weight = jax.random.normal(jax.random.PRNGKey(0), (3, 2, 3, 3))

    standardized = norm.conv_weight(weight)

    np.testing.assert_allclose(
        norm.conv_weight(7 * weight - 2), standardized, atol=1e-5
    )
    np.testing.assert_allclose(standardized.mean(axis=(1, 2, 3)), 0, atol=1e-6)
    np.testing.assert_allclose(
        np.sum(np.square(standardized), axis=(1, 2, 3)), 0.25, rtol=1e-4
    )


@pytest.mark.parametrize("norm", ["fixed", "ws"])
def test_norm_free_branches_start_at_about_unit_variance(norm):
    # the fixed scales stand in for normalization at initialization: each block's first
    # convolution's output has about unit variance, as BatchNorm's would (within a few times:
    # `residual_stats` takes features as independent, which they aren't), however deep
    net, state = make_variant("global pooling", norm=norm, num_blocks=6)
    rng = np.random.default_rng(0)
    x = jnp.asarray(
        np.moveaxis(observations(rng, batch_size=64), -1, 1), dtype=jnp.float32
    )
    dtype = jnp.dtype("float32")

    def branches(x):
        assert net.stem_norm is not None
        h, _ = conv_norm(net.stem_conv, net.stem_norm, x, state, dtype)
        h = jax.nn.relu(h)
        outputs = []
        for block in net.blocks:
            assert isinstance(block, ResidualBlock)
            y, _ = conv_norm(block.conv1, block.norm1, h, state, dtype)
            outputs.append(y)
            h, _ = block(h, state, dtype)
        return jnp.stack(outputs)

    # each channel's variance over the batch and the board, as BatchNorm normalizes it
    y = jax.vmap(branches)(x)
    variances = np.asarray(y.var(axis=(0, 3, 4)).mean(axis=-1))
    assert np.all((0.3 < variances) & (variances < 5)), variances


def test_residual_stats_grow_with_depth():
    stats = residual_stats(4)

    moments, variances = np.array(stats).T
    # a ReLU of a standard normal: E[x^2] 1/2, variance 1/2 - 1/(2 pi)
    np.testing.assert_allclose(moments[0], 0.5, atol=0.01)
    np.testing.assert_allclose(variances[0], 0.5 - 1 / (2 * np.pi), atol=0.01)
    assert np.all(np.diff(moments) > 0.5) and np.all(np.diff(variances) > 0.2)


@pytest.mark.parametrize("name", VARIANTS)
def test_pre_activation_batch_norm_infers_as_it_trained_on_its_batch(name):
    # BatchNorm before each convolution: after one training step, inference on the same batch
    # gives the outputs training did, as in the post-activation layout
    net, state = make_variant(name, layout="pre")
    x = observations(np.random.default_rng(0), batch_size=8)

    @eqx.filter_jit
    def run(net, state, x):
        training, state = apply_nn(net, state, x)
        inference, _ = apply_nn(eqx.nn.inference_mode(net), state, x)
        return training, inference

    training, inference = run(net, state, x)

    for a, b in zip(inference, training, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("norm", ["fixed", "ws"])
def test_fixed_scales_refuse_the_pre_activation_layout(norm):
    with pytest.raises(ValueError, match="pre"):
        make_variant("pooled heads", norm=norm, layout="pre")


@pytest.mark.parametrize("name", ["global pooling", "bottleneck"])
def test_the_pre_activation_trunk_starts_as_the_stem(name):
    # x + branch, each branch's last convolution starting at zero: at initialization the trunk is
    # the stem's output however deep, and the branches add to it once trained
    net, state = make_variant(name, norm="channel", layout="pre")
    x = jnp.moveaxis(observations(np.random.default_rng(0), 1)[0], -1, 0)
    dtype = jnp.dtype("float32")

    stem = net.stem_conv(x.astype(dtype))
    trunk = stem
    for block in net.blocks:
        trunk, _ = block(trunk, state, dtype)

    np.testing.assert_allclose(trunk, stem, atol=1e-6)
