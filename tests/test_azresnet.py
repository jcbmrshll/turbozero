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
    BranchedDepthwise,
    Depthwise,
    DynamicDepthwise,
    FixedScale,
    GatedScan,
    GroupNorm,
    PreActConvNeXt,
    ResidualBlock,
    batch_norm,
    conv,
    conv_norm,
    inference_affine,
    norm_maker,
    residual_stats,
    square_taps,
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


# variants only the pre-activation layout takes
PRE_VARIANTS = {
    # without a normalization after the stem, expand and project
    "convnext": {
        "num_blocks": 2,
        "num_channels": 6,
        "block": "convnext",
        "expand": 2,
        "dw_kernel": 3,
        "pool_every": 2,
        "pool_channels": 2,
        "layout": "pre",
        "stem_norm": False,
        "expand_norm": False,
        "project_norm": False,
    },
    # with one after each convolution, a ConvNeXt trunk's default
    "normalized convnext": {
        "num_blocks": 2,
        "num_channels": 6,
        "block": "convnext",
        "expand": 2,
        "dw_kernel": 3,
        "pool_every": 2,
        "pool_channels": 2,
        "layout": "pre",
    },
}
# the normalized ConvNeXt block's options
for _name, _options in {
    "branched convnext": {
        "dw_branches": (((0, 1), (0, -1)), ((1, 0), (-1, 0)), ((0, 0),))
    },
    "dynamic convnext": {"dw_experts": 3},
    "scanned convnext": {
        "scan_steps": ((0, 1), (1, 0), (0, -1), (-1, 0)),
        "scan_gates": 3,
    },
    "attention convnext": {"attention_last": 1, "attention_heads": 2},
    "positioned convnext": {"position_bias": True},
    "expert convnext": {"mlp_experts": 3},
}.items():
    PRE_VARIANTS[_name] = {**PRE_VARIANTS["normalized convnext"], **_options}


def make_variant(name: str, **options):
    variant = {**VARIANTS, **PRE_VARIANTS}[name]
    config = AZResnetConfig(policy_head_out_size=10, **{**variant, **options})
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


@pytest.mark.parametrize("name", [*VARIANTS, *PRE_VARIANTS])
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


@pytest.mark.parametrize("name", ["global pooling", "bottleneck", *PRE_VARIANTS])
def test_the_pre_activation_trunk_starts_as_the_stem(name):
    # x + branch, each branch's last convolution starting at zero: at initialization the trunk is
    # the stem's output however deep, and the branches add to it once trained
    net, state = make_variant(name, norm="channel", layout="pre")
    x = jnp.moveaxis(observations(np.random.default_rng(0), 1)[0], -1, 0)
    dtype = jnp.dtype("float32")

    stem = net.stem_conv(x.astype(dtype))
    if net.stem_norm is not None:
        stem, _ = conv_norm(net.stem_conv, net.stem_norm, x, state, dtype)
    trunk = stem
    for block in net.blocks:
        trunk, _ = block(trunk, state, dtype)

    np.testing.assert_allclose(trunk, stem, atol=1e-6)


def test_convnext_blocks_refuse_the_post_activation_layout():
    with pytest.raises(ValueError, match="pre"):
        make_variant("convnext", layout="post")


@pytest.mark.parametrize("norm", ["layer", "channel"])
def test_convnext_blocks_with_batch_independent_norms_infer_one_input_at_a_time(norm):
    net, state = make_variant("convnext", norm=norm)
    rng = np.random.default_rng(0)
    params, static = eqx.partition(net, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda x: x + 0.3 * rng.normal(size=x.shape).astype(x.dtype), params
    )
    net = eqx.combine(params, static)
    x = jnp.asarray(observations(rng, batch_size=4))
    (policy, value), _ = apply_nn(net, state, x)
    (one_policy, one_value), _ = apply_nn(eqx.nn.inference_mode(net), state, x[:1])
    assert policy.shape == (4, 10) and value.shape == (4, 1)
    np.testing.assert_allclose(one_policy, policy[:1], atol=1e-5)
    np.testing.assert_allclose(one_value, value[:1], atol=1e-5)


def reference_depthwise(weight: np.ndarray, taps, x: np.ndarray) -> np.ndarray:
    """y[c, i, j] = sum over taps of weight[c, t] * x[c, i + dy, j + dx], zero off the grid."""
    channels, height, width = x.shape
    y = np.zeros_like(x)
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                for t, (dy, dx) in enumerate(taps):
                    if 0 <= i + dy < height and 0 <= j + dx < width:
                        y[c, i, j] += weight[c, t] * x[c, i + dy, j + dx]
    return y


@pytest.mark.parametrize(
    "taps",
    [square_taps(3), ((0, 0), (0, 2), (-1, 1), (2, -2), (-3, 0))],
    ids=["3x3", "sparse"],
)
def test_depthwise_filters_each_channel_at_its_taps_only(taps):
    rng = np.random.default_rng(0)
    dw = Depthwise(3, taps, key=jax.random.PRNGKey(0))
    weight = rng.normal(size=dw.weight.shape).astype(np.float32)
    dw = eqx.tree_at(lambda d: d.weight, dw, jnp.asarray(weight))
    x = rng.normal(size=(3, 5, 6)).astype(np.float32)

    y = dw(jnp.asarray(x), jnp.dtype("float32"))

    np.testing.assert_allclose(y, reference_depthwise(weight, taps, x), atol=1e-5)


def test_depthwise_filters_have_a_weight_per_tap_and_the_square_that_holds_them():
    dw = Depthwise(4, ((0, 0), (0, 3), (-1, -1)), key=jax.random.PRNGKey(0))
    assert dw.weight.shape == (4, 3) and dw.kernel_size == 7
    dense = np.asarray(dw.dense())
    assert dense.shape == (4, 1, 7, 7)
    assert np.count_nonzero(dense[0, 0]) == 3
    with pytest.raises(ValueError, match="distinct"):
        Depthwise(4, ((0, 0), (0, 0)), key=jax.random.PRNGKey(0))


def test_convnext_blocks_take_their_taps_from_the_config():
    taps = ((0, 0), (0, 1), (2, 2))
    for options, expected in [({"dw_taps": taps}, taps), ({}, square_taps(3))]:
        net, _ = make_variant("convnext", **options)
        for block in net.blocks:
            assert isinstance(block, PreActConvNeXt)
            assert isinstance(block.dw, Depthwise) and block.dw.taps == expected


def test_normalizations_are_options_of_the_layouts_and_blocks_that_have_them():
    with pytest.raises(ValueError, match="stem_norm"):
        make_variant("global pooling", stem_norm=True)
    with pytest.raises(ValueError, match="convnext"):
        make_variant("global pooling", layout="pre", expand_norm=True)
    with pytest.raises(ValueError, match="convnext"):
        make_variant("global pooling", layout="pre", project_norm=True)


def test_a_normalized_projects_scale_starts_at_zero_and_its_weights_do_not():
    # a zeroed convolution under a normalization would output 0 and never learn
    net, _ = make_variant("normalized convnext")
    for block in net.blocks:
        assert isinstance(block, PreActConvNeXt) and block.project_norm is not None
        assert np.all(np.asarray(block.project_norm.weight) == 0)
        assert np.abs(np.asarray(block.project.weight)).min() > 0


@pytest.mark.parametrize(
    "name, invariant", [("convnext", False), ("normalized convnext", True)]
)
def test_a_normalized_convnext_trunk_ignores_the_scale_of_its_convolutions(
    name, invariant
):
    # in training mode, with each convolution followed by a BatchNorm, scaling any of them changes
    # nothing: weight decay shrinking them can't change the function. Without, scaling the stem or
    # a block's MLP does
    net, state = make_variant(name)
    rng = np.random.default_rng(0)
    params, static = eqx.partition(net, eqx.is_inexact_array)
    # every weight nonzero, the scales of the branches' outputs included
    params = jax.tree.map(
        lambda x: x + 0.3 * rng.normal(size=x.shape).astype(x.dtype), params
    )
    net = eqx.combine(params, static)
    x = jnp.asarray(observations(rng, batch_size=8))

    def scaled_block(block, f):
        return eqx.tree_at(
            lambda b: (b.dw.weight, b.expand.weight, b.project.weight),
            block,
            (block.dw.weight * f, block.expand.weight * f, block.project.weight * f),
        )

    def scaled(net, factor):
        return eqx.tree_at(
            lambda n: (n.stem_conv.weight, n.blocks),
            net,
            (
                net.stem_conv.weight * factor,
                [scaled_block(b, factor ** (i + 2)) for i, b in enumerate(net.blocks)],
            ),
        )

    (policy, value), _ = apply_nn(net, state, x)
    (scaled_policy, scaled_value), _ = apply_nn(scaled(net, 0.7), state, x)

    # up to BatchNorm's epsilon, against changes of about 0.1 or more
    change = max(
        float(jnp.abs(scaled_policy - policy).max()),
        float(jnp.abs(scaled_value - value).max()),
    )
    assert change < 1e-3 if invariant else change > 1e-2, change


def test_a_normalized_convnext_blocks_pooling_bias_comes_after_its_norm():
    # in training mode, a shift of the pooling bias the same for every input changes the outputs:
    # the norm doesn't take it out, so training sees all of the bias, and an average of the
    # weights moves only what training saw
    net, state = make_variant("normalized convnext")
    rng = np.random.default_rng(0)
    params, static = eqx.partition(net, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda x: x + 0.3 * rng.normal(size=x.shape).astype(x.dtype), params
    )
    net = eqx.combine(params, static)
    x = jnp.asarray(observations(rng, batch_size=8))
    pooled = [i for i, b in enumerate(net.blocks) if b.pool_linear is not None]
    assert pooled
    i = pooled[0]
    shifted = eqx.tree_at(
        lambda n: n.blocks[i].pool_linear.bias,
        net,
        net.blocks[i].pool_linear.bias + 1.0,
    )

    (policy, value), _ = apply_nn(net, state, x)
    (shifted_policy, shifted_value), _ = apply_nn(shifted, state, x)

    assert float(jnp.abs(shifted_value - value).max()) > 1e-3
    assert float(jnp.abs(shifted_policy - policy).max()) > 1e-3


def test_a_convnext_trunk_is_normalized_after_every_convolution_by_default():
    net, _ = make_variant("normalized convnext")
    assert net.stem_norm is not None
    for block in net.blocks:
        assert isinstance(block, PreActConvNeXt)
        assert block.expand_norm is not None and block.project_norm is not None
    # other pre-activation trunks start from the stem as it is
    plain, _ = make_variant("global pooling", layout="pre")
    assert plain.stem_norm is None


def bn_with_random_inference(norm, state, rng):
    """`norm` (BatchNorm) with a random affine map, and `state` with random running statistics."""
    norm = with_affine(norm, rng)
    return norm, with_statistics(
        norm, state, *random_statistics(rng, norm.weight.shape[0])
    )


def test_branches_fold_into_one_filter_of_their_taps():
    # each branch a filter then an affine map per channel: in inference their sum is one filter
    # on the union of their taps, plus a bias
    rng = np.random.default_rng(0)
    branches = (((0, 1), (0, 2)), ((1, 0), (0, 1)), ((0, 0),))
    make_norm = norm_maker()
    mixer, state = eqx.nn.make_with_state(BranchedDepthwise)(
        3, branches, key=jax.random.PRNGKey(0), make_norm=make_norm, stats=(1.0, 1.0)
    )
    norms = []
    for norm in mixer.norms:
        norm, state = bn_with_random_inference(norm, state, rng)
        norms.append(norm)
    mixer = eqx.nn.inference_mode(eqx.tree_at(lambda m: m.norms, mixer, norms))
    x = jnp.asarray(rng.normal(size=(3, 4, 5)), jnp.float32)

    y, _ = mixer(x, state, jnp.dtype(jnp.float32))
    folded, bias = mixer.fold(state)

    assert set(folded.taps) == {(0, 1), (0, 2), (1, 0), (0, 0)}
    expected = folded(x, jnp.dtype(jnp.float32)) + bias.reshape(-1, 1, 1)
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)


def test_a_dynamic_filter_starts_as_the_mean_of_its_experts():
    # the router starts at zero: every input gets the experts equally weighted
    taps = square_taps(3)
    mixer = DynamicDepthwise(4, taps, 3, key=jax.random.PRNGKey(0))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(4, 5, 5)), jnp.float32)
    mean = eqx.tree_at(
        lambda d: d.weight,
        Depthwise(4, taps, key=jax.random.PRNGKey(1)),
        mixer.weight.mean(axis=0),
    )

    y, _ = mixer(x, eqx.nn.State(mixer), jnp.dtype(jnp.float32))

    np.testing.assert_allclose(mixer.mixing(x), np.full(3, 1 / 3), rtol=1e-6)
    np.testing.assert_allclose(y, mean(x, jnp.dtype(jnp.float32)), rtol=1e-5, atol=1e-5)


def test_a_dynamic_filters_experts_depend_on_the_input():
    mixer = DynamicDepthwise(2, square_taps(3), 2, key=jax.random.PRNGKey(0))
    router = eqx.tree_at(
        lambda r: r.weight,
        mixer.router,
        jnp.ones_like(mixer.router.weight).at[1].set(-1),
    )
    mixer = eqx.tree_at(lambda m: m.router, mixer, router)
    a, b = jnp.zeros((2, 3, 3)), jnp.ones((2, 3, 3))
    assert not np.allclose(mixer.mixing(a), mixer.mixing(b))


def gated_scan(gate: float, channels: int = 2, length: int = 3):
    """A scan along rows and columns whose gates are all `gate` (a constant gate logit), with
    random weights, and its state."""
    scan, state = eqx.nn.make_with_state(GatedScan)(
        channels,
        ((0, 1), (1, 0), (0, -1), (-1, 0)),
        length,
        1,
        key=jax.random.PRNGKey(0),
        make_norm=norm_maker(),
        stats=(1.0, 1.0),
    )
    weight = np.random.default_rng(1).uniform(0.5, 1.5, scan.weight.shape)
    logit = float(np.log(gate / (1 - gate)))
    zero = jnp.zeros_like(scan.gate_conv.weight)
    scan = eqx.tree_at(
        lambda s: (s.weight, s.gate_conv.weight, s.gate_offset),
        scan,
        (
            jnp.asarray(weight, jnp.float32),
            zero,
            jnp.full_like(scan.gate_offset, logit),
        ),
    )
    return eqx.nn.inference_mode(scan), state


def line_filter(scan: GatedScan, gate: float) -> Depthwise:
    """The depthwise filter a scan whose gates are all `gate` computes: the tap at distance k
    along a line weighs gate^(k - 1)."""
    taps, weights = [(0, 0)], [scan.weight[:, 0]]
    for d, (dy, dx) in enumerate(scan.steps):
        for k in range(1, scan.length + 1):
            taps.append((k * dy, k * dx))
            weights.append(
                scan.weight[:, 1 + d * scan.length + k - 1] * gate ** (k - 1)
            )
    dw = Depthwise(scan.weight.shape[0], tuple(taps), key=jax.random.PRNGKey(0))
    return eqx.tree_at(lambda d: d.weight, dw, jnp.stack(weights, axis=1))


@pytest.mark.parametrize("gate", [0.999999, 0.5, 1e-6])
def test_gated_lines_with_equal_gates_are_a_line_filter_fading_by_the_gate(gate):
    # open gates: the plain filter along the lines; closed ones: only the nearest square each way
    scan, state = gated_scan(gate)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 5, 5)), jnp.float32)

    y, _ = scan(x, state, jnp.dtype(jnp.float32))

    expected = line_filter(scan, gate)(x, jnp.dtype(jnp.float32))
    np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-4)


def test_a_gated_line_stops_at_a_closed_square():
    # one square's gate closed (its own input decides it): what lies beyond it along a line
    # no longer reaches the square filtered
    scan, state = gated_scan(0.999999, channels=1, length=4)
    gate_conv = eqx.tree_at(
        lambda c: c.weight, scan.gate_conv, jnp.full_like(scan.gate_conv.weight, -100.0)
    )
    scan = eqx.tree_at(lambda s: s.gate_conv, scan, gate_conv)
    x = jnp.zeros((1, 1, 5)).at[0, 0, 2].set(1.0)  # the blocking square, at column 2
    far = x.at[0, 0, 4].set(1.0)  # and something beyond it

    near, _ = scan(x, state, jnp.dtype(jnp.float32))
    both, _ = scan(far, state, jnp.dtype(jnp.float32))

    # column 0 sees column 2 (one square between, open) but not column 4 (behind column 2)
    np.testing.assert_allclose(both[0, 0, 0], near[0, 0, 0], atol=1e-6)
    # column 3 sees column 4 directly
    assert abs(float(both[0, 0, 3] - near[0, 0, 3])) > 0.1


def unrolled_scan(scan: GatedScan, x, state, dtype):
    """GatedScan's definition computed step by step, the reference for its fast form: along
    each direction, shift and multiply by each square passed."""
    channels, height, width = x.shape
    g, state = scan.gates(x, state, dtype)
    g = jnp.repeat(g, channels // g.shape[0], axis=0)
    n = scan.length * max(abs(s) for step in scan.steps for s in step)
    pad = ((0, 0), (n, n), (n, n))
    xf = x.astype(jnp.float32)
    xp, gp = jnp.pad(xf, pad), jnp.pad(g, pad)

    def shifted(a, dy, dx):
        return a[:, n + dy : n + dy + height, n + dx : n + dx + width]

    y = scan.weight[:, 0, None, None] * xf
    for d, (dy, dx) in enumerate(scan.steps):
        through = jnp.ones_like(xf)
        for k in range(1, scan.length + 1):
            tap = scan.weight[:, 1 + d * scan.length + k - 1, None, None]
            y = y + tap * through * shifted(xp, k * dy, k * dx)
            if k < scan.length:
                through = through * shifted(gp, k * dy, k * dx)
    return y.astype(dtype), state


EIGHT_WAYS = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)


@pytest.mark.parametrize(
    "steps, length, board, offset",
    [
        (EIGHT_WAYS, 7, (8, 8), 2.0),
        # nearly closed gates, and so closed that their products underflow
        (EIGHT_WAYS, 7, (8, 8), -4.0),
        (EIGHT_WAYS, 7, (8, 8), -60.0),
        # a grid that isn't square, lines longer than it, one way only
        (((0, 1), (1, 1)), 6, (3, 5), 0.0),
        # steps that skip squares, and two that reach one square by different paths
        (((1, 2), (0, 1), (0, 2)), 4, (6, 7), 1.0),
        # no square between any two: nothing is gated
        (((0, 1), (-1, 0)), 1, (4, 4), 2.0),
    ],
)
def test_a_gated_scan_is_its_unrolled_definition_with_its_gradients(
    steps, length, board, offset
):
    rng = np.random.default_rng(0)
    scan, state = eqx.nn.make_with_state(GatedScan)(
        6,
        steps,
        length,
        3,
        key=jax.random.PRNGKey(0),
        make_norm=norm_maker(),
        stats=(1.0, 1.0),
    )
    scan = eqx.tree_at(lambda s: s.gate_offset, scan, jnp.full((3,), offset))
    params, static = eqx.partition(scan, eqx.is_inexact_array)
    x = jnp.asarray(rng.normal(size=(3, 6, *board)), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=x.shape), jnp.float32)

    def outputs_and_gradients(call):
        # batched and training, as the trainer runs it
        def batched(params, x):
            scan = eqx.combine(params, static)
            fresh = jax.tree.map(lambda a: a, state)  # a state is used once
            return jax.vmap(
                lambda x: call(scan, x, fresh, jnp.dtype(jnp.float32))[0],
                axis_name=BATCH_AXIS,
            )(x)

        @jax.jit
        def run(params, x):
            y, pull = jax.vjp(batched, params, x)
            return y, pull(cotangent), batched(params, x)  # also undifferentiated

        return jax.tree.leaves(run(params, x))

    actual = outputs_and_gradients(GatedScan.__call__)
    expected = outputs_and_gradients(unrolled_scan)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_a_gated_scans_matrix_products_ask_for_full_float32():
    # a GPU computes float32 matrix products with 10-bit mantissas unless asked not to, which
    # tests on the CPU can't see: the scan was 1e-3 off there
    scan, state = gated_scan(0.5)
    params, static = eqx.partition(scan, eqx.is_inexact_array)
    x = jnp.ones((2, 5, 5))

    def forward_and_backward(params, x):
        def apply(params, x):
            fresh = jax.tree.map(lambda a: a, state)
            scan = eqx.combine(params, static)
            return scan(x, fresh, jnp.dtype(jnp.float32))[0]

        y, pull = jax.vjp(apply, params, x)
        return pull(y)

    products = [
        eqn.params["precision"]
        for eqn in jax.make_jaxpr(forward_and_backward)(params, x).eqns
        if eqn.primitive.name == "dot_general"
    ]

    assert (
        len(products) >= 3
    )  # the paths' gates, their transpose, and the taps' weights
    highest = jax.lax.Precision.HIGHEST
    assert all(p is not None and set(p) == {highest} for p in products)


def test_equal_mlp_experts_are_one_convolution():
    # experts equal to the block's own convolutions: the mixture is that convolution
    net, state = make_variant("expert convnext")
    block = net.blocks[0]
    assert isinstance(block, PreActConvNeXt)
    assert block.expand_experts is not None and block.project_experts is not None
    plain = eqx.tree_at(
        lambda b: (b.mlp_router, b.expand_experts, b.project_experts),
        block,
        (None, None, None),
        is_leaf=lambda x: x is None,
    )
    same = eqx.tree_at(
        lambda b: (b.expand_experts, b.project_experts),
        block,
        (
            jnp.broadcast_to(block.expand.weight, block.expand_experts.shape),
            jnp.broadcast_to(block.project.weight, block.project_experts.shape),
        ),
    )
    same = eqx.tree_at(lambda b: b.project_norm.weight, same, jnp.ones(6))
    plain = eqx.tree_at(lambda b: b.project_norm.weight, plain, jnp.ones(6))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(6, 3, 3)), jnp.float32)

    @eqx.filter_jit
    def run(block, state, x):
        return jax.vmap(
            lambda x: block(x, state, jnp.dtype(jnp.float32))[0],
            in_axes=0,
            axis_name=BATCH_AXIS,
        )(x)

    xs = jnp.stack([x, -x])
    expected = run(plain, jax.tree.map(jnp.copy, state), xs)
    np.testing.assert_allclose(run(same, state, xs), expected, rtol=1e-5, atol=1e-5)


def test_input_planes_are_added_to_the_stems_input():
    def doubled(x):
        return 1 - x.astype(jnp.float32)

    net, state = make_variant(
        "normalized convnext", board_plane=True, input_features=doubled
    )
    # 2 input planes, 2 computed from them, 1 of ones
    assert net.stem_conv.weight.shape[1] == 5
    x = jnp.asarray(observations(np.random.default_rng(0)))
    (policy, value), _ = eqx.filter_jit(apply_nn)(net, state, x)
    assert policy.shape == (4, 10) and value.shape == (4, 1)


def test_the_block_options_belong_to_convnext_blocks():
    with pytest.raises(ValueError, match="convnext"):
        make_variant("global pooling", mlp_experts=2)
    with pytest.raises(ValueError, match="one step between squares"):
        make_variant("dynamic convnext", scan_steps=((0, 1),))
