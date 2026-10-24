import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.networks.transformer import BoardTransformer, BoardTransformerConfig
from core.networks.utils import apply_nn

INPUT_SHAPE = (3, 3, 2)


def make_net(**options):
    """A tiny BoardTransformer; the same parameters whatever its static options."""
    config = BoardTransformerConfig(
        policy_head_out_size=10,
        num_layers=2,
        dim=8,
        num_heads=2,
        ffn_dim=8,
        value_channels=2,
        value_hidden_size=4,
        **options,
    )
    return eqx.nn.make_with_state(BoardTransformer)(
        config, INPUT_SHAPE, key=jax.random.PRNGKey(0)
    )


def observations(rng: np.random.Generator, batch_size=4):
    return rng.random((batch_size, *INPUT_SHAPE)) < 0.5


def loss(net, state, x):
    (policy, value), state = apply_nn(net, state, x)
    return (policy**2).mean() + (value**2).mean(), state


@pytest.mark.parametrize("norm", ["post", "pre"])
@pytest.mark.parametrize("dtype, tolerance", [("float32", 1e-6), ("bfloat16", 5e-2)])
def test_inference_gives_the_outputs_training_does(norm, dtype, tolerance):
    net, state = make_net(norm=norm, inference_dtype=dtype)
    x = observations(np.random.default_rng(0))

    @eqx.filter_jit
    def run(net, state, x):
        return apply_nn(net, state, x)[0], apply_nn(
            eqx.nn.inference_mode(net), state, x
        )[0]

    (policy, value), inference = run(net, state, x)

    assert policy.shape == (4, 10) and value.shape == (4, 1)
    for a, b in zip(inference, (policy, value), strict=True):
        assert a.dtype == jnp.float32
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


def test_static_options_do_not_change_training():
    # inference_dtype only applies in inference; remat only changes what the backward pass
    # computes again
    @eqx.filter_jit
    def run(nets, state, x):
        grad_fn = eqx.filter_value_and_grad(loss, has_aux=True)
        return [jax.tree.leaves(grad_fn(net, state, x)) for net in nets]

    plain, state = make_net()
    nets = (plain, make_net(inference_dtype="bfloat16")[0], make_net(remat=True)[0])
    trained, trained16, trained_remat = run(
        nets, state, observations(np.random.default_rng(0))
    )

    for a, b in zip(trained, trained16, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(trained, trained_remat, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)


def test_mixed_precision_training_stays_close_to_float32():
    @eqx.filter_jit
    def run(net, state, x):
        (value, _), grads = eqx.filter_value_and_grad(loss, has_aux=True)(net, state, x)
        return value, grads

    net32, state = make_net()
    net16, _ = make_net(train_dtype="bfloat16")
    x = observations(np.random.default_rng(0), batch_size=16)
    loss32, grads32 = run(net32, state, x)
    loss16, grads16 = run(net16, state, x)

    np.testing.assert_allclose(loss16, loss32, rtol=2e-2)
    leaves16, leaves32 = jax.tree.leaves(grads16), jax.tree.leaves(grads32)
    assert all(a.dtype == b.dtype == jnp.float32 for a, b in zip(leaves16, leaves32))
    a, b = (
        np.concatenate([np.ravel(x) for x in leaves]) for leaves in (leaves16, leaves32)
    )
    assert np.linalg.norm(a - b) <= 0.15 * np.linalg.norm(b)


def test_the_attention_bias_is_learned_per_layer_head_and_pair_of_squares():
    net, state = make_net()
    without, _ = make_net(attention_bias=False)
    grads = eqx.filter_grad(
        lambda net: loss(net, state, observations(np.random.default_rng(0)))[0]
    )(net)

    assert all(layer.bias is None for layer in without.layers)
    for layer in grads.layers:
        assert layer.bias is not None and layer.bias.shape == (2, 9, 9)
        assert np.abs(layer.bias).sum() > 0


def test_the_policy_must_be_a_logit_per_square_and_a_pass():
    with pytest.raises(ValueError, match="10 actions, not 9"):
        eqx.nn.make_with_state(BoardTransformer)(
            BoardTransformerConfig(
                policy_head_out_size=9, num_layers=1, dim=4, num_heads=2, ffn_dim=4
            ),
            INPUT_SHAPE,
            key=jax.random.PRNGKey(0),
        )
