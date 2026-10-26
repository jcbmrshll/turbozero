import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from core.networks.azresnet import AZResnet, AZResnetConfig
from core.training.train import update_size_metrics, weight_decay_mask


def resnet_params():
    """The params of a small AZResnet, as the trainer optimizes them."""
    net, _ = eqx.nn.make_with_state(AZResnet)(
        AZResnetConfig(policy_head_out_size=5, num_blocks=1, num_channels=4),
        (3, 3, 2),
        key=jax.random.PRNGKey(0),
    )
    return eqx.filter(net, eqx.is_inexact_array)


def by_path(tree) -> dict[str, jax.Array]:
    return {
        jax.tree_util.keystr(path): x
        for path, x in jax.tree_util.tree_leaves_with_path(tree)
    }


def test_the_mask_selects_the_convolution_and_linear_weights():
    mask = by_path(weight_decay_mask(resnet_params()))

    assert {path for path, decayed in mask.items() if decayed} == {
        ".stem_conv.weight",
        ".blocks[0].conv1.weight",
        ".blocks[0].conv2.weight",
        ".heads.policy_conv.weight",
        ".heads.policy_linear.weight",
        ".heads.value_conv.weight",
        ".heads.value_hidden.weight",
        ".heads.value_linear.weight",
    }
    # the rest: the linear layers' biases and the BatchNorms' scales and offsets
    assert {path for path, decayed in mask.items() if not decayed} == {
        ".heads.policy_linear.bias",
        ".heads.value_hidden.bias",
        ".heads.value_linear.bias",
    } | {
        f".{bn}.{param}"
        for bn in (
            "stem_bn",
            "blocks[0].bn1",
            "blocks[0].bn2",
            "heads.policy_bn",
            "heads.value_bn",
        )
        for param in ("weight", "bias")
    }


def test_adamw_decays_the_masked_params_only():
    params = resnet_params()
    # BatchNorm starts with scale 1 and offset 0: make every param nonzero, to see it decay
    params = jax.tree.map(lambda x: x + 0.5, params)
    optimizer = optax.inject_hyperparams(optax.adamw, static_args=("mask",))(
        learning_rate=0.1, weight_decay=0.5, mask=weight_decay_mask
    )

    # with no gradient, Adam moves nothing, and only the weight decay is left
    zeros = jax.tree.map(jnp.zeros_like, params)
    updates, _ = optimizer.update(zeros, optimizer.init(params), params)
    stepped = optax.apply_updates(params, updates)

    mask = by_path(weight_decay_mask(params))
    for path, before in by_path(params).items():
        expected = before * (1 - 0.1 * 0.5) if mask[path] else before
        np.testing.assert_allclose(by_path(stepped)[path], expected, rtol=1e-6)


def test_update_size_is_the_step_relative_to_the_decayed_weights():
    params = {"conv": jnp.full((2, 2), 3.0), "bias": jnp.full((4,), 100.0)}
    updates = {"conv": jnp.full((2, 2), 0.3), "bias": jnp.full((4,), 50.0)}

    metrics = update_size_metrics(params, updates)

    # the bias isn't decayed, so it counts for neither: ||conv|| = 6, ||its update|| = 0.6
    np.testing.assert_allclose(metrics["weight_norm"], 6.0, rtol=1e-6)
    np.testing.assert_allclose(metrics["update_ratio"], 0.1, rtol=1e-6)
