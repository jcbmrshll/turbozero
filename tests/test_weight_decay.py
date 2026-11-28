from functools import partial
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from core.networks.azresnet import AZResnet, AZResnetConfig
from core.training.train import adamw, sgdw, update_size_metrics, weight_decay_mask


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
            "stem_norm",
            "blocks[0].norm1",
            "blocks[0].norm2",
            "heads.policy_norm",
            "heads.value_norm",
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


def sgd_steps(decoupled, grads, params, steps=2):
    """`params` after `steps` steps of `sgdw` (lr 0.1, momentum 0.5, decay 0.2) on `grads`."""
    optimizer = optax.inject_hyperparams(sgdw, static_args=("mask", "decoupled"))(
        learning_rate=0.1,
        momentum=0.5,
        weight_decay=0.2,
        decoupled=decoupled,
        mask=weight_decay_mask,
    )
    state = optimizer.init(params)
    for _ in range(steps):
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    return params


@pytest.mark.parametrize("decoupled", [False, True])
def test_sgd_decays_the_masked_params_with_momentum(decoupled):
    params = {"w": jnp.full((2, 2), 1.0), "b": jnp.full((2,), 1.0)}
    grads = {"w": jnp.full((2, 2), 0.5), "b": jnp.full((2,), 0.5)}

    stepped = cast(dict, sgd_steps(decoupled, grads, params))

    # by hand: m <- 0.5 m + g (+ 0.2 w for L2); w <- w - 0.1 (m (+ 0.2 w decoupled))
    def by_hand(w, decayed):
        m = 0.0
        for _ in range(2):
            if decoupled:
                m = 0.5 * m + 0.5
                w = w - 0.1 * (m + (0.2 * w if decayed else 0.0))
            else:
                m = 0.5 * m + 0.5 + (0.2 * w if decayed else 0.0)
                w = w - 0.1 * m
        return w

    np.testing.assert_allclose(stepped["w"], by_hand(1.0, True), rtol=1e-6)
    # the bias isn't decayed: both kinds just follow the gradient
    np.testing.assert_allclose(stepped["b"], by_hand(1.0, False), rtol=1e-6)
    assert by_hand(1.0, True) != by_hand(1.0, False)


# weight decay that follows the learning rate

PARAMS = {"w": jnp.full((2, 2), 2.0), "b": jnp.full((2,), 2.0)}
ZEROS = jax.tree.map(jnp.zeros_like, PARAMS)
# 1e-2 for 2 steps, then 1e-3
DROPPING = optax.piecewise_constant_schedule(1e-2, {2: 0.1})


def decays(optimizer, steps=4):
    """Each of `steps` zero-gradient updates' decay of the weight: -update / w (Adam moves
    nothing without a gradient), and the learning rate the update used."""
    state = optimizer.init(PARAMS)
    out = []
    for _ in range(steps):
        updates, state = optimizer.update(ZEROS, state, PARAMS)
        np.testing.assert_array_equal(updates["b"], 0.0)
        out.append(
            (
                float(-updates["w"][0, 0] / PARAMS["w"][0, 0]),
                float(state.hyperparams["learning_rate"]),
            )
        )
    return out


@pytest.mark.parametrize(
    "factory, static",
    [
        (adamw, ("mask", "decay_lr")),
        (partial(sgdw, momentum=0.0), ("mask", "decay_lr")),
        (
            partial(sgdw, momentum=0.0, decoupled=True),
            ("mask", "decay_lr", "decoupled"),
        ),
    ],
)
def test_the_decay_follows_the_learning_rate(factory, static):
    def optimizer(decay_lr):
        return optax.inject_hyperparams(factory, static_args=static)(
            learning_rate=DROPPING,
            weight_decay=0.5,
            mask=weight_decay_mask,
            decay_lr=decay_lr,
        )

    # 0.5 at 1e-2, 0.05 at 1e-3: each step's decay lr * 0.5 * lr / 1e-2, so lr^2 * 50
    for decay, lr in decays(optimizer(1e-2)):
        assert decay == pytest.approx(lr * lr * 50, rel=1e-5)
    assert [lr for _, lr in decays(optimizer(1e-2))] == pytest.approx(
        [1e-2] * 2 + [1e-3] * 2
    )
    # without decay_lr the decay is held: lr * 0.5
    for decay, lr in decays(optimizer(None)):
        assert decay == pytest.approx(lr * 0.5, rel=1e-5)


def test_the_injected_weight_decay_is_the_one_at_decay_lr():
    optimizer = optax.inject_hyperparams(adamw, static_args=("mask", "decay_lr"))(
        learning_rate=DROPPING, weight_decay=0.5, mask=weight_decay_mask, decay_lr=1e-2
    )
    state: Any = optimizer.init(PARAMS)
    for _ in range(3):
        _, state = optimizer.update(ZEROS, state, PARAMS)
    assert float(state.hyperparams["weight_decay"]) == 0.5


def test_its_state_is_optax_adamws():
    # so that states saved under optax.adamw load, branch and fork into it, and back
    def state(factory, **kwargs):
        static = ("mask", *kwargs)
        return optax.inject_hyperparams(factory, static_args=static)(
            learning_rate=DROPPING, weight_decay=0.5, mask=weight_decay_mask, **kwargs
        ).init(PARAMS)

    expected = state(optax.adamw)
    for got in (state(adamw, decay_lr=1e-2), state(adamw, decay_lr=None)):
        assert jax.tree.structure(got) == jax.tree.structure(expected)


def test_decay_lr_needs_the_learning_rates_value():
    with pytest.raises(TypeError, match="inject_hyperparams"):
        adamw(DROPPING, weight_decay=0.5, decay_lr=1e-2)
    # a number is fine
    adamw(1e-3, weight_decay=0.5, decay_lr=1e-2).init(PARAMS)
