import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.training.noise_scale import noise_scale_estimates, noise_scale_metrics

# the estimators


def test_the_estimators_recover_the_gradient_and_the_noise_from_micro_batches():
    # per-sample gradients G + noise, the noise's covariance diagonal: tr(Sigma) = sum(var)
    dim, b, n, steps = 16, 8, 4, 4000
    true_grad = jnp.linspace(-0.3, 0.3, dim)
    std = jnp.linspace(0.5, 2.0, dim)
    samples = true_grad + std * jax.random.normal(
        jax.random.PRNGKey(0), (steps, n, b, dim)
    )
    micro = samples.mean(axis=2)
    micro_sq = jnp.square(micro).sum(-1).mean(-1)
    batch_sq = jnp.square(micro.mean(axis=1)).sum(-1)

    tr_sigma, grad_sq = noise_scale_estimates(micro_sq, batch_sq, b, n * b)

    # unbiased: their means over many steps are the truth
    np.testing.assert_allclose(tr_sigma.mean(), jnp.square(std).sum(), rtol=0.02)
    np.testing.assert_allclose(grad_sq.mean(), jnp.square(true_grad).sum(), rtol=0.05)


def test_the_estimators_need_a_batch_bigger_than_its_micro_batches():
    with pytest.raises(ValueError, match="must be bigger"):
        noise_scale_estimates(jnp.array(1.0), jnp.array(1.0), 8, 8)


def test_metrics_give_the_noise_scale_and_the_signal_share():
    # tr(Sigma) 8 x 16 / (16 - 8) x (2 - 1) = 16, |G|^2 (16 x 1 - 8 x 2) / 8 = 0: all noise
    noise = noise_scale_metrics(jnp.array(2.0), jnp.array(1.0), 8, 16)
    assert float(noise["gns_tr_sigma"]) == pytest.approx(16.0)
    assert float(noise["gns_grad_sq"]) == pytest.approx(0.0)
    assert math.isinf(noise["gns_noise_scale"])
    assert float(noise["gns_signal_share"]) == 0.0

    # tr(Sigma) 16 x (1.5 - 1) = 8, |G|^2 (16 - 12) / 8 = 0.5: noise scale 16, half of 16 signal
    half = noise_scale_metrics(jnp.array(1.5), jnp.array(1.0), 8, 16)
    assert float(half["gns_noise_scale"]) == pytest.approx(16.0)
    assert float(half["gns_signal_share"]) == pytest.approx(0.5)
