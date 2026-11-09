import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.training.noise_scale import (
    NoiseBatchProgress,
    NoiseBatchSize,
    noise_scale_estimates,
    noise_scale_metrics,
)

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


# the batch size rule


def rule(**kwargs):
    return NoiseBatchSize(
        **{
            "start_batch": 64,
            "min_batch": 32,
            "max_batch": 256,
            "micro_batch": 16,
            "decay": 0.0,
            **kwargs,
        }
    )


def after(r, progress, noise_scale, grad_sq=1.0):
    return r.after_epoch(progress, noise_scale * grad_sq, grad_sq)


@pytest.mark.parametrize(
    "noise_scale, batch",
    [
        (64, 64),
        # the nearest power of 2 times the micro-batch, in log terms
        (90, 64),
        (91, 128),
        # one doubling or halving an epoch
        (1000, 128),
        (1, 32),
    ],
)
def test_the_batch_is_the_noise_scale_rounded_and_moving_one_step_at_a_time(
    noise_scale, batch
):
    r = rule()
    assert after(r, r.start(), noise_scale).batch == batch


def test_the_batch_stays_between_its_bounds_and_the_learning_rate_takes_over_past_the_cap():
    r = rule(start_batch=256)
    capped = after(r, r.start(), 1024)
    assert (capped.batch, capped.lr_scale) == (256, 0.25)
    # below the cap the rate is back to 1
    assert after(r, capped, 256).lr_scale == 1.0

    r = rule(start_batch=32)
    assert after(r, r.start(), 1).batch == 32


def test_the_batch_follows_the_noise_scale_by_its_factor():
    r = rule(factor=2.0)
    assert after(r, r.start(), 64).batch == 128


def test_estimates_are_averaged_before_their_ratio():
    r = rule(decay=0.5)
    first = r.after_epoch(r.start(), 64.0, 1.0)
    second = r.after_epoch(first, 192.0, 1.0)
    assert (second.tr_sigma, second.grad_sq) == (128.0, 1.0)
    assert r.noise_scale(second) == 128.0
    assert second.batch == 128


@pytest.mark.parametrize(
    "tr_sigma, grad_sq", [(math.nan, 1.0), (1.0, math.inf), (64.0, -1.0)]
)
def test_an_epoch_without_a_usable_noise_scale_changes_nothing(tr_sigma, grad_sq):
    r = rule()
    progress = NoiseBatchProgress(tr_sigma=None, grad_sq=None, batch=128, lr_scale=0.5)
    assert r.after_epoch(progress, tr_sigma, grad_sq).batch == 128
    assert r.after_epoch(progress, tr_sigma, grad_sq).lr_scale == 0.5


def test_steps_keep_the_samples_an_epoch_trains_on():
    r = rule()
    assert [r.steps(batch, 4096) for batch in (32, 64, 256, 8192)] == [128, 64, 16, 1]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"start_batch": 48}, "power of 2 times the micro-batch"),
        ({"max_batch": 250}, "power of 2 times the micro-batch"),
        ({"min_batch": 16, "start_batch": 16}, "at least two micro-batches"),
        ({"start_batch": 512}, "between min_batch"),
        ({"factor": 0.0}, "factor must be positive"),
        ({"decay": 1.0}, "decay in"),
    ],
)
def test_bad_rules_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rule(**kwargs)
