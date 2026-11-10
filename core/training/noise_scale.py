"""The gradient noise scale, measured from a training step's micro-batches (or at any weights, see
`Trainer.noise_scale_at`): for watching a run, nothing follows it.

The noise scale (McCandlish et al. 2018, "An Empirical Model of Large-Batch Training") is
B_noise = tr(Sigma) / |G|^2: the per-sample gradient covariance's trace over the true gradient's
squared norm. A batch of B samples has a gradient whose expected squared norm is
|G|^2 + tr(Sigma) / B, so about B / (B + B_noise) of a step is signal: at B = B_noise, half.

Gradient accumulation (the trainer's `grad_accum_steps`) computes the gradients of n micro-batches
of b samples before averaging them into the batch's, of B = n b samples. From the micro-batches'
mean squared norm |g_b|^2 and the batch gradient's |g_B|^2, McCandlish's unbiased estimators
(their appendix A.1) are
    |G|^2    = (B |g_B|^2 - b |g_b|^2) / (B - b)
    tr(Sigma) = (|g_b|^2 - |g_B|^2) / (1/b - 1/B)
Each step's estimates are noisy (|G|^2 can come out negative), so they're averaged over steps
before the ratio is taken.

Measured at the weights training moves, the noise scale is far too low when the weights jitter: a
snapshot sits off the centre of where they wander, and its gradient mostly points back, a pull
that adds to |G|^2 (on a 4x32 Othello net late in training, 300 at the weights against 5083 at
their 4-epoch average). `Trainer.noise_scale_at` measures it at a weight average instead. Even
there, at a constant learning rate |G|^2 includes the step's displacement of the average from where
smaller steps would take it, so the noise scale rises as the rate drops (on a stalled 12x144 about
3500 just after a 10x drop, 120000 thirty epochs on): a reading to watch, not one to set the
learning rate from, which cools it into the ground.
"""

import jax
import jax.numpy as jnp


def noise_scale_estimates(
    micro_sq: jax.Array, batch_sq: jax.Array, micro_batch: int, batch: int
) -> tuple[jax.Array, jax.Array]:
    """McCandlish's unbiased estimates of tr(Sigma) and |G|^2 (see the module's docstring).

    Args:
        micro_sq: the micro-batches' gradients' mean squared norm, |g_b|^2
        batch_sq: the squared norm of their mean, the batch's gradient, |g_B|^2
        micro_batch: samples in a micro-batch, b
        batch: samples in the batch, B, more than b

    Returns:
        (tr(Sigma), |G|^2): the per-sample gradient covariance's trace, and the true gradient's
            squared norm
    """
    if batch <= micro_batch:
        raise ValueError(
            f"the batch ({batch}) must be bigger than its micro-batches ({micro_batch})"
        )
    tr_sigma = (micro_sq - batch_sq) / (1 / micro_batch - 1 / batch)
    grad_sq = (batch * batch_sq - micro_batch * micro_sq) / (batch - micro_batch)
    return tr_sigma, grad_sq


def noise_scale_metrics(
    micro_sq: jax.Array, batch_sq: jax.Array, micro_batch: int, batch: int
) -> dict[str, jax.Array]:
    """An epoch's noise scale from its steps' mean |g_b|^2 and |g_B|^2 (see
    `noise_scale_estimates`):
    - `gns_tr_sigma`, `gns_grad_sq`: the estimates of tr(Sigma) and |G|^2
    - `gns_noise_scale`: their ratio, B_noise (infinite if |G|^2 came out at or below 0)
    - `gns_signal_share`: B / (B + B_noise), the share of a step at this batch that is signal
    """
    tr_sigma, grad_sq = noise_scale_estimates(micro_sq, batch_sq, micro_batch, batch)
    noise_scale = jnp.where(grad_sq > 0, tr_sigma / grad_sq, jnp.inf)
    return {
        "gns_tr_sigma": tr_sigma,
        "gns_grad_sq": grad_sq,
        "gns_noise_scale": noise_scale,
        "gns_signal_share": batch / (batch + noise_scale),
    }
