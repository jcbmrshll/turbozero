"""The gradient noise scale, measured from a training step's micro-batches, and a batch size set by
it (see `NoiseBatchSize`).

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

At the weights training moves, though, |G|^2 is mostly something else: the weights jitter around
where they're heading, and each snapshot's gradient points back to that centre. On a 4x32 Othello
net late in training this pull made |G|^2 about 20 times what it was at the weights averaged over
4 epochs, and the noise scale about 17 times smaller (300 against 5083). So the noise scale that
matters, at the centre, is measured from the mean of an epoch's T batch gradients instead: to first
order it's the gradient at the epoch's mean weights, where the pull mostly cancels, and its
squared norm is |G|^2 plus the noise of a mean of T B samples, so
    |G|^2 = |mean of the batch gradients|^2 - tr(Sigma) / (T B)
with tr(Sigma) from the micro-batches as above (the jitter barely changes it).
"""

import math
from dataclasses import dataclass

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
    micro_sq: jax.Array,
    batch_sq: jax.Array,
    micro_batch: int,
    batch: int,
    epoch_sq: jax.Array,
    steps: int,
) -> dict[str, jax.Array]:
    """An epoch's noise scale (see the module's docstring), from its `steps` steps' mean |g_b|^2
    and |g_B|^2, and `epoch_sq`, the squared norm of the mean of their batch gradients:
    - `gns_tr_sigma`: the estimate of tr(Sigma), from the micro-batches
    - `gns_grad_sq`, `gns_noise_scale`: |G|^2 at the weights the steps were at, from the
      micro-batches, and tr(Sigma) over it (low while the weights jitter)
    - `gns_epoch_grad_sq`, `gns_epoch_noise_scale`: |G|^2 from the mean of the steps' gradients,
      about the gradient at the epoch's mean weights, and tr(Sigma) over it: the noise scale
    - `gns_signal_share`: B / (B + the noise scale), the share of a step at this batch that is
      signal
    Noise scales are infinite where their |G|^2 came out at or below 0.
    """
    tr_sigma, grad_sq = noise_scale_estimates(micro_sq, batch_sq, micro_batch, batch)
    epoch_grad_sq = epoch_sq - tr_sigma / (steps * batch)

    def ratio(g: jax.Array) -> jax.Array:
        return jnp.where(g > 0, tr_sigma / g, jnp.inf)

    return {
        "gns_tr_sigma": tr_sigma,
        "gns_grad_sq": grad_sq,
        "gns_noise_scale": ratio(grad_sq),
        "gns_epoch_grad_sq": epoch_grad_sq,
        "gns_epoch_noise_scale": ratio(epoch_grad_sq),
        "gns_signal_share": batch / (batch + ratio(epoch_grad_sq)),
    }


@dataclass(frozen=True)
class NoiseBatchProgress:
    """What a `NoiseBatchSize` rule has seen, on the host: its averages of the epochs' estimates
    (None before the first), and the batch size and learning rate scale it chose for the next
    epoch."""

    tr_sigma: float | None = None
    grad_sq: float | None = None
    batch: int = 0
    lr_scale: float = 1.0


@dataclass(frozen=True)
class NoiseBatchSize:
    """A training batch size set from the gradient noise scale, the trainer's `batch_size_rule`:
    after every epoch, the batch for the next is `factor` times the noise scale, rounded to a power
    of 2 times the micro-batch, between `min_batch` and `max_batch`, and at most `max_doublings`
    doublings (or halvings) from the last. The trainer keeps the samples an epoch trains on
    fixed, so a bigger batch takes fewer steps.

    At a fixed learning rate a bigger batch is like a lower learning rate for SGD (Smith et al.
    2017, "Don't Decay the Learning Rate, Increase the Batch Size"): the noise in the weights goes
    with the learning rate over the batch. So as the noise scale grows (the gradient shrinks as
    training converges), the batch grows with it and training anneals, at a constant share of
    signal per step. Past `max_batch` the learning rate takes over: the update is scaled by
    `max_batch` over the batch the rule wanted.

    The noise scale is measured from each step's micro-batches and the mean of an epoch's
    gradients (see `noise_scale_metrics`: `gns_tr_sigma` and `gns_epoch_grad_sq`), so every batch
    must have at least two micro-batches. Its estimates are averaged over epochs, exponentially, the
    newest epoch's weight `1 - decay`.

    Attributes:
        start_batch: the batch size of the first epoch
        min_batch: the smallest batch size, at least two micro-batches
        max_batch: the biggest batch size
        micro_batch: the micro-batch size, samples a gradient is computed on at once
        factor: the batch size, in noise scales
        decay: how slowly the average of the estimates follows the newest epoch's, in [0, 1)
        max_doublings: the most the batch size doubles (or halves) from one epoch to the next
    """

    start_batch: int
    min_batch: int
    max_batch: int
    micro_batch: int
    factor: float = 1.0
    decay: float = 0.5
    max_doublings: int = 1

    def __post_init__(self):
        for name in ("start_batch", "min_batch", "max_batch"):
            size = getattr(self, name)
            if size % self.micro_batch or not _is_power_of_2(size // self.micro_batch):
                raise ValueError(
                    f"{name} ({size}) must be a power of 2 times the micro-batch "
                    f"({self.micro_batch})"
                )
        if self.min_batch < 2 * self.micro_batch:
            raise ValueError(
                f"min_batch ({self.min_batch}) must hold at least two micro-batches of "
                f"{self.micro_batch}, to measure the noise scale"
            )
        if not self.min_batch <= self.start_batch <= self.max_batch:
            raise ValueError(
                f"start_batch ({self.start_batch}) must be between min_batch "
                f"({self.min_batch}) and max_batch ({self.max_batch})"
            )
        if self.factor <= 0 or not 0 <= self.decay < 1 or self.max_doublings < 1:
            raise ValueError(
                "factor must be positive, decay in [0, 1), and max_doublings at least 1"
            )

    def start(self) -> NoiseBatchProgress:
        """The rule's progress before any epoch."""
        return NoiseBatchProgress(batch=self.start_batch)

    def noise_scale(self, progress: NoiseBatchProgress) -> float | None:
        """The averaged noise scale, None before any estimate, infinite if the averaged |G|^2 is at
        or below 0."""
        if progress.tr_sigma is None or progress.grad_sq is None:
            return None
        if progress.grad_sq <= 0:
            return math.inf
        return progress.tr_sigma / progress.grad_sq

    def after_epoch(
        self, progress: NoiseBatchProgress, tr_sigma: float, grad_sq: float
    ) -> NoiseBatchProgress:
        """The rule's progress after an epoch whose estimates were `tr_sigma` and `grad_sq` (see
        `noise_scale_metrics`), with the next epoch's batch size and learning rate scale. An epoch
        whose estimates aren't finite changes nothing, and while the averaged |G|^2 is at or
        below 0 (all noise, as far as it can tell) the batch and learning rate stay as they
        were."""
        if not (math.isfinite(tr_sigma) and math.isfinite(grad_sq)):
            return progress
        if progress.tr_sigma is None or progress.grad_sq is None:
            averaged = (tr_sigma, grad_sq)
        else:
            averaged = (
                self.decay * progress.tr_sigma + (1 - self.decay) * tr_sigma,
                self.decay * progress.grad_sq + (1 - self.decay) * grad_sq,
            )
        progress = NoiseBatchProgress(*averaged, progress.batch, progress.lr_scale)
        noise_scale = self.noise_scale(progress)
        if noise_scale is None or not math.isfinite(noise_scale):
            # all noise as far as the average can tell: no basis for a change
            return progress
        wanted = self.factor * noise_scale
        # the nearest power of 2 times the micro-batch, in log terms
        doublings = round(math.log2(max(wanted, self.micro_batch) / self.micro_batch))
        batch = self.micro_batch * 2**doublings
        batch = min(batch, progress.batch * 2**self.max_doublings)
        batch = max(batch, progress.batch // 2**self.max_doublings)
        batch = min(max(batch, self.min_batch), self.max_batch)
        lr_scale = self.max_batch / wanted if wanted > self.max_batch else 1.0
        return NoiseBatchProgress(*averaged, batch, lr_scale)

    def steps(self, batch: int, samples: int) -> int:
        """Training steps of `batch` samples that take about `samples` samples, at least one."""
        return max(1, round(samples / batch))

    def get_config(self) -> dict:
        return {
            "start_batch": self.start_batch,
            "min_batch": self.min_batch,
            "max_batch": self.max_batch,
            "micro_batch": self.micro_batch,
            "factor": self.factor,
            "decay": self.decay,
            "max_doublings": self.max_doublings,
        }


def _is_power_of_2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0
