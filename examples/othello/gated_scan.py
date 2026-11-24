"""Othello's gated scan: a ConvNeXt block's step between squares that filters along the rays from
each square, each tap gated by the squares between (see `GatedScan`). It stands in for the blocks'
depthwise filter as `core.networks.azresnet.AZResnetConfig.mixer`, named in a network spec as
scan=rays (see `game.SPEC_OPTIONS`)."""

import functools
import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from core.networks.azresnet import Norm, NormMaker, Stats, conv, conv_norm

# Othello's 8 directions, the lines along which a move flips discs
DIRECTIONS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def scan_mixer(
    steps: tuple[tuple[int, int], ...], length: int = 7, gates: int = 8
) -> functools.partial:
    """Makes the ConvNeXt blocks' `GatedScan`s of `steps`, `length` and `gates`, as
    `AZResnetConfig.mixer`."""
    return functools.partial(GatedScan, steps=steps, length=length, gates=gates)


class GatedScan(eqx.Module):
    """A depthwise filter along lines that the board can block: from each square, `length` squares
    each way along each of `steps` (dy, dx) directions, as `azresnet.Depthwise`'s taps along them
    would, but each tap's weight multiplied by the gates of the squares between it and the square
    filtered:
    y[c, p] = w[c, 0] x[c, p] + sum over directions d and distances k of
    w[c, d, k] * g[p + d] ... g[p + (k - 1) d] * x[c, p + k d].
    A gate is in (0, 1), a sigmoid of a 1x1 convolution of the input, normalized, so a line can
    stop at a square, e.g. an empty one (a selective scan along lines, as VMamba's along rows and
    columns). `gates` gate channels, each shared by a group of consecutive channels. Gates start
    near 1 (an offset of 2): the filter starts close to the plain filter along the lines."""

    weight: jax.Array
    gate_conv: eqx.nn.Conv2d
    gate_norm: Norm
    gate_offset: jax.Array
    steps: tuple[tuple[int, int], ...] = eqx.field(static=True)
    length: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        steps: tuple[tuple[int, int], ...],
        length: int,
        gates: int,
        *,
        key: jax.Array,
        make_norm: NormMaker,
        stats: Stats,
    ):
        if channels % gates:
            raise ValueError(f"{gates} gates don't divide {channels} channels")
        keys = jax.random.split(key, 2)
        self.steps = tuple((int(dy), int(dx)) for dy, dx in steps)
        self.length = length
        taps = 1 + len(steps) * length
        bound = 1 / math.sqrt(taps)
        self.weight = jax.random.uniform(
            keys[0], (channels, taps), minval=-bound, maxval=bound
        )
        self.gate_conv = conv(channels, gates, 1, keys[1])
        self.gate_norm = make_norm(gates, stats, 1)
        self.gate_offset = jnp.full((gates,), 2.0)

    @property
    def kernel_size(self) -> int:
        return 2 * self.length + 1

    def gate_logits(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        """Each gate channel's logit at each square, (gates, height, width), float32."""
        g, state = conv_norm(self.gate_conv, self.gate_norm, x, state, dtype)
        offset = self.gate_offset.reshape(-1, 1, 1)
        return g.astype(jnp.float32) + offset, state

    def gates(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        """Each gate channel's gate at each square, (gates, height, width), float32."""
        logits, state = self.gate_logits(x, state, dtype)
        return jax.nn.sigmoid(logits), state

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, dtype: jnp.dtype
    ) -> tuple[jax.Array, eqx.nn.State]:
        """The scan applied to `x`, computing in float32. Each path's product of gates is the exp
        of its gates' log-sigmoids summed, one matrix product for all paths, which also keeps the
        gradient's digits where gates are near 1 (a sigmoid's, g (1 - g), loses them). The sum
        along the lines is `line_sum`. Its matrix products ask for full precision: a GPU's
        default for float32 ones (TF32, 10-bit mantissas) was 1e-3 off."""
        channels, height, width = x.shape
        logits, state = self.gate_logits(x, state, dtype)
        gates, squares = logits.shape[0], height * width
        shape = (self.steps, self.length, height, width)
        through = jnp.exp(
            jnp.einsum(
                "eq,gq->ge",
                scan_lines(*shape).between,
                jax.nn.log_sigmoid(logits.reshape(gates, squares)),
                precision="highest",
            )
        )
        group = channels // gates
        y = line_sum(
            shape,
            self.weight.astype(jnp.float32).reshape(gates, group, -1),
            through,
            x.astype(jnp.float32).reshape(gates, group, squares),
        )
        return y.reshape(x.shape).astype(dtype), state


class Slots(NamedTuple):
    """The terms of some sums of products, a row a sum, short rows padded with terms of mask 0:
    each term's three indices. For `slot_sum` they are into the first factor, into the second and
    into the paths' gate products. `ScanLines.by_path` holds (p, q, tap) in the same fields."""

    first: np.ndarray
    second: np.ndarray
    path: np.ndarray
    mask: np.ndarray


class ScanLines(NamedTuple):
    """What `GatedScan` sums, as tables over its (square p, tap t) pairs that stay inside, tap t
    joining p to square q along a path of the squares between them.

    Attributes:
        between: (paths, squares), how many times each path passes each square. Path 0 passes
            none (it is always open), and the two ways along a line share a path.
        forward: a row for each p, its terms weight[t] * x[q]
        backward: a row for each q, its terms weight[t] * ybar[p]: the transpose
        by_path: a row for each path but 0's (empty), its terms' (p, q, t): the paths' cotangents
        by_tap: rows of up to `TAP_ROW` terms ybar[p] * x[q], each row one tap's
        tap_rows: (rows of by_tap, taps), 1 where the row is the tap's
    """

    between: np.ndarray
    forward: Slots
    backward: Slots
    by_path: Slots
    by_tap: Slots
    tap_rows: np.ndarray


# the terms in a row of the weights' cotangent: on a GPU 8 to 32 ran alike, 64 over 3x slower
TAP_ROW = 16


@functools.cache
def scan_lines(
    steps: tuple[tuple[int, int], ...], length: int, height: int, width: int
) -> ScanLines:
    """The tables of a `GatedScan` of `steps` and `length` on (height, width) inputs."""
    squares = height * width
    taps = [((0, 0), 0)] + [(step, k) for step in steps for k in range(1, length + 1)]
    paths: dict[tuple, int] = {(): 0}
    between = [np.zeros(squares, np.float32)]
    pairs = []  # (p, q, tap, path)
    for t, ((dy, dx), k) in enumerate(taps):
        for p in range(squares):
            row, col = divmod(p, width)
            if not (0 <= row + k * dy < height and 0 <= col + k * dx < width):
                continue
            q = p + k * (dy * width + dx)
            passed = tuple(sorted(p + j * (dy * width + dx) for j in range(1, k)))
            line = (min(p, q), max(p, q), passed) if passed else ()
            if line not in paths:
                paths[line] = len(between)
                counts = np.bincount(np.array(passed), minlength=squares)
                between.append(counts.astype(np.float32))
            pairs.append((p, q, t, paths[line]))

    by_p = [[] for _ in range(squares)]
    by_q = [[] for _ in range(squares)]
    by_path = [[] for _ in between]
    by_t = [[] for _ in taps]
    for p, q, t, path in pairs:
        by_p[p].append((t, q, path))
        by_q[q].append((t, p, path))
        if path:
            by_path[path].append((p, q, t))
        by_t[t].append((p, q, path))
    tap_rows, owners = [], []
    for t, terms in enumerate(by_t):
        for i in range(0, len(terms), TAP_ROW):
            tap_rows.append(terms[i : i + TAP_ROW])
            owners.append(t)
    return ScanLines(
        between=np.stack(between),
        forward=slots(by_p),
        backward=slots(by_q),
        by_path=slots(by_path),
        by_tap=slots(tap_rows),
        tap_rows=np.eye(len(taps), dtype=np.float32)[owners],
    )


def slots(rows: list[list[tuple[int, int, int]]]) -> Slots:
    """`rows` of (first, second, third index) terms as `Slots`."""
    table = np.zeros((3, len(rows), max(1, *map(len, rows))), np.int32)
    mask = np.zeros(table.shape[1:], np.float32)
    for r, row in enumerate(rows):
        for s, term in enumerate(row):
            table[:, r, s] = term
            mask[r, s] = 1
    return Slots(table[0], table[1], table[2], mask)


def slot_sum(
    table: Slots, first: jax.Array, second: jax.Array, through: jax.Array
) -> jax.Array:
    """For each row of `table`, the sum over its terms of first * second * through at the term's
    indices, (gates, group, rows), `first` and `second` (gates, group, ...) and `through`
    (gates, paths).

    It is written for how XLA compiles it on a GPU, into one kernel: the terms added one by one
    (a reduce over them ran 5x slower, its loads one after another), and `through` broadcast to
    the channels before it is indexed (indexed first, each term's is an array of its own)."""
    through = jnp.broadcast_to(through[:, None], (*first.shape[:2], through.shape[-1]))
    total = None
    for s in range(table.mask.shape[1]):
        left = first[:, :, table.first[:, s]]
        if not table.mask[:, s].all():
            left = left * table.mask[:, s]
        term = left * through[:, :, table.path[:, s]] * second[:, :, table.second[:, s]]
        total = term if total is None else total + term
    assert total is not None
    return total


def line_sum_primal(
    shape: tuple, weight: jax.Array, through: jax.Array, x: jax.Array
) -> jax.Array:
    """`GatedScan`'s sum along the lines, y[g, m, p] = sum over the taps t from square p that
    stay inside, to square q by path e, of weight[g, m, t] * through[g, e] * x[g, m, q], for
    gate g and its group's channel m. `shape` is `scan_lines`' arguments.

    Its transpose is written out (so it has no forward-mode derivative): autodiff's has a
    scatter for each lookup, and a training step took 18x as long with it. The optimization
    barriers keep each kernel on its own: without them XLA fuses them into the normalization's
    reduces and the block's sum, where they ran 2 to 4 times slower."""
    return slot_sum(scan_lines(*shape).forward, weight, x, through)


line_sum = jax.custom_vjp(line_sum_primal, nondiff_argnums=(0,))


def line_sum_fwd(shape: tuple, weight: jax.Array, through: jax.Array, x: jax.Array):
    y = jax.lax.optimization_barrier(line_sum(shape, weight, through, x))
    return y, (weight, through, x)


def line_sum_bwd(shape: tuple, saved: tuple, ybar: jax.Array):
    weight, through, x = saved
    lines = scan_lines(*shape)
    xbar = slot_sum(lines.backward, weight, ybar, through)
    p, q, t, mask = lines.by_path
    through_bar = jnp.sum(
        mask * weight[:, :, t] * ybar[:, :, p] * x[:, :, q], axis=(1, 3)
    )
    by_row = slot_sum(lines.by_tap, ybar, x, through)
    # flattened under its barrier, so the kernel keeps the batch first and reads coherently;
    # without one, XLA merges it with the batch's sum and writes gigabytes of intermediates
    by_row = jax.lax.optimization_barrier(by_row.reshape(-1)).reshape(by_row.shape)
    xbar, through_bar = jax.lax.optimization_barrier((xbar, through_bar))
    weight_bar = jnp.matmul(by_row, lines.tap_rows, precision="highest")
    return weight_bar, through_bar, xbar


line_sum.defvjp(line_sum_fwd, line_sum_bwd)
