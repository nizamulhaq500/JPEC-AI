"""Entropy models: the parts that turn a latent tensor into bits.

Two models, both from Balle et al., ICLR 2018 ("Variational image compression
with a scale hyperprior"), and both needed by every later phase of this project:

* :class:`FactorizedPrior` -- the **non-parametric density model** of Appendix
  6.1, used for the hyper latent ``z``, which has no prior of its own. JPEG AI's
  reference software calls this `FactorizedProbModel(chs_ls, max_symbol=z_range-1)`
  and instantiates one per branch (`common_modules.py:116`).
* :class:`GaussianConditional` -- ``p(y_hat | sigma)``, a discretised Gaussian
  whose scale is predicted by the hyper decoder. This is what JPEG AI's
  `sigma_quant_*` constants parameterise, and what MCM (Phase 6) conditions.

Both are written here rather than imported from compressai. Three reasons, in
order of importance:

1. **Phase 9 replaces the coder, not the model.** me-tANS needs the PMF/CDF
   construction to be ours so we can switch to `tans_mass_bits: 8` with an
   escape symbol. If the model came from compressai, Phase 9 would be a rewrite
   instead of a swap.
2. **Phase 11 needs bit-exactness.** JPEG AI's decoder is specified in integer
   arithmetic. You cannot make someone else's float pipeline bit-exact.
3. Understanding. The rate that training minimises and the bytes the coder emits
   are computed by two different code paths, and the gate for this phase is that
   they agree to within 1-2%. That is only debuggable if both paths are ours.

The rANS coder itself is still compressai's (`compressai.ans`) -- writing a
correct rANS implementation is Phase 9's job, and borrowing one now keeps this
phase's gate focused on the model.

Quantisation, and why there are two kinds
-----------------------------------------
`config.train.quantisation: {rate: noise, distortion: ste}`. These are not
interchangeable, and using one for both is a real (if popular) mistake:

* The **rate** branch adds uniform noise. The entropy model is a *continuous*
  density evaluated as ``c(y+0.5) - c(y-0.5)``; that expression is exactly the
  probability of a uniformly-dithered value, so noise is the estimator that
  matches the thing being differentiated. Rounding here would give the model
  zero gradient almost everywhere.
* The **distortion** branch rounds, with a straight-through gradient. The
  synthesis transform must see the values it will actually see at inference. Feed
  it noise and it learns to denoise, then behaves differently on real rounded
  input -- a train/test mismatch that shows up as a decode that looks worse than
  the training curve promised.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from jpegai.models.cdf import MAX_PRECISION, build_cdf_table

# ---------------------------------------------------------------------------
# Small numerical primitives
# ---------------------------------------------------------------------------
class _LowerBoundFn(torch.autograd.Function):
    """``max(x, b)`` whose gradient still flows when x is clamped but improving.

    A plain `clamp` kills the gradient for every clamped element, so a scale that
    starts below the bound can never learn its way out -- it is permanently
    frozen at the boundary. Letting the gradient through when it points back into
    the valid region fixes that while still forbidding the forward value from
    going below the bound.
    """

    @staticmethod
    def forward(ctx, x: Tensor, bound: Tensor):
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, bound = ctx.saved_tensors
        # Pass the gradient if we are inside the valid region, or if the update
        # would move us back towards it (grad_out < 0 increases x).
        pass_through = (x >= bound) | (grad_out < 0)
        return pass_through.type_as(grad_out) * grad_out, None


class LowerBound(nn.Module):
    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.tensor([float(bound)]))

    def forward(self, x: Tensor) -> Tensor:
        return _LowerBoundFn.apply(x, self.bound.to(x.dtype))


class _ClampFn(torch.autograd.Function):
    """Two-sided `LowerBound`. Same rule at both ends.

    Phase 5's `Iσ` needs this: it is an index into a fixed 3968-entry table, so
    both ends are hard limits, and a plain `clamp` would freeze any element that
    initialised outside the range -- a whole channel of the scale decoder can start
    saturated and, with the gradient killed, stay that way for the entire run.
    """

    @staticmethod
    def forward(ctx, x: Tensor, lo: Tensor, hi: Tensor):
        ctx.save_for_backward(x, lo, hi)
        return x.clamp(float(lo), float(hi))

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, lo, hi = ctx.saved_tensors
        # grad_out < 0 increases x, so it is "improving" only below the range.
        pass_through = ((x >= lo) & (x <= hi)) | ((x < lo) & (grad_out < 0)) \
            | ((x > hi) & (grad_out > 0))
        return pass_through.type_as(grad_out) * grad_out, None, None


class Clamp(nn.Module):
    """`clamp(x, lo, hi)` with the gradient still flowing when it would help."""

    def __init__(self, lo: float, hi: float):
        super().__init__()
        if not lo < hi:
            raise ValueError(f"need lo < hi, got {lo}, {hi}")
        self.register_buffer("lo", torch.tensor(float(lo)))
        self.register_buffer("hi", torch.tensor(float(hi)))

    def forward(self, x: Tensor) -> Tensor:
        return _ClampFn.apply(x, self.lo.to(x.dtype), self.hi.to(x.dtype))

    def extra_repr(self) -> str:
        return f"lo={float(self.lo):g}, hi={float(self.hi):g}"


def quantize_ste(x: Tensor) -> Tensor:
    """``round(x)`` with an identity gradient (straight-through estimator).

    Written as ``x + (round(x) - x).detach()`` rather than a custom autograd
    Function so it composes with autocast and torch.compile without special
    cases. The forward value is exactly `round(x)`; the backward is exactly 1.
    """
    return x + (torch.round(x) - x).detach()


def add_uniform_noise(x: Tensor) -> Tensor:
    """``x + U(-0.5, 0.5)`` -- the rate branch's quantisation proxy."""
    return x + torch.empty_like(x).uniform_(-0.5, 0.5)


def likelihood_to_bits(likelihood: Tensor) -> Tensor:
    """Total bits implied by a likelihood tensor. Clamped, because log(0) = -inf.

    The clamp bound matters: 1e-9 costs ~30 bits for a symbol the model thought
    impossible, which is a large but finite penalty the optimiser can act on.
    Without it a single impossible symbol makes the whole loss nan and the run is
    dead with no diagnostic.
    """
    return -torch.log2(likelihood.clamp_min(1e-9)).sum()


# ---------------------------------------------------------------------------
# Scale table -- JPEG AI eq. (13) geometry
# ---------------------------------------------------------------------------
def build_scale_table(
    minimum: float = 0.11, maximum: float = 54.82, levels: int = 32
) -> list[float]:
    """Log-uniform sigma table: ``sigma_k = minimum * exp(k * step)``.

    Defaults are JPEG AI's normative values (`entropy.sigma_quant_min`,
    `sigma_quant_max`, `sigma_quant_level`; see docs/06). The paper's eq. (13)
    with ``step = (ln(max) - ln(min)) / (levels - 1)`` is precisely a geometric
    series, so this one line implements it.

    Worth knowing: compressai's `SCALES_MIN` is **also 0.11**, and for the same
    reason -- both descend from Balle's tensorflow-compression, where 0.11 is the
    smallest scale at which a discretised Gaussian still has non-negligible mass
    outside its centre bin. Below it the distribution is a point mass and the
    entropy model stops being able to express uncertainty at all. JPEG AI kept
    the floor and lowered the ceiling (54.82 vs 256) with fewer levels (32 vs
    64), which shrinks the CDF table -- the paper's "memory-efficient" goal.

    What 32 levels costs, measured
    ------------------------------
    The training loss evaluates the rate at the *continuous* sigma out of `h_s`;
    the coder can only index one of these 32 rows. That difference is real rate,
    and it has been measured three independent ways, all agreeing:

    * closed form -- second-order expansion of the rate around a log-spaced grid;
    * synthetic -- log-uniform sigma, 32 levels: **+0.01521 bits/symbol**;
    * end to end -- Kodak through the full codec: **+1.86% to +1.92%** of a
      ~1.07 bits/symbol rate, i.e. +0.0199 bits/symbol.

    Two consequences worth stating here rather than rediscovering:

    1. It shows up as **rate, never as escapes**. `build_indexes` rounds sigma
       *up* to the next table entry, so the coder always uses a distribution at
       least as wide as the model predicted, whose CDF is at least as
       heavy-tailed. No symbol the model thought likely can fall outside the
       table. Rounding down would trade this 1.9% for out-of-range symbols, which
       is a far worse deal -- 0.63% escapes was measured at -17% rate.
    2. Any reported bitrate must come from **actual bytes**, not from the loss's
       estimate, which is optimistic by exactly this amount.

    Raising `levels` to 64 would roughly quarter the cost (the error is
    second-order in the grid step) at double the table size. JPEG AI chose 32, so
    32 is what this implements; `docs/06-normative-constants.md` carries the
    trade-off curve.
    """
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")
    if not 0 < minimum < maximum:
        raise ValueError(f"need 0 < minimum < maximum, got {minimum}, {maximum}")
    return list(np.exp(np.linspace(np.log(minimum), np.log(maximum), levels)))


# ---------------------------------------------------------------------------
# Base: CDF buffers + real coding
# ---------------------------------------------------------------------------
class EntropyModel(nn.Module):
    """Shared machinery: quantised CDF buffers, rANS encode/decode.

    Subclasses fill three buffers in :meth:`update`:

    ``_cdf``     [N, Lmax+2] int32 -- one row per distribution
    ``_cdf_len``  [N] int32        -- valid length of each row
    ``_offset``   [N] int32        -- symbol value that row i's bin 0 represents

    ``_offset`` is the one that bites. A Gaussian table for sigma=10 covers
    symbols [-60, +60], so symbol -60 must index bin 0: ``bin = symbol - offset``
    with ``offset = -60``. Get its sign wrong and encoding still "works" -- it
    just codes the wrong bins, and the decoder faithfully returns the wrong
    values. There is no exception, only a corrupt image.
    """

    def __init__(self, precision: int = MAX_PRECISION):
        super().__init__()
        if not 1 <= precision <= MAX_PRECISION:
            raise ValueError(f"precision must be in [1, {MAX_PRECISION}]")
        self.precision = int(precision)
        # Empty until update(); sized on first update, hence the load hook below.
        self.register_buffer("_cdf", torch.IntTensor())
        self.register_buffer("_cdf_len", torch.IntTensor())
        self.register_buffer("_offset", torch.IntTensor())

    # -- state_dict interop ------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Resize the CDF buffers to match the checkpoint before loading.

        Without this, loading a trained checkpoint into a freshly constructed
        model raises a size-mismatch on `_cdf` (empty vs [N, L]), because the
        table's width depends on the *learned* quantiles. Every learned-codec
        codebase needs this hook and it is always discovered the hard way.
        """
        for name in ("_cdf", "_cdf_len", "_offset"):
            key = prefix + name
            if key in state_dict:
                buf = getattr(self, name)
                incoming = state_dict[key]
                if buf.shape != incoming.shape:
                    setattr(self, name, incoming.clone())
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def ready(self) -> bool:
        """True once :meth:`update` has built the tables. Coding raises before it."""
        return self._cdf.numel() > 0

    def _check_ready(self) -> None:
        if not self.ready:
            raise RuntimeError(
                f"{type(self).__name__}: CDF tables not built. Call model.update() "
                "after training and before compress()/decompress()."
            )

    def table_bytes(self) -> int:
        """Size of the CDF tables in bytes -- the decoder-side memory the paper
        cares about (section VI-D, 'memory-efficient'). Reported by selftest."""
        return int(self._cdf.numel() * 4)

    # -- real coding -------------------------------------------------------
    def _encode(self, symbols: Tensor, indexes: Tensor) -> bytes:
        from compressai.ans import RansEncoder

        self._check_ready()
        if symbols.shape != indexes.shape:
            raise ValueError(f"shape mismatch {tuple(symbols.shape)} vs "
                             f"{tuple(indexes.shape)}")
        return RansEncoder().encode_with_indexes(
            symbols.reshape(-1).to(torch.int32).tolist(),
            indexes.reshape(-1).to(torch.int32).tolist(),
            self._cdf.tolist(),
            self._cdf_len.reshape(-1).to(torch.int32).tolist(),
            self._offset.reshape(-1).to(torch.int32).tolist(),
        )

    def _decode(self, stream: bytes, indexes: Tensor) -> Tensor:
        from compressai.ans import RansDecoder

        self._check_ready()
        values = RansDecoder().decode_with_indexes(
            stream,
            indexes.reshape(-1).to(torch.int32).tolist(),
            self._cdf.tolist(),
            self._cdf_len.reshape(-1).to(torch.int32).tolist(),
            self._offset.reshape(-1).to(torch.int32).tolist(),
        )
        return torch.tensor(values, dtype=torch.int32,
                            device=indexes.device).reshape(indexes.shape)


# ---------------------------------------------------------------------------
# Non-parametric factorised prior  (Balle 2018 Appendix 6.1)
# ---------------------------------------------------------------------------
class FactorizedPrior(EntropyModel):
    """Learned per-channel density with no side information.

    The model is a small monotonically-increasing MLP ``c: R -> (0, 1)`` per
    channel, representing the CDF directly. Monotonicity is *structural*, not a
    penalty: every weight matrix is passed through softplus so it is positive,
    and the nonlinearity ``g(u) = u + a * tanh(u)`` with ``a = tanh(a_hat) in
    (-1, 1)`` has derivative ``1 + a * sech^2(u) > 0``. A composition of
    increasing functions is increasing, so ``c`` is a valid CDF by construction
    at every point in training, including step 0.

    That is the whole trick of Appendix 6.1, and it is why this beats fitting a
    parametric family: no distributional assumption, but still a guaranteed CDF.

    The probability of an integer symbol is then just a difference of the CDF at
    the bin edges, ``c(y + 0.5) - c(y - 0.5)``.
    """

    def __init__(
        self,
        channels: int,
        filters: tuple[int, ...] = (3, 3, 3, 3),
        init_scale: float = 10.0,
        tail_mass: float = 1e-9,
        precision: int = MAX_PRECISION,
    ):
        super().__init__(precision=precision)
        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        dims = (1,) + self.filters + (1,)
        # Spread init_scale over the layers so the composed function starts with
        # roughly the intended input range rather than one layer dominating.
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))

        for i in range(len(self.filters) + 1):
            # softplus(init) = 1 / (scale * fan_out): pre-invert softplus so the
            # *effective* weight is what we want, since the forward pass applies
            # softplus. log(expm1(v)) is the exact inverse of log1p(exp(v)).
            init = float(np.log(np.expm1(1 / scale / dims[i + 1])))
            self.register_parameter(
                f"_matrix{i}",
                nn.Parameter(torch.full((self.channels, dims[i + 1], dims[i]), init)),
            )
            bias = torch.empty(self.channels, dims[i + 1], 1).uniform_(-0.5, 0.5)
            self.register_parameter(f"_bias{i}", nn.Parameter(bias))
            if i < len(self.filters):
                # Start at zero: tanh(0) = 0, so the model begins as a pure
                # monotone linear chain and grows curvature only as needed.
                self.register_parameter(
                    f"_factor{i}",
                    nn.Parameter(torch.zeros(self.channels, dims[i + 1], 1)),
                )

        # Three tracked points of each channel's distribution: the tail_mass/2
        # quantile, the median, and the 1-tail_mass/2 quantile. They are learned
        # by an auxiliary loss (see `aux_loss`) and used at update() time to
        # decide how wide that channel's CDF table needs to be.
        q = torch.tensor([-self.init_scale, 0.0, self.init_scale])
        self.quantiles = nn.Parameter(q.repeat(self.channels, 1, 1))
        target = float(np.log(2 / self.tail_mass - 1))
        self.register_buffer("target", torch.tensor([-target, 0.0, target]))

    # -- the monotone MLP --------------------------------------------------
    def _logits_cumulative(self, x: Tensor, stop_gradient: bool = False) -> Tensor:
        """Pre-sigmoid CDF ("logits") for input shaped [C, 1, N].

        Returning logits rather than probabilities is deliberate: the difference
        of two sigmoids loses all precision when both are near 0 or near 1, which
        is exactly the tail where rare symbols live and where rate is decided.
        :meth:`_likelihood` reconstructs the difference stably from the logits.
        """
        logits = x
        for i in range(len(self.filters) + 1):
            m = getattr(self, f"_matrix{i}")
            b = getattr(self, f"_bias{i}")
            if stop_gradient:
                m, b = m.detach(), b.detach()
            logits = torch.matmul(F.softplus(m), logits) + b
            if i < len(self.filters):
                a = getattr(self, f"_factor{i}")
                if stop_gradient:
                    a = a.detach()
                logits = logits + torch.tanh(a) * torch.tanh(logits)
        return logits

    def _likelihood(self, x: Tensor, stop_gradient: bool = False):
        lower = self._logits_cumulative(x - 0.5, stop_gradient)
        upper = self._logits_cumulative(x + 0.5, stop_gradient)
        # Evaluate on whichever side of the distribution keeps both sigmoids in
        # their accurate range. If the interval sits in the upper tail, both
        # sigmoid()s are ~1 and their difference is catastrophic cancellation;
        # flipping the sign reflects the interval into the lower tail where
        # sigmoid is accurate to full precision. |.| because the flip reverses
        # the order of the two terms. sign is detached: it is a branch, not a
        # quantity to differentiate.
        sign = -torch.sign(lower + upper).detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        return likelihood, lower, upper

    def medians(self) -> Tensor:
        """Per-channel median, shaped [C, 1, 1]. Subtracted before rounding so
        the quantisation grid is centred on each channel's mode instead of on
        zero -- worth real bits when a channel's distribution is offset."""
        return self.quantiles[:, :, 1:2]

    def aux_loss(self) -> Tensor:
        """Drives `quantiles` to the actual tail_mass/2, 0.5 and 1-tail_mass/2
        points of the learned CDF.

        This is a **separate optimiser** from the RD loss, on purpose: it does not
        trade off against rate or distortion, it only keeps the table extent
        honest. If it is left out, `update()` builds tables from whatever the
        quantiles were initialised to, and symbols fall outside the table -- which
        is one of the two ways "estimated rate != actual bytes" happens.
        """
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        return torch.abs(logits - self.target).sum()

    # -- forward -----------------------------------------------------------
    def forward(self, x: Tensor, *, noise: bool = None, ste: bool = True):
        """Returns ``(x_hat, likelihood)``, both shaped like `x`.

        `noise` defaults to `self.training`. With both `noise` and `ste` set, the
        two branches are computed separately: likelihood from the noisy value,
        `x_hat` from the rounded one. That is `config.train.quantisation`.
        """
        if noise is None:
            noise = self.training
        b, c = x.shape[0], x.shape[1]
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")

        # [B, C, ...] -> [C, 1, B*...]: the MLP is per-channel, and matmul wants
        # the channel dimension first so each channel gets its own weights.
        x_c = x.transpose(0, 1).reshape(c, 1, -1)
        med = self.medians()
        centred = x_c - med

        rate_in = add_uniform_noise(centred) if noise else torch.round(centred)
        likelihood, _, _ = self._likelihood(rate_in)

        hat_c = (quantize_ste(centred) if ste else torch.round(centred)) + med

        def back(t: Tensor) -> Tensor:
            return t.reshape(c, b, *x.shape[2:]).transpose(0, 1)

        return back(hat_c), back(likelihood)

    # -- tables and coding -------------------------------------------------
    @torch.no_grad()
    def _density_extent(self, max_half_width: int = 512):
        """Table extent read off the learned density rather than off `quantiles`.

        Returns the smallest integer ``(minima, maxima)`` per channel whose
        *outside* mass is at most ``tail_mass / 2`` on each side, measured in the
        centred coordinate `forward` uses -- so the returned interval is
        ``[-minima, +maxima]`` in symbol space, and it need not be symmetric.

        This exists because `quantiles` is only an *estimate* of those points,
        maintained by a separate optimiser (:meth:`aux_loss`) that converges on
        its own schedule. Reading the density directly is exact by construction
        and costs one MLP evaluation on a small grid.
        """
        # self.target[0] is logit(tail_mass / 2) -- the same threshold aux_loss
        # drives `quantiles[:, :, 0]` toward, so a converged model agrees with the
        # quantile-derived extent and this method changes nothing.
        thr = float(self.target[0])
        dev = self.quantiles.device
        half = 16
        while True:
            v = torch.arange(-half, half + 1, dtype=torch.float32, device=dev)
            grid = v[None, None, :].expand(self.channels, 1, -1)
            lower = self._logits_cumulative(grid - 0.5, stop_gradient=True)[:, 0, :]
            upper = self._logits_cumulative(grid + 0.5, stop_gradient=True)[:, 0, :]
            # The CDF is monotone in v (all matrices pass through softplus), so
            # `keep_lo` is a True prefix and `keep_hi` a True suffix. Counting is
            # therefore enough to locate each boundary -- no search needed.
            keep_lo = lower <= thr            # P(symbol < v - 0.5) <= tail_mass/2
            keep_hi = upper >= -thr           # P(symbol > v + 0.5) <= tail_mass/2
            covered = bool(keep_lo[:, 0].all()) and bool(keep_hi[:, -1].all())
            if covered or half >= max_half_width:
                break
            half *= 2
        n = 2 * half + 1
        lo_i = torch.clamp(keep_lo.sum(dim=1) - 1, min=0)
        hi_i = torch.clamp(n - keep_hi.sum(dim=1), max=n - 1)
        minima = torch.clamp(-v[lo_i], min=0).ceil().int()
        maxima = torch.clamp(v[hi_i], min=0).ceil().int()
        return minima, maxima

    @torch.no_grad()
    def update(self, force: bool = False) -> bool:
        """Build the quantised CDF table from the learned density.

        Table extent is read off the density by :meth:`_density_extent`: channel k
        covers ``[median - minima_k, median + maxima_k]``. Channels with a tight
        distribution get a short row; the array is padded to the widest.
        """
        if self.ready and not force:
            return False

        med = self.quantiles[:, 0, 1]
        # Extent from the density, not from `quantiles`. `quantiles` is trained by a
        # separate optimiser (:meth:`aux_loss`) and on an unconverged model it is
        # wrong in two ways at once: `med` sits off the density's mode, so symbols
        # are not centred on zero, and the interval is too narrow to reach where
        # they actually land. The result is symbols outside the table, each paying
        # an escape symbol plus a bypass-coded raw value -- roughly 8 bits where an
        # in-table rare symbol would have cost a fraction of one.
        #
        # Measured on the phase-6 two-branch chroma hyper-latent (`z_uv`, 96
        # channels): 2 channels had |med| ~ 1.8 with a 3-symbol row [-1, +1] while
        # their symbols reached +-2, giving 2072 escapes over the 24 Kodak images and
        # **+11.5%** on that stream at beta=0.012, +3.8% at 0.03. Reading the extent
        # off the density removes every escape on all eight ladder checkpoints
        # measured, and lands `z_uv` at -0.5% / +1.3% of the `forward()` estimate.
        # Payload totals move -1.06% (`ladder_p6` beta 0.002) to -0.41%
        # (`ladder_p5` beta 0.012); the two 3,000-step probe ladders, which had no
        # escapes to recover, cost +0.02%, which is ~25 B of CDF quantisation noise.
        #
        # `med` itself is deliberately left alone: `forward` centres on it, so the
        # rate loss was trained against the density evaluated at `x - med`, and
        # changing it here would make the table describe a different distribution
        # than the one the model was optimised for. Only the extent is at issue.
        #
        # This is a coder-side change only. Decoded latents and `x_hat` are
        # bit-identical before and after, so no checkpoint is invalidated.
        minima, maxima = self._density_extent()

        pmf_length = (minima + maxima + 1).int()
        max_length = int(pmf_length.max().item())

        # Row k is sampled at `-minima_k + [0 .. max_length-1]`, i.e. at the SYMBOL
        # values themselves; entries past pmf_length[k] are junk and
        # build_cdf_table ignores them.
        #
        # `-minima`, not `median - minima`. This has to match `forward`, which
        # centres first (`centred = x - median`) and then evaluates the density at
        # `round(centred)` -- so the density's input coordinate *is* the symbol
        # value, and bin `v` must be sampled at `v`. Sampling at `median + v`
        # instead makes the table a shifted copy of the distribution the rate loss
        # was trained against, and the coder then pays for the shift: measured at
        # **+63%** on a two-branch chroma hyper-latent stream (median ~ +-1.4) and
        # +3.9% on the luma one (median ~ +-0.24). The error is proportional to
        # |median|, which is why it hides on a converged model -- the aux loss
        # drives `median` toward the density's own median, and at the fixed point
        # both samplings agree. It only shows up while `median` is still moving,
        # which is every run that has not fully converged, and every early gate
        # check in every run. `_offset = -minima` is already the symbol value of
        # bin 0, so it needs no matching change.
        dev = med.device
        start = (-minima).to(torch.float32)
        samples = torch.arange(max_length, device=dev, dtype=torch.float32)
        samples = samples[None, :] + start[:, None]

        pmf, lower, upper = self._likelihood(samples.unsqueeze(1), stop_gradient=True)
        pmf = pmf[:, 0, :]
        # Mass outside the table, both sides, as one escape symbol. sigmoid(lower)
        # is P(below the table); sigmoid(-upper) is P(above it).
        tail = (torch.sigmoid(lower[:, 0, 0]) + torch.sigmoid(-upper[:, 0, -1]))

        cdfs, lengths = build_cdf_table(
            pmf.cpu().numpy(), tail.cpu().numpy(), pmf_length.cpu().numpy(),
            precision=self.precision,
        )
        self._cdf = torch.from_numpy(cdfs).to(dev)
        self._cdf_len = torch.from_numpy(lengths).to(dev)
        self._offset = (-minima).to(torch.int32).to(dev)
        return True

    def _indexes_like(self, shape: torch.Size, device) -> Tensor:
        """Row index per element: for a factorised model that is just the channel."""
        view = [1] * len(shape)
        view[1] = -1
        idx = torch.arange(self.channels, device=device).reshape(view).int()
        return idx.expand(shape[0], self.channels, *shape[2:]).contiguous()

    def compress(self, x: Tensor) -> list[bytes]:
        """One bytes object per batch element."""
        self._check_ready()
        med = self.medians().detach()
        med = med.reshape(1, self.channels, *([1] * (x.dim() - 2))).to(x.device)
        symbols = torch.round(x - med).to(torch.int32)
        idx = self._indexes_like(x.shape, x.device)
        return [self._encode(symbols[i:i + 1], idx[i:i + 1]) for i in range(x.shape[0])]

    def decompress(self, streams: list[bytes], spatial: tuple[int, ...],
                   device=None) -> Tensor:
        self._check_ready()
        device = device or self._cdf.device
        shape = torch.Size((len(streams), self.channels, *spatial))
        idx = self._indexes_like(shape, device)
        out = torch.empty(shape, dtype=torch.float32, device=device)
        med = self.medians().detach().reshape(
            1, self.channels, *([1] * len(spatial))).to(device)
        for i, s in enumerate(streams):
            out[i:i + 1] = self._decode(s, idx[i:i + 1]).float()
        return out + med


# ---------------------------------------------------------------------------
# Conditional Gaussian
# ---------------------------------------------------------------------------
class GaussianConditional(EntropyModel):
    """``p(y_hat | sigma)`` (and optionally mu) as a discretised Gaussian.

    ``P(y) = Phi((y + 0.5 - mu)/sigma) - Phi((y - 0.5 - mu)/sigma)``

    The scale is *quantised to a table* before coding, because the coder needs a
    finite set of CDFs. That quantisation is not a shortcut -- it is normative in
    JPEG AI, where `Isigma` is an integer index and the whole RVS/LSBS tooling
    (Phase 10) is indexed by sigma class. Using the table for the *estimated*
    rate too would be more faithful still, but it is non-differentiable; we
    follow the reference and use continuous sigma in training, then measure the
    resulting mismatch in :mod:`jpegai.models.selftest` rather than assume it
    away.
    """

    def __init__(
        self,
        scale_table: list[float] | None = None,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        precision: int = MAX_PRECISION,
    ):
        super().__init__(precision=precision)
        table = list(scale_table) if scale_table is not None else build_scale_table()
        if len(table) < 1:
            raise ValueError("scale_table must be non-empty")
        if list(table) != sorted(table) or any(s <= 0 for s in table):
            raise ValueError("scale_table must be sorted and strictly positive")
        self.tail_mass = float(tail_mass)
        self.register_buffer("scale_table", torch.tensor(table, dtype=torch.float32))
        self.lower_bound_scale = LowerBound(scale_bound)

    @staticmethod
    def _standard_cdf(x: Tensor) -> Tensor:
        """Standard normal CDF via erfc.

        ``0.5 * erfc(-x/sqrt2)`` rather than ``0.5 * (1 + erf(x/sqrt2))``: for
        very negative x the erf form computes ``1 + (-0.999...)`` and loses every
        significant digit, while erfc returns the small number directly. The tail
        is where rare symbols are, and rare symbols are where the bits are.
        """
        return 0.5 * torch.erfc(-(2 ** -0.5) * x)

    def _likelihood(self, x: Tensor, scales: Tensor, means: Tensor | None = None):
        values = x if means is None else x - means
        scales = self.lower_bound_scale(scales)
        # |values| folds the two tails together, which halves the erfc range and
        # keeps the evaluation on the accurate side for both signs.
        values = torch.abs(values)
        upper = self._standard_cdf((0.5 - values) / scales)
        lower = self._standard_cdf((-0.5 - values) / scales)
        return upper - lower

    def forward(self, x: Tensor, scales: Tensor, means: Tensor | None = None,
                *, noise: bool = None, ste: bool = True):
        if noise is None:
            noise = self.training
        centred = x if means is None else x - means
        rate_in = add_uniform_noise(centred) if noise else torch.round(centred)
        likelihood = self._likelihood(rate_in, scales)
        hat = quantize_ste(centred) if ste else torch.round(centred)
        if means is not None:
            hat = hat + means
        return hat, likelihood

    @torch.no_grad()
    def update(self, force: bool = False) -> bool:
        if self.ready and not force:
            return False
        from scipy.stats import norm

        # How far out the table must reach so that only tail_mass is left over.
        # For tail_mass=1e-9 this is ~6.11 sigma.
        multiplier = float(-norm.ppf(self.tail_mass / 2))
        centre = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = (2 * centre + 1).int()
        max_length = int(pmf_length.max().item())

        # Row i is centred: symbol 0 of the row is -centre[i]. Folding with abs()
        # gives the distance from the mean, which is all the Gaussian needs.
        dist = torch.abs(
            torch.arange(max_length, device=centre.device).int() - centre[:, None]
        ).float()
        s = self.scale_table.unsqueeze(1)
        upper = self._standard_cdf((0.5 - dist) / s)
        lower = self._standard_cdf((-0.5 - dist) / s)
        pmf = upper - lower
        tail = 2 * lower[:, 0]              # symmetric, so both tails at once

        cdfs, lengths = build_cdf_table(
            pmf.cpu().numpy(), tail.cpu().numpy(), pmf_length.cpu().numpy(),
            precision=self.precision,
        )
        dev = self.scale_table.device
        self._cdf = torch.from_numpy(cdfs).to(dev)
        self._cdf_len = torch.from_numpy(lengths).to(dev)
        self._offset = (-centre).to(torch.int32).to(dev)
        return True

    def build_indexes(self, scales: Tensor) -> Tensor:
        """Map each continuous scale to its row in the table.

        Implemented as a count of table entries below the scale, which is a
        searchsorted without the import and works elementwise on any shape.
        Equivalent to JPEG AI's `Isigma`, and the reason `sigma_quant_level` is
        a real constraint rather than a resolution knob.

        Phase 5 note: this is the *float* path, and it is not bit-exact across
        implementations -- a σ that lands one float32 ULP above a table entry moves
        one row. Where an integer `Iσ` exists, pass `SigmaIndex.table_row(Iσ)` to
        `compress`/`decompress` as `indexes=` instead. See `jpegai.models.hyper`.
        """
        scales = self.lower_bound_scale(scales)
        idx = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            idx = idx - (scales <= s).int()
        return idx

    def _rows(self, scales: Tensor, indexes: Tensor | None) -> Tensor:
        """The CDF rows to code with: explicit if given, else derived from σ.

        `indexes` exists so the integer-`Iσ` path can bypass `build_indexes`
        entirely rather than round-trip through a float σ and hope the two sides
        compute the same exponential. Validated rather than trusted: an index
        tensor of the wrong shape or out of range would otherwise produce a
        bitstream that decodes to noise, with the coder reporting success.
        """
        if indexes is None:
            return self.build_indexes(scales)
        if indexes.shape != scales.shape:
            raise ValueError(f"indexes shape {tuple(indexes.shape)} != scales "
                             f"shape {tuple(scales.shape)}")
        rows = indexes.to(torch.int32)
        n = len(self.scale_table)
        lo, hi = int(rows.min()), int(rows.max())
        if lo < 0 or hi >= n:
            raise ValueError(f"indexes must be in [0, {n - 1}], got [{lo}, {hi}]")
        return rows

    def compress(self, x: Tensor, scales: Tensor, means: Tensor | None = None, *,
                 indexes: Tensor | None = None) -> list[bytes]:
        self._check_ready()
        idx = self._rows(scales, indexes)
        values = x if means is None else x - means
        symbols = torch.round(values).to(torch.int32)
        return [self._encode(symbols[i:i + 1], idx[i:i + 1]) for i in range(x.shape[0])]

    def decompress(self, streams: list[bytes], scales: Tensor,
                   means: Tensor | None = None, *,
                   indexes: Tensor | None = None) -> Tensor:
        self._check_ready()
        idx = self._rows(scales, indexes)
        out = torch.empty_like(scales)
        for i, s in enumerate(streams):
            out[i:i + 1] = self._decode(s, idx[i:i + 1]).float()
        return out if means is None else out + means
