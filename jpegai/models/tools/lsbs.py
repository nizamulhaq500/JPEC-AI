r"""Phase 10, tool 2: Latent Scaling Before Synthesis (LSBS, §VI-H).

0.4 pp, and it shares everything expensive with RVS: it runs on the same reconstructed
latent, before synthesis, and indexes by the same pooled σ (eq. 7). Where RVS rescales the
residual, LSBS is a σ-dependent affine reweighting of *prediction versus residual* -- it
decides, per region, how much to trust the context model's guess `μ = p̈` against the
decoded correction `r̂`.

  eq. 10  ŷ += (r̂·TR[modelID, σ] + μ·TP[modelID, σ] + 2¹²) >> 13

`TP`, `TR` are `[4, 3968]` (4 rate models × 3968 σ buckets), learned, and initialised to
zero -- with TR = TP = 0 the bracket is `2¹² >> 13 = 0`, so LSBS is the identity until the
tables move. The shift is round-to-nearest at 13-bit fixed point (`+2¹²` then `>>13`).

The `>>13` floor has no useful gradient, so training reaches TP/TR through a straight-
through estimator: the forward value is the exact fixed-point result, the backward passes
the gradient of the pre-shift expression. Same trick the entropy quantiser uses.

`σ` is the pooled index from RVS (`pool_sigma`); LSBS never recomputes it, matching the
spec's "the same pooled σ_Y/σ_UV". `μ` is recovered as `ŷ − r̂`, which is exactly `p̈`.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .rvs import pool_sigma

FIXED = 13     # eq. 10 fixed-point shift
HALF = 1 << (FIXED - 1)   # the +2¹² round-to-nearest term


def _ste_shift(val: Tensor) -> Tensor:
    """`(val + 2¹²) >> 13` with a straight-through gradient of `val / 2¹³`.

    Forward is the true integer fixed-point result (floor after the half-add); backward
    behaves as the smooth `val / 8192`, so gradient reaches TP/TR through the bracket.
    """
    hard = torch.floor((val + HALF) / float(1 << FIXED))
    soft = val / float(1 << FIXED)
    return hard.detach() + (soft - soft.detach())


class LatentScalingBeforeSynthesis(nn.Module):
    """§VI-H. Owns TP/TR; applied by the codec after RVS, before synthesis.

    Identity at initialisation. `forward` takes the (possibly RVS-rescaled) residual `r`,
    the reconstructed latent `y_hat = p̈ + r`, and the pooled σ, and returns the reweighted
    latent. The pooled σ is required from the caller so RVS and LSBS share one pooling.
    """

    def __init__(self, *, model_ids: int = 4, buckets: int = 3968):
        super().__init__()
        self.buckets = buckets
        self.TR = nn.Parameter(torch.zeros(model_ids, buckets))
        self.TP = nn.Parameter(torch.zeros(model_ids, buckets))

    def forward(self, y_hat: Tensor, r: Tensor, sigma_pool: Tensor | None = None, *,
                i_sigma: Tensor | None = None, model_id: int = 0) -> Tensor:
        if sigma_pool is None:
            if i_sigma is None:
                raise ValueError("LSBS needs either sigma_pool or i_sigma")
            sigma_pool = pool_sigma(i_sigma, self.buckets)
        mu = y_hat - r                                   # = p̈
        tr = self.TR[model_id][sigma_pool]
        tp = self.TP[model_id][sigma_pool]
        delta = _ste_shift(r * tr + mu * tp)             # eq. 10
        return y_hat + delta
