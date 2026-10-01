r"""Phase 10, tool 1: Residual and Variance Scaling (RVS, §VI-G).

The single biggest post-processing tool in the paper -- 2.2 pp of the 20.2 pp the tool
set is worth -- and it costs zero MACs: it is two table look-ups on the reconstructed
latent, decoder-side, before synthesis. It buys back rate on the metrics the training
loss never saw (VMAF, FSIM, NLPD) by rescaling the decoded residual `r̂` and the variance
index `Iσ` per spatial region, keyed on how noisy that region's latent is.

Three equations (docs/02 §VI-G):

  eq. 7  σ = (32 + Σ_{8×8} Iσ) >> 6           a rounded mean of Iσ over a non-overlapping
                                              8×8 block of the /16 latent, i.e. one value
                                              per /128 grid cell; boundary blocks are
                                              completed with the pad value 1411.
  eq. 8  Iσ  += T1[modelID, id[c], σ]         additive refinement of the variance index.
  eq. 9  r̂  *= T2[modelID, id[c], σ] / 2¹⁶    16-bit fixed-point rescale of the residual.

`id[c] = GRFS_Y[c] + 2·rvs_enable_flag[comp]`, so with RVS on `id ∈ {2, 3}` and the
encoder's per-channel GRFS bit picks the curve. The tables are `[4, 4, 3968]` (4 rate
models × 4 ids × 3968 σ buckets) and the **same** tables serve luma and chroma.

Why the tables are learned here rather than copied from the standard: Part 1's normative
T1/T2 are not published as literals (docs/06). The plan's recipe is to make them trainable
parameters, freeze the rest of the codec, and fit them to the seven-metric objective. They
are initialised to the identity -- T1 = 0, T2 = 2¹⁶ -- so a checkpoint trained before RVS
existed decodes bit-for-bit the same until the tables move.

A subtlety worth stating plainly: eq. 8 updates `Iσ`, but nothing downstream of entropy
decoding in this codec reads the updated index (synthesis consumes `ŷ = p̈ + r̂`, not `Iσ`;
LSBS re-uses the pooled σ from the *original* index, per spec). So in our pipeline the
reconstruction moves through eq. 9 (T2) -- and through LSBS's eq. 10 when that tool is on.
eq. 8 is implemented for fidelity to the syntax and so the refined index is available to
any later consumer, not because it changes these pixels on its own.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

POOL = 8       # eq. 7 pools non-overlapping 8×8 blocks of the /16 latent -> /128 grid
PAD = 1411     # eq. 7 boundary pad ("the average Iσ_Y"), docs/06 isigma_pad_value
FIXED = 16     # eq. 9 fixed-point shift: r̂ · T2 / 2¹⁶


def pool_sigma(i_sigma: Tensor, buckets: int) -> Tensor:
    """eq. 7: one rounded-mean σ per 8×8 block, broadcast back to the /16 grid.

    Returns a `long` index tensor the shape of `i_sigma`, clamped into `[0, buckets)`,
    where every latent element carries the pooled value of the block that contains it.
    Both RVS and LSBS index their tables by this map, so it is computed once and shared.
    """
    b, c, h, w = i_sigma.shape
    ph, pw = (-h) % POOL, (-w) % POOL
    padded = F.pad(i_sigma.float(), (0, pw, 0, ph), value=float(PAD))
    block_sum = F.avg_pool2d(padded, POOL, stride=POOL) * (POOL * POOL)  # Σ over 8×8
    sigma = torch.div(block_sum + (POOL * POOL) // 2, POOL * POOL,
                      rounding_mode="floor")                            # (Σ + 32) >> 6
    up = F.interpolate(sigma, scale_factor=POOL, mode="nearest")[..., :h, :w]
    return up.long().clamp_(0, buckets - 1)


class ResidualVarianceScaling(nn.Module):
    """§VI-G. Owns T1/T2; applied by the codec to `(r̂, Iσ)` before synthesis.

    Identity at initialisation. `forward` takes the residual `r = ŷ − p̈` and the integer
    index, and returns the rescaled residual, the refined index, and the pooled σ (so the
    codec can hand the very same σ to LSBS). `ids` is an optional per-channel `id[c]` in
    `{0..3}`; it defaults to 2 everywhere (RVS enabled, GRFS bit 0).
    """

    def __init__(self, *, model_ids: int = 4, ids: int = 4, buckets: int = 3968):
        super().__init__()
        self.buckets = buckets
        self.T1 = nn.Parameter(torch.zeros(model_ids, ids, buckets))
        self.T2 = nn.Parameter(torch.full((model_ids, ids, buckets), float(1 << FIXED)))

    def _gather(self, table: Tensor, model_id: int, ids: Tensor,
                sigma_pool: Tensor) -> Tensor:
        # table[model_id] is [ids, buckets]; select [id[c], σ[b,c,i,j]] elementwise.
        idc = ids.view(1, -1, 1, 1).expand_as(sigma_pool)
        return table[model_id][idc, sigma_pool]

    def forward(self, r: Tensor, i_sigma: Tensor, *, model_id: int = 0,
                ids: Tensor | None = None, sigma_pool: Tensor | None = None) -> dict:
        if sigma_pool is None:
            sigma_pool = pool_sigma(i_sigma, self.buckets)
        if ids is None:
            ids = torch.full((r.shape[1],), 2, dtype=torch.long, device=r.device)
        t1 = self._gather(self.T1, model_id, ids, sigma_pool)
        t2 = self._gather(self.T2, model_id, ids, sigma_pool)
        r_out = r * t2 / float(1 << FIXED)                       # eq. 9
        i_out = (i_sigma + t1).clamp(0, self.buckets - 1)        # eq. 8 (inert here)
        return {"r": r_out, "i_sigma": i_out, "sigma_pool": sigma_pool}
