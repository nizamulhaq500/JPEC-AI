r"""Phase 10, tools 3-6: the four decoder-side post-processing filters (§VI-M).

Unlike RVS/LSBS these run *after* synthesis, in the pixel (plane) domain, and unlike the two
table tools they carry real parameters and MACs. They are the tail of Table IV, and the
paper's own numbers put two of them slightly negative on the seven-metric average while they
lift chroma PSNR sharply -- so a filter that hurts the average is a reproduced result, not a
bug.

Four scopes, from the plan's realisation table:

  * ``LumaEnhancementFilter`` (LEF, primary)   -- edge-gated residual sharpening on luma.
  * ``InterComponentInformation`` (ICCI, both) -- cross-component: luma refines chroma and
        chroma refines luma. The only post-filter with real MAC cost (paper: 4.6 of 28
        kMAC/pxl), because it is two conv stacks and it resamples across the chroma grid.
  * ``EdgeFreeEnhancementNonlinear`` (EFE nonlinear, secondary) -- a small ReLU CNN on chroma.
  * ``EdgeFreeEnhancementLinear`` (EFE linear, secondary)       -- a conv-only (activation
        free) filter on chroma.

Every one is a *residual* whose last convolution is zero-initialised, so at init it adds
exactly zero and the codec reconstructs byte-for-byte what it did before the filter was
attached. That is the same identity-at-init contract RVS/LSBS keep, and it is what lets the
ablation harness read each filter as a clean delta against the tools-off codec; it also means
the order the codec applies them in does not matter until they are trained. All four are
optional at decode even when signalled (§VI-M) -- the codec enforces that through
`_resolve_tools`, so the modules themselves are pure functions of their input planes and hold
no enable flag of their own.

The filters are rate-model-agnostic: one set of coefficients per checkpoint, trained with the
backbone frozen, rather than a per-rate table like RVS/LSBS. The paper signals EFE
coefficients in the tile/object header, but a single learned set is the honest realisation of
"suggested" architectures the standard does not fix normatively.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _zero_init(conv: nn.Conv2d) -> nn.Conv2d:
    """Zero a residual's output conv so the filter is identity until it is trained."""
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


class LumaEnhancementFilter(nn.Module):
    """LEF (§VI-M, primary). Edge-gated residual sharpening on the luma plane.

    A two-conv body produces a correction; a fixed Sobel gradient magnitude gates it per
    pixel, so the correction lands where there is structure and is suppressed on flat regions
    -- the "edge-aware" in the plan's description. The gate is an operator, not a parameter;
    the body is learned and its output conv is zero-init, so LEF is identity at start whatever
    the gate says.
    """

    def __init__(self, *, width: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(1, width, kernel, padding=pad)
        self.conv2 = _zero_init(nn.Conv2d(width, 1, kernel, padding=pad))
        sobel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("kx", sobel.view(1, 1, 3, 3))
        self.register_buffer("ky", sobel.t().contiguous().view(1, 1, 3, 3))

    def _gate(self, luma: Tensor) -> Tensor:
        gx = F.conv2d(luma, self.kx, padding=1)
        gy = F.conv2d(luma, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        peak = mag.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        return mag / peak                                  # scale-free per image, in [0, 1]

    def forward(self, luma: Tensor) -> Tensor:
        corr = self.conv2(F.relu(self.conv1(luma)))
        return luma + self._gate(luma) * corr


class EdgeFreeEnhancementLinear(nn.Module):
    """EFE linear (§VI-M, secondary). A single activation-free conv residual on chroma.

    "Linear" is literal: one convolution, no nonlinearity, so the whole filter is an affine
    map of the chroma plane. Zero-init makes it the identity affine map at start. This is the
    cheapest of the four and, in the paper, one of the two that slightly hurt the average.
    """

    def __init__(self, *, channels: int = 2, kernel: int = 3):
        super().__init__()
        self.conv = _zero_init(nn.Conv2d(channels, channels, kernel, padding=kernel // 2))

    def forward(self, chroma: Tensor) -> Tensor:
        return chroma + self.conv(chroma)


class EdgeFreeEnhancementNonlinear(nn.Module):
    """EFE nonlinear (§VI-M, secondary). A small ReLU CNN residual on chroma.

    The nonlinear sibling of the linear EFE: same scope and residual form, one hidden ReLU
    layer so it can do more than an affine map. Output conv zero-init, hence identity at start.
    """

    def __init__(self, *, channels: int = 2, width: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(channels, width, kernel, padding=pad)
        self.conv2 = _zero_init(nn.Conv2d(width, channels, kernel, padding=pad))

    def forward(self, chroma: Tensor) -> Tensor:
        return chroma + self.conv2(F.relu(self.conv1(chroma)))


class InterComponentInformation(nn.Module):
    """ICCI (§VI-M, both). Cross-component refinement in both directions at once.

    Luma conditions a chroma correction and chroma conditions a luma correction. At 4:2:0 and
    4:2:2 the two planes live on different grids, so each direction resamples the *other*
    component onto its own grid before concatenating -- luma is area-pooled down for the chroma
    branch, chroma is nearest-upsampled for the luma branch. Both directions read the planes as
    they arrive (not each other's corrected output), so ICCI is one simultaneous cross-refine
    rather than a two-step cascade, and both output convs are zero-init so each direction is
    independently identity at start.

    This is the wide tool -- two conv stacks plus resampling -- and the plan flags it as the
    only post-filter carrying real MAC cost.
    """

    def __init__(self, *, luma_ch: int = 1, chroma_ch: int = 2, width: int = 32,
                 kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.c1 = nn.Conv2d(chroma_ch + luma_ch, width, kernel, padding=pad)
        self.c2 = _zero_init(nn.Conv2d(width, chroma_ch, kernel, padding=pad))
        self.l1 = nn.Conv2d(luma_ch + chroma_ch, width, kernel, padding=pad)
        self.l2 = _zero_init(nn.Conv2d(width, luma_ch, kernel, padding=pad))

    def forward(self, luma: Tensor, chroma: Tensor) -> tuple[Tensor, Tensor]:
        lhw, chw = luma.shape[-2:], chroma.shape[-2:]
        same = lhw == chw
        luma_on_c = luma if same else F.interpolate(luma, size=chw, mode="area")
        chroma_on_l = chroma if same else F.interpolate(chroma, size=lhw, mode="nearest")
        chroma_ref = chroma + self.c2(F.relu(self.c1(torch.cat([chroma, luma_on_c], -3))))
        luma_ref = luma + self.l2(F.relu(self.l1(torch.cat([luma, chroma_on_l], -3))))
        return luma_ref, chroma_ref

