"""Two-branch YCbCr codec — JPEG AI §VI-A. Phase 4 items 2-6.

Phase 3's model codes RGB through one autoencoder. JPEG AI does not: luma and chroma
go through *separate* transforms with separate latents and separate hyperpriors, and
they exchange information at exactly two places. This module is that structure.

    x_Y   [1, H, W]      --4 stride-2-->  y_Y  [N_luma,   H/16, W/16]
    x_UV  [2, H/2, W/2]  --3 stride-2-->  y_UV [N_chroma, H/16, W/16]
                             ^                     |
                     x_Y downsampled          concat with y_Y  (eq. 3)
                     (link 1, encoder)        (link 2, decoder)

Why two branches at all
-----------------------
Three reasons, and the third is the one the paper cares about most:

1. Chroma carries far less information per sample, so spending a third of a
   full-resolution latent on it is waste. At 4:2:0 the chroma branch starts from a
   quarter of the samples.
2. The luma branch can be decoded *alone* -- see `luma_only` on
   :meth:`TwoBranchCodec.decompress`. Machine-consumption tasks mostly want luma, and
   this makes that a partial decode rather than a full one.
3. It lets the two components have different tools. MCM is luma-only (Phase 6), and
   the chroma post-filters (EFE, Phase 10) are chroma-only.

The stride schedule, which the paper leaves half-specified
----------------------------------------------------------
Both latents must land on the *same* grid or eq. (3) cannot concatenate them. The
paper says 3 stride-2 stages for 4:2:0 and 4 for 4:4:4, and says nothing about 4:2:2 --
where the chroma grid is already half width, so it needs 16x vertical and 8x
horizontal downsampling. That is not achievable with isotropic stride 2, and
:func:`stride_schedule` therefore emits an anisotropic stage. Our reading, not the
standard's; flagged in docs/06 as an open question.

What this module does not yet do
--------------------------------
The entropy coding is still Phase 3's: a scale or mean-scale hyperprior per branch,
no MCM (Phase 6), no residual/latent-domain prediction split (Phase 5), one synthesis
transform per branch rather than three selectable ones (Phase 7). The branch
*structure* is what Phase 4 buys, and it is what the later phases plug into.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from jpegai.models.colour import (
    FORMATS, get_format, luma_for_secondary, merge_planes, rgb_to_ycbcr_bt709,
    split_planes, ycbcr_to_rgb_bt709,
)
from jpegai.models.entropy import (
    FactorizedPrior, GaussianConditional, build_scale_table,
)
from jpegai.models.hyper import SigmaIndex, SplitHyperBranch
from jpegai.models.gain import (reject_gain, spatial_bits, spatial_reconstruct,
                                spatial_residual)
from jpegai.models.hyperprior import (
    AnalysisTransform, HyperAnalysis, HyperSynthesisScale, SynthesisTransform,
)
from jpegai.models.layers import activation, conv, deconv, pad_to_multiple, unpad

#: Total downsampling from full resolution to the latent grid. Four stride-2 stages.
LATENT_STRIDE = 16


def stride_schedule(fmt, total: int = LATENT_STRIDE) -> list[tuple[int, int]]:
    """Per-stage `(vertical, horizontal)` strides from a chroma grid to the latent.

    The chroma plane is already downsampled by `(f.ver, f.hor)` relative to luma, so
    the branch only has to make up the difference. Returns one tuple per stage:

        4:4:4 -> [(2,2), (2,2), (2,2), (2,2)]     4 stages, matching luma
        4:2:0 -> [(2,2), (2,2), (2,2)]            3 stages, the paper's number
        4:2:2 -> [(2,2), (2,2), (2,2), (2,1)]     4 vertical, 3 horizontal

    Isotropic stages come first so the tensor shrinks on both axes as early as
    possible -- the anisotropic stage is the most expensive one to run at width, and
    putting it last runs it on the smallest tensor.
    """
    f = get_format(fmt)
    need_v, need_h = total // f.ver, total // f.hor
    for n, axis in ((need_v, "vertical"), (need_h, "horizontal")):
        if n < 1 or (n & (n - 1)):
            raise ValueError(
                f"chroma format {f.name} needs {axis} downsampling of {n}x to reach "
                f"the /{total} latent grid, which is not a power of two"
            )
    nv, nh = need_v.bit_length() - 1, need_h.bit_length() - 1     # log2 of each
    # Stride-2 stages first on each axis independently, so the axis needing fewer
    # of them runs its stride-1 stages last -- which is what makes the isotropic
    # (2,2) stages lead. See the docstring.
    sched = [(2 if i < nv else 1, 2 if i < nh else 1) for i in range(max(nv, nh))]
    assert math.prod(s[0] for s in sched) == need_v
    assert math.prod(s[1] for s in sched) == need_h
    return sched


class SecondaryAnalysis(nn.Module):
    """Chroma (plus the luma link) -> the chroma latent, on the luma latent's grid.

    Takes `2 + 1` input channels: Cb, Cr, and the downsampled luma of Phase 4 item 4.
    That third channel is the paper's single encoder-side cross-component link, and
    it is cheap for what it does -- chroma edges sit where luma edges sit, so telling
    the chroma encoder about them saves it from having to infer structure it can see
    for free.
    """

    def __init__(self, latent: int, widths, fmt="420", *,
                 in_channels: int = 3, activation_name: str = "relu",
                 kernel: int = 5):
        super().__init__()
        self.fmt = get_format(fmt)
        self.schedule = stride_schedule(self.fmt)
        widths = self._fit_widths(list(widths), len(self.schedule), latent)

        layers: list[nn.Module] = []
        prev = in_channels
        for i, (w, s) in enumerate(zip(widths, self.schedule)):
            layers.append(conv(prev, w, kernel, stride=s))
            if i < len(widths) - 1:                  # latent stays unbounded
                layers.append(activation(activation_name, w))
            prev = w
        self.body = nn.Sequential(*layers)

    @staticmethod
    def _fit_widths(widths, stages: int, latent: int) -> list[int]:
        """Reuse the primary branch's width list for a shorter branch.

        The config carries one `analysis_width` of four entries. A 4:2:0 secondary
        branch has three stages, so one has to go: drop from the *front*, keeping the
        wide middle stage and the final projection. Dropping from the back would drop
        the projection and silently give the wrong latent width.
        """
        widths = list(widths)[-stages:] if len(widths) >= stages else \
            [widths[0]] * (stages - len(widths)) + list(widths)
        widths[-1] = latent
        return widths

    def forward(self, uv: Tensor, luma_supp: Tensor) -> Tensor:
        if luma_supp.shape[-2:] != uv.shape[-2:]:
            raise ValueError(
                f"the luma link must be on the chroma grid: got luma "
                f"{tuple(luma_supp.shape[-2:])} vs chroma {tuple(uv.shape[-2:])}"
            )
        return self.body(torch.cat([uv, luma_supp], dim=-3))


class SecondarySynthesis(nn.Module):
    """eq. (3): `concat(ŷ_UV, ŷ_Y)` -> reconstructed chroma, at the internal format.

    The luma latent is concatenated at full width rather than projected down first.
    That is what the reference software does (`*_sec.py` take `chs_ls=<chroma>`,
    `chs_supp=<luma>`) and it is the decoder-side half of the cross-component
    design: chroma reconstruction gets to see everything luma reconstruction saw,
    which is why chroma can afford so few of its own bits.
    """

    def __init__(self, latent: int, supp: int, widths, fmt="420", *,
                 out_channels: int = 2, activation_name: str = "relu",
                 kernel: int = 5):
        super().__init__()
        self.fmt = get_format(fmt)
        self.schedule = list(reversed(stride_schedule(self.fmt)))
        self.latent, self.supp = int(latent), int(supp)

        n = len(self.schedule)
        widths = list(widths)[:n - 1] if len(widths) >= n - 1 else \
            list(widths) + [widths[-1]] * (n - 1 - len(widths))

        layers: list[nn.Module] = []
        prev = latent + supp                          # <- eq. (3)
        for w, s in zip(widths, self.schedule[:-1]):
            layers.append(deconv(prev, w, kernel, stride=s))
            layers.append(activation(activation_name, w, inverse=True))
            prev = w
        layers.append(deconv(prev, out_channels, kernel, stride=self.schedule[-1]))
        self.body = nn.Sequential(*layers)

    def forward(self, y_uv: Tensor, y_luma: Tensor) -> Tensor:
        if y_uv.shape[-2:] != y_luma.shape[-2:]:
            raise ValueError(
                f"eq. (3) needs both latents on one grid: chroma "
                f"{tuple(y_uv.shape[-2:])} vs luma {tuple(y_luma.shape[-2:])}. "
                f"Check stride_schedule against the internal chroma format."
            )
        return self.body(torch.cat([y_uv, y_luma], dim=-3))


def _no_gain(delta_beta, q_index) -> None:
    """Refuse a quality map on the Phase 4 mean-scale branch.

    This branch's `h_s` emits sigma directly as a raw network output. There is no
    integer sigma index for an additive offset to live on, so eq. (10)'s change of
    units has nothing to change units *into*: the gain would have to become a
    multiply on a float sigma, which is a different mechanism with a different
    quantisation error and would not interoperate with the split-hyper bitstream.
    `TwoBranchCodec.__init__` already rejects `gain=True` without `split_hyper=True`;
    these three methods accept the kwargs only so the two branch types stay
    signature-compatible, and refuse so that a rate request is never dropped.
    """
    reject_gain(delta_beta, q_index, "HyperpriorBranch",
                "It predicts sigma directly and has no integer index to offset; "
                "gain=True requires split_hyper=True.")


class HyperpriorBranch(nn.Module):
    """One branch's side-information path: h_a, h_s, and the factorised prior for z.

    Factored out rather than duplicated because the two branches differ only in
    width, and because `compress`/`decompress` contain the z-before-sigma ordering
    that Phase 3 established is easy to get wrong and impossible to detect without a
    round-trip test. One copy, two users.

    The `GaussianConditional` is *passed in*, not owned: its table is indexed by
    quantised sigma level and has no channel dimension, so both branches share one.
    Two copies would double `table_bytes()` while representing the same 32-level
    grid, and JPEG AI has one sigma grid.
    """

    def __init__(self, latent: int, hyper: int, *, mean_scale: bool = True,
                 activation_name: str = "relu", precision: int = 16):
        super().__init__()
        self.latent, self.hyper = int(latent), int(hyper)
        self.mean_scale = bool(mean_scale)
        self.h_a = HyperAnalysis(latent, hyper, activation_name=activation_name)
        if mean_scale:
            mid = (hyper * 3) // 2
            self.h_s = nn.Sequential(
                deconv(hyper, mid, 5, stride=2),
                activation(activation_name, mid, inverse=True),
                deconv(mid, mid, 5, stride=2),
                activation(activation_name, mid, inverse=True),
                conv(mid, 2 * latent, 3, stride=1),
            )
        else:
            self.h_s = HyperSynthesisScale(hyper, latent,
                                           activation_name=activation_name)
        self.entropy_bottleneck = FactorizedPrior(hyper, precision=precision)

    def params(self, z_hat: Tensor) -> tuple[Tensor, Tensor | None]:
        out = self.h_s(z_hat)
        if not self.mean_scale:
            return out, None
        scales, means = out.chunk(2, dim=1)
        return scales, means

    def forward(self, y: Tensor, gc: GaussianConditional, *,
                noise: bool | None = None, ste: bool = True,
                delta_beta=0, q_index: Tensor | None = None) -> dict:
        _no_gain(delta_beta, q_index)
        z = self.h_a(y)
        z_hat, z_lik = self.entropy_bottleneck(z, noise=noise, ste=ste)
        scales, means = self.params(z_hat)
        y_hat, y_lik = gc(y, scales, means, noise=noise, ste=ste)
        return {"y_hat": y_hat, "y_lik": y_lik, "z_lik": z_lik,
                "z": z, "z_hat": z_hat, "scales": scales, "means": means}

    @torch.no_grad()
    def compress(self, y: Tensor, gc: GaussianConditional, *,
                 delta_beta=0, q_index: Tensor | None = None) -> dict:
        _no_gain(delta_beta, q_index)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        # From the *decoded* z_hat, never from z. See ScaleHyperprior.compress.
        z_hat = self.entropy_bottleneck.decompress(
            z_strings, tuple(z.shape[-2:]), device=z.device)
        scales, means = self.params(z_hat)
        return {"y_strings": gc.compress(y, scales, means),
                "z_strings": z_strings, "z_shape": tuple(z.shape[-2:])}

    @torch.no_grad()
    def decompress(self, part: dict, gc: GaussianConditional, device, *,
                   delta_beta=0, q_index: Tensor | None = None) -> dict:
        """`{y_hat, z_hat}`. Both, because the gate checks the hyper latent too.

        Returning only `y_hat` would leave `z_hat` unverifiable from the decoder
        side, and a `z_hat` mismatch is the failure that makes every sigma wrong --
        it shows up as a large rate gap with no other symptom.
        """
        _no_gain(delta_beta, q_index)
        z_hat = self.entropy_bottleneck.decompress(
            part["z_strings"], tuple(part["z_shape"]), device=device)
        scales, means = self.params(z_hat)
        return {"y_hat": gc.decompress(part["y_strings"], scales, means),
                "z_hat": z_hat}


class TwoBranchCodec(nn.Module):
    """The Phase 4 codec: primary luma branch + secondary chroma branch.

    Input and output are **RGB in [0,1]**, same as Phase 3's model, so the training
    loop, the loss and the benchmark need no changes. The YCbCr conversion and the
    chroma resampling happen inside, which also means the 4:2:0 chroma upsampling is
    inside the autograd graph -- correct, since the codec really does have to emit
    RGB, and the upsampling blur is a real part of its distortion.

    Padding happens once, in 4:4:4 YCbCr, before the split. Padding the planes
    separately would let the chroma plane round up to a size that is not the luma
    size divided by the subsampling factor, and the two latents would land on grids
    that differ by one -- which eq. (3) cannot concatenate.
    """

    def __init__(
        self,
        *,
        luma_latent: int = 96,
        chroma_latent: int = 48,
        luma_hyper: int = 96,
        chroma_hyper: int = 48,
        analysis_width=(96, 96, 128, 96),
        synthesis_width=(96, 128, 96, 96),
        internal_format: str = "420",
        mean_scale: bool = True,
        split_hyper: bool = False,
        fused_hyper: bool = False,
        mcm: bool = False,
        mcm_stages: int = 4,
        mcm_order=None,
        scale_layers: int = 2,
        activation_name: str = "relu",
        scale_min: float = 0.11,
        scale_max: float = 54.82,
        scale_levels: int = 32,
        sigma_precision: int = 7,
        precision: int = 16,
        pad_multiple: int = 64,
        gain: bool = False,
        scaler_precision: int = 10,
    ):
        super().__init__()
        self.fmt = get_format(internal_format)
        self.luma_latent, self.chroma_latent = int(luma_latent), int(chroma_latent)
        self.luma_hyper, self.chroma_hyper = int(luma_hyper), int(chroma_hyper)
        # RGB in, RGB out. The YCbCr split is internal, so `summarise`'s MAC probe
        # feeds the model the same 3-channel image the single-branch model takes.
        self.in_channels = 3
        self.pad_multiple = int(pad_multiple)
        self.mean_scale = bool(mean_scale)
        self.split_hyper = bool(split_hyper)
        self.fused_hyper = bool(fused_hyper)
        self.mcm = bool(mcm)
        self.mcm_stages = int(mcm_stages)
        # Phase 8. Variable rate needs the split hyper path: the gain offset is added
        # to `Isigma`, and `HyperpriorBranch` has no `Isigma` -- it predicts sigma
        # directly, with no integer grid for an integer offset to live on.
        self.gain = bool(gain)
        if gain and not split_hyper:
            raise ValueError(
                "gain=True needs split_hyper=True. The quality map is an additive "
                "offset on the integer sigma index, which only the split hyper path "
                "has; on the mean-scale path sigma is a raw network output and there "
                "is no index to offset. Set entropy.split_hyper_decoder: true."
            )
        if split_hyper and not mean_scale:
            raise ValueError(
                "split_hyper implies mean_scale: eq. (1)/(2) code the residual "
                "`y - p̈`, so a prediction is not optional in Phase 5's branch."
            )
        if fused_hyper and not split_hyper:
            raise ValueError(
                "fused_hyper is the --single-hyper-decoder *ablation of* the split "
                "branch; it has no meaning with the Phase 4 hyperprior branch."
            )
        if mcm and not split_hyper:
            raise ValueError(
                "MCM needs Phase 5's split branch: it refines `p̈` and leaves `Iσ` "
                "alone, which only exists as a separate output on the split path."
            )
        if mcm and fused_hyper:
            raise ValueError(
                "MCM and --single-hyper-decoder are alternatives, not a stack. MCM "
                "reads the hyper decoder's pre-shuffle `[4*chs, /32]` tensor as four "
                "per-coset predictions; the fused decoder emits `[2*chs, /16]` with "
                "Iσ folded in, and there is no coset structure in it to read."
            )

        # -- primary (luma): identical in structure to Phase 3, 1 channel in/out
        self.g_a_y = AnalysisTransform(1, luma_latent, analysis_width,
                                       activation_name=activation_name)
        self.g_s_y = SynthesisTransform(luma_latent, 1, synthesis_width,
                                        activation_name=activation_name)
        # -- secondary (chroma): Cb, Cr + the luma link in; eq. (3) on the way out
        self.g_a_uv = SecondaryAnalysis(chroma_latent, analysis_width, self.fmt,
                                        in_channels=3,
                                        activation_name=activation_name)
        self.g_s_uv = SecondarySynthesis(chroma_latent, luma_latent,
                                         synthesis_width, self.fmt,
                                         out_channels=2,
                                         activation_name=activation_name)

        # One sigma grid for the whole codec, and the `SigmaIndex` that reads it must
        # be built from the *same* three constants -- a scale table and an index
        # codebook that disagree produce a rate gap with no other symptom.
        self.sigma_index = SigmaIndex(minimum=scale_min, maximum=scale_max,
                                      levels=scale_levels,
                                      precision=sigma_precision) \
            if split_hyper else None

        def make_branch(latent, hyper, *, context: bool = False):
            if not split_hyper:
                return HyperpriorBranch(latent, hyper, mean_scale=mean_scale,
                                        activation_name=activation_name,
                                        precision=precision)
            if context:
                from jpegai.models.mcm import GROUP_ORDER, MCMBranch
                return MCMBranch(latent, hyper, sigma_index=self.sigma_index,
                                 stages=self.mcm_stages,
                                 order=mcm_order or GROUP_ORDER,
                                 scale_layers=scale_layers,
                                 activation_name=activation_name,
                                 precision=precision, gain=gain,
                                 scaler_precision=scaler_precision)
            return SplitHyperBranch(latent, hyper, sigma_index=self.sigma_index,
                                    fused=fused_hyper, scale_layers=scale_layers,
                                    activation_name=activation_name,
                                    precision=precision, gain=gain,
                                    scaler_precision=scaler_precision)

        # MCM is luma-only: `entropy.mcm_on_secondary: false`, and the paper says why
        # outright -- "considering the trade-off between complexity and coding gain,
        # it was concluded that usage of MCM process for the secondary component is
        # not needed". The chroma branch keeps Phase 5's single-pass prediction, which
        # also means `luma_only` decoding still pays for MCM and nothing else.
        self.branch_y = make_branch(luma_latent, luma_hyper, context=self.mcm)
        self.branch_uv = make_branch(chroma_latent, chroma_hyper)
        self.gaussian_conditional = GaussianConditional(
            build_scale_table(scale_min, scale_max, scale_levels),
            scale_bound=scale_min, precision=precision,
        )

    # -- shared plumbing ---------------------------------------------------
    def _to_planes(self, x_rgb: Tensor):
        """RGB -> padded (luma, chroma, luma-on-chroma-grid) plus the pad spec."""
        ycc, pad = pad_to_multiple(rgb_to_ycbcr_bt709(x_rgb), self.pad_multiple)
        y, uv = split_planes(ycc, self.fmt)
        return y, uv, luma_for_secondary(y, uv.shape[-2:]), pad

    def _to_rgb(self, y_hat: Tensor, uv_hat: Tensor, pad) -> Tensor:
        return unpad(ycbcr_to_rgb_bt709(merge_planes(y_hat, uv_hat)), tuple(pad))

    # -- Phase 8: the rate request -----------------------------------------
    def split_delta_beta(self, delta_beta) -> tuple:
        """One rate request -> the two `D_beta` the two components actually use.

        The paper signals **two** `D_beta`, one per component, "usually set to the same
        value" -- and that "usually" is the flexible colour bit allocation, so the
        plumbing has to keep them separate even though every ordinary call sets them
        alike. Accepts a scalar (both components), a 2-sequence, or a mapping with
        `y`/`uv` keys, and refuses anything that only names one component: a rate
        request that silently left chroma at its anchor would be very hard to see.
        """
        if isinstance(delta_beta, dict):
            unknown = set(delta_beta) - {"y", "uv"}
            if unknown:
                raise KeyError(f"delta_beta keys must be 'y'/'uv', got {sorted(unknown)}")
            if set(delta_beta) != {"y", "uv"}:
                raise KeyError(f"delta_beta needs both 'y' and 'uv', got "
                               f"{sorted(delta_beta)}")
            return delta_beta["y"], delta_beta["uv"]
        if isinstance(delta_beta, (tuple, list)):
            if len(delta_beta) != 2:
                raise ValueError(f"delta_beta as a sequence must be (y, uv), got "
                                 f"{len(delta_beta)} values")
            return tuple(delta_beta)
        return delta_beta, delta_beta

    def gain_parameters(self):
        """Just the gain vectors -- Table II's stage IV trains these and nothing else.

        Returned as a list so an optimiser can take it directly, and empty rather than
        raising when there is no gain unit, so a training script can ask unconditionally
        and discover it has nothing to train from the length.
        """
        return [b.gain.vector for b in (self.branch_y, self.branch_uv)
                if getattr(b, "gain", None) is not None]

    # -- differentiable path -----------------------------------------------
    def forward(self, x: Tensor, *, noise: bool | None = None,
                ste: bool = True, delta_beta=0, q_index: Tensor | None = None) -> dict:
        y, uv, supp, pad = self._to_planes(x)

        y_lat = self.g_a_y(y)
        uv_lat = self.g_a_uv(uv, supp)

        # One spatial map serves both branches, and that is not an approximation: the
        # secondary analysis lands its latent on the *luma* latent's grid (see
        # `SecondaryAnalysis` and `stride_schedule`), so both are /16 of the input,
        # which is exactly the paper's "the spatial dimensions of the primary and
        # secondary components are identical in the latent space".
        d_y, d_uv = self.split_delta_beta(delta_beta)
        out_y = self.branch_y(y_lat, self.gaussian_conditional,
                              noise=noise, ste=ste,
                              delta_beta=d_y, q_index=q_index)
        out_uv = self.branch_uv(uv_lat, self.gaussian_conditional,
                                noise=noise, ste=ste,
                                delta_beta=d_uv, q_index=q_index)

        y_rec = self.g_s_y(out_y["y_hat"])
        uv_rec = self.g_s_uv(out_uv["y_hat"], out_y["y_hat"])      # eq. (3)

        out = {
            "x_hat": self._to_rgb(y_rec, uv_rec, pad),
            # Four streams now, not two. The loss sums -log2 over all of them; a
            # missing key here is a silently under-reported rate, so the keys are
            # named after the streams rather than lumped into one tensor.
            "likelihoods": {"y": out_y["y_lik"], "z": out_y["z_lik"],
                            "y_uv": out_uv["y_lik"], "z_uv": out_uv["z_lik"]},
            "y": y_lat, "y_hat": out_y["y_hat"],
            "y_uv": uv_lat, "y_uv_hat": out_uv["y_hat"],
            "z": out_y["z"], "z_hat": out_y["z_hat"],
            "scales": out_y["scales"], "means": out_y["means"],
            # The same four diagnostics for the secondary branch. Present so the
            # round-trip gate can check *both* branches' out-of-range fractions: a
            # secondary branch quietly escaping to the bypass path would show up
            # only as an unexplained negative rate gap otherwise.
            "z_uv": out_uv["z"], "z_uv_hat": out_uv["z_hat"],
            "scales_uv": out_uv["scales"], "means_uv": out_uv["means"],
            "planes": {"luma": y_rec, "chroma": uv_rec,
                       "luma_src": y, "chroma_src": uv},
        }
        # Phase 5 only, and only if the branch produced one. Present so the training
        # log can watch the index distribution: a scale decoder pinned at 0 or at
        # 3967 is a dead branch, and in the loss it looks exactly like a healthy one.
        if "i_sigma" in out_y:
            out["i_sigma"] = out_y["i_sigma"]
            out["i_sigma_uv"] = out_uv["i_sigma"]
        # Phase 6 only. `r_hat` is what the coder actually writes on the luma branch,
        # and its statistics are the most direct evidence that the context model is
        # doing anything at all: a `gather` that was built but never reached leaves the
        # residual exactly where Phase 5 left it, trains fine, and codes fine.
        if "r_hat" in out_y:
            out["r_hat"] = out_y["r_hat"]
            out["mcm_y_hat"] = out_y["mcm_y_hat"]
        # Phase 8 only. The requested rate point, and the multipliers it produced, so
        # the training log can watch a gain vector that has stopped moving and the
        # round-trip gate can check that `Isigma + o` stayed inside the table.
        if out_y.get("gain") is not None:
            out["delta_beta"] = (d_y, d_uv)
            out["gain"], out["gain_uv"] = out_y["gain"], out_uv["gain"]
            out["gain_offset"] = out_y["gain_offset"]
            out["gain_offset_uv"] = out_uv["gain_offset"]
        return out

    # -- optimiser bookkeeping ---------------------------------------------
    def aux_loss(self) -> Tensor:
        return (self.branch_y.entropy_bottleneck.aux_loss()
                + self.branch_uv.entropy_bottleneck.aux_loss())

    def aux_parameters(self):
        return [self.branch_y.entropy_bottleneck.quantiles,
                self.branch_uv.entropy_bottleneck.quantiles]

    def main_parameters(self):
        aux = {id(p) for p in self.aux_parameters()}
        return [p for p in self.parameters() if id(p) not in aux]

    @torch.no_grad()
    def update(self, force: bool = False) -> bool:
        a = self.branch_y.entropy_bottleneck.update(force=force)
        b = self.branch_uv.entropy_bottleneck.update(force=force)
        c = self.gaussian_conditional.update(force=force)
        return a or b or c

    @property
    def tables_ready(self) -> bool:
        return (self.branch_y.entropy_bottleneck.ready
                and self.branch_uv.entropy_bottleneck.ready
                and self.gaussian_conditional.ready)

    def table_bytes(self) -> int:
        return (self.branch_y.entropy_bottleneck.table_bytes()
                + self.branch_uv.entropy_bottleneck.table_bytes()
                + self.gaussian_conditional.table_bytes())

    # -- reporting protocol ------------------------------------------------
    def summary_title(self) -> str:
        if self.split_hyper:
            prior = "fused-hyper" if self.fused_hyper else "split-hyper"
            if self.mcm:
                prior += f"+mcm{self.mcm_stages}"
        else:
            prior = "mean-scale" if self.mean_scale else "scale-only"
        if self.gain:
            prior += "+gain"
        return (f"{type(self).__name__}  internal {self.fmt.name}  {prior}  "
                f"luma={self.luma_latent}/{self.luma_hyper}  "
                f"chroma={self.chroma_latent}/{self.chroma_hyper}  "
                f"concat={self.chroma_latent + self.luma_latent}")

    def summary_parts(self):
        """`[(label, module, is_decoder_side)]`, eight parts across two branches.

        The hyper networks are listed individually rather than as `branch_y` /
        `branch_uv`, because a branch mixes encoder-only (`h_a`) with decoder-side
        (`h_s`) work and lumping them would inflate the decoder kMAC/pxl figure --
        the one number here that is compared against the paper's 8 / 28 / 215.

        Phase 5's scale decoders get their own rows for a sharper reason than
        tidiness: the acceptance criterion is *"the scale decoder is under 5% of
        decoder MACs"*, and that is only measurable if it is a bucket. Folded into
        `h_s_*` the claim would be unfalsifiable.
        """
        parts = [
            ("g_a_y", self.g_a_y, False), ("g_s_y", self.g_s_y, True),
            ("h_a_y", self.branch_y.h_a, False),
            ("h_s_y", self.branch_y.h_s, True),
            ("g_a_uv", self.g_a_uv, False), ("g_s_uv", self.g_s_uv, True),
            ("h_a_uv", self.branch_uv.h_a, False),
            ("h_s_uv", self.branch_uv.h_s, True),
            # Listed separately rather than as one row: they are two independent
            # factorised priors with their own quantiles, and one row would
            # silently report half the parameters.
            ("eb_y", self.branch_y.entropy_bottleneck, True),
            ("eb_uv", self.branch_uv.entropy_bottleneck, True),
        ]
        for suffix, br in (("_y", self.branch_y), ("_uv", self.branch_uv)):
            if getattr(br, "h_scale", None) is not None:
                parts.append((f"h_scale{suffix}", br.h_scale, True))
            # Phase 6's context model gets its own bucket for the same reason as the
            # scale decoders: the acceptance criterion is that decode cost grows by a
            # *constant* four passes, and that is only checkable if MCM is a number.
            if getattr(br, "mcm", None) is not None:
                parts.append((f"mcm{suffix}", br.mcm, True))
        return parts

    def gate_branches(self):
        """`[(suffix, entropy_bottleneck)]` -- two, so the gate checks both."""
        return [("", self.branch_y.entropy_bottleneck),
                ("_uv", self.branch_uv.entropy_bottleneck)]

    def training_parts(self) -> dict:
        """`{part: [(label, module)]}` -- Table II's "Trained Codec Parts" as modules.

        Four buckets, and the surprising one is `entropy`. The paper's stage III trains
        "decoder, entropy network", and it would be easy to read the hyper *analysis*
        `h_a` as encoder-side work, since it only ever runs on the encoder. The
        reference software settles it: `Net.parameters()` returns four groups and
        `entropy` holds "both hyper-encoders, both hyper-decoders, both hyper-scale-
        decoders, both hyper entropy models, `context_Y`", while `analysis` is
        `encoder_Y`/`encoder_UV` alone (`docs/architecture/13-training.md` section 8.4).
        So `h_a` is part of the entropy network, not of the encoder, and a stage III
        that froze it would be training something the standard's own schedule does not.

        Named after the paper's vocabulary rather than the software's, because
        `config.rate.models[*].stage3` is written in the paper's words. The aliases are
        `analysis -> encoder`, `synthesis -> decoder`, `gain_unit -> gain`; see
        :data:`jpegai.train.stages.ALIASES`.

        Every parameter of this model appears under exactly one part -- asserted, not
        assumed, by :func:`jpegai.train.stages.check_partition`. That invariant is the
        whole value of this method: a freeze implemented as "everything except these
        modules" is only correct if the buckets are a partition, and a parameter that
        belongs to no bucket would be silently frozen in every stage while a parameter
        in two would be trained by a stage that Table II says should not touch it.
        """
        parts: dict[str, list] = {
            "encoder": [("g_a_y", self.g_a_y), ("g_a_uv", self.g_a_uv)],
            "decoder": [("g_s_y", self.g_s_y), ("g_s_uv", self.g_s_uv)],
            "entropy": [],
            "gain": [],
        }
        for suffix, br in (("_y", self.branch_y), ("_uv", self.branch_uv)):
            parts["entropy"] += [(f"h_a{suffix}", br.h_a), (f"h_s{suffix}", br.h_s),
                                 (f"eb{suffix}", br.entropy_bottleneck)]
            for name in ("h_scale", "mcm"):
                mod = getattr(br, name, None)
                if mod is not None:
                    parts["entropy"].append((f"{name}{suffix}", mod))
            # `branch.gain` is the GainUnit; `self.gain` is the bool that built it.
            unit = getattr(br, "gain", None)
            if unit is not None:
                parts["gain"].append((f"gain{suffix}", unit))
        return parts

    def coder_rows(self, out: dict, suffix: str = "") -> Tensor:
        """The CDF row each `y{suffix}` symbol will actually be coded with.

        The gate needs this rather than `build_indexes(scales)` because on the split
        path the two disagree. `build_indexes` counts table entries below a float σ;
        the coder indexes through `SigmaIndex.table_row` on the integer `Iσ`. They
        agree on 3957 of the 3968 indices and differ on 11 -- one float32 ULP at the
        exact grid points, see `jpegai.models.hyper`. Measuring out-of-range and
        `est_q` against the wrong row would put a small permanent bias in `gap_q_pct`,
        the one number whose job is to read zero when the coder is correct.
        """
        if not self.split_hyper:
            return self.gaussian_conditional.build_indexes(out[f"scales{suffix}"])
        si = self.sigma_index
        return si.table_row(si.quantise(out[f"i_sigma{suffix}"]))

    # -- real bitstream ----------------------------------------------------
    @torch.no_grad()
    def precompress(self, x: Tensor) -> dict:
        """Fig. 9's cache: everything about `x` that the rate point cannot change.

        Bit-rate matching encodes the same picture roughly ten times -- two probes for
        eq. (14)'s linear fit, a bisection, and a validation -- and all but the
        arithmetic coding is identical across those. Caching is possible only because
        the gain sits *after* the prediction: `z`, `mu` and `Isigma` come off the
        ungained latent, so `code_cached` re-runs no network at all.

        Requires the gain unit, because without one there is only one rate point and a
        cache would be an invitation to a stale-tensor bug for no gain.
        """
        if not self.tables_ready:
            raise RuntimeError("entropy tables not built; call model.update()")
        if not self.gain:
            raise RuntimeError("precompress() is for the rate search, which needs a "
                               "gain unit; use compress() on a fixed-rate model")
        y, uv, supp, pad = self._to_planes(x)
        return {"luma": self.branch_y.precode(self.g_a_y(y)),
                "chroma": self.branch_uv.precode(self.g_a_uv(uv, supp)),
                "shape": tuple(x.shape[-2:]), "pad": tuple(pad),
                "internal_format": self.fmt.name}

    @torch.no_grad()
    def compress_cached(self, cache: dict, *, delta_beta=0,
                        q_index: Tensor | None = None) -> dict:
        """One packet at one rate point, from a `precompress` cache."""
        d_y, d_uv = self.split_delta_beta(delta_beta)
        packet = {
            "luma": self.branch_y.code_cached(cache["luma"],
                                              self.gaussian_conditional,
                                              delta_beta=d_y, q_index=q_index),
            "chroma": self.branch_uv.code_cached(cache["chroma"],
                                                 self.gaussian_conditional,
                                                 delta_beta=d_uv, q_index=q_index),
            "shape": cache["shape"], "pad": cache["pad"],
            "internal_format": cache["internal_format"],
        }
        return self._add_header(packet, d_y, d_uv, q_index)

    def _add_header(self, packet: dict, d_y, d_uv, q_index) -> dict:
        """The picture header, such as it is.

        Present only when a gain unit exists, so a Phase 5 packet is byte-for-byte what
        it was and `packet_bytes` keeps meaning the same thing. `jpegai/codestream/`
        will turn these into two 12-bit fields and a substream; until it exists they
        travel as python objects, and `header_bytes` is what charges them to the rate.
        """
        if self.gain:
            packet["delta_beta"] = (int(d_y), int(d_uv))
            if q_index is not None:
                packet["q_residual"] = spatial_residual(q_index).cpu()
        return packet

    @torch.no_grad()
    def compress(self, x: Tensor, *, delta_beta=0,
                 q_index: Tensor | None = None) -> dict:
        if not self.tables_ready:
            raise RuntimeError("entropy tables not built; call model.update()")
        y, uv, supp, pad = self._to_planes(x)
        y_lat = self.g_a_y(y)
        uv_lat = self.g_a_uv(uv, supp)
        d_y, d_uv = self.split_delta_beta(delta_beta)
        packet = {
            "luma": self.branch_y.compress(y_lat, self.gaussian_conditional,
                                           delta_beta=d_y, q_index=q_index),
            "chroma": self.branch_uv.compress(uv_lat, self.gaussian_conditional,
                                              delta_beta=d_uv, q_index=q_index),
            "shape": tuple(x.shape[-2:]),
            "pad": tuple(pad),
            "internal_format": self.fmt.name,
        }
        return self._add_header(packet, d_y, d_uv, q_index)

    @torch.no_grad()
    def decompress(self, packet: dict, device=None, *,
                   luma_only: bool = False) -> dict:
        """Decode. `luma_only` skips the entire secondary branch (Phase 4 item 6).

        Not a debug switch: it is the machine-consumption path. A vision model that
        only wants luma should not pay for chroma entropy decoding *or* for the
        secondary synthesis transform, and here it pays for neither -- the chroma
        strings are simply never read. The output is a grey image with correct luma,
        which is what `Cb = Cr = 0.5` means.

        The rate point is read from the packet, never from an argument. A decoder that
        had to be *told* `D_beta` would not be a decoder.
        """
        if not self.tables_ready:
            raise RuntimeError("entropy tables not built; call model.update()")
        device = device or next(self.parameters()).device
        d_y, d_uv = packet.get("delta_beta", (0, 0))
        q_index = None
        if packet.get("q_residual") is not None:
            q_index = spatial_reconstruct(packet["q_residual"]).to(device)

        dec_y = self.branch_y.decompress(packet["luma"], self.gaussian_conditional,
                                         device, delta_beta=d_y, q_index=q_index)
        y_hat = dec_y["y_hat"]
        y_rec = self.g_s_y(y_hat)

        dec_uv = None
        if luma_only:
            uv_rec = torch.full((y_rec.shape[0], 2, *y_rec.shape[-2:]), 0.5,
                                dtype=y_rec.dtype, device=y_rec.device)
        else:
            dec_uv = self.branch_uv.decompress(
                packet["chroma"], self.gaussian_conditional, device,
                delta_beta=d_uv, q_index=q_index)
            uv_rec = self.g_s_uv(dec_uv["y_hat"], y_hat)

        x_hat = self._to_rgb(y_rec, uv_rec, packet["pad"])
        return {"x_hat": x_hat.clamp_(0, 1),
                "y_hat": y_hat, "z_hat": dec_y["z_hat"],
                # None under luma_only rather than absent: a caller comparing
                # against the forward pass should get a clean TypeError, not a
                # KeyError that reads like a plumbing bug.
                "y_uv_hat": None if dec_uv is None else dec_uv["y_hat"],
                "z_uv_hat": None if dec_uv is None else dec_uv["z_hat"],
                "luma": y_rec, "chroma": uv_rec, "luma_only": bool(luma_only)}

    # -- accounting --------------------------------------------------------
    @staticmethod
    def header_bytes(packet: dict) -> int:
        """The side information Phase 8 adds: two `D_beta` fields plus a spatial map.

        Charged, not waved away. Two 12-bit fields are three bytes, which is under
        0.01% of a Kodak packet -- but a spatial quality map is per-picture data, and
        omitting it would report a rate the decoder cannot actually work from. The
        map's cost is `spatial_bits`' order-0 estimate rounded up to whole bytes;
        Phase 12's substream will replace the estimate with a measurement.
        """
        if "delta_beta" not in packet:
            return 0
        total = 3                                     # two 12-bit header fields
        if packet.get("q_residual") is not None:
            total += -(-int(spatial_bits(packet["q_residual"])) // 8)
        return total

    @staticmethod
    def packet_bytes(packet: dict, *, luma_only: bool = False) -> int:
        """Payload bytes. `luma_only` counts what a luma-only decoder must receive."""
        parts = ["luma"] if luma_only else ["luma", "chroma"]
        total = 0
        for p in parts:
            for key in ("y_strings", "z_strings"):
                total += sum(len(s) for s in packet[p][key])
        # A luma-only decoder still needs the header: it needs `D_beta` for luma, and
        # the spatial map is shared between the components rather than duplicated.
        return total + TwoBranchCodec.header_bytes(packet)

    @staticmethod
    def stream_bytes(packet: dict) -> dict[str, int]:
        """Bytes per individual stream, keyed the way `forward()` keys its likelihoods.

        `packet_bytes` returns one total, which is the right thing for a rate but the
        wrong thing for a diagnosis: a table/estimate disagreement in *one* stream
        arrives as a small aggregate number with no indication of where it came from.
        The Phase 5 median-shift bug read as +1.85% overall and was +63% on `z_uv`
        alone; the aggregate was too small to act on and the split was unambiguous.
        Keys match `likelihoods` so the gate can pair each stream with its estimate
        without a translation table. Phase 8 adds a fifth key, `header`, **only when
        the packet carries one** -- the two `D_beta` fields and the spatial map are not
        entropy-coded by this model, so there is no likelihood to pair them with, and
        the rate gate skips any stream it has no estimate for. Adding the key
        unconditionally would break the `sorted(likelihoods) == sorted(stream_bytes)`
        identity for every Phase 5 and Phase 6 model, which is worth more than a
        uniform key set: the sum staying equal to `packet_bytes` is what catches an
        under-reported rate, and it holds either way.
        """
        out = {
            "y": sum(len(s) for s in packet["luma"]["y_strings"]),
            "z": sum(len(s) for s in packet["luma"]["z_strings"]),
            "y_uv": sum(len(s) for s in packet["chroma"]["y_strings"]),
            "z_uv": sum(len(s) for s in packet["chroma"]["z_strings"]),
        }
        header = TwoBranchCodec.header_bytes(packet)
        if header:
            out["header"] = header
        return out

    @staticmethod
    def estimated_bits(out: dict) -> tuple[float, float]:
        """(y_bits, z_bits) summed over *both* branches, for the rate gate."""
        lik = out["likelihoods"]
        def bits(*keys):
            return float(sum(-torch.log2(lik[k].clamp_min(1e-12)).sum().item()
                             for k in keys if k in lik))
        return bits("y", "y_uv"), bits("z", "z_uv")


def build_two_branch(config, *, mean_scale: bool = True,
                     split_hyper: bool = False, fused_hyper: bool = False,
                     mcm: bool = False, mcm_stages: int | None = None,
                     gain: bool = False):
    """Instantiate from a loaded `jpegai.config` object.

    Chroma widths come from `secondary_latent`/`hyper_secondary_latent`, which are
    96/48 in Tier A and 160/96 in full -- see docs/06 §2. Do not hardcode either.

    Coder precision is left at the class default (16), matching
    `hyperprior.build_model`. The config's `sigma_precision: 7` is the precision of
    the *sigma representation* and `scaler_precision: 10` belongs to the gain
    vectors; neither is the rANS coder's, and passing one of them here would quietly
    shrink the CDF tables. `sigma_precision` *is* passed, but to the `SigmaIndex`,
    which is the one place it belongs -- and `scaler_precision` is now passed too, to
    the `GainUnit`, which is the other.

    `mcm_stages=None` takes `entropy.mcm_stages`, which is 4 and marked STRUCTURAL.
    The coset order comes from `entropy.mcm_group_order` rather than from a default
    in the model, because that list is the thing docs/06 §5 derived from the reference
    software and the config validator already checks it covers the 2x2 tile once.

    `gain` is a *caller's* flag rather than a config field, exactly like `split_hyper`
    and `mcm`: which phase's model to build is a decision of the script doing the
    building, and putting it in the yaml would make two configs differ by something
    that is not a codec constant.
    """
    ch, ent = config.channels, config.entropy
    return TwoBranchCodec(
        luma_latent=ch.primary_latent,
        chroma_latent=ch.secondary_latent,
        luma_hyper=ch.hyper_latent,
        chroma_hyper=ch.hyper_secondary_latent,
        analysis_width=ch.analysis_width,
        synthesis_width=ch.synthesis_width,
        internal_format=config.colour.internal_format,
        mean_scale=mean_scale,
        split_hyper=split_hyper,
        fused_hyper=fused_hyper,
        mcm=mcm,
        mcm_stages=ent.mcm_stages if mcm_stages is None else mcm_stages,
        mcm_order=[tuple(g) for g in ent.mcm_group_order],
        scale_min=ent.sigma_quant_min,
        scale_max=ent.sigma_quant_max,
        scale_levels=ent.sigma_quant_level,
        sigma_precision=ent.sigma_precision,
        pad_multiple=config.geometry.total_downsample,
        gain=gain,
        scaler_precision=ent.scaler_precision,
    )


if __name__ == "__main__":
    print("stride schedules from the chroma grid to the /16 latent grid")
    for name in FORMATS:
        sched = stride_schedule(name)
        v = math.prod(s[0] for s in sched)
        h = math.prod(s[1] for s in sched)
        print(f"  {name}: {len(sched)} stages {sched}  -> /{v} vertical /{h} horizontal")

    m = TwoBranchCodec()
    n = sum(p.numel() for p in m.parameters())
    x = torch.rand(1, 3, 128, 128)
    out = m(x)
    print(f"\nTwoBranchCodec  {n:,} params  ({n * 4 / 2**20:.1f} MiB fp32)")
    print(f"  x        {tuple(x.shape)}")
    print(f"  y_Y      {tuple(out['y'].shape)}")
    print(f"  y_UV     {tuple(out['y_uv'].shape)}   <- same grid, eq. (3) is valid")
    print(f"  x_hat    {tuple(out['x_hat'].shape)}")
    print(f"  streams  {sorted(out['likelihoods'])}")
