"""Rate-distortion loss.

    L = beta * D + R

`R` is bits per pixel from the entropy models. `D` is a weighted MSE on the
0-255 scale, luma-weighted 6:1:1 per `config.train.distortion_weights`.

Which of the two terms `beta` multiplies is not stated in the overview paper, and
getting it backwards silently inverts the whole RD ladder, so here is the
derivation. Three facts from the config, all confirmed from the WG1 reference
software (`quantization/gain_unit/params.py`, `CCS_SGMM/params.py`):

* the RD ladder is 18 betas from 0.0002 to 3.0;
* the four *trained* base models sit at beta = 0.002, 0.012, 0.075, 0.5;
* `betaDisplacementLog` is sampled uniformly in [-40, 40] with
  `beta_displacement_precision = 5`.

Take the displacement first. A log-domain displacement with precision 5 means the
effective beta is `base * 2**(disp / 2**5)`, so the range [-40, 40] spans
`2**(+-1.25)` = a factor of 0.42 to 2.38, i.e. **5.66x end to end**. Now look at
the gaps between adjacent base models: 0.012/0.002 = 6.0, 0.075/0.012 = 6.25,
0.5/0.075 = 6.67. The gain unit's displacement range covers exactly the gap
between neighbouring base models -- which is what the mechanism is *for*, and
strong evidence the numbers are being read correctly rather than coincidentally.

Now the direction. `beta = 0.002` times a 0-255 MSE is `0.002 * 255**2 = 130`
times a [0,1] MSE. compressai's published MSE lambdas for this exact
architecture, on the same scale, are 117 / 228 / 436 / 845 / 1625 / 3140 from
lowest to highest quality. So JPEG AI's lowest base beta lands within 11% of
compressai's lowest lambda, and its highest (0.5 -> 32500) sits above
compressai's highest. That only works if **beta multiplies the distortion**, and
larger beta means higher quality and higher rate. Reasoned inference, not a
normative citation -- but it is consistent with three independent constants.

The MS-SSIM term (`config.train.loss.ms_ssim`) is added because the paper's
primary metric set is perceptual: six of its seven metrics are structural or
perceptual and none is PSNR. Optimising pure MSE gives a model that wins on a
metric nobody is grading and loses on the AVG that decides the BD-rate.

One loss serves both the single-branch RGB model and Phase 4's two-branch YCbCr
model. It never touches the model's internal planes -- see
:meth:`RateDistortionLoss.distortion` for why the padded, subsampled internal
representation is the wrong place to measure -- so adding branches only adds
entropy-stream keys, which :meth:`branch_split` groups for the log.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from jpegai.models.colour import rgb_to_ycbcr_bt709

#: MSE is reported and weighted on the 0-255 scale, so that beta values are
#: directly comparable to the reference software's and to compressai's lambdas.
MSE_SCALE = 255.0 ** 2


#: Re-exported, not reimplemented. There used to be a local copy here with the
#: constants written out as literals; Phase 4 gave `jpegai.models.colour` a
#: canonical version with the coefficients asserted to sum to 1, so this module
#: now uses that one. `jpegai.eval.metrics` still keeps its own copy, and that
#: one has to stay: eval must run against JPEG and AVIF with no torch model
#: present, so it cannot import the model package, and the model package cannot
#: import eval without a cycle. Train has no such constraint.
PLANE_NAMES = ("y", "u", "v")


class RateDistortionLoss(nn.Module):
    """`L = beta * D + R`, with D luma-weighted in YCbCr and R in bits/pixel.

    Args:
        beta: distortion weight. Larger = higher quality and higher rate.
        weights: per-plane distortion weights, `{y, u, v}`. The paper says
            "prioritise the quality of the luma component during training"; the
            config's 6:1:1 is our reading of that.
        ms_ssim_weight: weight on `1 - MS-SSIM`, added to the MSE distortion.
        colour_space: "ycbcr" applies the weights in YCbCr (what the paper does);
            "rgb" is unweighted RGB MSE, for apples-to-apples comparison with
            published compressai numbers.
    """

    def __init__(self, beta: float = 0.002, *, weights=None,
                 ms_ssim_weight: float = 0.0, colour_space: str = "ycbcr"):
        super().__init__()
        self.beta = float(beta)
        w = dict(weights or {"y": 6.0, "u": 1.0, "v": 1.0})
        # Normalise so the *total* distortion scale is independent of the weights.
        # Without this, changing 6:1:1 to 8:1:1 silently rescales D by 25% and
        # every beta in the ladder means something different -- so an ablation on
        # the weights would also be an unintended ablation on the rate point.
        total = w["y"] + w["u"] + w["v"]
        self.register_buffer(
            "plane_weights",
            torch.tensor([w["y"], w["u"], w["v"]], dtype=torch.float32)
            * (3.0 / total),
        )
        self.ms_ssim_weight = float(ms_ssim_weight)
        if colour_space not in ("ycbcr", "rgb"):
            raise ValueError(f"colour_space must be ycbcr or rgb, got {colour_space!r}")
        self.colour_space = colour_space
        self._ms_ssim_fn = None

    # -- pieces ------------------------------------------------------------
    def distortion(self, x: Tensor, x_hat: Tensor):
        """(weighted MSE on the 0-255 scale, plain RGB MSE, per-plane MSE dict).

        Measured on the **unpadded RGB output**, not on the model's internal
        planes, even for the two-branch codec which hands both to us. Two reasons,
        and they point the same way:

        * The internal planes are still reflect-padded. Those border pixels are
          cropped before anyone sees them, so distortion spent there buys nothing
          -- exactly what `pad_to_multiple`'s docstring warns about.
        * At 4:2:0 the internal chroma plane is half resolution, so measuring
          there leaves the chroma *upsampler* outside the autograd graph and the
          secondary synthesis never learns to compensate for it. The metrics that
          decide the BD-rate all look at full-resolution output, so the loss
          should too.

        The RGB -> YCbCr -> RGB round trip this implies is exact to ~1e-7 (see
        `tests/test_colour.py`), so nothing is lost by going through RGB, and it
        keeps one loss working for both the single-branch and two-branch models.
        """
        plain = F.mse_loss(x_hat, x) * MSE_SCALE
        if self.colour_space == "rgb" or x.shape[1] != 3:
            return plain, plain, {}
        a = rgb_to_ycbcr_bt709(x)
        b = rgb_to_ycbcr_bt709(x_hat)
        # Mean over batch and space, then weight per plane. Weighting before the
        # spatial mean would make the result depend on the plane resolutions,
        # which matters from Phase 4 on when chroma is subsampled.
        per_plane = ((a - b) ** 2).mean(dim=(0, 2, 3))
        w = self.plane_weights.to(per_plane.device)
        parts = {k: v * MSE_SCALE for k, v in zip(PLANE_NAMES, per_plane)}
        return (per_plane * w).sum() / w.sum() * MSE_SCALE, plain, parts

    def _ms_ssim(self, x: Tensor, x_hat: Tensor) -> Tensor:
        if self._ms_ssim_fn is None:
            from pytorch_msssim import ms_ssim as f
            self._ms_ssim_fn = f
        # MS-SSIM's 5-level pyramid needs > 160 px; below that fall back to
        # nothing rather than crashing, so a 64px smoke test still runs.
        if min(x.shape[-2:]) < 161:
            return x.new_zeros(())
        val = self._ms_ssim_fn(x_hat.clamp(0, 1), x, data_range=1.0, size_average=True)
        return 1.0 - val

    @staticmethod
    def bpp(likelihoods: dict, num_pixels: int) -> tuple[Tensor, dict]:
        """Total bits/pixel plus a per-stream breakdown, differentiable."""
        parts, total = {}, None
        for name, lik in likelihoods.items():
            bits = -torch.log2(lik.clamp_min(1e-9)).sum()
            parts[name] = bits / num_pixels
            total = bits if total is None else total + bits
        return total / num_pixels, parts

    @staticmethod
    def branch_split(bpp_parts: dict) -> dict:
        """Group the per-stream rates into primary and secondary branch totals.

        Phase 4 turns two streams into four, and `bpp_y`/`bpp_z`/`bpp_y_uv`/
        `bpp_z_uv` individually are hard to read in a training log. What you
        actually want to watch is the *chroma share*: if the secondary branch is
        spending 40% of the rate on two subsampled planes weighted 1:1 against
        luma's 6, the 6:1:1 weighting is not doing its job. Returns `{}` for a
        single-branch model, so callers need no special case.
        """
        uv = [v for k, v in bpp_parts.items() if k.endswith("_uv")]
        if not uv:
            return {}
        luma = [v for k, v in bpp_parts.items() if not k.endswith("_uv")]
        b_luma, b_uv = sum(luma), sum(uv)
        return {"bpp_luma": b_luma, "bpp_chroma": b_uv,
                "chroma_share": b_uv / (b_luma + b_uv).clamp_min(1e-9)}

    # -- the loss ----------------------------------------------------------
    def forward(self, out: dict, x: Tensor) -> dict:
        n, _, h, w = x.shape
        num_pixels = n * h * w
        x_hat = out["x_hat"]

        bpp, bpp_parts = self.bpp(out["likelihoods"], num_pixels)
        mse, mse_rgb, mse_planes = self.distortion(x, x_hat)

        d = mse
        ms = None
        if self.ms_ssim_weight > 0:
            ms = self._ms_ssim(x, x_hat)
            # MS-SSIM's loss is in [0,1] while MSE_255 is in the tens; scaling by
            # MSE_SCALE puts the two on one scale so `loss.ms_ssim: 0.1` in the
            # config means "a tenth as important as MSE" rather than "a tenth of
            # a percent", which is what the raw ratio would give.
            d = d + self.ms_ssim_weight * ms * MSE_SCALE

        loss = self.beta * d + bpp

        def psnr(m):
            return 10 * torch.log10(MSE_SCALE / m.detach().clamp_min(1e-9))

        result = {
            "loss": loss,
            "bpp": bpp,
            "mse": mse.detach(),
            "mse_rgb": mse_rgb.detach(),
            "psnr": psnr(mse_rgb),
            # Per-plane PSNR, so chroma quality is visible while training rather
            # than discovered at evaluation time -- Phase 4's own pitfall list
            # says "don't forget chroma PSNR", and it is the number that
            # justifies Phase 10's EFE filters.
            **{f"psnr_{k}": psnr(v) for k, v in mse_planes.items()},
            **{f"bpp_{k}": v.detach() for k, v in bpp_parts.items()},
            **{k: v.detach() for k, v in self.branch_split(bpp_parts).items()},
        }
        if ms is not None:
            result["ms_ssim"] = (1.0 - ms).detach()
        return result


def loss_from_config(config, beta: float | None = None, **kw) -> RateDistortionLoss:
    t = config.train
    return RateDistortionLoss(
        beta=beta if beta is not None else config.rate.base_model_beta,
        weights=dict(t.distortion_weights),
        ms_ssim_weight=t.loss.get("ms_ssim", 0.0),
        **kw,
    )


if __name__ == "__main__":
    from jpegai.config import load_config

    cfg = load_config("tierA")
    crit = loss_from_config(cfg)
    print(f"beta {crit.beta}  plane weights {crit.plane_weights.tolist()}  "
          f"ms_ssim {crit.ms_ssim_weight}")

    # Sanity: the loss must be minimised by x_hat == x, and D must be monotone in
    # the amount of corruption. If either fails, nothing downstream is meaningful.
    x = torch.rand(2, 3, 256, 256)
    print(f"\n{'noise sigma':>12} {'mse_255':>9} {'psnr':>7} {'ms_ssim':>8} {'loss':>10}")
    for sigma in (0.0, 0.01, 0.03, 0.1):
        out = {"x_hat": (x + sigma * torch.randn_like(x)).clamp(0, 1),
               "likelihoods": {"y": torch.full((2, 96, 16, 16), 0.5),
                               "z": torch.full((2, 96, 4, 4), 0.5)}}
        r = crit(out, x)
        print(f"{sigma:12.3f} {r['mse']:9.3f} {r['psnr']:7.2f} "
              f"{r.get('ms_ssim', float('nan')):8.4f} {r['loss']:10.4f}")

    print("\nbeta ladder -> effective compressai-equivalent MSE lambda")
    for b in cfg.rate.beta_list:
        print(f"  beta {b:<8} == lambda*255^2 {b * MSE_SCALE:>10.1f}")
