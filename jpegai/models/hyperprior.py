"""Scale hyperprior -- Phase 3's deliverable and the gate for the whole project.

This is the architecture of Balle, Minnen, Singh, Hwang, Johnston, ICLR 2018
("Variational image compression with a scale hyperprior"), which is the direct
ancestor of JPEG AI's entropy model. It is *not* JPEG AI: no colour branches, no
MCM, no gain units, one synthesis head, scale-only (no mean). Every one of those
arrives in a later phase.

Building it first is deliberate. The plan's Phase 3 acceptance test is:

    1. Loss decreases and the RD curve beats JPEG comfortably.
    2. len(encode(x)) * 8 / npixels is within 1-2% of the estimated rate -log2 p.
    3. decode(encode(x)) reproduces the *latent* yhat exactly.

Criterion 2 is the one that matters. Training only ever sees the estimated rate;
the bytes are produced by a completely separate code path (quantised CDF tables +
rANS). If those two disagree, every RD curve this project ever plots is fiction,
and no amount of later architecture fixes it. So the smallest model that exercises
both paths end to end gets built and gated before anything else.

Why scale-only when JPEG AI predicts both mean and scale? Because a scale-only
hyperprior has exactly one thing the decoder must reproduce bit-exactly (sigma via
its quantised index), whereas mean-and-scale has two, and the mean is the one that
breaks: `round(y - mu)` at the encoder and `decode() + mu` at the decoder agree
only if `mu` is bit-identical on both sides. Phase 5 introduces that problem
deliberately, on top of a foundation already known to be sound.

Shapes, for a 256x256x3 input at Tier A (primary_latent 96, hyper_latent 96):

    x       [B,  3, 256, 256]
    y = g_a(x)      [B, 96, 16, 16]      four stride-2 stages -> /16
    z = h_a(|y|)    [B, 96,  4,  4]      two more             -> /64
    sigma = h_s(zhat)   [B, 96, 16, 16]  one sigma per latent element
    xhat = g_s(yhat)    [B,  3, 256, 256]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from jpegai.models.entropy import (
    FactorizedPrior,
    GaussianConditional,
    build_scale_table,
)
from jpegai.models.layers import activation, conv, deconv, pad_to_multiple, unpad


class AnalysisTransform(nn.Module):
    """g_a: image -> latent. Four stride-2 stages, so the latent sits at /16.

    `widths[i]` is the output width of stage i, and `widths[-1]` must equal the
    latent width -- the last stage *is* the projection to the latent, there is no
    separate 1x1 afterwards. `jpegai.config` asserts that invariant so a config
    typo fails at load time rather than as a shape error 40 layers deep.
    """

    def __init__(self, in_channels: int, latent: int, widths, *,
                 activation_name: str = "relu", kernel: int = 5):
        super().__init__()
        widths = list(widths)
        if len(widths) != 4:
            raise ValueError(f"analysis_width must have 4 entries, got {widths}")
        if widths[-1] != latent:
            raise ValueError(
                f"analysis_width[-1] ({widths[-1]}) must equal the latent width "
                f"({latent}): the final analysis stage is the projection to the latent."
            )

        layers: list[nn.Module] = []
        prev = in_channels
        for i, w in enumerate(widths):
            layers.append(conv(prev, w, kernel, stride=2))
            # No nonlinearity after the last stage. The latent is what the
            # entropy model sees, and it must be free to be negative and
            # unbounded -- a ReLU here would clamp half of it to zero and the
            # Gaussian prior would be modelling a distribution that cannot occur.
            if i < len(widths) - 1:
                layers.append(activation(activation_name, w))
            prev = w
        self.body = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class SynthesisTransform(nn.Module):
    """g_s: latent -> image. Four stride-2 upsampling stages, mirroring g_a.

    `widths` here are the three *hidden* widths; the fourth upsampling stage
    projects straight to `out_channels`.

    That is Balle's form, and it is chosen over JPEG AI's (a fourth stage to
    `final_channels`, then a 3x3 to the output) purely on cost at this phase: a
    deconv 96->96 at full resolution is ~57 kMAC/pixel, against ~1.8 kMAC/pixel
    for 96->3. Thirty times the arithmetic in the one layer that runs at full
    resolution is not affordable on a laptop, and it buys nothing that Phase 3 is
    testing. `channels.synthesis_width[-1]` is therefore unused here on purpose;
    it is Phase 7's per-decoder `final_channels`, where the three heads get
    designed to a MAC budget and the cost is the point.
    """

    def __init__(self, latent: int, out_channels: int, widths, *,
                 activation_name: str = "relu", kernel: int = 5):
        super().__init__()
        widths = list(widths)[:3]
        if len(widths) != 3:
            raise ValueError(f"need at least 3 synthesis widths, got {widths}")

        layers: list[nn.Module] = []
        prev = latent
        for w in widths:
            layers.append(deconv(prev, w, kernel, stride=2))
            layers.append(activation(activation_name, w, inverse=True))
            prev = w
        layers.append(deconv(prev, out_channels, kernel, stride=2))
        self.body = nn.Sequential(*layers)

    def forward(self, y: Tensor) -> Tensor:
        return self.body(y)


class HyperAnalysis(nn.Module):
    """h_a: latent -> hyper latent. One stride-1 stage then two stride-2, so /64.

    Consumes ``abs(y)`` rather than ``y``, matching both Balle's scale-only model
    and JPEG AI (`entropy.abs_in_hyperprior: true`, confirmed in the reference
    software's hyper encoder). The reason is that this network's only job is to
    predict a *scale*, and scale is a property of magnitude; handing it the sign
    means spending capacity learning that sigma(y) = sigma(-y).
    """

    def __init__(self, latent: int, hyper: int, *, activation_name: str = "relu"):
        super().__init__()
        self.body = nn.Sequential(
            conv(latent, hyper, 3, stride=1),
            activation(activation_name, hyper),
            conv(hyper, hyper, 5, stride=2),
            activation(activation_name, hyper),
            conv(hyper, hyper, 5, stride=2),
        )

    def forward(self, y: Tensor) -> Tensor:
        return self.body(torch.abs(y))


class HyperSynthesisScale(nn.Module):
    """h_s: hyper latent -> per-element sigma. Two stride-2 upsamples, then 3x3.

    Ends in ReLU so sigma >= 0. That is Balle's original and it looks careless --
    a dead unit produces sigma exactly 0, which is an infinitely confident
    prediction. It is safe only because `GaussianConditional` clamps through
    `LowerBound(0.11)`, which is also JPEG AI's `sigma_quant_min`. The clamp is
    load-bearing, not defensive: without it the rate term is `-log2(0)`.
    """

    def __init__(self, hyper: int, latent: int, *, activation_name: str = "relu"):
        super().__init__()
        self.body = nn.Sequential(
            deconv(hyper, hyper, 5, stride=2),
            activation(activation_name, hyper),
            deconv(hyper, hyper, 5, stride=2),
            activation(activation_name, hyper),
            conv(hyper, latent, 3, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.body(z)


class ScaleHyperprior(nn.Module):
    """The complete Phase 3 codec: g_a, g_s, h_a, h_s, two entropy models.

    Two entropy models because there are two things to transmit and only one of
    them has a learned conditional prior:

    * `y` (the image latent) is coded with a Gaussian whose sigma is *predicted*
      from side information -- that is the whole idea of a hyperprior.
    * `z` (the side information itself) has nothing to condition on, so it gets a
      factorised, non-parametric, learned marginal. It is small (1/16 the elements
      of `y`) which is what makes paying for it worthwhile.

    Subclassing contract for Phase 5: override `_build_hyper_synthesis` to emit
    `2 * latent` channels and `_split_params` to return `(scales, means)`. Nothing
    else in this class assumes scale-only.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent: int = 96,
        hyper: int = 96,
        analysis_width=(96, 96, 128, 96),
        synthesis_width=(96, 128, 96, 96),
        *,
        activation_name: str = "relu",
        scale_min: float = 0.11,
        scale_max: float = 54.82,
        scale_levels: int = 32,
        precision: int = 16,
        pad_multiple: int = 64,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent = int(latent)
        self.hyper = int(hyper)
        self.pad_multiple = int(pad_multiple)

        self.g_a = AnalysisTransform(in_channels, latent, analysis_width,
                                     activation_name=activation_name)
        self.g_s = SynthesisTransform(latent, in_channels, synthesis_width,
                                      activation_name=activation_name)
        self.h_a = HyperAnalysis(latent, hyper, activation_name=activation_name)
        self.h_s = self._build_hyper_synthesis(hyper, latent, activation_name)

        self.entropy_bottleneck = FactorizedPrior(hyper, precision=precision)
        self.gaussian_conditional = GaussianConditional(
            build_scale_table(scale_min, scale_max, scale_levels),
            scale_bound=scale_min,
            precision=precision,
        )

    # -- subclass hooks ----------------------------------------------------
    def _build_hyper_synthesis(self, hyper: int, latent: int, act: str) -> nn.Module:
        return HyperSynthesisScale(hyper, latent, activation_name=act)

    def _split_params(self, params: Tensor) -> tuple[Tensor, Tensor | None]:
        return params, None

    # -- differentiable path ----------------------------------------------
    def forward(self, x: Tensor, *, noise: bool | None = None,
                ste: bool = True) -> dict:
        """Returns everything the loss needs, plus the pieces the gate compares.

        `noise` / `ste` implement `config.train.quantisation`: uniform noise for
        the rate branch (so the entropy model sees a continuous relaxation whose
        differential entropy is the discrete entropy it will actually pay) and
        straight-through rounding for the distortion branch (so the synthesis
        transform sees the values it will actually receive). Using noise for both
        trains a decoder that has never seen a rounded latent; using STE for both
        trains an entropy model on a gradient that is a lie about the density.
        """
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihood = self.entropy_bottleneck(z, noise=noise, ste=ste)
        scales, means = self._split_params(self.h_s(z_hat))
        y_hat, y_likelihood = self.gaussian_conditional(
            y, scales, means, noise=noise, ste=ste)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihood, "z": z_likelihood},
            "y": y, "y_hat": y_hat, "z": z, "z_hat": z_hat,
            "scales": scales, "means": means,
        }

    def aux_loss(self) -> Tensor:
        """Loss for the *separate* quantile optimiser. See FactorizedPrior."""
        return self.entropy_bottleneck.aux_loss()

    def aux_parameters(self):
        return [self.entropy_bottleneck.quantiles]

    def main_parameters(self):
        aux = {id(p) for p in self.aux_parameters()}
        return [p for p in self.parameters() if id(p) not in aux]

    @torch.no_grad()
    def update(self, force: bool = False) -> bool:
        """Build both CDF tables. Must run after training, before any compress."""
        a = self.entropy_bottleneck.update(force=force)
        b = self.gaussian_conditional.update(force=force)
        return a or b

    @property
    def tables_ready(self) -> bool:
        return self.entropy_bottleneck.ready and self.gaussian_conditional.ready

    def table_bytes(self) -> int:
        return (self.entropy_bottleneck.table_bytes()
                + self.gaussian_conditional.table_bytes())

    # -- reporting protocol ------------------------------------------------
    def summary_title(self) -> str:
        return f"{type(self).__name__}  latent={self.latent}  hyper={self.hyper}"

    def summary_parts(self):
        """`[(label, module, is_decoder_side)]` for :func:`summarise`.

        `is_decoder_side` is what makes the kMAC/pxl figure comparable to the
        paper's 8 / 28 / 215: those are decoder costs, and `g_a`/`h_a` never run
        on the device that has to meet the budget.
        """
        return [("g_a", self.g_a, False), ("g_s", self.g_s, True),
                ("h_a", self.h_a, False), ("h_s", self.h_s, True),
                ("entropy_bottleneck", self.entropy_bottleneck, True)]

    def training_parts(self) -> dict:
        """`{part: [(label, module)]}` -- see `TwoBranchCodec.training_parts`.

        Present on the single-branch model too, even though Table II's schedule is a
        two-branch variable-rate thing and this model has no gain unit. Two reasons:
        the freeze machinery in `jpegai.train.stages` then works for a Phase 3 model
        as well, so stages I and II are runnable on it; and the partition invariant
        becomes testable here, which is where it is cheapest to see broken.

        `gain` is present and **empty** rather than absent. A stage that asks to train
        the gain unit of a fixed-rate model must fail loudly on "nothing to train", not
        on `KeyError`.
        """
        parts = {
            "encoder": [("g_a", self.g_a)],
            "decoder": [("g_s", self.g_s)],
            "entropy": [("h_a", self.h_a), ("h_s", self.h_s),
                        ("entropy_bottleneck", self.entropy_bottleneck)],
            "gain": [],
        }
        if getattr(self, "h_scale", None) is not None:
            parts["entropy"].append(("h_scale", self.h_scale))
        return parts

    def gate_branches(self):
        """`[(suffix, entropy_bottleneck)]` for the Phase 3 round-trip gate.

        One branch here; Phase 4's two-branch codec returns two, and the gate walks
        whatever it is given rather than reaching for `model.entropy_bottleneck`.
        The suffix indexes into the forward pass's output dict: `""` means the keys
        are `y`/`z`/`scales`/`means`, `"_uv"` means `y_uv`/`z_uv`/... .
        """
        return [("", self.entropy_bottleneck)]

    # -- real bitstream ----------------------------------------------------
    @torch.no_grad()
    def compress(self, x: Tensor) -> dict:
        """Produce actual bytes. Returns strings + the header the decoder needs.

        The order of operations is the part that has to be exactly right, and it
        is the mirror of `decompress`:

            z -> zhat -> encode(zhat)      # side info first, unconditionally
            sigma = h_s(zhat)              # from the QUANTISED zhat, never from z
            encode(y | sigma)

        Deriving sigma from `z` rather than `z_hat` is the classic asymmetry bug:
        it trains fine, it encodes fine, and the decoder -- which only ever has
        `z_hat` -- computes different sigmas, picks different CDF rows, and
        desynchronises. Nothing downstream can detect it except a round-trip test.
        """
        if not self.tables_ready:
            raise RuntimeError(
                "entropy tables not built; call model.update() after loading "
                "weights and before compress()"
            )
        x_pad, pad = pad_to_multiple(x, self.pad_multiple)
        y = self.g_a(x_pad)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(
            z_strings, tuple(z.shape[-2:]), device=z.device)

        scales, means = self._split_params(self.h_s(z_hat))
        y_strings = self.gaussian_conditional.compress(y, scales, means)

        return {
            "strings": [y_strings, z_strings],
            "z_shape": tuple(z.shape[-2:]),
            "shape": tuple(x.shape[-2:]),
            "pad": tuple(pad),
        }

    @torch.no_grad()
    def decompress(self, packet: dict, device=None) -> dict:
        """Decode a packet from :meth:`compress` back to pixels.

        Note that `z_hat` is obtained by *decoding*, not by re-running `h_a`:
        this method never sees `x`. That is the property criterion 3 of the gate
        checks, and it is why the gate compares latents rather than pixels -- a
        pixel comparison would also pass if the latents were slightly wrong and
        the synthesis transform happened to smooth over it.
        """
        if not self.tables_ready:
            raise RuntimeError("entropy tables not built; call model.update()")
        device = device or next(self.parameters()).device
        y_strings, z_strings = packet["strings"]

        z_hat = self.entropy_bottleneck.decompress(
            z_strings, tuple(packet["z_shape"]), device=device)
        scales, means = self._split_params(self.h_s(z_hat))
        y_hat = self.gaussian_conditional.decompress(y_strings, scales, means)
        x_hat = self.g_s(y_hat)
        x_hat = unpad(x_hat, tuple(packet["pad"]))
        return {"x_hat": x_hat.clamp_(0, 1), "y_hat": y_hat, "z_hat": z_hat}

    # -- accounting --------------------------------------------------------
    @staticmethod
    def packet_bytes(packet: dict) -> int:
        """Payload size only -- the entropy-coded strings, no container.

        Deliberately excludes the header (`shape`, `pad`, `z_shape`), which in a
        real codestream is a handful of bytes in the picture header and here is a
        python dict. Counting a JSON dict as codec overhead would make bpp depend
        on how the demo serialises, which is not a property of the codec. Phase 9
        replaces this with the real marker-based container and then the header
        genuinely does count.
        """
        return sum(len(s) for group in packet["strings"] for s in group)

    @staticmethod
    def stream_bytes(packet: dict) -> dict[str, int]:
        """Bytes per individual stream, keyed the way `forward()` keys its likelihoods.

        Same contract as `TwoBranchCodec.stream_bytes`; see there for why the gate
        wants the split rather than only `packet_bytes`' total. `packet["strings"]` is
        ordered `[y, z]` by `compress`, and that order is what `decompress` reads, so
        it is not a convention this method is free to reinterpret.
        """
        y_strings, z_strings = packet["strings"]
        return {"y": sum(len(s) for s in y_strings),
                "z": sum(len(s) for s in z_strings)}

    @staticmethod
    def estimated_bits(out: dict) -> tuple[float, float]:
        """(y_bits, z_bits) from the differentiable model, summed over the batch."""
        ly = out["likelihoods"]["y"]
        lz = out["likelihoods"]["z"]
        y_bits = float(-torch.log2(ly.clamp_min(1e-12)).sum().item())
        z_bits = float(-torch.log2(lz.clamp_min(1e-12)).sum().item())
        return y_bits, z_bits


class MeanScaleHyperprior(ScaleHyperprior):
    """Minnen, Balle, Toderici, NeurIPS 2018 -- h_s predicts mean *and* scale.

    Present here rather than in Phase 5 because it is nine lines given the
    subclass hooks, and having it now means the Phase 3 gate can be run against
    both: if the round-trip is bit-exact for scale-only but not for mean-scale,
    the fault is in mean handling and not in the CDF tables. That is a much
    faster diagnosis than discovering it two phases later.

    The mean makes the prior a *shifted* Gaussian, so the symbol coded is
    `round(y - mu)`. `mu` is not transmitted -- both sides compute it from `z_hat`
    -- which is exactly why the encoder must use the decoded `z_hat`.
    """

    def _build_hyper_synthesis(self, hyper: int, latent: int, act: str) -> nn.Module:
        # No final ReLU: the mean is signed. The scale half is made positive by
        # GaussianConditional's LowerBound instead, which is the only mechanism
        # that has to hold for the rate term to be finite.
        mid = (hyper * 3) // 2
        return nn.Sequential(
            deconv(hyper, mid, 5, stride=2),
            activation(act, mid, inverse=True),
            deconv(mid, mid * 3 // 2, 5, stride=2),
            activation(act, mid * 3 // 2, inverse=True),
            conv(mid * 3 // 2, 2 * latent, 3, stride=1),
        )

    def _split_params(self, params: Tensor) -> tuple[Tensor, Tensor | None]:
        scales, means = params.chunk(2, dim=1)
        return scales, means


# ---------------------------------------------------------------------------
# Construction from config
# ---------------------------------------------------------------------------
def build_model(config, *, kind: str = "scale", in_channels: int = 3):
    """Instantiate from a loaded `jpegai.config` object.

    `kind` is "scale" (Phase 3) or "mean-scale" (Phase 5 preview). The channel
    widths, sigma quantisation and coder precision all come from the config, so
    switching tierA -> full is a flag and not an edit.
    """
    ch = config.channels
    ent = config.entropy
    cls = {"scale": ScaleHyperprior, "mean-scale": MeanScaleHyperprior}[kind]
    return cls(
        in_channels=in_channels,
        latent=ch.primary_latent,
        hyper=ch.hyper_latent,
        analysis_width=ch.analysis_width,
        synthesis_width=ch.synthesis_width,
        scale_min=ent.sigma_quant_min,
        scale_max=ent.sigma_quant_max,
        scale_levels=ent.sigma_quant_level,
        pad_multiple=config.geometry.total_downsample,
    )


def summarise(model, *, crop: int = 256) -> str:
    """One-block report: parameters, MAC/pixel, table size. Used by the selftest.

    Works on any model exposing `summary_parts()` -> `[(label, module, decoder)]`
    and `summary_title()`. Phase 4's two-branch codec has eight parts across two
    branches rather than four, and its hyper networks are nested one level deeper,
    so hardcoding `model.g_a` here would have meant a second near-identical
    reporting function -- and then two places to keep the decoder-side MAC
    convention correct in.
    """
    from jpegai.utils import count_parameters, human_bytes, macs_breakdown

    parts = model.summary_parts()
    try:
        macs = macs_breakdown(model, (1, model.in_channels, crop, crop),
                              parts=[(n, m) for n, m, _ in parts])
    except Exception as exc:                                   # pragma: no cover
        macs = {"_error": str(exc)}

    # `params` means model size, so it counts every parameter, frozen or not. That is
    # only worth spelling out because `count_parameters` defaults to trainable-only,
    # and under Table II's stage IV -- which freezes all but the two gain vectors --
    # the default turned this block into a freeze report claiming a 576 B codec. The
    # `trainable` column appears only when a freeze is actually in effect, so every
    # unfrozen run's output is unchanged.
    n_all = count_parameters(model, trainable_only=False)
    n_fit = count_parameters(model)
    frozen = n_fit != n_all

    head = f"  {'part':18} {'params':>11}  {'kMAC/pxl':>9}"
    lines = [model.summary_title(), head + (f"  {'trainable':>11}" if frozen else "")]
    for name, mod, _ in parts:
        k = macs.get(name)
        cell = f"{k / 1e3:9.1f}" if k else " " * 9
        row = f"  {name:18} {count_parameters(mod, trainable_only=False):>11,}  {cell}"
        if frozen:
            row += f"  {count_parameters(mod):>11,}"
        lines.append(row)

    total_k = macs.get("TOTAL", 0.0) / 1e3
    if frozen:
        lines.append(f"  {'TOTAL':18} {n_all:>11,}  {total_k:9.1f}  {n_fit:>11,}"
                     f"   ({human_bytes(n_all * 4)} fp32, "
                     f"{100 * n_fit / max(n_all, 1):.1f}% trainable)")
    else:
        lines.append(f"  {'TOTAL':18} {n_all:>11,}  {total_k:9.1f}"
                     f"   ({human_bytes(n_all * 4)} fp32)")
    # The paper's kMAC/pxl figures are decoder-side, so this is the number to
    # compare against 8 / 28 / 215 -- not the total above.
    dec_k = sum(macs.get(n, 0.0) for n, _, is_dec in parts if is_dec) / 1e3
    lines.append(f"  {'decoder':18} {'':>11}  {dec_k:9.1f}"
                 f"   <- compare to paper SOP 8 / BOP 28 / HOP 215")
    if model.tables_ready:
        lines.append(f"  tables             {human_bytes(model.table_bytes()):>11}")
    return "\n".join(lines)


if __name__ == "__main__":
    from jpegai.config import load_config

    for tier in ("tierA", "full"):
        cfg = load_config(tier)
        for kind in ("scale", "mean-scale"):
            m = build_model(cfg, kind=kind)
            print(f"\n=== {tier} / {kind} " + "=" * 34)
            print(summarise(m, crop=cfg.train.crop))
            x = torch.rand(1, 3, 128, 128)
            out = m(x, noise=False, ste=True)
            yb, zb = m.estimated_bits(out)
            npix = 128 * 128
            mse = float(torch.mean((out["x_hat"].detach().clamp(0, 1) - x) ** 2))
            psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
            print(f"  forward   x_hat {tuple(out['x_hat'].shape)}  "
                  f"y {tuple(out['y'].shape)}  z {tuple(out['z'].shape)}")
            # An untrained h_s ends in ReLU whose random output is mostly 0, so
            # every sigma clamps to sigma_quant_min = 0.11 and the model claims
            # near-certainty about a latent that is in fact near zero. The rate is
            # therefore *low* and the PSNR is garbage -- which is the correct
            # signature of an untrained hyperprior, not a bug. Printing both
            # together makes that unambiguous.
            print(f"  rate      {(yb + zb) / npix:.4f} bpp estimated "
                  f"({yb / npix:.4f} y + {zb / npix:.4f} z)")
            print(f"  quality   {psnr:.2f} dB PSNR   <- untrained: random weights, "
                  f"sigma pinned at the 0.11 bound")
