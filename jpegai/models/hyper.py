r"""Phase 5: the split hyper decoders and the integer sigma index.

JPEG AI does not have one hyper decoder producing `(mu, sigma)`. It has **two
separate networks** off the same `zhat` (§VI-E), and that asymmetry is the paper's own
architectural contribution:

    zhat --> hyper_decoder        --> p_dd  (the prediction, eq. 1/2's `p̈`)
         \-> hyper_scale_decoder  --> Isigma (an integer index, eq. 13)

They are split because the two outputs are used at different times and at different
precisions. `p̈` is a *prediction* that must be reproduced bit-exactly at the decoder
(any drift and `ŷ = r̂ + p̈` decodes to the wrong latent), and it feeds the context
model in Phase 6, so it is the expensive one. `Iσ` only has to select one of 32 CDF
rows, so it can be a tiny network -- and it is deliberately *integer*, because Phase 10's
RVS and LSBS tools index tables by `Iσ` and Phase 11 needs it identical across devices.
Fusing them, as Balle's `h_s` does, forces the cheap output to be computed by the
expensive network. `--single-hyper-decoder` measures what the decoupling costs; the
paper never publishes that number.

Structures here are the reference software's, confirmed in `docs/06` §1-2, not the
paper's figures:

* hyper encoder: **five `conv3x3(chs, chs)`**, two of them stride-2, so `/16 -> /64`.
  Channel-preserving throughout, which is why there is no independent hyper width.
* hyper decoder: body at `/32`, ending `conv3x3(chs, 4*chs) -> PixelShuffle(2)`, so
  `p̈_Y` is `[640, /32, /32]` before the shuffle and `[160, /16, /16]` after.
* hyper scale decoder: body entirely at `/64`, ending
  `conv1x1(chs, 16*chs) -> PixelShuffle(4)`. One shuffle covers `/64 -> /16`, which is
  what makes it nearly free: every multiply happens at 1/16 the latent's area.

The rounding rule, and why it is certain
----------------------------------------
`Iσ` carries `sigma_precision = 7` fractional bits, so `t = Iσ / 128` is the exact
position on the 32-entry log-spaced grid, and the CDF row is `t` rounded to an integer.
Which way it rounds was an open question, and it is settled by
`sigma_idx_max_value = (levels - 1) * 2**precision - 1 = 3967`:

* rounding **up**, `ceil(3967/128) = 31 = levels - 1`. The maximum index lands exactly
  on the last row.
* rounding **down**, `3967 >> 7 = 30`. Row 31 would be unreachable -- a CDF row that
  can never be selected, in a design whose stated goal is a small table.

So round-up it is, and that independently agrees with Phase 3's measurement: rounding
σ up means the coder always uses a distribution at least as wide as the model
predicted, so the σ grid costs **rate and never escapes** (rounding down was measured
at 0.63% escapes for -17% rate).

Why the integer index is not merely a storage format
----------------------------------------------------
`table_row(Iσ)` and `GaussianConditional.build_indexes(sigma(Iσ))` implement the same
rule, and they agree on 3957 of the 3968 indices. They disagree on the other **11**,
and every one of them is an exact multiple of 128 -- an `Iσ` sitting exactly on a grid
point. The cause is one float32 ULP:

    I = 256:  sigma(I)  = 0.16422072052955627   (min * exp(log_k * 2), torch float32)
              table[2]  = 0.16422070562839508   (exp(linspace(...)), numpy float64->32)
                          ~1.5e-7 relative, and sigma(I) lands on the high side

`build_indexes` counts entries strictly below σ, so that last bit pushes it to row 3
while the exact arithmetic says row 2. Neither answer is unsafe -- row 3 is merely
wider -- but they are *different*, and a bitstream whose encoder used one rule and
whose decoder used the other is undecodable for 0.28% of its symbols.

That is the whole argument for JPEG AI keeping `Iσ` as an integer, and it is stronger
than "it saves space": `table_row` on an int is exact on every device and in every
build, while `build_indexes` on a reconstructed float σ is at the mercy of how the two
sides happened to compute an exponential. Phase 11's cross-device bit-exactness
requirement is unachievable through the float path. So the rule for this project is
that the split-hyper codec indexes CDF rows **only** through `table_row`, and
`tests/test_hyper.py` pins both the agreement and the exact 11 exceptions, so a change
to either path shows up as a test failure rather than as a corrupt bitstream.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from jpegai.models.entropy import Clamp, FactorizedPrior, GaussianConditional
from jpegai.models.layers import activation, conv, conv_shuffle


class SigmaIndex(nn.Module):
    """The `Iσ` <-> σ mapping of eq. (13), plus the CDF row lookup.

    Holds no parameters -- it is the fixed codebook the scale decoder's output is
    interpreted through. A `nn.Module` rather than a set of free functions for two
    practical reasons: the two-sided bound's buffers then follow the model across
    `.to(device)`, and the constants show up in `print(model)`, where a σ grid that
    silently disagrees with the `GaussianConditional`'s is visible rather than
    something you deduce from a bad rate.
    """

    def __init__(self, *, minimum: float = 0.11, maximum: float = 54.82,
                 levels: int = 32, precision: int = 7):
        super().__init__()
        if levels < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")
        if not 0 < minimum < maximum:
            raise ValueError(f"need 0 < minimum < maximum, got {minimum}, {maximum}")
        self.minimum, self.maximum = float(minimum), float(maximum)
        self.levels, self.precision = int(levels), int(precision)
        #: `2**precision`. The number of `Iσ` steps per CDF row.
        self.step = 1 << self.precision
        #: eq. (13)'s `k`: one row of the grid is `exp(log_k)` in σ.
        self.log_k = (math.log(self.maximum) - math.log(self.minimum)) \
            / (self.levels - 1)
        #: `(levels-1) * 2**precision - 1` = 3967 for the normative constants.
        self.max_index = (self.levels - 1) * self.step - 1
        self.clamp = Clamp(0.0, float(self.max_index))

    def sigma(self, index: Tensor) -> Tensor:
        """`Iσ -> σ`, differentiable. Accepts a float index during training."""
        return self.minimum * torch.exp(self.log_k * self.clamp(index) / self.step)

    def quantise(self, index: Tensor) -> Tensor:
        """`Iσ -> round(Iσ)` as an integer tensor, clamped into the table.

        Only at inference. During training the float index feeds `sigma()` directly:
        rounding it would zero the gradient of the entire scale decoder, and Phase 11
        adds the quantisation-aware finetune that makes the rounded version optimal.
        """
        return torch.round(self.clamp(index)).to(torch.int32)

    def table_row(self, index: Tensor) -> Tensor:
        """`Iσ -> CDF row`, rounding **up**. See this module's docstring.

        Integer `ceil(a/b)` as `(a + b - 1) // b`, so it is exact for an int tensor
        and does not depend on float rounding at the boundary -- which matters,
        because the boundary is where the round-up rule earns its keep.

        Every operation is out-of-place: `index` may already be int64, in which case
        an in-place clamp would silently rewrite the caller's tensor.
        """
        idx = index.to(torch.int64).clamp(0, self.max_index)
        row = (idx + self.step - 1) // self.step
        return row.clamp(0, self.levels - 1).to(torch.int32)

    def from_sigma(self, sigma: Tensor) -> Tensor:
        """`σ -> Iσ` (float). The inverse of `sigma()`, for tests and for the
        `--single-hyper-decoder` ablation, whose fused network predicts σ directly
        and still has to report an `Iσ` so the two paths stay comparable."""
        s = sigma.clamp_min(self.minimum)
        return self.clamp(torch.log(s / self.minimum) / self.log_k * self.step)

    def extra_repr(self) -> str:
        return (f"levels={self.levels}, precision={self.precision}, "
                f"sigma=[{self.minimum:g}, {self.maximum:g}], "
                f"index=[0, {self.max_index}]")


class HyperEncoder(nn.Module):
    """h_a: `y [latent, /16] -> z [latent, /64]`. Five conv3x3, two stride-2.

    The confirmed reference structure (docs/06 §1), which differs from Phase 3's
    `HyperAnalysis` -- that one is three convs with 5x5 kernels. Both reach /64; this
    one is what JPEG AI actually does, and it is channel-preserving, which is *why*
    there is no independent hyper width to configure. `hyper` is accepted only so a
    mismatch fails loudly here rather than as a shape error two modules later.

    Consumes `abs(y)`: `entropy.abs_in_hyperprior: true`. This network predicts a
    scale, scale is a property of magnitude, and handing it the sign spends capacity
    learning that σ(y) = σ(−y).
    """

    def __init__(self, latent: int, hyper: int, *, activation_name: str = "relu"):
        super().__init__()
        if hyper != latent:
            raise ValueError(
                f"the hyper autoencoder is channel-preserving, so hyper must equal "
                f"latent; got hyper={hyper}, latent={latent}. See docs/06 §1 -- the "
                f"reference software builds every hyper module with chs=chs_ls."
            )
        c = int(latent)
        # Strides on layers 2 and 4 rather than 1 and 2: one stride-1 conv at the
        # latent resolution first, so the network sees the latent's own neighbourhood
        # before any of it is thrown away.
        self.body = nn.Sequential(
            conv(c, c, 3, stride=1), activation(activation_name, c),
            conv(c, c, 3, stride=2), activation(activation_name, c),
            conv(c, c, 3, stride=1), activation(activation_name, c),
            conv(c, c, 3, stride=2), activation(activation_name, c),
            conv(c, c, 3, stride=1),
        )

    def forward(self, y: Tensor) -> Tensor:
        return self.body(torch.abs(y))


class HyperDecoder(nn.Module):
    """`zhat [chs, /64] -> p̈ [chs, /16]`, the prediction of eq. (1)/(2).

    Two upsamples: `conv_shuffle` from /64 to /32, then the confirmed final layer
    `conv3x3(chs, 4*chs) -> PixelShuffle(2)` from /32 to /16 -- which is the same
    primitive, so it is one call. The intermediate `[4*chs, /32, /32]` tensor is the
    paper's `[640, Ḣ/32, Ẇ/32]` at full width.

    No activation on the output. `p̈` is a *prediction of a latent*, which is signed
    and roughly zero-mean; a ReLU here would forbid negative predictions and the
    residual `r̂ = round(y − p̈)` would carry the whole negative half of the latent
    distribution. (`HyperSynthesisScale` ends in ReLU for the opposite reason -- σ
    must be non-negative.)

    `shuffle=False` stops one step short and returns the `[4*chs, /32]` tensor. That
    is Phase 6's entry point: `/32` is exactly one MCM coset's grid, so the four
    `chs`-wide slices of that tensor are four per-coset explicit predictions with no
    upsampling needed, which is what `HyperToContext9x1b` reads in the reference
    software and what `channels.pred_primary_preshuffle` in the config has recorded
    since Phase 1. The parameters are the same either way -- `PixelShuffle` has none --
    so a Phase 5 checkpoint's `h_s` loads into a Phase 6 model unchanged.
    """

    def __init__(self, chs: int, *, activation_name: str = "relu",
                 width: int | None = None, shuffle: bool = True):
        super().__init__()
        c = int(chs)
        mid = int(width) if width else c
        self.shuffle = bool(shuffle)
        self.body = nn.Sequential(
            conv_shuffle(c, mid, factor=2, kernel=3),      # /64 -> /32
            activation(activation_name, mid, inverse=True),
            conv(mid, mid, 3, stride=1),
            activation(activation_name, mid, inverse=True),
            conv_shuffle(mid, c, factor=2, kernel=3),      # /32 -> /16, 4*chs inside
        )

    def forward(self, z_hat: Tensor) -> Tensor:
        if self.shuffle:
            return self.body(z_hat)
        # Everything but the final `PixelShuffle`. Walked explicitly rather than by
        # slicing `self.body`, so the parameter names stay exactly what they are with
        # the shuffle on -- which is the whole point of the flag.
        mods = list(self.body)
        h = z_hat
        for m in mods[:-1]:
            h = m(h)
        return mods[-1][0](h)                              # the conv, not the shuffle


class HyperScaleDecoder(nn.Module):
    """`zhat [chs, /64] -> Iσ [chs, /16]`, a float index during training.

    Deliberately tiny, and the acceptance test is a number: **under 5% of decoder
    MACs**. That is achievable only because every multiply happens at /64 -- the one
    `PixelShuffle(4)` at the end covers the whole /64 -> /16 gap, so the expensive
    16x area increase happens after the last conv rather than before it. A version
    that upsampled first and convolved at /16 would compute the same function for 16x
    the cost, which is exactly the trap the split exists to avoid.

    The output is *unactivated*. `Iσ` is linear in log σ, so predicting it raw is
    predicting log σ up to an affine map -- the natural parameterisation. `SigmaIndex`
    applies the two-sided bound; nothing here needs to.

    `init_index` is load-bearing, not a nicety
    ------------------------------------------
    A conv stack with default init outputs values near zero, so an uninitialised
    scale decoder predicts `Iσ ~ 0`, which is `σ = sigma_quant_min = 0.11` for every
    element of the latent -- the narrowest distribution in the table, asserted with
    total confidence over a latent that is still random. Every symbol escapes, the
    rate term is enormous, and the gradient it produces is dominated by the escapes
    rather than by anything about the image. Setting the final bias puts the initial
    prediction mid-table, at `max_index/2 = 1983.5`, whose σ is 2.4537 -- a half step
    below the grid's exact geometric mean of `sqrt(0.11 * 54.82) = 2.4556`, because
    `max_index` is odd. That is the honest "I don't know yet" starting point: the
    middle of a log-spaced grid is the value that is the same factor away from both
    ends.
    """

    def __init__(self, chs: int, *, layers: int = 2, init_index: float = 0.0,
                 activation_name: str = "relu"):
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        c = int(chs)
        body: list[nn.Module] = []
        for _ in range(layers - 1):
            body += [conv(c, c, 3, stride=1), activation(activation_name, c)]
        # conv1x1 for the shuffle, per docs/06 §1. A 3x3 here would quadruple this
        # network's parameters (chs*16*chs*9) to no purpose: the spatial mixing was
        # already done by the layers above, and this one only redistributes channels
        # into the 4x4 output block.
        body.append(conv_shuffle(c, c, factor=4, kernel=1))
        self.body = nn.Sequential(*body)
        if init_index:
            # The Conv2d inside the final conv_shuffle; PixelShuffle only permutes,
            # so a constant bias there is a constant offset on Isigma.
            nn.init.constant_(self.body[-1][0].bias, float(init_index))

    def forward(self, z_hat: Tensor) -> Tensor:
        return self.body(z_hat)


class FusedHyperDecoder(nn.Module):
    """The `--single-hyper-decoder` ablation: one Balle-style `h_s` for both.

    Predicts `[2*chs, /16]` and splits it into `(p̈, Iσ)`. Present so Phase 13 can
    measure what JPEG AI's decoupling costs in RD -- the paper argues the split on
    complexity and bit-exactness grounds and never quantifies the RD price, so this
    is a number the project can report that the paper does not.

    Deliberately given the *same* two-upsample shape and roughly the same width as
    `HyperDecoder`, so the comparison isolates the decoupling rather than measuring a
    capacity difference. It therefore has slightly more parameters than the split
    pair, not fewer -- which is the honest direction for an ablation that expects the
    fused version to win on RD.
    """

    def __init__(self, chs: int, *, activation_name: str = "relu",
                 init_index: float = 0.0):
        super().__init__()
        c = int(chs)
        self.body = nn.Sequential(
            conv_shuffle(c, c, factor=2, kernel=3),
            activation(activation_name, c, inverse=True),
            conv(c, c, 3, stride=1),
            activation(activation_name, c, inverse=True),
            conv_shuffle(c, 2 * c, factor=2, kernel=3),
        )
        self.chs = c
        if init_index:
            # Bias only the Isigma half, exactly as the split version does -- p_dd
            # must still start near zero. `PixelShuffle(r)` sends conv output
            # channels [r*r*j, r*r*(j+1)) to shuffled channel j, so `chunk(2)`'s
            # second half comes from the conv's *upper* 4c channels and nothing else.
            with torch.no_grad():
                self.body[-1][0].bias[4 * c:].fill_(float(init_index))

    def forward(self, z_hat: Tensor) -> tuple[Tensor, Tensor]:
        pred, i_sigma = self.body(z_hat).chunk(2, dim=1)
        return pred, i_sigma


class SplitHyperBranch(nn.Module):
    """One branch's side information, Phase 5 style: `h_a` + two decoders + `z` prior.

    Drop-in for Phase 4's `HyperpriorBranch` -- same `forward`/`compress`/`decompress`
    contract, so `TwoBranchCodec` selects between them with a flag and the training
    loop, the loss and the benchmark need no changes.

    Two things are *shared*, not owned, and for the same reason in both cases: one
    `GaussianConditional` (its table is indexed by σ row and has no channel
    dimension) and one `SigmaIndex` (JPEG AI has one σ grid). Giving each branch its
    own would double the table bytes to represent the same 32 rows, and -- worse for
    the scale decoder -- would let the two branches drift onto different grids, which
    is undetectable until a rate gap appears with no other symptom.

    Residual coding (eqs 1, 2) is `GaussianConditional`'s `means` argument doing its
    job: `compress` codes `round(y - p̈)` and `decompress` returns `r̂ + p̈`. What
    Phase 5 changes is not that arithmetic but *where `p̈` comes from* -- a network of
    its own, so the prediction can be good without making the σ path expensive.
    """

    def __init__(self, latent: int, hyper: int, *, sigma_index: SigmaIndex,
                 fused: bool = False, scale_layers: int = 2,
                 activation_name: str = "relu", precision: int = 16):
        super().__init__()
        self.latent, self.hyper = int(latent), int(hyper)
        self.fused = bool(fused)
        self.sigma_index = sigma_index
        self.h_a = HyperEncoder(latent, hyper, activation_name=activation_name)
        # Mid-table (sigma ~ 2.45, the middle of the log-spaced grid). See
        # HyperScaleDecoder's docstring -- without it every latent element starts at
        # sigma_quant_min and the run opens with a rate explosion.
        init = sigma_index.max_index / 2.0
        if fused:
            self.h_s = FusedHyperDecoder(latent, activation_name=activation_name,
                                         init_index=init)
            self.h_scale = None
        else:
            self.h_s = HyperDecoder(latent, activation_name=activation_name)
            self.h_scale = HyperScaleDecoder(latent, layers=scale_layers,
                                             init_index=init,
                                             activation_name=activation_name)
        self.entropy_bottleneck = FactorizedPrior(hyper, precision=precision)

    # -- prediction ---------------------------------------------------------
    def predict(self, z_hat: Tensor, *, quantise: bool = False) -> dict:
        """`{means, i_sigma, scales, rows}` from `zhat` alone.

        `means` is `p̈`; `i_sigma` is the raw float index; `scales` is the σ that
        index denotes; `rows` is the CDF row, or None unless `quantise`.

        **Nothing here may depend on `y`.** That is the one invariant of eq. (1)/(2):
        the decoder has only `zhat` when it computes `p̈`, so any `y` dependence
        produces an encoder-only prediction and the decoder drifts. Taking `z_hat` as
        the sole argument makes that structural rather than a rule to remember.

        `quantise=True` is the inference path: round `Iσ` to an integer first, then
        derive both σ and the row from the *integer*, so the two are consistent with
        each other and reproducible on any device.
        """
        if self.fused:
            means, i_sigma = self.h_s(z_hat)
        else:
            means, i_sigma = self.h_s(z_hat), self.h_scale(z_hat)
        if not quantise:
            return {"means": means, "i_sigma": i_sigma,
                    "scales": self.sigma_index.sigma(i_sigma), "rows": None}
        i_int = self.sigma_index.quantise(i_sigma)
        return {"means": means, "i_sigma": i_int,
                "scales": self.sigma_index.sigma(i_int.float()),
                "rows": self.sigma_index.table_row(i_int)}

    def params(self, z_hat: Tensor) -> tuple[Tensor, Tensor]:
        """`(scales, means)`, for callers that predate the split."""
        p = self.predict(z_hat)
        return p["scales"], p["means"]

    # -- training -----------------------------------------------------------
    def forward(self, y: Tensor, gc: GaussianConditional, *,
                noise: bool | None = None, ste: bool = True) -> dict:
        z = self.h_a(y)
        z_hat, z_lik = self.entropy_bottleneck(z, noise=noise, ste=ste)
        p = self.predict(z_hat)
        y_hat, y_lik = gc(y, p["scales"], p["means"], noise=noise, ste=ste)
        return {"y_hat": y_hat, "y_lik": y_lik, "z_lik": z_lik,
                "z": z, "z_hat": z_hat,
                "scales": p["scales"], "means": p["means"],
                # Exposed so the round-trip gate can check the index the coder will
                # actually use, and so training can watch the index distribution --
                # a scale decoder pinned at 0 or at 3967 is a dead branch, and it
                # looks identical to a well-behaved one in the loss.
                "i_sigma": p["i_sigma"]}

    # -- real bitstream -----------------------------------------------------
    @torch.no_grad()
    def compress(self, y: Tensor, gc: GaussianConditional) -> dict:
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        # From the *decoded* z_hat, never from z: the decoder has only the former,
        # and the two differ wherever the factorised prior's rounding disagrees.
        z_hat = self.entropy_bottleneck.decompress(
            z_strings, tuple(z.shape[-2:]), device=z.device)
        p = self.predict(z_hat, quantise=True)
        return {"y_strings": gc.compress(y, p["scales"], p["means"],
                                         indexes=p["rows"]),
                "z_strings": z_strings, "z_shape": tuple(z.shape[-2:])}

    @torch.no_grad()
    def decompress(self, part: dict, gc: GaussianConditional, device) -> dict:
        z_hat = self.entropy_bottleneck.decompress(
            part["z_strings"], tuple(part["z_shape"]), device=device)
        p = self.predict(z_hat, quantise=True)
        return {"y_hat": gc.decompress(part["y_strings"], p["scales"], p["means"],
                                       indexes=p["rows"]),
                "z_hat": z_hat}

