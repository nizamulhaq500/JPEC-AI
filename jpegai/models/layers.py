"""Convolutional building blocks shared by every transform in this project.

Kept deliberately small. Two things here are worth more than the line count
suggests:

* **Padding conventions.** `conv(k=5, s=2)` and `deconv(k=5, s=2)` are exact
  inverses in *shape* only for the specific padding/output_padding pair used
  below. Getting it wrong shifts the reconstruction by half a pixel, which looks
  like a mildly blurry decode rather than an obvious bug, and costs ~1 dB
  forever. See :func:`deconv`.
* **GDN.** Implemented because the paper's decoderID 2 (HOP) is allowed to use it
  and Phase 13 ablates it, but **not** the default. The paper restricts SOP/BOP
  to conv/ReLU/ReLU6 precisely because GDN is awkward in fixed point, and
  `docs/03` flags it as unstable on MPS. ReLU is both faster and closer to the
  standard.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from jpegai.models.entropy import LowerBound


def conv(in_ch: int, out_ch: int, kernel: int = 5, stride=2) -> nn.Conv2d:
    """Downsampling conv with 'same'-style padding.

    ``padding = kernel // 2`` gives ``out = ceil(in / stride)``, which is the
    convention the whole project depends on: four stages of stride 2 take a
    64-multiple input to exactly ``in / 16``.

    `stride` may be a `(vertical, horizontal)` pair. Phase 4's secondary branch
    needs that for internal 4:2:2, where the chroma grid is already half width and
    so must be downsampled 16x vertically but only 8x horizontally to land on the
    same latent grid as luma.
    """
    return nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)


def deconv(in_ch: int, out_ch: int, kernel: int = 5, stride=2
           ) -> nn.ConvTranspose2d:
    """Upsampling transposed conv that exactly inverts :func:`conv`'s shape.

    For ``k=5, s=2`` the pair ``padding=2, output_padding=1`` gives
    ``out = 2 * in``. The general form below is
    ``output_padding = stride - 1`` with ``padding = kernel // 2``, which
    satisfies ConvTranspose2d's shape formula

        out = (in - 1) * stride - 2 * padding + kernel + output_padding

    only when ``kernel`` is odd. An even kernel with these settings is off by
    one, so it is rejected rather than silently producing a mis-sized tensor
    that later gets cropped and looks "nearly right".

    `stride` may be a `(vertical, horizontal)` pair, in which case
    `output_padding` is computed per axis -- a scalar ``stride - 1`` would raise on
    a tuple, and getting it wrong on one axis only is exactly the mis-sized tensor
    the check above exists to prevent.
    """
    if kernel % 2 == 0:
        raise ValueError(
            f"deconv needs an odd kernel to invert conv's padding exactly, got "
            f"{kernel}. Phase 7's 4x4 transposed convs use PixelShuffle instead."
        )
    out_pad = (tuple(s - 1 for s in stride)
               if isinstance(stride, (tuple, list)) else stride - 1)
    return nn.ConvTranspose2d(
        in_ch, out_ch, kernel, stride=stride,
        padding=kernel // 2, output_padding=out_pad,
    )


def conv_shuffle(in_ch: int, out_ch: int, factor: int = 2, kernel: int = 3
                 ) -> nn.Sequential:
    """Upsample by `factor` via ``conv -> PixelShuffle``.

    This is JPEG AI's upsampling primitive: the paper's SOP uses it exclusively
    and every decoder uses it for the final layer. It does the arithmetic at the
    *lower* resolution -- a conv producing ``out_ch * factor**2`` channels at
    H x W costs the same as one producing ``out_ch`` at H x W, but yields
    ``factor*H x factor*W``, so the MAC/pixel count drops by ``factor**2``
    relative to a transposed conv. It also has no checkerboard artefact, because
    every output pixel comes from its own filter rather than from overlapping
    strided taps.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * factor * factor, kernel, stride=1,
                  padding=kernel // 2),
        nn.PixelShuffle(factor),
    )


class GDN(nn.Module):
    """Generalised Divisive Normalisation (Balle et al., ICLR 2016).

        y_i = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2)

    A learned, invertible, *channel-coupling* normalisation. It works well for
    compression because it whitens across channels, which is what a factorised
    prior wants -- Balle's original hyperprior gains roughly 0.5-1 dB from GDN
    over ReLU.

    Not used by default here. `beta` and `gamma` must stay positive, so they are
    reparameterised through a square (the original paper's trick) with a lower
    bound; that reparameterisation is the part that misbehaves in reduced
    precision and on MPS, and it is why JPEG AI's SOP/BOP profiles forbid GDN
    outright. Provided so Phase 13 can ablate "GDN vs ReLU" with real numbers
    instead of citing the literature.
    """

    def __init__(self, channels: int, inverse: bool = False,
                 beta_min: float = 1e-6, gamma_init: float = 0.1):
        super().__init__()
        self.inverse = bool(inverse)
        # Reparameterise as sqrt: the forward pass squares, so the effective
        # value is positive by construction and the bound only has to keep the
        # square root away from zero.
        self.beta_bound = float(beta_min ** 0.5)
        self.gamma_bound = 1e-6 ** 0.5
        self.lower_beta = LowerBound(self.beta_bound)
        self.lower_gamma = LowerBound(self.gamma_bound)
        self.beta = nn.Parameter(torch.ones(channels))
        eye = torch.eye(channels)
        self.gamma = nn.Parameter((gamma_init ** 0.5) * eye)

    def forward(self, x: Tensor) -> Tensor:
        c = x.shape[1]
        beta = self.lower_beta(self.beta) ** 2
        gamma = (self.lower_gamma(self.gamma) ** 2).reshape(c, c, 1, 1)
        norm = F.conv2d(x * x, gamma, beta)
        norm = torch.sqrt(norm)
        return x * norm if self.inverse else x / norm


def activation(name: str, channels: int, inverse: bool = False) -> nn.Module:
    """Factory so a config string selects the nonlinearity.

    `relu6` is included because it is in the paper's allowed-op list for SOP and
    BOP: a bounded activation is what makes 8-bit fixed-point inference viable,
    which is the whole point of restricting those profiles.
    """
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    if name == "gdn":
        return GDN(channels, inverse=inverse)
    raise ValueError(f"unknown activation {name!r}; expected one of "
                     "relu, relu6, leaky_relu, gdn")


# ---------------------------------------------------------------------------
# Size handling
# ---------------------------------------------------------------------------
def pad_to_multiple(x: Tensor, multiple: int = 64) -> tuple[Tensor, tuple[int, ...]]:
    """Reflect-pad so H and W are multiples of `multiple`. Returns (padded, pad).

    `multiple` is 64 for this architecture: four analysis stages then two hyper
    stages, so a dimension not divisible by 64 leaves the hyper latent grid
    ragged and the hyper decoder's output no longer aligns with the latent it is
    supposed to describe.

    Reflect rather than zero padding: a zero border is a hard edge the analysis
    transform must spend bits describing, and those bits buy nothing because the
    region is cropped away. Reflection extends the image's own statistics.

    This is the crude version of the paper's section VI-J. Phase 12 replaces it
    with the two normative options (`layer_cropping` and `display_window`).
    """
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph == 0 and pw == 0:
        return x, (0, 0, 0, 0)
    # Reflect padding requires pad < dim; fall back to replicate for tiny inputs.
    mode = "reflect" if (ph < h and pw < w) else "replicate"
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    return F.pad(x, pad, mode=mode), pad


def unpad(x: Tensor, pad: tuple[int, ...]) -> Tensor:
    """Undo :func:`pad_to_multiple`. `pad` is (left, right, top, bottom)."""
    left, right, top, bottom = pad
    h, w = x.shape[-2:]
    return x[..., top:h - bottom if bottom else h, left:w - right if right else w]
