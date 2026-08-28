"""Colour space and chroma format handling — JPEG AI §VI-A/§VI-B.

This is Phase 4's first piece and it is deliberately separate from the model: the
two-branch architecture needs to split an image into a luma plane and a chroma pair
*at possibly different resolutions*, and put them back together at the end, and
every one of those steps has an off-by-one that only shows up on odd-sized images.

Four things live here:

1. **BT.709 full-range RGB <-> YCbCr**, an exact inverse pair.
2. **Chroma formats** 4:4:4 / 4:2:2 / 4:2:0, with the ceilings the paper's
   `⌈H/2⌉` implies, and the *internal* format handled independently of the
   *output* format (`c_ver_minus1`/`c_hor_minus1` vs `s_ver_minus1`/`s_hor_minus1`).
3. **`colour_transform_idx`** 0/1/2, including the encoder-signalled 3x3 matrix and
   bias of eq. (5).
4. **eq. (6)**, the final scale-and-clip to integer samples.

The paper's eq. (4) as printed does not work
--------------------------------------------
Both its first and second lines index `x̂_UV[1]`, so Cr is used where Cb belongs,
and `0.0722` is printed `0.07222`. Implemented as printed, the inverse is not the
inverse of anything: a grey ramp comes back magenta. What is implemented here is the
standard BT.709 inverse, which round-trips to 6e-8 in float32 (see
`tests/test_colour.py`). Noted in docs/02 §VI-B; needs a look at T.840-1 to confirm
the standard text agrees, which we cannot do without ITU access.

The three luma coefficients are the only magic numbers
------------------------------------------------------
`1.5748` and `1.8556` are *derived* below as `2(1-K_R)` and `2(1-K_B)` rather than
typed in, and asserted against the paper's printed values at import. Two constants
that must agree with three others is exactly the kind of thing that silently drifts
when someone switches BT.709 for BT.601.

What is a stand-in, and why it is isolated
------------------------------------------
The chroma **resampling filters** are normative in JPEG AI Part 1, which we do not
have. `subsample_chroma` uses an exact box average and `upsample_chroma` bilinear
interpolation -- reasonable, standard, and *wrong in detail*. They are confined to
those two functions, called from nowhere else, so swapping in the normative filters
later is a two-function change and not an archaeology project. Any number this
project reports for a non-4:4:4 internal format carries that caveat.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# BT.709 constants
# ---------------------------------------------------------------------------
#: ITU-R BT.709 luma coefficients. These three are the primary source of truth.
K_R, K_G, K_B = 0.2126, 0.7152, 0.0722

#: Chroma normalisation, derived so they cannot disagree with the K above.
CB_SCALE = 2.0 * (1.0 - K_B)          # 1.8556
CR_SCALE = 2.0 * (1.0 - K_R)          # 1.5748

assert abs(K_R + K_G + K_B - 1.0) < 1e-12, "luma coefficients must sum to 1"
assert abs(CB_SCALE - 1.8556) < 1e-12, CB_SCALE
assert abs(CR_SCALE - 1.5748) < 1e-12, CR_SCALE


def rgb_to_ycbcr_bt709(x: Tensor) -> Tensor:
    """RGB in [0,1] -> YCbCr with Y in [0,1] and Cb/Cr centred on 0.5.

    Full range (no 16-235 studio swing): the paper's models are trained on full-range
    BT.709, and a studio-range pipeline would quietly lose 7% of the dynamic range
    at both ends of every plane.
    """
    if x.shape[-3] != 3:
        raise ValueError(f"expected 3 channels, got shape {tuple(x.shape)}")
    r, g, b = x[..., 0:1, :, :], x[..., 1:2, :, :], x[..., 2:3, :, :]
    y = K_R * r + K_G * g + K_B * b
    cb = (b - y) / CB_SCALE + 0.5
    cr = (r - y) / CR_SCALE + 0.5
    return torch.cat([y, cb, cr], dim=-3)


def ycbcr_to_rgb_bt709(x: Tensor) -> Tensor:
    """The exact inverse of :func:`rgb_to_ycbcr_bt709`. Not clipped.

    Deliberately returns values outside [0,1] when the input is outside the RGB
    gamut, rather than clipping here. Clipping belongs in eq. (6) at the very end of
    decoding, and doing it twice hides the fact that a reconstruction went out of
    range -- which is a real signal, not noise, when a synthesis transform is
    misbehaving.
    """
    if x.shape[-3] != 3:
        raise ValueError(f"expected 3 channels, got shape {tuple(x.shape)}")
    y, cb, cr = x[..., 0:1, :, :], x[..., 1:2, :, :], x[..., 2:3, :, :]
    r = y + CR_SCALE * (cr - 0.5)
    b = y + CB_SCALE * (cb - 0.5)
    # From y = K_R r + K_G g + K_B b. Not a fourth independent constant.
    g = (y - K_R * r - K_B * b) / K_G
    return torch.cat([r, g, b], dim=-3)


# ---------------------------------------------------------------------------
# Chroma formats
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChromaFormat:
    """A chroma sampling format, in the paper's own syntax elements.

    `ver`/`hor` are the subsampling *factors*, i.e. `c_ver_minus1 + 1` and
    `c_hor_minus1 + 1`. Stored as factors rather than as the minus-one form because
    every piece of arithmetic downstream wants the factor, and carrying the
    minus-one form around is how an off-by-one gets in.
    """

    name: str
    ver: int
    hor: int

    @property
    def ver_minus1(self) -> int:
        return self.ver - 1

    @property
    def hor_minus1(self) -> int:
        return self.hor - 1

    @property
    def is_444(self) -> bool:
        return self.ver == 1 and self.hor == 1

    def chroma_size(self, h: int, w: int) -> tuple[int, int]:
        """Chroma plane size for a luma plane of `h x w`. **Ceiling**, not floor.

        A 4:2:0 chroma plane for a 321x499 image is 161x250, not 160x249: the paper
        writes ⌈H/2⌉ and it matters, because flooring drops the last row and column
        of colour and the reconstruction then has a grey edge that no metric
        attributes to the codec.
        """
        return (-(-h // self.ver), -(-w // self.hor))


FORMATS: dict[str, ChromaFormat] = {
    "444": ChromaFormat("444", 1, 1),
    "422": ChromaFormat("422", 1, 2),      # horizontally halved only
    "420": ChromaFormat("420", 2, 2),
}


def get_format(fmt) -> ChromaFormat:
    """Accept a name, a ChromaFormat, or a (ver_minus1, hor_minus1) pair."""
    if isinstance(fmt, ChromaFormat):
        return fmt
    if isinstance(fmt, str):
        key = fmt.replace(":", "")
        if key not in FORMATS:
            raise ValueError(f"unknown chroma format {fmt!r}; "
                             f"expected one of {sorted(FORMATS)}")
        return FORMATS[key]
    if isinstance(fmt, (tuple, list)) and len(fmt) == 2:
        ver, hor = int(fmt[0]) + 1, int(fmt[1]) + 1
        for f in FORMATS.values():
            if (f.ver, f.hor) == (ver, hor):
                return f
        return ChromaFormat(f"{ver}x{hor}", ver, hor)
    raise TypeError(f"cannot interpret {fmt!r} as a chroma format")


def subsample_chroma(uv: Tensor, fmt) -> Tensor:
    """Chroma at full resolution -> chroma at `fmt`'s resolution. Box average.

    `count_include_pad=False` with `ceil_mode=True` is what makes the odd-size case
    correct: the final partial window averages over the samples that exist instead
    of averaging in zeros, which would darken the last row and column.
    """
    f = get_format(fmt)
    if f.is_444:
        return uv
    return F.avg_pool2d(uv, kernel_size=(f.ver, f.hor), stride=(f.ver, f.hor),
                        ceil_mode=True, count_include_pad=False)


def upsample_chroma(uv: Tensor, size: tuple[int, int], *,
                    mode: str = "bilinear") -> Tensor:
    """Chroma -> the given `(h, w)`, normally the luma plane's size.

    Takes an explicit target size rather than a scale factor. With a scale factor,
    a 4:2:0 chroma plane of 161x250 upsamples to 322x500 and not back to the 321x499
    it came from, and the caller is then left cropping a row off -- an operation
    nobody remembers to do consistently.
    """
    if tuple(uv.shape[-2:]) == tuple(size):
        return uv
    kw = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    return F.interpolate(uv, size=tuple(size), mode=mode, **kw)


def split_planes(x: Tensor, fmt="420") -> tuple[Tensor, Tensor]:
    """YCbCr image -> (luma `[N,1,H,W]`, chroma `[N,2,h,w]`) for the two branches."""
    f = get_format(fmt)
    y = x[..., 0:1, :, :]
    uv = subsample_chroma(x[..., 1:3, :, :], f)
    return y, uv


def merge_planes(y: Tensor, uv: Tensor, *, mode: str = "bilinear") -> Tensor:
    """(luma, chroma) -> a full-resolution YCbCr image. Chroma is upsampled to luma."""
    return torch.cat([y, upsample_chroma(uv, y.shape[-2:], mode=mode)], dim=-3)


def convert_chroma_format(uv: Tensor, luma_hw: tuple[int, int], src, dst, *,
                          mode: str = "bilinear") -> Tensor:
    """Chroma from one format's grid to another's, given the luma size.

    This is the function that makes the *internal* format (`c_ver_minus1`,
    `c_hor_minus1`) independent of the *output* format (`s_ver_minus1`,
    `s_hor_minus1`): the codec may reconstruct at 4:2:0 and be asked to emit 4:2:2.

    Routes through full resolution rather than resampling directly. A direct
    4:2:0 -> 4:2:2 filter would be better quality, but composing two known-correct
    steps guarantees the sizes come out right for every pair including the
    non-integer ones, and the filters here are stand-ins regardless -- there is no
    point optimising the quality of a kernel we already know is not the normative
    one.
    """
    s, d = get_format(src), get_format(dst)
    if (s.ver, s.hor) == (d.ver, d.hor):
        return uv
    full = upsample_chroma(uv, luma_hw, mode=mode)
    return subsample_chroma(full, d)


def to_output_format(y: Tensor, uv: Tensor, *, internal="420", output="444",
                     mode: str = "bilinear") -> tuple[Tensor, Tensor]:
    """Reconstructed planes at the internal format -> planes at the output format.

    Returns `(luma, chroma)` rather than one interleaved tensor, because at anything
    other than 4:4:4 they have different spatial sizes and cannot be one tensor.
    """
    return y, convert_chroma_format(uv, y.shape[-2:], internal, output, mode=mode)


def luma_for_secondary(y: Tensor, uv_size: tuple[int, int]) -> Tensor:
    """Downsample luma to the chroma grid, for the encoder's cross-component link.

    §VI-A: cross-component information flows at exactly two points, and this is the
    first -- the secondary analysis transform consumes luma alongside chroma. Box
    average rather than bilinear, to match `subsample_chroma`: the two tensors are
    concatenated and fed to one convolution, so resampling them differently would
    put a systematic phase offset between the channels of a single input.
    """
    h, w = y.shape[-2:]
    th, tw = uv_size
    if (h, w) == (th, tw):
        return y
    ver = max(1, round(h / th))
    hor = max(1, round(w / tw))
    out = F.avg_pool2d(y, kernel_size=(ver, hor), stride=(ver, hor),
                       ceil_mode=True, count_include_pad=False)
    if tuple(out.shape[-2:]) != (th, tw):
        # Non-integer ratios (4:1:1-style, or a padded latent grid) fall back to
        # interpolation. Rare, and better than returning a mismatched shape that
        # would fail deep inside a conv with an unreadable message.
        out = F.interpolate(y, size=(th, tw), mode="area")
    return out


# ---------------------------------------------------------------------------
# colour_transform_idx  (§VI-B, eq. 4 / 5)  and eq. 6
# ---------------------------------------------------------------------------
#: The mandatory conversion at the *end* of decoding, after the post-filters.
#: Indices are the paper's, and index 1 -- not 0 -- is the identity.
TRANSFORM_YCBCR_TO_RGB = 0
TRANSFORM_NONE = 1
TRANSFORM_CUSTOM = 2


def apply_colour_transform(x: Tensor, idx: int = TRANSFORM_YCBCR_TO_RGB, *,
                           matrix: Tensor | None = None,
                           bias: Tensor | None = None) -> Tensor:
    """`colour_transform_idx` 0/1/2. Not clipped -- eq. (6) does that.

    `idx=2` is eq. (5): a 3x3 matrix and 3-vector the *encoder* chooses and signals
    in the picture header, which is how JPEG AI supports colour spaces its models
    were not trained on without adding a transform per space.
    """
    if idx == TRANSFORM_NONE:
        return x
    if idx == TRANSFORM_YCBCR_TO_RGB:
        return ycbcr_to_rgb_bt709(x)
    if idx == TRANSFORM_CUSTOM:
        if matrix is None:
            raise ValueError("colour_transform_idx=2 needs the signalled 3x3 matrix")
        a = torch.as_tensor(matrix, dtype=x.dtype, device=x.device).reshape(3, 3)
        # einsum over the channel axis, so this works for [C,H,W] and [N,C,H,W]
        # alike -- the decoder calls it on both.
        out = torch.einsum("ij,...jhw->...ihw", a, x)
        if bias is not None:
            b = torch.as_tensor(bias, dtype=x.dtype, device=x.device).reshape(3, 1, 1)
            out = out + b
        return out
    raise ValueError(f"colour_transform_idx must be 0, 1 or 2, got {idx}")


def scale_and_clip(x: Tensor, bitdepth: int = 8, *, round_output: bool = True) -> Tensor:
    """eq. (6): `clip(0, 2^bd - 1, x * (2^bd - 1))`.

    `round_output` is on by default because the codec's output is an integer image
    and the alternative -- truncation -- biases every sample down by half a level,
    which is a free ~0.02 dB of PSNR left on the floor. Turn it off only to keep the
    result differentiable.
    """
    if not 1 <= bitdepth <= 16:
        raise ValueError(f"bitdepth must be in 1..16, got {bitdepth}")
    peak = float(2 ** bitdepth - 1)
    out = x * peak
    if round_output:
        out = torch.round(out)
    return out.clamp(0.0, peak)


def decode_output(x: Tensor, *, idx: int = TRANSFORM_YCBCR_TO_RGB,
                  bitdepth: int = 8, matrix=None, bias=None,
                  round_output: bool = True) -> Tensor:
    """The decoder's last two steps in the paper's order: transform, then eq. (6).

    Provided as one function because the order is not interchangeable -- clipping to
    [0, 255] before the colour transform clips *YCbCr*, and Cb/Cr legitimately reach
    outside the luma range, so doing it that way desaturates every saturated colour
    in the picture.
    """
    return scale_and_clip(
        apply_colour_transform(x, idx, matrix=matrix, bias=bias),
        bitdepth, round_output=round_output,
    )


#: The signalled matrix is nine integers at this scale: `matrix = ints / 255.0`.
#: CONFIRMED in the reference software, docs/06 §9.
MATRIX_SCALE = 255.0


def quantise_signalled_matrix(matrix) -> Tensor:
    """A float 3x3 -> the nine integers a picture header would actually carry.

    docs/06 §9: JPEG AI fixes *no* colour coefficients. The picture header carries
    `clr_tr_matrix` as nine 8-bit integers and the decoder inverts them numerically,
    so eq. (4) is one instance of a signalled matrix rather than a constant of the
    standard -- which makes our textbook BT.709 inverse a legal configuration and not
    a deviation.

    **Open question, and the reason this returns signed integers:** docs/06 records
    the entries as "8-bit", and `/255.0` implies an unsigned 0..255 range -- but no
    RGB<->YCbCr matrix in either direction is entirely non-negative, so an unsigned
    range cannot express one. Either the integers are signed, or there is an offset
    the note does not mention. Signed is assumed here because it is the reading under
    which BT.709 is representable at all. Do not treat the exact integers this
    returns as normative.
    """
    m = torch.as_tensor(matrix, dtype=torch.float64).reshape(3, 3)
    return torch.round(m * MATRIX_SCALE)


def invert_signalled_matrix(ints, *, sample_scale: float = 1.0) -> Tensor:
    """The decoder's side: nine integers -> the inverse matrix it will apply.

    Mirrors `inv_matrix = torch.inverse(clr_tr_matrix / 255.0) * 255.0` from the
    reference software. That trailing `* 255.0` is absorbing *their* 0..255 sample
    scale, not part of the matrix inversion -- our tensors are in [0,1], so
    `sample_scale` defaults to 1. Getting that wrong scales every decoded pixel by
    255, which is not subtle, but it is the kind of thing that gets copied verbatim.
    """
    m = torch.as_tensor(ints, dtype=torch.float64).reshape(3, 3) / MATRIX_SCALE
    return (torch.inverse(m) * sample_scale).to(torch.float32)


def bt709_forward_matrix() -> Tensor:
    """The RGB -> YCbCr matrix as a 3x3, for signalling through the header path.

    The same transform as :func:`rgb_to_ycbcr_bt709` minus the +0.5 chroma offset,
    which travels as the bias of eq. (5) rather than in the matrix.
    """
    return torch.tensor([
        [K_R, K_G, K_B],
        [-K_R / CB_SCALE, -K_G / CB_SCALE, (1.0 - K_B) / CB_SCALE],
        [(1.0 - K_R) / CR_SCALE, -K_G / CR_SCALE, -K_B / CR_SCALE],
    ], dtype=torch.float32)


def signalled_matrix_error(matrix=None) -> tuple[float, float]:
    """`(round_trip, deviation_from_the_float_matrix)`, both max abs in [0,1].

    Two different questions, and conflating them gives the wrong answer about what
    8-bit signalling costs:

    * **Round trip** is essentially exact (~1e-7). It *has* to be: the decoder
      inverts the very integers the encoder signalled, so however coarsely those
      integers were quantised, the forward and inverse matrices remain an exact pair.
      This is a real virtue of deriving the inverse numerically rather than shipping
      a second signalled matrix -- the two can never disagree.
    * **Deviation** is the actual cost, ~4e-3: the transform applied is not quite
      BT.709 but the nearest matrix expressible in ninths of 1/255. It is a small
      global colour shift, identical on encoder and decoder, and it is invisible to
      any round-trip test.

    So Phase 4's "< 1e-5" acceptance is reachable through index 2 after all -- it
    just does not mean what it appears to. Passing it says the pair is consistent,
    not that the colour is right.
    """
    m = bt709_forward_matrix() if matrix is None else torch.as_tensor(matrix).float()
    ints = quantise_signalled_matrix(m)
    fwd = (ints / MATRIX_SCALE).to(torch.float32)
    inv = invert_signalled_matrix(ints)

    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    mid = torch.einsum("ij,...jhw->...ihw", fwd, x)
    back = torch.einsum("ij,...jhw->...ihw", inv, mid)
    exact = torch.einsum("ij,...jhw->...ihw", m, x)
    return float((back - x).abs().max()), float((mid - exact).abs().max())



# ---------------------------------------------------------------------------
def describe() -> None:
    print("BT.709 full range")
    print(f"  K_R {K_R}  K_G {K_G}  K_B {K_B}   (sum {K_R + K_G + K_B})")
    print(f"  Cb scale 2(1-K_B) = {CB_SCALE}")
    print(f"  Cr scale 2(1-K_R) = {CR_SCALE}")
    print("\nchroma formats: luma 321x499 (deliberately odd)")
    for name, f in FORMATS.items():
        ch, cw = f.chroma_size(321, 499)
        print(f"  {name}  ver {f.ver} hor {f.hor}  "
              f"c_ver_minus1={f.ver_minus1} c_hor_minus1={f.hor_minus1}  "
              f"-> chroma {ch}x{cw}")

    x = torch.rand(1, 3, 64, 64)
    err = float((ycbcr_to_rgb_bt709(rgb_to_ycbcr_bt709(x)) - x).abs().max())
    print(f"\nRGB -> YCbCr -> RGB max abs error: {err:.3e}  "
          f"(Phase 4 acceptance: < 1e-5)")

    print("\ncolour_transform_idx")
    print(f"  {TRANSFORM_YCBCR_TO_RGB} predefined YCbCr -> RGB (standard BT.709 "
          f"inverse; the paper's eq. 4 has two typos)")
    print(f"  {TRANSFORM_NONE} identity")
    print(f"  {TRANSFORM_CUSTOM} encoder-signalled 3x3 matrix + bias, eq. (5)")

    rt, dev = signalled_matrix_error()
    print(f"\nsignalled-matrix path (docs/06 §9): 8-bit coefficients, decoder inverts")
    print(f"  round trip through a signalled BT.709:  {rt:.3e}  "
          f"(exact by construction --")
    print(f"     the decoder inverts the same integers, so the pair cannot disagree)")
    print(f"  deviation from true BT.709:             {dev:.3e}  "
          f"<- the actual cost of")
    print(f"     8-bit signalling: a small global colour shift no round trip can see")

    print("\nnote: chroma resampling filters are normative in Part 1, which we do "
          "not have.\n      Box down / bilinear up are documented stand-ins, "
          "isolated in two functions.")


if __name__ == "__main__":
    describe()
