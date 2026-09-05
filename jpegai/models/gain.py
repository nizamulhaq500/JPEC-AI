r"""Phase 8: the gain unit, the 3D quality map, and beta displacement.

Nine trained checkpoints gave us nine rate points. JPEG AI gets a *continuous* rate
curve out of **four** models, and this module is the mechanism: one trained vector per
branch, plus one integer in the picture header.

The mechanism, from arXiv:2503.16288 ("Overview of Variable Rate Coding in JPEG AI")
------------------------------------------------------------------------------------
The quality map multiplies the *residual*, not the latent (eqs. 7 and 8):

    r_Y/UV         = y_Y/UV - mu_Y/UV                    (8)   what we already code
    r'_Y/UV[c,i,j] = m_Y/UV[c,i,j] * r_Y/UV[c,i,j]       (7)   what goes on the wire

`m` is the 3D quality map, the same shape as the residual: `(C, H/16, W/16)`. It is
never signalled -- it is *generated* from a channel-wise map `(1,1,C)` extended
spatially, a spatial map `Q (H/16,W/16,1)` extended across channels, or the product of
the two (Fig. 4a/4b/4c). Only the control parameters are coded.

The channel-wise map is a **single trained gain vector** extrapolated in the log
domain by an integer offset (Fig. 5). Cui et al.'s original gain unit holds a *matrix*
of N vectors and interpolates between neighbours with `m_f = m_n^ft * m_n+1^(1-ft)`;
JPEG AI drops to one vector because searching for the bracketing pair costs too much
at encode time. A target `beta_test` becomes that integer:

    d_beta = beta_test / beta_train                       (9)
    D_beta = floor(ln(d_beta) * P_beta / S_sigma)         (10)

with `S_sigma` "the quantization step for the entropy model" and `P_beta = 2**7`,
because `D_beta` carries seven fractional bits. It is a 12-bit signed header field
clamped to `[-1069, 702]` -- asymmetric because performance falls off faster above the
anchor point than below it (Fig. 6). Negative lowers the rate, positive raises it.

Why this costs us almost nothing to add
---------------------------------------
`S_sigma` is not a new constant. `SigmaIndex.log_k` is `(ln 54.82 - ln 0.11)/31 =
0.200365`, and the paper's `S_sigma = 0.2` is that number to one significant figure --
the same quantity, because both are "the entropy model's quantisation step". And
`SigmaIndex.step` is `2**sigma_precision = 128`, which is `P_beta`. So eq. (10) reads,
in our own units, `D_beta = ln(d_beta) expressed in Isigma units`, and the entire gain
unit is an **additive offset on Isigma**:

    o       = gain_vector + D_beta      (Isigma units, seven fractional bits)
    m       = exp(log_k * o / step)     (eq. 7's linear multiplier)
    Isigma' = Isigma + o                (so sigma' = m * sigma, exactly)

which is what makes it correct rather than merely plausible: scaling a residual by `m`
scales its standard deviation by `m`, so the coder must widen its Gaussian by the same
`m`, and on a log-spaced grid adding `o` to the index does precisely that. Nothing in
`GaussianConditional` changes, no CDF table changes, and `SigmaIndex.clamp` already
bounds the shifted index.

We use `log_k`, not a literal 0.2. Both sides of the codec use the same constant
either way, so either is self-consistent; with `log_k` the identity `sigma' = d_beta *
sigma` is exact instead of off by a factor `d_beta**0.0018`.

What a zero gain vector buys
----------------------------
`vector = 0` with `D_beta = 0` gives `m = 1` and `Isigma' = Isigma`: bit-for-bit the
Phase 5 codec. So a gain unit bolts onto an existing checkpoint as a no-op and trains
from there -- which is exactly Table II's stage IV, twelve epochs of the gain vector
alone with everything else frozen. Phase 8 is a fine-tune, not a training run.

The saturation hazard
---------------------
`Isigma + o` is clamped into `[0, 3967]`, identically at both ends of the codec, so a
saturated index is never a *mismatch*. It is worse than that at the top: `r' = m*r`
keeps growing while the coder's sigma stops at 54.82, which produces out-of-range
escapes rather than merely expensive symbols. That failure mode has cost this project
once already, so `GainUnit.saturation` reports the fraction and the round-trip gate
checks it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

#: Table I of arXiv:2503.16288 -- the 17 spatial quality-map scales, indexed by
#: `q_index + 8`. Not a geometric ladder: it is finer around 1.0 and coarser at the
#: ends, so an ROI adjustment near the anchor point is gentle and a large one is
#: reachable in few indices.
Q_SCALE_TABLE: tuple[float, ...] = (
    0.25, 0.3125, 0.375, 0.4375, 0.5,      # -8 .. -4
    0.625, 0.75, 0.875, 1.0, 1.25,         # -3 ..  1
    1.4375, 1.6875, 2.0, 2.4375, 2.875,    #  2 ..  6
    3.375, 4.0,                            #  7 ..  8
)
Q_INDEX_MIN, Q_INDEX_MAX = -8, 8
#: Where scale 1.0 sits, i.e. what turns a signed q index into a table position.
Q_ZERO = -Q_INDEX_MIN

#: `entropy.bdl_clipping_range` in the configs, confirmed against the reference
#: software's `quantization/params.py`. A 12-bit signed field would allow +-2048; the
#: standard is tighter, and asymmetric.
DELTA_BETA_MIN, DELTA_BETA_MAX = -1069, 702


# ---------------------------------------------------------------------------
# eqs. (9) and (10): a rate target as a header field
# ---------------------------------------------------------------------------
def beta_displacement(beta_test: float, beta_train: float, *,
                      log_k: float, step: int, clip: bool = True) -> int:
    """`beta_test` -> the integer `D_beta` that goes in the picture header.

    Pass `SigmaIndex.log_k` as `log_k` and `SigmaIndex.step` as `step`; the paper's
    `S_sigma` and `P_beta` are those two quantities, so the result comes out in the
    same units as `Isigma` and is simply added to it.

    The floor is the paper's and not a convenience. `D_beta` is a header field, so the
    encoder has to commit to an integer, and flooring biases every rate point very
    slightly *low* -- the safe direction when the point of the exercise is to land
    under a rate target.
    """
    if beta_test <= 0 or beta_train <= 0:
        raise ValueError(f"betas must be positive, got {beta_test} / {beta_train}")
    d = math.floor(math.log(beta_test / beta_train) * step / log_k)
    if clip:
        d = max(DELTA_BETA_MIN, min(DELTA_BETA_MAX, d))
    return int(d)


def beta_ratio(delta_beta: int, *, log_k: float, step: int) -> float:
    """The `d_beta` an integer `D_beta` actually achieves -- eq. (10) inverted.

    Worth having separately from `beta_displacement`, because the two do not compose
    to the identity: asking for twice the anchor's beta gets you
    `beta_ratio(beta_displacement(2 * b, b))`, which is 2 only up to one floor step
    (`exp(log_k/step)` = 1.00157, so at most 0.16% low). Reporting the requested ratio
    as though it were achieved is a small lie that accumulates across a rate ladder.
    """
    return math.exp(delta_beta * log_k / step)


def clip_delta_beta(delta_beta: int) -> int:
    """`D_beta` into its normative range, as a named operation rather than a `min`."""
    return int(max(DELTA_BETA_MIN, min(DELTA_BETA_MAX, delta_beta)))


# ---------------------------------------------------------------------------
# The gain unit
# ---------------------------------------------------------------------------
class GainUnit(nn.Module):
    """One branch's channel-wise quality map: a trained vector plus an integer offset.

    The vector lives in `Isigma` units -- `step` of them to one CDF row -- and is
    initialised to **zero**, so a fresh gain unit is the identity and an existing
    checkpoint keeps the exact rate point it was trained at. It is deliberately
    unconstrained: spending more bits on some channels than others is the entire point
    of a channel-wise map, and the combined offset is bounded downstream by
    `SigmaIndex.clamp` rather than here, where a clamp would kill the gradient.

    `delta_beta` is neither a parameter nor a buffer. It is a per-picture header field,
    so it arrives as an argument; storing it on the module would make a model's rate
    depend on hidden state that no checkpoint records and no bitstream carries.

    One gain unit per branch, not one shared: `C_Y = 160` and `C_UV = 96` differ, and
    the paper signals two `D_beta`, one per component, "usually set to the same value".
    Separate units are what make the flexible colour bit allocation expressible.
    """

    def __init__(self, channels: int, *, log_k: float, step: int,
                 scaler_precision: int = 10):
        super().__init__()
        self.channels = int(channels)
        self.log_k, self.step = float(log_k), int(step)
        #: `entropy.scaler_precision`. Extra fractional bits the reference software
        #: keeps on the scaler beyond `Isigma`'s seven, which is where its
        #: `scaled_sigma_precision = 17 = 10 + 7` comes from. Inferred, not confirmed:
        #: only used when `quantise=True`, and only Phase 11's integer path needs it.
        self.scaler_precision = int(scaler_precision)
        self.vector = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        # ln(q_scale) in Isigma units, so the spatial map composes with the vector by
        # addition like everything else here (Fig. 4c is a *product* of maps, and a
        # product of exponentials is a sum of exponents). A buffer rather than a
        # recomputation: it depends on log_k and step, and a `.to(device)` copy has to
        # carry the same numbers rather than re-derive them.
        q = torch.tensor(Q_SCALE_TABLE, dtype=torch.float32)
        self.register_buffer("q_offsets", torch.log(q) * self.step / self.log_k)

    # -- the maps ----------------------------------------------------------
    def offset(self, delta_beta: int | float | Tensor = 0,
               q_index: Tensor | None = None, *, quantise: bool = False) -> Tensor:
        """The combined log-domain offset `o`, broadcastable onto the residual.

        Shape `(1, C, 1, 1)` for a channel-wise map alone -- Fig. 4a, the map extended
        spatially by broadcasting rather than by materialising `(C, H/16, W/16)` floats
        we would only multiply elementwise anyway. Supplying `q_index` gives Fig. 4c,
        `(N, C, H, W)`, the joint map; supplying `q_index` to a unit whose vector is
        still zero gives Fig. 4b, the spatial map alone.

        `q_index` is Table I's integer map, shape `(N, 1, H, W)` on the latent grid, so
        one element per 16x16 image block, values in `[-8, 8]`.

        `quantise=True` rounds `o` onto the reference software's fixed-point grid.
        Off by default because our coder is float throughout; on, it is what Phase 11
        will need for cross-device bit-exactness.
        """
        o = self.vector + _as_tensor(delta_beta, self.vector)
        if q_index is not None:
            o = o + self.q_offsets[_check_q(q_index) + Q_ZERO]
        if quantise:
            grid = float(1 << self.scaler_precision)
            o = torch.round(o * grid) / grid
        return o

    def scale(self, offset: Tensor) -> Tensor:
        """`o -> m`, eq. (7)'s linear multiplier. `exp` of an offset in Isigma units."""
        return torch.exp(self.log_k * offset / self.step)

    def forward(self, delta_beta: int | float | Tensor = 0,
                q_index: Tensor | None = None,
                *, quantise: bool = False) -> tuple[Tensor, Tensor]:
        """`(o, m)`. A `forward` so hooks, `print(model)` and DDP see the gain unit."""
        o = self.offset(delta_beta, q_index, quantise=quantise)
        return o, self.scale(o)

    # -- diagnostics -------------------------------------------------------
    @torch.no_grad()
    def saturation(self, i_sigma: Tensor, offset: Tensor, *,
                   max_index: float) -> dict:
        """What fraction of `Isigma + o` runs off the end of the sigma table.

        Not a correctness check -- both ends of the codec clamp the same way. It is a
        *rate* check, and at the top end an escape check: past `max_index` the coder's
        sigma stops growing while `m*r` does not, so the tail of the residual leaves
        the CDF's support. Anything above a fraction of a percent at `high` means the
        requested `D_beta` is outside what this checkpoint can actually deliver.
        """
        shifted = i_sigma + offset
        n = shifted.numel()
        return {"low": (shifted < 0).sum().item() / n,
                "high": (shifted > max_index).sum().item() / n,
                "min": shifted.min().item(), "max": shifted.max().item()}

    def extra_repr(self) -> str:
        return (f"channels={self.channels}, log_k={self.log_k:.6g}, "
                f"step={self.step}, scaler_precision={self.scaler_precision}")


def _as_tensor(delta_beta: int | float | Tensor, like: Tensor) -> Tensor:
    """`D_beta` as something addable to the gain vector, whatever the caller passed.

    A python int for the ordinary single-picture case, or a tensor of shape `(N,1,1,1)`
    when a batch carries a different `D_beta` per image -- which is how the variable
    rate feature gets *trained*: sampling `D_beta` across the batch is what stops the
    gain vector from overfitting the one rate point the backbone was trained at.
    """
    if isinstance(delta_beta, Tensor):
        d = delta_beta.to(device=like.device, dtype=like.dtype)
        return d.reshape(-1, 1, 1, 1) if d.dim() == 1 else d
    return torch.as_tensor(float(delta_beta), device=like.device, dtype=like.dtype)


def _check_q(q_index: Tensor) -> Tensor:
    """A spatial map as int64 indices, or a clear error. Bounds first, always: an
    out-of-range index would silently wrap into the other end of Table I."""
    if q_index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"q_index must be an integer tensor, got {q_index.dtype}")
    lo, hi = int(q_index.min()), int(q_index.max())
    if lo < Q_INDEX_MIN or hi > Q_INDEX_MAX:
        raise ValueError(f"q_index out of Table I's range [{Q_INDEX_MIN}, "
                         f"{Q_INDEX_MAX}]: got [{lo}, {hi}]")
    return q_index.to(torch.int64)


def q_scales(q_index: Tensor) -> Tensor:
    """Table I as a lookup: integer indices -> float quantisation scales."""
    table = torch.tensor(Q_SCALE_TABLE, dtype=torch.float32, device=q_index.device)
    return table[_check_q(q_index) + Q_ZERO]


def reject_gain(delta_beta, q_index, where: str, why: str) -> None:
    """Refuse a rate request a branch cannot honour, rather than ignore one.

    Branches that predate Phase 8 still accept `delta_beta`/`q_index`, so callers do
    not have to know which branch they are holding -- but accepting an argument and
    dropping it would be the worst outcome available here. The bitstream would decode,
    the picture would look right, and the rate would simply not be the one that was
    asked for, which is invisible until a rate ladder comes out with two identical
    points on it.
    """
    asked = q_index is not None or (delta_beta.any() if isinstance(delta_beta, Tensor)
                                    else bool(delta_beta))
    if asked:
        raise NotImplementedError(
            f"{where} cannot apply a quality map (delta_beta={delta_beta!r}, "
            f"q_index={'set' if q_index is not None else None}). {why}"
        )


# ---------------------------------------------------------------------------
# eqs. (11) and (12): coding the spatial quality map
# ---------------------------------------------------------------------------
# The spatial map is the one part of the quality-map machinery that has to go *in the
# bitstream*: the channel-wise map is derivable from the model weights plus D_beta, but
# Q is a per-picture ROI decision the decoder cannot guess. So it is coded, losslessly,
# with a linear predictor and its residuals in a substream of their own.
#
# The predictor is the classic MED-without-the-gradient-test: the average of left and
# above in the interior, the single available neighbour on the edges, zero at the
# origin. On a map that is piecewise constant -- which is what an ROI mask is -- almost
# every residual is zero, which is why the paper can call the cost "minimal".
#
# `/2` in eq. (11) is written as division and has to be an integer operation for the
# residuals to be integers. Which way it breaks for negative sums is not stated;
# **floor** is ours, and the only requirement is that both ends of the codec agree,
# which they do by both calling these two functions.
def spatial_predict(q_left: Tensor, q_up: Tensor) -> Tensor:
    """eq. (11)'s interior case, factored out so encoder and decoder cannot diverge."""
    return torch.div(q_left + q_up, 2, rounding_mode="floor")


def spatial_residual(q: Tensor) -> Tensor:
    """eqs. (11) and (12): the integer map -> the residuals that go on the wire.

    Vectorised, and legitimately so: the map is coded losslessly, so the reconstructed
    neighbours the decoder predicts from are bit-identical to the original neighbours
    the encoder has in hand. The inverse gets no such shortcut.

    `q` is `(N, 1, H, W)` on the latent grid, integer, values in `[-8, 8]`.
    """
    q = _check_q(q)
    left, up = torch.zeros_like(q), torch.zeros_like(q)
    left[..., :, 1:] = q[..., :, :-1]
    up[..., 1:, :] = q[..., :-1, :]
    qp = spatial_predict(left, up)          # i > 0, j > 0
    qp[..., 0, :] = left[..., 0, :]         # i == 0        -> q[i, j-1]
    qp[..., :, 0] = up[..., :, 0]           # j == 0        -> q[i-1, j]
    qp[..., 0, 0] = 0                       # origin, last so it wins both edges
    return q - qp


def spatial_reconstruct(dq: Tensor) -> Tensor:
    """The decoder's side of eq. (12): residuals -> the integer map.

    Sequential by necessity. `qp[i,j]` reads neighbours that are themselves still being
    reconstructed, so there is no vectorised form -- the same structural reason
    raster-scan context models are slow, on a tensor 256 times smaller than the latent.
    A 512x768 picture makes this 32x48 = 1536 iterations, which is nothing next to one
    convolution.
    """
    dq = dq.to(torch.int64)
    q = torch.zeros_like(dq)
    height, width = dq.shape[-2:]
    for i in range(height):
        for j in range(width):
            if i and j:
                qp = spatial_predict(q[..., i, j - 1], q[..., i - 1, j])
            elif i:
                qp = q[..., i - 1, j]
            elif j:
                qp = q[..., i, j - 1]
            else:
                qp = torch.zeros_like(dq[..., 0, 0])
            q[..., i, j] = dq[..., i, j] + qp
    return q


def spatial_bits(dq: Tensor) -> float:
    """The ideal order-0 cost of a residual map, in bits.

    An estimate, and labelled as one: the real substream is an adaptive coder that
    Phase 12 will write, and this is the entropy of the residual histogram, which is
    what the paper's "minimal amount of bits" claim amounts to. Useful now because it
    lets `runbench` charge the spatial map to the rate instead of quietly omitting it.
    """
    flat = dq.reshape(-1).to(torch.int64)
    if flat.numel() == 0:
        return 0.0
    _, counts = torch.unique(flat, return_counts=True)
    p = counts.to(torch.float64) / flat.numel()
    # `max(0, ...)` so a constant map reports 0.0 rather than `-0.0`: a negative bit
    # count in a rate table is the sort of thing someone spends an hour chasing.
    return max(0.0, float(-(p * torch.log2(p)).sum() * flat.numel()))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # `SigmaIndex` is imported here and not at module scope on purpose: `hyper.py`
    # imports *this* module, so a top-level import would close the cycle.
    from jpegai.models.hyper import SigmaIndex

    si = SigmaIndex()
    print("the constant identity")
    print(f"  SigmaIndex.log_k = {si.log_k:.6f}   paper's S_sigma = 0.2      "
          f"-> {abs(si.log_k - 0.2) / 0.2 * 100:.2f}% apart")
    print(f"  SigmaIndex.step  = {si.step}        paper's P_beta   = 2**7 = 128")
    assert si.step == 128 and abs(si.log_k - 0.2) < 0.002

    gain = GainUnit(160, log_k=si.log_k, step=si.step)
    o0 = gain.offset(0)
    print(f"\nidentity at D_beta=0, untrained vector: |o| max {o0.abs().max():.3e}, "
          f"m == 1 everywhere: {bool(torch.all(gain.scale(o0) == 1.0))}")
    assert torch.all(gain.scale(o0) == 1.0)

    print("\neq. (10) round trip, and what sigma actually does")
    print(f"  {'beta_test/train':>15} {'D_beta':>7} {'achieved':>9} "
          f"{'sigma ratio':>12} {'err':>9}")
    for ratio in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0):
        d = beta_displacement(0.075 * ratio, 0.075, log_k=si.log_k, step=si.step)
        got = beta_ratio(d, log_k=si.log_k, step=si.step)
        # The claim under test: Isigma' = Isigma + D_beta multiplies sigma by d_beta.
        base = torch.tensor([[[[1900.0]]]])
        s_ratio = float(si.sigma(base + d) / si.sigma(base))
        print(f"  {ratio:>15g} {d:>7} {got:>9.5f} {s_ratio:>12.5f} "
              f"{abs(s_ratio - got) / got:>9.1e}")
        assert abs(s_ratio - got) / got < 1e-5

    lo = beta_ratio(DELTA_BETA_MIN, log_k=si.log_k, step=si.step)
    hi = beta_ratio(DELTA_BETA_MAX, log_k=si.log_k, step=si.step)
    print(f"\nD_beta in [{DELTA_BETA_MIN}, {DELTA_BETA_MAX}] spans d_beta "
          f"[{lo:.4f}, {hi:.4f}] = {hi / lo:.1f}x in sigma")
    print(f"  = [{DELTA_BETA_MIN / si.step:+.2f}, {DELTA_BETA_MAX / si.step:+.2f}] of "
          f"{si.levels - 1} CDF rows, so one model reaches "
          f"{(DELTA_BETA_MAX - DELTA_BETA_MIN) / si.step:.1f} rows of quantiser")

    print("\nTable I lookup")
    idx = torch.arange(Q_INDEX_MIN, Q_INDEX_MAX + 1, dtype=torch.int64)
    got = q_scales(idx.reshape(1, 1, 1, -1)).reshape(-1).tolist()
    print("  " + " ".join(f"{v:g}" for v in got))
    assert got == list(Q_SCALE_TABLE)
    # A spatial scale of s must be the same thing as a D_beta that multiplies by s.
    q_off = gain.q_offsets[Q_ZERO + 4]          # index +4 -> scale 2.0
    print(f"  index +4 (scale 2.0) is {float(q_off):.2f} Isigma units = "
          f"{float(q_off) * si.log_k / si.step:.5f} in ln, ln(2) = {math.log(2):.5f}")
    assert abs(float(q_off) * si.log_k / si.step - math.log(2.0)) < 1e-4

    print("\nspatial map round trip, eqs. (11) and (12)")
    torch.manual_seed(0)
    maps = {
        "random": torch.randint(Q_INDEX_MIN, Q_INDEX_MAX + 1, (1, 1, 32, 48)),
        "flat +0": torch.zeros(1, 1, 32, 48, dtype=torch.int64),
        "roi box": torch.zeros(1, 1, 32, 48, dtype=torch.int64),
        "gradient": (torch.arange(48).reshape(1, 1, 1, 48)
                     .expand(1, 1, 32, 48).clone() // 3 - 8),
    }
    maps["roi box"][..., 8:20, 12:30] = 5
    for name, q in maps.items():
        dq = spatial_residual(q)
        back = spatial_reconstruct(dq)
        ok = bool(torch.equal(back, q.to(torch.int64)))
        bits = spatial_bits(dq)
        print(f"  {name:9} nonzero residuals {int((dq != 0).sum()):5}/{dq.numel()}  "
              f"{bits:8.1f} bits ({bits / dq.numel():.3f} b/elem)  "
              f"lossless={'yes' if ok else 'NO'}")
        assert ok, name

    print("\nsaturation reporting (the escape hazard)")
    i_sigma = torch.full((1, 160, 32, 48), 1900.0)
    for d in (0, DELTA_BETA_MIN, DELTA_BETA_MAX, 2500):
        s = gain.saturation(i_sigma, gain.offset(d), max_index=si.max_index)
        print(f"  D_beta {d:>6}: index range [{s['min']:8.1f}, {s['max']:8.1f}]  "
              f"low {s['low']:6.1%}  high {s['high']:6.1%}")

    print("\nall gain-unit invariants hold")
