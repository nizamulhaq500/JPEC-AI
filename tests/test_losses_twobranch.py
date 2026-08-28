"""Phase 4 item 7 — the 6:1:1 Y:U:V loss weighting, against the two-branch model.

The weighting was implemented in Phase 2 and has been sitting there unverified,
because a single-branch RGB model cannot really test it: with one shared set of
parameters producing all three planes, "more gradient on luma" has no separately
observable consequence. The two-branch model does have two disjoint parameter
sets, so the claim becomes falsifiable — and that is what most of this file does:
compare the gradient norm reaching `g_a_y`/`g_s_y` against `g_a_uv`/`g_s_uv`
under 6:1:1 versus 1:1:1.

A weighting that is merely *present in the config* and a weighting that actually
reaches the secondary branch's parameters are different things, and the difference
is invisible in the loss value.
"""

from __future__ import annotations

import pytest
import torch

from jpegai.models.twobranch import TwoBranchCodec
from jpegai.train.losses import MSE_SCALE, RateDistortionLoss


def _model(fmt="420"):
    torch.manual_seed(0)
    return TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                          chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                          synthesis_width=(24, 16, 16, 16), internal_format=fmt)


def _grad_norms(model, weights, x):
    """Total gradient norm on the primary and secondary branch parameters."""
    model.zero_grad(set_to_none=True)
    crit = RateDistortionLoss(beta=0.01, weights=weights)
    crit(model(x), x)["loss"].backward()

    def norm(*mods):
        return sum(p.grad.pow(2).sum().item()
                   for m in mods for p in m.parameters() if p.grad is not None) ** 0.5

    return norm(model.g_a_y, model.g_s_y), norm(model.g_a_uv, model.g_s_uv)


# ---------------------------------------------------------------------------
# the weighting reaches the branches
# ---------------------------------------------------------------------------
def test_luma_weighting_shifts_gradient_from_the_secondary_to_the_primary_branch():
    """The whole point of item 7. Neutral weights are the control."""
    x = torch.rand(2, 3, 128, 128)
    even_y, even_uv = _grad_norms(_model(), {"y": 1.0, "u": 1.0, "v": 1.0}, x)
    six_y, six_uv = _grad_norms(_model(), {"y": 6.0, "u": 1.0, "v": 1.0}, x)

    assert six_uv < even_uv, "6:1:1 must reduce the secondary branch's gradient"
    assert six_y > even_y, "...and increase the primary branch's"
    # The ratio moves by more than noise: 6:1:1 normalised is (2.25, 0.375,
    # 0.375) against (1, 1, 1), so chroma is down 2.7x and luma up 2.25x.
    assert (six_y / six_uv) > 2.0 * (even_y / even_uv)


def test_the_secondary_branch_still_gets_a_gradient():
    """6:1:1 down-weights chroma; it must not silence it, or the branch is dead."""
    y, uv = _grad_norms(_model(), {"y": 6.0, "u": 1.0, "v": 1.0},
                        torch.rand(2, 3, 128, 128))
    assert uv > 0.0 and torch.isfinite(torch.tensor(uv))


def test_the_luma_link_carries_gradient_into_the_primary_analysis():
    """The encoder-side cross-component link (item 4) must be differentiable.

    `luma_for_secondary` box-averages `x_Y`, which is a function of the *input*,
    not of `g_a_y` -- so this is really about eq. (3): the concatenated luma
    latent means a chroma-only distortion still reaches the primary branch.
    """
    m, x = _model(), torch.rand(2, 3, 128, 128)
    out = m(x)
    # A distortion that exists only in chroma.
    (out["planes"]["chroma"] ** 2).mean().backward()
    g = sum(p.grad.abs().sum().item() for p in m.g_a_y.parameters()
            if p.grad is not None)
    assert g > 0, "eq. (3) must let chroma error reach the primary analysis"


# ---------------------------------------------------------------------------
# beta must keep meaning the same thing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("weights", [
    {"y": 1.0, "u": 1.0, "v": 1.0},
    {"y": 6.0, "u": 1.0, "v": 1.0},
    {"y": 8.0, "u": 1.0, "v": 1.0},
])
def test_the_weights_are_normalised_so_beta_means_one_thing(weights):
    """Otherwise a weight ablation is secretly a rate-point ablation too."""
    crit = RateDistortionLoss(weights=weights)
    assert crit.plane_weights.sum().item() == pytest.approx(3.0)


def test_a_uniform_rgb_offset_is_a_pure_luma_error():
    """Worth pinning, because it is not obvious and it makes the next test exact.

    Adding the same constant to R, G and B leaves chroma *exactly* unchanged: Y
    moves by that constant because the coefficients sum to 1, and Cb = (B - Y)/k
    then sees the offset cancel. So the classic "add a small offset" test image is
    a luma-only distortion, and a 6:1:1 loss scores it 2.25x higher than a 1:1:1
    loss -- 2.25 being the normalised luma weight, 6 * 3/8, not 6.
    """
    x = torch.rand(2, 3, 64, 64) * 0.9        # headroom, so +0.02 never clips
    x_hat = x + 0.02
    even, six = ({"y": 1.0, "u": 1.0, "v": 1.0}, {"y": 6.0, "u": 1.0, "v": 1.0})

    _, _, parts = RateDistortionLoss(weights=even).distortion(x, x_hat)
    assert parts["u"].item() == pytest.approx(0.0, abs=1e-6)
    assert parts["v"].item() == pytest.approx(0.0, abs=1e-6)

    d_even = RateDistortionLoss(weights=even).distortion(x, x_hat)[0]
    d_six = RateDistortionLoss(weights=six).distortion(x, x_hat)[0]
    assert (d_six / d_even).item() == pytest.approx(6.0 * 3.0 / 8.0, rel=1e-4)


def test_an_equal_error_in_every_plane_gives_the_same_distortion_whatever_the_weights():
    """The actual normalisation guarantee, and it is exact rather than approximate.

    D = sum(w_i e_i^2) / sum(w_i), so with e equal in all three planes D = e^2 for
    any weighting. That is what keeps a weight ablation from also being a rate
    ablation. Built in YCbCr and converted back, because -- see above -- an equal
    error in RGB is *not* an equal error per plane.
    """
    from jpegai.models.colour import rgb_to_ycbcr_bt709, ycbcr_to_rgb_bt709

    x = torch.rand(2, 3, 64, 64)
    x_hat = ycbcr_to_rgb_bt709(rgb_to_ycbcr_bt709(x) + 0.02)
    ds = [RateDistortionLoss(weights=w).distortion(x, x_hat)[0].item()
          for w in ({"y": 1.0, "u": 1.0, "v": 1.0}, {"y": 6.0, "u": 1.0, "v": 1.0},
                    {"y": 8.0, "u": 3.0, "v": 1.0})]
    for d in ds:
        assert d == pytest.approx(0.02 ** 2 * MSE_SCALE, rel=1e-3)


# ---------------------------------------------------------------------------
# reporting: chroma quality and the chroma rate share
# ---------------------------------------------------------------------------
def test_per_plane_psnr_is_reported():
    """Phase 4's own pitfall list: "don't forget chroma PSNR"."""
    m = _model()
    x = torch.rand(2, 3, 128, 128)
    r = RateDistortionLoss(beta=0.01)(m(x), x)
    for k in ("psnr", "psnr_y", "psnr_u", "psnr_v"):
        assert k in r and torch.isfinite(r[k]), k
    # Untrained, so no quality claim is possible -- but chroma is subsampled and
    # then bilinearly upsampled, which luma never is, so the planes cannot be
    # numerically identical. Equality here would mean the planes got mixed up.
    assert r["psnr_y"] != r["psnr_u"]


def test_the_branch_split_accounts_for_the_whole_rate():
    m = _model()
    x = torch.rand(2, 3, 128, 128)
    r = RateDistortionLoss(beta=0.01)(m(x), x)
    assert r["bpp_luma"] + r["bpp_chroma"] == pytest.approx(r["bpp"].item(), rel=1e-6)
    assert 0.0 < r["chroma_share"] < 1.0
    # The four streams must each land in exactly one group.
    assert r["bpp_luma"] == pytest.approx((r["bpp_y"] + r["bpp_z"]).item(), rel=1e-6)
    assert r["bpp_chroma"] == pytest.approx(
        (r["bpp_y_uv"] + r["bpp_z_uv"]).item(), rel=1e-6)


def test_the_branch_split_is_absent_for_a_single_branch_model():
    """Callers must need no special case for the Phase 3 model."""
    assert RateDistortionLoss.branch_split({"y": torch.tensor(1.0)}) == {}


def test_the_loss_is_finite_and_the_rate_counts_four_streams():
    m = _model()
    x = torch.rand(2, 3, 128, 128)
    out = m(x)
    r = RateDistortionLoss(beta=0.01)(out, x)
    assert torch.isfinite(r["loss"]) and r["loss"].requires_grad
    assert {"bpp_y", "bpp_z", "bpp_y_uv", "bpp_z_uv"} <= set(r)


@pytest.mark.parametrize("fmt", ["444", "422", "420"])
def test_the_loss_works_at_every_internal_chroma_format(fmt):
    m = _model(fmt)
    x = torch.rand(1, 3, 128, 192)
    r = RateDistortionLoss(beta=0.01)(m(x), x)
    assert torch.isfinite(r["loss"]) and torch.isfinite(r["psnr_u"])


def test_rgb_colour_space_bypasses_the_weighting_entirely():
    """The apples-to-apples mode for comparing against published compressai runs."""
    x = torch.rand(2, 3, 64, 64)
    x_hat = (x + 0.05).clamp(0, 1)
    crit = RateDistortionLoss(weights={"y": 6.0, "u": 1.0, "v": 1.0},
                              colour_space="rgb")
    d, plain, parts = crit.distortion(x, x_hat)
    assert d is plain and parts == {}
    assert d.item() == pytest.approx(
        ((x_hat - x) ** 2).mean().item() * MSE_SCALE, rel=1e-6)
