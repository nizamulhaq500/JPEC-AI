"""Tests for the colour pipeline — Phase 4 item 1.

The plan's two acceptance criteria for this piece are tested directly:

* RGB -> YCbCr -> RGB max abs error < 1e-5 at 4:4:4
* all nine {internal} x {output} chroma format combinations produce correct sizes

Beyond those, three things here are worth more than the usual amount of testing:

**The paper's eq. (4) as printed.** We deliberately did *not* implement it. Two tests
implement it as printed and show it fails the 1e-5 criterion, so the deviation is
recorded as a measurement rather than as a comment somebody may later "fix".

**The ceiling rule.** ⌈H/2⌉ vs ⌊H/2⌋ on an odd-sized image is one row of colour. It
does not crash, it does not look obviously wrong on a thumbnail, and it costs PSNR
that gets attributed to the codec.

**The duplicate forward transforms.** There were three copies of RGB→YCbCr: this
module's derived one, and hand-written literal ones in `jpegai.train.losses` and
`jpegai.eval.metrics`. Phase 4 collapsed the train copy to a re-export — train may
import models — leaving two. `jpegai.eval.metrics` keeps its own and must: eval has
to run against JPEG and AVIF with no torch model loaded, so it cannot import the
model package, and the model package cannot import eval without a cycle. A test
compares the two, so "derived" cannot drift away from "literal" unnoticed, and
another test asserts the train copy is still the *same object* rather than a fresh
divergent copy.
"""

from __future__ import annotations

import pytest
import torch

from jpegai.eval.metrics import rgb_to_ycbcr_bt709 as metrics_fwd
from jpegai.models.colour import (
    CB_SCALE, CR_SCALE, FORMATS, K_B, K_G, K_R, MATRIX_SCALE, TRANSFORM_CUSTOM,
    TRANSFORM_NONE, TRANSFORM_YCBCR_TO_RGB, ChromaFormat, apply_colour_transform,
    bt709_forward_matrix, convert_chroma_format, decode_output, get_format,
    invert_signalled_matrix, luma_for_secondary, merge_planes,
    quantise_signalled_matrix, rgb_to_ycbcr_bt709, scale_and_clip,
    signalled_matrix_error, split_planes, subsample_chroma, to_output_format,
    upsample_chroma, ycbcr_to_rgb_bt709,
)
from jpegai.train.losses import rgb_to_ycbcr_bt709 as losses_fwd

TOL = 1e-5                       # the plan's acceptance threshold
FMT_NAMES = ["444", "422", "420"]


# ---------------------------------------------------------------------------
# acceptance criterion 1: exact round trip
# ---------------------------------------------------------------------------
def test_round_trip_beats_the_acceptance_threshold():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 33, 47)
    err = (ycbcr_to_rgb_bt709(rgb_to_ycbcr_bt709(x)) - x).abs().max()
    assert err < TOL, err
    assert err < 1e-6, f"expected float32 noise (~1e-7), got {err}"


def test_round_trip_at_the_corners_of_the_cube():
    """Random pixels never land on pure primaries, which is where a sign error hides."""
    corners = torch.tensor([[0., 0., 0.], [1., 1., 1.], [1., 0., 0.],
                            [0., 1., 0.], [0., 0., 1.], [1., 1., 0.],
                            [0., 1., 1.], [1., 0., 1.], [0.5, 0.5, 0.5]])
    x = corners.T.reshape(1, 3, 3, 3)
    assert (ycbcr_to_rgb_bt709(rgb_to_ycbcr_bt709(x)) - x).abs().max() < TOL


def test_grey_has_no_chroma_and_luma_equal_to_the_grey_level():
    for level in (0.0, 0.25, 0.5, 1.0):
        ycc = rgb_to_ycbcr_bt709(torch.full((1, 3, 4, 4), level))
        assert torch.allclose(ycc[:, 0], torch.full((1, 4, 4), level), atol=1e-6)
        assert torch.allclose(ycc[:, 1:], torch.full((1, 2, 4, 4), 0.5), atol=1e-6)


def test_luma_of_the_primaries_is_the_bt709_coefficients():
    """Guards against a BT.601 swap, which round-trips perfectly and is still wrong."""
    prim = torch.eye(3).T.reshape(1, 3, 1, 3)          # R, G, B in three pixels
    y = rgb_to_ycbcr_bt709(prim)[0, 0, 0]
    assert torch.allclose(y, torch.tensor([K_R, K_G, K_B]), atol=1e-6)


def test_the_derived_chroma_scales_match_the_papers_printed_values():
    assert CB_SCALE == pytest.approx(1.8556, abs=1e-12)
    assert CR_SCALE == pytest.approx(1.5748, abs=1e-12)
    assert K_R + K_G + K_B == pytest.approx(1.0, abs=1e-12)


def test_the_remaining_literal_copy_of_the_forward_transform_agrees():
    """`jpegai.eval.metrics` carries its own, for a documented dependency reason."""
    torch.manual_seed(1)
    x = torch.rand(2, 3, 8, 8)
    assert torch.allclose(rgb_to_ycbcr_bt709(x), metrics_fwd(x), atol=1e-7)


def test_the_training_loss_uses_this_module_rather_than_its_own_copy():
    """Identity, not allclose: a re-export cannot drift, a copy can."""
    assert losses_fwd is rgb_to_ycbcr_bt709


def test_forward_accepts_unbatched_input():
    """The decoder calls these on [C,H,W]; the other two copies cannot."""
    x = torch.rand(3, 8, 8)
    assert rgb_to_ycbcr_bt709(x).shape == (3, 8, 8)
    assert (ycbcr_to_rgb_bt709(rgb_to_ycbcr_bt709(x)) - x).abs().max() < TOL


def test_wrong_channel_count_is_rejected():
    for fn in (rgb_to_ycbcr_bt709, ycbcr_to_rgb_bt709):
        with pytest.raises(ValueError, match="3 channels"):
            fn(torch.rand(1, 1, 8, 8))


def test_inverse_does_not_clip_out_of_gamut_values():
    """Clipping belongs in eq. (6). Doing it here would hide a broken synthesis."""
    ycc = torch.tensor([0.5, 1.0, 1.0]).reshape(1, 3, 1, 1)   # far outside RGB
    rgb = ycbcr_to_rgb_bt709(ycc)
    assert rgb.max() > 1.0


# ---------------------------------------------------------------------------
# the paper's eq. (4) as printed
# ---------------------------------------------------------------------------
def test_eq4_as_printed_uses_cr_for_blue_and_fails_the_threshold():
    """Both printed lines index x̂_UV[1], so Cr drives blue as well as red."""
    torch.manual_seed(2)
    x = torch.rand(1, 3, 16, 16)
    ycc = rgb_to_ycbcr_bt709(x)
    y, cb, cr = ycc[:, 0:1], ycc[:, 1:2], ycc[:, 2:3]

    r = y + CR_SCALE * (cr - 0.5)
    b = y + CB_SCALE * (cr - 0.5)                    # the typo: cr, not cb
    g = (y - K_R * r - K_B * b) / K_G
    err = (torch.cat([r, g, b], 1) - x).abs().max()

    assert err > 0.01, "the typo must be detectable, or this test proves nothing"
    assert err > TOL


def test_eq4s_second_typo_alone_also_fails_the_threshold():
    """0.07222 for 0.0722 looks harmless. On saturated blue it is 2.8e-5 of green."""
    x = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3, 1, 1)
    ycc = rgb_to_ycbcr_bt709(x)
    y, cb, cr = ycc[:, 0:1], ycc[:, 1:2], ycc[:, 2:3]

    r = y + CR_SCALE * (cr - 0.5)
    b = y + CB_SCALE * (cb - 0.5)
    g = (y - K_R * r - 0.07222 * b) / K_G            # the typo: 0.07222
    err = (torch.cat([r, g, b], 1) - x).abs().max()

    assert err > TOL, f"{err} — expected the 5th digit to matter at this tolerance"
    assert err < 1e-4                                 # but only just


# ---------------------------------------------------------------------------
# chroma formats: sizes, ceilings, acceptance criterion 2
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,ver,hor", [("444", 1, 1), ("422", 1, 2), ("420", 2, 2)])
def test_format_factors_and_minus_one_syntax(name, ver, hor):
    f = FORMATS[name]
    assert (f.ver, f.hor) == (ver, hor)
    assert (f.ver_minus1, f.hor_minus1) == (ver - 1, hor - 1)
    assert f.is_444 == (name == "444")


def test_chroma_size_uses_a_ceiling_on_odd_dimensions():
    assert FORMATS["420"].chroma_size(321, 499) == (161, 250)
    assert FORMATS["422"].chroma_size(321, 499) == (321, 250)
    assert FORMATS["444"].chroma_size(321, 499) == (321, 499)
    # the floor answers, which must not appear
    assert FORMATS["420"].chroma_size(321, 499) != (160, 249)


@pytest.mark.parametrize("h,w", [(32, 32), (33, 32), (32, 33), (33, 47), (1, 1)])
@pytest.mark.parametrize("name", FMT_NAMES)
def test_subsample_produces_exactly_the_declared_size(name, h, w):
    f = FORMATS[name]
    out = subsample_chroma(torch.rand(1, 2, h, w), f)
    assert tuple(out.shape[-2:]) == f.chroma_size(h, w)


@pytest.mark.parametrize("internal", FMT_NAMES)
@pytest.mark.parametrize("output", FMT_NAMES)
def test_all_nine_internal_by_output_combinations_are_correctly_sized(internal, output):
    """Acceptance criterion 2, on a deliberately odd image."""
    h, w = 33, 47
    x = rgb_to_ycbcr_bt709(torch.rand(1, 3, h, w))
    y, uv = split_planes(x, internal)
    assert tuple(uv.shape[-2:]) == FORMATS[internal].chroma_size(h, w)

    y_out, uv_out = to_output_format(y, uv, internal=internal, output=output)
    assert tuple(y_out.shape[-2:]) == (h, w)
    assert tuple(uv_out.shape[-2:]) == FORMATS[output].chroma_size(h, w)
    assert uv_out.shape[-3] == 2
    assert torch.isfinite(uv_out).all()


def test_a_format_conversion_to_itself_is_the_identity_not_a_resample():
    uv = torch.rand(1, 2, 16, 24)
    out = convert_chroma_format(uv, (32, 48), "420", "420")
    assert out is uv, "same-format conversion must not lose a generation of filtering"


def test_merge_restores_the_original_luma_size_on_odd_images():
    """Chroma 161x250 scaled by 2 is 322x500, not the 321x499 it came from."""
    x = rgb_to_ycbcr_bt709(torch.rand(1, 3, 321, 499))
    y, uv = split_planes(x, "420")
    assert tuple(merge_planes(y, uv).shape) == (1, 3, 321, 499)


def test_444_split_and_merge_is_lossless():
    x = rgb_to_ycbcr_bt709(torch.rand(1, 3, 16, 16))
    y, uv = split_planes(x, "444")
    assert torch.equal(merge_planes(y, uv), x)
    rgb = ycbcr_to_rgb_bt709(merge_planes(y, uv))
    assert (rgb - ycbcr_to_rgb_bt709(x)).abs().max() < TOL


def test_box_average_is_exact_and_does_not_average_in_zeros_at_the_edge():
    """A constant plane of odd width must stay constant, including the last column."""
    uv = torch.full((1, 2, 5, 5), 0.75)
    out = subsample_chroma(uv, "420")
    assert tuple(out.shape[-2:]) == (3, 3)
    assert torch.allclose(out, torch.full_like(out, 0.75), atol=1e-6), \
        "edge window averaged in padding, which darkens the last row and column"


def test_box_average_is_the_mean_of_its_window():
    uv = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    out = subsample_chroma(uv, "420")
    assert out[0, 0, 0, 0] == pytest.approx((0 + 1 + 4 + 5) / 4)
    assert out[0, 0, 1, 1] == pytest.approx((10 + 11 + 14 + 15) / 4)


def test_upsample_takes_a_size_not_a_scale_factor():
    uv = torch.rand(1, 2, 161, 250)
    assert tuple(upsample_chroma(uv, (321, 499)).shape[-2:]) == (321, 499)
    assert upsample_chroma(uv, (161, 250)) is uv          # already there, no filtering


def test_get_format_accepts_names_pairs_and_instances():
    assert get_format("4:2:0") is FORMATS["420"]
    assert get_format("420") is FORMATS["420"]
    assert get_format((1, 1)) is FORMATS["420"]           # c_ver_minus1, c_hor_minus1
    assert get_format((0, 1)) is FORMATS["422"]
    assert get_format((0, 0)) is FORMATS["444"]
    f = ChromaFormat("custom", 4, 1)
    assert get_format(f) is f
    with pytest.raises(ValueError, match="unknown chroma format"):
        get_format("411")
    with pytest.raises(TypeError):
        get_format(2.0)


# ---------------------------------------------------------------------------
# the encoder's cross-component link
# ---------------------------------------------------------------------------
def test_luma_for_secondary_lands_on_the_chroma_grid():
    for h, w in [(32, 32), (33, 47), (321, 499)]:
        y = torch.rand(1, 1, h, w)
        for name in FMT_NAMES:
            target = FORMATS[name].chroma_size(h, w)
            assert tuple(luma_for_secondary(y, target).shape[-2:]) == target


def test_luma_for_secondary_is_a_box_average_matching_subsample_chroma():
    """Both feed one convolution; different filters would phase-shift the channels."""
    plane = torch.rand(1, 1, 32, 32)
    ours = luma_for_secondary(plane, (16, 16))
    same = subsample_chroma(plane, "420")
    assert torch.allclose(ours, same, atol=1e-6)


def test_luma_for_secondary_at_444_is_a_no_op():
    y = torch.rand(1, 1, 17, 19)
    assert luma_for_secondary(y, (17, 19)) is y


# ---------------------------------------------------------------------------
# colour_transform_idx  (eq. 4 / 5)
# ---------------------------------------------------------------------------
def test_transform_index_1_is_the_identity_not_index_0():
    """Easy to assume 0 means 'none'. In JPEG AI, 0 is the YCbCr->RGB conversion."""
    x = torch.rand(1, 3, 8, 8)
    assert torch.equal(apply_colour_transform(x, TRANSFORM_NONE), x)
    assert not torch.equal(apply_colour_transform(x, TRANSFORM_YCBCR_TO_RGB), x)


def test_transform_index_0_is_the_bt709_inverse():
    x = torch.rand(1, 3, 8, 8)
    assert torch.equal(apply_colour_transform(x, TRANSFORM_YCBCR_TO_RGB),
                       ycbcr_to_rgb_bt709(x))


def test_a_signalled_matrix_can_reproduce_the_builtin_transform():
    """eq. (5) with the BT.709 inverse coefficients must equal index 0 exactly.

    This is the check that the einsum indexes the channel axis the way the matrix
    convention says it does -- a transposed matrix also runs, silently.
    """
    a = torch.tensor([
        [1.0, 0.0, CR_SCALE],
        [1.0, -K_B * CB_SCALE / K_G, -K_R * CR_SCALE / K_G],
        [1.0, CB_SCALE, 0.0],
    ])
    b = torch.tensor([-0.5 * CR_SCALE,
                      0.5 * (K_B * CB_SCALE + K_R * CR_SCALE) / K_G,
                      -0.5 * CB_SCALE])
    x = torch.rand(1, 3, 8, 8)
    got = apply_colour_transform(x, TRANSFORM_CUSTOM, matrix=a, bias=b)
    assert (got - ycbcr_to_rgb_bt709(x)).abs().max() < TOL


def test_a_signalled_matrix_works_unbatched_and_bias_is_optional():
    a = torch.eye(3) * 2.0
    x = torch.rand(3, 5, 7)
    assert torch.allclose(apply_colour_transform(x, TRANSFORM_CUSTOM, matrix=a), x * 2)
    out = apply_colour_transform(x, TRANSFORM_CUSTOM, matrix=a, bias=[1., 0., 0.])
    assert torch.allclose(out[0], x[0] * 2 + 1)
    assert torch.allclose(out[1], x[1] * 2)


def test_custom_transform_without_a_matrix_is_an_error_not_a_silent_identity():
    with pytest.raises(ValueError, match="3x3 matrix"):
        apply_colour_transform(torch.rand(1, 3, 4, 4), TRANSFORM_CUSTOM)


def test_unknown_transform_index_is_rejected():
    with pytest.raises(ValueError, match="must be 0, 1 or 2"):
        apply_colour_transform(torch.rand(1, 3, 4, 4), 3)


# ---------------------------------------------------------------------------
# the signalled matrix (docs/06 §9)
# ---------------------------------------------------------------------------
def test_the_signalled_matrix_round_trips_however_coarsely_it_is_quantised():
    """The decoder inverts the encoder's own integers, so the pair cannot disagree.

    This is the property that makes numerical inversion the right design: there is no
    second signalled matrix to drift out of sync with the first. Checked at an absurd
    quantisation to show the round trip does not depend on precision at all.
    """
    m = bt709_forward_matrix()
    for scale in (MATRIX_SCALE, 8.0):          # 8-bit, then a deliberately awful 3-bit
        ints = torch.round(m.double() * scale)
        fwd = (ints / scale).float()
        inv = torch.inverse(ints / scale).float()
        x = torch.rand(1, 3, 8, 8)
        mid = torch.einsum("ij,...jhw->...ihw", fwd, x)
        back = torch.einsum("ij,...jhw->...ihw", inv, mid)
        assert (back - x).abs().max() < 1e-4, scale


def test_eight_bit_signalling_costs_colour_accuracy_not_consistency():
    """The real cost is a global shift of about 1/255, invisible to a round trip."""
    round_trip, deviation = signalled_matrix_error()
    assert round_trip < TOL
    assert deviation > TOL, "if this passed, the quantisation is not being applied"
    assert deviation == pytest.approx(1.0 / 255.0, rel=0.5)


def test_quantise_signalled_matrix_yields_integers_in_the_8_bit_range():
    ints = quantise_signalled_matrix(bt709_forward_matrix())
    assert ints.shape == (3, 3)
    assert torch.equal(ints, torch.round(ints)), "must be integer-valued"
    assert ints.abs().max() <= 255, ints
    assert (ints < 0).any(), "a colour matrix has negative entries; see the docstring"


def test_invert_signalled_matrix_matches_the_reference_expression():
    """`torch.inverse(m / 255.0) * 255.0` -- with the 255 being *their* sample scale."""
    ints = quantise_signalled_matrix(bt709_forward_matrix())
    ours = invert_signalled_matrix(ints)
    theirs = torch.inverse(ints.double() / MATRIX_SCALE).float()
    assert torch.allclose(ours, theirs, atol=1e-5)
    scaled = invert_signalled_matrix(ints, sample_scale=255.0)
    assert torch.allclose(scaled, theirs * 255.0, atol=1e-3)


def test_the_forward_matrix_agrees_with_the_hand_written_transform():
    """`bt709_forward_matrix` and `rgb_to_ycbcr_bt709` must be the same transform."""
    x = torch.rand(1, 3, 8, 8)
    viamatrix = torch.einsum("ij,...jhw->...ihw", bt709_forward_matrix(), x)
    viamatrix = viamatrix + torch.tensor([0., 0.5, 0.5]).reshape(1, 3, 1, 1)
    assert torch.allclose(viamatrix, rgb_to_ycbcr_bt709(x), atol=1e-6)


def test_identity_is_the_default_signalled_matrix():
    """docs/06: `clr_tr_matrix` defaults to the identity, i.e. no colour change."""
    ints = quantise_signalled_matrix(torch.eye(3))
    assert torch.equal(ints, torch.eye(3).double() * 255.0)
    assert torch.allclose(invert_signalled_matrix(ints), torch.eye(3), atol=1e-5)


# ---------------------------------------------------------------------------
# eq. (6)
# ---------------------------------------------------------------------------
def test_scale_and_clip_maps_the_unit_interval_onto_the_sample_range():
    x = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 1, 3)
    assert torch.equal(scale_and_clip(x, 8), torch.tensor([0., 128., 255.]).reshape(1, 1, 1, 3))
    assert scale_and_clip(x, 10).max() == 1023.0
    assert scale_and_clip(x, 12).max() == 4095.0


def test_scale_and_clip_clips_both_ends():
    x = torch.tensor([-0.4, 1.7]).reshape(1, 1, 1, 2)
    out = scale_and_clip(x, 8)
    assert out.min() == 0.0 and out.max() == 255.0


def test_rounding_is_on_by_default_and_can_be_turned_off_for_gradients():
    x = torch.tensor([[[[0.501]]]], requires_grad=True)
    assert scale_and_clip(x, 8).item() == 128.0
    soft = scale_and_clip(x, 8, round_output=False)
    assert soft.item() == pytest.approx(0.501 * 255)
    soft.backward()
    assert x.grad is not None and x.grad.item() == pytest.approx(255.0)


def test_bad_bitdepth_is_rejected():
    for bd in (0, 17, -8):
        with pytest.raises(ValueError, match="bitdepth"):
            scale_and_clip(torch.rand(1, 1, 2, 2), bd)


def test_in_gamut_rgb_never_leaves_the_unit_interval_in_ycbcr():
    """Why the chroma scales are 2(1-K): they make the bound exactly tight.

    Pure blue gives Cb = 1.0 and pure red Cr = 1.0, exactly -- so no *source* image
    can produce chroma outside [0,1]. Overshoot is always the reconstruction's.
    """
    torch.manual_seed(4)
    corners = torch.tensor([[0., 0., 0.], [1., 1., 1.], [1., 0., 0.], [0., 1., 0.],
                            [0., 0., 1.], [1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])
    for x in (torch.rand(1, 3, 32, 32), corners.T.reshape(1, 3, 1, 8)):
        ycc = rgb_to_ycbcr_bt709(x)
        assert ycc.min() >= -1e-6 and ycc.max() <= 1.0 + 1e-6
    blue = rgb_to_ycbcr_bt709(torch.tensor([0., 0., 1.]).reshape(1, 3, 1, 1))
    red = rgb_to_ycbcr_bt709(torch.tensor([1., 0., 0.]).reshape(1, 3, 1, 1))
    assert blue[0, 1, 0, 0] == pytest.approx(1.0, abs=1e-6)     # Cb
    assert red[0, 2, 0, 0] == pytest.approx(1.0, abs=1e-6)      # Cr


def test_decode_output_transforms_before_clipping():
    """Clipping YCbCr first would desaturate every saturated colour in the picture.

    The overshoot below is not a contrived input: no source image produces chroma
    outside [0,1] (see the test above), but a synthesis transform ringing at a
    saturated edge does it constantly, and that is precisely when the order of these
    two operations becomes visible.

    Clipping Cr to 1.0 first does not just cap the red -- it changes the green the
    inverse computes from it, mixing green into what should be a pure red overshoot.
    """
    ycc = torch.tensor([0.5, 0.5, 1.4]).reshape(1, 3, 1, 1)     # strong red overshoot

    right = decode_output(ycc, bitdepth=8)
    wrong = scale_and_clip(ycbcr_to_rgb_bt709(ycc.clamp(0.0, 1.0)), 8)

    assert right[0, 0] == 255 and wrong[0, 0] == 255             # both cap the red
    assert right[0, 1] < wrong[0, 1] - 8, \
        f"clip-first must add green: {right[0, 1].item()} vs {wrong[0, 1].item()}"
    assert torch.equal(right[0, 2], wrong[0, 2])                # blue is untouched


def test_decode_output_round_trips_an_8_bit_image_exactly():
    """The whole pipeline, on real integer samples: encode-side in, decode-side out."""
    torch.manual_seed(3)
    src = torch.randint(0, 256, (1, 3, 16, 16)).float()
    ycc = rgb_to_ycbcr_bt709(src / 255.0)
    out = decode_output(ycc, idx=TRANSFORM_YCBCR_TO_RGB, bitdepth=8)
    assert torch.equal(out, src), (out - src).abs().max()
