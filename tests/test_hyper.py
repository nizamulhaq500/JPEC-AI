"""Phase 5: split hyper decoders and the integer sigma index.

Most of this file exists to pin **one specific hazard**. `SigmaIndex.table_row` and
`GaussianConditional.build_indexes` implement the same rule two different ways, they
agree on 3957 of the 3968 valid indices, and they differ on 11 -- one float32 ULP at
the exact grid points. An encoder using one and a decoder using the other produces a
bitstream that decodes to the wrong latent for 0.28% of its symbols, with both sides
reporting success. So:

* the 11 exceptions are enumerated by index, not counted, so a change to either path
  fails a test rather than corrupting a bitstream;
* the mismatch is *demonstrated* end to end rather than argued, in
  `test_mixing_the_two_index_paths_corrupts_the_latent`;
* the round-up rounding rule is tested against the thing that proves it -- that
  `max_index` reaches the last CDF row, which round-down cannot do.

The other claims here are the ones Phase 5's acceptance criteria are written in:
`predict` sees `zhat` and nothing else, and the scale decoder costs under 5% of
decoder MACs.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from jpegai.config import load_config
from jpegai.models import build_any_model
from jpegai.models.entropy import GaussianConditional, build_scale_table
from jpegai.models.hyper import (
    FusedHyperDecoder, HyperDecoder, HyperEncoder, HyperScaleDecoder, SigmaIndex,
    SplitHyperBranch,
)
from jpegai.utils import macs_breakdown

#: The exact indices where the integer and float paths disagree. Every one is a
#: multiple of 128, i.e. an `Iσ` sitting exactly on a grid point, which is where a
#: single ULP decides the comparison. Recorded as data because the *value* of this
#: test is that the list cannot drift silently.
ULP_DISAGREEMENTS = [256, 1152, 1280, 1536, 1664, 2176, 2304, 2560, 3200, 3328, 3456]


def _index():
    return SigmaIndex()


def _gc(si):
    return GaussianConditional(
        build_scale_table(si.minimum, si.maximum, si.levels), scale_bound=si.minimum)


def _branch(fused=False, latent=32):
    torch.manual_seed(0)
    si = _index()
    return SplitHyperBranch(latent, latent, sigma_index=si, fused=fused).eval(), si


# -- the rounding rule ----------------------------------------------------------
def test_the_maximum_index_reaches_the_last_cdf_row():
    """The whole argument for round-up, as an assertion.

    `sigma_idx_max_value = 3967` is a reference-software constant. Under round-up it
    maps to row 31; under `>> 7` it maps to row 30 and row 31 is unreachable for
    every possible `Iσ` -- a dead CDF row in a design whose stated goal is a small
    table. That asymmetry is what settles the rule, so it is what gets tested.
    """
    si = _index()
    assert si.max_index == 3967
    top = torch.tensor([si.max_index])
    assert int(si.table_row(top)) == si.levels - 1 == 31
    assert si.max_index >> si.precision == 30      # what round-down would have given


def test_every_cdf_row_is_reachable():
    si = _index()
    rows = si.table_row(torch.arange(si.max_index + 1))
    assert set(rows.tolist()) == set(range(si.levels))


def test_table_row_is_monotone_and_lands_exactly_on_grid_points():
    si = _index()
    idx = torch.arange(si.max_index + 1)
    rows = si.table_row(idx)
    assert torch.all(rows[1:] >= rows[:-1])
    # A grid point is an exact multiple of 2**precision, and round-up must map it to
    # itself rather than to the row above -- `(a + b - 1) // b` is what guarantees it.
    for k in range(si.levels - 1):
        assert int(si.table_row(torch.tensor([k * si.step]))) == k
        # And one step past it moves up, so no row swallows two grid points.
        assert int(si.table_row(torch.tensor([k * si.step + 1]))) == k + 1


def test_table_row_clamps_instead_of_indexing_past_the_table():
    si = _index()
    wild = torch.tensor([-10_000, -1, si.max_index + 1, 10_000])
    rows = si.table_row(wild)
    assert rows.tolist() == [0, 0, si.levels - 1, si.levels - 1]


def test_table_row_does_not_mutate_its_argument():
    """`.to(int64)` on an int64 tensor is a no-op, so an in-place clamp inside
    `table_row` would rewrite the caller's tensor. It did, once."""
    si = _index()
    idx = torch.tensor([-5, 99_999], dtype=torch.int64)
    si.table_row(idx)
    assert idx.tolist() == [-5, 99_999]


# -- the ULP disagreement -------------------------------------------------------
def test_the_two_index_paths_disagree_on_exactly_eleven_indices():
    si = _index()
    gc = _gc(si)
    idx = torch.arange(si.max_index + 1)
    integer = si.table_row(idx)
    via_float = gc.build_indexes(si.sigma(idx.float()))
    bad = (integer != via_float).nonzero().flatten().tolist()
    assert bad == ULP_DISAGREEMENTS
    # Every one is exactly on a grid point. If a disagreement ever appeared *between*
    # grid points the cause would not be a ULP and this whole analysis would be wrong.
    assert all(i % si.step == 0 for i in bad)
    # The float path is the one that is off, and always by one row upward: sigma(I)
    # lands a ULP above the table entry, and `build_indexes` counts entries strictly
    # below sigma.
    for i in bad:
        assert int(via_float[i]) == int(integer[i]) + 1


def test_the_disagreement_is_one_float32_ulp_and_nothing_larger():
    si = _index()
    table = torch.tensor(build_scale_table(si.minimum, si.maximum, si.levels))
    for i in ULP_DISAGREEMENTS:
        k = i // si.step
        got = float(si.sigma(torch.tensor(float(i))))
        want = float(table[k])
        assert got > want                                  # high side, hence the +1 row
        assert abs(got - want) / want < 1e-6               # and only just


def test_mixing_the_two_index_paths_corrupts_the_latent():
    """The failure the integer path exists to prevent, demonstrated.

    Not a hypothetical: encode with the integer row, decode with the float row, and
    the returned latent is wrong -- while both calls return normally. This is why the
    project's rule is that the split-hyper codec never touches `build_indexes`.
    """
    si = _index()
    gc = _gc(si)
    gc.update(force=True)
    i_sigma = torch.tensor([[float(i) for i in ULP_DISAGREEMENTS]]).reshape(1, 1, 1, -1)
    scales = si.sigma(i_sigma)
    y = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0]]
                     ).reshape(1, 1, 1, -1)
    integer_rows = si.table_row(si.quantise(i_sigma))
    float_rows = gc.build_indexes(scales)
    assert not torch.equal(integer_rows, float_rows)

    strings = gc.compress(y, scales, indexes=integer_rows)
    matched = gc.decompress(strings, scales, indexes=integer_rows)
    assert torch.equal(matched, y)

    mismatched = gc.decompress(strings, scales, indexes=float_rows)
    assert not torch.equal(mismatched, y)


def test_the_gaussian_conditional_validates_explicit_indexes():
    si = _index()
    gc = _gc(si)
    gc.update(force=True)
    scales = torch.full((1, 2, 1, 3), 1.0)
    with pytest.raises(ValueError, match="shape"):
        gc.compress(torch.zeros_like(scales), scales,
                    indexes=torch.zeros(1, 2, 1, 4, dtype=torch.int32))
    with pytest.raises(ValueError, match=r"\[0, 31\]"):
        gc.compress(torch.zeros_like(scales), scales,
                    indexes=torch.full((1, 2, 1, 3), 32, dtype=torch.int32))
    with pytest.raises(ValueError, match=r"\[0, 31\]"):
        gc.compress(torch.zeros_like(scales), scales,
                    indexes=torch.full((1, 2, 1, 3), -1, dtype=torch.int32))


# -- the sigma codebook ---------------------------------------------------------
def test_the_largest_representable_sigma_falls_just_short_of_the_grid_top():
    """A consequence of `max_index = 3967` rather than 3968, and it sharpens the
    round-up argument. `Iσ` can never *denote* 54.82: the most it reaches is 54.734,
    one 128th of a grid step below. The top table entry is reachable only as a CDF
    **row**, which round-up delivers and round-down does not. So at the very top of
    the range the coder uses a distribution slightly wider than the one predicted --
    which is the safe direction, and the only direction available.
    """
    si = _index()
    assert float(si.sigma(torch.tensor(0.0))) == pytest.approx(si.minimum, rel=1e-6)
    top = float(si.sigma(torch.tensor(float(si.max_index))))
    assert top == pytest.approx(54.734, abs=1e-3)
    assert top < si.maximum
    # One further step would have landed exactly on it; the clamp is what stops it.
    assert si.minimum * math.exp(si.log_k * (si.max_index + 1) / si.step) \
        == pytest.approx(si.maximum, rel=1e-6)
    # The row it selects is nonetheless the one holding 54.82.
    table = build_scale_table(si.minimum, si.maximum, si.levels)
    assert table[int(si.table_row(torch.tensor([si.max_index])))] \
        == pytest.approx(si.maximum, rel=1e-6)


def test_from_sigma_inverts_sigma():
    si = _index()
    idx = torch.linspace(0, si.max_index, 97)
    assert torch.allclose(si.from_sigma(si.sigma(idx)), idx, atol=1e-2)


def test_the_clamp_passes_gradient_only_when_it_points_back_into_range():
    """A plain `clamp` would freeze a saturated channel forever. `Clamp` lets the
    gradient through when it is trying to bring the value back."""
    si = _index()
    for start, grad, expect in [(-500.0, -1.0, True),    # below, pushed up: pass
                                (-500.0, +1.0, False),   # below, pushed down: block
                                (9e9, +1.0, True),       # above, pushed down: pass
                                (9e9, -1.0, False),      # above, pushed up: block
                                (100.0, +1.0, True)]:    # in range: always
        x = torch.tensor([start], requires_grad=True)
        si.clamp(x).backward(torch.tensor([grad]))
        assert (float(x.grad) != 0.0) is expect, (start, grad)


def test_quantise_keeps_sigma_and_the_cdf_row_consistent():
    """The inference path derives both from the *same* integer. Deriving sigma from
    the float index and the row from the rounded one would let them disagree."""
    br, si = _branch()
    z = torch.randn(1, 32, 4, 4)
    p = br.predict(z, quantise=True)
    assert p["i_sigma"].dtype == torch.int32
    assert torch.equal(p["scales"], si.sigma(p["i_sigma"].float()))
    assert torch.equal(p["rows"], si.table_row(p["i_sigma"]))


# -- the networks ---------------------------------------------------------------
def test_the_hyper_encoder_refuses_a_non_preserving_width():
    with pytest.raises(ValueError, match="channel-preserving"):
        HyperEncoder(96, 64)


def test_the_hyper_encoder_ignores_the_sign_of_the_latent():
    """`abs_in_hyperprior: true`. Scale is a property of magnitude, so feeding the
    sign would spend capacity learning that sigma(y) == sigma(-y)."""
    torch.manual_seed(0)
    h = HyperEncoder(16, 16).eval()
    y = torch.randn(1, 16, 8, 8)
    with torch.no_grad():
        assert torch.equal(h(y), h(-y))


def test_the_hyper_decoder_can_predict_negative_values():
    """A ReLU on `p̈` would forbid negative predictions, and the residual
    `r = round(y - p̈)` would then carry the whole negative half of the latent."""
    torch.manual_seed(0)
    d = HyperDecoder(16).eval()
    with torch.no_grad():
        out = d(torch.randn(4, 16, 4, 4) * 5)
    assert float(out.min()) < 0 < float(out.max())


def test_the_hyper_decoder_upsamples_by_four():
    torch.manual_seed(0)
    assert HyperDecoder(16).eval()(torch.zeros(1, 16, 3, 5)).shape == (1, 16, 12, 20)


def test_the_scale_decoder_upsamples_by_four_in_one_shuffle():
    torch.manual_seed(0)
    assert HyperScaleDecoder(16).eval()(torch.zeros(1, 16, 3, 5)).shape \
        == (1, 16, 12, 20)


def test_every_convolution_in_the_scale_decoder_runs_below_the_latent_grid():
    """Why the scale decoder is nearly free, as a structural assertion rather than a
    MAC number: the single `PixelShuffle(4)` is the *last* operation, so no multiply
    ever happens at the latent resolution."""
    torch.manual_seed(0)
    d = HyperScaleDecoder(16).eval()
    seen = []
    for m in d.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(lambda mo, i, o: seen.append(tuple(o.shape[-2:])))
    with torch.no_grad():
        out = d(torch.zeros(1, 16, 4, 4))
    assert seen and all(s == (4, 4) for s in seen)
    assert out.shape[-2:] == (16, 16)


def test_the_scale_decoder_starts_mid_table_rather_than_at_the_floor():
    """Without this the run opens with sigma pinned at 0.11 over a random latent:
    every symbol escapes and the gradient is about escapes, not about the image."""
    br, si = _branch()
    p = br.predict(torch.randn(2, 32, 4, 4))
    assert float(p["i_sigma"].mean().detach()) \
        == pytest.approx(si.max_index / 2, rel=0.05)
    assert float(p["scales"].mean().detach()) == pytest.approx(2.45, rel=0.05)
    # And the prediction itself must still start near zero -- it is a latent, not a
    # scale, and biasing it would put a constant offset into every residual.
    assert abs(float(p["means"].mean().detach())) < 0.5


def test_the_fused_decoder_biases_only_the_sigma_half():
    """`PixelShuffle(r)` sends conv output channels [r^2*j, r^2*(j+1)) to shuffled
    channel j, so `chunk(2)`'s second half comes from the conv's upper 4c channels
    and nothing else. If that were not true, biasing the tail of the bias vector
    would leak an offset into `p̈`."""
    torch.manual_seed(0)
    d = FusedHyperDecoder(8, init_index=1983.5).eval()
    with torch.no_grad():
        pred, i_sigma = d(torch.randn(2, 8, 4, 4))
    assert abs(float(pred.mean())) < 1.0
    assert float(i_sigma.mean()) == pytest.approx(1983.5, rel=0.01)


# -- the branch contract --------------------------------------------------------
def test_predict_takes_z_hat_and_nothing_else():
    """The one invariant of eq. (1)/(2): the decoder has only `zhat` when it computes
    `p̈`. Any `y` argument here would make the prediction encoder-only and the decoder
    would drift. Enforced on the signature so it cannot be added quietly."""
    sig = inspect.signature(SplitHyperBranch.predict)
    assert [p.name for p in sig.parameters.values()] == ["self", "z_hat", "quantise"]
    assert sig.parameters["quantise"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("fused", [False, True])
def test_the_branch_round_trips_the_latent_bit_exactly(fused):
    br, si = _branch(fused=fused)
    gc = _gc(si)
    gc.update(force=True)
    br.entropy_bottleneck.update(force=True)
    y = torch.randn(1, 32, 8, 8) * 3
    with torch.no_grad():
        part = br.compress(y, gc)
        dec = br.decompress(part, gc, torch.device("cpu"))
    # Not `y`: the coder round-trips the *quantised* latent, so the reference is
    # `round(y - p̈) + p̈` computed from the decoded z_hat.
    p = br.predict(dec["z_hat"], quantise=True)
    assert torch.equal(dec["y_hat"], torch.round(y - p["means"]) + p["means"])


@pytest.mark.parametrize("fused", [False, True])
def test_the_branch_exposes_the_index_during_training(fused):
    br, si = _branch(fused=fused)
    gc = _gc(si)
    out = br(torch.randn(2, 32, 8, 8), gc, noise=False, ste=True)
    assert set(out) >= {"y_hat", "y_lik", "z_lik", "z", "z_hat", "scales", "means",
                        "i_sigma"}
    assert out["i_sigma"].shape == out["scales"].shape


def test_the_split_branch_has_a_scale_decoder_and_the_fused_one_does_not():
    assert _branch(fused=False)[0].h_scale is not None
    assert _branch(fused=True)[0].h_scale is None


# -- the codec ------------------------------------------------------------------
def _codec(kind):
    torch.manual_seed(0)
    m = build_any_model(load_config("tierA"), kind).eval()
    m.update(force=True)
    return m


@pytest.mark.parametrize("kind", ["twobranch-split", "twobranch-fused"])
def test_the_codec_round_trips_and_reports_the_index(kind):
    m = _codec(kind)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        out = m(x, noise=False, ste=True)
        dec = m.decompress(m.compress(x))
    assert torch.equal(dec["y_hat"], out["y_hat"])
    assert "i_sigma" in out and "i_sigma_uv" in out


def test_phase_four_stays_on_the_float_path():
    """`twobranch` must keep behaving exactly as it did, or its checkpoints are
    wrong weights rather than a load error."""
    m = _codec("twobranch")
    with torch.no_grad():
        out = m(torch.rand(1, 3, 64, 64), noise=False, ste=True)
    assert "i_sigma" not in out
    assert m.sigma_index is None
    rows = m.coder_rows(out, "")
    assert torch.equal(rows, m.gaussian_conditional.build_indexes(out["scales"]))


def test_the_split_codec_gates_on_the_integer_row():
    m = _codec("twobranch-split")
    with torch.no_grad():
        out = m(torch.rand(1, 3, 64, 64), noise=False, ste=True)
    si = m.sigma_index
    assert torch.equal(m.coder_rows(out, ""),
                       si.table_row(si.quantise(out["i_sigma"])))
    assert torch.equal(m.coder_rows(out, "_uv"),
                       si.table_row(si.quantise(out["i_sigma_uv"])))


def test_the_scale_decoders_are_under_five_percent_of_decoder_macs():
    """Phase 5 acceptance criterion 2, measured rather than asserted by design."""
    m = _codec("twobranch-split")
    parts = m.summary_parts()
    mb = macs_breakdown(m, (1, 3, 128, 128), parts=[(n, s) for n, s, _ in parts])
    decoder = {n: mb[n] for n, _, is_dec in parts if is_dec and n in mb}
    assert {"h_scale_y", "h_scale_uv"} <= set(decoder)
    share = (decoder["h_scale_y"] + decoder["h_scale_uv"]) / sum(decoder.values())
    assert share < 0.05, f"scale decoders are {100 * share:.2f}% of decoder MACs"


def test_the_split_hyper_decoder_is_cheaper_than_phase_fours_fused_one():
    """The complexity claim behind §VI-E, as a comparison rather than an absolute.
    Phase 4's mean-scale `h_s` is a deconv stack that does two of its three layers at
    the latent grid; the confirmed structure keeps every multiply at /64 or /32."""
    def h_s_macs(kind):
        m = _codec(kind)
        parts = m.summary_parts()
        mb = macs_breakdown(m, (1, 3, 128, 128), parts=[(n, s) for n, s, _ in parts])
        return mb["h_s_y"]
    assert h_s_macs("twobranch-split") < 0.25 * h_s_macs("twobranch")


def test_the_configuration_flags_refuse_meaningless_combinations():
    from jpegai.models.twobranch import TwoBranchCodec
    with pytest.raises(ValueError, match="mean_scale"):
        TwoBranchCodec(split_hyper=True, mean_scale=False)
    with pytest.raises(ValueError, match="ablation"):
        TwoBranchCodec(fused_hyper=True, split_hyper=False)


def test_the_three_two_branch_kinds_are_distinct_architectures():
    """They share a `--model` string namespace and a checkpoint format, so two kinds
    that built the same thing would silently load each other's weights."""
    keys = {k: set(_codec(k).state_dict()) for k in
            ("twobranch", "twobranch-split", "twobranch-fused")}
    assert keys["twobranch"] != keys["twobranch-split"]
    assert keys["twobranch-split"] != keys["twobranch-fused"]
    assert any("h_scale" in k for k in keys["twobranch-split"])
    assert not any("h_scale" in k for k in keys["twobranch-fused"])
