"""Tests for Phase 10's decoder-side latent-domain tools — RVS (§VI-G) and LSBS (§VI-H).

These tools share one unusual contract, and every property below is a facet of it: they
run on the *reconstructed* latent, after entropy decoding and before synthesis, and they
never touch the bitstream. That is what makes them the ablation lever — the identical
packet decodes with the tools on or off — and it is also what makes them dangerous, since
a bug here changes the picture with nothing in the rate to flag it.

**Identity at initialisation must be exact, not approximate.** T1 = 0 / T2 = 2¹⁶ for RVS
and TR = TP = 0 for LSBS are chosen so a freshly-attached tool is a bit-exact no-op: an
existing Phase 5/6/8 checkpoint has to reconstruct the same tensor it always did until the
tables are trained. "Almost identity" would mean every prior BD-rate number was measured on
a subtly different codec.

**The pooling is eq. 7 to the integer, boundary pad included.** `σ = (32 + Σ_{8×8} Iσ) >> 6`
is a fixed-point block mean with round-to-nearest, and the pad value 1411 is what a
partial edge block sees. RVS and LSBS must pool identically — the spec ties LSBS to "the
same pooled σ" — so both go through `pool_sigma`.

**eq. 9 and eq. 10 are fixed-point, and the shift floors.** RVS scales the residual by
`T2/2¹⁶`; LSBS adds `(r·TR + μ·TP + 2¹²) >> 13`. The floors have no gradient, so training
reaches the tables through a straight-through estimator — tested by checking a gradient
actually arrives at the table it is supposed to move.

**Skip-anyway decode semantics.** `_resolve_tools` intersects the request with what was
built: a present tool can be skipped at decode, but asking for an absent one cannot conjure
it. This is the one gate the ablation harness turns, so it is tested directly rather than
only through a round trip.
"""

from __future__ import annotations

import pytest
import torch

from jpegai.models.tools.lsbs import FIXED as LSBS_FIXED, LatentScalingBeforeSynthesis
from jpegai.models.tools.postfilter import (
    EdgeFreeEnhancementLinear, EdgeFreeEnhancementNonlinear,
    InterComponentInformation, LumaEnhancementFilter,
)
from jpegai.models.tools.rvs import FIXED as RVS_FIXED, PAD, POOL, ResidualVarianceScaling, pool_sigma
from jpegai.eval.ablation import (_bpp_is_subset_invariant, build_report,
                                  measure_subsets, tool_macs, tool_subsets)
from jpegai.eval.metrics import PAPER_SEVEN
from jpegai.models.twobranch import ALL_TOOLS, LATENT_TOOLS, PIXEL_TOOLS, TwoBranchCodec
from jpegai.train.stages import (PARTS, ToolsStage, apply_freeze, aux_is_trained,
                                 check_partition, part_parameters, tools_stage)

BUCKETS = 3968     # SigmaIndex().max_index + 1; the σ-table extent both tools index by


def _codec(*, tools=("rvs", "lsbs"), split_hyper=True, mcm=False, gain=False):
    """A small split-hyper two-branch codec with the requested Phase 10 tools attached.

    Deliberately tiny (the entropy tables are what make `update()` slow) but a real codec:
    the tools are constructed exactly as `build_two_branch` would build them, so the
    integration tests exercise the same wiring the training path does.
    """
    return TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                          chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                          synthesis_width=(24, 16, 16, 16),
                          internal_format="420", mean_scale=True,
                          split_hyper=split_hyper, mcm=mcm, gain=gain, tools=tools).eval()


# ---------------------------------------------------------------------------
# eq. 7: pooled σ is a fixed-point 8×8 block mean, with a 1411 boundary pad
# ---------------------------------------------------------------------------
def test_pool_sigma_is_the_block_mean_rounded_to_nearest():
    """A uniform block pools to its own value, and a ramp pools to the rounded mean.

    `(32 + Σ) >> 6` over 64 elements is `(Σ + 32)//64`. A constant block sums to `64v`, and
    `(64v + 32)//64 = v` exactly. The `arange(64)` ramp sums to 2016, and `(2016+32)//64 =
    32` — the round-to-nearest, not a truncation, is what the `+32` buys.
    """
    const = torch.full((1, 1, POOL, POOL), 7, dtype=torch.long)
    assert torch.equal(pool_sigma(const, BUCKETS), torch.full_like(const, 7))

    ramp = torch.arange(POOL * POOL).view(1, 1, POOL, POOL)
    assert int(pool_sigma(ramp, BUCKETS).unique().item()) == 32


def test_pool_sigma_pads_partial_edge_blocks_with_1411():
    """A 4×4 tile is a single partial block: 16 real samples, 48 padded at value 1411.

    The pooled index is therefore `(16·v + 48·1411 + 32)//64`, upsampled and cropped back to
    4×4. For v = 0 that is `(48·1411 + 32)//64 = 1058`, uniform across the tile — the pad is
    not a wraparound or a zero, it is the spec's fixed edge σ.
    """
    tile = torch.zeros(1, 1, 4, 4, dtype=torch.long)
    pooled = pool_sigma(tile, BUCKETS)
    assert pooled.shape == (1, 1, 4, 4)
    assert torch.equal(pooled, torch.full_like(pooled, (48 * PAD + 32) // 64))


def test_pool_sigma_clamps_into_the_table():
    """σ over-range clamps to the last bucket rather than indexing past the tables.

    An index that pooled to ≥ buckets would be an out-of-bounds gather at eq. 8/9/10; the
    clamp is what keeps a pathological Iσ from becoming a crash instead of a saturated σ.
    """
    huge = torch.full((1, 1, POOL, POOL), 10 ** 6, dtype=torch.long)
    assert int(pool_sigma(huge, BUCKETS).max().item()) == BUCKETS - 1


# ---------------------------------------------------------------------------
# §VI-G: RVS scales the residual (eq. 9) and shifts the index (eq. 8)
# ---------------------------------------------------------------------------
def _rr():
    """A residual and a matching in-range σ index, deterministic across runs."""
    torch.manual_seed(0)
    r = torch.randn(1, 2, 16, 16)
    i_sigma = torch.randint(0, BUCKETS, (1, 2, 16, 16))
    return r, i_sigma


def test_rvs_is_identity_at_init():
    """T1 = 0, T2 = 2¹⁶: the residual is scaled by 1 and the index shifted by 0.

    This is the property the whole no-op contract rests on. Powers of two multiply and
    divide exactly in float32, so `r · 2¹⁶ / 2¹⁶` is `r` to the bit, not merely close.
    """
    rvs = ResidualVarianceScaling(buckets=BUCKETS)
    r, i_sigma = _rr()
    out = rvs(r, i_sigma)
    assert torch.equal(out["r"], r)
    assert torch.equal(out["i_sigma"], i_sigma)


def test_rvs_scales_the_residual_by_T2_over_2_16():
    """eq. 9 exactly: T2 = 2¹⁵ everywhere halves the residual (2¹⁵/2¹⁶ = ½)."""
    rvs = ResidualVarianceScaling(buckets=BUCKETS)
    rvs.T2.data.fill_(float(1 << (RVS_FIXED - 1)))
    r, i_sigma = _rr()
    assert torch.equal(rvs(r, i_sigma)["r"], r * 0.5)


def test_rvs_id_vector_routes_channels_to_table_rows():
    """`id[c]` selects which table row each channel reads; the default is 2.

    Row 2 is halved, every other row left at identity. With the default ids the residual
    halves; with an explicit all-zero id vector it passes through untouched — the gather is
    per-channel and keyed on `id`, not a single shared row.
    """
    rvs = ResidualVarianceScaling(buckets=BUCKETS)
    rvs.T2.data[0, 2, :] = float(1 << (RVS_FIXED - 1))
    r, i_sigma = _rr()
    assert torch.equal(rvs(r, i_sigma)["r"], r * 0.5)                 # default id = 2
    row0 = torch.zeros(r.shape[1], dtype=torch.long)
    assert torch.equal(rvs(r, i_sigma, ids=row0)["r"], r)            # row 0 is identity


def test_rvs_index_shift_is_eq8_and_clamps():
    """eq. 8 adds `T1[…,σ]` to the index and clamps into the table.

    Inert downstream in our reconstruction pipeline (synthesis consumes ŷ = p̈ + r̂), but the
    module still owns the arithmetic, so it is checked here: +5 in range lands at v+5, and
    +5 at the ceiling saturates rather than running off the end of the tables.
    """
    rvs = ResidualVarianceScaling(buckets=BUCKETS)
    rvs.T1.data[0, 2, :] = 5.0
    r = torch.zeros(1, 1, 8, 8)
    lo = torch.full((1, 1, 8, 8), 10)
    assert torch.equal(rvs(r, lo)["i_sigma"], torch.full_like(lo, 15))
    hi = torch.full((1, 1, 8, 8), BUCKETS - 1)
    assert torch.equal(rvs(r, hi)["i_sigma"], torch.full_like(hi, BUCKETS - 1))


def test_rvs_gradient_reaches_T2():
    """A gradient on the scaled residual arrives at the T2 buckets it read.

    RVS is trained with the backbone frozen, so the tables are the *only* thing the loss can
    move; a table that received no gradient would sit at its init forever and the tool would
    be dead weight. `∂(r·T2/2¹⁶)/∂T2 = r/2¹⁶`, nonzero wherever r is.
    """
    rvs = ResidualVarianceScaling(buckets=BUCKETS)
    r, i_sigma = _rr()
    rvs(r, i_sigma)["r"].sum().backward()
    assert rvs.T2.grad is not None and rvs.T2.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# §VI-H: LSBS reweights prediction vs residual (eq. 10), fixed-point with an STE
# ---------------------------------------------------------------------------
def test_lsbs_is_identity_at_init():
    """TR = TP = 0: the bracket is `2¹² >> 13 = 0`, so ŷ is returned untouched.

    The half-add makes the floor round to nearest, and at init the only thing inside it is
    the half itself — `floor(4096/8192) = 0` — which is why zero tables are a true no-op and
    not an off-by-one nudge.
    """
    lsbs = LatentScalingBeforeSynthesis(buckets=BUCKETS)
    y_hat = torch.randn(1, 2, 16, 16)
    r = torch.randn(1, 2, 16, 16)
    sp = torch.zeros(1, 2, 16, 16, dtype=torch.long)
    assert torch.equal(lsbs(y_hat, r, sp), y_hat)


def test_lsbs_residual_term_is_eq10():
    """TR = 2¹³, TP = 0: the added term is `floor((r·2¹³ + 2¹²) >> 13) = floor(r + ½)`."""
    lsbs = LatentScalingBeforeSynthesis(buckets=BUCKETS)
    lsbs.TR.data[0, :] = float(1 << LSBS_FIXED)
    torch.manual_seed(1)
    y_hat, r = torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)
    sp = torch.zeros(1, 2, 8, 8, dtype=torch.long)
    assert torch.equal(lsbs(y_hat, r, sp), y_hat + torch.floor(r + 0.5))


def test_lsbs_prediction_term_recovers_mu_as_yhat_minus_r():
    """TP = 2¹³, TR = 0: the term is `floor(μ + ½)` with μ reconstructed as ŷ − r = p̈.

    The paper writes eq. 10 in terms of the prediction μ, but the decoder only has ŷ and r̂;
    getting the μ term right is a claim that `ŷ − r` really is the context model's mean. A
    sign slip there would reweight toward the residual instead of the prediction.
    """
    lsbs = LatentScalingBeforeSynthesis(buckets=BUCKETS)
    lsbs.TP.data[0, :] = float(1 << LSBS_FIXED)
    torch.manual_seed(2)
    y_hat, r = torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)
    sp = torch.zeros(1, 2, 8, 8, dtype=torch.long)
    assert torch.equal(lsbs(y_hat, r, sp), y_hat + torch.floor((y_hat - r) + 0.5))


def test_lsbs_straight_through_gradient_reaches_both_tables():
    """The `>>13` floor has no gradient, so training must reach TP/TR through the STE.

    The estimator passes `val/2¹³` backward; a gradient that failed to arrive would leave
    the tables frozen at their identity init exactly as if the tool were disabled.
    """
    lsbs = LatentScalingBeforeSynthesis(buckets=BUCKETS)
    torch.manual_seed(3)
    y_hat, r = torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)
    sp = torch.zeros(1, 2, 8, 8, dtype=torch.long)
    lsbs(y_hat, r, sp).sum().backward()
    assert lsbs.TR.grad is not None and lsbs.TR.grad.abs().sum() > 0
    assert lsbs.TP.grad is not None and lsbs.TP.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Integration: the tools as the codec constructs, gates, and applies them
# ---------------------------------------------------------------------------
def test_tools_require_the_split_hyper_path():
    """Without an integer σ index there is nothing to key the tables on, so it refuses.

    Same shape of refusal the gain unit makes: a codec that quietly built the tools on the
    fused-hyper path would index them by a σ that does not exist.
    """
    with pytest.raises(ValueError, match="split_hyper"):
        _codec(tools=("rvs",), split_hyper=False)


def test_unknown_tool_name_is_rejected():
    """A misspelled `--tools` entry is a hard error, not a silently-ignored no-op."""
    with pytest.raises(ValueError, match="unknown Phase 10 tool"):
        _codec(tools=("rvs", "bogus"))


def test_resolve_tools_intersects_never_unions():
    """`None` → all built; a request is intersected with what exists, never unioned.

    The asymmetry is the point: a built tool can be dropped at decode (the ablation), but a
    tool that was never constructed cannot be summoned by naming it.
    """
    both = _codec(tools=("rvs", "lsbs"))
    assert both._resolve_tools(None) == frozenset({"rvs", "lsbs"})
    assert both._resolve_tools(set()) == frozenset()
    assert both._resolve_tools(["rvs"]) == frozenset({"rvs"})
    assert both._resolve_tools(["rvs", "bogus"]) == frozenset({"rvs"})
    only_rvs = _codec(tools=("rvs",))
    assert only_rvs._resolve_tools(["rvs", "lsbs"]) == frozenset({"rvs"})


def test_partition_names_the_tools_bucket():
    """The four table Parameters (T1, T2, TR, TP) land in the `"tools"` training bucket.

    `check_partition` asserts `training_parts()` is a partition of `parameters()`; reaching
    the count means the tools were added to exactly one bucket and none was dropped. A codec
    with no tools leaves the bucket empty rather than absent.
    """
    assert check_partition(_codec(tools=("rvs", "lsbs")))["tools"] == 4
    assert check_partition(_codec(tools=("rvs",)))["tools"] == 2
    assert check_partition(_codec(tools=()))["tools"] == 0


def test_forward_is_identity_at_init_with_tools_on():
    """The forward synthesis path is byte-identical with tools on vs. off at init."""
    m = _codec(tools=("rvs", "lsbs"))
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    on = m(x, apply_tools=None)["x_hat"]
    off = m(x, apply_tools=set())["x_hat"]
    assert torch.equal(on, off)


def test_same_bitstream_decodes_with_tools_on_or_off_at_init():
    """One packet, two decodes: the tools are decoder-side and identity at init.

    This is the ablation contract in miniature — `compress` never sees `apply_tools`, and the
    identical `packet` reconstructs the same picture whether the tables are applied or not.
    """
    m = _codec(tools=("rvs", "lsbs"))
    m.update()
    x = torch.rand(1, 3, 64, 64)
    packet = m.compress(x)
    on = m.decompress(packet, apply_tools=None)["x_hat"]
    off = m.decompress(packet, apply_tools=set())["x_hat"]
    assert torch.equal(on, off)


def test_each_tool_moves_the_latent_once_its_tables_move():
    """After the tables leave their init, RVS and LSBS each change the reconstruction.

    Driven through `_refine_latent` with a synthetic non-zero residual rather than a trained
    checkpoint: the point is that the wiring is live and the two tools compose, not that a
    particular checkpoint gains a particular amount. RVS-alone differs from off, and adding
    LSBS differs from RVS-alone — neither is a no-op that the other is silently masking.
    """
    m = _codec(tools=("rvs", "lsbs"))
    m.rvs.T2.data[0, 2, :] = float(1 << (RVS_FIXED - 1))   # halve the residual (id 2)
    m.lsbs.TP.data[0, :] = float(1 << LSBS_FIXED)          # add floor(μ + ½)
    torch.manual_seed(4)
    means = torch.randn(1, 2, 16, 16)
    r = torch.randn(1, 2, 16, 16) * 4.0                    # a few latent units, quantised away at init
    y_hat = means + r
    i_sigma = torch.randint(0, BUCKETS, (1, 2, 16, 16))

    off = m._refine_latent(y_hat, means, i_sigma, frozenset())
    rvs_only = m._refine_latent(y_hat, means, i_sigma, frozenset({"rvs"}))
    both = m._refine_latent(y_hat, means, i_sigma, frozenset({"rvs", "lsbs"}))
    assert torch.equal(off, y_hat)
    assert not torch.allclose(rvs_only, off)
    assert not torch.allclose(both, rvs_only)


def test_refine_latent_skips_when_there_is_no_index():
    """No σ index (e.g. a branch that produced none) means the tools cannot pool: skip.

    The guard returns ŷ untouched rather than fabricating a σ, so a caller that forgets to
    thread `i_sigma` degrades to the tools-off codec instead of indexing garbage.
    """
    m = _codec(tools=("rvs", "lsbs"))
    m.rvs.T2.data.fill_(float(1 << (RVS_FIXED - 1)))
    y_hat = torch.randn(1, 2, 8, 8)
    means = torch.zeros(1, 2, 8, 8)
    assert torch.equal(
        m._refine_latent(y_hat, means, None, frozenset({"rvs", "lsbs"})), y_hat)


# ---------------------------------------------------------------------------
# §VI-M: the four pixel-domain post-filters, as standalone modules
# ---------------------------------------------------------------------------
def test_post_filters_are_identity_at_init():
    """Every filter's output conv is zero-init, so each adds exactly zero to start.

    This is the pixel-domain half of the tools-off contract: a checkpoint that gains these
    modules reconstructs the same planes until the filters are trained, whatever their input.
    """
    torch.manual_seed(0)
    luma = torch.randn(1, 1, 16, 16)
    chroma = torch.randn(1, 2, 8, 8)                       # 4:2:0 grid: half the luma size

    assert torch.equal(LumaEnhancementFilter()(luma), luma)
    assert torch.equal(EdgeFreeEnhancementLinear()(chroma), chroma)
    assert torch.equal(EdgeFreeEnhancementNonlinear()(chroma), chroma)
    l_ref, c_ref = InterComponentInformation()(luma, chroma)
    assert torch.equal(l_ref, luma) and torch.equal(c_ref, chroma)


def test_post_filter_output_convs_receive_gradient():
    """The zero-init output conv of each residual is what a gradient must reach to unfreeze it.

    A residual block whose output conv never received a gradient would stay at its identity
    init forever — indistinguishable from the tool being disabled. (The hidden conv legitimately
    gets no gradient on the very first step, because the zeroed output conv gates it; the output
    conv is the one that has to move.)
    """
    torch.manual_seed(0)
    luma = torch.randn(1, 1, 16, 16)
    chroma = torch.randn(1, 2, 8, 8)

    lef = LumaEnhancementFilter(); lef(luma).sum().backward()
    assert lef.conv2.weight.grad.abs().sum() > 0

    lin = EdgeFreeEnhancementLinear(); lin(chroma).sum().backward()
    assert lin.conv.weight.grad.abs().sum() > 0

    nl = EdgeFreeEnhancementNonlinear(); nl(chroma).sum().backward()
    assert nl.conv2.weight.grad.abs().sum() > 0

    icci = InterComponentInformation()
    l_ref, c_ref = icci(luma, chroma)
    (l_ref.sum() + c_ref.sum()).backward()
    assert icci.l2.weight.grad.abs().sum() > 0     # luma-refinement direction
    assert icci.c2.weight.grad.abs().sum() > 0     # chroma-refinement direction


def test_icci_bridges_the_chroma_grid_both_ways():
    """ICCI resamples across the 4:2:0 grid mismatch and preserves each plane's own size.

    Cross-component only means anything if luma can actually reach the chroma grid and back;
    the shapes coming out must match the shapes going in, or `_to_rgb` gets planes it cannot
    merge. Checked at 4:2:0 (2× mismatch) where the resampling is not a no-op.
    """
    icci = InterComponentInformation()
    icci.l2.weight.data.normal_(); icci.c2.weight.data.normal_()   # off identity
    luma = torch.randn(1, 1, 32, 32)
    chroma = torch.randn(1, 2, 16, 16)
    l_ref, c_ref = icci(luma, chroma)
    assert l_ref.shape == luma.shape and c_ref.shape == chroma.shape
    assert not torch.allclose(l_ref, luma) and not torch.allclose(c_ref, chroma)


# ---------------------------------------------------------------------------
# §VI-M: the post-filters as the codec builds, gates, and applies them
# ---------------------------------------------------------------------------
def test_pixel_tools_do_not_require_the_split_hyper_path():
    """The post-filters read planes, not the σ index, so they build without split-hyper.

    Only RVS/LSBS need the integer σ index. A codec that demanded split-hyper for a plane-
    domain filter would be coupling two unrelated things — and a *mixed* request that includes
    a latent tool must still refuse, because that half genuinely needs the index.
    """
    m = _codec(tools=PIXEL_TOOLS, split_hyper=False)       # builds, no raise
    assert m.lef is not None and m.icci is not None
    with pytest.raises(ValueError, match="split_hyper"):
        _codec(tools=("lef", "rvs"), split_hyper=False)


def test_partition_counts_every_tool_in_one_bucket():
    """All six tools' parameters land in `"tools"` and nowhere else.

    22 Parameters with the full stack: RVS (T1, T2) + LSBS (TR, TP) = 4, LEF 4, ICCI 8, EFE
    nonlinear 4, EFE linear 2. `check_partition` would raise if any were double-counted or
    orphaned, so the number standing is the real assertion.
    """
    assert check_partition(_codec(tools=ALL_TOOLS))["tools"] == 22
    assert check_partition(_codec(tools=("lef",)))["tools"] == 4


def test_full_tool_stack_is_identity_at_init_end_to_end():
    """With all six tools on, forward and a compress/decompress round trip are byte-neutral.

    The whole point of the identity-at-init design across both tool families: attaching the
    entire Table IV stack to a trained checkpoint changes nothing until the tools are trained.
    """
    m = _codec(tools=ALL_TOOLS)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    assert torch.equal(m(x, apply_tools=None)["x_hat"], m(x, apply_tools=set())["x_hat"])
    m.update()
    packet = m.compress(x)
    assert torch.equal(m.decompress(packet, apply_tools=None)["x_hat"],
                       m.decompress(packet, apply_tools=set())["x_hat"])


def _perturb_output_conv(m, name):
    """Push one filter's zero-init output conv off identity so it actually does something."""
    convs = {"lef": lambda: [m.lef.conv2], "efe-nonlinear": lambda: [m.efe_nl.conv2],
             "efe-linear": lambda: [m.efe_lin.conv],
             "icci": lambda: [m.icci.l2, m.icci.c2]}[name]()
    for conv in convs:
        conv.weight.data.normal_(std=0.1)


@pytest.mark.parametrize("name", PIXEL_TOOLS)
def test_each_post_filter_changes_the_planes_once_trained(name):
    """Each filter, on its own, moves the reconstruction after its output conv leaves zero.

    Run through `_postfilter` on synthetic planes so the test needs no checkpoint: the claim is
    that the filter is wired live and gated by `active`, filter by filter, not that it helps.
    """
    m = _codec(tools=PIXEL_TOOLS)
    _perturb_output_conv(m, name)
    torch.manual_seed(5)
    luma = torch.randn(1, 1, 64, 64)
    chroma = torch.randn(1, 2, 32, 32)                     # 4:2:0
    off_l, off_c = m._postfilter(luma, chroma, frozenset())
    on_l, on_c = m._postfilter(luma, chroma, frozenset({name}))
    assert torch.equal(off_l, luma) and torch.equal(off_c, chroma)
    moved = (not torch.allclose(on_l, luma)) or (not torch.allclose(on_c, chroma))
    assert moved, f"{name} left both planes unchanged"


def test_luma_only_decode_skips_the_chroma_scope_filters():
    """Under `luma_only` the chroma plane is a placeholder, so only LEF may touch anything.

    The chroma-scope filters (EFE ×2) and the cross-component ICCI must not run on the grey
    fill — doing so would stamp a learned pattern onto chroma the decoder was told to ignore.
    LEF reads luma alone and still applies.
    """
    m = _codec(tools=("lef", "icci", "efe-linear"))
    for name in ("lef", "icci", "efe-linear"):
        _perturb_output_conv(m, name)
    torch.manual_seed(6)
    luma = torch.randn(1, 1, 64, 64)
    chroma = torch.randn(1, 2, 32, 32)
    out_l, out_c = m._postfilter(luma, chroma, m.tool_names, luma_only=True)
    assert torch.equal(out_c, chroma)          # chroma-scope + ICCI skipped
    assert not torch.allclose(out_l, luma)     # LEF still sharpened luma


# ---------------------------------------------------------------------------
# Phase 10 training stage: ToolsStage duck-types a Table II Stage
# ---------------------------------------------------------------------------
def test_tools_stage_trains_only_the_tools_bucket():
    """The stage trains `tools` and freezes everything else -- the paper's own logic."""
    s = tools_stage(beta=0.01)
    assert s.parts == ("tools",)
    assert set(s.frozen) == set(PARTS) - {"tools"}
    assert s.is_tools is True


def test_tools_stage_loss_defaults_to_mix():
    """`mix` keeps MS-SSIM on -- six of the seven graded metrics are perceptual, and it
    matches `Stage.loss_kwargs`: mix passes nothing, only a pure-MSE stage forces zero."""
    assert tools_stage(beta=0.01).loss == "mix"
    assert tools_stage(beta=0.01).loss_kwargs() == {}
    assert ToolsStage(beta=0.01, loss="mse").loss_kwargs() == {"ms_ssim_weight": 0.0}


def test_tools_stage_carries_the_attributes_the_loop_reads_off_a_stage():
    """The loop drives it through the `Stage` seam, so it must quack like one."""
    s = tools_stage(beta=0.02, sample_delta_beta=True, model_id=3)
    for attr in ("name", "beta", "loss", "epochs", "parts", "sample_delta_beta",
                 "model_id", "frozen"):
        assert hasattr(s, attr), attr
    assert callable(s.loss_kwargs) and callable(s.summary)
    assert s.beta == 0.02 and s.model_id == 3 and s.sample_delta_beta is True
    # A plain Stage has no `is_tools`, so the loop's step-budget guard reads False there.
    assert getattr(s, "is_tools", False) is True


def test_tools_stage_rejects_a_non_tools_part_list_and_bad_loss():
    with pytest.raises(ValueError):
        ToolsStage(beta=0.01, parts=("encoder",))
    with pytest.raises(ValueError):
        ToolsStage(beta=0.01, loss="bogus")


def test_freeze_all_but_tools_leaves_only_tool_params_trainable():
    """`apply_freeze(model, ("tools",))` is exactly the freeze the tools stage applies:
    every tool parameter trains, every backbone parameter is frozen, and the optimiser's
    list (`part_parameters`) is precisely the tool parameters -- no more, no less."""
    m = _codec(tools=ALL_TOOLS, gain=True)
    check_partition(m)                                   # partition holds with tools
    apply_freeze(m, ("tools",))
    tool_ids = {id(p) for _, mod in m.training_parts()["tools"] for p in mod.parameters()}
    for p in m.parameters():
        assert p.requires_grad == (id(p) in tool_ids)
    got = {id(p) for p in part_parameters(m, ("tools",))}
    assert got == tool_ids
    # The entropy network is frozen, so the aux optimiser is off for a tools stage.
    assert aux_is_trained(m, ("tools",)) is False


# ---------------------------------------------------------------------------
# Phase 10 Table IV: the switchable-tools ablation over one common bitstream
# ---------------------------------------------------------------------------
def test_tool_subsets_orders_all_on_then_each_off_then_none():
    """`tool_subsets` names the exact decode configurations Table IV compares.

    `all-on` first (the anchor), then one `<tool> off` per tool in fixed `ALL_TOOLS`
    order (so the table reads the same however the model was built), then `none`. Each
    `<t> off` set is all-on minus exactly that tool -- the ablation's one-tool delta.
    """
    subs = tool_subsets(ALL_TOOLS)
    names = list(subs)
    assert names[0] == "all-on" and names[-1] == "none"
    assert names[1:-1] == [f"{t} off" for t in ALL_TOOLS]
    assert subs["all-on"] == frozenset(ALL_TOOLS)
    assert subs["none"] == frozenset()
    for t in ALL_TOOLS:
        assert subs[f"{t} off"] == frozenset(ALL_TOOLS) - {t}


def _vr_ablation_curves(model, *, n_images=1, crop=96, deltas=(-400, -200, 0, 200, 400)):
    """Sweep a variable-rate tools codec over >=4 delta-betas -> Table IV curves.

    Random images are enough: the ablation asserts *relationships between subsets of one
    bitstream* (equal at init, moved by a trained tool), never an absolute quality, so the
    content need only be something the codec can round-trip. >=4 points lets `build_report`
    reach BD-rate; a small crop keeps the whole sweep on CPU in a second or two.
    """
    torch.manual_seed(11)
    tensors = [torch.rand(1, 3, crop, crop) for _ in range(n_images)]
    subsets = tool_subsets(model.tool_names)
    rate_points = [(model, int(d)) for d in deltas]
    model.update(force=True)               # build the entropy tables compress() needs
    return measure_subsets(rate_points, tensors, subsets, "cpu"), subsets


def test_ablation_bitstream_is_subset_invariant_and_identity_at_init():
    """The decoder-side claim, end to end: at init every subset decodes the *same* picture.

    All six tools start at identity, so decoding one packet under any subset yields a
    byte-identical reconstruction (equal `psnr_y`) -- and because the tools never touch the
    bitstream, the bpp column is identical across subsets *by construction*, trained or not.
    With >=4 rate points `build_report` also produces the Table IV skeleton (one BD-rate row
    per `<tool> off`, anchored on all-on) even though every delta here is zero.
    """
    m = _codec(tools=ALL_TOOLS, gain=True)
    curves, subsets = _vr_ablation_curves(m)
    ref = curves["all-on"]
    for name, c in curves.items():
        assert c["psnr_y"] == pytest.approx(ref["psnr_y"]), f"{name} moved at init"
        assert c["bpp"] == pytest.approx(ref["bpp"])
    assert _bpp_is_subset_invariant(curves) is True

    report = build_report(curves, m, crop=96)
    assert report["anchor"] == "all-on" and report["bpp_subset_invariant"] is True
    assert report["n_rate_points"] == 5
    assert report["tools"] == sorted(ALL_TOOLS)
    # Table IV has one row per "<tool> off"; the anchor is not its own row.
    assert set(report["table_iv"]) == {f"{t} off" for t in ALL_TOOLS} | {"none"}


def test_ablation_moves_only_the_subset_that_drops_a_trained_tool():
    """Push one post-filter (LEF) off identity and only the subsets that *drop it* change.

    This is the whole ablation argument in one packet: the `<T> off` curve differs from
    all-on exactly when T is doing work, so `lef off` and `none` move while every other
    `<tool> off` is a no-op. LEF is the honest lever here because the latent tools (RVS/LSBS)
    are inert on this tiny untrained codec -- its decoded residual `r = ŷ − p̈` is exactly
    zero and the means are ~0, so eq. 9/10's fixed-point floors swallow any table change;
    their gating is proven on a synthetic non-zero residual in
    `test_each_tool_moves_the_latent_once_its_tables_move`. bpp stays subset-invariant
    regardless -- the tools are decoder-side, the bytes never moved.
    """
    m = _codec(tools=ALL_TOOLS, gain=True)
    _perturb_output_conv(m, "lef")                 # pixel tool: non-identity output conv
    curves, _ = _vr_ablation_curves(m)
    ref = curves["all-on"]["psnr_y"]

    for moved in ("lef off", "none"):              # each drops LEF, which now does work
        assert curves[moved]["psnr_y"] != pytest.approx(ref), f"{moved} should differ"
    for inert in ("rvs off", "lsbs off", "icci off", "efe-nonlinear off", "efe-linear off"):
        assert curves[inert]["psnr_y"] == pytest.approx(ref), f"{inert} should be a no-op"
    assert _bpp_is_subset_invariant(curves) is True    # decoder-side: bytes never moved


def test_tool_macs_are_zero_for_the_latent_lookups_and_positive_for_icci():
    """The kMAC/pxl column: RVS and LSBS are table look-ups (0 MAC), ICCI convolves (>0).

    The paper's point about the latent tools is that they buy rate for free; the harness
    has to show a literal 0 for them and a real cost for ICCI, the only post-filter that
    lands meaningfully above zero. RVS/LSBS never enter the conv-counting breakdown, so
    they read exactly 0.0 by construction, not by rounding.
    """
    macs = tool_macs(_codec(tools=ALL_TOOLS, gain=True), crop=96)
    assert set(macs) == set(ALL_TOOLS)
    assert macs["rvs"] == 0.0 and macs["lsbs"] == 0.0
    assert macs["icci"] > 0.0
    assert macs["icci"] >= max(macs[t] for t in PIXEL_TOOLS)   # ICCI is the costliest

