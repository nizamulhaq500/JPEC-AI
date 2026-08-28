"""Tests for Phase 6 — MCM, the 4-stage checkerboard context model (§VI-D, Fig. 2).

The one non-negotiable assertion in the whole phase, quoted from the plan:

    ŷ_decoder == ŷ_encoder exactly, tensor-for-tensor. Non-negotiable. Assert in CI.

It is non-negotiable because getting it wrong does **not** crash. Stage `k`'s
prediction depends on stages `< k`, so an encoder that computes its contexts by any
route the decoder does not repeat produces residuals the decoder cannot undo — and
what comes out is a codestream that decodes without complaint into a reconstruction
that drifts a little further with every stage. There is no error to catch; the only
detector is an equality test. Hence `test_the_decoder_rebuilds_the_encoders_latent`
below, at four sizes and three chroma formats, through a real arithmetic coder.

The second thing tested hard is that the context is actually *used*. "MCM gives 4-9%
BD-rate over Phase 5. If < 2%, context nets aren't seeing previous groups" is an RD
claim that needs a training run, but the structural half of it does not: a wiring
mistake that leaves `gather` disconnected still trains, still codes, and still loses
only a couple of percent. `test_a_later_coset_sees_an_earlier_one` and its negative
twin pin the dependency graph directly instead of waiting for a BD-rate number.

Third: §VI-E's decoupling. The residual field must come off the bitstream in **one**
call with no network in the loop, which means `gc.decompress` is called exactly once
with `means=None`. That is a property of the decoder's *shape*, not of its output, so
it is tested by watching the call rather than the pixels.
"""

from __future__ import annotations

import pytest
import torch

from jpegai.models import KINDS, MCM_STAGES, build_any_model
from jpegai.models.entropy import GaussianConditional, build_scale_table
from jpegai.models.hyper import HyperDecoder, SigmaIndex
from jpegai.models.mcm import (
    GROUP_ORDER, ContextStage, MCMBranch, MultiStageContextModel, chs2group,
    join_cosets, split_cosets, split_pred, stage_cosets,
)
from jpegai.models.twobranch import TwoBranchCodec

FMT_NAMES = ["444", "422", "420"]


def _codec(fmt="420", *, stages=4, mcm=True, split_hyper=True, **kw):
    """A small two-branch codec with MCM on the luma branch.

    `luma_latent=32` is the smallest width `chs2group`'s assert allows, which is the
    point: it exercises the `groups == 1` corner rather than Tier A's 3.
    """
    return TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                          chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                          synthesis_width=(24, 16, 16, 16),
                          internal_format=fmt, mean_scale=True,
                          split_hyper=split_hyper,
                          mcm=mcm, mcm_stages=stages, **kw).eval()


def _branch(chs=32, hyper=32, *, stages=4):
    """An `MCMBranch` on its own, with the coder and σ grid it would be handed."""
    gc = GaussianConditional(build_scale_table(0.11, 54.82, 32), scale_bound=0.11)
    br = MCMBranch(chs, hyper, sigma_index=SigmaIndex(), stages=stages).eval()
    return br, gc


# ---------------------------------------------------------------------------
# the coset partition
# ---------------------------------------------------------------------------
def test_the_order_is_diagonal_first_and_matches_the_config():
    """docs/06 §5 derives this twice from the reference software; the config records
    it. If the three ever disagree, the config wins and this test is the alarm."""
    from jpegai.config import load_config
    assert GROUP_ORDER == ((0, 0), (1, 1), (0, 1), (1, 0))
    cfg = load_config("tierA")
    assert [tuple(g) for g in cfg.entropy.mcm_group_order] == list(GROUP_ORDER)
    assert cfg.entropy.mcm_stages == len(GROUP_ORDER)


def test_the_four_cosets_tile_the_grid_exactly_once():
    """Not "the pieces are the right size" — every position covered exactly once.

    A partition that double-covered one position and missed another would still
    round-trip through split/join for most tensors, because the missed position keeps
    whatever the double-covered one wrote.
    """
    marker = torch.arange(8 * 8, dtype=torch.float32).reshape(1, 1, 8, 8)
    parts = split_cosets(marker)
    seen = torch.cat([p.reshape(-1) for p in parts]).sort().values
    assert torch.equal(seen, marker.reshape(-1).sort().values)


def test_split_and_join_are_exact_inverses():
    x = torch.randn(2, 5, 6, 10)
    assert torch.equal(join_cosets(split_cosets(x)), x)


def test_the_cosets_are_the_ones_the_order_names():
    x = torch.randn(1, 3, 4, 4)
    for part, (a, b) in zip(split_cosets(x), GROUP_ORDER):
        assert torch.equal(part, x[..., a::2, b::2])


def test_an_odd_latent_is_refused_rather_than_half_covered():
    """The codec pads to /64 so this cannot fire from `compress`; it fires when
    somebody hands MCM a hand-made tensor, and then it should say so."""
    with pytest.raises(ValueError, match="even latent grid"):
        split_cosets(torch.randn(1, 2, 5, 4))


def test_join_checks_the_coset_count():
    with pytest.raises(ValueError, match="expected 4 cosets"):
        join_cosets([torch.randn(1, 2, 3, 3)] * 3)


# ---------------------------------------------------------------------------
# split_pred — the pre-shuffle channel layout
# ---------------------------------------------------------------------------
def test_the_prediction_is_cut_the_way_pixel_shuffle_would_cut_it():
    """The identity that makes a Phase 5 warm start a warm start.

    At initialisation every `ContextStage` returns `pred + ~0`, so the assembled mean
    field must be *exactly* the field Phase 5's `pixel_shuffle` produced from the same
    `h_s` output. That is only true if the per-coset slices are the ones the shuffle
    itself would have taken.
    """
    pred = torch.randn(2, 4 * 5, 6, 7)
    assert torch.equal(join_cosets(split_pred(pred)),
                       torch.pixel_shuffle(pred, 2))


def test_the_contiguous_chunk_does_not_give_that_identity():
    """The negative half, and the reason `split_pred` exists at all.

    `chunk` returns four tensors of exactly the right shape, so it passes every
    round-trip and gradient test in this file — it is a permutation of the same
    numbers. The only thing it breaks is the warm start, silently. If someone
    "simplifies" `split_pred` back to `chunk`, this is the test that objects.
    """
    pred = torch.randn(2, 4 * 5, 6, 7)
    chunked = list(pred.chunk(4, dim=-3))
    assert not torch.equal(join_cosets(chunked), torch.pixel_shuffle(pred, 2))
    # ...and it really is the same multiset of numbers, which is why nothing else
    # notices.
    assert torch.equal(join_cosets(chunked).reshape(-1).sort().values,
                       torch.pixel_shuffle(pred, 2).reshape(-1).sort().values)


def test_split_pred_slices_are_strided_by_four():
    pred = torch.randn(1, 4 * 3, 2, 2)
    for part, (i, j) in zip(split_pred(pred), GROUP_ORDER):
        assert torch.equal(part, pred[:, 2 * i + j::4])


def test_split_pred_refuses_a_layout_it_cannot_mean():
    with pytest.raises(ValueError, match="2x2 tile"):
        split_pred(torch.randn(1, 8, 2, 2), order=((0, 0), (1, 1)))
    with pytest.raises(ValueError, match="multiple of"):
        split_pred(torch.randn(1, 10, 2, 2))


# ---------------------------------------------------------------------------
# chs2group and the stage schedule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chs,groups", [(32, 1), (96, 3), (160, 5), (320, 10)])
def test_chs2group_reproduces_the_references_formula(chs, groups):
    """`max(1, chs // 32)`, docs/06 §2. Tier A's 96 and the full model's 160 both
    land on real group counts, which is why `primary_latent % 32 == 0` is a validated
    config constraint and not a style preference."""
    assert chs2group(chs) == groups


def test_chs2group_refuses_a_non_multiple_instead_of_rounding():
    """The reference asserts. Softening it to `max(1, ...)` would give a *working*
    model with a non-standard structure — the failure this project could not detect."""
    with pytest.raises(ValueError, match="multiple of 32"):
        chs2group(48)


@pytest.mark.parametrize("stages,expected", [
    (4, [(0,), (1,), (2,), (3,)]),
    (2, [(0, 1), (2, 3)]),
    (1, [(0, 1, 2, 3)]),
])
def test_the_stage_schedule_is_consecutive_groups_of_the_order(stages, expected):
    """2 stages must be `{(0,0),(1,1)}` then `{(0,1),(1,0)}` — one checkerboard colour
    and then the other. That is only true because the order is diagonal-first, and it
    is what makes the 2-stage ablation the classic checkerboard model rather than an
    arbitrary half-measure."""
    assert stage_cosets(4, stages) == expected


def test_three_stages_do_not_divide_four_cosets():
    with pytest.raises(ValueError, match="does not divide"):
        stage_cosets(4, 3)


@pytest.mark.parametrize("stages,visible", [
    (4, [0, 1, 2, 3]),          # each coset sees every earlier one
    (2, [0, 0, 2, 2]),          # within a stage, nothing sees a sibling
    (1, [0, 0, 0, 0]),          # nothing sees anything: Phase 5 plus a refiner
])
def test_a_coset_sees_exactly_the_cosets_of_strictly_earlier_stages(stages, visible):
    assert MultiStageContextModel(32, stages=stages).visible == visible


def test_there_is_one_context_network_per_coset_at_every_stage_count():
    """The ablation must not be a capacity comparison. `stages` changes what each
    network may condition on, never how many networks there are."""
    counts = {s: len(MultiStageContextModel(32, stages=s).nets) for s in (1, 2, 4)}
    assert counts == {1: 4, 2: 4, 4: 4}


def test_stages_is_the_number_of_sequential_passes():
    for s in (1, 2, 4):
        assert MultiStageContextModel(32, stages=s).stages == s


def test_an_order_that_is_not_a_partition_of_the_tile_is_refused():
    with pytest.raises(ValueError, match="exactly once"):
        MultiStageContextModel(32, order=((0, 0), (0, 0), (1, 1), (1, 0)))


# ---------------------------------------------------------------------------
# the loop — one function, both directions
# ---------------------------------------------------------------------------
def _loop(chs=32, h=8, w=12, *, stages=4, seed=0):
    """A model, a random pre-shuffle prediction, and a random latent."""
    torch.manual_seed(seed)
    mcm = MultiStageContextModel(chs, stages=stages).eval()
    pred = torch.randn(1, chs * 4, h // 2, w // 2)
    y = torch.randn(1, chs, h, w) * 3.0
    return mcm, pred, y


@pytest.mark.parametrize("stages", [1, 2, 4])
def test_the_decoder_rebuilds_the_encoders_latent_from_the_residual_alone(stages):
    """**The non-negotiable one**, at the level of the loop itself.

    The encoder is given `y` and produces `r̂`; the decoder is given that `r̂` and
    nothing else, and must arrive at the same `ŷ`. Tested at every stage count because
    the failure mode is stage-dependent: a 1-stage model has no history to get wrong.
    """
    mcm, pred, y = _loop(stages=stages)
    enc = mcm.reconstruct(pred, y=y, ste=False)
    dec = mcm.reconstruct(pred, r_hat=enc["r_hat"], ste=False)
    assert torch.equal(dec["y_hat"], enc["y_hat"])
    assert torch.equal(dec["means"], enc["means"])
    assert torch.equal(dec["r_hat"], enc["r_hat"])


def test_the_residual_is_integral_and_the_latent_is_mean_plus_residual():
    """`ŷ = r̂ + ctx` with `r̂` an integer field is the whole of eqs (1)/(2) on this
    branch. If `r̂` were not integral the arithmetic coder would round it a second
    time and the decoder's `ŷ` would differ by that rounding."""
    mcm, pred, y = _loop()
    out = mcm.reconstruct(pred, y=y, ste=False)
    assert torch.equal(out["r_hat"], torch.round(out["r_hat"]))
    assert torch.allclose(out["y_hat"], out["r_hat"] + out["means"], atol=0)


def test_reconstruct_takes_exactly_one_direction():
    """Not a defensive nicety: a default would turn a forgotten argument into a
    silent encoder-only reconstruction, which is the one bug this phase cannot see."""
    mcm, pred, y = _loop()
    with pytest.raises(ValueError, match="exactly one of"):
        mcm.reconstruct(pred)
    with pytest.raises(ValueError, match="exactly one of"):
        mcm.reconstruct(pred, y=y, r_hat=y)


def test_a_post_shuffle_prediction_is_caught_with_a_useful_message():
    mcm, pred, y = _loop()
    with pytest.raises(ValueError, match="shuffle=True"):
        mcm.reconstruct(pred[:, :32], y=y)


# ---------------------------------------------------------------------------
# the dependency graph — is the context actually being used?
# ---------------------------------------------------------------------------
def _sharpen(mcm, gain=100.0):
    """Undo the near-identity init so the networks are genuinely sensitive.

    These tests are about *wiring*, not about initialisation: at `init_gain=0.01` a
    disconnected `gather` and a connected one differ by a number small enough that a
    tolerance would have to be guessed. Scaling the last layer back to its default
    makes "does coset 2 respond to coset 0 at all" a question about the graph.
    """
    with torch.no_grad():
        for net in mcm.nets:
            net.fuse[-1].weight.mul_(gain)
    return mcm


def _means_per_coset(mcm, pred, y):
    return split_cosets(mcm.reconstruct(pred, y=y, ste=False)["means"], mcm.order)


def test_the_first_coset_is_predicted_from_the_hyper_decoder_alone():
    """Stage 0 is the one MCM cannot help, and that must be exact, not approximate:
    it is what the decoder can compute before it has reconstructed anything."""
    mcm, pred, y = _loop()
    _sharpen(mcm)
    a = _means_per_coset(mcm, pred, y)
    b = _means_per_coset(mcm, pred, y + 50.0)
    assert torch.equal(a[0], b[0])


def test_a_later_coset_sees_an_earlier_one():
    """The structural half of "if < 2%, context nets aren't seeing previous groups".

    Perturbing the latent inside coset 0 must move the context of every coset in a
    later stage. A `gather` that was built but never reached would leave all four
    unchanged and still train, code and decode.
    """
    mcm, pred, y = _loop()
    _sharpen(mcm)
    bumped = y.clone()
    bumped[..., 0::2, 0::2] += 20.0                       # coset (0,0) only
    a, b = _means_per_coset(mcm, pred, y), _means_per_coset(mcm, pred, bumped)
    assert torch.equal(a[0], b[0]), "stage 0 must not see its own coset's latent"
    for k in (1, 2, 3):
        assert not torch.equal(a[k], b[k]), f"coset {mcm.order[k]} ignored coset (0,0)"


def test_cosets_in_the_same_stage_do_not_see_each_other():
    """What makes a stage one parallel pass. With 2 stages, `(1,1)` is coded beside
    `(0,0)`, so it must not condition on it — otherwise the "4 stages regardless of
    image size" complexity claim is a claim about a model nobody is running."""
    mcm, pred, y = _loop(stages=2)
    _sharpen(mcm)
    bumped = y.clone()
    bumped[..., 0::2, 0::2] += 20.0
    a, b = _means_per_coset(mcm, pred, y), _means_per_coset(mcm, pred, bumped)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[2], b[2]) and not torch.equal(a[3], b[3])


def test_one_stage_is_phase_five_plus_a_refiner():
    """The zero point the "4-9% over Phase 5" claim is measured against. With one
    stage nothing may depend on `y` at all, which is exactly `SplitHyperBranch`'s
    invariant — so the 1-stage ablation really is Phase 5 with extra convolutions,
    not a differently-initialised model that happens to be nearby."""
    mcm, pred, y = _loop(stages=1)
    _sharpen(mcm)
    a = _means_per_coset(mcm, pred, y)
    b = _means_per_coset(mcm, pred, torch.randn_like(y) * 7.0)
    assert all(torch.equal(x, z) for x, z in zip(a, b))


# ---------------------------------------------------------------------------
# ContextStage
# ---------------------------------------------------------------------------
def test_the_context_is_a_correction_to_the_hyper_prediction():
    """At init `ctx ≈ p̈` to a few digits, so an untrained MCM model codes what Phase
    5's model codes and the run does not open with a rate explosion. Scaled down
    rather than zeroed — see the next test for why."""
    stage = ContextStage(32, 2).eval()
    pred = torch.randn(1, 32, 6, 6)
    out = stage(pred, [torch.randn(1, 32, 6, 6), torch.randn(1, 32, 6, 6)])
    assert torch.allclose(out, pred, atol=5e-2)
    assert not torch.equal(out, pred)


def test_the_last_fusion_layer_is_small_but_not_zero():
    """An exact zero leaves every layer above it with zero gradient on step 0, and
    `selftest`'s "every parameter receives a finite non-zero gradient" check would
    fail — correctly, because that state is indistinguishable from a network that was
    never wired up."""
    stage = ContextStage(32, 1)
    assert float(stage.fuse[-1].weight.detach().abs().max()) > 0
    assert torch.equal(stage.fuse[-1].bias.detach(),
                       torch.zeros_like(stage.fuse[-1].bias))

def test_stage_zero_has_no_gather_network_at_all():
    """Registered as `None`, not as an identity, so `print(model)` shows the asymmetry
    — the first stage is the one MCM cannot help, and that is the shape of the model."""
    assert ContextStage(32, 0).gather is None
    assert ContextStage(32, 1).gather is not None


def test_the_gather_collapses_channels_before_it_mixes_space():
    """`conv1x1(k*chs, chs)` then grouped `conv3x3(chs, chs)`. The ordering is what
    makes the per-stage cost nearly independent of the stage index — a 3x3 straight
    onto `k*chs` channels would cost `k` times as much, and "four stages at fixed
    complexity" would stop being true down the chain."""
    for k in (1, 2, 3):
        g = ContextStage(32, k).gather
        assert (g[0].in_channels, g[0].out_channels, g[0].kernel_size) == \
               (k * 32, 32, (1, 1))
        assert (g[-1].in_channels, g[-1].out_channels, g[-1].kernel_size) == \
               (32, 32, (3, 3))
        assert g[-1].groups == chs2group(32)


def test_a_stage_refuses_a_history_of_the_wrong_length():
    with pytest.raises(ValueError, match="out of step"):
        ContextStage(32, 2)(torch.randn(1, 32, 4, 4), [torch.randn(1, 32, 4, 4)])


# ---------------------------------------------------------------------------
# MCMBranch — the prediction arrives pre-split
# ---------------------------------------------------------------------------
def test_the_hyper_decoder_stops_before_the_shuffle():
    """`/32` is exactly one coset's grid, so the four `chs`-wide slices of the
    pre-shuffle tensor *are* the four per-coset predictions. Nothing is upsampled and
    every context convolution runs at a quarter of the latent's area."""
    br, _ = _branch(chs=32)
    assert br.h_s.shuffle is False
    z_hat = torch.randn(1, 32, 2, 3)                      # /64
    pred = br.h_s(z_hat)
    assert tuple(pred.shape) == (1, 4 * 32, 4, 6)         # [4*chs, /32]


def test_shuffle_off_and_on_are_the_same_parameters_rearranged():
    """`PixelShuffle` has no weights, so a Phase 5 checkpoint's `h_s` loads into a
    Phase 6 model unchanged. Tested as an equality of tensors, not of key names: the
    shuffled output must be exactly the pixel-shuffle of the unshuffled one."""
    torch.manual_seed(0)
    on = HyperDecoder(32)
    off = HyperDecoder(32, shuffle=False)
    off.load_state_dict(on.state_dict())
    z_hat = torch.randn(1, 32, 2, 3)
    with torch.no_grad():
        assert torch.equal(on(z_hat), torch.pixel_shuffle(off(z_hat), 2))


def test_predict_returns_no_mean_on_purpose():
    """The mean is only defined once earlier stages exist, so any caller has to go
    through `reconstruct()` and therefore has to declare which side of the codec it
    is. A stale `p̈` here would be silently codeable."""
    br, _ = _branch()
    p = br.predict(torch.randn(1, 32, 2, 2))
    assert p["means"] is None
    assert tuple(p["pred"].shape) == (1, 128, 4, 4)


def test_sigma_still_comes_from_the_scale_decoder_and_zhat_alone():
    """σ is *not* context-modelled, and that is the load-bearing design decision of
    the phase: it is what lets the whole residual field be decoded in one pass. If σ
    ever gained a `y` dependence the entropy engine would have to stop four times and
    wait for the accelerator, and §VI-E's decoupling would be gone."""
    br, _ = _branch()
    z_hat = torch.randn(1, 32, 2, 2)
    a = br.predict(z_hat, quantise=True)
    b = br.predict(z_hat, quantise=True)
    assert torch.equal(a["scales"], b["scales"]) and torch.equal(a["rows"], b["rows"])
    assert a["rows"].shape == a["scales"].shape
    assert br.h_scale is not None and br.fused is False


def test_a_phase_five_checkpoint_warm_starts_a_phase_six_model():
    """The reason the near-identity init is worth the trouble. Every Phase 5 parameter
    must land, by name and by shape, and the only thing missing must be MCM itself."""
    from jpegai.models.hyper import SplitHyperBranch
    idx = SigmaIndex()
    old = SplitHyperBranch(32, 32, sigma_index=idx, fused=False)
    new = MCMBranch(32, 32, sigma_index=idx)
    report = new.load_state_dict(old.state_dict(), strict=False)
    assert report.unexpected_keys == []
    assert all(k.startswith("mcm.") for k in report.missing_keys)
    assert report.missing_keys, "nothing was added? then this is not MCM"
    for k, v in old.state_dict().items():
        assert torch.equal(new.state_dict()[k], v), k


def test_the_warm_started_mean_field_is_phase_fives_mean_field():
    """The warm start measured in means, not in tensor names.

    `test_a_phase_five_checkpoint_warm_starts_a_phase_six_model` proves every weight
    landed; it cannot see whether the weights are *used* the way Phase 5 used them.
    With `pred.chunk(4)` in `reconstruct` that test still passes while the mean field
    comes out permuted -- two of the four cosets fed the wrong prediction -- so the
    Phase 6 run opens from a worse point than the checkpoint it inherited. The only
    detector is to compute both mean fields and compare them.

    Not `torch.equal`: the near-identity init is deliberately near, so each stage adds
    its 1%-scaled fusion output. The bound is on the *relative* deviation, which is
    what "the warm start starts where Phase 5 stopped" actually claims.
    """
    from jpegai.models.hyper import SplitHyperBranch
    idx = SigmaIndex()
    torch.manual_seed(0)
    old = SplitHyperBranch(32, 32, sigma_index=idx, fused=False).eval()
    new = MCMBranch(32, 32, sigma_index=idx).eval()
    new.load_state_dict(old.state_dict(), strict=False)

    z_hat = torch.randn(1, 32, 4, 4)
    y = torch.randn(1, 32, 16, 16)
    with torch.no_grad():
        p5 = old.h_s(z_hat)                                  # [chs, /16], shuffled
        p6 = new.mcm.reconstruct(new.h_s(z_hat), y=y)["means"]
    assert p5.shape == p6.shape
    rel = (p6 - p5).abs().max() / p5.abs().max()
    assert rel < 0.05, f"the warm start is not near-identity: {rel:.3f} relative"

    # And the permuted alternative is off by whole feature maps, not by 1%. This is the
    # number that was actually observed with `chunk`, and it is why the bound above is
    # meaningful rather than vacuous.
    with torch.no_grad():
        wrong = join_cosets(list(new.h_s(z_hat).chunk(4, dim=-3)))
    assert (wrong - p5).abs().max() / p5.abs().max() > 0.2


# ---------------------------------------------------------------------------
# the branch through a real arithmetic coder
# ---------------------------------------------------------------------------
def test_the_loss_and_the_synthesis_transform_see_the_same_latent():
    """`forward` returns two `ŷ`s by two routes — the loop's own, and the coder's
    `mean + quantise(y - mean)`. They must be the same tensor, or the rate the loss
    optimises and the latent the decoder reconstructs have quietly come apart."""
    br, gc = _branch()
    torch.manual_seed(0)
    out = br(torch.randn(1, 32, 8, 8) * 3.0, gc, noise=False, ste=True)
    assert torch.equal(out["y_hat"], out["mcm_y_hat"])
    assert out["means"] is not None
    assert torch.equal(out["r_hat"], torch.round(out["r_hat"]))


def test_the_branch_survives_the_bitstream_bit_exactly():
    br, gc = _branch()
    gc.update(force=True)
    br.entropy_bottleneck.update(force=True)
    torch.manual_seed(0)
    y = torch.randn(1, 32, 8, 12) * 3.0
    packet = br.compress(y, gc)
    dec = br.decompress(packet, gc, y.device)
    ref = br(y, gc, noise=False, ste=True)
    assert torch.equal(dec["y_hat"], ref["y_hat"])
    assert torch.equal(dec["r_hat"], ref["r_hat"])


def test_the_entropy_decoder_never_enters_the_loop():
    """§VI-E's decoupling, tested as a property of the decoder's *shape*.

    The residual field comes off the bitstream in **one** call with `means=None`, and
    only then do the four stages turn `r̂` into `ŷ`. A design that fed each stage's
    context back into the coder would produce identical pixels and would destroy the
    self-contained entropy engine the whole section exists to build — so the pixels
    cannot be the test. The call is.
    """
    br, gc = _branch()
    gc.update(force=True)
    br.entropy_bottleneck.update(force=True)
    torch.manual_seed(0)
    y = torch.randn(1, 32, 8, 12) * 3.0
    packet = br.compress(y, gc)

    calls = []
    real = gc.decompress

    def spy(streams, scales, means=None, **kw):
        calls.append(means)
        return real(streams, scales, means, **kw)

    gc.decompress = spy
    try:
        br.decompress(packet, gc, y.device)
    finally:
        gc.decompress = real
    assert len(calls) == 1, f"the coder was re-entered {len(calls)} times"
    assert calls[0] is None, "the coder was handed a mean, so it waited for a network"


def test_the_packet_has_one_y_stream_per_branch_exactly_as_phase_five():
    """MCM changes the prediction, not the layout. If the residual had to be coded as
    four substreams then `stream_bytes`, `packet_bytes`, `estimated_bits` and the rate
    gate would all need Phase 6 versions; they do not, and this is why."""
    br, gc = _branch()
    gc.update(force=True)
    br.entropy_bottleneck.update(force=True)
    packet = br.compress(torch.randn(1, 32, 8, 8) * 3.0, gc)
    assert sorted(packet) == ["y_strings", "z_shape", "z_strings"]
    assert len(packet["y_strings"]) == 1 and len(packet["z_strings"]) == 1


def test_every_parameter_receives_a_finite_non_zero_gradient():
    """`selftest`'s standing check, applied to the new module while it is small enough
    to name the offender. This is the test the exact-zero init would have failed."""
    br, gc = _branch()
    torch.manual_seed(0)
    out = br(torch.randn(1, 32, 8, 8) * 3.0, gc, noise=True, ste=True)
    loss = out["y_hat"].square().mean() - out["y_lik"].clamp_min(1e-9).log().mean()
    loss.backward()
    for name, p in br.mcm.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0, name


# ---------------------------------------------------------------------------
# the whole codec
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", FMT_NAMES)
@pytest.mark.parametrize("h,w", [(128, 192), (192, 128)])
def test_the_decoder_rebuilds_the_encoders_latent(fmt, h, w):
    """**The non-negotiable one**, end to end: through the chroma subsampling, both
    branches, the shared σ table and a real range coder.

    Parameterised over the anisotropic formats and over a non-square input in both
    orientations because the coset split is the one place in the codec where an `h`/`w`
    transposition is invisible: `x[..., a::2, b::2]` with `a` and `b` swapped is a
    perfectly well-formed partition, just not the same one on both sides.
    """
    torch.manual_seed(0)
    m = _codec(fmt)
    m.update(force=True)
    x = torch.rand(1, 3, h, w)
    packet = m.compress(x)
    decoded = m.decompress(packet)
    reference = m(x, noise=False, ste=True)
    assert torch.equal(decoded["y_hat"], reference["y_hat"])
    assert torch.equal(decoded["y_uv_hat"], reference["y_uv_hat"])


@pytest.mark.parametrize("stages", [1, 2, 4])
def test_every_stage_count_codes_and_decodes(stages):
    """The ablation has to be runnable, not just constructible."""
    torch.manual_seed(0)
    m = _codec(stages=stages)
    m.update(force=True)
    x = torch.rand(1, 3, 64, 64)
    decoded = m.decompress(m.compress(x))
    assert torch.equal(decoded["y_hat"], m(x, noise=False, ste=True)["y_hat"])
    assert m.branch_y.mcm.stages == stages


def test_the_stream_layout_is_unchanged_from_phase_five():
    """Same four streams, same names. `stream_bytes`, `packet_bytes`,
    `estimated_bits`, `gate_branches` and the rate gate therefore need no Phase 6
    version — which is the practical payoff of σ not being context-modelled."""
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    keys = []
    for mcm in (False, True):
        m = _codec(mcm=mcm)
        m.update(force=True)
        keys.append(sorted(m.stream_bytes(m.compress(x))))
        assert sorted(m(x)["likelihoods"]) == sorted(keys[-1])
    assert keys[0] == keys[1] == ["y", "y_uv", "z", "z_uv"]


def test_mcm_is_on_luma_only():
    """`entropy.mcm_on_secondary: false`. The paper's reason is that chroma is already
    predicted from luma by eq. (3), so a context model there would be spending
    sequential passes on the branch that has the least left to gain."""
    m = _codec()
    assert isinstance(m.branch_y, MCMBranch)
    assert not isinstance(m.branch_uv, MCMBranch)
    assert getattr(m.branch_uv, "mcm", None) is None
    assert m.branch_uv.h_s.shuffle is True


def test_luma_only_decoding_still_works_and_still_pays_for_mcm():
    """MCM sits on the branch a luma-only decoder does decode, so the complexity
    claim for that mode has to include it — and the mode has to keep working."""
    torch.manual_seed(0)
    m = _codec()
    m.update(force=True)
    packet = m.compress(torch.rand(1, 3, 128, 128))
    full = m.decompress(packet)
    partial = m.decompress(packet, luma_only=True)
    assert torch.equal(partial["y_hat"], full["y_hat"])
    assert partial["luma_only"] is True and partial["y_uv_hat"] is None


# ---------------------------------------------------------------------------
# the complexity claim: four passes, regardless of image size
# ---------------------------------------------------------------------------
def _count_stage_calls(m, x):
    """How many times a `ContextStage` runs during one **decode**.

    The packet is built before the hooks go on, because the encoder runs the identical
    loop and counting both would report eight passes for a decoder that makes four.
    """
    packet = m.compress(x)
    calls = []
    handles = [net.register_forward_hook(lambda *a: calls.append(1))
               for net in m.branch_y.mcm.nets]
    try:
        m.decompress(packet)
    finally:
        for h in handles:
            h.remove()
    return len(calls)


def test_the_number_of_network_passes_does_not_grow_with_the_image():
    """The acceptance criterion behind "decode-time vs megapixels: two parallel lines".

    The wall-clock version of this needs a GPU and a plot; the structural version does
    not, and it is the one that can regress silently. An autoregressive context model
    would run one pass per *sample* and would still decode correctly — the only symptom
    is a decoder that is unusable at 4K, which no correctness test would ever see.
    """
    torch.manual_seed(0)
    m = _codec()
    m.update(force=True)
    small = _count_stage_calls(m, torch.rand(1, 3, 64, 64))
    large = _count_stage_calls(m, torch.rand(1, 3, 256, 384))
    assert small == large == 4, (small, large)


@pytest.mark.parametrize("stages,passes", [(1, 4), (2, 4), (4, 4)])
def test_fewer_stages_means_fewer_sequential_passes_not_fewer_networks(stages, passes):
    """All four networks always run; `stages` changes how many of those runs have to
    wait for each other. So the ablation trades latency for prediction quality at a
    fixed parameter count, which is the comparison the plan asks for."""
    torch.manual_seed(0)
    m = _codec(stages=stages)
    m.update(force=True)
    assert _count_stage_calls(m, torch.rand(1, 3, 64, 64)) == passes
    assert m.branch_y.mcm.stages == stages


def test_mcm_is_its_own_mac_bucket():
    """"Decode cost grows by a constant four passes" is only checkable if MCM is a
    number. Folded into `h_s_y` the claim would be unfalsifiable — the same reason the
    scale decoders got their own rows in Phase 5."""
    from jpegai.utils import macs_breakdown
    m = _codec()
    parts = m.summary_parts()
    names = [n for n, _, _ in parts]
    assert "mcm_y" in names and "mcm_uv" not in names
    rows = macs_breakdown(m, (1, 3, 128, 128),
                          parts=[(n, s) for n, s, _ in parts])
    assert rows["mcm_y"] > 0


def test_the_title_names_the_stage_count():
    """`summary_title` ends up in logs and in checkpoint metadata, where "split-hyper"
    alone would not distinguish a 4-stage run from a 1-stage ablation."""
    assert "+mcm4" in _codec(stages=4).summary_title()
    assert "+mcm1" in _codec(stages=1).summary_title()
    assert "mcm" not in _codec(mcm=False).summary_title()


# ---------------------------------------------------------------------------
# construction: the guards, and the kind strings that are on-disk format
# ---------------------------------------------------------------------------
def test_mcm_needs_phase_fives_split_branch():
    """MCM refines `p̈` and leaves `Iσ` alone, which only exists as a separate output
    on the split path."""
    with pytest.raises(ValueError, match="split branch"):
        _codec(mcm=True, split_hyper=False)


def test_mcm_and_the_single_hyper_decoder_ablation_are_alternatives_not_a_stack():
    """The fused decoder emits `[2*chs, /16]` with Iσ folded in; there is no coset
    structure in it for MCM to read. Refused rather than silently mis-sliced."""
    with pytest.raises(ValueError, match="alternatives, not a stack"):
        _codec(mcm=True, fused_hyper=True)


def test_the_kind_strings_exist_and_carry_the_stage_count():
    """`meta["model"]` is all `jpegai.eval.neural` has when it rebuilds an
    architecture from a `.pt` months later. A stage count kept *outside* the string
    would load a 4-stage model from 2-stage weights and the shapes would fit, because
    the ablation deliberately changes only the conditioning — so the digit is in the
    name."""
    assert MCM_STAGES == {"twobranch-mcm": 4, "twobranch-mcm2": 2, "twobranch-mcm1": 1}
    for kind in MCM_STAGES:
        assert kind in KINDS


def test_the_phase_four_and_five_kind_strings_still_mean_what_they_meant():
    """These strings are on-disk format. Redefining one turns an existing checkpoint
    into wrong weights rather than into a load error."""
    for kind in ("scale", "mean-scale", "twobranch", "twobranch-split",
                 "twobranch-fused"):
        assert kind in KINDS


@pytest.mark.parametrize("kind,stages", sorted(MCM_STAGES.items()))
def test_build_any_model_routes_each_kind_to_the_right_stage_count(kind, stages):
    from jpegai.config import load_config
    m = build_any_model(load_config("tierA"), kind)
    assert isinstance(m.branch_y, MCMBranch)
    assert m.branch_y.mcm.stages == stages
    assert m.branch_y.mcm.chs == m.luma_latent == 96
    assert chs2group(m.luma_latent) == 3
    assert m.branch_y.mcm.order == GROUP_ORDER


def test_an_unknown_kind_lists_the_ones_that_exist():
    from jpegai.config import load_config
    with pytest.raises(ValueError, match="twobranch-mcm"):
        build_any_model(load_config("tierA"), "twobranch-mcm3")
