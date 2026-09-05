"""Phase 8: Table II's staged schedule, and the gain unit's neutrality.

Two kinds of test here, and the second kind is the reason the file exists.

The cheap ones cover `jpegai.train.stages` as arithmetic and bookkeeping: the part
vocabulary, the apportionment, which beta eq. (9) anchors on. Those are unit tests in
the ordinary sense -- they would catch a typo in Table II's transcription.

The expensive one (`test_zero_gain_is_byte_identical`) is the real gate. Table II's
stage IV bolts a gain unit onto an already-trained checkpoint, so the *whole premise*
of the phase is that a zero gain vector at `Delta_beta = 0` is the exact identity: if
it is not, every variable-rate result is measured against a codec that is already
different from the Phase 5 ladder it warm-started from, and the comparison is
meaningless. The training loop's `delta_beta_check` tests this arithmetically, on the
offset. This tests it *end to end*, on the bytes, against a separately built
`twobranch-split` model with matched weights -- which is the claim that actually
matters and the one that cannot be true by construction.
"""

from __future__ import annotations

import math

import pytest
import torch

from jpegai.config import load_config
from jpegai.models import build_any_model
from jpegai.train.stages import (ALIASES, LOSSES, PARTS, STAGE_NAMES, Stage,
                                 anchor_beta, apply_freeze, aux_is_trained,
                                 canonical_part, canonical_parts, check_partition,
                                 find_stage, model_entry, model_ids, part_modules,
                                 part_parameters, sample_delta_beta, schedule,
                                 stage_steps, steps_for)

KINDS_UNDER_TEST = ("scale", "mean-scale", "twobranch", "twobranch-split",
                    "twobranch-fused", "twobranch-mcm", "twobranch-mcm1",
                    "twobranch-vr")


@pytest.fixture(scope="module")
def cfg():
    return load_config("tierA")


@pytest.fixture(scope="module")
def models(cfg):
    """Every kind, built once. Building `twobranch-mcm` is not cheap."""
    return {k: build_any_model(cfg, k) for k in KINDS_UNDER_TEST}


@pytest.fixture(scope="module")
def grid(cfg):
    """eq. (10)'s two unit-conversion constants, `S_sigma` and `P_beta`.

    Built from the config rather than from a model, because the arithmetic tests below
    are about the *units* and building a codec to read two floats off it would make
    them depend on the architecture they are meant to be independent of.
    """
    from jpegai.models.hyper import SigmaIndex

    ent = cfg.entropy
    si = SigmaIndex(minimum=ent.sigma_quant_min, maximum=ent.sigma_quant_max,
                    levels=ent.sigma_quant_level, precision=ent.sigma_precision)
    return {"log_k": si.log_k, "step": si.step}


# ---------------------------------------------------------------------------
# The part vocabulary
# ---------------------------------------------------------------------------
def test_aliases_resolve():
    """The reference software's `--frozen_part` words and the paper's long form.

    Both vocabularies appear in documents this project is read against, so both have to
    be accepted -- `analysis` is what `13-training.md` calls the encoder.
    """
    assert canonical_part("analysis") == "encoder"
    assert canonical_part("synthesis") == "decoder"
    assert canonical_part("gain_unit") == "gain"
    assert canonical_part("entropy network") == "entropy"
    assert canonical_part("  ENCODER ") == "encoder"
    for alias, target in ALIASES.items():
        assert canonical_part(alias) == target
        assert target in PARTS


def test_unknown_part_raises():
    with pytest.raises(ValueError, match="unknown codec part"):
        canonical_part("hyper")


def test_canonical_parts_is_order_independent():
    """Two spellings of one stage must compare equal.

    `canonical_parts` sorts into `PARTS` order and de-duplicates, so a stage read off
    the paper ("decoder, entropy network, gain unit") equals one read off the config
    (`[gain, entropy, decoder]`). Without this, `Stage.__eq__` on a frozen dataclass
    would call two identical stages different and the schedule would be untestable.
    """
    a = canonical_parts(["decoder", "entropy network", "gain_unit"])
    b = canonical_parts(["gain", "entropy", "decoder"])
    assert a == b == ("decoder", "entropy", "gain")
    assert canonical_parts(["gain", "gain", "gain_unit"]) == ("gain",)
    assert canonical_parts([]) == ()


# ---------------------------------------------------------------------------
# The partition invariant -- the one that has no symptom when it breaks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
def test_parts_partition_every_model(models, kind):
    """`training_parts()` covers every parameter exactly once, on every kind.

    Freezing is the complement of the stage's part list, so a parameter in no bucket is
    frozen in *every* stage: it trains nowhere, the loss still falls, and the codec is
    quietly worse. There is no test that catches that downstream, which is why this one
    runs over every architecture rather than just the variable-rate one.
    """
    counts = check_partition(models[kind])
    assert set(counts) == set(PARTS)
    assert sum(counts.values()) == len(list(models[kind].parameters()))


@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
def test_gain_bucket_present_and_empty_without_a_gain_unit(models, kind):
    """`gain` is present-and-empty on a fixed-rate model, not absent.

    So `--stage IV --model twobranch-split` fails on "this stage trains nothing", which
    names the actual problem, rather than on a `KeyError: 'gain'` from inside the freeze
    machinery.
    """
    mods = part_modules(models[kind])
    assert "gain" in mods
    assert bool(mods["gain"]) == (kind == "twobranch-vr")
    if kind == "twobranch-vr":
        # One vector per branch, and JPEG AI's simplification is that it is *one*
        # vector, not the reference software's 18-entry interpolated table.
        assert len(mods["gain"]) == 2
        assert len(part_parameters(models[kind], ["gain"])) == 2


def test_entropy_owns_the_hyper_analysis(models):
    """`h_a` is entropy, not encoder -- `13-training.md` section 8.4, verbatim.

    The section lists `Net.parameters()`'s four groups, and the `entropy` group is
    "both hyper-encoders, both hyper-decoders, both hyper-scale-decoders, both hyper
    entropy models, context_Y". This decides what Table II's stage III ("decoder,
    entropy network") actually trains, and getting it the other way round would freeze
    two networks the standard's own schedule fine-tunes.
    """
    labels = {p: {lab for lab, _ in mods}
              for p, mods in part_modules(models["twobranch-vr"]).items()}
    assert {"h_a_y", "h_a_uv"} <= labels["entropy"]
    assert labels["encoder"] == {"g_a_y", "g_a_uv"}
    assert labels["decoder"] == {"g_s_y", "g_s_uv"}
    assert not any(lab.startswith("h_") for lab in labels["encoder"])


def test_mcm_lands_in_entropy(models):
    """Same section: `context_Y` is in the entropy group.

    Worth its own assertion because MCM sits between the hyper decoder and the coder
    and could plausibly have been filed under either -- and stage III trains entropy,
    so the choice is load-bearing.
    """
    labels = {lab for lab, _ in part_modules(models["twobranch-mcm"])["entropy"]}
    assert "mcm_y" in labels


@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
def test_apply_freeze_agrees_with_part_parameters(models, kind):
    """The two halves of freezing must select the same tensors.

    `apply_freeze` clears `requires_grad`; `part_parameters` is what the loop hands to
    Adam. If they disagreed, the optimiser would hold a tensor whose gradient is always
    `None` -- which torch tolerates silently, and then weight decay or a momentum buffer
    can still move a weight the schedule says is frozen.
    """
    model = models[kind]
    try:
        for parts in ([], ["gain"], ["decoder"], ["decoder", "entropy"],
                      ["decoder", "entropy", "gain"], list(PARTS)):
            apply_freeze(model, parts)
            on = {id(p) for p in model.parameters() if p.requires_grad}
            assert on == {id(p) for p in part_parameters(model, parts)}
    finally:
        apply_freeze(model, PARTS)   # leave the fixture trainable for other tests


def test_freeze_is_reversible(models):
    """A stage must not permanently disable anything.

    Stages chain -- IV freezes almost everything, and the next model's stage I trains
    the lot -- so `apply_freeze(model, PARTS)` has to restore every tensor. It is set
    rather than cleared for exactly this reason.
    """
    model = models["twobranch-vr"]
    apply_freeze(model, ["gain"])
    assert sum(p.requires_grad for p in model.parameters()) == 2
    apply_freeze(model, PARTS)
    assert all(p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------------------
# Table II itself
# ---------------------------------------------------------------------------
def test_schedule_matches_table_ii(cfg):
    """The paper's four rows, transcribed and checked against the config.

    Hardcoded here on purpose. The config is the editable copy and this is the
    assertion that it still says what arXiv 2503.16288 p. 9 says, so a future edit to
    `tierA.yaml` that changes the schedule has to change this file too and therefore
    has to be deliberate.
    """
    expect = {
        0: (0.002, 0.002, ("decoder", "entropy"), ("gain",)),
        1: (0.007, 0.03, ("decoder", "entropy", "gain"), ("decoder", "gain")),
        2: (0.075, 0.2, ("decoder", "entropy", "gain"), ("decoder", "gain")),
        3: (0.5, 1.0, ("decoder", "entropy", "gain"), ("decoder", "gain")),
    }
    assert set(model_ids(cfg)) == set(expect)
    for mid, (b_early, b_late, s3, s4) in expect.items():
        one, two, three, four = schedule(cfg, mid)
        assert [s.name for s in (one, two, three, four)] == list(STAGE_NAMES)
        # Stages I and II are the same row twice apart from the loss column.
        assert one.beta == two.beta == pytest.approx(b_early)
        assert (one.loss, two.loss) == ("mse", "mix")
        assert one.parts == two.parts == ("encoder", "decoder", "entropy")
        # III and IV are where the models differ, and both differences are the config's.
        assert three.beta == four.beta == pytest.approx(b_late)
        assert (three.loss, four.loss) == ("mix", "mix")
        assert three.parts == s3 and four.parts == s4
        assert [s.epochs for s in (one, two, three, four)] == [64, 32, 20, 12]


def test_model_zero_is_the_exception(cfg):
    """Model 0 does not raise beta and trains the gain unit alone in stage IV.

    Both because it "is designed to cover the low bit rate range": there is nothing
    below it, so there is no raised beta to adapt to, and the backbone it inherits is
    already the one the ladder anchors on.
    """
    m0 = model_entry(cfg, 0)
    assert m0["beta_train"] == m0["beta_stage34"]
    assert find_stage(cfg, 0, "IV").parts == ("gain",)
    assert find_stage(cfg, 0, "III").parts == ("decoder", "entropy")
    for mid in (1, 2, 3):
        assert model_entry(cfg, mid)["beta_stage34"] > model_entry(cfg, mid)["beta_train"]
        assert "decoder" in find_stage(cfg, mid, "IV").parts


def test_anchor_is_stage_ones_beta(cfg, grid):
    """eq. (9) divides by stage I's beta, not the stage being run.

    The paper's reason is the epoch count -- stage I is 64 of the 128 epochs, "and thus
    the model is most adapted to the betatrain in this stage". Using model 1's raised
    0.03 would shift the whole ladder by `beta_displacement(0.03, 0.007) = +929`, which
    is past the +702 clamp: every rate point would collapse onto the top rung.
    """
    from jpegai.models.gain import DELTA_BETA_MAX, beta_displacement

    for mid in model_ids(cfg):
        assert anchor_beta(cfg, mid) == pytest.approx(model_entry(cfg, mid)["beta_train"])
        assert anchor_beta(cfg, mid) == pytest.approx(find_stage(cfg, mid, "I").beta)
    assert anchor_beta(cfg, 1) == pytest.approx(0.007)
    assert find_stage(cfg, 1, "III").beta == pytest.approx(0.03)
    # `clip=False` on purpose: with the clamp on, the mistake is invisible -- it saturates
    # at +702 and looks like a legitimate top rung. The unclamped value is what shows the
    # ladder has been pushed off the end.
    wrong = beta_displacement(find_stage(cfg, 1, "III").beta, anchor_beta(cfg, 1),
                              clip=False, **grid)
    assert wrong > DELTA_BETA_MAX
    assert wrong == 929          # the number quoted in `anchor_beta`'s docstring


def test_model_entry_is_by_id_not_position(cfg):
    """Ids are the paper's model numbers and appear in checkpoint names.

    List order is a YAML detail. Looking up by position would silently follow a
    different schedule the first time the config is reordered.
    """
    assert model_entry(cfg, 3)["beta_train"] == pytest.approx(0.5)
    with pytest.raises(KeyError, match="no model with id"):
        model_entry(cfg, 7)


def test_stage_rejects_nonsense():
    with pytest.raises(ValueError, match="stage name"):
        Stage("V", 0.002, "mix", 12, ("gain",))
    with pytest.raises(ValueError, match="loss must be"):
        Stage("I", 0.002, "mixed", 12, ("gain",))
    with pytest.raises(ValueError, match="trains nothing"):
        Stage("IV", 0.002, "mix", 12, ())
    with pytest.raises(KeyError, match="no stage"):
        find_stage(load_config("tierA"), 0, "V")


def test_frozen_is_the_complement(cfg):
    for mid in model_ids(cfg):
        for st in schedule(cfg, mid):
            assert set(st.parts) | set(st.frozen) == set(PARTS)
            assert not set(st.parts) & set(st.frozen)


def test_loss_kwargs_only_overrides_stage_one(cfg):
    """Stage I is "MSE" in Table II, so the configured MS-SSIM weight must be zeroed.

    Stages II-IV are "Mix" and inherit `train.loss.ms_ssim` -- which is the knob the
    reference software makes rate-dependent (0.5 / 0.5 / 0.4 / 0.3 per model) and we
    keep flat. Passing no override is what lets that stay a Phase 12 experiment rather
    than a value hardcoded in two places.
    """
    one, two, three, four = schedule(cfg, 0)
    assert one.loss_kwargs() == {"ms_ssim_weight": 0.0}
    assert two.loss_kwargs() == three.loss_kwargs() == four.loss_kwargs() == {}
    assert set(LOSSES) == {"mse", "mix"}


def test_aux_optimiser_follows_the_entropy_bucket(cfg):
    """The quantiles must stop moving when the entropy network is frozen.

    They decide how wide `update()` builds the CDF table. Letting them drift under a
    density that is no longer changing changes the *bytes* a frozen entropy model
    produces -- so a stage IV run would not reproduce the stage III checkpoint's rate
    even though Table II says stage IV does not touch the entropy model.
    """
    for mid in model_ids(cfg):
        one, two, three, four = schedule(cfg, mid)
        # `None` for the model: the question is answered from the part list alone, and
        # this asserts it stays that way -- if it ever needs the model, this breaks
        # loudly rather than reading a stale answer.
        assert aux_is_trained(None, one.parts) and aux_is_trained(None, two.parts)
        assert aux_is_trained(None, three.parts)     # III trains entropy for all four
        assert not aux_is_trained(None, four.parts)  # IV never does


# ---------------------------------------------------------------------------
# Epochs -> steps
# ---------------------------------------------------------------------------
def test_steps_for_sums_exactly(cfg):
    """Largest-remainder apportionment: the parts sum to the budget, always.

    Naive rounding loses or gains a few steps, which does not matter to the training --
    but a `--stage IV` whose step count disagrees with what the schedule reported makes
    two runs of "the same" schedule non-comparable, and that does.
    """
    sch = schedule(cfg, 0)
    for total in (0, 1, 7, 12, 100, 999, 12_345, 400_000, 1_000_003):
        per = steps_for(sch, total)
        assert sum(per) == total
        assert all(n >= 0 for n in per)
    # The paper's 64/32/20/12 over 400k, which is the number the loop prints.
    assert steps_for(sch, 400_000) == (200_000, 100_000, 62_500, 37_500)


def test_steps_for_preserves_the_epoch_ordering(cfg):
    """Stage I is half the run and stage IV a tenth -- that is what the epochs encode.

    The dataset here is not CTTC's 5264 sequences, so the absolute epoch counts do not
    transfer; the proportions are the transferable part.
    """
    sch = schedule(cfg, 0)
    per = steps_for(sch, 400_000)
    assert per[0] > per[1] > per[2] > per[3]
    denom = sum(s.epochs for s in sch)
    for st, n in zip(sch, per):
        assert n == pytest.approx(400_000 * st.epochs / denom, abs=1.0)


def test_stage_steps_agrees_with_steps_for(cfg):
    sch = schedule(cfg, 2)
    per = steps_for(sch, 250_000)
    for st, n in zip(sch, per):
        assert stage_steps(cfg, 2, st.name, 250_000) == n
        assert stage_steps(cfg, 2, st.name.lower(), 250_000) == n
    with pytest.raises(ValueError, match="non-negative"):
        steps_for(sch, -1)


# ---------------------------------------------------------------------------
# Delta_beta sampling  (OURS)
# ---------------------------------------------------------------------------
def test_sampled_delta_beta_is_an_in_range_integer(cfg):
    """Integer, because `Delta_beta` is a header field.

    A fractional value would train the gain vector against offsets no bitstream can
    ever request, so the vector would be optimal for a rate the coder cannot signal.
    """
    lo, hi = (int(v) for v in cfg.rate.beta_train_sample)
    g = torch.Generator().manual_seed(0)
    draws = [sample_delta_beta(cfg, g) for _ in range(400)]
    assert all(isinstance(d, int) and lo <= d <= hi for d in draws)
    # Both signs, and reaching most of the range -- a sampler stuck near zero would
    # pass the bounds check and teach the gain vector nothing.
    assert min(draws) < lo + 0.1 * (hi - lo) and max(draws) > hi - 0.1 * (hi - lo)
    assert any(d < 0 for d in draws) and any(d > 0 for d in draws)


def test_sample_range_is_inside_the_clamp(cfg):
    """`beta_train_sample` must not exceed what a bitstream can carry.

    It is deliberately narrower: the clamp's last few hundred are where `Isigma + o`
    runs off the end of the sigma table, and past that the gradient measures escapes
    rather than rate.
    """
    from jpegai.models.gain import DELTA_BETA_MAX, DELTA_BETA_MIN

    lo, hi = (int(v) for v in cfg.rate.beta_train_sample)
    assert DELTA_BETA_MIN <= lo < 0 < hi <= DELTA_BETA_MAX
    assert (lo, hi) != (DELTA_BETA_MIN, DELTA_BETA_MAX)


# ---------------------------------------------------------------------------
# The premise of the whole phase: a zero gain vector is the exact identity
# ---------------------------------------------------------------------------
@torch.no_grad()
def test_zero_gain_is_byte_identical(cfg):
    """A fresh `twobranch-vr` encodes the same bytes as a matched `twobranch-split`.

    This is Table II stage IV's precondition, stated end to end. Stage IV starts from a
    trained fixed-rate checkpoint and adds the gain unit; if adding it perturbed the
    bitstream at `Delta_beta = 0`, every variable-rate number in Phase 8 would be
    measured against a codec that already differs from the Phase 5 ladder it inherited,
    and the comparison would mean nothing.

    Stronger than the loop's `delta_beta_check`, which proves the same thing
    arithmetically on the offset. That check can pass while the *plumbing* still
    differs -- an extra header field, a different coset order, a gain applied before
    rather than after quantisation. Only comparing the actual strings against a model
    built without the gain unit at all rules that out.
    """
    vr = build_any_model(cfg, "twobranch-vr").eval()
    ref = build_any_model(cfg, "twobranch-split").eval()

    # Name-matched load, so the two models are the same codec except for the gain unit.
    # `strict=False` is expected to drop exactly the two gain vectors and add nothing.
    missing, unexpected = ref.load_state_dict(vr.state_dict(), strict=False)
    assert not missing, f"twobranch-split wants parameters vr does not have: {missing}"
    gain_only = {k for k in unexpected if ".gain." in k or k.endswith(".gain.vector")}
    assert set(unexpected) == gain_only and gain_only, (
        f"the two kinds differ by more than the gain unit: "
        f"{sorted(set(unexpected) - gain_only)}")

    vr.update(force=True)
    ref.update(force=True)
    x = torch.rand(1, 3, 128, 128)

    a = vr.compress(x, delta_beta=0)
    b = ref.compress(x)
    for branch in ("luma", "chroma"):
        for key in ("y_strings", "z_strings"):
            assert a[branch][key] == b[branch][key], (
                f"{branch}/{key} differs with a zero gain vector at Delta_beta = 0")

    # And the same reconstruction, bit for bit -- the gain unit must not perturb
    # dequantisation either.
    ya = vr.decompress(a, device=torch.device("cpu"))["y_hat"]
    yb = ref.decompress(b, device=torch.device("cpu"))["y_hat"]
    assert torch.equal(ya, yb)


@torch.no_grad()
def test_delta_beta_moves_the_rate_monotonically(cfg):
    """The loop's gate, run here so `pytest` covers it without a training run.

    Monotone, not strictly increasing: `R(Delta_beta)` is a step function whose
    plateaus are one CDF row wide (~128-136 in `Delta_beta` units on an untrained
    model), so adjacent probes legitimately return identical rates. A *decrease* is the
    real fault -- it means a sign-flipped or saturated gain vector, and it would break
    the bisection in `jpegai.coder.brm`, whose only precondition is monotonicity.
    """
    from jpegai.train.loop import delta_beta_check

    model = build_any_model(cfg, "twobranch-vr")
    valid = [torch.rand(3, 128, 128) for _ in range(1)]
    db = delta_beta_check(model, valid, torch.device("cpu"),
                          points=cfg.rate.beta_eval_points)

    assert db["beta_neutral"], "Delta_beta = 0 does not leave the offset untouched"
    assert db["beta_unit_gain"], "a fresh gain vector is not unity"
    assert db["beta_cache_ok"], "precompress/compress_cached disagrees with compress"
    assert db["beta_exact"], f"decode is not bit-exact (maxerr {db['beta_maxerr']:.3g})"
    assert db["beta_monotone"], f"rate decreases by {db['beta_worst_drop']:.4f} bpp"
    assert db["beta_ok"]

    bpps = db["beta_bpps"]
    assert len(bpps) == len(cfg.rate.beta_eval_points)
    assert bpps == sorted(bpps)
    # One checkpoint has to span a useful range or the mechanism buys nothing. The
    # paper's clamp is 5.3x down and 3.0x up in sigma; the realised *rate* span is
    # smaller because rate is not linear in sigma, so this asserts only that the ladder
    # is worth calling a ladder.
    assert db["beta_span"] > 2.0
    assert db["beta_anchor_bpp"] == pytest.approx(
        bpps[cfg.rate.beta_eval_points.index(0)])


def test_delta_beta_check_is_empty_without_a_gain_unit(cfg):
    """It has to be callable on every model, so the loop needs no `if` around it.

    Every phase before this one trains a fixed-rate codec, and the gate is printed from
    the same place for all of them.
    """
    from jpegai.train.loop import delta_beta_check

    model = build_any_model(cfg, "twobranch-split")
    assert delta_beta_check(model, [torch.rand(3, 64, 64)], torch.device("cpu")) == {}


def test_eq_ten_closes_against_the_sigma_grid(cfg, grid):
    """eq. (10) is `Isigma`'s own unit conversion, not a free parameter.

    `Delta_beta = floor(ln(beta_test/beta_train) * P_beta / S_sigma)` with `P_beta =
    2^7` and `S_sigma` the sigma grid's log step. The point of asserting it here rather
    than only in `test_gain.py` is that `anchor_beta` above is only meaningful if this
    holds: the anchor is what the ratio is taken against.
    """
    from jpegai.models.gain import beta_displacement

    assert beta_displacement(0.002, 0.002, **grid) == 0
    lo = beta_displacement(cfg.rate.beta_list[0], 0.002, **grid)
    hi = beta_displacement(cfg.rate.beta_list[-1], 0.002, **grid)
    assert lo < 0 < hi
    # Monotone in beta_test, which is what makes the ladder orderable at all.
    d = [beta_displacement(b, 0.002, clip=False, **grid) for b in cfg.rate.beta_list]
    assert d == sorted(d)
    # And it really is a logarithm: doubling the ratio adds a constant.
    step = beta_displacement(0.004, 0.002, clip=False, **grid)
    assert beta_displacement(0.008, 0.002, clip=False, **grid) == pytest.approx(
        2 * step, abs=1)
    # The paper's stated S_sigma is 0.2 to one significant figure; the exact grid step
    # is 0.200365, which is the 0.18% discrepancy recorded in the config.
    assert grid["log_k"] == pytest.approx(0.200365, abs=1e-6)
    assert grid["step"] == 128
    assert step == pytest.approx(math.log(2) * 128 / 0.200365, abs=1)
