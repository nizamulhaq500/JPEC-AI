"""Tests for bit-rate matching — the encoder search of section III-B2.

BRM is non-normative, which changes what is worth testing about it. Nothing here can
produce an undecodable stream, so correctness means something narrower and sharper: the
search must return **the point the paper's rule names**, and it must get there in a
number of encodes that makes the feature affordable. Four properties.

**The rule is "closest to the model's default rate", not "closest to the target".**
Fig. 8's chosen point is the one inside the tolerance band nearest Δβ = 0, so the
achieved rate always errs *toward the anchor* — high for a target below the anchor, low
for one above it — by very nearly the full tolerance. An implementation that bisected
to the target instead would look better on rate error and worse on BD-rate, and the
error's *sign* is what tells the two apart.

**The anchor shortcut has to fire.** If Δβ = 0 is already inside the band the answer is
0 and no search runs. That case is why BRM costs 3.6–4.8 pp instead of far more, so a
test that only exercised the search would miss the property the design rests on.

**Fig. 9's cache must serve every probe.** One `precompress` per model, however many
Δβ are tried; the whole encode-time budget depends on it, and a cache that quietly
re-ran the analysis transform would be invisible except in the wall clock.

**The bisection needs monotone rate.** Tested against real packet bytes on a real
codec, because the search bisects on measured bytes and a non-monotone rate curve makes
a bisection converge to nonsense.

The search logic is tested against an analytic rate curve rather than a trained model.
That is deliberate: an untrained codec's rate is a coarse step function of Δβ — the
plateaus are one CDF row wide, so `R(Δβ)` takes about thirteen distinct values across
the whole clamp — and band-edge selection cannot be asserted to a tenth of a percent on
a curve whose steps are five percent tall. `Curve` supplies the smooth monotone rate
that a trained model has; the real codec is then used for the properties that are about
the codec rather than about the search.
"""

from __future__ import annotations

import math

import pytest
import torch

from jpegai.coder.brm import (
    BISECT_WINDOW, DELTA_BETA_MAX, DELTA_BETA_MIN, MAX_ITERATIONS, TOLERANCE,
    BRMResult, RatePoint, RateSearch, _lower_bound, beta_ladder, beta_ratio_of, brm,
    delta_beta_for, linear_fit, match_rate, rate_ladder, select_model, solve_fit,
)
from jpegai.models.hyper import SigmaIndex
from jpegai.models.twobranch import TwoBranchCodec

CLAMP_SPAN = DELTA_BETA_MAX - DELTA_BETA_MIN      # 1771


# ---------------------------------------------------------------------------
# an analytic stand-in for a trained codec
# ---------------------------------------------------------------------------
class Curve:
    """A model whose rate is a smooth, monotone, analytic function of Δβ.

    `bytes(Δβ) = anchor · exp(decay · w(Δβ))`, where `w` is the identity by default. So
    `anchor_bytes` is literally the rate at Δβ = 0 and the rate spans `span` across the
    whole clamp. A real codec's rate grows more slowly than the σ multiplier does — the
    measured Tier A model spans about 4.5× of rate over 16× of σ — which does not matter
    here: what is under test is whether the search lands on the point its rule names,
    and that is a question about the curve's *shape*, not its slope.

    `bend > 0` replaces `w` with a saturating `k·tanh(Δβ/k)`, `k = CLAMP_SPAN/bend`.
    Still monotone, so the bisection's precondition holds, but no longer log-linear —
    which is the point. Eq. (14) fits a chord between the two clamp ends, and against a
    curve that flattens at both ends that chord is too shallow, so Δβ,1 lands past the
    answer by more than the ±100 window can absorb. `bend = 0` keeps the curve exactly
    log-linear, which is what lets the fit itself be tested for exactness.

    Counts its own calls, so a test can assert the encode budget and the cache.
    """

    def __init__(self, anchor_bytes: int = 24576, span: float = 16.0,
                 bend: float = 0.0):
        self.anchor_bytes = int(anchor_bytes)
        self.decay = math.log(span) / CLAMP_SPAN
        self.bend = float(bend)
        self.passes = 0
        self.encodes = 0

    def _warp(self, delta_beta: int) -> float:
        if not self.bend:
            return float(delta_beta)
        k = CLAMP_SPAN / self.bend
        return k * math.tanh(delta_beta / k)

    def precompress(self, x):
        self.passes += 1
        return {"pixels": int(x.shape[-1]) * int(x.shape[-2])}

    def compress_cached(self, cache, *, delta_beta=0, q_index=None):
        self.encodes += 1
        return {"delta_beta": int(delta_beta), "q_index": q_index}

    def packet_bytes(self, packet) -> int:
        return max(1, round(self.anchor_bytes
                            * math.exp(self.decay * self._warp(packet["delta_beta"]))))


def _x(h: int = 512, w: int = 768) -> torch.Tensor:
    """A picture-shaped tensor. `Curve` reads only its shape."""
    return torch.zeros(1, 3, h, w)


def _anchor_bpp(model, x=None) -> float:
    x = _x() if x is None else x
    return RateSearch(model, x).default().bpp


# ---------------------------------------------------------------------------
# the search rule (eq. 14 and Fig. 8), on the analytic curve
# ---------------------------------------------------------------------------
def test_the_anchor_is_the_answer_when_it_is_already_in_tolerance():
    """The case that keeps BRM cheap: no search, one encode, Δβ exactly 0."""
    c = Curve()
    r = match_rate(c, _x(), _anchor_bpp(Curve()))
    assert r.delta_beta == 0
    assert r.in_tolerance and not r.clamped
    assert r.encodes == 1 and r.passes == 1
    assert [p.delta_beta for p in r.trace] == [0]


@pytest.mark.parametrize("frac", [0.91, 0.95, 1.0, 1.05, 1.09])
def test_the_shortcut_covers_the_whole_band_not_just_the_centre(frac):
    """Anywhere inside ±10% of the anchor, the answer is the anchor.

    A tolerance is permission to be 10% off, and the paper spends that permission on
    staying at Δβ = 0. Testing only `frac == 1.0` would pass on an implementation that
    searched whenever the target was not exactly the anchor rate.
    """
    r = match_rate(Curve(), _x(), _anchor_bpp(Curve()) * frac)
    assert r.delta_beta == 0 and r.encodes == 1


@pytest.mark.parametrize("frac", [0.25, 0.5, 0.8, 1.25, 2.0, 2.8])
def test_the_achieved_rate_always_errs_toward_the_anchor(frac):
    """Fig. 8's rule, and the one assertion that separates it from bisect-to-target.

    A target below the anchor is met from *above*, a target above it from *below*, and
    in both cases by very nearly the whole tolerance -- that is what "closest to the
    default rate point" means on a curve fine enough to land anywhere.
    """
    anchor = _anchor_bpp(Curve())
    target = anchor * frac
    r = match_rate(Curve(), _x(), target)
    assert r.in_tolerance and not r.clamped
    if target < anchor:
        assert r.rate_error > 0, "a cheap target should be met from above"
        assert TOLERANCE - 0.005 < r.rate_error <= TOLERANCE
    else:
        assert r.rate_error < 0, "an expensive target should be met from below"
        assert -TOLERANCE <= r.rate_error < -TOLERANCE + 0.005


@pytest.mark.parametrize("frac", [0.3, 0.7, 1.4, 2.5])
def test_aiming_at_the_target_is_more_accurate_and_further_from_the_anchor(frac):
    """`stay_near_anchor=False` is the trade the paper declined, made explicit.

    Turning the rule off must improve rate accuracy and move Δβ further from zero.
    Both directions matter: an implementation where the flag did nothing would pass an
    accuracy check alone, and one that overshot would pass a distance check alone.
    """
    anchor = _anchor_bpp(Curve())
    target = anchor * frac
    near = match_rate(Curve(), _x(), target)
    exact = match_rate(Curve(), _x(), target, stay_near_anchor=False)
    assert abs(exact.rate_error) < abs(near.rate_error)
    assert abs(exact.rate_error) < 0.001
    assert abs(exact.delta_beta) > abs(near.delta_beta)


def test_the_search_costs_a_bisection_not_a_scan():
    """The encode budget: anchor, two fit probes, and a closing bisection.

    The number matters as much as the answer -- BRM's whole cost story is 3-5x encode
    time, and a search that probed even fifty points would be a different feature.
    `CLAMP_SPAN` in the assertion is the alternative being ruled out: 1771 encodes.
    """
    c = Curve()
    r = match_rate(c, _x(), _anchor_bpp(Curve()) * 0.4)
    assert r.in_tolerance
    # 1 anchor + 2 fit + ceil(log2(200)) = 8 bisection probes + 1 confirmation.
    assert r.encodes <= 3 + MAX_ITERATIONS
    assert r.encodes == c.encodes < CLAMP_SPAN


def test_one_precompress_serves_every_probe():
    """Fig. 9, as an assertion. The cache is the reason validation can be exact."""
    c = Curve()
    r = match_rate(c, _x(), _anchor_bpp(Curve()) * 0.4)
    assert c.passes == 1 == r.passes
    assert r.encodes > 1, "this target must actually search, or the test is vacuous"


def test_probes_are_memoised_so_the_bisection_never_repeats_itself():
    """A bisection revisits its own endpoints; re-encoding them would be pure waste."""
    c = Curve()
    s = RateSearch(c, _x())
    for d in (0, -400, 0, -400, 702, 702, 0):
        s.probe(d)
    assert c.encodes == 3
    assert [p.delta_beta for p in s.trace()] == [-400, 0, 702]


def test_a_probe_outside_the_clamp_is_the_clamp_not_an_error():
    """Δβ is a 12-bit field, so 5000 is not a rate request, it is `DELTA_BETA_MAX`."""
    s = RateSearch(Curve(), _x())
    assert s.probe(50_000).delta_beta == DELTA_BETA_MAX
    assert s.probe(-50_000).delta_beta == DELTA_BETA_MIN
    assert s.probe(DELTA_BETA_MAX) is s.probe(50_000)


# ---------------------------------------------------------------------------
# eq. (14): the fit, and what happens when it is wrong
# ---------------------------------------------------------------------------
def test_the_fit_inverts_an_exactly_log_linear_curve():
    """`log R = a·Δβ + b` recovered from two probes, then solved back at `log Rt`."""
    s = RateSearch(Curve(), _x())
    lo, hi = s.probe(DELTA_BETA_MIN), s.probe(DELTA_BETA_MAX)
    a, b = linear_fit(lo, hi)
    assert a > 0
    for d in (-900, -300, 0, 250, 650):
        assert solve_fit(a, b, s.probe(d).bpp) == pytest.approx(d, abs=2)


def test_the_fit_needs_two_distinct_points():
    s = RateSearch(Curve(), _x())
    with pytest.raises(ValueError, match="two distinct"):
        linear_fit(s.probe(0), s.probe(0))


def test_a_flat_fit_falls_back_to_the_anchor():
    """Both clamp ends giving the same rate means the σ table saturated at both. There
    is no line to solve, and Δβ = 0 is as good a place to start bisecting as any."""
    assert solve_fit(0.0, math.log(0.5), 0.9) == 0


def test_a_wrong_fit_is_widened_to_the_whole_clamp():
    """The documented deviation from the paper, and proof that it is load-bearing.

    `bend=3` flattens the rate curve at both ends, so eq. (14)'s chord is too shallow
    and Δβ,1 lands 146 past the answer -- outside the ±100 window, which the paper
    specifies and does not say what to do about. Everything inside that window is 17-25%
    below the target, so returning the window's best would be a miss. Widening to the
    full clamp costs one more bisection on an already-warm cache and answers instead.

    Asserted three ways, because "in tolerance" alone could have come from luck: the
    answer must be outside the window, the trace must show probes outside it, and the
    same target with the widening unreachable (`window=1`) must still be legal.
    """
    c = Curve(bend=3.0)
    target = _anchor_bpp(Curve(bend=3.0)) * 0.8
    s = RateSearch(c, _x())
    a, b = linear_fit(s.probe(DELTA_BETA_MIN), s.probe(DELTA_BETA_MAX))
    d1 = solve_fit(a, b, target)

    r = match_rate(Curve(bend=3.0), _x(), target)
    assert r.in_tolerance and not r.clamped, r.summary()
    assert abs(r.delta_beta - d1) > BISECT_WINDOW, "the window did bracket it after all"
    assert max(p.delta_beta for p in r.trace) > d1 + BISECT_WINDOW
    assert r.encodes <= 3 + 2 * MAX_ITERATIONS, "two bisections, not a scan"

    # A window too small to bracket even the target still yields a legal rate: the fit
    # is a starting point, so the fallback has something to work with either way.
    assert match_rate(Curve(bend=3.0), _x(), target, window=1).in_tolerance


def test_a_curved_rate_function_defeats_the_fit_and_not_the_search():
    """The bisection is what delivers the accuracy; the fit only has to start it.

    Across a bent curve the chord misses by up to 185 Δβ, in both directions and by
    different amounts at every target. Every one of these still lands inside the band,
    which is the property that matters -- eq. (14) is a heuristic, and the search must
    not inherit its error.
    """
    c = Curve(bend=2.0)
    anchor = _anchor_bpp(Curve(bend=2.0))
    for frac in (0.3, 0.5, 0.8, 1.3, 2.0):
        r = match_rate(c, _x(), anchor * frac)
        assert r.in_tolerance, f"{frac}: {r.summary()}"


def test_the_bisection_returns_the_first_index_reaching_the_goal():
    """`_lower_bound` is the primitive everything else is built on, so it is tested as
    one: the answer must reach the goal and its neighbour below must not."""
    s = RateSearch(Curve(), _x())
    goal = s.probe(0).bpp * 1.3
    d, edge = _lower_bound(s, -1069, 702, goal, MAX_ITERATIONS)
    assert not edge
    assert s.probe(d).bpp >= goal > s.probe(d - 1).bpp


def test_the_bisection_reports_an_unreachable_goal_rather_than_guessing():
    s = RateSearch(Curve(), _x())
    top = s.probe(DELTA_BETA_MAX).bpp
    d, edge = _lower_bound(s, -1069, 702, top * 10, MAX_ITERATIONS)
    assert edge and d == DELTA_BETA_MAX
    d, edge = _lower_bound(s, -1069, 702, 0.0, MAX_ITERATIONS)
    assert edge and d == DELTA_BETA_MIN


# ---------------------------------------------------------------------------
# out of reach
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frac", [0.05, 0.1, 5.0, 20.0])
def test_an_unreachable_target_clamps_and_says_so(frac):
    """`clamped` and `in_tolerance` are separate fields for exactly this case.

    One model spans 16x in σ and rather less in rate. Reporting a 60%-off result as a
    success would hide the failure mode the four-model ladder exists to avoid.
    """
    r = match_rate(Curve(), _x(), _anchor_bpp(Curve()) * frac)
    assert r.clamped and not r.in_tolerance
    assert r.delta_beta in (DELTA_BETA_MIN, DELTA_BETA_MAX)
    assert (r.delta_beta == DELTA_BETA_MIN) == (frac < 1)


def test_landing_on_the_clamp_and_hitting_the_target_is_not_clamped():
    """`clamped` means the range ran out, not that the answer sits at the edge of it.

    The target is chosen so the band's floor falls between the last two Δβ, which is the
    only way to reach the clamp on a curve fine enough to land anywhere: one Δβ step is
    0.16% of rate against a 10% band, so any coarser target is met well short of 702.
    """
    c = Curve()
    s = RateSearch(c, _x())
    goal = (s.probe(DELTA_BETA_MAX - 1).bpp + s.probe(DELTA_BETA_MAX).bpp) / 2
    r = match_rate(Curve(), _x(), goal / (1 - TOLERANCE))
    assert r.delta_beta == DELTA_BETA_MAX
    assert r.in_tolerance and not r.clamped


def test_a_nonpositive_target_is_refused():
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError, match="must be positive"):
            match_rate(Curve(), _x(), bad)


def test_rate_matching_is_per_picture():
    """A batch would need one Δβ per image, and Δβ is a per-picture header field."""
    with pytest.raises(ValueError, match="per picture"):
        RateSearch(Curve(), torch.zeros(4, 3, 64, 64))


# ---------------------------------------------------------------------------
# eq. (13): model selection
# ---------------------------------------------------------------------------
def test_selection_picks_the_relatively_closest_default_rate():
    models = [Curve(anchor_bytes=6144), Curve(anchor_bytes=24576),
              Curve(anchor_bytes=98304)]
    x = _x()
    mid = _anchor_bpp(Curve(anchor_bytes=24576))
    idx, drs, searches = select_model(models, x, mid * 1.05)
    assert idx == 1
    assert drs[1] < drs[0] and drs[1] < drs[2]
    assert [s.encodes for s in searches] == [1, 1, 1], "one anchor probe each"


def test_dividing_by_the_default_rate_favours_the_more_expensive_model():
    """Eq. (13) divides by `Rd`, not by `Rt`, and that is Fig. 6's asymmetry as a rule.

    The worked case: at `Rt = 0.3`, a model at `Rd = 0.2` scores `0.1/0.2 = 0.50` and
    one at `Rd = 0.45` scores `0.15/0.45 = 0.33`. The second wins although it is 50%
    further away in absolute bpp, because pulling a rate down costs less quality than
    pushing one up. Dividing by `Rt` would have picked the other one.
    """
    pixels = 512 * 768
    cheap = Curve(anchor_bytes=round(0.20 * pixels / 8))
    dear = Curve(anchor_bytes=round(0.45 * pixels / 8))
    idx, drs, _ = select_model([cheap, dear], _x(), 0.30)
    assert drs == pytest.approx([0.50, 1 / 3], abs=1e-3)
    assert idx == 1
    # And the alternative rule really would have disagreed.
    by_target = [abs(d) for d in (0.20 - 0.30, 0.45 - 0.30)]
    assert min(range(2), key=lambda i: by_target[i]) == 0


def test_selection_hands_its_cache_to_the_search():
    """The point of returning the `RateSearch` objects: the winner is not re-encoded.

    Four models means four `precompress` calls whatever happens; what must not happen
    is a fifth for the model that won.
    """
    models = [Curve(anchor_bytes=6144), Curve(anchor_bytes=24576)]
    r = brm(models, _x(), _anchor_bpp(Curve(anchor_bytes=24576)) * 0.5)
    assert [m.passes for m in models] == [1, 1]
    assert r.passes == 2
    assert r.encodes == sum(m.encodes for m in models)


def test_a_single_model_ladder_is_legal():
    """The honest way to report what one model can do, which is what we measure."""
    c = Curve()
    r = brm([c], _x(), _anchor_bpp(Curve()) * 0.5)
    assert r.model_index == 0 and r.dr == pytest.approx(0.5, abs=0.01)
    assert r.in_tolerance


def test_selecting_from_nothing_is_refused():
    with pytest.raises(ValueError, match="no models"):
        select_model([], _x(), 0.5)


def test_the_result_summarises_itself():
    r = match_rate(Curve(), _x(), _anchor_bpp(Curve()) * 0.5)
    text = r.summary()
    assert "model 0" in text and "encodes" in text and "ok" in text
    assert f"{r.bpp:.4f}" in text


# ---------------------------------------------------------------------------
# eqs. (9) and (10): addressing a rate by β
# ---------------------------------------------------------------------------
def test_delta_beta_for_reads_its_constants_off_the_live_sigma_grid():
    """`S_σ` *is* the grid's log step, so the grid is the argument. Passing the two
    constants separately is what would let the two ends of the codec disagree."""
    si = SigmaIndex()
    assert delta_beta_for(0.075, 0.075, si) == 0
    assert delta_beta_for(0.15, 0.075, si) > 0
    assert delta_beta_for(0.0375, 0.075, si) < 0
    assert beta_ratio_of(delta_beta_for(0.15, 0.075, si), si) == pytest.approx(2.0,
                                                                              rel=2e-3)


def test_the_achieved_ratio_is_reported_low_not_as_requested():
    """Eq. (10) floors, so a requested ratio comes back up to one step low. Reporting
    the request as though it were achieved is a small lie that bends a BD-rate curve."""
    si = SigmaIndex()
    got = beta_ratio_of(delta_beta_for(2.0 * 0.075, 0.075, si), si)
    assert got <= 2.0
    assert got > 2.0 * math.exp(-si.log_k / si.step)


# ---------------------------------------------------------------------------
# the real codec
# ---------------------------------------------------------------------------
def _codec(**kw):
    """The same small codec `tests/test_gain.py` uses, so the two files agree on what
    a Phase 8 model is."""
    m = TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                       chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                       synthesis_width=(24, 16, 16, 16), internal_format="420",
                       mean_scale=True, split_hyper=True, mcm=False,
                       gain=True, **kw).eval()
    m.update(force=True)
    return m


@pytest.fixture(scope="module")
def codec():
    torch.manual_seed(0)
    return _codec()


@pytest.fixture(scope="module")
def picture():
    torch.manual_seed(1)
    return torch.rand(1, 3, 128, 192)


def test_the_cached_path_and_a_plain_encode_produce_the_same_bytes(codec, picture):
    """`compress = code_cached(precode(y))`, so the two must be byte-identical.

    This is the load-bearing test for Fig. 9. If the cached path drifted from the plain
    one, the search would optimise a rate that the real encoder never produces -- and
    the packets both decode, so nothing else would notice.
    """
    cache = codec.precompress(picture)
    for d in (DELTA_BETA_MIN, -400, 0, 300, DELTA_BETA_MAX):
        cached = codec.compress_cached(cache, delta_beta=d)
        plain = codec.compress(picture, delta_beta=d)
        assert codec.packet_bytes(cached) == codec.packet_bytes(plain)
        assert cached["luma"]["y_strings"] == plain["luma"]["y_strings"]
        assert cached["chroma"]["y_strings"] == plain["chroma"]["y_strings"]
        assert cached["delta_beta"] == plain["delta_beta"] == (d, d)


def test_rate_is_monotone_in_delta_beta_on_real_bytes(codec, picture):
    """The bisection's one precondition, measured on coded bytes rather than estimated.

    Non-decreasing rather than strictly increasing: the coding σ is quantised onto the
    Iσ grid, so the rate is a step function whose plateaus are one CDF row wide. A
    plateau is fine for a bisection; a dip is not.
    """
    ladder = rate_ladder(codec, picture)
    bpps = [p.bpp for p in ladder]
    assert bpps == sorted(bpps)
    assert bpps[-1] > bpps[0] * 2, "the clamp should move the rate substantially"


def test_the_fixed_rate_ladder_matches_the_configured_eval_points(codec, picture):
    """`rate_ladder`'s default sweep is a copy of `rate.beta_eval_points`; the config
    is the source of truth, and a copy that drifts is worse than no copy."""
    from jpegai.config import load_config

    cfg = load_config("tierA")
    assert [p.delta_beta for p in rate_ladder(codec, picture)] \
        == list(cfg.rate.beta_eval_points)


def test_the_ladder_costs_one_forward_pass(codec, picture):
    """Nine rate points, one analysis transform. Asserted by counting calls into the
    luma analysis transform, because the saving is invisible in the output."""
    calls = []
    hook = codec.g_a_y.register_forward_pre_hook(lambda *a: calls.append(1))
    try:
        points = rate_ladder(codec, picture)
    finally:
        hook.remove()
    assert len(points) == 9 and len(calls) == 1


def test_every_matched_rate_decodes(codec, picture):
    """Validation means a real encode, so the answer is a packet, not an estimate."""
    anchor = _anchor_bpp(codec, picture)
    for frac in (0.5, 0.8, 1.0, 1.3):
        r = match_rate(codec, picture, anchor * frac)
        out = codec.decompress(r.packet)
        assert out["x_hat"].shape == picture.shape
        assert torch.isfinite(out["x_hat"]).all()


def test_the_answer_carries_the_delta_beta_the_decoder_will_read(codec, picture):
    """A decoder that had to be *told* the rate point would not be a decoder.

    The Δβ the search settled on has to be in the packet, in both branch fields, and the
    packet has to be the one a plain `compress` at that Δβ would have produced -- the
    search returns its own last encode, so a mismatch here means it returned a stale one.
    """
    r = match_rate(codec, picture, _anchor_bpp(codec, picture) * 0.55)
    assert r.packet["delta_beta"] == (r.delta_beta, r.delta_beta)
    ref = codec.compress(picture, delta_beta=r.delta_beta)
    assert codec.packet_bytes(ref) == codec.packet_bytes(r.packet)
    assert codec.packet_bytes(ref) * 8.0 / (128 * 192) == pytest.approx(r.bpp)


def test_the_search_charges_the_header_to_the_rate(codec, picture):
    """`packet_bytes` includes the two 12-bit fields, so the search optimises the rate
    the decoder actually receives -- `runbench.py`'s convention, not a variant of it."""
    s = RateSearch(codec, picture)
    p = s.probe(0)
    assert p.nbytes == codec.packet_bytes(p.packet)
    assert codec.header_bytes(p.packet) == 3
    assert p.bpp == pytest.approx(p.nbytes * 8.0 / (128 * 192))


def test_a_spatial_quality_map_rides_through_the_whole_search(codec, picture):
    """A rate target and an ROI map are independent requests, and BRM has to honour
    both: the map is fixed, Δβ is searched, and the answer still carries the map."""
    q = torch.zeros(1, 1, 8, 12, dtype=torch.int64)
    q[..., 2:6, 3:9] = 4
    plain = match_rate(codec, picture, _anchor_bpp(codec, picture) * 0.6)
    roi = match_rate(codec, picture, _anchor_bpp(codec, picture) * 0.6, q_index=q)
    assert roi.packet.get("q_residual") is not None
    assert roi.in_tolerance
    # The map raises σ inside the box, so the same rate needs a lower Δβ.
    assert roi.delta_beta < plain.delta_beta
    assert codec.decompress(roi.packet)["x_hat"].shape == picture.shape


def test_the_beta_ladder_maps_betas_to_the_delta_beta_that_imitates_them(codec,
                                                                        picture):
    """One checkpoint answering a β sweep -- what makes "one model, nine rate points"
    comparable against "nine models, nine rate points"."""
    betas = (0.015, 0.03, 0.075, 0.2)
    pairs = beta_ladder(codec, picture, betas, 0.075)
    assert [b for b, _ in pairs] == list(betas)
    assert pairs[2][1].delta_beta == 0, "β_test == β_train is the anchor"
    assert [p.delta_beta for _, p in pairs] == sorted(p.delta_beta
                                                     for _, p in pairs)
    assert [p.bpp for _, p in pairs] == sorted(p.bpp for _, p in pairs)


def test_betas_beyond_the_clamp_collapse_onto_it(codec, picture):
    """One model reaches 5.3x down and 3.0x up. Beyond that two βs give the same Δβ,
    and the returned Δβ is what says so -- there is no error to raise, because the
    clamp is the standard's answer, not ours."""
    pairs = beta_ladder(codec, picture, (0.5, 1.0, 3.0), 0.075)
    assert all(p.delta_beta == DELTA_BETA_MAX for _, p in pairs)
    assert len({p.nbytes for _, p in pairs}) == 1


def test_brm_over_a_real_two_model_ladder(codec, picture):
    """The whole algorithm on real packets: selection, search, validation, decode."""
    torch.manual_seed(0)
    dearer = _codec()
    with torch.no_grad():
        for v in dearer.gain_parameters():
            v.fill_(300.0)
    anchor = _anchor_bpp(codec, picture)
    cheap = brm([codec, dearer], picture, anchor * 0.95)
    dear = brm([codec, dearer], picture, RateSearch(dearer, picture).default().bpp)
    assert cheap.model_index == 0 and dear.model_index == 1
    for r in (cheap, dear):
        assert r.passes == 2, "one precompress per model, no more"
        assert codec.decompress(r.packet)["x_hat"].shape == picture.shape


def test_a_fixed_rate_model_has_no_cache_to_search(codec, picture):
    """`precompress` needs a gain unit. Without one there is a single rate point, and a
    cache would be an invitation to a stale-tensor bug in exchange for nothing."""
    torch.manual_seed(0)
    m = TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                       chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                       synthesis_width=(24, 16, 16, 16), internal_format="420",
                       mean_scale=True, split_hyper=True, mcm=False, gain=False).eval()
    m.update(force=True)
    with pytest.raises(RuntimeError, match="gain unit"):
        RateSearch(m, picture)


def test_searching_before_the_tables_are_built_is_refused(picture):
    """An un-`update()`d model has no CDFs, so a probe would return garbage bytes."""
    torch.manual_seed(0)
    m = TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                       chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                       synthesis_width=(24, 16, 16, 16), internal_format="420",
                       mean_scale=True, split_hyper=True, mcm=False, gain=True).eval()
    with pytest.raises(RuntimeError, match="entropy tables"):
        RateSearch(m, picture)


# ---------------------------------------------------------------------------
# the records themselves
# ---------------------------------------------------------------------------
def test_rate_error_is_signed_and_relative_to_the_target():
    r = BRMResult(0.50, -200, 0.45, {})
    assert r.rate_error == pytest.approx(-0.10)
    assert r.in_tolerance
    assert BRMResult(0.50, 0, 0.56, {}).rate_error == pytest.approx(0.12)
    assert not BRMResult(0.50, 0, 0.56, {}).in_tolerance


def test_a_rate_point_compares_on_what_was_asked_and_measured():
    """`packet` is excluded from equality: two probes of the same Δβ are the same rate
    point, and comparing dicts of `bytes` objects would be both slow and meaningless."""
    a = RatePoint(-200, 100, 0.5, {"a": 1})
    b = RatePoint(-200, 100, 0.5, {"b": 2})
    assert a == b
    assert RatePoint(-201, 100, 0.5, {}) != a
    assert "packet" not in repr(a)
