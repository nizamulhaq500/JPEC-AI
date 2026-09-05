"""Bit-rate matching -- the encoder search that turns a rate *target* into a Δβ.

This is section III-B2 of the paper, and it is **non-normative**: nothing here is in
the bitstream, and a different encoder may search differently and still be compliant.
It matters anyway, for two reasons.

First, without it the codec cannot hit a requested rate at all. The paper's own
measurement is a maximum rate error of **327%** for the main profile and 324% for the
high profile when the four β_train models are used as-is. Four fixed rate points do not
make a rate control.

Second, it costs real BD-rate: −16.7% becomes −13.1% for the main profile and −24.0%
becomes −19.2% for the high profile, a penalty of 3.6-4.8 pp, and encode time rises
from 5 to 15 minutes (main) or 13 to 67 (high). So when comparing a fixed-rate ladder
against JPEG AI's published numbers, the honest column is the fixed-rate one -- see
`docs/` and the report's decomposition table. This module is what would be needed to
compete against the *other* column.

The algorithm has three parts, and the paper names them:

**Model selection** (eq. 13). Each of the four models is encoded once at its own anchor
(β_test = β_train, i.e. Δβ = 0) to get a default rate `Rd`, and the model minimising
`Dr = |Rd − Rt| / Rd` wins. Dividing by `Rd` rather than by `Rt` is not a detail: it
makes the measure asymmetric in favour of models whose default rate is *above* the
target, which is deliberate -- Fig. 6 shows quality falls off faster when a model is
pushed up in rate than when it is pulled down.

**Δβ search** (eq. 14). `log R ≈ a·Δβ + b`. Two probes at the ends of the clamp fit the
line, solving it at `log Rt` gives `Δβ,1`, and a bisection inside `[Δβ,1 − 100,
Δβ,1 + 100]` finds the answer. The linear model is only approximate -- it is what Fig. 7
looks like, not a derivation -- so the bisection is what actually delivers the accuracy
and the fit only has to land within 100.

**Δβ validation.** Every candidate is really encoded, so the rate is measured and never
estimated. Fig. 9 is what makes that affordable: the latent, the hyper latent, μ and Iσ
all come off the *ungained* path, so they are computed once and reused. See
`SplitHyperBranch.precode`.

One subtlety in the selection rule that is easy to read past. The paper says the chosen
point is the one "which can generate a rate that is lower than the maximum rate
difference threshold and is also closest to the current model's default rate" -- not
closest to the *target*. So BRM deliberately stops at the near edge of the tolerance
band rather than in the middle of it: staying close to the anchor is what preserves the
variable-rate quality, and a 10% tolerance is treated as permission to be 10% off. When
the anchor itself is inside the band, the answer is Δβ = 0 and no search happens at all,
which is why the BD-rate penalty is a few points rather than many.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from jpegai.models.gain import (DELTA_BETA_MAX, DELTA_BETA_MIN, beta_displacement,
                                beta_ratio, clip_delta_beta)

TOLERANCE = 0.10          # PAPER: the "maximum rate difference threshold"
BISECT_WINDOW = 100       # PAPER: [Δβ,1 − 100, Δβ,1 + 100]
MAX_ITERATIONS = 12       # an integer bisection of a 200-wide window needs 8


@dataclass(frozen=True)
class RatePoint:
    """One encode: what was asked for, and what came out."""

    delta_beta: int
    nbytes: int
    bpp: float
    packet: dict = field(repr=False, compare=False)


@dataclass
class BRMResult:
    """The outcome of a rate match, including the parts that went wrong.

    `clamped` and `in_tolerance` are separate on purpose. A clamped search means the
    model could not reach the target even at the end of its range -- the answer is the
    best available, and reporting it as a success would hide the very failure mode the
    four-model design exists to avoid.
    """

    target_bpp: float
    delta_beta: int
    bpp: float
    packet: dict = field(repr=False)
    #: Arithmetic-coding passes, every model included. This is the number the paper's
    #: 5 -> 15 minute encode time is about.
    encodes: int = 0
    #: Network forward passes -- one `precompress` per model considered. Cheap probes
    #: are the whole point of Fig. 9's cache, so counting only `encodes` would flatter
    #: the result: four models cost four analysis transforms whatever the search does.
    passes: int = 0
    model_index: int = 0
    dr: float = 0.0
    clamped: bool = False
    trace: list[RatePoint] = field(default_factory=list, repr=False)

    @property
    def rate_error(self) -> float:
        """Signed relative miss, `(R − Rt)/Rt`. Negative means under the target."""
        return (self.bpp - self.target_bpp) / self.target_bpp

    @property
    def in_tolerance(self) -> bool:
        return abs(self.rate_error) <= TOLERANCE + 1e-12

    def summary(self) -> str:
        flag = "ok" if self.in_tolerance else ("clamped" if self.clamped else "MISS")
        return (f"model {self.model_index}  Δβ {self.delta_beta:+5d}  "
                f"{self.bpp:.4f} bpp for {self.target_bpp:.4f} target  "
                f"({self.rate_error:+.1%}, {self.encodes} encodes / {self.passes} "
                f"passes, {flag})")


class RateSearch:
    """A cached encoder for one picture: `probe(Δβ)` returns real coded bytes.

    Holds Fig. 9's cache, so every probe after the first costs one arithmetic-coding
    pass and nothing else. Probes are memoised as well -- the bisection revisits its
    own endpoints, and re-encoding them would be pure waste.

    `encodes` counts arithmetic-coding passes and `passes` the network forward pass,
    which is always exactly one. Keeping them apart is the only way to see what the
    cache bought.
    """

    def __init__(self, model, x: Tensor, *, q_index: Tensor | None = None):
        if x.shape[0] != 1:
            raise ValueError(f"rate matching is per picture, got a batch of "
                             f"{x.shape[0]}")
        self.model = model
        self.q_index = q_index
        self.pixels = int(x.shape[-1]) * int(x.shape[-2])
        self.cache = model.precompress(x)
        self._seen: dict[int, RatePoint] = {}
        self.encodes = 0
        self.passes = 1

    def probe(self, delta_beta: int) -> RatePoint:
        d = clip_delta_beta(int(delta_beta))
        if d in self._seen:
            return self._seen[d]
        packet = self.model.compress_cached(self.cache, delta_beta=d,
                                            q_index=self.q_index)
        # `packet_bytes` includes the header, so the two Δβ fields and the spatial map
        # are charged to every probe -- the search must optimise the rate the decoder
        # actually receives, which is `runbench.py`'s convention too.
        nbytes = self.model.packet_bytes(packet)
        self.encodes += 1
        point = RatePoint(d, nbytes, nbytes * 8.0 / self.pixels, packet)
        self._seen[d] = point
        return point

    def default(self) -> RatePoint:
        """`Rd`: the anchor, β_test = β_train."""
        return self.probe(0)

    def trace(self) -> list[RatePoint]:
        """Every probe made, in Δβ order -- the rate curve the search walked."""
        return sorted(self._seen.values(), key=lambda p: p.delta_beta)


def _lower_bound(search: RateSearch, lo: int, hi: int, goal_bpp: float,
                 budget: int) -> tuple[int, bool]:
    """Smallest Δβ in `[lo, hi]` whose rate reaches `goal_bpp`, and whether it hit
    the edge of the window rather than a genuine crossing.

    An integer bisection, which is the right kind here: Δβ *is* an integer field, so
    there is no tolerance to choose and no risk of grinding on a flat step. It ends
    when the bracket closes, in `ceil(log2(hi - lo))` probes -- 8 for the paper's
    200-wide window.

    It assumes `R(Δβ)` is non-decreasing, which is the whole reason the gain unit can
    be rate-controlled at all: a larger Δβ scales every residual up by the same
    factor and widens every σ to match, so more bits come out. The assumption is not
    exactly true byte-for-byte -- an arithmetic coder can spend one byte fewer on a
    slightly wider distribution -- and it does not need to be. A byte of local
    non-monotonicity moves the answer by one Δβ step, which is 0.16% of rate; the
    tolerance band is 10%.
    """
    lo, hi = clip_delta_beta(lo), clip_delta_beta(hi)
    if search.probe(lo).bpp >= goal_bpp:
        return lo, True                       # the goal is below the whole window
    if search.probe(hi).bpp < goal_bpp:
        return hi, True                       # unreachable inside the window
    for _ in range(budget):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        if search.probe(mid).bpp >= goal_bpp:
            hi = mid
        else:
            lo = mid
    return hi, False


def linear_fit(low: RatePoint, high: RatePoint) -> tuple[float, float]:
    """Eq. (14)'s `a` and `b` from two probes, fitted on `log R`.

    Natural log, matching the paper's `log(R)`; the base only rescales `a` and `b`
    together, so the solved Δβ is the same either way.
    """
    if high.delta_beta == low.delta_beta:
        raise ValueError("eq. (14) needs two distinct Δβ")
    dl, dh = float(low.delta_beta), float(high.delta_beta)
    rl, rh = math.log(max(low.bpp, 1e-9)), math.log(max(high.bpp, 1e-9))
    a = (rh - rl) / (dh - dl)
    return a, rl - a * dl


def solve_fit(a: float, b: float, target_bpp: float) -> int:
    """Δβ,1: where the fitted line crosses `log Rt`. Clamped, and rounded to the
    integer field. A flat fit means the probes gave the same rate, which happens when
    both ends saturate the σ table; the anchor is then as good a starting point as
    any."""
    if abs(a) < 1e-12:
        return 0
    return clip_delta_beta(round((math.log(max(target_bpp, 1e-9)) - b) / a))


def _band_edge(s: RateSearch, lo: int, hi: int, goal: float, below: bool,
               budget: int) -> tuple[int, bool]:
    """The Δβ in `[lo, hi]` sitting on the near edge of the tolerance band.

    Two cases, one bisection. Coming up from a cheap anchor we want the *smallest* Δβ
    whose rate reaches the band's floor. Coming down from an expensive one we want the
    *largest* Δβ still under the band's ceiling, which is (smallest above it) − 1.
    """
    if below:
        return _lower_bound(s, lo, hi, goal, budget)
    d_above, edge = _lower_bound(s, lo, hi, goal * (1 + 1e-12), budget)
    return clip_delta_beta(d_above - 1), edge


def _nearest(s: RateSearch, d: int, target_bpp: float) -> RatePoint:
    """`d` or `d − 1`, whichever lands closer to the target.

    Only used with `stay_near_anchor=False`. The bisection returns the first Δβ at or
    above the target, so its neighbour below is the other candidate and one of the two
    is the closest achievable rate. Both are usually already in the probe cache.
    """
    here = s.probe(d)
    if d <= DELTA_BETA_MIN:
        return here
    prev = s.probe(d - 1)
    return prev if abs(prev.bpp - target_bpp) < abs(here.bpp - target_bpp) else here


def _within(bpp: float, target: float, tolerance: float) -> bool:
    return abs(bpp - target) <= tolerance * target + 1e-12


def match_rate(model, x: Tensor, target_bpp: float, *,
               q_index: Tensor | None = None,
               tolerance: float = TOLERANCE,
               window: int = BISECT_WINDOW,
               max_iterations: int = MAX_ITERATIONS,
               stay_near_anchor: bool = True,
               search: RateSearch | None = None) -> BRMResult:
    """Find the Δβ that puts one picture at `target_bpp`, for one model.

    This is the Δβ search and validation stages; `brm()` adds model selection in
    front. Pass an existing `search` to reuse a cache that model selection already
    built -- the whole point of Fig. 9 is not to build it twice.

    `stay_near_anchor=True` is the paper's rule and the default. Setting it False aims
    at the target itself, which lands *more accurately* and codes slightly *worse*:
    that is the trade the paper made in the other direction, and having both here is
    what lets the report quantify it rather than assert it.
    """
    if target_bpp <= 0:
        raise ValueError(f"target_bpp must be positive, got {target_bpp}")
    s = search if search is not None else RateSearch(model, x, q_index=q_index)
    anchor = s.default()

    # The anchor first. If it is already inside the band it *is* the answer -- the
    # paper's rule is "closest to the default rate point", and nothing is closer.
    # This is the case that keeps BRM's BD-rate penalty at a few points: no search
    # runs, no quality is given up, and the encode costs one arithmetic pass.
    if stay_near_anchor and _within(anchor.bpp, target_bpp, tolerance):
        return BRMResult(target_bpp, 0, anchor.bpp, anchor.packet,
                         encodes=s.encodes, passes=s.passes, trace=[anchor])

    # Eq. (14): fit on the two ends of the clamp, then solve at log(Rt).
    lo_pt, hi_pt = s.probe(DELTA_BETA_MIN), s.probe(DELTA_BETA_MAX)
    a, b = linear_fit(lo_pt, hi_pt)
    d1 = solve_fit(a, b, target_bpp)

    # Which edge of the tolerance band to aim for. Rate rises with Δβ, so when the
    # anchor is too cheap we want the *lowest* rate that qualifies, and when it is
    # too expensive we want the highest -- either way, the one nearest the anchor.
    below = anchor.bpp < target_bpp
    goal = target_bpp
    if stay_near_anchor:
        goal *= (1 - tolerance) if below else (1 + tolerance)

    def pick(lo: int, hi: int) -> tuple[RatePoint, bool]:
        """One bisection over `[lo, hi]`: the chosen point, and whether the window
        turned out not to bracket the answer at all."""
        if not stay_near_anchor:
            d_at, edge = _lower_bound(s, lo, hi, goal, max_iterations)
            return _nearest(s, d_at, target_bpp), edge
        d, edge = _band_edge(s, lo, hi, goal, below, max_iterations)
        return s.probe(d), edge

    best, edge = pick(d1 - window, d1 + window)

    # The window may have missed -- eq. (14)'s fit is a chord across a curve that is
    # concave in log space, so it can put Δβ,1 on the wrong side of the anchor
    # entirely. Widening to the full clamp is ours, not the paper's: refusing to
    # answer because a heuristic window was 100 too narrow would be worse than one
    # extra bisection, and the bisection is 11 probes on a cache that is already warm.
    if edge and not _within(best.bpp, target_bpp, tolerance):
        best, _ = pick(DELTA_BETA_MIN, DELTA_BETA_MAX)

    # Clamped means "the range ran out", which is only true if the end of the range
    # is genuinely on the wrong side of the target -- Δβ = 702 landing exactly on a
    # requested rate is a success, not a failure.
    at_end = best.delta_beta in (DELTA_BETA_MIN, DELTA_BETA_MAX)
    clamped = at_end and not _within(best.bpp, target_bpp, tolerance)
    return BRMResult(target_bpp, best.delta_beta, best.bpp, best.packet,
                     encodes=s.encodes, passes=s.passes, clamped=clamped,
                     trace=s.trace())


def select_model(models, x: Tensor, target_bpp: float, *,
                 q_index: Tensor | None = None) -> tuple[int, list[float], list]:
    """Eq. (13): the model whose default rate is relatively closest to the target.

    Returns the winning index, every model's `Dr`, and the `RateSearch` objects, so
    the caller can hand the winner's cache straight to `match_rate` instead of
    re-encoding the picture a fifth time.

    `Dr = |Rd − Rt| / Rd` divides by the model's own default rate, not by the target.
    That tilts ties toward the model with the *higher* `Rd`, i.e. toward pulling a
    rate down rather than pushing one up -- Fig. 6's asymmetry, expressed as a
    selection rule. Worth checking by hand: at `Rt = 0.3` a model at `Rd = 0.2` scores
    0.50 and one at `Rd = 0.45` scores 0.33, so the more expensive model wins even
    though it is further away in absolute bpp.
    """
    if not models:
        raise ValueError("no models to select from")
    searches = [RateSearch(m, x, q_index=q_index) for m in models]
    drs = [abs(s.default().bpp - target_bpp) / max(s.default().bpp, 1e-9)
           for s in searches]
    return min(range(len(drs)), key=lambda i: drs[i]), drs, searches


@torch.no_grad()
def brm(models, x: Tensor, target_bpp: float, *,
        q_index: Tensor | None = None, **kw) -> BRMResult:
    """The whole algorithm: select a model, search Δβ, validate.

    `models` is the ladder of trained β_train checkpoints -- four of them in JPEG AI.
    A single-element list is legal and skips straight to the search, which is the
    honest way to report what *one* model can do.

    The reported `encodes` and `passes` are the totals across every model, selection
    included. Selection is not free: four anchors is four analysis transforms and four
    factorised-prior compressions, and on a four-model ladder that is the larger half
    of the bill even when the search itself stops immediately.
    """
    idx, drs, searches = select_model(models, x, target_bpp, q_index=q_index)
    out = match_rate(models[idx], x, target_bpp, q_index=q_index,
                     search=searches[idx], **kw)
    out.model_index, out.dr = idx, drs[idx]
    out.encodes = sum(s.encodes for s in searches)
    out.passes = sum(s.passes for s in searches)
    return out


@torch.no_grad()
def rate_ladder(model, x: Tensor, delta_betas=None, *,
                q_index: Tensor | None = None) -> list[RatePoint]:
    """One model, many rate points -- the fixed-rate ladder, for comparison.

    This is what our own results are measured on: no rate target, no search, just Δβ
    swept over its range. Uses the same cache, so a nine-point ladder costs one
    forward pass and nine arithmetic-coding passes.

    The default sweep is a copy of `rate.beta_eval_points`, which is the real source of
    truth; it lives here too so the function is usable without a config, and the two
    are checked against each other in `tests/test_brm.py`.
    """
    if delta_betas is None:
        delta_betas = (DELTA_BETA_MIN, -800, -600, -400, -200, 0, 200, 450,
                       DELTA_BETA_MAX)
    s = RateSearch(model, x, q_index=q_index)
    return [s.probe(d) for d in delta_betas]


def delta_beta_for(beta_test: float, beta_train: float, sigma_index) -> int:
    """Eqs. (9) and (10) against a live σ grid -- the ladder driver's entry point.

    Takes the grid rather than the two constants because the whole derivation is that
    `S_σ` *is* the grid's log step; passing them separately invites the two ends of
    the codec to disagree about it.
    """
    return beta_displacement(beta_test, beta_train,
                             log_k=sigma_index.log_k, step=sigma_index.step)


def beta_ratio_of(delta_beta: int, sigma_index) -> float:
    """The `δβ` an integer Δβ achieves -- eq. (10) inverted, for reporting.

    Not the inverse of `delta_beta_for`: eq. (10) floors, so a requested ratio comes
    back up to 0.16% low. Reporting the requested ratio as though it had been achieved
    is the kind of small lie that shows up as a kink in a BD-rate curve.
    """
    return beta_ratio(delta_beta, log_k=sigma_index.log_k, step=sigma_index.step)


@torch.no_grad()
def beta_ladder(model, x: Tensor, betas, beta_train: float, *,
                q_index: Tensor | None = None) -> list[tuple[float, RatePoint]]:
    """A ladder addressed by β rather than by Δβ -- the driver Table II implies.

    JPEG AI's 18-entry `rate.beta_list` is the ladder the reference encoder exposes,
    and each β on it becomes a Δβ against the checkpoint's own `beta_train` (eqs. 9
    and 10). This is the *other* way to ask for a rate, and it is the honest one for a
    lambda sweep: the previous phases got one checkpoint per β, so pairing each β with
    the Δβ that claims to imitate it is what makes "one model, nine rate points"
    comparable against "nine models, nine rate points".

    Requesting a β outside `[β_train·0.1876, β_train·3.0008]` clamps, and the returned
    Δβ shows it: two βs mapping to the same Δβ is the clamp, not a bug.
    """
    si = model.sigma_index
    s = RateSearch(model, x, q_index=q_index)
    return [(float(b), s.probe(delta_beta_for(float(b), beta_train, si)))
            for b in betas]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A self-test on a tiny random codec. The gain unit is untrained, so the rates are
    # meaningless as *rates* -- what is under test is that the search machinery finds
    # the point it claims to find, which does not depend on the weights being good.
    from jpegai.models.twobranch import TwoBranchCodec

    torch.manual_seed(0)
    model = TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                           chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                           synthesis_width=(24, 16, 16, 16), internal_format="420",
                           mean_scale=True, split_hyper=True, mcm=False,
                           gain=True).eval()
    model.update(force=True)
    x = torch.rand(1, 3, 128, 192)

    print("the fixed-rate ladder (one forward pass, nine coding passes)")
    print(f"  {'Δβ':>6} {'bytes':>7} {'bpp':>8} {'δβ':>7}")
    ladder = rate_ladder(model, x)
    for p in ladder:
        ratio = beta_ratio_of(p.delta_beta, model.sigma_index)
        print(f"  {p.delta_beta:>6} {p.nbytes:>7} {p.bpp:>8.4f} {ratio:>7.3f}")
    monotone = all(b.bpp >= a.bpp for a, b in zip(ladder, ladder[1:]))
    print(f"  rate non-decreasing in Δβ: {monotone}")

    lo, hi = ladder[0].bpp, ladder[-1].bpp
    print(f"\nreachable rate span for this one model: {lo:.4f} .. {hi:.4f} bpp "
          f"({hi / lo:.2f}x)")

    print("\nbit-rate matching, one model")
    print(f"  {'target':>7} -> {'Δβ':>6} {'bpp':>8} {'err':>8} {'enc':>4} "
          f"{'pass':>5}  flag")
    anchor_bpp = ladder[5].bpp
    for frac in (0.45, 0.9, 1.0, 1.1, 2.0, 8.0):
        target = anchor_bpp * frac
        r = match_rate(model, x, target)
        flag = "ok" if r.in_tolerance else ("clamped" if r.clamped else "MISS")
        print(f"  {target:>7.4f} -> {r.delta_beta:>6} {r.bpp:>8.4f} "
              f"{r.rate_error:>+8.1%} {r.encodes:>4} {r.passes:>5}  {flag}")
        # Every answer is a real encode, so the packet must decode.
        assert model.decompress(r.packet)["x_hat"].shape == x.shape

    print("\nthe anchor shortcut: a target the anchor already meets costs one encode")
    r = match_rate(model, x, anchor_bpp)
    print(f"  {r.summary()}")
    assert r.delta_beta == 0 and r.encodes == 1 and r.in_tolerance

    print("\nstay_near_anchor: the paper's rule vs aiming at the target")
    for frac in (0.6, 1.6):
        target = anchor_bpp * frac
        near = match_rate(model, x, target, stay_near_anchor=True)
        exact = match_rate(model, x, target, stay_near_anchor=False)
        print(f"  target {target:.4f}   near-anchor Δβ {near.delta_beta:+5d} "
              f"({near.rate_error:+.1%})   at-target Δβ {exact.delta_beta:+5d} "
              f"({exact.rate_error:+.1%})")
        assert abs(exact.rate_error) <= abs(near.rate_error) + 1e-9
        assert abs(exact.delta_beta) >= abs(near.delta_beta)

    print("\neq. (13) model selection over a two-model ladder")
    other = TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                           chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                           synthesis_width=(24, 16, 16, 16), internal_format="420",
                           mean_scale=True, split_hyper=True, mcm=False,
                           gain=True).eval()
    other.update(force=True)
    with torch.no_grad():
        for v in other.gain_parameters():
            v.fill_(300.0)
    for frac in (0.5, 1.0, 3.0):
        r = brm([model, other], x, anchor_bpp * frac)
        print(f"  {r.summary()}  Dr {r.dr:.3f}")

    print("\nβ-addressed ladder (eqs. 9 and 10, against β_train = 0.075)")
    print(f"  {'β_test':>8} {'Δβ':>6} {'bpp':>8}")
    for b, p in beta_ladder(model, x, (0.015, 0.03, 0.075, 0.2, 0.5), 0.075):
        print(f"  {b:>8g} {p.delta_beta:>6} {p.bpp:>8.4f}")

    print("\nall BRM invariants hold")
