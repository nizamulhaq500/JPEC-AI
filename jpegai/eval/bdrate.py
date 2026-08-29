"""Bjontegaard delta-rate.

The one number every codec comparison reduces to. Written from scratch rather
than pulled from a library because (a) it is 30 lines, (b) every learned-codec
paper computes it slightly differently and you need to know which variant you
used, and (c) getting the overlap handling wrong silently produces flattering
numbers.

Convention throughout this project, matching the paper:

    negative BD-rate = the test codec needs FEWER bits for the same quality
                     = the test codec is BETTER

The paper (Tables III-VI) reports BD-rate *per metric*, then averages the seven
metrics. That is not the same as averaging the metrics first and computing one
BD-rate -- see `bd_rate_table`.

Interpolant: **monotone piecewise cubic (PCHIP)**, not the global cubic of the
original 2001 Bjontegaard note. The difference is not cosmetic and it bit this
project once, so it is worth stating why.

Bjontegaard's cubic assumes both curves are well described by one polynomial over
the whole integration range. That holds for PSNR against log-rate. It fails badly
for the *saturating* metrics in this project's set -- `fsim`, `ms_ssim`, `iw_ssim`
all crowd into the last 1% of their range, so quality is a near-vertical function
of log-rate at the top and a gentle one at the bottom. A single cubic cannot be
both, and when the two codecs' quality ranges overlap only partially the fit is
dominated by anchor points *outside* the integration window -- which then set the
answer inside it.

Measured on `results/bench_p5.json` (Kodak, our tier-full ladder vs JPEG), where
our 5 points overlap only 7 of JPEG's 11:

    metric      global cubic     PCHIP     linear
    fsim              +56.0%     -16.6%     -11.0%
    ms_ssim            -1.1%     -26.2%     -21.5%
    iw_ssim           +12.5%      -5.6%      -1.9%
    psnr_hvs          +30.9%     +30.4%     +30.6%

The cubic reported +56% on a metric where our codec uses fewer bits than JPEG at
*every* measured quality -- an impossible sign. Two independent local methods agree
to a few percent everywhere; the cubic is the outlier, and only where the curve
saturates (`psnr_hvs`, which does not saturate, is stable across all three). It
flattered the anchors too: webp `fsim` read -16.2% under the cubic and -3.4% here.

PCHIP is monotone and local: it interpolates the measured points exactly, never
overshoots between them, and a point outside the overlap cannot influence the
integral inside it. It is what the JVET common test conditions moved to for the
same reason.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = ["bd_rate", "bd_psnr", "bd_rate_table", "overlap_coverage"]

# Points in the numeric integration over the overlapping interval. PCHIP is cheap
# and the integrand is smooth, so this is far more than needed for 4 decimals.
_GRID = 4001


def _integrate(x_a, y_a, x_t, y_t):
    """Mean of (test - anchor) over the overlapping x-range, via monotone PCHIP.

    Returns NaN when the ranges do not overlap -- the honest answer, since the
    curves are then not comparable. Shared by `bd_rate` and `bd_psnr`, which
    differ only in which variable is the abscissa.
    """
    lo = max(x_a.min(), x_t.min())
    hi = min(x_a.max(), x_t.max())
    if not (hi > lo):
        return float("nan")

    grid = np.linspace(lo, hi, _GRID)
    diff = PchipInterpolator(x_t, y_t)(grid) - PchipInterpolator(x_a, y_a)(grid)
    return float(np.trapezoid(diff, grid) / (hi - lo))


def _prep(rate, qual):
    """Sort by quality, drop non-finite points, return (log-rate, quality).

    PCHIP needs a *strictly* increasing abscissa, so ties in quality are collapsed
    to one point at their mean log-rate. Ties happen for real: a saturating metric
    can return the same value at two adjacent quality settings once it has run out
    of headroom, and the old global fit silently tolerated that.
    """
    rate = np.asarray(rate, dtype=np.float64).ravel()
    qual = np.asarray(qual, dtype=np.float64).ravel()
    if rate.size != qual.size:
        raise ValueError(f"rate/quality length mismatch: {rate.size} vs {qual.size}")
    keep = np.isfinite(rate) & np.isfinite(qual) & (rate > 0)
    rate, qual = rate[keep], qual[keep]
    order = np.argsort(qual, kind="stable")
    lr, q = np.log(rate[order]), qual[order]

    uniq, inverse = np.unique(q, return_inverse=True)
    if uniq.size != q.size:
        lr = np.bincount(inverse, weights=lr) / np.bincount(inverse)
        q = uniq
    return lr, q


def bd_rate(rate_anchor, qual_anchor, rate_test, qual_test, *, min_points: int = 4) -> float:
    """Average bitrate difference of `test` vs `anchor` at equal quality, in %.

    Interpolates (quality -> log rate) for each codec with a monotone PCHIP,
    integrates the difference over the *overlapping* quality range, and
    exponentiates back.

    Negative = test is better.

    Needs >= `min_points` rate points per codec. Returns NaN if the quality ranges
    do not overlap, which is the honest answer -- the curves are not comparable.
    """
    lr_a, q_a = _prep(rate_anchor, qual_anchor)
    lr_t, q_t = _prep(rate_test, qual_test)

    if lr_a.size < min_points or lr_t.size < min_points:
        raise ValueError(
            f"need >= {min_points} distinct-quality points, "
            f"got {lr_a.size} and {lr_t.size}"
        )

    # Overlapping quality interval only. Extrapolating outside the measured range
    # is how people accidentally report -60% BD-rate.
    avg_log_diff = _integrate(q_a, lr_a, q_t, lr_t)
    if not np.isfinite(avg_log_diff):
        return float("nan")
    return float((np.exp(avg_log_diff) - 1.0) * 100.0)


def bd_psnr(rate_anchor, qual_anchor, rate_test, qual_test, *, min_points: int = 4) -> float:
    """Average quality difference at equal rate. Positive = test is better.

    The dual of `bd_rate`, and the more informative one when the two codecs' rate
    ranges overlap better than their quality ranges -- which is exactly the case
    for a 5-point neural ladder against an 11-point anchor sweep.
    """
    lr_a, q_a = _prep(rate_anchor, qual_anchor)
    lr_t, q_t = _prep(rate_test, qual_test)

    if lr_a.size < min_points or lr_t.size < min_points:
        raise ValueError(f"need >= {min_points} points, got {lr_a.size} and {lr_t.size}")

    # Abscissa and ordinate swap: integrate quality over the shared log-rate range.
    # `_prep` sorted by quality; PCHIP needs log-rate increasing, so re-sort.
    oa, ot = np.argsort(lr_a), np.argsort(lr_t)
    return _integrate(lr_a[oa], q_a[oa], lr_t[ot], q_t[ot])


def overlap_coverage(anchor: dict, test: dict, metric: str) -> tuple[int, int]:
    """How many of the anchor's points fall inside the shared quality range.

    The BD-rate of a 5-point ladder against an 11-point anchor sweep is only as
    trustworthy as that shared range. When the test curve covers a small slice of
    the anchor's, the integral is decided by a handful of anchor points and the
    right response is to *train lower-rate points*, not to quote the number harder.
    Returns (points_in_overlap, total_anchor_points).
    """
    _, q_a = _prep(anchor["bpp"], anchor[metric])
    _, q_t = _prep(test["bpp"], test[metric])
    lo, hi = max(q_a.min(), q_t.min()), min(q_a.max(), q_t.max())
    if not (hi > lo):
        return 0, int(q_a.size)
    return int(((q_a >= lo) & (q_a <= hi)).sum()), int(q_a.size)


def bd_rate_table(anchor: dict, test: dict, metrics: list[str] | None = None) -> dict:
    """Per-metric BD-rate plus the average, the way the paper's tables do it.

    Both `anchor` and `test` are dicts of the shape produced by
    `jpegai.eval.runbench`::

        {"bpp": [...], "ms_ssim": [...], "vif": [...], ...}

    One BD-rate is computed per metric, and "AVG" is the unweighted mean of
    those -- NOT a BD-rate of averaged metrics. Metrics missing from either
    dict, or with too few points, are skipped and reported in "_skipped".

    "_coverage" carries the worst `overlap_coverage` across the metrics, as
    (points_in_overlap, total). Callers should surface it: a low ratio is the
    condition under which BD-rate is fragile no matter how it is interpolated.

    Returns e.g. {"ms_ssim": -33.0, "vif": 1.4, ..., "AVG": -20.2, "_skipped": []}
    """
    if metrics is None:
        metrics = [m for m in anchor if m != "bpp" and m in test]

    out: dict[str, float] = {}
    skipped: list[str] = []
    worst: tuple[int, int] | None = None
    for m in metrics:
        try:
            out[m] = bd_rate(anchor["bpp"], anchor[m], test["bpp"], test[m])
            cov = overlap_coverage(anchor, test, m)
        except (ValueError, KeyError) as exc:  # too few points, or absent
            skipped.append(f"{m}: {exc}")
            continue
        if worst is None or cov[0] < worst[0]:
            worst = cov

    finite = [v for v in out.values() if np.isfinite(v)]
    out["AVG"] = float(np.mean(finite)) if finite else float("nan")
    out["_skipped"] = skipped
    out["_coverage"] = worst if worst is not None else (0, 0)
    return out


if __name__ == "__main__":
    # Sanity checks. `python -m jpegai.eval.bdrate`
    rate = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
    qual = np.array([26.0, 29.0, 32.0, 35.0, 38.0])

    identical = bd_rate(rate, qual, rate, qual)
    print(f"self vs self          : {identical:+.6f} %   (must be 0)")
    assert abs(identical) < 1e-9

    # Test codec needs exactly half the bits at every quality -> -50%.
    half = bd_rate(rate, qual, rate / 2, qual)
    print(f"half the bitrate      : {half:+.4f} %   (must be -50)")
    assert abs(half + 50.0) < 1e-6

    doubled = bd_rate(rate, qual, rate * 2, qual)
    print(f"double the bitrate    : {doubled:+.4f} %   (must be +100)")
    assert abs(doubled - 100.0) < 1e-6

    # No quality overlap -> NaN, not a fabricated number.
    disjoint = bd_rate(rate, qual, rate, qual + 100)
    print(f"disjoint quality range: {disjoint}          (must be nan)")
    assert np.isnan(disjoint)

    # The case that made this module change interpolants, reduced to arithmetic.
    #
    # A saturating metric (q -> 1 as rate -> inf) sampled by an anchor sweep, and a
    # 5-point ladder that only reaches the *top* of that range -- which is exactly a
    # neural rate ladder against libjpeg on Kodak. The ladder is built to need half
    # the anchor's bits at equal quality, so the true answer is -50% and nothing else.
    #
    # The property under test is invariance: all four anchor sweeps below pass
    # through the integration window identically and differ only in how far *below*
    # it they extend. A BD-rate cannot legitimately depend on that. PCHIP is local,
    # so it does not; the global cubic is fitted to all the points at once, so the
    # region outside the window sets the answer inside it.
    q_t = np.linspace(0.977, 0.9989, 5)        # the top slice only, 5 points
    r_t = (1.0 / (1.0 - q_t)) / 2.0            # half the bits at the same quality

    def _cubic(ra, qa, rt, qt):
        """The global cubic this module used to fit. Kept as an executable footnote."""
        pa, pt = np.polyfit(qa, np.log(ra), 3), np.polyfit(qt, np.log(rt), 3)
        lo, hi = max(qa.min(), qt.min()), min(qa.max(), qt.max())
        ia = np.polyval(np.polyint(pa), [lo, hi])
        it = np.polyval(np.polyint(pt), [lo, hi])
        return (np.exp(((it[1] - it[0]) - (ia[1] - ia[0])) / (hi - lo)) - 1.0) * 100.0

    print("\nsaturating metric, ladder covers only the top of the anchor's range.")
    print("truth is -50% for every row; only the anchor's *outside* sampling varies.")
    print(f"  {'anchor rate span':>20s} {'overlap':>10s} {'PCHIP':>9s} {'cubic':>9s}")
    pchip_all, cubic_all = [], []
    for lo_r, hi_r in [(4.0, 1600.0), (2.0, 3000.0), (1.5, 4000.0), (1.33, 8000.0)]:
        r_a = np.geomspace(lo_r, hi_r, 11)
        q_a = 1.0 - 1.0 / r_a
        n, tot = overlap_coverage({"bpp": r_a, "m": q_a}, {"bpp": r_t, "m": q_t}, "m")
        p, c = bd_rate(r_a, q_a, r_t, q_t), _cubic(r_a, q_a, r_t, q_t)
        pchip_all.append(p)
        cubic_all.append(c)
        print(f"  {lo_r:7.2f} .. {hi_r:<8.0f} {n:4d} of {tot:<3d} {p:+9.2f} {c:+9.2f}")

    print(f"\n  PCHIP spread across the four : {max(pchip_all) - min(pchip_all):5.2f} points")
    print(f"  cubic spread across the four : {max(cubic_all) - min(cubic_all):5.2f} points")

    # PCHIP: close to truth, and the same answer however the anchor was sampled.
    assert all(abs(p + 50.0) < 3.0 for p in pchip_all), pchip_all
    assert max(pchip_all) - min(pchip_all) < 0.5, pchip_all
    # The cubic: neither. This is what made `fsim` read +56% on real data.
    assert max(cubic_all) - min(cubic_all) > 10.0, cubic_all

    print("\nbdrate.py: all checks passed")
