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
"""

from __future__ import annotations

import numpy as np

__all__ = ["bd_rate", "bd_psnr", "bd_rate_table"]


def _prep(rate, qual):
    """Sort by quality, drop non-finite points, return (log-rate, quality)."""
    rate = np.asarray(rate, dtype=np.float64).ravel()
    qual = np.asarray(qual, dtype=np.float64).ravel()
    if rate.size != qual.size:
        raise ValueError(f"rate/quality length mismatch: {rate.size} vs {qual.size}")
    keep = np.isfinite(rate) & np.isfinite(qual) & (rate > 0)
    rate, qual = rate[keep], qual[keep]
    order = np.argsort(qual)
    return np.log(rate[order]), qual[order]


def bd_rate(rate_anchor, qual_anchor, rate_test, qual_test, *, degree: int = 3) -> float:
    """Average bitrate difference of `test` vs `anchor` at equal quality, in %.

    Fits a cubic to (quality -> log rate) for each codec, integrates the
    difference over the *overlapping* quality range, and exponentiates back.

    Negative = test is better.

    Needs >= degree+1 rate points per codec (4 for a cubic). Returns NaN if the
    quality ranges do not overlap, which is the honest answer -- the curves are
    not comparable.
    """
    lr_a, q_a = _prep(rate_anchor, qual_anchor)
    lr_t, q_t = _prep(rate_test, qual_test)

    n = degree + 1
    if lr_a.size < n or lr_t.size < n:
        raise ValueError(
            f"need >= {n} points for degree-{degree} fit, got {lr_a.size} and {lr_t.size}"
        )

    # Overlapping quality interval only. Extrapolating a cubic outside the
    # measured range is how people accidentally report -60% BD-rate.
    lo = max(q_a.min(), q_t.min())
    hi = min(q_a.max(), q_t.max())
    if not (hi > lo):
        return float("nan")

    p_a = np.polyfit(q_a, lr_a, degree)
    p_t = np.polyfit(q_t, lr_t, degree)

    int_a = np.polyval(np.polyint(p_a), [lo, hi])
    int_t = np.polyval(np.polyint(p_t), [lo, hi])

    avg_log_diff = ((int_t[1] - int_t[0]) - (int_a[1] - int_a[0])) / (hi - lo)
    return float((np.exp(avg_log_diff) - 1.0) * 100.0)


def bd_psnr(rate_anchor, qual_anchor, rate_test, qual_test, *, degree: int = 3) -> float:
    """Average quality difference at equal rate. Positive = test is better.

    The dual of bd_rate. Less commonly reported but useful when the two codecs'
    rate ranges overlap better than their quality ranges.
    """
    lr_a, q_a = _prep(rate_anchor, qual_anchor)
    lr_t, q_t = _prep(rate_test, qual_test)

    n = degree + 1
    if lr_a.size < n or lr_t.size < n:
        raise ValueError(f"need >= {n} points, got {lr_a.size} and {lr_t.size}")

    lo = max(lr_a.min(), lr_t.min())
    hi = min(lr_a.max(), lr_t.max())
    if not (hi > lo):
        return float("nan")

    p_a = np.polyfit(lr_a, q_a, degree)
    p_t = np.polyfit(lr_t, q_t, degree)
    int_a = np.polyval(np.polyint(p_a), [lo, hi])
    int_t = np.polyval(np.polyint(p_t), [lo, hi])
    return float(((int_t[1] - int_t[0]) - (int_a[1] - int_a[0])) / (hi - lo))


def bd_rate_table(anchor: dict, test: dict, metrics: list[str] | None = None) -> dict:
    """Per-metric BD-rate plus the average, the way the paper's tables do it.

    Both `anchor` and `test` are dicts of the shape produced by
    `jpegai.eval.runbench`::

        {"bpp": [...], "ms_ssim": [...], "vif": [...], ...}

    One BD-rate is computed per metric, and "AVG" is the unweighted mean of
    those -- NOT a BD-rate of averaged metrics. Metrics missing from either
    dict, or with too few points, are skipped and reported in "_skipped".

    Returns e.g. {"ms_ssim": -33.0, "vif": 1.4, ..., "AVG": -20.2, "_skipped": []}
    """
    if metrics is None:
        metrics = [m for m in anchor if m != "bpp" and m in test]

    out: dict[str, float] = {}
    skipped: list[str] = []
    for m in metrics:
        try:
            out[m] = bd_rate(anchor["bpp"], anchor[m], test["bpp"], test[m])
        except (ValueError, KeyError) as exc:  # too few points, or absent
            skipped.append(f"{m}: {exc}")
            continue

    finite = [v for v in out.values() if np.isfinite(v)]
    out["AVG"] = float(np.mean(finite)) if finite else float("nan")
    out["_skipped"] = skipped
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

    print("\nbdrate.py: all checks passed")
