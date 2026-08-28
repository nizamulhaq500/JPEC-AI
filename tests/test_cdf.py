"""Tests for the quantized-CDF layer -- the part where a subtle error costs rate
silently instead of crashing.

    python tests/test_cdf.py
    pytest tests/test_cdf.py

Numpy only, except for two tests at the end that check the table against the real
rANS coder; those skip if compressai is missing.

What is worth testing here, and what is not. `pmf_to_quantized_cdf` cannot really
be checked against a reference value -- the "right" answer is whatever keeps the
coder consistent with itself. What *can* be checked is that its invariants hold on
adversarial inputs, and that the **coding cost it implies stays close to the
entropy of the distribution it came from**. That second property is the one that
matters: a table can satisfy every structural invariant and still cost 20% extra
rate, and the resulting codec looks correct while being quietly bad.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpegai.models.cdf import (  # noqa: E402
    MAX_PRECISION,
    build_cdf_table,
    cdf_cost_bits,
    pmf_to_quantized_cdf,
)


# ---------------------------------------------------------------------------
# PMF generators, including the awkward cases
# ---------------------------------------------------------------------------
def _gauss_pmf(n: int, sigma: float) -> np.ndarray:
    x = np.arange(n) - (n - 1) / 2.0
    return np.exp(-0.5 * (x / sigma) ** 2)


def _pmf_zoo(rng) -> list[tuple[str, np.ndarray]]:
    """Distributions that break naive implementations."""
    zoo = [
        ("uniform-2", np.ones(2)),
        ("uniform-33", np.ones(33)),
        ("gauss-wide", _gauss_pmf(65, 12.0)),
        ("gauss-narrow", _gauss_pmf(65, 0.4)),        # nearly a point mass
        ("gauss-tiny-sigma", _gauss_pmf(65, 0.11)),   # sigma_quant_min
        ("one-hot", np.eye(1, 40, 17).ravel()),       # exact zeros everywhere else
        ("with-zeros", np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.3])),
        ("unnormalised", np.array([4.0, 2.0, 1.0])),  # sums to 7, not 1
        ("denormal-tail", np.concatenate([[1.0], np.full(30, 1e-300)])),
        ("descending", np.exp(-np.arange(50) / 3.0)),
        ("monotone-spike", np.concatenate([np.full(20, 1e-8), [1.0]])),
    ]
    for i in range(12):
        n = int(rng.integers(2, 90))
        zoo.append((f"random-{i}", rng.random(n) ** 4 + 1e-12))
    return zoo


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------
def test_invariants_hold_on_every_pmf():
    rng = np.random.default_rng(7)
    for name, pmf in _pmf_zoo(rng):
        for precision in (8, 12, 16):
            n = pmf.size
            if n + 1 > (1 << precision):
                continue
            cdf = pmf_to_quantized_cdf(pmf, precision)
            w = np.diff(cdf.astype(np.int64))
            assert cdf.dtype == np.int32, name
            assert cdf.size == n + 1, f"{name}: {cdf.size} != {n + 1}"
            assert cdf[0] == 0, name
            assert int(cdf[-1]) == 1 << precision, f"{name}: total {cdf[-1]}"
            assert w.min() >= 1, f"{name} p{precision}: zero-width bin"
            assert int(w.sum()) == 1 << precision, f"{name}: mass not conserved"


def test_mass_is_conserved_not_created():
    """Widening zero bins must steal, never mint. Otherwise the total drifts and
    the decoder desynchronises after the first few thousand symbols."""
    for precision in (8, 10, 16):
        pmf = np.concatenate([[1.0], np.full(60, 1e-40)])
        if 61 + 1 > (1 << precision):
            continue
        cdf = pmf_to_quantized_cdf(pmf, precision)
        assert int(np.diff(cdf.astype(np.int64)).sum()) == 1 << precision


def test_ordering_is_preserved_for_well_separated_symbols():
    """A symbol with clearly more mass must not end up with a narrower bin.

    Only asserted where the float gap exceeds one quantisation unit -- below that
    the ordering genuinely may invert, and demanding otherwise would be demanding
    more resolution than the table has.
    """
    precision = 16
    unit = 1.0 / (1 << precision)
    rng = np.random.default_rng(11)
    pmf = rng.random(40) + 1e-3
    p = pmf / pmf.sum()
    cdf = pmf_to_quantized_cdf(pmf, precision)
    w = np.diff(cdf.astype(np.int64))
    bad = [(i, j) for i in range(40) for j in range(40)
           if p[i] - p[j] > 4 * unit and w[i] < w[j]]
    assert not bad, f"ordering inverted for {bad[:4]}"


def test_deterministic():
    """Same input, same bytes -- across calls and across dtypes of the input.

    Non-determinism here would make a codestream depend on how the PMF was
    constructed, which would break Phase 11's bit-exactness requirement in a way
    that is almost impossible to find later.
    """
    pmf = _gauss_pmf(63, 3.0)
    a = pmf_to_quantized_cdf(pmf, 16)
    b = pmf_to_quantized_cdf(pmf, 16)
    c = pmf_to_quantized_cdf(pmf.astype(np.float32).astype(np.float64), 16)
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_rejects_bad_input():
    for bad, why in [
        (np.array([]), "empty"),
        (np.array([0.0, 0.0]), "sums to zero"),
        (np.array([1.0, -0.5]), "negative"),
        (np.array([1.0, np.nan]), "nan"),
        (np.array([1.0, np.inf]), "inf"),
    ]:
        try:
            pmf_to_quantized_cdf(bad, 16)
        except ValueError:
            continue
        raise AssertionError(f"accepted {why} pmf")

    for precision in (0, -1, MAX_PRECISION + 1):
        try:
            pmf_to_quantized_cdf(np.ones(4), precision)
        except ValueError:
            continue
        raise AssertionError(f"accepted precision {precision}")


def test_alphabet_too_large_for_precision_raises():
    """More symbols than units of mass is unrepresentable. It must raise, because
    the alternative -- silently merging two symbols -- produces a table the
    decoder cannot invert."""
    try:
        pmf_to_quantized_cdf(np.ones(300), precision=8)   # 300 symbols, 256 units
    except ValueError as exc:
        assert "do not fit" in str(exc) or "widen" in str(exc), str(exc)
        return
    raise AssertionError("accepted an alphabet larger than the mass")


# ---------------------------------------------------------------------------
# The property that actually matters: cost
# ---------------------------------------------------------------------------
def test_quantisation_overhead_is_small():
    """Expected cost under the quantized table vs the float entropy.

    The table can only lose: it is a coarser model of the same distribution. The
    question is how much. Anything under a few percent is quantisation; a large
    gap means the construction is wrong.

    This is the numpy-side analogue of the end-to-end gate in
    `jpegai.models.selftest`, and it isolates the table from the coder -- if this
    passes and the end-to-end gate fails, the bug is in the coder, and vice versa.
    """
    precision = 16
    worst = 0.0
    for name, pmf in _pmf_zoo(np.random.default_rng(5)):
        if pmf.size + 1 > (1 << precision):
            continue
        p = pmf / pmf.sum()
        cdf = pmf_to_quantized_cdf(pmf, precision)
        entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        cost = sum(p[i] * cdf_cost_bits(cdf, i, precision)
                   for i in range(pmf.size) if p[i] > 0)
        # Absolute floor as well as relative: for a near-deterministic
        # distribution the entropy is ~0 bits and any relative bound is
        # meaningless, but the absolute overhead must still be tiny.
        overhead = cost - entropy
        assert overhead > -1e-9, f"{name}: table beats entropy by {-overhead:.4g} bits"
        assert overhead < 0.02 or overhead < 0.01 * entropy, \
            f"{name}: +{overhead:.4g} bits over entropy {entropy:.4g}"
        worst = max(worst, overhead)
    assert worst < 0.02, f"worst-case overhead {worst:.4g} bits"


def test_overhead_shrinks_with_precision():
    """Overhead must fall monotonically as precision rises, and never rise."""
    pmf = _gauss_pmf(65, 2.0)
    p = pmf / pmf.sum()
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    prev = None
    for precision in (8, 10, 12, 14, 16):
        cdf = pmf_to_quantized_cdf(pmf, precision)
        cost = sum(p[i] * cdf_cost_bits(cdf, i, precision) for i in range(pmf.size))
        over = cost - entropy
        if prev is not None:
            assert over <= prev + 1e-12, f"p{precision}: overhead grew {prev} -> {over}"
        prev = over
    assert prev < 0.01, f"16-bit overhead {prev:.4g} bits is too large"


def test_overhead_is_the_widening_tax_and_nothing_else():
    """Pin *why* the residual overhead exists, so a real regression is visible.

    At 16-bit precision the overhead is not a fixed small number -- it scales with
    how many symbols are too rare to earn a bin. Measured:

        n=65  sigma=12    0 dead symbols   overhead 0.000000 bits
        n=17  sigma=2     0 dead           overhead 0.000001
        n=33  sigma=2    14 dead           overhead 0.000468
        n=65  sigma=2    46 dead           overhead 0.002721
        n=129 sigma=2   110 dead           overhead 0.005790

    Every dead symbol is given one unit of mass stolen from the frequent bins, so
    the frequent symbols pay about `-log2(1 - stolen / 2**precision)`. That
    quantity accounts for 40-100% of the measured overhead in each case; the
    remainder is ordinary half-unit rounding of the surviving bins.

    It is an *estimate* rather than a strict lower bound: a surviving bin can round
    up as well as down, and for a near-deterministic distribution (n=65,
    sigma=0.4) the measured overhead lands ~1e-5 *below* it. So the assertion
    below brackets the overhead between half and five times the estimate.

    Testing the *relationship* rather than an absolute constant is what makes this
    a real test: a construction that minted mass instead of stealing it, or stole
    from the widest bin instead of the narrowest, would break the bracket while
    still satisfying every structural invariant above.
    """
    precision = 16
    unit = 1 << precision
    for n, sigma in [(65, 12.0), (17, 2.0), (33, 2.0), (65, 2.0), (129, 2.0),
                     (65, 0.4)]:
        pmf = _gauss_pmf(n, sigma)
        p = pmf / pmf.sum()
        entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        cdf = pmf_to_quantized_cdf(pmf, precision)
        widths = np.diff(cdf.astype(np.int64))
        cost = sum(p[i] * cdf_cost_bits(cdf, i, precision) for i in range(n))
        over = cost - entropy

        rare = p * unit < 0.5                     # would have rounded to width 0
        stolen = int(widths[rare].sum())
        assert stolen == int(rare.sum()), \
            f"n={n}: widened bins should hold exactly 1 unit each, got {stolen}"
        est = -np.log2(1 - stolen / unit) if stolen else 0.0

        if stolen == 0:
            assert over < 1e-5, f"n={n} sigma={sigma}: nothing widened, " \
                                f"so overhead should be ~0, got {over:.3g}"
        else:
            assert over >= 0.5 * est - 1e-9, \
                f"n={n}: overhead {over:.3g} far below the widening tax {est:.3g}"
            assert over <= 5 * est + 5e-4, \
                f"n={n}: overhead {over:.3g} exceeds 5x the widening tax {est:.3g}"


def test_cost_matches_bin_width():
    cdf = pmf_to_quantized_cdf(np.array([1.0, 1.0, 2.0]), precision=8)
    # 256 units split 64/64/128 exactly -> 2, 2, 1 bits.
    assert [cdf_cost_bits(cdf, i, 8) for i in range(3)] == [2.0, 2.0, 1.0]
    try:
        cdf_cost_bits(np.array([0, 0, 4], dtype=np.int32), 0, 8)
    except ValueError:
        return
    raise AssertionError("zero-width bin did not raise")


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------
def test_table_lengths_and_padding():
    pmfs = np.zeros((3, 8))
    pmfs[0, :3] = [0.5, 0.3, 0.2]
    pmfs[1, :8] = _gauss_pmf(8, 2.0)
    pmfs[2, :2] = [0.9, 0.1]
    lengths_in = np.array([3, 8, 2])
    tails = np.array([1e-3, 1e-6, 0.2])
    cdfs, lengths = build_cdf_table(pmfs, tails, lengths_in, precision=16)

    assert cdfs.shape == (3, 10), cdfs.shape           # Lmax + 2
    assert list(lengths) == [5, 10, 4]                 # pmf_length + 2
    for i, L in enumerate(lengths):
        row = cdfs[i, :L].astype(np.int64)
        assert row[0] == 0
        assert row[-1] == 1 << 16, f"row {i} total {row[-1]}"
        assert np.all(np.diff(row) >= 1), f"row {i} not strictly increasing"
        # Padding past the row's length must stay zero: the decoder trusts
        # `lengths` and reading beyond it must be obviously wrong, not plausible.
        assert np.all(cdfs[i, L:] == 0), f"row {i} has junk in the padding"


def test_escape_symbol_carries_the_tail_mass():
    """The escape bin is the last real entry, and its width must track the tail.

    If the tail mass were dropped, every out-of-range symbol would be
    unencodable; if it were given a fixed width, the rate cost of escapes would
    not depend on how much mass actually escapes.
    """
    pmf = _gauss_pmf(16, 3.0)
    pmf = pmf / pmf.sum()
    widths = []
    for tail in (1e-9, 1e-4, 1e-2, 0.1):
        cdfs, lengths = build_cdf_table(pmf[None, :], np.array([tail]),
                                        np.array([16]), precision=16)
        row = cdfs[0, :lengths[0]].astype(np.int64)
        widths.append(int(row[-1] - row[-2]))          # the escape bin
    assert all(w >= 1 for w in widths), widths
    assert widths == sorted(widths), f"escape width not monotone in tail: {widths}"
    assert widths[-1] > widths[0] * 100, f"tail mass barely reflected: {widths}"


def test_build_rejects_shape_mismatch():
    pmfs = np.ones((3, 8))
    for tails, lens, why in [
        (np.ones(2), np.array([3, 3, 3]), "short tail_masses"),
        (np.ones(3), np.array([3, 3]), "short pmf_lengths"),
        (np.ones(3), np.array([3, 3, 99]), "length beyond row width"),
    ]:
        try:
            build_cdf_table(pmfs, tails, lens, precision=16)
        except ValueError:
            continue
        raise AssertionError(f"accepted {why}")


# ---------------------------------------------------------------------------
# Against the real coder
# ---------------------------------------------------------------------------
def _rans():
    try:
        from compressai.ans import RansDecoder, RansEncoder
    except ImportError:                                # pragma: no cover
        return None
    return RansEncoder, RansDecoder


def test_rans_roundtrip_uses_our_table():
    """The table we build is the table the coder consumes, bit for bit.

    This is the seam between our numpy CDF construction and compressai's rANS. It
    is the one place where a convention mismatch (an off-by-one in `cdf_length`,
    or the escape bin in the wrong position) shows up as a decode failure rather
    than as extra rate.
    """
    rans = _rans()
    if rans is None:
        print("  skip: compressai not installed")
        return
    RansEncoder, RansDecoder = rans

    precision = 16
    rng = np.random.default_rng(3)
    n_dist, lmax = 8, 32
    pmfs = np.zeros((n_dist, lmax))
    lengths_in = np.zeros(n_dist, dtype=np.int64)
    for i in range(n_dist):
        L = int(rng.integers(4, lmax + 1))
        pmfs[i, :L] = _gauss_pmf(L, 1.0 + 4.0 * rng.random())
        pmfs[i, :L] /= pmfs[i, :L].sum()
        lengths_in[i] = L
    cdfs, lengths = build_cdf_table(pmfs, np.full(n_dist, 1e-5), lengths_in,
                                    precision)

    # Sample symbols from the distributions they will be coded with, so the test
    # exercises the common path rather than only the tails.
    n = 4000
    idx = rng.integers(0, n_dist, n)
    sym = np.array([rng.choice(int(lengths_in[k]), p=pmfs[k, :int(lengths_in[k])])
                    for k in idx], dtype=np.int32)

    cdf_list = [cdfs[i, :lengths[i]].tolist() for i in range(n_dist)]
    offsets = [0] * n_dist
    enc = RansEncoder()
    stream = enc.encode_with_indexes(sym.tolist(), idx.astype(np.int32).tolist(),
                                     cdf_list, lengths.tolist(), offsets)
    dec = RansDecoder()
    back = dec.decode_with_indexes(stream, idx.astype(np.int32).tolist(),
                                   cdf_list, lengths.tolist(), offsets)
    assert list(back) == sym.tolist(), "rANS round-trip is not exact"

    # And the byte count must match the table's own predicted cost. This is the
    # tightest available check on the construction: it compares bits computed from
    # bin widths against bits the coder actually emitted.
    predicted = sum(cdf_cost_bits(cdfs[idx[i]], int(sym[i]), precision)
                    for i in range(n))
    actual = len(stream) * 8
    gap = (actual - predicted) / predicted
    assert abs(gap) < 0.01, f"predicted {predicted:.0f} bits, coder wrote {actual}"


def test_rans_handles_the_escape_path():
    """Symbols outside the table must survive via escape + bypass.

    Worth its own test because the escape path is *cheap* -- often cheaper than
    the model's estimate for a symbol it considered impossible -- so a table that
    is too narrow shows up as `actual < estimated`, a negative gap that reads like
    good news. See `train.loop.out_of_range_fraction`.
    """
    rans = _rans()
    if rans is None:
        print("  skip: compressai not installed")
        return
    RansEncoder, RansDecoder = rans

    precision, L = 16, 8
    pmf = _gauss_pmf(L, 1.5)
    pmf /= pmf.sum()
    cdfs, lengths = build_cdf_table(pmf[None, :], np.array([1e-3]),
                                    np.array([L]), precision)
    cdf_list = [cdfs[0, :lengths[0]].tolist()]

    # Deliberately include values past the end of the row.
    sym = [0, 3, 7, 8, 40, 3, 200, 1]
    idx = [0] * len(sym)
    enc, dec = RansEncoder(), RansDecoder()
    stream = enc.encode_with_indexes(sym, idx, cdf_list, lengths.tolist(), [0])
    back = dec.decode_with_indexes(stream, idx, cdf_list, lengths.tolist(), [0])
    assert list(back) == sym, f"escape path lost data: {list(back)} != {sym}"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:                        # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
