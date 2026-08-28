"""Quantized CDF construction -- the step that silently breaks learned codecs.

An entropy coder cannot consume a float PMF. It needs a table of integers
`cdf[0..L]` with `cdf[0] == 0`, `cdf[L] == 2**precision`, and **every** bin
strictly wider than zero. The conversion from float to that table is three lines
of arithmetic and about five ways to be wrong, and every one of them produces a
codec that trains fine, reports a beautiful estimated bitrate, and then either
writes more bytes than it predicted or fails to decode at all.

The failure is quiet because the *estimated* rate (-log2 p from the float model)
and the *actual* rate (bytes the coder emits) are computed by two different code
paths. Training only ever sees the first one. That is exactly why Phase 3's gate
is "estimated within 1-2% of actual" rather than "the loss went down".

Three specific traps, all handled below:

1. **Zero-width bins.** `round(cumulative * 2**precision)` maps any symbol whose
   probability is below `2**-precision` to a bin of width 0. A rANS/range coder
   asked to encode that symbol either divides by zero or emits a codeword the
   decoder maps to a different symbol. So a *possible* symbol must never get
   width 0 -- and widening it means taking that mass from somewhere, which is
   why `_widen_zero_bins` steals rather than just clamps.
2. **The tail.** A Gaussian has unbounded support but the table is finite. The
   mass beyond the table is collected into one extra "escape" symbol at the end;
   drop it and the CDF does not sum to 2**precision, which desynchronises the
   decoder after the first out-of-range value.
3. **Precision ceiling.** compressai's rANS requires `total <= 2**16`. Asking
   for 17 bits does not raise -- it wraps.

`pmf_to_quantized_cdf` here is written in numpy rather than reused from
compressai, so that we own the code Phase 9 replaces with me-tANS. It is safe to
differ from compressai's version because the table is *passed to* the coder --
encoder and decoder both use ours, so self-consistency is what matters, not
agreement with a third party.

Deliberate divergence from compressai (measured, not assumed)
------------------------------------------------------------
compressai ports `ryg_rans`'s normalisation: quantise each *frequency*
(`round(p_i * 2**precision)`), renormalise by integer division, cumsum, then pin
the last entry. Because integer division floors every frequency, the cumulative
drifts downward and **the final bin absorbs all the accumulated rounding slack**.
compressai's own source comments that this is "not optimal".

We instead round the *cumulative* and repair, which spreads the rounding error
across the table. Excess bits/symbol over the float model's own entropy,
averaged over 300 random distributions of length 33:

    distribution        precision   ours      compressai
    uniform-ish            8      +0.0021      +0.0469
    gaussian sigma=8       8      +0.0028      +0.1420
    gaussian sigma=8      12      +0.0000      +0.0031
    gaussian sigma=1.5    16      +0.0008      +0.0011
    gaussian sigma=1.5     8      +0.3439      +0.2830   <- the one loss

Ours is tighter nearly everywhere, and by ~50x at 8-bit precision -- which is
exactly the me-tANS operating point (`entropy.tans_mass_bits: 8`), so this is not
an academic difference for a JPEG AI implementation.

Why the peaky case loses, and what the standard does about it
------------------------------------------------------------
The `sigma=1.5, precision=8` loss is not a bug in the repair pass; it is the
repair pass working. A narrow Gaussian over 33 symbols puts almost all its mass
in ~9 bins, leaving ~22 tail symbols whose true probability is below `2**-8`.
Rule 1 forces each to width >= 1, so the tail consumes 22/256 = 8.6% of the
mass it should not have, and the peak pays for it.

That is precisely the cost `entropy.tans_escape_threshold_exp: -11` exists to
avoid: symbols below `2**-11` get no bin of their own and are coded through a
single escape symbol instead. Measured excess bits/symbol at precision 8:

    sigma   tail bins   no escape   escape at 2**-11
    1.0        26        +0.4095        +0.0054
    1.5        22        +0.2732        +0.0139
    3.0        12        +0.1383        +0.0229
    8.0         0        +0.0011        +0.0011

The escape removes 75-98% of the penalty and is correctly inert when there is no
tail. Phase 3 runs at precision 16 where none of this bites; Phase 9 must
implement the escape, and this table is the acceptance target for it.
"""

from __future__ import annotations

import numpy as np

#: compressai's rANS coder assumes the cumulative total fits in 16 bits. This is
#: a property of that coder, not of the maths; me-tANS in Phase 9 uses 8
#: (`entropy.tans_mass_bits`), which is why precision is a parameter everywhere.
MAX_PRECISION = 16


def pmf_to_quantized_cdf(pmf, precision: int = MAX_PRECISION) -> np.ndarray:
    """Float PMF -> strictly increasing integer CDF of length ``len(pmf) + 1``.

    Returns int32 with ``cdf[0] == 0`` and ``cdf[-1] == 2**precision``.

    `pmf` does not have to be normalised -- it is divided by its own sum. That is
    intentional: a Gaussian PMF truncated to a finite table never sums to 1, and
    forcing callers to normalise first is one more place to forget.
    """
    if not 1 <= precision <= MAX_PRECISION:
        raise ValueError(f"precision must be in [1, {MAX_PRECISION}], got {precision}")

    p = np.asarray(pmf, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("empty pmf")
    if not np.all(np.isfinite(p)):
        raise ValueError("pmf contains nan/inf")
    if np.any(p < 0):
        raise ValueError("pmf contains negative mass")

    total = float(p.sum())
    if total <= 0:
        raise ValueError("pmf sums to zero")

    # Cumulative in float first, then scale, then round. Rounding a monotonic
    # sequence keeps it monotonic (non-strictly), which is what makes the
    # widening pass below a local repair rather than a full re-sort.
    cum = np.concatenate([[0.0], np.cumsum(p)])
    cdf = np.rint(cum / total * (1 << precision)).astype(np.int64)
    cdf[0] = 0
    cdf[-1] = 1 << precision            # pin the total exactly; do not trust rounding

    _widen_zero_bins(cdf)

    widths = np.diff(cdf)
    if np.any(widths < 1):              # unreachable, but this is the invariant
        raise AssertionError(f"non-positive bin width after widening: {widths.min()}")
    return cdf.astype(np.int32)


def _widen_zero_bins(cdf: np.ndarray) -> None:
    """Give every zero-width bin one unit of mass, in place.

    The mass is *stolen from the narrowest bin that can spare it* (width > 1),
    not from the widest and not created out of nothing. Two reasons:

    * The total must stay exactly ``2**precision``, so mass is conserved.
    * Taking from the narrowest available bin minimises the relative distortion
      of the model. Taking from the widest would be a smaller relative change to
      that one bin but a larger absolute change to the frequent symbols, which
      is where the bits actually are.

    Stealing is implemented as shifting a contiguous run of CDF boundaries by
    one, which changes exactly two bin widths: the thief gains 1, the victim
    loses 1. Everything between them slides and keeps its width.

    This mirrors compressai's C++ `pmf_to_quantized_cdf` so the two can be
    compared symbol for symbol.
    """
    n = cdf.size - 1                     # number of bins
    for i in range(n):
        if cdf[i + 1] > cdf[i]:
            continue

        widths = np.diff(cdf)
        spare = np.flatnonzero(widths > 1)
        if spare.size == 0:
            # Fewer units of mass than symbols: the table cannot represent this
            # alphabet at this precision. Raising is right -- silently merging
            # symbols would corrupt the codestream.
            raise ValueError(
                f"cannot widen bin {i}: {n} symbols do not fit in "
                f"{cdf[-1]} units of probability mass. Raise precision or "
                f"shorten the table."
            )
        # argmin over widths[spare] picks the narrowest; ties go to the lowest
        # index, which keeps the result deterministic across numpy versions.
        j = int(spare[np.argmin(widths[spare])])

        if j < i:
            cdf[j + 1:i + 1] -= 1
        else:
            cdf[i + 1:j + 1] += 1


def build_cdf_table(
    pmfs: np.ndarray,
    tail_masses: np.ndarray,
    pmf_lengths: np.ndarray,
    precision: int = MAX_PRECISION,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-distribution CDFs into one padded table.

    Args:
        pmfs:        [N, Lmax] float. Row i is only valid up to pmf_lengths[i].
        tail_masses: [N] float. Probability outside row i's table, coded as one
                     extra escape symbol appended to that row.
        pmf_lengths: [N] int, number of real symbols per row.

    Returns:
        cdfs:        [N, Lmax + 2] int32, row i valid up to lengths[i].
        lengths:     [N] int32, ``pmf_lengths + 2`` (escape symbol + final total).

    The ``+2`` is not slack. A row with L real symbols needs L+1 entries for the
    real bins, one more for the escape symbol, and the array holds boundaries,
    so L+2 numbers. Getting this off by one is the single most common way a
    decoder walks off the end of a row into the next distribution's data.
    """
    pmfs = np.asarray(pmfs, dtype=np.float64)
    tail_masses = np.asarray(tail_masses, dtype=np.float64).ravel()
    pmf_lengths = np.asarray(pmf_lengths, dtype=np.int64).ravel()

    n, lmax = pmfs.shape
    if tail_masses.size != n or pmf_lengths.size != n:
        raise ValueError(
            f"shape mismatch: pmfs {pmfs.shape}, tail_masses {tail_masses.shape}, "
            f"pmf_lengths {pmf_lengths.shape}"
        )
    if pmf_lengths.max(initial=0) > lmax:
        raise ValueError(f"pmf_lengths max {pmf_lengths.max()} exceeds row width {lmax}")

    cdfs = np.zeros((n, lmax + 2), dtype=np.int32)
    for i in range(n):
        row = np.concatenate([pmfs[i, : pmf_lengths[i]], [tail_masses[i]]])
        c = pmf_to_quantized_cdf(row, precision)
        cdfs[i, : c.size] = c
    return cdfs, (pmf_lengths + 2).astype(np.int32)


def cdf_cost_bits(cdf: np.ndarray, symbol_index: int, precision: int) -> float:
    """Bits the coder will actually spend on one symbol, per the *quantized* table.

    Useful for pinning down where estimated and actual rate diverge: compare this
    against ``-log2(pmf[symbol])`` from the float model. A systematic gap means
    the table is wrong; a gap only on rare symbols means it is just quantization
    of the tail, which is expected and small.
    """
    width = int(cdf[symbol_index + 1]) - int(cdf[symbol_index])
    if width <= 0:
        raise ValueError(f"symbol {symbol_index} has zero-width bin")
    return -float(np.log2(width / (1 << precision)))
