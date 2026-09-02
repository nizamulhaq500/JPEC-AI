"""The entropy models' one non-negotiable invariant, and the bug it was hiding.

**The invariant.** The rate the training loss sees and the rate the coder charges must
be the same function of the same data. `forward()` returns a likelihood; `update()`
builds a quantised CDF table; `compress()` writes bytes against that table. If any two
of those three disagree, the model optimises one thing and the codestream pays for
another, and *nothing crashes* -- the pictures come back bit-exact and only the bitrate
is wrong. No round-trip test can see it.

**The bug this file exists to prevent.** `update()` sampled the density at
`median + v` while `forward()` evaluated it at `v`, so the table was a shifted copy of
the distribution the rate loss was trained against. The shift is proportional to
`|median|`, and `median` is a *learned* quantile that starts at 0 and only drifts once
the aux loss starts moving -- so on a freshly-constructed model the two samplings agree
exactly and every existing test passed. It cost a measured **+63%** on a two-branch
chroma hyper-latent stream (medians around +-1.4) and +3.9% on the luma one (+-0.24).

Hence the rule these tests enforce: **exercise the table with a non-zero median.** A
test on a fresh prior is not a test of this code path at all.
"""

from __future__ import annotations

import math

import pytest
import torch

from jpegai.models.entropy import (
    FactorizedPrior,
    GaussianConditional,
    build_scale_table,
)


def _prior(channels: int = 6, *, medians=None, half_width: float = 6.0):
    """A prior whose learned quantiles are set by hand.

    Setting them directly is the point: it puts the model in the state a *partly
    trained* one is in -- density fitted to one thing, medians pointing somewhere
    else -- which is where the table build and the forward pass can diverge. Waiting
    for the aux loss to produce that state would make the test slow and flaky.
    """
    torch.manual_seed(0)
    eb = FactorizedPrior(channels).eval()
    if medians is not None:
        med = torch.tensor(medians, dtype=torch.float32)
        assert med.numel() == channels
        q = torch.stack([med - half_width, med, med + half_width], dim=-1)
        eb.quantiles.data = q.reshape(channels, 1, 3)
    return eb


def _table_bits(eb, z) -> tuple[float, int]:
    """(bits, escapes) implied by the coder's own quantised CDF for `z`'s symbols.

    Deliberately reimplemented from the buffers rather than calling any helper: this
    has to be an independent reading of the table, or it would agree with `update()`
    by construction and test nothing.
    """
    med = eb.medians().detach().reshape(1, -1, 1, 1)
    sym = torch.round(z - med).to(torch.int64)
    total, escapes = 0.0, 0
    for k in range(sym.shape[1]):
        length = int(eb._cdf_len[k]) - 2                 # real symbols in row k
        bins = sym[0, k].reshape(-1) - int(eb._offset[k])
        inside = (bins >= 0) & (bins < length)
        escapes += int((~inside).sum())
        row = eb._cdf[k].double() / (1 << eb.precision)
        b = bins.clamp(0, max(length - 1, 0))
        p = (row[b + 1] - row[b]).clamp_min(1e-30)
        total += float((-torch.log2(p))[inside].sum())
    return total, escapes


def _forward_bits(eb, z) -> float:
    with torch.no_grad():
        _, lik = eb(z, noise=False, ste=False)
    return float(-torch.log2(lik.clamp_min(1e-12)).sum())


# -- the invariant, at the median offsets that break it -------------------------
@pytest.mark.parametrize("medians", [
    [0.0] * 6,                                    # the case every other test covers
    [1.455, -1.170, 0.014, 1.257, -0.6, 0.9],     # measured off a real chroma branch
    [2.5, -2.5, 0.5, -0.5, 3.4, -3.4],            # and further out, both signs
])
def test_the_coder_table_charges_what_the_rate_loss_predicted(medians):
    """The whole point. Same data, same model, two independent readings of the rate.

    Sensitivity, measured by reintroducing the bug: `medians0` passes (it is the
    case with no shift to detect), `medians1` **also passes**, and `medians2` fails.
    That is worth stating rather than hiding -- a freshly constructed prior has
    `init_scale = 10`, so its density is very broad, and shifting a broad density by
    1.4 barely changes any bin. The real chroma branch had a *narrow* fitted density,
    which is exactly why the same 1.4 shift cost it 63% there and almost nothing
    here.

    So this test is the readable statement of the invariant, not the sensitive
    detector. `test_the_table_does_not_depend_on_the_median_at_all` is the sensitive
    one: it catches every shift exactly, including `medians1`.
    """
    eb = _prior(medians=medians)
    eb.update(force=True)
    torch.manual_seed(1)
    z = torch.randn(1, 6, 8, 8) * 1.5 + torch.tensor(medians).reshape(1, -1, 1, 1)

    table, escapes = _table_bits(eb, z)
    forward = _forward_bits(eb, z)
    assert escapes == 0
    # 0.5% covers the CDF's own 16-bit quantisation; the bug was 63%.
    assert abs(table - forward) / forward < 5e-3, (
        f"table {table / 8:.1f} B vs forward {forward / 8:.1f} B "
        f"= {100 * (table - forward) / forward:+.2f}%")


def test_the_table_does_not_depend_on_the_median_at_all():
    """The sharpest statement of the fix, and it needs no bit counting.

    The table is indexed by *symbol*, and symbols are median-relative by
    construction (`compress` writes `round(z - median)`). So two priors with the same
    density and the same tail widths must produce **byte-identical** tables no matter
    where their medians sit. Under the shift bug they did not: the table was a copy of
    the density sampled at `median + v`, so every distinct median gave a distinct
    table. This is the assertion that fails loudly if the shift ever comes back.
    """
    fresh = _prior(medians=[0.0] * 6)
    fresh.update(force=True)
    for offsets in ([3.0] * 6, [1.455, -1.170, 0.014, 1.257, -0.6, 0.9],
                    [-4.0, 4.0, -2.5, 2.5, 0.25, -0.25]):
        shifted = _prior(medians=offsets)
        shifted.update(force=True)
        assert torch.equal(shifted._cdf, fresh._cdf), offsets
        assert torch.equal(shifted._cdf_len, fresh._cdf_len), offsets
        assert torch.equal(shifted._offset, fresh._offset), offsets


def test_a_nonzero_median_is_what_makes_these_tests_bite():
    """Guards the fixture from being quietly defanged.

    If `_prior` stopped applying `medians`, the invariant tests above would still
    pass and would stop checking anything. So assert the offset state is real and
    that it reaches the two places that matter: `medians()` and the symbols the
    coder derives from it.
    """
    eb = _prior(medians=[3.0] * 6)
    assert float(eb.medians().detach().abs().min()) > 2.9
    z = torch.randn(1, 6, 4, 4) + 3.0
    med = eb.medians().detach().reshape(1, -1, 1, 1)
    # The median must actually move the quantisation grid, or `compress` and
    # `forward` are both operating on un-centred values and the fixture is inert.
    assert not torch.equal(torch.round(z - med), torch.round(z))
    with torch.no_grad():
        z_hat, _ = eb(z, noise=False, ste=False)
    assert torch.equal(z_hat, torch.round(z - med) + med)


def test_the_table_peak_sits_where_the_density_peak_sits():
    """A sharper localisation of the same bug, independent of any bit count.

    The most likely *symbol* under the table must be the most likely symbol under
    the density. Under the shift bug the table's peak moved to `-median` while the
    density's stayed at 0, which is visible with no arithmetic at all.
    """
    eb = _prior(medians=[2.0, -2.0, 0.0, 1.0, -1.0, 3.0])
    eb.update(force=True)
    for k in range(6):
        length = int(eb._cdf_len[k]) - 2
        row = eb._cdf[k].double()
        pmf = row[1:length + 1] - row[0:length]
        table_peak = int(pmf.argmax()) + int(eb._offset[k])

        syms = torch.arange(-12, 13, dtype=torch.float32)
        probe = torch.zeros(1, 6, 1, syms.numel())
        probe[0, k, 0] = syms + float(eb.medians().detach()[k, 0, 0])   # symbol -> value
        with torch.no_grad():
            _, lik = eb(probe, noise=False, ste=False)
        density_peak = int(syms[int(lik[0, k, 0].argmax())])
        assert abs(table_peak - density_peak) <= 1, (k, table_peak, density_peak)


def test_bin_zero_of_every_row_is_the_symbol_the_offset_names():
    """`_offset[k]` is documented as "the symbol value that row k's bin 0
    represents". The decoder trusts it absolutely -- an off-by-one there shifts every
    decoded symbol in the channel -- so it is checked against `minima`, the quantity
    it is derived from, rather than assumed."""
    medians = [1.455, -1.170, 0.014, 1.257, -0.6, 0.9]
    eb = _prior(medians=medians, half_width=6.0)
    eb.update(force=True)
    minima, maxima = eb._density_extent()
    assert torch.equal(eb._offset.cpu().int(), (-minima).cpu())
    # And the row holds exactly the range the extent claims.
    assert torch.equal(eb._cdf_len.cpu().int() - 2, (minima + maxima + 1).cpu())


def test_the_table_reaches_every_symbol_the_density_puts_mass_on():
    """`update` reads its extent off the density, so escapes are what the test
    measures -- not the extent arithmetic that produces them.

    A channel whose `median` sits away from its density's mode is the case that
    broke: `forward` centres on `median`, so the symbols land near
    ``mode - median`` while a row centred on zero cannot reach them. Below,
    `median` is pushed up to 2 bins off a narrow density, which is the phase-6
    `z_uv` signature (|median| ~ 1.8 against a 3-symbol row).
    """
    eb = _prior(channels=6, medians=[0.0] * 6, half_width=0.45)
    with torch.no_grad():                     # move the medians off the mode
        shift = torch.tensor([0.0, 1.9, -1.9, 2.4, -2.4, 0.6])
        eb.quantiles[:, 0, :] += shift[:, None]
    eb.update(force=True)

    med = eb.medians().detach().reshape(1, -1, 1, 1)
    torch.manual_seed(5)
    # Data sits at the density's mode (0), not at `median`.
    z = torch.randn(1, 6, 12, 12) * 0.2
    table, escapes = _table_bits(eb, z)
    assert escapes == 0, f"{escapes} symbols outside their row"

    # And the bytes written stay close to the table's own estimate, which is the
    # user-visible consequence: every escape costs an escape symbol plus a
    # bypass-coded raw value, roughly 8 bits where the symbol was worth a fraction.
    strings = eb.compress(z)
    actual = sum(len(s) for s in strings) * 8
    assert (actual - table) / table < 0.05, (
        f"actual {actual / 8:.0f} B vs table {table / 8:.1f} B")
    assert torch.equal(eb.decompress(strings, tuple(z.shape[-2:]), device=z.device),
                       torch.round(z - med) + med)


def test_a_real_round_trip_costs_what_the_table_says():
    """Closes the loop: table bits vs *bytes actually written*, not another estimate.

    Only the rANS flush and the 16-bit CDF quantisation may sit between them.
    """
    eb = _prior(channels=8, medians=[1.4, -1.4, 0.0, 2.2, -2.2, 0.7, -0.7, 1.9])
    eb.update(force=True)
    torch.manual_seed(2)
    z = torch.randn(1, 8, 16, 16) * 1.2 + eb.medians().detach().reshape(1, -1, 1, 1)

    strings = eb.compress(z)
    z_hat = eb.decompress(strings, tuple(z.shape[-2:]), device=z.device)
    med = eb.medians().detach().reshape(1, -1, 1, 1)
    assert torch.equal(z_hat, torch.round(z - med) + med)

    table, _ = _table_bits(eb, z)
    actual = sum(len(s) for s in strings) * 8
    assert 0 <= (actual - table) / table < 0.05, (
        f"actual {actual / 8:.0f} B vs table {table / 8:.1f} B")


# -- the same invariant on the Gaussian side ------------------------------------
def test_the_gaussian_table_charges_what_its_likelihood_predicted():
    """The Gaussian conditional has no median parameter, so it cannot suffer the bug
    above -- but the invariant is the same one, and it should be pinned on both
    models rather than only on the one that broke.

    `y` is drawn *from* the scales rather than independently of them. Drawing
    `y ~ N(0, 4)` against scales in [0.2, 6.2] puts symbols 20 sigma into their own
    tail, where the estimate charges `-log2(p)` of an astronomically small number and
    the coder escapes them at a flat cost instead -- the actual bytes then come out
    *below* the estimate and the comparison measures escape handling, not the table.
    """
    torch.manual_seed(0)
    gc = GaussianConditional(build_scale_table(0.11, 54.82, 32), scale_bound=0.11)
    gc.update(force=True)
    scales = torch.rand(1, 4, 16, 16) * 6 + 0.2
    y = torch.randn(1, 4, 16, 16) * scales
    idx = gc.build_indexes(scales)
    scales_q = gc.scale_table[idx.long()]

    lik = gc._likelihood(torch.round(y), scales_q)
    est = float(-torch.log2(lik.clamp_min(1e-12)).sum())
    actual = sum(len(s) for s in gc.compress(y, scales)) * 8
    assert abs(actual - est) / est < 0.02, (
        f"actual {actual / 8:.0f} B vs est {est / 8:.1f} B "
        f"= {100 * (actual - est) / est:+.2f}%")


def test_the_escape_symbol_carries_the_mass_outside_the_table():
    """`build_cdf_table` gets `tail` = P(below) + P(above). If that were dropped the
    table's real symbols would sum to 1 and an out-of-range symbol would be
    unencodable; if it were double-counted every symbol would be slightly overpriced.
    """
    eb = _prior(medians=[0.0] * 6, half_width=3.0)
    eb.update(force=True)
    for k in range(6):
        length = int(eb._cdf_len[k]) - 2
        row = eb._cdf[k].double() / (1 << eb.precision)
        real = float(row[length] - row[0])
        escape = float(row[length + 1] - row[length])
        assert real + escape == pytest.approx(1.0, abs=2e-4)
        assert escape > 0.0, "an escape bin of zero width cannot be coded"


def test_the_density_is_a_valid_cdf_at_a_shifted_median():
    """Monotonicity is structural (softplus weights), so it cannot break -- but the
    *table* is where a valid CDF turns into a coder, and a non-monotone row is
    undecodable. Checked at the offset medians, since that is the untested path."""
    eb = _prior(medians=[2.0, -2.0, 0.5, -0.5, 4.0, -4.0])
    eb.update(force=True)
    for k in range(6):
        length = int(eb._cdf_len[k])
        row = eb._cdf[k][:length]
        assert torch.all(row[1:] > row[:-1]), f"row {k} is not strictly increasing"
        assert int(row[0]) == 0
        assert int(row[-1]) == 1 << eb.precision


def test_the_median_earns_its_complexity():
    """Why the median exists at all, stated as the comparison that justifies it.

    A channel whose mass sits at +2.5 is coded by a median-aware model as symbol 0 --
    the density's own peak -- and by a median-blind one as symbol 2 or 3, out where the
    density is thin. So the median-aware table must cost strictly fewer bits on the
    same data. Establishing that here matters because the fix above made the table
    median-*independent*, and the obvious next "simplification" is to drop the median
    entirely, which would silently cost real bits.

    Note what is *not* asserted: an absolute bits-per-symbol figure. A freshly
    constructed prior has `init_scale = 10`, so its density is deliberately very
    broad and its mode carries only a few percent of the mass -- around 4.6 bits per
    symbol even when everything is correct. Any threshold on that number would be a
    statement about the initialisation, not about the median.
    """
    offset = 2.5
    z = torch.full((1, 2, 8, 8), offset)

    aware = _prior(channels=2, medians=[offset, offset])
    aware.update(force=True)
    centred, _ = aware(z, noise=False, ste=False)
    assert torch.allclose(centred, z, atol=1e-6)   # on a grid point: zero round error
    aware_bits, aware_esc = _table_bits(aware, z)

    blind = _prior(channels=2, medians=[0.0, 0.0])
    blind.update(force=True)
    blind_bits, blind_esc = _table_bits(blind, z)

    assert aware_esc == blind_esc == 0
    assert aware_bits < blind_bits, (aware_bits, blind_bits)
