"""The mid-training rate gate, and the failure it was blind to.

`roundtrip_check` is the only thing standing between a broken entropy coder and a
finished training run. It has one hard job: notice when the bytes on disk cost more
than the rate loss was told they would.

For most of this project it reported that as **one aggregate percentage**, and that is
what let the Phase 5 median-shift bug survive three ladders. The aggregate divides one
stream's error by *every* stream's bits, so a stream that is 63% wrong shows up as
+1.85% overall -- above the +-0.5% gate, but only just, and with nothing to say about
where it came from. On a model whose bad stream carried a smaller share of the bits it
would have passed outright.

So the gate now reports per stream, and these tests pin both halves of that: the
numbers are correct on a healthy model, and the detector actually fires on a sick one.
The second half is the point. A diagnostic that has never been shown to trip is not a
diagnostic.
"""

from __future__ import annotations

import torch

from jpegai.models.hyperprior import build_model
from jpegai.models.twobranch import build_two_branch
from jpegai.models.entropy import FactorizedPrior
from jpegai.config import load_config
from jpegai.train.loop import roundtrip_check


def _valid(n: int = 1, h: int = 64, w: int = 64):
    """A stand-in for the validation dataset: `roundtrip_check` only ever indexes it.

    Structured content, not noise. `torch.rand` is incompressible, so every stream
    sits at its maximum rate and a rate *regression* has almost no room to show --
    which would make the injection test below pass for the wrong reason.
    """
    torch.manual_seed(0)
    ys = torch.linspace(0, 1, h).reshape(1, h, 1)
    xs = torch.linspace(0, 1, w).reshape(1, 1, w)
    base = (ys * xs).expand(3, h, w)
    return [(base + 0.05 * torch.randn(3, h, w)).clamp(0, 1) for _ in range(n)]


def _two_branch(*, uv_init_scale: float | None = None):
    """A two-branch codec, optionally with a **peaked** chroma hyper-latent prior.

    `uv_init_scale` matters more than it looks. `FactorizedPrior`'s default
    `init_scale=10` is a deliberately very broad density: measured, its table is 21
    bins wide and its most likely symbol carries p=0.025. A density that flat is
    *insensitive to being shifted* -- moving the table three bins under it changes the
    cost of almost nothing, which is why the first version of the injection test below
    reproduced the historical bug's mechanism exactly and measured no effect at all.

    At `init_scale=0.5` the table is 3 bins with peak p=0.44, which is the regime a
    trained prior is in and the regime where a misaligned table costs real bytes. This
    is the same sensitivity trap documented in `tests/test_entropy.py`: a fresh prior
    is not a fixture for anything to do with table alignment.
    """
    torch.manual_seed(0)
    m = build_two_branch(load_config("tierA"), mean_scale=True)
    if uv_init_scale is not None:
        old = m.branch_uv.entropy_bottleneck
        m.branch_uv.entropy_bottleneck = FactorizedPrior(
            old.channels, init_scale=uv_init_scale)
    return m


def _shift_uv_table(m, bins: int):
    """Make the chroma hyper-latent's table describe symbols `bins` away from the real
    ones, *after* `update()` has built it correctly.

    This is the historical bug's signature reproduced at the cheapest possible point.
    The real bug sampled the density on a grid offset by the learned median, which
    makes the table a shifted copy of the density the rate loss was trained against.
    Shifting `_offset` produces the same misalignment, and crucially the same
    *symptoms*: the table stays a valid strictly-increasing CDF, `_offset` is applied
    symmetrically by encoder and decoder so the latent still round-trips **bit-exact**,
    and only the bitrate is wrong.
    """
    eb = m.branch_uv.entropy_bottleneck
    real = eb.update

    def shifted(*a, **kw):
        rv = real(*a, **kw)
        eb._offset += bins
        return rv

    eb.update = shifted
    return m


# -- the numbers are right on a healthy model -----------------------------------
def test_the_per_stream_gaps_are_reported_for_every_stream():
    rt = roundtrip_check(_two_branch(), _valid(), torch.device("cpu"))
    for name in ("y", "z", "y_uv", "z_uv"):
        assert f"gap_{name}_pct" in rt, name
        assert f"excess_{name}_b" in rt, name
    assert rt["worst_stream"] in ("y", "z", "y_uv", "z_uv")


def test_the_single_branch_gate_reports_its_two_streams_and_no_others():
    """The gate must not assume four streams. A `mean-scale` model has two, and a
    `KeyError` here would take out the gate on every Phase 3 run."""
    m = build_model(load_config("tierA"), kind="mean-scale")
    rt = roundtrip_check(m, _valid(), torch.device("cpu"))
    assert {k for k in rt if k.startswith("gap_")} == {
        "gap_pct", "gap_q_pct", "gap_y_pct", "gap_z_pct"}


def test_the_worst_stream_is_ranked_by_bytes_not_by_percentage():
    """`worst_stream` is what the ladder summary prints and what a reader acts on.

    Ranked by excess bytes deliberately. At random init a stream can carry almost no
    rate -- the untrained `y` predicts its own near-zero symbols nearly perfectly, so
    its estimate is a fraction of a bit against 8 real bytes of rANS flush, and its
    *percentage* reads in the hundreds of thousands. Measured on this very fixture:
    `gap_y_pct` is around +5e5 and `gap_y_uv_pct` around +1e6, while both streams are
    8 bytes and entirely healthy. A percentage ranking would name one of them on every
    early rtcheck of every run.
    """
    rt = roundtrip_check(_two_branch(), _valid(), torch.device("cpu"))
    excess = {k[7:-2]: v for k, v in rt.items() if k.startswith("excess_")}
    assert rt["worst_stream"] == max(excess, key=lambda k: excess[k])
    assert rt["worst_stream_b"] == excess[rt["worst_stream"]]
    assert rt["worst_stream_pct"] == rt[f"gap_{rt['worst_stream']}_pct"]
    # The premise, asserted so this test explains itself if it ever fails: the
    # percentage really is degenerate here, and really does disagree with the ranking.
    pcts = {k[4:-4]: v for k, v in rt.items()
            if k.startswith("gap_") and k not in ("gap_pct", "gap_q_pct")}
    assert max(pcts.values()) > 1e4, pcts


def test_a_healthy_model_passes_the_per_stream_arm():
    """The floor, measured over 72 stream-readings on trained checkpoints (both 3k
    ladders x 3 beta an octave apart x 4 validation images): median +5.3 B, full spread
    -18.3 to +14.5 B, near-independent of stream size -- rANS flush plus 16-bit CDF
    quantisation, plus a two-sided sigma-grid term on the `y` streams. The 16 B / 2%
    conjunction has to sit above that, or the gate cries wolf on every correct run and
    gets ignored.

    The margin is on the conjunction, not on either arm. +14.5 B eats 91% of the byte
    arm, but it came from a 26 KB stream at +0.055% -- 3% of the percentage arm. The
    closest any healthy reading comes to firing is a 972 B `z_uv` at +6.9 B / +0.71%,
    which is 0.36x of its binding arm."""
    rt = roundtrip_check(_two_branch(), _valid(), torch.device("cpu"))
    assert rt["streams_ok"], {k: v for k, v in rt.items() if k.startswith(("gap_", "excess_"))}


# -- and the detector fires on a sick one ---------------------------------------
def test_a_stream_that_beats_its_estimate_is_not_a_failure_and_is_still_named():
    """The excess is **two-sided**, and both halves of that have bitten.

    Measured over 72 readings on trained checkpoints, the per-stream excess spans -18.3
    to +14.5 B. The negative end is not a bug: a `y` stream is scored against `est_q`,
    and quantising sigma onto the 64-entry log grid can round sigma *up*, so the estimate
    is pessimistic and the real bytes come in *under* it. Coming in cheap is never the
    fault this gate hunts.

    Two things therefore have to hold, and this test forces both by reporting byte counts
    below every estimate. The gate must stay green -- hence `excess > 16` rather than
    `|excess| > 16`. And `worst_stream` must still name a stream: the ranking floor is
    `-inf` for exactly this case, because a finite floor (it was `-1.0`) leaves the name
    empty and the ladder summary and rtcheck line then print nothing at all, with no
    error to say why.
    """
    m = _two_branch()
    real = m.stream_bytes
    # Zero, not one. At random init the `y` streams predict their own near-zero symbols
    # so well that their estimate is a *fraction of a bit*, so even a single byte is an
    # excess of +0.999 and two of the four streams would stay positive.
    m.stream_bytes = lambda packet: {k: 0 for k in real(packet)}

    rt = roundtrip_check(m, _valid(), torch.device("cpu"))
    excess = {k[7:-2]: v for k, v in rt.items() if k.startswith("excess_")}
    assert excess and max(excess.values()) < 0.0, excess
    assert rt["streams_ok"]
    assert rt["worst_stream"] == max(excess, key=lambda k: excess[k])
    assert rt["worst_stream_b"] == excess[rt["worst_stream"]]


def test_a_shifted_table_is_caught_and_named():
    """The injection test, and the reason this file exists.

    Injected on the chroma hyper-latent, which is where the real bug did the most
    damage (+63%, 848 excess bytes): it is the smallest stream, so it is the one an
    aggregate hides best.

    Two claims are asserted alongside the detection, because together they are what
    makes this bug class so hard to see. The latent still round-trips **bit-exact** --
    `_offset` is applied symmetrically, so nothing is *incorrect*, only expensive. And
    the other three streams are untouched, so a gate that reported only a total, or
    only correctness, would call this a clean run.
    """
    m = _shift_uv_table(_two_branch(uv_init_scale=0.5), 2)
    rt = roundtrip_check(m, _valid(), torch.device("cpu"))

    assert not rt["streams_ok"]
    assert rt["worst_stream"] == "z_uv", (rt["worst_stream"], rt["worst_stream_b"])
    assert rt["worst_stream_b"] > 16.0, rt["worst_stream_b"]
    assert rt["y_exact"] and rt["z_exact"]
    for clean in ("y", "z", "y_uv"):
        assert rt[f"excess_{clean}_b"] < 16.0, (clean, rt[f"excess_{clean}_b"])


def test_a_broad_density_cannot_detect_a_shifted_table_and_that_is_not_a_gate_bug():
    """The honest limit of the injection above, recorded rather than hidden.

    The same shift applied to a model whose chroma prior is at the **default**
    `init_scale=10` moves almost no bytes: that density is flat enough over its own
    21-bin table (peak p = 0.025) that sliding the table under it barely changes any
    symbol's cost. So the gate does not fire -- and it *should* not, because on that
    model the misalignment genuinely is nearly free.

    This is worth a test rather than a comment because it is the trap the first version
    of the test above fell into: reproduce the historical bug's mechanism faithfully on
    a fresh model and measure nothing, then conclude the detector is broken. It is the
    fixture that is broken. The real bug cost 63% precisely because the real prior was
    trained and therefore narrow.
    """
    m = _shift_uv_table(_two_branch(), 3)          # default init_scale=10
    rt = roundtrip_check(m, _valid(), torch.device("cpu"))
    assert rt["streams_ok"]
    assert rt["excess_z_uv_b"] < 16.0, rt["excess_z_uv_b"]


def test_the_aggregate_gate_alone_understates_it():
    """The justification for adding a second arm, stated as a measurement.

    If the aggregate caught everything the per-stream arm catches, the per-stream arm
    would be noise. So: inject the fault and compare. The aggregate moves, but by a
    fraction of the per-stream number, because it divides the chroma hyper-latent's
    error by all four streams' bits. Scale the model up, or the image, and that
    fraction shrinks further while the fault stays identical -- which is exactly how
    the real bug read +1.85% overall while one stream was +63%.
    """
    m = _shift_uv_table(_two_branch(uv_init_scale=0.5), 2)
    rt = roundtrip_check(m, _valid(), torch.device("cpu"))
    assert rt["gap_z_uv_pct"] > 4 * abs(rt["gap_q_pct"]), (
        f"per-stream {rt['gap_z_uv_pct']:+.2f}% vs aggregate {rt['gap_q_pct']:+.2f}%")


def test_the_two_arms_of_the_threshold_are_both_load_bearing():
    """Neither arm alone has the right shape, so neither may be dropped.

    * Bytes alone: 16 B is nothing on a 3 kB stream, so a 5% fault there would pass.
    * Percent alone: the measured 3-7 B floor is 0.1% on a 3 kB stream but ~4% on a
      200 B one, so a small healthy stream would fail -- and at random init, where a
      stream's estimate is near zero, the percentage runs to six figures on 8 bytes of
      flush.

    Asserted against the real floor rather than as arithmetic on invented numbers: the
    healthy model's own excess must clear the byte arm, which is what makes the
    conjunction quiet, *and* its percentage must breach the percent arm, which is what
    proves the byte arm is doing the work.
    """
    rt = roundtrip_check(_two_branch(), _valid(), torch.device("cpu"))
    excess = [v for k, v in rt.items() if k.startswith("excess_")]
    assert excess and max(excess) < 16.0, excess
    pcts = [v for k, v in rt.items()
            if k.startswith("gap_") and k not in ("gap_pct", "gap_q_pct")]
    assert max(pcts) > 2.0, pcts          # percent arm alone would fail this run
    assert rt["streams_ok"]               # the conjunction does not
