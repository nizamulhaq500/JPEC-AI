"""The luma-only decode fast path -- Phase 4 acceptance criterion 4.

The criterion is "`--luma-only` decodes and produces a grey image with correct luma,
at reduced decode time you can actually measure and report". Three claims, and the
first two are the ones a test can pin exactly:

* **Correct luma.** The primary branch must decode to the *same* tensor whether or
  not chroma was decoded. Bit-exact, not close: the two paths run identical code on
  identical inputs, so any difference is a bug.
* **Grey.** R == G == B after the inverse colour transform. Checked on the RGB
  output rather than on the internal chroma planes, because a sign error in
  `_to_rgb` would leave the planes correct and the picture tinted.
* **Fewer bytes.** Measured by *deleting* the chroma strings before decoding, which
  is the only way to prove the fast path never reads them. A decoder that decodes
  chroma and discards it passes both of the above.

The timing claim is deliberately *not* asserted as a threshold. Wall-clock on a
shared CI machine is not a testable property -- what is testable is that the harness
reports the saving honestly, so the tests here check the reporting logic (that the
noise column flags an unreportable measurement, that the arithmetic share is
computed from the `_uv` parts only) with hand-made rows instead of real timings.
"""

from __future__ import annotations

import copy

import torch

from jpegai.config import load_config
from jpegai.eval.lumaonly import decoder_kmac, measure, report
from jpegai.models import build_any_model


def _model():
    torch.manual_seed(0)
    m = build_any_model(load_config("tierA"), "twobranch").eval()
    m.update(force=True)
    return m


def _row(**kw):
    """A `measure` row with plausible defaults, for testing `report` alone."""
    base = {"size": "64x64", "pixels": 4096, "bytes_full": 1000, "bytes_luma": 660,
            "bpp_full": 1.95, "ms_full": 100.0, "ms_luma": 76.0,
            "ms_full_spread": 2.0, "ms_luma_spread": 2.0,
            "luma_identical": True, "grey_max_dev": 0.0}
    return {**base, **kw}


# -- the three hard claims ------------------------------------------------------
@torch.no_grad()
def test_the_luma_plane_is_bit_identical_with_and_without_chroma():
    m = _model()
    packet = m.compress(torch.rand(1, 3, 96, 128))
    full = m.decompress(packet)
    part = m.decompress(packet, luma_only=True)
    assert torch.equal(part["luma"], full["luma"])
    assert torch.equal(part["y_hat"], full["y_hat"])


@torch.no_grad()
def test_the_luma_only_output_is_grey():
    m = _model()
    x = m.decompress(m.compress(torch.rand(1, 3, 96, 128)), luma_only=True)["x_hat"]
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    assert torch.allclose(r, g, atol=1e-6) and torch.allclose(g, b, atol=1e-6)
    # And not a *constant* grey: the luma must actually vary, or "correct luma" is
    # being satisfied by a decoder that returns 0.5 everywhere.
    assert float(x.std()) > 0.01


@torch.no_grad()
def test_the_chroma_strings_are_never_read():
    """Delete them, then decode. The only proof that survives an eager decoder."""
    m = _model()
    packet = m.compress(torch.rand(1, 3, 96, 128))
    ref = m.decompress(packet, luma_only=True)
    stripped = {k: v for k, v in packet.items() if k != "chroma"}
    out = m.decompress(stripped, luma_only=True)
    assert torch.equal(out["luma"], ref["luma"])
    assert out["y_uv_hat"] is None and out["z_uv_hat"] is None


@torch.no_grad()
def test_a_full_decode_of_a_stripped_packet_fails_loudly():
    """The complement. If it did not raise, the previous test would prove nothing."""
    m = _model()
    stripped = {k: v for k, v in m.compress(torch.rand(1, 3, 96, 128)).items()
                if k != "chroma"}
    try:
        m.decompress(stripped)
    except KeyError:
        return
    raise AssertionError("full decode silently tolerated a missing chroma stream")


@torch.no_grad()
def test_the_luma_only_payload_is_a_strict_subset_of_the_full_one():
    m = _model()
    packet = m.compress(torch.rand(1, 3, 96, 128))
    full, luma = m.packet_bytes(packet), m.packet_bytes(packet, luma_only=True)
    assert 0 < luma < full
    # Chroma is two subsampled planes against one full-resolution one, so its share
    # of the payload sits well under half -- but it is not negligible either, and a
    # share near zero would mean the secondary branch has nothing to say.
    assert 0.1 < (full - luma) / full < 0.5


# -- the harness ----------------------------------------------------------------
@torch.no_grad()
def test_measure_reports_a_saving_and_verifies_correctness_in_one_pass():
    rows = measure(_model(), torch.device("cpu"), sizes=((96, 128),), repeats=2)
    assert len(rows) == 1
    r = rows[0]
    assert r["size"] == "128x96"
    assert r["luma_identical"] and r["grey_max_dev"] < 1e-5
    assert r["bytes_luma"] < r["bytes_full"]
    assert r["ms_full"] > 0 and r["ms_luma"] > 0


def test_the_arithmetic_share_counts_only_the_secondary_branch():
    kmac = decoder_kmac(_model(), crop=128)
    assert set(kmac) == {"g_s_y", "h_s_y", "g_s_uv", "h_s_uv", "eb_y", "eb_uv"}
    # The encoder-side halves must be absent: billing g_a into a "decoder kMAC/pxl"
    # figure is exactly the mistake `summary_parts`'s is_decoder flag exists to stop.
    assert not any(k.startswith("g_a") or k.startswith("h_a") for k in kmac)
    skipped = kmac["g_s_uv"] + kmac["h_s_uv"]
    assert 0.1 < skipped / sum(kmac.values()) < 0.35


def test_the_report_flags_a_measurement_swamped_by_noise():
    quiet = report([_row(ms_full_spread=1.0, ms_luma_spread=1.0)])
    loud = report([_row(ms_full_spread=40.0, ms_luma_spread=1.0)])
    assert "WARNING" not in quiet
    assert "WARNING" in loud and "not reportable" in loud


def test_the_report_explains_a_saving_above_the_arithmetic_share():
    kmac = {"g_s_y": 100.0, "g_s_uv": 20.0}          # 16.7% skipped
    big = report([_row(ms_luma=70.0)], decoder_kmac=kmac)   # 30% saved
    small = report([_row(ms_luma=95.0)], decoder_kmac=kmac)  # 5% saved
    assert "rANS decoding is not" in big
    assert "cheap" in small and "rANS" not in small


def test_the_report_marks_a_wrong_luma_plane_rather_than_hiding_it():
    assert "MISMATCH" in report([_row(luma_identical=False)])


@torch.no_grad()
def test_the_fast_path_leaves_the_packet_untouched():
    """`decompress` must not consume the strings: benchmarks decode the same one
    many times, and a coder that mutated its input would give a saving on the
    second call that had nothing to do with skipping chroma."""
    m = _model()
    packet = m.compress(torch.rand(1, 3, 96, 128))
    before = copy.deepcopy(packet)
    m.decompress(packet, luma_only=True)
    m.decompress(packet)
    assert m.packet_bytes(packet) == m.packet_bytes(before)
    assert packet["luma"]["y_strings"] == before["luma"]["y_strings"]
    assert packet["chroma"]["y_strings"] == before["chroma"]["y_strings"]
