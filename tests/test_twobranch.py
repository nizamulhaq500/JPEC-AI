"""Tests for the two-branch YCbCr codec — Phase 4 items 2-6.

The structural invariant that everything else depends on is that both latents land on
the **same spatial grid**, because eq. (3) concatenates them. It holds trivially at
4:2:0 with square inputs, which is exactly why it is tested at 4:2:2 (anisotropic
stride) and on odd sizes (ceilings) as well.

The other thing tested hard is `luma_only`. It is easy to write a version that decodes
the chroma stream and then throws the result away — that produces identical pixels and
a completely false complexity claim. The test therefore *deletes the chroma part of the
packet* and requires the decode to still succeed.
"""

from __future__ import annotations

import math

import pytest
import torch

from jpegai.models.colour import FORMATS
from jpegai.models.layers import conv, deconv
from jpegai.models.twobranch import (
    LATENT_STRIDE, HyperpriorBranch, SecondaryAnalysis, SecondarySynthesis,
    TwoBranchCodec, stride_schedule,
)

FMT_NAMES = ["444", "422", "420"]


def _model(fmt="420", **kw):
    """A deliberately small codec. These tests are about shapes and plumbing."""
    return TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                          chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                          synthesis_width=(24, 16, 16, 16),
                          internal_format=fmt, **kw).eval()


# ---------------------------------------------------------------------------
# the stride schedule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt,stages", [("444", 4), ("422", 4), ("420", 3)])
def test_stage_count_matches_the_paper_where_the_paper_says(fmt, stages):
    assert len(stride_schedule(fmt)) == stages


@pytest.mark.parametrize("fmt", FMT_NAMES)
def test_the_schedule_lands_exactly_on_the_latent_grid(fmt):
    f, sched = FORMATS[fmt], stride_schedule(fmt)
    assert math.prod(s[0] for s in sched) * f.ver == LATENT_STRIDE
    assert math.prod(s[1] for s in sched) * f.hor == LATENT_STRIDE


def test_422_needs_an_anisotropic_stage_and_gets_exactly_one():
    """The case the paper does not mention. Chroma is already half width."""
    sched = stride_schedule("422")
    assert sched.count((2, 1)) == 1
    assert sched.count((2, 2)) == 3


def test_isotropic_stages_come_first_so_the_tensor_shrinks_early():
    sched = stride_schedule("422")
    areas = [s[0] * s[1] for s in sched]
    assert areas == sorted(areas, reverse=True), sched


def test_a_format_that_cannot_reach_the_latent_grid_is_rejected():
    from jpegai.models.colour import ChromaFormat
    with pytest.raises(ValueError, match="not a power of two"):
        stride_schedule(ChromaFormat("411", 1, 3))


def test_anisotropic_deconv_exactly_inverts_anisotropic_conv():
    """`output_padding = stride - 1` had to become per-axis for the 4:2:2 stage."""
    x = torch.rand(1, 4, 16, 16)
    down = conv(4, 4, 5, stride=(2, 1))(x)
    assert tuple(down.shape[-2:]) == (8, 16)
    up = deconv(4, 4, 5, stride=(2, 1))(down)
    assert tuple(up.shape[-2:]) == (16, 16)


# ---------------------------------------------------------------------------
# eq. (3): one grid, and the concatenation width
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", FMT_NAMES)
@pytest.mark.parametrize("h,w", [(128, 128), (192, 256), (129, 130)])
def test_both_latents_land_on_the_same_grid(fmt, h, w):
    out = _model(fmt)(torch.rand(1, 3, h, w))
    assert out["y"].shape[-2:] == out["y_uv"].shape[-2:]
    assert out["y"].shape[1] == 32 and out["y_uv"].shape[1] == 16
    assert out["x_hat"].shape == (1, 3, h, w)


def test_secondary_synthesis_consumes_chroma_plus_luma_at_full_width():
    """eq. (3): the luma latent is concatenated, not projected down first."""
    g = SecondarySynthesis(latent=16, supp=32, widths=(24, 16, 16), fmt="420")
    first = next(m for m in g.body if hasattr(m, "in_channels"))
    assert first.in_channels == 16 + 32


def test_secondary_analysis_takes_the_luma_link_as_a_third_channel():
    g = SecondaryAnalysis(latent=16, widths=(16, 16, 24, 32), fmt="420")
    first = next(m for m in g.body if hasattr(m, "in_channels"))
    assert first.in_channels == 3, "Cb, Cr and the downsampled luma"


def test_a_grid_mismatch_raises_instead_of_broadcasting_silently():
    g = SecondarySynthesis(latent=16, supp=32, widths=(24, 16, 16), fmt="420")
    with pytest.raises(ValueError, match="eq. \\(3\\) needs both latents on one grid"):
        g(torch.rand(1, 16, 8, 8), torch.rand(1, 32, 4, 4))

    a = SecondaryAnalysis(latent=16, widths=(16, 16, 24, 32), fmt="420")
    with pytest.raises(ValueError, match="luma link must be on the chroma grid"):
        a(torch.rand(1, 2, 16, 16), torch.rand(1, 1, 32, 32))


def test_the_final_analysis_width_is_forced_to_the_latent_width():
    """The config carries four widths; a 3-stage branch drops one from the front."""
    g = SecondaryAnalysis(latent=16, widths=(16, 16, 24, 32), fmt="420")
    convs = [m for m in g.body if hasattr(m, "out_channels")]
    assert len(convs) == 3
    assert convs[-1].out_channels == 16, "last stage must project to the latent"
    assert convs[0].out_channels == 16 and convs[1].out_channels == 24


# ---------------------------------------------------------------------------
# four streams
# ---------------------------------------------------------------------------
def test_there_are_four_streams_and_the_rate_counts_all_of_them():
    m = _model()
    out = m(torch.rand(1, 3, 128, 128))
    assert set(out["likelihoods"]) == {"y", "z", "y_uv", "z_uv"}

    y_bits, z_bits = m.estimated_bits(out)
    luma_only = {"likelihoods": {k: v for k, v in out["likelihoods"].items()
                                if not k.endswith("_uv")}}
    y_luma, z_luma = m.estimated_bits(luma_only)
    assert y_bits > y_luma > 0, "chroma must contribute to the rate"
    assert z_bits > z_luma > 0


def test_two_aux_parameters_one_per_branch_and_disjoint_from_the_main_set():
    m = _model()
    aux = m.aux_parameters()
    assert len(aux) == 2
    main_ids = {id(p) for p in m.main_parameters()}
    assert not any(id(p) in main_ids for p in aux)
    assert len(main_ids) + len(aux) == len(list(m.parameters()))
    assert torch.isfinite(m.aux_loss())


def test_one_shared_sigma_table_not_two():
    """Both branches index the same 32-level grid; two copies would double the tables."""
    m = _model()
    assert m.branch_y is not m.branch_uv
    gc = [mod for mod in m.modules()
          if type(mod).__name__ == "GaussianConditional"]
    assert len(gc) == 1, "the sigma table must be shared between branches"


def test_tables_are_not_ready_until_update_is_called():
    m = _model()
    assert not m.tables_ready
    with pytest.raises(RuntimeError, match="entropy tables not built"):
        m.compress(torch.rand(1, 3, 64, 64))
    m.update(force=True)
    assert m.tables_ready
    assert m.table_bytes() > 0


# ---------------------------------------------------------------------------
# the real bitstream
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", FMT_NAMES)
def test_the_luma_latent_survives_the_bitstream_bit_exactly(fmt):
    """The Phase 3 gate criterion, now with two branches sharing one coder."""
    torch.manual_seed(0)
    m = _model(fmt)
    m.update(force=True)
    x = torch.rand(1, 3, 128, 192)
    packet = m.compress(x)
    decoded = m.decompress(packet)
    reference = m(x, noise=False, ste=True)
    assert torch.equal(decoded["y_hat"], reference["y_hat"])


@pytest.mark.parametrize("h,w", [(128, 192), (130, 194), (65, 65)])
def test_decode_returns_the_original_size_on_odd_inputs(h, w):
    m = _model()
    m.update(force=True)
    packet = m.compress(torch.rand(1, 3, h, w))
    assert tuple(m.decompress(packet)["x_hat"].shape) == (1, 3, h, w)


def test_output_is_in_range():
    m = _model()
    m.update(force=True)
    x_hat = m.decompress(m.compress(torch.rand(1, 3, 64, 64)))["x_hat"]
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0


def test_the_packet_records_the_internal_format_it_was_coded_with():
    """A decoder that guesses 4:2:0 for a 4:2:2 stream desynchronises."""
    for fmt in FMT_NAMES:
        m = _model(fmt)
        m.update(force=True)
        assert m.compress(torch.rand(1, 3, 64, 64))["internal_format"] == fmt


# ---------------------------------------------------------------------------
# luma_only — Phase 4 item 6
# ---------------------------------------------------------------------------
def test_luma_only_never_reads_the_chroma_stream():
    """Deleted, not ignored. A decoder that reads then discards is not a fast path."""
    m = _model()
    m.update(force=True)
    packet = m.compress(torch.rand(1, 3, 128, 128))
    full = m.decompress(packet)

    del packet["chroma"]
    partial = m.decompress(packet, luma_only=True)

    assert partial["luma_only"] is True
    assert torch.equal(partial["luma"], full["luma"]), \
        "the luma plane must be identical to a full decode"


def test_luma_only_produces_a_grey_image_with_the_right_luma():
    m = _model()
    m.update(force=True)
    out = m.decompress(m.compress(torch.rand(1, 3, 96, 96)), luma_only=True)
    assert torch.allclose(out["chroma"], torch.full_like(out["chroma"], 0.5))
    r, g, b = out["x_hat"][:, 0], out["x_hat"][:, 1], out["x_hat"][:, 2]
    assert torch.allclose(r, g, atol=1e-5) and torch.allclose(g, b, atol=1e-5), \
        "Cb = Cr = 0.5 must decode to R = G = B"


def test_luma_only_costs_strictly_fewer_bytes():
    m = _model()
    m.update(force=True)
    packet = m.compress(torch.rand(1, 3, 128, 128))
    assert 0 < m.packet_bytes(packet, luma_only=True) < m.packet_bytes(packet)


def test_packet_bytes_counts_every_stream_exactly_once():
    m = _model()
    m.update(force=True)
    packet = m.compress(torch.rand(1, 3, 64, 64))
    by_hand = sum(len(s) for part in ("luma", "chroma")
                  for key in ("y_strings", "z_strings")
                  for s in packet[part][key])
    assert m.packet_bytes(packet) == by_hand


def test_stream_bytes_partitions_the_packet_and_names_the_streams_like_forward():
    """Two properties, both load-bearing for the rate gate.

    **It is a partition.** The per-stream bytes must sum to `packet_bytes` exactly --
    no stream counted twice, none omitted. A gap attribution that does not add up to
    the total invites exactly the wrong conclusion: "all four streams look fine, so
    the aggregate must be wrong".

    **The keys match `likelihoods`.** The gate pairs each stream's bytes with that
    stream's estimate by name. If the two dicts drifted apart, `est_q_per_stream.get`
    would return `None`, every pairing would be skipped, and the per-stream gap would
    silently report *nothing at all* while the gate went on printing "ok".
    """
    m = _model()
    m.update(force=True)
    x = torch.rand(1, 3, 64, 64)
    packet = m.compress(x)
    sb = m.stream_bytes(packet)

    assert sum(sb.values()) == m.packet_bytes(packet)
    assert set(sb) == set(m(x)["likelihoods"])
    assert all(v > 0 for v in sb.values()), sb


def test_stream_bytes_agrees_with_packet_bytes_on_the_luma_only_subset():
    """`packet_bytes(luma_only=True)` and `stream_bytes` derive the "which streams
    does a luma decoder need" answer independently -- one by selecting packet parts,
    the other by selecting keys. They must agree, or `--luma-only`'s headline saving
    is measured against a different set of streams than the gate attributes bytes to.
    """
    m = _model()
    m.update(force=True)
    packet = m.compress(torch.rand(1, 3, 64, 64))
    sb = m.stream_bytes(packet)
    assert sb["y"] + sb["z"] == m.packet_bytes(packet, luma_only=True)


# ---------------------------------------------------------------------------
# construction from config
# ---------------------------------------------------------------------------
def test_scale_only_and_mean_scale_both_build_and_code():
    for mean_scale in (False, True):
        m = _model(mean_scale=mean_scale)
        m.update(force=True)
        out = m(torch.rand(1, 3, 64, 64))
        assert (out["means"] is not None) == mean_scale
        d = m.decompress(m.compress(torch.rand(1, 3, 64, 64)))
        assert torch.isfinite(d["x_hat"]).all()


def test_build_two_branch_takes_the_widths_from_the_config():
    from jpegai.config import load_config
    from jpegai.models.twobranch import build_two_branch
    for name, luma, chroma in [("tierA", 96, 48), ("full", 160, 96)]:
        cfg = load_config(name)
        m = build_two_branch(cfg)
        assert (m.luma_latent, m.chroma_latent) == (luma, chroma), name
        assert m.fmt.name == cfg.colour.internal_format


def test_a_branch_is_reusable_on_its_own():
    """HyperpriorBranch is shared between the two branches; it must stand alone."""
    from jpegai.models.entropy import GaussianConditional, build_scale_table
    gc = GaussianConditional(build_scale_table(0.11, 54.82, 32), scale_bound=0.11)
    b = HyperpriorBranch(16, 16, mean_scale=True)
    out = b(torch.rand(1, 16, 8, 8), gc)
    assert out["y_hat"].shape == (1, 16, 8, 8)
    assert out["means"] is not None and out["scales"].shape == (1, 16, 8, 8)
