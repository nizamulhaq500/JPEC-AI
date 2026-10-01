"""Tests for the `.jpegai` container and the `jpegai` CLI (Phase 14).

The container has one job: be lossless. The bytes `compress` produced come back
identical, the rate the file reports recomputes to the same number, and a packet with a
gain header and a spatial quality map round-trips values-exact. The CLI tests drive
encode/decode/inspect/conform against a *tiny* codec held in memory -- no checkpoint on
disk, `_load_model` monkeypatched -- so they stay fast while still exercising ROI,
luma-only, crop, and the honest-scope guards (format assertion, gain-only knobs).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from jpegai import cli
from jpegai.codestream import coded_bytes, describe, read_packet, write_packet
from jpegai.codestream.container import CONTAINER_VERSION
from jpegai.models.twobranch import TwoBranchCodec


def _tiny(**kw):
    """A deliberately small codec; these tests are about serialization, not quality."""
    m = TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                       chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                       synthesis_width=(24, 16, 16, 16), internal_format="420",
                       **kw).eval()
    m.update(force=True)
    return m


def _q_index(model, x, boost=3):
    y, _u, _s, _p = model._to_planes(x)
    lh, lw = model.g_a_y(y).shape[-2:]
    q = torch.zeros(1, 1, lh, lw, dtype=torch.int64)
    q[..., : lh // 2, :] = boost
    return q


def _png(path, h=64, w=64):
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path)
    return str(path)


@pytest.fixture
def img():
    torch.manual_seed(0)
    return torch.rand(1, 3, 64, 64)


# --- the container is lossless -------------------------------------------------------

def test_plain_roundtrip_is_byte_identical(tmp_path, img):
    m = _tiny()
    with torch.no_grad():
        pkt = m.compress(img)
    path = tmp_path / "p.jpegai"
    disk = write_packet(path, pkt)
    back = read_packet(path)

    for br in ("luma", "chroma"):
        for key in ("y_strings", "z_strings"):
            a = [bytes(s) for s in pkt[br][key]]
            b = [bytes(s) for s in back[br][key]]
            assert a == b, f"{br}.{key} differs"
        assert back[br]["z_shape"] == pkt[br]["z_shape"]
    assert back["shape"] == pkt["shape"] and back["pad"] == pkt["pad"]
    assert back["internal_format"] == pkt["internal_format"]
    assert "delta_beta" not in back and "q_residual" not in back
    # the model's own rate reads identically before and after the file
    assert TwoBranchCodec.packet_bytes(back) == TwoBranchCodec.packet_bytes(pkt)
    with torch.no_grad():
        assert torch.equal(m.decompress(pkt)["x_hat"], m.decompress(back)["x_hat"])
    assert disk >= coded_bytes(pkt)   # the file never undercounts the rate it carries


def test_gain_header_and_quality_map_roundtrip(tmp_path, img):
    m = _tiny(split_hyper=True, gain=True)
    q = _q_index(m, img)
    with torch.no_grad():
        pkt = m.compress(img, delta_beta=(1, 2), q_index=q)
    assert "delta_beta" in pkt and pkt.get("q_residual") is not None

    path = tmp_path / "g.jpegai"
    write_packet(path, pkt)
    back = read_packet(path)

    assert tuple(back["delta_beta"]) == (1, 2)
    assert torch.equal(back["q_residual"].to(torch.int64),
                       pkt["q_residual"].to(torch.int64)), "quality map not lossless"
    assert TwoBranchCodec.header_bytes(back) == TwoBranchCodec.header_bytes(pkt)
    assert TwoBranchCodec.packet_bytes(back) == TwoBranchCodec.packet_bytes(pkt)
    with torch.no_grad():
        assert torch.equal(m.decompress(pkt)["x_hat"], m.decompress(back)["x_hat"])


def test_quality_map_stored_at_minimal_width(tmp_path, img):
    """Table I residuals fit in int8; the container must not pad them to int64 and then
    charge the eightfold width to its own overhead (that would flatter nothing, but it
    inflates the honest `overhead_bytes` number `inspect` prints)."""
    m = _tiny(split_hyper=True, gain=True)
    with torch.no_grad():
        pkt = m.compress(img, delta_beta=(1, 1), q_index=_q_index(m, img))
    path = tmp_path / "n.jpegai"
    write_packet(path, pkt)
    d = describe(path)
    assert d["q_residual"]["dtype"] == "|i1"
    assert d["q_residual"]["nbytes"] == pkt["q_residual"].numel()  # one byte per position


def test_describe_separates_rate_from_disk(tmp_path, img):
    m = _tiny()
    with torch.no_grad():
        pkt = m.compress(img)
    path = tmp_path / "d.jpegai"
    disk = write_packet(path, pkt)
    d = describe(path)
    assert d["version"] == CONTAINER_VERSION
    assert d["coded_bytes"] == TwoBranchCodec.packet_bytes(pkt)
    assert d["disk_bytes"] == disk
    assert d["overhead_bytes"] == disk - d["coded_bytes"] >= 0
    assert d["delta_beta"] is None and d["q_residual"] is None


def test_non_container_is_rejected(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"definitely not a JPAI container")
    with pytest.raises(ValueError, match="not a .jpegai container"):
        read_packet(bad)


def test_truncated_header_is_rejected(tmp_path):
    short = tmp_path / "short.bin"
    short.write_bytes(b"JP")   # fewer bytes than the fixed header
    with pytest.raises(ValueError, match="truncated"):
        read_packet(short)


# --- the CLI, driven against an in-memory tiny codec ---------------------------------

def _patch_model(monkeypatch, model):
    """`jpegai` loads a checkpoint from disk; swap that for a codec we already hold."""
    monkeypatch.setattr(cli, "_load_model",
                        lambda ckpt, device=None: (model, {"model": "twobranch"},
                                                    torch.device("cpu")))


def test_cli_encode_inspect_conform_decode(tmp_path, monkeypatch):
    _patch_model(monkeypatch, _tiny())
    src = _png(tmp_path / "in.png")
    out = str(tmp_path / "out.jpegai")
    rec = str(tmp_path / "rec.png")
    assert cli.main(["encode", src, out, "--checkpoint", "x"]) == 0
    assert cli.main(["inspect", out]) == 0
    assert cli.main(["conform", out]) == 0
    assert cli.main(["decode", out, rec, "--checkpoint", "x"]) == 0
    assert np.asarray(Image.open(rec)).shape == (64, 64, 3)


def test_cli_luma_only_and_crop(tmp_path, monkeypatch):
    _patch_model(monkeypatch, _tiny())
    src = _png(tmp_path / "in.png")
    out = str(tmp_path / "out.jpegai")
    rec = str(tmp_path / "rec.png")
    cli.main(["encode", src, out, "--checkpoint", "x"])
    assert cli.main(["decode", out, rec, "--checkpoint", "x",
                     "--luma-only", "--crop", "0,0,32,16"]) == 0
    assert np.asarray(Image.open(rec)).shape == (16, 32, 3)   # crop is h=16, w=32


def test_cli_roi_writes_a_quality_map(tmp_path, monkeypatch):
    _patch_model(monkeypatch, _tiny(split_hyper=True, gain=True))
    src = _png(tmp_path / "in.png")
    mask = np.zeros((64, 64), np.uint8)
    mask[:32] = 255                       # boost the top half
    Image.fromarray(mask).save(tmp_path / "mask.png")
    out = str(tmp_path / "out.jpegai")
    assert cli.main(["encode", src, out, "--checkpoint", "x",
                     "--roi", str(tmp_path / "mask.png"), "--roi-boost", "4"]) == 0
    assert describe(out)["q_residual"] is not None


def test_cli_roi_requires_a_gain_checkpoint(tmp_path, monkeypatch):
    _patch_model(monkeypatch, _tiny())    # fixed-rate: no quality map to steer
    src = _png(tmp_path / "in.png")
    Image.fromarray(np.full((64, 64), 255, np.uint8)).save(tmp_path / "mask.png")
    with pytest.raises(SystemExit, match="gain"):
        cli.main(["encode", src, str(tmp_path / "o.jpegai"), "--checkpoint", "x",
                  "--roi", str(tmp_path / "mask.png")])


def test_cli_asserts_internal_format(tmp_path, monkeypatch):
    _patch_model(monkeypatch, _tiny())    # codes 420
    src = _png(tmp_path / "in.png")
    with pytest.raises(SystemExit, match="420"):
        cli.main(["encode", src, str(tmp_path / "o.jpegai"), "--checkpoint", "x",
                  "--internal-format", "444"])


def test_cli_betas_require_a_gain_checkpoint(tmp_path, monkeypatch):
    _patch_model(monkeypatch, _tiny())
    src = _png(tmp_path / "in.png")
    with pytest.raises(SystemExit, match="gain"):
        cli.main(["encode", src, str(tmp_path / "o.jpegai"), "--checkpoint", "x",
                  "--beta-luma", "100"])


def test_cli_bench_forwards_to_runbench(monkeypatch):
    """`bench` is a thin pass-through: everything after `--` reaches runbench verbatim
    and its exit code propagates (None -> 0)."""
    import jpegai.eval.runbench as runbench
    seen = {}

    def fake_main(argv):
        seen["argv"] = list(argv)
        return 3

    monkeypatch.setattr(runbench, "main", fake_main)
    rc = cli.main(["bench", "--", "--neural", "foo", "--out", "bar"])
    assert rc == 3
    assert seen["argv"][-4:] == ["--neural", "foo", "--out", "bar"]
