"""Headless tests for the demo engine (`demo/engine.py`).

The Streamlit app cannot bind a port in the sandbox, so the UI is a thin shell and the
real behaviour lives in `engine`, which is pure -- arrays in, arrays/numbers out. These
drive it on the same deliberately tiny in-memory codec the container tests use, so they
stay fast while exercising the exact path the app shows: container accounting, the
gain-only guards, the luma-only base rate, region PSNR, and the latent bit maps.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from demo import engine
from jpegai.models.twobranch import TwoBranchCodec

DEV = torch.device("cpu")


def _tiny(**kw):
    m = TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                       chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                       synthesis_width=(24, 16, 16, 16), internal_format="420",
                       **kw).eval()
    m.update(force=True)
    return m


def _rgb(h=64, w=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.fixture
def fixed():
    return _tiny()


@pytest.fixture
def gain():
    return _tiny(split_hyper=True, gain=True)


# --- container accounting the app surfaces -------------------------------------------

def test_capabilities_lists_both_sides_disjointly():
    assert engine.CAPABILITIES["in"] and engine.CAPABILITIES["out"]
    assert not (set(engine.CAPABILITIES["in"]) & set(engine.CAPABILITIES["out"]))


def test_encode_reports_two_sizes_and_round_trips(tmp_path, fixed):
    rgb = _rgb()
    path = tmp_path / "a.jpegai"
    res = engine.encode(fixed, rgb, DEV, path=str(path))
    assert (res.width, res.height) == (64, 64)
    assert res.coded_bytes > 0 and res.bpp > 0
    assert res.disk_bytes >= res.coded_bytes       # the file never undercounts the rate
    assert 0 <= res.scaffolding_pct < 100
    d = engine.inspect(str(path))
    assert d["coded_bytes"] == res.coded_bytes and d["disk_bytes"] == res.disk_bytes
    rec = engine.decode(fixed, res.packet, DEV)
    assert rec.shape == (64, 64, 3) and rec.dtype == np.uint8


def test_encode_without_path_falls_back_to_coded(fixed):
    res = engine.encode(fixed, _rgb(), DEV)
    assert res.disk_bytes == res.coded_bytes and res.scaffolding_pct == 0.0


def test_luma_only_base_is_a_cheaper_prefix(fixed):
    res = engine.encode(fixed, _rgb(), DEV)
    full = engine.decode(fixed, res.packet, DEV)
    base = engine.decode(fixed, res.packet, DEV, luma_only=True)
    assert full.shape == base.shape == (64, 64, 3)
    assert TwoBranchCodec.packet_bytes(res.packet, luma_only=True) < res.coded_bytes


# --- quality metrics ------------------------------------------------------------------

def test_psnr_inf_when_identical_and_finite_otherwise():
    a = _rgb()
    assert engine.psnr(a, a) == float("inf")
    b = a.copy(); b[0, 0, 0] ^= 0xFF
    assert np.isfinite(engine.psnr(a, b))


def test_region_psnr_splits_inside_from_outside():
    a = _rgb()
    b = a.copy(); b[:32] = 0                        # damage the top half only
    mask = np.zeros((64, 64), np.uint8); mask[:32] = 255
    inside, outside = engine.region_psnr(a, b, mask)
    assert np.isfinite(inside) and outside == float("inf")   # bottom half untouched


def test_region_psnr_nan_on_empty_side():
    a = _rgb()
    inside, _ = engine.region_psnr(a, a.copy(), np.zeros((64, 64), np.uint8))
    assert np.isnan(inside)                          # nothing inside -> undefined


# --- the gain-only RoI path -----------------------------------------------------------

def test_mask_to_q_blocks_a_fixed_rate_checkpoint(fixed):
    with pytest.raises(ValueError, match="gain"):
        engine._mask_to_q(fixed, engine.cli._to_tensor(_rgb(), DEV),
                          np.full((64, 64), 255, np.uint8), 4)


def test_mask_to_q_lands_on_the_luma_grid_and_clamps(gain):
    x = engine.cli._to_tensor(_rgb(), DEV)
    half = np.zeros((64, 64), np.uint8); half[:32] = 255
    q = engine._mask_to_q(gain, x, half, boost=4)
    y, _u, _s, _p = gain._to_planes(x)
    lh, lw = gain.g_a_y(y).shape[-2:]
    assert q.shape == (1, 1, lh, lw)
    assert int(q.max()) == 4 and int(q.min()) == 0
    big = engine._mask_to_q(gain, x, np.full((64, 64), 255, np.uint8), boost=99)
    assert int(big.max()) <= 8                       # clamped into Table I's range


def test_roi_encode_writes_a_quality_map(gain):
    mask = np.zeros((64, 64), np.uint8); mask[:32] = 255
    res = engine.encode(gain, _rgb(), DEV, roi_mask=mask, roi_boost=4)
    assert res.packet.get("q_residual") is not None


# --- the latent bit-allocation explorer -----------------------------------------------

def test_latent_maps_live_on_the_luma_grid(gain):
    maps = engine.latent_maps(gain, _rgb(), DEV)
    x = engine.cli._to_tensor(_rgb(), DEV)
    y, _u, _s, _p = gain._to_planes(x)
    lh, lw = gain.g_a_y(y).shape[-2:]
    assert maps["grid"] == (lh, lw)
    assert maps["bits"].shape == (lh, lw)
    assert maps["sigma"] is not None and maps["sigma"].shape == (lh, lw)
    assert maps["total_bits"] == pytest.approx(float(maps["bits"].sum()))
    assert np.isfinite(maps["bits"]).all()


def test_latent_maps_delta_beta_moves_the_budget(gain):
    lo = engine.latent_maps(gain, _rgb(), DEV, delta_beta=(-400, 0))["total_bits"]
    hi = engine.latent_maps(gain, _rgb(), DEV, delta_beta=(400, 0))["total_bits"]
    assert lo != hi                                  # Δβ reweights the latent


# --- the UI is a thin shell that imports cleanly --------------------------------------

def test_app_imports_and_wires_the_engine():
    import demo.app as app
    assert callable(app.main) and app.engine is engine
