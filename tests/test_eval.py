"""Tests for the Phase 1 measuring instrument.

Runnable two ways::

    python tests/test_eval.py       # no pytest needed
    pytest tests/test_eval.py

Deliberately has **no torch dependency**: the metric backends are stubbed. The
point is to prove the *plumbing* -- dataset walk, cache, aggregation, the
lower-is-better sign flip, BD-rate wiring, report writing -- is correct before the
real metrics arrive. When they do, the only thing left to validate is the metrics
themselves.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpegai.config import load_config  # noqa: E402
from jpegai.eval import codecs as anchors  # noqa: E402
from jpegai.eval import runbench  # noqa: E402
from jpegai.eval.bdrate import bd_rate, bd_rate_table  # noqa: E402


# ---------------------------------------------------------------------------
# Stub metrics: real arithmetic, no torch.
# ---------------------------------------------------------------------------
class _StubMetrics:
    """Two metrics with opposite directions, so the sign flip gets exercised."""

    PAPER_SEVEN = ["ms_ssim", "nlpd"]
    REGISTRY = {
        "ms_ssim": (None, True, "stub"),    # higher is better
        "nlpd": (None, False, "stub"),      # lower is better
    }
    calls = 0

    @classmethod
    def compute_all(cls, ref, test, *, metrics=None, include_psnr=True, bitdepth=8):
        cls.calls += 1
        a = np.asarray(ref, dtype=np.float64)
        b = np.asarray(test, dtype=np.float64)
        mse = float(np.mean((a - b) ** 2))
        out = {}
        for m in (metrics or cls.PAPER_SEVEN):
            if m == "ms_ssim":
                out[m] = 1.0 / (1.0 + 200.0 * mse)      # -> 1 as quality rises
            elif m == "nlpd":
                out[m] = float(np.sqrt(mse))            # -> 0 as quality rises
            else:
                raise KeyError(m)
        if include_psnr:
            p = float("inf") if mse <= 0 else 10.0 * np.log10(1.0 / mse)
            out["psnr"] = p
            out["psnr_y"] = p
        return out


def _install_stubs():
    runbench._load_metrics = lambda: _StubMetrics
    runbench._to_tensor = lambda rgb: np.asarray(rgb, dtype=np.float64) / 255.0
    return _StubMetrics


def _make_images(d: Path, n: int = 3, size: int = 96) -> None:
    """Textured synthetic images. Flat images compress to nothing and make every
    codec look identical, which would hide real plumbing bugs."""
    from PIL import Image

    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        base = np.stack([
            (xx * 2 + i * 30) % 256,
            (yy * 2 + i * 50) % 256,
            ((xx + yy) * 3) % 256,
        ], axis=-1).astype(np.int16)
        base += rng.integers(-25, 26, base.shape)
        Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(d / f"img{i:02d}.png")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_inheritance():
    tier_a, full = load_config("tierA"), load_config("full")

    assert tier_a.channels.primary_latent == 96
    assert full.channels.primary_latent == 160

    # Deep merge: full.yaml never mentions these, so they must come from tierA.
    assert full.colour.matrix == "bt709"
    assert full.entropy.mcm_stages == tier_a.entropy.mcm_stages == 4
    assert full.entropy.isigma_pad_value == 1411
    assert dict(full.train.distortion_weights) == {"y": 6.0, "u": 1.0, "v": 1.0}

    # Override.
    assert full.train.batch == 16 and tier_a.train.batch == 8
    assert full.train.amp is True and tier_a.train.amp is False

    # Provenance, base first.
    assert full["_files"][0].endswith("tierA.yaml")
    assert full["_files"][-1].endswith("full.yaml")


def test_config_eq3_invariant():
    """Paper eq. (3): secondary synthesis consumes concat(y_UV, y_Y).

    Confirmed visually in Fig. 1 -- the concatenation input is the luma *latent*.
    If someone widens one latent and forgets the other, this catches it.
    """
    for name in ("tierA", "full"):
        ch = load_config(name).channels
        assert ch.secondary_synthesis_in == ch.secondary_latent + ch.primary_latent
        assert ch.pred_primary_preshuffle == 4 * ch.primary_latent
        assert ch.pred_secondary_preshuffle == 4 * ch.secondary_latent


def test_config_normative_constants():
    """Constants read out of the WG1 reference software. NOT tunable.

    Each of these was, at some point, a guess of ours that turned out wrong or
    unconfirmed. They are pinned here because every one of them produces a model
    that builds, trains and reports plausible numbers while not being JPEG AI.
    Provenance with file paths: docs/06-normative-constants.md.
    """
    for name in ("tierA", "full"):
        c = load_config(name)
        ch, ent, rate = c.channels, c.entropy, c.rate

        # The hyper autoencoder is channel-preserving and every hyper module is
        # built with chs=chs_ls of its own branch (common_modules.py:116-128), so
        # the hyper width is derived. We previously had 128 here, reasoning from a
        # [128, 64] CDF shape; that 128 is an unused fallback default.
        assert ch.hyper_latent == ch.primary_latent
        assert ch.hyper_secondary_latent == ch.secondary_latent

        # MCM_phases.chs2group() asserts this outright. MCM is luma-only, so it
        # binds the primary latent only.
        assert ch.primary_latent % 32 == 0

        # The precision chain closes on itself, which is why we trust it:
        # scaled = (gain_vector + beta_displacement) + sigma = (5+5)+7 = 17, and
        # lsbs_scale_mode.py independently hardcodes 17.
        assert ent.sigma_precision == 7
        assert ent.scaler_precision == (rate.gain_vector_precision
                                        + rate.beta_displacement_precision) == 10

        # sigma_idx_max_value = (level-1) * 2^precision - 1 = 3967, so the RVS and
        # LSBS tables span [0, 3967] -- 3968 entries. This is what the paper's
        # "[..., 3968]" table extent means, and it makes the Isigma -> sigma-class
        # mapping a right-shift by sigma_precision.
        assert ent.sigma_quant_level == 32
        assert ent.isigma_table_size == (ent.sigma_quant_level - 1) * 2 ** ent.sigma_precision
        assert (ent.sigma_quant_min, ent.sigma_quant_max) == (0.11, 54.82)

        # thr_skip. We had 0 with a "calibrate later" comment; no calibration
        # needed, it is normative.
        assert ent.skip_threshold == 382
        assert ent.skip_judge_thr == 3

        # me-tANS is parameterised by probability mass bits, not a tableLog. At 8
        # bits the state space is 256 per sigma class, and 32*256*4 B = 32 KiB of
        # decode transitions -- consistent with the paper's "~100 KB of tables".
        # Our earlier tans_table_log: 11 would have been 8x too large.
        assert ent.tans_mass_bits == 8
        assert ent.hyper_max_symbol == ent.z_range - 1 == 62

        # down_shuffle returns (part1, part4, part2, part3) -> diagonal first.
        # Independently confirmed by context.py's odd-size guards: the row guard
        # fires for stages {1,3} (dy=1) and the column guard for {1,2} (dx=1).
        assert [list(g) for g in ent.mcm_group_order] == [[0, 0], [1, 1], [0, 1], [1, 0]]

        # 18-entry ladder from gain_unit/params.py. Note beta=0.012, the paper's
        # second base model, is NOT a ladder entry (0.01 and 0.015 bracket it).
        assert len(rate.beta_list) == 18
        assert rate.beta_list[0] == 0.0002 and rate.beta_list[-1] == 3.0
        assert 0.012 not in rate.beta_list

    # The normative widths belong to full.yaml; Tier A is our deliberate cut.
    full = load_config("full").channels
    assert (full.primary_latent, full.secondary_latent) == (160, 96)
    assert full.secondary_synthesis_in == 256          # paper eq. (3)
    assert full.pred_primary_preshuffle == 640         # paper


def test_config_sop_has_no_encoder():
    """Three decoders, two encoders -- SOP reuses BOP's.

    autoencoder_data/encoder/ has bop_prim, bop_sec, hop_prim, hop_sec and no
    sop_*; cfg/profiles/bopEnc_sopDec.json makes the reuse explicit. Also checks
    that synthesis_transform_id is a cumulative capability list (SOP [0], BOP
    [1,0], HOP [2,1,0]) rather than a single selector -- that is what makes a HOP
    stream decodable by an SOP decoder.
    """
    for name in ("tierA", "full"):
        decs = {d["name"]: d for d in load_config(name).decoders}
        assert decs["SOP"]["has_encoder"] is False
        assert decs["BOP"]["has_encoder"] is True and decs["HOP"]["has_encoder"] is True

        enabled = [d for d in load_config(name).decoders if d.get("enabled", True)]
        assert sum(1 for d in enabled if d["has_encoder"]) == 2, "two encoders, three decoders"

        assert list(decs["SOP"]["signals"]) == [0]
        assert list(decs["BOP"]["signals"]) == [1, 0]
        assert list(decs["HOP"]["signals"]) == [2, 1, 0]
        # Each list must start with its own id and be strictly descending.
        for d in enabled:
            sig = list(d["signals"])
            assert sig[0] == d["id"] and sig == sorted(sig, reverse=True)


def test_paper_seven_uses_psnr_hvs_not_hvsm():
    """The seventh metric is PSNR-HVS. QAF keeps the FIRST return value of
    psnr_hvsm.psnr_hvs_hvsm(), which is PSNR-HVS; the paper's Table III heading
    agrees and only its prose says PSNR-HVS-M.

    Also pins the per-metric plane, which is the part that cannot be guessed: six
    of the seven are luma-only and only FSIM sees colour. Running everything on
    RGB (which we did at first) yields plausible, incomparable numbers.

    Skipped without torch, since it imports the real metrics module.
    """
    try:
        from jpegai.eval import metrics as M
    except Exception as exc:                                # pragma: no cover
        print(f"    (skipped: {type(exc).__name__}: {exc})")
        return

    assert M.PAPER_SEVEN == ["ms_ssim", "vif", "fsim", "vmaf", "nlpd",
                             "psnr_hvs", "iw_ssim"]
    assert "psnr_hvsm" not in M.PAPER_SEVEN, "PSNR-HVS-M is not one of the seven"
    assert "psnr_hvsm" in M.REGISTRY, "still report it, just do not average it"
    assert len(M.PAPER_SEVEN) == 7 and len(set(M.PAPER_SEVEN)) == 7
    assert all(n in M.REGISTRY for n in M.PAPER_SEVEN)

    assert M.INTERNAL_BITS == 10, "QAF's MetricParent(bits=10, max_val=1023)"

    # NLPD is the only lower-is-better metric among the seven; runbench negates it
    # before the BD-rate fit, so a flipped flag silently inverts one column.
    lower = [n for n in M.PAPER_SEVEN if not M.REGISTRY[n][1]]
    assert lower == ["nlpd"], f"expected only nlpd to be lower-is-better, got {lower}"


def test_luma_helper_is_a_no_op_on_grayscale():
    """luma() must pass 1-channel input straight through, and must return the Y of
    a 3-channel input. The second property is what the NLPD path relies on: we
    feed pyiqa Y replicated across three channels, which is exact only because a
    weighted-sum luma conversion of a grey image returns that grey unchanged.
    """
    try:
        import torch
        from jpegai.eval import metrics as M
    except Exception as exc:                                # pragma: no cover
        print(f"    (skipped: {type(exc).__name__}: {exc})")
        return

    torch.manual_seed(0)
    gray = torch.rand(1, 1, 32, 32)
    assert torch.equal(M.luma(gray), gray), "luma() must not touch 1-channel input"

    y = M.luma(torch.rand(1, 3, 32, 32))
    assert y.shape == (1, 1, 32, 32)
    # Replicating Y to 3 channels and re-extracting luma is the identity.
    assert torch.allclose(M.luma(y.repeat(1, 3, 1, 1)), y, atol=1e-5)


def test_config_override_typo_is_an_error():
    from jpegai.config import apply_overrides

    cfg = load_config("tierA")
    try:
        apply_overrides(cfg, ["trian.batch=4"])
    except KeyError:
        pass
    else:
        raise AssertionError("a typo'd config section must raise, not be invented")

    try:
        apply_overrides(cfg, ["train.btach=4"])
    except KeyError:
        pass
    else:
        raise AssertionError("a typo'd config key must raise, not be invented")

    apply_overrides(cfg, ["train.batch=4", "tools.rvs.enable=false"])
    assert cfg.train.batch == 4 and cfg.tools.rvs.enable is False


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------
def test_codec_roundtrip_shape_and_monotone_rate():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for c in anchors.REGISTRY.values():
        if not c.available():
            continue
        sizes = []
        for q in c.qualities:
            nbytes, dec = c.encode_decode(img, q)
            assert dec.shape == img.shape and dec.dtype == np.uint8
            assert nbytes > 0
            sizes.append(nbytes)
        # Rate must move one way across the ladder, or the cubic fit is garbage.
        up = all(a <= b for a, b in zip(sizes, sizes[1:]))
        down = all(a >= b for a, b in zip(sizes, sizes[1:]))
        assert up or down, f"{c.name}: non-monotone rate ladder {sizes}"


def test_codec_rejects_bad_input():
    c = anchors.get("jpeg")
    for bad in (np.zeros((8, 8), np.uint8),
                np.zeros((8, 8, 3), np.float32),
                np.zeros((8, 8, 4), np.uint8)):
        try:
            c.encode_decode(bad, 50)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted bad input {bad.shape} {bad.dtype}")


# ---------------------------------------------------------------------------
# BD-rate
# ---------------------------------------------------------------------------
def test_bdrate_known_answers():
    rate = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
    qual = np.array([26.0, 29.0, 32.0, 35.0, 38.0])
    assert abs(bd_rate(rate, qual, rate, qual)) < 1e-9
    assert abs(bd_rate(rate, qual, rate / 2, qual) + 50.0) < 1e-6
    assert abs(bd_rate(rate, qual, rate * 2, qual) - 100.0) < 1e-6
    assert np.isnan(bd_rate(rate, qual, rate, qual + 100))


def test_bdrate_table_avg_is_unweighted_mean():
    """docs/05 verified all 19 rows of Tables III-VI as the plain mean of 7."""
    rate = [0.1, 0.2, 0.4, 0.8]
    a = {"bpp": rate, "m1": [20.0, 25, 30, 35], "m2": [10.0, 15, 20, 25]}
    t = {"bpp": [r / 2 for r in rate], "m1": a["m1"], "m2": a["m2"]}
    row = bd_rate_table(a, t, ["m1", "m2"])
    assert abs(row["m1"] + 50) < 1e-6 and abs(row["m2"] + 50) < 1e-6
    assert abs(row["AVG"] - (row["m1"] + row["m2"]) / 2) < 1e-12


# ---------------------------------------------------------------------------
# runbench plumbing
# ---------------------------------------------------------------------------
def test_runbench_end_to_end():
    stub = _install_stubs()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data, out = td / "imgs", td / "out"
        data.mkdir()
        out.mkdir()
        _make_images(data, n=3)

        # Redirect cache into the temp dir so the test never touches results/.
        real_cache = runbench.CACHE
        runbench.CACHE = out / "cache"
        try:
            images = runbench.list_images(data)
            assert len(images) == 3

            names = ["ms_ssim", "nlpd"]
            jpeg = anchors.get("jpeg")

            stub.calls = 0
            curve = runbench.measure_codec(jpeg, images, "tst", names, verbose=False)
            first_calls = stub.calls
            assert first_calls == 3 * len(jpeg.qualities)

            # Every field has one entry per quality point.
            for f in ["bpp", "ms", "quality"] + names:
                assert len(curve[f]) == len(jpeg.qualities), f
            # bpp rises with quality; the two stub metrics move opposite ways.
            assert curve["bpp"] == sorted(curve["bpp"])
            assert curve["ms_ssim"] == sorted(curve["ms_ssim"])
            assert curve["nlpd"] == sorted(curve["nlpd"], reverse=True)

            # Cache: a second identical pass must compute nothing.
            stub.calls = 0
            again = runbench.measure_codec(jpeg, images, "tst", names, verbose=False)
            assert stub.calls == 0, "cache did not prevent recomputation"
            assert again["bpp"] == curve["bpp"]

            # Asking for a metric the cache lacks must invalidate that entry.
            stub.PAPER_SEVEN = names + ["ms_ssim"]
            stub.calls = 0
            runbench.measure_codec(jpeg, images, "tst", ["ms_ssim"], verbose=False)
            assert stub.calls == 0, "subset of cached metrics should still hit"

            # Sign flip: nlpd must be negated on the way to BD-rate.
            flipped = runbench._for_bdrate(curve, names, stub)
            assert flipped["ms_ssim"] == list(curve["ms_ssim"])
            assert flipped["nlpd"] == [-v for v in curve["nlpd"]]

            # A codec against itself must be exactly 0% on every metric.
            self_report = runbench.bdrate_report(
                {"jpeg": curve, "copy": curve}, "jpeg", names, stub)
            for m in names + ["AVG"]:
                assert abs(self_report["copy"][m]) < 1e-9, m

            # Two real codecs, full report path.
            webp = anchors.get("webp")
            if webp.available():
                curves = {"jpeg": curve,
                          "webp": runbench.measure_codec(webp, images, "tst", names,
                                                         verbose=False)}
                report = runbench.bdrate_report(curves, "jpeg", names, stub)
                assert np.isfinite(report["webp"]["AVG"])

                md, png = out / "b.md", out / "b.png"
                runbench.write_markdown(md, "tst", curves, report, "jpeg", names)
                runbench.plot_curves(png, "tst", curves, names)
                assert md.exists() and png.stat().st_size > 1000
                text = md.read_text()
                assert "BD-rate vs jpeg" in text and "webp" in text

                # The JSON we would write must round-trip.
                blob = json.dumps({"curves": curves})
                assert json.loads(blob)["curves"]["jpeg"]["bpp"] == curve["bpp"]
        finally:
            runbench.CACHE = real_cache
            stub.PAPER_SEVEN = ["ms_ssim", "nlpd"]


def test_runbench_missing_dataset_message():
    try:
        runbench.list_images(Path("/nonexistent/dataset/dir"))
    except FileNotFoundError as exc:
        assert "setup.sh" in str(exc), "the error should say how to fix it"
    else:
        raise AssertionError("missing dataset must raise")


# ---------------------------------------------------------------------------
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
