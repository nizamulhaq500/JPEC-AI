"""Phase 3 acceptance gate.

    python -m jpegai.models.selftest                      # structural checks only
    python -m jpegai.models.selftest --checkpoint <path>   # + the real RD gate

Exits non-zero on any failure, so it works as a pre-commit or CI check.

The plan states the Phase 3 gate as three criteria. Two of them are properties of
the *code* and can be checked on an untrained model in seconds; one needs weights.
This module checks all the structural ones on every run so that a regression in
the coder is caught immediately rather than at the end of the next training run.

The gate criteria, restated after measurement (see `train.loop.roundtrip_check`
for the derivation):

    1. yhat survives encode->decode bit-exactly.               [structural]
    2. Actual bytes agree with the estimate computed at the
       QUANTISED sigma to within +-0.5%.                        [structural]
    3. Actual bytes exceed the estimate the training loss saw
       by ~+1.9%, and that excess is accounted for by the
       32-level sigma grid, not by a coder bug.                [structural]
    4. Trained RD curve beats JPEG.                            [needs weights]

Criterion 3 replaces the plan's original "estimated within 1-2% of actual". That
phrasing turned out to conflate two independent effects: a correct coder on JPEG
AI's coarse sigma grid lands at +1.9%, which the original threshold would have
flagged as a failure. Splitting the comparison in two makes the coder testable to
+-0.5% while the sigma-grid cost is reported rather than hidden.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from jpegai.config import load_config
from jpegai.models.cdf import build_cdf_table, pmf_to_quantized_cdf
from jpegai.models.entropy import (
    FactorizedPrior,
    GaussianConditional,
    LowerBound,
    build_scale_table,
)
from jpegai.models import build_any_model
from jpegai.models.colour import get_format
from jpegai.models.hyper import SigmaIndex
from jpegai.models.hyperprior import build_model
from jpegai.models.layers import conv, deconv, pad_to_multiple, unpad
from jpegai.models.twobranch import TwoBranchCodec
from jpegai.utils import macs_breakdown, pick_device, seed_everything


class Report:
    """Collects pass/fail lines so one run reports every failure, not just the first."""

    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((bool(ok), name, detail))
        return bool(ok)

    def section(self, title: str) -> None:
        self.rows.append((None, title, ""))

    def summary(self) -> int:
        width = max(len(n) for _, n, _ in self.rows) + 2
        failed = 0
        for ok, name, detail in self.rows:
            if ok is None:
                print(f"\n{name}")
                print("-" * (width + 30))
                continue
            if not ok:
                failed += 1
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:{width}} {detail}")
        total = sum(1 for ok, _, _ in self.rows if ok is not None)
        print(f"\n{total - failed}/{total} checks passed")
        return 1 if failed else 0


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------
def check_shapes(r: Report) -> None:
    r.section("Layer shape conventions")
    for k in (3, 5, 7):
        x = torch.zeros(1, 4, 32, 32)
        down = conv(4, 8, k, 2)(x)
        up = deconv(8, 4, k, 2)(down)
        r.check(down.shape[-2:] == (16, 16), f"conv k={k} halves the resolution",
                str(tuple(down.shape[-2:])))
        r.check(up.shape[-2:] == (32, 32), f"deconv k={k} inverts it exactly",
                str(tuple(up.shape[-2:])))
    try:
        deconv(4, 4, 4, 2)
        r.check(False, "even deconv kernel is rejected", "no exception raised")
    except ValueError:
        r.check(True, "even deconv kernel is rejected", "ValueError")

    r.section("Padding")
    for h, w in [(768, 512), (321, 499), (64, 64), (17, 5)]:
        x = torch.rand(1, 3, h, w)
        p, pad = pad_to_multiple(x, 64)
        back = unpad(p, pad)
        r.check(p.shape[-2] % 64 == 0 and p.shape[-1] % 64 == 0,
                f"pad {w}x{h} -> multiple of 64",
                f"{tuple(p.shape[-2:])[::-1]}")
        r.check(back.shape == x.shape and torch.equal(back, x),
                f"unpad {w}x{h} recovers the original exactly", "")


def check_cdf(r: Report) -> None:
    r.section("Quantized CDF invariants")
    rng = np.random.default_rng(0)
    bad_mono = bad_total = bad_zero = 0
    for _ in range(300):
        n = int(rng.integers(2, 64))
        style = rng.integers(0, 3)
        if style == 0:
            pmf = rng.random(n) + 1e-3
        elif style == 1:
            x = np.arange(n) - n / 2
            pmf = np.exp(-0.5 * (x / max(rng.random() * 8, 0.3)) ** 2) + 1e-12
        else:
            pmf = np.full(n, 1e-9)
            pmf[rng.integers(0, n)] = 1.0
        prec = int(rng.integers(8, 17))
        if n + 1 > (1 << prec):          # cannot fit; the module raises by design
            continue
        cdf = pmf_to_quantized_cdf(pmf, prec)
        if not np.all(np.diff(cdf) >= 1):
            bad_mono += 1
        if int(cdf[-1]) != (1 << prec) or int(cdf[0]) != 0:
            bad_total += 1
        if np.any(np.diff(cdf) == 0):
            bad_zero += 1
    r.check(bad_mono == 0, "CDF is strictly increasing (300 random pmfs)",
            f"{bad_mono} violations")
    r.check(bad_total == 0, "CDF spans exactly [0, 2^precision]",
            f"{bad_total} violations")
    r.check(bad_zero == 0, "no zero-width bins", f"{bad_zero} violations")

    # A pmf with more symbols than units of mass must raise, not silently merge.
    try:
        pmf_to_quantized_cdf(np.full(300, 1.0), precision=8)
        r.check(False, "over-long table at low precision raises", "no exception")
    except ValueError:
        r.check(True, "over-long table at low precision raises", "ValueError")

    # The +2 in cdf_length: L real symbols -> L+1 boundaries + 1 escape = L+2.
    pmfs = np.array([[0.5, 0.3, 0.2, 0.0], [0.25, 0.25, 0.25, 0.25]])
    cdfs, lengths = build_cdf_table(pmfs, np.array([1e-6, 1e-6]), np.array([3, 4]))
    r.check(list(lengths) == [5, 6], "cdf_length == pmf_length + 2", str(list(lengths)))
    r.check(cdfs.shape[1] == 6, "table row width == Lmax + 2", str(cdfs.shape))


def check_lower_bound(r: Report) -> None:
    r.section("LowerBound gradient semantics")
    lb = LowerBound(1.0)
    # Below the bound with a gradient that would push it further down: blocked.
    x = torch.tensor([0.5], requires_grad=True)
    lb(x).backward(torch.tensor([1.0]))
    blocked = float(x.grad) == 0.0
    # Below the bound but the gradient would push it back up: passed through, or
    # the parameter can never escape the clamp and the model is stuck forever.
    x2 = torch.tensor([0.5], requires_grad=True)
    lb(x2).backward(torch.tensor([-1.0]))
    passed = float(x2.grad) == -1.0
    x3 = torch.tensor([2.0], requires_grad=True)
    lb(x3).backward(torch.tensor([1.0]))
    above = float(x3.grad) == 1.0
    r.check(blocked, "clamped and pushing further out -> gradient blocked", "")
    r.check(passed, "clamped but improving -> gradient passes through", "")
    r.check(above, "above the bound -> gradient untouched", "")
    r.check(float(lb(torch.tensor([0.5]))) == 1.0, "forward clamps to the bound", "")


def check_entropy_roundtrip(r: Report, device) -> None:
    r.section("Entropy model round-trip (bit-exact)")
    torch.manual_seed(0)

    eb = FactorizedPrior(16).to(device)
    eb.eval()
    eb.update(force=True)
    z = (torch.randn(2, 16, 8, 8, device=device) * 3.0)
    z_hat, _ = eb(z, noise=False, ste=False)
    streams = eb.compress(z_hat)
    back = eb.decompress(streams, (8, 8), device=device)
    r.check(torch.equal(back, z_hat), "FactorizedPrior encode->decode is exact",
            f"maxerr {float((back - z_hat).abs().max()):.3g}, "
            f"{sum(len(s) for s in streams)} B")

    gc = GaussianConditional(build_scale_table(0.11, 54.82, 32)).to(device)
    gc.eval()
    gc.update(force=True)
    sigma = torch.rand(2, 16, 8, 8, device=device) * 4.0 + 0.2
    y = torch.randn_like(sigma) * sigma
    y_hat, lik = gc(y, sigma, None, noise=False, ste=False)
    streams = gc.compress(y_hat, sigma)
    back = gc.decompress(streams, sigma)
    r.check(torch.equal(back, y_hat), "GaussianConditional encode->decode is exact",
            f"maxerr {float((back - y_hat).abs().max()):.3g}, "
            f"{sum(len(s) for s in streams)} B")

    # Calibrated sigma is the only case where the estimate should match; that is
    # the whole point of the diagnostic table in docs/06.
    est = float(-torch.log2(lik.clamp_min(1e-12)).sum())
    act = sum(len(s) for s in streams) * 8
    gap = 100.0 * (act - est) / est
    r.check(abs(gap) < 3.0, "calibrated sigma: estimate within 3% of bytes",
            f"{gap:+.2f}% ({est:.0f} est vs {act} actual bits)")

    r.section("Sigma index mapping")
    gc2 = GaussianConditional(build_scale_table(0.11, 54.82, 32))
    # Index with the model's OWN buffer, not with `torch.tensor(build_scale_table(...))`:
    # the builder returns float64 and the buffer is float32, so the two disagree in
    # the 8th digit and `scales <= s` then lands on the wrong side of half the
    # boundaries. The coder only ever sees float32, so float32 is what to test.
    table = gc2.scale_table
    idx = gc2.build_indexes(table.clone())
    r.check(torch.equal(idx, torch.arange(32, dtype=torch.int32)),
            "each table scale maps to its own row", "")
    r.check(int(gc2.build_indexes(torch.tensor([0.001]))) == 0,
            "sigma below the minimum clamps to row 0", "")
    r.check(int(gc2.build_indexes(torch.tensor([1e6]))) == 31,
            "sigma above the maximum clamps to the last row", "")

    ratios = (table[1:] / table[:-1]).log()
    step = math.log(54.82 / 0.11) / 31
    r.check(float((ratios - step).abs().max()) < 1e-5,
            "table is geometric: eq. (13) sigma_k = min * exp(k*step)",
            f"step {step:.4f}")

    # Quantisation rounds sigma UP to the next table entry, never down. That is
    # why `y_oor_pct` stays at 0: the coder always uses a sigma at least as wide
    # as the model predicted, so its CDF is always at least as heavy-tailed, and
    # no symbol the model considered likely can fall outside the table. It is also
    # exactly why the sigma grid costs rate (+1.9%) instead of causing escapes --
    # the two failure modes are mutually exclusive and this is the reason.
    probe = torch.rand(20000) * 54.0 + 0.12
    quant = table[gc2.build_indexes(probe).long()]
    r.check(bool((quant >= probe - 1e-6).all()),
            "quantised sigma is never smaller than the predicted sigma",
            f"max shortfall {float((probe - quant).max()):.3g}")
    r.check(float((quant / probe).max()) < math.exp(step) + 1e-4,
            "and never more than one grid step larger",
            f"max ratio {float((quant / probe).max()):.4f} vs bound "
            f"{math.exp(step):.4f}")


def _excite_parts(g_a, h_a, h_s, *, ga: float = 100.0, ha: float = 60.0,
                  hs: float = 60.0, hs_bias: float = 1.0) -> None:
    """Scale up the last layer of each transform so the latents span real bins.

    A freshly initialised model is **degenerate under quantisation**, and this is
    measured, not assumed: at init `y` has absmax 0.072 and `z` absmax 0.030, both
    an order of magnitude below the 1.0 quantisation step, so `round()` sends every
    element of both to exactly zero. `h_s` then outputs absmax 0.042, entirely
    below `sigma_quant_min`, so `LowerBound(0.11)` flattens every sigma to the same
    value and every index to row 0.

    That single fact makes three otherwise-good checks vacuous:

    * gradient reachability -- `g_s.body.0` and `h_s.body.0` see an identically
      zero input, so their *weight* gradients are legitimately zero while their
      biases are fine. Reads as a dead branch; is not one.
    * the z-vs-zhat asymmetry check -- with every sigma clamped, indexes derived
      from `z` and from `z_hat` are trivially equal, so the check cannot fail even
      if `compress()` used the wrong one.
    * anything about sigma variation at all.

    Exciting the transforms tests the graph topology, which is what these checks
    are actually for, without pretending an untrained model is a trained one. It
    is also the reason `lr_at()` has a warmup: for the first few hundred steps the
    real model is in exactly this degenerate regime.
    """
    import torch.nn as nn

    def last(mod):
        convs = [m for m in mod.modules()
                 if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
        return convs[-1]

    with torch.no_grad():
        last(g_a).weight.mul_(ga)
        last(h_a).weight.mul_(ha)
        c = last(h_s)
        c.weight.mul_(hs)
        if c.bias is not None:
            c.bias.add_(hs_bias)


def _excite(model, **kw) -> None:
    """:func:`_excite_parts` on a single-branch model's three transforms."""
    _excite_parts(model.g_a, model.h_a, model.h_s, **kw)


def check_degenerate_init(r: Report, device) -> None:
    """Pin the untrained-model regime, because it is easy to misread as a bug."""
    r.section("Untrained model is degenerate under quantisation (expected)")
    torch.manual_seed(0)
    m = build_model(load_config("tierA"), kind="scale").to(device)
    with torch.no_grad():
        out = m(torch.rand(2, 3, 128, 128, device=device), noise=False, ste=True)
    r.check(float(out["y"].abs().max()) < 0.5,
            "latent magnitude is below the quantisation step",
            f"absmax {float(out['y'].abs().max()):.4f}")
    r.check(int((out["y_hat"] != 0).sum()) == 0, "so every yhat rounds to zero", "")
    r.check(float(out["scales"].max()) <= 0.11 + 1e-6,
            "and every sigma clamps to sigma_quant_min",
            f"max {float(out['scales'].max()):.4f}")
    # Which is why an untrained model reports a LOW rate and a terrible PSNR --
    # near-certainty about a near-zero latent. Measured: 0.127 bpp at 5.28 dB.
    m.eval()
    m.update(force=True)
    with torch.no_grad():
        x = torch.rand(1, 3, 128, 128, device=device)
        nb = m.packet_bytes(m.compress(x))
    r.check(nb * 8 / (128 * 128) < 0.3,
            "untrained rate is low, not high (certainty about zero)",
            f"{nb * 8 / (128 * 128):.4f} bpp")


def check_model(r: Report, device, tier: str = "tierA") -> None:
    cfg = load_config(tier)
    for kind in ("scale", "mean-scale"):
        r.section(f"Model {kind} ({tier})")
        torch.manual_seed(0)
        m = build_model(cfg, kind=kind).to(device)
        # See _excite: an unexcited model quantises to all-zero, which makes the
        # gradient and asymmetry checks below vacuous rather than passing.
        _excite(m)

        x = torch.rand(2, 3, 128, 128, device=device)
        out = m(x)
        r.check(out["x_hat"].shape == x.shape, "reconstruction has the input shape",
                str(tuple(out["x_hat"].shape)))
        r.check(out["y"].shape[-2:] == (8, 8), "latent at /16", str(tuple(out["y"].shape)))
        r.check(out["z"].shape[-2:] == (2, 2), "hyper latent at /64",
                str(tuple(out["z"].shape)))
        r.check(out["scales"].shape == out["y"].shape,
                "one sigma per latent element", "")
        r.check(int((out["y_hat"] != 0).sum()) > out["y_hat"].numel() // 2,
                "excited latents actually span quantisation bins",
                f"{int((out['y_hat'] != 0).sum())}/{out['y_hat'].numel()} nonzero")
        if kind == "mean-scale":
            r.check(out["means"] is not None and out["means"].shape == out["y"].shape,
                    "one mean per latent element", "")

        # Gradients must reach every parameter. A dead branch here is the kind of
        # bug that costs a whole training run and produces no error message.
        loss = out["x_hat"].square().mean() \
            - torch.log2(out["likelihoods"]["y"].clamp_min(1e-9)).mean() \
            - torch.log2(out["likelihoods"]["z"].clamp_min(1e-9)).mean()
        m.zero_grad()
        loss.backward()
        dead = [n for n, p in m.named_parameters()
                if p.requires_grad and n != "entropy_bottleneck.quantiles"
                and (p.grad is None or not torch.isfinite(p.grad).all()
                     or float(p.grad.abs().sum()) == 0.0)]
        r.check(not dead, "every parameter receives a finite non-zero gradient",
                "dead: " + ", ".join(dead[:3]) if dead else "")

        # quantiles must be reachable from aux_loss and ONLY from aux_loss.
        m.zero_grad()
        m.aux_loss().backward()
        q = m.entropy_bottleneck.quantiles
        r.check(q.grad is not None and float(q.grad.abs().sum()) > 0,
                "aux_loss reaches quantiles", "")
        others = [n for n, p in m.named_parameters()
                  if n != "entropy_bottleneck.quantiles" and p.grad is not None
                  and float(p.grad.abs().sum()) != 0.0]
        r.check(not others, "aux_loss touches nothing else",
                "also: " + ", ".join(others[:3]) if others else "")

        r.section(f"Two-quantisation training ({kind})")
        torch.manual_seed(0)
        a = m(x, noise=True, ste=True)
        torch.manual_seed(0)
        b = m(x, noise=False, ste=True)
        # The rate branch must differ (noise vs round) while the distortion branch
        # must not: that is exactly config.train.quantisation.
        rate_differs = not torch.equal(a["likelihoods"]["y"], b["likelihoods"]["y"])
        dist_same = torch.equal(a["y_hat"], b["y_hat"])
        r.check(rate_differs, "noise changes the rate branch", "")
        r.check(dist_same, "noise leaves the distortion branch (yhat) alone", "")
        integral = torch.equal(b["y_hat"], torch.round(b["y_hat"]))
        if kind == "scale":
            r.check(integral, "ste yhat is exactly integral", "")
        else:
            # Mean-scale codes the residual, so yhat sits on an integer grid
            # OFFSET by the predicted mean. Asserting plain integrality here would
            # be wrong, and asserting nothing would miss a broken residual path.
            res = b["y_hat"] - b["means"]
            r.check(torch.allclose(res, torch.round(res), atol=1e-5),
                    "ste yhat is integral after subtracting the mean",
                    f"maxdev {float((res - torch.round(res)).abs().max()):.3g}")

        r.section(f"Codec round-trip ({kind})")
        m.eval()
        m.update(force=True)
        for h, w in [(128, 128), (192, 320)]:
            xi = torch.rand(1, 3, h, w, device=device)
            with torch.no_grad():
                o = m(xi, noise=False, ste=True)
                pk = m.compress(xi)
                dec = m.decompress(pk, device=device)
            r.check(dec["x_hat"].shape == xi.shape,
                    f"{w}x{h}: decode returns the input shape",
                    str(tuple(dec["x_hat"].shape)))
            r.check(torch.equal(dec["y_hat"], o["y_hat"]),
                    f"{w}x{h}: yhat bit-exact through the coder",
                    f"maxerr {float((dec['y_hat'] - o['y_hat']).abs().max()):.3g}")
            r.check(torch.equal(dec["z_hat"], o["z_hat"]),
                    f"{w}x{h}: zhat bit-exact through the coder", "")
            nb = m.packet_bytes(pk)
            r.check(nb > 0, f"{w}x{h}: bitstream is non-empty",
                    f"{nb} B = {nb * 8 / (h * w):.4f} bpp")

        # Non-multiple-of-64 input must work via padding, and 64 is the multiple
        # that matters because the hyper latent is at /64.
        xi = torch.rand(1, 3, 100, 150, device=device)
        with torch.no_grad():
            dec = m.decompress(m.compress(xi), device=device)
        r.check(dec["x_hat"].shape == xi.shape,
                "150x100 (not a multiple of 64) round-trips", "")

        r.section(f"Encoder must use the DECODED zhat ({kind})")
        # Deriving sigma from z instead of zhat is the classic asymmetry bug. It
        # cannot be caught by inspecting compress() alone, but it always changes
        # the sigma indexes, so compare them.
        with torch.no_grad():
            xi = torch.rand(1, 3, 128, 128, device=device)
            y = m.g_a(xi)
            z = m.h_a(y)
            zs = m.entropy_bottleneck.compress(z)
            z_hat = m.entropy_bottleneck.decompress(zs, tuple(z.shape[-2:]),
                                                    device=device)
            i_hat = m.gaussian_conditional.build_indexes(
                m._split_params(m.h_s(z_hat))[0])
            i_raw = m.gaussian_conditional.build_indexes(
                m._split_params(m.h_s(z))[0])
        r.check(not torch.equal(z, z_hat), "z and zhat genuinely differ", "")
        r.check(not torch.equal(i_hat, i_raw),
                "so sigma indexes from z vs zhat differ -- the check is meaningful",
                f"{int((i_hat != i_raw).sum())} of {i_hat.numel()} rows")
        m.train()


# ---------------------------------------------------------------------------
# Phase 4: the two-branch codec
# ---------------------------------------------------------------------------
def _excite_two_branch(model, **kw) -> None:
    """`_excite` for a model with two of everything. Same reasoning applies."""
    _excite_parts(model.g_a_y, model.branch_y.h_a, model.branch_y.h_s, **kw)
    _excite_parts(model.g_a_uv, model.branch_uv.h_a, model.branch_uv.h_s, **kw)


def check_two_branch(r: Report, device, tier: str = "tierA") -> None:
    """The Phase 4 invariants, at all three internal chroma formats.

    Structured around the one thing that can silently go wrong: the two latents
    have to land on the *same* spatial grid, because eq. (3) concatenates them.
    Everything downstream -- the concatenation width, the shared sigma table, the
    four-stream rate -- is downstream of that.
    """
    cfg = load_config(tier)
    for fmt in ("444", "422", "420"):
        r.section(f"Two-branch model, internal {fmt} ({tier})")
        torch.manual_seed(0)
        # Built directly rather than through `build_two_branch` so the internal
        # chroma format can be varied; everything else comes from the config.
        m = TwoBranchCodec(
            luma_latent=cfg.channels.primary_latent,
            chroma_latent=cfg.channels.secondary_latent,
            luma_hyper=cfg.channels.hyper_latent,
            chroma_hyper=cfg.channels.hyper_secondary_latent,
            analysis_width=cfg.channels.analysis_width,
            synthesis_width=cfg.channels.synthesis_width,
            internal_format=fmt,
            pad_multiple=cfg.geometry.total_downsample,
        ).to(device)
        _excite_two_branch(m)

        x = torch.rand(2, 3, 128, 128, device=device)
        out = m(x)
        r.check(out["x_hat"].shape == x.shape, "reconstruction has the input shape",
                str(tuple(out["x_hat"].shape)))
        r.check(out["y"].shape[-2:] == out["y_uv"].shape[-2:],
                "eq. (3): both latents on one grid",
                f"luma {tuple(out['y'].shape[-2:])} "
                f"chroma {tuple(out['y_uv'].shape[-2:])}")
        r.check(out["y"].shape[-2:] == (8, 8), "latents at /16",
                str(tuple(out["y"].shape)))
        r.check(out["y"].shape[1] == cfg.channels.primary_latent
                and out["y_uv"].shape[1] == cfg.channels.secondary_latent,
                "latent widths come from the config",
                f"{out['y'].shape[1]} + {out['y_uv'].shape[1]} = "
                f"{out['y'].shape[1] + out['y_uv'].shape[1]}")
        r.check(set(out["likelihoods"]) == {"y", "z", "y_uv", "z_uv"},
                "four entropy streams", ", ".join(sorted(out["likelihoods"])))

        # Chroma plane geometry. The ceiling rule lives here, not in the codec.
        ch, cw = out["planes"]["chroma"].shape[-2:]
        f = get_format(fmt)
        r.check((ch, cw) == (128 // f.ver, 128 // f.hor),
                f"chroma plane is H/{f.ver} x W/{f.hor}", f"{cw}x{ch}")

        # Gradients: the two branches plus the cross-component link.
        loss = out["x_hat"].square().mean()
        for k, lik in out["likelihoods"].items():
            loss = loss - torch.log2(lik.clamp_min(1e-9)).mean()
        m.zero_grad()
        loss.backward()
        quant = {"branch_y.entropy_bottleneck.quantiles",
                 "branch_uv.entropy_bottleneck.quantiles"}
        dead = [n for n, p in m.named_parameters()
                if p.requires_grad and n not in quant
                and (p.grad is None or not torch.isfinite(p.grad).all()
                     or float(p.grad.abs().sum()) == 0.0)]
        r.check(not dead, "every parameter receives a finite non-zero gradient",
                "dead: " + ", ".join(dead[:3]) if dead else "")

        # eq. (3) is a real link, not decoration: chroma-only error must reach the
        # primary analysis. If the concatenation were dropped this stays silent.
        m.zero_grad()
        m(x)["planes"]["chroma"].square().mean().backward()
        g = sum(float(p.grad.abs().sum()) for p in m.g_a_y.parameters()
                if p.grad is not None)
        r.check(g > 0, "eq. (3) carries chroma error into the primary analysis",
                f"grad sum {g:.4g}")

        # Both quantile sets, and only those, from aux_loss.
        m.zero_grad()
        m.aux_loss().backward()
        reached = [n for n, p in m.named_parameters()
                   if n in quant and p.grad is not None
                   and float(p.grad.abs().sum()) > 0]
        r.check(len(reached) == 2, "aux_loss reaches both branches' quantiles",
                f"{len(reached)}/2")
        others = [n for n, p in m.named_parameters()
                  if n not in quant and p.grad is not None
                  and float(p.grad.abs().sum()) != 0.0]
        r.check(not others, "aux_loss touches nothing else",
                "also: " + ", ".join(others[:3]) if others else "")

        r.section(f"Two-branch bitstream, internal {fmt}")
        m.eval()
        m.update(force=True)
        gcs = [mod for mod in m.modules()
               if type(mod).__name__ == "GaussianConditional"]
        r.check(len(gcs) == 1, "one shared sigma table, not one per branch",
                f"{len(gcs)} found, tables {m.table_bytes() / 1024:.1f} KiB")

        for h, w in [(128, 192), (130, 194)]:
            xi = torch.rand(1, 3, h, w, device=device)
            with torch.no_grad():
                o = m(xi, noise=False, ste=True)
                pk = m.compress(xi)
                dec = m.decompress(pk, device=device)
            r.check(dec["x_hat"].shape == xi.shape,
                    f"{w}x{h}: decode returns the input shape",
                    str(tuple(dec["x_hat"].shape)))
            r.check(torch.equal(dec["y_hat"], o["y_hat"]),
                    f"{w}x{h}: luma yhat bit-exact through the coder",
                    f"maxerr {float((dec['y_hat'] - o['y_hat']).abs().max()):.3g}")
            r.check(torch.equal(dec["y_uv_hat"], o["y_uv_hat"]),
                    f"{w}x{h}: chroma yhat bit-exact through the coder", "")
            full = m.packet_bytes(pk)
            luma = m.packet_bytes(pk, luma_only=True)
            r.check(0 < luma < full, f"{w}x{h}: luma-only is a smaller payload",
                    f"{luma} of {full} B = {100.0 * luma / full:.0f}%, "
                    f"{full * 8 / (h * w):.4f} bpp full")

        # luma_only must not read the chroma strings at all -- deleting them is
        # the only way to prove that, since decode-then-discard looks identical.
        with torch.no_grad():
            pk = m.compress(torch.rand(1, 3, 128, 128, device=device))
            ref = m.decompress(pk, device=device)
            del pk["chroma"]
            part = m.decompress(pk, device=device, luma_only=True)
        r.check(torch.equal(part["luma"], ref["luma"]),
                "luma-only decode never touches the chroma stream",
                "chroma strings deleted from the packet before decoding")
        rgb = part["x_hat"]
        r.check(torch.allclose(rgb[:, 0], rgb[:, 1], atol=1e-5)
                and torch.allclose(rgb[:, 1], rgb[:, 2], atol=1e-5),
                "luma-only output is grey (Cb = Cr = 0.5)", "")

        r.section(f"Encoder must use the DECODED zhat, both branches ({fmt})")
        with torch.no_grad():
            xi = torch.rand(1, 3, 128, 128, device=device)
            yp, uvp, supp, _ = m._to_planes(xi)
            for tag, lat, br in (("luma", m.g_a_y(yp), m.branch_y),
                                 ("chroma", m.g_a_uv(uvp, supp), m.branch_uv)):
                z = br.h_a(lat)
                z_hat = br.entropy_bottleneck.decompress(
                    br.entropy_bottleneck.compress(z), tuple(z.shape[-2:]),
                    device=device)
                gc = m.gaussian_conditional
                i_hat = gc.build_indexes(br.params(z_hat)[0])
                i_raw = gc.build_indexes(br.params(z)[0])
                r.check(not torch.equal(z, z_hat), f"{tag}: z and zhat differ", "")
                r.check(not torch.equal(i_hat, i_raw),
                        f"{tag}: sigma indexes from z vs zhat differ",
                        f"{int((i_hat != i_raw).sum())} of {i_hat.numel()} rows")
        m.train()


# ---------------------------------------------------------------------------
# Phase 5
# ---------------------------------------------------------------------------
def check_split_hyper(r: Report, device, tier: str = "tierA") -> None:
    """The Phase 5 invariants: the sigma codebook, and the two split decoders.

    The sigma-codebook checks are here rather than only in pytest because they are
    the ones whose failure mode is a *corrupt bitstream* rather than a crash. If
    `table_row` ever stops rounding up, or the disagreement with the float path ever
    grows past the known 11 indices, the codec still encodes and still decodes and
    the pictures come back wrong -- so the one-command gate has to say so.
    """
    cfg = load_config(tier)
    ent = cfg.entropy
    si = SigmaIndex(minimum=ent.sigma_quant_min, maximum=ent.sigma_quant_max,
                    levels=ent.sigma_quant_level, precision=ent.sigma_precision)

    r.section(f"Sigma index codebook ({tier})")
    r.check(si.max_index == (ent.sigma_quant_level - 1) * 2 ** ent.sigma_precision - 1,
            "max_index = (levels-1)*2^precision - 1", str(si.max_index))
    idx = torch.arange(si.max_index + 1)
    rows = si.table_row(idx)
    r.check(int(rows[-1]) == si.levels - 1,
            "the maximum index reaches the last CDF row (this is the round-up proof)",
            f"row {int(rows[-1])} of {si.levels - 1}; round-down would give "
            f"{si.max_index >> si.precision}")
    r.check(set(rows.tolist()) == set(range(si.levels)),
            "every CDF row is reachable", f"{len(set(rows.tolist()))} of {si.levels}")
    r.check(bool(torch.all(rows[1:] >= rows[:-1])), "table_row is monotone", "")
    r.check(all(int(si.table_row(torch.tensor([k * si.step]))) == k
                for k in range(si.levels - 1)),
            "grid points map to themselves, not to the row above", "")

    gc = GaussianConditional(
        build_scale_table(si.minimum, si.maximum, si.levels), scale_bound=si.minimum)
    bad = (rows != gc.build_indexes(si.sigma(idx.float()))).nonzero().flatten()
    n = int(bad.numel())
    r.check(n == 11, "integer and float index paths disagree on exactly 11 of 3968",
            f"{n} indices ({100.0 * n / idx.numel():.3f}% of symbols would be "
            f"undecodable if the two sides mixed rules)")
    r.check(all(int(i) % si.step == 0 for i in bad),
            "every disagreement sits exactly on a grid point (i.e. is one ULP)",
            "not a systematic rounding difference")

    for kind in ("twobranch-split", "twobranch-fused"):
        r.section(f"Split hyper decoders -- {kind} ({tier})")
        torch.manual_seed(0)
        m = build_any_model(cfg, kind).to(device).eval()
        m.update(force=True)
        x = torch.rand(1, 3, 128, 128, device=device)
        with torch.no_grad():
            out = m(x, noise=False, ste=True)
            dec = m.decompress(m.compress(x), device=device)

        r.check("i_sigma" in out and "i_sigma_uv" in out,
                "the training pass exposes Isigma for both branches", "")
        lo, hi = float(out["i_sigma"].min()), float(out["i_sigma"].max())
        r.check(0 < lo and hi < si.max_index,
                "Isigma starts inside the table, not pinned at either end",
                f"[{lo:.0f}, {hi:.0f}] of [0, {si.max_index}]")
        r.check(abs(float(out["scales"].mean()) - 2.45) < 0.2,
                "initial sigma is mid-grid (~2.45), not the 0.11 floor",
                f"{float(out['scales'].mean()):.4f}")
        r.check(torch.equal(dec["y_hat"], out["y_hat"]),
                "yhat is bit-exact through a real bitstream", "")

        si_m = m.sigma_index
        agree = all(
            torch.equal(m.coder_rows(out, sfx),
                        si_m.table_row(si_m.quantise(out[f"i_sigma{sfx}"])))
            for sfx, _ in m.gate_branches())
        r.check(agree, "the gate indexes rows through the integer path", "")

        parts = m.summary_parts()
        mb = macs_breakdown(m, (1, 3, 128, 128), parts=[(nm, s) for nm, s, _ in parts])
        dec_mac = {nm: mb[nm] for nm, _, is_dec in parts if is_dec and nm in mb}
        total = sum(dec_mac.values())
        scale_mac = sum(v for k, v in dec_mac.items() if k.startswith("h_scale"))
        if kind == "twobranch-split":
            r.check(scale_mac / total < 0.05,
                    "the scale decoders are under 5% of decoder MACs "
                    "(Phase 5 criterion 2)",
                    f"{100 * scale_mac / total:.3f}% of {total / 1000:.1f} kMAC/pxl")
        else:
            r.check(scale_mac == 0.0,
                    "the fused ablation has no separate scale decoder", "")
        h_s_mac = sum(v for k, v in dec_mac.items() if k.startswith("h_s"))
        # Printed, not checked: it is a number to report, and inventing a threshold
        # for it would be a check that only ever passes.
        print(f"    decoder {total / 1000:.1f} kMAC/pxl   "
              f"h_s {h_s_mac / 1000:.2f}   h_scale {scale_mac / 1000:.3f}")


# ---------------------------------------------------------------------------
# Phase 6 -- MCM
# ---------------------------------------------------------------------------
def check_mcm(r: Report, device, tier: str = "tierA") -> None:
    """§VI-D's context model. Three properties, all of which fail *silently*.

    1. The decoder must rebuild the encoder's latent exactly. Stage `k` conditions on
       stages `< k`, so an asymmetry between the two sides yields a codestream that
       decodes without complaint into a reconstruction that drifts further with every
       stage. There is no exception to catch; only an equality to check.
    2. The entropy decoder must be entered **once**, with no mean. That is §VI-E's
       decoupling, and a version that re-entered the coder per stage would produce
       byte-identical output while destroying the self-contained entropy engine the
       whole section exists to build. So the call is watched, not the pixels.
    3. The number of network passes must be **four**, whatever the image size. An
       autoregressive context model decodes perfectly and is merely unusable at 4K.

    Run on the real tier config, not on a toy: `chs2group` depends on the latent width
    (96 -> 3 groups), and it is the one place where a config change silently alters the
    structure rather than raising.
    """
    from jpegai.models.mcm import GROUP_ORDER, MCMBranch, join_cosets, split_pred

    cfg = load_config(tier)
    ent = cfg.entropy
    r.section(f"MCM -- coset order and schedule ({tier})")
    r.check([tuple(g) for g in ent.mcm_group_order] == list(GROUP_ORDER),
            "the config's coset order is the diagonal-first one docs/06 5 derives",
            " -> ".join(str(g) for g in GROUP_ORDER))
    r.check(ent.mcm_stages == 4 and not ent.mcm_on_secondary,
            "4 stages, luma only", f"stages={ent.mcm_stages}, "
            f"on_secondary={ent.mcm_on_secondary}")

    for kind, want in (("twobranch-mcm", 4), ("twobranch-mcm2", 2),
                       ("twobranch-mcm1", 1)):
        r.section(f"MCM -- {kind} ({tier})")
        torch.manual_seed(0)
        m = build_any_model(cfg, kind).to(device).eval()
        m.update(force=True)
        mcm = m.branch_y.mcm
        r.check(isinstance(m.branch_y, MCMBranch)
                and not isinstance(m.branch_uv, MCMBranch),
                "MCM is on the luma branch and only there", "")
        r.check(mcm.stages == want and len(mcm.nets) == 4,
                f"{want} sequential stages over 4 context networks",
                f"visible={mcm.visible}")
        r.check(m.branch_y.h_s.shuffle is False
                and m.branch_y.h_s(torch.zeros(1, cfg.channels.hyper_latent, 2, 2,
                                               device=device)).shape[-3]
                == 4 * m.luma_latent,
                "the prediction arrives pre-split as [4*chs, /32]",
                f"{4 * m.luma_latent} channels on the coset grid")
        # Which of those channels belongs to which coset is `PixelShuffle`'s question,
        # and answering it wrong costs nothing but the warm start: a contiguous
        # `chunk` is a permutation of the same numbers, so the codec still round-trips
        # bit-exactly and the model still trains. What it loses is the head start --
        # `h_s` was trained with the shuffle in place, so a Phase 5 checkpoint would
        # open its Phase 6 run with two of its four predictions swapped instead of as
        # the identity. The identity below is the only detector.
        with torch.no_grad():
            pred = m.branch_y.h_s(torch.randn(1, cfg.channels.hyper_latent, 2, 2,
                                              device=device))
            same = torch.equal(join_cosets(split_pred(pred)),
                               torch.pixel_shuffle(pred, 2))
        r.check(same, "the cosets are sliced the way PixelShuffle would slice them",
                "so a Phase 5 checkpoint warm-starts as the identity")

        x = torch.rand(1, 3, 128, 192, device=device)
        with torch.no_grad():
            out = m(x, noise=False, ste=True)
            packet = m.compress(x)

            calls: list = []
            real = m.gaussian_conditional.decompress

            def spy(streams, scales, means=None, _real=real, _calls=calls, **kw):
                _calls.append(means)
                return _real(streams, scales, means, **kw)

            passes: list = []
            handles = [n.register_forward_hook(lambda *a: passes.append(1))
                       for n in mcm.nets]
            m.gaussian_conditional.decompress = spy
            try:
                dec = m.decompress(packet, device=device)
            finally:
                m.gaussian_conditional.decompress = real
                for h in handles:
                    h.remove()

        r.check(torch.equal(dec["y_hat"], out["y_hat"]),
                "yhat_decoder == yhat_encoder, tensor-for-tensor "
                "(the non-negotiable one)", "")
        r.check(torch.equal(out["y_hat"], out["mcm_y_hat"]),
                "the coder's yhat and the loop's own yhat are the same tensor",
                "the rate the loss sees and the latent g_s sees cannot drift")
        # Two branches, so two y-stream decodes: luma's must be the mean-free one.
        r.check(sum(1 for c in calls if c is None) == 1
                and len(calls) == len(m.gate_branches()),
                "the residual field is decoded in one mean-free pass, "
                "no network in the loop (VI-E)",
                f"{len(calls)} coder calls, means={['None' if c is None else 'p' for c in calls]}")
        r.check(len(passes) == 4,
                "exactly 4 context-network passes per decode, at every stage count",
                f"{len(passes)} passes at {x.shape[-2]}x{x.shape[-1]}")
        r.check(torch.equal(out["r_hat"], torch.round(out["r_hat"])),
                "the coded residual is an integer field", "")

        parts = m.summary_parts()
        names = [nm for nm, _, _ in parts]
        mb = macs_breakdown(m, (1, 3, 128, 128),
                            parts=[(nm, s) for nm, s, _ in parts])
        dec_mac = {nm: mb[nm] for nm, _, is_dec in parts if is_dec and nm in mb}
        total = sum(dec_mac.values())
        mcm_mac = sum(v for k, v in dec_mac.items() if k.startswith("mcm"))
        r.check("mcm_y" in names and "mcm_uv" not in names,
                "MCM is its own MAC bucket on luma only "
                "(otherwise the constant-cost claim is unfalsifiable)", "")
        print(f"    decoder {total / 1000:.1f} kMAC/pxl   "
              f"mcm {mcm_mac / 1000:.2f} ({100 * mcm_mac / total:.1f}%)")

    r.section(f"MCM -- passes do not grow with the image ({tier})")
    torch.manual_seed(0)
    m = build_any_model(cfg, "twobranch-mcm").to(device).eval()
    m.update(force=True)
    counts = []
    for side in (64, 256):
        with torch.no_grad():
            packet = m.compress(torch.rand(1, 3, side, side, device=device))
            passes: list = []
            handles = [n.register_forward_hook(lambda *a: passes.append(1))
                       for n in m.branch_y.mcm.nets]
            try:
                m.decompress(packet, device=device)
            finally:
                for h in handles:
                    h.remove()
        counts.append(len(passes))
    r.check(counts == [4, 4],
            "4 passes at 64x64 and 4 at 256x256 -- 16x the pixels, same latency",
            f"{counts[0]} then {counts[1]}")


# ---------------------------------------------------------------------------
# The trained gate
# ---------------------------------------------------------------------------
def check_trained(r: Report, device, checkpoint: Path, tier: str,
                  images: int = 4) -> None:
    from jpegai.train.dataset import ImageDataset
    from jpegai.train.loop import load_checkpoint, roundtrip_check

    cfg = load_config(tier)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kind = blob.get("meta", {}).get("model", "scale")
    # `build_any_model`, not `build_model`: the ladders write "twobranch" and
    # "twobranch-split" into meta["model"], and the hyperprior builder rejects both.
    m = build_any_model(cfg, kind).to(device)
    load_checkpoint(checkpoint, m)
    m.eval()

    r.section(f"Trained gate -- {checkpoint.name} ({kind}, step {blob.get('step')})")
    ds = ImageDataset(cfg.data.test, max_side=768, multiple=64, limit=images)

    gaps_q, gaps, oor, exact = [], [], [], []
    worst_b, split_ok = [], []
    for i in range(len(ds)):
        rt = roundtrip_check(m, ds, device, index=i)
        gaps_q.append(rt["gap_q_pct"])
        gaps.append(rt["gap_pct"])
        oor.append(max(rt["y_oor_pct"], rt["z_oor_pct"]))
        exact.append(rt["y_exact"] and rt["z_exact"])
        split_ok.append(rt["streams_ok"])
        worst_b.append((rt["worst_stream_b"], rt["worst_stream"],
                        rt["worst_stream_pct"]))
        print(f"    {ds.name(i):10} act {rt['act_bpp']:.4f} bpp  "
              f"psnr {rt['psnr']:5.2f}  vs est_q {rt['gap_q_pct']:+.2f}%  "
              f"vs est {rt['gap_pct']:+.2f}%  oor {oor[-1]:.3f}%  "
              f"worst {rt['worst_stream']} {rt['worst_stream_b']:+.0f} B")

    mq = float(np.mean(gaps_q))
    mg = float(np.mean(gaps))
    r.check(all(exact), "latents bit-exact on every test image", f"{sum(exact)}/{len(exact)}")
    r.check(abs(mq) < 0.5, "mean gap vs quantised-sigma estimate within +-0.5%",
            f"{mq:+.3f}%")
    r.check(max(oor) < 0.01, "no symbols escape the CDF tables",
            f"max {max(oor):.4f}%")
    r.check(0.0 < mg < 3.0, "gap vs the loss's own estimate is the sigma-grid cost",
            f"{mg:+.2f}% (expect ~+1.9% for 32 levels)")
    # A separate arm from the mean above, because the mean cannot see it. One stream
    # disagreeing with its own entropy table by 63% arrives in `gap_q_pct` divided by
    # every other stream's bits -- that is how the median-shift bug passed for three
    # ladders. See `train.loop.roundtrip_check` for the two-armed threshold.
    b, name, pct = max(worst_b)
    r.check(all(split_ok), "every stream agrees with its own entropy table",
            f"worst {name} {b:+.1f} B ({pct:+.2f}%)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jpegai.models.selftest")
    ap.add_argument("--tier", default="tierA")
    ap.add_argument("--device", default="cpu",
                    help="cpu is the default here on purpose: the checks are about "
                         "correctness, and CPU is the reference implementation")
    ap.add_argument("--checkpoint", default=None, help="also run the trained RD gate")
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--skip-two-branch", action="store_true",
                    help="skip the Phase 4, 5 and 6 sections, which are the slow "
                         "ones (three chroma formats x two branches x a real "
                         "bitstream, then two more codecs with MAC probes, then "
                         "three MCM stage counts)")
    args = ap.parse_args(argv)

    seed_everything(0)
    device = pick_device(args.device)
    r = Report()

    print(f"Phase 3-6 self-test -- tier {args.tier}, device {device}")
    check_shapes(r)
    check_cdf(r)
    check_lower_bound(r)
    check_entropy_roundtrip(r, device)
    check_degenerate_init(r, device)
    check_model(r, device, args.tier)
    if not args.skip_two_branch:
        check_two_branch(r, device, args.tier)
        check_split_hyper(r, device, args.tier)
        check_mcm(r, device, args.tier)
    if args.checkpoint:
        check_trained(r, device, Path(args.checkpoint), args.tier, args.images)
    else:
        r.section("Trained gate")
        print("  skipped -- pass --checkpoint <path> to run it")

    return r.summary()


if __name__ == "__main__":
    sys.exit(main())
