"""The seven JPEG AI evaluation metrics, plus PSNR.

Paper reference: section VII-A. The metric set was chosen in WG1 N85013 (Nov 2019)
because PSNR alone is a poor predictor of human judgement, and learned codecs
exploit exactly that gap.

    MS-SSIM      structural, multiscale
    VIF          information-theoretic (mutual information through an HVS channel)
    FSIM         feature-based (phase congruency + gradient magnitude)
    VMAF         Netflix's learned fusion metric
    NLPD         normalised Laplacian pyramid distance
    PSNR-HVS     DCT-domain PSNR with contrast masking
    IW-SSIM      information-weighted SSIM

We add PSNR-Y/U/V because the paper's EFE post-filter discussion (section VI-M) is
justified entirely by chroma PSNR, which is *not* one of the seven.

**The definitions here follow `ref/jpeg-ai-qaf/metrics.py`**, the WG1 Objective
Quality Assessment Framework that produced the paper's tables. Three of its
conventions are not guessable from the paper and change every number:

1. **Six of the seven are luma-only.** Only FSIM sees colour (``channels=3``).
   MS-SSIM, VIF, NLPD, IW-SSIM, PSNR-HVS and VMAF all run on Y alone. Running
   them on RGB -- the obvious reading -- inflates scores and makes BD-rate
   incomparable with the paper.
2. **The internal precision is 10-bit**, not the source's 8 (``MetricParent``
   defaults ``bits=10, max_val=1023``). 8-bit input is scaled up, not rounded down.
3. **Each metric wants its own input range**: MS-SSIM at 0..1023, VIF/NLPD/
   PSNR-HVS in [0,1], IW-SSIM in [0,255]. PSNR-HVS additionally replicate-pads
   to a multiple of 8 before the DCT.

Also: the seventh metric is **PSNR-HVS**, not PSNR-HVS-M. The reference computes
both and reports ``p_hvs`` (``qaf/metrics.py::PSNR_HVS.calc`` returns the first of
the pair). We keep PSNR-HVS-M available since we compute it anyway.

Availability is uneven across libraries, so every metric goes through a registry
with an explicit backend and an `available` flag. Nothing silently returns a wrong
number: a metric either works or reports why it doesn't. `describe()` prints the
current state -- run it right after setup.

Where the reference's exact backend is not installed we substitute and say so in
`BACKEND_NOTES`; those metrics are directionally right but not bit-comparable
with the paper.

Convention: all inputs are float tensors in [0, 1], shape [N, C, H, W], RGB.
Metrics are computed on the *final output* (post-clip, at output bit depth), per
paper eq. (6) -- see `quantise_to_bitdepth`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional backends. Import failures are recorded, never raised at import time.
# ---------------------------------------------------------------------------
_BACKEND_ERR: dict[str, str] = {}

try:
    import piq  # MS-SSIM, VIF, FSIM, IW-SSIM
except Exception as exc:  # pragma: no cover
    piq = None
    _BACKEND_ERR["piq"] = str(exc)

try:
    import pyiqa  # NLPD
except Exception as exc:  # pragma: no cover
    pyiqa = None
    _BACKEND_ERR["pyiqa"] = str(exc)

try:
    from pytorch_msssim import ms_ssim as _pt_ms_ssim  # the reference's MS-SSIM
except Exception as exc:  # pragma: no cover
    _pt_ms_ssim = None
    _BACKEND_ERR["pytorch_msssim"] = str(exc)

try:
    from psnr_hvsm import psnr_hvs_hvsm as _ref_psnr_hvs  # the reference's PSNR-HVS
except Exception as exc:  # pragma: no cover
    _ref_psnr_hvs = None
    _BACKEND_ERR["psnr_hvsm"] = str(exc)

#: Where we deviate from ref/jpeg-ai-qaf, and why. Printed by describe() so a
#: report can never quote a number without its provenance.
BACKEND_NOTES: dict[str, str] = {
    "vif": "reference uses IQA_pytorch.VIFs(channels=1); we use piq.vif_p on Y",
    "fsim": "reference uses IQA_pytorch.FSIM(channels=3); we use piq.fsim on RGB",
    "nlpd": "reference uses IQA_pytorch.NLPD(channels=1); we use pyiqa nlpd on Y",
    "iw_ssim": "reference uses its own IW_SSIM_PyTorch (needs pyrtools); "
               "we use piq.information_weighted_ssim on Y",
}

#: Internal precision the reference framework evaluates at, regardless of source
#: bit depth (qaf/metrics.py::MetricParent defaults bits=10, max_val=1023).
INTERNAL_BITS = 10

_PYIQA_CACHE: dict[str, object] = {}


def _pyiqa_metric(name: str):
    """Lazily build and cache a pyiqa metric. Returns None if unavailable."""
    if pyiqa is None:
        return None
    if name not in _PYIQA_CACHE:
        try:
            _PYIQA_CACHE[name] = pyiqa.create_metric(name, device="cpu")
        except Exception as exc:
            _BACKEND_ERR[f"pyiqa:{name}"] = str(exc)
            _PYIQA_CACHE[name] = None
    return _PYIQA_CACHE[name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quantise_to_bitdepth(x: torch.Tensor, bitdepth: int = 8) -> torch.Tensor:
    """Paper eq. (6): scale to the integer range, round, clip, scale back.

    Metrics must be computed on what the decoder actually outputs, not on
    unclipped floats. Skipping this flatters every learned codec slightly,
    because it hides clipping loss in saturated regions.
    """
    peak = float(2**bitdepth - 1)
    return torch.clamp(torch.round(x * peak), 0.0, peak) / peak


def _as_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"expected [N,C,H,W] or [C,H,W], got shape {tuple(x.shape)}")
    return x.float()


def rgb_to_ycbcr_bt709(x: torch.Tensor) -> torch.Tensor:
    """BT.709 full-range RGB -> YCbCr, for the per-component PSNRs.

    The codec's own colour pipeline lives in jpegai/models -- this is only for
    measurement, kept here so eval has no dependency on the model package.
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556 + 0.5
    cr = (r - y) / 1.5748 + 0.5
    return torch.cat([y, cb, cr], dim=1)


def luma(x: torch.Tensor) -> torch.Tensor:
    """The Y plane as [N,1,H,W] in [0,1].

    Six of the seven metrics are luma-only in the reference framework, so this is
    the default input path -- not a special case.
    """
    x = _as_4d(x)
    if x.shape[1] == 1:
        return x
    return rgb_to_ycbcr_bt709(x)[:, 0:1]


def _pad_to_multiple(x: torch.Tensor, mult: int = 8) -> torch.Tensor:
    """Replicate-pad H and W up to a multiple of `mult`.

    Matches qaf/metrics.py::PSNR_HVS.pad_img. Necessary because the metric works
    on whole 8x8 DCT blocks; cropping instead would silently discard an edge
    strip, and zero-padding would invent a hard edge the codec never coded.
    """
    h, w = x.shape[-2:]
    return F.pad(x, (0, (-w) % mult, 0, (-h) % mult), mode="replicate")


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------
def psnr(ref: torch.Tensor, test: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(_as_4d(test), _as_4d(ref)).item()
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def psnr_yuv(ref: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    """PSNR per BT.709 component. Y is the one usually quoted; U/V justify EFE."""
    a, b = rgb_to_ycbcr_bt709(_as_4d(ref)), rgb_to_ycbcr_bt709(_as_4d(test))
    out = {}
    for i, name in enumerate(("psnr_y", "psnr_u", "psnr_v")):
        mse = F.mse_loss(b[:, i], a[:, i]).item()
        out[name] = float("inf") if mse <= 0 else float(10.0 * np.log10(1.0 / mse))
    return out


# ---------------------------------------------------------------------------
# PSNR-HVS-M  (own implementation -- not in piq or pyiqa)
# ---------------------------------------------------------------------------
# Transcribed from the reference implementation of Ponomarenko et al.,
# "On between-coefficient contrast masking of DCT basis functions" (2007).
#
# NOTE: these tables are transcribed from the published reference and have not
# been machine-verified against it here. They affect all codecs identically, so
# relative comparisons (which is all BD-rate uses) are robust to small errors;
# do not quote absolute PSNR-HVS-M values against the literature without
# checking them against the reference MATLAB.
_CSF = np.array(
    [
        [1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
        [2.339554, 2.144755, 1.851790, 1.608443, 1.211893, 0.732298, 0.418619, 0.284084],
        [2.573509, 1.851790, 1.744214, 1.529301, 1.098930, 0.684873, 0.377353, 0.248235],
        [1.608443, 1.608443, 1.529301, 1.339561, 0.943549, 0.611868, 0.353971, 0.242179],
        [1.072295, 1.211893, 1.098930, 0.943549, 0.687873, 0.473293, 0.285024, 0.215153],
        [0.643377, 0.732298, 0.684873, 0.611868, 0.473293, 0.348298, 0.221909, 0.177301],
        [0.504610, 0.418619, 0.377353, 0.353971, 0.285024, 0.221909, 0.153270, 0.132178],
        [0.421887, 0.284084, 0.248235, 0.242179, 0.215153, 0.177301, 0.132178, 0.126988],
    ]
)
_MASK = np.array(
    [
        [0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
        [0.826446, 0.694444, 0.517028, 0.390625, 0.221885, 0.081633, 0.026874, 0.012385],
        [1.000000, 0.517028, 0.462493, 0.355556, 0.180074, 0.070248, 0.021684, 0.009490],
        [0.390625, 0.390625, 0.355556, 0.271267, 0.134613, 0.056000, 0.019255, 0.009155],
        [0.173611, 0.221885, 0.180074, 0.134613, 0.071831, 0.033864, 0.012238, 0.006988],
        [0.062500, 0.081633, 0.070248, 0.056000, 0.033864, 0.018588, 0.007565, 0.004769],
        [0.038447, 0.026874, 0.021684, 0.019255, 0.012238, 0.007565, 0.003688, 0.002846],
        [0.026874, 0.012385, 0.009490, 0.009155, 0.006988, 0.004769, 0.002846, 0.002453],
    ]
)


def _dct_matrix(n: int = 8) -> np.ndarray:
    """Orthonormal DCT-II matrix, so D @ blk @ D.T matches MATLAB's dct2."""
    k = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    d = np.cos(np.pi * (2 * j + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    d[0] /= np.sqrt(2.0)
    return d


_D8 = _dct_matrix(8)


def _blocks_8x8(img: np.ndarray) -> np.ndarray:
    """[H,W] -> [nblocks, 8, 8], non-overlapping, trailing partial blocks dropped."""
    h, w = (img.shape[0] // 8) * 8, (img.shape[1] // 8) * 8
    if h == 0 or w == 0:
        return np.empty((0, 8, 8))
    return (
        img[:h, :w]
        .reshape(h // 8, 8, w // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )


def _mask_energy(blk: np.ndarray, blk_dct: np.ndarray) -> np.ndarray:
    """Per-block contrast-masking energy. blk/blk_dct: [N,8,8] -> [N]."""
    m = np.einsum("nkl,kl->n", blk_dct**2, _MASK)
    total_var = blk.reshape(len(blk), -1).var(axis=1)
    quad_var = sum(
        blk[:, r : r + 4, c : c + 4].reshape(len(blk), -1).var(axis=1)
        for r, c in ((0, 0), (0, 4), (4, 4), (4, 0))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pop = np.where(total_var != 0, quad_var / total_var, 0.0)
    return np.sqrt(np.maximum(m * pop, 0.0)) / 32.0


def psnr_hvs_m(ref: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    """PSNR-HVS-M (with masking) and PSNR-HVS (without), on the BT.709 luma.

    Both are returned because the paper reports PSNR-HVS-M but the unmasked
    variant is a useful debugging companion: if they diverge wildly, the masking
    stage is likely wrong.
    """
    y_ref = (rgb_to_ycbcr_bt709(_as_4d(ref))[:, 0] * 255.0).cpu().numpy()
    y_tst = (rgb_to_ycbcr_bt709(_as_4d(test))[:, 0] * 255.0).cpu().numpy()

    s_masked, s_plain, count = 0.0, 0.0, 0
    for a_img, b_img in zip(y_ref, y_tst):
        a, b = _blocks_8x8(a_img), _blocks_8x8(b_img)
        if len(a) == 0:
            continue
        a_dct = np.einsum("ik,nkl,jl->nij", _D8, a, _D8)
        b_dct = np.einsum("ik,nkl,jl->nij", _D8, b, _D8)

        # Masking uses whichever image tolerates more distortion at this block.
        mask = np.maximum(_mask_energy(a, a_dct), _mask_energy(b, b_dct))

        diff = np.abs(a_dct - b_dct)
        s_plain += float(np.sum((diff * _CSF) ** 2))

        # DC (0,0) is never masked; AC coefficients are reduced by the threshold.
        thresh = mask[:, None, None] / _MASK[None, :, :]
        thresh[:, 0, 0] = 0.0
        reduced = np.maximum(diff - thresh, 0.0)
        s_masked += float(np.sum((reduced * _CSF) ** 2))
        count += a.size  # nblocks * 64 coefficients

    if count == 0:
        return {"psnr_hvsm": float("nan"), "psnr_hvs": float("nan")}

    def _to_psnr(sse: float) -> float:
        mse = sse / count
        return float("inf") if mse <= 1e-12 else float(10.0 * np.log10(255.0**2 / mse))

    return {"psnr_hvsm": _to_psnr(s_masked), "psnr_hvs": _to_psnr(s_plain)}


# ---------------------------------------------------------------------------
# NLPD fallback (approximation -- clearly labelled as such)
# ---------------------------------------------------------------------------
_NLPD_FILT = torch.tensor([0.05, 0.25, 0.4, 0.25, 0.05])


def _nlpd_approx(ref: torch.Tensor, test: torch.Tensor, scales: int = 6) -> float:
    """Laplacian-pyramid + divisive-normalisation distance.

    APPROXIMATION. The published NLPD (Laparra et al. 2016) uses per-scale
    normalisation filters and sigmas fitted to psychophysical data, which we do
    not have. This keeps the structure -- Laplacian pyramid, local divisive
    normalisation, per-scale root-mean-square difference -- with generic
    parameters. Use pyiqa's `nlpd` when available; this exists so the metric
    slot is never silently empty.

    Lower is better, so runbench negates it before BD-rate (which assumes
    higher-is-better quality).
    """
    k = (_NLPD_FILT[:, None] * _NLPD_FILT[None, :]).to(ref)
    k = (k / k.sum()).expand(1, 1, 5, 5)

    def luma(x):
        return rgb_to_ycbcr_bt709(_as_4d(x))[:, 0:1]

    a, b = luma(ref), luma(test)
    dists = []
    for s in range(scales):
        if min(a.shape[-2:]) < 8:
            break
        if s < scales - 1:
            a_lo = F.conv2d(F.pad(a, (2, 2, 2, 2), mode="reflect"), k)[..., ::2, ::2]
            b_lo = F.conv2d(F.pad(b, (2, 2, 2, 2), mode="reflect"), k)[..., ::2, ::2]
            a_up = F.interpolate(a_lo, size=a.shape[-2:], mode="bilinear", align_corners=False)
            b_up = F.interpolate(b_lo, size=b.shape[-2:], mode="bilinear", align_corners=False)
            a_band, b_band = a - a_up, b - b_up
        else:
            a_lo = b_lo = None
            a_band, b_band = a, b

        # Divisive normalisation by local amplitude, shared denominator floor.
        def norm(band):
            amp = F.conv2d(F.pad(band.abs(), (2, 2, 2, 2), mode="reflect"), k)
            return band / (0.17 + amp)

        dists.append(torch.sqrt(F.mse_loss(norm(a_band), norm(b_band)) + 1e-12).item())
        if a_lo is None:
            break
        a, b = a_lo, b_lo

    return float(np.sqrt(np.mean(np.square(dists)))) if dists else float("nan")


# ---------------------------------------------------------------------------
# VMAF (shells out to ffmpeg/libvmaf)
# ---------------------------------------------------------------------------
def _have_vmaf() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        return "libvmaf" in out
    except Exception:
        return False


def vmaf(ref: torch.Tensor, test: torch.Tensor) -> float:
    """VMAF via ffmpeg's libvmaf filter. Returns NaN if ffmpeg lacks libvmaf.

    Slow (process spawn + PNG round-trip per image), so runbench caches results.
    """
    if not _have_vmaf():
        return float("nan")
    from PIL import Image

    def _save(t: torch.Tensor, path: Path) -> None:
        arr = (_as_4d(t)[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr).save(path)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _save(ref, td / "ref.png")
        _save(test, td / "tst.png")
        log = td / "vmaf.json"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(td / "tst.png"), "-i", str(td / "ref.png"),
            "-lavfi", f"libvmaf=log_fmt=json:log_path={log}",
            "-f", "null", "-",
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            import json
            data = json.loads(log.read_text())
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except Exception as exc:
            _BACKEND_ERR["vmaf"] = str(exc)
            return float("nan")


# ---------------------------------------------------------------------------
# Registry
#
# Each entry runs on the plane and in the range the reference framework uses.
# `on` selects the input transform, so the plane convention is declared once per
# metric instead of being buried in a lambda.
# ---------------------------------------------------------------------------
def _piq_metric(fn_name: str, *, on: str = "y", scale: float = 1.0):
    """piq metric on the Y plane (default) or RGB, scaled to `scale`."""
    def call(ref, test):
        if piq is None:
            return float("nan")
        fn = getattr(piq, fn_name, None)
        if fn is None:
            _BACKEND_ERR[f"piq:{fn_name}"] = "not present in installed piq"
            return float("nan")
        a = (luma(test) if on == "y" else _as_4d(test)).clamp(0, 1) * scale
        b = (luma(ref) if on == "y" else _as_4d(ref)).clamp(0, 1) * scale
        try:
            return float(fn(a, b, data_range=scale))
        except Exception as exc:
            _BACKEND_ERR[f"piq:{fn_name}"] = str(exc)
            return float("nan")
    return call


def ms_ssim(ref: torch.Tensor, test: torch.Tensor) -> float:
    """MS-SSIM on Y at `INTERNAL_BITS` precision, per qaf::MSSSIMTorch.

    Uses `pytorch_msssim`, the reference's own backend, so this metric *is*
    comparable with the paper. piq is only a fallback: its multi_scale_ssim
    rejects images below 161x161, which the reference accepts.
    """
    peak = float(2**INTERNAL_BITS - 1)
    a, b = luma(test).clamp(0, 1) * peak, luma(ref).clamp(0, 1) * peak
    if _pt_ms_ssim is not None:
        try:
            return float(_pt_ms_ssim(a, b, data_range=peak))
        except Exception as exc:
            _BACKEND_ERR["pytorch_msssim:ms_ssim"] = str(exc)
    return _piq_metric("multi_scale_ssim")(ref, test)


def psnr_hvs(ref: torch.Tensor, test: torch.Tensor) -> float:
    """PSNR-HVS on Y, per qaf::PSNR_HVS -- the paper's seventh metric.

    Prefers the reference's `psnr_hvsm` package. Our own implementation (below)
    agrees on the algorithm but not necessarily to the last decimal, because the
    CSF and masking tables are quoted to different precision in the literature.
    """
    if _ref_psnr_hvs is not None:
        try:
            a = _pad_to_multiple(luma(test).clamp(0, 1)).squeeze().double().cpu().numpy()
            b = _pad_to_multiple(luma(ref).clamp(0, 1)).squeeze().double().cpu().numpy()
            p_hvs, _p_hvs_m = _ref_psnr_hvs(b, a)
            return float(p_hvs)
        except Exception as exc:
            _BACKEND_ERR["psnr_hvsm:psnr_hvs"] = str(exc)
    return psnr_hvs_m(ref, test)["psnr_hvs"]


def _nlpd(ref, test) -> float:
    """NLPD on Y. Lower is better -- negated by runbench for BD-rate.

    The reference builds `IQA_pytorch.NLPD(channels=1)` and feeds it Y. pyiqa's
    nlpd hard-rejects 1-channel input, so we replicate Y across three channels.
    That is exact, not a fudge: any weighted-sum luma conversion of a grey image
    returns the same grey (the coefficients sum to 1), and if pyiqa instead runs
    per-channel and averages, three identical channels average to the single-Y
    value. Either path reproduces NLPD(channels=1) on Y.
    """
    m = _pyiqa_metric("nlpd")
    if m is not None:
        try:
            y_t = luma(test).clamp(0, 1).repeat(1, 3, 1, 1)
            y_r = luma(ref).clamp(0, 1).repeat(1, 3, 1, 1)
            return float(m(y_t, y_r))
        except Exception as exc:
            _BACKEND_ERR["pyiqa:nlpd"] = str(exc)
    return _nlpd_approx(ref, test)


# name -> (callable, higher_is_better, backend description)
REGISTRY: dict[str, tuple] = {
    "ms_ssim":   (ms_ssim,                                  True,  "pytorch_msssim (Y, 10-bit)"),
    "vif":       (_piq_metric("vif_p"),                     True,  "piq (Y)"),
    "fsim":      (_piq_metric("fsim", on="rgb"),            True,  "piq (RGB)"),
    "iw_ssim":   (_piq_metric("information_weighted_ssim", scale=255.0),
                                                            True,  "piq (Y, 0-255)"),
    "nlpd":      (_nlpd,                                    False, "pyiqa (Y), else own approximation"),
    "vmaf":      (vmaf,                                     True,  "ffmpeg/libvmaf"),
    "psnr_hvs":  (psnr_hvs,                                 True,  "psnr_hvsm pkg (Y), else own"),
    "psnr_hvsm": (lambda r, t: psnr_hvs_m(r, t)["psnr_hvsm"], True, "own implementation"),
}

#: The seven metrics the paper averages for its BD-rate "AVG" column.
#: PSNR-HVS, not PSNR-HVS-M -- see the module docstring.
PAPER_SEVEN = ["ms_ssim", "vif", "fsim", "vmaf", "nlpd", "psnr_hvs", "iw_ssim"]


def compute_all(
    ref: torch.Tensor,
    test: torch.Tensor,
    *,
    bitdepth: int = 8,
    metrics: list[str] | None = None,
    include_psnr: bool = True,
) -> dict[str, float]:
    """All requested metrics for one image pair, as a flat dict.

    Both images are quantised to `bitdepth` first (paper eq. 6) so we measure
    what a decoder actually emits.

    Sign convention: values are returned as the metric natively defines them.
    Lower-is-better metrics (currently NLPD) are flagged in REGISTRY; runbench
    negates them before handing anything to BD-rate.
    """
    ref = quantise_to_bitdepth(_as_4d(ref).clamp(0, 1), bitdepth)
    test = quantise_to_bitdepth(_as_4d(test).clamp(0, 1), bitdepth)

    names = metrics if metrics is not None else PAPER_SEVEN
    out: dict[str, float] = {}
    for name in names:
        if name not in REGISTRY:
            raise KeyError(f"unknown metric {name!r}; known: {sorted(REGISTRY)}")
        out[name] = REGISTRY[name][0](ref, test)

    if include_psnr:
        out["psnr"] = psnr(ref, test)
        out.update(psnr_yuv(ref, test))
        # psnr_hvs is one of PAPER_SEVEN now, and the registry entry prefers the
        # reference `psnr_hvsm` package. Only fill it in from our own
        # implementation if it was not requested as a metric -- otherwise this
        # would silently overwrite the comparable value with the approximate one.
        if "psnr_hvs" not in out:
            out["psnr_hvs"] = psnr_hvs_m(ref, test)["psnr_hvs"]
    return out


def describe() -> None:
    """Print which metrics work right now, and why the others don't.

    Run this immediately after setup.sh: `python -m jpegai.eval.metrics`
    """
    # 192x192: above piq's 161x161 floor for the 5-scale pyramid metrics, so the
    # smoke test exercises the same path Kodak (768x512) will.
    torch.manual_seed(0)
    ref = torch.rand(1, 3, 192, 192)
    test = (ref + 0.02 * torch.randn_like(ref)).clamp(0, 1)

    print("backend availability")
    for lib, ok, hint in [
        ("piq", piq is not None, ""),
        ("pyiqa", pyiqa is not None, ""),
        ("pytorch_msssim", _pt_ms_ssim is not None, "the reference MS-SSIM backend"),
        ("psnr_hvsm", _ref_psnr_hvs is not None,
         "unavailable on macOS/arm64 -- see docs/07 1.2; our own psnr_hvs is used"),
        ("ffmpeg+vmaf", _have_vmaf(), "brew install ffmpeg"),
    ]:
        state = "ok" if ok else f"MISSING{'  (' + hint + ')' if hint else ''}"
        print(f"  {lib:15} {state}")

    print(f"\nmetric smoke test ({tuple(ref.shape[-2:])} noise pair, "
          f"internal precision {INTERNAL_BITS}-bit)")
    print(f"  {'metric':12} {'value':>12}  {'dir':4} backend")
    for name in PAPER_SEVEN:
        fn, higher, backend = REGISTRY[name]
        try:
            val = fn(ref, test)
        except Exception as exc:
            val, backend = float("nan"), f"ERROR: {exc}"
        flag = "up" if higher else "down"
        print(f"  {name:12} {val:12.5f}  {flag:4} {backend}")

    extra = {"psnr": psnr(ref, test), **psnr_yuv(ref, test), **psnr_hvs_m(ref, test)}
    for k, v in extra.items():
        print(f"  {k:12} {v:12.5f}  up   own")

    if _BACKEND_ERR:
        print("\nrecorded backend problems")
        for k, v in _BACKEND_ERR.items():
            print(f"  {k}: {v}")

    missing = [n for n in PAPER_SEVEN if not np.isfinite(REGISTRY[n][0](ref, test))]
    print(
        f"\n{len(PAPER_SEVEN) - len(missing)}/7 paper metrics working."
        + (f" Missing: {', '.join(missing)}" if missing else " All present.")
    )


if __name__ == "__main__":
    describe()
