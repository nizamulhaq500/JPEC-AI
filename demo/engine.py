"""Pure, UI-agnostic engine behind the Streamlit demo (`demo/app.py`).

Every function here takes arrays/tensors and returns arrays/numbers -- no Streamlit, no
globals -- so the demo's real behaviour is unit-tested headlessly (`tests/test_demo.py`)
and the app layer stays a thin shell. It drives the exact encode/decode path the CLI uses
(`jpegai.cli`, `jpegai.codestream`), so what the demo shows is what `jpegai encode/decode`
produces, not a parallel reimplementation.

Scope is the CLI's scope, honestly (see `CAPABILITIES`): variable-rate Delta_beta, RoI via
the gain quality map, the luma-only base-vs-full decode on one codestream, and the latent
bit-allocation view the forward pass already exposes. Tiling, region random-access,
progressive truncation, skip mode, and multiple normative decoder IDs are *not* in this
build, and the demo names them as absent rather than faking them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jpegai import cli
from jpegai.codestream import coded_bytes, describe, read_packet, write_packet

# What the demo can honestly show, and what it deliberately cannot. The app renders this
# verbatim, so the honesty boundary is on screen, not buried in a footnote.
CAPABILITIES = {
    "in": [
        "Variable rate -- luma/chroma Δβ on a gain checkpoint, one set of weights",
        "Region of interest -- a mask boosts quality on the latent grid, same codec",
        "Base vs full decode -- one codestream, luma-only or full chroma",
        "Latent bit-allocation -- per-position bits and the σ-index map",
        "Container accounting -- coded rate vs on-disk bytes (the scaffolding gap)",
    ],
    "out": [
        "Progressive decode -- needs residual truncation; not in this build",
        "Random-access crop -- needs region partitioning; not in this build",
        "Skip-mode overlay -- Phase 9 skip mode not built",
        "Three normative decoder IDs -- only base/full here (Track B)",
        "Synthesis tiling for bounded-memory 4K decode -- not in this build",
    ],
}


def load(checkpoint: str, device: str | None = None):
    """The CLI's own loader, so the demo codes with the exact trained model."""
    return cli._load_model(checkpoint, device)


def _mask_to_q(model, x, mask, boost: int):
    """A greyscale/boolean mask (H×W array) -> integer quality map on the luma latent grid.

    The array-in-memory twin of `cli._roi_q_index` (which opens a PNG): the demo's canvas
    hands us pixels, not a file. Same arithmetic -- resize to the luma latent exactly (no
    assumed stride), threshold, scale by the boost, clamp into Table I's range -- and the
    one map drives both branches, since their latents share a grid.
    """
    if not getattr(model, "gain", False):
        raise ValueError("RoI needs a gain (variable-rate) checkpoint; this one is "
                         "fixed-rate and has no quality map to steer")
    from PIL import Image

    y, _uv, _supp, _pad = model._to_planes(x)
    lh, lw = model.g_a_y(y).shape[-2:]
    m = Image.fromarray(np.asarray(mask)).convert("L").resize((lw, lh), Image.NEAREST)
    on = torch.from_numpy((np.asarray(m) > 127).astype("int64"))
    return (on * int(boost)).clamp(-8, 8).view(1, 1, lh, lw).to(x.device)


@dataclass
class EncodeResult:
    packet: dict
    coded_bytes: int
    disk_bytes: int
    bpp: float
    width: int
    height: int

    @property
    def scaffolding_pct(self) -> float:
        return 100.0 * (self.disk_bytes - self.coded_bytes) / max(self.disk_bytes, 1)


def encode(model, rgb: np.ndarray, dev, *, delta_beta=(0, 0),
           roi_mask=None, roi_boost: int = 4, path=None) -> EncodeResult:
    """Encode an H×W×3 uint8 array exactly as `jpegai encode` would.

    `path` writes a real `.jpegai` container and reports its on-disk size; omit it and
    `disk_bytes` falls back to the coded size (the demo often only needs the rate).
    """
    x = cli._to_tensor(rgb, dev)
    q_index = _mask_to_q(model, x, roi_mask, roi_boost) if roi_mask is not None else None
    with torch.no_grad():
        packet = model.compress(x, delta_beta=(int(delta_beta[0]), int(delta_beta[1])),
                                q_index=q_index)
    h, w = rgb.shape[:2]
    cb = coded_bytes(packet)
    disk = write_packet(path, packet) if path is not None else cb
    return EncodeResult(packet, cb, disk, cb * 8.0 / (h * w), w, h)


def decode(model, packet: dict, dev, *, luma_only: bool = False) -> np.ndarray:
    """Decode a packet to an H×W×3 uint8 array (the full codec, or the luma-only base)."""
    with torch.no_grad():
        rec = model.decompress(packet, device=dev, luma_only=luma_only)
    return _to_uint8(rec["x_hat"])


def _to_uint8(x_hat) -> np.ndarray:
    out = (x_hat.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return out.squeeze(0).permute(1, 2, 0).cpu().numpy()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """RGB PSNR in dB between two uint8 arrays; inf when identical."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def region_psnr(original: np.ndarray, recon: np.ndarray, mask: np.ndarray
                ) -> tuple[float, float]:
    """PSNR inside vs outside a full-resolution mask (>127 = inside).

    The RoI claim made measurable: boost a region and this should show the inside gaining
    dB over a baseline encode while the outside gives some back. Masked MSE over all three
    channels, so a flat colour shift counts the same as a texture loss.
    """
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    inside = m > 127
    o, r = original.astype(np.float64), recon.astype(np.float64)
    se = ((o - r) ** 2).mean(axis=2)   # per-pixel MSE across RGB

    def _db(sel):
        if not sel.any():
            return float("nan")
        mse = float(se[sel].mean())
        return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)

    return _db(inside), _db(~inside)


@torch.no_grad()
def latent_maps(model, rgb: np.ndarray, dev, *, delta_beta=(0, 0)) -> dict:
    """Per-position luma bits (the bit-allocation map) and the mean σ-index map.

    `bits[h,w] = -Σ_c log2 p(ŷ[c,h,w])` straight from the forward likelihoods -- literally
    where the coder spends its budget, and the single most striking latent view: bright
    where the image is hard. The σ-index map is the gain unit's own quality signal
    (`i_sigma`), averaged over channels. Both live on the luma latent grid. Returns `None`
    maps on a fused-hyper codec, which exposes no integer σ.
    """
    x = cli._to_tensor(rgb, dev)
    out = model.forward(x, delta_beta=(int(delta_beta[0]), int(delta_beta[1])))
    lik = out["likelihoods"]["y"].clamp_min(1e-9)
    bits = (-torch.log2(lik)).sum(dim=1).squeeze(0).cpu().numpy()
    sigma = None
    if "i_sigma" in out:
        sigma = out["i_sigma"].float().mean(dim=1).squeeze(0).cpu().numpy()
    return {"bits": bits, "sigma": sigma,
            "total_bits": float(bits.sum()), "grid": tuple(bits.shape)}


def inspect(path: str) -> dict:
    """`describe` passthrough -- the container accounting panel reads this."""
    return describe(path)
