"""Traditional codec anchors for the rate-distortion comparison.

The paper's anchor is **VTM** (VVC intra), which we cannot use: VTM intra on one
Kodak image takes minutes, the encoder is a C++ build, and the JPEG AI common test
conditions specify configuration we do not have. So this project uses two anchor
strategies, and reports which one every number came from:

1. **Codecs we can actually run** -- JPEG, WebP, AVIF, JPEG 2000. These give us a
   real, reproducible RD curve on our own machine, at our own bit depths, on our
   own datasets. JPEG is the honest headline comparison for a student project
   ("we beat JPEG by X%"), and AVIF is the strong modern anchor.
2. **VTM points read from a file** -- `jpeg-ai-anchors` publishes anchor RD data.
   `runbench --anchor-json` consumes it, which lets us compute BD-rate against
   the *same* anchor the paper used without running VTM. This is the only path to
   a number directly comparable with Tables III-VI.

Interface: every codec is a `Codec` with

    encode_decode(rgb_uint8_hwc, quality) -> (nbytes, decoded_rgb_uint8_hwc)

so the caller never handles file paths, and bpp is computed from the real
compressed size including all headers. Counting only entropy-coded payload would
flatter the anchors by a few percent at low rates -- exactly where the comparison
matters most.

Quality ladders are chosen to span roughly 0.05-2.0 bpp on photographic content,
which brackets JPEG AI's operating range with margin on both sides. BD-rate needs
>= 4 points per curve for a cubic fit; every ladder here has at least 8, so
dropping a failed point still leaves a valid fit.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_ERR: dict[str, str] = {}

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    _ERR["pillow"] = str(exc)

# pillow-avif-plugin registers the AVIF format with PIL as an import side effect.
try:
    import pillow_avif  # noqa: F401
    _HAVE_AVIF = True
except Exception as exc:  # pragma: no cover
    _HAVE_AVIF = False
    _ERR["pillow-avif-plugin"] = str(exc)


@dataclass
class Codec:
    name: str
    qualities: list          #: values passed to `encode_decode` as `quality`
    save_kwargs: Callable    #: quality -> dict of PIL save kwargs
    pil_format: str
    note: str = ""
    _available: bool | None = field(default=None, repr=False)

    def available(self) -> bool:
        """True if a 32x32 round trip actually works. Cached."""
        if self._available is None:
            if Image is None:
                self._available = False
            else:
                try:
                    probe = (np.arange(32 * 32 * 3, dtype=np.uint8)
                             .reshape(32, 32, 3))
                    self.encode_decode(probe, self.qualities[len(self.qualities) // 2])
                    self._available = True
                except Exception as exc:
                    _ERR[self.name] = str(exc)
                    self._available = False
        return self._available

    def encode_decode(self, rgb: np.ndarray, quality) -> tuple[int, np.ndarray]:
        """Compress, measure, decompress. Returns (bytes on the wire, decoded RGB).

        The decode goes through the *same buffer* that was measured, so there is
        no way for the reported size and the reported image to disagree -- a
        failure mode that quietly invalidates RD curves.
        """
        if Image is None:
            raise RuntimeError(f"Pillow unavailable: {_ERR.get('pillow')}")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected uint8 [H,W,3], got {rgb.dtype} {rgb.shape}")

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format=self.pil_format, **self.save_kwargs(quality))
        nbytes = buf.tell()
        buf.seek(0)
        dec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)

        if dec.shape != rgb.shape:
            raise RuntimeError(
                f"{self.name} q={quality}: decoded {dec.shape} != source {rgb.shape}"
            )
        return nbytes, dec


# ---------------------------------------------------------------------------
# The anchors
# ---------------------------------------------------------------------------
# JPEG: 4:2:0 chroma subsampling, matching JPEG AI's own internal 4:2:0 format,
# and `optimize=True` for optimal Huffman tables. Without `optimize`, libjpeg
# uses the generic example tables from the 1992 spec and wastes 2-5% -- an
# artificially weak anchor, which would inflate our reported gain.
JPEG = Codec(
    name="jpeg",
    qualities=[10, 18, 25, 32, 40, 50, 62, 75, 85, 92, 96],
    save_kwargs=lambda q: {"quality": int(q), "subsampling": 2, "optimize": True},
    pil_format="JPEG",
    note="libjpeg, 4:2:0, optimised Huffman tables",
)

# WebP: `method=6` is the slowest and best-compressing search. Lossless mode is
# never used -- we want the lossy RD curve.
WEBP = Codec(
    name="webp",
    qualities=[8, 15, 25, 35, 45, 55, 68, 80, 90, 95],
    save_kwargs=lambda q: {"quality": int(q), "method": 6, "lossless": False},
    pil_format="WEBP",
    note="libwebp, method 6",
)

# AVIF: AV1 intra. The strongest anchor we can run, and the closest available
# proxy for VTM -- AV1 and VVC intra are within a few percent of each other on
# still images, so "we are N% behind AVIF" is a defensible stand-in for the
# paper's VTM comparison. speed=4 balances encode time against compression;
# speed=0 is ~10x slower for ~1% gain.
AVIF = Codec(
    name="avif",
    qualities=[12, 20, 28, 36, 44, 52, 62, 72, 82, 90],
    save_kwargs=lambda q: {"quality": int(q), "speed": 4},
    pil_format="AVIF",
    note="libavif/AV1 intra, speed 4 -- closest runnable proxy for VTM",
)

# JPEG 2000: parameterised by *compression ratio*, not a 0-100 quality, so its
# ladder is a different kind of number. Included because it is the previous
# generation's "advanced" still codec and makes the historical point that
# wavelets lost to block transforms on photographic content.
JP2 = Codec(
    name="jp2",
    qualities=[240, 160, 110, 75, 50, 34, 22, 15, 10, 7],
    save_kwargs=lambda r: {
        "quality_mode": "rates",
        "quality_layers": [float(r)],
        "irreversible": True,
    },
    pil_format="JPEG2000",
    note="OpenJPEG, irreversible 9/7 wavelet; quality values are compression ratios",
)

REGISTRY: dict[str, Codec] = {c.name: c for c in (JPEG, WEBP, AVIF, JP2)}

#: Sensible default set: one legacy, one mid, one modern.
DEFAULT_CODECS = ["jpeg", "webp", "avif"]


def get(name: str) -> Codec:
    if name not in REGISTRY:
        raise KeyError(f"unknown codec {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def describe() -> None:
    """Print which anchors work here, and how to install the ones that do not."""
    print(f"{'codec':8} {'status':10} {'points':>7}  note")
    for name, c in REGISTRY.items():
        ok = c.available()
        print(f"  {name:6} {'ok' if ok else 'MISSING':10} {len(c.qualities):>7}  {c.note}")
    if not _HAVE_AVIF:
        print("\navif is the strongest anchor we can run. To enable it:")
        print("    pip install pillow-avif-plugin")
    if _ERR:
        print("\nrecorded problems")
        for k, v in _ERR.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    describe()

    # Round-trip sanity: a real gradient+noise image, checking that bitrate is
    # monotone in quality. If a codec's ladder is not monotone, the RD curve will
    # zig-zag and the cubic fit becomes meaningless -- better to know now.
    if Image is not None:
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:256, 0:256]
        img = np.stack([
            (xx * 255 // 255).astype(np.uint8),
            (yy * 255 // 255).astype(np.uint8),
            ((xx + yy) % 256).astype(np.uint8),
        ], axis=-1)
        img = np.clip(img.astype(np.int16) + rng.integers(-12, 13, img.shape), 0, 255).astype(np.uint8)

        print("\nmonotonicity check on a 256x256 gradient+noise image")
        for name, c in REGISTRY.items():
            if not c.available():
                continue
            bpps = []
            for q in c.qualities:
                nbytes, _ = c.encode_decode(img, q)
                bpps.append(nbytes * 8 / (256 * 256))
            mono = all(b <= a + 1e-9 for a, b in zip(bpps[1:], bpps[:-1])) or \
                   all(a <= b + 1e-9 for a, b in zip(bpps[:-1], bpps[1:]))
            print(f"  {name:6} {bpps[0]:6.3f} -> {bpps[-1]:6.3f} bpp   "
                  f"monotone={'yes' if mono else 'NO'}")
