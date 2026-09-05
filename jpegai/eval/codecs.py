"""Traditional codec anchors for the rate-distortion comparison.

The paper's anchor is **VTM** (VVC intra), which we cannot use: VTM intra on one
Kodak image takes minutes, the encoder is a C++ build, and the JPEG AI common test
conditions specify configuration we do not have. So this project uses two anchor
strategies, and reports which one every number came from:

1. **Codecs we can actually run** -- JPEG, WebP, AVIF, JPEG 2000, and VVC through
   VVenC. These give us a real, reproducible RD curve on our own machine, at our own
   bit depths, on our own datasets. JPEG is the honest headline comparison for a
   student project ("we beat JPEG by X%"); the VVenC row is the one to quote against
   the paper, because it is the paper's own codec class rather than a proxy for it.
2. **VTM points read from a file** -- `jpeg-ai-anchors` publishes anchor RD data, and
   `runbench --anchor-json` would consume it, giving BD-rate against the *same*
   encoder the paper used rather than a different VVC implementation. Not implemented.
   The VVenC row made it much less urgent than it was when this note was written.

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
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
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

# AVIF: AV1 intra. The strongest anchor Pillow can give us, and until VVenC landed it
# stood in for VTM on the argument that AV1 and VVC intra sit within a few percent of
# each other on still images. Measured on Kodak, that argument is wrong: AVIF needs
# **18.9% more bits than VVenC** for the same quality (`results/p6_9pt_vs_vvc.md`), so
# the stand-in was understating the paper's anchor by about a fifth, and every "we are N%
# behind AVIF" claim was correspondingly flattering. Some of that is encoder effort
# rather than format -- speed=4 against preset slower is not a matched comparison -- but
# not nineteen points of it. Anything compared with the paper should cite the vvc row.
# speed=4 balances encode time against compression; speed=0 is ~10x slower for ~1% gain.
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

# ---------------------------------------------------------------------------
# VVC intra via VVenC -- the paper's own anchor class
# ---------------------------------------------------------------------------
# Every headline number in the paper is BD-rate against VTM, the VVC reference
# encoder, and until now this project had no VVC point at all: the AVIF row stood in
# for it, on the argument that AV1 and VVC intra sit within a few percent of each
# other. That argument was carrying the entire comparison the report rests on, so
# measure it instead of asserting it.
#
# VVenC is Fraunhofer's production VVC encoder. It is *not* VTM -- at `--preset
# slower` it lands within a few percent of VTM intra, which makes it a far tighter
# proxy than AVIF, and unlike VTM it finishes 24 Kodak images in minutes instead of
# hours. Report it as VVenC. Never relabel it VTM.
#
#     brew install vvenc          # provides vvencapp
#
# Decoding needs no second install: ffmpeg >= 7 ships a native VVC decoder, and this
# machine has 9.0.1. The colour conversions run through ffmpeg in both directions
# with matched full-range BT.709, so the only losses are VVC's own plus 4:2:0 chroma
# subsampling -- exactly what JPEG, WebP and AVIF each do internally.
#
# The sanity check is *not* "QP 1 should clear 50 dB". 4:2:0 discards three quarters of
# the chroma samples before VVC ever sees the image, and that is irreversible, so it
# caps RGB PSNR however fine the quantiser gets. Measured on kodim01: rgb -> yuv420 ->
# rgb with no codec at all tops out at 44.17 dB (R 42.90, G 48.38, B 43.10 -- green
# survives because Y carries most of it). QP 1 lands at 44.01 dB, 0.16 dB under a
# ceiling set by the format rather than by the encoder, which is what a matched pair of
# conversions looks like. So compare QP 1 against a conversion-only round trip, not
# against a fixed number; a gap of more than a few tenths there is a colour bug being
# read as coding loss.
VVENC_PRESET = "slower"

#: Flag spellings, kept in one place because they are the only fragile part here.
#: `vvencapp --help` pins them; a mismatch fails the 32x32 probe in milliseconds
#: rather than 24 images in.
VVENC_ENCODE = ("{bin} --input {yuv} --size {w}x{h} --format yuv420 --frames 1 "
                "--framerate 1 --qp {qp} --preset {preset} --output {bs}")
FFMPEG_TO_YUV = ("{bin} -v error -y -f rawvideo -pix_fmt rgb24 -s {w}x{h} -i {rgb} "
                 "-vf scale=in_range=full:out_range=full -pix_fmt yuv420p "
                 "-f rawvideo {yuv}")
FFMPEG_FROM_VVC = ("{bin} -v error -y -i {bs} -frames:v 1 "
                   "-vf scale=in_range=full:out_range=full -pix_fmt rgb24 "
                   "-f rawvideo {rgb}")


def _run(cmd: str) -> None:
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"exit {r.returncode}: {cmd}\n{r.stdout[-1500:]}")


@dataclass
class SubprocessCodec:
    """A codec that lives in external binaries, with `Codec`'s public surface.

    Duck-typed rather than a subclass: `Codec.encode_decode` is hard-wired to a PIL
    in-memory buffer, which is the right design for the three PIL anchors and no use
    at all for a pair of command-line tools that only speak files.
    """

    name: str
    qualities: list
    note: str = ""
    needs: tuple[str, ...] = ()
    _available: bool | None = field(default=None, repr=False)

    def available(self) -> bool:
        if self._available is None:
            missing = [b for b in self.needs if shutil.which(b) is None]
            if missing:
                _ERR[self.name] = f"not on PATH: {', '.join(missing)}"
                self._available = False
            else:
                try:
                    probe = (np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3))
                    self.encode_decode(probe, self.qualities[len(self.qualities) // 2])
                    self._available = True
                except Exception as exc:
                    _ERR[self.name] = str(exc)
                    self._available = False
        return self._available

    def encode_decode(self, rgb: np.ndarray, quality) -> tuple[int, np.ndarray]:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected uint8 [H,W,3], got {rgb.dtype} {rgb.shape}")
        h, w = rgb.shape[:2]
        if w % 2 or h % 2:
            raise ValueError(f"{self.name}: 4:2:0 needs even dimensions, got {w}x{h}")
        ff, enc = shutil.which("ffmpeg"), shutil.which("vvencapp")

        with tempfile.TemporaryDirectory(prefix="vvc_") as td:
            d = Path(td)
            src, yuv, bs, out = d / "s.rgb", d / "s.yuv", d / "c.266", d / "d.rgb"
            src.write_bytes(rgb.tobytes())
            _run(FFMPEG_TO_YUV.format(bin=ff, w=w, h=h, rgb=src, yuv=yuv))
            _run(VVENC_ENCODE.format(bin=enc, yuv=yuv, w=w, h=h, qp=int(quality),
                                     preset=VVENC_PRESET, bs=bs))
            nbytes = bs.stat().st_size
            _run(FFMPEG_FROM_VVC.format(bin=ff, bs=bs, rgb=out))
            dec = np.frombuffer(out.read_bytes(), dtype=np.uint8)

        if dec.size != h * w * 3:
            raise RuntimeError(f"{self.name} q={quality}: decoded {dec.size} bytes, "
                               f"expected {h * w * 3}")
        return nbytes, dec.reshape(h, w, 3)


# QP descends so the ladder ascends in rate, matching every other codec here, and
# spans roughly 0.05-3 bpp on photographic content to bracket JPEG's own range.
VVC = SubprocessCodec(
    name="vvc",
    qualities=[47, 44, 41, 38, 35, 32, 29, 26, 22, 18],
    note=f"VVenC preset {VVENC_PRESET}, VVC intra, 4:2:0 -- proxy for the paper's VTM",
    needs=("vvencapp", "ffmpeg"),
)

REGISTRY: dict[str, Codec] = {c.name: c for c in (JPEG, WEBP, AVIF, JP2, VVC)}

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
    if not VVC.available():
        print("\nvvc is the paper's own anchor class -- every table in it is BD-rate")
        print("against VTM, and without this row the report has no VVC number at all.")
        print("    brew install vvenc          # ffmpeg already decodes VVC")
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
