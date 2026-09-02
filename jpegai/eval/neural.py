"""Our trained codec, wrapped so `runbench` can measure it like any anchor.

    python -m jpegai.eval.runbench --codecs jpeg,webp,avif --neural checkpoints/ladder

The point of routing our codec through the *same* harness that produced the
WebP −16.5% / AVIF −42.2% numbers is that nothing about the measurement changes
when the codec under test becomes ours. Same seven metrics, same BD-rate
integration, same anchor, same cache. A separate evaluation path for our own codec
would be the easiest possible way to report a number that is not comparable.

Two decisions here are about honesty rather than code:

**Bytes come from the bitstream, not the model.** `encode_decode` returns
`packet_bytes()`, the length of the actual rANS payload, and the image it returns
is decoded *from that same payload*. The estimate the training loss saw is
optimistic by ~1.9% (the σ grid, see docs/06 §3.1) and is never used here.

**The header is counted.** `packet_bytes()` is payload only, which is the
convention in the learned-compression literature and in compressai. But a JPEG
file on disk includes its own headers, so comparing a bare payload against a
complete JPEG quietly favours us. `header_bytes` adds the minimum a real decoder
needs -- image width and height, from which the latent and hyper-latent shapes and
the padding all follow -- so the comparison is between two self-contained files.
Four bytes is ~0.02% of a Kodak image at 0.4 bpp, so this changes no conclusion;
it just removes a thumb from the scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jpegai.config import PROJECT_ROOT

#: Width and height as uint16 each. Everything else the decoder needs -- latent
#: resolution, hyper-latent resolution, how much padding to strip -- is a fixed
#: function of these two numbers and the (constant) architecture.
DEFAULT_HEADER_BYTES = 4

#: Bumped by any change that moves a bitstream for unchanged weights, so that
#: `fingerprint` invalidates the benchmark cache. A checkpoint's mtime cannot see a
#: change to the coder, and a stale cached rate against new code is indistinguishable
#: from a real measurement.
#:
#:   1 -- through 2026-08-31.
#:   2 -- 2026-09-01: `FactorizedPrior.update` reads its table extent off the
#:        density instead of off the learned quantiles, removing out-of-range
#:        escapes on the factorised hyper-latent streams.
CODER_VERSION = 2


class NeuralCodec:
    """Duck-types `eval.codecs.Codec`: `.name`, `.qualities`, `.encode_decode`.

    Not a subclass, because `Codec` is a dataclass built around PIL save kwargs
    and inheriting it would mean carrying three fields that mean nothing here.
    `measure_codec` only ever touches those three members.

    `qualities` are the rate points, labelled by the β they were trained at, so
    the benchmark's per-quality cache keys stay meaningful across runs.
    """

    def __init__(self, checkpoints: dict, *, name: str = "jpegai",
                 tier: str = "tierA", device=None,
                 header_bytes: int = DEFAULT_HEADER_BYTES,
                 note: str = ""):
        if not checkpoints:
            raise ValueError("no checkpoints")
        self.name = name
        self.tier = tier
        self.header_bytes = int(header_bytes)
        self.note = note or f"{len(checkpoints)} trained rate points"
        self._paths = dict(checkpoints)
        # Sort by β so the RD curve comes out monotone in rate, which the BD-rate
        # integration assumes.
        self.qualities = sorted(self._paths, key=lambda k: float(k))
        self._device = device
        self._models: dict = {}
        self.cache_name = f"{self.name}-{self.fingerprint()}"

    def fingerprint(self) -> str:
        """Short digest of *which weights these are*, for cache keying.

        The benchmark caches per (dataset, codec, image, quality). For JPEG that is
        safe: quality 80 means the same thing forever. For us, β 0.002 means
        "whatever beta0.002/final.pt holds right now", so retraining a point and
        re-running the benchmark would silently report the *old* model's rate and
        PSNR -- the worst kind of wrong number, because it looks like a normal run.

        Keyed on size and mtime rather than content: checkpoints are hundreds of
        megabytes and are written exactly once by training, so hashing their bytes
        on every benchmark run would cost more than the measurements. The failure
        mode this misses -- a file rewritten with identical size in the same
        nanosecond -- cannot happen to a torch.save.

        The weights are not the whole story, though: the same checkpoint produces
        different *bytes* whenever the entropy coder changes, and the coder is our
        code, not the checkpoint's. `CODER_VERSION` is therefore folded in, and must
        be bumped by any change that moves a bitstream. Without it, landing the
        table-extent fix (`entropy.FactorizedPrior._density_extent`, which cut
        `ladder_p6` beta 0.002 by 1.06%) would have left every cached row reporting
        the old rate against the new code -- the same silent-wrong-number failure
        this method exists to prevent, one level up.
        """
        import hashlib

        h = hashlib.sha256()
        h.update(f"coder{CODER_VERSION}|".encode())
        for label in self.qualities:
            st = Path(self._paths[label]).stat()
            h.update(f"{label}:{st.st_size}:{st.st_mtime_ns}|".encode())
        return h.hexdigest()[:8]

    # -- construction ------------------------------------------------------
    @classmethod
    def from_directory(cls, root, *, pattern: str = "final.pt", **kw) -> "NeuralCodec":
        """Collect one checkpoint per subdirectory named `beta<value>`.

        Matches the layout `train.runladder` writes:
            checkpoints/ladder/beta0.002/final.pt
            checkpoints/ladder/beta0.012/final.pt
        """
        root = Path(root)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        found = {}
        for sub in sorted(root.iterdir() if root.is_dir() else []):
            ck = sub / pattern
            if not ck.exists():
                continue
            label = sub.name[4:] if sub.name.startswith("beta") else sub.name
            found[label] = ck
        if not found:
            raise FileNotFoundError(
                f"no {pattern} under {root}/*/ -- run: "
                f"python -m jpegai.train.runladder"
            )
        return cls(found, **kw)

    # -- model cache -------------------------------------------------------
    def _model(self, quality):
        """Load and prepare one rate point's model. Cached; tables built once.

        `update()` is what turns learned distributions into the integer CDF tables
        the coder needs, and it must happen after loading and before any
        `compress`. Doing it once per model rather than once per image matters:
        it rebuilds 32 Gaussian rows plus one row per hyper channel.
        """
        if quality in self._models:
            return self._models[quality]

        import torch

        from jpegai.config import load_config
        from jpegai.models import build_any_model
        from jpegai.train.loop import load_checkpoint
        from jpegai.utils import pick_device

        if self._device is None:
            self._device = pick_device(None)
        path = self._paths[quality]
        blob = torch.load(path, map_location="cpu", weights_only=False)
        meta = blob.get("meta", {})
        cfg = load_config(meta.get("tier", self.tier))
        # `meta["model"]` selects the architecture, so a two-branch checkpoint
        # rebuilds as two-branch without the caller having to know. Defaulting to
        # "scale" keeps the Phase 3 checkpoints, written before the key existed,
        # loadable.
        model = build_any_model(cfg, meta.get("model", "scale")).to(self._device)
        load_checkpoint(path, model)
        model.eval()
        model.update(force=True)
        self._models[quality] = model
        return model

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return all(Path(p).exists() for p in self._paths.values())

    # -- the measurement ---------------------------------------------------
    def encode_decode(self, rgb: np.ndarray, quality) -> tuple[int, np.ndarray]:
        """Real bitstream in, real pixels out. Signature matches `Codec`."""
        import torch

        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected uint8 [H,W,3], got {rgb.dtype} {rgb.shape}")

        model = self._model(quality)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        x = x.float().div_(255.0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            packet = model.compress(x)
            nbytes = model.packet_bytes(packet) + self.header_bytes
            # Decode from the packet, never from the forward pass. If these two
            # ever disagree the reported size and the reported image would
            # describe different things, which is the one error that cannot be
            # detected downstream.
            x_hat = model.decompress(packet, device=self._device)["x_hat"]

        # Round rather than truncate: truncation biases every channel down by
        # ~0.5/255 and costs a measurable ~0.02 dB of PSNR for no reason.
        out = (x_hat.clamp(0, 1) * 255.0).round().to(torch.uint8)
        dec = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if dec.shape != rgb.shape:
            raise RuntimeError(f"decoded {dec.shape} != source {rgb.shape}")
        return int(nbytes), np.ascontiguousarray(dec)


def describe(root="checkpoints/ladder") -> None:
    try:
        codec = NeuralCodec.from_directory(root)
    except FileNotFoundError as exc:
        print(exc)
        return
    print(f"{codec.name}: {len(codec.qualities)} rate points from {root}")
    for q in codec.qualities:
        print(f"  beta {q:<10} {codec._paths[q]}")


if __name__ == "__main__":
    import sys

    describe(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ladder")
