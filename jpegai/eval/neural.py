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

`VariableRateCodec` at the bottom is the same wrapper with the quality axis moved
from *which checkpoint* to *which `Delta_beta`*, which is how Phase 8's one-model
ladder gets measured by the identical harness. It counts the same four header bytes,
even though a real variable-rate decoder also needs the 7-bit `Delta_beta` field
itself: rounding that up to a fifth byte would move every rate point by 0.005% and
make the two curves non-comparable for no gain in honesty, so the field is noted
here and the constant left alone.
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
        self.qualities = self._sorted_qualities()
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
        for label, path in self._fingerprint_rows():
            st = Path(path).stat()
            h.update(f"{label}:{st.st_size}:{st.st_mtime_ns}|".encode())
        return h.hexdigest()[:8]

    def _fingerprint_rows(self) -> list[tuple[str, Path]]:
        """`(label, checkpoint)` pairs whose size and mtime identify these weights.

        A seam, so `VariableRateCodec` -- whose quality points are `Delta_beta` values
        that all resolve to the *same* checkpoint -- can hash that checkpoint once
        instead of once per rung. Hashing per rung would work but would make the digest
        depend on the point list, so adding a tenth rate point would invalidate the
        cached measurements of the nine that did not change.
        """
        return [(str(q), Path(self._paths[q])) for q in self.qualities]

    # -- seams for the variable-rate subclass ------------------------------
    def _sorted_qualities(self) -> list:
        """The quality axis, ascending in rate. β labels for a trained ladder.

        Sorted numerically because the BD-rate integration assumes the RD curve arrives
        monotone in rate, and `"0.075"` sorts before `"0.2"` as a string.

        A seam and not an inline `sorted` because `VariableRateCodec` keys `_paths` by one
        fixed string -- `float("vr")` raises -- and a subclass cannot repair that after the
        fact: `__init__` sorts, and then fingerprints, before the subclass regains control.
        """
        return sorted(self._paths, key=float)

    def _checkpoint_label(self, quality):
        """Which checkpoint a quality point comes from. Identity for a trained ladder.

        One checkpoint per rate point is the whole shape of a Phase 5/6 ladder, so this
        is the identity here and only interesting in `VariableRateCodec`, where nine
        rate points share one set of weights.
        """
        return quality

    def _compress_kwargs(self, quality) -> dict:
        """Extra arguments for `model.compress`. Empty for a fixed-rate ladder.

        Exists so `encode_decode` stays one function. That function holds the two
        decisions this module's docstring is about -- bytes come from the bitstream, the
        header is counted -- and a variable-rate copy of it would be the easiest way to
        end up reporting rates measured two different ways on one plot.
        """
        return {}

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

        Cached on the *checkpoint* label rather than on the quality point, so a
        variable-rate sweep whose nine rungs share one set of weights loads and
        `update()`s them once.
        """
        label = self._checkpoint_label(quality)
        if label in self._models:
            return self._models[label]

        import torch

        from jpegai.config import load_config
        from jpegai.models import build_any_model
        from jpegai.train.loop import load_checkpoint
        from jpegai.utils import pick_device

        if self._device is None:
            self._device = pick_device(None)
        path = self._paths[label]
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
        self._models[label] = model
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
            packet = model.compress(x, **self._compress_kwargs(quality))
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


class VariableRateCodec(NeuralCodec):
    """One checkpoint, the whole rate ladder, measured through `Delta_beta`.

        python -m jpegai.eval.runbench --neural checkpoints/vr_m0 --delta-beta

    This is Phase 8's headline claim turned into a measurement. A `NeuralCodec` needs
    one trained checkpoint per rate point, which is what every phase up to 6 produced
    and what costs four trainings for four points. Here the quality axis is the header
    field itself: nine `Delta_beta` values, one set of weights, and the RD curve comes
    out of the *same* `measure_codec` loop with the *same* metrics and the *same*
    BD-rate integration as the four-checkpoint version. That is the only way the
    comparison means anything -- "variable rate costs us X% BD-rate" is a statement
    about two curves, so both curves have to be produced identically.

    Quality labels are the signed integers, `-1069` through `+702`, rather than an
    equivalent beta. Two reasons. `Delta_beta` is what the bitstream actually carries,
    so a labelled point is a decodable file rather than a training intention; and
    converting to beta would need the anchor, which lives in the checkpoint's own
    metadata and would silently change the labels -- and therefore invalidate the
    benchmark cache -- if a later run were pointed at a checkpoint trained at a
    different `beta_train`.

    Rate is monotone in `Delta_beta` (the training loop's `delta_beta_check` gates
    exactly this), so sorting the labels ascending sorts the curve ascending in bpp,
    which is what the BD-rate integration assumes.
    """

    #: The single checkpoint's key in `_paths`. Any fixed string works; this one shows
    #: up in the cache fingerprint, so it is spelled to be recognisable there.
    LABEL = "vr"

    def __init__(self, checkpoint, points, *, name: str = "jpegai-vr",
                 tier: str = "tierA", device=None,
                 header_bytes: int = DEFAULT_HEADER_BYTES, note: str = ""):
        pts = sorted({int(d) for d in points})
        if not pts:
            raise ValueError("no Delta_beta points to sweep")
        # Refuse rather than clamp. A clamped point is a duplicate of the end rung, so
        # the curve would silently carry two identical rate points -- which the PCHIP
        # interpolation in `bdrate.py` cannot fit, and which would look like a codec
        # that stopped responding to the rate request rather than like a bad argument.
        from jpegai.models.gain import DELTA_BETA_MAX, DELTA_BETA_MIN
        out = [d for d in pts if not DELTA_BETA_MIN <= d <= DELTA_BETA_MAX]
        if out:
            raise ValueError(
                f"Delta_beta must be within [{DELTA_BETA_MIN}, {DELTA_BETA_MAX}] -- the "
                f"header field's normative range -- but got {out}")
        # Before `super().__init__`, which reads it back through `_sorted_qualities`.
        self._points = pts
        super().__init__({self.LABEL: checkpoint}, name=name, tier=tier,
                         device=device, header_bytes=header_bytes,
                         note=note or (f"one checkpoint swept over {len(pts)} "
                                       f"Delta_beta values, {pts[0]:+d}..{pts[-1]:+d}"))

    def _sorted_qualities(self):
        """The `Delta_beta` rungs, already sorted and de-duplicated by `__init__`.

        Overriding this rather than reassigning `qualities` afterwards because the base
        `__init__` both sorts *and* fingerprints, so a late fix would leave the interim
        `float("vr")` to raise.
        """
        return list(self._points)

    def _fingerprint_rows(self):
        """One row, not one per rung -- see the base class's note on this seam."""
        return [(self.LABEL, Path(self._paths[self.LABEL]))]

    def _checkpoint_label(self, quality):
        return self.LABEL

    def _compress_kwargs(self, quality) -> dict:
        return {"delta_beta": int(quality)}

    def anchor_beta(self) -> float | None:
        """The checkpoint's own `beta`, for the report's note. `None` if unrecorded.

        Read from metadata rather than taken as an argument because it is not a knob:
        eq. (9) divides by the beta the weights were trained at, and only the checkpoint
        knows that. It is reported and not used -- the sweep is in `Delta_beta` units,
        which need no anchor -- so a checkpoint written before the key existed still
        measures fine and simply says nothing about its anchor.
        """
        import torch

        blob = torch.load(self._paths[self.LABEL], map_location="cpu",
                          weights_only=False)
        meta = blob.get("meta", {})
        beta = meta.get("anchor_beta", meta.get("beta"))
        return None if beta is None else float(beta)

    @classmethod
    def from_checkpoint(cls, path, points=None, *, tier: str = "tierA",
                        pattern: str = "final.pt", **kw) -> "VariableRateCodec":
        """Accept either the `.pt` itself or the directory holding it.

        Directory form so `--neural checkpoints/vr_m0` reads the same way with and
        without `--delta-beta`, which is what makes the two runs easy to diff.

        `points` defaults to `config.rate.beta_eval_points` -- nine values spanning the
        clamp with the anchor included -- so the default sweep is the one the config
        documents rather than one chosen at the command line.
        """
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_dir():
            ck = p / pattern
            if not ck.exists():
                raise FileNotFoundError(
                    f"no {pattern} in {p} -- a --delta-beta sweep needs one "
                    f"variable-rate checkpoint, not a ladder of them")
            p = ck
        if not p.exists():
            raise FileNotFoundError(f"no checkpoint at {p}")
        if points is None:
            from jpegai.config import load_config
            points = load_config(tier).rate.beta_eval_points
        return cls(p, points, tier=tier, **kw)

    def check_gain_unit(self) -> None:
        """Fail early and by name if the checkpoint has no gain unit.

        Without this the sweep runs, every rung ignores its `delta_beta` argument or
        raises deep inside `compress`, and the failure reads as a coder bug. The
        likeliest cause is pointing `--delta-beta` at a Phase 5 or Phase 6 ladder, which
        is an easy mistake precisely because the directory layout is the same.
        """
        model = self._model(self.qualities[0])
        if not getattr(model, "gain", False):
            raise ValueError(
                f"{self._paths[self.LABEL]} is not a variable-rate checkpoint "
                f"(no gain unit), so Delta_beta has nothing to act on. Train one with "
                f"--model twobranch-vr --stage IV, or drop --delta-beta to measure "
                f"this as an ordinary rate ladder")


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
