"""Measure the luma-only decode fast path -- Phase 4's fourth acceptance criterion.

The criterion is deliberately worded as "at reduced decode time you can actually
measure and report", not "it works", because a luma-only decoder is trivially easy
to write wrong in a way that produces the correct pixels and saves nothing: decode
the chroma stream, then discard it. That version passes every correctness test.

So this measures three separate things, and the *rate* saving and the *time* saving
are reported apart from each other, because they come from different halves of the
work:

* **Bytes not received.** The chroma streams are never read, so a transport that
  knows the consumer is luma-only can drop them.
* **Entropy decoding not done.** Two of the four streams, and their CDF lookups.
* **Synthesis not run.** `g_s_uv` -- 26.2 of the model's 132.4 decoder kMAC/pxl at
  Tier A -- plus the chroma upsample inside `merge_planes`.

The expected saving is *not* proportional to the chroma branch's share of the
parameters. At Tier A the skipped convolutions are 27.0 of 132.4 decoder kMAC/pxl
(`g_s_uv` 26.2 + `h_s_uv` 0.8) = 20.4%, because `g_s_y` alone is 102 and it still
runs. So a saving in that neighbourhood is what the arithmetic predicts.

Measurement, though, comes out slightly *above* the arithmetic share -- ~24% on
CPU -- and the reason is worth stating rather than rounding away: rANS decoding is
not multiply-accumulate work, so it is invisible to a kMAC count, and skipping two
of the four streams removes real wall-clock the 20.4% figure never included. The
arithmetic share is therefore a rough anchor, not a ceiling. Anyone quoting one
number should quote the *rate* saving, which is the larger and more robust of the
two: chroma is a third of the payload on every image, on every device.

Run:
    .venv/bin/python -m jpegai.eval.lumaonly --device cpu
    .venv/bin/python -m jpegai.eval.lumaonly --checkpoint checkpoints/tb/final.pt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

from jpegai.config import load_config
from jpegai.models import build_any_model
from jpegai.utils import describe_device, pick_device, seed_everything


def _time_decode(model, packet, *, luma_only: bool, device, repeats: int,
                 warmup: int = 2) -> list[float]:
    """Per-call wall time in ms. Median of `repeats`, after `warmup` untimed calls.

    Median, not mean: on a laptop a single scheduling hiccup can double one sample,
    and a mean over ten samples is then reporting the hiccup. Warmup matters because
    the first call allocates the workspace and, on MPS, compiles kernels -- charging
    that to whichever variant happens to run first would invent a difference.
    """
    for _ in range(warmup):
        model.decompress(packet, device=device, luma_only=luma_only)
    out = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        model.decompress(packet, device=device, luma_only=luma_only)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        out.append(1000.0 * (time.perf_counter() - t0))
    return out


@torch.no_grad()
def measure(model, device, *, sizes=((512, 768), (1024, 1024)), repeats: int = 9
            ) -> list[dict]:
    """One row per image size. Model must already be `eval()`ed and `update()`d."""
    rows = []
    for h, w in sizes:
        x = torch.rand(1, 3, h, w, device=device)
        packet = model.compress(x)

        full_b = model.packet_bytes(packet)
        luma_b = model.packet_bytes(packet, luma_only=True)
        t_full = _time_decode(model, packet, luma_only=False, device=device,
                              repeats=repeats)
        t_luma = _time_decode(model, packet, luma_only=True, device=device,
                              repeats=repeats)

        # Correctness alongside the timing, in the same call: a timing harness that
        # reports a speedup for a decoder that stopped producing the right luma is
        # worse than no harness.
        ref = model.decompress(packet, device=device)
        part = model.decompress(packet, device=device, luma_only=True)
        rgb = part["x_hat"]
        rows.append({
            "size": f"{w}x{h}",
            "pixels": h * w,
            "bytes_full": full_b,
            "bytes_luma": luma_b,
            "bpp_full": full_b * 8 / (h * w),
            "ms_full": statistics.median(t_full),
            "ms_luma": statistics.median(t_luma),
            "ms_full_spread": max(t_full) - min(t_full),
            "ms_luma_spread": max(t_luma) - min(t_luma),
            "luma_identical": bool(torch.equal(part["luma"], ref["luma"])),
            # R == G == B, the visible half of "correct luma, grey chroma". Checked
            # after `_to_rgb`, so it also covers the inverse colour transform: a sign
            # error there would still give a bit-exact `luma` plane.
            "grey_max_dev": float((rgb - rgb.mean(dim=1, keepdim=True)).abs().max()),
        })
    return rows


def report(rows: list[dict], *, decoder_kmac: dict | None = None) -> str:
    lines = [
        f"  {'size':>10} {'bytes':>9} {'-> luma':>9} {'rate':>7}   "
        f"{'ms full':>8} {'ms luma':>8} {'time':>7} {'noise':>6}   luma  grey",
        "  " + "-" * 89,
    ]
    for r in rows:
        rate = 100.0 * (1 - r["bytes_luma"] / r["bytes_full"])
        tsave = 100.0 * (1 - r["ms_luma"] / r["ms_full"])
        # Spread over the difference. Above 1 means the run-to-run jitter is as big
        # as the effect and the time column says nothing; printing it beats printing
        # a confident-looking percentage that a rerun would contradict.
        gap = r["ms_full"] - r["ms_luma"]
        noise = max(r["ms_full_spread"], r["ms_luma_spread"]) / max(abs(gap), 1e-9)
        lines.append(
            f"  {r['size']:>10} {r['bytes_full']:>9,} {r['bytes_luma']:>9,} "
            f"{-rate:>6.1f}%   {r['ms_full']:>8.1f} {r['ms_luma']:>8.1f} "
            f"{-tsave:>6.1f}% {noise:>5.2f}x   "
            f"{'ok' if r['luma_identical'] else 'MISMATCH'}  "
            f"{r['grey_max_dev']:.1e}"
        )
    if any(max(r["ms_full_spread"], r["ms_luma_spread"])
           > abs(r["ms_full"] - r["ms_luma"]) for r in rows):
        lines.append("  WARNING: noise >= 1x on some row -- the time column is not "
                     "reportable. Rerun\n           with more --repeats on an idle "
                     "machine.")
    if decoder_kmac:
        total = sum(decoder_kmac.values())
        skipped = sum(v for k, v in decoder_kmac.items() if k.endswith("_uv"))
        share = 100.0 * skipped / max(total, 1e-9)
        measured = statistics.median(
            100.0 * (1 - r["ms_luma"] / r["ms_full"]) for r in rows)
        lines += [
            "",
            f"  convolutions skipped: {skipped:.1f} of {total:.1f} decoder kMAC/pxl "
            f"= {share:.1f}%   (g_s_y, 102.0, still runs)",
            f"  time saved:           {measured:.1f}%",
        ]
        # Above the arithmetic share is the expected outcome, not a suspicious one --
        # but say why, because "the speedup beats the FLOP count" reads like a
        # measurement error unless the missing term is named.
        if measured > share:
            lines.append(
                "  -- above the arithmetic share because rANS decoding is not "
                "multiply-accumulate\n     work: two of the four streams go unread, "
                "and a kMAC count cannot see that.")
        else:
            lines.append(
                "  -- at or below the arithmetic share: the two skipped entropy "
                "streams are cheap\n     relative to the convolutions on this device.")
        lines.append(
            "  Quote the rate saving, not the time saving: it is larger, and it does "
            "not\n  depend on the device.")
    return "\n".join(lines)


def decoder_kmac(model, crop: int = 256) -> dict:
    """Decoder-side kMAC/pxl per part, keyed so `_uv` parts can be summed."""
    from jpegai.utils import macs_breakdown

    parts = model.summary_parts()
    macs = macs_breakdown(model, (1, model.in_channels, crop, crop),
                          parts=[(n, m) for n, m, _ in parts])
    return {n: macs.get(n, 0.0) / 1e3 for n, _, is_dec in parts if is_dec}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jpegai.eval.lumaonly")
    ap.add_argument("--tier", default="tierA")
    ap.add_argument("--checkpoint", default=None,
                    help="trained weights; random init is fine for timing, since "
                         "decode cost does not depend on what was learned -- but "
                         "the byte counts do, so quote those from a trained model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--sizes", default="512x768,1024x1024")
    args = ap.parse_args(argv)

    seed_everything(0)
    device = pick_device(args.device, verbose=True)
    cfg = load_config(args.tier)

    kind = "twobranch"
    if args.checkpoint:
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        kind = blob.get("meta", {}).get("model", "twobranch")
    # Any of the two-branch kinds will do -- the fast path is a property of the
    # branch split, not of which entropy model sits inside a branch. Matching on the
    # prefix means Phase 5's checkpoints work here without another edit.
    if not kind.startswith("twobranch"):
        print(f"error: {args.checkpoint} is a {kind!r} model; the luma-only path "
              "only exists on the two-branch codec", file=sys.stderr)
        return 2

    model = build_any_model(cfg, kind).to(device)
    if args.checkpoint:
        from jpegai.train.loop import load_checkpoint
        load_checkpoint(Path(args.checkpoint), model)
    model.eval()
    model.update(force=True)

    sizes = []
    for tok in args.sizes.split(","):
        w, h = tok.lower().split("x")
        sizes.append((int(h), int(w)))

    print(f"\nluma-only decode -- {model.summary_title()}")
    print(f"device {describe_device(device)}  "
          f"weights {'trained' if args.checkpoint else 'random init'}  "
          f"median of {args.repeats}\n")
    rows = measure(model, device, sizes=tuple(sizes), repeats=args.repeats)
    print(report(rows, decoder_kmac=decoder_kmac(model, crop=cfg.train.crop)))
    print()
    ok = all(r["luma_identical"] and r["grey_max_dev"] < 1e-5 for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
