#!/usr/bin/env python3
"""Per-image outlier scan over a benchmark's cache. Finds decoder desyncs.

`runbench` reports dataset averages, which is what BD-rate needs and exactly what
hides a single catastrophically broken image. The training-time round-trip check
found two of these in `ladder_p3f`: beta 0.002 at step 10,000 decoded to 15.71 dB
against 28.86 dB forward, and beta 0.2 at step 18,000 to 13.49 dB. Averaged over 24
images a 13 dB hole moves the mean by half a decibel and looks like a bad rate point.

The mechanism is an entropy-coder desync. `mean-scale` has no `coder_rows`, so it
indexes the sigma table from float sigma on both sides of the channel; when the two
sides round a boundary differently the decoder walks off the table and every
subsequent symbol is garbage. Same class as the fixed z_uv escape bug. It is not a
gradual quality loss -- it is total, on one image, at one rate point, and the bpp
stays perfectly normal because the *encoder* was fine.

So: flag any image whose PSNR sits far below its peers at the same quality, using a
median/MAD rule rather than a mean/sigma one, because the outlier we are hunting would
drag a mean and inflate a sigma enough to hide itself.

**Outcome, 2026-09-04.** `ladder_p3f` and the nine-point `ladder_p6` both scan clean on
all 24 Kodak images. So neither desync survived to `final.pt` -- they were transient
states of a mid-training checkpoint, and both affected beta are in the benched ladder,
so this is a real negative rather than an untested one. Every published rate point in
`results/p3f_kodak.*` and `results/p6_9pt.*` is a genuine measurement.

Run it with the project interpreter. Resolving a ladder name to its cache directory
goes through the checkpoint fingerprint, and that needs torch -- see `_fingerprint`
for why the directory name alone cannot identify a ladder.

    .venv/bin/python tools/scan_per_image.py --list
    .venv/bin/python tools/scan_per_image.py --codec ladder_p3f
    .venv/bin/python tools/scan_per_image.py --codec ladder_p3f --drop 3.0 --show 4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
CKPT = ROOT / "checkpoints"


def _fingerprint(ladder: str) -> tuple[str | None, str]:
    """The cache fingerprint `runbench` keys `ladder`'s checkpoints under.

    Returns `(fingerprint, "")`, or `(None, reason)` -- the reason matters, because
    "that ladder was never benched" and "this interpreter has no torch" need opposite
    responses from whoever ran the command.

    Matching on the directory name cannot work, which is worth stating plainly because
    the first version of this tool tried it. `runbench` names a neural cache directory
    `<codec>-<fingerprint>`, and the codec half is `jpegai` when exactly one `--neural`
    directory was benched, `ours-<dirname>` only when several were (runbench.py:576).
    The single-ladder case is the common one -- it is what every command in this
    project's notes runs -- and it leaves the ladder's name nowhere in the path.

    The fingerprint is the only real link, so ask `NeuralCodec` for it rather than
    reimplementing the recipe here: it folds in `CODER_VERSION` as well as each
    checkpoint's size and mtime, and a copy of that logic in a scan tool would drift
    and start reading a stale directory as if it were the fresh one.
    """
    root = Path(ladder) if Path(ladder).is_dir() else CKPT / ladder
    if not root.is_dir():
        return None, f"{root} is not a directory, so it is not a ladder"
    # Running `tools/scan_per_image.py` puts `tools/` on sys.path, not the repo root,
    # so `import jpegai` fails from here without this even under `.venv/bin/python`.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from jpegai.eval.neural import NeuralCodec
    except Exception as exc:
        return None, (f"cannot import jpegai.eval.neural ({exc}); run this with the "
                      f"project interpreter, .venv/bin/python")
    try:
        return NeuralCodec.from_directory(root).fingerprint(), ""
    except Exception as exc:
        return None, f"{root} holds no beta*/final.pt ({exc})"


def _inventory(dirs: list[Path]) -> str:
    """Every cache directory with what it actually holds.

    A bare list of `jpegai-<hex>` names is no help in picking one, which is exactly
    what this tool printed the first time it failed to resolve. The beta labels and
    the image count identify a ladder on sight.
    """
    out = ["cached codecs for this dataset:"]
    for d in dirs:
        files = sorted(d.glob("*.json"))
        quals: list[str] = []
        if files:
            try:
                blob = json.loads(files[0].read_text())
                quals = sorted(blob.get("points", blob), key=lambda q: float(q))
            except Exception:
                pass
        stamp = time.strftime("%b %d %H:%M", time.localtime(d.stat().st_mtime))
        out.append(f"  {d.name:31} {len(files):3} images  {stamp}"
                   + (f"  q={','.join(quals)}" if quals else ""))
    return "\n".join(out) + "\n"


def _dirs(dataset: str) -> list[Path]:
    root = CACHE / dataset
    if not root.is_dir():
        sys.exit(f"no cache for dataset {dataset!r} under {CACHE}")
    return sorted((d for d in root.iterdir() if d.is_dir()),
                  key=lambda d: d.stat().st_mtime, reverse=True)


def _resolve(dataset: str, codec: str) -> Path:
    """Find the cache directory holding `codec`'s per-image points.

    Four ways, most exact first: the literal directory name, the checkpoint
    fingerprint, then a substring, newest mtime winning a tie because the older
    fingerprints for one ladder are previous training states.
    """
    dirs = _dirs(dataset)
    exact = CACHE / dataset / codec
    if exact.is_dir():
        return exact

    fp, why = _fingerprint(codec)
    if fp:
        for d in dirs:
            if d.name.endswith(f"-{fp}"):
                return d

    hits = [d for d in dirs if codec in d.name]
    if len(hits) > 1:
        print(f"note: {len(hits)} directories match {codec!r}; using the newest "
              f"({hits[0].name})", file=sys.stderr)
    if hits:
        return hits[0]

    tail = (f"\n{codec!r} fingerprints to {fp!r}, which is not among them: those "
            f"checkpoints have never\nbeen benched on {dataset}, or were rewritten "
            f"since. Run runbench first.\n" if fp else
            f"\nNo fingerprint either: {why}.\n")
    sys.exit(f"no cache directory for {codec!r} in {CACHE / dataset}\n\n"
             + _inventory(dirs) + tail)


def _load(d: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(d.glob("*.json")):
        blob = json.loads(f.read_text())
        pts = blob.get("points", blob)
        if isinstance(pts, dict):
            out[f.stem] = pts
    if not out:
        sys.exit(f"{d} holds no cached points")
    return out


def scan(dataset: str, codec: str, drop: float, show: int, field: str,
         rate_tol: float) -> int:
    d = _resolve(dataset, codec)
    per_image = _load(d)
    quals = sorted({q for pts in per_image.values() for q in pts},
                   key=lambda q: float(q))
    print(f"{d.relative_to(CACHE)}   {len(per_image)} images, "
          f"{len(quals)} rate point(s), flagging {field} more than {drop:.1f} dB "
          f"below the median\n")

    bad = 0
    for q in quals:
        rows = [(n, p[q]) for n, p in per_image.items() if q in p and field in p[q]]
        if len(rows) < 3:
            continue
        vals = [e[field] for _, e in rows]
        med = statistics.median(vals)
        # MAD, not stdev: one 13 dB hole inflates a stdev enough to stop being an
        # outlier by its own measure.
        mad = statistics.median([abs(v - med) for v in vals]) or 1e-9
        rows.sort(key=lambda r: r[1][field])

        bmed = statistics.median([e["bpp"] for _, e in rows])
        # Two conditions, not one. A hard image -- kodim13, the rocky stream -- sits
        # 5-7 dB below the median on EVERY codec including JPEG, which cannot desync,
        # and it does so while spending 45-92% more bits than the median: the encoder
        # tried and the content is simply expensive. A desync spends the NORMAL number
        # of bits and returns garbage, because the encoder was fine and only the
        # decoder walked off the table. So the rate condition is what separates a
        # broken codec from a difficult picture, and without it this scan just
        # rediscovers kodim13 nine times.
        flags = [(n, e) for n, e in rows
                 if med - e[field] > drop and e["bpp"] <= bmed * (1.0 + rate_tol)]
        head = f"beta {q:<8} median {field} {med:6.2f} dB  MAD {mad:4.2f}"
        if flags:
            bad += len(flags)
            print(f"{head}   *** {len(flags)} OUTLIER(S) ***")
        else:
            print(f"{head}   clean   (worst {rows[0][0]} {rows[0][1][field]:.2f})")
        for n, e in flags:
            print(f"    {n:12} {field} {e[field]:6.2f} "
                  f"({e[field] - med:+.2f} dB, {(med - e[field]) / mad:5.1f} MAD)"
                  f"  bpp {e['bpp']:.4f} ({e['bpp'] / bmed - 1:+.1%} vs median)"
                  f"  psnr_y {e.get('psnr_y', float('nan')):.2f}"
                  f"  ms_ssim {e.get('ms_ssim', float('nan')):.4f}")
        for n, e in rows[:show] if not flags else []:
            print(f"    {n:12} {field} {e[field]:6.2f}  bpp {e['bpp']:.4f}")

    print()
    if bad:
        print(f"{bad} outlier(s). A drop of >5 dB at normal bpp is a decoder desync, "
              f"not a hard image:\nthe encoder's rate was fine, so the two sides "
              f"disagreed about the sigma table.\nDo not put this ladder in a results "
              f"table until it scans clean.")
    else:
        print(f"clean: no image is more than {drop:.1f} dB below its rate point's "
              f"median.\nThe dataset averages in the bench table are safe to quote.")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codec",
                    help="a ladder directory under checkpoints/ (resolved through its "
                         "fingerprint), a cache directory name, or a substring of one")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="show every cached codec for the dataset, with its rate "
                         "points, and exit")
    ap.add_argument("--dataset", default="kodak")
    ap.add_argument("--drop", type=float, default=5.0,
                    help="dB below the median that counts as an outlier (default 5)")
    ap.add_argument("--field", default="psnr",
                    help="metric to scan (default psnr; psnr_y isolates luma)")
    ap.add_argument("--show", type=int, default=2,
                    help="worst N images to print for a clean rate point")
    ap.add_argument("--rate-tol", type=float, default=0.15, dest="rate_tol",
                    help="an image is only an outlier if its bpp is within this "
                         "fraction above the rate point's median (default 0.15). "
                         "Hard images cost more bits; desyncs do not")
    a = ap.parse_args()
    if a.list_only:
        print(_inventory(_dirs(a.dataset)), end="")
        sys.exit(0)
    if not a.codec:
        ap.error("give --codec, or --list to see what is cached")
    sys.exit(scan(a.dataset, a.codec, a.drop, a.show, a.field, a.rate_tol))
