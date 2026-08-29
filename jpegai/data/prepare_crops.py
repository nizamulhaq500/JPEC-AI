"""One-time extraction of fixed-size training crops.

    python -m jpegai.data.prepare_crops
    python -m jpegai.data.prepare_crops --sources div2k_train --per-image 8
    python -m jpegai.data.prepare_crops --crop 256 --limit 20      # quick trial

Why crop offline instead of cropping on the fly in the dataloader:

* **Decode cost.** A DIV2K PNG is ~2040x1356. Decoding 2.8 megapixels to keep
  0.065 of them wastes ~97% of the dataloader's CPU. On a machine with no CUDA,
  where the GPU is not the bottleneck, that is the difference between a training
  step waiting on data and not.
* **Reproducibility.** The crop set becomes a fixed, inspectable artifact with a
  manifest. Two runs see identical data, so a loss curve difference means a code
  difference.
* **Flat-crop filtering happens once**, not every epoch.

Two rules this script follows that matter for a *compression* model specifically:

1. **Never resize.** Downsampling destroys sensor noise and high-frequency detail
   -- exactly the content that is expensive to code. A model trained on
   downsampled data learns an easier problem than it will be asked to solve, and
   reports flattering bitrates that collapse on real photographs. Crops are taken
   at native resolution or not at all.
2. **Crop size must be a multiple of the total downsampling factor** (64: four
   analysis stages then two hyper stages). Otherwise the latent grid does not
   divide evenly and training silently exercises padding logic that inference on
   real images will hit differently. 256 = 4 x 64.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from jpegai.config import PROJECT_ROOT

#: Named source datasets -> directory, relative to the project root.
SOURCES: dict[str, str] = {
    "div2k_train": "data/div2k/DIV2K_train_HR",
    "div2k_valid": "data/div2k/DIV2K_valid_HR",
    "flickr2k": "data/flickr2k/Flickr2K",
    "kodak": "data/kodak",
}

#: Defaults per source. Flickr2K has ~3.3x more images than DIV2K train, so fewer
#: crops each keeps the mixture roughly balanced rather than letting one dataset's
#: photographic style dominate.
PER_IMAGE: dict[str, int] = {
    "div2k_train": 8,
    "div2k_valid": 2,
    "flickr2k": 4,
    "kodak": 0,          # test set -- never train on it
}

IMAGE_EXT = {".png", ".ppm", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def _job(task: tuple) -> dict:
    """Extract crops from one source image. Runs in a worker process."""
    src, out_dir, crop, n, min_std, tries, compress = task
    from PIL import Image

    src, out_dir = Path(src), Path(out_dir)
    result = {"src": src.name, "written": [], "skipped_flat": 0, "bytes": 0,
              "error": None}

    targets = [out_dir / f"{src.stem}_c{i:02d}.png" for i in range(n)]
    if all(t.exists() for t in targets):
        result["written"] = [t.name for t in targets]
        result["bytes"] = sum(t.stat().st_size for t in targets)
        result["cached"] = True
        return result

    try:
        img = Image.open(src)
        img = img.convert("RGB") if img.mode != "RGB" else img
        arr = np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        result["error"] = f"open failed: {exc}"
        return result

    h, w = arr.shape[:2]
    if h < crop or w < crop:
        result["error"] = f"too small ({w}x{h} < {crop})"
        return result

    # Seed from a stable hash of the filename, so the crop set is identical across
    # runs, machines and worker counts. Python's built-in hash() is randomised per
    # process and would silently break that.
    rng = np.random.default_rng(zlib.crc32(src.name.encode()) & 0xFFFFFFFF)

    for target in targets:
        if target.exists():
            result["written"].append(target.name)
            result["bytes"] += target.stat().st_size
            continue

        chosen = None
        for _ in range(tries):
            y = int(rng.integers(0, h - crop + 1))
            x = int(rng.integers(0, w - crop + 1))
            patch = arr[y:y + crop, x:x + crop]
            # Reject near-flat crops: clear sky, blown highlights, solid borders.
            # They cost a full training step and teach the model almost nothing,
            # because they are already nearly free to code.
            if patch.std() >= min_std:
                chosen = patch
                break
        if chosen is None:
            result["skipped_flat"] += 1
            continue

        Image.fromarray(chosen).save(target, format="PNG", compress_level=compress)
        result["written"].append(target.name)
        result["bytes"] += target.stat().st_size

    return result


def prepare(
    sources: list[str],
    out_root: Path,
    *,
    crop: int = 256,
    per_image: dict[str, int] | None = None,
    min_std: float = 8.0,
    tries: int = 12,
    workers: int = 0,
    limit: int | None = None,
    compress: int = 3,
) -> dict:
    per_image = per_image or PER_IMAGE
    manifest: dict = {"crop": crop, "min_std": min_std, "sources": {}}
    grand_total = grand_bytes = 0

    for name in sources:
        rel = SOURCES.get(name)
        root = (PROJECT_ROOT / rel) if rel else Path(name)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        n_per = per_image.get(name, 4)

        # The output subdirectory is derived from the *basename*, never from the
        # raw source string. `out_root / name` with an absolute `name` resolves to
        # `name` itself -- which would write crops straight into the source
        # dataset. Caught by tests/test_data.py; do not "simplify" this back.
        label = name if name in SOURCES else (root.name or "source")

        if n_per <= 0:
            print(f"{label:14} skipped (per-image = 0; this is a test set)")
            continue
        if not root.is_dir():
            print(f"{label:14} MISSING  {root}\n"
                  f"{'':14} run `bash setup.sh` to download it")
            manifest["sources"][label] = {"status": "missing", "root": str(root)}
            continue

        images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        if limit:
            images = images[:limit]
        if not images:
            print(f"{label:14} no images under {root}")
            continue

        out_dir = out_root / label
        if out_dir.resolve() == root.resolve():
            raise ValueError(
                f"refusing to write crops into the source directory ({root}). "
                "Choose a different --out."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks = [(str(p), str(out_dir), crop, n_per, min_std, tries, compress)
                 for p in images]

        print(f"{label:14} {len(images):5} images x {n_per} crops -> {out_dir.name}/",
              flush=True)
        t0 = time.perf_counter()
        results = []
        if workers and workers > 1:
            with Pool(workers) as pool:
                for i, r in enumerate(pool.imap_unordered(_job, tasks, chunksize=4), 1):
                    results.append(r)
                    if i % 100 == 0 or i == len(tasks):
                        print(f"{'':14} {i}/{len(tasks)}", flush=True)
        else:
            for i, t in enumerate(tasks, 1):
                results.append(_job(t))
                if i % 100 == 0 or i == len(tasks):
                    print(f"{'':14} {i}/{len(tasks)}", flush=True)
        dt = time.perf_counter() - t0

        written = sum(len(r["written"]) for r in results)
        cached = sum(1 for r in results if r.get("cached"))
        flat = sum(r["skipped_flat"] for r in results)
        nbytes = sum(r["bytes"] for r in results)
        errors = [(r["src"], r["error"]) for r in results if r["error"]]

        grand_total += written
        grand_bytes += nbytes
        manifest["sources"][label] = {
            "status": "ok", "root": str(root), "out": str(out_dir),
            "images": len(images), "per_image": n_per, "crops": written,
            "images_already_done": cached, "rejected_flat": flat,
            "bytes": nbytes, "seconds": round(dt, 1),
            "errors": errors[:20], "n_errors": len(errors),
        }
        print(f"{'':14} {written} crops, {nbytes / 2**20:.0f} MiB, {dt:.0f}s"
              + (f", {cached} images already done" if cached else "")
              + (f", {flat} flat crops rejected" if flat else "")
              + (f", {len(errors)} errors" if errors else ""))
        for src, err in errors[:5]:
            print(f"{'':16} {src}: {err}")

    manifest["total_crops"] = grand_total
    manifest["total_bytes"] = grand_bytes
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m jpegai.data.prepare_crops",
        description="Extract fixed-size training crops (one-time).",
    )
    ap.add_argument("--sources", default="div2k_train,flickr2k",
                    help="comma-separated: " + ", ".join(SOURCES))
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--crop", type=int, default=256,
                    help="must be a multiple of 64 (the total downsampling factor)")
    ap.add_argument("--per-image", type=int, default=None,
                    help="override the per-source defaults")
    ap.add_argument("--min-std", type=float, default=8.0,
                    help="reject crops flatter than this (0-255 scale)")
    ap.add_argument("--tries", type=int, default=12,
                    help="attempts to find a non-flat crop before giving up")
    ap.add_argument("--workers", type=int, default=0,
                    help="processes; 0 or 1 runs serially")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N images per source (for a quick trial)")
    ap.add_argument("--compress", type=int, default=3, choices=range(10),
                    help="PNG compress_level; 3 trades ~15%% size for ~3x write speed")
    args = ap.parse_args(argv)

    if args.crop % 64:
        print(f"error: --crop {args.crop} is not a multiple of 64.\n"
              "The architecture downsamples by 64 (4 analysis + 2 hyper stages); a\n"
              "non-multiple makes the latent grid uneven and exercises padding\n"
              "paths in training that inference will hit differently.")
        return 2

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    per_image = dict(PER_IMAGE)
    if args.per_image is not None:
        per_image = {s: args.per_image for s in sources}

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"crop {args.crop}x{args.crop}, min std {args.min_std}, "
          f"{args.workers or 1} worker(s) -> {out_root}\n")
    manifest = prepare(
        sources, out_root,
        crop=args.crop, per_image=per_image, min_std=args.min_std,
        tries=args.tries, workers=args.workers, limit=args.limit,
        compress=args.compress,
    )

    mpath = out_root / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1))

    total, nbytes = manifest["total_crops"], manifest["total_bytes"]
    print(f"\n{total} crops, {nbytes / 2**30:.2f} GiB total")
    print(f"manifest {mpath.relative_to(PROJECT_ROOT)}")
    if total == 0:
        print("\nNo crops written. The source datasets are not downloaded yet:")
        print("    bash setup.sh")
        return 1
    steps_per_epoch = total // 8
    print(f"\nAt batch 8 that is {steps_per_epoch:,} steps per epoch; the Tier A "
          f"budget of 400,000 iterations is {400_000 / max(steps_per_epoch, 1):.1f} epochs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
