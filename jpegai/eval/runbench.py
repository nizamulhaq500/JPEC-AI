"""Rate-distortion benchmark harness.

    python -m jpegai.eval.runbench --codecs jpeg,webp,avif --dataset kodak

Produces, in `results/`:

    bench_<dataset>.json     every rate/metric point, per codec and per quality
    bench_<dataset>.png      RD curve grid, one subplot per metric
    bench_<dataset>.md       BD-rate table, ready to paste into the report

This is Phase 1's deliverable and it exists **before** any model code, on purpose.
A codec project without a trustworthy measuring instrument produces numbers nobody
should believe, including its author. Everything here -- the datasets, the metric
set, the aggregation convention, the BD-rate anchor -- is fixed now so that when
our own decoder arrives it drops into a harness that has already been validated
against codecs whose behaviour we know.

Aggregation convention, matching the JPEG AI common test conditions: for each
quality point, average bpp over the dataset and average each metric over the
dataset, then fit **one** RD curve to those averaged points. The alternative --
per-image BD-rate, then average -- gives different numbers and is not what the
paper does. Verified against Tables III-VI: see docs/05.

Adding our own codec later means one more entry in the codec loop; nothing else
in this file changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from jpegai.config import PROJECT_ROOT
from jpegai.eval import codecs as anchors
from jpegai.eval.bdrate import bd_rate_table

RESULTS = PROJECT_ROOT / "results"
CACHE = RESULTS / "cache"

#: Where each named dataset lives, relative to the project root.
DATASETS: dict[str, str] = {
    "kodak": "data/kodak",
    "div2k_valid": "data/div2k/DIV2K_valid_HR",
    "div2k_train": "data/div2k/DIV2K_train_HR",
    "clic": "data/clic",
}

IMAGE_EXT = {".png", ".ppm", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".webp"}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def resolve_dataset(name: str) -> tuple[str, Path]:
    """Accept a registered name or any directory path. Returns (label, dir)."""
    if name in DATASETS:
        return name, PROJECT_ROOT / DATASETS[name]
    p = Path(name)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.name or name, p


def list_images(root: Path, limit: int | None = None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"dataset directory not found: {root}\n"
            f"Run `bash setup.sh` to download Kodak and DIV2K, or pass a path."
        )
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if not files:
        raise FileNotFoundError(f"no images found under {root}")
    return files[:limit] if limit else files


# ---------------------------------------------------------------------------
# Metrics bridge. Imported lazily so --list and --dry-run work without torch.
# ---------------------------------------------------------------------------
def _load_metrics():
    try:
        from jpegai.eval import metrics
    except ImportError as exc:
        raise SystemExit(
            f"cannot import jpegai.eval.metrics: {exc}\n"
            "The metric backends are not installed yet. Run:\n"
            "    bash setup.sh\n"
            "then `source .venv/bin/activate` and retry."
        ) from exc
    return metrics


def _to_tensor(rgb: np.ndarray):
    """uint8 [H,W,3] -> float torch tensor [1,3,H,W] in [0,1]."""
    import torch

    arr = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(arr).float().unsqueeze(0) / 255.0


# ---------------------------------------------------------------------------
# Cache. One file per (dataset, codec, image), keyed inside by quality.
# ---------------------------------------------------------------------------
CACHE_VERSION = 1


def _cache_path(label: str, codec: str, image: Path) -> Path:
    return CACHE / label / codec / f"{image.stem}.json"


def _cache_load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except Exception:
        return {}
    if blob.get("_version") != CACHE_VERSION:
        return {}
    return blob.get("points", {})


def _cache_save(path: Path, points: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"_version": CACHE_VERSION, "points": points}, indent=1))
    tmp.replace(path)  # atomic, so a Ctrl-C never leaves a half-written cache


# ---------------------------------------------------------------------------
# The measurement loop
# ---------------------------------------------------------------------------
#: Fields `compute_all(include_psnr=True)` returns for free alongside the requested
#: metrics. Reported and plotted, but never part of the BD-rate AVG -- the paper's
#: AVG is the mean of its seven metrics and nothing else (verified in docs/05).
_EXTRA_FIELDS = ["psnr", "psnr_y", "psnr_u", "psnr_v"]


def measure_codec(
    codec: anchors.Codec,
    images: list[Path],
    label: str,
    metric_names: list[str],
    *,
    use_cache: bool = True,
    verbose: bool = True,
) -> dict:
    """Run one codec over one dataset at every quality point.

    Returns {"bpp": [...], "<metric>": [...], "quality": [...], "n_images": int}
    with one entry per quality point, averaged over the dataset.
    """
    from PIL import Image

    metrics = _load_metrics()
    per_quality: dict = {
        q: {"bpp": [], "ms": [],
            **{m: [] for m in dict.fromkeys(list(metric_names) + _EXTRA_FIELDS)}}
        for q in codec.qualities
    }

    for idx, img_path in enumerate(images, 1):
        # `cache_name` lets a codec whose behaviour is not fixed by its name -- ours,
        # whose quality labels point at trainable weights -- invalidate its own cache
        # when those weights change. Anchors have no such attribute and use `name`.
        cpath = _cache_path(label, getattr(codec, "cache_name", codec.name), img_path)
        cached = _cache_load(cpath) if use_cache else {}
        src = None  # loaded lazily -- a fully cached image is never even opened
        dirty = False

        for q in codec.qualities:
            key = str(q)
            entry = cached.get(key)
            if entry is None or any(m not in entry for m in metric_names):
                if src is None:
                    src = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
                t0 = time.perf_counter()
                nbytes, dec = codec.encode_decode(src, q)
                ms = (time.perf_counter() - t0) * 1e3

                h, w = src.shape[:2]
                vals = metrics.compute_all(
                    _to_tensor(src), _to_tensor(dec),
                    metrics=metric_names, include_psnr=True,
                )
                entry = {"bpp": nbytes * 8.0 / (w * h), "ms": ms,
                         "w": w, "h": h, **{k: float(v) for k, v in vals.items()}}
                cached[key] = entry
                dirty = True

            per_quality[q]["bpp"].append(entry["bpp"])
            per_quality[q]["ms"].append(entry.get("ms", float("nan")))
            for m in metric_names:
                per_quality[q][m].append(entry[m])
            # compute_all(include_psnr=True) also returns plain PSNR and per-plane
            # PSNR. They cost nothing extra and the paper's EFE discussion is about
            # chroma PSNR, so carry them through instead of computing and throwing
            # them away. `.get` with nan because a cache entry written by an older
            # run may predate these keys -- absent is not a reason to recompute.
            for m in _EXTRA_FIELDS:
                per_quality[q][m].append(float(entry.get(m, float("nan"))))

        if dirty and use_cache:
            _cache_save(cpath, cached)
        if verbose:
            state = "computed" if dirty else "cached"
            print(f"  [{idx:3}/{len(images)}] {codec.name:5} {img_path.name:22} {state}",
                  flush=True)

    curve: dict = {"quality": list(codec.qualities), "n_images": len(images),
                   "note": codec.note}
    for field in ["bpp", "ms"] + metric_names:
        curve[field] = [float(np.nanmean(per_quality[q][field])) for q in codec.qualities]
    # Drop an extra field entirely if no image produced a value for it, so the
    # report and the plot never show an all-nan row.
    for field in _EXTRA_FIELDS:
        if field in metric_names:
            continue
        col = [float(np.nanmean(per_quality[q][field])) for q in codec.qualities]
        if not all(np.isnan(v) for v in col):
            curve[field] = col
    return curve


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _for_bdrate(curve: dict, metric_names: list[str], metrics_mod) -> dict:
    """Flip lower-is-better metrics so BD-rate's higher-is-better assumption holds.

    NLPD is the only one today. Getting this wrong inverts its sign and quietly
    poisons the AVG column, so it is done in exactly one place.
    """
    out = {"bpp": curve["bpp"]}
    for m in metric_names:
        higher_better = metrics_mod.REGISTRY[m][1]
        vals = np.asarray(curve[m], dtype=float)
        out[m] = list(vals if higher_better else -vals)
    return out


def bdrate_report(curves: dict, anchor: str, metric_names: list[str], metrics_mod) -> dict:
    """BD-rate of every codec against `anchor`, per metric plus AVG."""
    if anchor not in curves:
        raise KeyError(f"anchor {anchor!r} not among measured codecs {sorted(curves)}")
    a = _for_bdrate(curves[anchor], metric_names, metrics_mod)
    return {
        name: bd_rate_table(a, _for_bdrate(c, metric_names, metrics_mod), metric_names)
        for name, c in curves.items() if name != anchor
    }


def write_markdown(path: Path, label: str, curves: dict, report: dict,
                   anchor: str, metric_names: list[str]) -> None:
    n = next(iter(curves.values()))["n_images"]
    lines = [
        f"# RD benchmark -- {label}",
        "",
        f"{n} images. Anchor: **{anchor}**. Negative BD-rate = fewer bits for equal "
        "quality = better.",
        "",
        "AVG is the unweighted mean of the per-metric BD-rates, matching the paper's "
        "Tables III-VI (verified in docs/05).",
        "",
        "## BD-rate vs " + anchor,
        "",
        "| codec | AVG | " + " | ".join(metric_names) + " |",
        "|---|---|" + "---|" * len(metric_names),
    ]
    for name, row in report.items():
        cells = [f"{row.get(m, float('nan')):+.1f}%" for m in metric_names]
        lines.append(f"| {name} | **{row['AVG']:+.1f}%** | " + " | ".join(cells) + " |")

    lines += ["", "## Rate points (dataset averages)", "",
              "| codec | quality | bpp | " + " | ".join(metric_names) + " |",
              "|---|---|---|" + "---|" * len(metric_names)]
    for name, c in curves.items():
        for i, q in enumerate(c["quality"]):
            cells = [f"{c[m][i]:.4f}" for m in metric_names]
            lines.append(f"| {name} | {q} | {c['bpp'][i]:.4f} | " + " | ".join(cells) + " |")

    # PSNR is reported separately and deliberately excluded from AVG: the paper's
    # AVG is the mean of its seven metrics, and none of them is PSNR. It is here
    # because the EFE tools trade BD-rate for chroma PSNR (Table IV), so we need a
    # chroma-PSNR column to reproduce that trade-off in Phase 13.
    extras = [f for f in _EXTRA_FIELDS
              if f not in metric_names and any(f in c for c in curves.values())]
    if extras:
        lines += ["", "## PSNR (dB, reported only -- never part of AVG)", "",
                  "| codec | quality | bpp | " + " | ".join(extras) + " |",
                  "|---|---|---|" + "---|" * len(extras)]
        for name, c in curves.items():
            for i, q in enumerate(c["quality"]):
                cells = [f"{c[f][i]:.2f}" if f in c else "--" for f in extras]
                lines.append(f"| {name} | {q} | {c['bpp'][i]:.4f} | "
                             + " | ".join(cells) + " |")

    skipped = {k: v["_skipped"] for k, v in report.items() if v.get("_skipped")}
    if skipped:
        lines += ["", "## Skipped metrics", ""]
        for k, v in skipped.items():
            for s in v:
                lines.append(f"- `{k}`: {s}")

    path.write_text("\n".join(lines) + "\n")


def plot_curves(path: Path, label: str, curves: dict, metric_names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Only plot fields at least one curve actually carries. `psnr`/`psnr_y` are
    # appended as a convenience because they are cheap and useful for the EFE
    # chroma discussion, but they are absent whenever the curve builder was
    # restricted to `--metrics`. Requesting them unconditionally drew blank axes.
    wanted = list(dict.fromkeys(list(metric_names) + _EXTRA_FIELDS))
    fields = [f for f in wanted if any(f in c for c in curves.values())]
    if not fields:
        return
    ncol = min(3, len(fields))
    nrow = (len(fields) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow), squeeze=False)

    for i, field in enumerate(fields):
        ax = axes[i // ncol][i % ncol]
        for name, c in curves.items():
            if field not in c:
                continue
            ax.plot(c["bpp"], c[field], marker="o", ms=3.5, lw=1.4, label=name)
        ax.set_xlabel("bitrate (bpp)")
        ax.set_ylabel(field)
        ax.set_title(field)
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_xscale("log")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(len(fields), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(f"Rate-distortion -- {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m jpegai.eval.runbench",
        description="Rate-distortion benchmark for anchor codecs (and later, ours).",
    )
    ap.add_argument("--codecs", default=",".join(anchors.DEFAULT_CODECS),
                    help="comma-separated: " + ",".join(anchors.REGISTRY))
    ap.add_argument("--neural", action="append", default=None, metavar="DIR",
                    help="also measure our trained codec, one rate point per "
                         "beta<value>/final.pt under DIR (see train.runladder). "
                         "Repeatable, or comma-separated: several ladders become "
                         "several curves on one plot, named after their directories, "
                         "so --anchor can name one of them and BD-rate comes out "
                         "phase-against-phase instead of only against JPEG")
    ap.add_argument("--dataset", default="kodak",
                    help="name (" + ", ".join(DATASETS) + ") or a directory path")
    ap.add_argument("--metrics", default=None,
                    help="comma-separated; default is the paper's seven")
    ap.add_argument("--anchor", default="jpeg", help="BD-rate reference codec")
    ap.add_argument("--out", default=None, metavar="STEM",
                    help="write results/<STEM>.{json,md,png} instead of "
                         "results/bench_<dataset>. Give each ladder its own stem, or "
                         "the next run overwrites the last one's table")
    ap.add_argument("--limit", type=int, default=None, help="first N images only")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--list", action="store_true", help="show codecs/datasets and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the work that would be done, then exit")
    args = ap.parse_args(argv)

    if args.list:
        anchors.describe()
        print("\ndatasets")
        for name, rel in DATASETS.items():
            root = PROJECT_ROOT / rel
            try:
                n = len(list_images(root))
                print(f"  {name:12} {n:4} images  {rel}")
            except FileNotFoundError:
                print(f"  {name:12}  --  missing    {rel}")
        return 0

    label, root = resolve_dataset(args.dataset)
    images = list_images(root, args.limit)

    names = [c.strip() for c in args.codecs.split(",") if c.strip()]
    chosen, unavailable = [], []
    for n in names:
        c = anchors.get(n)
        (chosen if c.available() else unavailable).append(c)
    if unavailable:
        print("skipping unavailable codecs: "
              + ", ".join(f"{c.name} ({anchors._ERR.get(c.name, 'unavailable')})"
                          for c in unavailable), file=sys.stderr)
    if args.neural:
        # Appended last so the anchors are measured first: if a torch problem
        # kills the run, the anchor curves are already cached.
        from jpegai.eval.neural import NeuralCodec

        dirs = [d.strip() for spec in args.neural for d in spec.split(",") if d.strip()]
        for d in dirs:
            # One ladder keeps the plain name, so an existing cache and every command
            # already written down still hit. Several have to be told apart, and the
            # directory name is what the user chose to call the phase -- `ladder_p6`
            # becomes `ours-ladder_p6`, which is also what `--anchor` then takes.
            name = "jpegai" if len(dirs) == 1 else f"ours-{Path(d).name}"
            ours = NeuralCodec.from_directory(d, name=name)
            if not ours.available():
                print(f"neural codec at {d} is not loadable", file=sys.stderr)
                return 1
            chosen.append(ours)
    if not chosen:
        print("no usable codecs", file=sys.stderr)
        return 1

    if args.metrics:
        metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics_probe = _load_metrics()
        metric_names = list(metrics_probe.PAPER_SEVEN)

    total = len(images) * sum(len(c.qualities) for c in chosen)
    print(f"dataset  {label}  ({len(images)} images from {root})")
    print(f"codecs   {', '.join(c.name for c in chosen)}")
    print(f"metrics  {', '.join(metric_names)}")
    print(f"work     {total} encode/decode/measure operations")
    if "vmaf" in metric_names:
        print("note     vmaf spawns ffmpeg per measurement; drop it with --metrics "
              "for a fast pass")
    if args.dry_run:
        return 0
    print()

    metrics_mod = _load_metrics()
    curves = {}
    for c in chosen:
        curves[c.name] = measure_codec(
            c, images, label, metric_names,
            use_cache=not args.no_cache, verbose=not args.quiet,
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    # `label` still names the dataset inside the files and on the plot; only the
    # filename stem moves, so four ladders can be measured without each run
    # overwriting the previous one's table.
    stem = args.out or f"bench_{label}"
    out_json = RESULTS / f"{stem}.json"
    out_json.write_text(json.dumps(
        {"dataset": label, "root": str(root), "n_images": len(images),
         "metrics": metric_names, "anchor": args.anchor, "curves": curves},
        indent=1))

    anchor = args.anchor if args.anchor in curves else chosen[0].name
    if anchor != args.anchor:
        print(f"\nanchor {args.anchor!r} unavailable; using {anchor!r}", file=sys.stderr)
    report = bdrate_report(curves, anchor, metric_names, metrics_mod)

    out_md = RESULTS / f"{stem}.md"
    write_markdown(out_md, label, curves, report, anchor, metric_names)
    out_png = RESULTS / f"{stem}.png"
    plot_curves(out_png, label, curves, metric_names)

    print(f"\nBD-rate vs {anchor}   (negative = better)")
    width = max(len(n) for n in report) if report else 6
    print(f"  {'codec':{width}}  {'AVG':>8}   " + "  ".join(f"{m:>9}" for m in metric_names))
    for name, row in report.items():
        cells = "  ".join(f"{row.get(m, float('nan')):>+9.1f}" for m in metric_names)
        print(f"  {name:{width}}  {row['AVG']:>+8.1f}   {cells}")

    print(f"\nwrote {out_json.relative_to(PROJECT_ROOT)}")
    print(f"      {out_md.relative_to(PROJECT_ROOT)}")
    print(f"      {out_png.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
