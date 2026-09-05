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

Phase 8 adds `--delta-beta`, which changes what a `--neural` rate point *is* -- one
checkpoint swept over the `Delta_beta` header field, instead of one checkpoint per
rate. Nothing else moves: same images, same metrics, same aggregation, same BD-rate
anchor. That is the point. The cost of variable rate is the gap between two curves,
so measuring the two curves through one code path is what makes the gap a number
rather than an argument. The flag decides how every `--neural` directory in a run is
read, so those two curves come from two runs; `--rerender A,B` pools their stored
measurements into one table without re-encoding anything.
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


def psnr_bdrate_report(curves: dict, anchor: str) -> dict:
    """BD-rate on plain and per-plane PSNR. Diagnostic, never part of AVG.

    The paper's seven metrics say whether the codec is competitive; these say *where*
    it is not. `psnr_y` against `psnr_u`/`psnr_v` separates the luma and chroma
    branches, which is the difference between "train longer" and "the luma branch
    needs the context model" -- and none of the four saturates, so unlike the seven
    they are immune to the interpolant trouble documented in docs/07 §5.4.

    All four are higher-is-better, so no sign flip is needed.
    """
    if anchor not in curves:
        return {}
    fields = [f for f in _EXTRA_FIELDS if all(f in c for c in curves.values())]
    if not fields:
        return {}
    a = curves[anchor]
    return {name: bd_rate_table(a, c, fields)
            for name, c in curves.items() if name != anchor}


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
        "BD-rate is interpolated with a monotone PCHIP over the *shared* quality range, "
        "not a global cubic -- see `jpegai/eval/bdrate.py` for why that choice changes "
        "answers by tens of percent on the saturating metrics.",
        "",
        "## BD-rate vs " + anchor,
        "",
        "| codec | AVG | " + " | ".join(metric_names) + " | overlap |",
        "|---|---|" + "---|" * len(metric_names) + "---|",
    ]
    for name, row in report.items():
        cells = [f"{row.get(m, float('nan')):+.1f}%" for m in metric_names]
        got, tot = row.get("_coverage", (0, 0))
        lines.append(f"| {name} | **{row['AVG']:+.1f}%** | " + " | ".join(cells)
                     + f" | {got}/{tot} |")

    # "overlap" is how many of the anchor's rate points fall inside the shared
    # quality range. It is a caveat on the row, not a result, so it is explained
    # here rather than left as a bare fraction.
    thin = {n: r["_coverage"] for n, r in report.items()
            if r.get("_coverage", (0, 0))[1] and r["_coverage"][0] < 0.7 * r["_coverage"][1]}
    lines += ["",
              f"`overlap` = how many of {anchor}'s rate points lie inside that codec's "
              "shared quality range. BD-rate averages over the overlap only, so a low "
              "count means the number rests on few anchor points."]
    if thin:
        names = ", ".join(f"`{n}` ({g}/{t})" for n, (g, t) in thin.items())
        lines += ["",
                  f"**Caveat:** {names} span only part of {anchor}'s range. Their AVG is "
                  "not measured over the same ground as the other rows. The fix is "
                  "lower-rate points in the ladder."]

    lines += ["", "## Rate points (dataset averages)", "",
              "| codec | quality | bpp | " + " | ".join(metric_names) + " |",
              "|---|---|---|" + "---|" * len(metric_names)]
    for name, c in curves.items():
        for i, q in enumerate(c["quality"]):
            cells = [f"{c[m][i]:.4f}" for m in metric_names]
            lines.append(f"| {name} | {q} | {c['bpp'][i]:.4f} | " + " | ".join(cells) + " |")

    # What the `quality` column means differs per row and is not guessable: a JPEG
    # quality, a trained beta, or -- for a variable-rate sweep -- a signed Delta_beta
    # header value. Each codec already carries a one-line `note`; printing them here is
    # the difference between a table a reader can interpret and one where `0` and
    # `0.002` sit in the same column meaning unrelated things.
    notes = {n: c["note"] for n, c in curves.items() if c.get("note")}
    if notes:
        lines += ["", "`quality` column: "
                  + "; ".join(f"**{n}** -- {v}" for n, v in notes.items())]

    # PSNR is reported separately and deliberately excluded from AVG: the paper's
    # AVG is the mean of its seven metrics, and none of them is PSNR. It is here
    # because the EFE tools trade BD-rate for chroma PSNR (Table IV), so we need a
    # chroma-PSNR column to reproduce that trade-off in Phase 13.
    extras = [f for f in _EXTRA_FIELDS
              if f not in metric_names and any(f in c for c in curves.values())]
    if extras:
        psnr_bd = psnr_bdrate_report(curves, anchor)
        if psnr_bd:
            cols = [f for f in _EXTRA_FIELDS if all(f in c for c in curves.values())]
            lines += ["", f"### PSNR BD-rate vs {anchor} (diagnostic, not in AVG)", "",
                      "Separates the two branches: `psnr_y` is the luma branch, "
                      "`psnr_u`/`psnr_v` the chroma one. None of these saturates, so they "
                      "are the most robust rows in this file.", "",
                      "| codec | " + " | ".join(cols) + " |",
                      "|---|" + "---|" * len(cols)]
            for name, row in psnr_bd.items():
                cells = [f"{row[f]:+.1f}%" if f in row else "--" for f in cols]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")

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
def _rel(path: Path) -> str:
    """Project-relative path for display, falling back to absolute.

    `Path.relative_to` raises when the target is outside the project, which it is
    whenever `RESULTS` has been pointed elsewhere. A crash while printing where a
    file was written would be an absurd way to lose a finished benchmark.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_report(report: dict, anchor: str, metric_names: list[str]) -> None:
    """The console BD-rate table, including the thin-overlap caveat."""
    print(f"\nBD-rate vs {anchor}   (negative = better)")
    width = max(len(n) for n in report) if report else 6
    print(f"  {'codec':{width}}  {'AVG':>8}   " + "  ".join(f"{m:>9}" for m in metric_names)
          + "   overlap")
    for name, row in report.items():
        cells = "  ".join(f"{row.get(m, float('nan')):>+9.1f}" for m in metric_names)
        got, tot = row.get("_coverage", (0, 0))
        print(f"  {name:{width}}  {row['AVG']:>+8.1f}   {cells}   {got:2}/{tot}")

    # BD-rate is an average over the *shared* quality range. A ladder that only
    # reaches the top of the anchor's range is compared over a slice of it, and the
    # answer then rests on a handful of anchor points. Saying so beats letting the
    # AVG be read as if the curves covered the same ground.
    thin = {n: r["_coverage"] for n, r in report.items()
            if r.get("_coverage", (0, 0))[1] and r["_coverage"][0] < 0.7 * r["_coverage"][1]}
    if thin:
        print("\nNOTE: these curves span only part of the anchor's quality range, so their")
        print("      BD-rate is an average over that slice, not over the whole sweep:")
        for n, (got, tot) in thin.items():
            print(f"        {n:{width}}  {got} of {tot} {anchor} points in the overlap")
        print("      The fix is lower-rate points in the ladder, not a different anchor.")


def print_psnr_report(psnr_bd: dict, anchor: str, curves: dict) -> None:
    """The luma/chroma diagnostic table. Says *where* a codec loses, not whether."""
    if not psnr_bd:
        return
    cols = [f for f in _EXTRA_FIELDS if all(f in c for c in curves.values())]
    width = max(len(n) for n in psnr_bd)
    print(f"\nPSNR BD-rate vs {anchor}   (diagnostic, not in AVG)")
    print(f"  {'codec':{width}}  " + "  ".join(f"{f:>9}" for f in cols))
    for name, row in psnr_bd.items():
        print(f"  {name:{width}}  "
              + "  ".join(f"{row[f]:>+9.1f}" if f in row else f"{'--':>9}" for f in cols))


def parse_window(arg: str | None) -> tuple[float, float] | None:
    """`"0.12:1.0"` -> `(0.12, 1.0)`. An open end is spelled by omitting it."""
    if not arg:
        return None
    lo, _, hi = arg.partition(":")
    if not _:
        raise SystemExit(f"--rate-window wants LO:HI, got {arg!r}")
    lo_f = float(lo) if lo.strip() else 0.0
    hi_f = float(hi) if hi.strip() else float("inf")
    if hi_f <= lo_f:
        raise SystemExit(f"--rate-window {arg!r}: hi must exceed lo")
    return lo_f, hi_f


def window_curves(curves: dict, window: tuple[float, float]) -> dict:
    """Drop every rate point outside `window` bpp, keeping the lists parallel.

    JPEG AI's common test conditions measure BD-rate at exactly five target rates --
    0.12, 0.25, 0.5, 0.75 and 1.0 bpp -- and nothing above 1.0. Our own ladders run to
    2.4 bpp, and BD-rate over the union of two ranges is not the same quantity as
    BD-rate over theirs: it weights in the high-rate end, where an autoencoder is
    saturating against its own reconstruction ceiling and a block transform is not, so
    the wider the window the worse we look for reasons the paper's number never sees.
    Comparing our figure with a published one therefore means matching the window
    first.

    Filtering is analysis, not measurement: the same cached points, fewer of them, so
    this is a pure function of the stored JSON and costs no encoding. Every metric list
    is indexed by the same mask as `bpp`, because a curve whose `bpp` and `vmaf` lists
    disagree in length would silently pair the wrong rate with the wrong score.
    """
    lo, hi = window
    out = {}
    for name, c in curves.items():
        keep = [i for i, b in enumerate(c["bpp"]) if lo <= float(b) <= hi]
        out[name] = {k: ([v[i] for i in keep] if isinstance(v, list)
                         and len(v) == len(c["bpp"]) else v)
                     for k, v in c.items()}
    return out


def check_metrics(names: list[str]) -> bool:
    """`False`, with a message, if any name is not a registered metric.

    Called before the encoding starts rather than where the names are used, because where
    they are used is after every encode: `bdrate_report` is the first thing to index the
    registry, so an unrecognised name used to spend the whole run and *then* raise
    `KeyError` with the JSON already written.

    `psnr` is the name that invites the mistake. It is a column in the reports, but it
    comes from the separate diagnostic table -- PSNR is reported and deliberately kept out
    of the paper's seven-metric AVG -- so it is not selectable here.
    """
    known = _load_metrics().REGISTRY
    unknown = [m for m in names if m not in known]
    if not unknown:
        return True
    hint = ("  psnr, psnr_y, psnr_u and psnr_v always appear in the diagnostic table "
            "below the main one and are not selected here."
            if set(unknown) & {"psnr", "psnr_y", "psnr_u", "psnr_v"} else "")
    print(f"unknown metric(s) {', '.join(unknown)}; --metrics takes any of "
          f"{', '.join(known)}.{hint}", file=sys.stderr)
    return False


def merge_saved(stems: list[str]) -> tuple[dict, str, list[str], str | None, int] | None:
    """Pool the curves of several stored benchmarks into one set. `None` on refusal.

    Phase 8's headline number is a comparison between two curves that cannot be produced
    in one run: `--delta-beta` decides how *every* `--neural` directory is read, so a
    fixed-rate ladder and a one-checkpoint sweep are necessarily two invocations. Their
    JSONs already hold finished measurements, and BD-rate is a pure function of those, so
    pooling the curves costs milliseconds and re-encodes nothing.

    What it refuses is what would make the pooled table a lie:

    * A different dataset or a different image count. BD-rate between curves measured on
      different pictures is not a comparison of codecs.
    * The same codec name carrying different numbers. `jpeg` appears in both files and is
      legitimately identical -- that is the shared anchor, and pooling is what makes it
      shared. Two *different* curves under one name would silently drop one of them,
      which is the failure this whole module keeps guarding against.

    Metrics are intersected, because BD-rate needs a column present in every curve, and
    the drop is printed rather than absorbed.
    """
    curves: dict = {}
    label = None
    metric_sets: list[list[str]] = []
    anchor = None
    n_images = None
    for stem in stems:
        src = RESULTS / f"{stem}.json"
        if not src.exists():
            print(f"no such benchmark: {_rel(src)}", file=sys.stderr)
            return None
        saved = json.loads(src.read_text())
        ds, n = saved.get("dataset", stem), saved.get("n_images")
        if label is None:
            label, n_images, anchor = ds, n, saved.get("anchor")
        elif (ds, n) != (label, n_images):
            print(f"{stem}.json measured {ds} on {n} images, but the first stem "
                  f"measured {label} on {n_images} -- BD-rate across different "
                  f"pictures is not a comparison of codecs", file=sys.stderr)
            return None
        metric_sets.append(saved["metrics"])
        for name, c in saved["curves"].items():
            if name in curves and curves[name]["bpp"] != c["bpp"]:
                print(f"codec {name!r} appears in more than one stem with different "
                      f"rates, so pooling would drop one of them -- re-run one of "
                      f"them under a different --neural name", file=sys.stderr)
                return None
            curves.setdefault(name, c)
    metrics = [m for m in metric_sets[0] if all(m in s for s in metric_sets)]
    if not metrics:
        print(f"the stems share no metrics ({' vs '.join(map(str, metric_sets))})",
              file=sys.stderr)
        return None
    dropped = [m for s in metric_sets for m in s if m not in metrics]
    if dropped:
        print(f"pooled on {len(metrics)} shared metric"
              f"{'' if len(metrics) == 1 else 's'}; dropped "
              f"{', '.join(sorted(set(dropped)))}")
    return curves, label, metrics, anchor, n_images


def rerender(stem: str, anchor_req: str | None, metrics_arg: str | None,
             window: tuple[float, float] | None = None,
             out: str | None = None) -> int:
    """Rebuild `results/<stem>.{md,png}` from `results/<stem>.json`, no encoding.

    The JSON holds the measurement -- rate and metric values per codec per quality.
    The markdown and the plot are renderings of it, and BD-rate is a pure function of
    it. So when the *analysis* changes rather than the data (§5.4 of docs/07 replaced
    the BD-rate interpolant), re-deriving the reports costs milliseconds and cannot
    drift from the numbers that were actually measured, whereas re-running the
    benchmark costs hours and needs every codec and image still present.

    `stem` may be several stems, comma-separated, in which case their curves are pooled
    -- see `merge_saved` for what that refuses. A pooled report needs somewhere to go
    that is not one of its sources, so it writes `<a>+<b>` unless `out` names a stem.
    """
    stems = [s.strip() for s in stem.split(",") if s.strip()]
    merged = merge_saved(stems)
    if merged is None:
        return 1
    curves, label, saved_metrics, saved_anchor, n_images = merged
    metric_names = ([m.strip() for m in metrics_arg.split(",") if m.strip()]
                    if metrics_arg else saved_metrics)
    if not check_metrics(metric_names):
        return 1

    anchor = anchor_req or saved_anchor or "jpeg"
    if anchor not in curves:
        print(f"anchor {anchor!r} not in {'+'.join(stems)} ({', '.join(curves)})",
              file=sys.stderr)
        return 1

    # A windowed report is a different quantity from the full-range one, so it gets its
    # own stem rather than overwriting the table that every earlier note quotes.
    out_stem = out or "+".join(stems)
    if window:
        before = {n: len(c["bpp"]) for n, c in curves.items()}
        curves = window_curves(curves, window)
        lo, hi = window
        out_stem = f"{out_stem}_w{lo:g}-{hi:g}".replace(".", "p")
        print(f"rate window [{lo:g}, {hi:g}] bpp")
        for n, c in curves.items():
            span = (f"{min(c['bpp']):.3f}-{max(c['bpp']):.3f} bpp" if c["bpp"]
                    else "nothing left in the window")
            print(f"  {n:34} {len(c['bpp']):2} of {before[n]:2} points   {span}")
        if anchor in curves and len(curves[anchor]["bpp"]) < 4:
            print(f"\nanchor {anchor!r} keeps only {len(curves[anchor]['bpp'])} point(s) "
                  f"in this window; BD-rate needs 4 and will report nan.", file=sys.stderr)
        print()

    report = bdrate_report(curves, anchor, metric_names, _load_metrics())
    out_md, out_png = RESULTS / f"{out_stem}.md", RESULTS / f"{out_stem}.png"
    write_markdown(out_md, label, curves, report, anchor, metric_names)
    plot_curves(out_png, label, curves, metric_names)

    print(f"re-rendered from {', '.join(_rel(RESULTS / f'{s}.json') for s in stems)} "
          f"({n_images} images, {len(curves)} codecs, no re-encoding)")
    print_report(report, anchor, metric_names)
    print_psnr_report(psnr_bdrate_report(curves, anchor), anchor, curves)
    print(f"\nwrote {_rel(out_md)}")
    print(f"      {_rel(out_png)}")
    return 0


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
    ap.add_argument("--delta-beta", nargs="?", const="auto", default=None,
                    metavar="LIST", dest="delta_beta",
                    help="read each --neural DIR as ONE variable-rate checkpoint and "
                         "sweep Delta_beta instead, so a single set of weights produces "
                         "the whole RD curve (Phase 8). Bare --delta-beta uses "
                         "config.rate.beta_eval_points; pass a comma-separated list of "
                         "signed integers in [-1069, 702] to choose the rungs. The "
                         "quality column then holds Delta_beta, not beta. It applies to "
                         "every --neural DIR in the run, so measuring what variable rate "
                         "costs against a fixed-rate ladder is two runs with two --out "
                         "stems, pooled afterwards with --rerender A,B --anchor ...")
    ap.add_argument("--dataset", default="kodak",
                    help="name (" + ", ".join(DATASETS) + ") or a directory path")
    ap.add_argument("--metrics", default=None,
                    help="comma-separated; default is the paper's seven")
    ap.add_argument("--anchor", default=None,
                    help="BD-rate reference codec (default jpeg, or whatever the "
                         "benchmark being re-rendered used)")
    ap.add_argument("--out", default=None, metavar="STEM",
                    help="write results/<STEM>.{json,md,png} instead of "
                         "results/bench_<dataset>. Give each ladder its own stem, or "
                         "the next run overwrites the last one's table. With --rerender "
                         "it names the re-rendered .md/.png, which is what a pooled "
                         "report needs since it has no single source stem")
    ap.add_argument("--limit", type=int, default=None, help="first N images only")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--list", action="store_true", help="show codecs/datasets and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the work that would be done, then exit")
    ap.add_argument("--rerender", default=None, metavar="STEM[,STEM...]",
                    help="recompute results/<STEM>.{md,png} from the stored "
                         "results/<STEM>.json without re-encoding anything. For when "
                         "the analysis changed and the measurement did not. Several "
                         "comma-separated stems are pooled into one table, which is how "
                         "a fixed-rate ladder and a --delta-beta sweep -- necessarily "
                         "two runs -- get compared to each other; --out names the "
                         "pooled stem, --anchor picks which curve is the reference")
    ap.add_argument("--rate-window", default=None, metavar="LO:HI", dest="rate_window",
                    help="restrict BD-rate to points inside LO..HI bpp. Use 0.12:1.0 to "
                         "match JPEG AI's common test conditions, whose five target "
                         "rates stop at 1.0 bpp; any figure compared with a published "
                         "one has to use the published window")
    args = ap.parse_args(argv)
    window = parse_window(args.rate_window)
    if args.delta_beta and not args.neural:
        print("--delta-beta only changes how --neural is read; nothing to sweep",
              file=sys.stderr)
        return 1

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

    if args.rerender:
        return rerender(args.rerender, args.anchor, args.metrics, window, args.out)

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
        from jpegai.eval.neural import NeuralCodec, VariableRateCodec

        points = None
        if args.delta_beta and args.delta_beta != "auto":
            try:
                points = [int(v) for v in args.delta_beta.replace(" ", ",").split(",")
                          if v.strip()]
            except ValueError:
                print(f"--delta-beta wants signed integers, got "
                      f"{args.delta_beta!r}", file=sys.stderr)
                return 1
            if not points:
                print("--delta-beta got an empty point list", file=sys.stderr)
                return 1

        dirs = [d.strip() for spec in args.neural for d in spec.split(",") if d.strip()]
        for d in dirs:
            # One ladder keeps the plain name, so an existing cache and every command
            # already written down still hit. Several have to be told apart, and the
            # directory name is what the user chose to call the phase -- `ladder_p6`
            # becomes `ours-ladder_p6`, which is also what `--anchor` then takes.
            name = "jpegai" if len(dirs) == 1 else f"ours-{Path(d).name}"
            if args.delta_beta:
                # `-vr` in the name even for a single curve, because the two kinds of
                # run write the same files and a plot legend reading `jpegai` would not
                # say which one produced it. It also keeps the two caches separate,
                # which matters: the quality keys collide (`0` is a valid Delta_beta and
                # `0` would be a beta label) and a shared cache would cross-read them.
                try:
                    ours = VariableRateCodec.from_checkpoint(
                        d, points, name=f"{name}-vr" if name == "jpegai" else name)
                    ours.check_gain_unit()
                except (FileNotFoundError, ValueError) as exc:
                    print(f"--delta-beta on {d}: {exc}", file=sys.stderr)
                    return 1
                anchor_b = ours.anchor_beta()
                span = f"{ours.qualities[0]:+d}..{ours.qualities[-1]:+d}"
                print(f"neural   {ours.name}: one checkpoint, "
                      f"{len(ours.qualities)} Delta_beta points ({span})"
                      + (f", trained at beta {anchor_b:g}"
                         if anchor_b is not None else ""))
            else:
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
        if not check_metrics(metric_names):
            return 1
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
    anchor_req = args.anchor or "jpeg"
    # `label` still names the dataset inside the files and on the plot; only the
    # filename stem moves, so four ladders can be measured without each run
    # overwriting the previous one's table.
    stem = args.out or f"bench_{label}"
    out_json = RESULTS / f"{stem}.json"
    out_json.write_text(json.dumps(
        {"dataset": label, "root": str(root), "n_images": len(images),
         "metrics": metric_names, "anchor": anchor_req, "curves": curves},
        indent=1))

    anchor = anchor_req if anchor_req in curves else chosen[0].name
    if anchor != anchor_req:
        print(f"\nanchor {anchor_req!r} unavailable; using {anchor!r}", file=sys.stderr)
    # The JSON above is the measurement and stays full-range; only the analysis below
    # sees the window, so re-widening it later never needs a re-encode.
    if window:
        curves = window_curves(curves, window)
        print(f"\nrate window [{window[0]:g}, {window[1]:g}] bpp: "
              + ", ".join(f"{n} {len(c['bpp'])}pt" for n, c in curves.items()))
    report = bdrate_report(curves, anchor, metric_names, metrics_mod)

    out_md = RESULTS / f"{stem}.md"
    write_markdown(out_md, label, curves, report, anchor, metric_names)
    out_png = RESULTS / f"{stem}.png"
    plot_curves(out_png, label, curves, metric_names)

    print_report(report, anchor, metric_names)
    print_psnr_report(psnr_bdrate_report(curves, anchor), anchor, curves)

    print(f"\nwrote {_rel(out_json)}")
    print(f"      {_rel(out_md)}")
    print(f"      {_rel(out_png)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
