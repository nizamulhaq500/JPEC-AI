r"""Phase 10, Table IV: the switchable-tools ablation, over one common bitstream.

Table IV of the paper reads each switchable tool as a delta against the codec with
every tool on -- "turn this one off, how many more bits for the same quality". The
honest way to measure that is the thing the standard's decode semantics already give
us: all six tools are *decoder-side* (RVS/LSBS refine the latent before synthesis,
the four post-filters run on the output planes), so a bitstream written once decodes
to a *different reconstruction* under every tool subset while the bytes never move.

So this harness compresses each image exactly once per rate point and then decodes
that same packet once per subset -- `decompress(packet, apply_tools=subset)`. The
bpp column is therefore identical across every subset by construction (asserted), and
the only thing that varies is distortion. The Table IV number for tool ``T`` is the
AVG BD-rate of the "``T`` off" curve against "all-on": positive means switching ``T``
off costs bits, i.e. ``T`` was helping.

Two ways to get the >=4 rate points BD-rate needs, mirroring `runbench`:

  * a single variable-rate checkpoint swept over `config.rate.beta_eval_points`
    (`--delta-betas`), one set of weights covering the whole ladder; or
  * a ladder of fixed-rate checkpoints, one rate point each.

The paper's own findings this is built to reproduce: RVS moves the perceptual metrics
(FSIM/VMAF) most; the two EFE filters lift chroma PSNR sharply while sitting slightly
negative on the seven-metric AVG -- a real inversion, not a bug; and ICCI is the only
post-filter with meaningful MAC cost, which the kMAC/pxl column makes visible.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from jpegai.config import PROJECT_ROOT, load_config
from jpegai.eval import metrics as _metrics
from jpegai.eval.metrics import PAPER_SEVEN
from jpegai.eval.runbench import (_EXTRA_FIELDS, _to_tensor, bdrate_report,
                                  list_images, psnr_bdrate_report, resolve_dataset)
from jpegai.models import build_any_model
from jpegai.models.twobranch import ALL_TOOLS
from jpegai.utils import macs_breakdown, pick_device

RESULTS = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# The subsets Table IV compares
# ---------------------------------------------------------------------------
def tool_subsets(tools) -> "OrderedDict[str, frozenset]":
    """The ordered set of decode configurations Table IV needs.

    ``all-on`` (the anchor) first, then one "``<tool>`` off" per tool in the paper's
    fixed ``ALL_TOOLS`` order so the table reads the same whatever order the model was
    built in, then ``none`` (every tool off) as the whole-stack delta. Each value is
    the ``apply_tools`` set handed to ``decompress`` -- the model intersects it with
    what it actually holds, so an "off" subset is exactly "all-on minus that one".
    """
    have = frozenset(tools)
    ordered = [t for t in ALL_TOOLS if t in have]
    subsets: "OrderedDict[str, frozenset]" = OrderedDict()
    subsets["all-on"] = have
    for t in ordered:
        subsets[f"{t} off"] = have - {t}
    subsets["none"] = frozenset()
    return subsets


# ---------------------------------------------------------------------------
# Loading a tools checkpoint
# ---------------------------------------------------------------------------
def load_tools_model(path, device=None):
    """Rebuild the exact architecture a tools checkpoint was trained with.

    Mirrors `jpegai.eval.neural`'s loader, plus the one thing that path also now does:
    read `meta["tools"]` and pass it to `build_any_model`, so the tool modules exist to
    receive their weights. Raises if the checkpoint carries no tools -- an ablation over
    nothing is a mistake worth naming, not an empty table.
    """
    import torch
    from jpegai.train.loop import load_checkpoint

    device = device or pick_device(None)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob.get("meta", {})
    cfg = load_config(meta.get("tier", "full"))
    model = build_any_model(cfg, meta.get("model", "scale"),
                            tools=tuple(meta.get("tools", ()))).to(device)
    load_checkpoint(path, model)
    if not getattr(model, "tool_names", None):
        raise SystemExit(f"{Path(path).name} has no switchable tools to ablate; it was "
                         f"trained without --tools (meta['tools'] is empty)")
    model.eval()
    model.update(force=True)
    return model, meta, cfg


# ---------------------------------------------------------------------------
# Measurement: compress once per rate point, decode once per subset
# ---------------------------------------------------------------------------
def reconstruct_subsets(model, packet, device, subsets):
    """Decode one packet under every subset -> `{subset_name: x_hat}`.

    The ablation seam in one call: the bytes are fixed, `apply_tools` is not. Kept
    separate from the metric loop so a test can assert the identity-at-init property
    (every subset byte-identical output) without computing a single metric.
    """
    return {name: model.decompress(packet, device=device, apply_tools=sub)["x_hat"]
            for name, sub in subsets.items()}


def measure_subsets(rate_points, tensors, subsets, device, *,
                    metric_names=PAPER_SEVEN, verbose=False):
    """Build one RD curve per subset, averaged over `tensors`.

    `rate_points` is a list of `(model, delta_beta)` -- a variable-rate sweep reuses
    one model across many `delta_beta`, a fixed-rate ladder pairs each checkpoint with
    `None`. Every model must expose the same tools (the subsets are shared). Returns
    `{subset_name: {"bpp": [...], <metric>: [...], "psnr_y": [...], ..., "n_images"}}`
    with one entry per rate point.
    """
    fields = list(dict.fromkeys(list(metric_names) + _EXTRA_FIELDS))
    # acc[name][r] = {"bpp": [...over images], <field>: [...]}
    acc = {name: [{"bpp": [], **{f: [] for f in fields}} for _ in rate_points]
           for name in subsets}
    for i, x in enumerate(tensors, 1):
        x = x.to(device)
        for r, (model, d) in enumerate(rate_points):
            packet = model.compress(x) if d is None else model.compress(x, delta_beta=int(d))
            nbytes = model.packet_bytes(packet) + model.header_bytes(packet)
            bpp = nbytes * 8.0 / (x.shape[-1] * x.shape[-2])
            for name, x_hat in reconstruct_subsets(model, packet, device, subsets).items():
                vals = _metrics.compute_all(x, x_hat, metrics=list(metric_names),
                                            include_psnr=True)
                acc[name][r]["bpp"].append(bpp)
                for f in fields:
                    if f in vals:
                        acc[name][r][f].append(float(vals[f]))
        if verbose:
            print(f"  [{i:3}/{len(tensors)}] measured", flush=True)

    curves: dict = {}
    for name in subsets:
        curve = {"n_images": len(tensors), "note": name}
        curve["bpp"] = [float(np.mean(acc[name][r]["bpp"])) for r in range(len(rate_points))]
        for f in fields:
            col = [acc[name][r][f] for r in range(len(rate_points))]
            if all(len(c) for c in col):
                curve[f] = [float(np.mean(c)) for c in col]
        curves[name] = curve
    return curves


# ---------------------------------------------------------------------------
# Table IV assembly
# ---------------------------------------------------------------------------
def tool_macs(model, crop: int = 256) -> "OrderedDict[str, float]":
    """kMAC/pxl each tool adds, via `training_parts()['tools']` as the MAC buckets.

    RVS and LSBS are integer table lookups with no convolutions, so they never appear
    in the conv-counting breakdown and are reported as a literal 0.0 -- which is the
    point the paper makes about them (~0 MAC). ICCI is the only one that lands
    meaningfully above zero, and this column is what shows it.
    """
    parts = model.training_parts()["tools"]
    bd = macs_breakdown(model, (1, 3, crop, crop), parts=parts) if parts else {}
    out: "OrderedDict[str, float]" = OrderedDict()
    for label, _mod in parts:
        out[label] = round(bd.get(label, 0.0) / 1e3, 4)      # MAC/pxl -> kMAC/pxl
    return out


def _bpp_is_subset_invariant(curves) -> bool:
    """The decoder-side claim, checked: every subset priced the same bits per point."""
    ref = curves["all-on"]["bpp"]
    return all(np.allclose(c["bpp"], ref, atol=1e-9) for c in curves.values())


def build_report(curves, model, *, metric_names=PAPER_SEVEN, crop: int = 256) -> dict:
    """The Table IV payload: per-subset AVG BD-rate vs all-on, PSNR deltas, MACs."""
    n_points = len(curves["all-on"]["bpp"])
    report = {
        "anchor": "all-on",
        "n_images": curves["all-on"]["n_images"],
        "n_rate_points": n_points,
        "bpp_subset_invariant": bool(_bpp_is_subset_invariant(curves)),
        "kmac_per_pixel": dict(tool_macs(model, crop)),
        "tools": sorted(model.tool_names),
    }
    if n_points >= 4:
        report["bdrate"] = bdrate_report(curves, "all-on", metric_names, _metrics)
        report["psnr_bdrate"] = psnr_bdrate_report(curves, "all-on")
        # Table IV's Δ column, pulled out for the eye: AVG BD-rate of each "off" curve.
        report["table_iv"] = {name: tbl.get("AVG")
                              for name, tbl in report["bdrate"].items()}
    else:
        report["note"] = (f"only {n_points} rate point(s); BD-rate needs >=4. Report "
                          f"carries per-subset metric means for inspection instead.")
        report["metric_means"] = {
            name: {m: c.get(m) for m in list(metric_names) + _EXTRA_FIELDS if m in c}
            for name, c in curves.items()}
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def write_markdown(path: Path, label: str, report: dict, curves: dict) -> None:
    macs = report["kmac_per_pixel"]
    lines = [
        f"# Table IV -- switchable-tools ablation ({label})",
        "",
        f"{report['n_images']} images, {report['n_rate_points']} rate point(s). "
        f"Anchor: **all-on**. Tools: {', '.join(report['tools'])}.",
        "",
        "Each row switches one tool **off** and decodes the *same* bitstream; a "
        "positive BD-rate means turning it off costs bits, i.e. the tool was helping. "
        f"bpp identical across subsets: **{report['bpp_subset_invariant']}** "
        "(the tools are decoder-side, so they move no bytes).",
        "",
    ]
    if "table_iv" in report:
        lines += ["| Tool off | AVG BD-rate % | kMAC/pxl |", "| --- | ---: | ---: |"]
        for name, avg in report["table_iv"].items():
            tool = name[:-4] if name.endswith(" off") else name
            mac = macs.get(tool)
            lines.append(f"| {name} | {avg:+.2f} | "
                         f"{'' if mac is None else f'{mac:.3f}'} |")
        # The chroma-PSNR-up / AVG-down inversion the paper flags for the EFE filters is
        # only visible with the per-plane PSNR beside the AVG, so print both.
        pbd = report.get("psnr_bdrate", {})
        if pbd:
            lines += ["", "| Tool off | psnr_y % | psnr_u % | psnr_v % |",
                      "| --- | ---: | ---: | ---: |"]
            for name, tbl in pbd.items():
                lines.append(f"| {name} | {tbl.get('psnr_y', float('nan')):+.2f} | "
                             f"{tbl.get('psnr_u', float('nan')):+.2f} | "
                             f"{tbl.get('psnr_v', float('nan')):+.2f} |")
    else:
        lines += [report.get("note", ""), "",
                  "| Subset | " + " | ".join(PAPER_SEVEN) + " |",
                  "| --- " + "| ---: " * len(PAPER_SEVEN) + "|"]
        for name, mm in report["metric_means"].items():
            cells = " | ".join(f"{mm.get(m, float('nan')):.4f}" for m in PAPER_SEVEN)
            lines.append(f"| {name} | {cells} |")
    lines += ["", "kMAC/pxl is decoder-side, counted over Conv2d/ConvTranspose2d; RVS "
              "and LSBS are table lookups and read 0 by construction.", ""]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _center_crop(x, crop):
    if not crop:
        return x
    _, _, h, w = x.shape
    top, left = max(0, (h - crop) // 2), max(0, (w - crop) // 2)
    return x[..., top:top + crop, left:left + crop]


def _rate_points(models, cfg, delta_betas):
    """`[(model, delta_beta)]`. One VR checkpoint sweeps; a ladder is one point each."""
    first = models[0]
    if delta_betas is not None or (len(models) == 1 and getattr(first, "gain", False)):
        if len(models) != 1:
            raise SystemExit("a Delta_beta sweep reads ONE variable-rate checkpoint; "
                             "pass a single --checkpoint (or drop --delta-betas for a "
                             "fixed-rate ladder)")
        if not getattr(first, "gain", False):
            raise SystemExit("--delta-betas needs a variable-rate checkpoint (one with "
                             "a gain unit); this one is fixed-rate")
        pts = delta_betas if delta_betas else list(cfg.rate.beta_eval_points)
        return [(first, int(d)) for d in pts]
    return [(m, None) for m in models]


def run_ablation(checkpoints, dataset="kodak", *, limit=None, delta_betas=None,
                 crop=None, device=None, out=None, metric_names=PAPER_SEVEN,
                 verbose=True) -> dict:
    """End to end: load, sweep, decode-per-subset, report. Writes `<out>.{json,md}`."""
    from PIL import Image

    device = pick_device(device) if not hasattr(device, "type") else device
    loaded = [load_tools_model(p, device) for p in checkpoints]
    models = [m for m, _meta, _cfg in loaded]
    tools0 = models[0].tool_names
    if any(m.tool_names != tools0 for m in models):
        seen = " vs ".join(",".join(sorted(m.tool_names)) or "(none)" for m in models)
        raise SystemExit(f"every checkpoint in a ladder must carry the same tools: {seen}")
    cfg = loaded[0][2]
    subsets = tool_subsets(tools0)
    rate_points = _rate_points(models, cfg, delta_betas)

    label, root = resolve_dataset(dataset)
    tensors = []
    for p in list_images(root, limit):
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        tensors.append(_center_crop(_to_tensor(rgb), crop))
    if verbose:
        print(f"ablation {label}: {len(tensors)} images x {len(rate_points)} rate "
              f"point(s) x {len(subsets)} subsets, tools {sorted(tools0)}", flush=True)

    curves = measure_subsets(rate_points, tensors, subsets, device,
                             metric_names=metric_names, verbose=verbose)
    report = build_report(curves, models[0], metric_names=metric_names,
                          crop=crop or 256)

    stem = out or f"ablation_{label}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stem}.json").write_text(
        json.dumps({"report": report, "curves": curves}, indent=1))
    write_markdown(RESULTS / f"{stem}.md", label, report, curves)
    if verbose:
        print(f"wrote results/{stem}.json and results/{stem}.md", flush=True)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_deltas(spec, cfg):
    if spec in (None, "auto"):
        return None if spec is None else list(cfg.rate.beta_eval_points)
    return [int(v) for v in str(spec).split(",") if v.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m jpegai.eval.ablation",
        description="Phase 10 Table IV: switchable-tools ablation on one bitstream.")
    ap.add_argument("checkpoints", nargs="+", metavar="CKPT",
                    help="one variable-rate tools checkpoint to sweep, or a ladder of "
                         "fixed-rate ones (each a rate point). All must share tools")
    ap.add_argument("--dataset", default="kodak",
                    help="registered name or a directory path")
    ap.add_argument("--limit", type=int, default=None, help="first N images only")
    ap.add_argument("--delta-betas", nargs="?", const="auto", default=None,
                    dest="delta_betas", metavar="LIST",
                    help="sweep ONE variable-rate checkpoint over Delta_beta instead of "
                         "reading a ladder. Bare flag uses config.rate.beta_eval_points; "
                         "or pass a comma-separated list of signed integers")
    ap.add_argument("--crop", type=int, default=None,
                    help="centre-crop every image to CROPxCROP first (faster; the tools "
                         "are shift-invariant so the deltas are unchanged)")
    ap.add_argument("--metrics", default=None,
                    help="comma-separated; default is the paper's seven")
    ap.add_argument("--out", default=None, metavar="STEM",
                    help="write results/<STEM>.{json,md} (default ablation_<dataset>)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    # `--delta-betas` needs the config, which needs a loaded checkpoint to know the
    # tier; resolve it from the first checkpoint's meta the same way the loader does.
    import torch
    tier = torch.load(args.checkpoints[0], map_location="cpu",
                      weights_only=False).get("meta", {}).get("tier", "full")
    deltas = _parse_deltas(args.delta_betas, load_config(tier))
    metric_names = (args.metrics.split(",") if args.metrics else PAPER_SEVEN)
    run_ablation(args.checkpoints, args.dataset, limit=args.limit, delta_betas=deltas,
                 crop=args.crop, device=args.device, out=args.out,
                 metric_names=metric_names, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






