"""`jpegai` command line: encode / decode / inspect / bench / conform.

A real round-trip over the on-disk `.jpegai` container (see `jpegai.codestream`): encode a
PNG to a self-describing file that carries the exact entropy substreams, decode it back to
pixels, and inspect or conformance-check the file -- the last two without a checkpoint.

Scope is kept honest. The architecture, internal format, and synthesis transform are fixed
by the trained checkpoint, so there is no `--model/--encoder-id/--decoder-id` selection
here: those are JPEG AI's normative identifiers and belong to Track B (T.840-1), which is
gated on tables not in hand (docs/03 Section 0). Region-of-interest coding (`--roi`) is
real, routed through the gain unit's quality map; spatial tiling and independent regions
are not in this build. `bench` forwards to `jpegai.eval.runbench`.

    python -m jpegai encode  in.png out.jpegai --checkpoint checkpoints/p8_vr_mcm_IV/final.pt
    python -m jpegai decode  out.jpegai rec.png --checkpoint checkpoints/p8_vr_mcm_IV/final.pt
    python -m jpegai inspect out.jpegai
    python -m jpegai conform out.jpegai
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jpegai.codestream import coded_bytes, describe, read_packet, write_packet

_TOOLS_HELP = "comma-separated subset of the checkpoint's built tools (or 'none')"


def _load_model(checkpoint: str, device: str | None = None):
    """Rebuild the exact model a checkpoint was trained as, with coding tables ready."""
    import torch

    from jpegai.config import load_config
    from jpegai.models import build_any_model
    from jpegai.train.loop import load_checkpoint
    from jpegai.utils import pick_device

    dev = pick_device(device)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = blob.get("meta", {})
    cfg = load_config(meta.get("tier", "tierA"))
    model = build_any_model(cfg, meta.get("model", "scale"),
                            tools=tuple(meta.get("tools", ()))).to(dev)
    load_checkpoint(checkpoint, model)
    model.eval()
    model.update(force=True)
    return model, meta, dev

def _read_rgb(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _to_tensor(rgb: np.ndarray, device):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return x.float().div_(255.0).unsqueeze(0).to(device)


def _save_rgb(path: str, x_hat) -> np.ndarray:
    import torch
    from PIL import Image
    out = (x_hat.clamp(0, 1) * 255.0).round().to(torch.uint8)
    arr = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)
    return arr


def _parse_tools(spec):
    """`None` -> leave as-is; 'none' -> ablate all; else the named subset (lowercased)."""
    if spec is None:
        return None
    spec = spec.strip().lower()
    if spec in ("", "none"):
        return frozenset()
    return frozenset(t.strip() for t in spec.split(",") if t.strip())


def _roi_q_index(model, x, mask_path: str, boost: int):
    """An ROI mask (white = boost) -> Table I's integer quality map on the latent grid.

    Sized to the luma latent exactly, by running the analysis transform, so it never has
    to assume the stride. The gain unit applies the same map to both branches (their
    latents share a grid), which is why one tensor suffices.
    """
    import torch
    from PIL import Image

    if not getattr(model, "gain", False):
        raise SystemExit("--roi needs a gain (variable-rate) checkpoint; this one is "
                         "fixed-rate and has no quality map to steer")
    y, _uv, _supp, _pad = model._to_planes(x)
    lh, lw = model.g_a_y(y).shape[-2:]
    mask = Image.open(mask_path).convert("L").resize((lw, lh), Image.NEAREST)
    on = torch.from_numpy((np.asarray(mask) > 127).astype("int64"))
    return (on * int(boost)).clamp(-8, 8).view(1, 1, lh, lw).to(x.device)

def cmd_encode(args) -> int:
    import torch

    rgb = _read_rgb(args.input)
    model, meta, dev = _load_model(args.checkpoint, args.device)

    if args.internal_format and args.internal_format != model.fmt.name:
        raise SystemExit(f"checkpoint codes {model.fmt.name}, not {args.internal_format}; "
                         f"the internal format is fixed at training time")
    if (args.beta_luma or args.beta_chroma) and not getattr(model, "gain", False):
        raise SystemExit("--beta-luma/--beta-chroma need a gain checkpoint; this one is "
                         "fixed-rate (its rate is set by the weights, not a header)")

    tools = _parse_tools(args.tools)
    if tools:
        missing = tools - set(model.tool_names)
        if missing:
            raise SystemExit(f"checkpoint has no tool(s) {sorted(missing)}; it was built "
                             f"with {sorted(model.tool_names) or 'none'}")

    x = _to_tensor(rgb, dev)
    q_index = _roi_q_index(model, x, args.roi, args.roi_boost) if args.roi else None
    with torch.no_grad():
        packet = model.compress(x, delta_beta=(args.beta_luma, args.beta_chroma),
                                q_index=q_index)
    if tools is not None:
        packet["tools"] = sorted(tools)

    disk = write_packet(args.output, packet)
    cb = coded_bytes(packet)
    h, w = rgb.shape[:2]
    bpp = cb * 8 / (h * w)
    print(f"encoded {args.input} -> {args.output}")
    print(f"  {w}x{h}  {model.fmt.name}  model={meta.get('model', 'scale')}"
          f"  coded {cb:,} B ({bpp:.4f} bpp)")
    print(f"  on disk {disk:,} B  (+{disk - cb:,} B container scaffolding, "
          f"{100 * (disk - cb) / disk:.1f}%)")
    if q_index is not None:
        print(f"  roi quality map boosted by +{args.roi_boost} on the latent grid")
    if tools:
        print(f"  decoder tools flagged: {sorted(tools)}")
    return 0

def cmd_decode(args) -> int:
    import torch

    packet = read_packet(args.input)
    model, meta, dev = _load_model(args.checkpoint, args.device)

    apply_tools = _parse_tools(args.tools)
    if apply_tools:
        missing = apply_tools - set(model.tool_names)
        if missing:
            raise SystemExit(f"checkpoint has no tool(s) {sorted(missing)}; built with "
                             f"{sorted(model.tool_names) or 'none'}")
    with torch.no_grad():
        rec = model.decompress(packet, device=dev, luma_only=args.luma_only,
                               apply_tools=apply_tools)
    arr = _save_rgb(args.output, rec["x_hat"])

    if args.crop:
        try:
            cx, cy, cw, ch = (int(v) for v in args.crop.split(","))
        except ValueError:
            raise SystemExit("--crop wants x,y,w,h (four integers)")
        arr = arr[cy:cy + ch, cx:cx + cw]
        from PIL import Image
        Image.fromarray(np.ascontiguousarray(arr)).save(args.output)
    h, w = arr.shape[:2]
    print(f"decoded {args.input} -> {args.output}  ({w}x{h}"
          f"{', luma only' if args.luma_only else ''})")
    if apply_tools is not None:
        print(f"  tools applied: {sorted(apply_tools) or 'none (ablated)'}")
    return 0


def cmd_inspect(args) -> int:
    d = describe(args.input)
    h, w = d["shape"]
    cb, disk = d["coded_bytes"], d["disk_bytes"]
    print(f"{args.input}")
    print(f"  container v{d['version']}   {w}x{h}   internal format {d['internal_format']}")
    print(f"  coded    {cb:>10,} B   {cb * 8 / (h * w):.4f} bpp   (model accounting; the rate)")
    print(f"  on disk  {disk:>10,} B   +{d['overhead_bytes']:,} B "
          f"({100 * d['overhead_bytes'] / disk:.1f}% container scaffolding)")
    print(f"  header   {d['header_bytes']:>10,} B   streams {d['streams']}")
    if d["delta_beta"] is not None:
        print(f"  delta_beta (luma, chroma) = {d['delta_beta']}")
    if d["tools"]:
        print(f"  decoder tools: {d['tools']}")
    if d["q_residual"] is not None:
        q = d["q_residual"]
        print(f"  roi map  shape {tuple(q['shape'])}  ({q['nbytes']} B raw, {q['dtype']})")
    return 0

def cmd_bench(args) -> int:
    """Forward to the real benchmark. `jpegai bench -- --neural <dir> ...`."""
    from jpegai.eval import runbench
    return int(runbench.main(args.rest) or 0)


def cmd_conform(args) -> int:
    """Container conformance: structural + accounting self-consistency.

    This checks the `.jpegai` *container*, not the normative T.840-1 codestream -- the
    latter is Track B and its marker/substream tables are not in hand (docs/03 Section 0),
    so a true bitstream-conformance pass cannot be written yet, and claiming one would be
    the dishonest move. What it does verify: the file is a well-formed container, its
    declared stream lengths cover the payload exactly, it decodes back to a packet, and the
    rate it advertises recomputes from that packet.
    """
    from jpegai.codestream.container import _read_header
    from jpegai.models.twobranch import TwoBranchCodec

    path = Path(args.input)
    with open(path, "rb") as f:
        header, version = _read_header(f)   # raises on bad magic / version
        payload = f.read()

    declared = 0
    for br in ("luma", "chroma"):
        rec = header["branches"][br]
        declared += sum(rec["y"]) + sum(rec["z"])
    if "q_residual" in header:
        declared += header["q_residual"]["nbytes"]

    pkt = read_packet(path)
    recomputed = TwoBranchCodec.packet_bytes(pkt)
    stored = int(header["accounting"]["coded_bytes"])

    checks = [
        (f"well-formed container (magic, v{version})", True, ""),
        ("declared stream lengths cover the payload",
         declared == len(payload), f"{declared} vs {len(payload)} B"),
        ("packet rate recomputes from the file",
         recomputed == stored, f"{recomputed} vs {stored} B"),
    ]
    ok = all(c[1] for c in checks)
    for name, passed, detail in checks:
        tail = f"   [{detail}]" if detail and not passed else ""
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}{tail}")
    print(f"\nconform: {'PASS' if ok else 'FAIL'}  (container format; the normative "
          f"T.840-1 codestream is Track B and is not checked here)")
    return 0 if ok else 1

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jpegai", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="PNG -> .jpegai")
    e.add_argument("input"); e.add_argument("output")
    e.add_argument("--checkpoint", required=True, help="trained .pt to code with")
    e.add_argument("--beta-luma", type=int, default=0,
                   help="luma Delta_beta rate offset (gain checkpoints only)")
    e.add_argument("--beta-chroma", type=int, default=0,
                   help="chroma Delta_beta rate offset (gain checkpoints only)")
    e.add_argument("--internal-format", help="assert the checkpoint's format (420/444)")
    e.add_argument("--roi", help="ROI mask PNG (white = boost quality)")
    e.add_argument("--roi-boost", type=int, default=4, help="quality-map boost on the ROI")
    e.add_argument("--tools", help=_TOOLS_HELP)
    e.add_argument("--device", default=None)
    e.set_defaults(func=cmd_encode)

    d = sub.add_parser("decode", help=".jpegai -> PNG")
    d.add_argument("input"); d.add_argument("output")
    d.add_argument("--checkpoint", required=True, help="the same architecture that encoded")
    d.add_argument("--luma-only", action="store_true",
                   help="skip the chroma branch entirely (machine-consumption path)")
    d.add_argument("--crop", help="post-decode crop x,y,w,h")
    d.add_argument("--tools", help=_TOOLS_HELP + "; overrides the packet's own flags")
    d.add_argument("--device", default=None)
    d.set_defaults(func=cmd_decode)

    i = sub.add_parser("inspect", help="print a container's sizes and headers")
    i.add_argument("input")
    i.set_defaults(func=cmd_inspect)

    b = sub.add_parser("bench", help="forward to jpegai.eval.runbench")
    b.add_argument("rest", nargs=argparse.REMAINDER,
                   help="arguments passed through to runbench")
    b.set_defaults(func=cmd_bench)

    c = sub.add_parser("conform", help="container conformance check")
    c.add_argument("input")
    c.set_defaults(func=cmd_conform)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())





