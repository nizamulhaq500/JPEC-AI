# Strip a ladder's final.pt checkpoints down to what `--warm-start` actually reads.
#
# Run this ON THE MAC, then upload cloud/seeds/ to the GPU box.
#
# Why it exists: `checkpoints/ladder_p5/beta0.012/final.pt` is 144 MB, but 101 MB of
# that is Adam's two moment buffers over 12.6 M parameters, and loop.py's
# `--warm-start` deliberately does not load them (it starts a fresh optimiser at
# step 0 -- see loop.py:391-400). So the upload is 3x larger than the information
# in it. Stripped, one seed is ~48 MB.
#
#   python cloud/make_seed.py                    # just beta0.012 (the two single-beta runs)
#   python cloud/make_seed.py --all              # all five betas (also unlocks ladder_p5_long)
#
# Output layout mirrors the checkpoint tree, so on the cloud box it untars straight
# into place and `--warm-start-from checkpoints/ladder_p5` finds it with no flags.

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

# Run by path, not as a module (`python cloud/make_seed.py`), so put the repo root on
# the path before importing jpegai.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from jpegai.config import PROJECT_ROOT

ALL_BETAS = ["0.002", "0.012", "0.03", "0.075", "0.2"]


def strip_one(src: Path, dst: Path) -> tuple[int, int]:
    """Weights + step + meta only. Same shape `_cross_ladder_seed` would have written."""
    blob = torch.load(src, map_location="cpu", weights_only=False)
    meta = dict(blob.get("meta", {}))
    # Keep `meta['model']`: loop.py prints it in the warm-start banner, and a banner
    # that says "? @ step 50,000" is how you fail to notice you seeded the wrong arch.
    out = {"step": int(blob.get("step", 0)), "model": blob["model"], "meta": meta}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return src.stat().st_size, dst.stat().st_size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="checkpoints/ladder_p5")
    ap.add_argument("--betas", default="0.012",
                    help="comma-separated; ignored when --all is given")
    ap.add_argument("--all", action="store_true", help="every beta in the ladder")
    ap.add_argument("--out", default="cloud/seeds")
    args = ap.parse_args(argv)

    root = PROJECT_ROOT / args.ladder
    out_root = PROJECT_ROOT / args.out
    betas = ALL_BETAS if args.all else [b.strip() for b in args.betas.split(",") if b.strip()]

    made: list[Path] = []
    tot_in = tot_out = 0
    for b in betas:
        src = root / f"beta{b}" / "final.pt"
        if not src.exists():
            print(f"  beta {b:<6} MISSING {src.relative_to(PROJECT_ROOT)}")
            continue
        rel = Path(args.ladder) / f"beta{b}" / "final.pt"
        dst = out_root / rel
        a, c = strip_one(src, dst)
        tot_in += a
        tot_out += c
        made.append(rel)
        print(f"  beta {b:<6} {a / 2**20:>6.0f} MB -> {c / 2**20:>5.0f} MB  {rel}")

    if not made:
        print("nothing to do -- no checkpoints matched", file=sys.stderr)
        return 1

    tar = out_root / "seeds.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for rel in made:
            tf.add(out_root / rel, arcname=str(rel))
    print(f"\n{tar.relative_to(PROJECT_ROOT)}  {tar.stat().st_size / 2**20:.0f} MB "
          f"({tot_in / 2**20:.0f} MB of checkpoint -> {tot_out / 2**20:.0f} MB of weights)")
    print("upload it to /root/JPEC-AI/ on the GPU box, then run cell 2b.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
