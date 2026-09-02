"""Train the whole rate ladder, then hand the checkpoints to the benchmark.

    python -m jpegai.train.runladder                       # the default 5 points
    python -m jpegai.train.runladder --betas 0.002,0.075   # just two
    python -m jpegai.train.runladder --bench               # and run the RD comparison

One β is one trained model is one point on the RD curve. Phase 3's last acceptance
criterion -- "the RD curve beats JPEG" -- is not a property of any single model, so
it cannot be checked until a ladder exists.

Why these five β
----------------
`config.rate.beta_list` has 18 entries spanning 0.0002 to 3.0, which is the range
JPEG AI's *variable-rate* mechanism covers with a handful of trained models plus
gain units (Phase 8). Training 18 models is pointless here. The default five are
chosen so that four of them are JPEG AI's own **base model** β (0.002, 0.012,
0.075) plus ladder members that fill the gaps, together spanning roughly
0.12-1.3 bpp -- which is where JPEG's own quality range sits on Kodak, and BD-rate
is only defined over the overlap.

β = 0.5 (the fourth base model) is deliberately left out of the default: at
λ·255² = 32513 it is a ~2 bpp operating point that needs the most training to look
good and contributes least to a comparison against JPEG. Add it with `--betas` when
there is GPU time.

Warm starting
-------------
Points are trained in ascending β, each initialised from the previous one's
weights. A codec at β and one at 2β differ mostly in how finely the latent is
used, not in what the transforms have learned, so the second model starts from
something already useful instead of from noise. It is purely a wall-clock
optimisation, and it makes the models *not* independent -- so `--no-warm-start`
exists for the Phase 13 ablation, where independence is the point.

`--warm-start-from ANOTHER_LADDER` is the better warm start where it applies, and it
wins over the one above: it seeds each point from the *same* β of another ladder, so a
Phase 6 point starts from a Phase 5 point that already learned this exact
rate/distortion trade-off and differs only by the module being added.

    python -m jpegai.train.runladder --model twobranch-mcm --name ladder_p6 \
        --warm-start-from checkpoints/ladder_p5

Weights only, by name, `strict=False` -- the new phase's fresh parameters stay at
their init and the LR schedule restarts from warmup.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from jpegai.config import PROJECT_ROOT, load_config
from jpegai.models import KINDS
from jpegai.train.loop import CHECKPOINT_ROOT, main as train_main

#: Ascending, so warm starting always goes from coarser to finer.
DEFAULT_BETAS = [0.002, 0.012, 0.03, 0.075, 0.2]


def _label(beta: float) -> str:
    """Directory-safe, round-trips through float() and sorts numerically."""
    return f"{beta:g}"


def run_ladder(args) -> int:
    cfg = load_config(args.tier)
    betas = sorted(float(b) for b in args.betas.split(",") if b.strip())
    root = CHECKPOINT_ROOT / args.name
    root.mkdir(parents=True, exist_ok=True)

    iters = args.iterations or cfg.train.iterations
    print(f"ladder   {args.name}  ({args.tier}, {args.model})")
    print(f"betas    {', '.join(_label(b) for b in betas)}")
    print(f"steps    {iters:,} per point  ->  {iters * len(betas):,} total")
    print(f"warm     {'yes (each point starts from the previous)' if args.warm_start else 'no (independent)'}")
    if args.warm_start_from:
        print(f"seed     {args.warm_start_from}  (same beta, weights only)")
    print(f"out      {_rel(root)}")
    print()

    summary: list[dict] = []
    prev_final: Path | None = None
    t_start = time.perf_counter()

    for i, beta in enumerate(betas, 1):
        label = _label(beta)
        sub = root / f"beta{label}"
        final = sub / "final.pt"

        if final.exists() and args.skip_done:
            print(f"[{i}/{len(betas)}] beta {label}: final.pt exists, skipping")
            prev_final = final
            summary.append(_read_summary(final, beta))
            continue

        sub.mkdir(parents=True, exist_ok=True)
        argv = [
            "--tier", args.tier, "--model", args.model,
            "--beta", repr(beta), "--iterations", str(iters),
            "--name", f"{args.name}/beta{label}",
            "--workers", str(args.workers),
            "--colour-space", args.colour_space,
            "--log-every", str(args.log_every),
            "--valid-every", str(args.valid_every),
            "--rtcheck", str(args.rtcheck),
        ]
        if args.batch:
            argv += ["--batch", str(args.batch)]
        if args.device:
            argv += ["--device", args.device]

        # Two different warm starts, and the cross-ladder one wins where it exists.
        seed_from = _cross_ladder_seed(args.warm_start_from, label)
        if seed_from is not None:
            # `--warm-start` (not `--resume`): weights only, step 0, fresh optimiser.
            argv += ["--warm-start", str(seed_from)]
        elif args.warm_start and prev_final is not None:
            # `--resume` restores the optimiser state and the step counter too,
            # which is wrong for a *new* rate point: the LR schedule must restart
            # from warmup, or the new beta gets whatever tiny LR the previous run
            # decayed to and barely moves. So copy the weights into place under a
            # name the loop will load, and let it start at step 0.
            seed_ck = sub / "warmstart.pt"
            _copy_weights_only(prev_final, seed_ck)
            argv += ["--resume", "warmstart.pt"]

        if seed_from is not None:
            how = f"  warm start from {seed_from.parent.parent.name}/beta{label}"
        elif args.warm_start and prev_final is not None:
            how = "  warm start"
        else:
            how = ""
        print(f"[{i}/{len(betas)}] beta {label} "
              f"(lambda*255^2 = {beta * 255 ** 2:.0f})" + how
              + "\n" + "-" * 70)
        rc = train_main(argv)
        if rc != 0:
            print(f"\nbeta {label} failed with code {rc}; stopping the ladder.",
                  file=sys.stderr)
            return rc
        prev_final = final
        summary.append(_read_summary(final, beta))
        _write_summary(root, summary, time.perf_counter() - t_start)

    _print_summary(summary)
    _write_summary(root, summary, time.perf_counter() - t_start)

    if args.bench:
        print("\nrunning the RD comparison\n" + "=" * 70)
        from jpegai.eval.runbench import main as bench_main

        bargv = ["--dataset", args.dataset, "--neural", str(root),
                 "--codecs", args.anchors, "--anchor", "jpeg"]
        if args.bench_metrics:
            bargv += ["--metrics", args.bench_metrics]
        if args.limit:
            bargv += ["--limit", str(args.limit)]
        return bench_main(bargv)
    else:
        rel = _rel(root)
        print(f"\nnext: python -m jpegai.eval.runbench --neural {rel} "
              f"--codecs jpeg,webp,avif")
    return 0


def _rel(path: Path) -> Path | str:
    """`path` relative to the project root when it is inside it, else unchanged."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _cross_ladder_seed(ladder_dir: str | None, label: str) -> Path | None:
    """The *same-beta* checkpoint of another ladder, or None.

    A Phase 5 point at β is a far better initialisation for a Phase 6 point at β than
    a Phase 6 point at the previous β is: it has already learned this exact
    rate/distortion trade-off and differs only by the module being added. `loop.py`'s
    `--warm-start` loads it by name with `strict=False`, so the new phase's fresh
    parameters (`mcm.*`) simply stay at their init.

    Returns None -- and says why -- when that ladder has no point at this β, so a
    ladder whose β list is not a subset of its seed's still trains every point,
    falling back to the intra-ladder warm start for the ones it cannot seed.
    """
    if not ladder_dir:
        return None
    src = Path(ladder_dir)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    cand = src / f"beta{label}" / "final.pt"
    if cand.exists():
        return cand
    print(f"         no {_rel(cand)}; falling back to the intra-ladder warm start")
    return None


def _copy_weights_only(src: Path, dst: Path) -> None:
    """Copy model weights, drop optimiser state and the step counter.

    Keeping the Adam state across a β change would carry second-moment estimates
    that were accumulated for a different objective, and keeping the step counter
    would skip the LR warmup. Both are subtly wrong in ways that show up as a
    worse model rather than an error.
    """
    import torch

    blob = torch.load(src, map_location="cpu", weights_only=False)
    torch.save({"step": 0, "model": blob["model"],
                "meta": {"warm_start_from": str(src)}}, dst)


def _read_summary(final: Path, beta: float) -> dict:
    import torch

    blob = torch.load(final, map_location="cpu", weights_only=False)
    meta = blob.get("meta", {})
    v = meta.get("valid", {})
    rt = meta.get("rtcheck", {})
    return {
        "beta": beta,
        "lambda255": beta * 255 ** 2,
        "step": int(blob.get("step", 0)),
        "valid_bpp": v.get("bpp"),
        "valid_psnr": v.get("psnr"),
        "valid_set": meta.get("valid_set"),
        "act_bpp": rt.get("act_bpp"),
        "gap_q_pct": rt.get("gap_q_pct"),
        "gap_pct": rt.get("gap_pct"),
        "y_oor_pct": rt.get("y_oor_pct"),
        "z_oor_pct": rt.get("z_oor_pct"),
        "y_exact": rt.get("y_exact"),
        # Phase 5. Old checkpoints predate these keys, hence the defaults: `True`
        # for `streams_ok` so a pre-Phase-5 ladder is not reported as failing a gate
        # its checkpoints could not have recorded, and the summary prints "--" for
        # the missing worst-stream columns rather than inventing a zero.
        "streams_ok": rt.get("streams_ok", True),
        "worst_stream": rt.get("worst_stream"),
        "worst_stream_b": rt.get("worst_stream_b"),
        "worst_stream_pct": rt.get("worst_stream_pct"),
        "path": str(final),
    }


def _print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("ladder summary   (act bpp and psnr are from the REAL bitstream)")
    print("-" * 78)
    print(f"{'beta':>8} {'lambda':>8} {'steps':>8} {'est bpp':>8} {'act bpp':>8} "
          f"{'psnr':>7} {'gap_q':>7} {'oor y':>7} {'oor z':>7} {'exact':>6} "
          f"{'worst stream':>16}")
    for r in rows:
        def f(key, spec):
            v = r.get(key)
            return format(v, spec) if isinstance(v, (int, float)) else "--"
        # The worst per-stream disagreement, named. An aggregate `gap_q` averages a
        # bad stream against every good one, so it can pass while one stream is
        # badly wrong -- which is exactly how the median-shift bug survived three
        # ladders. Whichever stream is worst gets named on every row, not only on
        # failing rows, so the column has a history to be read against.
        w = r.get("worst_stream")
        worst = (f"{w} {r['worst_stream_b']:+.0f} B"
                 if w and isinstance(r.get("worst_stream_b"), float) else "--")
        print(f"{r['beta']:>8g} {r['lambda255']:>8.0f} {r['step']:>8,} "
              f"{f('valid_bpp', '8.4f')} {f('act_bpp', '8.4f')} "
              f"{f('valid_psnr', '7.2f')} {f('gap_q_pct', '+7.2f')} "
              f"{f('y_oor_pct', '7.3f')} {f('z_oor_pct', '7.3f')} "
              f"{str(r.get('y_exact', '--')):>6} {worst:>16}")
    print("-" * 78)
    # The ladder is only meaningful if rate is monotone in beta. If it is not, a
    # point is undertrained -- say so rather than letting BD-rate silently
    # integrate a non-monotone curve.
    #
    # Checked on `valid_bpp` (the 8-image validation average) rather than
    # `act_bpp`, which `roundtrip_check` measures on a single image: one image is
    # too noisy a basis for a warning, and two adjacent beta can easily order
    # differently on one picture than on the set. The two agree to ~2% (the sigma
    # grid), so anything this would miss is a ladder too dense to be worth having.
    bpps = [r["valid_bpp"] for r in rows if isinstance(r.get("valid_bpp"), float)]
    if len(bpps) > 1 and bpps != sorted(bpps):
        print("WARNING: validation bpp is NOT monotone in beta. At least one point")
        print("         is undertrained; BD-rate over a non-monotone curve is not")
        print("         meaningful. Train the offending point(s) longer.")
    bad = [r for r in rows
           if isinstance(r.get("gap_q_pct"), float) and abs(r["gap_q_pct"]) >= 0.5]
    if bad:
        print("WARNING: the coder/table gate failed at beta "
              + ", ".join(f"{r['beta']:g} ({r['gap_q_pct']:+.2f}%)" for r in bad))
    else:
        print("gate:    every rate point agrees with its quantised-sigma estimate "
              "to within +-0.5%")

    # Escapes get their own arm because they are not reliably visible in any of the
    # arms above. An escape makes the coder fall back to a uniform range for that
    # symbol, which is *more* expensive than the table -- so it pushes the real
    # bytes up and the gap *down*, and both the aggregate gate (+-0.5% two-sided)
    # and `streams_ok` (one-sided, `excess > 16.0`) can read that as fine. beta0.002
    # of ladder_p6 did exactly this: 1.82% of z_uv symbols escaped, the z stream came
    # in 78 B under its own estimate, `streams_ok` stayed True and gap_q was -0.14%,
    # inside the band. Rebuilding the tables with current code cost 0.6% of real
    # bytes with no retraining. `out_of_range_fraction`'s docstring already says this
    # number has to be watched directly rather than inferred from a gap; this is that.
    esc = [r for r in rows
           if max(r.get("y_oor_pct") or 0.0, r.get("z_oor_pct") or 0.0) > 0.01]
    if esc:
        print("WARNING: symbols escaped the entropy tables at beta "
              + ", ".join(
                  f"{r['beta']:g} (y {r.get('y_oor_pct') or 0.0:.3f}%, "
                  f"z {r.get('z_oor_pct') or 0.0:.3f}%)" for r in esc))
        print("         Escapes cost real bytes and are the one failure here that a")
        print("         passing gate does not clear. Re-gate the checkpoint before")
        print("         retraining it -- a stale table rebuilds for free.")

    # A separate arm from the aggregate above, for the same reason `roundtrip_check`
    # keeps them separate: one stream can disagree with its own table by tens of
    # percent while the aggregate stays under 0.5%, because the aggregate divides
    # that error by every stream's bits.
    split = [r for r in rows if r.get("streams_ok") is False]
    if split:
        print("WARNING: a stream disagrees with its own entropy table at beta "
              + ", ".join(f"{r['beta']:g} ({r.get('worst_stream')} "
                          f"{r.get('worst_stream_b') or 0.0:+.0f} B, "
                          f"{r.get('worst_stream_pct') or 0.0:+.1f}%)" for r in split))
        print("         The coder is faithful. Two causes, and the `oor` columns tell")
        print("         them apart: a nonzero `oor` for that stream means symbols fell")
        print("         OUTSIDE the table and each paid an escape plus ~8 raw bits --")
        print("         a table-extent problem, fixed in `update()`. With `oor` at zero")
        print("         the table's shape is wrong instead: compare it against")
        print("         `forward()`'s density. Either way, not the rANS layer.")

    # A ladder whose points were validated on different image sets has no
    # comparable rate column at all, so this outranks the monotonicity warning.
    # It happens for real: `build_loaders` falls back to the test set when the
    # validation directory is empty, so recovering DIV2K_valid part-way through a
    # ladder silently changes the yardstick between points.
    sets = {json.dumps(r.get("valid_set")) for r in rows}
    if len(sets) > 1:
        print("WARNING: points were validated on DIFFERENT image sets:")
        for r in rows:
            print(f"           beta {r['beta']:<8g} {r.get('valid_set')}")
        print("         The bpp and psnr columns above are not comparable across")
        print("         points. Retrain the ladder with --retrain.")
    elif rows and rows[0].get("valid_set"):
        print(f"valid:   {', '.join(rows[0]['valid_set'])}")


def _write_summary(root: Path, rows: list[dict], elapsed: float) -> None:
    (root / "ladder.json").write_text(json.dumps(
        {"points": rows, "elapsed_s": round(elapsed, 1)}, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jpegai.train.runladder")
    ap.add_argument("--betas", default=",".join(f"{b:g}" for b in DEFAULT_BETAS))
    ap.add_argument("--tier", default="tierA")
    ap.add_argument("--model", default="mean-scale", choices=list(KINDS),
                    help="mean-scale by default: same cost, strictly better RD, "
                         "and it is what Phase 5 builds on. twobranch is Phase 4's "
                         "YCbCr split -- run one ladder of each to test the "
                         "'two-branch beats single-branch at equal rate' claim. "
                         "twobranch-split is Phase 5's residual coding with a "
                         "separate scale decoder; twobranch-fused is its ablation. "
                         "twobranch-mcm is Phase 6's 4-stage context model, with "
                         "twobranch-mcm2 / twobranch-mcm1 as the stage ablation "
                         "(mcm1 is the Phase 5 zero point to measure MCM against)")
    ap.add_argument("--name", default="ladder", help="subdirectory of checkpoints/")
    ap.add_argument("--iterations", type=int, default=None,
                    help="steps PER rate point; default config.train.iterations")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--colour-space", default="ycbcr", choices=["ycbcr", "rgb"])
    ap.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    ap.add_argument("--warm-start-from", default=None, metavar="LADDER_DIR",
                    help="another ladder's directory (e.g. checkpoints/ladder_p5). "
                         "Each point is seeded from that ladder's SAME beta, weights "
                         "only, which beats the intra-ladder previous-beta seed and is "
                         "how a Phase 6 ladder starts from Phase 5. Points with no "
                         "matching beta fall back to the intra-ladder warm start")
    ap.add_argument("--retrain", dest="skip_done", action="store_false",
                    help="retrain points that already have a final.pt")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--valid-every", type=int, default=2000)
    ap.add_argument("--rtcheck", type=int, default=2000)
    ap.add_argument("--bench", action="store_true",
                    help="run the RD comparison against the anchors afterwards")
    ap.add_argument("--dataset", default="kodak")
    ap.add_argument("--anchors", default="jpeg,webp,avif")
    ap.add_argument("--bench-metrics", default=None)
    ap.add_argument("--limit", type=int, default=None)
    return run_ladder(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
