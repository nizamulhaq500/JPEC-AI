# JPEG AI — cloud training on an RTX PRO 6000 (marimo notebook)
#
# Paste each CELL block below into its own marimo cell, in order.
# Cells 1-5 are setup and run once. Cell 6 launches. Cells 7-9 monitor and collect.
#
# MARIMO RULES THIS FILE OBEYS. They are not Jupyter's, and breaking them is a hard
# error rather than a warning -- an earlier draft of this file broke all four:
#   * ONE DEFINITION PER NAME across the whole notebook. Two cells that each write
#     `for p in ...` gives "This cell redefines variables from other cells", not
#     shadowing. So every cell body below lives inside a function, and only the
#     handful of names other cells genuinely need stay at top level.
#   * A LEADING UNDERSCORE makes a top-level name cell-LOCAL. That is why the helpers
#     are `_boot`, `_prepare_data`, `_launch`: several cells define a name in that
#     shape on purpose and marimo keeps them apart.
#   * CELLS RUN IN DEPENDENCY ORDER, not file order, and re-run whenever a dependency
#     changes. Editing cell 1 therefore re-triggers cell 6, so cells 4 and 6 are
#     idempotent: they guard on an artefact existing rather than redoing the work.
#   * NO SHELL MODE and no `!` magic. A bare `git clone` in a cell is parsed as
#     Python and raises SyntaxError. Everything shells out through `sh()`.
#
# The only names that cross cells: ROOT, REPO, sh (cell 1); and TIER, COMMON, SEED,
# TIERS, REQUIRE_SEED, JOBS (cell 6). Everything else is local to its cell.
#
# Design notes that matter:
#   * batch stays 8 and steps stay 50,000 for every run except `ladder_p6_long`,
#     so the results can sit in the same tables as the three MPS ladders.
#   * the unit of comparability is the LADDER, not the rate point: every Mac ladder
#     chained betas ascending from a cold beta0.002, and ladder_p6 additionally
#     seeded each point from ladder_p5 at the same beta. A single-beta run therefore
#     has to be seeded the same way or it is not comparable to anything -- hence
#     cell 2b's 45 MB upload. See cell 6's SEED block.
#   * the dataloader is replaced with an in-RAM uint8 tensor: 4 vCPUs cannot feed
#     this GPU through PNG decode, and concurrent runs would fight over them.
#   * OMP is pinned to 1 thread per job. 4 vCPUs against 4 torch processes each
#     defaulting to 4 OMP threads is 16 threads on 4 cores, and that alone can cost
#     more than the concurrency gains.
#   * every run is launched detached with setsid+nohup, NOT inside a cell, so a
#     reaped notebook session does not kill training.

# ======================= CELL 1 — clone, install, verify =======================
import os, pathlib, subprocess, sys

ROOT = pathlib.Path("/root/JPEC-AI")          # not /tmp: marimo runs from /tmp/marimo_*
REPO = "https://github.com/nizamulhaq500/JPEC-AI.git"


def sh(cmd, cwd=None, check=True, quiet=False):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not quiet:
        print(r.stdout[-4000:], flush=True)
    if check and r.returncode:
        raise RuntimeError(f"exit {r.returncode}: {cmd}")
    return r.stdout


def _boot():
    if not (ROOT / ".git").exists():
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        sh(f"git clone {REPO} {ROOT}")
    else:
        sh("git pull --ff-only", cwd=ROOT)
    sh(f"{sys.executable} -m pip install -q -r requirements.txt", cwd=ROOT)

    # compressai carries C++ extensions and is a HARD requirement: the round-trip
    # gate every 2000 steps needs its rANS coder. Without it the runs still work
    # with `--rtcheck 0`, but then nothing here proves the codec emits real bytes --
    # which is objective 3 of the project, so fix it rather than skip it.
    try:
        import compressai.ans  # noqa: F401
        print("compressai OK")
    except Exception as exc:
        print(f"compressai MISSING ({exc}); trying a source build")
        sh(f"{sys.executable} -m pip install -q --no-binary :all: compressai",
           check=False)

    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        print(f"{pr.name}  {pr.total_memory / 2**30:.1f} GiB  sm_{pr.major}{pr.minor}")
    else:
        print("NO CUDA -- STOP HERE. Every job in cell 6 passes --device cuda and "
              "will die on launch.\nSwitch the runtime to the GPU tier (the "
              "'4 CPU - 32 GiB' dropdown in the header) and re-run.")
    print(f"cpus {os.cpu_count()}")


_boot()

# ===================== CELL 2 — data (deterministic crops) =====================
# prepare_crops.py seeds its RNG with crc32(filename) and walks sorted() order, so
# the 6,400 crops regenerated here are byte-identical to the ones the Mac ladders
# trained on. That is what keeps the new runs comparable; do not change --sources,
# --crop, --per-image or --min-std.


def _prepare_data():
    div2k = "http://data.vision.ee.ethz.ch/cvl/DIV2K"
    data = ROOT / "data" / "div2k"
    data.mkdir(parents=True, exist_ok=True)

    for part, need in [("DIV2K_train_HR", 800), ("DIV2K_valid_HR", 100)]:
        d = data / part
        have = len(list(d.glob("*.png"))) if d.is_dir() else 0
        if have >= need:
            print(f"{part}: {have} images, skipping")
            continue
        z = data / f"{part}.zip"
        if not z.exists():
            sh(f"wget -q --show-progress -O {z} {div2k}/{part}.zip", cwd=data)
        sh(f"unzip -q -o {z} -d {data}", cwd=data)
        z.unlink()
        print(f"{part}: {len(list(d.glob('*.png')))} images")

    # Only div2k_train, matching data/crops/manifest.json on the Mac. flickr2k is
    # deliberately absent: adding it would change the training distribution and every
    # comparison with ladders #0-#2 along with it.
    sh(f"{sys.executable} -m jpegai.data.prepare_crops --sources div2k_train "
       f"--crop 256 --per-image 8 --min-std 8.0 --workers 4", cwd=ROOT)

    n = len(list((ROOT / "data/crops/div2k_train").glob("*.png")))
    assert n == 6400, f"expected 6400 crops, got {n} -- do NOT train until this matches"
    print(f"crops OK: {n}")


_prepare_data()

# ==================== CELL 2b — the warm-start seed (upload) ===================
# WHERE IS /root/JPEC-AI? Inside this container, not on the Mac and not in the folder
# the notebook lives in. marimo itself runs from /tmp/marimo_*, so browsing for
# "JPEC-AI" in the notebook's own directory finds nothing -- cell 1 cloned the repo to
# /root/JPEC-AI and every cell here uses that absolute path. Nothing is wrong.
#
# So the only real task is getting one 45 MB file from the Mac into the container.
# Two ways, and this cell handles both:
#
#   A. Upload it with molab's file browser (the file-tree icon in the left sidebar)
#      and do not worry where it lands. This cell hunts /root, /tmp, the cwd, the
#      home directory and /tmp/*/ for seeds.tar.gz and moves it into place itself.
#
#   B. No upload button, or it fails on 45 MB? Serve it over HTTP instead. The repo
#      is public and the container clearly has egress -- it just pulled 4 GB of
#      DIV2K -- so a GitHub release asset is the least-effort route. On the Mac:
#
#        gh release create seeds-p5 cloud/seeds/seeds.tar.gz \
#            --title "ladder_p5 warm-start seed" \
#            --notes "Stripped ladder_p5 weights for --warm-start-from. Not a git
#        object: checkpoints/ are gitignored and exceed GitHub's 100 MB blob limit."
#
#      then set SEED_URL below and re-run this cell:
#
#        SEED_URL = ("https://github.com/nizamulhaq500/JPEC-AI/releases/download/"
#                    "seeds-p5/seeds.tar.gz")
#
# Why the seed is not optional. ladder_p6/beta0.012 -- the run every single-beta job
# here is measured against -- did not start from random weights. It started from
# ladder_p5/beta0.012 (`--warm-start-from`, see logs/ladder_p6.log). A cold run at
# beta 0.012 therefore differs from it in TWO ways at once, budget and initialisation,
# and `ladder_p6_long` exists specifically to measure budget alone. So cell 6
# hard-stops that run when this seed is missing rather than spend 200,000 steps on a
# number nothing can be compared to.
#
# checkpoints/ is gitignored, which is why this cannot come down with `git pull`.
# make_seed.py drops Adam's moment buffers (loop.py's --warm-start never reads them),
# taking 144 MB to 48 MB with a verified-identical load: 117 tensors loaded,
# 28 initialised fresh, same as the original.



SEED_URL = ""      # plan B: a public URL to fetch seeds.tar.gz from, see above


def _check_seed():
    import shutil

    p5 = ROOT / "checkpoints" / "ladder_p5" / "beta0.012" / "final.pt"
    if p5.exists():
        print(f"seed present: {p5.relative_to(ROOT)} "
              f"({p5.stat().st_size / 2**20:.0f} MB)")
        return

    dst = ROOT / "seeds.tar.gz"
    if SEED_URL and not dst.exists():
        sh(f"wget -q --show-progress -O {dst} {SEED_URL}", check=False)

    # Look everywhere an upload plausibly lands rather than insisting on one path.
    hunt = [ROOT, pathlib.Path.cwd(), pathlib.Path("/root"), pathlib.Path("/tmp"),
            pathlib.Path.home(), pathlib.Path("/content"), pathlib.Path("/mnt/data")]
    found = next((d / "seeds.tar.gz" for d in hunt if (d / "seeds.tar.gz").is_file()),
                 None)
    if found is None:
        found = next(iter(pathlib.Path("/tmp").glob("*/seeds.tar.gz")), None)
    if found is None:
        print(f"NO SEED. seeds.tar.gz is not in this container yet.\n"
              f"  cwd is {pathlib.Path.cwd()}\n"
              f"  searched: {', '.join(str(d) for d in hunt)} and /tmp/*/\n"
              f"  Drop it anywhere in that list -- this cell moves it into place.\n"
              f"  No upload button? Set SEED_URL above and re-run.\n"
              f"  tier 1's sweep still runs cold (all three runs share seed 1234, so "
              f"the\n  three-way ranking is intact) -- but ladder_p6_long will refuse "
              f"to start,\n  and sweep_w6 will not double as the CUDA-vs-MPS bridge.")
        return

    print(f"found {found}  ({found.stat().st_size / 2**20:.0f} MB)")
    if found.parent != ROOT:
        shutil.move(str(found), str(dst))     # may cross filesystems, so not rename
        print(f"moved -> {dst}")
    sh(f"tar xzf {dst.name}", cwd=ROOT)
    sh("ls -la checkpoints/ladder_p5/beta*/final.pt", cwd=ROOT, check=False)
    if not p5.exists():
        print("EXTRACTED BUT WRONG SHAPE -- expected "
              "checkpoints/ladder_p5/beta0.012/final.pt inside the tar.\n"
              "Re-make it on the Mac with cloud/make_seed.py.")


_check_seed()

# ================= CELL 3 — in-RAM dataset + CUDA dataloader ==================
# 6,400 x 3 x 256 x 256 uint8 = 1.26 GiB. Held once in RAM, shared by every run
# via the OS page cache on the .npy, so 4 concurrent jobs do not decode 4x the PNGs.
#
# Written as a patch module rather than an edit to jpegai/train/dataset.py, so
# `git pull` never conflicts and the Mac tree is untouched.
#
# One honest caveat: the crop PIXELS are identical to the Mac's, but num_workers
# drops to 0, so the augmentation draw ORDER differs. The hardware-bridge run
# therefore measures (CUDA vs MPS) + (dataloader order) together, not CUDA alone.

_INRAM_SRC = '''"""In-RAM crop dataset. Import for its side effect: it monkeypatches
`jpegai.train.dataset.build_loaders` to serve crops from one uint8 tensor.
"""
from __future__ import annotations
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from jpegai.config import PROJECT_ROOT
from jpegai.train import dataset as _ds

CACHE = PROJECT_ROOT / "data" / "crops" / "_pack_256.npy"


def _pack(files, crop=256):
    if CACHE.exists():
        a = np.load(CACHE, mmap_mode="r")
        if a.shape == (len(files), 3, crop, crop):
            print(f"in-RAM crops: {CACHE.name} {a.shape} (cached)", flush=True)
            return np.ascontiguousarray(a)
    out = np.empty((len(files), 3, crop, crop), dtype=np.uint8)
    for i, f in enumerate(files):
        out[i] = _ds._load_rgb(f).transpose(2, 0, 1)
        if i % 1000 == 0:
            print(f"  packing {i}/{len(files)}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, out)
    print(f"in-RAM crops: packed {out.shape} -> {CACHE.name}", flush=True)
    return out


class InRamCrops(Dataset):
    """Same augmentation as CropDataset: the dihedral group, exact permutations."""

    def __init__(self, roots, crop=256, augment=True, seed=None):
        files = _ds.list_images(roots)
        if not files:
            raise FileNotFoundError(f"no crops under {roots}")
        self.data = torch.from_numpy(_pack(files, crop))
        self.augment = bool(augment)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        t = self.data[i]
        if self.augment:
            k = int(torch.randint(0, 4, ()))
            if k:
                t = torch.rot90(t, k, dims=(1, 2))
            if bool(torch.randint(0, 2, ())):
                t = torch.flip(t, dims=(2,))
        return t.contiguous().float().div_(255.0)


_orig = _ds.build_loaders


def build_loaders(config, *, batch=None, workers=4, valid_limit=8, seed=1234):
    train = InRamCrops(config.data.train, crop=config.train.crop, augment=True)
    _, valid = _orig(config, batch=batch, workers=0, valid_limit=valid_limit, seed=seed)
    loader = DataLoader(train, batch_size=batch or config.train.batch, shuffle=True,
                        num_workers=0, pin_memory=torch.cuda.is_available(),
                        drop_last=True)
    return loader, valid


_ds.build_loaders = build_loaders
print("patched: in-RAM crops, num_workers=0, pin_memory on CUDA", flush=True)
'''


def _write_inram():
    patch = ROOT / "jpegai" / "train" / "_inram.py"
    patch.write_text(_INRAM_SRC)
    print(f"wrote {patch.relative_to(ROOT)}")
    # Build the pack once, now, so four jobs do not all try to write it at once.
    sh(f"{sys.executable} -c \"import jpegai.train._inram as m; "
       f"from jpegai.config import load_config; c=load_config('full'); "
       f"m.InRamCrops(c.data.train, c.train.crop)\"", cwd=ROOT)


_write_inram()

# ================== CELL 4 — sanity gate before spending hours =================
# 210 checks over the whole encode -> bytes -> decode -> metrics path, then a
# 200-step throughput probe so cell 6's ETAs are measured rather than guessed.
# If this is not clean on CUDA, nothing launched below is worth its GPU time.
#
# Guarded on checkpoints/_probe, because marimo re-runs a cell whenever one of its
# dependencies changes and this one costs minutes.


def _gate():
    probe = ROOT / "checkpoints" / "_probe"
    if probe.exists():
        print(f"probe already ran; delete checkpoints/_probe to re-measure")
        return
    sh(f"{sys.executable} -m jpegai.models.selftest --device cuda",
       cwd=ROOT, check=False)
    sh(f"{sys.executable} -m jpegai.train.loop --tier full --model twobranch-mcm "
       f"--beta 0.012 --iterations 200 --batch 8 --workers 0 --device cuda "
       f"--name _probe --log-every 50 --valid-every 100000 --rtcheck 0", cwd=ROOT)


_gate()

# ================= CELL 5 — config variants for the weight sweep ===============
# distortion_weights is OURS, not normative (report 26.3), and it is the cheapest
# untested hypothesis for the luma deficit -- Kodak now puts it at psnr_y +7.2%
# against psnr_u -60.0%. Two variants around our 6:1:1.


def _write_variants():
    for tag, wy in [("w4", 4.0), ("w8", 8.0)]:
        (ROOT / "jpegai" / "config" / f"full_{tag}.yaml").write_text(
            f"# Distortion-weight sweep, {wy:g}:1:1. OURS -- see report 26.3, "
            f"27.1 item 5.\n"
            f"_base: full.yaml\n"
            f"name: full_{tag}\n"
            f"train:\n"
            f"  distortion_weights: {{y: {wy:g}, u: 1.0, v: 1.0}}\n")
        print(f"wrote full_{tag}.yaml")


_write_variants()

# ========================= CELL 6 — launch, detached ==========================
# Four processes on one card in tier 1. The model is 12.6 M params at batch 8, so
# VRAM is a few GiB each out of 95 and the card is nowhere near full -- concurrency
# within a tier is close to free, and 4 vCPUs is the real ceiling.
#
# TIERED, because the hours are limited. Tiers are sequential, jobs within a tier run
# concurrently, and tier 1 launches the LONGEST job first so the three 50k runs
# finish inside its shadow at no extra wall clock.
#
#   TIER 1   ladder_p6_long (200k steps) + the 3-way weight sweep (50k each)
#   TIER 2   ladder_p5_cont (the MCM control, 50k) + ladder_p3f (5 betas x 50k)
#
# Tier 1 is exactly four jobs for exactly four vCPUs. Tier 2 is the expensive one:
# 300,000 steps against tier 1's 350,000, but with far less concurrency to hide it.
#
# NOTHING is at reduced steps. The sweep runs at the same 50,000 as every Mac ladder,
# so its ranking is a result rather than a hint -- and sweep_w6 comes out an exact
# recipe match for ladder_p6/beta0.012, which makes it the CUDA-vs-MPS bridge for
# free. That is why there is no separate `ladder_hwbridge` any more.
#
# Set TIER and re-run this cell for each tier you can afford. Relaunching is safe:
# a job whose log already exists is skipped, which is what keeps marimo's automatic
# re-execution from starting a second copy of a 200,000-step run.

TIER = 1

# --iterations is on EVERY job on purpose: the config defaults are 600,000 (full) and
# 400,000 (tierA), so a missing flag is a twelvefold overrun rather than a typo.
COMMON = ("--batch 8 --workers 0 --device cuda --colour-space ycbcr "
          "--log-every 200 --valid-every 2000 --rtcheck 2000")

SEED = "--warm-start-from checkpoints/ladder_p5"

TIERS = {
    # ---- TIER 1: the budget probe and the weight sweep, all at full steps ------
    # Longest job first. p6_long is the only thing here that is not 50,000 steps, and
    # it is the wall clock for the whole tier; the sweep rides along inside it.
    1: {
        # 26.1's confound in its cheapest decisive form: 4x the steps at ONE beta,
        # same architecture, and the same seed weights ladder_p6/beta0.012 started
        # from -- so the only difference left is the budget. Until this runs, phase
        # 6's +0.60 dB is an upper bound. Needs cell 2b; see REQUIRE_SEED.
        "ladder_p6_long": f"--model twobranch-mcm --tier full --name ladder_p6_long "
                          f"--betas 0.012 --iterations 200000 {SEED}",

        # distortion_weights is OURS, not normative (report 26.3), and the cheapest
        # untested hypothesis for the luma deficit -- Kodak has psnr_y at +7.2% while
        # psnr_u is -60.0%. Three runs sharing config seed 1234 and differing in
        # exactly one key, so the ranking is clean with or without the warm start.
        # sweep_w6 is the 6:1:1 control AND, with the seed present, a step-for-step
        # rerun of ladder_p6/beta0.012 on CUDA -- i.e. the hardware bridge.
        "sweep_w6": f"--model twobranch-mcm --tier full    --name sweep_w6 "
                    f"--betas 0.012 --iterations 50000 {SEED}",
        "sweep_w4": f"--model twobranch-mcm --tier full_w4 --name sweep_w4 "
                    f"--betas 0.012 --iterations 50000 {SEED}",
        "sweep_w8": f"--model twobranch-mcm --tier full_w8 --name sweep_w8 "
                    f"--betas 0.012 --iterations 50000 {SEED}",
    },
    # ---- TIER 2: the MCM attribution control, and phase 3 at full width --------
    # ladder_p5_cont is the sharpest single result available for 50,000 steps, and it
    # is here rather than tier 1 only because tier 1 already has one job per vCPU.
    # ladder_p6/beta0.012 IS ladder_p5/beta0.012 plus 50,000 steps plus the MCM. Run
    # the same 50,000 steps from the same weights WITHOUT the MCM and the difference
    # is the MCM alone -- which turns phase 6's +0.60 dB from an upper bound into an
    # attributed number. Same REQUIRE_SEED logic: cold it measures nothing.
    #
    # ladder_p3f separates "the phase 3 architecture" from "the tier width". Default 5
    # betas and the default intra-ladder warm start, because that is exactly how
    # ladders #0 and #1 were run (logs/ladder.log, logs/ladder_p5.log) and the ladder
    # -- not the rate point -- is the unit of comparison. No SEED: mean-scale cannot
    # usefully load twobranch weights, and #0/#1 started cold too. 250,000 steps
    # total, so it is the expensive job here, and the only one that carries its own
    # BD-rate.
    2: {
        "ladder_p5_cont": f"--model twobranch-split --tier full --name ladder_p5_cont "
                          f"--betas 0.012 --iterations 50000 {SEED}",
        "ladder_p3f": "--model mean-scale --tier full --name ladder_p3f "
                      "--iterations 50000",
    },
}

# A job that measures a BUDGET or a MODULE difference must not also carry an
# INITIALISATION difference, so these refuse to start cold rather than produce a
# number nothing can be compared against.
REQUIRE_SEED = {"ladder_p6_long", "ladder_p5_cont"}

JOBS = {k: v for t in sorted(TIERS) if t <= TIER for k, v in TIERS[t].items()}

def _launch():
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)

    # A tiny runner rather than a `python -c` one-liner: the argument strings contain
    # commas and braces, and quoting them through nohup twice is how launch bugs
    # happen.
    runner = ROOT / "cloud_run.py"
    runner.write_text(
        "import sys\n"
        "import torch\n"
        "import jpegai.train._inram          # noqa: F401  -- in-RAM loader patch\n"
        "# Shapes are fixed (batch 8, 256x256) for 50k-200k steps, so autotuning the\n"
        "# conv algorithms pays for itself inside the first minute. TF32 is\n"
        "# deliberately NOT enabled: a 10-bit mantissa underneath an entropy model is\n"
        "# not worth the doubt, and fp32 is what the three Mac ladders ran.\n"
        "torch.backends.cudnn.benchmark = True\n"
        "from jpegai.train.runladder import main\n"
        "sys.exit(main(sys.argv[1:]))\n")

    have_seed = (ROOT / "checkpoints/ladder_p5/beta0.012/final.pt").exists()
    print(f"tier {TIER}: launching {len(TIERS[TIER])}, monitoring {len(JOBS)}, "
          f"seed {'present' if have_seed else 'MISSING (see cell 2b)'}\n")

    # 4 vCPUs against 4 torch processes that each default to 4 OMP threads is 16
    # threads on 4 cores. Pinning to 1 is worth more here than anything else.
    #
    # PYTHONUNBUFFERED because stdout redirected to a file is block-buffered at
    # 8 KiB: the progress lines flush themselves (loop.py:514) but the ladder header
    # does not, so without this the log stays EMPTY until step 200 -- and an empty
    # log is exactly when you most want to read the header and confirm the warm
    # start took. Costs nothing at one line per 200 steps.
    env = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1"

    for name, args in TIERS[TIER].items():
        if name in REQUIRE_SEED and not have_seed:
            print(f"{name:16s} SKIPPED -- needs "
                  f"checkpoints/ladder_p5/beta0.012/final.pt. Cold it would\n"
                  f"{'':17s} measure budget and initialisation together, which is "
                  f"the one thing it exists to avoid.")
            continue
        log = logs / f"cloud_{name}.log"
        if log.exists():
            print(f"{name:16s} log exists, not relaunching "
                  f"(delete {log.name} to force)")
            continue
        argv = " ".join(args.split()) + " " + COMMON
        cmd = (f"setsid env {env} nohup {sys.executable} {runner.name} {argv} "
               f"> {log} 2>&1 &")
        subprocess.Popen(cmd, shell=True, cwd=ROOT)
        print(f"launched {name:16s} -> logs/{log.name}")


_launch()

# ============================ CELL 7 — monitor ================================
# Re-run this cell whenever you want a status board. Safe to run any time; it only
# reads. json/time are imported inside the function so they are not notebook-wide
# definitions that a later cell would collide with.


def _status():
    import json, time

    print(time.strftime("%H:%M:%S"))
    sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,"
       "temperature.gpu --format=csv,noheader", check=False)
    # Liveness, separately from the logs. An empty log is ambiguous on its own --
    # a job still in startup and a job that died before its first flush look the
    # same in the file -- and this is what tells them apart.
    sh("pgrep -af cloud_run.py | grep -o -- '--name [A-Za-z0-9_]*' | sort | uniq -c"
       " || echo '  (no cloud_run.py processes)'", check=False)

    for name in JOBS:
        log = ROOT / "logs" / f"cloud_{name}.log"
        if not log.exists():
            print(f"\n=== {name}: no log")
            continue
        st = log.stat()
        lines = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
        prog = [ln for ln in lines if "/" in ln and "it/s" in ln]
        print(f"\n=== {name}   {st.st_size:,} B, last write "
              f"{(time.time() - st.st_mtime) / 60:.1f} min ago")
        if prog:
            print("   " + prog[-1].strip())
        elif lines:
            print("   " + lines[-1][:150])
        else:
            # An empty log is the NORMAL state for the first few minutes and used to
            # crash this cell on `lines[-1]`. Only the progress and validation lines
            # flush (loop.py:514, 557, 564); the ladder header does not, so with
            # stdout redirected to a file it sits in an 8 KiB block buffer until the
            # first `--log-every` line pushes it out. Before that there is genuinely
            # nothing in the file, and startup is not instant either: torch import,
            # the 677 MiB crop tensor, then cudnn.benchmark autotuning the convs.
            print("   empty -- nothing has flushed yet. First write is the step-200"
                  " progress line;\n       until then the header is still in stdout's"
                  " buffer. Check pgrep above:\n       named there = alive and"
                  " starting up, absent = it died, so read the log in full.")

        j = ROOT / "checkpoints" / name / "ladder.json"
        if j.exists():
            for pt in json.loads(j.read_text()).get("points", []):
                print(f"   beta {pt['beta']:<7g} step {pt.get('step', 0):>7,} "
                      f"bpp {pt.get('valid_bpp') or float('nan'):.4f} "
                      f"psnr {pt.get('valid_psnr') or float('nan'):.2f} "
                      f"gap_q {pt.get('gap_q_pct') or float('nan'):+.2f}% "
                      f"exact {pt.get('y_exact')}")


_status()

# ===================== CELL 8 — package results for download ==================
# Run when a job finishes. Assume this filesystem is NOT durable: pull the tarball
# down before the session ends. ~60-80 MB per rate point.


def _package():
    import tarfile

    out = ROOT / "cloud_results.tar.gz"
    with tarfile.open(out, "w:gz") as tf:
        for name in JOBS:
            for rel in (f"checkpoints/{name}/ladder.json", f"logs/cloud_{name}.log"):
                src = ROOT / rel
                if src.exists():
                    tf.add(src, arcname=rel)
            for f in (ROOT / "checkpoints" / name).rglob("final.pt"):
                tf.add(f, arcname=str(f.relative_to(ROOT)))
    print(f"{out}  {out.stat().st_size / 2**20:.0f} MiB")
    print("download this, then on the Mac:")
    print("  tar xzf cloud_results.tar.gz")
    print("  .venv/bin/python -m jpegai.eval.runbench --neural "
          "checkpoints/ladder_p3f --codecs jpeg,webp,avif")


_package()

# ============ CELL 9 — fill ladder_p6's rate hole (50k steps, ~40 min) ========
# This closes the one real methodological weakness in the headline BD-rate.
# ladder_p6 jumps beta 0.002 -> 0.012, which is 0.456 -> 0.963 bpp: a 2.1x hole
# with FOUR of JPEG's eleven Kodak points inside it, so PCHIP interpolates the
# neural curve straight across the anchor's densest region to produce the number.
# One point at beta 0.005 (lambda*255^2 = 325, ~0.6 bpp) removes it.
#
# Running it HERE rather than on the Mac is defensible now that sweep_w6 has
# landed: same model, tier, beta, batch, seed and step count as the Mac's
# ladder_p6/beta0.012, and it came in at 0.9571 bpp / 32.58 dB against the Mac's
# 0.9628 / 32.59 -- 0.6% cheaper at equal PSNR, so the CUDA and MPS paths agree
# to 0.01 dB. One CUDA point in an MPS ladder costs far less than the hole does.
# Say so in the report rather than hiding it.
#
# The warm start has to be ladder_p6/beta0.002, which is what --skip-done would
# have chained from on the Mac. On the Mac:
#
#   .venv/bin/python cloud/make_seed.py --ladder checkpoints/ladder_p6 \
#       --betas 0.002 --out cloud/seeds_p6
#
# then upload cloud/seeds_p6/seeds.tar.gz. Its NAME does not matter: this cell
# reads the CONTENTS of every archive it can find and takes the one holding the
# member it needs, which is also why the stale seeds.tar*.gz copies already in
# the notebook folder cannot fool it.
#
# It calls loop.py directly for a single point instead of going through
# runladder -- exactly how beta0.0005 and beta0.001 were trained -- so the
# checkpoint lands inside the existing ladder_p6 tree. ladder.json is stale
# afterwards by design; regenerate it on the Mac, as with those two.
#
# This is a 4th concurrent job against 4 vCPUs, i.e. back to tier 1's load. The
# three tier-2 runs slow proportionally; nothing else changes.

def _fill_hole():
    import tarfile

    want = "checkpoints/ladder_p6/beta0.002/final.pt"
    seed = ROOT / want
    log = ROOT / "logs" / "cloud_p6_hole.log"

    if log.exists():
        print(f"log exists, not relaunching (delete logs/{log.name} to force)")
        return

    if not seed.exists():
        # Content-addressed rather than name-addressed: three stale seeds.tar*.gz
        # are already sitting next to the notebook and none of them holds this
        # member, so matching on the filename would pick the wrong archive.
        cands = []
        for d in {pathlib.Path.cwd(), ROOT, pathlib.Path("/root"),
                  pathlib.Path("/tmp"), pathlib.Path.home()}:
            cands += sorted(d.glob("*.tar*gz"))
        cands += sorted(pathlib.Path("/tmp").glob("*/*.tar*gz"))
        for c in cands:
            try:
                with tarfile.open(c) as t:
                    names = t.getnames()
            except Exception:
                continue                      # not a tar, or a partial upload
            if any(n.endswith("ladder_p6/beta0.002/final.pt") for n in names):
                print(f"seed found inside {c}")
                subprocess.run(["tar", "xzf", str(c), "-C", str(ROOT)], check=True)
                break
        else:
            print(f"NO SEED for the hole point -- need {want} here.\n"
                  f"  On the Mac:  .venv/bin/python cloud/make_seed.py "
                  f"--ladder checkpoints/ladder_p6 --betas 0.002 --out cloud/seeds_p6\n"
                  f"  then upload cloud/seeds_p6/seeds.tar.gz anywhere in the box.\n"
                  f"  Scanned {len(cands)} archive(s); none held that member.\n"
                  f"  Refusing to start cold: an unseeded point is not comparable to "
                  f"the ladder it is joining, which is the entire purpose here.")
            return

    if not seed.exists():
        print(f"extracted, but {want} is still missing -- the archive has a "
              f"different internal layout than make_seed.py writes")
        return

    _HOLE_LAUNCH(want, log)


def _HOLE_LAUNCH(want, log):
    # A second runner beside cloud_run.py, because this one enters loop.main (a
    # single rate point) rather than runladder.main (a whole ladder). Same two
    # preambles: the in-RAM loader patch and cudnn autotuning, TF32 still off.
    runner = ROOT / "cloud_point.py"
    runner.write_text(
        "import sys\n"
        "import torch\n"
        "import jpegai.train._inram          # noqa: F401  -- in-RAM loader patch\n"
        "torch.backends.cudnn.benchmark = True\n"
        "from jpegai.train.loop import main\n"
        "sys.exit(main(sys.argv[1:]))\n")

    # --name carries the beta subdirectory itself, which is the convention
    # runladder uses (`f"{name}/beta{label}"`), so the checkpoint lands at
    # checkpoints/ladder_p6/beta0.005/ and runbench picks it up with no flags.
    argv = (f"--model twobranch-mcm --tier full --beta 0.005 --iterations 50000 "
            f"--name ladder_p6/beta0.005 --warm-start {want} {COMMON}")
    cmd = (f"setsid env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 "
           f"nohup {sys.executable} {runner.name} {argv} > {log} 2>&1 &")
    subprocess.Popen(cmd, shell=True, cwd=ROOT)
    print(f"launched ladder_p6/beta0.005  (lambda*255^2 = {0.005 * 255 ** 2:.0f}, "
          f"expect ~0.6 bpp)  -> logs/{log.name}")
    print("cell 7 will not show it -- it iterates JOBS. Tail it with:")
    print("  print((ROOT / 'logs' / 'cloud_p6_hole.log').read_text()[-1500:])")


_fill_hole()

# ---- the other thing worth doing with spare hours -----------------------------
# `ladder_p5_cont` in tier 2 gives the MCM attribution at ONE rate point, which is
# enough to state the number. `ladder_p5_long` gives it at all five, which is what
# a BD-rate needs -- the same control, but able to say "the MCM is worth X% of the
# rate" instead of "+Y dB at 0.4 bpp". 250,000 steps, and it needs all five seed
# checkpoints rather than one:
#
#   .venv/bin/python cloud/make_seed.py --all     # 720 MB of checkpoints -> ~240 MB
#
# upload, re-run cell 2b, then paste this as a cell of its own:
#
#   def _long():
#       sh("setsid env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 nohup "
#          f"{sys.executable} cloud_run.py "
#          "--model twobranch-split --tier full --name ladder_p5_long "
#          "--warm-start-from checkpoints/ladder_p5 --iterations 50000 "
#          f"{COMMON} > logs/cloud_ladder_p5_long.log 2>&1 &", cwd=ROOT)
#   _long()
#
# No --betas: the default grid is already 0.002/0.012/0.03/0.075/0.2, which is what
# ladders #0-#2 used. It is second in line behind the hole point because 117ef49
# already put the 4-stage MCM under 1%, so this refines a number known to be small,
# at 5x the steps and 5x the upload.

