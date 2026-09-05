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

# ============== CELL 1b — limits, and prove a file can get out ================
# Runs before anything expensive, because two containers have now been lost and
# both times the archives could not be reached from the file browser. Three
# questions, measured rather than assumed:
#
#   1. How much RAM does this container actually get? Each training process holds
#      its own copy of the crop pack unless cell 3 returns the mmap -- 1.17 GiB
#      per process, 8.2 GiB across seven jobs, all of it the same bytes. Cell 3
#      now returns the map; this prints both numbers so the fix is visible.
#   2. How much disk? loop.py:572-573 writes best.pt and latest.pt on every
#      validation as well as final.pt, so the checkpoint bill is 3x the obvious
#      one -- ~5 GiB across eleven rate points, not 1.7.
#   3. Can a file reach the Mac AT ALL? Click the button below. If a 1 KiB file
#      cannot get down, nothing else in this notebook matters, and the place to
#      discover that is here rather than four hours in.


def _preflight():
    import shutil

    # cgroup v2 then v1. An unlimited cgroup reads "max", which fails int() and
    # falls through to MemTotal.
    cap = None
    for f in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = int(pathlib.Path(f).read_text().split()[0])
        except (OSError, ValueError):
            continue
        if 0 < v < 2**50:
            cap = v
            break
    host = next((int(x.split()[1]) * 1024
                 for x in pathlib.Path("/proc/meminfo").read_text().splitlines()
                 if x.startswith("MemTotal:")), 0)
    ram = min(x for x in (cap, host) if x)
    print(f"RAM      {host / 2**30:6.1f} GiB host, cgroup cap "
          f"{f'{cap / 2**30:.1f} GiB' if cap else 'none'}"
          f"  -> {ram / 2**30:.1f} GiB usable, {os.cpu_count()} vCPU")

    pack = 6400 * 3 * 256 * 256          # the crop tensor, uint8
    proc = 2.0 * 2**30                   # torch + CUDA context + cuDNN, per process
    for n in (7, 4):
        shared, private = pack + n * proc, n * (pack + proc)
        print(f"  {n} jobs {shared / 2**30:6.1f} GiB shared (mmap, what cell 3 does"
              f" now)   {private / 2**30:5.1f} GiB if each copies"
              f"   {'OK' if shared < 0.8 * ram else 'TIGHT -- run tier 1 only'}")

    du = shutil.disk_usage(ROOT if ROOT.exists() else "/")
    need = (7.6 * 2**30                      # DIV2K train+valid; cell 2 deletes the zips
            + 1.5 * 2**30                    # 6400 crop PNGs
            + pack                           # _pack_256.npy
            + 3 * (6 * 177 + 5 * 114) * 2**20    # final + latest + best, 11 points
            + 1.3 * 2**30)                   # the archives cell 8 writes
    print(f"disk     {du.free / 2**30:6.1f} GiB free of {du.total / 2**30:.1f}"
          f"   campaign needs ~{need / 2**30:.1f} GiB"
          f"   {'OK' if need < 0.85 * du.free else 'WILL NOT FIT'}")

    # The thing both losses actually turned on: ROOT is not what the sidebar shows.
    probe = pathlib.Path.cwd() / "EXFIL_TEST.txt"
    probe.write_text("If this reached the Mac, the download path works.\n")
    print(f"\nwrote {probe}\n  cwd is what the file browser lists. {ROOT} is where"
          f" every archive gets written\n  and it has never appeared there -- which"
          f" is why cells 7 and 8 now stage into cwd.")
    try:
        import marimo as mo
        return mo.download(probe.read_bytes(), filename=probe.name)
    except Exception as exc:
        print(f"  no button ({exc.__class__.__name__}) -- open the file-tree icon in"
              f" the left sidebar\n  and look for {probe.name}. If it is not there,"
              f" STOP: fix the route before training.")


_preflight()

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
import numpy as np, torch, warnings
from torch.utils.data import Dataset, DataLoader
from jpegai.config import PROJECT_ROOT
from jpegai.train import dataset as _ds

CACHE = PROJECT_ROOT / "data" / "crops" / "_pack_256.npy"


def _pack(files, crop=256):
    if CACHE.exists():
        a = np.load(CACHE, mmap_mode="r")
        if a.shape == (len(files), 3, crop, crop):
            print(f"in-RAM crops: {CACHE.name} {a.shape} (mmap)", flush=True)
            # Return the map, NOT a copy. np.save writes C order, so the mmap is
            # already contiguous and ascontiguousarray materialised a private
            # 1.17 GiB in every process -- 8.2 GiB across seven concurrent jobs,
            # all of it the same bytes off the same file. Through the page cache
            # it is 1.17 GiB once, shared. Nothing writes to it: __getitem__ does
            # .float(), which copies, before div_ touches anything.
            return a
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
        a = _pack(files, crop)
        # from_numpy warns on a read-only mmap and hands back a tensor it says is
        # unsafe to write. Nothing here writes to it, so the warning is noise in
        # seven job logs -- but if some torch version turns that into an error,
        # fall back to a private copy rather than failing at step 0.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.data = torch.from_numpy(a)
        except (ValueError, TypeError):
            self.data = torch.from_numpy(np.array(a))
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
# untested hypothesis for the luma deficit -- Kodak puts psnr_y at +6.4% against
# psnr_u at -60.5%.
#
# THE RANGE IS WIDER THAN IT WAS, and that is the whole point of running this again.
# The first sweep tried 4:1:1 / 6:1:1 / 8:1:1 and came back 32.55 / 32.58 / 32.60 dB
# overall, Y 32.85 / 32.96 / 33.02, U 40.44 / 40.36 / 40.29 -- monotone in the right
# direction, ~1% of rate end to end, and therefore a null result. It was null because
# the range was too narrow to be informative, not because the knob does nothing:
# chroma is reconstructing ~7 dB ABOVE luma, so a 2x change in the luma weight moves
# an allocation that is already lopsided by ~7 dB nowhere near enough to see. The
# informative span is 1:1:1 (no luma preference at all) to 24:1:1 (four times ours),
# which brackets our 6:1:1 by a factor of 6 in each direction instead of 1.5.
#
# 6:1:1 stays in as the control, and it is also the exact recipe of
# ladder_p6/beta0.012 -- so it doubles as the CUDA-vs-MPS bridge, for free, exactly
# as sweep_w6 did last time (0.9571 bpp / 32.58 dB on CUDA against the Mac's
# 0.9628 / 32.59, i.e. agreement to ~0.01 dB after rate normalisation).
#
# Same three runs, same 50,000 steps, same 150,000 GPU-steps as before. The only
# difference is that this time the answer can be something other than "no effect".


def _write_variants():
    for tag, wy in [("w1", 1.0), ("w24", 24.0)]:
        (ROOT / "jpegai" / "config" / f"full_{tag}.yaml").write_text(
            f"# Distortion-weight sweep, {wy:g}:1:1. OURS -- see report 26.3, "
            f"27.1 item 5.\n"
            f"_base: full.yaml\n"
            f"name: full_{tag}\n"
            f"train:\n"
            f"  distortion_weights: {{y: {wy:g}, u: 1.0, v: 1.0}}\n")
        print(f"wrote full_{tag}.yaml   ({wy:g}:1:1)")


_write_variants()

# ========================= CELL 6 — launch, detached ==========================
# LAUNCH EVERYTHING AT ONCE. The first run of this notebook went tier-by-tier, four
# jobs at a time, on the theory that 4 vCPUs is the ceiling. The monitor board then
# showed the card at 66% utilisation and 1,543 MiB of 95 GiB in use -- so the ceiling
# was neither VRAM (each job is ~400 MiB; the card would hold two hundred of them) nor
# the GPU itself. Holding runs back in a second tier bought nothing and cost the
# wall clock of a whole extra pass.
#
# The arithmetic that matters, all of it measured on this card last time: one job
# alone runs at 15.88 it/s, four concurrent jobs aggregate to 60.7 it/s (15.2 each),
# so concurrency inside a tier was already nearly free. Total work here is 700,000
# steps. Spread over 7 lanes that is ~2 h of card time -- but `ladder_p6_long` is a
# single indivisible 200,000-step job, so the critical path is 200,000 / ~15 =
# **~3.7 h whatever else runs beside it**. Every other run therefore belongs in
# p6_long's shadow, which is what TIER = 0 does.
#
#   ladder_p6_long   200k   the step budget: 4x the steps at one beta. The biggest
#                           single finding last time (+0.55 dB at matched rate).
#   ladder_p3f       250k   5 betas x 50k. Single-branch mean-scale, and the only run
#                           here that carries its own Kodak BD-rate.
#   ladder_p5_cont    50k   the MCM control: same seed, same steps, no MCM.
#   sweep_w1/6/24    150k   the luma-weight sweep, on a range wide enough to answer
#                           (cell 5). w6 doubles as the CUDA-vs-MPS bridge.
#   (cell 9)          50k   beta 0.0002, which takes the Kodak overlap to 8/11.
#
# 7 jobs on 4 vCPUs is oversubscribed, and whether that helps or hurts is a question
# about kernel-launch overhead that is cheaper to MEASURE than to predict: cell 7
# prints aggregate it/s, so compare it against 60.7 within the first two minutes. If
# it came out lower, kill `sweep_w1` and `sweep_w24` (the least valuable pair) and the
# rest speeds back up. Set TIER = 1 or 2 to fall back to the old sequential scheme.
#
# NOTHING is at reduced steps. The sweep runs at the same 50,000 as every Mac ladder,
# so its ranking is a result rather than a hint.
#
# Relaunching is safe: a job whose log already exists is skipped, which is what keeps
# marimo's automatic re-execution from starting a second copy of a 200,000-step run.

TIER = 0      # 0 = everything concurrently (recommended); 1 or 2 = that tier only

# --iterations is on EVERY job on purpose: the config defaults are 600,000 (full) and
# 400,000 (tierA), so a missing flag is a twelvefold overrun rather than a typo.
COMMON = ("--batch 8 --workers 0 --device cuda --colour-space ycbcr "
          "--log-every 200 --valid-every 2000 --rtcheck 2000")

SEED = "--warm-start-from checkpoints/ladder_p5"

TIERS = {
    # The two groups below are no longer a schedule -- TIER = 0 launches all of them
    # together. They are kept as GROUPS because the split still records something
    # true: group 1 is the runs that share a warm start and a beta, group 2 is the
    # two that stand alone. Set TIER = 1 or 2 to get the old sequential behaviour.
    #
    # ---- 1: the budget probe and the weight sweep, all at full steps -----------
    # Longest job first, so p6_long's 200,000 steps start before anything else
    # competes for the card.
    1: {
        # 26.1's confound in its cheapest decisive form: 4x the steps at ONE beta,
        # same architecture, and the same seed weights ladder_p6/beta0.012 started
        # from -- so the only difference left is the budget. Until this runs, phase
        # 6's +0.60 dB is an upper bound. Needs cell 2b; see REQUIRE_SEED.
        "ladder_p6_long": f"--model twobranch-mcm --tier full --name ladder_p6_long "
                          f"--betas 0.012 --iterations 200000 {SEED}",

        # distortion_weights is OURS, not normative (report 26.3), and the cheapest
        # untested hypothesis for the luma deficit -- Kodak has psnr_y at +6.4% while
        # psnr_u is -60.5%. Three runs sharing config seed 1234 and differing in
        # exactly one key, so the ranking is clean with or without the warm start.
        # The span is 1:1:1 to 24:1:1 this time, not 4 to 8: see cell 5 for why the
        # narrow version could only ever come back null.
        # sweep_w6 is the 6:1:1 control AND, with the seed present, a step-for-step
        # rerun of ladder_p6/beta0.012 on CUDA -- i.e. the hardware bridge.
        "sweep_w6": f"--model twobranch-mcm --tier full     --name sweep_w6 "
                    f"--betas 0.012 --iterations 50000 {SEED}",
        "sweep_w1": f"--model twobranch-mcm --tier full_w1  --name sweep_w1 "
                    f"--betas 0.012 --iterations 50000 {SEED}",
        "sweep_w24": f"--model twobranch-mcm --tier full_w24 --name sweep_w24 "
                     f"--betas 0.012 --iterations 50000 {SEED}",
    },
    # ---- 2: the MCM attribution control, and phase 3 at full width -------------
    # ladder_p5_cont is the sharpest single result available for 50,000 steps.
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
    # BD-rate. Last time it also came back with `exact False` at beta 0.03 -- watch
    # that row on cell 7's board rather than discovering it at bench time.
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

# Every job, in tier order, which is also longest-first -- so the 200,000-step
# critical path starts before anything queues behind it. TIER = 0 launches all of
# them; a nonzero TIER keeps the old sequential behaviour for when the card is shared.
JOBS = {k: v for t in sorted(TIERS) for k, v in TIERS[t].items()}


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

    want = JOBS if TIER == 0 else TIERS[TIER]
    have_seed = (ROOT / "checkpoints/ladder_p5/beta0.012/final.pt").exists()

    def _steps(a):
        """GPU-steps a job will actually run: --iterations is PER rate point."""
        n = (len(a.split("--betas")[1].split()[0].split(","))
             if "--betas" in a else 5)      # no --betas = runladder's default grid
        return int(a.split("--iterations")[1].split()[0]) * n

    total = sum(_steps(a) for a in want.values())
    print(f"{'all tiers' if TIER == 0 else f'tier {TIER}'}: launching {len(want)} job(s), "
          f"{total:,} GPU-steps, seed "
          f"{'present' if have_seed else 'MISSING (see cell 2b)'}\n")

    # 4 vCPUs against 4 torch processes that each default to 4 OMP threads is 16
    # threads on 4 cores. Pinning to 1 is worth more here than anything else.
    #
    # PYTHONUNBUFFERED because stdout redirected to a file is block-buffered at
    # 8 KiB: the progress lines flush themselves (loop.py:514) but the ladder header
    # does not, so without this the log stays EMPTY until step 200 -- and an empty
    # log is exactly when you most want to read the header and confirm the warm
    # start took. Costs nothing at one line per 200 steps.
    env = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1"

    for name, args in want.items():
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
# reads. json/re/time are imported inside the function so they are not notebook-wide
# definitions that a later cell would collide with.
#
# THIS BOARD IS THE BACKUP. A rented container gets reset, and when one did, eleven
# checkpoints and ~150 KB of logs went with it -- every number that survived did so
# because it had been pasted into a chat window. So this cell prints each rate point
# in FULL rather than a headline, and the habit that costs nothing is to paste the
# output somewhere durable each time you poll. Downloading cell 8's numbers archive
# is the belt; this is the braces, and it needs no download at all.
#
# It also prints aggregate it/s. The card did 60.7 it/s across four concurrent jobs
# last time at 66% utilisation, so with seven jobs the question is whether
# oversubscribing 4 vCPUs helps or hurts -- read the number off this board in the
# first two minutes rather than guessing. Lower than ~60 means kill sweep_w1 and
# sweep_w24.


def _status():
    import json, re, time

    print(time.strftime("%H:%M:%S"))
    sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,"
       "temperature.gpu --format=csv,noheader", check=False)
    # Liveness, separately from the logs. An empty log is ambiguous on its own --
    # a job still in startup and a job that died before its first flush look the
    # same in the file -- and this is what tells them apart.
    sh("pgrep -af 'cloud_run.py|cloud_point.py' | grep -o -- '--name [A-Za-z0-9_./]*'"
       " | sort | uniq -c || echo '  (no training processes)'", check=False)

    rate_total = 0.0
    live = 0

    # Iterate the LOGS, not JOBS: cell 9's low point is launched outside the tier
    # dictionary, and keying off JOBS silently omitted it. Anything that writes
    # logs/cloud_*.log shows up here now, whoever launched it.
    for log in sorted((ROOT / "logs").glob("cloud_*.log")):
        name = log.stem[len("cloud_"):]
        st = log.stat()
        lines = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
        prog = [ln for ln in lines if "/" in ln and "it/s" in ln]
        idle = (time.time() - st.st_mtime) / 60
        print(f"\n=== {name}   {st.st_size:,} B, last write {idle:.1f} min ago")
        if prog:
            print("   " + prog[-1].strip())
            # Only a LIVE log contributes to the aggregate. A finished job's last
            # line still carries an it/s, and counting it would inflate the total
            # with throughput the card is no longer producing.
            m = re.search(r"([\d.]+)\s*it/s", prog[-1])
            if m and idle < 3.0:
                rate_total += float(m.group(1))
                live += 1
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

        # The full row, not a headline. These lines are the durable record if the
        # container dies, so they carry the columns a table would need: the real
        # bitstream's bpp beside the estimate, the escape fractions, and the worst
        # per-stream disagreement. See runladder._print_summary for what each gate
        # means; this is the same data before the warnings are applied to it.
        j = ROOT / "checkpoints" / name / "ladder.json"
        if j.exists():
            pts = json.loads(j.read_text()).get("points", [])
            if pts:
                print(f"   {'beta':>7} {'step':>7} {'est bpp':>8} {'act bpp':>8} "
                      f"{'psnr':>6} {'gap_q':>7} {'oor y':>6} {'oor z':>6} "
                      f"{'exact':>5}  worst")
            for pt in pts:
                def g(k, spec):
                    v = pt.get(k)
                    return format(v, spec) if isinstance(v, (int, float)) else "--"
                w = pt.get("worst_stream")
                worst = (f"{w} {pt['worst_stream_b']:+.0f} B"
                         if w and isinstance(pt.get("worst_stream_b"), float) else "--")
                print(f"   {pt['beta']:>7g} {pt.get('step', 0):>7,} "
                      f"{g('valid_bpp', '8.4f')} {g('act_bpp', '8.4f')} "
                      f"{g('valid_psnr', '6.2f')} {g('gap_q_pct', '+7.2f')} "
                      f"{g('y_oor_pct', '6.3f')} {g('z_oor_pct', '6.3f')} "
                      f"{str(pt.get('y_exact', '--')):>5}  {worst}")
        else:
            # A single point launched through loop.main (cell 9) never writes a
            # ladder.json -- its numbers exist only in this log, so surface them here
            # rather than leaving the one run with no durable row.
            for ln in [x for x in lines if "bpp" in x.lower()][-3:]:
                print("   " + ln.strip()[:160])

    print(f"\naggregate {rate_total:>6.1f} it/s across {live} live job(s)"
          f"   (4-job baseline was 60.7)")
    if live > 4 and rate_total and rate_total < 55:
        print("  Oversubscription is costing you. Kill the two least valuable runs")
        print("  (trailing space in the pattern: `pkill -f` is a regex over the whole")
        print("  cmdline, and `sweep_w1` without it also matches nothing useful):")
        print("    sh(\"pkill -f 'name sweep_w1 '\", check=False)")
        print("    sh(\"pkill -f 'name sweep_w24 '\", check=False)")
    print("PASTE THIS BOARD SOMEWHERE DURABLE, then click the button below.")


def _bank():
    """Rebuild the numbers archive and return it as a download button.

    Cell 8 did this once, at the end, and the container was reclaimed before
    anyone pressed anything -- twice, 4.3 h of GPU each time. So it happens on
    every poll now instead. Three of the five results here (step budget, MCM
    attribution, weight sweep) are single-beta DIV2K comparisons whose every
    number lives in a ladder.json, which is why this archive is ~100 KiB and
    still carries them all.

    Nothing on the box is durable. /root/JPEC-AI is gone after a reset and so is
    the notebook's own directory -- /marimo looked persistent because a seed
    survived there once, but that was a kernel restart, not a container reset.
    molab keeps the notebook source and nothing else. The Mac is the only storage.
    """
    import tarfile

    out = ROOT / "cloud_numbers.tar.gz"
    n = 0
    with tarfile.open(out, "w:gz") as tf:
        for d, pat in [(ROOT / "checkpoints", "*/ladder.json"),
                       (ROOT / "logs", "cloud_*.log"),
                       (ROOT / "results", "*.json")]:
            for p in sorted(d.glob(pat)) if d.is_dir() else []:
                arc = (f"cloud_results/{p.parent.name}_ladder.json"
                       if p.name == "ladder.json" else f"cloud_results/{p.name}")
                tf.add(p, arcname=arc)
                n += 1
    print(f"banked {out.name}  {out.stat().st_size / 2**10:.0f} KiB, {n} file(s)")

    # Staged into the notebook's directory too: a download button that fails leaves
    # nothing behind, and the file browser is the only other way out of here.
    try:
        import shutil
        shutil.copy2(out, pathlib.Path.cwd() / out.name)
    except Exception as exc:
        print(f"  stage failed: {exc}")
    try:
        import marimo as mo
        return mo.download(out.read_bytes(), filename=out.name)
    except Exception as exc:
        print(f"  no button ({exc.__class__.__name__}) -- take it from the file "
              f"browser at {pathlib.Path.cwd() / out.name}")


_status()
_bank()

# ===================== CELL 8 — package results for download ==================
# Run when a job finishes. Assume this filesystem is NOT durable: pull the files
# down before the session ends.
#
# Two kinds of output with very different urgency, so two kinds of archive:
#   * cloud_numbers.tar.gz -- every ladder.json and every log. Under a megabyte,
#     and it holds every number that ends up in the report. Take it first; once it
#     is on the Mac, losing the container costs GPU hours but no results.
#   * cloud_w<n>_<name>.tar.gz -- weights, one archive per run, ranked by what is
#     blocked without them. 100-150 MB each, so pulling them through a browser is
#     a decision rather than a formality. Stop once you have what you need.
#
# This walks the checkpoint tree rather than JOBS on purpose. Cell 9's low point
# writes to checkpoints/ladder_p6/beta0.0002, which is not a JOBS key, and it is the
# most valuable file here -- keying off JOBS left it behind without saying so, and
# that is precisely how a 200,000-step run's weights went missing once already.


def _package():
    import tarfile

    # The seed tree came FROM the Mac, so sending it back is a 100 MB round trip
    # for files already there. Matched on path COMPONENTS, not string prefixes:
    # "ladder_p5_cont".startswith("ladder_p5") is true and would drop a real run.
    # _probe is cell 4's 200-step throughput probe. It is not a seed, it is junk --
    # but it writes a full-tier final.pt with optimiser state, so it packaged as a
    # 161.7 MiB "cloud_w8_other.tar.gz" that took a round trip to identify.
    seeded = [("ladder_p5",), ("ladder_p6", "beta0.0005"), ("_probe",)]
    # Ranked by what cannot be done without it. #1 takes the Kodak overlap from 7/11
    # to 8/11 and does it at the end of the curve where the codec is 21% ahead; #2
    # settles single-branch vs two-branch on Kodak; #3 puts the step-budget result
    # there too. The sweeps come last because their conclusion is already in
    # ladder.json -- three rate points at one beta do not need re-benching to be
    # reported, and cell 7's board has already captured the numbers.
    rank = [("ladder_p6", "beta0.0002"), ("ladder_p3f",), ("ladder_p6_long",),
            ("ladder_p5_cont",), ("sweep_w6",), ("sweep_w24",), ("sweep_w1",)]
    # ladder.json files the MAC owns. Never ship these back: checkpoints/ladder_p6/
    # ladder.json on the Mac is the authoritative seven-point ladder, and an
    # extract-in-place of a box copy would quietly overwrite it.
    mac_owned = {"ladder_p5", "ladder_p6"}

    # Arcnamed under cloud_results/ so that extracting it can never overwrite
    # anything -- the point is to read these, not to slot them into the tree.
    numbers = ROOT / "cloud_numbers.tar.gz"
    with tarfile.open(numbers, "w:gz") as tf:
        for p in sorted(ROOT.glob("checkpoints/*/ladder.json")):
            tf.add(p, arcname=f"cloud_results/{p.parent.name}_ladder.json")
        for p in sorted((ROOT / "logs").glob("cloud_*.log")):
            tf.add(p, arcname=f"cloud_results/{p.name}")
    print(f"{numbers.name:<38} {numbers.stat().st_size / 2**20:>6.1f} MiB"
          f"   <-- take this first: every number, no weights")

    groups: dict[int, list] = {}
    for f in sorted((ROOT / "checkpoints").rglob("final.pt")):
        rel = f.relative_to(ROOT / "checkpoints")
        if any(rel.parts[:len(s)] == s for s in seeded):
            continue
        i = next((n for n, r in enumerate(rank) if rel.parts[:len(r)] == r), len(rank))
        groups.setdefault(i, []).append(f)

    print()
    written = []
    for i in sorted(groups):
        label = "_".join(rank[i]) if i < len(rank) else "other"
        out = ROOT / f"cloud_w{i + 1}_{label.replace('.', '')}.tar.gz"
        with tarfile.open(out, "w:gz") as tf:
            for f in groups[i]:
                # Real paths here, unlike the numbers archive: these have to land
                # where runbench looks for them.
                tf.add(f, arcname=str(f.relative_to(ROOT)))
            run = rank[i][0] if i < len(rank) else None
            j = ROOT / "checkpoints" / run / "ladder.json" if run else None
            if j and j.exists() and run not in mac_owned:
                tf.add(j, arcname=str(j.relative_to(ROOT)))
        print(f"{out.name:<38} {out.stat().st_size / 2**20:>6.1f} MiB"
              f"   {len(groups[i])} checkpoint(s)")
        written.append(out)

    # Staged where the file browser can actually see them: the notebook's own
    # directory, not ROOT. Cell 1 clones to /root/JPEC-AI and the sidebar has never
    # shown it, which is the same asymmetry that made cell 2b hunt six directories
    # for the upload. Hard-linked when the two share a filesystem so it costs no
    # disk, copied when they do not. Top three only -- w4..w7 are 559 MiB of
    # checkpoints whose conclusions are already in ladder.json.
    stage = pathlib.Path.cwd()
    if stage != ROOT:
        for out in written[:3]:
            tgt = stage / out.name
            if tgt.exists() and tgt.stat().st_size == out.stat().st_size:
                continue
            try:
                tgt.unlink(missing_ok=True)
                os.link(out, tgt)
            except OSError:
                import shutil
                shutil.copy2(out, tgt)
        print(f"\nstaged the first {min(3, len(written))} in {stage} -- "
              f"they show up in the file browser from there")

    # Derived from `written`, never hardcoded: this line named beta0005 for a while
    # after cell 9 moved to beta0.0002, i.e. it told you to extract a file that did
    # not exist. An instruction that can drift out of step with the thing it
    # describes should not be a literal.
    print("\nPULL EACH ARCHIVE AS ITS RUN FINISHES, not after all seven. Both losses")
    print("so far happened in the gap between the last job ending and anything being")
    print("downloaded -- ladder_p3f is done around the one-hour mark, four hours")
    print("before the campaign is. Everything that is not weights is already in")
    print("cloud_numbers.tar.gz, which cell 7 now rebuilds every time you poll.")
    print("\non the Mac, in whatever order you got them:")
    print("  tar xzf cloud_numbers.tar.gz     # lands in cloud_results/, "
          "overwrites nothing")
    if written:
        print(f"  tar xzf {written[0].name}   # weights, into checkpoints/ in place")
    # runladder rather than runbench first: every point is already trained, so
    # --skip-done (the default) only reads the checkpoints and rewrites ladder.json --
    # which is what regenerates the gate warnings and the summary table the report
    # quotes. A single point trained through loop.main never wrote one.
    print("  .venv/bin/python -m jpegai.train.runladder --model twobranch-mcm "
          "--tier full \\")
    print("      --name ladder_p6 --betas "
          "0.0002,0.0005,0.001,0.002,0.005,0.012,0.03,0.075,0.2")
    print("  .venv/bin/python -m jpegai.eval.runbench --neural checkpoints/ladder_p6 "
          "\\")
    print("      --codecs jpeg,webp,avif --anchor jpeg --dataset kodak --out p6_9pt")


_package()

# ========= CELL 9 — the low-rate point that widens the overlap (50k, ~50 min) =====
# The rate hole this cell used to fill is FILLED: checkpoints/ladder_p6/beta0.005 is
# trained, banked on the Mac, and already inside the -16.2% AVG headline (it moved the
# number 0.8 points on its own, from -15.4). Do not retrain it. This cell now goes
# after the one remaining structural weakness in that headline, which is the OVERLAP.
#
# BD-rate is only defined where the two curves overlap on the METRIC axis, and the
# Kodak run reports 7 of JPEG's 11 quality points. Four are excluded, and they fail
# for two completely different reasons:
#
#   q85 41.17 dB, q92 44.62, q96 47.92   -- above our psnr_hvs ceiling of 40.65. This
#       is the Tier-A capacity finding again, one tier up: no amount of rate reaches
#       them, so they are out of scope for any run, and 250,000 steps would not help.
#   q10 25.09 dB                          -- BELOW our floor of 25.25. Missed by
#       0.16 dB. One cheap point fixes it.
#
# So 8/11 is one 50,000-step run away and 9/11 is not available at any price. Worth
# doing because the truncation is currently CONSERVATIVE in a way that costs us the
# best part of the curve: the excluded low end is exactly where the codec is ahead of
# JPEG by 21.2%, so extending the floor pulls the integration window down into
# favourable territory rather than merely making it wider.
#
# WHY beta 0.0002 AND NOT 0.0003. The floor has to drop by at least 0.16 dB and the
# error is asymmetric: overshooting costs nothing at all (a point below the anchor's
# own floor simply does not get integrated, and it still improves the curve's shape
# where PCHIP needs it), while undershooting wastes the entire 50,000 steps and
# leaves the overlap at 7/11. beta 0.0005 -> 0.0002 is a 2.5x cut in lambda*255^2
# (32.5 -> 13.0) against a required 0.16 dB, which is generous on purpose. 0.0002 is
# also the bottom of config.rate.beta_list, so it is a documented grid member rather
# than a number invented for this run.
#
# The warm start must be ladder_p6/beta0.0005 -- the adjacent point, and the one
# --skip-done would have chained from on the Mac. Build it there:
#
#   .venv/bin/python cloud/make_seed.py --ladder checkpoints/ladder_p6 \
#       --betas 0.0005 --out cloud/seeds_p6low
#   mv cloud/seeds_p6low/seeds.tar.gz cloud/seeds_p6low/seed_p6low.tar.gz
#
# The rename is only so it cannot be confused with the ladder_p5 seed cell 2b wants;
# this cell matches on archive CONTENTS, not names, so any name in fact works.
#
# It calls loop.py directly for a single point instead of going through runladder --
# exactly how beta0.0005, beta0.001 and beta0.005 were trained -- so the checkpoint
# lands inside the existing ladder_p6 tree. ladder.json is stale afterwards by
# design; regenerate it on the Mac, as with those three.

def _low_point():
    import tarfile

    want = "checkpoints/ladder_p6/beta0.0005/final.pt"
    seed = ROOT / want
    log = ROOT / "logs" / "cloud_p6_low.log"

    if log.exists():
        print(f"log exists, not relaunching (delete logs/{log.name} to force)")
        return

    if not seed.exists():
        # Content-addressed rather than name-addressed: several stale seeds.tar*.gz
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
            if any(n.endswith("ladder_p6/beta0.0005/final.pt") for n in names):
                print(f"seed found inside {c}")
                subprocess.run(["tar", "xzf", str(c), "-C", str(ROOT)], check=True)
                break
        else:
            print(f"NO SEED for the low point -- need {want} here.\n"
                  f"  On the Mac:  .venv/bin/python cloud/make_seed.py "
                  f"--ladder checkpoints/ladder_p6 --betas 0.0005 --out cloud/seeds_p6low\n"
                  f"  then upload cloud/seeds_p6low/seeds.tar.gz anywhere in the box.\n"
                  f"  Scanned {len(cands)} archive(s); none held that member.\n"
                  f"  Refusing to start cold: an unseeded point is not comparable to "
                  f"the ladder it is joining, which is the entire purpose here.")
            return

    if not seed.exists():
        print(f"extracted, but {want} is still missing -- the archive has a "
              f"different internal layout than make_seed.py writes")
        return

    _LOW_LAUNCH(want, log)


def _LOW_LAUNCH(want, log):
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
    # checkpoints/ladder_p6/beta0.0002/ and runbench picks it up with no flags.
    argv = (f"--model twobranch-mcm --tier full --beta 0.0002 --iterations 50000 "
            f"--name ladder_p6/beta0.0002 --warm-start {want} {COMMON}")
    cmd = (f"setsid env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 "
           f"nohup {sys.executable} {runner.name} {argv} > {log} 2>&1 &")
    subprocess.Popen(cmd, shell=True, cwd=ROOT)
    print(f"launched ladder_p6/beta0.0002  (lambda*255^2 = {0.0002 * 255 ** 2:.1f}, "
          f"expect well under 0.2 bpp)  -> logs/{log.name}")
    print("cell 7 picks it up automatically -- it iterates logs/cloud_*.log, and with")
    print("no ladder.json for a single point it falls back to the log's own bpp lines.")


_low_point()

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

