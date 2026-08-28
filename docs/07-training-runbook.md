# Training runbook — what to run, in what order, and what to watch

Everything in `jpegai/` runs offline. This file is the short list of things that
need **your** machine: network access for the last few packages and datasets, and a
GPU/MPS device for the long training runs.

Read §1, run §2 (30 seconds), then start §4 and leave it. §7 is the current list of
what is outstanding and what each item costs.

---

## 1. Outstanding network commands

My sandbox has no outbound network, so these are yours. They are independent — run any
subset, in any order. **As of 2026-08-28 §1 is closed:** §1.1 is installed, §1.2 turned
out to be impossible on this hardware and is documented as such, and §1.3/§1.4 are
optional.

### 1.1 ffmpeg — done

```bash
brew install ffmpeg          # already installed: /opt/homebrew/bin/ffmpeg, libvmaf present
```

VMAF is live, so `python -m jpegai.eval.metrics` now reports **7/7 paper metrics
working**. This mattered because VMAF is one of the seven that make up the AVG column in
Tables III–VI: without it our AVG would be a mean of six things and not comparable to
the paper's mean of seven.

### 1.2 The reference `psnr_hvs` backend — a dead end on this machine, closed

```bash
.venv/bin/pip install --no-deps psnr-hvsm      # ERROR: no matching distribution
```

**Do not retry this.** The package exists on PyPI but publishes **wheels only**, for
`manylinux_2_17_x86_64` and `win_amd64`. There is no macOS build, no arm64 build, and no
source distribution to compile from — so on an Apple Silicon Mac pip correctly reports
"from versions: none". It also pins `numpy<2` while this project runs numpy 2.5.2, so even
an x86 machine would need a separate environment.

So `psnr_hvs` and `psnr_hvsm` run permanently on our own DCT-domain implementation
(35.88 dB and 50.58 dB on the smoke pair), not on the package the QAF reference calls.

**Why this is acceptable rather than merely unavoidable.** It would matter a great deal if
we were reporting absolute `psnr_hvs` scores against the paper's. We are not: every number
we publish is a **BD-rate**, and both sides of every BD-rate — our codec and the anchor —
are measured by the same function on the same images through the same harness. A
systematic offset in one metric's absolute scale largely cancels in a rate difference
computed from two curves that share it. The residual risk is that our `psnr_hvs` has a
different *shape*, not just a different offset, and that is worth stating in the report as
a known limitation of one of the seven metrics rather than papering over.

Nothing else is missing: `nlpd` runs on `pyiqa`; `ms_ssim`, `vif`, `fsim`, `iw_ssim` on
`pytorch_msssim`/`piq`; VMAF on ffmpeg/libvmaf; WebP and AVIF anchors on Pillow +
`pillow-avif-plugin`. `python -m jpegai.eval.metrics` reports **7/7 paper metrics
working** and prints the live backend for each.


### 1.3 DIV2K validation set — complete, 100/100

**Done, nothing left to do here.** The history is worth keeping because the failure was
silent: `DIV2K_valid_HR.zip` arrived truncated at 379 MiB of 428 MiB, `unzip` refused the
whole archive, the validation directory stayed empty, and training therefore fell back to
validating on Kodak — our *test* set. Nothing errored. I first recovered 88 of the 100
images straight out of the partial download, each verified against the CRC32 stored in the
archive:

```bash
.venv/bin/python -m jpegai.data.salvage_zip data/div2k/DIV2K_valid_HR.zip data/div2k
```

You then resumed the download rather than restarting it:

```bash
curl -C - -fL -o data/div2k/DIV2K_valid_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
```

which completed the archive at 448,993,893 B. `zipfile.testzip()` now reports **no corrupt
entry across all 100 members**, and the remaining 12 (`0803, 0805, 0817, 0840, 0842, 0846,
0850, 0855, 0861, 0874, 0886, 0888`) were extracted on **2026-08-28**, between ladder #0
and ladder #1. All 100 decode as RGB at ≥256 px. The `.zip` is now dead weight and can be
deleted.

> **Timing matters, and this was timed correctly.** Every checkpoint records the image list
> it validated on, and `runladder` compares those lists across rate points. The extraction
> happened *after* ladder #0 finished and *before* ladder #1 started, so #0's five points
> share one 88-image yardstick and every future ladder shares one 100-image yardstick.
> Doing it mid-ladder would have changed the yardstick between rate points, and the summary
> would have refused to treat the bpp column as comparable — correctly — and told you to
> retrain.
>
> The consequence to remember when reading results: **ladder #0's `est bpp` column is not
> directly comparable with later ladders' `est bpp`.** It never was going to be a
> cross-ladder number anyway — BD-rate is computed by `runbench` on Kodak, which is
> untouched by any of this.

### 1.4 Flickr2K — optional

```bash
bash setup.sh          # step 5 retries it; DIV2K alone is workable
```

800 DIV2K images at 256 px give roughly 100k distinct crops, which is enough for
the phases up to about 8. More data mainly helps the largest configurations.

---

## 2. Pre-flight, 30 seconds

Run this **before** starting a long job. It exercises every code path a training
run will hit — the transforms, the CDF construction, the rANS round-trip — on the
device you are about to train on:

```bash
.venv/bin/python -m jpegai.models.selftest --device mps
```

Expect `210/210 checks passed`. I cannot test MPS here (this sandbox is CPU-only), and
the entropy coder is the part most likely to behave differently on another backend:
rANS runs on CPU, so every `compress`/`decompress` crosses the device boundary. If
something is going to break on MPS, this is where it surfaces — in half a minute
rather than four hours in.

If MPS fails and CPU passes, train with `--device cpu` and tell me the error.

---

## 3. How long a run takes

**Measured on the M2 Pro, MPS, mean-scale at batch 8 / 256 px: 5.87 it/s.** That is
2.3× the 2.55 it/s I measure on CPU here, and it makes

```
hours ≈ steps_per_point × points ÷ it_per_s ÷ 3600
```

come out at **2.4 h per rate point, ~12 h for a five-point ladder** at 50k steps. The
Mac is a perfectly adequate training machine for this project; a rented GPU would only
compress the wall clock.

Two-branch models are heavier than mean-scale — 5.63 M parameters and 157 kMAC/pxl for
`twobranch-mcm` against 3.75 M and 134 for `mean-scale` — so expect their ladders to run
slower than 12 h. Rather than scale a guess: start the run and read `it/s` off the first
log line, which appears within a minute.

Three useful budgets:

| what | steps/point | why |
|---|---|---|
| smoke | 2,000 | proves the pipeline end to end; the model is still noise |
| **demo** | **50,000** | beats JPEG clearly; what to show a professor |
| full | 300,000+ | approaches published quality; cloud GPU territory |

Warm starting means only the first point pays full price — each later β begins from
the previous one's weights, so it starts from a working codec rather than noise.

**A warm-started point looks worse before it looks better,** and that is not a fault.
The LR warms up linearly over the first 1,000 steps to 1e-4, and the optimiser state is
deliberately *not* inherited across a β change (its second moments were accumulated for a
different objective). So for the first few hundred steps the loss can drift upward while
the model is pulled toward its new rate target. Judge the point on the trend after
warmup and on the `rtcheck` gate line, not on the first four log lines.

---

## 4. The training command

```bash
.venv/bin/python -u -m jpegai.train.runladder --device mps --iterations 50000 --batch 8 2>&1 | tee logs/ladder.log
```

Five rate points — β = 0.002, 0.012, 0.03, 0.075, 0.2 — into
`checkpoints/ladder/beta<value>/final.pt`. One β is one trained model is one point
on the RD curve.

**`--model` decides which phase you are measuring, and it defaults to Phase 3's
`mean-scale`.** The command above is therefore the Phase 3 baseline, not the newest
codec. Add `--model` and `--name` to train a later phase, and give each ladder its own
`--name` so they can be compared afterwards:

| phase | flag | what its ladder is for |
|---|---|---|
| 3 | *(default)* `--model mean-scale` | the single-branch baseline everything else is measured against |
| 4 | `--model twobranch` | "two-branch beats single-branch at equal rate" |
| 5 | `--model twobranch-split` | residual coding + split hyper decoders; `twobranch-fused` is its ablation |
| 6 | `--model twobranch-mcm` | the context model; `-mcm2` / `-mcm1` are the 2- and 1-stage ablation |

A Phase 6 ladder does not have to start from noise. `--warm-start-from` points at
*another ladder* and seeds each rate point from that ladder's **same β** — weights only,
by name, not the optimiser and not the step counter — so MCM initialises as a
near-identity refiner on top of a codec that already learned this exact rate/distortion
trade-off:

```bash
.venv/bin/python -u -m jpegai.train.runladder --model twobranch-mcm --name ladder_p6 \
  --warm-start-from checkpoints/ladder_p5 --device mps --iterations 50000 --batch 8
```

It prints how many tensors landed and how many were initialised fresh (117 / 28 at Tier
A). The same-β cross-ladder seed **wins over** the ordinary intra-ladder warm start; a β
with no matching point in the seed ladder says so and falls back to it. Use
`--warm-start`, never `--resume`, across a change of `--model`: `--resume` also restores
the Adam state, whose parameter groups do not match a different architecture.

For a single point rather than a ladder, `loop.py` takes the checkpoint directly:

```bash
.venv/bin/python -u -m jpegai.train.loop --model twobranch-mcm --beta 0.03 --iterations 50000 \
  --warm-start checkpoints/ladder_p5/beta0.03/final.pt --device mps --batch 8
```

It is safe to stop and restart: finished points are skipped, so a re-run resumes at
the first point that has no `final.pt`. Add `--retrain` to force all of them.

### What to watch

Two lines matter. Every 500 steps:

```
   1,000/50,000  loss   4.5347  bpp 0.5622  psnr  17.46  aux  3903.93  ...
```

`loss` should fall and `psnr` should rise. If `loss` goes to `nan` the run stops
itself and writes `nan.pt` — send me that and the last few log lines.

Every 2,000 steps, the gate:

```
        rtcheck  act 0.1269  vs est +0.21%  vs est_q +0.21%  ...  yhat exact  ok
```

`vs est_q` is the one to watch: it compares the bytes actually written against what
the entropy model predicts using the σ the coder really indexes with. It should stay
inside ±0.5% and end with `ok`. `vs est` legitimately drifts to about **+1.9%** as
the model trains — that is the cost of JPEG AI's 32-level σ grid, not a bug
(docs/06 §3.1). A **negative** `vs est_q` is the bad sign: it means symbols are
escaping the CDF tables.

At the end, a summary table plus warnings if the ladder is not monotone in β or if
points disagree about which images they were validated on.

---

## 5. The measurement

```bash
.venv/bin/python -m jpegai.eval.runbench --codecs jpeg,webp,avif --neural checkpoints/ladder
```

Writes `results/bench_kodak.{json,md,png}`: the RD curves and the BD-rate table.
Our codec goes through the *same* harness that produced the WebP −16.5% /
AVIF −42.2% anchor numbers — same seven metrics, same BD-rate integration, same
anchor, same cache — because a separate evaluation path for one's own codec is the
easiest way to report a number that is not comparable to anything.

Rate comes from the actual bitstream (`packet_bytes` plus a 4-byte width/height
header), never from the entropy model's estimate, and the decoded image is decoded
from that same bitstream.

Add `--limit 4 --metrics ms_ssim` for a two-minute sanity pass before the full run.

BD-rate needs **at least 4 rate points** and an overlapping bitrate range with the
anchor; with fewer, the table prints `nan` and says why under "Skipped metrics".

### 5.1 Ladder #0's result, and how to read the summary table

Completed 2026-08-28, `mean-scale`, 5 points × 50,000 steps, DIV2K 88-image validation:

| β | λ·255² | est bpp | act bpp | PSNR | gap_q | oor | ŷ exact |
|---|---|---|---|---|---|---|---|
| 0.002 | 130 | 0.3977 | 0.3012 | 28.75 | +0.31% | 0.000% | ✓ |
| 0.012 | 780 | 0.8124 | 0.6462 | 30.92 | +0.17% | 0.000% | ✓ |
| 0.03 | 1951 | 1.1264 | 0.9144 | 32.07 | +0.04% | 0.000% | ✓ |
| 0.075 | 4877 | 1.3719 | 1.1533 | 32.53 | +0.05% | 0.000% | ✓ |
| 0.2 | 13005 | 1.5735 | 1.3525 | 32.71 | +0.04% | 0.000% | ✓ |

**Everything that is a gate, passed.** Monotone in β; every point inside ±0.5% of its
quantised-σ estimate; nothing escaping the CDF tables; `ŷ` bit-exact through the coder at
all five points. That is the coder and the entropy tables cleared across the whole rate
range, which is what this ladder was for.

**`est bpp` and `act bpp` are not the same measurement and should not be differenced.**
`est bpp` is the 8-image DIV2K validation average; `act bpp` is `roundtrip_check` on a
single image. Different content, so the ~20% spread between the columns is content, not
error. The column that *does* measure coder fidelity is `gap_q`, and it is ≤0.31%
everywhere. Compare bpp across ladders only via `runbench` on Kodak.

**One thing to watch in the RD curve: the high-rate end is flattening.** From β=0.03 to
β=0.2 the actual rate rises 48% (0.9144 → 1.3525 bpp) for +0.64 dB. Healthy codecs give
noticeably more than that. Two candidate explanations, and they are distinguishable rather
than a matter of opinion: either 50k steps is not enough for the high-rate points — the
high-β end has the most to learn and is trained last from a warm start — or Tier A's 96
`/16` latent channels are near their capacity, in which case extra rate has nowhere to go.
If it is capacity, `twobranch` (two branches, 96 + 96) should not show the same flattening
at the same rates; if it is training length, it will. Either way it costs BD-rate at the
top of the range, so it matters for the headline number and is worth naming in the report
rather than discovering in the plot.

> **Settled in §5.2, and it was capacity.** The unquantised autoencoder tops out at the
> same 32.3 dB, so the flattening is the transform's width, not the number of steps and
> not the coder. It cost the whole BD-rate comparison, which is why §7 now runs at
> `--tier full`.

### 5.2 Ladder #0's BD-rate, and the measurement that explains it

Run on 2026-08-28, 24 Kodak images, all seven metrics, JPEG anchor:

| codec | AVG | ms_ssim | vif | fsim | vmaf | nlpd | psnr_hvs | iw_ssim |
|---|---|---|---|---|---|---|---|---|
| webp | −15.3 | −17.6 | −24.2 | −16.2 | −8.3 | −19.9 | −1.8 | −18.9 |
| avif | −41.0 | −48.8 | −41.4 | −49.2 | −33.3 | −40.9 | −24.6 | −48.5 |
| **jpegai (ladder #0)** | **+15.6** | −9.4 | −0.3 | +5.6 | +46.6 | +9.6 | +40.5 | +16.3 |

**Ladder #0 loses to JPEG.** It wins on MS-SSIM (−9.4%), ties VIF, and loses badly on
VMAF and `psnr_hvs`. The anchor numbers reproduced (−15.3 / −41.0 against the −16.5 /
−42.2 measured earlier on a different image subset), so the harness is not at fault.

The RD points say where it goes wrong. Ours against JPEG at matched rate:

| bpp | ours PSNR | JPEG PSNR | ours `psnr_hvs` | JPEG `psnr_hvs` |
|---|---|---|---|---|
| ~0.36 | 28.34 | 28.80 @ 0.42 | 26.71 | 28.34 @ 0.42 |
| ~1.05 | 31.34 | 33.08 @ 1.04 | 32.16 | 35.71 @ 1.04 |
| ~1.5 | 31.98 | 34.52 @ 1.34 | 33.39 | 38.12 @ 1.34 |

Rate rises 42% from 1.07 to 1.52 bpp and buys **+0.64 dB**. That is not a codec that is
merely undertrained, it is a codec against a wall — so the wall was measured rather than
guessed at.

**Where the wall is.** Reconstructing the same images three ways through the β=0.2
checkpoint — the autoencoder with the bottleneck *disabled*, the same thing with the
latent rounded, and the real bitstream:

| | autoencoder (no quantiser) | rounded latent | real bitstream |
|---|---|---|---|
| β=0.03 | 31.90 dB | 31.74 | 31.74 |
| β=0.2 | **32.30 dB** | 32.26 | 32.27 |

With the quantiser switched off and rate therefore infinite, the transforms reach
**32.30 dB**. Quantisation costs 0.04 dB and the entropy coder costs nothing measurable —
consistent with every `gap_q` gate passing. **So no entropy-model or coder work can move
this number, and neither can more β.** The analysis/synthesis pair is the ceiling.

**Why the pair, and which fix applies.** Tier A's `primary_latent: 96` at `/16` maps
16×16×3 = 768 input dimensions onto 96 latent channels: an **8:1** dimensionality
reduction. The optimal *linear* transform at that width — block PCA/KLT on Kodak's own
16×16×3 patches — is the reference:

| latent channels | ratio | optimal linear PSNR | |
|---|---|---|---|
| 96 | 8.0:1 | **30.91 dB** | Tier A, ladder #0 |
| 160 | 4.8:1 | **35.02 dB** | the paper's luma width (`full.yaml`) |
| 192 | 4.0:1 | 37.11 dB | |
| 320 | 2.4:1 | 46.75 dB | CompressAI `mbt2018-mean` M=320 |

Our learned transform reaches 32.30 dB where the best possible *linear* transform of the
same width reaches 30.91 — it is **1.4 dB better than the linear bound**, which is a
transform doing its job. A conv transform sees far more than one 16×16 block, so it
should beat that line, and it does.

So the diagnosis is **width, not steps, and not the coder**: 96 channels cannot exceed
about 32–33 dB however long it trains, and JPEG reaches 34.52 dB at the top of the
comparison range. Tier A was always the reduced tier for CPU development — the mistake
was running the *headline* ladders at it. `full.yaml` already holds the paper's own
160/96, whose linear bound alone (35.02 dB) clears JPEG.

**Ladder #0 is not wasted and should stay in the report.** It is the Phase 3 baseline at
Tier A, every gate on it is clean, and the three measurements above are a better piece of
evidence than a curve that happened to work: the bottleneck was located, not guessed.

---

## 6. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: torch` | using system python | use `.venv/bin/python` |
| `WARNING: DataLoader workers unavailable` | macOS shared-memory manager blocked | harmless, already falls back to 0 workers |
| `WARNING: validation set ... is empty` | §1.3, now fixed permanently — 100/100 images | re-run the salvage command |
| `WARNING: points were validated on DIFFERENT image sets` | the validation directory changed mid-ladder | see §1.3; retrain the ladder, or read only its `act bpp` column |
| `nan` loss | learning rate or a bad batch | send `checkpoints/*/nan.pt` and the log |
| `vs est_q` drifts past ±0.5% | entropy tables and coder disagree | send the log — this is a real bug, not tuning |
| BD-rate is `nan` | fewer than 4 points, or no rate overlap | train more points, or widen the β range |

---

## 7. What is outstanding, and what it costs (as of 2026-08-28)

Phases 3–6 are all built, and every criterion that can be checked without trained
weights is checked: 326 pytest tests, 210 self-test checks, 215 with a checkpoint. What
is left in Phases 4, 5 and 6 is the same kind of thing in each case — an **RD
comparison**, which is GPU time, not code.

**§5.2 changed what to spend that time on.** Every ladder from here runs at
**`--tier full`** (the paper's 160/96 latent widths), because Tier A's 96 channels cap
reconstruction at ~32.3 dB and JPEG reaches 34.5 dB inside the comparison range. Four
more Tier A ladders would all have hit the same wall, and MCM's contribution would have
been measured against it rather than against Phase 5.

Tier full costs **1.58×** a Tier A step for `mean-scale` and **2.14×** for
`twobranch-mcm` (measured, CPU, batch 2 / 256 px). Against the 5.87 it/s measured on MPS
in §3 that is:

| `--tier full --model` | it/s (est.) | per point | per 5-point ladder |
|---|---|---|---|
| `mean-scale` | ~3.7 | ~3.8 h | **~19 h** |
| `twobranch-split` | ~3.0 | ~4.6 h | **~23 h** |
| `twobranch-mcm` | ~2.7 | ~5.1 h | **~26 h** |

Read the real `it/s` off the first log line rather than trusting the estimate.

| # | ladder | `--model` | settles |
|---|---|---|---|
| 0 | `ladder` | `mean-scale` | Tier A Phase 3 — **done 2026-08-28, 5/5 clean, +15.6% vs JPEG; see §5.2** |
| 1 | `ladder_p5` | `twobranch-split` | Phase 5's RD, **and it is the seed for #3** |
| 2 | `ladder_p4` | `twobranch` | "two-branch beats single-branch at equal rate" (vs #6) |
| 3 | `ladder_p6` | `twobranch-mcm` | "MCM gives 4–9% BD-rate over Phase 5"; `--warm-start-from checkpoints/ladder_p5` |
| 4 | `ladder_p5f` | `twobranch-fused` | the single-hyper-decoder ablation |
| 5 | `ladder_p6a` | `twobranch-mcm2`, `-mcm1` | the 1 / 2 / 4-stage ablation |
| 6 | `ladder_p3f` | `mean-scale` | the Phase 3 baseline **at tier full**, so #2 has a same-tier reference |

**Order matters for exactly one reason:** #1 before #3, because #3 warm-starts from it.
If there is time for only two, run **#1 and #3** — that pair gives the Phase 6 headline,
which is the newest claim in the project, and it is a *relative* BD-rate that
`--anchor ours-ladder_p5` computes directly, so it does not depend on either curve
beating JPEG. #6 is what gives the absolute "we beat JPEG" number. #4 and #5 are
ablations: they strengthen the report but no criterion outside their own phase needs them.

Then the measurement, once per ladder, plus one combined run for the report:

```bash
.venv/bin/python -m jpegai.eval.runbench --codecs jpeg,webp,avif --neural checkpoints/ladder_p6
```

Two things are also outstanding that need no GPU at all and can run on the Mac whenever
it is idle — `runbench` on the two 3k-step ladders already on disk (`ladder_cpu3k`,
`ladder_tb3k`), which is a rehearsal of the comparison rather than a result, and the
`runladder --bench` path, which has never been exercised end to end.

### 7.1 The exact sequence, copy-paste

Step 1 is done. Steps 2 and 6 need no GPU and take minutes. **Run step 3 before step 4** —
nothing else is ordered, so anything can be dropped from the bottom.

**1. ~~The Tier A baseline BD-rate.~~ Done — §5.2.** `results/bench_kodak.{json,md,png}`.

**2. Pre-flight before every long run (30 s).**

```bash
.venv/bin/python -m jpegai.models.selftest --device mps
```

**3. Ladder #1 — Phase 5 at tier full (~23 h). The seed for #3, so it goes first.**

```bash
.venv/bin/python -u -m jpegai.train.runladder --tier full --model twobranch-split --name ladder_p5 --device mps --iterations 50000 --batch 8 2>&1 | tee logs/ladder_p5.log
```

**4. Ladder #3 — Phase 6, the headline (~26 h), warm-started from #1.**

```bash
.venv/bin/python -u -m jpegai.train.runladder --tier full --model twobranch-mcm --name ladder_p6 --warm-start-from checkpoints/ladder_p5 --device mps --iterations 50000 --batch 8 2>&1 | tee logs/ladder_p6.log
```

Expect `warm start from ladder_p5/beta<β>` on every point, and a `tensors loaded /
initialised fresh` line. A point that says *falling back to the intra-ladder warm start*
found no `final.pt` at that β in ladder #1 — it still trains, just from a worse start.

**5. Ladder #6 then #2 — the tier-full Phase 3 baseline (~19 h), then Phase 4 (~23 h).**

```bash
.venv/bin/python -u -m jpegai.train.runladder --tier full --model mean-scale --name ladder_p3f --device mps --iterations 50000 --batch 8 2>&1 | tee logs/ladder_p3f.log
```

```bash
.venv/bin/python -u -m jpegai.train.runladder --tier full --model twobranch --name ladder_p4 --device mps --iterations 50000 --batch 8 2>&1 | tee logs/ladder_p4.log
```

**6. Measure (~15 min each, CPU, safe to run while a ladder trains).** One run per
question, each with its own `--out` so nothing overwrites anything:

```bash
.venv/bin/python -m jpegai.eval.runbench --codecs jpeg,webp,avif --neural checkpoints/ladder_p6 --out bench_p6_vs_jpeg
```

The Phase 6 claim is *relative*, and `--neural` takes several ladders so BD-rate can be
computed phase-against-phase instead of only against JPEG:

```bash
.venv/bin/python -m jpegai.eval.runbench --codecs jpeg --neural checkpoints/ladder_p5,checkpoints/ladder_p6 --anchor ours-ladder_p5 --out bench_mcm_gain
```

That prints `ours-ladder_p6` against `ours-ladder_p5` directly — the 4–9% MCM number.
The same shape gives Phase 4's claim (`--neural checkpoints/ladder_p3f,checkpoints/ladder_p4
--anchor ours-ladder_p3f`) and the report's combined figure (all ladders in one
`--neural`, `--anchor jpeg`).

**7. The ablations, if there is time.** `ladder_p5f` (`twobranch-fused`), then
`ladder_p6a` twice (`twobranch-mcm2` and `twobranch-mcm1`, each with its own `--name`).

### 7.2 What to check on the first tier-full point, before letting 100 h run

The whole point of moving tiers is the ceiling, so verify it moved. When
`ladder_p5/beta0.2` finishes, its summary `psnr` should be **≥35 dB**. At Tier A the same
point read 32.71.

* **≥35 dB** — width was the binding constraint, as §5.2 concluded. Carry on.
* **32–34 dB** — width helped but 50k steps is now the limit; the model is 2.6× larger and
  needs longer. Raise `--iterations` before spending time on the remaining ladders.
* **≈32.7 dB, unchanged** — width was *not* the constraint and §5.2's conclusion is wrong.
  Stop and say so; nothing below matters until that is understood.

Stopping and restarting any ladder is safe: finished points are skipped, so a re-run picks
up at the first β with no `final.pt`. Add `--retrain` to force all of them.
