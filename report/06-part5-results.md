<div class="page-break"></div>

# Part V — Training and Results

*Every number in this part was produced on the author's own laptop, from a real bitstream, and
every bitrate is the size of that bitstream divided by the pixel count. Nothing is estimated from
the loss.*

## 16. Hardware, and what a laptop can and cannot do

### 16.1 The machine

| | |
|---|---|
| machine | Apple MacBook Pro, **M2 Pro** |
| accelerator | Apple **MPS** (Metal Performance Shaders) via PyTorch |
| interpreter | `.venv/bin/python`, Python 3.12 |
| training data | 6,400 crops, 256×256, from DIV2K train |
| validation | 100 images, DIV2K valid (full resolution) |
| benchmark | 24 images, Kodak, 768×512 |

There is no CUDA GPU in this project. Everything below ran on integrated Apple silicon.

### 16.2 The measured cost model

Throughput was measured rather than assumed, because the whole schedule depends on it:

| configuration | it/s (batch 8, 256 px, MPS) | per 50k-step point | per 5-point ladder |
|---|---|---|---|
| mean-scale, tier A | **5.87** | 2.4 h | **≈ 12 h** |
| mean-scale, tier full | 3.71 (1.58× slower) | 3.7 h | ≈ 19 h |
| `twobranch-split`, tier full | ≈ 3.0 | 4.6 h | ≈ 23 h |
| `twobranch-mcm`, tier full | 2.74 (2.14× slower) | 5.1 h | ≈ 26 h |

So a single rate ladder is between half a day and a full day of continuous laptop work, and the
project's total training bill so far is on the order of **60 hours**. This is the constraint that
shapes every decision in Part V: we can afford about one ladder per day, which means **the number
of ablations is severely limited** and each one has to be chosen for information value rather than
completeness. It is also why tier A exists.

### 16.3 What this constraint costs, honestly

The paper's models are trained on far more data, for far longer, with a multi-stage schedule and
learning-rate annealing across hundreds of thousands of steps. Ours get **50,000 steps** on 6,400
crops. This is by far the largest confound in every comparison in this report, it is not
quantifiable without the compute to run the control, and §26.1 says so again in the limitations.

Anything in this part that reads as a deficiency of the *architecture* may instead be a deficiency
of the *budget*. The two measurements in chapter 20 exist precisely to separate those two cases
for one specific question, and they are the only place in the report where that separation is
actually established rather than argued.

## 17. Methodology

### 17.1 How a rate point is trained

One optimisation run per rate point, five points per ladder, β fixed within a run.

| | |
|---|---|
| objective | `L = β · D₂₅₅ + R`, §5.2 |
| distortion | weighted MSE on YCbCr, weights `{y:6, u:1, v:1}` normalised to Σ = 3 — **OURS** |
| rate | `−log₂ p(ŷ)` under the model's own likelihoods, plus the hyper stream |
| quantisation | additive uniform noise on the rate branch, **STE** on the distortion branch |
| optimiser | Adam, two parameter groups (main / auxiliary quantile loss) |
| steps | 50,000 for reported ladders; 3,000 for smoke ladders |
| batch | 8 crops of 256×256 |
| lr | 1e-4 with a linear warm-up over 1,000 steps and a cosine tail |
| β values | 0.002, 0.012, 0.03, 0.075, 0.2 |

The β set is not arbitrary: 0.002, 0.012, 0.075 are three of the four **base-model** β values read
out of the reference software (§5.2), so the ladder is anchored on the standard's own operating
points rather than on a range picked to look good.

### 17.2 How a result is measured

The chain, with no shortcuts anywhere in it:

1. Encode a Kodak image with the trained model → **a file of bytes**.
2. bpp = 8 × filesize / (width × height).
3. Decode **that file** → a reconstruction.
4. Compute all seven paper metrics plus PSNR and per-plane PSNR-Y/U/V on that reconstruction.
5. Repeat for all 24 images; average per rate point.
6. Repeat for the anchors (JPEG at 11 quality points, WebP, AVIF).
7. BD-rate per metric via **PCHIP** over the overlapping quality window, then average the seven.

Step 3 is the one that most published learned-codec results skip, and it is the one that makes the
round-trip gate of §15.2 meaningful.

### 17.3 Warm starting, and its one hazard

Later phases warm-start from earlier ones — phase 6 from phase 5's checkpoints — because training
each architecture from scratch at 50,000 steps × 5 points is a day per phase we do not have.

The loader matches parameters by name and shape, loads what matches, and **reports what it did**:

```
117 tensors loaded, 28 initialised fresh
  (e.g. branch_y.mcm.nets.0.fuse.0.weight)
```

That report line exists because of §22.2: a warm start that silently mismatched channel layout
degraded the model while printing nothing at all. The rule now is that **any partial load prints
its own manifest**, and 28 fresh tensors for a model that gained an MCM is exactly what should be
fresh.

The hazard this introduces is a confound: phase 6's ladder has effectively had *more total
training* than phase 5's, so any improvement is architecture **plus** extra steps. §27 proposes the
control (`ladder_p5_long`) that removes it, and until that runs, phase 6's gain must be read as an
upper bound.

## 18. Ladder by ladder

Five ladders have been run. Two are 50,000-step results; two are 3,000-step smoke tests that exist
to validate the pipeline, not to produce numbers; one is in flight.

### 18.1 `ladder` — #0, mean-scale hyperprior, tier A

The first real codec. Single-branch RGB, mean-scale entropy model, 96-channel latent.

| β | λ·255² | steps | est bpp | **act bpp** | **PSNR** | gap_q | oor | `ŷ` exact |
|---|---|---|---|---|---|---|---|---|
| 0.002 | 130 | 50,000 | 0.3977 | **0.3012** | **28.75** | +0.31% | 0.000% | ✓ |
| 0.012 | 780 | 50,000 | 0.8124 | **0.6462** | **30.92** | +0.17% | 0.000% | ✓ |
| 0.03 | 1951 | 50,000 | 1.1264 | **0.9144** | **32.07** | +0.04% | 0.000% | ✓ |
| 0.075 | 4877 | 50,000 | 1.3719 | **1.1533** | **32.53** | +0.05% | 0.000% | ✓ |
| 0.2 | 13005 | 50,000 | 1.5735 | **1.3525** | **32.71** | +0.04% | 0.000% | ✓ |

Three things to read out of this table.

**The gate passes everywhere.** `gap_q` ≤ 0.31%, `oor` exactly zero, `ŷ` bit-exact at every point.
The coder is sound.

**The estimated bpp is consistently *above* the actual bpp** — 0.3977 vs 0.3012, a 24% overshoot at
the low end. That is not an error. The estimate is computed on 256×256 training crops with
continuous likelihoods; the actual is a whole 768×512 image with quantised CDF tables and
skip-coded near-zero regions. Larger images amortise the header and contain more coherent
low-detail area. The *gate* compares like with like (`gap_q`) and that is the number that must be
small.

**The curve saturates.** From β 0.075 to 0.2, the rate rises 17% and the PSNR rises **0.18 dB**.
Between β 0.03 and 0.2 — a 6.7× change in β — the total gain is 0.64 dB. This is the wall, and
chapter 20 identifies it.

### 18.2 `ladder_p5` — #1, `twobranch-split`, tier full

Phase 5's architecture at the paper's own widths: two branches (160 luma / 96 chroma), residual
coding, split hyper decoders.

| β | steps | est bpp | **act bpp** | **PSNR** | gap_q | oor | worst stream |
|---|---|---|---|---|---|---|---|
| 0.002 | 50,000 | 0.5353 | **0.4417** | **29.03** | **−3.30%** | 0.000% | `y_uv` +15 B |
| 0.012 | 50,000 | 0.9303 | **0.7225** | **31.80** | +0.28% | 0.000% | **`z_uv` +104 B (+4.9%)** |
| 0.03 | 50,000 | 1.3266 | **0.9831** | **33.52** | +0.20% | 0.000% | **`z_uv` +109 B (+9.5%)** |
| 0.075 | 50,000 | 1.8275 | **1.3445** | **34.83** | +0.05% | 0.000% | `z_uv` +21 B |
| 0.2 | 50,000 | 2.3985 | **1.7752** | **35.81** | +0.03% | 0.001% | `y` +13 B |

**Two gate warnings, and they are reported rather than suppressed:**

```
WARNING: the coder/table gate failed at beta 0.002 (-3.30%)
WARNING: a stream disagrees with its own entropy table at
         beta 0.012 (z_uv +104 B, +4.9%), 0.03 (z_uv +109 B, +9.5%)
```

`ŷ` is still bit-exact at every point — the codec is *correct*, the bytes decode to exactly what
was encoded. What is wrong is narrower and it is diagnosed in §22.4: the **chroma hyper stream's**
`update()`-built tables do not match the density its `forward()` rate loss was trained against.
The gate's own message says it precisely:

> the coder is faithful to a table that is not the density the rate loss was trained against

The cost is bounded and small — `z_uv` is a few hundred bytes on a payload of tens of kilobytes —
but it is a real open defect and it is listed as one.

**The gain over ladder #0, per β:**

| β | #0 PSNR (tier A) | #1 PSNR (tier full) | Δ |
|---|---|---|---|
| 0.002 | 28.75 | 29.03 | **+0.28** |
| 0.012 | 30.92 | 31.80 | **+0.88** |
| 0.03 | 32.07 | 33.52 | **+1.45** |
| 0.075 | 32.53 | 34.83 | **+2.30** |
| 0.2 | 32.71 | 35.81 | **+3.10** |

The Δ grows monotonically with β. That shape is itself informative: at low rate the two
configurations are nearly equivalent because neither is capacity-limited, and the gap widens
exactly as tier A runs into its ceiling. This is the same conclusion chapter 20 reaches by a
different route.

### 18.3 The two smoke ladders

3,000 steps each. These are **pipeline tests**, and their PSNRs (22–26 dB) are meaningless as
results. They are reported because one of them found a bug.

**`ladder_cpu3k`** — mean-scale, CPU, 3,000 steps:

| β | est bpp | act bpp | PSNR | gap_q | exact |
|---|---|---|---|---|---|
| 0.002 | 0.4362 | 0.3870 | 22.03 | +0.29% | ✓ |
| 0.03 | 0.8468 | 0.7473 | 25.63 | +0.21% | ✓ |
| 0.2 | 1.0165 | 0.9143 | 26.40 | +0.13% | ✓ |

Clean. Its purpose was to prove the pipeline runs without MPS at all, which matters for
portability, and it does.

**`ladder_tb3k`** — `twobranch`, 3,000 steps:

| β | est bpp | act bpp | PSNR | gap_q | exact |
|---|---|---|---|---|---|
| 0.002 | 0.4621 | 0.4127 | 23.02 | **+1.29%** | ✓ |
| 0.03 | 0.8686 | 0.7383 | 25.46 | **+1.85%** | ✓ |
| 0.2 | 1.0836 | 0.9503 | 26.41 | **+2.24%** | ✓ |

**All three points fail the gate.** This is the first appearance of the `z_uv` defect, and the
pattern across all four ladders is the diagnostic that localises it:

| ladder | architecture | gate |
|---|---|---|
| `ladder` | mean-scale, single-branch | **passes** |
| `ladder_cpu3k` | mean-scale, single-branch | **passes** |
| `ladder_tb3k` | two-branch | **fails** ×3 |
| `ladder_p5` | two-branch split | **fails** ×3 |

Both single-branch ladders pass; both two-branch ladders fail. The defect is therefore **in the
two-branch chroma hyper path and nowhere else** — which is exactly what §22.4 confirms. A smoke
test that "wasted" an hour of CPU paid for itself by making that table possible.

Note also that the failure is *larger* at 3,000 steps (+2.24%) than at 50,000 (+0.20%). The
mismatch shrinks as the model converges, which is why it is nearly invisible in a finished model
and glaring in a partly-trained one — and why the mid-training gate layer of §15.3 is the only
layer that could have caught it.

### 18.4 The monochrome fast path

A measurement enabled by §8.4's finding that the luma branch is completely independent of chroma:
if you only want a greyscale reconstruction, you can decode the luma stream alone and skip the
entire chroma branch.

| β | rate saving | | |
|---|---|---|---|
| 0.002 | **−11.9%** | | |
| 0.03 | **−12.3%** | | |
| 0.2 | **−17.0%** | | |

| resolution | full decode | `--luma-only` | speedup |
|---|---|---|---|
| 768×512 | 161.1 ms | **121.0 ms** | −24.9% |
| 1024×1024 | 426.6 ms | **328.1 ms** | −23.1% |

Two correctness properties hold: the luma output is **bit-identical** to the full decoder's luma,
and the chroma planes come out flat grey to within 1.2 × 10⁻⁷. So this is a genuine structural
property of the architecture, not an approximation.

**This table was first published from a randomly initialised model** and reported −33.2%. §21.2 is
that error.

### 18.5 `ladder_p6` — in flight

`twobranch-mcm`, tier full, warm-started from `ladder_p5`. At the time of writing it is **11,000 of
50,000 steps into its first rate point**, roughly 5.5 h remaining on that point and about 30 h on
the ladder.

```
 11,000/50,000  loss 0.9545  bpp 0.4849  psnr 28.00  aux 60.55
   lr 1.00e-04   1.98 it/s   eta 5.46 h
   Y/U/V 29.12/37.29/37.29   chroma 25.0%
```

Health checks all pass: `ŷ` exact, `oor` 0.000%, and the loss is descending. At 11,000 steps it is
already at **28.00 dB** where `ladder_p5`'s finished β = 0.002 point reached 29.03 dB, which is a
reasonable trajectory for a warm start.

The `Y/U/V 29.12/37.29/37.29` breakdown is the number worth watching, and it restates the project's
central finding in a single line: **chroma is 8 dB better than luma.** Chroma consumes 25.0% of the
bits and is 8 dB ahead. That is not a balance any sensible rate allocation would choose, and it is
why the MCM — attached to the luma branch only — is the right next tool.

## 19. Headline results

### 19.1 BD-rate against JPEG on Kodak, seven-metric average

24 images, PCHIP interpolant, per-metric BD-rate then unweighted mean. **Negative is better.**

| codec | **AVG** | ms_ssim | vif | fsim | vmaf | nlpd | psnr_hvs | iw_ssim | overlap |
|---|---|---|---|---|---|---|---|---|---|
| **WebP** | **−10.6** | −13.3 | −24.0 | −3.4 | −1.7 | −20.0 | −1.8 | −10.2 | 9/11 |
| **AVIF** | **−36.1** | −42.3 | −41.2 | −37.7 | −26.5 | −40.7 | −24.6 | −39.5 | 10/11 |
| ours #0, tier A | **−0.4** | −31.5 | −3.9 | −29.7 | **+30.0** | +3.0 | **+37.8** | −8.6 | 4/11 |
| ours #1, tier full | **+1.8** | −26.2 | −4.2 | −16.6 | **+28.1** | +6.3 | **+30.4** | −5.6 | 6/11 |
| *paper, Table V, dec 0* | *−7.5* | — | — | — | — | — | — | — | — |

**The anchors validate the harness.** WebP at −10.6% and AVIF at −36.1% are where a correct
implementation should put them. That is the check that makes our own row believable — and the
check that failed loudly when the BD-rate code was wrong (§22.1).

**Read our rows plainly: we are level with JPEG.** Not ahead of it on average, and not near VVC.

**But the per-metric spread is the actual result**, and it is enormous:

| decisively **ahead** | decisively **behind** |
|---|---|
| MS-SSIM −31.5 / −26.2 | PSNR-HVS **+37.8 / +30.4** |
| FSIM −29.7 / −16.6 | VMAF **+28.1 / +30.0** |
| IW-SSIM −8.6 / −5.6 | NLPD +3.0 / +6.3 |

A ~60-point spread between MS-SSIM and PSNR-HVS on the same bitstreams. Both metrics are computed
on the same reconstruction, at 10-bit precision, on the luma plane. So this is not a measurement
artefact — it is a statement about what the model learned. **Structural similarity is good;
pixel-accurate fidelity is bad.** The model is producing plausible texture in roughly the right
place rather than the exact pixel values, which is the classic signature of an MSE-trained
autoencoder that is capacity-limited: it spends its bits on what reduces average squared error
across a patch, and squared error is minimised by getting structure right and detail approximately
right.

### 19.1.1 A warning about comparing the two AVGs

**−0.4% and +1.8% must not be differenced.** Their overlap coverages are 4/11 and 6/11. They are
BD-rates over *different quality windows* — #0's window is narrower and lower, because tier A
cannot reach the quality where JPEG's high-quality points live. A BD-rate over 4 points and one
over 10 points answer different questions, and §5.9.4 is why we print the overlap column at all.

The direct architecture comparison is §19.2's matched-rate measurement, which involves no
interpolation whatsoever.

### 19.2 The tier change, measured at a matched rate: +2.12 dB

The clean comparison, and the reason it is clean is that it compares **two real bitstreams of
almost the same size**:

| | act bpp | PSNR |
|---|---|---|
| ladder #0, tier A, β = 0.2 | 1.3525 | **32.71 dB** |
| ladder #1, tier full, β = 0.075 | 1.3445 | **34.83 dB** |
| | −0.6% smaller | **+2.12 dB** |

The tier-full bitstream is *slightly smaller* and 2.12 dB better. No curve fitting, no
extrapolation, no interpolation — two files and two PSNRs.

This is the measurement that justified spending a day of training on tier full and it is the
strongest single result in the project.

**One caveat, stated because it matters:** #1 differs from #0 in *two* ways — the tier (96→160
luma) and the architecture (mean-scale single-branch → two-branch split residual). The +2.12 dB is
the joint effect. Chapter 20's PCA bound is what attributes the bulk of it to the width: it
predicts 30.91 dB for 96 channels and 35.02 dB for 160, a 4.1 dB span, which brackets the observed
2.12 dB and makes width the dominant term.

### 19.3 Where the deficit is: entirely in luma

The PSNR-plane BD-rates, which is the diagnostic that localises everything:

| codec | psnr | psnr_y | **psnr_u** | **psnr_v** |
|---|---|---|---|---|
| WebP | −33.2 | −31.6 | −33.3 | −34.8 |
| AVIF | −47.3 | −43.7 | **−59.1** | **−56.9** |
| ours #0 | +28.1 | **+48.6** | **−43.2** | −37.8 |
| ours #1 | +14.0 | **+28.2** | **−54.6** | **−47.0** |

Read the bottom row across. **Our chroma is at AVIF's level** — −54.6 and −47.0 against AVIF's
−59.1 and −56.9. Our luma is **+28.2%**, i.e. 28% *worse* than JPEG.

That is a **75-percentage-point spread between two branches of the same model**, trained by the
same recipe, with the same optimiser, the same data, the same entropy coder and the same number of
steps. Nothing in the training procedure is plane-specific except the 6:1:1 distortion weight,
which favours luma.

So this is a statement about the branches themselves, and it has three consequences that direct
everything remaining:

1. **The architecture is not broken.** A broken codec is bad everywhere. Ours is competitive with
   AVIF on two of three planes.
2. **The luma branch is where all remaining work belongs.** It is 160 channels against chroma's 96
   and it is the one carrying the detail.
3. **The MCM is the right next tool**, because in JPEG AI the MCM attaches to the **luma branch
   only** (§8.4). The standard's own designers put their strongest entropy model exactly where our
   deficit is. `ladder_p6` is that experiment and it is running.

There is also a plausible reading of *why*, worth stating as a hypothesis rather than a result: the
6:1:1 weighting is **OURS**, not the standard's. The paper says only "prioritise luma". If 6:1:1
over-weights luma distortion relative to its rate, the optimiser will pour bits into luma while
still failing to reach fidelity there — and the observed 25% chroma bit share with 8 dB better
chroma is consistent with chroma being *over*-served. Testing this needs a weight sweep, which is
in §27.

### 19.4 The rate–distortion curves

![Kodak rate-distortion curves, all codecs](results/bench_kodak.png)

![Phase 5 rate-distortion curves](results/bench_p5.png)

Kodak operating points, from real bitstreams:

| point | #0 bpp | #0 PSNR | #1 bpp | #1 PSNR |
|---|---|---|---|---|
| 1 | 0.3614 | 28.34 | 0.4833 | 28.67 |
| 2 | 0.7601 | 30.26 | 0.8870 | 31.59 |
| 3 | 1.0710 | 31.34 | 1.2707 | 33.79 |
| 4 | 1.3140 | 31.77 | 1.7473 | 35.38 |
| 5 | 1.5168 | 31.98 | 2.2802 | 36.47 |

The shapes differ in the way that matters. Ladder #0 **flattens**: its last three points gain 0.64
dB for 42% more rate. Ladder #1 keeps climbing: 33.79 → 36.47 dB over its last three points. Tier
A is saturating and tier full is not, which is the visual form of chapter 20's ceiling.

### 19.5 Complexity, measured

Parameter counts and kMAC/pixel from `models/complexity.py`, which counts multiply-accumulates
analytically from layer shapes rather than by profiling.

| model | params | total kMAC/pxl | decoder kMAC/pxl |
|---|---|---|---|
| single-branch RGB (phase 3) | 3,751,627 | 134.4 | 111.6 |
| two-branch YCbCr (phase 4) | 4,903,491 | 160.0 | 132.4 |
| `twobranch-split` (phase 5) | **4,575,603** | — | **128.9** |
| `twobranch-fused` (phase 5 ablation) | 4,700,451 | — | 129.2 |
| `twobranch-mcm` (phase 6) | 5,627,571 | — | 129.9 |
| `twobranch-mcm2` | 5,498,355 | — | 129.8 |
| `twobranch-mcm1` | 5,239,923 | — | 129.5 |

Three results here.

**A whole second branch costs +19% decoder complexity.** 111.6 → 132.4 kMAC/pixel. Chroma is 33.0
of 160.0 total and 27.0 of 132.4 decoder — cheap, because it runs at half the spatial resolution
in each dimension, i.e. a quarter of the samples.

**The split hyper decoder is 6.8× cheaper than the fused one** at the `h_s` module: **0.49 vs 3.33
kMAC/pixel**. Confirmed by hand arithmetic: 81 + 81 + 324 = 486 MAC/pixel for split against 337.5
+ 2025 + 972 = 3,334 for fused. And the accuracy it costs is **0.055%** of rate. That is the
design decision of §6.3 vindicated numerically: two narrow specialised heads beat one wide shared
one on both axes at once.

**MCM is nearly free at decode.** +1.0 kMAC/pixel for the full 4-stage version — 129.9 vs 128.9,
under 1%. It costs 1.05 M parameters and four sequential passes, but almost no arithmetic. Which
is the whole point of §6.5: the expensive thing about an autoregressive context model was never the
arithmetic, it was the serialisation, and the checkerboard fixes the serialisation without adding
arithmetic.

## 20. The ceiling — the most important measurement in the project

### 20.1 The question, and why guessing was not acceptable

Ladder #0 stopped improving at 32.3 dB. Three explanations were available and they imply
completely different next actions:

1. **The entropy coder is lossy or miscalibrated** → fix the coder.
2. **Training has not converged** → train longer.
3. **The latent is too narrow to carry the information** → widen the model.

Guessing wrong here costs a day of laptop time in the best case and a wrong architectural
conclusion in the worst. So instead of guessing, two bounds were measured.

### 20.2 Bound one: disable the quantiser

Reconstruct the same images with the quantiser switched off entirely — pass the continuous latent
straight through. This is **infinite bitrate**: no rounding, no entropy coding, no rate cost at
all. Whatever the transforms can do, they can do here.

| condition | β = 0.03 | β = 0.2 |
|---|---|---|
| **quantiser disabled** (infinite rate) | 31.90 dB | **32.30 dB** |
| latent rounded | 31.74 | 32.26 |
| decoded from the **real bitstream** | 31.74 | **32.27** |

Two conclusions, and they eliminate two of the three explanations outright.

**The coder costs 0.03 dB.** 32.30 with no quantiser at all against 32.27 through the full
encode→bytes→decode chain. Explanation 1 is dead: there is nothing to recover in the coder.

**More training cannot help.** The ceiling holds *with the quantiser removed*, so it is a property
of the transforms, not of the rate–distortion trade-off. Explanation 2 is dead too: a model cannot
be trained past the reconstruction quality it achieves at infinite bitrate.

By elimination, explanation 3. But elimination is not proof, so:

### 20.3 Bound two: the best possible linear transform of the same width

Compute the optimal *linear* transform at each latent width — block PCA / KLT on Kodak's own
pixels, which is provably the best linear transform for that compression ratio — and reconstruct.

| latent channels | compression ratio | PCA/KLT PSNR |
|---|---|---|
| **96** (tier A) | 8.0 : 1 | **30.91 dB** |
| **160** (tier full) | 4.8 : 1 | **35.02 dB** |
| 192 | 4.0 : 1 | 37.11 dB |
| 320 | 2.4 : 1 | 46.75 dB |

Now everything closes.

**At 96 channels the bound is 30.91 dB and our learned transform reaches 32.30 dB.** The learned
transform is **1.4 dB better than the best possible linear transform of the same size.** That is
the transform doing its job — it is exploiting exactly the non-linear structure that §1.2's
"consequence 1" promised, and it is a positive result hiding inside a negative one.

**And the width is the wall.** The PCA curve rises steeply with width — 30.91 → 35.02 dB from 96 to
160 channels. So the fix is width, and nothing else.

### 20.4 The prediction, and its test

The bounds predicted: widening 96 → 160 should be worth roughly the PCA span, on the order of a
few dB.

**Measured: +2.12 dB at a matched 1.34 bpp** (§19.2).

Prediction and measurement agree in sign and magnitude. This is the only place in the report where
a quantitative prediction was made *before* the experiment and then confirmed by it, and it is why
chapter 20 is titled as it is: the value was not the 32.3 dB number, it was that two cheap
measurements converted an architectural guess into a decided question, and then a third confirmed
the decision.

### 20.5 What this means for the remaining work

| explanation | verdict | evidence |
|---|---|---|
| the coder is lossy | **eliminated** | 32.30 vs 32.27 dB — 0.03 dB total |
| training is unconverged | **eliminated** for tier A | the ceiling holds at infinite bitrate |
| the latent is too narrow | **confirmed** | PCA 30.91 @ 96 vs 35.02 @ 160; measured +2.12 dB |

Tier A is closed as a line of investigation — it cannot be pushed past ~32.3 dB by any amount of
training or coder work, and every tier A number in this report should be read as a
*development-configuration* result.

Note what this does **not** settle: tier full's own ceiling. Ladder #1 reaches 35.81 dB where PCA
at 160 channels predicts 35.02, so tier full is already *above* its linear bound and still
climbing at its top rate point (§19.4). Whether it is capacity-limited or budget-limited is
**unresolved**, and the same two measurements would settle it. That is in §27.
