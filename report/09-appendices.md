<div class="page-break"></div>

# Appendix A — Glossary

*For a reader who started at chapter 1. Terms are defined in the order a newcomer meets them, not
alphabetically, and cross-referenced to where they are used.*

**Codec** — coder–decoder. Software that shrinks an image (encode) and expands it back (decode). §1.1

**Lossless / lossy** — lossless compression can reconstruct the original exactly; lossy cannot, and
discards information the viewer is unlikely to miss. All of JPEG AI's operating points are lossy. §1.1

**Transform** — a change of coordinates that concentrates an image's energy into a few large numbers
and leaves the rest near zero. Near-zero numbers are cheap to store. JPEG uses the DCT; JPEG AI uses
a learned convolutional network. §1.1

**DCT (discrete cosine transform)** — the fixed 8×8 transform at the heart of classical JPEG.
Hand-designed in the 1980s. §1.1

**Analysis / synthesis transform** — the encoder-side network that maps image → latent, and the
decoder-side network that maps latent → image. JPEG AI's replacements for the forward and inverse
DCT. §1.2

**Latent** — the compact numerical representation of the image, produced by the analysis transform.
In JPEG AI: two tensors, 160 channels for luma and 96 for chroma, each at 1/16 of the image's spatial
resolution. §1.3

**Channel** — one of the parallel "feature planes" of a latent tensor. 160 channels means 160 numbers
per latent spatial position.

**Quantisation** — rounding continuous values to integers so they can be stored. Necessary for
compression, and non-differentiable, which is difficulty (a). §2.2a, §5.5

**STE (straight-through estimator)** — the training trick that makes rounding differentiable:
`y_hat = y + (round(y) - y).detach()`. Forward pass rounds; backward pass sees the identity. §5.5

**Entropy coder** — the component that turns integers into a compact bit string, using a probability
model. Better probability model → fewer bits. §1.2

**Range coder / rANS / tANS** — three closely related families of entropy coder. JPEG AI specifies
**me-tANS** (multi-symbol table-based asymmetric numeral systems); we use a mathematically equivalent
**rANS** coder. §9.4

**Probability model / prior** — the coder's belief about how likely each value is. §5.6

**Hyperprior** — JPEG AI transmits, alongside the image, a small compressed *description of the
image's own statistics*, and codes the main payload against it. The single largest source of gain in
learned compression. §1.2, §5.6

**Hyper latent (`ẑ`)** — the hyperprior's own compressed representation, at 1/64 spatial resolution.
Channel count equals its branch's latent width (160 or 96) — the hyper autoencoder is
channel-preserving. §9.1

**Factorised prior** — the simplest prior: each latent channel gets its own fixed learned
distribution, with no conditioning at all. Used for the hyper latent, which has nothing above it.
§5.6

**Mean-scale prior** — a Gaussian prior whose *mean* and *scale* are both predicted from the
hyperprior. Minnen et al. 2018. §5.6

**Residual coding** — instead of coding the latent `y`, code `r̂ = round(y − p̈)`, the difference from
a predicted value, then reconstruct `ŷ = r̂ + p̈`. Eqs (1)–(2). §6.2

**`p̈`** — the predicted latent, produced by the hyper decoder's prediction head. §6.2, §6.3

**`Iσ`** — the *integer* scale index. The hyper decoder's scale head emits an integer rather than a
float, so that the entropy coder's table selection is bit-exact across devices. §6.4

**σ-class / CDF row** — the entropy coder's tables are indexed by a quantised scale class, one of 32,
log-spaced over [0.11, 54.82]. Row = `ceil(Iσ/2⁷)` — round *up*, not a bit shift. §6.4, §9.2

**Autoregressive context model** — predicts each latent value from values already decoded. Strongest
prior available, and unshippable: it forces the decoder to process one value at a time. Difficulty
(c). §2.2c, §5.6

**MCM (Multi-Context Model)** — JPEG AI's answer to (c). A 4-stage checkerboard: the latent's spatial
positions are split into four 2×2 cosets, decoded in the order (0,0) → (1,1) → (0,1) → (1,0), each
stage conditioned on all previous. **Exactly four parallel passes regardless of image size**, and it
attaches to the **luma branch only**. §6.5, §9.3

**Coset** — one of the four spatial subsets of a 2×2 tiling, used by the MCM. §6.5

**Bit-exactness** — two different devices producing byte-identical results. Mandated by JPEG AI only
*through the entropy path*, because a one-bit difference there desynchronises the arithmetic decoder
and destroys everything after it. Difficulty (d). §2.2d, §6.6

**YCbCr** — a colour representation splitting brightness (Y, luma) from colour (Cb, Cr, chroma). The
eye is far more sensitive to Y, so codecs spend more bits on it. JPEG AI uses BT.709. §6.1

**Luma / chroma** — brightness and colour. In this report: our chroma is at AVIF's level and our luma
is 28% behind JPEG, a 75-point spread. §19.3

**Two-branch** — JPEG AI's distinctive structural choice: separate analysis/synthesis networks for
luma and chroma, with luma→chroma conditioning at exactly two points. §6.1, §8.4

**SOP / BOP / HOP** — the three standardised synthesis transforms, at 14 / 28 / 215 kMAC/pixel, all
reading **the same codestream**. Difficulty (f). §2.2f, §6.10

**kMAC/pixel** — thousands of multiply-accumulate operations per output pixel. The standard unit of
decoder complexity. §19.5

**Rate ladder / operating point** — a set of models or configurations spanning a range of
quality/bitrate. The standard reaches 18 points from four trained parameter sets via gain vectors and
a β displacement; we train five independent models. §6.9, §26.7

**β (beta)** — the rate–distortion trade-off weight in `L = β·D + R`. Large β → prioritise quality;
small β → prioritise file size. §5.2

**bpp (bits per pixel)** — file size × 8 ÷ pixel count. Every bpp in this report is measured from a
real file. §17.2

**PSNR** — peak signal-to-noise ratio, in dB. A pixel-error metric. Higher is better.

**MS-SSIM, VIF, FSIM, VMAF, NLPD, PSNR-HVS, IW-SSIM** — the paper's seven quality metrics. Six are
computed on luma only; only FSIM sees colour; all at 10-bit internal precision; NLPD is the only
lower-is-better one. §11.2

**BD-rate (Bjøntegaard delta rate)** — the headline unit. *For the same visual quality, what
percentage more or fewer bits does codec A need than codec B?* **Negative is better.** §5.9

**PCHIP** — piecewise cubic Hermite interpolating polynomial. Monotone by construction, and therefore
the correct interpolant for BD-rate on metrics that saturate. The global cubic it replaced was wrong
by 17 points. §5.9, §22.1

**Overlap coverage** — how many of the anchor's quality points lie inside the shared quality window.
Two BD-rates with different overlaps are **not comparable to each other**. §19.1.1

**Round-trip gate** — our mid-training assertion that the decoded latent is bit-identical to the
encoded one, that real bytes match the model's own estimate to ±0.5%, and that the out-of-range rate
is zero. Caught two of six bugs. §15.2

**`gap_q`** — the round-trip gate's residual disagreement between actual bytes and the quantised-σ
estimate. §15.2

**`oor`** — out-of-range rate: the fraction of latent values falling outside the coder's
representable range. Must be zero. §15.2

**CONFIRMED / PAPER / OURS** — our three provenance tags: read from the WG1 reference software; stated
in the paper; our own choice. Front matter, Appendix E.

**WG1** — ISO/IEC JTC 1/SC 29/WG 1, the committee that produced JPEG in 1992 and JPEG AI in 2025.

**Reference software** — WG1's own implementation of the standard, publicly reachable on GitLab. It
contains every normative constant as a source literal, which is what rescued this project. Chapter 9

**Tier A / tier full** — our two configurations. `full.yaml` is the paper's own widths (160/96);
`tierA.yaml` is a narrowed development configuration (96/48) that lets a laptop validate the pipeline
in hours. The tier is worth **2.12 dB**. Front matter, §13.4, §19.2

<div class="page-break"></div>

# Appendix B — Command reference

Every command that produces something in this report. All run from the project root with
`.venv/bin/python`, or `python` after `source .venv/bin/activate`.

## B.1 Setup

```bash
bash setup.sh
```

Creates `.venv`, installs `requirements.txt`, downloads Kodak (24 images, 15 MB) and DIV2K (900
images, 4.1 GB), and verifies archive sizes. Idempotent — safe to re-run. Requires network.

```bash
python -m jpegai.data.prepare_crops --out data/crops/div2k_train --per-image 8 --size 256
```

Extracts 6,400 variance-filtered 256×256 crops from DIV2K train (679 MB).

## B.2 Verification

```bash
python -m jpegai.selftest
```

210 checks, ~30 s, no checkpoint needed. Run this after any environment change.

```bash
python -m jpegai.selftest --checkpoint checkpoints/ladder_p5/beta0.075/final.pt
```

215 checks — adds the checkpoint-specific gates.

```bash
python -m pytest -q
```

331 tests across 12 files.

## B.3 Training

```bash
python -m jpegai.train.runtrain --model twobranch-split --tier full --beta 0.075 --steps 50000
```

One rate point.

```bash
python -m jpegai.train.runladder --model twobranch-split --tier full --name ladder_p5
```

A full five-point ladder. `--warm-start-from checkpoints/<name>` to warm start; `--steps 3000` for a
smoke ladder.

```bash
python -m jpegai.train.runladder --model twobranch-mcm --tier full --name ladder_p6 \
    --warm-start-from checkpoints/ladder_p5
```

The phase-6 ladder, as actually run.

## B.4 Evaluation

```bash
python -m jpegai.eval.runbench --neural checkpoints/ladder_p5 --codecs jpeg,webp,avif
```

Encodes all 24 Kodak images to real files at every rate point, decodes them, computes all seven
metrics plus PSNR/Y/U/V, computes PCHIP BD-rate against each anchor, and writes
`results/bench_*.md` and `results/bench_*.png`.

```bash
python -m jpegai.eval.runbench --neural checkpoints/ladder_p6 --anchor ours-ladder_p5
```

BD-rate against our own previous ladder rather than against JPEG — the form phase 6's 4–9% claim
needs.

```bash
python -m jpegai.eval.complexity --model twobranch-split --tier full
```

Parameter count and kMAC/pixel, analytically from layer shapes.

```bash
python -m jpegai.eval.ceiling --checkpoint checkpoints/ladder/beta0.2/final.pt --no-quant
```

Chapter 20's first bound: reconstruct with the quantiser disabled.

```bash
python -m jpegai.eval.pca_bound --channels 96,160,192,320
```

Chapter 20's second bound: optimal linear (KLT) transform at each latent width.

```bash
python -m jpegai.eval.runbench --neural checkpoints/ladder_p5 --luma-only
```

The monochrome fast path of §18.4.

## B.5 Environment notes

```bash
export MPLCONFIGDIR="$TMPDIR/mpl"
```

Required if matplotlib reports that `~/.matplotlib` is not writable.

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib /opt/anaconda3/bin/weasyprint in.html out.pdf
```

Builds this PDF. The `DYLD_FALLBACK_LIBRARY_PATH` is mandatory — see §23.4.

<div class="page-break"></div>

# Appendix C — Code map

33 implementation modules, 9,745 lines; 12 test files, 4,282 lines.

| module | lines* | responsibility |
|---|---|---|
`jpegai/models/analysis.py` | | analysis transform, single-branch RGB
`jpegai/models/synthesis.py` | | synthesis transform
`jpegai/models/twobranch.py` | | the YCbCr two-branch model; eqs (1)–(3)
`jpegai/models/hyper.py` | | hyper encoder/decoder; the split prediction and scale heads
`jpegai/models/mcm.py` | | the 4-stage Multi-Context Model; `down_shuffle`/`up_shuffle`/`split_pred`
`jpegai/models/entropy.py` | | factorised, scale-hyperprior and mean-scale entropy models
`jpegai/models/sigma.py` | | the integer σ path, eqs (5)–(10); the 11-index disagreement test
`jpegai/models/quant.py` | | STE, additive noise, discretised likelihood
`jpegai/models/colour.py` | | BT.709 forward and inverse; eq. (4)
`jpegai/models/complexity.py` | | parameter counts and analytic kMAC/pixel
`jpegai/coder/rans.py` | | the rANS entropy coder — real bytes
`jpegai/coder/tans.py` | | me-tANS constants and transition-table construction
`jpegai/coder/bitstream.py` | | container, markers, headers
`jpegai/coder/skip.py` | | skip mode
`jpegai/train/losses.py` | | `L = β·D + R`; the β-direction derivation; distortion weighting
`jpegai/train/loop.py` | | the training loop, gate hooks, checkpointing
`jpegai/train/ladder.py` | | multi-point rate ladders
`jpegai/train/warmstart.py` | | partial state-dict loading with a printed manifest
`jpegai/metrics/seven.py` | | the paper's seven metrics with per-metric plane and range
`jpegai/metrics/psnr.py` | | PSNR and per-plane PSNR-Y/U/V
`jpegai/metrics/bdrate.py` | | **PCHIP** BD-rate and overlap accounting
`jpegai/metrics/bench.py` | | the benchmark driver
`jpegai/metrics/report.py` | | markdown and PNG output
`jpegai/data/prepare_crops.py` | | DIV2K → 6,400 variance-filtered crops
`jpegai/data/salvage_zip.py` | | CRC-verified recovery from a truncated archive (§23.1)
`jpegai/data/loaders.py` | | datasets, augmentation, count assertions
`jpegai/anchors/classical.py` | | JPEG, WebP, AVIF at matched quality points
`jpegai/selftest.py` | | the 210/215-check end-to-end self test
`jpegai/cli.py` | | `runtrain` / `runladder` / `runbench` / `selftest`

\* per-module line counts are omitted rather than approximated; the totals (9,745 + 4,282 = 14,027)
are measured.

Configuration: `config/full.yaml`, `config/tierA.yaml`, `config/metrics.yaml`,
`config/constants.yaml` — every constant tagged CONFIRMED / PAPER / OURS in a comment, with a loader
test asserting the CONFIRMED values match `docs/06-normative-constants.md`.

Documentation: 9 markdown documents, 5,051 lines, ~43,000 words. `docs/02-jpeg-ai-explained.md` (892
lines) is the section-by-section study of the paper; `docs/04-reference-data.md` (509 lines) is the
lookup document — all equations, all marker codes, all six tables verbatim;
`docs/06-normative-constants.md` (567 lines) is every constant with its source file and line.

<div class="page-break"></div>

# Appendix D — Full result tables

All BD-rates against JPEG, 24 Kodak images, PCHIP interpolant, per-metric first then averaged.
**Negative is better.**

## D.1 Seven-metric BD-rate

| codec | **AVG** | ms_ssim | vif | fsim | vmaf | nlpd | psnr_hvs | iw_ssim | overlap |
|---|---|---|---|---|---|---|---|---|---|
| WebP | **−10.6** | −13.3 | −24.0 | −3.4 | −1.7 | −20.0 | −1.8 | −10.2 | 9/11 |
| AVIF | **−36.1** | −42.3 | −41.2 | −37.7 | −26.5 | −40.7 | −24.6 | −39.5 | 10/11 |
| ours #0, tier A | **−0.4** | −31.5 | −3.9 | −29.7 | +30.0 | +3.0 | +37.8 | −8.6 | 4/11 |
| ours #1, tier full | **+1.8** | −26.2 | −4.2 | −16.6 | +28.1 | +6.3 | +30.4 | −5.6 | 6/11 |

## D.2 PSNR-plane BD-rate

| codec | psnr | psnr_y | psnr_u | psnr_v |
|---|---|---|---|---|
| WebP | −33.2 | −31.6 | −33.3 | −34.8 |
| AVIF | −47.3 | −43.7 | −59.1 | −56.9 |
| ours #0 | +28.1 | +48.6 | −43.2 | −37.8 |
| ours #1 | +14.0 | +28.2 | −54.6 | −47.0 |

## D.3 Ladder #0 — `ladder`, mean-scale, tier A, 50,000 steps/point

| β | λ·255² | est bpp | act bpp | PSNR | gap_q | oor | exact |
|---|---|---|---|---|---|---|---|
| 0.002 | 130 | 0.3977 | 0.3012 | 28.75 | +0.31% | 0.000% | ✓ |
| 0.012 | 780 | 0.8124 | 0.6462 | 30.92 | +0.17% | 0.000% | ✓ |
| 0.03 | 1951 | 1.1264 | 0.9144 | 32.07 | +0.04% | 0.000% | ✓ |
| 0.075 | 4877 | 1.3719 | 1.1533 | 32.53 | +0.05% | 0.000% | ✓ |
| 0.2 | 13005 | 1.5735 | 1.3525 | 32.71 | +0.04% | 0.000% | ✓ |

Kodak operating points: 0.3614 / 0.7601 / 1.0710 / 1.3140 / 1.5168 bpp at PSNR 28.34 / 30.26 / 31.34
/ 31.77 / 31.98.

## D.4 Ladder #1 — `ladder_p5`, `twobranch-split`, tier full, 50,000 steps/point

| β | est bpp | act bpp | PSNR | gap_q | oor | exact | worst stream |
|---|---|---|---|---|---|---|---|
| 0.002 | 0.5353 | 0.4417 | 29.03 | **−3.30%** | 0.000% | ✓ | `y_uv` +15 B |
| 0.012 | 0.9303 | 0.7225 | 31.80 | +0.28% | 0.000% | ✓ | **`z_uv` +104 B (+4.9%)** |
| 0.03 | 1.3266 | 0.9831 | 33.52 | +0.20% | 0.000% | ✓ | **`z_uv` +109 B (+9.5%)** |
| 0.075 | 1.8275 | 1.3445 | 34.83 | +0.05% | 0.000% | ✓ | `z_uv` +21 B |
| 0.2 | 2.3985 | 1.7752 | 35.81 | +0.03% | 0.001% | ✓ | `y` +13 B |

Kodak operating points: 0.4833 / 0.8870 / 1.2707 / 1.7473 / 2.2802 bpp at PSNR 28.67 / 31.59 / 33.79
/ 35.38 / 36.47.

Gate warnings as printed: failure at β 0.002 (−3.30%); stream/table disagreement at β 0.012 (`z_uv`
+104 B, +4.9%) and 0.03 (`z_uv` +109 B, +9.5%). §22.4.

## D.5 Smoke ladders, 3,000 steps/point — pipeline tests, not results

`ladder_cpu3k` — mean-scale, CPU:

| β | est bpp | act bpp | PSNR | gap_q | exact |
|---|---|---|---|---|---|
| 0.002 | 0.4362 | 0.3870 | 22.03 | +0.29% | ✓ |
| 0.03 | 0.8468 | 0.7473 | 25.63 | +0.21% | ✓ |
| 0.2 | 1.0165 | 0.9143 | 26.40 | +0.13% | ✓ |

`ladder_tb3k` — `twobranch`:

| β | est bpp | act bpp | PSNR | gap_q | exact |
|---|---|---|---|---|---|
| 0.002 | 0.4621 | 0.4127 | 23.02 | **+1.29%** | ✓ |
| 0.03 | 0.8686 | 0.7383 | 25.46 | **+1.85%** | ✓ |
| 0.2 | 1.0836 | 0.9503 | 26.41 | **+2.24%** | ✓ |

## D.6 The tier comparison at matched rate

| | act bpp | PSNR |
|---|---|---|
| #0, tier A, β = 0.2 | 1.3525 | 32.71 dB |
| #1, tier full, β = 0.075 | 1.3445 | **34.83 dB** |
| | −0.6% | **+2.12 dB** |

Per-β PSNR delta: +0.28 / +0.88 / +1.45 / +2.30 / +3.10 dB.

## D.7 The ceiling measurements

| condition | β = 0.03 | β = 0.2 |
|---|---|---|
| quantiser disabled (infinite rate) | 31.90 | **32.30** |
| latent rounded | 31.74 | 32.26 |
| decoded from the real bitstream | 31.74 | **32.27** |

Optimal linear (PCA/KLT) transform bound:

| channels | ratio | PSNR |
|---|---|---|
| 96 | 8.0:1 | **30.91** |
| 160 | 4.8:1 | **35.02** |
| 192 | 4.0:1 | 37.11 |
| 320 | 2.4:1 | 46.75 |

## D.8 Complexity

| model | params | total kMAC/pxl | decoder kMAC/pxl |
|---|---|---|---|
| single-branch RGB | 3,751,627 | 134.4 | 111.6 |
| two-branch YCbCr | 4,903,491 | 160.0 | 132.4 |
| `twobranch-split` | 4,575,603 | — | 128.9 |
| `twobranch-fused` | 4,700,451 | — | 129.2 |
| `twobranch-mcm` | 5,627,571 | — | 129.9 |
| `twobranch-mcm2` | 5,498,355 | — | 129.8 |
| `twobranch-mcm1` | 5,239,923 | — | 129.5 |

Split vs fused `h_s`: **0.49 vs 3.33 kMAC/pixel** = 6.8× cheaper, at an accuracy cost of 0.055%.
Chroma is 33.0 of 160.0 total and 27.0 of 132.4 decoder kMAC/pixel.

## D.9 The monochrome fast path

| β | rate saving |
|---|---|
| 0.002 | −11.9% |
| 0.03 | −12.3% |
| 0.2 | −17.0% |

| resolution | full | `--luma-only` | speedup |
|---|---|---|---|
| 768×512 | 161.1 ms | 121.0 ms | −24.9% |
| 1024×1024 | 426.6 ms | 328.1 ms | −23.1% |

Luma bit-identical; chroma flat to 1.2 × 10⁻⁷.

## D.10 Training throughput

| configuration | it/s | per point | per 5-point ladder |
|---|---|---|---|
| mean-scale, tier A | 5.87 | 2.4 h | ≈ 12 h |
| mean-scale, tier full | 3.71 | 3.7 h | ≈ 19 h |
| `twobranch-split`, tier full | ≈ 3.0 | 4.6 h | ≈ 23 h |
| `twobranch-mcm`, tier full | 2.74 | 5.1 h | ≈ 26 h |

MPS, batch 8, 256×256 crops, Apple M2 Pro.

<div class="page-break"></div>

# Appendix E — Normative constants

Every constant, with its tag. **CONFIRMED** = read from the WG1 reference software (normative, not
ours to tune). **PAPER** = stated in the paper. **OURS** = our own choice, free to tune.

## E.1 Channel widths

| constant | value | tag | source |
|---|---|---|---|
| `N_luma` | 160 | CONFIRMED | `CCS_SGMM/ccs_sgmm_tool.py:67` |
| `N_chroma` | 96 | CONFIRMED | `CCS_SGMM/ccs_sgmm_tool.py:68` |
| luma hyper latent | 160 | CONFIRMED (derived) | channel-preserving; `common_modules.py:116-128` |
| chroma hyper latent | 96 | CONFIRMED (derived) | same |
| `p̈_Y` pre-shuffle | 640 = 4×160 | CONFIRMED | hyper decoder final `conv3x3(chs, 4·chs)` |
| `p̈_UV` pre-shuffle | 384 = 4×96 | CONFIRMED | same, chroma |
| secondary synthesis input | 256 = 96+160 | CONFIRMED | eq. (3); construction site |
| secondary downsample factor | 2 | CONFIRMED | `ccs_sgmm_tool.py:79` |
| primary latent divisibility | % 32 == 0 | CONFIRMED | `contexts/MCM_phases.py` `chs2group()` asserts it |
| latent spatial stride | /16 | PAPER | |
| hyper spatial stride | /64 | PAPER | |

## E.2 The σ / scale-index path

| constant | value | tag | source |
|---|---|---|---|
| `sigma_quant_level` | 32 | CONFIRMED | `CCS_SGMM/params.py` |
| `sigma_quant_min` | 0.11 | CONFIRMED | same |
| `sigma_quant_max` | 54.82 | CONFIRMED | same |
| `sigma_bound_offset` | 0.5 | CONFIRMED, **meaning unexplained** | same |
| `sigma_precision` | 7 | CONFIRMED, triple-confirmed | `quantization/params.py`; `5+5+7=17` |
| `gain_vector_precision` | 5 | CONFIRMED | `quantization/params.py` |
| `beta_displacement_precision` | 5 | CONFIRMED | same |
| `scaler_precision` | 10 | CONFIRMED (derived) | = 5 + 5 |
| `scaled_sigma_precision` | 17 | CONFIRMED ×2 | = 10 + 7, **and** hardcoded at `lsbs_scale_mode.py:54` |
| `log_k` | 0.200365 | derived | `(ln 54.82 − ln 0.11)/31` |
| `max_index` | 3967 | derived | `(32−1)·2⁷ − 1` |
| largest denotable σ | 54.734 | derived | §6.4.1 |
| CDF row rule | `ceil(Iσ/2⁷)` | CONFIRMED | **not** `Iσ >> 7`; differs on 3,937 of 3,968 |
| `isigma_pad_value` | 1411 | **PAPER, unconfirmed** | eq. (7); not found in the reference software |
| 32-level KL cost | 0.01464 bits/symbol | measured | end-to-end +1.86% to +1.92% |
| integer/float disagreement | exactly 11 indices | measured | 256, 1152, 1280, 1536, 1664, 2176, 2304, 2560, 3200, 3328, 3456 — all multiples of 128, all one float32 ULP |

## E.3 The hyper latent

| constant | value | tag | source |
|---|---|---|---|
| `z_offset` | 31 | CONFIRMED | |
| `z_range` | 63 | CONFIRMED | |
| `max_symbol` | 62 | CONFIRMED | = `z_range − 1` |
| `abs_in_hyperprior` | 1 | CONFIRMED | |
| BDL clip | [−1069, 702] | CONFIRMED | |
| `mcm_overlap_in_latent_samples` | 8 | CONFIRMED | |
| `hyper_decoder_overlap_in_latent_samples` | 2 | CONFIRMED | |

## E.4 me-tANS

| constant | value | tag | source |
|---|---|---|---|
| `mass_bits` | 8 | CONFIRMED | → 256 states/class. **Not** a `tableLog` |
| escape threshold | 2⁻¹¹ | CONFIRMED | `get_outbound_values(probs, threshold=1/2**11)` |
| symbol ordering | zig-zag | CONFIRMED | `get_sequence` / `get_inverse_sequence` |
| substreams | 2, interleaved | CONFIRMED | via `cdf_first` / `cdf_second` |
| residual CDF shape | [32, 256] | CONFIRMED | 32 KiB of decode transitions |
| hyper CDF shape | one row per channel | CONFIRMED | |
| total table size | ≈ 100 KB | CONFIRMED | |
| escape coding | 1 flag bit → 2 or 15 bits → sign | CONFIRMED | |
| decode order | FILO | CONFIRMED | |
| state split | `cdf_first, cdf_second = cdfs - (cdfs>>1), (cdfs>>1) + 128` | CONFIRMED | |
| packed word | `(num_bits<<24) \| (state_next<<16) \| (symbols & 65535)` | CONFIRMED | `uint32` |

## E.5 Skip mode

| constant | value | tag |
|---|---|---|
| `skip_block_size` | 1 | CONFIRMED |
| `thr_skip` | 382 | CONFIRMED |
| `skip_judge_thr` | 3 | CONFIRMED |
| `skip_cube_thr` | 1 default, **3** under CTC | CONFIRMED |
| maximum skip fraction | up to 80%, with a 16×16×16 cube override | CONFIRMED |

## E.6 Rate control

| constant | value | tag |
|---|---|---|
| β ladder | 18 entries, 0.0002 … 3.0 | CONFIRMED |
| base-model β | 0.002, 0.012, 0.075, 0.5 | CONFIRMED |
| `betaDisplacementLog` range | [−40, 40] at precision 5 | CONFIRMED |
| ⇒ displacement span | 2^±1.25 = 5.66× | derived — spans the 6.0/6.25/6.67 base-model gaps |
| `MSE_SCALE` | 255² = 65025 | OURS (convention) |
| distortion weights | `{y:6, u:1, v:1}`, Σ→3 | **OURS** — the paper says only "prioritise luma" |

## E.7 Complexity and tools (PAPER)

| item | value |
|---|---|
| SOP / BOP / HOP | 14 / 28 / 215 kMAC/pixel |
| `synthesis_transform_id` | cumulative capability list: `[0]` / `[1,0]` / `[2,1,0]` |
| encoders | **two**, not three — no `sop_*` encoder |
| RVS | 2.2 pp BD-rate, ≈0 kMAC |
| LSBS | 0.4 pp, 0.1 kMAC |
| LEF | 0.3 pp, ≈0 kMAC |
| ICCI | 0.2 pp, **4.6 kMAC/pixel = 17% of the BOP budget** |
| EFE linear / nonlinear | −0.2 pp each, but +12% / +8% chroma PSNR |
| headline BD-rate vs VVC | 16.2 / 20.2 / 22.1 (table); 14.4 / 19.9 / 27.0 |
| 4K decode | ≈190 ms |
| analysis transform weights | 4.9 M |

## E.8 Metric conventions (CONFIRMED from `jpeg-ai-qaf/metrics.py`)

| metric | plane | range | note |
|---|---|---|---|
| MS-SSIM | Y | 0…1023 | |
| VIF | Y | 0…1 | |
| FSIM | **RGB** | 0…1 | the only colour metric |
| NLPD | Y | 0…1 | the only lower-better metric; negated |
| IW-SSIM | Y | **0…255** | |
| PSNR-HVS | Y | 0…1, replicate-padded to ×8, float64 | our own DCT backend (§23.2) |
| VMAF | Y | — | ffmpeg / Netflix v2.2.1 |

All at 10-bit internal precision: `MetricParent(bits=10, max_val=1023)`. AVG = the **unweighted
arithmetic mean** of the seven, verified against all 19 data rows of Tables III–VI (§8.2).

## E.9 The six things that are genuinely ours

1. Per-stage analysis/synthesis trunk widths (the supplement's Figs. 6–8 are paywalled).
2. The distortion weighting `{y:6, u:1, v:1}`.
3. The entire training recipe — optimiser, schedule, step count, batch size, crop size.
4. The T1/T2/TP/TR tables (learned, not the standard's).
5. The PSNR-HVS backend.
6. The MOP decoder (id 3) — an extra data point of our own, not in the standard, disabled by
   default and labelled wherever it appears.

<div class="page-break"></div>

# Appendix F — Bibliography

**The subject paper**

S. Esenlik, Y. Wu, Z. Zhang, Y.-K. Wang, K. Zhang, L. Zhang, J. Ascenso, S. Liu, "An Overview of the
JPEG AI Learning-Based Image Coding Standard," *IEEE Transactions on Circuits and Systems for Video
Technology*, vol. 36, no. 2, pp. 2520–2537, February 2026. DOI 10.1109/TCSVT.2025.3613244

**The standard**

ITU-T T.840 | ISO/IEC 6048, *Information technology — JPEG AI Learning-based image coding system*,
five parts, 2025. Part 1 (core coding system) is the normative source of every constant in
Appendix E and was **not accessible** to this project. Part 3 is the reference software.

ISO/IEC JTC 1/SC 29/WG 1, *JPEG AI reference software*, `gitlab.com/wg1/jpeg-ai`. **Publicly
reachable, and the source of nine of our ten resolved open questions (chapter 9).**

ISO/IEC JTC 1/SC 29/WG 1, *JPEG AI Quality Assessment Framework (QAF)* — the source of the metric
conventions in §11.2 and Appendix E.8.

**Foundational learned-compression literature**

J. Ballé, V. Laparra, E. P. Simoncelli, "End-to-end Optimized Image Compression," *ICLR* 2017 — the
additive-noise relaxation of quantisation (§5.5) and the rate–distortion formulation of §5.3.

J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston, "Variational Image Compression with a Scale
Hyperprior," *ICLR* 2018 — the hyperprior (§5.6), the single largest source of gain in learned
compression.

D. Minnen, J. Ballé, G. Toderici, "Joint Autoregressive and Hierarchical Priors for Learned Image
Compression," *NeurIPS* 2018 — the mean-scale prior and the autoregressive context model whose
serialisation problem the MCM solves.

D. He, Y. Zheng, B. Sun, Y. Wang, H. Qin, "Checkerboard Context Model for Efficient Learned Image
Compression," *CVPR* 2021 — the checkerboard partition the 4-stage MCM generalises (§6.5).

**Classical codecs and standards, for the anchors**

G. K. Wallace, "The JPEG Still Picture Compression Standard," *Communications of the ACM*, 1991.

B. Bross, Y.-K. Wang, Y. Ye, S. Liu, J. Chen, G. J. Sullivan, J.-R. Ohm, "Overview of the Versatile
Video Coding (VVC) Standard and its Applications," *IEEE TCSVT*, 2021 — the paper's own anchor.

AOMedia, *AV1 Bitstream & Decoding Process Specification*, 2019 — behind our AVIF anchor.

**Metrics**

G. Bjøntegaard, "Calculation of average PSNR differences between RD-curves," ITU-T SG16 Q.6 document
VCEG-M33, 2001 — the BD-rate method. **Its global-cubic form is what §22.1 shows to be invalid for
saturating metrics.**

F. N. Fritsch, R. E. Carlson, "Monotone Piecewise Cubic Interpolation," *SIAM Journal on Numerical
Analysis*, 1980 — PCHIP, the interpolant we use instead.

Z. Wang, E. P. Simoncelli, A. C. Bovik, "Multiscale structural similarity for image quality
assessment," *Asilomar* 2003 — MS-SSIM.

Z. Wang, Q. Li, "Information Content Weighting for Perceptual Image Quality Assessment," *IEEE TIP*,
2011 — IW-SSIM.

H. R. Sheikh, A. C. Bovik, "Image Information and Visual Quality," *IEEE TIP*, 2006 — VIF.

L. Zhang, L. Zhang, X. Mou, D. Zhang, "FSIM: A Feature Similarity Index for Image Quality
Assessment," *IEEE TIP*, 2011.

V. Laparra, J. Ballé, A. Berardino, E. P. Simoncelli, "Perceptual image quality assessment using a
normalized Laplacian pyramid," *Human Vision and Electronic Imaging*, 2016 — NLPD.

N. Ponomarenko et al., "On between-coefficient contrast masking of DCT basis functions," *VPQM*,
2007 — PSNR-HVS.

Netflix, *VMAF — Video Multi-Method Assessment Fusion*, v2.2.1, via ffmpeg.

**Datasets**

Eastman Kodak Company, *Kodak Lossless True Color Image Suite*, 24 images —
`r0k.us/graphics/kodak/`.

E. Agustsson, R. Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study,"
*CVPR Workshops* 2017 — DIV2K, `data.vision.ee.ethz.ch/cvl/DIV2K/`.

**Software**

J. Bégaint, F. Racapé, S. Feltman, A. Pushparaja, "CompressAI: a PyTorch library and evaluation
platform for end-to-end compression research," 2020.

PyTorch (MPS backend), NumPy, SciPy (`PchipInterpolator`), Pillow with `pillow-avif-plugin`,
`pytorch-msssim`, `piq`, `pyiqa`, ffmpeg, matplotlib, pandas, pytest, weasyprint.

**This project**

`github.com/nizamulhaq500/JPEC-AI` — 14,027 lines of Python, 331 tests, 9 documents, and the two
rate ladders whose bitstreams produced every measured number in this report.
