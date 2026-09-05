# JPEG AI — a from-scratch implementation of the learning-based image coding standard

A working neural image codec built to the architecture described in:

> S. Esenlik, Y. Wu, Z. Zhang, Y.-K. Wang, K. Zhang, L. Zhang, J. Ascenso, S. Liu,
> "An Overview of the JPEG AI Learning-Based Image Coding Standard,"
> *IEEE Trans. Circuits Syst. Video Technol.*, vol. 36, no. 2, pp. 2520–2537, Feb. 2026.
> [DOI: 10.1109/TCSVT.2025.3613244](https://doi.org/10.1109/TCSVT.2025.3613244)

Real bitstreams, not estimates: every reported rate comes from bytes actually written by
the entropy coder, and every reported image is decoded from those same bytes.

## Status

Seven of the [14 planned phases](docs/03-implementation-plan.md) are built. The plan now
carries a [revised roadmap](docs/03-implementation-plan.md#phase-map-revised) naming which of
the remaining phases get finished, which are cut, and why.

| | | |
|---|---|---|
| **Phases 1–2** | environment, datasets, anchor codecs, 7 metrics, BD-rate harness | done |
| **Phase 3** | mean-scale hyperprior, rANS entropy coder, full training loop | done |
| **Phase 4** | two-branch YCbCr architecture (separate luma / chroma latents) | done |
| **Phase 5** | residual coding with split hyper decoders | done |
| **Phase 6** | the 4-stage Multi-Context Model (§VI-D), coset order from the WG1 reference software | done |
| **Phase 8** | gain unit, 3D quality map / RoI, bit-rate matching, the paper's 4-stage schedule | built, untrained |
| **Phase 10** | RVS, LSBS, the four post-filters | next |
| **Phases 13–14** | ablation table, encoder/decoder CLI, demo, report | the deliverable |

Verification: **498 pytest tests**, **210 self-test checks** (215 with a checkpoint), 7/7 of
the paper's metrics live, and the coder gated to ±0.5% against its own entropy model at
every rate point.

## Measured result

Nine rate points (β 0.0002 … 0.2) at the paper's own latent widths — luma 160, chroma 96 —
50,000 steps each, each point warm-started from the one below it. Measured on 24 Kodak
images over all seven of the paper's metrics against JPEG, WebP, AVIF and VVC, the last
through VVenC at `preset slower` standing in for VTM. BD-rate, negative = fewer bits at
equal quality:

| vs JPEG | AVG | ms_ssim | iw_ssim | fsim | vif | nlpd | vmaf | psnr_hvs | overlap |
|---|---|---|---|---|---|---|---|---|---|
| **this codec** | **−21.4%** | −47.8 | −38.6 | −36.2 | −23.9 | −19.5 | **+4.6** | **+11.3** | 8/11 |
| WebP | −10.6% | −13.3 | −10.2 | −3.4 | −24.0 | −20.0 | −1.7 | −1.8 | 9/11 |
| AVIF | −36.1% | −42.3 | −39.5 | −37.7 | −41.2 | −40.7 | −26.5 | −24.6 | 10/11 |
| VVC | −45.9% | −49.1 | −47.9 | −44.4 | −53.9 | −50.6 | −41.1 | −34.2 | 9/11 |

Comfortably past JPEG and WebP; ahead of AVIF on the two structural metrics and behind it
overall; behind VVC. `overlap` counts how many of JPEG's eleven rate points fall inside the
shared quality range — BD-rate averages over that overlap only, so reporting it is not
optional.

Full report: [`results/p6_9pt_vs_vvc.md`](results/p6_9pt_vs_vvc.md). It regenerates from the
stored JSON with `runbench --rerender p6_9pt_vs_vvc`, re-encoding nothing.

### Where the gap to the standard is

Re-anchored on VVC, which is how the paper reports (it anchors on VTM), this codec sits at
**+47.1%** AVG. The standard's own Kodak figures are **−7.5% / −12.9% / −21.1%** for its three
decoder complexities ([docs/05](docs/05-tables-figures-verified.md) checks that arithmetic) —
and those are with the optional tools *on*, which this codec does not have, so the like-for-like
target is a few points weaker than they look. Either way the gap is around 60 points, and the
diagnostic table says where it lives:

| vs VVC | psnr_y | psnr_u | psnr_v | ms_ssim | iw_ssim | fsim | vmaf | psnr_hvs |
|---|---|---|---|---|---|---|---|---|
| this codec | **+172.1** | +4.3 | +7.0 | **−1.9** | +17.6 | +14.6 | +79.2 | +80.9 |
| JPEG | +128.4 | +181.5 | +140.2 | +96.3 | +92.1 | +79.7 | +69.7 | +51.9 |

Two branches, two verdicts. **Chroma is VVC-class** — +4.3% and +7.0% where JPEG is +181% and
+140% — and MS-SSIM is level with VVC. **Luma is the entire deficit** at +172.1%, and it drags
every pixel-error metric with it.

MS-SSIM is also a worked example of why the `overlap` column exists. Against JPEG, VVC's −49.1%
beats this codec's −47.8%; measured directly against VVC over their 6/10 shared range, this
codec comes out marginally *ahead* at −1.9%. Both are correct over their own overlap window,
and BD-rate across different windows is not transitive. The honest reading is "competitive with
VVC on MS-SSIM", not a winner either way.

Three causes, ordered by how much is actually known about each:

1. **Training budget.** 50,000 steps per rate point, against hundreds of thousands per model
   × four models for the verification model. Measured rather than assumed: 4× the steps at
   one β is worth **+0.90 dB at matched rate**, or equivalently **15.6% fewer bits at matched
   PSNR** — one rate point on Kodak, interpolated with the same PCHIP the harness uses, so it
   is a rate saving on PSNR and not a seven-metric BD-rate. The curve had not flattened.
2. **Missing tools.** No RVS, LSBS or post-filters yet. The paper attributes **3.0–4.2 pp** of
   AVG to those and puts the gain in **FSIM and VMAF** — precisely the metrics this codec
   loses. That is Phase 10, and it is why Phase 10 is next.
3. **MAC efficiency.** This decoder costs **325 kMAC/pxl** against the paper's 8 / 28 / 215 for
   SOP / BOP / HOP: one synthesis transform doing the work of three, and less efficiently per
   MAC. Phase 7 is the fix, and it is deliberately off the critical path — see the roadmap.

### Two results that came out against the plan

**Phase 6's multi-stage context model is worth 1.8% rate, not the +0.60 dB it was credited
with — and all of it is in the stage count.** The original figure compared the MCM against a
checkpoint that had also had 50,000 fewer steps, so it was mostly measuring the budget. Three
runs at β 0.012 fix that: same seed weights, same 200,000 steps, same tier, differing only in
the context model. On 24 Kodak images:

| MCM stages | bpp | psnr | psnr_y | ms_ssim | psnr_hvs | vmaf |
|---|---|---|---|---|---|---|
| none | 0.9620 | 33.735 | 34.524 | 0.9950 | 34.577 | 90.74 |
| 1 | 0.9636 | 33.727 | 34.544 | 0.9950 | 34.592 | 90.66 |
| **4 (the paper's)** | **0.9446** | 33.738 | 34.513 | 0.9950 | 34.588 | **91.24** |

Quality is matched to within 0.011 dB PSNR and 0.015 dB PSNR-HVS with MS-SSIM equal to four
decimals, so the bpp column is a clean rate comparison: the 4-stage model spends **1.81% fewer
bits** than no context model at all, and 1.97% fewer than a 1-stage one — while the 1-stage
model is worth **nothing** (+0.17%, i.e. marginally worse than none). The paper's specific
design choice is the part that pays; a context model as such is not. Single-β runs cannot
produce a BD-rate, which is why the `+nan%` rows in
[`results/p6_200k_3way.md`](results/p6_200k_3way.md) are correct and why this is reported at
matched quality instead. At the shorter 50,000-step budget the tool is invisible: 1-stage and
4-stage land on 32.690 and 32.687 dB at the same 0.917 bpp, indistinguishable. This tool needs
budget before it shows up at all.

**BD-rate was wrong by up to 72 points**, from a global cubic fit that is invalid on metrics
that saturate near 1.0. [docs/07 §5.4](docs/07-training-runbook.md) is the post-mortem; the
harness now uses monotone PCHIP and carries an invariance regression test.

Development before this ran at a reduced "Tier A" width (96 latent channels), which has a hard
ceiling: with the quantiser *disabled* those transforms still top out at **32.30 dB**, and
block PCA at the same width tops out at 30.91 dB — so the bottleneck was dimensionality, not
the entropy coder ([docs/07 §5.2](docs/07-training-runbook.md)). Moving to the paper's widths
was worth **+2.12 dB** at a matched 1.34 bpp. Every number on this page is at the paper's
widths.

## What happens next

The [revised roadmap](docs/03-implementation-plan.md#phase-map-revised) is the authority; in
short:

1. **Train the Phase 8 checkpoint.** The variable-rate path — gain unit, the four-stage
   schedule, the Δβ sweep through the benchmark harness — is built and verified end to end
   against a smoke checkpoint. What it needs is one real Stage III/IV run, which then produces
   the ≥10× rate span from a single model and the variable-rate BD-rate penalty that the paper
   does not publish.
2. **Phase 10 — RVS, LSBS, post-filters.** The only remaining phase with a measured reason to
   expect gain, for the reason in cause 2 above.
3. **Phases 13–14 — the deliverable.** Ablation table, `jpegai encode` / `decode` / `inspect`
   CLI, the RoI and progressive-decode demo, and the written report.

Cut, with reasons recorded in the roadmap: Phase 11 (cross-device integer bit-exactness),
most of Phase 12 (arbitrary image sizes kept), me-tANS from Phase 9 (skip mode kept). Phase 7
is optional and runs only if 13–14 land early.

## Read the docs in this order

| # | File | What it is |
|---|---|---|
| 0 | [00-START-HERE.md](docs/00-START-HERE.md) | Index, current status, and what each phase settled |
| 1 | [01-foundations.md](docs/01-foundations.md) | The prerequisites the paper assumes: rate–distortion, VAEs as codecs, hyperpriors, context models, ANS |
| 2 | [02-jpeg-ai-explained.md](docs/02-jpeg-ai-explained.md) | The paper section by section, with every shape, equation and rationale |
| 3 | [03-implementation-plan.md](docs/03-implementation-plan.md) | The 14-phase build plan, and the revised roadmap over the top of it |
| 4 | [04-reference-data.md](docs/04-reference-data.md) | Extracted tables, marker codes, tensor-shape cheat sheet |
| 5 | [05-tables-figures-verified.md](docs/05-tables-figures-verified.md) | Tables III–VI checked arithmetically; establishes Kodak as the yardstick |
| 6 | [06-normative-constants.md](docs/06-normative-constants.md) | Every constant not in the paper, traced to the WG1 reference software |
| 7 | [07-training-runbook.md](docs/07-training-runbook.md) | **How to run it**: pre-flight, the training commands, what the log lines mean, results so far |

## Quick start

```bash
bash setup.sh
```

Then confirm the whole codec path — transforms, CDF construction, rANS round-trip — on your
device in about 30 seconds:

```bash
.venv/bin/python -m jpegai.models.selftest --device mps
```

Train the rate ladder that produced the numbers above (see
[docs/07 §7](docs/07-training-runbook.md) for wall-clock estimates — this is a GPU-hours job,
not a laptop one):

```bash
.venv/bin/python -u -m jpegai.train.runladder --tier full --model twobranch-mcm --name ladder_p6 --betas 0.0002,0.0005,0.001,0.002,0.005,0.012,0.03,0.075,0.2 --iterations 50000 --batch 8 --device cuda
```

Measure it against the four anchors through the same harness that produced the table above:

```bash
.venv/bin/python -m jpegai.eval.runbench --codecs jpeg,webp,avif,vvc --neural checkpoints/ladder_p6 --out bench_p6
```

Or sweep one variable-rate checkpoint instead of a ladder of them — this is the Phase 8 path,
built and verified end to end but with no trained weights behind it yet:

```bash
.venv/bin/python -m jpegai.eval.runbench --codecs jpeg --neural checkpoints/vr --delta-beta -1069,-800,-600,-400,-200,0,200,450,702 --out bench_vr
```

## Layout

```
jpegai/
  models/        transforms, hyperpriors, entropy models, CDF tables, MCM, gain unit, self-test
  models/tools/  reserved for Phase 10's switchable tools (RVS, LSBS, post-filters) — empty
  coder/         encoder-side and non-normative: bit-rate matching. Nothing here enters a bitstream
  codestream/    reserved for Phase 9's marker-level codestream — empty
  train/         training loop, losses, the paper's 4-stage schedule, the rate-ladder driver
  eval/          metrics, anchor codecs, BD-rate, the benchmark harness
  data/          dataset loaders and the DIV2K zip salvage tool
  config/        tierA.yaml (development) and full.yaml (the paper's widths)
cloud/           the marimo notebook that drives a rented GPU box
cloud_results/   logs and ladder.json from those runs — the provenance of the numbers above
docs/            the written study and the runbook
results/         benchmark JSON, markdown and plots, all regenerable without re-encoding
tests/           498 tests
tools/           scan_per_image.py — median/MAD outlier scan that catches entropy-coder desyncs
```

The `coder/` ↔ `codestream/` split is deliberate and worth one sentence: a change under
`coder/` can cost BD-rate and encode time but can never make a stream undecodable, and a
change under `codestream/` can.

## What is not in this repo

Datasets (`data/`), trained weights (`checkpoints/`, except each run's `ladder.json`), the WG1
reference software (`ref/`, a separate project under its own terms), and the paper's own text
and figures (IEEE copyright). `setup.sh` fetches the datasets;
[docs/07-training-runbook.md](docs/07-training-runbook.md) produces the weights.

The written report is also absent on purpose. A draft exists locally, but it predates the
results on this page and is being rewritten from scratch as part of Phase 14 — once as the
report proper, and once at more depth for my own use.
