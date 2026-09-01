<div class="page-break"></div>

# Part IV — The Build

*What the plan was, what each phase produced, what the code looks like, and how it is checked.*

## 13. The phase-wise plan

### 13.1 Why fourteen phases and not "build the codec"

The plan was fixed before any codec code was written, and it has one organising principle:
**measurement precedes the thing measured.** Phases 1 and 2 build no codec at all. They build the
anchors (JPEG, WebP, AVIF), the seven metrics, and the BD-rate harness. Only then does phase 3
build a codec, and it is immediately measurable.

The reason is the one stated in §2.3. A neural codec that is silently broken reports plausible
numbers. If the measurement machinery is built *after* the codec, its first output is
unfalsifiable — you cannot tell a bad codec from a bad measurement. If it is built first and
validated against codecs whose performance is already published (WebP should land near −10%, AVIF
near −36% on Kodak against JPEG), then when the codec's number is strange, the strangeness is
attributable.

That is exactly what happened. §22.1's BD-rate bug was caught because *WebP's* number was wrong,
not because ours was.

Each phase has explicit **completion criteria** written before the phase starts. A phase is not
"done" when the code exists; it is done when its criteria pass. Several phases have criteria still
outstanding and they are named as such.

### 13.2 The fourteen phases

| P | Name | Delivers | Status |
|---|---|---|---|
| **1** | Foundation & anchors | dataset pipeline, JPEG/WebP/AVIF anchors, project skeleton | **complete** |
| **2** | Metrics & BD-rate | all 7 paper metrics + PSNR/Y/U/V, PCHIP BD-rate, plots, report writer | **complete** |
| **3** | Baseline neural codec | mean-scale hyperprior, rANS coder, training loop, round-trip gate | **complete, trained** |
| **4** | Two-branch YCbCr | separate luma (160) / chroma (96) branches, cross-branch conditioning | **complete, smoke-trained** |
| **5** | Residual + split hyper decoders | eqs (1)–(3), prediction head and scale head, integer `Iσ` | **complete, trained** |
| **6** | Multi-Context Model | the 4-stage checkerboard of §VI-D | **complete, training now** |
| **7** | Three synthesis transforms | SOP / BOP / HOP at 14 / 28 / 215 kMAC/pxl on one codestream | not started |
| **8** | Variable rate | per-channel gain vectors, β displacement, the 18-point ladder from 4 parameter sets | not started |
| **9** | me-tANS + codestream | the real entropy coder, markers, headers, skip mode | not started |
| **10** | Coding tools | RVS, LSBS, LEF, ICCI, EFE ×2 | not started |
| **11** | Integer bit-exactness | the 8-bit/32-bit integer hyper scale decoder, cross-device conformance | partially — the integer scale decoder exists (§6.4) |
| **12** | Functionality | ROI, progressive decoding, tiling, arbitrary sizes, HDR metadata | not started |
| **13** | Final evaluation | the full 18-point ladder, all datasets, the ablation table | not started |
| **14** | CLI & packaging | a `jpegai` command that encodes and decodes files | not started |

### 13.3 The "minimum defensible core", and why it lands at phase 6

A judgement call was needed early: what is the smallest subset of the standard that can honestly
be called *an implementation of JPEG AI*, as opposed to a generic learned codec?

The answer is **phases 1–6**, and the argument is that these six phases contain every part of the
architecture that is *specific to JPEG AI* rather than generic to learned compression:

- the **two-branch YCbCr** split with cross-component conditioning (P4) — no research codec does
  this; it is JPEG AI's distinctive structural choice;
- the **residual coding** against a learned prediction, eqs (1)–(3) (P5) — likewise;
- the **split hyper decoders** producing a *prediction* and an *integer scale index* separately
  (P5) — this is the design that makes bit-exactness possible without integerising the whole
  network;
- the **4-stage MCM** (P6) — JPEG AI's answer to the sequential-context problem, and the reason
  its decode latency is constant in image size.

Phases 7–14 are, in a precise sense, *engineering around* that core: complexity scalability,
rate scalability, the specific entropy coder, the post-filters, the conformance apparatus, and
the file format. They matter enormously for a shipped standard and they add roughly 4–6% of
BD-rate between them. But an implementation that has 1–6 and lacks 7–14 is a correct JPEG AI
core with missing tools; an implementation that has 7–14 and lacks 4–6 is not JPEG AI at all.

This is why phase 6 was chosen as the point at which the report is written.

### 13.4 The two tiers

Every phase is implemented once and configured twice.

| | `full.yaml` (the paper) | `tierA.yaml` (development) |
|---|---|---|
| luma latent | 160 | **96** |
| chroma latent | 96 | **48** |
| hyper latent | = branch width | = branch width |
| trunk widths | wider | narrower |
| params | 4,903,491 | ~3.75 M |
| decoder kMAC/pxl | 132.4 | ~112 |
| train time, 5-point ladder | ~19–26 h | ~12 h |

Tier A exists because a five-point ladder at full width takes about a day and a half of laptop
time, and the pipeline needed dozens of iterations. It is a **development** configuration, not a
result — §20 shows it caps the codec at 32.3 dB, and every tier A number in this report is
labelled.

## 14. The code

14,027 lines of Python: 9,745 implementation across 33 modules, 4,282 test across 12 files.

### 14.1 Layout

```
jpegai/
  models/        the codec itself
    analysis.py       analysis transform (RGB single-branch)
    synthesis.py      synthesis transform
    twobranch.py      the YCbCr two-branch model, eqs (1)-(3)
    hyper.py          hyper encoder / decoder, split heads
    mcm.py            the 4-stage Multi-Context Model
    entropy.py        factorised, scale-hyperprior, mean-scale models
    sigma.py          the integer sigma-index path, eq. (5)-(10)
    quant.py          STE, additive noise, discretised likelihood
    colour.py         BT.709 forward and inverse, eq. (4)
    complexity.py     parameter counts and kMAC/pixel accounting
  coder/         real bytes
    rans.py           the rANS entropy coder
    tans.py           me-tANS constants and table construction
    bitstream.py      container, markers, headers
    skip.py           skip mode
  train/
    losses.py         L = beta*D + R, the distortion weighting
    loop.py           the training loop, gate hooks, checkpointing
    ladder.py         multi-point rate ladders
    warmstart.py      partial state-dict loading across architectures
  metrics/
    seven.py          the paper's seven metrics with per-metric conventions
    psnr.py           PSNR and per-plane PSNR-Y/U/V
    bdrate.py         PCHIP BD-rate, overlap accounting
    bench.py          the benchmark driver
    report.py         markdown and PNG output
  data/
    prepare_crops.py  DIV2K -> 6,400 variance-filtered 256x256 crops
    salvage_zip.py    CRC-verified recovery from a truncated archive
    loaders.py        datasets and augmentation
  anchors/
    classical.py      JPEG, WebP, AVIF at matched quality points
  selftest.py         the 210-check end-to-end self test
  cli.py              runtrain / runladder / runbench / selftest
config/
  full.yaml  tierA.yaml  metrics.yaml  constants.yaml
tests/            12 files, 331 tests
docs/             9 documents, 5,051 lines
```

### 14.2 The modules that carry the most weight

**`models/sigma.py`** is the most delicate file in the project and the one most tightly bound to
the standard. It implements eqs (5)–(10): the integer scale-index path, the log-spaced 32-level σ
quantisation, the `ceil(Iσ/2⁷)` CDF row selection, and the integer hyper scale decoder with 8-bit
multipliers and 32-bit accumulators. It also contains the measurement of §6.4.2 — the exact
eleven indices where the integer and float paths disagree, all of them multiples of 128, all of
them one float32 ULP apart — and a test that asserts the disagreement set is *exactly* those
eleven, so any drift is caught immediately.

**`models/mcm.py`** implements the 4-stage checkerboard. Its subtlety is not the algorithm but the
*channel layout*: `PixelShuffle`'s inverse is not `chunk`, and getting that wrong produced §22.2's
bug — a model that trains fine and whose context model is predicting from permuted channels. The
module docstring now spells out the layout explicitly, and a test asserts the round trip
`up_shuffle(down_shuffle(y)) == y` plus the mean-field deviation bound that caught it.

**`metrics/bdrate.py`** is the file whose bug cost the most (§22.1). It now uses
`scipy.interpolate.PchipInterpolator` — monotone piecewise-cubic Hermite — instead of a global
cubic, and it *always* returns `overlap_coverage` alongside the BD-rate, because the two numbers
are meaningless apart.

**`train/losses.py`** carries the β-direction derivation as a docstring, because the sign
convention is the single easiest thing to get backwards in this literature and getting it
backwards produces a codec that optimises for *large* files. §5.2.

### 14.3 Configuration and provenance tags

Constants live in YAML, and each carries its provenance tag in a comment:

```yaml
sigma:
  quant_level:   32        # CONFIRMED  CCS_SGMM/params.py
  quant_min:     0.11      # CONFIRMED  CCS_SGMM/params.py
  quant_max:     54.82     # CONFIRMED  CCS_SGMM/params.py
  precision:     7         # CONFIRMED  quantization/params.py (triple-confirmed, see 9.2)
  bound_offset:  0.5       # CONFIRMED as a constant; meaning unexplained (12, item 3)
  isigma_pad:    1411      # PAPER, unconfirmed - not found in reference software
distortion:
  weights: {y: 6, u: 1, v: 1}   # OURS - the paper says only "prioritise luma"
```

A loader test asserts that every constant tagged CONFIRMED matches the value recorded in
`docs/06-normative-constants.md`, so the documentation and the code cannot drift apart.

## 15. The verification harness

Three layers, and they catch different things.

### 15.1 The three layers

**Layer 1 — 331 unit tests.** Shape agreement, gradient flow, the σ table's exact disagreement
set, the shuffle round trip, BD-rate's invariance property, the metric conventions, the colour
transform's inverse. These catch *code* errors.

**Layer 2 — the 210-check self-test.** One command, ~30 seconds, runs the entire encode→bytes→
decode→metrics path on real images and asserts 210 properties of the result (215 when a checkpoint
is supplied, which adds the checkpoint-specific gates). This is what the user runs after any
environment change. It catches *integration* errors — the ones where every unit passes and the
assembly is wrong.

**Layer 3 — the mid-training gates.** Assertions that run *during* training, at checkpoint
boundaries, on the model as it currently stands. These catch the errors that only appear in a
partly-trained model, and §22.3 is a bug that was visible **only** through this layer.

### 15.2 The round-trip gate — the single most valuable test in the project

Objective 3 said: real bytes, always. The gate that enforces it runs at every checkpoint and
asserts three things.

**(a) The decoded latent is bit-identical to the encoded latent.**

```
yhat exact ✓
```

Not "close". Identical. If the entropy decoder returns even one different integer, the assertion
fires. This is the property that separates a codec from a research prototype: it means the decoder
is genuinely reconstructing from the bitstream, not peeking at the encoder's tensors.

**(b) The actual byte count agrees with the model's own rate estimate.** The training loss
computes an *estimated* rate from the likelihoods; the coder produces an *actual* byte count. The
gate compares them:

```
est bpp 1.3719   act bpp 1.1533   gap_q +0.05%
```

`gap_q` is the residual disagreement after accounting for the known estimate/actual difference
(the estimate uses continuous likelihoods; the actual uses the quantised CDF tables). Across
ladder #0's five points it is **+0.31 / +0.17 / +0.04 / +0.05 / +0.04 %** — well inside the ±0.5%
criterion. A wrong CDF table, a wrong σ row, a wrong offset — any of these blows `gap_q` up
immediately, and this is precisely how §22.3's 1.8% payload bug was found.

**(c) The out-of-range rate is zero.**

```
oor y 0.000%
```

`oor` counts latent values falling outside the coder's representable range, which would have to be
escape-coded. Zero means the σ tables span the actual distribution.

### 15.3 The invariance test — how a measurement tool is verified

A BD-rate implementation cannot be checked against a known answer, because there is no reference
value to compare to. So it is checked against an **invariance** instead, which is the standard
technique when ground truth is unavailable.

The property: BD-rate measures a ratio between two curves *inside their overlapping quality
window*. Points of the anchor that lie **outside** that window must not affect the answer. So
build four anchor sweeps that are **identical inside the window and different below it**, and
require the four BD-rates to agree.

| interpolant | spread across the four sweeps |
|---|---|
| **PCHIP** (monotone piecewise cubic Hermite) | **0.04 points** |
| global cubic (textbook Bjøntegaard) | **17.08 points** |

The global cubic fails the invariance by 17 points. That is the whole of §22.1's bug, exhibited by
a four-line test — and it is the reason the test exists.

### 15.4 What the harness does not cover

Named for honesty:

- **Conformance with the real standard.** Untestable without the normative tables. Our gates prove
  our decoder inverts our encoder; they cannot prove either matches WG1's.
- **Cross-device bit-exactness.** The integer path exists (§6.4) but has only been run on one
  machine, so its portability is argued rather than demonstrated.
- **Perceptual quality by human observers.** All seven metrics are proxies. The paper's own
  subjective test results are not reproducible by us.
- **The `runladder --bench` combined path.** Written, not yet exercised end to end.
