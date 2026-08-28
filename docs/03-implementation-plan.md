# 03 — Implementation Plan

**Goal:** a working, demonstrable JPEG AI–class codec that implements *every architectural
idea in the paper*, that you understand end to end, and that you can defend in front of your
professor.

This plan is deliberately not minimal. It is also deliberately honest about what a single
student on a laptop can and cannot reproduce. Read §0 before anything else — it decides
which of three tracks you commit to.

---

## §0. Hardware reality check, and the three tracks

### What you have

Apple M2 Pro, 10 cores, 16 GB unified memory, **no CUDA**. PyTorch runs on the **MPS**
backend, roughly 3–8× slower than an A100 for conv nets, with occasional ops falling back to
CPU (set `PYTORCH_ENABLE_MPS_FALLBACK=1`).

### What the paper's numbers cost

The VM was trained on the CTTC protocol. Realistically: hundreds of thousands of iterations,
batch 8–16 at 256×256, **per model**, and there are **four** model parameter sets. On a
V100-class GPU that is on the order of a week per model. On your machine, the full
160-channel configuration at four rate points is **out of reach**, and you should say so out
loud in your report rather than quietly producing bad numbers. Being explicit about
compute-boundedness is a sign of maturity, not weakness.

### Therefore, three tracks

| Track | What it is | Effort | What it proves |
|---|---|---|---|
| **A — JPEG-AI-Lite (primary)** | Full architecture, *reduced width*. Every tool from the paper implemented and ablatable. Trained yourself. | The bulk of the project | You understand and can build the whole system |
| **B — Conformance decoder (stretch)** | Decoder only, matching the real standard's syntax, validated against official ONNX models / conformance codestreams | 2–3 focused weeks, gated on getting Part 1 + ONNX | You can implement a *standard*, not just a paper |
| **C — Full-width training (opportunistic)** | Tier A code, 160/96 channels, trained on Colab/Kaggle free GPU | Background, weeks of wall-clock | Your numbers get close to publishable |

**Commit to A. Run C in the background from Phase 8 onward. Attempt B only after Phase 10,
and only if you obtain T.840-1 and the ONNX models.**

### Tier A configuration (the one number set that makes this feasible)

| Parameter | Paper | Tier A | Why |
|---|---|---|---|
| Primary latent channels | 160 | **96** | ~2.8× fewer MACs in the widest layers |
| Secondary latent channels | 96 | **48** | |
| Hyper latent channels | 128/160 | **96** | |
| Latent stride | /16 | /16 | **Do not change** — it's structural |
| Hyper stride | /64 | /64 | **Do not change** |
| MCM stages | 4 | 4 | **Do not change** — it's the paper's contribution |
| Synthesis transforms | 3 (SOP/BOP/HOP) | 3 | **Do not change** — multi-branch is the headline |
| Trained models | 4 λ values | **1 + gain units** | See below |
| Training crops | 256×256 | 256×256 | |
| Batch | 8–16 | 8 | Memory-bound |

**The single most important shortcut:** the paper's own **gain unit / rate adaptation
mechanism (§VI-I, eqs 11–15)** turns one trained model into a continuous rate ladder. So
train **one** model with *randomly sampled* `betaDisplacementLog` per training step, and you
get the whole RD curve from one training run. This is not a hack — it is literally what
JPEG AI does between its four models, and reference [42] is the paper about it. You save 4×
the compute *by implementing a feature of the standard*. Lead with this in your report.

---

## Phase map

```
 P1  Environment, data, baselines            ┐
 P2  Metrics + BD-rate harness               │ Weeks 1–2   foundation
 P3  Reproduce a scale hyperprior            ┘
 P4  Two-branch YCbCr architecture           ┐
 P5  Mean+scale, split hyper decoders        │ Weeks 3–5   the codec core
 P6  MCM (4-stage checkerboard)              ┘ ◄── MINIMUM DEFENSIBLE CORE
 P7  Three synthesis transforms              ┐
 P8  Variable rate: gain / 3D gain / RoI     │ Weeks 6–8   the JPEG AI part
 P9  me-tANS + skip mode + real codestream   ┘
 P10 RVS + LSBS + post-filters               ┐
 P11 Integer entropy path + bit-exactness    │ Weeks 9–11  the standards part
 P12 Tiling, regions, arbitrary size, progressive ┘
 P13 Evaluation, ablation, benchmark report  ┐ Weeks 12–14 the deliverable
 P14 Demo app, CLI, report, slides           ┘
 (B) Conformance decoder against real ONNX   — optional, parallel
```

Each phase below has: **objective**, **build**, **paper reference**, **acceptance test**
(the thing that proves it works), **pitfalls**.

---

## Phase 1 — Environment, data, and baselines

**Objective:** a reproducible environment, a dataset, and *classical codec numbers to beat*.
Do not write a single line of model code until you can measure.

**Build**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy pillow tqdm matplotlib scipy
pip install compressai            # scaffolding + baselines + range coder
pip install piq pytorch-msssim    # metrics
pip install streamlit             # demo later
pip install onnx onnxruntime      # Track B later
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Datasets:
- **Training:** DIV2K (800 train, 2K resolution) + Flickr2K (2650). ~15 GB.
  Pre-extract to random 256×256 PNG crops once — on-the-fly cropping of 2K PNGs will
  bottleneck your dataloader on CPU.
- **Validation:** DIV2K valid (100 images).
- **Test:** **Kodak** (24 images, 768×512) — mandatory, it's in Table V so you can compare
  directly. Plus **CLIC 2024 validation** if you can get it (Table VI).
- Optional: a handful of screen/synthetic images to reproduce the paper's honesty about
  synthetic content (§VII-D).

Baselines (anchors). VTM-11.1 is the paper's anchor but is a heavy build; use practical
anchors and treat VTM as a stretch:

| Anchor | How | Priority |
|---|---|---|
| JPEG | `pillow`, quality sweep | must |
| WebP | `pillow` / `cwebp` | must |
| **AVIF** (= AV1 intra ≈ VVC-class) | `pillow-avif-plugin` or `libavif` | **must — this is your real competitor** |
| JPEG XL | `cjxl`/`djxl` | should |
| VVC Intra (VTM-11.1) | build VTM from vcgit.hhi.fraunhofer.de | stretch — enables direct comparison to the paper's tables |
| CompressAI `bmshj2018-hyperprior`, `mbt2018`, `cheng2020` pretrained | pip, one line | must — these are your *learned* baselines |

**Directory layout** — set this up now, it will save you later:

```
jpegai/
  config/            YAML configs (one per operating point / ablation)
  data/              dataset + dataloaders + crop extraction
  models/
    transforms.py    analysis/synthesis, primary + secondary
    hyper.py         hyper encoder, hyper decoder, hyper scale decoder
    entropy.py       factorised prior, conditional Gaussian, CDF building
    mcm.py           4-stage context model
    tools/
      rvs.py  lsbs.py  gain.py  filters.py  skip.py
    codec.py         the full nn.Module
  coder/
    rangecoder.py    Phase 3-8 (from compressai)
    metans.py        Phase 9  — your me-tANS
    bitwriter.py     bit/byte-level IO, byte alignment
  codestream/
    markers.py  headers.py  writer.py  reader.py
  eval/
    metrics.py  bdrate.py  runbench.py  plots.py
  cli.py             jpegai encode / decode / bench
  demo/              streamlit app
  tests/
docs/                (this)
paper/               (this)
```

**Acceptance test:** `python -m jpegai.eval.runbench --codecs jpeg,webp,avif --dataset kodak`
produces an RD curve PNG and a JSON of rate/metric points. **You have a measuring
instrument before you have a codec.**

**Pitfalls**
- MPS + `num_workers>0` in DataLoader can deadlock. Use `num_workers=4,
  persistent_workers=True` and test early.
- Compare bitrates as **total file bytes × 8 / pixels**, including all headers. Learned-codec
  papers sometimes quietly exclude header cost; don't.

---

## Phase 2 — Metrics and BD-rate

**Objective:** all seven of the paper's metrics, plus a BD-rate implementation you trust.

**Paper reference:** §VII-A; the metric list is from WG1 N85013.

**Build**

The plane and input range for each metric are **not** free choices — the QAF reference
fixes them, and six of the seven run on **luma only at 10-bit internal precision**.
Getting this wrong yields plausible numbers that cannot be compared to the paper at all.
Table below now reflects what we actually shipped; see
[06-normative-constants.md §11](06-normative-constants.md) for the QAF provenance.

| Metric | Plane / range | Source we use | QAF's own backend |
|---|---|---|---|
| MS-SSIM | Y, 0…1023 | `pytorch_msssim.ms_ssim` | same |
| VIF | Y, 0…1 | `piq.vif_p` | `IQA_pytorch.VIFs(channels=1)` |
| FSIM | **RGB**, 0…1 | `piq.fsim` | `IQA_pytorch.FSIM(channels=3)` |
| NLPD | Y, 0…1 | `pyiqa` (Y replicated to 3ch) | `IQA_pytorch.NLPD(channels=1)` |
| **PSNR-HVS** | Y, 0…1, replicate-pad to /8, float64 | `psnr_hvsm.psnr_hvs_hvsm()[0]` | same |
| IW-SSIM | Y, **0…255** | `piq.information_weighted_ssim` | QAF's own `IW_SSIM_PyTorch` |
| VMAF | Y | `ffmpeg -lavfi libvmaf` | Netflix binary v2.2.1 |
| PSNR (Y, U, V) | — | ours; the EFE discussion needs chroma PSNR | not one of the seven |

The seventh metric is **PSNR-HVS**, not PSNR-HVS-M: QAF keeps the *first* return value
of `psnr_hvs_hvsm()`. We compute PSNR-HVS-M too, but it is excluded from AVG.

Note `piq`'s 5-scale pyramid metrics (MS-SSIM, IW-SSIM) require inputs of at least
161×161 and return NaN below that. Kodak is 768×512, so this only bites in unit tests.
Using `pytorch_msssim` for MS-SSIM avoids the floor as well as matching QAF.

Write `eval/bdrate.py` yourself — it's 30 lines and understanding it matters:

```python
def bd_rate(r_anchor, q_anchor, r_test, q_test):
    """Bjøntegaard delta rate. Negative = test needs fewer bits. Returns %."""
    lr1, lr2 = np.log(r_anchor), np.log(r_test)
    p1 = np.polyfit(q_anchor, lr1, 3)          # log-rate as f(quality)
    p2 = np.polyfit(q_test,   lr2, 3)
    lo = max(min(q_anchor), min(q_test))       # overlap only
    hi = min(max(q_anchor), max(q_test))
    i1 = np.polyval(np.polyint(p1), [lo, hi])
    i2 = np.polyval(np.polyint(p2), [lo, hi])
    return (np.exp(((i2[1]-i2[0]) - (i1[1]-i1[0])) / (hi-lo)) - 1) * 100
```

Then **the per-metric aggregation the paper uses**: compute BD-rate independently per
metric, then average the seven. (Not: average the metrics then compute one BD-rate.)

**Acceptance test:** BD-rate of WebP vs JPEG on Kodak should land around −25 to −35%.
BD-rate of a codec against itself must be exactly 0.0. Reproduce **Table V's decoderID 1
row** shape qualitatively with the CompressAI `mbt2018` pretrained model vs AVIF — you
should see the same *pattern*: strongly negative MS-SSIM/VMAF, positive PSNR-HVS.

**Pitfalls**
- VMAF expects YUV or video input; wrap it carefully and cache results — it's slow.
- Metrics must be computed in the **output colour space at output bit depth**, after
  clipping (paper eq. 6), not on float tensors in [0,1] pre-clip.
- 4 rate points is the minimum for a cubic fit. Use 5–7.

---

## Phase 3 — Reproduce a scale hyperprior from scratch

**Objective:** before touching JPEG AI specifics, build and train Ballé 2018 yourself, in
your own repo, and match CompressAI's pretrained numbers within a couple of dB. This
de-risks everything downstream: if Phase 6 doesn't train, you'll know it isn't the plumbing.

**Paper reference:** doc 01 §4b; paper's reference [10].

**Build**
- `g_a`: 4× (Conv `k5s2` → GDN or ReLU). `g_s`: mirror with 4× (ConvT `k5s2` → IGDN/ReLU).
- `h_a`: 2–3 stride-2 convs → `z`. `h_s`: mirror → σ.
- Factorised prior for `z`: implement the **non-parametric density model** (Ballé 2018
  Appendix 6.1) — a small monotonic MLP producing a CDF. Or use
  `compressai.entropy_models.EntropyBottleneck` for now and write your own in Phase 9.
- Conditional Gaussian for `y`: `p(ŷ|σ) = Φ((ŷ+0.5−μ)/σ) − Φ((ŷ−0.5−μ)/σ)` with
  `σ = exp(clamp(logσ, −10, 10))`, and a lower bound on σ (`SCALES_MIN = 0.11`).
- Loss: `L = R_y + R_z + λ·255²·MSE`. Sweep `λ ∈ {0.0018, 0.0035, 0.0067, 0.013, 0.025,
  0.0483}` (CompressAI's set) for the baseline only.
- Quantisation: **additive uniform noise for the rate branch, STE for the distortion
  branch** (doc 01 §3).
- Real coding: use `compressai.ans` / range coder so you can produce actual bytes.

**Acceptance test — the gate for the whole project:**
1. Training loss decreases and RD curve beats JPEG comfortably.
2. **`len(encode(x)) * 8 / npixels` is within 1–2% of the model's estimated rate
   `−log₂ p`.** If these diverge, your CDF construction or your coder is wrong, and every
   later phase will be built on sand. Test this obsessively.
3. `decode(encode(x))` reproduces the *latent* `ŷ` exactly (compare tensors, not pixels).

**Pitfalls**
- Off-by-one in CDF quantisation is the classic bug. Build CDFs as
  `cdf = round(cumulative_prob * 2**precision)`, enforce strict monotonicity by adding 1 to
  any zero-width bin, and append the total.
- GDN can be unstable on MPS; ReLU/LeakyReLU works nearly as well and is what JPEG AI uses
  anyway (`decoderID` 0/1 are ReLU/ReLU6-only). **Prefer ReLU — it aligns with the paper.**

---

## Phase 4 — Two-branch YCbCr architecture

**Objective:** the paper's primary/secondary component split (§VI-A, VI-B).

**Build**
1. **Colour pipeline.** RGB → YCbCr BT.709, full range, float. Support internal 4:4:4,
   4:2:2, 4:2:0 (`c_ver_minus1`, `c_hor_minus1`) *independently* of the output format
   (`s_ver_minus1`, `s_hor_minus1`), with a normative-style upsampler when they differ.
   Implement `colour_transform_idx` = 0/1/2, including the encoder-signalled 3×3 matrix +
   bias (eq. 5) and the final scale-and-clip (eq. 6).
   > **Use the textbook BT.709 inverse, not the paper's eq. 4 as printed** — see
   > `02-jpeg-ai-explained.md` §VI-B for the two typos.
2. **Primary branch:** `x_Y [1,H,W]` → 4 stride-2 stages → `y_Y [N_luma, Ḣ/16, Ẇ/16]`.
3. **Secondary branch:** `x_UV [2, H/c_v, W/c_h]` → **3** stride-2 stages for 4:2:0 (4 for
   4:4:4) → `y_UV [N_chroma, Ḣ/16, Ẇ/16]`. The number of stages must be chosen so `y_UV`
   lands on the same grid as `y_Y` — that's what makes eq. 3 valid.
4. **Preprocessing:** the one encoder-side cross-component link — secondary analysis
   consumes `x_Y`. Simplest faithful version: downsample `x_Y` to `x_UV`'s resolution and
   concatenate as an extra input channel.
5. **Concatenation (eq. 3):** `ŷᶜ_UV = concat(ŷ_UV, ŷ_Y)` → `[N_chroma + N_luma, Ḣ/16,
   Ẇ/16]`; the secondary synthesis transform takes that.
   > **Take the widths from the config, never from this document.** `N_luma = 160` and
   > `N_chroma = 96` in `full.yaml` — both confirmed against the reference software, so
   > eq. (3)'s `256 = 96 + 160` is correct as printed. `tierA.yaml` is deliberately half
   > that (96 / 48) to fit this laptop, which makes `144` the Tier A concatenation width.
   > Earlier drafts of this plan quoted the Tier A pair as if it were the paper's; see
   > `06-normative-constants.md` §1–2. Also: `N_luma % 32 == 0` is asserted by MCM's
   > `chs2group()` in Phase 6, so the primary width is not freely choosable.
6. **Monochrome fast path:** a `--luma-only` decode flag that skips the entire secondary
   branch. Cheap to add, and it demonstrates the machine-consumption motivation.
7. Loss weighting: weight luma distortion higher than chroma (the paper's "prioritise the
   quality of the luma component during the training process"). Start 6:1:1 for Y:U:V.

**Acceptance test**
- Round-trip colour conversion (RGB→YCbCr→RGB) max abs error < 1e-5 at 4:4:4. ✅
- All nine combinations of {internal 4:4:4, 4:2:2, 4:2:0} × {output 4:4:4, 4:2:2, 4:2:0}
  produce correctly-sized output. ✅
- `--luma-only` decodes and produces a grey image with correct luma, at reduced decode time
  you can actually measure and report. ✅ `python -m jpegai.eval.lumaonly` — −33.2% payload,
  −23.8% decode time on CPU, luma bit-identical, grey to 7.5e-09.
  > Prove the saving by **deleting** the chroma strings before decoding. A decoder that
  > decodes chroma and discards it produces the right pixels and saves nothing, and passes
  > every other form of this test. Report the *rate* saving as the headline: it is larger
  > and device-independent. The time saving came out slightly above the skipped-kMAC share,
  > because rANS decoding is not multiply–accumulate work and a MAC count cannot see it.
- Two-branch beats single-branch-RGB at equal rate on your validation set (it should — this
  justifies the design). ⏳ needs two matched trained ladders; the comparison path is wired
  (`runladder --model twobranch` → `runbench --neural`) and exercised on a 3,000-step CPU run.

**Pitfalls**
- Chroma at 4:2:0 for an odd-sized image: `⌈H/2⌉`. Ceilings everywhere, and they interact
  with Phase 12's padding.
- Don't forget chroma PSNR in your metrics — Phase 10's EFE filters are justified by it.
  Now logged every step as `Y/U/V xx.xx/xx.xx/xx.xx` alongside `chroma NN.N%`, the
  secondary branch's share of the rate. Those two together are the only live signal that
  6:1:1 is working: a chroma share climbing while `psnr_u`/`psnr_v` stay flat means the
  secondary branch is buying rate and not quality.
- Measure distortion on the **unpadded RGB output**, never on the internal planes. Two
  separate reasons and both bite: the internal planes are still reflect-padded and those
  pixels get cropped, so distortion spent there buys nothing; and at 4:2:0 the internal
  chroma plane is half resolution, so measuring there leaves the chroma **upsampler**
  outside the autograd graph and the secondary synthesis never learns to compensate.
- The 6:1:1 weighting is not testable against a single-branch model — there are no separate
  chroma parameters to measure. Test it by gradient norm on the two branches, against a
  1:1:1 control. And beware the obvious test image: a **uniform RGB offset is a pure luma
  error** (`K_R+K_G+K_B = 1` makes it cancel exactly in Cb and Cr), so 6:1:1 scores it at
  exactly `6·3/8`, not at some fuzzy "more than 1:1:1".
- Normalise the plane weights to `Σw = 3`. Otherwise changing the weighting also changes
  the overall distortion scale, and a weighting ablation is secretly a rate-point ablation.

---

## Phase 5 — Mean+scale prediction and the split hyper decoders

**Objective:** the paper's actual entropy structure — residual coding plus **two separate**
networks off `ẑ` (§VI-E). This is the paper's own architectural contribution and the
cheapest big win in your report.

**Build**
1. **Hyper encoder** (per branch): `y` → 2 stride-2 convs → `z` at `Ḣ/64, Ẇ/64`,
   96 channels. Quantise → `ẑ`, coded with the factorised prior.
2. **Hyper decoder** `ẑ → p̈`. In the real standard `p̈_Y` is `[640, Ḣ/32, Ẇ/32]` and is
   **pixel-shuffled ×2 to `[160, Ḣ/16, Ẇ/16]`**. Reproduce that structure at Tier A width
   (`[384, /32, /32]` → shuffle → `[96, /16, /16]`) — it's a real design detail worth having.
3. **Hyper *scale* decoder** `ẑ → Iσ`, a *separate, deliberately tiny* network producing
   `[96, /16, /16]`. Keep it to 2–3 layers with few channels. **Its output is an integer
   log-domain index, not a float σ.** Store as int; map to σ via
   `σ = exp(Iσ · step / 2**sigmaPrecision)` (mirroring eq. 13's form). Pick
   `sigmaPrecision = 8`, `step` such that your `Iσ` range is a few thousand — matching the
   paper's `[…, 3968]` table extents makes Phase 10 drop straight in.

   > **As built, this differs from the guess above in three ways** — see
   > `docs/06-normative-constants.md` §3. `sigmaPrecision` is **7**, not 8; the mapping
   > carries the σ floor, `σ = 0.11 · exp(log_k · Iσ / 2⁷)` with
   > `log_k = (ln 54.82 − ln 0.11)/31`; and the extent is `3967`, not 3968, which is
   > exactly what forces the CDF row to be `ceil(Iσ/2⁷)` rather than `Iσ >> 7`.
4. **Residual coding** (eqs 1, 2): encoder `r̂ = round(y − p̈)`; decoder `ŷ = r̂ + p̈`. For
   the secondary branch that's the whole story — no context model, ever (paper's explicit
   decision).
5. **Ablation switch** `--single-hyper-decoder` that fuses the two back into one Ballé-style
   `h_s`. You will use this in Phase 13 to *measure the cost of the decoupling*, which is a
   number the paper never publishes. That is genuinely novel work you can present.

**Acceptance test**
- Residual coding beats plain scale-hyperprior latent coding by 3–8% BD-rate.
- The scale decoder is < 5% of total decoder MACs (count them — write a MAC counter now, you
  need kMAC/pxl for Phase 13).
- `--single-hyper-decoder` vs split: report the delta. Expect the split to be slightly worse
  in RD, and say so — that's the paper's trade.

**Status — built, `jpegai/models/hyper.py` + `--model twobranch-split`**

| criterion | state |
|---|---|
| scale decoder < 5% of decoder MACs | **passed — 0.055%** (0.07 of 128.9 kMAC/pxl, both branches) |
| residual coding beats scale-hyperprior by 3–8% | open, needs weights |
| `--single-hyper-decoder` delta reported | ablation built (`--model twobranch-fused`), delta needs weights |

Structurally verified beyond the plan's asks: bit-exact round trip through a real
bitstream for both new kinds; `predict()` provably sees `ẑ` only, pinned on the
signature; every conv in the scale decoder provably runs below the latent grid; the
integer and float index paths disagree on exactly 11 of 3968 indices and the codec is
wired so it can never mix them. Two findings the paper does not state: the confirmed
`h_s` structure is **6.8× cheaper** than a deconv-based mean-scale `h_s` (0.49 vs 3.33
kMAC/pxl) purely because `conv_shuffle` keeps every multiply at /64 and /32; and the
largest σ that `Iσ` can denote is **54.734**, not 54.82.

**Pitfalls**
- `p̈` must be **identical** at encoder and decoder. Any dependence on `y` (rather than only
  on `ẑ`) is a bug that will only surface as decoder drift.
- Quantising `Iσ` to int throws away gradient. During training, use the float log-σ for the
  rate loss and quantise only at inference — then add a quantisation-aware finetune in
  Phase 11.

---

## Phase 6 — MCM: the 4-stage checkerboard context model  ◄ MINIMUM DEFENSIBLE CORE

**Objective:** §VI-D and Fig. 2. Luma only.

**Paper reference:** §VI-D, Fig. 2 (`paper/imgs/table_p8_Fm0.png`); doc 01 §4e; He et al.
CVPR 2021.

**Build**

Partition the `Ḣ/16 × Ẇ/16` latent grid into 4 groups by `(i mod 2, j mod 2)`, identical in
every channel:

```
 g0 g1 g0 g1        stage 1: (0,0)   conditioned on p̈ only
 g2 g3 g2 g3        stage 2: (1,1)   conditioned on p̈, g0
 g0 g1 g0 g1        stage 3: (0,1)   conditioned on p̈, g0, g1(=g3 done)
 g2 g3 g2 g3        stage 4: (1,0)   conditioned on p̈, all previous
```

(The paper doesn't state the group *order*; `(0,0) → (1,1) → (0,1) → (1,0)` is the natural
one — diagonal first maximises the context available to stages 3–4. Note your choice as an
assumption.)

> **No longer an assumption.** [06-normative-constants.md](06-normative-constants.md) §5
> derives exactly this order twice from the WG1 reference software —
> `ContextUtils.down_shuffle` returns `(part1, part4, part2, part3)`, and `context.py`'s
> odd-size guards force `dy=1` on stages {1,3} and `dx=1` on {1,2}, which only the
> diagonal-first order satisfies. `cfg.entropy.mcm_group_order` records it and
> `tests/test_mcm.py` asserts the code, the config and the derivation agree; if they ever
> diverge the config wins and that test is the alarm.

Per stage `k`:
```
ŷ_partial  = ŷ with only groups < k filled, zeros elsewhere
ctx_k      = ContextNet_k( concat(ŷ_partial, p̈, stage_onehot) )      # a few convs
ŷ[group_k] = r̂[group_k] + ctx_k[group_k]
```
Four separate small `ContextNet_k` (the paper says per-stage structure is in Appendix C, so
per-stage weights are the safe reading). Use 3×3 convs; a mask isn't needed because you
zero the not-yet-decoded groups explicitly.

Crucially: **the encoder must run the identical 4-stage loop** to compute the `r̂` values it
codes, because stage `k`'s prediction depends on stages `< k`. Write it once and share it.

**Acceptance test**
- **`ŷ_decoder == ŷ_encoder` exactly, tensor-for-tensor.** Non-negotiable. Assert it in CI.
- MCM gives 4–9% BD-rate over Phase 5. If it gives < 2%, your context nets aren't seeing
  the previous groups — check your masking.
- Decode wall time grows by a *constant* 4 network passes, not with image size. Plot
  decode-time vs megapixels for MCM-on and MCM-off: **two parallel lines**. That plot is
  the single best slide in your presentation.
- Ablation: 1 / 2 / 4 stages. You'll get the paper's conclusion (4 is the sweet spot) or
  learn something interesting.

> ### Stop-and-assess checkpoint
> After Phase 6 you have: a two-branch YCbCr hyperprior codec with residual coding, split
> hyper decoders, a 4-stage parallel context model, real entropy-coded bytes, seven metrics
> and BD-rate against AVIF/WebP/JPEG and three learned baselines. **If you run out of time,
> this is already a defensible, complete project.** Everything after this is what makes it
> *JPEG AI* rather than *a learned codec*.

---

## Phase 7 — Three synthesis transforms on one codestream

**Objective:** the paper's headline structural feature (§VI-F, §VIII). One `.jpegai` file,
three decoders, three quality/complexity points.

**Build**

| decoderID | Name | Upsampling | Final channels (Tier A) | Layer restriction |
|---|---|---|---|---|
| 0 | **SOP** | 2×2 conv + pixel shuffle | 24 | conv, ReLU, ReLU6 **only** |
| 1 | **BOP** | 4×4 transposed conv; pixel shuffle **only in the final layer** | 48 | conv, ReLU, ReLU6 only |
| 2 | **HOP** | richer: residual blocks, attention, GDN, deeper stacks | 96 | unrestricted |

Enforce the layer restriction *in code* — a `assert_simple_ops(module)` check that walks the
module tree and rejects anything outside the allowed set for IDs 0 and 1. Then you can
truthfully claim conformance to the operating-point constraints.

**Training:** all three heads share one analysis transform and one entropy model. Train
jointly: `L = R + λ·(w0·D0 + w1·D1 + w2·D2)`. Start `w = (1,1,1)`; if SOP drags the shared
encoder down, try 0.5/1.0/1.0, and consider a short per-head finetune with the encoder
frozen. **Also implement the two-encoder finding:** train a second analysis transform
optimised for HOP only (`encoderID 1`, `L = R + λ·D2`) and confirm the paper's Table III
result that encoderID 1 is worse with SOP/BOP but markedly better with HOP. Reproducing a
non-obvious published finding is exactly the kind of thing professors reward.

**Acceptance test**
- One file, three decodes, monotone quality SOP < BOP < HOP at fixed rate.
- MAC counts roughly in the paper's ratio (1 : 2 : ~8–15).
- `--decoder-id` on the CLI; the ID also written into the picture header.
- Table III reproduced *in shape*: a 2×3 grid of BD-rates for (encoderID, decoderID).

**Pitfalls**
- Pixel shuffle channel ordering: `nn.PixelShuffle` expects `[C·r², H, W]` in a specific
  interleave. Get it wrong and you'll see a checkerboard artefact — which is a good visual
  debug signal.
- Odd sizes + transposed conv → off-by-one. Phase 12 fixes this properly; until then, pad
  to a multiple of 64 and crop.

---

## Phase 8 — Variable rate: gain unit, 3D gain unit, quality map, RoI

**Objective:** §VI-I, eqs 11–15. Also the mechanism that saves you 4× the training compute.

**Build**
1. **`mref[modelID, comp, c]`** — a learned per-channel gain vector per component. Store as
   12-bit signed ints as the paper specifies.
2. **`betaDisplacementLog[comp]`** — the rate knob, per component, in the picture header.
   Independent luma/chroma quality control; expose both on the CLI.
3. **Decoder-side (eqs 13–15):**
   ```
   mlog = betaDisplacementLog[comp] + mref[modelID, comp, c] (+ Gain3d[i,j])
   minv = exp(-mlog * step / 2**sigmaPrecision)        # LUT this, per the paper
   r̂   *= minv
   Iσ  += mlog
   ```
   Encoder scales the unquantised residual by `1/minv` before rounding. Verify the sign
   convention: larger `betaDisplacementLog` → higher rate and higher quality.
4. **Variable-rate training:** sample `betaDisplacementLog` uniformly from your target range
   each iteration (or each batch element). The rate term automatically tracks it because
   `Iσ` shifts. **One training run, whole RD curve.**
5. **3D gain unit + quality map:** `Gain3d[i,j]` derived from a spatial quality map, coded in
   an `SOQ` segment. Implement the map as a coarse grid (say, one value per 8×8 latent
   block), delta-coded and entropy coded. Then:
   - `--roi mask.png --roi-boost +4` on the encoder
   - a **saliency-driven automatic mode**: use a cheap saliency or face detector to build
     the map (this is a great demo — "spend bits on the faces")
6. **LUT check:** implement `exp()` as a lookup table and assert it matches the float
   version to within your fixed-point precision.

**Acceptance test**
- One trained model + a `betaDisplacementLog` sweep produces a smooth RD curve spanning
  ≥ 10× bitrate range. (Paper claims ~20× across its four *models*; from one model you
  should get a good fraction of that. Report what you actually achieve.)
- Compare against 3–4 separately-trained fixed-λ models on a subset: quantify the
  variable-rate penalty in BD-rate. **This is a real, publishable-quality experiment** and
  the paper does not give this number.
- RoI: mask region visibly sharper at the same total file size; measure PSNR inside vs
  outside the mask.
- Independent luma/chroma: `--beta-luma 0 --beta-chroma -8` visibly degrades colour only.

---

## Phase 9 — me-tANS, skip mode, and the real codestream

**Objective:** replace the research range coder with the standard's actual entropy coder
(§VI-C, Algorithm 1), and produce real files with real headers (§V).

This is the phase that turns "a PyTorch experiment" into "a codec". Budget generously.

**Build**

### 9a. tANS core
1. Quantise your `Iσ` range into **32 σ-classes** and build a `[32, 256]` CDF table (the
   paper's dimensions). Discretised zero-mean Gaussian per class, plus an escape symbol.
2. Build the four tANS tables per class: `transition_table_symbol`,
   `transition_table_nBits`, `transition_table_stateNext`, `bound_table`. Read Yann Collet's
   FSE articles for the construction; the spread function and table size (`2^tableLog`,
   typically 2^10–2^12) are the design choices.
3. **Decoder = Algorithm 1 verbatim.** FILO: pointer starts at the *last* symbol and moves
   backwards. Fast path: table read → symbol, read `nBits`, `state = stateNext | value`.
4. **Escape / outbound coding:** if `r̂ + bound_table[Iσ] == 0`, read 1 flag bit → field
   width 2 or 15 bits → value → sign bit. Implement exactly this.
5. **Encoder** runs backwards relative to the decoder. Write it as the mirror of Algorithm 1
   and fuzz-test.
6. **Dual-state interleaving** even single-threaded, as the paper describes — two ANS states
   over two substreams "mimicking a dual-threaded setup". Measure the speedup; it's a nice
   result.
7. Hyper samples: separate table set, `[128, 64]`, selected by **channel index** not σ.

### 9b. Skip mode
- `if Iσ[c,i,j] < skip_threshold: r̂ = 0`, code nothing.
- Report the **skip ratio** per image. The paper says up to 80%; see what you get.
- **Cube-based skip override:** partition `r̂` into **16×16×16** cubes; one flag per cube can
  revert skipping in that cube. Encoder decides by measuring distortion with/without.
- Add a visual demo: skip-mask overlay on the image. Beautiful slide material.

### 9c. Codestream
Implement the byte-exact container from §V and Table II:

```
SOC(ff80) PIH(ff82) [TOH(ff83)] [SOQ(ff8b)] [RDI(ff84)] [UDI(ff8c)]
          SOZ(ff88) SORp(ff89)+ SORs(ff8a)+ EOC(ff81)
```
- 2-byte marker, variable-length size field, payload; **byte-align after the size field and
  after the payload**.
- **PIH:** profile/level, picture size, output bit depth, internal + output subsampling
  (`c_*`, `s_*`), `modelID`, `encoderID`?, `decoderID`, colour-transform idx (+ matrix/bias),
  region/tile params, multithreading substream counts, RVS flags (`grfs_enable_flag`,
  `rvs_enable_flag`, `GRFS_Y[c]`), `betaDisplacementLog[2]`, skip params,
  `diff_display_img_width/height`.
- **TOH:** `lsbs_enable_flag[2]`, filters header (4 enable flags + per-filter control data).
  **Absent TOH ⇒ all flags 0.**
- Multiple `SORp`/`SORs` with distinct `region_idx` (feeds Phase 12).
- Substream offsets at segment start; counts in PIH.
- A **`jpegai inspect file.jpegai`** command that dumps every marker, segment size and
  syntax element as a tree. You will use this constantly, and it demos beautifully.

**Acceptance test**
- me-tANS output size within **0.5%** of your Phase 3 range coder on the same symbols and
  the same model. If it's worse, your table construction is lossy.
- me-tANS decode throughput ≥ 3× the range coder on the same machine. Measure and report
  Msymbols/s.
- Fuzz: 10⁶ random symbol streams, encode→decode, assert exact equality.
- Table memory footprint measured and reported against the paper's ~100 KB.
- `jpegai inspect` round-trips: parse your own file, re-serialise, get identical bytes.
- **Truncate the file after `SOZ` and confirm the decoder produces a sane (if bad) image** —
  that's Phase 12's progressive decoding pre-validated.

**Pitfalls**
- **FILO is the #1 source of bugs.** Write the encoder as literally reversed, and build the
  round-trip test *first*.
- Symbol clamping: your Gaussian CDF must cover the full symbol alphabet you can produce, or
  the escape path must catch everything else. An uncaught out-of-range symbol corrupts the
  entire rest of the stream (and will look like "the decoder works on 90% of images").
- Byte alignment: forget one and every downstream segment is misparsed.

---

## Phase 10 — RVS, LSBS, and the four post-processing filters

**Objective:** §VI-G, VI-H, VI-M. These are the switchable tools of Table IV, and
implementing them gives you a real, reproducible ablation table.

**Build**

### RVS (worth 2.2% in the paper)
1. Pooling (eq. 7): `σ = (32 + Σ_{8×8} Iσ) >> 6`, padding value **1411**.
2. Tables `T1[modelID, id, 3968]`, `T2[modelID, id, 3968]`. You don't have the normative
   values, so **learn them**: make them trainable parameters (or a tiny MLP over
   `(id, σ_bucket)` that you then bake into a table), trained with the *seven-metric*
   objective — or at least with FSIM + VMAF + NLPD added to the loss, since that's what RVS
   exists to fix. Freeze the rest of the codec and train only `T1`/`T2`. This is fast, and it
   *is* the paper's own logic.
3. `id[c] = GRFS_Y[c] + 2·rvs_enable_flag[comp]`, `GRFS_Y[c] ∈ {0,1}` per-channel flags in
   the PIH. Encoder chooses `GRFS_Y[c]` by trying both and keeping the better.
4. Eqs 8, 9: `Iσ += T1[...]`, `r̂ = r̂ · T2[...] / 2¹⁶` (16-bit fixed point).

### LSBS (worth 0.4%)
Eq. 10, gated by `lsbs_enable_flag[comp]` in the TOH:
```
μ = ŷ − r̂
ŷ += (r̂·TR[modelID, σ] + μ·TP[modelID, σ] + 2¹²) >> 13
```
`TP[4,3968]`, `TR[4,3968]`, learned the same way. Semantically: a σ-dependent reweighting of
prediction-vs-residual trust.

### Post-filters
| Filter | Scope | Suggested realisation |
|---|---|---|
| **EFE linear** | secondary | small linear (conv-only) filter, coefficients signalled in TOH |
| **EFE nonlinear** | secondary | small CNN with ReLU, few kernels |
| **ICCI** | both | cross-component: refine chroma using luma edges *and* luma using chroma |
| **LEF** | primary | edge-aware sharpening on luma, e.g. gradient-gated conv |
All four **optional at decode time even when enabled in the syntax** — implement that
skip-anyway semantics explicitly, it's a stated requirement.

**Acceptance test — this produces your version of Table IV:**

| TEST | your AVG BD-rate | paper |
|---|---|---|
| all on | — | −20.2% |
| RVS off | — | −18.0% (Δ 2.2) |
| LSBS off | — | −19.8% (Δ 0.4) |
| LEF off | — | −19.9% (Δ 0.3) |
| ICCI off | — | −20.0% (Δ 0.2) |
| EFE nonlin off | — | −20.4% (Δ −0.2) |
| EFE lin off | — | −20.4% (Δ −0.2) |

Also reproduce the *qualitative* findings:
- RVS gains should concentrate in **FSIM and VMAF** (paper: 6.7 and 6.1 points) and barely
  move MS-SSIM.
- EFE filters should **slightly hurt** the 7-metric average while improving **chroma PSNR**
  substantially (paper: +12% / +8%). If you reproduce that inversion, you've reproduced the
  paper's most interesting engineering observation.
- ICCI should be the only tool with real MAC cost (paper: 4.6 of 28 kMAC/pxl).

---

## Phase 11 — Integer entropy path and cross-device bit-exactness

**Objective:** §III's central claim. This is the phase that makes your project about a
*standard* rather than a model, and almost nobody attempting this will do it.

**Build**
1. **Quantise the hyper scale decoder to integer arithmetic**: 8-bit weights, 8-bit
   activations, **32-bit accumulators**. Per-tensor or per-channel scales, power-of-two
   requantisation shifts so requantisation is a shift not a divide. Write it as an explicit
   integer reference implementation (numpy int32) *separate from* the PyTorch model — this is
   what the standard would specify.
2. **Overflow audit.** For every layer compute the worst-case accumulator magnitude
   (`Σ |w| · max|a|`) and assert `< 2³¹`. The paper cites [39] for a proof that 32-bit
   registers never overflow; do the arithmetic version of that proof for your network and
   put the table in your report.
3. **Quantisation-aware finetune** so the int model's RD ≈ the float model's.
4. **Bit-exactness harness:**
   ```
   for device in [cpu, mps, cpu-float64, another-machine]:
       r̂, ẑ = entropy_decode(codestream, device)
       assert hash(r̂) == reference_hash and hash(ẑ) == reference_hash
   ```
   Then, immediately after, show that **`x̂` differs slightly across devices and that this is
   fine** — measure the PSNR between CPU and MPS reconstructions (should be > 60 dB) and
   state the conformance rule: bit-exact through entropy decoding, tolerant thereafter.
5. **The killer demo:** deliberately perturb `Iσ` by ±1 at one position and show the
   decoded image turning into garbage from that point on. That single before/after pair
   *explains* why the conformance point sits where it does, better than any paragraph.

**Acceptance test**
- Integer and float entropy paths give identical `r̂` on your whole test set.
- Int quantisation costs < 0.5% BD-rate after finetuning.
- CPU vs MPS: `r̂` identical (byte-hash equal), `x̂` different but > 60 dB apart.
- Overflow-bound table for every layer, all margins > 2×.

---

## Phase 12 — Tiling, region partitioning, arbitrary sizes, progressive decoding

**Objective:** §VI-J, VI-K, VI-L. The functionality features. Individually small, and each
is a demo.

**Build**
1. **Layer-based cropping (§VI-K, mechanism 1).** A padding layer before each of the 4
   analysis downsampling stages and each of the 2 hyper-encoder stages; a matching cropping
   layer after each corresponding upsampling stage in synthesis / hyper decoder / hyper
   scale decoder. Record the per-stage pad amounts (they're derivable from the image size, so
   nothing is signalled). Test on prime-number image sizes: **997×661**.
2. **Display window (§VI-K, mechanism 2).** Alternative path: pad the input to a multiple of
   **64**, encode, and crop the reconstruction using `diff_display_img_width/height`.
   Implement both and **measure the difference** — rate overhead of the padded version vs the
   static-shape benefit. The paper's justification is that dynamic intermediate shapes break
   static-graph NPU compilers; you can demonstrate this concretely by exporting both to ONNX
   and showing one has dynamic axes and one doesn't. Excellent, concrete result.
3. **Synthesis transform tiling.** Tile `ŷ` into blocks; run synthesis per tile with a
   halo/overlap region to avoid seams; blend or crop. Measure **peak memory** vs tile size
   and plot it. Add `--tile-size`. Show a 4K image decoded in bounded memory.
4. **Region partitioning.** Split `r_Y-stream`/`r_UV-stream` into multiple `SORp`/`SORs`
   segments with distinct `region_idx`. With
   `region_residual_in_its_own_substream_flag = 1`, enforce both normative constraints:
   each region contains an integer number of synthesis tiles, and MCM/synthesis must not
   read across region boundaries. Then implement **`jpegai decode --crop x,y,w,h`** that
   parses the headers, decodes *only* the intersecting regions, and never touches the rest
   of the file. **Report the byte count actually read and the time saved.** This is a
   genuinely impressive demo: random access into a compressed image.
5. **Progressive decoding.** Zero part of `r̂` and decode. Implement three drop strategies
   and compare: (a) drop by MCM stage (decode stage 1 only, then 1–2, …) — the natural fit;
   (b) drop by channel importance; (c) drop by spatial region. Produce an animated GIF of the
   image improving. Also implement the *truncated-file* case: decode a file cut short after
   `SOZ`.

**Acceptance test**
- 997×661, 1×1, 4096×2160, 1-pixel-wide, and 4:2:0-with-odd-dimensions all round-trip.
- Tiled vs untiled synthesis: PSNR difference < 0.01 dB (halo big enough), peak memory
  reduced measurably.
- `--crop` reads < 30% of the file for a 25% crop and produces the same pixels as a full
  decode of that area (with independent regions).
- Progressive GIF, and a rate-vs-quality curve for the truncation points.

---

## Phase 13 — Evaluation, ablation, and the benchmark report

**Objective:** the numbers you show your professor. Automated, reproducible, honest.

**Build** one command, `jpegai bench --config config/full.yaml`, that produces:

1. **RD curves** on Kodak, DIV2K-valid, CLIC-2024 (if available), plus a small
   synthetic/screen set — one plot per metric (7 + PSNR-Y/U/V), your codec's three decoders
   plus every anchor.
2. **Your Table III:** BD-rate per metric, for the 2×3 grid of (encoderID, decoderID),
   against your chosen anchor, plus **kMAC/pxl** (count them properly, per operating point)
   and **ms/megapixel** (decode time, broken down: entropy decode / latent reconstruction /
   synthesis / post-filters — the paper's Part 3 profiling tooling does exactly this).
3. **Your Table IV:** the tool-off ablation from Phase 10, plus these extra ablations that
   the paper *doesn't* publish and that make your work original:
   - split vs fused hyper decoder (cost of the decoupling)
   - MCM 1 / 2 / 4 stages
   - variable-rate (gain) vs separately-trained fixed-λ models
   - integer vs float entropy path
   - two-branch YCbCr vs single-branch RGB
   - skip mode on/off, and skip ratio vs bitrate
   - layer-based cropping vs pad-to-64 display window
4. **Your Tables V/VI:** cross-dataset generalisation. Expect the paper's pattern — worse on
   Kodak (small images) than on the 1K–4K set.
5. **Subjective comparison sheets** (paper §VII-D / Fig. 3): at ~0.08 bpp and ~0.3 bpp, with
   post-filters **disabled**, side-by-side crops of your codec (BOP, HOP) vs AVIF vs
   original. Include a screen-content image, and be upfront if text degrades — the paper is.
6. **Complexity/quality Pareto plot**: kMAC/pxl on x, BD-rate on y, your three operating
   points plus baselines. The paper's 14/28/215 → −16.2/−20.2/−22.1 is a steep curve; ask
   the question the paper doesn't: what sits at 50–80 kMAC/pxl? **Train a fourth,
   intermediate synthesis head and answer it.** That's a genuine research contribution, and
   it's cheap because everything else is already built.

**Acceptance test:** the whole report regenerates from a clean checkout with one command,
and every number in your written report is traceable to a JSON produced by it. Fix random
seeds. Log versions.

**Honesty requirements** (state these explicitly in the report):
- Tier A is reduced-width and trained on far less compute than the VM, so absolute BD-rate
  will fall short of the paper's. **Report the gap and explain it.**
- VIF and PSNR-HVS will likely be positive (worse than anchor). That matches the paper.
  Explain why (perceptual training objective).
- If you use AVIF rather than VTM-11.1 as anchor, say so and don't compare your percentages
  directly to the paper's.

---

## Phase 14 — Demo, CLI, report, slides

**Objective:** the thing your professor actually sees.

### CLI
```bash
jpegai encode in.png out.jpegai --model 0 --encoder-id 0 \
       --beta-luma 0 --beta-chroma 0 --internal-format 420 \
       --roi mask.png --roi-boost 4 --tiles 512 --regions 2x2 --tools rvs,lsbs,lef,icci
jpegai decode out.jpegai rec.png --decoder-id 1 [--luma-only] [--crop 100,100,512,512]
jpegai inspect out.jpegai              # marker/segment/syntax tree
jpegai bench  --config config/full.yaml
jpegai conform out.jpegai              # bit-exactness harness across devices
```

### Interactive demo (Streamlit or Gradio)
- Upload an image; sliders for `betaDisplacementLog` (luma and chroma separately).
- **Radio for decoderID 0/1/2 decoding the same codestream** — with file size shown *once*,
  and quality/time changing. This is the single clearest way to show what multi-branch
  decoding means, and it lands in five seconds.
- Paint an RoI mask on the canvas → re-encode at the same total size → show the boost.
- **Progressive decode animation** (Phase 12).
- **Skip-mask overlay** and skip-% readout (Phase 9).
- **Latent explorer:** visualise `ŷ` channels, `Iσ` as a heat map (this *is* the local
  bit-allocation map, and it's visually striking), `r̂` sparsity, and the 4 MCM stage masks.
- **Random-access panel:** crop selector, showing bytes read vs total.
- **Bit-exactness panel:** hash of `r̂` on CPU vs MPS (equal), PSNR of `x̂` CPU vs MPS
  (not equal, > 60 dB) — with the ±1 `Iσ` perturbation demo next to it.
- Live metric table vs AVIF/JPEG at matched bitrate.

### Written report (~20–30 pages)
1. Introduction and standardisation context (§I–II).
2. Background: learned image compression (doc 01).
3. The JPEG AI design: architecture, the three design decisions (§III–IV).
4. Codestream and syntax (§V).
5. Tools (§VI), each with your implementation notes and deviations.
6. **A table mapping every paper section → your module → your test.** Professors love this
   table; it proves coverage at a glance.
7. Experimental results (Phase 13), with the honesty section.
8. What I could not reproduce and why (compute, missing normative tables, missing supplement).
9. Conclusions and future work (§XII–XIII, plus your own).

### Slides (~15)
1 the problem · 2 classical vs learned · 3 rate–distortion in one equation · 4 JPEG AI in
the standards landscape · 5 architecture diagram · 6 **the three design decisions** ·
7 MCM with the parallel-lines timing plot · 8 me-tANS and skip mode · 9 variable rate + RoI
demo · 10 the three operating points, one file (live) · 11 bit-exactness + the ±1 garbage
demo · 12 results: RD curves and your Table III · 13 ablation: your Table IV · 14 limits and
honesty · 15 what I'd do next.

---

## Track B (optional) — a real conformance decoder

Attempt only after Phase 10, and only if you obtain:
1. **ITU-T T.840-1** (free from itu.int if available) — for exact layer configurations, CDF
   tables, `T1`/`T2`/`TP`/`TR` tables, and full syntax.
2. **The four ONNX model parameter sets** (link is inside Part 1).
3. **Part 4 conformance codestreams** (target finalisation Oct 2025).
4. Optionally **VM 7.0** from `gitlab.com/wg1/jpeg-ai/jpeg-ai-vm` to generate your own test
   streams.

Then: parse a real codestream with your Phase 9 reader, run the real ONNX graphs via
`onnxruntime`, and check your reconstruction against the reference. **If you can decode one
official conformance codestream correctly, that alone is a stronger result than everything
in Track A**, because it means you implemented a published international standard. Treat it
as the stretch goal it is, not the plan.

---

## Track C (background) — full-width training

From Phase 8 onward, whenever the architecture is stable, push a training job to a free
GPU (Colab / Kaggle T4/P100, or your university's cluster if you can get an account) at the
paper's real widths (160/96 primary/secondary channels). Checkpoint to Drive every ~30
minutes so preemption doesn't cost you. Your Tier A code should be width-parameterised from
day one (`config/tierA.yaml`, `config/full.yaml`) so this is a config change, not a port.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Training doesn't converge | high, at least once | Phase 3 gate; copy CompressAI's LR schedule (1e-4, drop to 1e-5 for last 10%); clip gradients; check for NaN in the σ path (clamp log σ) |
| Encoder/decoder latent mismatch in MCM | high | Assert exact tensor equality in CI from Phase 6 onward; never let it regress |
| me-tANS bugs | high | Round-trip fuzz test *before* integrating; keep the range coder as a switchable fallback |
| Not enough compute for good numbers | certain | Tier A + gain-unit variable rate; be explicit in the report; Track C in background |
| Missing normative tables (`T1`,`T2`,`TP`,`TR`, CDFs) | certain unless you get Part 1 | Learn them; document clearly as a deviation |
| Missing supplement (hyper decoder / MCM / synthesis layer configs) | certain unless you get it from Xplore | Design your own at Tier A width; document as a deviation |
| Scope creep | high | The Phase 6 checkpoint exists for this. Ship Phase 6 quality, then extend |
| MPS op gaps / silent CPU fallback | medium | `PYTORCH_ENABLE_MPS_FALLBACK=1`, profile per-op early, keep a CPU-only correctness path |

---

## What to obtain, in priority order

1. **ITU-T T.840-1** (= ISO/IEC 6048-1), free on itu.int if published there. Unlocks
   everything normative.
2. **The paper's supplementary material** from IEEE Xplore
   (DOI 10.1109/TCSVT.2025.3613244) — Appendices A/B/C and Figs. 6/7/8 contain the exact
   network configurations for the hyper decoder, hyper scale decoder, MCM stages, and all
   three synthesis transforms. **This is your single biggest missing piece.**
3. **Reference [42]** — Jia et al., "Overview of variable rate coding in JPEG AI", TCSVT
   35(9), 2025. Everything about gain units, quality maps, and an example encoder rate
   control.
4. **Reference [40]** — Zhang et al., "End-to-end learning-based image compression with a
   decoupled framework", TCSVT 34(5), 2024. Explains the split hyper decoder.
5. **VM 7.0** — `gitlab.com/wg1/jpeg-ai/jpeg-ai-vm`. Even reading it is instructive; running
   it gives you ground truth.
6. **The four ONNX models** (link inside Part 1) and Part 4 conformance streams — the gate
   for Track B.

Ask your professor whether the university has IEEE Xplore and ISO access. Item 2 in
particular is behind Xplore and is the difference between guessing network configurations
and knowing them.

---

## If you have only four weeks

Do Phases 1, 2, 3, 4, 5, 6 and a cut-down 14 (CLI + Streamlit with the decoderID switch
stubbed to two heads + a short report). That is still a two-branch YCbCr hyperprior codec
with residual coding, split hyper decoders and a 4-stage parallel context model, measured on
seven metrics against AVIF with BD-rate — comfortably a good project. Then add phases in this
order as time allows: **7 (three decoders) → 8 (variable rate + RoI) → 9 (me-tANS +
codestream) → 12 (progressive + random access, cheap demos) → 10 (tools) → 11 (bit-exact) →
13 (full evaluation)**.
