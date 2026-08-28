# 01 — Foundations: What You Must Understand Before the Paper Makes Sense

The paper assumes you already know learned image compression. You probably don't yet, and
the paper never explains it — it is an *overview of a standard*, written for people who
already sat in the JPEG committee. This chapter is the missing prerequisite. Nothing here
is from the paper; it is the surrounding literature that JPEG AI is built on.

---

## 1. The problem: rate–distortion optimisation

Compression is a two-objective problem. You want the file small (**rate**, R, in bits per
pixel) and the decoded image close to the original (**distortion**, D). You cannot have
both, so you minimise a weighted sum:

```
L = R + λ·D          (or equivalently   L = λ_rate·R + D)
```

λ is the knob. Small λ → cares about rate → small files, ugly images. Large λ → cares
about distortion → big files, pretty images. Sweeping λ traces out the **RD curve**, and
"codec A is better than codec B" means A's RD curve sits below-and-right of B's.

Classical codecs (JPEG, JPEG 2000, HEVC, VVC) hand-design every stage: a fixed transform
(DCT, wavelet), a hand-tuned quantiser, a hand-tuned entropy coder. Learned codecs
replace all three with neural networks and **optimise them jointly against L by gradient
descent**. That is the entire idea.

### How rate becomes differentiable

You cannot backpropagate through "length of the output file". The trick: rate is the
expected code length, and by Shannon, the optimal code length of a symbol under model `p`
is `−log₂ p(symbol)`. So:

```
R = E[ −log₂ p_model(ŷ) ]
```

If `p_model` is a neural network, R *is* differentiable. Train the entropy model to
predict the latents well, and the true file size follows, because a real entropy coder
(arithmetic / ANS) achieves within a fraction of a bit of `−log₂ p`.

**This is the single most important idea in the field.** The entropy model is not a
post-processing step bolted on at the end; it is a *trained component whose loss is the
bitrate*.

---

## 2. The transform-coding autoencoder (Ballé et al., ICLR 2017)

Reference: J. Ballé, V. Laparra, E. Simoncelli, "End-to-end optimized image compression"
— this is paper [9] in your paper's bibliography.

```
x ──[ g_a : analysis transform ]──► y ──[ Q ]──► ŷ ──[ g_s : synthesis transform ]──► x̂
                                     │
                                     └──► entropy model p(ŷ) ──► rate
```

- `g_a` (**analysis transform** — the encoder-side CNN) is a stack of strided
  convolutions. Each stride-2 conv halves H and W while growing the channel count. It
  plays the role the DCT plays in JPEG: decorrelate and compact energy.
- `y` is the **latent representation**: a `[C, H/16, W/16]` tensor of floats.
- `Q` quantises `y` to integers `ŷ` (usually just rounding).
- `g_s` (**synthesis transform** — the decoder-side CNN) mirrors `g_a` with upsampling
  layers and reconstructs `x̂`.

The vocabulary matters because JPEG AI uses exactly these words. When the paper says
"primary component analysis transform" it means "the luma encoder CNN".

### Why 16× downsampling and not 8× or 32×?

A 4:1 spatial reduction per axis per stage is cheap; four stages give /16. At /16 with
160 channels you have `160/(16·16) = 0.625` latent values per pixel — a 1.6× spatial
compaction before the entropy coder even starts. Empirically 4 stages is the sweet spot:
3 stages leaves too much spatial redundancy for the entropy model to mop up, 5 stages
loses too much detail to recover. JPEG AI uses 4.

---

## 3. The quantisation problem and the straight-through estimator

`round()` has zero gradient almost everywhere, so it kills backpropagation. Three standard
fixes:

1. **Additive uniform noise** (Ballé 2017). During training, replace `ŷ = round(y)` with
   `ỹ = y + u`, `u ~ U(−0.5, 0.5)`. This makes the model a genuine VAE with a uniform
   posterior, and the rate term becomes a proper differential entropy. Clean theory, but
   train/test mismatch: at test time you round, not add noise.
2. **Straight-through estimator (STE)**. Forward pass rounds; backward pass pretends the
   round was identity: `ŷ = y + (round(y) − y).detach()`. No mismatch, biased gradient.
3. **Mixed / two-phase**. Noise for the rate branch (keeps the entropy model honest), STE
   for the distortion branch (keeps the synthesis transform honest). This is what most
   modern implementations, and what you should do, use.

JPEG AI's quantisation is plain rounding of a *residual* (see §5), which makes STE
directly applicable.

---

## 4. Entropy models, in increasing sophistication

This is the axis along which the whole field has progressed.

### 4a. Fully factorised prior

`p(ŷ) = Π_i p_i(ŷ_i)`, one learned 1-D distribution per channel, shared across all spatial
positions. Cheap, parallel, and weak: it cannot exploit the fact that neighbouring latents
are correlated.

### 4b. Scale hyperprior (Ballé et al., ICLR 2018) ← **JPEG AI's backbone**

Reference [10] in your paper. The key observation: the *scale* (standard deviation) of a
latent varies enormously across the image — flat sky regions have tiny latents, textured
regions have large ones. A single fixed distribution per channel must be wide enough for
the worst case, which wastes bits everywhere else.

Fix: send **side information** describing the local scale.

```
y ──[ h_a : hyper encoder ]──► z ──[ Q ]──► ẑ ──[ h_s : hyper decoder ]──► σ
                                            │
                                            └── coded with a fixed factorised prior
ŷ coded with  p(ŷ | σ) = N(0, σ²)  discretised
```

`z` is a second, much smaller latent (another /4 spatial reduction → /64 of the image). It
costs very few bits (a few percent of the total) but lets you model `ŷ` with a
*per-sample* Gaussian. This is called a **hyperprior** because it is a prior over the
parameters of a prior. It is a two-level latent-variable model, i.e. a hierarchical VAE.

The name "variational autoencoder with hyperprior" in the paper's abstract refers exactly
to this architecture.

**Where JPEG AI's tensors come from:**

| Ballé 2018 | JPEG AI name | JPEG AI tensor |
|---|---|---|
| `g_a` | analysis transform | produces `y_Y`, `y_UV` |
| `y` | latent | `y_Y [160, H/16, W/16]` |
| `h_a` | hyper encoder | produces `z` |
| `ẑ` | quantised hyper tensor | `ẑ_Y`, `ẑ_UV`, at /64 |
| `h_s` → σ | **hyper scale decoder** | `Iσ_Y`, `Iσ_UV` (log-domain) |
| — (new) | **hyper decoder** | `p̈_Y`, `p̈_UV` (the *mean* prediction) |
| `g_s` | synthesis transform | produces `x̂` |

Note JPEG AI splits Ballé's single `h_s` into **two** networks: one producing the scale
(σ) and one producing the mean (p̈). That split is not cosmetic — see §7 of the paper
explanation doc; it is the "decoupling" that lets the entropy stage run on CPU while the
rest runs on the NPU.

### 4c. Mean-and-scale hyperprior (Minnen et al., NeurIPS 2018)

Predict both `μ` and `σ`, and code `ŷ ~ N(μ, σ²)`. Equivalently: predict `μ`, code the
**residual** `r = ŷ − μ`, and code `r ~ N(0, σ²)`. JPEG AI does exactly this — and it is
why the paper talks about `r̂_Y`/`r̂_UV` ("quantized residual tensors") rather than coding
the latents directly. Equation (1) and (2) of the paper are literally
`r̂ = round(y − p̈)` and `ŷ = r̂ + p̈`.

### 4d. Autoregressive context model (Minnen 2018)

Also condition on already-decoded neighbours: `p(ŷ_i | ŷ_{<i}, ẑ)` via a masked
convolution. Gains ~5–10% BD-rate. **Fatal flaw:** decoding becomes sequential per
*sample*. A 4K latent tensor has ~30 million samples; a per-sample neural-network forward
pass is unshippable. This is why no product used learned codecs for years.

### 4e. Checkerboard / multi-stage context (He et al., CVPR 2021) ← **JPEG AI's MCM**

Reference: "Checkerboard Context Model for Efficient Learned Image Compression". Instead
of a raster-scan autoregression over N samples (N sequential steps), split the latent
grid into K interleaved groups. Decode group 1 conditioned only on the hyperprior; decode
group 2 conditioned on the hyperprior *and all of group 1*; and so on. Within a group,
every sample is independent, so each stage is one fully-parallel network pass.

- 2 groups (classic checkerboard) → 2 passes, recovers most of the context gain.
- **4 groups (2×2 pattern) → 4 passes.** This is JPEG AI's **MCM (Multi-stage Context
  Modeling)** and Fig. 2 of the paper is precisely this 2×2 grouping.

So JPEG AI gets spatial-context coding gain at a fixed cost of 4 network passes,
independent of image size. The paper calls this out as "a key distinction of JPEG AI from
many other state-of-the-art learning-based image codecs" (§VI-D) — most competitors
dropped context models entirely to stay fast.

### 4f. What came after (and why JPEG AI v1 doesn't use it)

- **Channel-wise autoregression** (Minnen & Singh 2020): split channels into ~10 slices,
  autoregress across slices. Strong, but 10 sequential passes.
- **ELIC** (He et al., CVPR 2022): uneven channel grouping + spatial checkerboard.
  State-of-the-art RD for a while.
- **Transformer / Swin-based transforms** (SwinT-ChARM, TIC, Contextformer).
- **Generative / perceptual codecs** (HiFiC, and diffusion-based decoders): astonishing
  at very low bitrate, but they *hallucinate* detail. Unacceptable for a general-purpose
  standard where fidelity claims must hold.
- **Implicit neural / online-overfitted** (COIN, C3): encode by overfitting a tiny network
  per image. Encoding takes minutes.

The paper's §XIII names diffusion, transformers and implicit/online-training as the things
to watch for JPEG AI v2. v1 deliberately stopped at a hyperprior + 4-stage checkerboard
because everything past that either breaks the complexity budget or breaks fidelity.

---

## 5. Perceptual metrics — and why there are seven of them

PSNR/MSE is a terrible predictor of human judgement, and learned codecs exploit exactly
the gap: optimise pure MSE and you get blurry-but-high-PSNR images; optimise a perceptual
loss and PSNR drops while humans prefer the result. JPEG AI's committee therefore scored
proposals on **seven** metrics:

| Metric | Family | What it captures |
|---|---|---|
| **MS-SSIM** | structural, multiscale | luminance/contrast/structure similarity across scales |
| **VIF** | information-theoretic | mutual information through an HVS channel |
| **FSIM** | feature-based | phase congruency + gradient magnitude |
| **VMAF** | learned fusion (Netflix) | SVM/ensemble over VIF+DLM+motion features |
| **NLPD** | normalised Laplacian pyramid | divisive-normalisation model of early vision |
| **PSNR-HVS-M** | DCT-domain masked PSNR | between-coefficient contrast masking |
| **IW-SSIM** | information-weighted SSIM | SSIM weighted by local information content |

Critically, JPEG AI **trains** on a weighted MSE + MS-SSIM combination but is **evaluated**
on all seven. That mismatch is not an oversight — it is the entire justification for the
RVS tool (§VI-G of the paper), which is a decoder-side correction that buys back BD-rate
on VMAF/FSIM/NLPD specifically *because* those were not in the training loss.

Notice in Table III: JPEG AI is hugely better on MS-SSIM (−34.8%), IW-SSIM, FSIM, VMAF,
NLPD — and roughly a wash or worse on **VIF** and **PSNR-HVS** (both positive = worse than
VVC at decoderID 0). That is the fingerprint of a perceptually-trained codec. Be honest
about this when you present.

### BD-rate (Bjøntegaard Delta rate)

The standard way to reduce two RD curves to one number: fit a cubic through
(log rate, quality) points for each codec, integrate the difference over the overlapping
quality range, and report the average % bitrate change at equal quality. **Negative =
better** (you need fewer bits for the same quality). Every number in Tables III–VI is a
BD-rate against VVC Intra (VTM-11.1) as the anchor.

---

## 6. Entropy coders: arithmetic coding vs ANS

The entropy *model* gives you probabilities; the entropy *coder* turns symbols +
probabilities into bits.

- **Arithmetic coding (AC)**: maintains a shrinking interval `[low, high)`. Optimal to
  within ~2 bits total. Per-symbol cost: a couple of multiplies/divides plus
  renormalisation. Inherently sequential.
- **Range coding**: byte-oriented AC. What CompressAI and most research code use.
- **ANS / rANS / tANS** (Duda, 2013): encodes into a single integer state; decoding is a
  table lookup plus a shift/OR. Same compression as AC, but **much** faster —
  especially **tANS** (table-driven ANS), where the entire state machine is precomputed
  into lookup tables, so decoding a symbol is ~2 ALU ops and one table read. Cost: the
  tables. If you need a different table per probability distribution and you have
  thousands of distributions, tables explode.

JPEG AI's answer is **me-tANS** ("memory-efficient tabulated ANS"): quantise σ into a
small number of classes (the residual CDF table is `[32, 256]` — 32 distributions × 256
symbols), so you only need ~32 transition tables, totalling **~100 KB**. That is
cache-resident. This is why JPEG AI can claim CPU-side entropy decoding at high
throughput.

ANS decodes in **FILO** order — last symbol encoded is first decoded. Algorithm 1 in the
paper says "move the pointer in the bitstream to the last symbol position; the pointer
moves backwards". Your encoder must therefore run backwards relative to the decoder, or
buffer and reverse. This trips up everyone implementing ANS the first time.

---

## 7. Why colour is handled the way it is

The human visual system has far more luminance resolution than chrominance resolution.
Every practical codec exploits this: convert RGB → YCbCr, subsample chroma
(4:2:0 = quarter-resolution chroma), spend most bits on luma.

- **4:4:4** — no chroma subsampling
- **4:2:2** — chroma halved horizontally
- **4:2:0** — chroma halved both ways (the common case)

JPEG AI adds a second motivation: **complexity**. Running one big joint network over all
three components is more expensive than running a big network on luma and a small one on
chroma. So JPEG AI splits into a **primary** (luma) branch and a **secondary** (chroma)
branch, and lets them talk at exactly two points — once at the start of analysis, once at
the start of synthesis. Everything else is independent. That costs some coding gain
(luma and chroma *are* correlated) but buys a monochrome fast path: for computer-vision
consumers who only need luma, you skip the chroma branch entirely.

---

## 8. Standards vocabulary you need

Learned-compression papers and standards documents use different words for the same thing.

| Term | Meaning |
|---|---|
| **Normative** | You must do it exactly this way to be compliant. |
| **Informative** | Explanation/example; not binding. |
| **Codestream** | The compressed bytes. (JPEG-family word for "bitstream".) |
| **Syntax element** | A named field in the codestream. |
| **Marker** | 2-byte tag identifying a codestream segment. JPEG uses `0xFFxx`. |
| **Profile** | A subset of the syntax/tools an implementation must support. |
| **Level** | Numeric limits (max resolution, max throughput) within a profile. |
| **Conformance** | A test suite; pass it and you may claim compliance. |
| **Verification Model (VM)** | The committee's reference software. |
| **CTTC** | Common Training and Test Conditions — the agreed experimental protocol. |
| **CfP** | Call for Proposals. |
| **Anchor** | The baseline codec you measure against (here: VVC Intra / VTM-11.1). |
| **Operating point** | A complexity/quality configuration. JPEG AI has three: SOP/BOP/HOP. |

**Only the decoder is normative.** This is the most important structural fact about every
modern coding standard, and it is why the paper says the encoder description is "for
informational purposes". The standard defines: given these bytes, produce this image.
How you *chose* those bytes is your business. That is why competing encoders (x264 vs
whatever) can differ enormously in quality while producing equally compliant streams —
and why JPEG AI ships two example encoders (`encoderID` 0 and 1) that are explicitly
"not a normative part of the standard".

---

## 9. Reading list, ordered by usefulness to this project

1. Ballé, Minnen, Singh, Hwang, Johnston, **"Variational image compression with a scale
   hyperprior"**, ICLR 2018. *The* architecture. Read this twice.
2. Minnen, Ballé, Toderici, **"Joint autoregressive and hierarchical priors for learned
   image compression"**, NeurIPS 2018. Mean+scale, and the context model.
3. He, Zheng, Sun, Wang, Qin, **"Checkerboard context model for efficient learned image
   compression"**, CVPR 2021. The idea behind MCM.
4. Zhang, Esenlik, Wu, Wang, Zhang, Zhang, **"End-to-end learning-based image compression
   with a decoupled framework"**, TCSVT 34(5), 2024. Reference [40] — the decoupling of
   entropy decoding from latent reconstruction, by the same authors. Directly explains
   JPEG AI's two-hyper-decoder split.
5. Jia, Brand, Yu, Karabutov, Alshina, Kaup, **"Overview of variable rate coding in JPEG
   AI"**, TCSVT 35(9), 2025. Reference [42] — the deep dive on gain units, quality maps
   and rate adaptation, including an example encoder rate-control algorithm. **Get this
   one; it fills the biggest gap in the overview paper.**
6. Ascenso, Alshina, Ebrahimi, **"The JPEG AI standard: providing efficient human and
   machine visual data consumption"**, IEEE MultiMedia 30(1), 2023. Reference [8] —
   the vision/requirements paper.
7. Alshina, Ascenso, Ebrahimi, IEEE MultiMedia 31(4), 2024. Reference [16] — status
   update.
8. Duda, **"Asymmetric numeral systems"**, arXiv 2013 — plus Yann Collet's FSE blog posts,
   which are far more readable than the paper for actually implementing tANS.
9. Bégaint et al., **CompressAI** (InterDigital). Not a paper — a PyTorch library with
   pretrained Ballé/Minnen/Cheng models and a working range coder. Your scaffolding.

---

## 10. Self-check before moving on

If you can answer these, read the next doc. If not, reread the relevant section.

1. Why is the rate term differentiable when file size isn't?
2. What does the hyperprior buy you over a factorised prior, and what does it cost?
3. Why does JPEG AI code `r̂ = round(y − p̈)` instead of `round(y)`?
4. A raster-scan context model and a 4-stage checkerboard model both use decoded
   neighbours. Why is only one of them shippable?
5. Why does the committee use seven metrics, and what does it mean that VIF and PSNR-HVS
   come out *worse* than VVC in Table III?
6. Why is tANS faster than arithmetic coding, and what does "memory-efficient" buy in
   me-tANS?
7. Why is only the decoder normative, and what does that let JPEG AI vendors do?
