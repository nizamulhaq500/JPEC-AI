# 02 — The JPEG AI Standard, Explained

A section-by-section walkthrough of Esenlik et al., *"An Overview of the JPEG AI
Learning-Based Image Coding Standard"*, TCSVT 36(2):2520–2537, Feb 2026 — with every
number, tensor shape and equation, plus what the paper leaves unsaid.

Symbols follow the paper: `Y` = primary (luma) component, `UV` = secondary (chroma)
component, `Ḣ`/`Ẇ` = *padded* height/width, `x̂` = reconstructed, `ŷ` = quantised latent,
`r̂` = quantised residual, `ẑ` = quantised hyper tensor, `p̈` = prediction tensor,
`Iσ` = variance tensor **in the logarithm domain** (an integer index, not a float σ).

---

## §I–II. What JPEG AI is, and how it got here

**Scope** (from the abstract): "a practical learning-based image coding standard offering a
single-stream, compact compressed domain representation, targeting both human
visualization and machine consumption."

Two phrases carry the whole design:

- **"single-stream"** — one codestream, decodable at several complexity/quality points. Not
  a layered/scalable stream with separate enhancement layers; the *same bytes* feed three
  different decoders.
- **"machine consumption"** — the latent `ŷ` is meant to be usable directly for vision
  tasks without reconstructing pixels. **Deferred to v2.** Version 1 is human-vision only.

### Timeline

| When | What |
|---|---|
| 2019 | JPEG AI project established (ISO/IEC JTC 1/SC 29/WG 1) |
| Nov 2019 | N85013: analysis of objective metrics for learned codecs → the 7-metric choice |
| Jan 2022 | Final **Call for Proposals** (N100095) |
| Jan 2023 | **CTTC** published (N100421) — common training & test conditions |
| Jul 2022 | 7 CfP responses; **Bytedance** [32] and **Huawei** [33] selected |
| Oct 2022 | The two harmonised → **VM 1.0** (Verification Model) |
| 2023–24 | Core experiments; device implementations on iPhone/smartphones |
| 2025 | Part 1 (core coding), Part 3 (ref. software), Part 5 (file format) finalised |
| Oct 2025 | Part 4 (conformance) target finalisation |
| Feb 2026 | This overview paper published |

### Real-device evidence (§II)

These numbers matter because they are the standard's whole justification:

- **1024×1024 decoded in < 20 ms** on an already-shipping smartphone [36], [37]
- **4K image decoded in ≈ 190 ms** on a smartphone [38]

These are unoptimised demos. The point of citing them is that the committee refused to
adopt any tool that could not be demonstrated on real hardware.

### The five parts (Table I)

| Part | Name | ITU-T Rec. \| ISO/IEC |
|---|---|---|
| 1 | Core coding systems | T.840-1 \| 6048-1 |
| 2 | Profiling | T.840-2 \| 6048-2 |
| 3 | Reference software | T.840-3 \| 6048-3 |
| 4 | Conformance | T.840-4 \| 6048-4 |
| 5 | File format | T.840-5 \| 6048-5 |

> **Practical tip:** ITU-T Recommendations are published free of charge on itu.int. If
> **T.840-1** is available there, that single document gives you the exact layer
> configurations, the CDF tables, the RVS/LSBS lookup tables, and the complete syntax —
> everything the overview paper compresses into prose. It is by far the highest-leverage
> document to obtain. The paper also states the four sets of trained model parameters are
> distributed **in ONNX format** via a link inside Part 1.

---

## §III. Design philosophy — the three decisions that shaped everything

This is the most important section in the paper, and the easiest to skim past.

### 1. Multi-branch decoding

One codestream, **three synthesis transforms** (`decoderID = 0, 1, 2`). Any of the three
produces a *conformant* reconstruction of the same bytes, at different quality and
different cost. Compare: in H.264/HEVC/VVC, a compliant decoder must produce one specific
output. JPEG AI deliberately admits three.

Why: a codestream in a cloud photo library will be decoded by a $80 phone, a laptop
without an NPU, and a workstation GPU. Rather than force the encoder to pick the lowest
common denominator, let each device pick its own operating point.

### 2. The bit-exact conformance point is *after entropy decoding only*

This is the subtle, clever one. JPEG AI mandates bit-exactness only up to and including
arithmetic decoding — i.e. the tensors `r̂_Y`, `r̂_UV` (and hence `ẑ_Y`, `ẑ_UV`) must be
*identical* on every device. Everything downstream (latent reconstruction, synthesis,
post-filters) may diverge slightly from the reference.

Why this is necessary: entropy decoding is a *feedback loop*. If your σ prediction differs
by one LSB from mine, we select a different CDF row, decode a different symbol, and the
entire rest of the stream desynchronises into garbage. Bit-exactness there is
non-negotiable. But the synthesis transform is *open-loop* — a 1e-6 float difference in a
conv output changes the last bit of a pixel and nothing else. Demanding bit-exactness
there would force everyone onto identical float semantics, killing NPU/GPU deployment.

**How they achieve it:** the entropy-decoding pipeline uses **integer arithmetic neural
network operations implementable with 8-bit multipliers and 32-bit accumulators**, and
[39] contains a proof that 32-bit registers cannot overflow anywhere in the pipeline. So
the hyper scale decoder is a fully integer network. This is the single most
implementation-critical fact in the paper.

### 3. Only ReLU / ReLU6 / convolution in the low operating points

`decoderID` 0 and 1 are restricted to layers "widely supported by parallel processing
accelerators on the market". Only HOP (`decoderID = 2`) uses richer layers, because it
targets high-end and future hardware. Memory operations like pixel shuffle were also
chosen with care (see §VI-F).

---

## §IV. The architecture, end to end

### Colour and component layout

Input is converted to **YCbCr** (ITU-R BT.709 — the paper writes "R.709-6"), at 4:2:0,
4:2:2 or 4:4:4.

- Primary component tensor: `x_Y [1, H, W]`
- Secondary component tensor: `x_UV [2, ⌈H/c_v⌉, ⌈W/c_h⌉]` where `c_v`, `c_h` are the
  chroma subsampling factors.

### Encoder path (non-normative — informative only)

```
x_Y ─────────────────► [ Primary analysis transform ] ──► y_Y  [160, Ḣ/2⁴, Ẇ/2⁴]
                                    ▲
x_UV ─► [ Preprocessing ]───────────┘ (uses x_Y)
        └──────────────► [ Secondary analysis transform ] ──► y_UV [96, Ḣ/2⁴, Ẇ/2⁴]

y_Y  ──► [ Primary Hyper Encoder ]   ──► Q ──► ẑ_Y   at Ḣ/2⁶, Ẇ/2⁶  ──► AE ──► z_Y-stream
y_UV ──► [ Secondary Hyper Encoder ] ──► Q ──► ẑ_UV  at Ḣ/2⁶, Ẇ/2⁶  ──► AE ──► z_UV-stream

ẑ_Y ──► [ Primary hyper decoder ]       ──► p̈_Y  [640, Ḣ/2⁵, Ẇ/2⁵]
ẑ_Y ──► [ Primary hyper scale decoder ] ──► Iσ_Y [160, Ḣ/2⁴, Ẇ/2⁴]

(y_Y, p̈_Y) ──► [ MCM + quantisation, 4 stages ] ──► r̂_Y, ŷ_Y  [160, Ḣ/2⁴, Ẇ/2⁴]
(y_UV, p̈_UV) ──► subtract + round ──────────────► r̂_UV       [96,  Ḣ/2⁴, Ẇ/2⁴]

r̂_Y  + Iσ_Y  ──► AE (N(0, σ²)) ──► r_Y-stream
r̂_UV + Iσ_UV ──► AE (N(0, σ²)) ──► r_UV-stream
```

Equation (1) — the secondary component skips MCM entirely:

```
r̂_UV[c,i,j] = round( y_UV[c,i,j] − p̈_UV[c,i,j] )        for 0 ≤ c < 96
```

The paper's reasoning: "from the perspective of the trade-off between computational
complexity and coding gain, it was concluded that usage of MCM process for the secondary
component is not justified." Chroma is already cheap; a 4-stage context model on it is not
worth the silicon.

Note: **in the encoder the primary and secondary branches are fully independent except for
the preprocessing stage** at the start of secondary analysis, which consumes `x_Y`.

### Decoder path (normative)

```
z_Y-stream  ──► AD ──► ẑ_Y   ──┬──► [ hyper scale decoder ] ──► Iσ_Y
                               └──► [ hyper decoder ]       ──► p̈_Y
z_UV-stream ──► AD ──► ẑ_UV  ──┬──► [ hyper scale decoder ] ──► Iσ_UV
                               └──► [ hyper decoder ]       ──► p̈_UV

r_Y-stream  + Iσ_Y  ──► AD ──► r̂_Y
r_UV-stream + Iσ_UV ──► AD ──► r̂_UV
        └──────── END OF BIT-EXACT CONFORMANCE POINT ────────┘

(r̂_Y, p̈_Y) ──► [ MCM, 4 stages ] ──► ŷ_Y
r̂_UV + p̈_UV ─────────────────────► ŷ_UV                          … eq (2)
ŷ_UV ⊕ ŷ_Y  ─────────────────────► ŷᶜ_UV [256, Ḣ/2⁴, Ẇ/2⁴]      … eq (3)

ŷ_Y    ──► [ Primary synthesis transform   (decoderID 0|1|2) ] ──► x̂_Y
ŷᶜ_UV  ──► [ Secondary synthesis transform (decoderID 0|1|2) ] ──► x̂_UV

(x̂_Y, x̂_UV) ──► post-processing filters ──► colour space conversion ──► scale/clip ──► x̂
```

Equation (2):  `ŷ_UV[c,i,j] = r̂_UV[c,i,j] + p̈_UV[c,i,j]`,  `0 ≤ c < 96`

Equation (3) — the second (and last) cross-component link:

```
ŷᶜ_UV[c,i,j] = ŷ_UV[c,i,j]   for   0 ≤ c <  96
             = ŷ_Y [c,i,j]   for  96 ≤ c < 256
```

So the chroma synthesis transform sees the *luma latent* concatenated onto the chroma
latent — 96 + 160 = 256 channels. Chroma reconstruction is conditioned on luma structure,
which is where the chroma branch gets its edge fidelity from without paying for a joint
transform.

> **Inference (not stated in the paper, but forced by eq. 3):** since `ŷ_UV` and `ŷ_Y`
> must have *identical spatial dimensions* to concatenate, the secondary analysis
> transform must use **one fewer downsampling stage than the primary for 4:2:0 input**
> (3 vs 4), because `x_UV` already arrives at half resolution. For 4:4:4 it needs all 4.
> That is exactly why `c_ver_minus1`/`c_hor_minus1` in the picture header are described as
> "controlling the coded picture used in internal processing, i.e. **the layer
> configuration of the synthesis transform**". Get this wrong and nothing lines up.

### Two design decisions the paper explicitly flags (§IV, end)

1. **Arithmetic decoding is sequential → likely CPU.** So the two networks interleaved
   *between* the two arithmetic-decoding stages (the primary and secondary hyper **scale**
   decoders) are deliberately made computationally tiny, so the whole entropy pipeline can
   live on the CPU without a round-trip to the accelerator.
2. **Integer arithmetic in the entropy pipeline** → bit-exact `r̂` and `ẑ` across devices.
   8-bit multipliers, 32-bit registers, overflow impossible per [39].

---

## §V. Codestream organisation and high-level syntax

Structure: `SOC` … codestream segments … `EOC`. Each segment = 2-byte marker +
variable-length size + payload, with **byte alignment enforced after the size field and
after the payload**.

### Markers (Table II — recovered from the PDF's vector graphics)

| Symbol | Code | Payload | M/O |
|---|---|---|---|
| **SOC** | `0xff80` | none (Start Of Codestream) | Mandatory |
| **EOC** | `0xff81` | none (End Of Codestream) | Mandatory |
| **PIH** | `0xff82` | Picture header | Mandatory |
| **SOZ** | `0xff88` | `z_Y-stream` **and** `z_UV-stream` | Mandatory |
| **SORp** | `0xff89` | `r_Y-stream` | Mandatory |
| **SORs** | `0xff8a` | `r_UV-stream` | Mandatory |
| **TOH** | `0xff83` | Tools header | Optional |
| **SOQ** | `0xff8b` | Quality map information | Optional |
| **UDI** | `0xff8c` | User defined information | Optional |
| **RDI** | `0xff84` | Rendering information | Optional |

Both hyper substreams share one `SOZ` segment because `ẑ` is a small fraction of the
total bits. Multiple `SORp`/`SORs` segments may appear provided each carries a distinct
`region_idx` (`region_idx[0]` for primary, `region_idx[1]` for secondary) — that is the
mechanism behind region partitioning (§VI-J).

### PIH — Picture header (mandatory)

Carries: profile/level IDs, picture size, output bit-depth, internal **and** output
subsampling modes, model indices, plus control parameters for:
colour space conversion · region/tile partitioning · multithreaded entropy decoding ·
**RVS** · rate adaptation · skip mode.

### TOH — Tools header (optional)

Everything applied *after* the entropy pipeline:
- `lsbs_enable_flag[comp]` × 2 (**LSBS** for primary and secondary)
- a **filters header** with four enable flags: EFE linear, **ICCI**, EFE nonlinear, **LEF**
  — plus each enabled filter's control data.

**If TOH is absent, all these flags are inferred to be 0.** So the minimum viable codestream
is `SOC · PIH · SOZ · SORp · SORs · EOC` — a useful first implementation target.

### SOQ — Quality map information

A 2-D quality map tensor enabling different quantisation step sizes per spatial location
("**3D gain unit**"). This is the RoI-coding mechanism. Detail in §VI-I.

### UDI / RDI

- **UDI**: arbitrary application-defined bytes, any format. Escape hatch for
  application standards built on JPEG AI.
- **RDI**: rendering metadata — colour primaries, transfer characteristics, matrix
  coefficients, full-range flag, chroma sample location; mastering-display colour volume
  (primaries, white point, luminance range); nominal target brightness upper bounds; and
  HDR dynamic metadata. This is how wide-gamut/HDR support is delivered.

---

## §VI. The coding techniques, tool by tool

### VI-A. Separated processing of colour components

Motivation, in the paper's own priority order: (1) HVS is more sensitive to luminance;
(2) YUV decorrelates better than RGB; (3) it lets you *prioritise luma during training*;
(4) legacy device compatibility. Plus the dominant one: **complexity**.

Four picture-header flags:
- `s_ver_minus1`, `s_hor_minus1` → subsampling of the **intended output** picture
- `c_ver_minus1`, `c_hor_minus1` → subsampling of the **coded** picture (internal)

If internal ≠ output format, a **normative upsampling process** runs at the end.

Cross-component information flows at exactly **two** points: start of analysis
(preprocessing), start of synthesis (concatenation, eq. 3). Nothing else — *except* the
non-normative post-processing filters (ICCI in particular crosses components).

The paper is candid that this costs coding gain: "may not be ideal in terms of the coding
gain, as there are usually significant correlations between the samples of the luma and
chroma components." Accepted for complexity. Bonus: **monochrome fast path** — decode luma
only, skip the chroma branch, for CV applications.

### VI-B. Colour space conversion

Models were trained on BT.709 YCbCr, so performance degrades in other internal colour
spaces. JPEG AI puts a **mandatory** conversion at the very *end* of decoding (after
post-filters), controlled by `colour_transform_idx`:

- `= 1` → no conversion
- `= 0` → predefined YCbCr → RGB, eq (4):
  ```
  x̂[0] = x̂_Y[0] + 1.5748·(x̂_UV[1] − 0.5)
  x̂[1] = x̂_Y[0] + 1.8556·(x̂_UV[1] − 0.5)
  x̂[2] = ( x̂_Y[0] − 0.2126·x̂[0] − 0.07222·x̂[1] ) / 0.7152
  ```
  > **Caution — the paper appears to have typos here.** Standard BT.709 inverse is
  > `R = Y + 1.5748·Cr`, `B = Y + 1.8556·Cb`, `G = (Y − 0.2126·R − 0.0722·B)/0.7152`.
  > As printed, both the first and second equations index `x̂_UV[1]` (they should be
  > `[1]`=Cr and `[0]`=Cb respectively), and `0.07222` should be `0.0722`. Implement the
  > standard BT.709 inverse; verify against T.840-1.
- `= 2` → encoder-defined 3×3 matrix `a[3,3]` + bias `b[3]` signalled in the PIH, eq (5).

Final scaling/clipping, eq (6):
`x̂[c,i,j] ← clip(0, 2^bitdepth − 1, x̂[c,i,j] · (2^bitdepth − 1))`

### VI-C. Entropy coding pipeline

Three steps: decode `ẑ` → compute `Iσ` → decode `r̂` using `Iσ`.

**Decoding `ẑ`:** fixed CDF table, **`[128, 64]`**, specified in Part 1. Table row selected
by **channel index** `c`. Coder: **me-tANS**.

**Computing `Iσ`:** `ẑ` → hyper scale decoder → `Iσ_Y`, `Iσ_UV`. Residuals are modelled as
**zero-mean Gaussian** with these per-sample standard deviations. The hyper scale decoder
is "designed with minimal complexity and is quantized to provide identical results across
different architectures and devices."

**Decoding `r̂`:** precalculated CDF table **`[32, 256]`** — 32 distributions × 256 symbol
values. Row selected by `Iσ[c,i,j]`. Using precomputed distributions is where the
throughput comes from (hence "*tabulated* ANS").

> **Note the dimension mismatch to resolve in Part 1:** `Iσ` values range over thousands
> (the RVS tables are indexed `[…, 3968]`), yet the residual CDF table has only 32 rows. So
> there must be a normative mapping from `Iσ` to one of 32 σ-classes (a quantisation of the
> log-σ index). Same question for `ẑ`: the paper writes `ẑ_Y` with 160 channels while the
> hyper CDF table has 128 rows. **Both are things to pin down from T.840-1 before coding
> the entropy stage.**

#### Skip mode

If `Iσ[c,i,j] <` a fixed predefined threshold, **skip coding that residual sample
entirely** (it is inferred zero). "Depending on the content, coding of up to **80%** of the
residual samples might be skipped." Enormous throughput win.

But: the fixed threshold "may cause visual artefacts in rare cases". Safeguard:
**cube-based skip mode** — the skip decision can be explicitly reverted per
**16×16×16 partition** of the residual tensor, signalled in the codestream. A classic
standards-committee move: keep the fast heuristic, add an explicit override for the
pathological cases.

#### Multithreading

`z_Y`, `z_UV`, `r_Y`, `r_UV` and `q` streams can each be split into independently
decodable substreams. Substream **offsets** are signalled at the start of the segment; the
substream **count** in the picture header.

#### me-tANS (Algorithm 1)

Four precomputed tables: `transition_table_symbol`, `transition_table_nBits`,
`transition_table_stateNext`, `bound_table`. Table selected by `Iσ[c,i,j]` for residuals,
by channel `c` for hyper samples. Total table memory **≈ 100 KB** at the decoder.

```
Initialisation:
  flatten Iσ[] to 1-D
  point at the LAST symbol position; pointer moves BACKWARDS   ← FILO
  s ← parse 8 bits

for i in 0 .. n_symbols/4:
    for j in 0..3:                                    # fast path, 4 symbols at a time
        r̂[4i+j] ← transition_table_symbol[ Iσ[4i+j] ][s]
        n       ← transition_table_nBits [ Iσ[4i+j] ][s]
        value   ← parse n bits
        s       ← transition_table_stateNext[ Iσ[4i+j] ][s] | value      # just an OR
    for j in 0..3:                                    # escape path
        if r̂[4i+j] + bound_table[ Iσ[4i+j] ] == 0:
            ind   ← parse 1 bit
            m     ← ind ? 2 : 15
            value ← parse m bits
            sign  ← parse 1 bit
            r̂[4i+j] ← bound_table[ Iσ[4i+j] ] + value × sign
reorganise r̂[] to 3-D
```

Read that carefully — it is the most implementation-dense paragraph in the paper:

- **Decoding a symbol costs one table read, a bitwise OR and an addition.** That is the
  entire point of tANS.
- **Two-tier symbol coding.** Small-magnitude symbols come straight from the table.
  When the decoded value hits the table's boundary (`r̂ + bound_table == 0`, i.e. the
  symbol equals the escape marker), an **outbound value** is read: 1 flag bit selecting a
  2-bit or 15-bit field, then the value, then a sign bit. So the table only has to cover
  the high-probability core of the distribution while remaining able to code rare large
  values. This is the ANS analogue of Golomb/Exp-Golomb escape coding.
- **Even single-threaded operation interleaves two substreams and maintains two ANS
  states**, "effectively mimicking the behavior of a dual-threaded setup" — instruction-level
  parallelism, for free, on one core.
- The `/4` loop structure is deliberate: 4 symbols per inner iteration, matching the
  4-symbol grouping used for the escape pass.

### VI-D. Latent sample reconstruction — MCM

`(r̂_Y, ẑ_Y)` → `ŷ_Y` via hyper decoder + **Multi-stage Context Modeling** in **4 stages**.
`ŷ_UV` needs only the hyper decoder (eq. 2).

Each MCM stage reconstructs one group of latent samples using: the corresponding samples of
`r̂_Y`, the prediction `p̈_Y`, **and every latent sample already reconstructed in earlier
stages**. Fig. 2 shows the grouping: a **2×2 spatial checkerboard pattern, identical in
every channel** — group 1 = (even row, even col), group 2 = (even, odd), group 3 = (odd,
even), group 4 = (odd, odd), by the natural reading of the figure.

- Within a stage: all samples independent → one fully parallel network pass.
- Across stages: 4 sequential passes total, **independent of image size**.

"During the development of JPEG AI, various latent sample reconstruction schemes have been
evaluated, with the current design with 4 processing stages achieving the best compromise
between complexity and coding efficiency."

Per-stage structure is in Appendix C of the supplementary material; hyper decoder and hyper
scale decoder in Appendices A and B. **You do not have the supplement — this is your
single biggest missing piece.** Options: get the supplement from IEEE Xplore
(`doi.org/10.1109/TCSVT.2025.3613244`, "supplementary downloadable material"), or read the
layer configs from T.840-1, or inspect the ONNX models directly.

### VI-E. Decoupled entropy decoding and latent reconstruction

The reason there are **two** networks consuming `ẑ`:

| Network | Output | Belongs to | Runs on |
|---|---|---|---|
| hyper **scale** decoder | `Iσ` | the entropy decoding pipeline | CPU (integer, bit-exact) |
| hyper decoder | `p̈` | latent sample reconstruction | NPU/GPU (float, tolerant) |

Ballé's original design has a single `h_s`. Splitting it means the entropy pipeline is a
self-contained engine: feed it a codestream, get `r̂` out, no accelerator round-trip and no
float-determinism requirement on the accelerator. Latent reconstruction and synthesis then
run entirely on the NPU. Full analysis in [40] (Zhang et al., TCSVT 2024) — same authors.

This is the kind of design decision that only comes out of a standards process, and it is
worth highlighting in your presentation: **it is an architectural change motivated purely
by deployment, at a small cost in coding efficiency.**

### VI-F. Synthesis transforms — the three operating points

"As this stage concentrates most of the codec's computational burden."

| decoderID | Name | Target | Upsampling | Final-layer channels | kMAC/pxl |
|---|---|---|---|---|---|
| 0 | **SOP** (Simple) | laptops w/o NN accel; mid/low-end mobile; tight power/timing | 2×2 conv + **pixel shuffle** | **32** | **14** |
| 1 | **BOP** (Base) | high-end mobile with NPU/GPU | **4×4 deconvolution**, pixel shuffle only in the final layer | **64** | **28** |
| 2 | **HOP** (High) | desktops with GPUs | richer layers beyond ReLU/ReLU6/conv | — | **215** |

The `decoderID 0` vs `1` distinction is a lovely piece of real-world engineering:
pixel shuffle is *cheaper* in theory than deconvolution, but "implementations of pixel
shuffling might be inefficient on certain devices" — some NPUs handle the memory
reshuffle badly. So BOP uses 4×4 deconvolution everywhere and confines pixel shuffle to
the last layer. The 32-vs-64 final channel count is what makes SOP roughly half the cost.

HOP is 7.7× the MACs of BOP for ~2 percentage points of average BD-rate (encoderID 0) —
but 7 points when paired with encoderID 1 (see Table III). Diminishing but real.

Network diagrams: Appendix figures 6 (decoderID 0), 7 (decoderID 1), 8 (decoderID 2) of
the supplement.

### VI-G. Residual and Variance Scaling (RVS) — the biggest single tool

**The problem it solves:** models are trained on a weighted MSE + MS-SSIM loss, but scored
on seven metrics. RVS is a decoder-side, conditional rescaling of `Iσ` and `r̂` that buys
back BD-rate on **VMAF, FSIM and NLPD** specifically because those are *not* in the
training loss. It contributes **2.2%** average gain — more than all other switchable tools
combined.

Control: `grfs_enable_flag[comp]` and `rvs_enable_flag[comp]` in the PIH
(`comp` = 0 primary, 1 secondary). Identical process for both components.

**Step 1** — average-pool `Iσ_Y` over 8×8 blocks, eq (7):

```
σ_Y[c,i,j] = ( 32 + Σ_{i'=0..7} Σ_{j'=0..7} Iσ_Y[c, 8i+i', 8j+j'] ) >> 6
```
for `0 ≤ c < 160`, `0 ≤ i,j < Ḣ/128, Ẇ/128`.

That is a **rounded integer mean** over 64 samples (`+32` then `>>6`). Boundary padding
value = **1411**, "selected since it represents the average value of the samples of `Iσ_Y`".
Resolution check: `Iσ` at /16 of the image, pooled 8× → /128. ✓

**Step 2** — update `Iσ_Y` and `r̂_Y`, eqs (8), (9):

```
Iσ_Y[c,i,j] ← Iσ_Y[c,i,j] + T1[ modelID, id[c], σ_Y[c, ⌊i/8⌋, ⌊j/8⌋] ]
r̂_Y[c,i,j] ← r̂_Y[c,i,j] · T2[ modelID, id[c], σ_Y[c, ⌊i/8⌋, ⌊j/8⌋] ] / 2¹⁶
```
with `id[c] = GRFS_Y[c] + 2·rvs_enable_flag[0]`, over `0 ≤ c,i,j < 160, Ḣ/16, Ẇ/16`.

- `T1[4, 4, 3968]`, `T2[4, 4, 3968]` — normative tables in Part 1, which **also specifies
  how to compute the entries on the fly** so implementations need not store them.
  Dimensions: 4 modelIDs × 4 `id` values × 3968 σ buckets.
- `GRFS_Y[c]` ∈ {0,1} are per-channel flags signalled in the PIH when
  `grfs_enable_flag[0]` is true. So `id[c]` ∈ {0,1,2,3} selects one of four
  correction curves per channel — a cheap per-channel, per-σ correction with almost no
  side information.
- Additive in the log domain (`Iσ`) = multiplicative on σ. `/2¹⁶` = 16-bit fixed point.

Same `T1`/`T2` tables serve the secondary component.

### VI-H. Latent Scaling Before Synthesis (LSBS)

Inputs: `r̂` (after entropy decoding) and `ŷ` (after latent reconstruction), plus the same
pooled `σ_Y`/`σ_UV` from RVS. Controlled by `lsbs_enable_flag[comp]` in the **tools
header**.

Let `μ_Y[c,i,j] = ŷ_Y[c,i,j] − r̂_Y[c,i,j]` — i.e. recover the *prediction* part by
subtracting the residual out of the reconstructed latent. Then, eq (10):

```
ŷ_Y[c,i,j] ← ŷ_Y[c,i,j]
           + ( r̂_Y[c,i,j] · TR[ modelID, σ_Y[c,⌊i/8⌋,⌊j/8⌋] ]
             + μ_Y[c,i,j] · TP[ modelID, σ_Y[c,⌊i/8⌋,⌊j/8⌋] ]
             + 2¹² ) >> 13
```

`TP[4, 3968]`, `TR[4, 3968]`, normative, also computable on the fly. `+2¹² then >>13` =
round-to-nearest in 13-bit fixed point.

Read it plainly: **a σ-dependent affine reweighting of "how much to trust the prediction
versus the residual" applied to the latent just before synthesis.** In low-σ (flat)
regions you want to lean on the prediction; in high-σ (textured) regions on the residual.
The optimal balance differs from what the trained network produces, and LSBS is the
learned correction. Worth **0.4%**.

### VI-I. Rate adaptation and quality map

**Three** independent quality controls:

1. **Model selection.** Four model parameter sets, trained for four Lagrange multipliers
   `β_train`, stored as **ONNX**, selected by `modelID` in the PIH. Roughly a **20× BPP
   range** between the lowest- and highest-quality models (content- and
   encoder-dependent).
2. **Gain unit** — a channel-wise adjustment for fine rate control between models.
3. **3D gain unit** — gain unit + a **spatial** quality map, for flexible spatial bit
   allocation (RoI coding).

The decoder-side process, eqs (11)–(15):

```
(11)  mlog[comp,c,i,j] = betaDisplacementLog[comp] + mref[modelID, comp, c]
(12)  if gain_3D_enable_flag:  mlog[comp,c,i,j] += Gain3d[i,j]
(13)  m⁻¹[comp,c,i,j] = exp( −mlog[comp,c,i,j] · step / 2^sigmaPrecision )
(14)  r̂_Y  ← r̂_Y  · m⁻¹[0,c,i,j]        r̂_UV ← r̂_UV · m⁻¹[1,c,i,j]
(15)  Iσ_Y ← Iσ_Y + mlog[0,c,i,j]       Iσ_UV ← Iσ_UV + mlog[1,c,i,j]
```
over `0 ≤ i,j < Ḣ/16, Ẇ/16`; `c < 160` (primary), `c < 96` (secondary).

- `betaDisplacementLog[comp]` — **the rate knob**, signalled per component in the PIH.
  Luma and chroma quality are independently adjustable.
- `mref[modelID, comp, c]` — predefined per-channel reference gains: the channel-wise gain
  unit.
- `Gain3d[i,j]` — derived from the **quality map** in the `SOQ` segment: spatial rate
  control → RoI.
- `step`, `sigmaPrecision`, `mref` are predefined **12-bit signed integers**. The `exp()`
  in (13) "can be replaced with a lookup-table in practical implementations."

**Sign convention** (worth working out because the paper doesn't spell it out): the encoder
must additionally scale the *unquantised* residuals `r_Y`, `r_UV` "in an opposite way",
i.e. multiply by `m = 1/m⁻¹ = exp(+mlog·step/2^sp)` before rounding. For `mlog > 0` that
scales *up* before rounding → **finer effective quantisation → higher rate**, and the
matching `Iσ += mlog` widens the modelled distribution to fit the now-larger integers.
Consistent. Higher `betaDisplacementLog` → higher quality.

This is the mechanism that turns 4 trained models into a **continuous** rate ladder — the
same trick as "modulated autoencoders" / gain-unit variable-rate coding in the literature.
Deep dive: reference **[42]**, Jia et al., *"Overview of variable rate coding in JPEG AI"*,
TCSVT 35(9) 2025, which also gives an example encoder rate-control algorithm. **Get it.**

### VI-J. Tiling and spatial random access

Two *independent* partitioning schemes targeting different pipeline stages:

| Scheme | Targets | Purpose | Flag |
|---|---|---|---|
| **Synthesis transform tiling** | the synthesis transform | tile `ŷ` so reconstruction is patch-by-patch → bounds **peak memory** on large images | `synthesis_tile_enable[comp]` |
| **Region partitioning** | entropy decoding + latent reconstruction | split `r_Y-stream`/`r_UV-stream` into separate codestream segments, each decodable to its own region of `ŷ` → **spatial random access** | `region_partitioning_flag` |

`region_residual_in_its_own_substream_flag` decides whether regions are truly independent.
When true:
- each region **shall contain an integer multiple of synthesis transform tiles**, and
- latent sample reconstruction (MCM) and synthesis **must not** use samples from
  neighbouring regions.

Use cases named: 360° imaging, VR, RoI coding, thumbnails, transcoding. Both schemes can
be combined. Example partitioning figure is in the supplement.

Note the trade-off, unstated but real: independent regions kill context and boundary
prediction across region edges, so random access costs coding efficiency — exactly like
slices/tiles in HEVC/VVC.

### VI-K. Arbitrary image sizes — two mechanisms

**1. Layer-based cropping.** The analysis network has a **padding layer in front of each of
its 4 downsampling layers**, and the hyper encoder in front of each of its **2**. When a
tensor dimension is odd before halving, pad by one sample. Each encoder padding layer has a
**corresponding cropping layer after the matching upsampling layer** in the synthesis,
hyper decoder and hyper scale decoder networks. Called "right-on-point padding": an
intermediate tensor grows by at most one line per direction, so almost no redundant
computation.

This is where `Ḣ`, `Ẇ` (padded dimensions) in all the tensor shapes come from.

**2. Display window.** Layer-based cropping "cannot be gracefully supported by some device
implementations, since it requires changing the intermediate tensor size between two
layers of a subnetwork" — i.e. it defeats static-shape graph compilation, which is how most
NPU toolchains work. So JPEG AI alternatively lets you **pad the original image to a
multiple of 64** in both directions, encode that, and crop the *reconstruction* at the end.
No intermediate cropping at all → fully static shapes → faster on such devices.
Controlled by `diff_display_img_width` and `diff_display_img_height` in the PIH.

64 = 2⁶ = the total downsampling factor of analysis (2⁴) × hyper encoder (2²).

### VI-L. Progressive decoding

Almost comically simple: **zero out part of `r̂_Y` / `r̂_UV` and run latent reconstruction and
synthesis on the partially filled tensors.** No normative machinery; the standard just
guarantees it works. Gives previews/low-quality reconstructions from a truncated
codestream — useful on slow links or weak devices. The *method* of choosing what to drop is
deliberately left unspecified.

### VI-M. Post-processing filters

Four filters, all **optional** — "the decoder can skip their application even if they are
enabled by the syntax". Control data lives in the tools header.

| Filter | Modifies | Ablation: avg BD-rate contribution |
|---|---|---|
| **EFE linear** (Enhancement Filtering Extensions) | secondary only | **−0.2%** (slight loss) but **+12% chroma PSNR** |
| **EFE nonlinear** | secondary only | **−0.2%** (slight loss) but **+8% chroma PSNR** |
| **ICCI** (Inter-Channel Correlation Information) | **both** primary and secondary | **+0.2%** |
| **LEF** (Luma Edge Filter) | primary only | **+0.3%** |

Purpose: recover colour, detail and contrast lost to quantisation and reconstruction.
Note ICCI is the *third* cross-component path, outside the two normative ones — the paper
flags this in §VI-A ("except during the non-normative post-processing filtering stages").

The EFE filters are the interesting case: they *hurt* the 7-metric average (their side
information costs bits) but massively improve chroma PSNR, "which is not among the metrics
used in the development of JPEG AI". The committee kept them anyway. A good illustration of
metric-driven development having blind spots that engineering judgement has to cover.

---

## §VII. Experimental results

### Setup

- **Datasets:** JPEG AI test set (50 natural images, 1K–4K; the CTTC common condition);
  JPEG AI synthetic set (36 images: animation, screen, game content); **Kodak** (24
  images); **CLIC 2024** validation set.
- **Anchor:** VVC Intra via **VTM-11.1**; PNG→YUV with FFmpeg.
- **Test codec:** JPEG AI **VM 7.0**.
- **Metrics:** the seven, plus **BD-rate**; complexity in **kMAC/pxl**; decode time
  normalised to **ms/megapixel**.
- **Hardware:** NVIDIA Tesla V100 (32 GB) + Intel Xeon Platinum 8336C @ 2.30 GHz.

### Table III — main result (BD-rate vs VVC Intra; negative = JPEG AI better)

| | | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS | kMAC/pxl | ms/MPx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **encoderID 0** | decoderID 0 | **−16.2%** | −30.9% | +6.9% | −22.3% | −13.4% | −26.6% | −30.0% | +2.9% | 14 | 285 |
| | decoderID 1 | **−20.2%** | −33.0% | +1.4% | −26.9% | −17.3% | −29.1% | −34.8% | −1.9% | 28 | 266 |
| | decoderID 2 | **−22.1%** | −34.8% | −2.0% | −27.7% | −19.3% | −31.2% | −37.3% | −2.5% | 215 | 323 |
| **encoderID 1** | decoderID 0 | **−14.4%** | −30.3% | +9.7% | −20.4% | −11.6% | −25.1% | −29.2% | +6.1% | 14 | 246 |
| | decoderID 1 | **−19.9%** | −33.0% | +1.5% | −26.4% | −16.7% | −28.4% | −35.8% | −0.8% | 28 | 271 |
| | decoderID 2 | **−27.0%** | −37.6% | −8.5% | −34.7% | −23.6% | −33.7% | −42.4% | −8.8% | 215 | 332 |

> The prose in §VII-B quotes slightly different averages (16.0 / 20.2 / 21.1 and
> 13.9 / 19.7 / 27) than the table (16.2 / 20.2 / 22.1 and 14.4 / 19.9 / 27.0). Cite the
> table.

Things to actually notice:

1. **Two analysis transforms exist** (`encoderID` 0 and 1) and they are *not* ordered —
   encoderID 0 is better with decoders 0 and 1; encoderID 1 is dramatically better with
   decoder 2 (−27.0% vs −22.1%). "The encoder that is used in the bitstream generation
   depends on the target decoders." Encoder-side matching to the expected decoder is a real
   deployment decision.
2. **MS-SSIM and VMAF carry the win** (−30% to −42%). VIF and PSNR-HVS are the weak spots
   and are *positive* (worse than VVC) at decoderID 0. Perceptually-trained-codec
   fingerprint.
3. **kMAC/pxl and wall-clock decouple completely.** 14 → 215 kMAC/pxl is 15× the
   arithmetic, but decode time only moves 285 → 323 ms/MPx on a V100. The GPU is not
   compute-bound; the sequential entropy decoding dominates. On a phone the ranking would
   invert — which is exactly why the three operating points exist and why the paper says
   "the capabilities of the target device are the ultimate determining factor."
4. Total decode-time range across all six combinations: **246–332 ms/MPx**.

### Table IV — tool-off ablation (encoderID 0, decoderID 1)

| TEST | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS | kMAC/pxl | ms/MPx |
|---|---|---|---|---|---|---|---|---|---|---|
| **All on** | **−20.2%** | −33.0% | 1.4% | −26.9% | −17.3% | −29.1% | −34.9% | −1.9% | 27.7 | 266 |
| RVS off | −18.0% | −33.1% | 0.8% | −20.2% | −14.8% | −29.0% | −28.8% | −0.7% | 27.7 | 249 |
| LSBS off | −19.8% | −33.1% | 1.8% | −27.4% | −17.5% | −29.1% | −30.5% | −2.7% | 27.6 | 249 |
| LEF off | −19.9% | −33.2% | 1.2% | −27.3% | −18.1% | −29.0% | −29.0% | −3.7% | 27.7 | 251 |
| ICCI off | −20.0% | −32.8% | 1.7% | −26.9% | −17.2% | −29.0% | −34.1% | −2.0% | 23.1 | 245 |
| EFE nonlinear off | −20.4% | −33.3% | 1.2% | −26.2% | −17.6% | −29.4% | −35.1% | −2.1% | 27.7 | 246 |
| EFE linear off | −20.4% | −33.3% | 1.1% | −26.1% | −17.6% | −29.4% | −35.2% | −2.2% | 28.6 | 250 |

Ranking: **RVS 2.2% ≫ LSBS 0.4% > LEF 0.3% > ICCI 0.2% > EFE nonlinear/linear −0.2%.**

Look at *where* RVS pays: FSIM −20.2 → −26.9 (6.7 points) and VMAF −28.8 → −34.9 (6.1
points), while MS-SSIM barely moves. Exactly as designed — RVS targets the metrics absent
from the training loss. Also note ICCI is the only tool with real compute cost
(23.1 → 27.7 kMAC/pxl, i.e. **4.6 of BOP's 28 kMAC/pxl, ~17%**, for 0.2% BD-rate).

### Tables V & VI — generalisation

**Kodak:**

| | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS |
|---|---|---|---|---|---|---|---|---|
| decoderID 0 | −7.5% | −29.8% | +18.1% | −19.7% | −0.2% | −24.1% | −22.3% | +25.3% |
| decoderID 1 | −12.9% | −32.1% | +11.4% | −22.9% | −6.3% | −26.8% | −28.4% | +14.5% |
| decoderID 2 | −21.1% | −37.3% | 0.0% | −28.8% | −15.6% | −32.0% | −38.3% | +4.4% |

**CLIC 2024 validation:**

| | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS |
|---|---|---|---|---|---|---|---|---|
| decoderID 0 | −12.1% | −25.7% | +22.6% | −30.8% | −7.6% | −25.0% | −25.4% | +7.3% |
| decoderID 1 | −16.8% | −28.4% | +15.4% | −34.6% | −12.2% | −27.9% | −32.0% | +1.9% |
| decoderID 2 | −24.9% | −34.5% | +2.8% | −42.3% | −19.9% | −33.5% | −40.7% | −6.3% |

The paper's summary — "roughly between 7% and 25% coding gain depending on the selected
decoder" — is fair, but note the honest reading: on Kodak at SOP the average gain collapses
to −7.5%, driven by **PSNR-HVS +25.3%** and **VIF +18.1%**. Kodak is 768×512, much smaller
than the 1K–4K CTTC images; learned codecs generally do relatively worse on small images
(less spatial context to exploit, and the hyperprior overhead is proportionally larger).
**Do not present "JPEG AI beats VVC by 20%" without a caveat** — it depends on decoder,
dataset and metric, and on two of seven metrics it loses.

### §VII-D. Subjective results (Fig. 3)

Two rate points, ≈0.08 bpp and ≈0.3 bpp, post-filters **disabled**, same codestream decoded
with BOP and HOP, versus VVC Intra.

- **Natural content:** JPEG AI preserves more textural detail than VVC Intra, especially at
  the lowest rate. BOP and HOP both beat VVC subjectively; HOP's colours are "more vibrant
  and closer to the original".
- **Synthetic content:** HOP > BOP in fidelity, but both "struggle with sharp straight
  lines"; overall on par with VVC Intra. **On screen-captured images containing letters,
  reconstruction quality is worse than VVC.**

The paper is admirably direct about this: "One limitation of JPEG AI is that its performance
is not consistent in synthetic content… The synthetic content coding has not been the main
focus in the development of the first version of JPEG AI due to time limitations."

---

## §VIII. Profiling (Part 2)

A **nested** profile structure: **stream profiles** (constraints on codestream syntax and
admissible values) each associated with one or more **decoder profiles** (subsets of the
decoder tool set). A codestream conforming to a stream profile is decodable by *all* its
associated decoder profiles. Both IDs live in the picture header.

Draft Part 2 defines **one stream profile** and **three decoder profiles**:

| Profile | Supported synthesis transforms |
|---|---|
| Main@Simple | decoderID 0 only |
| Main@Base | decoderID 0, 1 |
| Main@High | decoderID 0, 1, 2 |

In the Main stream profile, post-processing filters are **not mandatory**.

This is how "multi-branch decoding" becomes enforceable: the encoder declares a stream
profile, and every decoder profile associated with it is guaranteed to be able to decode.

## §IX. Conformance (Part 4)

Defines a **test suite per (profile, level) pair**. Pass all tests → compliant. As
established in §III, conformance permits non-bit-exact reconstruction downstream of entropy
decoding — this is the part that makes multi-vendor NPU implementations legal. Still in
draft at the time of writing, target Oct 2025.

## §X. Reference software (Part 3)

The **VM** — encoder + decoder, plus metric-calculation and testing scripts to reproduce
CTTC results, plus profiling tools for per-component timing and tool impact.
`https://gitlab.com/wg1/jpeg-ai/jpeg-ai-vm`. Explicitly "rarely used directly in practical
applications" — it is a research and verification tool, not a product.

## §XI. File format (Part 5)

Containers based on **ISOBMFF** (ISO/IEC 14496-12) and **HEIF** (ISO/IEC 23008-12):

- **Annex A — Motion JPEG AI**: timed sequences of JPEG AI images in ISOBMFF, with timing.
  One or more motion sequences; supports editing, display, interchange, streaming.
- **Annex B — HEIF encapsulation**: single images, image collections, image sequences, with
  defined file brands.

So JPEG AI gets a video-ish path (Motion JPEG AI = intra-only sequences, like Motion JPEG
2000) and a still-image path, without inventing a new container.

---

## §XII–XIII. Conclusions, limitations, and where this goes next

### Honest limitations, as stated

1. **Synthetic/screen content** — only basic support. No dedicated tools or model
   parameters. Generally worse than traditional codecs there.
2. **No bit-exact picture reconstruction** — deliberately excluded to maximise
   implementation flexibility.
3. **No lossless coding** — deliberately excluded from v1.
4. **Human vision only** — machine consumption deferred despite being in the original
   scope.

### Named directions for v2

- Improved synthetic-content coding (explicitly "likely one of the focus points").
- Bit-exact reconstruction and lossless coding.
- **Implicit neural representations / online (per-image) training**, **diffusion models**,
  **transformer architectures** — to be monitored for "clear evidence of advances in terms
  of compression efficiency and complexity".
- **Machine consumption of the latent.** The strongest idea in the paper's forward look:
  `ŷ` is available at the decoder *before* pixel reconstruction, and can feed object
  detection, recognition, segmentation, super-resolution, denoising, colour correction
  **directly**. Lower complexity than decode-then-analyse, and "in some cases, with higher
  accuracy, particularly at lower quality settings" — because you skip the lossy
  pixel-domain round trip.

### My critical read

**What is genuinely novel here is not the neural architecture — it is the engineering
discipline.** The core is a 2018 scale-hyperprior VAE with a 2021-style checkerboard
context model. Nothing in it would surprise a compression researcher. What *is* new:

- **A bit-exactness boundary placed exactly where the feedback loop is**, with an integer
  network and a formal overflow proof, and float freedom everywhere else. This is the piece
  the research literature completely ignores and without which no learned codec could ever
  ship as a standard.
- **Three decoders on one codestream.** A genuinely unusual normative structure.
- **Splitting the hyper decoder in two** so the entropy engine is CPU-local — a coding-gain
  sacrifice made purely for deployability.
- **Tools that exist only because train-time and test-time objectives differ** (RVS, LSBS).
  These are, frankly, learned patches over a metric mismatch. They work (2.2% is a lot) but
  they are a symptom: nobody knows how to train directly on seven perceptual metrics.
- **Refusing anything undemonstrable on a real phone.** Pixel shuffle restricted because
  some NPUs handle it badly; the display-window mechanism added because dynamic
  intermediate shapes break static graph compilers. This is what makes it a standard rather
  than a paper.

**Where I would push back:** the "up to 20% better than VVC" headline needs the asterisk
that it is an average over seven metrics of which two go the wrong way, that Kodak at SOP
gives only 7.5%, and that the anchor VTM-11.1 is a 2020-era software encoder configuration.
And the gap between 14 kMAC/pxl (SOP) and 215 (HOP) for ~6 BD-rate points is a very steep
price — the interesting engineering question, which the paper does not answer, is what
sits at 50–80 kMAC/pxl.

---

## One-page summary to memorise

- **Backbone:** VAE + scale-and-mean hyperprior, YCbCr, split luma (160-ch latent) /
  chroma (96-ch latent) branches, latents at 1/16 resolution, hyper latents at 1/64.
- **Coded quantities:** `ẑ` (hyper), and `r̂ = round(y − p̈)` (residual w.r.t. a
  hyperprior-predicted mean). Never the latent directly.
- **Entropy coder:** me-tANS, table-driven, ~100 KB tables, FILO, escape coding for large
  values, skip mode (up to 80% of samples) with a 16³ cube-based override, multithreaded
  substreams.
- **Context model:** MCM — 4-stage 2×2 checkerboard, luma only.
- **Bit-exactness:** mandated up to the end of entropy decoding, via an integer hyper scale
  decoder (8-bit multipliers, 32-bit accumulators, proven overflow-free). Free thereafter.
- **Three decoders on one stream:** SOP 14 / BOP 28 / HOP 215 kMAC/pxl →
  −16.2% / −20.2% / −22.1% BD-rate vs VVC Intra (encoderID 0).
- **Tools:** RVS (2.2%), LSBS (0.4%), LEF (0.3%), ICCI (0.2%), EFE linear + nonlinear
  (chroma PSNR +12%/+8%).
- **Rate control:** 4 ONNX models (~20× bpp range) × channel-wise gain × spatial quality
  map (3D gain) → continuous rate ladder and RoI.
- **Functionality:** synthesis tiling (memory), region partitioning (random access),
  progressive decoding (zero out residuals), arbitrary sizes (per-layer padding/cropping
  *or* pad-to-64 display window), HDR/wide-gamut metadata.
- **5 parts:** core coding / profiling / reference software / conformance / file format
  (ISOBMFF + HEIF, incl. Motion JPEG AI).
- **Not in v1:** lossless, bit-exact reconstruction, good screen content, machine-vision
  tasks.
