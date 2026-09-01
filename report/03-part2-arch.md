<div class="page-break"></div>

## 6. The JPEG AI architecture, as the standard specifies it

*This chapter is our reading of the paper, reorganised for an implementer. Everything here is
tagged PAPER unless marked otherwise; the constants the paper omits are in chapter 9 and
Appendix E.*

### 6.1 How the standard came to exist

| when | what |
|---|---|
| 2019 | WG1 launches the "JPEG AI" exploration activity |
| Nov 2019 | metric analysis — the seven-metric set is fixed (WG1 N85013) |
| Jan 2022 | final Call for Proposals, WG1 N100095 |
| Jul 2022 | **seven** CfP responses evaluated |
| — | two selected as the basis: Bytedance (ref [32]) and Huawei (ref [33]) |
| Oct 2022 | the two are harmonised into **Verification Model 1.0** |
| Jan 2023 | Common Training and Test Conditions fixed (WG1 N100421) |
| 2025 | Parts 1, 3 and 5 finalised; Part 2 in draft; Part 4 targeted for Oct 2025 |
| Feb 2026 | the overview paper — this project's subject — is published |

Two things are worth noting about that history. First, the standard is a **harmonisation of two
independent proposals**, which is visible in the architecture: several tools look like they were
each somebody's contribution rather than parts of one plan. Second, the metric set was fixed in
**2019**, before the architecture existed — so the codec was optimised for a target chosen in
advance rather than a target chosen to flatter it. That is good methodology and it is worth
saying, because it is the reason the seven-metric average is a meaningful number at all.

**The five parts (the paper's Table I):**

| Part | Name | ITU-T \| ISO/IEC | Status per the paper |
|---|---|---|---|
| 1 | Core coding system | T.840-1 \| 6048-1 | finalised 2025 |
| 2 | Profiling | T.840-2 \| 6048-2 | draft — 1 stream profile, 3 decoder profiles |
| 3 | Reference software | T.840-3 \| 6048-3 | finalised 2025 — `gitlab.com/wg1/jpeg-ai/jpeg-ai-vm` |
| 4 | Conformance | T.840-4 \| 6048-4 | target Oct 2025 |
| 5 | File format | T.840-5 \| 6048-5 | finalised 2025 — ISOBMFF ("Motion JPEG AI") + HEIF |

Four sets of trained model parameters are distributed in **ONNX** format via a link in Part 1.
Part 3 — the reference software — is the one we could reach, and it is the reason chapter 9
exists.

**Real-device evidence the paper cites:** 1024×1024 decoded in **under 20 ms** on a smartphone
(refs [36], [37]); **4K in about 190 ms** (ref [38]). These are the numbers that make the
"deployable" claim credible, and they are why the complexity discussion in §6.10 is not
academic.

### 6.2 The three decisions that shape everything

**Decision 1: only the decoder is normative.** The standard fixes the decoder's architecture,
weights and — in the entropy path — its exact integer arithmetic. The encoder is entirely free.
A better encoder can be deployed tomorrow and every existing decoder will read its files. Note
the striking structural consequence we found in the reference software (§9.1): there are **three
decoders but only two encoders** — the simplest operating point has no encoder of its own and
reuses the middle one's.

**Decision 2: bit-exactness is required only up to the entropy decoder's output.** Full
bit-exact reconstruction would force every implementation to reproduce every float operation
identically, which would kill hardware diversity. So the standard draws the line exactly where
it must: `r̂` and `ẑ` — the entropy-decoded integers — must be bit-exact on every device, because
a one-bit error there desynchronises the coder and destroys the rest of the stream. Everything
after that point (the synthesis transforms, the filters, the colour conversion) may differ in
the last few bits, and the image will be imperceptibly different rather than broken. The price
is that JPEG AI has **no bit-exact reconstruction and no lossless mode** — a stated v1
limitation.

**Decision 3: one codestream, three decoders.** SOP / BOP / HOP at 14 / 28 / 215 kMAC/pixel all
read the same file. The `synthesis_transform_id` field turns out to be a **cumulative capability
list** rather than a selector — SOP signals `[0]`, BOP `[1,0]`, HOP `[2,1,0]` — so a HOP stream
is decodable by an SOP decoder. That is the mechanism behind the scalability claim, and it is
not obvious from the paper; we found it in the reference software's profile configuration.

### 6.3 The encoder, step by step

```
  x (RGB)
    │
    ├─ colour transform → YCbCr BT.709,  internal subsampling 4:4:4 / 4:2:2 / 4:2:0
    │
    ├─────────────── PRIMARY (luma) ───────────────┐   ┌────── SECONDARY (chroma) ──────┐
    │                                              │   │                                │
    │  x_Y  [1, H, W]                              │   │  x_UV [2, H/c_v, W/c_h]         │
    │    ↓ g_a  (4 × stride-2)                     │   │    ↓ g_a,sec (one fewer stage   │
    │  y_Y  [160, Ḣ/16, Ẇ/16]                      │   │              at 4:2:0)          │
    │                                              │   │  y_UV [96, Ḣ/16, Ẇ/16]          │
    │    ↓ h_a  (2 × stride-2, on |y|)             │   │    ↓ h_a,sec  (sees y_Y too)    │
    │  z_Y  [160, Ḣ/64, Ẇ/64]  → round → ẑ_Y       │   │  z_UV → round → ẑ_UV            │
    │                                              │   │                                │
    │    ↓ h_s (prediction head)                   │   │    ↓ h_s,sec                    │
    │  p̈_Y  [640, Ḣ/32, Ẇ/32] → PixelShuffle ×2    │   │  p̈_UV [384,…] → shuffle         │
    │       → [160, Ḣ/16, Ẇ/16]                    │   │       → [96, Ḣ/16, Ẇ/16]        │
    │                                              │   │                                │
    │    ↓ h_s,scale (INTEGER network)             │   │    ↓ h_s,scale,sec              │
    │  Iσ_Y [160, Ḣ/16, Ẇ/16]  integer, log domain │   │  Iσ_UV [96,…]                   │
    │                                              │   │                                │
    │  MCM: 4 stages refine p̈_Y                    │   │  eq (1) directly:               │
    │  r̂_Y = round(y_Y − p̈_Y) per stage            │   │  r̂_UV = round(y_UV − p̈_UV)      │
    └──────────────────────────────────────────────┘   └────────────────────────────────┘
                                    │
                       me-tANS entropy coding, four streams:
                       ẑ_Y, ẑ_UV  (one shared SOZ substream)
                       r̂_Y (SORp),  r̂_UV (SORs)
                                    │
                             codestream (§6.7)
```

Two details in that diagram that are easy to miss and that we confirmed from the paper's Figure
1 rather than inferred:

- **the hyper encoder takes `y`, not `x`.** It compresses the *latent's* statistics, not the
  image's.
- **`abs_in_hyperprior = 1`**: the hyper encoder consumes `|y|`, the absolute value. It is
  modelling magnitude, which is what a scale parameter needs.

### 6.4 Split hyper decoders — the detail that is not cosmetic

The residual-coding rung of §5.6.4 needs two outputs, `μ` and `σ`. A research implementation
produces both from **one** network with `2N` output channels. JPEG AI uses **two separate
networks**:

| network | output | arithmetic |
|---|---|---|
| hyper **decoder** | `p̈` — the prediction | float, and *not* bit-exactness-critical |
| hyper **scale decoder** | `Iσ` — an **integer** log-domain scale index | **integer, normative, bit-exact** |

The reason is Decision 2. `Iσ` selects the CDF table row that the entropy coder uses, so it must
be identical on every device or the stream desynchronises. `p̈` is only added back to the decoded
residual, so a last-bit difference there produces an imperceptibly different image. **Splitting
the network is what allows the bit-exactness requirement to be confined to the small integer
half.** The paper backs this with an overflow proof (its ref [39]) showing 8-bit multipliers and
32-bit accumulators cannot overflow.

We measured the cost of the split, and it is essentially free — a finding worth having because
it is counter-intuitive that two networks would be cheaper than one:

| model | parameters | decoder kMAC/pixel |
|---|---|---|
| fused (one network, `2N` out) | 4,700,451 | 129.2 |
| **split (two networks)** | **4,575,603** | **128.9** |

The split is *smaller*. Why: the scale head only has to produce a coarse 32-level quantity, so
it can be far narrower than the prediction head. Our two `h_s` heads cost 0.49 kMAC/pixel
against a fused 3.33 — **6.8× cheaper** — which hand-arithmetic confirms (81 + 81 + 324 = 486
MAC/pixel for the split against 337.5 + 2025 + 972 = 3334 for the fused). And we measured the
scale decoder's accuracy cost at **0.055%** of rate.

#### 6.4.1 The σ index, and why it must be an integer

`Iσ` is an integer index into a **log-spaced** grid of 32 scales:

```
    σ(Iσ) = σ_min · exp( log_k · Iσ / 2^p )

      σ_min = 0.11        σ_max = 54.82        levels = 32     [CONFIRMED]
      p = sigma_precision = 7                                  [CONFIRMED]
      log_k = (ln 54.82 − ln 0.11) / 31 = 0.200365
      max_index = (32 − 1) × 2⁷ − 1 = 3967
```

So `Iσ` carries **7 fractional bits**, and the CDF table row — the σ-class — is those bits
removed. That is what explains the `[…, 3968]` extent of the standard's RVS and LSBS tables:
`[0, 3967]` is exactly 3968 entries.

Three things we established about this that the paper does not state, each of which would have
been a silent bug:

**(a) The row is `ceil(Iσ / 2⁷)`, not `Iσ >> 7`.** Under a right shift, `3967 >> 7 = 30`, which
leaves CDF row 31 **unreachable for every possible index** — in a design whose entire purpose is
a small table. Under round-up, `ceil(3967/128) = 31 = levels − 1`, so the maximum index lands
exactly on the last row. The two rules differ on every index that is *not* an exact multiple of
128 — that is 3,937 of 3,968 — so this was not a cosmetic slip.

**(b) The largest denotable σ is 54.734, not 54.82.** A consequence of `max_index = 3967` rather
than 3968. Reaching 54.82 would need `Iσ = 3968`, which the range excludes. So the top of the
grid is never *denoted* by an index; it is only ever selected as a *row*, which is precisely
what round-up does and round-down cannot. It corroborates (a) from a second direction.

**(c) There is a stronger reason for the integer than the paper gives.** The paper motivates
`Iσ` on storage and on the tables that index by it. We found a harder reason by measurement.
Computing the row two ways — exact integer arithmetic on `Iσ`, versus reconstructing the float σ
and searching a scale table — agrees on 3,957 of 3,968 indices and **disagrees on 11**:

```
    256, 1152, 1280, 1536, 1664, 2176, 2304, 2560, 3200, 3328, 3456
```

Every one is an exact multiple of 128 — an index sitting precisely on a grid point, the only
place a single bit can decide a comparison. The cause is **one float32 ULP** between two
computations of the same exponential:

| | value |
|---|---|
| `σ(256)` via `σ_min · exp(log_k · 2)` in torch | 0.164220**72052955627** |
| `scale_table[2]` via numpy `exp(linspace(…))` → float32 | 0.164220**70562839508** |

The table search counts entries strictly below σ, so that last bit pushes it to row 3 while
exact integer arithmetic says row 2. Neither row is *unsafe* — row 3 is merely wider — but they
are **different**, and a bitstream whose encoder used one rule and whose decoder used the other
decodes to the wrong latent for 0.28% of its symbols, **with both sides reporting success**.

So the argument for the integer index is not that it saves space. It is that integer arithmetic
is exact on every device and in every build, while a float reconstruction is at the mercy of how
each side happened to compute an exponential. **The standard's cross-device bit-exactness
requirement is unachievable through the float path.** Our code enforces this structurally: the
codec always passes the integer row explicitly and never derives it from a float σ, and a test
pins the exact 11 exceptions *and* pins the corruption — encode with one rule, decode with the
other, assert the latent comes back wrong.

#### 6.4.2 What 32 levels costs, measured

The training loss evaluates rate at the *continuous* σ the network predicts. The coder can only
index one of 32 rows. That difference is real rate the loss never sees, so we measured it rather
than assuming it small. Excess is `KL(p_σ ‖ p_σ-quantised)` in bits per symbol, averaged over
4,000 σ drawn log-uniformly on [0.11, 54.82]:

| levels | log step | excess (bits/symbol) | ratio to next |
|---|---|---|---|
| 8 | 0.8873 | 0.21345 | — |
| 16 | 0.4141 | 0.05493 | 3.89× |
| **32** | **0.2004** | **0.01464** | 3.75× |
| 64 | 0.0986 | 0.00365 | 4.01× |
| 128 | 0.0489 | 0.00093 | 3.92× |
| 256 | 0.0244 | 0.00023 | 4.04× |

The right-hand column is the point: **every halving of the step quarters the cost.** The error is
second-order in the grid step, exactly as a Taylor expansion of the rate around σ predicts. So 32
levels is not an arbitrary point on a flat curve — it sits where halving the table again would
still buy about 0.011 bits/symbol and one more halving only 0.003.

**End-to-end confirmation:** the full codec on Kodak measures **+1.86% to +1.92%** over the
loss's own estimate. At its ~1.07 bits/symbol that is +0.0199 bits/symbol against +0.01464
predicted here — the same effect at the same magnitude, differing because the real σ
distribution is not log-uniform.

Three things follow, all load-bearing for the rest of this report:

1. **It costs rate; it never escapes.** Rounding σ *up* means the coder's distribution is always
   at least as wide as predicted, so nothing the model thought likely can fall outside the
   table. Rounding down would trade this ~1.9% for out-of-range symbols, which is a far worse
   deal: a deliberately miscalibrated run at 0.63% escapes measured **−17% rate** — a bitstream
   that looks 17% *better* than the model while being broken.
2. **Reported bitrates must come from actual bytes.** The loss's estimate is optimistic by
   exactly this amount, so an RD curve built from it would claim a codec ~1.9% better than the
   one that exists.
3. **The round-trip gate must threshold against the σ-quantised estimate**, not the continuous
   one. Against the quantised σ our coder agrees to −0.11%, which is what actually tests the CDF
   construction. Our original plan said "estimate within 1–2% of actual", which would have
   flagged a *correct* codec as broken.

### 6.5 The Multi-Context Model (§VI-D)

Four stages over the four cosets of a 2×2 spatial tile, order `(0,0) → (1,1) → (0,1) → (1,0)`:

```
            ┌───────┬───────┐
            │   1   │   3   │        1 = (0,0)  coded with no context
            ├───────┼───────┤        2 = (1,1)  uses 1
            │   4   │   2   │        3 = (0,1)  uses 1, 2
            └───────┴───────┘        4 = (1,0)  uses 1, 2, 3
```

Per stage: the hyper decoder's `4N`-wide output is divided into four `N`-wide slices, one per
stage — **which is why `p̈` is 4 × `N` channels wide**. Stage 0 uses its slice directly; stages
1/2/3 concatenate every previously reconstructed residual, pass it through a `1×1` convolution
of `k·N → N` channels for `k = 1/2/3`, then a grouped `3×3`.

Properties: exactly **four** sequential passes for any image size (constant latency); **luma
only**; partitions space, never channels.

The reference software handles odd latent sizes with two guards that turn out to be a *proof* of
the stage order — a stage needs a row guard iff its vertical offset is 1, and a column guard iff
its horizontal offset is 1, and the guards are on stages `{1,3}` and `{1,2}` respectively, which
forces stage 1 = (1,1), stage 2 = (0,1), stage 3 = (1,0), stage 0 = (0,0). Same answer as the
shuffle code, derived from unrelated concerns. §9.3.

### 6.6 The entropy coder as specified: me-tANS

Algorithm 1 of the paper, transcribed:

```
Init:  flatten Iσ[] to 1-D
       point at the LAST symbol position     (pointer moves BACKWARDS — FILO)
       s ← parse 8 bits

for i = 0 .. n_symbols/4:
    # fast path — 4 symbols, pure table lookups
    for j = 0..3:
        k = 4i + j
        r̂[k] ← transition_table_symbol   [ Iσ[k] ][ s ]
        n    ← transition_table_nBits    [ Iσ[k] ][ s ]
        v    ← parse n bits
        s    ← transition_table_stateNext[ Iσ[k] ][ s ] | v
    # escape path — same 4 symbols
    for j = 0..3:
        k = 4i + j
        if r̂[k] + bound_table[ Iσ[k] ] == 0:
            ind  ← parse 1 bit
            m    ← ind ? 2 : 15
            v    ← parse m bits
            sign ← parse 1 bit
            r̂[k] ← bound_table[ Iσ[k] ] + v × sign
reorganise r̂[] to 3-D
```

Note the structure: the fast path is four table lookups with no arithmetic, and the escape path
is a *separate* loop over the same four symbols. That separation is what lets a hardware
implementation pipeline the common case. Table selection is by `Iσ` for residuals and by channel
index for hyper samples.

**Skip mode:** if `Iσ[c,i,j]` is below a threshold (`thr_skip = 382`, CONFIRMED), the residual is
not coded at all and is inferred zero. Up to **80%** of residual samples may be skipped. It is
overridable per **16×16×16** cube by a max-pool test, so a genuinely detailed region can opt back
in.

**Multithreading:** each of `ẑ_Y`, `ẑ_UV`, `r̂_Y`, `r̂_UV` and the quality map can be split into
independent substreams, with offsets at the segment start and counts in the picture header. And
even single-threaded, **two substreams and two ANS states are interleaved** — the paper's phrase
is "mimicking a dual-threaded setup".

**What we use instead, and why that is legitimate.** Our coder is **rANS** (`jpegai/coder/`),
which is mathematically equivalent — same Shannon-limit performance, same CDF tables, different
state mechanics. We cannot build the real me-tANS because its transition tables are *derived
from* the normative CDF tables, which are in the paywalled Part 1. What we *do* have is every
me-tANS constant (§9.4), so phase 9 is a table-construction exercise rather than a research
problem. The distinction matters for honesty: our **rates are correct** (both coders hit the same
entropy) but our **codestream is not conformant**.

### 6.7 The codestream (§V)

Marker-segment structure, exactly like every JPEG: `marker (2 bytes) | size | payload`, with
byte alignment after the size field and after the payload.

| Symbol | Code | Payload | M/O |
|---|---|---|---|
| SOC | `0xff80` | start of codestream | **M** |
| EOC | `0xff81` | end of codestream | **M** |
| PIH | `0xff82` | picture header | **M** |
| TOH | `0xff83` | tools header | O |
| RDI | `0xff84` | rendering information (HDR, colour volume) | O |
| SOZ | `0xff88` | `ẑ_Y` **and** `ẑ_UV` streams | **M** |
| SORp | `0xff89` | `r̂_Y` stream | **M** |
| SORs | `0xff8a` | `r̂_UV` stream | **M** |
| SOQ | `0xff8b` | quality-map information | O |
| UDI | `0xff8c` | user-defined information | O |

Three implementation observations we derived from this table:

1. **`0xff85`–`0xff87` are unassigned.** A conformant parser must skip unknown markers by
   length, not reject them, or it will break on a future extension.
2. **SOZ carries *both* hyper streams**, so four logical substreams live in three marker
   segments. The reference software confirms it: the primary and secondary `z` markers are the
   same value, so `ẑ_Y` and `ẑ_UV` share one threaded substream.
3. **Six mandatory markers** ⇒ the minimal legal codestream is
   `SOC · PIH · SOZ · SORp · SORs · EOC`. That is the smallest thing a conformance decoder must
   accept, and therefore the first thing to implement in phase 9.

The picture header carries: profile and level IDs; picture size; output bit depth; internal
subsampling (`c_ver_minus1`, `c_hor_minus1`) and output subsampling (`s_ver_minus1`,
`s_hor_minus1`); model indices; `decoderID`; `colour_transform_idx` (plus a 3×3 matrix and
3-vector when it is 2); region and tile partitioning; substream counts; the RVS flags;
`betaDisplacementLog` per component; the 3-D gain flag; skip-mode parameters; and the display
window. **If the TOH is absent, all its flags are inferred 0** — so post-filters and LSBS are off
by default, which is the correct default for the simplest decoder.

### 6.8 The coding tools, and what each is worth

The paper's Table IV is a tool ablation. Subtracting each row from the all-on row gives each
tool's contribution — a derivation the paper does not print, and it is more informative than the
table:

| tool | BD-rate value | cost (kMAC/pixel) | verdict |
|---|---|---|---|
| **RVS** — residual & variance scaling | **2.2 pp** | ~0 | free gain; the clear winner |
| **LSBS** — latent scaling before synthesis | 0.4 pp | 0.1 | cheap, keep |
| **LEF** — latent enhancement filter | 0.3 pp | ~0 | cheap, keep |
| **ICCI** — inter-component correlation improvement | 0.2 pp | **4.6** | **a bad trade — 17% of the BOP decoder budget for 0.2 pp** |
| **EFE linear** | **−0.2 pp** | −0.9 | *negative* BD-rate; kept for **+12%** chroma PSNR |
| **EFE nonlinear** | **−0.2 pp** | ~0 | *negative* BD-rate; kept for **+8%** chroma PSNR |

Three readings worth stating:

- **RVS pays where it was designed to.** It moves FSIM by 6.7 points and VMAF by 6.1, while
  MS-SSIM barely moves. It is a perceptual tool and it behaves like one.
- **ICCI is the only tool with real compute cost**, and the paper does not call out that its
  cost/benefit is poor. On a phone this would be the first thing to disable.
- **Both EFE variants have negative BD-rate value and are kept anyway.** They are retained for
  chroma PSNR, which the seven-metric average barely sees (six of the seven metrics are
  luma-only). This is a legitimate decision — chroma artefacts are visible even when the metrics
  do not punish them — but it is a decision made *against* the stated objective, and it is worth
  knowing about.

**Two anomalies in the paper's own Table IV**, which we found by arithmetic and which set a
noise floor on every timing number in the paper:

1. "EFE linear off" has **higher** kMAC/pixel (28.6) than all-on (27.7). Turning a tool off
   should not add compute.
2. **Every** ablation decodes *faster* than all-on, including the ones with identical MAC counts.
   Since disabling a free tool cannot speed up the decoder, this implies at least ~6% timing
   noise — so **differences below roughly 15 ms/megapixel in the paper are not real**.

The RVS equations, for completeness — the pooling, the variance offset, the residual scale:

```
    (7)  σ_Y[c,i,j] = ( 32 + Σ_{i'=0}^{7} Σ_{j'=0}^{7} Iσ_Y[c, 8i+i', 8j+j'] ) >> 6
                       boundary padding value = 1411
    (8)  Iσ_Y[c,i,j] += T1[ modelID, id[c], σ_Y[c, ⌊i/8⌋, ⌊j/8⌋] ]
    (9)  r̂_Y[c,i,j]  =  r̂_Y[c,i,j] · T2[ modelID, id[c], σ_Y[…] ] / 2¹⁶
```

with `T1[4,4,3968]` and `T2[4,4,3968]` normative. And LSBS:

```
         μ_Y[c,i,j] = ŷ_Y[c,i,j] − r̂_Y[c,i,j]
    (10) ŷ_Y[c,i,j] += ( r̂_Y·TR[modelID, σ_Y] + μ_Y·TP[modelID, σ_Y] + 2¹² ) >> 13
```

with `TP[4,3968]`, `TR[4,3968]`. All four tables are **learned weights** distributed inside the
reference software's Git-LFS checkpoints, not source literals — so they are on our "cannot
obtain" list (§7.2) and phase 10 learns its own.

### 6.9 Variable rate from four models

A standard cannot ship 40 models. JPEG AI ships **four** and reaches an 18-point ladder:

```
    (11) mlog[comp,c,i,j] = betaDisplacementLog[comp] + mref[modelID, comp, c]
    (12) if gain_3D_enable_flag:  mlog[comp,c,i,j] += Gain3d[i,j]
    (13) m⁻¹ = exp( −mlog · step / 2^sigmaPrecision )
    (14) r̂_Y ·= m⁻¹[0,·]     r̂_UV ·= m⁻¹[1,·]
    (15) Iσ_Y += mlog[0,·]   Iσ_UV += mlog[1,·]
```

Three mechanisms stacked: a learned **per-channel** gain vector `mref` (some channels matter more
than others, and the network knows which); a signalled scalar **displacement** per component
(this is the quality dial); and an optional **3-D gain map** from the SOQ segment, which is how
region-of-interest coding is done — paint a quality map, and eq. (12) spends more bits there.

`exp()` may be a lookup table. Note eqs (14) and (15) together: scaling the residual *and*
offsetting the σ index keeps the entropy model consistent with the scaled residual, which is
what makes this work without retraining.

The reference software's own ladder is 18 β values from 0.0002 to 3.0, and three of the paper's
four base models (0.002, 0.075, 0.5) are ladder entries. **The fourth, β = 0.012, is not** — the
ladder brackets it with 0.01 and 0.015. Either the paper rounds, or trained base models are not
required to coincide with ladder entries.

### 6.10 Operating points, profiles and levels

| decoderID | Name | Target | Upsampling | kMAC/pxl | Allowed layers |
|---|---|---|---|---|---|
| 0 | **SOP** | laptops without NN acceleration, mid/low-end mobile | 2×2 conv + pixel shuffle | **14** | conv, ReLU, ReLU6 |
| 1 | **BOP** | high-end mobile (NPU/GPU) | 4×4 deconv; shuffle only in the last layer | **28** | conv, ReLU, ReLU6 |
| 2 | **HOP** | desktop GPU | richer | **215** | unrestricted |

| Decoder profile | Supported decoderIDs |
|---|---|
| Main@Simple | 0 |
| Main@Base | 0, 1 |
| Main@High | 0, 1, 2 |

The **layer restriction** on decoders 0 and 1 — convolution, ReLU and ReLU6 only — is a
standardisation decision, not a compression one. It exists so that a fixed-function NPU can run
the decoder. Levels gate available models and picture size, from 6.2 to 398 megapixels.

### 6.11 Functionality features

These are what separate a codec from a research model, and each is a real engineering
requirement:

- **Arbitrary image sizes.** Pad to a multiple of **64 = 2⁶** — the total stride of analysis
  (2⁴) plus hyper encoder (2²) — then crop the reconstruction using the signalled display
  window. Layer-based cropping is also available, padding and cropping per stage instead.
- **Synthesis tiling.** Decode the image in tiles to bound peak memory. A phone cannot hold a
  full 4K activation tensor.
- **Region partitioning.** Multiple SORp/SORs segments with distinct `region_idx`, either
  offset-based dependent regions or marker-based independent ones. Independent regions give
  random-access crop decoding — decode a thumbnail region without touching the rest.
- **Progressive decoding.** Zero part of `r̂`, or truncate the codestream, and you get a valid
  lower-quality image. No separate encoding pass required.
- **Region-of-interest coding.** Eq. (12) plus the SOQ quality map.
- **HDR and wide gamut.** Carried as metadata in RDI — colour primaries, transfer
  characteristics, matrix coefficients, mastering display volume, dynamic metadata.

### 6.12 The paper's own results

**Table III — main results, BD-rate against VVC Intra (VTM-11.1) on the JPEG AI test set (50
images, 1K–4K).** Negative = JPEG AI better.

| encID | decID | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS | kMAC/pxl | ms/MPx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | **−16.2%** | −30.9 | +6.9 | −22.3 | −13.4 | −26.6 | −30.0 | +2.9 | 14 | 285 |
| 0 | 1 | **−20.2%** | −33.0 | +1.4 | −26.9 | −17.3 | −29.1 | −34.8 | −1.9 | 28 | 266 |
| 0 | 2 | **−22.1%** | −34.8 | −2.0 | −27.7 | −19.3 | −31.2 | −37.3 | −2.5 | 215 | 323 |
| 1 | 0 | **−14.4%** | −30.3 | +9.7 | −20.4 | −11.6 | −25.1 | −29.2 | +6.1 | 14 | 246 |
| 1 | 1 | **−19.9%** | −33.0 | +1.5 | −26.4 | −16.7 | −28.4 | −35.8 | −0.8 | 28 | 271 |
| 1 | 2 | **−27.0%** | −37.6 | −8.5 | −34.7 | −23.6 | −33.7 | −42.4 | −8.8 | 215 | 332 |

Hardware: NVIDIA Tesla V100 32 GB, Intel Xeon Platinum 8336C @ 2.30 GHz.

Four readings:

- **Encoder choice depends on the target decoder.** encID 0 wins with decoders 0 and 1; encID 1
  wins *decisively* with decoder 2 (−27.0 against −22.1). A single "best encoder" does not exist,
  which is the freedom Decision 1 bought.
- **VIF and PSNR-HVS are the weak metrics** — both positive at the simplest decoder. MS-SSIM and
  VMAF carry the win. Our own results show the same shape, which is reassuring: our weaknesses
  are the architecture's weaknesses, not our bugs.
- **15× the MACs costs 13% more time.** 285 → 323 ms/MPx for 14 → 215 kMAC/pixel. The GPU is not
  compute-bound; sequential entropy decoding dominates. On mobile the ranking inverts, which is
  the entire reason three decoders exist.
- **The paper's prose disagrees with its own table.** §VII-B quotes 16.0 / 20.2 / 21.1 and 13.9 /
  19.7 / 27 and 13 kMAC/pixel; the table says 16.2 / 20.2 / 22.1 and 14.4 / 19.9 / 27.0 and 14.
  **Cite the table.**

**Table V — Kodak (24 images, 768×512).** This is our comparable dataset.

| decID | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS |
|---|---|---|---|---|---|---|---|---|
| 0 | **−7.5%** | −29.8 | +18.1 | −19.7 | −0.2 | −24.1 | −22.3 | **+25.3** |
| 1 | **−12.9%** | −32.1 | +11.4 | −22.9 | −6.3 | −26.8 | −28.4 | +14.5 |
| 2 | **−21.1%** | −37.3 | 0.0 | −28.8 | −15.6 | −32.0 | −38.3 | +4.4 |

**Table VI — CLIC 2024 validation.**

| decID | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS |
|---|---|---|---|---|---|---|---|---|
| 0 | −12.1% | −25.7 | +22.6 | −30.8 | −7.6 | −25.0 | −25.4 | +7.3 |
| 1 | −16.8% | −28.4 | +15.4 | −34.6 | −12.2 | −27.9 | −32.0 | +1.9 |
| 2 | −24.9% | −34.5 | +2.8 | −42.3 | −19.9 | −33.5 | −40.7 | −6.3 |

**The dataset penalty, and why it changed our target.** Comparing Table III with Table V at
matched decoder:

| decoderID | test set | Kodak | penalty |
|---|---|---|---|
| 0 | −16.2% | −7.5% | **8.7 pp** |
| 1 | −20.2% | −12.9% | **7.3 pp** |
| 2 | −22.1% | −21.1% | 1.0 pp |

Kodak is *much* harder for a learned codec at the simplest decoder, and the gap closes as the
decoder gets richer. Why: Kodak's images are small (768×512), so there is less spatial context
for the transform to exploit and the hyperprior's overhead is relatively larger. Three
consequences for this project:

1. **Our honest target is −7.5%**, not −16.2%. Using the headline number would have overstated
   our shortfall by 8.7 points.
2. **PSNR-HVS at +25.3% on Kodak is normal, not a bug.** When our own run produced +37.8% we
   knew to look at the *magnitude* rather than the sign — the paper's own codec is positive there
   too.
3. **FSIM inverts between datasets by 10+ points** (−19.7 on Kodak against −30.8 on CLIC). Any
   single-metric conclusion is dataset-dependent.

### 6.13 Subjective results and profiling

§VII-D reports a subjective comparison at approximately **0.08 bpp** and **0.3 bpp** with
**post-filters disabled**, on a 36-image synthetic set (animation, screen content, game
capture). §VIII profiles the three decoder profiles on real hardware. The profiling section is
what supports the deployability claim; the subjective section is where the paper is most candid
about screen content, which is a stated weakness.

### 6.14 Stated limitations of version 1, and version 2 directions

The paper's own list, which any report on JPEG AI should repeat:

1. **Synthetic and screen content:** basic support only, no dedicated tools or models. Worse
   than traditional codecs on screen-captured text.
2. **No bit-exact reconstruction** — deliberate, for implementation flexibility (Decision 2).
3. **No lossless coding.**
4. **Machine consumption of the latent deferred to v2**, despite being in the original scope.

Named v2 directions: better synthetic content; bit-exact and lossless modes; implicit neural
representations and online training; diffusion models; transformer architectures; and
machine-vision tasks run **directly on `ŷ`** (detection, recognition, segmentation,
super-resolution, denoising, colour correction) at lower complexity and — the paper's own phrase
— "in some cases higher accuracy, particularly at lower quality settings".

That last one is the interesting research direction. If detection can run on the latent, the
decode step is skipped entirely for machine consumers, which is most of the traffic in a
surveillance or autonomous-vehicle context.
