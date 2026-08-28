# Tables and Figures — verified transcription

**Provenance.** Everything below was read from screenshots of the published paper
supplied by the user on 2026-08-26. It **supersedes** the output of
`paper/rasterize.py`, which reconstructed these tables from vector glyph outlines
and could not be independently checked at the time.

Verification result: the rasterizer's Table III numbers were **correct**. That is
worth recording, because it means the rest of `docs/02` — written from the same
extraction — is trustworthy.

An additional internal-consistency check passed on all 19 data rows of Tables
III–VI: the `AVG` column is the **unweighted arithmetic mean of the seven metric
columns**, reproducing the printed value to one decimal place every time. So:

* the seven metrics are equally weighted — no perceptual weighting, no
  normalisation, no dropping of outliers;
* BD-rate is computed **per metric first, then averaged** — never by averaging
  the metrics and computing one BD-rate;
* `jpegai/eval/bdrate.py::bd_rate_table` implements the right convention.

---

## Table I — JPEG AI Parts

| Part | Name | ITU-T Recommendation \| ISO/IEC International Standard |
|---|---|---|
| Part 1 | Core coding systems | T.840-1 \| 6048-1 |
| Part 2 | Profiling | T.840-2 \| 6048-2 |
| Part 3 | Reference software | T.840-3 \| 6048-3 |
| Part 4 | Conformance | T.840-4 \| 6048-4 |
| Part 5 | File format | T.840-5 \| 6048-5 |

Unchanged from the earlier transcription.

---

## Table II — JPEG AI Markers

| Symbol | Code | Payload | M/O |
|---|---|---|---|
| SOC  | `0xff80` | None | **Mandatory** |
| EOC  | `0xff81` | None | **Mandatory** |
| PIH  | `0xff82` | Picture header | **Mandatory** |
| SOZ  | `0xff88` | `z_Y`-stream **and** `z_UV`-stream | **Mandatory** |
| SORp | `0xff89` | `r_Y`-stream | **Mandatory** |
| SORs | `0xff8a` | `r_UV`-stream | **Mandatory** |
| TOH  | `0xff83` | Tools header | Optional |
| SOQ  | `0xff8b` | Quality map information | Optional |
| UDI  | `0xff8c` | User defined information | Optional |
| RDI  | `0xff84` | Rendering information | Optional |

Three implementation-relevant observations the table makes that the prose does not:

1. **`0xff85`, `0xff86`, `0xff87` are unassigned.** The mandatory markers occupy
   `80–84` plus `88–8a`; the gap sits exactly where a v1 draft would have had
   three more markers. Our parser must therefore treat unknown `0xff85..0xff87`
   as *reserved and skippable* (read length, skip payload), not as an error — that
   is the only forward-compatible reading, and it is free to implement.
2. **SOZ carries both hyper-latent streams in one marker segment.** So there are
   four logical substreams (`z_Y`, `z_UV`, `r_Y`, `r_UV`) but only **three**
   marker segments. The split between `z_Y` and `z_UV` inside SOZ has to be
   either length-prefixed or derivable from the PIH — a question for Part 1 or
   the reference software.
3. **Six mandatory markers, four optional.** A minimal conformant codestream is
   `SOC · PIH · SOZ · SORp · SORs · EOC`. That is our Phase 9 target, and every
   optional marker is a separate, independently testable increment.

---

## Table III — JPEG AI Common Test Conditions (anchor: VTM)

| enc | dec | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS | kMAC/pxl | ms/Mpxl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | **−16.2%** | −30.9% | +6.9% | −22.3% | −13.4% | −26.6% | −30.0% | +2.9% | 14 | 285 |
| 0 | 1 | **−20.2%** | −33.0% | +1.4% | −26.9% | −17.3% | −29.1% | −34.8% | −1.9% | 28 | 266 |
| 0 | 2 | **−22.1%** | −34.8% | −2.0% | −27.7% | −19.3% | −31.2% | −37.3% | −2.5% | 215 | 323 |
| 1 | 0 | **−14.4%** | −30.3% | +9.7% | −20.4% | −11.6% | −25.1% | −29.2% | +6.1% | 14 | 246 |
| 1 | 1 | **−19.9%** | −33.0% | +1.5% | −26.4% | −16.7% | −28.4% | −35.8% | −0.8% | 28 | 271 |
| 1 | 2 | **−27.0%** | −37.6% | −8.5% | −34.7% | −23.6% | −33.7% | −42.4% | −8.8% | 215 | 332 |

Note the column is headed **PSNR-HVS**, not PSNR-HVS-M. **RESOLVED from the QAF
repository:** `ref/jpeg-ai-qaf/metrics.py` calls `psnr_hvsm.psnr_hvs_hvsm()`, which
returns both values, and keeps the **first** — i.e. **PSNR-HVS**. The table heading is
correct and §VII-A's prose is loose. So `PAPER_SEVEN` in `jpegai/eval/metrics.py` uses
`psnr_hvs`; we still compute and report `psnr_hvsm` alongside it, but it is not one of
the seven and is not averaged into AVG. Also confirmed there: it is computed on **Y at
10-bit**, replicate-padded to a multiple of 8, in float64. See
[06-normative-constants.md §11](06-normative-constants.md).

### What this table actually says

* **VIF is JPEG AI's weak metric.** Positive (worse than VTM) at every operating
  point except the highest-complexity decoder, where it barely crosses zero.
  PSNR-HVS behaves the same way. The favourable AVG is carried by MS-SSIM, VMAF,
  IW-SSIM and FSIM.
* **Complexity buys little at the top.** decoderID 1 → 2 is a **7.7×** increase in
  kMAC/pxl (28 → 215) for **1.9 pp** of BD-rate at encoderID 0. The interesting
  engineering point is decoderID 1: 2/3 of the total gain at 13% of the compute.
* **encoderID 1 helps only the largest decoder.** It is *worse* than encoderID 0
  at decoderIDs 0 and 1 (−14.4 vs −16.2, −19.9 vs −20.2) and much better at
  decoderID 2 (−27.0 vs −22.1). So the two encoders are tuned for different
  operating points; there is no universally better encoder.
* **Decode time does not track kMAC/pxl.** decoderID 0 at 14 kMAC/pxl takes
  285 ms/Mpxl while decoderID 1 at twice the MACs takes 266. Entropy decoding —
  serial, CPU-side, MAC-free — dominates at low operating points. This is the
  empirical justification for me-tANS and for skip mode, and it is the single
  most important number in the paper for anyone building a fast decoder.

---

## Table IV — Tools Disable Test (anchor: VTM)

Baseline "All on" corresponds to encoderID 0 / decoderID 1.

| Test | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS | kMAC/pxl | ms/Mpxl |
|---|---|---|---|---|---|---|---|---|---|---|
| All on | −20.2% | −33.0% | +1.4% | −26.9% | −17.3% | −29.1% | −34.9% | −1.9% | 27.7 | 266 |
| RVS off | −18.0% | −33.1% | +0.8% | −20.2% | −14.8% | −29.0% | −28.8% | −0.7% | 27.7 | 249 |
| LSBS off | −19.8% | −33.1% | +1.8% | −27.4% | −17.5% | −29.1% | −30.5% | −2.7% | 27.6 | 249 |
| LEF off | −19.9% | −33.2% | +1.2% | −27.3% | −18.1% | −29.0% | −29.0% | −3.7% | 27.7 | 251 |
| ICCI off | −20.0% | −32.8% | +1.7% | −26.9% | −17.2% | −29.0% | −34.1% | −2.0% | **23.1** | 245 |
| EFE nonlinear off | −20.4% | −33.3% | +1.2% | −26.2% | −17.6% | −29.4% | −35.1% | −2.1% | 27.7 | 246 |
| EFE linear off | −20.4% | −33.3% | +1.1% | −26.1% | −17.6% | −29.4% | −35.2% | −2.2% | **28.6** | 250 |

### Derived tool value (BD-rate cost of switching the tool off)

| Tool | BD-rate gain | kMAC/pxl cost | pp per kMAC/pxl |
|---|---|---|---|
| RVS | **2.2 pp** | ~0 | ∞ — free |
| LSBS | 0.4 pp | 0.1 | 4.0 |
| LEF | 0.3 pp | ~0 | ∞ — free |
| ICCI | 0.2 pp | **4.6** | 0.04 |
| EFE nonlinear | **−0.2 pp** | ~0 | negative |
| EFE linear | **−0.2 pp** | −0.9 | negative |

* **RVS is the whole ballgame**: 2.2 pp of the 20.2 pp total, at no measurable
  compute cost, and it is a lookup-table rescaling of the entropy model rather
  than a network. Our Phase 10 must not treat it as optional polish. Its effect
  is concentrated in FSIM (−26.9 → −20.2, a 6.7 pp swing) and VMAF (−34.9 →
  −28.8, 6.1 pp), i.e. RVS is mostly buying *perceptual* quality, not MSE.
* **ICCI is a bad trade at BOP.** 4.6 kMAC/pxl — **17% of the entire decoder
  budget** — for 0.2 pp. Any complexity-constrained profile should drop it. The
  paper does not draw this conclusion; the table does. Good ablation material for
  our report.
* **Both EFE variants have *negative* BD-rate value** and are kept anyway,
  because section VI-M justifies them on chroma PSNR (+12% linear, +8%
  nonlinear) — a quantity that is not among the seven metrics. This is the
  clearest example in the paper of a tool retained against its own headline
  metric, and it is why `metrics.py` computes PSNR-U and PSNR-V even though the
  paper's tables never report them.

### Two anomalies in this table

1. **"EFE linear off" has *higher* kMAC/pxl than "all on"** (28.6 vs 27.7).
   Disabling a tool should not add compute. Most likely the linear EFE *replaces*
   a more expensive normative chroma upsampling path, so removing it re-enables
   that path. Worth confirming from the reference software; flagged so we do not
   copy the anomaly into our own ablation table.
2. **Every ablation decodes faster than "all on"** (245–251 vs 266 ms/Mpxl),
   including rows with identical kMAC/pxl. A 15 ms/Mpxl spread across
   computationally identical configurations means the timing column carries at
   least ~6% run-to-run noise. Do not read differences below ~15 ms/Mpxl in this
   paper as real. Our own timing harness must report a median over repeats.

Also: Table IV's baseline VMAF is **−34.9%** where Table III's matching row says
**−34.8%**, and kMAC/pxl is 27.7 vs 28. Rounding in Table III, not a
contradiction — but it means Table III's "28" is really 27.7.

---

## Table V — Kodak (anchor: VTM)

| Test | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS |
|---|---|---|---|---|---|---|---|---|
| decoderID 0 | **−7.5%** | −29.8% | +18.1% | −19.7% | −0.2% | −24.1% | −22.3% | **+25.3%** |
| decoderID 1 | **−12.9%** | −32.1% | +11.4% | −22.9% | −6.3% | −26.8% | −28.4% | +14.5% |
| decoderID 2 | **−21.1%** | −37.3% | 0.0% | −28.8% | −15.6% | −32.0% | −38.3% | +4.4% |

(The VIF cell is printed as `18.1%%` in the paper — a typo, not a second unit.)

## Table VI — CLIC 2024 validation (anchor: VTM)

| Test | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS |
|---|---|---|---|---|---|---|---|---|
| decoderID 0 | **−12.1%** | −25.7% | +22.6% | −30.8% | −7.6% | −25.0% | −25.4% | +7.3% |
| decoderID 1 | **−16.8%** | −28.4% | +15.4% | −34.6% | −12.2% | −27.9% | −32.0% | +1.9% |
| decoderID 2 | **−24.9%** | −34.5% | +2.8% | −42.3% | −19.9% | −33.5% | −40.7% | −6.3% |

### These two tables are the most important ones for *our* project

We will evaluate on **Kodak**, because it is 24 images and we can afford it. So
Table V, not Table III, is our yardstick — and it is dramatically less flattering:

| | CTC (Table III) | Kodak (Table V) | penalty |
|---|---|---|---|
| decoderID 0 | −16.2% | **−7.5%** | 8.7 pp |
| decoderID 1 | −20.2% | −12.9% | 7.3 pp |
| decoderID 2 | −22.1% | −21.1% | 1.0 pp |

Three consequences for how we set expectations and read our own results:

1. **Kodak costs the low-complexity decoder more than half its gain.** The
   penalty shrinks to ~1 pp at decoderID 2. Kodak's images are 768×512 — small,
   film-grainy, and high-frequency — which is exactly where a shallow synthesis
   transform cannot compete and where VTM's hand-tuned intra prediction is
   strongest. Tier A is a *reduced-width, low-complexity* configuration evaluated
   on Kodak. **The honest target for our headline number is Table V decoderID 0,
   about −7.5%, not −16.2%.** Quoting −16% as the goal would set up the project
   to look like a failure when it is working correctly.
2. **PSNR-HVS +25.3% on Kodak at decoderID 0 is normal.** If our model shows
   badly negative PSNR-HVS performance, that reproduces the standard rather than
   revealing a bug. Same for VIF at +18%. Knowing this in advance prevents days
   of chasing a non-bug.
3. **FSIM inverts between datasets.** Kodak −19.7% vs CLIC −30.8% at decoderID 0;
   CLIC is *better* on FSIM at every decoder but *worse* on MS-SSIM. So dataset
   choice moves individual metrics by 10+ pp in both directions. Any single-number
   comparison across papers that does not name the dataset is meaningless — we
   will state ours everywhere.

---

## Figure 1 — encoder / decoder block diagram (verified)

The figure confirms the dataflow in `docs/02` §IV. Confirmations that matter,
because each one closes an assumption:

**1. MCM is luma-only, and the chroma path is a bare addition.**
The luma encoder block is labelled *"MCM and quantization"*; the chroma one is
*"Subtraction and quantization"*. On the decoder side, luma goes through *"MCM"*
and chroma through *"Addition"*. Confirms `entropy.mcm_on_secondary: false`, and
confirms that chroma reconstruction is exactly `ŷ_UV = r̂_UV + p̈_UV` — one
tensor add, no context modelling, no serial dependency.

**2. `I_σ` feeds the arithmetic coder only — never the synthesis path.**
In both encoder and decoder, the arrow out of *"hyper scale decoder"* goes to the
*arithmetic encoder/decoder* box, and nowhere else. `p̈` goes to MCM/Addition.
This is the decoupling described in §VI-D, drawn explicitly: the scale exists to
build CDFs, the mean exists to reconstruct. **Architectural consequence: the
hyper *scale* decoder must run before entropy decoding and must be bit-exact,
while the hyper *decoder* only has to run before reconstruction and need not be.**
That is what lets the entropy stage stay on the CPU while the rest goes to an
NPU, and it is the reason for two separate networks off one `ẑ`.

**3. Concatenation takes the luma *latent* `ŷ_Y`, not the decoded image `x̂_Y`.**
The vertical arrow into *"Concatenation"* leaves the `ŷ_Y` line between MCM and
the primary synthesis transform. Confirms eq. (3) and confirms
`channels.secondary_synthesis_in = 96 + 160 = 256`. Also confirms the chroma
branch does **not** wait for the primary synthesis transform — the two synthesis
transforms can run concurrently.

**4. Cross-component information flows luma → chroma only, at exactly two points.**
*Preprocessing* takes both `x_Y` and `x_UV`; *Concatenation* takes `ŷ_Y`. There
is no arrow in the other direction anywhere in the figure. **The luma branch is
completely independent of chroma.** Three things follow, and all three are free:
   * the monochrome fast path (`functionality.monochrome_fast_path`) is trivially
     conformant — you simply never parse SORs;
   * luma decode latency never depends on chroma;
   * we can build, train and evaluate the entire primary branch before writing a
     line of the secondary branch. **This is why the phase ordering in
     `docs/03` is correct** — Phase 4's two-branch work is genuinely additive.

**5. Four arithmetic coders, three marker segments.** Confirms the Table II
reading above.

**6. The hyper encoder takes `y`, not `x`.** Standard hyperprior topology; the
arrow originates at the analysis transform's output.

### Figure labelling slip

All four hyper blocks on the chroma path are labelled *"Primary hyper …"* in the
published figure (*Primary Hyper Encoder*, *Primary hyper scale decoder*,
*Primary hyper Decoder*, twice on each side). They must be the **secondary**
modules: they consume `y_UV`, emit `ẑ_UV` / `I_σUV` / `p̈_UV`, and operate at
different channel counts. Our code names them `secondary_*`. Cosmetic, but worth
recording so nobody later "fixes" our naming to match the figure.

### What Figure 1 deliberately omits

No gain unit, quality map, RVS, LSBS, LEF, ICCI or EFE block appears. Figure 1 is
the *core* pipeline; every tool in §VI-G…VI-M is folded inside these boxes or
sits outside them. Useful confirmation that our layered plan — core first
(Phases 3–6), tools after (Phase 10) — matches how the standard itself is
decomposed.

---

## Figure 2 — the MCM checkerboard (partially verified)

Legend, in the figure's own order:

| swatch | stage |
|---|---|
| dark grey | 1st stage |
| white | 2nd stage |
| light grey | 3rd stage |
| black | 4th stage |

**Confirmed:**

* Four stages, drawn as a **2×2 spatial tile** — matches `mcm_stages: 4`.
* The grouping is drawn on a **3-D block**, and the colour of each cell is
  constant along the depth axis. Depth is the channel dimension, so **all
  channels at a given spatial position belong to the same stage**. MCM partitions
  space, never channels. This is the detail that makes the "fixed 4 passes
  regardless of image size" claim work, and it means our implementation
  partitions with a `[1, 1, H, W]` spatial mask broadcast over channels — not a
  per-channel schedule.
* The caption says *"primary component latent samples"* — again luma-only.

**Not resolved:** which of the four positions in the 2×2 tile is stage 1. At the
available raster resolution the dark-grey (stage 1) and black (stage 4) swatches
are not reliably separable in the small cells, and the two greys likewise.

`entropy.mcm_group_order: [[0,0], [1,1], [0,1], [1,0]]` therefore **remains an
assumption**, on the reasoning that a diagonal-first pair gives stages 3 and 4
the richest 4-neighbour context. It stays flagged in `docs/04` §7. The impact is
bounded and non-fatal: a wrong order costs coding efficiency but still decodes
correctly, because encoder and decoder share the constant. The **reference
software settles it definitively** — a much better source than any screenshot,
which is why the clone matters more than the figure.

---

## New prose captured from the Fig. 2 page (me-tANS)

The page containing Figure 2 carries me-tANS detail not in the earlier
extraction. Verbatim substance:

* Escape coding *"preserv[es] the ability to encode symbols with larger values
  when necessary"* — consistent with the 1-flag-bit → 2-or-15-bits → sign scheme
  in `docs/02` §VI-C.
* Algorithm 1 is the **single-threaded** procedure. *"Stream interleaving and
  stream concatenation techniques are employed to achieve parallel coding with
  multiple threads."*
* **"Even in single-threaded operation, two substreams are interleaved and two
  separate states are maintained to enhance parallelism — effectively mimicking
  the behavior of a dual-threaded setup."**
* *"Decoding requires only a bitwise OR operation and a simple addition"* thanks
  to the tabular design.
* Tables are *"on the order of 100 KB at the decoder side."*

**The dual-state detail is a hard requirement, not a note.** Two interleaved
substreams with two independent ANS states, even single-threaded, changes the
bitstream layout — a single-state decoder will not parse a conformant stream. It
must be designed into Phase 9 from the start rather than retrofitted. It exists
to break the serial dependency chain: ANS decode is inherently sequential
(state → symbol → state), so two states let a superscalar CPU keep two
dependency chains in flight and roughly double throughput with no extra tables.
Combined with the FILO decode order, this is why entropy decoding is fast enough
to matter at 14 kMAC/pxl.

`"bitwise OR + addition"` is also a useful correctness check on our table
design: if our inner decode loop needs a multiply, shift-heavy arithmetic, or a
branch per symbol, our tables are laid out wrong.

---

## Status of `docs/04` §7 open questions after this verification

| # | Question | Status |
|---|---|---|
| 1 | `ẑ_Y` channel count: 160 vs 128 | **open** — needs reference software |
| 2 | `I_σ` → 32 σ-class mapping | **open** — blocks the entropy stage |
| 3 | `step` / `sigmaPrecision` values | **open** |
| 4 | `skip_threshold` | **open** |
| 5 | MCM group ordering | **narrowed** — 2×2 spatial, channel-constant confirmed; position→stage still open |
| 6 | Secondary analysis stage count | **resolved** — Fig. 1's concatenation of `ŷ_UV` with `ŷ_Y` requires equal spatial size, so 4:2:0 chroma must use one fewer stage. Dimensionally forced. |
| 7 | tANS `tableLog` / spread function | **open**, plus a new requirement: two interleaved substreams, two states |
| 8 | `p̈_UV` pre-shuffle channel count | **open** |
| 9 | eq. (4) coefficients | **open** — still using the textbook BT.709 inverse |

Net: one resolved, one narrowed, seven still needing the reference software.
Also newly *added* to the list: how `z_Y` and `z_UV` are delimited inside the
single SOZ segment.
