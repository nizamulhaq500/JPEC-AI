# JPEG AI — Study & Implementation Project

Everything in `docs/` was produced from a close reading of:

> S. Esenlik, Y. Wu, Z. Zhang, Y.-K. Wang, K. Zhang, L. Zhang, J. Ascenso, S. Liu,
> "An Overview of the JPEG AI Learning-Based Image Coding Standard,"
> *IEEE Trans. Circuits Syst. Video Technol.*, vol. 36, no. 2, pp. 2520–2537, Feb. 2026.
> DOI: 10.1109/TCSVT.2025.3613244

## Read in this order

| # | File | What it is |
|---|------|-----------|
| 1 | [01-foundations.md](01-foundations.md) | The theory you need *before* the paper makes sense: rate–distortion, VAEs as codecs, hyperpriors, context models, quantization tricks, ANS. Not in the paper — this is the missing prerequisite chapter. |
| 2 | [02-jpeg-ai-explained.md](02-jpeg-ai-explained.md) | The paper itself, section by section, with every number, tensor shape, equation and design rationale, plus critical commentary. |
| 3 | [03-implementation-plan.md](03-implementation-plan.md) | 14-phase build plan for a working JPEG AI–class codec + demo you can show your professors. |
| 4 | [04-reference-data.md](04-reference-data.md) | Extracted tables, marker codes, tensor-shape cheat sheet, per-metric BD-rate results, implementation checklists. |
| 5 | [05-tables-figures-verified.md](05-tables-figures-verified.md) | Tables III–VI transcribed from the author-supplied screenshots and checked arithmetically. Establishes that AVG is the unweighted mean of the seven metrics, and that **Kodak (Table V) is our yardstick, not CTC (Table III)**. |
| 6 | [06-normative-constants.md](06-normative-constants.md) | Every numeric constant we could not get from the paper, extracted from the WG1 **reference software** with file-level provenance. Resolves nine of the ten open questions, and corrects two of my own earlier misreadings. Read this before touching `jpegai/config/*.yaml`. |
| 7 | [07-training-runbook.md](07-training-runbook.md) | **Your side of the work.** The remaining commands that need network access, a 30-second pre-flight, the training command, what the two log lines mean, and a troubleshooting table. |

## Current status

Phases 1–3 of the plan are done and verified. Phases 4 (two-branch YCbCr), 5 (split
hyper decoders, residual coding) and 6 (the multi-stage context model) are all built and
their *structural* criteria are measured; what remains in all three is the same thing —
RD comparisons, which need real training runs rather than more code.

Everything that can be checked without trained weights is checked on every commit:
**322 pytest tests** and **210 self-test checks**, the latter including a bit-exact
encode→decode round trip through a real rANS bitstream for all three internal chroma
formats × all six entropy-model kinds. Pointed at a trained checkpoint
(`--checkpoint <path>`) the self-test adds 5 more that only mean something once weights
have moved — including the one that would have caught the coder bug described at the end
of this file.

What Phases 1–3 delivered:

* **Config system** — `python -m jpegai.config` prints a summary and checks 17
  architectural invariants against both `tierA.yaml` and `full.yaml`. Eight of those
  invariants come from the reference software, not the paper.
* **Metrics** — all seven of the paper's metrics implemented at the QAF reference's own
  conventions (six luma-only, 10-bit internal precision, per-metric input ranges).
  6/7 run today; VMAF needs `ffmpeg`. `python -m jpegai.eval.metrics` reports what works.
* **BD-rate + benchmark harness** — `python -m jpegai.eval.runbench`. First real result,
  24 Kodak images, BD-rate vs JPEG: **WebP −16.5%, AVIF −42.2%**, which is where the
  published literature puts them. That is the Phase 1 gate: the measuring apparatus is
  trustworthy before we ask it to judge our own codec. Outputs in `results/`.
* **Scale hyperprior + real bitstream** — `jpegai/models/`, `jpegai/train/`. Our own
  entropy models (factorised non-parametric for `z`, Gaussian conditional for `y`), our
  own quantised-CDF construction, rANS from compressai. Both the scale-only and the
  mean-scale variants train and code.
* **Rate ladder + our codec in the benchmark** — `python -m jpegai.train.runladder`
  trains one model per β into `checkpoints/ladder/beta<value>/final.pt`, warm-starting
  each point from the previous one; `runbench --neural` then measures those checkpoints
  through the same harness as the anchors, from actual bitstream bytes.

**Phase 3 gate — passed.** `python -m jpegai.models.selftest` runs 83 structural checks
plus 4 more with `--checkpoint`; `pytest tests/` adds 40 (15 on the CDF layer, 9 on the
zip salvage).
On a trained checkpoint over Kodak:

| quantity | measured | meaning |
|---|---|---|
| actual vs σ-quantised estimate | **−0.11%** | the CDF tables and the rANS coder are correct |
| actual vs the estimate the loss saw | **+1.92%** | the cost of JPEG AI's 32-level σ grid, not a bug |
| symbols outside the CDF tables | **0.000%** | no escapes on either stream |
| `ŷ` through `decode(encode(x))` | **bit-exact** | on every test image |

Splitting that gap in two was the substantive finding of the phase. The plan originally
asked for "estimated within 1–2% of actual", which would have failed a *correct* codec:
the coder is accurate to −0.11%, and the remaining +1.9% is σ quantisation, confirmed
three independent ways (closed form, synthetic log-uniform σ, end-to-end). See
`06-normative-constants.md` §3.1 for the level-count trade-off curve. Two rules follow:
RD curves are computed from **actual bytes**, never the estimate, and the acceptance
threshold is on the σ-quantised comparison.

Also settled in Phase 3: **β multiplies the distortion** (`L = β·D₂₅₅ + R`), derived from
three independent config constants — see the derivation in `jpegai/train/losses.py`.

**The one criterion still open** is the first: *"RD curve beats JPEG comfortably."* That
needs trained weights, not code. The ladder driver, the benchmark wiring and the
measurement path are all built and verified end to end on a deliberately tiny run —
what remains is wall-clock on a machine with a GPU. See
[07-training-runbook.md](07-training-runbook.md).

```bash
python -m jpegai.train.runladder --device mps --iterations 50000 --batch 8
python -m jpegai.eval.runbench --codecs jpeg,webp,avif --neural checkpoints/ladder
```

**Validation was fixed along the way.** `DIV2K_valid_HR.zip` had been truncated at 379
of 428 MiB, so `unzip` rejected it, the validation directory was empty, and training
fell back to validating on **Kodak — our test set**. Model selection was therefore
peeking at the test data. `jpegai/data/salvage_zip.py` reads the partial archive
forward through its local file headers and recovered 88 of the 100 images, each
verified against its stored CRC32. Checkpoints now also record *which* image set their
validation numbers came from, and the ladder refuses to compare points that disagree.

Next: Phase 4, the two-branch YCbCr architecture.


## Phase 4 — two-branch YCbCr (built; two of four criteria measured)

`jpegai/models/twobranch.py`. A primary (luma) branch and a secondary (chroma) branch,
joined by the paper's two cross-component links: the encoder-side one (secondary
analysis consumes a downsampled `x_Y`) and eq. (3)'s decoder-side concatenation
`ŷᶜ_UV = concat(ŷ_UV, ŷ_Y)`. Four entropy streams (`y`, `z`, `y_uv`, `z_uv`) over
**one shared** σ table. Internal 4:4:4 / 4:2:2 / 4:2:0 all build, code and round-trip;
4:2:2 needs an anisotropic stride `(2,1)`, which the paper does not specify.

Train it with `--model twobranch`; `runladder` and `runbench --neural` both accept it.

**Complexity — the reportable architectural result.** At Tier A:

| | params | kMAC/pxl total | kMAC/pxl decoder |
|---|---|---|---|
| single-branch RGB | 3,751,627 | 134.4 | 111.6 |
| two-branch YCbCr | 4,903,491 | 160.0 | 132.4 |

A complete second branch with its own analysis, synthesis and hyperprior costs only
**+19% decoder complexity**, because it runs at quarter resolution. Chroma is 33.0 of
the 160.0 kMAC/pxl and 27.0 of the 132.4 decoder-side. The decoder figure is bucketed
per-part (`summary_parts()`) specifically so `h_a` is never billed into it — a branch
mixes encoder-only and decoder-side work, and lumping them inflates the one number
that gets compared to the paper's SOP 8 / BOP 28 / HOP 215.

**6:1:1 luma weighting — now verified, not just implemented.** It had been in the loss
since Phase 2 but was untestable: a single-branch RGB model has no separate chroma
parameters to measure. With two disjoint parameter sets the claim becomes falsifiable,
and `tests/test_losses_twobranch.py` measures it — against a 1:1:1 control, 6:1:1
lowers the secondary branch's gradient norm, raises the primary's, and shifts the
ratio by more than 2×, without silencing chroma. Two exact identities got pinned along
the way:

* A **uniform RGB offset is a pure luma error.** Because `K_R + K_G + K_B = 1`, adding
  a constant to R, G and B shifts Y by exactly that constant and cancels exactly in
  `Cb = (B−Y)/1.8556`. So the obvious "add a small offset" test image has *zero*
  chroma error, and 6:1:1 scores it exactly `6·3/8 = 2.25×` a 1:1:1 loss. The first
  version of that test asserted a fuzzy bound and failed against the sharp identity.
* Weights are normalised to `Σw = 3`, so an **equal error in every plane gives a
  weight-independent distortion**. Without that, a weighting ablation is secretly a
  rate-point ablation.

Distortion is measured on the **unpadded RGB output**, not the model's internal planes,
for two independent reasons: internal planes are still reflect-padded and those pixels
get cropped, so distortion spent there buys nothing; and at 4:2:0 the internal chroma
plane is half resolution, so measuring there would leave the chroma *upsampler* outside
the autograd graph and the secondary synthesis would never learn to compensate for it.

**`--luma-only`, the machine-consumption fast path.** `python -m jpegai.eval.lumaonly`.
Measured on CPU against the **trained** `ladder_tb3k` checkpoints — the tool's own help
warns that byte counts must come from trained weights, and an earlier version of this
table did not heed it (see the correction note below):

| β (rate point) | 512×768 payload | 1024×1024 payload | rate saving |
|---|---|---|---|
| 0.002 | 24,508 → 21,600 B | 65,932 → 58,140 B | **−11.9%** |
| 0.03 | 50,640 → 44,364 B | 135,044 → 118,492 B | **−12.3%** |
| 0.2 | 66,676 → 55,364 B | 178,484 → 148,096 B | **−17.0%** |

| | 768×512 | 1024×1024 |
|---|---|---|
| decode | 161.1 → 121.0 ms (**−24.9%**) | 426.6 → 328.1 ms (**−23.1%**) |
| luma plane | bit-identical to a full decode | bit-identical |
| output | grey to 1.2e-07 | grey to 1.2e-07 |

Two things to read off that. The **rate** saving is 12–17% and *grows with rate*, because
chroma's share of the payload grows with it. The **time** saving is flat at ~24% across
all three rate points, which is the expected shape: decode cost depends on the
architecture, not on what the bits say.

> **Correction.** This table previously read **−33.2%**, flat, at both sizes. That number
> came from a *randomly initialised* model, where the chroma latent is essentially noise
> and therefore costs its maximum — so chroma was hugely overrepresented in the payload
> and deleting it looked three times as valuable as it is. The `--checkpoint` help text
> had said "the byte counts do [depend on training], so quote those from a trained
> model" since the tool was written; the table was the one place that ignored it. The
> *timing* numbers were always safe at random init, and barely moved (23.8% → 24.0%).

Run-to-run jitter was 0.04× to 0.14× of the effect, so the timing is reportable; the
harness prints that ratio and refuses to stand behind a row where it reaches 1×.

The saving is proven by *deleting* the chroma strings before decoding — a decoder that
decodes chroma and throws it away passes every correctness test, so nothing weaker
proves the fast path exists. Worth noting honestly: the **time** saving (24.0%) comes
out slightly *above* the arithmetic saving (20.4% of decoder kMAC/pxl), because rANS
decoding is not multiply–accumulate work and a kMAC count cannot see two of four
streams going unread. The kMAC share is an anchor, not a ceiling. The number to quote
is the rate: an eighth to a sixth of the payload, on every image, on every device.

**Still open, needing trained weights:** the fourth criterion, *"two-branch beats
single-branch-RGB at equal rate"*. Two matched 3,000-step CPU ladders were run to
exercise the comparison path end to end, and both finished clean. Re-measured after the
median-shift fix (§"A coder bug that no correctness test could see", below) — `act bpp`
and `gap_q` both moved, `est bpp` and PSNR did not, because those come from the forward
pass and the fix was confined to the coder's table:

`ladder_cpu3k`, single-branch `mean-scale`:

| β | λ·255² | est bpp | act bpp | PSNR | gap_q | out-of-range | ŷ exact |
|---|---|---|---|---|---|---|---|
| 0.002 | 130 | 0.4362 | 0.3860 | 22.03 | +0.01% | 0.000% | yes |
| 0.03 | 1,951 | 0.8468 | 0.7460 | 25.63 | +0.03% | 0.000% | yes |
| 0.2 | 13,005 | 1.0165 | 0.9131 | 26.40 | −0.00% | 0.000% | yes |

`ladder_tb3k`, `twobranch`:

| β | λ·255² | est bpp | act bpp | PSNR | gap_q | out-of-range | ŷ exact |
|---|---|---|---|---|---|---|---|
| 0.002 | 130 | 0.4621 | 0.4077 | 23.02 | +0.07% | 0.000% | yes |
| 0.03 | 1,951 | 0.8686 | 0.7252 | 25.46 | +0.04% | 0.000% | yes |
| 0.2 | 13,005 | 1.0836 | 0.9296 | 26.41 | +0.01% | 0.000% | yes |

Every rate point on both ladders is inside the ±0.5% coder gate, at three rates an
octave apart. 3,000 steps is far too short for an RD *claim*; these exist to prove the
comparison machinery, and the real answer comes from the 50,000-step MPS ladder.

> **The tables above are re-measurements, not the stored metadata.** Both ladders
> finished *before* the fix, so their `final.pt` files still carry the pre-fix
> `rtcheck` block — re-rendering either ladder summary replays the old numbers and its
> stale `gate failed` warning. That is deliberate: the checkpoints are the evidence the
> bug happened, so they were left alone rather than rewritten. The `worst stream` column
> also reads `--` on them, since they predate the per-stream keys.

**Gate:** `python -m jpegai.models.selftest` is 210/210 checks, across all three
internal chroma formats and all six entropy-model kinds; with `--checkpoint` it adds
5 more. `pytest tests/` is 322.


## Phase 5 — split hyper decoders and the integer σ index (built; complexity criterion met)

`jpegai/models/hyper.py`, wired into `twobranch.py`. This is §VI-E: the hyper decoder
splits into **two heads over one shared `ẑ`** — a prediction head producing `p̈` and a
scale head producing the integer index `Iσ` — and the latent is coded as the residual
of eqs. (1)/(2), `r = round(y − p̈)`, `ŷ = r + p̈`.

Three `--model` kinds now share the two-branch checkpoint format:

| kind | entropy model | at Tier A |
|---|---|---|
| `twobranch` | Phase 4 mean-scale, one fused `h_s` | 4,903,491 params / 132.4 decoder kMAC/pxl |
| `twobranch-split` | **Phase 5 default** — separate `h_s` + `h_scale` | 4,575,603 / 128.9 |
| `twobranch-fused` | the `--single-hyper-decoder` ablation | 4,700,451 / 129.2 |

`twobranch` keeps meaning exactly what it meant, because these strings are written into
checkpoint `meta["model"]` — they are on-disk format, and redefining one silently turns
an old `.pt` into *wrong weights* rather than into a load error.

**Criterion 2 is met with three orders of magnitude to spare.** The plan asked for the
scale decoder to be under 5% of decoder complexity; the two scale decoders together are
0.06 + 0.01 = 0.07 of 128.9 kMAC/pxl = **0.055%**. They are reported as their own
`h_scale_y` / `h_scale_uv` rows in `summary_parts()` rather than folded into `h_s_*`
precisely so the claim is falsifiable — folded in, it would be unmeasurable.

**A complexity result the paper does not state.** The confirmed reference `h_s`
structure is **6.8× cheaper** than Phase 4's deconv-based mean-scale `h_s` — 0.49 vs
3.33 kMAC/pxl — and the reason is structural, not a matter of width. Every convolution
in the split decoder outputs at /64 (4×4) or /32 (8×8) and *never* at the latent grid
/16, because `conv_shuffle` does the arithmetic at the lower resolution and rearranges
afterwards. A deconv stack does two of its three layers at /16. Verified two ways: by
hand arithmetic (81 + 81 + 324 = 486 MAC/pxl against 337.5 + 2025 + 972 = 3334) and by
forward hooks asserting the output resolution of every `Conv2d` in the stack.

### `Iσ` is an integer, and the rounding direction is decidable

`σ = 0.11 · exp(log_k · Iσ / 2⁷)` with `log_k = (ln 54.82 − ln 0.11)/31 = 0.200365`.
`Iσ` is not a scale, it is a **fixed-point index in the log domain** with
`sigma_precision = 7`, and `max_index = (32−1)·2⁷ − 1 = 3967`.

The CDF row is `ceil(Iσ/2⁷)`, written `(idx + step − 1) // step` — **not** `Iσ >> 7`.
(`docs/06` §3 said `>> 7` and was wrong; the two differ on 3937 of 3968 indices, so it
was not cosmetic.) Three independent arguments settle the direction:

1. **Reachability.** `ceil(3967/128) = 31 = levels − 1`, so the last row is reachable.
   `3967 >> 7 = 30` leaves row 31 permanently dead — an unusable CDF row in a design
   whose stated goal is a small table.
2. **Measurement.** Phase 3 measured round-down at 0.63% escape symbols for a −17%
   rate saving; round-up escapes nothing.
3. **The ceiling.** `max_index` is 3967, not 3968, so the largest σ that `Iσ` can
   *denote* is **54.734**, one 128th of a step below the grid top. 54.82 is therefore
   reachable only as a *row*, which round-up delivers and round-down does not.

### The one-ULP hazard, and why the integer path is not optional

`SigmaIndex.table_row(Iσ)` and `GaussianConditional.build_indexes(σ(Iσ))` implement the
same rule two ways. They agree on 3,957 of 3,968 indices and **disagree on exactly 11**:

```
256, 1152, 1280, 1536, 1664, 2176, 2304, 2560, 3200, 3328, 3456
```

Every one is an exact multiple of 128 — an `Iσ` sitting precisely on a grid point, where
a single float32 ULP decides a `<` comparison. At index 256, torch computes
`0.16422072052955627` while the float32 `scale_table` buffer holds
`0.16422070562839508` (float64 truth: `0.16422071296529303`), and `build_indexes`
counts entries *strictly below* σ, so the float path is always the one that lands one
row too high.

0.277% of symbols. An encoder on one path and a decoder on the other produce a
bitstream that decodes to the wrong latent **while both sides report success** — the
worst failure mode a codec has. So the hazard is closed at every level: the split codec
never calls `build_indexes`; `GaussianConditional` takes an explicit `indexes=` kwarg
and validates it; `TwoBranchCodec.coder_rows()` is what the mid-training gate asks, so
`gap_q_pct` cannot acquire a permanent bias from measuring against the wrong row; and
`tests/test_hyper.py` both *enumerates* the 11 indices (so the list cannot drift
silently) and *demonstrates* the corruption end to end in
`test_mixing_the_two_index_paths_corrupts_the_latent`.

### Other invariants pinned

* **`p̈` depends on `ẑ` alone.** Enforced structurally — `predict(z_hat, *, quantise)`
  takes no `y` — and pinned by a signature assertion, because an encoder-only
  prediction would make the decoder drift and every test would still pass.
* **σ starts mid-table**, at `max_index/2 = 1983.5` → σ = 2.4537, by biasing the scale
  head's final layer. (A half-step below the grid's exact geometric mean
  `√(0.11·54.82) = 2.4556`, because `max_index` is odd.) Without it every latent element
  opens at the σ = 0.11 floor over a random latent, every symbol escapes, and the first
  few thousand gradient steps are about escapes rather than about the image.
* **`PixelShuffle(r)` sends conv channels `[r²j, r²(j+1))` to shuffled channel `j``**,
  which is what makes the fused ablation's `bias[4c:].fill_(init)` bias the `Iσ` half
  and *only* the `Iσ` half. Tested, not assumed.

**Still open, both needing trained weights:** criterion 1, residual coding beats plain
scale-hyperprior by 3–8% BD-rate; and the `twobranch-fused` vs `twobranch-split` RD
delta. Note the fused ablation has *more* parameters than the split pair (4,700,451 vs
4,575,603), which is the honest direction — it is expected to lose on RD despite being
larger, and that is the point of running it.

Next: Phase 6, the Multi-Context Model (4-stage checkerboard, Fig. 2).


## Phase 6 — the Multi-Context Model (built; complexity criterion met)

`jpegai/models/mcm.py`, wired into `twobranch.py`. This is §VI-D: the `/16` luma latent
is partitioned into four **cosets** by `(i mod 2, j mod 2)` and reconstructed in a fixed
order, each coset's prediction conditioned on every coset already reconstructed. Within
a stage all samples are independent, so a stage is **one parallel network pass** — four
of them, whatever the image size.

The coset order is `(0,0) → (1,1) → (0,1) → (1,0)`, diagonal first. It is *derived*, not
chosen: [06-normative-constants.md](06-normative-constants.md) §5 proves it twice from
the WG1 reference software. Diagonal-first is also what makes the 2-stage ablation the
classic one-shot checkerboard model rather than an arbitrary half-measure.

Three more `--model` kinds, all sharing the two-branch checkpoint format:

| kind | stages | params | decoder kMAC/pxl at Tier A | MCM's share |
|---|---|---|---|---|
| `twobranch-mcm` | **Phase 6 default** — 4 | 5,627,571 | 129.9 | +1.0 (+0.8%) |
| `twobranch-mcm2` | 2 (the checkerboard ablation) | 5,498,355 | 129.8 | +0.9 (+0.7%) |
| `twobranch-mcm1` | 1 (the Phase 5 zero point) | 5,239,923 | 129.5 | +0.6 (+0.5%) |

All three have the same **four** context networks; `stages` changes only what each one
may condition on. The parameter counts still differ, because a network that sees three
earlier cosets has a wider `conv1x1` than one that sees none — but the *number* of
networks is fixed, so the ablation is about context and not about capacity. (Compare
Phase 5's `twobranch-split`: 4,575,603 params, 128.9 decoder kMAC/pxl.)

**The entropy coder never enters the stage loop.** MCM refines only the *mean*; `Iσ`
comes from the scale head and depends on `ẑ` alone. So the whole residual field `r̂` is
entropy-decoded in **one** mean-free pass, and only then do the four stages turn `r̂`
into `ŷ`. That is §VI-E's decoupling, and it is why the packet layout is byte-for-byte
what Phase 5 wrote — one `y` stream and one `z` stream per branch. `stream_bytes`,
`packet_bytes`, `estimated_bits`, the rate gate and the training loop needed **zero**
changes. The self-test watches the call rather than the pixels (`means=None`, exactly
one call) because a version that re-entered the coder per stage would produce identical
bytes while destroying the self-contained entropy engine §VI-E exists to build.

**Cost is measured, not assumed.** Every context convolution runs at `/32` — the hyper
decoder's prediction arrives *pre-shuffle* as `[4·chs, /32]`, which is exactly one
coset's grid — so the four stages together add under 1% of decoder MACs. `PixelShuffle`
is parameter-free, so a Phase 5 `h_s` loads into a Phase 6 model unchanged and
`--warm-start` seeds an MCM ladder from the Phase 5 one.

**Verified without trained weights:** `tests/test_mcm.py` is 77 tests and
`check_mcm` is 33 self-test checks. The non-negotiable one — `ŷ_decoder == ŷ_encoder`,
tensor-for-tensor — is asserted through a real rANS bitstream at three chroma formats ×
two orientations for the whole codec, and at all three stage counts on the branch,
because getting it wrong does not crash: stage `k`
conditions on stages `< k`, so an encoder that computes its contexts by any route the
decoder does not repeat yields a codestream that decodes without complaint into a
reconstruction that drifts a little further with every stage.

**Still owed, and it needs GPU time:** "MCM gives 4–9% BD-rate over Phase 5; if < 2%,
the context nets aren't seeing previous groups", and the wall-clock half of the constant
-latency claim (decode time vs megapixels, MCM on and off — two parallel lines). The
*structural* halves of both are pinned: the dependency graph is asserted directly
(a later coset must change when an earlier one changes; same-stage siblings must not),
and the pass count is asserted to be 4 at 64² and at 256².


## The channel-layout bug that only the warm start could see

Worth recording next to the coder bug below, because it belongs to the same family and
was found the same way — by measuring something that was supposed to be trivially true.

`MultiStageContextModel.reconstruct` cut the hyper decoder's pre-shuffle `[4·chs, /32]`
prediction into four per-coset predictions with `pred.chunk(4, dim=-3)`. That is the
wrong cut. `PixelShuffle(2)` sends input channel `4c + 2i + j` to output channel `c` at
sub-position `(i, j)`, so coset `(i, j)`'s prediction is the **strided** slice
`pred[:, 2i+j :: 4]`. The contiguous chunk is a different permutation of the same
numbers.

**Nothing noticed.** Four tensors of exactly the right shape; the codec round-tripped
bit-exactly; every gradient was finite and non-zero; the model trained to a perfectly
good optimum. The only casualty was the warm start: `h_s` was trained *with* the shuffle
in place, so handing its slice `2i+j` to a different coset permutes the mean field, and
a Phase 5 checkpoint would open its Phase 6 run with two of its four predictions swapped
instead of as the identity — exactly the head start the near-identity init exists to
preserve.

Measured on `checkpoints/smoke_split/final.pt` warm-started into `twobranch-mcm`:

| | `chunk` | `split_pred` |
|---|---|---|
| mean field max abs deviation from Phase 5 | 0.0869 | **0.0003** |
| integer residual field vs Phase 5's | differs | **identical** |

Fixed by giving the slicing its own named function, `split_pred`, whose docstring says
why it is not inlined. Two tests now pin it —
`join_cosets(split_pred(pred)) == pixel_shuffle(pred, 2)` exactly, and its negative twin
asserting `chunk` does *not* give that identity while being the same multiset of numbers
— plus one self-test check per MCM kind, and a test that compares the two mean fields
numerically rather than comparing tensor *names*, since a name-level warm-start check
passes either way.

The lesson generalises past this bug: a check that a warm start *loaded* is not a check
that it warm-started.


## A coder bug that no correctness test could see

Found while gating Phase 5, and worth its own section because the *shape* of it is the
transferable lesson, not the one-line fix.

**The symptom.** The two-branch rate gate read **+1.85%** against its σ-quantised
estimate — outside the ±0.5% threshold, but only just, and with nothing to say about
where the excess came from. Meanwhile every correctness check passed: `ŷ` bit-exact
through a real bitstream, 0.000% of symbols escaping the CDF tables, decoded pictures
identical to the forward pass. Nothing was *wrong*; the bitstream was merely 1.85% more
expensive than the loss had been told.

**The bug.** `FactorizedPrior.forward` centres before evaluating the density —
`v = round(z − median)` — so the density's input coordinate *is* the symbol value.
`FactorizedPrior.update`, which builds the coder's quantised CDF, sampled the pmf for
bin `v` at `median + v` instead. The table was therefore a **shifted copy** of the
distribution the rate loss was trained against, with the shift equal to the learned
median.

**Why every test passed.** `median` is a learned quantile that starts at 0 and only
drifts as the aux loss moves it. At `median = 0` the two samplings are identical, so a
freshly constructed model — which is what most tests build — cannot see the bug at all.
And at convergence the aux loss has driven `median` to the density's own median, where
they agree again. The bug is visible only *in between*: on every partly-trained model,
which is every model at every mid-training gate check, and invisible at both ends.

**The diagnosis, which is the reusable part.** Strictly by elimination, each step ruling
out a whole layer:

1. **Not rANS framing.** A fixed per-stream overhead would shrink as a fraction of a
   growing rate. The observed gap *grew* with rate (+1.29% → +1.85% → +2.24%), so it was
   proportional to something, not additive.
2. **One stream, not all four.** Per-stream byte accounting put 854 of 991 excess bytes
   on `z_uv` alone — the chroma hyper-latent, the smallest of the four streams.
3. **The coder is faithful to its own table.** Bits read straight out of the quantised
   CDF matched the bytes written to +0.27%. So the fault was upstream of the coder: in
   the table.
4. **`forward` and `compress` round identically.** `z_hat − (round(z − med) + med)`
   differed by exactly `0.0000`. So both sides agreed on the *symbols*; they disagreed
   on the *probabilities*.
5. **The table pmf against the symbols actually coded.** Channel 0 of that stream has a
   learned median of **+1.455**, and every symbol in it lands on `v=1` or `v=2` —
   empirical entropy exactly 1.00 bit. Sampled the way `forward` samples, the table
   charges **1.55 bits/symbol**; sampled at `median + v` it charges **2.42**. The table
   was on a shifted grid, and the overcharge tracked `|median|` across channels: channel
   46, median −0.52, moved only 1.89 → 1.95 bits.

**The effect of the fix**, measured, no retraining — the rate loss uses `forward`'s
likelihood, which never changed:

| | before | after |
|---|---|---|
| two-branch gate (`gap_q_pct`) | +1.85% | **+0.04%** |
| single-branch gate | +0.21% | **+0.03%** |
| `z_uv` stream on one 768×768 image | 2,200 B | **1,352 B** (−38.5%) |
| whole payload, same image | 54,436 B | **53,468 B** (−1.78%) |

Consequences recorded honestly: every two-branch rate number reported before the fix was
~1.2–2.2% too high and every single-branch one ~0.1–0.3% too high, which is why the
ladder tables above carry re-measured `act bpp` columns. The three `**`-flagged gate
failures in `logs/ladder_tb3k.log` were this bug, not a coder or architecture fault.

**What the gate learned.** The aggregate `gap_q_pct` divides one stream's error by
*every* stream's bits, which is why a 63% fault surfaced as 1.85%. On a model whose bad
stream carried a smaller share of the total it would have passed outright. So
`roundtrip_check` now reports **per stream** — `gap_y_pct`, `gap_z_uv_pct`,
`excess_*_b`, and a named `worst_stream` — and fails on a second, independent arm.
Four design points, each forced by a measurement:

* **The z streams are measured against the plain forward likelihood**, not against a
  table reading. That deliberately spans all three links at once (forward → table →
  bytes), so it catches a table that disagrees with the trained density and not merely a
  coder that disagrees with its table. The old `est_q` mixed a table reading for `y`
  with a forward reading for `z`, which is precisely why a z-table fault had nowhere to
  appear except the total.
* **The threshold has two arms** — excess must exceed *both* 16 bytes and 2%. The floor
  was measured over **72 stream-readings** (both 3k ladders × 3 β an octave apart × 4
  validation images): median **+5.3 B**, full spread **−18.3 to +14.5 B**, and nearly
  independent of stream size — the +14.5 B reading came from a 26 kB stream, while a
  972 B stream cost +6.9 B. As a *percentage* that same floor is 0.055% on the big
  stream and 0.71% on the small one, and would be ~4% on a 200 B stream. So a
  percentage-only gate cries wolf on small streams and a byte-only gate goes blind on
  large ones.
* **The margin that matters is on the conjunction, not on either arm.** Taken alone the
  byte arm looks alarmingly tight — +14.5 B is 91% of 16 B. But that reading consumes
  only **3%** of the percentage arm, so it is nowhere near firing. Across all 72 readings
  the closest approach to a false alarm is a 972 B `z_uv` at +6.9 B / +0.71%, which
  consumes **0.36×** of its binding arm: about 2.8× of real headroom. This is also the
  argument for *not* raising the byte arm — doing so would only blind the gate on the
  small streams it exists to protect. The median-shift bug sat at 854 B / +63%, clear of
  the two arms by 53× and 31×.
* **The excess is two-sided, and the arm is deliberately not.** A `y` stream is measured
  against `est_q`, and quantising σ onto the 64-entry log grid can round σ *up* — making
  the estimate pessimistic and the real bytes *fewer* than predicted. Hence the −18.3 B
  end of the spread. Coming in under the estimate is never the fault this gate hunts, so
  the test is `excess > 16` and not `|excess| > 16`.
* **`worst_stream` ranks by bytes, not percentage.** At random init a stream can carry
  almost no rate — an untrained `y` predicts its own near-zero symbols nearly perfectly,
  so 8 bytes of pure flush reads as **+526,578%**. A percentage ranking would name that
  harmless stream on every early check of every run. The ranking floor is `-inf` rather
  than a small negative number, because the excess is two-sided: on a run where every
  stream came in under its estimate, a finite floor would name no stream at all.

`tests/test_entropy.py` (12 tests) pins the table/forward invariant, including the
sharpest form of it: **the quantised CDF table must not depend on the median at all**,
because it is indexed by symbol and symbols are median-relative by construction.
`tests/test_rate_gate.py` (9 tests) pins the detector, by injecting a shifted table and
requiring the gate to name the right stream. Both files record a sensitivity result
rather than hiding it: at the default `init_scale = 10` a fresh prior's density is flat
enough over its own table (peak p = 0.025 across 21 bins) that shifting the table under
it costs almost nothing, so **a fresh prior is not a valid fixture for anything about
table alignment.** The first version of the injection test reproduced the historical
bug's mechanism exactly and measured zero effect for that reason.


## Working artifacts already in this repo

```
paper/
  paper_text.txt        full extracted text of the paper (18 pages)
  rasterize.py          minimal PDF vector-path rasterizer (see note below)
  imgs/
    p03_0_Im1.png       Fig. 1 right half  (secondary/chroma branch, enc + dec)
    p03_1_Im0.png       Fig. 1 left half   (primary/luma branch, enc + dec)
    table_p2_Fm0.png    Table I    — the five parts of JPEG AI
    table_p5_Fm0.png    Table II   — codestream markers + hex codes
    table_p8_Fm0.png    Fig. 2     — MCM 4-stage checkerboard grouping
    table_p9_Fm0.png    Table III  — main results, per-metric BD-rate + complexity
    table_p12_Fm0.png   Table IV   — tool-off ablation
    table_p12_Fm1.png   Table V    — Kodak results
    table_p13_Fm0.png   Table VI   — CLIC 2024 validation results
```

**Note on `rasterize.py`:** Tables I–VI and Figs. 2 in the PDF are not text and not
embedded bitmaps — they are vector glyph outlines inside Form XObjects, so ordinary
text extraction returns nothing for them. Installing `poppler`/`pymupdf` was blocked
(no network in this environment), so `paper/rasterize.py` is a from-scratch renderer:
it walks the PDF content stream, tracks the CTM and fill colour, flattens cubic
Béziers, and scanline-fills with non-zero/even-odd winding. That is how the numbers in
`04-reference-data.md` were recovered. Usage:

```bash
python3 paper/rasterize.py <page_index_0based> /Fm0 7
```

## Environment note

This machine is an Apple M2 Pro (10 cores, 16 GB, no CUDA). That constrains the
training strategy — see §"Hardware reality check" in the implementation plan. Also,
outbound network was denied in this session, so `pip install` and dataset downloads
must be run by you (or the sandbox permissions relaxed).
