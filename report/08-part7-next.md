<div class="page-break"></div>

# Part VII — Limitations, Remaining Work, Open Questions

## 26. Honest limitations

Ordered by how much each one is likely to be distorting the numbers in Part V.

### 26.1 The training budget is the largest confound, and it is not quantified

50,000 steps on 6,400 crops of a single dataset, on a laptop. The paper's models are trained on far
more data, for far longer, with multi-stage schedules.

This is not a small caveat. Everything in Part V that reads as an architectural deficiency **may
instead be a budget deficiency**, and we cannot separate the two without the compute to run the
control. Chapter 20 separates them for exactly one question — tier A's ceiling — by measuring at
infinite bitrate. For every other question the confound stands.

The direction of the bias is knowable even if the magnitude is not: more training would improve our
numbers, so **every deficit reported here is an upper bound on the true architectural deficit.**

### 26.2 Eight of the standard's coding tools are absent

Phases 7–14. By the paper's own ablation (Table IV), the missing tools are worth roughly:

| tool | BD-rate |
|---|---|
| RVS | 2.2 pp |
| LSBS | 0.4 pp |
| LEF | 0.3 pp |
| ICCI | 0.2 pp |
| EFE ×2 | 0.2 pp each |

So the missing tools are worth about **3.5% of the rate**. That transfers between anchors to first
order, because it is a relative saving rather than a BD-rate against something: it would move
ladder #2 from 0.908× JPEG's bits to 0.876×, i.e. from −9.2% to −12.4% vs JPEG.

Set against the real gap that is a small share. §19.1.2 puts the paper's simplest decoder at
roughly **−41% vs JPEG**, so the gap from ladder #2 is ~32 points and these tools are ~11% of it.
An earlier draft of this chapter said they were "3.5 of the 8-point gap," which was ~44% of it —
that came from differencing our vs-JPEG BD-rate against the paper's vs-VVC-Intra one, and it made
the missing tools look like about half the answer when they are closer to a tenth. **Which leaves
~28 points that are the luma branch and the training budget.**

### 26.3 The distortion weighting is ours, and it is in the wrong place to be ignored

`{y:6, u:1, v:1}` is **OURS**. The paper says only "prioritise luma" and gives no numbers. Our
deficit is *entirely* in luma while chroma is at AVIF's level and consumes 25% of the bits (§19.3).
It is entirely possible that this weighting is misallocating the rate–distortion trade-off, and it is
the cheapest untested hypothesis in the project. §27 has the sweep.

### 26.4 Conformance is not, and cannot be, demonstrated

Our gates prove that **our** decoder inverts **our** encoder bit-exactly. They cannot prove either
matches WG1's, because that needs the normative CDF tables and the four ONNX parameter sets, both
distributed through the paywalled Part 1. This is a permanent limitation of the project as scoped,
not a to-do.

Related: the integer bit-exact path (§6.4) exists and is tested, but has only ever run on one
machine. Its cross-device portability is *argued* from the 8-bit/32-bit overflow bound, not
demonstrated.

### 26.5 Per-stage widths are ours, so complexity comparisons are approximate

The supplement's Figs. 6–8 carry the per-stage trunk widths and we do not have them (§7.2). Our
kMAC/pixel figures are therefore *our* architecture's, and any statement of the form "at matched
complexity" in this report is approximate.

### 26.6 One metric backend is ours

PSNR-HVS runs on our own DCT implementation because `psnr-hvsm` cannot be installed on macOS/arm64
(§23.2). BD-rate is a ratio between two curves measured with the same implementation, so a systematic
offset largely cancels — but not exactly. That column is internally consistent and only approximately
comparable to the paper's.

### 26.7 Two ladders are not a rate ladder

The standard defines an **18-point** quality ladder from four parameter sets, via learned per-channel
gains and a signalled β displacement (§6.9). We train **five independent models per ladder** with no
gain vectors at all. This is functionally different from the standard and it means our "rate ladder"
is a set of separate codecs rather than one variable-rate codec. Phase 8.

### 26.8 There is one decoder, not three

Phase 7 builds SOP/BOP/HOP. Today there is one synthesis transform, so nothing in this report
exercises the complexity-scalability claim, and the `synthesis_transform_id` capability-list
mechanism of §9.6 is documented but unimplemented.

### 26.9 Kodak is 24 small images

768×512, 24 of them, all natural photographs from 1993. It is the right choice because it is the only
dataset on which the paper publishes a comparable figure (§10). It is nonetheless a narrow benchmark:
no 4K content, no screen content, no HDR, and small enough that per-image variance matters.

## 27. Remaining work, in priority order

Each item has a cost estimate, because the whole project is compute-bound.

### 27.1 Immediate

**1. ~~Fix the `z_uv` gate failure (§22.4).~~ Done, 2026-09-01.** The proposed fix was wrong as well
as the diagnosis: no `coder_rows` accessor was needed. The cause was out-of-range escapes from a
table extent read off unconverged `quantiles`, and the fix reads the extent off the density instead
(`FactorizedPrior._density_extent`). Zero escapes on every checkpoint, −1.1% of total payload at
β 0.002, and no retraining — decoded latents are bit-identical. §22.4 has the measurement. *Actual
cost: ~3 h, no re-training.* **What remains is to re-run the benchmark**, because the fix changes the
bytes and therefore every BD-rate on a two-branch ladder (all of them are slightly pessimistic as
printed in §19):

```bash
python -m jpegai.eval.runbench --neural checkpoints/ladder_p6 --codecs jpeg,webp,avif
```

`CODER_VERSION` in `jpegai/eval/neural.py` invalidates the neural rows automatically; the JPEG/WebP/
AVIF anchors stay cached, so this costs only our own encode passes. `ladder_p6/beta0.001` is training
as this is written, so one run after it lands picks up the coder fix *and* the new rate point.

**2. Add VTM-11.1 intra as a fourth anchor.** This is now the top item, because §19.1.2's ~32-point
gap rests on AVIF standing in for VVC Intra and a ±6-point assumption about the difference between
them. Measuring VTM on the same 24 images replaces the assumption with a number and puts every row
in §19.1 on the paper's own anchor, which is the only way to state our position without a conversion
caveat attached. *Cost: encoder build plus ~11 rate points × 24 images; VTM intra is slow but this is
a one-time anchor measurement, and it is cached forever after.*

**3. Separate the MCM's effect from the warm start.**

```bash
python -m jpegai.train.runladder --model twobranch-split --name ladder_p5_long \
    --warm-start-from checkpoints/ladder_p5
```

The control that removes §17.3's confound. The `-mcm1` run already showed 4-stage MCM is worth
0.01–0.04 dB against 1-stage (§18.6), so ladder #2's +0.60 dB over ladder #1 is *not* the MCM's
multi-stage structure — but mcm1 shares both remaining candidates with mcm4, so what is left
undivided is **the extra 50,000 steps versus the presence of any context model at all.** This run
holds the architecture at phase 5 and gives it the extra steps, which separates them. Until it runs,
phase 6's gain must be read as an upper bound. *Cost: ~23 h.*

**4. Add one more low-β rate point.** β = 0.001 is training; β ≈ 0.0005 would take ladder #2's
overlap coverage to about 9/11, matching the anchors' 9–10/11. *Cost: ~7 h.* This improves the
*credibility* of the headline number rather than the number itself.

**5. Sweep the distortion weights (§26.3).** `{y:6,u:1,v:1}` versus `{4,1,1}` and `{8,1,1}` at a
single β. Three short runs, and it directly tests the leading hypothesis for the luma deficit.
*Cost: ~6 h at reduced steps.*

### 27.2 Then — the missing ladders and ablations

| run | model | purpose | cost |
|---|---|---|---|
| `ladder_p3f` | `mean-scale`, tier full | isolates tier from architecture in §19.2's +2.12 dB | ~19 h |
| `ladder_p4` | `twobranch` | the two-branch step without split hyper decoders | ~23 h |
| `ladder_p5f` | `twobranch-fused` | confirms the 6.8× / 0.055% split-vs-fused trade at scale | ~23 h |
| ~~`ladder_p6a`~~ | ~~`twobranch-mcm1`~~ | **done** — 4-stage MCM worth <1% over 1-stage | ~26 h |

`ladder_p3f` is the highest-value one: it is what turns §19.2's joint tier+architecture measurement
into two attributed measurements. `twobranch-mcm2` is no longer worth running: the `-mcm1` result
already answers the question 2-stage was meant to answer, from the other end.

### 27.3 Also outstanding

- **Tier full's own ceiling.** Repeat chapter 20's two measurements at 160 channels. Ladder #1 is
  already *above* its PCA bound (35.81 vs 35.02) and still climbing, so whether it is capacity- or
  budget-limited is genuinely unresolved — and the answer determines whether phase 7's wider decoders
  will help. *Cost: ~1 h. This is the best value-per-hour item in the entire list.*
- **A BD-rate for `ladder_p6a_mcm1`.** It has two rate points, and BD-rate needs three, so
  `results/bench_all.md` currently prints `nan` for it. The <1% MCM conclusion rests on matched-β
  per-point comparisons instead. A third point would let it appear in §19.1 like everything else.
- Phase 6's two unmet criteria: the 4–9% BD-rate claim (via `--anchor ours-ladder_p5`) and the
  wall-clock half of the constant-latency claim.
- `runbench` on `ladder_cpu3k` and `ladder_tb3k`.
- The untested `runladder --bench` combined path.
- `runladder` chains β ascending (`runladder.py:70`), so the lowest β always trains cold rather than
  warm-starting from the nearest trained point. Should chain from whichever end has trained points.
- Delete the redundant 428 MiB `data/div2k/DIV2K_valid_HR.zip`.

### 27.4 Phases 7–14

In plan order, with what each would be worth:

| P | what it adds | expected |
|---|---|---|
| 7 | SOP/BOP/HOP synthesis transforms | the complexity-scalability claim becomes testable |
| 8 | gain vectors + β displacement → the 18-point ladder | one variable-rate codec instead of five models |
| 9 | me-tANS + the real codestream with markers | conformant file format; §9.4's constants are ready |
| 10 | RVS, LSBS, LEF, ICCI, EFE ×2 | **≈ 3.5 pp** of BD-rate (§26.2) |
| 11 | full integer bit-exactness + cross-device conformance | the interoperability guarantee |
| 12 | ROI, progressive, tiling, arbitrary sizes, HDR | deployability |
| 13 | full 18-point evaluation on all datasets + the ablation table | the complete results table |
| 14 | a `jpegai` CLI that encodes and decodes files | usable as a tool |

Phase 10 is the one with a directly measurable BD-rate payoff. Phase 8 is the one that would make
the codec structurally comparable to the standard rather than functionally similar.

## 28. Open questions

Genuine ones, in the sense that we do not know the answer and the answer would change what we do.

**1. Is the luma deficit capacity, recipe, or rate allocation?** Three candidates, three different
fixes. The distortion-weight sweep (§27.1 item 4) tests recipe; tier full's ceiling measurement
(§27.3) tests capacity; the MCM result tests whether a stronger luma entropy model closes it.
**This is the project's central open question.**

**2. What does `sigma_bound_offset = 0.5` mean?** Confirmed as a constant, unexplained. Two mutually
exclusive readings: a rounding offset making the σ-class rule round-*nearest* — which would
contradict §6.4.1's reachability argument and mean our implementation is wrong at half the grid
points — or a widening of the CDF tail bound. The reachability argument and our own escape-rate
measurement both favour round-up, so round-up is implemented. **This is the one place where the
supplement's arrival would most change the code.**

**3. Does 2-stage MCM capture most of 4-stage's gain?** **Answered, and more sharply than the
question was posed** (§18.6). We ran 1-stage rather than 2-stage, and it matches 4-stage to within
0.01–0.04 dB and 0.5% of rate. So four sequential decode passes buy essentially nothing at our
training budget, and 2-stage need not be run. The question that replaces it: is that because the
multi-stage context genuinely adds little on Kodak-sized images, or because 50,000 steps is too few
for the later stages to learn to use their extra context?

**4. Why does Table IV show every ablation decoding faster than all-on?** (§8.3c.) Either the timing
noise is ≳6%, in which case the paper's finer timing distinctions cannot be read, or there is a
systematic effect we do not understand. It matters because we cite those timings.

**5. How much of the ~32-point gap is tools and how much is training?** §26.2 estimates 3.5 points
of tools, leaving ~28. That split is an estimate built on the paper's ablation being additive, which
ablations generally are not — and the residual is now so much larger than the tool contribution that
the interesting question has changed. It is no longer "which tools are missing" but "how much of a
~28-point deficit can 50,000 laptop steps be responsible for," and the only way to bound that is a
long run at fixed architecture.

**6. Is our +19% decoder cost for a second branch the same as the standard's?** Our per-stage widths
are ours (§26.5), so the number is our architecture's. It is a plausible sanity check on the design
rather than a reproduction of it.

---

## Closing statement

The project set out to answer whether an architecture described in a paper, with its numeric
specification behind a paywall, could be reconstructed, implemented, trained and **honestly**
measured on one laptop.

Reconstructed: yes — nine of ten open questions resolved from the reference software, one dissolved,
and two of our own readings corrected before they became silent bugs.

Implemented: the six phases that contain everything specific to JPEG AI rather than generic to
learned compression. 14,154 lines, 332 tests, real bytes at every rate point, `ŷ` bit-exact
throughout.

Trained: three complete ladders plus a two-point control, about 90 hours of laptop time.

Measured honestly — and the second attempt at "honestly" is the one that counts. Our best ladder is
**9.2% ahead of JPEG**, roughly WebP's level. Against the standard, an earlier draft of this report
put us "about 8 percentage points" short of the paper's −7.5%, which was arrived at by subtracting a
BD-rate measured against JPEG from one measured against VVC Intra. Converted onto a single anchor
(§19.1.2) the paper's simplest decoder is around −41% vs JPEG and the gap is **~32 points**: at
matched quality we spend about **1.5× its bits**. About 3.5 of those points are eight unbuilt coding
tools; the other ~28 are the luma branch and 50,000 steps on a laptop, undivided. That correction
makes the project's headline result worse by a factor of four, and it belongs in the report for
exactly that reason.

Seven bugs are documented with their wrong numbers printed next to their right ones, five of which
produced plausible results rather than crashes — the anchor-mismatch error being the most plausible
of all, since every number involved in it was individually correct. One defect remains open and is
disclosed rather than buried.

The single most useful thing the project produced is not a compression result. It is chapter 20: two
cheap measurements that converted "why has it stopped improving?" from a guess into a decided
question, and then a third that confirmed the decision at **+2.12 dB on two real bitstreams of
matched size**. That is the method the remaining phases inherit.
