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

So about **3.5 percentage points** of the 8-point gap to −7.5% is accounted for by tools we have not
built. Which leaves roughly 4.5 points that are the luma branch and the training budget.

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

### 27.1 Immediate — after `ladder_p6` finishes (~30 h remaining)

**1. Fix the `z_uv` gate failure (§22.4).** Expose a `coder_rows` accessor on the chroma hyper prior
so the gate interrogates the model for the table it will actually use, mirroring what the σ path
already does. Deferred only because `ladder_p6` is training against that module right now. *Cost:
~2 h of work, then a re-gate.*

**2. Benchmark `ladder_p6` and separate the MCM's effect from the warm start.** Two runs:

```bash
python -m jpegai.eval.runbench --neural checkpoints/ladder_p6 --codecs jpeg,webp,avif
```

```bash
python -m jpegai.train.runladder --model twobranch-split --name ladder_p5_long \
    --warm-start-from checkpoints/ladder_p5
```

The second is the control that removes §17.3's confound. Without it, phase 6's gain is architecture
**plus** extra steps and must be read as an upper bound. *Cost: ~23 h.*

**3. Add two low-β rate points.** β ≈ 0.0005 and 0.001 would raise ladder #1's overlap coverage from
6/11 to about 9/11, making its AVG comparable to the anchors' 9–10/11 and to ladder #0's. *Cost:
~9 h.* This is the cheapest improvement to the *credibility* of the headline number, as opposed to
the number itself.

**4. Sweep the distortion weights (§26.3).** `{y:6,u:1,v:1}` versus `{4,1,1}` and `{8,1,1}` at a
single β. Three short runs, and it directly tests the leading hypothesis for the luma deficit.
*Cost: ~6 h at reduced steps.*

### 27.2 Then — the missing ladders and ablations

| run | model | purpose | cost |
|---|---|---|---|
| `ladder_p3f` | `mean-scale`, tier full | isolates tier from architecture in §19.2's +2.12 dB | ~19 h |
| `ladder_p4` | `twobranch` | the two-branch step without split hyper decoders | ~23 h |
| `ladder_p5f` | `twobranch-fused` | confirms the 6.8× / 0.055% split-vs-fused trade at scale | ~23 h |
| `ladder_p6a` | `twobranch-mcm2`, `-mcm1` | does 2-stage MCM capture most of 4-stage's gain? | ~26 h each |

`ladder_p3f` is the highest-value one: it is what turns §19.2's joint tier+architecture measurement
into two attributed measurements.

### 27.3 Also outstanding

- **Tier full's own ceiling.** Repeat chapter 20's two measurements at 160 channels. Ladder #1 is
  already *above* its PCA bound (35.81 vs 35.02) and still climbing, so whether it is capacity- or
  budget-limited is genuinely unresolved — and the answer determines whether phase 7's wider decoders
  will help. *Cost: ~1 h. This is the best value-per-hour item in the entire list.*
- Phase 6's two unmet criteria: the 4–9% BD-rate claim (via `--anchor ours-ladder_p5`) and the
  wall-clock half of the constant-latency claim.
- `runbench` on `ladder_cpu3k` and `ladder_tb3k`.
- The untested `runladder --bench` combined path.
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

**3. Does 2-stage MCM capture most of 4-stage's gain?** The MCM costs four sequential decode passes
for +1.0 kMAC/pixel. If `-mcm2` gets most of the benefit at two passes, that is a materially better
latency/quality point than the standard's own choice, and worth reporting as such.

**4. Why does Table IV show every ablation decoding faster than all-on?** (§8.3c.) Either the timing
noise is ≳6%, in which case the paper's finer timing distinctions cannot be read, or there is a
systematic effect we do not understand. It matters because we cite those timings.

**5. How much of the 8-point gap is tools and how much is training?** §26.2 estimates 3.5 points of
tools, leaving ~4.5. That split is an estimate built on the paper's ablation being additive, which
ablations generally are not.

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
learned compression. 14,027 lines, 331 tests, real bytes at every rate point, `ŷ` bit-exact
throughout.

Trained: two complete ladders, a third in flight, about 60 hours of laptop time.

Measured honestly: we are **level with JPEG**, roughly 8 percentage points short of the paper's
comparable −7.5%, and we can say where those points are — the luma branch and eight unbuilt tools.
Six bugs are documented with their wrong numbers printed next to their right ones, four of which
produced plausible results rather than crashes. One defect remains open and is disclosed rather than
buried.

The single most useful thing the project produced is not a compression result. It is chapter 20: two
cheap measurements that converted "why has it stopped improving?" from a guess into a decided
question, and then a third that confirmed the decision at **+2.12 dB on two real bitstreams of
matched size**. That is the method the remaining phases inherit.
