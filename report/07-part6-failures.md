<div class="page-break"></div>

# Part VI — Failures, Bugs and Errors

*This part exists because §2.3 said the interesting word in our problem statement was "honestly".
Six substantive correctness bugs were found. **Four of them produced plausible numbers rather than
crashes**, which is the only kind that is actually dangerous. Each one below gets the wrong number,
the symptom that exposed it, the diagnosis, and the corrected number.*

## 21. Training runs that failed or misled

### 21.1 The first two-branch smoke test — a gate failure treated as information

`ladder_tb3k` failed its coder gate at all three rate points (+1.29%, +1.85%, +2.24%). The
temptation, and it is a strong one when a smoke test at 3,000 steps produces a 26 dB image, is to
dismiss it as "undertrained, will settle".

It did partly settle — the same defect is +0.20% at 50,000 steps in `ladder_p5`. But it never went
away, and treating the smoke failure as signal is what produced §18.3's four-ladder table, which
localised the defect to the two-branch chroma hyper path in one step. **The failure was worth more
than the run's actual results**, which are meaningless.

### 21.2 The `--luma-only` table published from an untrained model

The monochrome fast-path table of §18.4 was first produced and written down as **−33.2%** rate
saving. That number is wrong, and it is wrong because the measurement was run against a **randomly
initialised model**.

*Why a random model gives a wrong and flattering answer:* in an untrained model the chroma latent
carries near-noise, which is maximally expensive to entropy-code. So dropping it saves an enormous
fraction of the payload. In a trained model chroma is smooth, cheap, and compresses well — so
dropping it saves much less.

**Corrected: −11.9% / −12.3% / −17.0%** at β 0.002 / 0.03 / 0.2.

The lesson has been made procedural: any measurement script now **asserts that a checkpoint was
actually loaded** and prints its step count, so "I forgot the `--checkpoint` flag" cannot
silently produce a publishable table.

### 21.3 Tier A's saturation — a failure that was not a bug

Worth including because the correct diagnosis was "nothing is broken".

Ladder #0 flattening at 32.3 dB looks exactly like a bug. Chapter 20's two measurements established
that it is a hard capacity limit: the coder costs 0.03 dB, the ceiling holds at infinite bitrate,
and the learned transform is already 1.4 dB *better* than the optimal linear transform of the same
width.

**The failure mode being guarded against here is the opposite of the usual one** — not shipping a
bug, but spending days hunting a bug that does not exist. Two cheap measurements are much cheaper
than that.

## 22. Correctness bugs

### 22.1 The BD-rate interpolant — the most expensive bug, and it was in the measurement code

**The wrong numbers, as first reported:**

| codec | first reported | corrected | moved by |
|---|---|---|---|
| WebP | −15.3% | **−10.6%** | 4.7 |
| AVIF | −41.0% | **−36.1%** | 4.9 |
| ours, ladder #0 | **+15.6%** | **−0.4%** | **16.0** |
| ours, ladder #1 | **+20.6%** | **+1.8%** | **18.8** |

Both headline figures moved by about 17 percentage points, **in the wrong direction** — the bug was
making our codec look far worse than it is.

**The symptom that exposed it:** `fsim +56.0%`. FSIM is a structural-similarity metric on which our
codec is *good* (the corrected value is −29.7%). A +56% BD-rate on a metric where MS-SSIM says
−31.5% is not a plausible result; two structural metrics cannot disagree by 87 points on the same
bitstreams. That implausibility is what triggered the investigation.

**The diagnosis.** The textbook Bjøntegaard method fits a **single global cubic polynomial** through
all the rate–quality points of each curve and integrates the difference. That is fine for PSNR,
which is unbounded and roughly logarithmic in rate. It is **invalid for metrics that saturate**.

FSIM, MS-SSIM, VIF and IW-SSIM are all bounded above by 1.0 and approach it asymptotically. A global
cubic through such points must bend; having bent, it **overshoots outside the data range** and can
even become non-monotonic. Integrating a non-monotonic fit against a monotonic one produces
arbitrary numbers.

**The fix:** `scipy.interpolate.PchipInterpolator` — monotone piecewise cubic Hermite interpolation.
PCHIP is *constrained* to be monotone between knots and cannot overshoot.

**How the fix was verified**, and this is the part that matters more than the fix: there is no
reference BD-rate to compare against, so the implementation was verified against an **invariance**
instead (§15.3). BD-rate is defined over the *overlapping* quality window, so anchor points outside
that window must not affect the answer. Four anchor sweeps identical inside the window and different
below it:

| interpolant | spread across four sweeps |
|---|---|
| **PCHIP** | **0.04 points** |
| global cubic | **17.08 points** |

The cubic violates the invariance by 17 points — which is exactly the magnitude of the error in the
headline figures. The two numbers agree, which is what makes this a diagnosis and not a guess.

**Two permanent consequences:**

- `bdrate.py` uses PCHIP and there is a test asserting the invariance holds to 0.15 points.
- **`overlap_coverage` is now returned with every BD-rate and printed in every table**, because two
  BD-rates over different windows are not comparable to each other (§19.1.1).

The generalisable lesson: **the measurement tool is part of the system under test, and it deserves
the same scrutiny as the codec.** This bug lived in `metrics/bdrate.py`, was found by disbelieving
an *anchor's* number rather than our own, and cost more BD-rate points than any bug in the codec.

### 22.2 `chunk` is not the inverse of `PixelShuffle` — the MCM channel-layout bug

**The bug.** MCM's prediction tensor packs four stages' worth of parameters along the channel axis,
and they must be unpacked in `PixelShuffle`'s interleaved layout. The code did:

```python
p1, p2, p3, p4 = pred.chunk(4, dim=-3)     # WRONG
```

`chunk` splits into four *contiguous blocks*. `PixelShuffle` interleaves with a stride. So every
stage was reading a permuted set of channels — a *valid* tensor of the right shape, containing the
wrong numbers.

**Why it was dangerous.** Nothing crashed. Shapes matched. The model trained, the loss went down,
and `ŷ` remained bit-exact — because the *same* wrong permutation was applied on both the encode and
decode sides, so the codec was self-consistent. It was simply a worse context model than intended,
and there is no signal in the training curve that says so.

**The symptom.** A dedicated diagnostic: the MCM's prediction should approximate the true mean-field
conditional expectation, so the **mean-field deviation** was measured directly.

| | mean-field deviation |
|---|---|
| with `chunk` | **0.0869** |
| with correct `split_pred` | **0.0003** |

A factor of ~290. The correct layout:

```python
def split_pred(pred, n_stages=4):
    """PixelShuffle's inverse is a strided de-interleave, NOT a contiguous chunk."""
    B, C, H, W = pred.shape
    return pred.reshape(B, C // n_stages, n_stages, H, W).unbind(dim=2)
```

The module docstring now states the layout explicitly, and a test asserts
`up_shuffle(down_shuffle(y)) == y` plus the deviation bound.

**The same class of error, in a second place.** A warm start silently permuted channels for the same
reason — parameters matched by name and shape, loaded, and were wrong. This is why §17.3's warm-start
loader now prints its own manifest of what loaded and what did not.

### 22.3 `FactorizedPrior.update()` ≠ `forward()` — 1.8% of every payload

**The bug.** `compressai`'s `FactorizedPrior` has two paths that must agree: `forward()`, which
computes the likelihoods used in the training loss, and `update()`, which builds the discrete CDF
tables the entropy coder actually uses. `update()` applies a **median shift** that `forward()` does
not. The tables were therefore centred slightly off the density the model had been trained against.

**The symptom.** The round-trip gate, and only the round-trip gate:

| | gate `gap_q` |
|---|---|
| before | **+1.85%** |
| after | **+0.04%** |

Nothing else showed it. `ŷ` was bit-exact (the coder was internally consistent), the images looked
right, and PSNR was unaffected. The only visible effect was that the real byte count exceeded the
model's own estimate by 1.85% — which is precisely the property §15.2's gate exists to check.

**The cost, measured per stream:**

| | before | after | change |
|---|---|---|---|
| `z_uv` stream | 2,200 B | **1,352 B** | **−38.5%** |
| whole payload | — | — | **−1.78%** |

A 1.78% rate saving from a two-line fix, and it had been silently paid on every bitstream in the
project until then.

**Visible only on partly-trained models.** As the model converges, the learned density approaches
the median-shifted one and the discrepancy shrinks toward zero. So this bug is *invisible* in a
finished model and glaring at 3,000 steps — which is the single strongest argument for the
mid-training gate layer of §15.3. A gate that only runs at the end of training would never have
found it.

### 22.4 The `z_uv` chroma hyper gate failure — still open

**Disclosed as an open defect, not a fixed one.**

The same *class* of problem as §22.3, in the two-branch model's chroma hyper prior: `update()` and
`forward()` disagree, so the coder writes bytes that are faithful to a table which is not the
density the rate loss was trained against. The gate's own diagnostic says it in one line:

```
** z_uv disagrees with its own table by +104 B (+4.9%)
   -- the coder is faithful to a table that is not the density
      the rate loss was trained against
```

**The evidence localising it** is the four-ladder pattern of §18.3: both single-branch ladders pass
the gate, both two-branch ladders fail it. The chroma hyper path is the only thing they do not
share.

**Magnitude:**

| ladder | steps | failures |
|---|---|---|
| `ladder_tb3k` | 3,000 | +1.29% / +1.85% / +2.24% |
| `ladder_p5` | 50,000 | −3.30% at β 0.002; `z_uv` +104 B (+4.9%) and +109 B (+9.5%) |

**Why it is not a correctness catastrophe:** `ŷ` remains bit-exact at every rate point in every
ladder. The codec encodes and decodes correctly. What is wrong is *efficiency* — a few hundred bytes
on a payload of tens of kilobytes, i.e. well under 1% of total rate — plus a −3.30% anomaly at the
lowest rate point, where `z_uv` is a large fraction of a very small payload.

**Why it is not yet fixed:** the fix requires the chroma hyper prior to expose a `coder_rows`
accessor in the same way §6.4's σ path does, so the gate can interrogate the model for the exact
table it will use rather than rebuilding it. That is a change to a module currently being trained
against by `ladder_p6`; changing it mid-flight invalidates 11,000 steps. It is the first item in §27.

### 22.5 Every metric computed on RGB

Six of the paper's seven metrics are **luma-only** (§11.2). Our `metrics.py` originally ran all
seven on RGB.

No crash, no implausible value, no gate failure — just seven numbers that are **not the paper's
metrics** and therefore cannot be compared to the paper's tables at all. This is the purest example
in the project of a bug whose only symptom is being wrong.

Found by reading WG1's own Quality Assessment Framework source rather than by any test, which is
worth noting: **no amount of internal testing finds a wrong convention.** Only the external source
does. Fixed by making the plane and the input range per-metric properties, with a test pinning each
one.

### 22.6 The two constants we read wrong

Both are in §9.1 and both would have produced a model that trains happily and is not JPEG AI:

| | we had | correct | how it would have failed |
|---|---|---|---|
| hyper latent width | 128 | **160** | hyper AE at the wrong width throughout; the `[128,64]` table shape we reasoned from is an *unused default* |
| chroma latent width | 48 | **96** | the entire chroma branch at half width; eq. (3)'s 256 would have been 208 |

Both came from the same mistake: **reading a class attribute as a value.** In the reference software
a class attribute is a *default*; the construction site is what decides. Our config loader now
asserts `hyper_latent == primary_latent` so correction 1 cannot silently return.

## 23. Environment and tooling errors

Recorded because they cost real time and because they are the kind of thing that is never written
down and always re-encountered.

### 23.1 The truncated DIV2K download — validating on the test set

**The most dangerous non-code error in the project.**

`DIV2K_valid_HR.zip` downloaded to **379 MiB of 428 MiB** and the extraction *appeared* to succeed.
The unzip produced a partial set of images with no error surfaced, so the training loop's validation
step silently ran on a **different image set than intended** — for a period, effectively on data
that overlapped the benchmark. Validation numbers were meaningless and, worse, optimistic.

**The recovery, in two steps:**

1. `jpegai/data/salvage_zip.py` — written for this, it walks the ZIP's local file headers directly
   and extracts every member whose **CRC-32 verifies**, recovering **88 of 100** images with
   certainty about which 88.
2. A resumed download, `curl -C -`, to the full **448,993,893 bytes**, then a re-extract to the full
   100.

**The permanent fixes:** the dataset loader now asserts the expected file **count** before training
starts, and `setup.sh` verifies archive sizes after download. A silent partial dataset is now a hard
failure.

### 23.2 `psnr-hvsm` cannot be installed — a permanent dead end

```
ERROR: Could not find a version that satisfies the requirement psnr-hvsm
       (from versions: none)
```

PyPI has wheels for `manylinux_2_17_x86_64` and `win_amd64` **only** — no macOS build, no arm64
build, and **no source distribution**, so there is nothing to compile. It also pins `numpy<2` while
the project runs 2.5.2.

Not fixable on this machine. We compute PSNR-HVS with **our own DCT implementation**, and §11.3
states the consequence: because every published figure is a BD-rate — a ratio between two curves
measured with the *same* metric implementation — a systematic offset largely cancels. Not perfectly,
so the `psnr_hvs` column is internally consistent and only approximately comparable to the paper's.

This one is disclosed rather than solved, and it is listed in §12 as unverified item 5.

### 23.3 The sandbox constraints

Recorded because they shaped the project's whole division of labour:

| symptom | cause | workaround |
|---|---|---|
| every `pip install` / dataset download fails | **no network egress** in the implementation environment | every network action written as a command block for the user to run |
| cannot create `.git` | sandbox denies it | all git operations run by the user |
| `diff <(a) <(b)` → "Operation not permitted" | **process substitution blocked** | write both sides to temp files first |
| matplotlib: `/Users/nizam/.matplotlib is not a writable directory` | home dir not writable | `export MPLCONFIGDIR="$TMPDIR/mpl"` |
| `nice` → "operation not permitted" | sandbox | run without it |
| `ps`, `timeout`, `cat -A` not found | not in the sandbox image | poll log files instead of processes |
| MPS unavailable in-sandbox but works for the user | sandbox has no Metal access | the user runs all training |

### 23.4 The PDF toolchain

Building this document was itself a small engineering problem.

**Absent:** `pdflatex`, `xelatex`, `tectonic`, `pandoc`, `wkhtmltopdf`, `typst`. So no LaTeX route.

**The route that works:** Markdown → HTML via the `markdown` module (extensions `tables`,
`fenced_code`, `toc`, `attr_list`, `md_in_html`, `sane_lists`) → PDF via **weasyprint**.

**And weasyprint fails out of the box:**

```
OSError: cannot load library 'libpango-1.0-0'
```

It looks for the Linux shared-object name while Homebrew installs
`/opt/homebrew/lib/libpango-1.0.0.dylib` — a different filename for the same library. Fixed by
pointing the dynamic loader at Homebrew's lib directory:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib /opt/anaconda3/bin/weasyprint in.html out.pdf
```

The `GLib-CRITICAL` warnings it prints on stderr are harmless. Output verified with `pypdf`: correct
page count, extractable text, and `— β Δ ≈ ± ×` all render.

### 23.5 Four small ones

| error | cause | fix |
|---|---|---|
| test collection failure | a test named `..._without_re-encoding` — a hyphen is not valid in a Python identifier | renamed `..._without_reencoding` |
| `_stub_metrics()` does not exist | misremembered helper name | the real one is `_install_stubs()` returning `_StubMetrics` |
| `KeyError: 'n_images'` in `write_markdown` | `n_images` lives *inside each curve*, not at top level | added it to both synthetic test curves |
| `ValueError: '.../b.json' is not in the subpath of '/Users/nizam/JPEC AI'` | `rerender` crashed while **printing** where it had written a temp file | a `_rel()` helper routed through all 7 display sites |

The third is worth one more sentence: the failing assertion was `abs(AVG + 28.571) < 0.05`, which
came out at −28.643. The tolerance **was** loosened — to 0.15 — but only after diagnosing *why*:
PCHIP's derivative at a knot depends on its neighbours, and the anchor has knots beyond both ends of
the overlap window, so a small legitimate difference is expected. The reason is recorded in a comment
next to the tolerance. **Loosening a tolerance without recording why is how a test stops being a
test.**

## 24. Dead ends

| dead end | what happened |
|---|---|
| the paper's supplementary material | IEEE subscription; the professor was asked and it did not arrive in time. Per-stage widths remain **OURS** |
| ITU-T / ISO purchase of Part 1 | both paywalled. Redirected to the WG1 GitLab reference software, which turned out to be *better* — it has the constants as literals (chapter 9) |
| the T1/T2/TP/TR tables | Git-LFS objects inside checkpoints we skipped with `GIT_LFS_SKIP_SMUDGE=1`. Deferred to phase 10, which learns its own |
| `psnr-hvsm` | §23.2. Permanent |
| CLIC 2024 and the JPEG AI test set | not publicly downloadable. Tables III, IV and VI are **not reproducible by us**; Kodak/Table V is the only comparable figure and it is why the target is −7.5% |
| `sigma_bound_offset = 0.5` | confirmed as a constant, meaning still unknown. Two readings, mutually exclusive; §12 item 3 |
| Flickr2K | downloaded as optional extra training data, never used — DIV2K's 6,400 crops proved sufficient at our step budget |

## 25. Lessons

The transferable ones, each earned from a specific bug above.

**1. Build the measuring instrument before the thing being measured, and validate it on things
whose answer is already known.** The BD-rate bug (§22.1) was caught because **WebP's** number was
implausible, not ours. Without a codec of known performance in the harness, a 17-point error in the
measurement code is indistinguishable from a bad codec.

**2. The measurement tool is part of the system under test.** The most expensive bug in the project
was in `metrics/bdrate.py`, not in the codec.

**3. When there is no ground truth, test an invariance.** BD-rate has no reference value to check
against, so it was checked against a property it must satisfy: anchor points outside the overlap
window cannot change the answer. PCHIP 0.04, cubic 17.08 (§15.3). That single test both found the
bug and proved the fix.

**4. Assert on real bytes, at every checkpoint, during training.** The round-trip gate (§15.2)
caught §22.3 and localised §22.4. Two of six bugs were visible **only** through a mid-training gate
on actual bitstream sizes — invisible to unit tests, invisible to the loss curve, invisible in a
converged model.

**5. Prefer constants that confirm each other.** `sigma_precision = 7` is certain because
`5 + 5 + 7 = 17` and a *different* file independently hardcodes 17 (§9.2). A value read once might be
a default; a value satisfying an arithmetic identity with values from other files is a value.

**6. In someone else's codebase, a class attribute is a default, not a value. Find the construction
site.** Both of §22.6's wrong constants came from this one mistake, and both would have built a
model that trains happily and is not JPEG AI.

**7. No internal test finds a wrong convention.** Every metric on RGB (§22.5) passed every test we
had. Only WG1's own source settled it. For anything defined externally, read the external
definition.

**8. Compare against the honest figure, not the flattering one.** The paper's headline is −16.2% on
its own test set; the comparable figure for our dataset and decoder complexity is **−7.5%** (§8.4).
Using the headline would have overstated our shortfall by 8.7 points and could have made a *correct*
result look like a bug.

**9. Measure bounds before hunting bugs.** Tier A's saturation looked exactly like a defect. Two
cheap measurements — quantiser off, and PCA at the same width — proved it was a hard capacity limit
and saved days of searching for a bug that does not exist (§20).

**10. Report the failures.** `ladder_tb3k` is a meaningless 3,000-step run whose *gate failures*
produced §18.3's four-ladder table, which localised an open defect to one module. A suppressed
warning would have been worth nothing.
