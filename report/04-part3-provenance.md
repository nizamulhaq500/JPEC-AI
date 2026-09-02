<div class="page-break"></div>

# Part III — Provenance: where every piece of information came from

*A report that mixes "the standard says" with "we assumed" is not checkable. This part names the
source of every fact the implementation rests on, including the sources we could not reach and
what we did instead.*

## 7. What we could and could not obtain

### 7.1 The access problem, stated plainly

| source | status | consequence |
|---|---|---|
| The overview paper (IEEE TCSVT) | **obtained** — provided by the supervisor as a PDF | the architecture, all equations, all six tables, both figures |
| The paper's **supplementary material** (Figs. 6–8, Appendices A–C) | **not obtained** — IEEE Xplore subscription required | per-stage network widths unknown; ours are inferred |
| **ITU-T T.840-1 \| ISO/IEC 6048-1** (Part 1, the core standard) | **not obtained** — ITU and ISO both require purchase | every numeric constant, the normative CDF tables, the T1/T2/TP/TR tables |
| **WG1 GitLab** (`gitlab.com/wg1/jpeg-ai`) — Part 3, the reference software | **obtained** — publicly reachable | **this rescued the project.** See chapter 9 |
| The four **ONNX** trained parameter sets | **not obtained** — distributed via a link inside Part 1 | we train our own weights from scratch |
| The **T1/T2/TP/TR** learned tables | **not obtained** — Git-LFS objects inside the reference checkpoints | phase 10 learns its own |

The critical realisation, and it is worth stating as a research method: **the reference software
is the normative implementation of Part 1, and every constant in the standard appears in it as a
literal.** The standard's *text* is paywalled; the standard's *behaviour* is not. Nine of our ten
open questions were answered by grepping `params.py` files. Chapter 9 is that extraction.

### 7.2 What was therefore genuinely unavailable

Three things, and each is disclosed wherever it affects a number:

1. **Per-stage trunk widths** for the analysis and synthesis transforms. The paper gives totals
   and kMAC/pixel; the reference software exposes per-*decoder* hidden widths (§9.1) but not the
   analysis transform's stage widths, which are constructed from configuration we do not have.
   Our widths are **OURS**.
2. **The RVS/LSBS tables** `T1`, `T2`, `TP`, `TR`. Learned weights, LFS-tracked, not source
   literals. Phase 10 learns them with the rest of the codec frozen. **Documented deviation.**
3. **`isigma_pad_value = 1411`** — stated in the paper's eq. (7), *not* found anywhere in the
   reference software. Tagged **PAPER, unconfirmed**. Harmless: it is a boundary padding value.

### 7.3 The environment constraint that shaped how we worked

Worth recording because it determined the project's division of labour. The implementation
environment has **no network egress**. Every action requiring the network — installing a package,
downloading a dataset, a `git push` — had to be performed by the user in a terminal, from
commands written for them. This is why `setup.sh` exists as a single idempotent script rather
than a list of instructions, and why chapter 27's remaining work is written as copy-paste command
blocks.

## 8. The paper, and how its tables were recovered

### 8.1 The extraction problem

The paper's PDF has an unusual property: **Tables I–VI and Figure 2 are not text.** They are
vector glyph outlines inside PDF Form XObjects, which means text extraction returns nothing at
all for them. The prose extracted cleanly; every number we needed did not.

Two independent recoveries were performed, and they cross-check each other:

1. **`paper/rasterize.py`** — a from-scratch PDF vector-path rasteriser, written for this
   project, that walks the content streams of the Form XObjects and renders the glyph outlines to
   PNG. That produced nine images: both halves of Figure 1, Figure 2, and Tables I–VI.
2. **Screenshots supplied by the user** on 2026-08-26, read directly.

The screenshots are treated as authoritative where the two differ. In the event they did not
differ: the rasteriser's Table III numbers were confirmed **correct** against the screenshot.
That is a pleasant result for a 500-line rasteriser and it means the extracted figures can be
trusted for the tables we did not screenshot.

Recovered artifacts:

```
paper/paper_text.txt          full prose, 18 pages, page-delimited
paper/rasterize.py            the from-scratch vector-path rasterizer
paper/imgs/p03_1_Im0.png      Fig. 1 left  — primary/luma branch, encoder + decoder
paper/imgs/p03_0_Im1.png      Fig. 1 right — secondary/chroma branch
paper/imgs/table_p2_Fm0.png   Table I   — the five parts
paper/imgs/table_p5_Fm0.png   Table II  — markers and hex codes
paper/imgs/table_p8_Fm0.png   Fig. 2    — the MCM 4-stage checkerboard
paper/imgs/table_p9_Fm0.png   Table III — main results
paper/imgs/table_p12_Fm0.png  Table IV  — tool ablation
paper/imgs/table_p12_Fm1.png  Table V   — Kodak
paper/imgs/table_p13_Fm0.png  Table VI  — CLIC 2024
```

These files are not redistributed in the public repository (IEEE copyright).

### 8.2 Verifying the AVG column — establishing what the paper's headline number *is*

The paper's tables have an **AVG** column and never say how it is computed. Since every result in
this report is compared against it, that had to be settled. Three hypotheses: the unweighted mean
of the seven metrics; a weighted mean; or something else entirely.

Method: recompute the unweighted arithmetic mean of the seven per-metric BD-rates for **all 19
data rows** of Tables III, IV, V and VI, and compare to the printed AVG.

Result: **it matches on all 19 rows.** The AVG is the plain unweighted arithmetic mean of the
seven metric BD-rates. That is now a constant in our code:

```python
PAPER_SEVEN = ["ms_ssim", "vif", "fsim", "vmaf", "nlpd", "psnr_hvs", "iw_ssim"]
```

with the average taken across them unweighted, and BD-rate computed **per metric first**.

This is a small result that carries a lot of weight. Without it, every comparison in Part V would
be against a number whose definition we had guessed.

### 8.3 Errors and anomalies found in the paper itself

Stated not as criticism but because an implementer will hit them.

**(a) Two typos in eq. (4)**, the YCbCr→RGB conversion. As printed, the second line indexes
`x̂_UV[1]` where it must be `[0]` (Cb), and the third line's coefficient is `0.07222` where BT.709
is `0.0722`. Corrected against the textbook BT.709 inverse — which, as §9.5 shows, turns out to
be a *legal configuration choice* rather than a deviation, because the colour matrix is a
signalled header field with no normative value at all.

**(b) Prose disagrees with Table III.** §VII-B quotes AVG figures of 16.0 / 20.2 / 21.1 and 13.9
/ 19.7 / 27, and 13 / 28 / 215 kMAC/pixel. The table says 16.2 / 20.2 / 22.1, 14.4 / 19.9 / 27.0,
and 14 / 28 / 215. **We cite the table**, since the table's AVGs are the ones that reproduce
arithmetically from the per-metric columns (§8.2).

**(c) Two arithmetic anomalies in Table IV**, described in §6.8: an ablation with *higher* MAC
count than all-on, and *every* ablation decoding faster than all-on. Together they imply ≳6%
timing noise, which sets a floor on how finely the paper's timing numbers can be read.

**(d) A labelling slip in Figure 1.** All four blocks of the *chroma* hyper path are printed
"Primary hyper …". The architecture is unambiguous from the surrounding wiring; the labels are
not.

### 8.4 What Figure 1 confirmed that the text does not state

Reading the figure carefully settled six things that the prose leaves open, each of which changes
an implementation decision:

1. **MCM is luma-only.** The chroma branch's corresponding position is a bare addition — eq. (2)
   and nothing more.
2. **`Iσ` feeds only the arithmetic coder.** It does not enter the synthesis path. So the scale
   decoder's output is needed *before* entropy decoding and nowhere after it, which is exactly
   what makes the integer/float split of §6.4 possible.
3. **The eq.-(3) concatenation takes the luma *latent*, not the decoded luma image.** A natural
   misreading, and it would change the chroma synthesis input from 256 channels to 97.
4. **Cross-component flow is luma→chroma only, at exactly two points** — the chroma hyper encoder
   and the eq.-(3) concatenation. Therefore **the luma branch is completely independent of
   chroma**, which is what licenses the monochrome fast path we measured (§18.4).
5. **Four arithmetic coders, three marker segments.** Consistent with SOZ carrying both hyper
   streams.
6. **The hyper encoder takes `y`, not `x`.**

Figure 2 was partially resolvable from the raster: four stages, a 2×2 spatial tile, and colour
**constant along the channel axis** — so MCM partitions space and never channels. Which specific
cell is stage 1 was *not* resolvable from the image, and was settled from the reference software
instead (§9.3).

## 9. The WG1 reference software — nine of ten open questions

Checked out with `GIT_LFS_SKIP_SMUDGE=1` so the multi-gigabyte weight files remain pointer stubs.
Everything below is a literal in a source file, with the path recorded. Appendix E is the full
table; this chapter is the six extractions that mattered most, including **two places where our
own earlier readings were wrong**.

### 9.1 Channel widths — and two corrections to ourselves

**Correction 1: the hyper latent is 160, not 128.** We had reasoned that because the standard's
hyper CDF table is `[128, 64]`, the hyper latent must be 128 channels, and had configured it so.
That was wrong. The hyper autoencoder is **channel-preserving** — every hyper module is
constructed with `chs = chs_ls`, the latent width of its own branch:

```python
# coding_tools/core_models/CCS_SGMM/common_modules.py:116-128
self.hyper_entropy       = FactorizedProbModel(self.chs_ls, max_symbol=self.z_range - 1)
self.hyper_encoder       = ...create_instance(..., chs=self.chs_ls, ...)
self.hyper_decoder       = ...create_instance(..., chs=self.chs_ls, ...)
self.hyper_scale_decoder = ...create_instance(..., chs=self.chs_ls)
```

Channel preservation is visible in the modules themselves: the hyper encoder is five
`conv3x3(chs, chs)` (two of them stride 2), and the scale decoder ends
`conv1x1(chs, chs*16) → PixelShuffle(4)`, returning to `chs`. So there is **no independent hyper
width at all**, and the `128` we anchored on is an *unused fallback default*
(`kwargs.get('chs_ls', 128)`). The paper was right and we were wrong. Our config loader now
**asserts** `hyper_latent == primary_latent` so this cannot silently return.

**Correction 2: the chroma latent is 96, not 48.** We had recorded `IN_CHS: int = 48` and
`CHS_LS: int = 48` from the secondary component's source files. Those are **class-attribute
defaults**, overridden at construction. The top-level model sets both widths explicitly:

```python
# coding_tools/core_models/CCS_SGMM/ccs_sgmm_tool.py:67-82
N_luma   = 160
N_chroma = 96
model_y  = SepChannelsSGMMTool(1, chs_ls=N_luma,   ccs_id=0, ...)
model_uv = SepChannelsSGMMTool(2, chs_ls=N_chroma, chs_ls_supp=N_luma,
                               chs_in_supp=1, downsample_factor=2, ccs_id=1, ...)
```

So the paper's 96, and eq. (3)'s `256 = 96 + 160`, were correct. **Had we acted on the 48, the
chroma branch would have been built at half width — a model that trains happily and is silently
not JPEG AI.**

The lesson, and it is the single most transferable methodological point in this chapter: **in this
codebase a class attribute is a default, not a value. Always find the construction site.**

Confirmed widths:

| quantity | value | source |
|---|---|---|
| luma latent `N_luma` | **160** | `CCS_SGMM/ccs_sgmm_tool.py:67` |
| chroma latent `N_chroma` | **96** | `CCS_SGMM/ccs_sgmm_tool.py:68` |
| luma hyper latent | **160** | derived — channel-preserving |
| chroma hyper latent | **96** | derived — same |
| `p̈_Y` pre-shuffle | **640** = 4×160 | hyper decoder's last layer `conv3x3(chs, 4·chs)` |
| `p̈_UV` pre-shuffle | **384** = 4×96 | same, chroma |
| secondary synthesis input | **256** = 96+160 | eq. (3), confirmed at the construction site |
| secondary downsample factor | 2 | `ccs_sgmm_tool.py:79` |
| primary latent divisibility | **% 32 == 0** | `contexts/MCM_phases.py`'s `chs2group()` asserts it |

That last row is a real constraint, not trivia: `chs2group(chs)` asserts divisibility by 32 and
returns `max(1, chs // 32)` as the `groups` argument of the MCM convolutions. It constrains the
*primary* latent only (MCM is luma-only). 160 % 32 = 0 ✓, and our tier A 96 % 32 = 0 ✓ — which is
partly why 96 was chosen as the reduced width.

**And the structural asymmetry the paper never states:** the encoder directory contains
`bop_prim`, `bop_sec`, `hop_prim`, `hop_sec` — and **no `sop_*`**. There are three decoders and
two encoders; SOP reuses BOP's encoder, which a profile configuration file makes explicit. We
record this as `has_encoder: false` on decoder 0.

Per-decoder hidden widths, from the reference software's class attributes — the closest available
substitute for the supplement's Figs. 6–8:

| decoder | in | supp | out | hidden |
|---|---|---|---|---|
| SOP primary | 160 | — | 1 | 96, 64 |
| BOP primary | 160 | — | 1 | 64, 64, 96 |
| HOP primary | 160 | — | 1 | 128, 128 |
| SOP secondary | 96 | 160 | 2 | 64, 64 |
| BOP secondary | 96 | 160 | 2 | 64, 64, 128 |
| HOP secondary | 96 | 160 | 2 | 64, 64 |

### 9.2 The σ constants, and a precision chain that closes on itself

| constant | value | source |
|---|---|---|
| `sigma_quant_level` | **32** | `CCS_SGMM/params.py` |
| `sigma_quant_min` | **0.11** | same |
| `sigma_quant_max` | **54.82** | same |
| `sigma_bound_offset` | **0.5** | same — still unexplained, see below |
| `sigma_precision` | **7** | `coding_tools/quantization/params.py` |
| `gain_vector_precision` | **5** | same |
| `beta_displacement_precision` | **5** | same |
| `scaler_precision` | **10** | derived = 5 + 5 |
| `scaled_sigma_precision` | **17** | = 10 + 7 |

The last row is the useful cross-check. A *different* file — `lsbs_scale_mode.py:54` —
independently **hardcodes** `scaled_sigma_precision = 17`. Since `10 + 7 = 17`, the whole
precision chain closes on itself. So `sigma_precision = 7` is *certain* rather than merely read
once. (We had guessed 8.)

This is the methodological habit worth naming: **prefer constants that confirm each other.** A
value read once might be a default; a value that satisfies an arithmetic identity with two other
values read from other files is a value.

**One constant remains unexplained.** `sigma_bound_offset = 0.5` is confirmed as a constant and is
unused in our code. It has two plausible readings we cannot separate: a rounding offset applied
before the shift — which would make the σ-class rule round-*nearest* rather than round-*up*, and
would then contradict the reachability argument of §6.4.1 — or a widening of the CDF tail bound.
The reachability argument and our own escape-rate measurement both point at round-up, so round-up
is what is implemented, and this is flagged as the one place to revisit if the supplement arrives.

### 9.3 The MCM stage order — two independent confirmations

Our guess had been `(0,0) → (1,1) → (0,1) → (1,0)`, diagonal first. Both confirmations agree.

**First**, from the shuffle that forms the stages:

```python
# components/contexts/utils.py  ContextUtils.down_shuffle
y = y.reshape(B, iC, oH, factor_hw, oW, factor_hw)
y = y.permute(0, 1, 2, 4, 3, 5)
y = y.reshape(B, iC, oH, oW, factor_hw * factor_hw)
part1, part2, part3, part4 = torch.chunk(y, chunks=4, dim=4)
return part1.squeeze(4), part4.squeeze(4), part2.squeeze(4), part3.squeeze(4)
```

After that permute the raster order is `part1 = (0,0)`, `part2 = (0,1)`, `part3 = (1,0)`,
`part4 = (1,1)`. The **return** order is `(part1, part4, part2, part3)` — diagonal first.
`up_shuffle` unpacks the mirror image.

**Second**, and more convincing because it comes from an unrelated concern — the odd-size guards:

```python
# components/contexts/context.py
if h_ls % 2 == 1 and stage_id in [1, 3]:   # drop the redundant last ROW
if w_ls % 2 == 1 and stage_id in [1, 2]:   # drop the redundant last COL
```

A stage needs the row guard **iff** its vertical offset is 1, and the column guard **iff** its
horizontal offset is 1. Row guard on `{1,3}` and column guard on `{1,2}` forces stage 1 = (1,1),
stage 2 = (0,1), stage 3 = (1,0), and therefore stage 0 = (0,0). **Same answer, from code written
for a completely different purpose.** That is the strongest kind of confirmation available without
the standard text.

### 9.4 me-tANS constants — and a wrong premise corrected

Our open question had asked for the tANS `tableLog` and spread function. **The premise was
wrong**: the reference parameterises by *probability mass bits*, not table log.

| constant | value | note |
|---|---|---|
| `mass_bits` | **8** | → 2⁸ = 256 states per σ-class. **Not** a `tableLog` |
| escape threshold | **2⁻¹¹** | `get_outbound_values(probs, threshold=1/2**11)` |
| symbol ordering | **zig-zag** | `get_sequence` / `get_inverse_sequence` |
| substreams | **2, interleaved** | via the `cdf_first` / `cdf_second` state split |

Our guessed `tans_table_log: 11` would have produced 2,048 states per class and an **8× larger
table** — a config that works and is not the standard. Renamed to `tans_mass_bits: 8`.

The packed transition word, which is what makes the decoder arithmetic-free:

```python
cdf_first, cdf_second = cdfs - (cdfs >> 1), (cdfs >> 1) + 128
...
return ((num_bits << 24) | (state_next << 16) | (symbols & 65535)).astype(np.uint32)
```

### 9.5 Skip mode, and the colour transform question that dissolved

**Skip mode** — we had `skip_threshold: 0` with a comment saying "calibrate in phase 9". No
calibration is needed:

| constant | value |
|---|---|
| `skip_block_size` | **1** |
| `thr_skip` | **382** |
| `skip_judge_thr` | **3** |
| `skip_cube_thr` | **1** default, **3** under the common test conditions |

We use the CTC value 3, since CTC is what the paper's tables were produced under.

**The colour transform.** Our open question asked for "the exact eq. (4) coefficients", on the
theory that the printed equation had two typos (§8.3a). **There are no normative coefficients.**
The reference software reads a `clr_tr_matrix` of **nine 8-bit integers from the picture header**,
defaulting to the identity, and inverts it numerically at decode time:

```python
inv_matrix = torch.inverse(clr_tr_matrix / 255.0) * 255.0
```

So eq. (4) is one *instance* of a signalled matrix, not a constant of the standard. Our decision
to ship the textbook BT.709 inverse is therefore a **legal configuration choice rather than a
deviation** — a better position than we thought we were in. The question dissolved rather than
being answered.

### 9.6 Codestream details, and a cumulative capability list

All ten marker codes confirmed as transcribed. Three details worth recording:

- TOH is named `MARKER_TON` in the software (`"tool_header"`).
- **Mandatory substreams are PIH, SOZ, SORP, SORS**, confirming the minimal conformant stream
  `SOC · PIH · SOZ · SORp · SORs · EOC`.
- The primary and secondary `z` marker values are **equal**, so `ẑ_Y` and `ẑ_UV` share a single
  threaded SOZ substream rather than being separately delimited.
- **`synthesis_transform_id` is a cumulative capability list, not a selector**: SOP signals `[0]`,
  BOP `[1,0]`, HOP `[2,1,0]`. A HOP stream is therefore decodable by an SOP decoder. This is the
  mechanism behind the multi-branch scalability claim and it is not visible in the paper.

### 9.7 Summary: the ten open questions

| # | question | answer | outcome |
|---|---|---|---|
| 1 | `ẑ_Y` channel count | **160** — the hyper AE is channel-preserving | our 128 was **wrong** |
| 2 | `Iσ` → σ-class mapping | log-spaced, 32 levels over [0.11, 54.82]; class = `ceil(Iσ/2⁷)` | resolved |
| 3 | `step` / `sigmaPrecision` | `sigma_precision = 7` (we guessed 8) | resolved, triple-confirmed |
| 4 | `skip_threshold` | **382** | resolved |
| 5 | MCM stage ordering | `(0,0) → (1,1) → (0,1) → (1,0)` | our guess was **right** |
| 6 | secondary analysis stage count | `downsample_factor = 2` on the chroma branch | resolved |
| 7 | tANS `tableLog` / spread | `mass_bits = 8`, zig-zag ordering | **premise was wrong** |
| 8 | `p̈_UV` pre-shuffle channels | **384** = 4 × 96 | resolved |
| 9 | eq. (4) coefficients | there are none — it is a signalled header field | **dissolved** |
| 10 | `ẑ_Y` / `ẑ_UV` delimiting | one shared threaded SOZ substream | resolved |

Nine resolved, one dissolved, **two of our own readings corrected**, and both of the corrections
were the dangerous kind — a model that trains fine and is not JPEG AI.

## 10. Datasets — what, from where, and how prepared

| dataset | content | source | our use |
|---|---|---|---|
| **Kodak** | 24 images, 768×512, uncompressed PNG | `r0k.us/graphics/kodak/` | **the benchmark.** Matches the paper's Table V |
| **DIV2K train** | 800 images, 2K resolution | `data.vision.ee.ethz.ch/cvl/DIV2K/` | training data |
| **DIV2K valid** | 100 images, 2K | same | validation during training |
| **Flickr2K** | ~2,650 images | optional | not used; DIV2K alone proved sufficient |
| **CLIC 2024 validation** | — | not obtained | the paper's Table VI is therefore not reproducible by us |
| **JPEG AI test set (CTTC)** | 50 images, 1K–4K | not publicly downloadable | the paper's Tables III/IV are not reproducible by us |

On disk: 15 MB of Kodak, 4.1 GB of DIV2K, and 679 MB of extracted training crops.

**Why Kodak and not the paper's own test set.** The JPEG AI test set is not publicly
downloadable, so Tables III and IV cannot be reproduced by anyone outside WG1. Kodak can, and the
paper publishes Kodak results in Table V. That is the *only* directly comparable figure available
to us, and it is why §6.12's dataset-penalty table changed our target from −16.2% to −7.5%.

**Preparation.** `jpegai/data/prepare_crops.py` extracts **6,400** random 256×256 crops from the
800 DIV2K training images (8 per image), rejecting crops whose variance is below a threshold —
otherwise a substantial fraction of the training set is featureless sky, and the model spends
capacity learning to compress nothing. Crops are stored as PNG with a manifest so the training
set is reproducible.

**Validation is 100/100 DIV2K images**, and getting there involved the most dangerous bug in the
project — the download was silently truncated and validation ran on the *test* set for a period.
§23.2 is that story.

## 11. Software dependencies, and the metric backend problem

### 11.1 The stack

| package | why |
|---|---|
| `torch`, `torchvision` | the model. MPS backend for Apple Silicon |
| `compressai` | reference implementations of the entropy-model rungs of §5.6, pretrained baselines to benchmark against, and a working range coder |
| `numpy`, `scipy` | numerics; `scipy.interpolate.PchipInterpolator` is the BD-rate interpolant |
| `pillow`, `pillow-avif-plugin` | the JPEG / WebP / AVIF anchors |
| `pytorch-msssim`, `piq`, `pyiqa` | metric backends |
| `ffmpeg` (system, via Homebrew) | VMAF, using Netflix's own implementation |
| `matplotlib`, `pandas` | RD plots and tables |
| `pytest` | 332 tests |

### 11.2 The metric conventions that are not guessable

Read from `ref/jpeg-ai-qaf/metrics.py` — the committee's own Quality Assessment Framework. **All
metrics are computed at 10-bit internal precision, not 8.**

| metric | plane | input range | QAF's backend |
|---|---|---|---|
| MS-SSIM | **Y** | 0…1023 | `pytorch_msssim` |
| VIF | **Y** | 0…1 | `IQA_pytorch.VIFs(channels=1)` |
| FSIM | **RGB** | 0…1 | `IQA_pytorch.FSIM(channels=3)` |
| NLPD | **Y** | 0…1 | `IQA_pytorch.NLPD(channels=1)` |
| IW-SSIM | **Y** | **0…255** | QAF's own implementation |
| PSNR-HVS | **Y** | 0…1, replicate-padded to a multiple of 8, float64 | `psnr_hvsm` |
| VMAF | Y | — | Netflix binary v2.2.1 |

**Six of the seven are luma-only. Only FSIM sees colour.** Our `metrics.py` originally ran every
metric on RGB. That is a silent correctness bug of the worst kind: it produces entirely plausible
numbers that cannot be compared to the paper's at all. Fixed — the plane and range are now
selected per metric.

**And the seventh metric is PSNR-HVS, not PSNR-HVS-M.** QAF calls `psnr_hvs_hvsm(...)`, which
returns both, and keeps the **first**. We compute and report both, but only `psnr_hvs` is one of
the seven and only it is averaged into AVG.

### 11.3 Backend substitutions, and one permanent dead end

Two substitutions, both recorded in code as `BACKEND_NOTES`:

- `piq` in place of `IQA_pytorch` for VIF, FSIM and IW-SSIM;
- `pyiqa` for NLPD. `pyiqa`'s NLPD rejects single-channel input, so we feed Y replicated across
  three channels. **This is exact**, not an approximation: any weighted-sum luma conversion of a
  grey image returns that grey unchanged, because the coefficients sum to 1.

And one dead end that is worth documenting because it is *permanent*: **`psnr-hvsm` cannot be
installed on this machine.** PyPI has wheels for `manylinux_2_17_x86_64` and `win_amd64` only —
no macOS build, no arm64 build, and **no source distribution** — and the package pins `numpy<2`
while we run 2.5.2. So `psnr_hvs` and `psnr_hvsm` run on **our own DCT implementation** of the
metric.

Why that is acceptable, stated explicitly because it affects a headline number: every published
figure is a **BD-rate**, which is a ratio between two curves measured with the *same* metric
implementation. A systematic offset in the metric largely cancels. It does not cancel *perfectly*
— BD-rate is not exactly scale-invariant — so our `psnr_hvs` column is internally consistent and
only approximately comparable to the paper's. That is disclosed wherever the number appears.

## 12. What remains unverified

The honest residue. Nine items, in descending order of how much they would change.

| # | item | status | if wrong, the effect is |
|---|---|---|---|
| 1 | per-stage analysis/synthesis widths | **OURS** | our complexity numbers are not the standard's; BD-rate comparisons at "matched complexity" are approximate |
| 2 | T1/T2/TP/TR tables | **learned by us** | phase 10's RVS/LSBS gains will differ from the paper's 2.2/0.4 pp |
| 3 | `sigma_bound_offset = 0.5`'s meaning | **CONFIRMED as a constant, unexplained** | if it makes the σ-class rule round-nearest, §6.4.1's implementation is wrong at half the grid points |
| 4 | `isigma_pad_value = 1411` | **PAPER, unconfirmed** | a boundary artefact in RVS only |
| 5 | `psnr_hvs` backend | **ours** | the `psnr_hvs` column is approximately, not exactly, comparable |
| 6 | which Figure-2 cell is stage 1 | resolved from software, not from the figure | none — two independent code confirmations |
| 7 | our training recipe entirely | **OURS** | the largest confound in every comparison. §26.1 |
| 8 | the MOP decoder (id 3) | **entirely ours, not in the standard** | it is an extra data point, clearly labelled, disabled by default |
| 9 | 6:1:1 luma weighting | **OURS** | affects the luma/chroma balance directly, which is where our deficit is. §26.2 |

Items 7 and 9 deserve emphasis because they are the ones most likely to explain the gap between
our results and the paper's, and neither is a bug — they are *choices we had to make* because the
paper says only "prioritise luma" and gives no training recipe at all.
