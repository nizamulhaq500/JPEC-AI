# 06 — Normative constants, extracted from the WG1 reference software

**What this document is.** The overview paper (Esenlik et al., TCSVT 36(2):2520-2537)
describes the architecture completely but omits most numeric constants, and the
documents that carry them — ITU-T T.840 | ISO/IEC 6048 and the paper's own
supplementary material — are paywalled. The WG1 **reference software** is not: it is
the normative implementation of Rec. ITU-T T.840.1 | ISO/IEC 6048-1, and every
constant in the standard appears in it as a literal.

So this file records every constant we extracted, with the file it came from, so that
any number in `jpegai/config/*.yaml` can be traced to a source. Three tags are used
throughout the configs and this document:

| tag | meaning |
|---|---|
| **CONFIRMED** | read out of the reference software. Normative. Do not tune. |
| **PAPER** | stated in the overview paper's main text. |
| **OURS** | our choice, no normative source. Free to tune; listed in §7 below. |

Reference software root in this repo: `ref/jpeg-ai-reference-software/`. All paths
below are relative to `src/codec/` inside it unless stated otherwise.

---

## 1. What this changed, and one correction to my earlier notes

Nine of the ten items previously listed as open questions in
[04-reference-data.md §7](04-reference-data.md) are now resolved. Two of my earlier
readings were **wrong** and are corrected here:

**Correction 1 — the hyper latent is 160, not 128.** I had argued that because the
hyper CDF table reads `[128, 64]`, the hyper latent must be 128 channels, and set
`full.yaml: hyper_latent: 128` on that basis. That was wrong. The hyper autoencoder is
**channel-preserving**, and every hyper module is constructed with `chs=self.chs_ls` —
the latent width of its own branch:

```python
# coding_tools/core_models/CCS_SGMM/common_modules.py:116-128
self.hyper_entropy       = FactorizedProbModel(self.chs_ls, max_symbol=self.z_range - 1)
self.hyper_encoder       = ...create_instance(..., chs=self.chs_ls, ...)
self.hyper_decoder       = ...create_instance(..., chs=self.chs_ls, ...)
self.hyper_scale_decoder = ...create_instance(..., chs=self.chs_ls)
```

Channel-preservation is visible in the modules themselves: the hyper encoder is five
`conv3x3(chs, chs)` (two with stride 2), and the hyper scale decoder ends
`conv1x1(chs, chs*16) → PixelShuffle(4)`, which returns to `chs`. So there is no
independent hyper width at all, and the `128` I anchored on is the *unused fallback
default* `kwargs.get('chs_ls', 128)` in `CCS_SGMM/sep_chan_tool.py:69`. **The paper was
right and I was wrong.** Fixed in both configs; the loader now asserts
`hyper_latent == primary_latent`.

**Correction 2 — the chroma latent is 96, not 48.** While reading
`components/autoencoder_data/`, I recorded `IN_CHS: int = 48` / `CHS_LS: int = 48` from
the `*_sec.py` files as the chroma latent width. Those are **class-attribute defaults**
that are overridden at construction. The top-level model sets both widths explicitly:

```python
# coding_tools/core_models/CCS_SGMM/ccs_sgmm_tool.py:67-82
N_luma   = 160
N_chroma = 96
model_y  = SepChannelsSGMMTool(1, chs_ls=N_luma,   ccs_id=0, ...)
model_uv = SepChannelsSGMMTool(2, chs_ls=N_chroma, chs_ls_supp=N_luma,
                               chs_in_supp=1, downsample_factor=2, ccs_id=1, ...)
```

So the paper's 96, and eq. (3)'s `256 = 96 + 160`, were correct. Had I acted on the 48,
the chroma branch would have been built at half width — a model that trains happily and
is silently not JPEG AI. `full.yaml` needed **no** change for these three values.

The lesson worth carrying: in this codebase a class attribute is a *default*, not a
*value*. Always find the construction site.

---

## 2. Channel widths

| quantity | value | source |
|---|---|---|
| luma (primary) latent `N_luma` | **160** | `CCS_SGMM/ccs_sgmm_tool.py:67` |
| chroma (secondary) latent `N_chroma` | **96** | `CCS_SGMM/ccs_sgmm_tool.py:68` |
| luma hyper latent | **160** | derived: `chs=chs_ls`, channel-preserving |
| chroma hyper latent | **96** | derived, same |
| `p̈_Y` pre-shuffle | **640** = 4×160 | hyper decoder's last layer is `conv3x3(chs, 4*chs)`, `autoencoder_hyper/decoder/base.py` |
| `p̈_UV` pre-shuffle | **384** = 4×96 | same, chroma branch |
| secondary synthesis input | **256** = 96+160 | `*_sec.py` take `chs_ls=<chroma>`, `chs_supp=<luma>` |
| secondary downsample factor | 2 | `ccs_sgmm_tool.py:79` |
| primary latent divisibility | **% 32 == 0** | `contexts/MCM_phases.py` `chs2group()` asserts it |

`chs2group(chs)` asserts `chs % 32 == 0` and returns `max(1, chs // 32)`, used as the
`groups` argument of the MCM phase convolutions. MCM is luma-only, so this constrains
the primary latent only. 160 % 32 = 0 ✓, and our Tier A 96 % 32 = 0 ✓.

**Structural asymmetry the paper does not state: there are three decoders but only two
encoders.** `components/autoencoder_data/encoder/` contains `bop_prim`, `bop_sec`,
`hop_prim`, `hop_sec` — and no `sop_*`. SOP reuses the BOP encoder, which
`cfg/profiles/bopEnc_sopDec.json` makes explicit. Recorded as `has_encoder: false` on
decoder 0 in both configs.

Per-decoder hidden widths (the supplementary material's Figs. 6–8 equivalent), from the
`C1/C2/C3` class attributes:

| decoder | in | supp | out | hidden |
|---|---|---|---|---|
| SOP primary | 160 | — | 1 | 96, 64 |
| BOP primary | 160 | — | 1 | 64, 64, 96 |
| HOP primary | 160 | — | 1 | 128, 128 |
| SOP secondary | 96 | 160 | 2 | 64, 64 |
| BOP secondary | 96 | 160 | 2 | 64, 64, 128 |
| HOP secondary | 96 | 160 | 2 | 64, 64 |

All data decoders have `out_scale_factor = 16` (four stride-2 stages).

---

## 3. Scale (σ) quantisation — resolves open questions 2 and 3

`Iσ` is an integer index in the **log** domain. The mapping to a linear σ is
log-spaced over a fixed range:

```python
# coding_tools/core_models/CCS_SGMM/common_modules.py
self.log_k = (np.log(self.sigma_quant_max) - np.log(self.sigma_quant_min)) \
             / (self.sigma_quant_level - 1)
...
self.hyper_scale_decoder.sigma_idx_max_value = \
    (self.sigma_quant_level - 1) * (2 ** unscaled_sigma_precision) - 1
```

| constant | value | source |
|---|---|---|
| `sigma_quant_level` | **32** | `CCS_SGMM/params.py` |
| `sigma_quant_min` | **0.11** | `CCS_SGMM/params.py` |
| `sigma_quant_max` | **54.82** | `CCS_SGMM/params.py` |
| `sigma_bound_offset` | **0.5** | `CCS_SGMM/params.py` |
| `sigma_precision` | **7** | `coding_tools/quantization/params.py` |
| `gain_vector_precision` | **5** | `quantization/params.py` |
| `beta_displacement_precision` | **5** | `quantization/params.py` |
| `scaler_precision` | **10** | derived = 5 + 5, `quantization.py:141` |
| `scaled_sigma_precision` | **17** | = 10 + 7, `quantization.py:131` |

That last row is the useful cross-check: `ls_processing/lsbs/lsbs_scale_mode.py:54`
independently **hardcodes** `self.scaled_sigma_precision = 17`. Since
`scaled = scaler + sigma = (5 + 5) + 7 = 17`, the whole precision chain closes on
itself. Three constants confirm each other, so `sigma_precision = 7` is certain rather
than merely read once. (We had guessed 8.)

**The `[…, 3968]` table extent is now explained.** `sigma_idx_max_value =
(32 − 1) × 2⁷ − 1 = 3967`, so the RVS/LSBS tables are indexed over `[0, 3967]` —
exactly 3968 entries. Open question 2 was "how do thousands of `Iσ` values map to 32
CDF rows"; the answer is that `Iσ` carries 7 fractional bits, and the σ-class is those
bits removed. Our guessed `isigma_table_size: 3968` was right, and the loader now
derives it from the other two constants rather than hardcoding it.

**Correction (Phase 5).** An earlier version of this section said the σ-class is
`Iσ >> 7`. It is not — it is `ceil(Iσ / 2⁷)`. `3967 >> 7 = 30`, which would leave CDF
row 31 unreachable for every possible `Iσ`, in a design whose stated purpose is a
small table. Under round-up `ceil(3967 / 128) = 31 = sigma_quant_level − 1`, so the
maximum index lands exactly on the last row. The two rules differ on every index that
is not an exact multiple of 128 — i.e. on 3937 of 3968 — so this was not a cosmetic
slip. See §3.2, and `jpegai/models/hyper.py`, which implements the corrected rule.

### 3.1a The largest representable σ is 54.734, not 54.82

A small consequence of `max_index = 3967` rather than 3968, and it corroborates the
round-up rule from a second direction. `σ(Iσ) = 0.11 · exp(log_k · Iσ / 2⁷)` at
`Iσ = 3967` gives **54.734**; reaching 54.82 would need `Iσ = 3968`, which the range
excludes. So the top of the σ grid is never *denoted* by an index — it is only ever
selected as a CDF **row**, which is precisely what round-up does and round-down cannot.
At the very top of the range the coder therefore uses a distribution marginally wider
than the one predicted, which is the safe direction and the only one available.

### 3.2 Why `Iσ` must stay an integer — a stronger reason than the paper gives

The paper motivates the integer `Iσ` on storage and on the RVS/LSBS tables that index
by it. There is a harder reason, found by measurement in Phase 5.

`SigmaIndex.table_row(Iσ)` and `GaussianConditional.build_indexes(σ(Iσ))` implement the
same rule by two routes. Over all 3968 valid indices they agree on 3957 and **disagree
on 11**:

```
256, 1152, 1280, 1536, 1664, 2176, 2304, 2560, 3200, 3328, 3456
```

Every one is an exact multiple of 128 — an `Iσ` sitting precisely on a grid point,
which is the only place a single bit can decide the comparison. The cause is one
float32 ULP between two different computations of the same exponential:

| | value |
|---|---|
| `σ(256)`, torch `min · exp(log_k · 2)` | 0.164220**72052955627** |
| `scale_table[2]`, numpy `exp(linspace(…))` → float32 | 0.164220**70562839508** |

`build_indexes` counts table entries strictly below σ, so that last bit pushes it to
row 3 while the exact integer arithmetic says row 2. Neither row is *unsafe* — row 3 is
merely wider — but they are **different**, and a bitstream whose encoder used one rule
and whose decoder used the other decodes to the wrong latent for 0.28% of its symbols,
with both sides reporting success.

So the argument for the integer index is not that it saves space. It is that
`table_row` on an integer is exact on every device and in every build, while
`build_indexes` on a reconstructed float σ is at the mercy of how each side happened to
compute an exponential. **Phase 11's cross-device bit-exactness requirement is
unachievable through the float path.**

Two rules follow, both now enforced in code rather than remembered:

1. `GaussianConditional.compress`/`decompress` take an `indexes=` argument, and the
   split-hyper codec always passes `table_row(Iσ)`. It never calls `build_indexes`.
2. The mid-training round-trip gate asks the model for its rows
   (`TwoBranchCodec.coder_rows`) instead of deriving them from σ. Measuring
   out-of-range and `est_q` against the wrong row would put a small permanent bias in
   `gap_q_pct`, the one number whose job is to read zero when the coder is correct.

`tests/test_hyper.py` pins the agreement, the exact 11 exceptions, and the corruption
itself — encode with the integer row, decode with the float row, assert the latent
comes back wrong — so a change to either path fails a test instead of shipping a
broken bitstream.

**Still open: `sigma_bound_offset = 0.5`.** Confirmed as a constant, still unused in
our code, and it has two plausible readings we cannot yet separate: a rounding offset
applied before the shift (which would make the rule round-to-nearest rather than
round-up, and would then *contradict* the reachability argument above), or a widening
of the CDF tail bound. The reachability argument and Phase 3's escape measurement both
point at round-up, so that is what is implemented; if the supplementary material shows
otherwise, §3.2 is the section to revisit.

### 3.1 What 32 levels costs — measured

The training loss evaluates the rate at the **continuous** σ that `h_s` predicts.
The coder can only index one of 32 rows. The difference is real rate that the loss
curve never shows, and it is large enough to matter for every number this project
reports, so it was measured rather than estimated.

Excess is `KL(p_σ ‖ p_σq)` in bits/symbol: the true distribution is a discretised
Gaussian at the predicted σ, the coding distribution is the one at the grid entry
`build_indexes` selects. Averaged over 4000 σ drawn log-uniform on
`[0.11, 54.82]`:

| levels | log step | excess (bits/symbol) | as % of that test's 3.44 b/sym | ratio to next |
|---|---|---|---|---|
| 8 | 0.8873 | 0.21345 | 6.20% | — |
| 16 | 0.4141 | 0.05493 | 1.60% | 3.89× |
| **32** | **0.2004** | **0.01464** | **0.43%** | 3.75× |
| 64 | 0.0986 | 0.00365 | 0.11% | 4.01× |
| 128 | 0.0489 | 0.00093 | 0.03% | 3.92× |
| 256 | 0.0244 | 0.00023 | 0.01% | 4.04× |

The right-hand column is the point: every halving of the step **quarters** the
cost. The error is second-order in the grid step, exactly as a Taylor expansion of
the rate around σ predicts. So 32 levels is not an arbitrary choice on a flat
curve — it sits where halving the table again would still buy ~0.011 bits/symbol,
and one more halving after that only ~0.003.

**End-to-end confirmation.** The full codec on Kodak measures **+1.86% to +1.92%**
over the loss's own estimate. At its ~1.07 bits/symbol that is +0.0199
bits/symbol, against +0.01464 predicted here — the same effect at the same
magnitude, differing because the real σ distribution is not log-uniform.

Note the two percentages (0.43% and 1.9%) are not in conflict. A percentage of rate
depends entirely on the operating point; the synthetic test's mean rate is 3.44
b/sym because log-uniform σ is dominated by wide distributions. **Bits per symbol
is the transferable number; the percentage is not.**

Three things follow, all of them load-bearing:

1. **It costs rate, never escapes.** `build_indexes` rounds σ *up*, so the coder's
   distribution is always at least as wide as the model predicted and always at
   least as heavy-tailed. Nothing the model thought likely can land outside the
   table. Rounding down would trade this 1.9% for out-of-range symbols, which is a
   far worse deal — a deliberately miscalibrated run at 0.63% escapes measured
   −17% rate, i.e. a bitstream that looks 17% *better* than the model while being
   broken.
2. **Reported bitrates must come from actual bytes.** The loss's estimate is
   optimistic by exactly this amount, so an RD curve built from it would claim a
   codec ~1.9% better than the one that exists.
3. **The Phase 3 gate thresholds on the σ-quantised estimate**, not the
   continuous one. Against the quantised σ the coder agrees to −0.11%, which is
   what actually tests the CDF construction. The plan's original "estimated within
   1–2% of actual" would have flagged a correct codec as broken.

Reproduce with `python -m jpegai.models.selftest --checkpoint <ckpt>`, which
reports both gaps separately.

---

## 4. Hyper-latent (z) entropy coding

| constant | value | source |
|---|---|---|
| `z_offset` | **31** | `CCS_SGMM/params.py` |
| `z_range` | **63** | `CCS_SGMM/params.py` — symbols span [−31, 31] |
| factorized `max_symbol` | **62** | `= z_range − 1`, `common_modules.py:116` |
| `abs_in_hyperprior` | **1** | `CCS_SGMM/params.py` — hyper encoder consumes `abs(y)` |
| BDL clipping range | **[−1069, 702]** | `quantization/params.py` |
| `mcm_overlap_in_latent_samples` | **8** | `CCS_SGMM/params.py` |
| `hyper_decoder_overlap_in_latent_samples` | **2** | `CCS_SGMM/params.py` |

The hyper CDF has **one row per channel of its own branch** (`FactorizedProbModel(chs_ls,
...)`), so there is no fixed row count — 160 rows for luma, 96 for chroma. The `[128, 64]`
shape I built an argument on in §7 item 1 corresponds to neither; it matches the unused
`chs_ls=128` default, and `64 ≈ z_range` rounded up to a power of two.

---

## 5. MCM stage order — resolves open question 5, and confirms our guess

Two independent confirmations that the stage order is
**`(0,0) → (1,1) → (0,1) → (1,0)`** (diagonal first), which is exactly what
`mcm_group_order` already said.

**First**, the shuffle that produces the four stages:

```python
# components/contexts/utils.py  ContextUtils.down_shuffle
y = y.reshape(B, iC, oH, factor_hw, oW, factor_hw)
y = y.permute(0, 1, 2, 4, 3, 5)
y = y.reshape(B, iC, oH, oW, factor_hw * factor_hw)
part1, part2, part3, part4 = torch.chunk(y, chunks=4, dim=4)
return part1.squeeze(4), part4.squeeze(4), part2.squeeze(4), part3.squeeze(4)
```

After that permute the raster order is `part1=(0,0)`, `part2=(0,1)`, `part3=(1,0)`,
`part4=(1,1)`. The return order is `(part1, part4, part2, part3)` — diagonal first.
`up_shuffle` unpacks the mirror image, `part0, part3, part1, part2 = input_rec`.

**Second**, and more convincingly because it is derived from a different concern —
`components/contexts/context.py`'s odd-size guards:

```python
if h_ls % 2 == 1 and stage_id in [1, 3]:   # drop the redundant last ROW
if w_ls % 2 == 1 and stage_id in [1, 2]:   # drop the redundant last COL
```

A stage needs the row guard iff its `dy == 1`, and the column guard iff its `dx == 1`.
Row guard on `{1,3}` and column guard on `{1,2}` forces stage 1 = (1,1), stage 2 = (0,1),
stage 3 = (1,0), and therefore stage 0 = (0,0). Same answer, from unrelated code.

Phase network structure, from `contexts/MCM_phases.py`: phase 0 uses
`fusion_pred_net(pred_explicit)` alone; phases 1/2/3 apply `conv1x1(k*chs, chs)` for
k = 1/2/3 then `conv3x3(chs, chs, groups=chs2group(chs))`, concatenating every
previously reconstructed residual. `HyperToContext9x1b.forward` chunks the hyper
decoder's output into four `chs`-wide slices, one per stage — which is *why*
`p̈` is `4 × chs` wide.

### 5.1 Which channels belong to which stage is *our* question, not the reference's

The line above says the reference **chunks** its `4 × chs` prediction contiguously, and
for the reference that is right: it never uses `PixelShuffle`: it moves between the
`/16` grid and the four `/32` cosets with the explicit `down_shuffle` / `up_shuffle`
above, so the last convolution's channel `k*chs + c` is stage `k`'s channel `c` by
construction.

Our Phase 6 does not have that freedom, and the difference cost real debugging. Phase 5's
`HyperDecoder` ends `conv3x3(chs, 4*chs) → PixelShuffle(2)`; Phase 6 reuses the very same
convolution with `shuffle=False` so that a Phase 5 checkpoint warm-starts as the
identity. That convolution was therefore *trained* under `PixelShuffle`'s mapping —
input channel `4c + 2i + j` → output channel `c` at sub-position `(i, j)` — so for us
coset `(i, j)` is the **strided** slice `pred[:, 2i+j :: 4]`, and the contiguous chunk is
a different permutation of the same numbers.

Both layouts are internally consistent; neither is normative. What is not free is mixing
them, and the mixture is invisible: shapes match, the codec round-trips bit-exactly, the
model trains fine, and only the warm start silently degrades. `jpegai/models/mcm.py`
isolates the choice in `split_pred` so it has one place to be right, and
[00-START-HERE.md](00-START-HERE.md) records the measurement that caught it.

The lesson for the rest of the phases: a constant lifted from the reference is only
transferable together with the surrounding convention it was written under. This one was
`chunk` — correct there, wrong here.

---

## 6. me-tANS — resolves open question 7

From `entropy_coding/lib_wrappers/mans/utils.py` (and the identical
`cpp_exts/mans/utils.py`, which additionally defines `normalize_z`):

| constant | value | note |
|---|---|---|
| `mass_bits` | **8** | **not** a `tableLog` — see below |
| escape threshold | **2⁻¹¹** | `get_outbound_values(probs, threshold=1/2**11)` |
| symbol ordering | **zig-zag** | `get_sequence` / `get_inverse_sequence` |
| substreams | **2, interleaved** | via the `cdf_first` / `cdf_second` state split |

Open question 7 asked for `tableLog` and the spread function. The premise was slightly
wrong: the reference parameterises by **probability mass bits**, not table log. With
`mass_bits = 8` the state space is 2⁸ = 256 per σ-class. The two-substream split and
the packed transition words:

```python
cdf_first, cdf_second = cdfs - (cdfs >> 1), (cdfs >> 1) + 128
...
return ((num_bits << 24) | (state_next << 16) | (symbols & 65535)).astype(np.uint32)
```

Splitting the CDF into halves and offsetting the second by 128 is what lets the decoder
be **OR + addition** instead of a comparison chain — the paper's claim that decode needs
no multiply or divide.

This also lets us check the paper's "~100 KB of tables" arithmetically:
32 σ-classes × 256 states × 4 bytes = **32 KiB** for the decode transitions alone, with
encode transitions (`(delta_bits << 16) | adders`) and the CDF/PMF matrices making up
the rest. ~100 KB total is the right order of magnitude — the claim holds. Our guessed
`tans_table_log: 11` would have produced 2048 states per class and an 8× larger table;
renamed to `tans_mass_bits: 8`.

---

## 7. Skip mode — resolves open question 4

| constant | value | source |
|---|---|---|
| `skip_block_size` | **1** | `coding_tools/skip_ls/params.py` |
| `thr_skip` | **382** | `skip_ls/params.py` |
| `skip_judge_thr` | **3** | `skip_ls/params.py` |
| `skip_cube_thr` | **1** default, **3** under CTC | `skip_ls/params.py`; `cfg/oper_point/common.json` |

We had `skip_threshold: 0` with a comment saying "calibrate in Phase 9". No calibration
needed — it is 382. The cube override is a max-pool test:
`cubeflag = (maxpool(diff_yhat) > self.skip_cube_thr)`, `contexts/context.py`. We use
the CTC value 3 in both configs, since CTC is what the paper's tables were produced
under.

---

## 8. Rate ladder — the paper's four base models sit on an 18-entry ladder

```python
# coding_tools/quantization/gain_unit/params.py
beta_list = [0.0002, 0.0005, 0.001, 0.002, 0.004, 0.007, 0.01, 0.015, 0.03,
             0.05, 0.075, 0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 3.0]
```

Eighteen operating points; `CCS_SGMM/params.py` gives `base_model_beta = 0.002`.
Three of the paper's four base-model β values (0.002, 0.075, 0.5) are ladder entries.
**The fourth, β = 0.012, is not** — the ladder brackets it with 0.01 and 0.015. Either
the paper rounds, or the trained base models are not required to coincide with ladder
entries. Not load-bearing for us: the gain unit interpolates, so what matters is the
ladder, and we train one model and sweep β anyway.

Other quantisation bit-widths, all `quantization/params.py`:
`beta_displacement_log_bitdepth = 12`, `gain_vector_bitdepth = 6`,
`gain_vector_log_bitdepth = 12`.

---

## 9. Colour transform — resolves open question 9, by dissolving it

Open question 9 asked for "the exact eq. (4) coefficients", on the theory that the
printed equation had two typos. **There are no normative coefficients.**
`colour_processing/colour_transformation/colour_transformation.py` reads a
`clr_tr_matrix` of **nine 8-bit integers from the picture header**, defaulting to the
identity, and inverts it numerically at decode time:

```python
inv_matrix = torch.inverse(clr_tr_matrix / 255.0) * 255.0
```

So eq. (4) is one *instance* of a signalled matrix, not a constant of the standard.
`colour_transform_idx` selects the output convention — `0` = RGB out, `1` = YUV out,
`2` = custom, `None` = auto (`colour_transformation/params.py`). Our decision to ship
the textbook BT.709 inverse is therefore a legal configuration choice rather than a
deviation, which is a better position than we thought we were in.

---

## 10. Codestream layout — resolves open question 10

`bitstream_structure/layouts_def.py` confirms all ten marker codes as previously
transcribed. Two details worth recording:

* TOH is named **`MARKER_TON`** in the software (`"tool_header"`).
* **Mandatory** substreams are PIH, SOZ, SORP, SORS. That confirms the minimal
  conformant stream we derived: `SOC · PIH · SOZ · SORp · SORs · EOC`.
* `use_ae` applies to SOQ/SOZ/SORP/SORS; `has_regions` only to SORP/SORS.
* `MarkersPrimary["z"] == MarkersSecondary["z"] == 3` — so **`ẑ_Y` and `ẑ_UV` share a
  single threaded SOZ substream** rather than being separately delimited.

`region_residual_in_its_own_substream_flag` (`CCS_SGMM/params.py`) selects between
offset-based dependent regions (0) and marker-based independent regions (1).

`synthesis_transform_id` is a **cumulative capability list**, not a single selector:
SOP signals `[0]`, BOP `[1, 0]`, HOP `[2, 1, 0]` (`cfg/profiles/*.json`). A HOP stream
is therefore decodable by an SOP decoder — this is the mechanism behind the
multi-branch scalability claim, and it is not obvious from the paper. Levels gate the
available models: `{2}` / `{2,3}` / `{0,1,2,3}` with `max_pic_size` from 6 220 800 to
398 131 200 (`cfg/oper_point/levels.json`).

---

## 11. The seven metrics — conventions that are not guessable

From `ref/jpeg-ai-qaf/metrics.py`. `MetricParent(bits=10, max_val=1023)`: **all metrics
are computed at 10-bit internal precision**, not 8.

| metric | plane | input range | backend in QAF |
|---|---|---|---|
| MS-SSIM | **Y** | 0…1023 | `pytorch_msssim`, `data_range=max_val` |
| VIF | **Y** | 0…1 | `IQA_pytorch.VIFs(channels=1)` |
| FSIM | **RGB** | 0…1 | `IQA_pytorch.FSIM(channels=3)` |
| NLPD | **Y** | 0…1 | `IQA_pytorch.NLPD(channels=1)` |
| IW-SSIM | **Y** | **0…255** | own `IW_SSIM_PyTorch` |
| PSNR-HVS | **Y** | 0…1, replicate-padded to a multiple of 8, float64 | `psnr_hvsm.psnr_hvs_hvsm` |
| VMAF | Y | — | Netflix binary v2.2.1 |

**Six of the seven are luma-only; only FSIM sees colour.** Our `metrics.py` originally
ran every metric on RGB. That is a silent correctness bug of the worst kind — it
produces plausible numbers that cannot be compared to the paper's at all. Fixed;
`jpegai/eval/metrics.py` now selects the plane and range per metric.

**The seventh metric is PSNR-HVS, not PSNR-HVS-M.** QAF calls
`psnr_hvs_hvsm(...)`, which returns both, and keeps the **first**. `PAPER_SEVEN` in
`jpegai/eval/metrics.py` uses `psnr_hvs`; `psnr_hvsm` is still computed and reported,
but is not one of the seven and is not averaged into AVG.

Backend substitutions we make, all recorded in `metrics.py::BACKEND_NOTES`: `piq` for
VIF/FSIM/IW-SSIM and `pyiqa` for NLPD, in place of `IQA_pytorch` and QAF's own IW-SSIM.
pyiqa's NLPD rejects 1-channel input, so we feed Y replicated across three channels —
exact, because any weighted-sum luma conversion of a grey image returns that grey
unchanged (the coefficients sum to 1). Installing the exact backends
(`pip install psnr-hvsm IQA-pytorch pyrtools`) would remove the last substitutions;
until then the numbers are internally consistent but only approximately comparable to
published values.

---

## 12. Still genuinely ours

Everything above is normative. These remain our own choices, and should be labelled as
such in the report:

1. **Tier A channel widths** (96/48 rather than 160/96) — a deliberate reduction so the
   Mac can validate. `full.yaml` carries the normative widths.
2. **Per-stage trunk widths** `analysis_width` / `synthesis_width` — the paper gives
   totals and kMAC/pxl, not per-stage widths. Ours until the supplement arrives.
3. **T1/T2/TP/TR tables** for RVS/LSBS — not in the reference software as literals
   (they are learned weights in the LFS-tracked checkpoints). We learn them with the
   rest of the codec frozen. Documented deviation.
4. **The MOP decoder (id 3)** — not in the standard at all. Our own head, answering a
   question the paper leaves open: what sits between BOP's 28 and HOP's 215 kMAC/pxl.
   Disabled by default, enabled for the Phase 13 ablation.
5. **`isigma_pad_value: 1411`** — this one is PAPER, stated in eq. (7), but we have not
   found it in the reference software, so it is unconfirmed.
6. **Training recipe entirely** — loss weights, LR schedule, iteration budget,
   `distortion_weights: {y: 6, u: 1, v: 1}`. The paper says only "prioritise luma".

---

## 13. How to re-verify any of this

Every claim above is a `grep` away. The reference software is in `ref/`, checked out
with `GIT_LFS_SKIP_SMUDGE=1` so the multi-GB weights are pointer files. The constants
live in `params.py` files next to the tool that uses them:

```bash
find ref/jpeg-ai-reference-software/src/codec -name params.py
```

Two habits that mattered while doing this:

* **Find the construction site, not the class attribute.** Correction 2 above cost real
  time. `IN_CHS: int = 48` is a default; `chs_ls=N_chroma` is the value.
* **Prefer constants that confirm each other.** `sigma_precision = 7` is trustworthy
  not because one file says 7, but because a *different* file hardcodes 17 and
  5 + 5 + 7 = 17.
