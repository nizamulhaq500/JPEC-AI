# 04 — Reference Data

Everything extracted from the paper in one place: the six tables, the marker codes, all
tensor shapes, every equation, the normative table dimensions, and implementation
checklists. This is the lookup document — keep it open while coding.

---

## 1. Tensor shape cheat sheet

`H`, `W` = original size. `Ḣ`, `Ẇ` = padded size. `c_v`, `c_h` = chroma subsampling factors.

| Tensor | Shape | Where |
|---|---|---|
| `x_Y` | `[1, H, W]` | input luma |
| `x_UV` | `[2, ⌈H/c_v⌉, ⌈W/c_h⌉]` | input chroma |
| `y_Y` | `[160, Ḣ/2⁴, Ẇ/2⁴]` | primary latent |
| `y_UV` | `[96, Ḣ/2⁴, Ẇ/2⁴]` | secondary latent |
| `ẑ_Y` | `[160?, Ḣ/2⁶, Ẇ/2⁶]` | primary hyper latent — see §7 caveat |
| `ẑ_UV` | `[·, Ḣ/2⁶, Ẇ/2⁶]` | secondary hyper latent |
| `p̈_Y` | `[640, Ḣ/2⁵, Ẇ/2⁵]` → pixel-shuffle ×2 → `[160, Ḣ/2⁴, Ẇ/2⁴]` | primary prediction |
| `p̈_UV` | `[·, Ḣ/2⁴, Ẇ/2⁴]` (96 ch after shuffle) | secondary prediction |
| `Iσ_Y` | `[160, Ḣ/2⁴, Ẇ/2⁴]` integer, log domain | primary variance index |
| `Iσ_UV` | `[96, Ḣ/2⁴, Ẇ/2⁴]` integer, log domain | secondary variance index |
| `r̂_Y` | `[160, Ḣ/2⁴, Ẇ/2⁴]` integer | primary quantised residual |
| `r̂_UV` | `[96, Ḣ/2⁴, Ẇ/2⁴]` integer | secondary quantised residual |
| `ŷ_Y` | `[160, Ḣ/2⁴, Ẇ/2⁴]` | reconstructed primary latent |
| `ŷ_UV` | `[96, Ḣ/2⁴, Ẇ/2⁴]` | reconstructed secondary latent |
| `ŷᶜ_UV` | `[256, Ḣ/2⁴, Ẇ/2⁴]` = concat(`ŷ_UV` 96, `ŷ_Y` 160) | secondary synthesis input |
| `σ_Y` | `[160, Ḣ/2⁷, Ẇ/2⁷]` integer | RVS 8×8-pooled variance |
| `σ_UV` | `[96, Ḣ/2⁷, Ẇ/2⁷]` integer | RVS 8×8-pooled variance |

**Stride bookkeeping:** analysis = 4 stride-2 stages → /16. Hyper encoder = 2 more → /64.
Total 2⁶ = 64, which is why the display-window mechanism pads to a multiple of **64**. RVS
pools `Iσ` (at /16) by 8×8 → /128 = 2⁷. All consistent.

---

## 2. Equations

### Residual coding
```
(1)  r̂_UV[c,i,j] = round( y_UV[c,i,j] − p̈_UV[c,i,j] )          0 ≤ c < 96
(2)  ŷ_UV[c,i,j] = r̂_UV[c,i,j] + p̈_UV[c,i,j]                   0 ≤ c < 96
(3)  ŷᶜ_UV[c,i,j] = ŷ_UV[c,i,j]   for   0 ≤ c <  96
                  = ŷ_Y [c,i,j]   for  96 ≤ c < 256
```
The primary component uses MCM instead of (1)/(2) — same subtraction, but the prediction is
refined over 4 stages.

### Colour space conversion (§VI-B)
```
(4)  colour_transform_idx == 0   (YCbCr → RGB)
     x̂[0] = x̂_Y[0] + 1.5748·(x̂_UV[1] − 0.5)
     x̂[1] = x̂_Y[0] + 1.8556·(x̂_UV[1] − 0.5)      ← printed as [1]; should be [0] (Cb)
     x̂[2] = ( x̂_Y[0] − 0.2126·x̂[0] − 0.07222·x̂[1] ) / 0.7152   ← 0.07222 should be 0.0722

(5)  colour_transform_idx == 2   (encoder-defined)
     x̂[c] = Σ_k a[c,k]·(x̂_Y/x̂_UV components) + b[c]     a[3,3], b[3] in the PIH

(6)  x̂[c,i,j] = clip( 0, 2^bitdepth − 1, x̂[c,i,j] · (2^bitdepth − 1) )
```
`colour_transform_idx == 1` → no conversion. Conversion happens **after** post-filters.
Standard BT.709 inverse (use this): `R = Y + 1.5748·Cr`, `B = Y + 1.8556·Cb`,
`G = (Y − 0.2126·R − 0.0722·B)/0.7152`.

### RVS — Residual and Variance Scaling (§VI-G)
```
(7)  σ_Y[c,i,j] = ( 32 + Σ_{i'=0}^{7} Σ_{j'=0}^{7} Iσ_Y[c, 8i+i', 8j+j'] ) >> 6
                                                 boundary padding value = 1411
(8)  Iσ_Y[c,i,j] += T1[ modelID, id[c], σ_Y[c, ⌊i/8⌋, ⌊j/8⌋] ]
(9)  r̂_Y[c,i,j]  = r̂_Y[c,i,j] · T2[ modelID, id[c], σ_Y[c, ⌊i/8⌋, ⌊j/8⌋] ] / 2¹⁶

     id[c] = GRFS_Y[c] + 2 · rvs_enable_flag[0]          GRFS_Y[c] ∈ {0,1}
```
`T1[4, 4, 3968]`, `T2[4, 4, 3968]` — normative, and Part 1 also specifies how to compute
entries on the fly. Same tables for the secondary component.

### LSBS — Latent Scaling Before Synthesis (§VI-H)
```
     μ_Y[c,i,j] = ŷ_Y[c,i,j] − r̂_Y[c,i,j]
(10) ŷ_Y[c,i,j] += ( r̂_Y[c,i,j]·TR[ modelID, σ_Y[c,⌊i/8⌋,⌊j/8⌋] ]
                   + μ_Y[c,i,j]·TP[ modelID, σ_Y[c,⌊i/8⌋,⌊j/8⌋] ]
                   + 2¹² ) >> 13
```
`TP[4, 3968]`, `TR[4, 3968]` — normative, computable on the fly. Gated by
`lsbs_enable_flag[comp]` in the **tools header**.

### Rate adaptation / gain units (§VI-I)
```
(11) mlog[comp,c,i,j] = betaDisplacementLog[comp] + mref[modelID, comp, c]
(12) if gain_3D_enable_flag:  mlog[comp,c,i,j] += Gain3d[i,j]
(13) m⁻¹[comp,c,i,j]  = exp( −mlog[comp,c,i,j] · step / 2^sigmaPrecision )
(14) r̂_Y  ·= m⁻¹[0,·]      r̂_UV ·= m⁻¹[1,·]
(15) Iσ_Y += mlog[0,·]     Iσ_UV += mlog[1,·]
```
- `step`, `sigmaPrecision`, `mref` are predefined **12-bit signed integers**.
- `exp()` may be a LUT.
- Encoder scales unquantised `r_Y`, `r_UV` by `1/m⁻¹` before rounding.
- Higher `betaDisplacementLog` → finer quantisation → higher rate and quality.

---

## 3. Codestream markers (Table II)

| Symbol | Code | Payload | M/O |
|---|---|---|---|
| SOC | `0xff80` | — (start of codestream) | M |
| EOC | `0xff81` | — (end of codestream) | M |
| PIH | `0xff82` | picture header | M |
| TOH | `0xff83` | tools header | O |
| RDI | `0xff84` | rendering information | O |
| SOZ | `0xff88` | `z_Y-stream` and `z_UV-stream` | M |
| SORp | `0xff89` | `r_Y-stream` | M |
| SORs | `0xff8a` | `r_UV-stream` | M |
| SOQ | `0xff8b` | quality map information | O |
| UDI | `0xff8c` | user defined information | O |

Segment layout: `marker (2 bytes) | size (variable) | payload`, with **byte alignment after
the size field and after the payload**. Multiple `SORp`/`SORs` allowed with distinct
`region_idx` (`[0]` primary, `[1]` secondary). Minimal legal codestream:
`SOC · PIH · SOZ · SORp · SORs · EOC`.

### Syntax elements named in the paper

**PIH:** profile ID · level ID · picture size · output bit depth · internal subsampling
(`c_ver_minus1`, `c_hor_minus1`) · output subsampling (`s_ver_minus1`, `s_hor_minus1`) ·
model indices (`modelID`) · `decoderID` · `colour_transform_idx` (+ `a[3,3]`, `b[3]` when
= 2) · region/tile partitioning params (`region_partitioning_flag`,
`region_residual_in_its_own_substream_flag`, `synthesis_tile_enable[comp]`) ·
multithreaded-entropy-decoding substream counts · `grfs_enable_flag[comp]`,
`rvs_enable_flag[comp]`, `GRFS_Y[c]` · `betaDisplacementLog[comp]` ·
`gain_3D_enable_flag` · skip-mode params · `diff_display_img_width`,
`diff_display_img_height`.

**TOH:** `lsbs_enable_flag[comp]` (comp = 0,1) · filters header = 4 enable flags
(EFE linear, ICCI, EFE nonlinear, LEF) + per-filter control data.
**Absent TOH ⇒ all these flags inferred 0.**

**SOQ:** 2-D quality map → `Gain3d[i,j]`.

**RDI:** colour primaries · transfer characteristics · matrix coefficients · full-range
flag · chroma sample location · mastering display colour volume (primaries, white point,
luminance range) · nominal target brightness upper bounds · HDR dynamic metadata.

**UDI:** arbitrary application-defined bytes.

---

## 4. Normative table dimensions (all in Part 1)

| Table | Dimensions | Indexed by | Purpose |
|---|---|---|---|
| Hyper CDF | `[128, 64]` | channel index `c` | entropy coding of `ẑ` |
| Residual CDF | `[32, 256]` | `Iσ[c,i,j]` | entropy coding of `r̂` |
| `transition_table_symbol` | per σ-class / channel | tANS state | me-tANS decode |
| `transition_table_nBits` | per σ-class / channel | tANS state | me-tANS decode |
| `transition_table_stateNext` | per σ-class / channel | tANS state | me-tANS decode |
| `bound_table` | per σ-class / channel | — | escape/outbound threshold |
| `T1` | `[4, 4, 3968]` | modelID, `id[c]`, σ | RVS variance offset |
| `T2` | `[4, 4, 3968]` | modelID, `id[c]`, σ | RVS residual scale (`/2¹⁶`) |
| `TP` | `[4, 3968]` | modelID, σ | LSBS prediction weight (`>>13`) |
| `TR` | `[4, 3968]` | modelID, σ | LSBS residual weight (`>>13`) |
| `mref` | `[modelID, comp, c]` | — | channel-wise gain, 12-bit signed |

Total me-tANS decoder table memory: **≈ 100 KB**.

---

## 5. me-tANS decoding (Algorithm 1)

```
Init:  flatten Iσ[] to 1-D
       point at the LAST symbol position (pointer moves BACKWARDS — FILO)
       s ← parse 8 bits

for i = 0 .. n_symbols/4:
    # fast path — 4 symbols
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

Table selection: by `Iσ[c,i,j]` for residuals, by channel index `c` for hyper samples.
Even single-threaded, two substreams and two ANS states are interleaved, "mimicking a
dual-threaded setup".

**Skip mode:** `Iσ[c,i,j] < threshold` ⇒ residual not coded, inferred 0. Up to **80%** of
residual samples may be skipped. Overridable per **16×16×16** cube (cube-based skip mode).

**Multithreading:** `z_Y`, `z_UV`, `r_Y`, `r_UV`, `q` streams each splittable into
independent substreams; offsets at segment start, counts in PIH.

---

## 6. Operating points and profiles

| decoderID | Name | Target device | Upsampling | Final-layer channels | kMAC/pxl | Allowed layers |
|---|---|---|---|---|---|---|
| 0 | SOP | laptops w/o NN accel, mid/low-end mobile | 2×2 conv + pixel shuffle | 32 | 14 | conv, ReLU, ReLU6 |
| 1 | BOP | high-end mobile (NPU/GPU) | 4×4 deconv; pixel shuffle only in final layer | 64 | 28 | conv, ReLU, ReLU6 |
| 2 | HOP | desktop GPU | richer | — | 215 | unrestricted |

| Decoder profile | Supported decoderIDs |
|---|---|
| Main@Simple | 0 |
| Main@Base | 0, 1 |
| Main@High | 0, 1, 2 |

One stream profile (**Main**) in draft Part 2; post-processing filters **not mandatory** in
it. Profile and level IDs are in the picture header.

Network diagrams: supplement Figs. 6 (dID 0), 7 (dID 1), 8 (dID 2); hyper decoder =
Appendix A, hyper scale decoder = Appendix B, MCM stages = Appendix C.

---

## 7. Open questions — RESOLVED from the WG1 reference software

These were the things the overview paper left ambiguous. We expected to need ITU-T
T.840-1 for them. In the event, the **reference software** answered nine of ten, since
it is the normative implementation and carries every constant as a literal.

Full extraction with file paths, line numbers and cross-checks:
**[06-normative-constants.md](06-normative-constants.md)**. Summary:

| # | question | answer | status |
|---|---|---|---|
| 1 | `ẑ_Y` channel count | **160** — the hyper AE is channel-preserving | my 128 reading was **wrong** |
| 2 | `Iσ` → σ-class mapping | log-spaced, 32 levels over [0.11, 54.82]; class = `Iσ >> 7` | resolved |
| 3 | `step` / `sigmaPrecision` | `sigma_precision = 7` (we guessed 8) | resolved, triple-confirmed |
| 4 | `skip_threshold` | **382** (`thr_skip`) | resolved |
| 5 | MCM group ordering | `(0,0) → (1,1) → (0,1) → (1,0)` | our guess was **right** |
| 6 | secondary analysis stage count | `downsample_factor=2` on the chroma branch | resolved |
| 7 | tANS `tableLog` / spread | `mass_bits = 8`, zig-zag ordering | premise was wrong — see §6 there |
| 8 | `p̈_UV` pre-shuffle channels | **384** = 4 × 96 | resolved |
| 9 | eq. (4) coefficients | there are none — the matrix is a signalled header field | dissolved |
| 10 | `ẑ_Y` / `ẑ_UV` delimiting | one shared threaded SOZ substream | resolved |

Two of my earlier readings were wrong, and both are worth stating plainly because both
would have produced a model that trains fine and is not JPEG AI:

* **`hyper_latent` is 160, not 128.** I reasoned from the `[128, 64]` CDF table shape;
  the real cause of that 128 is an unused fallback default. The paper was right.
* **`secondary_latent` is 96, not 48.** I read a class-attribute default instead of the
  construction site. The paper (and eq. 3's `256 = 96 + 160`) were right.

Both are fixed in `jpegai/config/*.yaml`, and `python -m jpegai.config` now asserts
`hyper_latent == primary_latent` and `primary_latent % 32 == 0` so neither can silently
return.

Still genuinely unresolved, and now the only reason we still want the supplement:

* **Per-stage trunk widths** for the analysis and synthesis transforms. The paper gives
  totals and kMAC/pxl; the reference software's per-decoder hidden widths are in
  §2 of [06](06-normative-constants.md), but the *analysis* transform's stage widths are
  not exposed as literals.
* **T1/T2/TP/TR tables** for RVS and LSBS — learned weights inside the LFS checkpoints,
  not source literals. We learn our own.
* **`isigma_pad_value = 1411`** — stated in the paper's eq. (7), not found in the
  reference software. Unconfirmed but harmless.


---

## 8. Results — Table III (main, BD-rate vs VVC Intra / VTM-11.1, JPEG AI test set)

Negative = JPEG AI better.

| encID | decID | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS | kMAC/pxl | ms/MPx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | **−16.2%** | −30.9% | +6.9% | −22.3% | −13.4% | −26.6% | −30.0% | +2.9% | 14 | 285 |
| 0 | 1 | **−20.2%** | −33.0% | +1.4% | −26.9% | −17.3% | −29.1% | −34.8% | −1.9% | 28 | 266 |
| 0 | 2 | **−22.1%** | −34.8% | −2.0% | −27.7% | −19.3% | −31.2% | −37.3% | −2.5% | 215 | 323 |
| 1 | 0 | **−14.4%** | −30.3% | +9.7% | −20.4% | −11.6% | −25.1% | −29.2% | +6.1% | 14 | 246 |
| 1 | 1 | **−19.9%** | −33.0% | +1.5% | −26.4% | −16.7% | −28.4% | −35.8% | −0.8% | 28 | 271 |
| 1 | 2 | **−27.0%** | −37.6% | −8.5% | −34.7% | −23.6% | −33.7% | −42.4% | −8.8% | 215 | 332 |

> **Prose/table discrepancy.** §VII-B quotes averages of 16.0 / 20.2 / 21.1 (encID 0) and
> 13.9 / 19.7 / 27 (encID 1), and 13/28/215 kMAC/pxl. The table says 16.2 / 20.2 / 22.1,
> 14.4 / 19.9 / 27.0, and 14/28/215. **Cite the table.**

Notes:
- encoderID 0 wins with decoders 0 and 1; **encoderID 1 wins decisively with decoder 2**
  (−27.0 vs −22.1). Encoder choice depends on the target decoder.
- VIF and PSNR-HVS are the weak metrics (positive at dID 0). MS-SSIM and VMAF carry the win.
- 15× the MACs (14→215) costs only 285→323 ms/MPx on a V100 — the GPU isn't compute-bound;
  sequential entropy decoding dominates. On mobile the ranking inverts.
- Hardware: NVIDIA Tesla V100 32 GB + Intel Xeon Platinum 8336C @ 2.30 GHz.

## 9. Results — Table IV (tool ablation, encID 0, decID 1)

| TEST | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS | kMAC/pxl | ms/MPx |
|---|---|---|---|---|---|---|---|---|---|---|
| all on | **−20.2%** | −33.0% | 1.4% | −26.9% | −17.3% | −29.1% | −34.9% | −1.9% | 27.7 | 266 |
| RVS off | −18.0% | −33.1% | 0.8% | −20.2% | −14.8% | −29.0% | −28.8% | −0.7% | 27.7 | 249 |
| LSBS off | −19.8% | −33.1% | 1.8% | −27.4% | −17.5% | −29.1% | −30.5% | −2.7% | 27.6 | 249 |
| LEF off | −19.9% | −33.2% | 1.2% | −27.3% | −18.1% | −29.0% | −29.0% | −3.7% | 27.7 | 251 |
| ICCI off | −20.0% | −32.8% | 1.7% | −26.9% | −17.2% | −29.0% | −34.1% | −2.0% | 23.1 | 245 |
| EFE nonlin off | −20.4% | −33.3% | 1.2% | −26.2% | −17.6% | −29.4% | −35.1% | −2.1% | 27.7 | 246 |
| EFE lin off | −20.4% | −33.3% | 1.1% | −26.1% | −17.6% | −29.4% | −35.2% | −2.2% | 28.6 | 250 |

**Tool contributions:** RVS **2.2%** · LSBS **0.4%** · LEF **0.3%** · ICCI **0.2%** ·
EFE nonlinear **−0.2%** (but **+8%** chroma PSNR) · EFE linear **−0.2%** (but **+12%**
chroma PSNR).

RVS pays where it was designed to: FSIM −20.2 → −26.9 (6.7 pts) and VMAF −28.8 → −34.9
(6.1 pts), while MS-SSIM barely moves. ICCI is the only tool with real compute cost
(23.1 → 27.7 kMAC/pxl ≈ 4.6, ~17% of BOP).

## 10. Results — Table V (Kodak) and Table VI (CLIC 2024 validation)

**Kodak (24 images, 768×512)**

| decID | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS |
|---|---|---|---|---|---|---|---|---|
| 0 | −7.5% | −29.8% | +18.1% | −19.7% | −0.2% | −24.1% | −22.3% | +25.3% |
| 1 | −12.9% | −32.1% | +11.4% | −22.9% | −6.3% | −26.8% | −28.4% | +14.5% |
| 2 | −21.1% | −37.3% | 0.0% | −28.8% | −15.6% | −32.0% | −38.3% | +4.4% |

**CLIC 2024 validation**

| decID | AVG | MS-SSIM | VIF | FSIM | NLPD | IW-SSIM | VMAF | PSNR-HVS |
|---|---|---|---|---|---|---|---|---|
| 0 | −12.1% | −25.7% | +22.6% | −30.8% | −7.6% | −25.0% | −25.4% | +7.3% |
| 1 | −16.8% | −28.4% | +15.4% | −34.6% | −12.2% | −27.9% | −32.0% | +1.9% |
| 2 | −24.9% | −34.5% | +2.8% | −42.3% | −19.9% | −33.5% | −40.7% | −6.3% |

Paper's summary: "roughly between 7% and 25% coding gain depending on the selected decoder."
Note the honest reading: Kodak at SOP collapses to −7.5% because PSNR-HVS is **+25.3%** and
VIF **+18.1%**. Kodak's small images suit learned codecs less well (less context, relatively
larger hyperprior overhead).

## 11. Datasets and evaluation protocol

| Dataset | Content | Used for |
|---|---|---|
| JPEG AI test set (CTTC) | 50 natural images, 1K–4K | Tables III, IV |
| JPEG AI synthetic set | 36 images: animation, screen, game | §VII-D subjective |
| Kodak | 24 images, 768×512 | Table V |
| CLIC 2024 validation | — | Table VI |

Metrics: MS-SSIM, VIF, FSIM, VMAF, NLPD, **PSNR-HVS**, IW-SSIM + BD-rate. (PSNR-HVS,
not PSNR-HVS-M — QAF keeps the first return value of `psnr_hvs_hvsm()`. All but FSIM
are luma-only, at 10-bit internal precision. See
[06-normative-constants.md §11](06-normative-constants.md).)
Complexity: kMAC/pxl and ms/megapixel. Anchor: VVC Intra, VTM-11.1, PNG→YUV with FFmpeg.
Test codec: JPEG AI **VM 7.0**. Metric list rationale: WG1 N85013 (Nov 2019).

Subjective comparison (Fig. 3): ~0.08 bpp and ~0.3 bpp, **post-filters disabled**.

---

## 12. The five parts (Table I)

| Part | Name | ITU-T \| ISO/IEC | Status per paper |
|---|---|---|---|
| 1 | Core coding systems | T.840-1 \| 6048-1 | finalised 2025 |
| 2 | Profiling | T.840-2 \| 6048-2 | draft: 1 stream profile, 3 decoder profiles |
| 3 | Reference software | T.840-3 \| 6048-3 | finalised 2025; `gitlab.com/wg1/jpeg-ai/jpeg-ai-vm` |
| 4 | Conformance | T.840-4 \| 6048-4 | target finalisation Oct 2025 |
| 5 | File format | T.840-5 \| 6048-5 | finalised 2025; ISOBMFF (Annex A, "Motion JPEG AI") + HEIF (Annex B) |

Four sets of trained model parameters distributed in **ONNX** format via a link in Part 1.

Standardisation milestones: project started 2019 · metric analysis N85013 (Nov 2019) ·
final CfP N100095 (Jan 2022) · 7 CfP responses (Jul 2022) · Bytedance [32] + Huawei [33]
selected · harmonised into VM 1.0 (Oct 2022) · CTTC N100421 (Jan 2023).

Device demos: 1024×1024 in **< 20 ms** on a smartphone [36], [37]; **4K in ≈ 190 ms** [38].

---

## 13. Implementation checklists

### Decoder — normative pipeline order
```
[ ] parse SOC
[ ] parse PIH  (profile, level, size, bitdepth, c_*/s_*, modelID, decoderID,
                colour_transform_idx, regions/tiles, substream counts, RVS flags,
                betaDisplacementLog, skip params, display window)
[ ] parse optional TOH (lsbs flags, filters header), SOQ, RDI, UDI
[ ] SOZ  → me-tANS decode ẑ_Y, ẑ_UV      (fixed CDF [128,64], by channel)
[ ] hyper scale decoder (INTEGER) → Iσ_Y, Iσ_UV
[ ] rate adaptation eqs 11–13, 15 → Iσ += mlog
[ ] RVS eq 7 (pool), eq 8 (Iσ += T1)
[ ] SORp/SORs → me-tANS decode r̂_Y, r̂_UV (CDF [32,256], by Iσ; skip mode; cubes)
[ ] rate adaptation eq 14  → r̂ *= m⁻¹
[ ] RVS eq 9               → r̂ *= T2/2¹⁶
[ ] ================ END BIT-EXACT CONFORMANCE POINT ================
[ ] hyper decoder → p̈_Y, p̈_UV (incl. pixel shuffle)
[ ] MCM 4 stages → ŷ_Y                    (primary only)
[ ] eq 2         → ŷ_UV                   (secondary)
[ ] LSBS eq 10 (if lsbs_enable_flag)
[ ] eq 3: concat → ŷᶜ_UV [256, ...]
[ ] primary synthesis transform (decoderID) → x̂_Y
[ ] secondary synthesis transform (decoderID) → x̂_UV
[ ] post-filters: EFE linear, ICCI, EFE nonlinear, LEF   (all optional to apply)
[ ] internal→output subsampling conversion (if c_* ≠ s_*)
[ ] colour space conversion eqs 4/5
[ ] scale + clip eq 6
[ ] display-window crop (diff_display_img_*)
[ ] parse EOC
```

### Feature coverage checklist (tick as you implement)
```
Architecture
[ ] YCbCr BT.709, 4:4:4 / 4:2:2 / 4:2:0, internal ≠ output formats
[ ] primary/secondary split; preprocessing cross-link; eq-3 concat cross-link
[ ] monochrome (luma-only) fast decode path
[ ] 4-stage analysis (/16), 2-stage hyper encoder (/64)
[ ] hyper decoder (p̈) and hyper scale decoder (Iσ) as SEPARATE networks
[ ] residual coding r̂ = round(y − p̈)
[ ] MCM 4-stage 2×2 checkerboard, luma only
[ ] three synthesis transforms (SOP/BOP/HOP) on one codestream
[ ] layer restriction (conv/ReLU/ReLU6) enforced for decoderID 0 and 1

Entropy coding
[ ] me-tANS with 4 transition tables + bound_table, FILO
[ ] escape/outbound coding (1-bit flag → 2 or 15 bits → sign)
[ ] dual-state interleaving even single-threaded
[ ] hyper CDF [128,64] by channel; residual CDF [32,256] by Iσ
[ ] skip mode + 16×16×16 cube-based override; report skip %
[ ] multithreaded substreams with signalled offsets

Tools
[ ] RVS  (eqs 7–9, T1/T2, GRFS_Y[c], pad value 1411)
[ ] LSBS (eq 10, TP/TR)
[ ] gain unit (mref) + betaDisplacementLog per component
[ ] 3D gain unit + quality map (SOQ) → RoI coding
[ ] EFE linear, EFE nonlinear, ICCI, LEF — each skippable at decode

Functionality
[ ] layer-based cropping (per-stage padding/cropping)
[ ] display window (pad to multiple of 64 + crop reconstruction)
[ ] synthesis transform tiling (bounded peak memory)
[ ] region partitioning (multiple SORp/SORs, region_idx) + random-access crop decode
[ ] progressive decoding (zero part of r̂; truncated codestream)
[ ] HDR / wide-gamut metadata (RDI)

Standards conformance
[ ] full codestream: SOC/PIH/TOH/SOQ/RDI/UDI/SOZ/SORp/SORs/EOC + byte alignment
[ ] TOH absent ⇒ flags inferred 0
[ ] integer hyper scale decoder (8-bit mult, 32-bit accum) + overflow bound table
[ ] bit-exact r̂/ẑ across CPU/MPS; tolerant x̂ (measure and report)
[ ] profile/level IDs written and checked
[ ] `jpegai inspect` syntax dumper

Evaluation
[ ] 7 metrics + PSNR-Y/U/V; BD-rate (cubic, log-rate, overlap only, per-metric then average)
[ ] anchors: JPEG, WebP, AVIF (+ JPEG XL, VTM-11.1 stretch)
[ ] learned baselines: bmshj2018, mbt2018, cheng2020
[ ] kMAC/pxl counter; ms/MPx with per-stage breakdown
[ ] datasets: Kodak, DIV2K-valid, CLIC-2024, small synthetic/screen set
[ ] ablations: RVS/LSBS/LEF/ICCI/EFE ×2, MCM stages, split vs fused hyper decoder,
    variable-rate vs fixed-λ, int vs float, two-branch vs RGB, skip on/off,
    cropping vs display window
[ ] subjective sheets at ~0.08 and ~0.3 bpp with post-filters disabled
```

---

## 14. Limitations of JPEG AI v1 (state these in your report)

1. Synthetic/screen content: basic support only, no dedicated tools or models; worse than
   traditional codecs on screen-captured text.
2. No bit-exact picture reconstruction (deliberate, for implementation flexibility).
3. No lossless coding.
4. Machine consumption of the latent deferred to v2, despite being in the original scope.

**Named v2 directions:** better synthetic content · bit-exact + lossless · implicit neural
representations / online training · diffusion models · transformer architectures ·
machine-vision tasks directly on `ŷ` (detection, recognition, segmentation,
super-resolution, denoising, colour correction) at lower complexity and "in some cases
higher accuracy, particularly at lower quality settings".

---

## 15. Source artifacts in this repo

```
paper/paper_text.txt          full text, 18 pages, page-delimited
paper/rasterize.py            from-scratch PDF vector-path rasterizer
paper/imgs/p03_1_Im0.png      Fig. 1 left  — primary/luma branch (enc + dec)
paper/imgs/p03_0_Im1.png      Fig. 1 right — secondary/chroma branch (enc + dec)
paper/imgs/table_p2_Fm0.png   Table I   — the five parts
paper/imgs/table_p5_Fm0.png   Table II  — markers + hex codes
paper/imgs/table_p8_Fm0.png   Fig. 2    — MCM 4-stage checkerboard grouping
paper/imgs/table_p9_Fm0.png   Table III — main results
paper/imgs/table_p12_Fm0.png  Table IV  — tool ablation
paper/imgs/table_p12_Fm1.png  Table V   — Kodak
paper/imgs/table_p13_Fm0.png  Table VI  — CLIC 2024
```
Tables I–VI and Fig. 2 are vector glyph outlines in Form XObjects, invisible to text
extraction — `rasterize.py` is how the numbers above were recovered.
