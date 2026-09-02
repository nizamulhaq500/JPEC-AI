<div class="cover" markdown="1">

<div class="cover-title">JPEG AI</div>

<div class="cover-sub">Building a Learning-Based Image Codec from a Research Paper</div>

<div class="cover-tag">A complete project report — from the first page of the paper to the last measured bitstream</div>

<div class="cover-meta" markdown="1">

**Author:** Nizam ul Haq

**Subject paper:** S. Esenlik, Y. Wu, Z. Zhang, Y.-K. Wang, K. Zhang, L. Zhang, J. Ascenso, S. Liu,
"An Overview of the JPEG AI Learning-Based Image Coding Standard," *IEEE Transactions on Circuits
and Systems for Video Technology*, vol. 36, no. 2, pp. 2520–2537, February 2026.
DOI 10.1109/TCSVT.2025.3613244

**Report date:** 29 August 2026

**Repository:** `github.com/nizamulhaq500/JPEC-AI`

</div>

<div class="cover-note" markdown="1">

This report is written for a reader who has never heard the words "JPEG AI". It begins from
what an image codec is and why anyone would want to build one out of a neural network, and
ends with the exact bit-rates our implementation produced on the Kodak test set. Nothing is
assumed except school-level algebra and a general sense of what a photograph is.

Every number in this report was measured on this machine, from real bitstreams, unless it is
explicitly attributed to the paper. Where we got something wrong — and we got several things
badly wrong — the wrong number, the symptom, the diagnosis and the corrected number are all
printed. Part VI exists for exactly that purpose.

</div>

</div>

<div class="page-break"></div>

# Executive summary

**What JPEG AI is.** JPEG AI is the first international image-compression standard whose
compression engine is a trained neural network rather than a hand-designed mathematical
transform. It was finalised in 2025 by ISO/IEC JTC 1/SC 29/WG 1 — the same committee that
produced the original JPEG in 1992 — and published as ITU-T T.840 | ISO/IEC 6048. Where JPEG
uses the discrete cosine transform, a fixed formula written down by humans, JPEG AI uses an
*analysis transform*: a convolutional network whose 4.9 million weights were learned by
gradient descent from photographs. The paper that defines this report's scope is the
committee's own overview of that standard.

**The problem it solves.** Images are roughly 70% of the bytes on the public internet. A 20%
reduction in image bitrate at equal visual quality is therefore worth an enormous amount of
bandwidth, storage and battery. Classical codecs — JPEG, WebP, AVIF, VVC — have been
improving at a decelerating rate for thirty years, because a hand-designed transform can only
exploit the statistical structure a human noticed and wrote down. A learned transform can
exploit whatever structure exists in the training data. JPEG AI reports **16.2% to 27.0%**
fewer bits than VVC Intra — the strongest classical codec available — at equal quality on its
own test set, while decoding a 4K image in about 190 ms.

**What we built.** A working neural image codec, from scratch, in PyTorch, to the
architecture the paper describes: a two-branch YCbCr autoencoder (a 160-channel luma latent
and a 96-channel chroma latent, both at one-sixteenth resolution), a hyperprior that
transmits a compressed description of the latent's own statistics, residual coding against a
learned prediction, a 4-stage Multi-Context Model, and a real range/rANS entropy coder that
writes real bytes. Six of the fourteen planned phases are complete. The code is 14,154 lines
of Python across 33 modules and 13 test files, verified by **332 automated tests** and a
**210-check self-test** that exercises the entire encode–decode path on the user's own
hardware in 30 seconds.

**What we measured.** Three full rate ladders are trained — five operating points each, 50,000
optimisation steps per point, about 12 to 35 hours of wall-clock per ladder on an Apple M2 Pro.
On the 24 Kodak images, against JPEG, on the paper's own seven-metric average, they score
**−0.4%** (reduced-width development tier), **+1.8%** (the paper's own channel widths) and
**−9.2%** (adding the Multi-Context Model). Read plainly: the first two are level with JPEG and
the third is ahead of it, at roughly WebP's level. All three are decisively *ahead* on the
structural and perceptual metrics — MS-SSIM −26% to −35%, FSIM −17% to −30% — and *behind* on
the two that track pixel error, though that is closing fast: VMAF +30% → +12%, PSNR-HVS
+38% → +18% across the three ladders.

Splitting by colour plane locates the deficit precisely and identically in all three: our
**chroma** is at or past AVIF's level (−59% and −52% BD-rate on the two chroma planes, where
AVIF is −59% and −57%), while our **luma** is still worse than JPEG, at +10%. That is a
69-percentage-point spread between two branches of the same model, trained by the same recipe,
and it is the single most useful finding in the project so far. The luma figure has moved
+49% → +28% → +10% as we widened the branch and then gave it a context model, so it is
responding to treatment — but it has not yet crossed zero.

**What went wrong, and what that bought.** Seven substantive bugs were found and fixed, five of
them the kind that produce plausible numbers rather than crashes. The most expensive was in
our own BD-rate measurement code, not in the codec: a global cubic fit — the textbook
Bjøntegaard method — is invalid for metrics that saturate near their maximum, and it moved
both headline figures by about 17 percentage points in the wrong direction. Both were first
reported as +15.6% and +20.6%. A second bug in the entropy-model calibration was costing 1.8%
of every payload. A third silently degraded a warm start by permuting channels. In each case
this report prints the wrong number, the symptom that exposed it, and the measurement that
confirmed the fix.

**The most important measurement we made** was not a result but a *bound*. When the
reduced-tier codec stopped improving at 32.3 dB, we did not guess why. We reconstructed the
same images with the quantiser switched off — infinite bitrate — and the transforms still
topped out at 32.30 dB. Then we computed the optimal *linear* transform of the same width
(block PCA on Kodak's own pixels) and got 30.91 dB. Two numbers, and they settle it: the
entropy coder costs nothing, more training cannot help, and the ceiling is the latent width.
The learned transform is 1.4 dB *better* than the best possible linear transform of the same
size, which is the transform doing its job. Widening to the paper's own 160 channels was then
measured, not argued: **+2.12 dB at a matched 1.34 bits per pixel**.

**Honest position.** Three ladders are trained. The first two are level with JPEG; the third,
which attaches the multi-stage context model to the luma branch, is **9.2% ahead of JPEG** —
roughly WebP's level, and the first thing we trained that beats a shipped codec.

Against the standard we are much further behind than an earlier draft of this report claimed.
The paper's Kodak figure for the simplest decoder is −7.5%, but that is **against VVC Intra**,
whereas all of our numbers are against JPEG; earlier drafts differenced the two and reported a
gap of "about 8 percentage points." Converting properly (§19.1.2) puts the paper's decoder at
roughly **−41% vs JPEG**, so the real gap is **about 32 points**, and at matched quality we
spend about **1.5× the paper's bits**. About 3.5 points of that is the eight coding tools we
have not built (phases 7–14); the remaining ~28 are the luma branch and a training budget of
50,000 steps on a laptop, and we cannot yet separate those two.

<div class="page-break"></div>

<h1>Contents</h1>

[TOC]

<div class="page-break"></div>

# How to read this report

The report is in seven parts and six appendices. They are meant to be read in order, but
they are also meant to be skippable.

| If you want… | Read |
|---|---|
| the short version | the Executive Summary, then chapter 4 |
| to understand what JPEG AI *is*, from zero | Part I (chapters 1–4) |
| the mathematics and physics | Part II, chapter 5 |
| the standard's architecture in detail | Part II, chapter 6 |
| where every fact and file came from | Part III (chapters 7–12) |
| the plan and what is built | Part IV (chapters 13–15) |
| the results | Part V (chapters 16–20) |
| the failures, bugs and errors | Part VI (chapters 21–25) |
| what is left to do | Part VII (chapters 26–28) |
| definitions, commands, tables | the appendices |

**Three conventions used throughout.**

**BD-rate** is the headline unit. It answers: *for the same visual quality, what percentage
more or fewer bits does codec A need than codec B?* **Negative is better.** −20% means "20%
fewer bits for the same quality". It is explained properly in §5.9.

**Provenance tags.** Every numeric constant in our code carries one of three tags, and this
report uses the same three:

| tag | meaning |
|---|---|
| **CONFIRMED** | read directly out of the WG1 reference software. Normative — part of the standard. Not ours to tune. |
| **PAPER** | stated in the overview paper's own text or tables. |
| **OURS** | our own choice, with no normative source. Free to tune, and listed as ours wherever it appears. |

This matters because a report that silently mixes the three is not checkable. Appendix E
lists every constant with its source file.

**Two tiers.** `full.yaml` is the paper's own configuration: 160 luma latent channels, 96
chroma. `tierA.yaml` is a deliberately narrowed development configuration — 96 and 48 — that
exists so a laptop can validate the whole pipeline in hours instead of days. Results are
always labelled with the tier they came from, because §20 shows the tier is worth 2.12 dB.
