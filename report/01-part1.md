<div class="page-break"></div>

# Part I — Orientation

*This part assumes nothing. It explains what an image codec is, what makes JPEG AI different,
what problem it solves, what we set out to do about it, and where the project stands. A reader
who stops after chapter 4 will still be able to describe the project accurately.*

## 1. What is JPEG AI?

### 1.1 First, what a codec does

A digital photograph is a grid of numbers. A 12-megapixel phone photo is 12 million pixels,
each carrying three colour values (red, green, blue), each value one byte. That is 36
megabytes. Nobody stores or sends 36 MB per photo, so every photograph you have ever seen on a
screen has been through a **codec** — a *coder–decoder* — that shrank it to something between
1 and 5 MB and then expanded it back again.

Codecs shrink images by exploiting two facts about photographs.

**Fact one: neighbouring pixels are similar.** In almost every natural image, a pixel is a
good predictor of the pixel next to it. Sky next to sky. Skin next to skin. This is
*redundancy*, and removing it is lossless — you can put it back exactly. A codec does this
with a **transform**: a change of coordinates that concentrates the image's energy into a few
numbers and leaves the rest near zero. Near-zero numbers are cheap to store.

**Fact two: the human eye does not see everything.** It is far more sensitive to brightness
than to colour, and far more sensitive to smooth gradients than to fine texture. So a codec
can throw information away in the places the eye will not miss it. This is *irrelevancy*, and
removing it is lossy — you cannot put it back. This is why a JPEG at quality 50 looks fine and
is ten times smaller than the original.

Classical JPEG, standardised in 1992, does exactly this: it converts RGB to a
brightness-plus-colour representation, splits the image into 8×8 blocks, applies the
**discrete cosine transform** (DCT) to each block, divides the resulting coefficients by a
table of numbers chosen by hand in the 1980s to match human vision, rounds them to integers,
and then packs the integers with an entropy coder. Every one of those steps is a formula a
human wrote down.

### 1.2 What JPEG AI changes

JPEG AI replaces the hand-designed transform with a **learned** one.

Concretely: instead of the DCT, an *analysis transform* — a convolutional neural network with
about 4.9 million weights — maps the image to a compact numerical representation called the
**latent**. Instead of the inverse DCT, a *synthesis transform* — another network — maps the
latent back to an image. Neither network was designed. Both were trained, by showing the
system hundreds of thousands of photographs and adjusting the weights by gradient descent
until the combination of "how many bits this costs" and "how wrong the reconstruction is" was
as small as possible.

Three consequences follow, and they are the whole point of the standard.

**Consequence 1: the transform can exploit structure nobody wrote down.** The DCT knows about
sinusoids. A learned transform knows about whatever recurs in photographs — edges, textures,
skin, foliage, sky gradients, the particular way light falls off — because those are what
minimised the loss.

**Consequence 2: the probability model can be learned too, and conditioned on the image.**
JPEG's entropy coder uses fixed Huffman tables. JPEG AI transmits, alongside the image, a
small compressed *description of the image's own statistics*, and uses that to code the main
payload. This is the **hyperprior**, and it is the single largest source of gain in learned
compression. It is explained in §5.6.

**Consequence 3: the decoder is where the standard lives.** A neural network is only
interoperable if everyone runs the same weights. So JPEG AI standardises the **decoder** —
its architecture, its weights, and the bit-exact integer arithmetic in the entropy path — and
leaves the encoder free. Anyone may invent a better encoder tomorrow; every existing decoder
will still read its output. This is the same contract as every previous JPEG and MPEG
standard, and it is why the paper spends most of its length on the decoder.

### 1.3 What JPEG AI is, in one paragraph

> JPEG AI is an international standard (ITU-T T.840 | ISO/IEC 6048, five parts, finalised
> 2025) for compressing photographs with a neural network. Its encoder maps an image to two
> latent tensors — one for brightness at 160 channels, one for colour at 96 — each at
> one-sixteenth of the image's spatial resolution. It transmits those latents as integers,
> coded against a probability model that is itself transmitted in compressed form. Its decoder
> comes in three complexity grades (14, 28 and 215 thousand multiply-accumulates per pixel) so
> that the same file decodes on a low-end phone and on a desktop GPU, at different quality.
> Against VVC Intra — the strongest classical still-image codec — it reports 16.2% to 27.0%
> fewer bits at equal quality, and it decodes 4K in about 190 ms.

### 1.4 Why it is called "AI" and not "neural"

Nomenclature, not substance. WG1 named the activity "JPEG AI" in 2019 when the exploration
began; the original scope included *machine consumption* of the compressed representation —
running object detection directly on the latent, without ever reconstructing an image. That
half of the scope was deferred to version 2. Version 1, which this report is about, is a human
viewing codec that happens to be built out of neural networks.

## 2. The problem statement

### 2.1 The engineering problem

Three numbers frame it.

**Images are most of the web's bytes.** Estimates vary between 45% and 75% depending on
methodology and year, but every estimate agrees images dominate. Anything that reduces image
bitrate applies to a very large denominator.

**Classical codec progress is decelerating.** The generational gains are roughly: JPEG (1992)
→ JPEG 2000 (2000), about 10%; → HEVC/HEIF intra (2013), about 30% over JPEG 2000; →
VVC intra (2020), about 25% over HEVC. Each generation took longer, cost more decoder
complexity, and delivered less. The reason is structural: a hand-designed transform can only
capture regularities a human has noticed and formalised, and after thirty years the easy ones
are taken.

**Learned codecs passed the classical ones in the research literature around 2020.** Ballé et
al. (2018) and Minnen et al. (2018) showed an autoencoder with a hyperprior beating HEVC
intra; by 2020 the best research models beat VVC intra on perceptual metrics. But research
models are not codecs. They have no bitstream format, no interoperability, no complexity
bounds, no integer arithmetic, no conformance testing, and they are typically trained one
model per bitrate. Nothing you could put in a phone.

**So the problem statement of JPEG AI is not "can a neural network compress images".** That
was answered in 2018. It is:

> Can the research result be turned into a *standard* — a single normative decoder
> specification with a defined codestream, bounded complexity on real hardware, integer
> bit-exactness where interoperability demands it, one model family covering the whole
> quality range, and the functionality features (region of interest, progressive decoding,
> tiling, arbitrary image sizes, HDR metadata) that a deployable image format requires?

Every design decision in the paper is an answer to some part of that question, and a great
many of them are compromises that a research paper would never make. Recognising which parts
of the architecture exist for *compression* and which exist for *standardisation* is most of
what it takes to read the paper properly. Chapter 6 flags them as they arise.

### 2.2 The specific technical difficulties

Six problems have to be solved simultaneously, and they conflict.

**(a) Quantisation is not differentiable.** Compression requires rounding to integers.
Training requires gradients. The derivative of `round(x)` is zero almost everywhere, so naïve
training produces no learning signal at all. §5.5.

**(b) The rate is not differentiable either.** The thing to be minimised is the *length of the
compressed file*, which is an integer number of bytes produced by a discrete algorithm. §5.3
shows the substitution that makes it a smooth function of the weights.

**(c) A better probability model needs context, and context is sequential.** The strongest
research entropy models predict each latent value from the values already decoded — which
forces the decoder to process one value at a time. For a 4K image that is millions of
sequential steps and it is unshippable. §5.6 and §6.5 describe the fix JPEG AI adopts: a
4-stage checkerboard that gets most of the benefit in exactly four parallel passes,
*regardless of image size*.

**(d) Float arithmetic is not reproducible across devices.** Two GPUs can compute the same
convolution and differ in the last bit. In the entropy decoder, a one-bit difference does not
cause a slightly worse image — it desynchronises the arithmetic decoder and destroys
everything after that point. JPEG AI's answer is to mandate bit-exactness *only* through the
entropy path, using an integer network with 8-bit multipliers and 32-bit accumulators, and to
tolerate small float differences after it. §6.6.

**(e) One model per bitrate does not scale.** A research paper trains six models for six
quality levels. A standard cannot ship 40. JPEG AI ships **four** trained parameter sets and
reaches an 18-point quality ladder by *scaling the latent* with learned per-channel gains plus
a signalled displacement. §6.9.

**(f) One decoder cannot suit both a phone and a workstation.** JPEG AI defines three
synthesis transforms of 14, 28 and 215 kMAC/pixel that all read **the same codestream**. A
phone runs the cheap one and gets a slightly worse image; a desktop runs the expensive one and
gets a better one, from the identical file. §6.10.

### 2.3 The academic problem — this project's problem

Our problem is narrower and it is the one this report answers:

> The overview paper describes the architecture completely and specifies almost nothing
> numerically. The documents that carry the numbers — ITU-T T.840-1 | ISO/IEC 6048-1 and the
> paper's own supplementary material — are behind paywalls we do not have access to. Given only
> the paper, publicly reachable software, and one laptop: **can the architecture be
> reconstructed, implemented, trained and honestly measured?**

The interesting part of that question is the word *honestly*. It is very easy to produce a
neural codec that reports excellent numbers and is broken. Four of the six bugs in Part VI
were of exactly this kind — they produced plausible results. So a large fraction of this
project's effort went into building measurement machinery that can *catch* our own mistakes,
before building the thing being measured. Chapter 15 is about that machinery, and Part VI is
the evidence that it works.

## 3. What we set out to achieve

Six objectives, in the order they were fixed. Each has a verifiable test attached, because an
objective without a test is a hope.

**Objective 1 — Understand the paper completely.** Not a summary: every equation, every tensor
shape, every design rationale, every marker code, and every place the paper is ambiguous or
self-contradictory. *Test:* a written study that reproduces the architecture in enough detail
to implement it without further reference, and that lists the paper's own internal
discrepancies. **Status: done.** `docs/02-jpeg-ai-explained.md`, 892 lines, section by
section, including three arithmetic errors found in the paper itself (§8.3).

**Objective 2 — Reconstruct every missing constant, with provenance.** *Test:* every number in
the configuration files traceable to a named source file and line, tagged CONFIRMED / PAPER /
OURS. **Status: done for 9 of 10 open questions.** Chapter 9 and Appendix E. Two of our own
early readings were *wrong* and are corrected there.

**Objective 3 — Build a real codec, not a research prototype.** Specifically: real bytes from
a real entropy coder; every reported bitrate measured from the file, never estimated from the
loss; every reported image decoded from those same bytes. *Test:* a round-trip gate that runs
during training and asserts the decoded latent is **bit-identical** to the encoded one, and
that actual bytes agree with the model's own prediction to within ±0.5%. **Status: done and
passing at every rate point.** §15.2. This gate is what caught two of the six bugs.

**Objective 4 — Measure against the paper's own criteria.** All seven of the paper's metrics
(MS-SSIM, VIF, FSIM, VMAF, NLPD, PSNR-HVS, IW-SSIM), on the paper's own datasets, with
BD-rate computed the way the paper computes it. *Test:* the seven-metric average reproduced
arithmetically against all 19 published data rows of the paper's Tables III–VI. **Status:
done.** §8.2 — and this is how we established that the paper's AVG column is the plain
unweighted mean, which the paper never states.

**Objective 5 — Train it, and report what happens.** Including the runs that fail. *Test:*
complete rate ladders with per-point logs retained, and a results section that prints the
numbers that were wrong alongside the numbers that are right. **Status: two ladders complete,
a third training.** Part V and Part VI.

**Objective 6 — Be honest about the gap.** *Test:* compare against the paper's *comparable*
figure, not its most flattering one. **Status: done, and it changed the target.** §8.4: the
paper's headline −16.2% is on its own 50-image test set against VVC. Our comparable figure is
its Table V — Kodak, simplest decoder — which is **−7.5%**. Using the headline number as a
target would have overstated our shortfall by 8.7 percentage points and, worse, would have
made a *correct* result look like a bug.

### 3.1 What we explicitly did not attempt

Stating this matters, because an implementation that quietly omits things and then reports a
comparison is not measuring what it claims.

| Not attempted | Why |
|---|---|
| Bit-exact conformance with the real standard | needs the normative learned weight tables (T1/T2/TP/TR) and the ONNX parameter sets, both distributed via Part 1, which is paywalled |
| The `me-tANS` entropy coder as specified | its transition tables are derived from the normative CDF tables we do not have. We use a mathematically equivalent rANS coder and have the me-tANS *constants* (Appendix E, §6.6) for phase 9 |
| The four post-processing filters (EFE ×2, ICCI, LEF) | phase 10; they are worth 0.2–0.3% each per the paper's own ablation, so their absence costs about 1% |
| Training at the paper's scale | the paper's models are trained on far more data for far longer than 50,000 steps on a laptop. This is the largest single confound in our comparison and §26 quantifies it |

## 4. Where the project stands today

One page, no narrative.

### 4.1 Built

| Phase | Content | State |
|---|---|---|
| 1 | Anchor codecs (JPEG, WebP, AVIF), dataset pipeline | complete |
| 2 | All seven metrics + PSNR/Y/U/V, BD-rate harness | complete |
| 3 | Mean-scale hyperprior, rANS coder, training loop, round-trip gate | complete, trained |
| 4 | Two-branch YCbCr architecture (separate luma/chroma latents) | complete, smoke-trained |
| 5 | Residual coding with split hyper decoders (eqs 1–3) | complete, trained |
| 6 | 4-stage Multi-Context Model (§VI-D) | complete, training now |
| 7–14 | three synthesis transforms, variable rate, me-tANS + codestream, RVS/LSBS/post-filters, integer path, functionality, report, CLI | not started |

### 4.2 Verified

| | |
|---|---|
| automated tests | **331** pytest tests, 12 files, 4,282 lines of test code |
| self-test checks | **210** without a checkpoint, **215** with one — full encode/decode path, ~30 s |
| metrics live | **7 / 7** of the paper's set, plus PSNR and per-plane PSNR-Y/U/V |
| coder fidelity | actual bytes within **±0.5%** of the model's own prediction at every rate point |
| latent fidelity | `ŷ` **bit-exact** through `decode(encode(x))` at every rate point |
| code size | 14,027 lines of Python (9,745 implementation + 4,282 tests) |
| documentation | 9 markdown documents, 5,051 lines, ~43,000 words |

### 4.3 Measured

BD-rate against JPEG on the 24 Kodak images, seven-metric average. **Negative is better.**

| codec | AVG | overlap | comment |
|---|---|---|---|
| WebP | **−10.6%** | 9/11 | our sanity anchor: a real, shipped codec |
| AVIF | **−36.1%** | 10/11 | AV1 intra — our stand-in for VVC-class performance |
| ours, ladder #0 (tier A, 96 ch) | **−0.4%** | 4/11 | level with JPEG |
| ours, ladder #1 (tier full, 160 ch) | **+1.8%** | 6/11 | level with JPEG, over a wider range |
| *the paper, Table V, decoder 0* | *−7.5%* | — | *the honest target* |

`overlap` counts how many of JPEG's eleven quality points lie inside our shared quality range.
It is printed because a BD-rate over 4 of 11 points and one over 10 of 11 are **not comparable
to each other** — see §5.9.4. This is why the two ladders' AVGs must not be differenced.

### 4.4 The three findings that matter most

**1. The bottleneck was located, not guessed.** Tier A stopped improving at 32.3 dB. With the
quantiser *disabled* — infinite bitrate — the transforms still reached only 32.30 dB, and the
optimal linear transform of the same width reaches 30.91 dB. So: the coder costs nothing, more
training cannot help, and the latent width is the wall. §20.

**2. The tier change is worth +2.12 dB, measured at a matched rate.** 1.3525 bpp → 32.71 dB
(tier A) against 1.3445 bpp → 34.83 dB (tier full). Not extrapolated — two real bitstreams at
the same size. §19.2.

**3. The remaining deficit is entirely in luma.** Chroma BD-rate −54.6% and −47.0%; luma
+28.2%. A 75-point spread between two branches of one model. Nothing in the training recipe or
the entropy coder is plane-specific, so this is a statement about the branches themselves —
and it is what the currently-training ladder attacks, since the Multi-Context Model attaches to
the luma branch only. §19.3.
