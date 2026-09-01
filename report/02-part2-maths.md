<div class="page-break"></div>

# Part II — The science

*This part is the mathematics and the physics. Chapter 5 builds the theory from information
theory up to the specific machinery JPEG AI uses, with every formula we actually implement.
Chapter 6 is the standard's architecture in full detail. A reader who wants to know what the
project* does *rather than why it works can skip to Part IV.*

## 5. The mathematics and physics we use

### 5.1 Information theory: entropy is the price of a symbol

Everything in compression rests on one 1948 result. If a symbol `s` occurs with probability
`p(s)`, then the *cheapest possible* encoding spends

```
    ℓ(s) = −log₂ p(s)   bits
```

on it. A symbol you expected (`p = 0.5`) costs 1 bit. A symbol you were almost certain of
(`p = 0.99`) costs 0.0145 bits. A surprise (`p = 0.001`) costs 10 bits. Averaging over the
source gives its **entropy**

```
    H(S) = − Σ  p(s) · log₂ p(s)     bits per symbol
             s
```

and Shannon's source coding theorem says no lossless code can average fewer bits than `H(S)`,
while codes exist that get arbitrarily close.

Two consequences drive the entire design of a learned codec.

**Consequence A: compression is prediction.** The `−log₂ p` above uses the *true* probability.
If you code with a model `q` instead, you spend

```
    E_p[ −log₂ q(s) ] = H(p) + KL(p ‖ q)     bits per symbol
```

— the entropy you could never avoid, *plus* the Kullback–Leibler divergence between the truth
and your model. That KL term is pure waste, and it is the only part you can control. **So
"build a better codec" and "build a better probability model" are the same sentence.** This is
why a learned codec's entropy model is not a detail; it is where the gains live.

**Consequence B: this is why the hyperprior exists.** If your model `q` is fixed for all
images, it must be an average over all images, and `KL(p_this_image ‖ q_average)` is large.
If instead you can *spend a few bits telling the decoder about this particular image's
statistics*, you shrink KL by more than the bits you spent. That trade is the hyperprior
(§5.6.3), and it is the single largest architectural gain in the field.

We use this directly. Our rate is computed as `−log₂` of the model's own predicted probability,
summed over every latent value and divided by the number of pixels, and it is checked against
the byte count the coder actually produced. That check — the same quantity computed two ways
and required to agree — is our strongest correctness test (§15.2).

### 5.2 Rate–distortion: what is actually being minimised

Lossy compression has two costs that trade against each other: **rate** `R` (bits) and
**distortion** `D` (how wrong the reconstruction is). You cannot minimise both. What you can
do is minimise a weighted sum, which is the standard Lagrangian relaxation of "minimise `D`
subject to `R ≤ R_target`":

```
    L = R + λ · D
```

Sweep `λ` and you trace the **rate–distortion curve**: the set of achievable
(rate, distortion) pairs. Every codec comparison in this report is a comparison between two
such curves.

**Our sign convention, and why it needed a derivation.** We write

```
    L = β · D₂₅₅ + R
```

so `β` multiplies the *distortion*, and larger `β` means higher quality and higher rate. `D₂₅₅`
is a mean-squared error on the 0–255 scale. The paper does not state which term its `β`
multiplies, and getting it backwards silently inverts the entire rate ladder — you would train
five models, all of them fine, and the curve would run the wrong way. So the direction was
derived from three independent constants found in the WG1 reference software:

1. the standard's rate ladder is 18 values of `β` from 0.0002 to 3.0;
2. the four *trained* base models sit at `β` = 0.002, 0.012, 0.075, 0.5;
3. `betaDisplacementLog` is signalled in [−40, +40] with a precision of 5 bits.

Take (3) first: a log-domain displacement with precision 5 means the effective `β` is
`base · 2^(disp/2⁵)`, so [−40, +40] spans `2^±1.25` — a factor of 0.42 to 2.38, i.e. **5.66×
end to end**. Now look at the gaps between adjacent base models in (2): 0.012/0.002 = 6.0,
0.075/0.012 = 6.25, 0.5/0.075 = 6.67. **The displacement range covers exactly the gap between
neighbouring base models** — which is what the mechanism is *for*, and is strong evidence the
numbers are being read correctly rather than coincidentally.

Now the direction. `β = 0.002` against a 0–255 MSE equals `0.002 × 255² = 130` against a [0,1]
MSE. The published MSE Lagrange multipliers for this exact architecture family, on the same
scale, run 117 / 228 / 436 / 845 / 1625 / 3140 from lowest to highest quality. JPEG AI's lowest
base `β` lands within 11% of the lowest of those, and its highest (0.5 → 32,500) sits above the
highest. **That only works if `β` multiplies the distortion.** Reasoned inference from three
constants, not a normative citation — and it is recorded as such in `jpegai/train/losses.py`,
whose module docstring is this paragraph.

**What `D` is, for us.** A weighted MSE in YCbCr with luma weighted 6:1:1 against the two
chroma planes, plus a small MS-SSIM term:

```
    D = Σ  w_c · MSE(x̂_c, x_c) · 255²  +  γ · (1 − MS-SSIM(x̂, x))
        c∈{Y,U,V}

    w = {Y: 6, U: 1, V: 1} / normalised so Σw = 3          [OURS]
```

The 6:1:1 is **ours**: the paper says only "prioritise the quality of the luma component during
training". The MS-SSIM term is also ours, and it is there for a specific reason — six of the
paper's seven metrics are structural or perceptual and *none* of them is PSNR. Optimising pure
MSE produces a model that wins a metric nobody is grading and loses the average that decides
the BD-rate.

**How we verified the 6:1:1 weighting is applied correctly.** Add a uniform offset to all three
RGB channels. In YCbCr that is a pure luma error — the chroma planes are unchanged by
construction. So the weighted loss should score it exactly `6 × 3/8 = 2.25×` an unweighted
loss. It does, to numerical precision. That test is in `tests/test_losses_twobranch.py` and it
exists because a weighting bug is otherwise invisible.

### 5.3 Making the rate differentiable

`R` is the length of a file: an integer, produced by a discrete algorithm, with no derivative.
Training needs a gradient. The substitution that makes this work is the one from §5.1:

```
    R  ≈  E [ −log₂ p_model(ŷ) ]
```

The *expected code length under the model* is a smooth, differentiable function of the model's
parameters **and** of the latent values, and by Shannon it is within a fraction of a bit of what
a good arithmetic coder will actually produce. So we minimise the model's own prediction of the
file size, and then — crucially — we *check* against the real file size afterwards. §15.2. Those
two numbers agreeing to within 0.5% is the whole basis for trusting any rate we report.

### 5.4 Transform coding as an autoencoder

The classical pipeline and the learned pipeline are the same four boxes:

```
   classical:   x → [ DCT ]      → coeffs → [ quantise ] → [ Huffman ] → bits
                                                                ↓
   learned:     x → [ g_a: CNN ] →   y    → [ quantise ] → [ entropy   ] → bits
                                                            [ model p  ]
```

with an inverse chain on the other side. Formally:

```
    y  = g_a(x; φ)              analysis transform, learned
    ŷ  = Q(y)                   quantisation (rounding)
    x̂  = g_s(ŷ; θ)              synthesis transform, learned
    R  = −log₂ p(ŷ; ψ)          entropy model, learned
```

and all three parameter sets `φ, θ, ψ` are trained jointly by minimising `β·D + R`. This is
formally a **variational autoencoder** with a uniform posterior, which is where the term
"hyperprior" comes from — but for understanding the codec, "learned transform + learned
probability model" is the more useful reading.

**Why one-sixteenth resolution.** The analysis transform has four stride-2 stages, so the
latent is at `H/16 × W/16`. With 160 luma channels that is

```
    160 / (16 × 16) = 0.625 latent values per pixel
```

Compare: JPEG keeps one DCT coefficient per pixel and throws most of them away by
quantisation. A learned codec throws its information away *spatially, in the transform*, and
keeps a dense small tensor. The reason /16 rather than /8 or /32 is a compute/quality trade:
each halving of resolution quarters the number of latent positions, which quarters the entropy
coder's work and the context model's sequential depth, but costs reconstruction detail. /16
with a wide channel dimension is where the field settled empirically.

### 5.5 The quantisation problem, and the three fixes

`ŷ = round(y)` has derivative zero almost everywhere and undefined at half-integers. Gradient
descent through it learns nothing. Three fixes are used, and JPEG AI-class codecs use all
three, in different places:

**Fix 1 — additive uniform noise (Ballé 2017).** During training, replace rounding with
`ŷ = y + u`, `u ~ U(−½, ½)`. Rounding to the nearest integer perturbs a value by something in
`[−½, ½]`, so uniform noise has the same support and a similar effect, and it is differentiable
in `y`. It also has an exact interpretation: it makes the model a VAE whose posterior is a unit
uniform box, so the "rate" term is a genuine variational bound. **We use this for the rate
branch** — the branch that computes `R` — because the noise model is what makes the rate a
proper bound.

**Fix 2 — the straight-through estimator.** Round in the forward pass, pretend the derivative
is 1 in the backward pass:

```python
    y_hat = y + (torch.round(y) - y).detach()
```

The `.detach()` makes the correction term a constant, so the forward value is exactly
`round(y)` and the gradient flows as if it were the identity. **We use this for the distortion
branch**, because the decoder must be trained on exactly the values it will receive. Training
the synthesis transform on noisy latents and then feeding it rounded ones at inference is a
train/test mismatch worth several tenths of a dB.

**Fix 3 — a discretised likelihood.** The rate must be the probability of an *integer*, not a
density at a point. So every probability is computed as a difference of cumulative
distributions over the unit interval around the value:

```
    p(ŷ) = C(ŷ + ½) − C(ŷ − ½)
```

This is the same function the entropy coder's CDF tables encode, which is not a coincidence —
it is what makes the loss's rate estimate and the coder's byte count comparable at all.

Using different quantisation surrogates in the two branches sounds like a hack and is in fact
the standard practice, for the reason given: the rate branch wants the variational bound, the
distortion branch wants the real values.

### 5.6 The entropy-model ladder

This is the axis along which learned compression has actually improved, and every rung is worth
understanding because JPEG AI's design is a specific point on it. All five rungs exist in our
codebase; we built them in order, because each one is the previous one plus a term.

#### 5.6.1 Rung 0 — a fixed factorised prior

Assume every latent value is independent, with a per-channel distribution learned once and
shared by all images:

```
    p(ŷ) = Π  p_c ( ŷ[c,i,j] )
          c,i,j
```

`p_c` is a small monotone network trained to be a CDF. This is the JPEG-with-Huffman-tables
level of sophistication: no adaptation to the image at all. It is not competitive on its own,
but it is not useless either — it is exactly what JPEG AI uses for the **hyper-latent** `ẑ`,
because `ẑ` is small and there is nothing left to condition it on.

#### 5.6.2 Why that is not enough

Latent values are *not* independent, in two ways. Spatially: a latent position over sky has
small values, one over a face has large values, and neighbouring positions agree.
Cross-channel: channels respond to correlated features. A fixed factorised model has to average
over all of this, so its KL term (§5.1) is large.

#### 5.6.3 Rung 1 — the scale hyperprior (Ballé et al. 2018)

The key move in the field. **Transmit a compressed description of the latent's own local
scale.** A second, much smaller autoencoder runs on top of the latent:

```
    z  = h_a(y)                    hyper-encoder: 2 more stride-2 stages → /64
    ẑ  = round(z)                  coded with the factorised prior of §5.6.1
    σ  = h_s(ẑ)                    hyper-decoder → a per-value standard deviation
    p(ŷ | ẑ) = N(0, σ²) ⋆ U(−½,½)  a Gaussian, convolved with the unit box
```

The decoder decodes `ẑ` first — it is small, typically a few percent of the payload — then runs
`h_s` to obtain a *per-position, per-channel* `σ`, and uses those `σ` to decode the main
payload. The bits spent on `ẑ` buy a probability model tuned to this image. The trade is
strongly favourable: this single change was worth roughly 20% BD-rate when introduced.

Physically: the hyper-latent is a *side channel carrying the local variance field*. Sky gets
small `σ` and therefore cheap symbols; texture gets large `σ` and expensive ones. The codec has
learned where in the image the information is.

#### 5.6.4 Rung 2 — mean-scale, i.e. residual coding (Minnen et al. 2018)

Let the hyper-decoder predict a **mean** as well as a scale:

```
    (μ, σ) = h_s(ẑ)
    p(ŷ | ẑ) = N(μ, σ²) ⋆ U(−½,½)
```

Equivalently and more usefully: code the *residual* `y − μ` instead of `y`. If the prediction is
any good, the residual is smaller and centred on zero, so it costs fewer bits. Worth roughly
another 10%.

**This is exactly JPEG AI's equations (1) and (2)**, with the paper's notation `p̈` for the
prediction:

```
    (1)   r̂ = round( y − p̈ )
    (2)   ŷ = r̂ + p̈
```

The paper calls it *residual coding* rather than mean-scale, and — a detail with real
consequences — it uses **two separate networks** for the two outputs: a *hyper decoder* that
produces `p̈`, and a *hyper scale decoder* that produces the scale. §5.6.6 and §6.4 explain why
that split is not cosmetic.

#### 5.6.5 Rung 3 — autoregressive context, and why it is unshippable

Condition each latent value on the values already decoded:

```
    p(ŷ[i]) = f( ẑ, ŷ[1..i−1] )        e.g. a masked 5×5 convolution
```

This is the best-performing entropy model in the literature — another ~8–10% — and it is
fatal for deployment. Decoding position `i` requires position `i−1` to be finished, so the
decoder is a strictly sequential loop over every latent position. For a 4K image at /16 with
160 channels that is about two million serialised network evaluations. The paper says the
quiet part out loud: the GPU is not compute-bound, **sequential entropy decoding dominates**.
Note in the paper's own Table III that a 15× increase in decoder MACs (14 → 215 kMAC/pixel)
costs only 285 → 323 ms per megapixel. The convolutions are nearly free; the serialisation is
not.

#### 5.6.6 Rung 4 — the checkerboard / multi-stage context model

The compromise, and JPEG AI's choice. Split the latent's *spatial* positions into a small number
of groups. Code group 1 with no context. Code group 2 using group 1. Code group 3 using groups
1–2. And so on. Within a group everything is independent, so a group decodes **fully in
parallel**.

JPEG AI uses **four** groups, formed by a 2×2 spatial pattern — the four cosets of a 2×2 tile —
so the decoder always makes exactly **four** passes, *whatever the image size*. That is the
property that makes it shippable: constant latency in the number of sequential steps, and it is
what the paper calls the **Multi-Context Model (MCM)**.

Two facts about JPEG AI's MCM that we had to establish from the reference software rather than
the paper:

**The stage order is `(0,0) → (1,1) → (0,1) → (1,0)`** — diagonal first, then the two
off-diagonals. This is confirmed twice over, from unrelated code (§9.3).

**MCM partitions space, never channels.** In the paper's Figure 2 the colour is constant along
the channel axis and varies only over the 2×2 spatial tile. All 160 channels of a given spatial
position belong to the same stage.

And one more, which is the reason it helps us specifically: **MCM is applied to the luma branch
only.** Chroma uses the plain eq. (1)/(2) residual coding. Since §19.3 shows our deficit is
entirely in luma, this is the right tool aimed at the right place.

### 5.7 Arithmetic coding, range coding, and ANS

The entropy model gives probabilities. Something has to turn probabilities into bytes at the
Shannon limit. Three families:

**Arithmetic coding.** Represent the whole message as a single real number in [0,1), narrowing
the interval by `p(s)` at each symbol. Optimal to within one bit for the entire message.
Requires arbitrary-precision arithmetic in principle, and multiply/divide per symbol in
practice.

**Range coding.** Arithmetic coding on integers with renormalisation. This is what JPEG 2000,
HEVC, VVC and every practical arithmetic coder actually is. Still needs a multiply and usually a
divide per symbol.

**ANS — asymmetric numeral systems (Duda, 2009).** A different construction with the same
optimality. State `x`; encoding a symbol maps `x → x'` in a way that packs `−log₂ p(s)` bits of
information into the state, spilling bytes when it overflows. Two variants matter:

- **rANS** — range variant, arithmetic-based, table-free. Compact and fast. **This is what we
  use.**
- **tANS** — table variant. All state transitions are *precomputed into a lookup table*, so
  encode and decode are pure table lookups with **no multiplication and no division at all**.
  This is why it is chosen for hardware and for phones.

**JPEG AI uses "me-tANS": multi-symbol, escape-coded tANS.** Its properties, all confirmed from
the reference software (§9.4):

| property | value | why |
|---|---|---|
| probability mass bits | 8 → 256 states per σ-class | small tables |
| residual CDF table | `[32, 256]` — 32 σ-classes × 256 symbols | table selected by `Iσ` |
| hyper CDF table | one row per channel | table selected by channel index |
| total decoder tables | ≈ 100 KB | fits a phone's cache |
| decode direction | **FILO** — last symbol first | a property of ANS, not a design choice |
| escape coding | 1 flag bit → 2 or 15 bits → sign | keeps the table to 256 symbols while still coding rare huge values |
| skip mode | up to **80%** of residuals not coded at all | when `Iσ` is below a threshold, the residual is *inferred* zero |
| substreams | **2, interleaved, even single-threaded** | "mimicking a dual-threaded setup" — a hardware-friendliness decision |

The table split is the clever part. The CDF is split into halves with the second offset by 128:

```python
    cdf_first, cdf_second = cdfs - (cdfs >> 1), (cdfs >> 1) + 128
```

which lets the decoder be an **OR plus an addition** instead of a comparison chain. That is the
mechanism behind the paper's claim that decode needs no multiply and no divide.

**We verified the "~100 KB of tables" claim arithmetically:** 32 σ-classes × 256 states × 4
bytes = **32 KiB** for the decode transition table alone, with the encode transitions and the
CDF/PMF matrices making up the rest. ~100 KB is the right order of magnitude. The claim holds.

### 5.8 Colour, and why two branches

The eye is far more sensitive to brightness than to colour. Every codec since JPEG exploits
this by converting RGB to a luma-plus-chroma space and spending fewer bits on chroma. JPEG AI
uses **YCbCr BT.709**:

```
    Y  = 0.2126·R + 0.7152·G + 0.0722·B
    Cb = (B − Y) / 1.8556 + 0.5
    Cr = (R − Y) / 1.5748 + 0.5
```

(The coefficients sum to 1 — we assert that in code, because a typo there is otherwise
invisible.) The inverse, which is what the decoder runs:

```
    R = Y + 1.5748·(Cr − 0.5)
    B = Y + 1.8556·(Cb − 0.5)
    G = (Y − 0.2126·R − 0.0722·B) / 0.7152
```

**Two architectural options exist**, and the choice is not obvious:

**(a) One network, three input channels.** Feed YCbCr as a 3-channel tensor to a single
analysis transform. Simple; one latent; the network can learn cross-colour structure freely.

**(b) Two branches.** A *primary* branch on luma alone, and a *secondary* branch on the two
chroma planes, with separate latents, separate hyperpriors and separate entropy streams.

JPEG AI chooses **(b)**, and pays for it: our own measurement puts the second branch at **+19%
decoder complexity** (111.6 → 132.4 kMAC/pixel, of which chroma is 27.0). What does that buy?

1. **A monochrome fast path.** The luma branch is completely independent of chroma — we
   verified from the paper's own Figure 1 that information flows luma→chroma at exactly two
   points and never the other way. So a grey image, or a viewer that only needs luminance, can
   decode the luma stream alone and skip the rest. We measured this: **11.9% to 17.0% fewer
   bits** and **23–25% faster decode**, with the luma output *bit-identical* to the full path.
2. **Different rate control per plane.** Separate entropy streams and separate gain vectors
   mean luma and chroma quality can be traded independently, which a single latent cannot do.
3. **Native chroma subsampling.** 4:2:0 input means the chroma branch's input is already half
   resolution, so it uses one fewer downsampling stage (`downsample_factor = 2` in the
   reference software) and both latents land on the same /16 grid.

The cross-links, per eqs (1)–(3) and the paper's Figure 1:

- the **chroma hyper encoder** additionally sees the luma latent (helping predict `p̈_UV`);
- the **chroma synthesis transform** takes `ŷᶜ_UV = concat(ŷ_UV, ŷ_Y)` — the paper's eq. (3) —
  which is `96 + 160 = 256` channels. Note it takes the luma **latent**, not the decoded luma
  image.

Nothing flows chroma → luma. That asymmetry is the monochrome fast path.

### 5.9 BD-rate: how coding gain is measured, and how we got it wrong

#### 5.9.1 The definition

Two codecs cannot be compared at one operating point, because they will not be at the same
rate *and* the same quality. So:

1. Encode the test set at several quality settings with each codec. You get two curves of
   (rate, quality) points.
2. Plot quality against **log rate** — log because rate is multiplicative; a 10% saving means
   the same thing at every scale.
3. Interpolate each curve.
4. Over the **quality range the two curves share**, integrate the horizontal distance between
   them.
5. Exponentiate: the average *percentage* rate difference at equal quality.

```
                    1                  q_high
    BD-rate = exp( ─────────  ∫  [ log R_test(q) − log R_ref(q) ] dq ) − 1
                   q_hi − q_lo   q_low
```

Negative means the test codec needs fewer bits: better.

#### 5.9.2 The paper's protocol, which we follow

- Seven metrics: **MS-SSIM, VIF, FSIM, VMAF, NLPD, PSNR-HVS, IW-SSIM**.
- BD-rate computed **per metric first, then averaged**. Not "average the metrics, then
  BD-rate". These give different answers and the paper's is the former.
- The **AVG** column is the plain unweighted arithmetic mean of the seven. The paper never
  states this; we verified it arithmetically against all 19 published data rows of its Tables
  III–VI (§8.2).
- NLPD is the only metric where lower is better, so it is negated before integration.
- PSNR is *reported but never averaged in*, matching the paper. We additionally report per-plane
  PSNR-Y/U/V as a diagnostic, which turned out to be the most informative thing in our results.

#### 5.9.3 The interpolant, and a 17-point error

**The textbook method — Bjøntegaard's — fits a single global cubic** to each codec's whole
curve and integrates it over the overlap. We implemented that. It is **wrong for this metric
set**, and it moved both our headline numbers by about 17 percentage points.

The failure mode: *a global fit is global*. Anchor points **outside** the integration window
pull the polynomial **inside** it. For a well-behaved metric like PSNR this barely matters. For
a **saturating** metric it is catastrophic — and four of the paper's seven saturate hard.
MS-SSIM, FSIM and IW-SSIM all crowd into the last 1% of their range: at high rate the curve is
nearly vertical in log-rate, at low rate it is gentle. **One cubic cannot be both**, so it
compromises, and the compromise is set by points you are not integrating over.

The fix is **monotone PCHIP** — piecewise cubic Hermite interpolation. It is (a) *local*: a
point outside the window cannot influence the answer inside it; (b) *monotone*: it never
overshoots, so it cannot invent a quality reversal that the data does not contain; and (c)
*interpolating*: it passes through the measured points exactly. JVET's common test conditions
moved to it for the same reason.

The correction, printed in full in §22.1:

| | global cubic | PCHIP |
|---|---|---|
| ladder #0 | +15.6% | **−0.4%** |
| ladder #1 | +20.6% | **+1.8%** |
| WebP | −15.3% | −10.6% |
| AVIF | −41.0% | −36.1% |

**How to test a BD-rate implementation.** Not against a golden value — you will simply encode
your own bug into the expected number. Test the **invariance**: four anchor sweeps that pass
through the integration window *identically* and differ only in how far *below* it they extend
must produce the same answer. PCHIP spreads **0.04** percentage points across those four. The
cubic spreads **17.08**. That assertion is now a regression test.

#### 5.9.4 Overlap coverage, and why two of our own numbers must not be compared

A five-point ladder against an eleven-point anchor sweep may only *share quality range* with
four of the anchor's points. The BD-rate is then an average over a **slice** of the comparison
— and typically the low-rate slice, where a learned codec is at its strongest. So:

> **Two BD-rates with different overlap coverage are not comparable to each other.**

We print `overlap` in every table for this reason. It is why ladder #0's −0.4% (4/11) must not
be read as "better than" ladder #1's +1.8% (6/11), even though it is a smaller number. To
compare our own ladders we use matched-rate points, or re-anchor one ladder against the other.

### 5.10 The dimensionality bound: how we found the ceiling

This is the piece of mathematics that produced the project's most useful measurement, and it is
not from the paper.

The analysis transform maps a 16×16 region of a 3-channel image — **768 numbers** — to `N`
latent channels at one position. At tier A, `N = 96`: an **8:1** dimensionality reduction. That
is a hard constraint independent of training, architecture or bitrate. If 768 dimensions must
pass through 96, information is lost, and the amount is computable.

For the best *linear* transform the answer is exactly known. Principal component analysis —
equivalently the Karhunen–Loève transform — is the optimal linear map to `N` dimensions in the
mean-squared sense, and the residual error is the sum of the discarded eigenvalues:

```
    MSE_min(N) =  Σ  λ_k              λ_k the eigenvalues of the patch covariance,
                 k>N                  in descending order
```

So: take Kodak's own 16×16×3 patches, compute their covariance, take the eigenvalues, and you
get a **hard lower bound on the error of any linear transform at that width** — and a very
strong indicator for a nonlinear one, since a convolutional transform sees more than one block
and should therefore beat it.

Measured on Kodak:

| latent channels | reduction ratio | optimal *linear* PSNR | |
|---|---|---|---|
| 96 | 8.0 : 1 | **30.91 dB** | tier A |
| 160 | 4.8 : 1 | **35.02 dB** | the paper's luma width |
| 192 | 4.0 : 1 | 37.11 dB | |
| 320 | 2.4 : 1 | 46.75 dB | a research model's width |

Our tier-A learned transform reaches 32.30 dB, which is **1.4 dB better than the linear bound**
— the transform is doing its job. But 96 channels cannot exceed roughly 32–33 dB however long
it trains, and JPEG reaches 34.52 dB at the top of the comparison range. **That is the entire
explanation of ladder #0's result**, and it was obtained in an afternoon from an eigenvalue
decomposition rather than from a week of extra training. §20.
