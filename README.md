# JPEG AI — a from-scratch implementation of the learning-based image coding standard

A working neural image codec built to the architecture described in:

> S. Esenlik, Y. Wu, Z. Zhang, Y.-K. Wang, K. Zhang, L. Zhang, J. Ascenso, S. Liu,
> "An Overview of the JPEG AI Learning-Based Image Coding Standard,"
> *IEEE Trans. Circuits Syst. Video Technol.*, vol. 36, no. 2, pp. 2520–2537, Feb. 2026.
> [DOI: 10.1109/TCSVT.2025.3613244](https://doi.org/10.1109/TCSVT.2025.3613244)

Real bitstreams, not estimates: every reported rate comes from bytes actually written by
the entropy coder, and every reported image is decoded from those same bytes.

## Status

Phases 3–6 of the [14-phase plan](docs/03-implementation-plan.md) are built:

| | |
|---|---|
| **Phase 3** | mean-scale hyperprior, rANS entropy coder, full training loop |
| **Phase 4** | two-branch YCbCr architecture (separate luma / chroma latents) |
| **Phase 5** | residual coding with split hyper decoders |
| **Phase 6** | the 4-stage Multi-Context Model (§VI-D), coset order derived from WG1 reference software |

Verification: **326 pytest tests**, **210 self-test checks** (215 with a checkpoint),
7/7 of the paper's metrics live, and the coder gated to ±0.5% against its own entropy
model at every rate point.

**Measured result so far.** The first trained ladder (Tier A, 5 rate points × 50k steps)
scores **+15.6% BD-rate against JPEG** on Kodak — i.e. worse than JPEG — and
[docs/07 §5.2](docs/07-training-runbook.md) shows why with two measurements rather than a
guess: with the quantiser *disabled* the transforms still top out at 32.30 dB, and block
PCA at the same latent width tops out at 30.91 dB. The bottleneck is Tier A's 96 latent
channels (an 8:1 dimensionality reduction), not the entropy coder and not the training
length. Training at the paper's own 160/96 widths is in progress.

## Read the docs in this order

| # | File | What it is |
|---|---|---|
| 0 | [00-START-HERE.md](docs/00-START-HERE.md) | Index, current status, and what each phase settled |
| 1 | [01-foundations.md](docs/01-foundations.md) | The prerequisites the paper assumes: rate–distortion, VAEs as codecs, hyperpriors, context models, ANS |
| 2 | [02-jpeg-ai-explained.md](docs/02-jpeg-ai-explained.md) | The paper section by section, with every shape, equation and rationale |
| 3 | [03-implementation-plan.md](docs/03-implementation-plan.md) | The 14-phase build plan |
| 4 | [04-reference-data.md](docs/04-reference-data.md) | Extracted tables, marker codes, tensor-shape cheat sheet |
| 5 | [05-tables-figures-verified.md](docs/05-tables-figures-verified.md) | Tables III–VI checked arithmetically; establishes Kodak as the yardstick |
| 6 | [06-normative-constants.md](docs/06-normative-constants.md) | Every constant not in the paper, traced to the WG1 reference software |
| 7 | [07-training-runbook.md](docs/07-training-runbook.md) | **How to run it**: pre-flight, the training commands, what the log lines mean, results so far |

## Quick start

```bash
bash setup.sh
```

Then confirm the whole codec path — transforms, CDF construction, rANS round-trip — on
your device in about 30 seconds:

```bash
.venv/bin/python -m jpegai.models.selftest --device mps
```

Train one rate ladder (see [docs/07 §7](docs/07-training-runbook.md) for the full sequence
and wall-clock estimates):

```bash
.venv/bin/python -u -m jpegai.train.runladder --tier full --model twobranch-mcm --name ladder_p6 --device mps --iterations 50000 --batch 8
```

Measure it against JPEG, WebP and AVIF through the same harness that produced the anchor
numbers:

```bash
.venv/bin/python -m jpegai.eval.runbench --codecs jpeg,webp,avif --neural checkpoints/ladder_p6
```

## Layout

```
jpegai/
  models/      transforms, hyperpriors, entropy models, MCM, self-test
  coder/       rANS entropy coder and CDF table construction
  codestream/  packet layout and markers
  train/       training loop and the rate-ladder driver
  eval/        metrics, anchor codecs, BD-rate, the benchmark harness
  data/        dataset loaders and the DIV2K zip salvage tool
  config/      tierA.yaml (development) and full.yaml (the paper's widths)
docs/          the written study and the runbook
tests/         326 tests
```

## What is not in this repo

Datasets (`data/`), trained weights (`checkpoints/`), the WG1 reference software
(`ref/`, a separate project under its own terms), and the paper's own text and figures
(IEEE copyright). `setup.sh` fetches the datasets; `docs/07-training-runbook.md` produces
the weights.
