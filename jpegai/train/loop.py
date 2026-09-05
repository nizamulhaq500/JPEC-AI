"""Training loop.

    python -m jpegai.train.loop --tier tierA --beta 0.002 --iterations 60000
    python -m jpegai.train.loop --model twobranch-vr --model-id 1 --stage III

Two optimisers, which is not optional and is the part people get wrong:

* **main** (Adam, `config.train.lr`) trains everything that participates in the
  rate-distortion trade-off: g_a, g_s, h_a, h_s, and the factorised density's MLP.
* **aux** (Adam, 1e-3) trains *only* `entropy_bottleneck.quantiles`, against
  `aux_loss()`, which measures how badly the learned CDF's stated tail points
  disagree with the CDF itself.

They must be separate because the two objectives are not commensurable. The
quantiles do not affect the rate the RD loss sees -- they only decide **how wide a
table `update()` builds**. Put them on the main optimiser and the RD loss will
happily shrink the table to reduce nothing, symbols start falling outside it, and
the actual byte count diverges from the estimate. That failure appears only in
the round-trip check, hundreds of thousands of steps later.

So the loop runs the round-trip check *during* training (`--rtcheck`), not after.
It costs one image every few thousand steps and it is the only signal that says
the real coder agrees with the loss curve.

`--stage` runs one row of the variable-rate paper's Table II: it takes beta, the loss
type, the step budget and the **freeze list** from `jpegai.train.stages`, so a stage
III run trains the decoder and entropy network and provably nothing else. Stages chain
through `--warm-start`, not `--resume`: the optimiser's param groups change between
stages by construction, so loading the old optimiser state would raise before it ever
reached the weights.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from jpegai.config import PROJECT_ROOT, load_config
from jpegai.models import KINDS, build_any_model
from jpegai.models.hyperprior import summarise
from jpegai.train.dataset import build_loaders
from jpegai.train.losses import MSE_SCALE, loss_from_config
from jpegai.train.stages import (STAGE_NAMES, anchor_beta, apply_freeze,
                                 aux_is_trained, check_partition, find_stage,
                                 part_parameters, sample_delta_beta, stage_steps)
from jpegai.utils import describe_device, pick_device, seed_everything

CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def lr_at(step: int, total: int, lr: float, lr_final: float,
          warmup: int = 1000) -> float:
    """Linear warmup, flat, then cosine to `lr_final` over the last 10%.

    Flat-then-decay rather than cosine throughout: a learned codec's entropy model
    needs a long stretch at a constant learning rate to settle, and decaying from
    step 0 tends to lock in whatever crude latent distribution the first few
    thousand steps produced. The final decay is what buys the last ~0.3 dB.

    Warmup exists because step 0 has sigma pinned at the 0.11 lower bound
    everywhere -- the rate term is then near zero and its gradient is nearly flat,
    so a full-size step goes somewhere arbitrary.
    """
    if step < warmup:
        return lr * (step + 1) / warmup
    decay_start = int(total * 0.9)
    if step < decay_start:
        return lr
    t = (step - decay_start) / max(1, total - decay_start)
    return lr_final + 0.5 * (lr - lr_final) * (1 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, valid, criterion, device, *, limit: int | None = None) -> dict:
    """Estimated bpp / PSNR over whole validation images, no noise, STE rounding.

    `noise=False` matters: validation must measure what the *coder* will do, which
    is rounding, not the uniform-noise relaxation used for the rate gradient. A
    validation number computed with noise is optimistic by a few percent and moves
    around between runs.
    """
    model.eval()
    n = len(valid) if limit is None else min(limit, len(valid))
    acc = {"bpp": 0.0, "psnr": 0.0, "mse": 0.0, "loss": 0.0}
    for i in range(n):
        x = valid[i].unsqueeze(0).to(device)
        out = model(x, noise=False, ste=True)
        r = criterion(out, x)
        for k in acc:
            acc[k] += float(r[k])
    model.train()
    return {k: v / max(1, n) for k, v in acc.items()}


@torch.no_grad()
def out_of_range_fraction(entropy_model, symbols: torch.Tensor,
                          indexes: torch.Tensor) -> float:
    """Fraction of symbols that fall outside their CDF row, so escape to bypass.

    This is the *leading indicator* for the gate. rANS handles an out-of-range
    symbol by coding the escape and then writing the raw value on a bypass path,
    which is cheap -- often cheaper than the model's own estimate for a value the
    model considers nearly impossible. So an overconfident or too-narrow table
    shows up as **actual < estimated**, a negative gap, which reads like good news
    and is not.

    Measured earlier against deliberately miscalibrated sigma: 0.00% out of range
    gave a +0.69% gap, 0.63% out of range gave -17.15%. A tenth of a percent of
    escapes is worth more than ten percent of rate, so this number needs to be
    watched directly rather than inferred from the gap.
    """
    off = entropy_model._offset.to(symbols.device)[indexes.long()]
    # A row of length L covers symbol values [offset, offset + (L-2) - 1]: L-2
    # real symbols, since _cdf_len counts the escape bin and the final total.
    length = entropy_model._cdf_len.to(symbols.device)[indexes.long()] - 2
    below = symbols < off
    above = symbols >= off + length
    return float((below | above).float().mean())


@torch.no_grad()
def delta_beta_check(model, valid, device, *, index: int = 0,
                     points=None) -> dict:
    """The Phase 8 arm of the gate: is the rate ladder real, and is Delta_beta = 0 free?

    Empty for a model with no gain unit, so the caller can ask unconditionally.

    Four properties, each one a bug this project can actually produce.

    **1. Delta_beta = 0 adds exactly nothing.** eq. (10)'s offset is *additive on
    Isigma*, so `offset(0)` must be the gain vector bit-for-bit -- and while the vector
    is still at its zero initialisation the multiplier must be exactly 1.0, which is
    what lets Table II's stage IV bolt a gain unit onto a trained checkpoint as a no-op.
    Checked as **bitwise tensor equality**, not with a tolerance: the failure mode is
    `Delta_beta` arriving as a float and dragging the offset by an ULP, which moves a
    handful of symbols onto the neighbouring CDF row. That costs a fraction of a percent
    of rate -- far too little to see in a loss curve, and permanent.

    `beta_unit_gain` reports whether the multiplier is *still* exactly 1, which is true
    at initialisation and legitimately false after stage III trains the vector. It is
    reported, not gated: gating on it would fail every stage IV run.

    **2. Fig. 9's cache agrees with the direct path.** Bit-rate matching encodes one
    picture ten times through `precompress`/`compress_cached`, reusing everything up to
    the gain unit. If that cache ever goes stale the rate search optimises a rate the
    real encoder will not produce, and every BRM number afterwards is fiction. Compared
    as **byte equality of the four strings**, per rung, because a stale cache off by one
    training step produces a perfectly valid packet at a slightly wrong rate.

    **3. Rate is non-decreasing in Delta_beta.** This is the *precondition of eq. (14)'s
    bisection* -- the search assumes it and returns an arbitrary point without it.
    Measured on an untrained Tier A model, `R(Delta_beta)` is a step function with
    plateaus about 128-136 wide (one CDF row), because an untrained model's Isigma is
    nearly constant across elements; 222 probes at stride 8 across the whole clamp gave
    13 changes and **zero decreases**. So this checks monotonicity, not strict increase:
    equal neighbours are normal, a decrease is the fault.

    **4. Every rung decodes bit-exact.** The offset is applied symmetrically at the two
    ends, and this is the arm that catches a `Delta_beta` written to the header in one
    convention and read in another -- a bug that leaves the encoder's own numbers
    perfect and only shows up in the reconstruction.

    Also returns the ladder itself and its span, because the span is what says whether a
    checkpoint is worth benchmarking: a gain unit whose vector has collapsed passes all
    four arms above and delivers a span of 1.0x. And `beta_sat_high`, the fraction of
    `Isigma + o` that runs off the top of the sigma table at the ladder's top rung --
    past `max_index` the coder's sigma stops growing while the scaled residual does not,
    so the residual's tail leaves the CDF's support. Anything above a fraction of a
    percent there means `rate.beta_train_sample` is reaching further than this
    checkpoint can actually deliver.
    """
    if not getattr(model, "gain", False):
        return {}
    from jpegai.models.gain import DELTA_BETA_MAX, DELTA_BETA_MIN

    model.eval()
    model.update(force=True)
    x = valid[index].unsqueeze(0).to(device)
    ladder = sorted({int(d) for d in
                     (points if points is not None
                      else (DELTA_BETA_MIN, 0, DELTA_BETA_MAX))})

    # (1) the identity, as arithmetic on the offset rather than as a rate comparison.
    # A rate comparison cannot see this: `compress(x)` and `compress(x, delta_beta=0)`
    # are the same call, so comparing their bytes would assert nothing at all.
    neutral, unit = True, True
    for branch in (model.branch_y, model.branch_uv):
        g = branch.gain
        neutral = neutral and bool(torch.equal(g.offset(0), g.vector))
        unit = unit and bool(torch.equal(g.scale(g.offset(0)),
                                         torch.ones_like(g.vector)))

    npix = x.shape[-1] * x.shape[-2]
    cache = model.precompress(x)
    bpps, cache_ok, exact, worst_err = [], True, True, 0.0
    for d in ladder:
        packet = model.compress(x, delta_beta=d)
        bpps.append(model.packet_bytes(packet) * 8 / npix)

        # (2) the same rung through the search's cache.
        cached = model.compress_cached(cache, delta_beta=d)
        cache_ok = cache_ok and all(
            packet[p][k] == cached[p][k]
            for p in ("luma", "chroma") for k in ("y_strings", "z_strings"))

        # (4) every rung survives the round trip, not just the anchor.
        dec = model.decompress(packet, device=device)
        ref = model(x, noise=False, ste=True, delta_beta=d)
        err = (float((dec["y_hat"] - ref["y_hat"]).abs().max())
               if dec["y_hat"].shape == ref["y_hat"].shape else float("inf"))
        worst_err = max(worst_err, err)
        exact = exact and err == 0.0

    # (3) monotone, with a tolerance of exactly zero. A plateau is fine; a dip is not.
    drops = [bpps[i] - bpps[i - 1] for i in range(1, len(bpps))
             if bpps[i] < bpps[i - 1]]
    top = model(x, noise=False, ste=True, delta_beta=ladder[-1])
    sat = model.branch_y.gain.saturation(
        top["i_sigma"], top["gain_offset"], max_index=model.sigma_index.max_index)

    model.train()
    return {
        "beta_points": ladder,
        "beta_bpps": [round(b, 6) for b in bpps],
        "beta_anchor_bpp": bpps[ladder.index(0)] if 0 in ladder else 0.0,
        "beta_span": max(bpps) / max(min(bpps), 1e-9),
        "beta_neutral": neutral,
        "beta_unit_gain": unit,
        "beta_cache_ok": cache_ok,
        "beta_monotone": not drops,
        "beta_worst_drop": min(drops, default=0.0),
        "beta_exact": exact,
        "beta_maxerr": worst_err,
        "beta_header_b": model.header_bytes(model.compress(x, delta_beta=0)),
        "beta_sat_high": 100.0 * sat["high"],
        "beta_sat_low": 100.0 * sat["low"],
        "beta_ok": neutral and cache_ok and not drops and exact,
    }


@torch.no_grad()
def roundtrip_check(model, valid, device, *, index: int = 0) -> dict:
    """The Phase 3 gate, run mid-training. Estimated bits vs bytes actually written.

    The gap is reported **twice**, against two different estimates, because it has
    two independent causes and conflating them makes it undiagnosable:

    * `gap_pct` -- actual vs the rate the *training loss* saw, which uses the
      continuous sigma straight out of `h_s`. Measured at **+1.86%** on Kodak.
    * `gap_q_pct` -- actual vs the rate implied by the **quantised** sigma the
      coder actually indexes with, i.e. `scale_table[build_indexes(sigma)]`.
      Measured at **-0.11%**.

    That decomposition is the useful one. `gap_q_pct` near zero says the CDF
    construction and the rANS coder are correct. The residual in `gap_pct` is then
    *not a bug*: it is the cost of JPEG AI's 32-level logarithmic sigma grid
    (`entropy.sigma_quant_level`), which the differentiable model does not model
    and the codestream cannot avoid. Independently measured at +0.01521
    bits/symbol on synthetic log-uniform sigma; +1.86% of 1.07 bits/symbol here is
    +0.0199, the same effect at the same magnitude.

    Two consequences that matter for every number this project reports:

    1. RD curves must come from **actual bytes**, never from the estimate.
       Reporting the estimate would claim a codec ~1.9% better than the one that
       exists.
    2. The right acceptance threshold is on `gap_q_pct` (+-0.5%), not on
       `gap_pct`. A `gap_pct` of +2% with `gap_q_pct` at 0 is a correct codec on a
       coarse sigma grid; the plan's original "within 1-2%" would have flagged it.

    Also returns out-of-range fractions -- see :func:`out_of_range_fraction`. A
    **negative** gap always means escapes, never a lucky coder.

    Walks `model.gate_branches()` rather than reaching for
    `model.entropy_bottleneck`, so Phase 4's two-branch codec is checked on *both*
    branches. The reported `y_oor_pct`/`z_oor_pct` are the **worst** branch, not the
    average: a secondary branch escaping on 1% of its symbols while luma is clean
    must not be averaged down into looking like 0.5%. Per-branch values are also
    returned under their suffixed keys (`y_oor_uv_pct` and so on).

    Also returns a **per-stream gap** under `gap_y_pct` / `gap_z_pct` /
    `gap_y_uv_pct` / `gap_z_uv_pct`: that stream's real bytes against that stream's
    own estimate. `gap_q_pct` alone is not diagnosable, because it divides one
    stream's error by every stream's bits. The Phase 5 median-shift bug surfaced as
    an aggregate +1.85%, which is small enough to read as accumulated slop; per
    stream it was +0.01 / +0.14 / **+63%**, which names the culprit outright.

    The two stream families are measured against deliberately *different* estimates:

    * `y`, `y_uv` against the sigma-quantised estimate, for the reason above -- the
      raw forward estimate carries the +1.9% sigma-grid term on every correct codec,
      which is large enough to bury a real fault.
    * `z`, `z_uv` against the plain **forward** likelihood, which has no sigma table
      and so no grid term. This is the sensitive choice and it is the point: it spans
      all three links at once (forward -> table -> bytes), so it catches a table that
      disagrees with the density the rate loss was trained against, and not only a
      coder that disagrees with its table. The median-shift bug lived in exactly that
      first link and left the other two perfect.

    `worst_stream` is ranked by **excess bytes** (`worst_stream_b`), not by the
    percentage, and reports the percentage alongside. The percentage is the unstable
    quantity: a stream that is still near zero rate -- a random-init `y`, which
    predicts its own near-zero symbols almost perfectly -- has an estimate of a
    fraction of a bit against 8 real bytes of rANS flush, which reads as +526578%.
    That would win a percentage ranking on every run for the first few hundred steps.
    Excess bytes is stable, is what the pass/fail arm is anchored on, and is the
    quantity a reader can act on: it is what fixing the stream would save.
    """
    model.eval()
    model.update(force=True)
    x = valid[index].unsqueeze(0).to(device)

    out = model(x, noise=False, ste=True)
    y_bits, z_bits = model.estimated_bits(out)
    npix = x.shape[-1] * x.shape[-2]

    packet = model.compress(x)
    actual_bits = model.packet_bytes(packet) * 8
    dec = model.decompress(packet, device=device)

    # Compare latents, not pixels: the synthesis transform is smooth enough to
    # hide a small latent error, so a pixel comparison can pass while the coder
    # is broken. Criterion 3 of the gate is deliberately about yhat.
    y_ref = out["y_hat"]
    same_shape = dec["y_hat"].shape == y_ref.shape
    y_max = float((dec["y_hat"] - y_ref).abs().max()) if same_shape else float("inf")
    z_max = float((dec["z_hat"] - out["z_hat"]).abs().max())

    # Why each stream's gap is what it is, per branch.
    gc = model.gaussian_conditional
    per_branch: dict[str, float] = {}
    est_q_per_stream: dict[str, float] = {}
    y_bits_q = 0.0
    for suffix, eb in model.gate_branches():
        z = out[f"z{suffix}"]
        z_med = eb.medians().detach().reshape(1, -1, 1, 1).to(z.device)
        z_sym = torch.round(z - z_med).to(torch.int32)
        z_idx = eb._indexes_like(z.shape, z.device)
        per_branch[f"z_oor{suffix}_pct"] = 100.0 * out_of_range_fraction(
            eb, z_sym, z_idx)

        y = out[f"y{suffix}"]
        means = out[f"means{suffix}"]
        scales = out[f"scales{suffix}"]
        y_vals = y if means is None else y - means
        y_sym = torch.round(y_vals).to(torch.int32)
        # Ask the model which row its coder will use. Phase 5's split branch indexes
        # through the integer `Iσ`, which differs from `build_indexes(σ)` on 11 of
        # 3968 indices; measuring the gate against the wrong one would bias
        # `gap_q_pct` permanently. Models without the hook have only the float path.
        rows = getattr(model, "coder_rows", None)
        y_idx = rows(out, suffix) if rows else gc.build_indexes(scales)
        per_branch[f"y_oor{suffix}_pct"] = 100.0 * out_of_range_fraction(
            gc, y_sym, y_idx)

        # The same likelihood function, evaluated at the sigma the coder will use.
        scales_q = gc.scale_table.to(y_idx.device)[y_idx.long()]
        lik_q = gc._likelihood(y_sym.float(), scales_q,
                               None if means is None else torch.zeros(()))
        this_y_q = float(-torch.log2(lik_q.clamp_min(1e-12)).sum())
        y_bits_q += this_y_q

        # Per-stream estimates, so the aggregate gap can be attributed. An aggregate
        # alone is not diagnosable: a +1.85% total on the two-branch codec turned out
        # to be +0.01% on `y`, +0.14% on `y_uv` and **+63%** on `z_uv`, and the useful
        # information was entirely in the split. Reported against est_q for the
        # Gaussian streams and against the plain estimate for the factorised ones,
        # which have no sigma table and so no sigma-grid term.
        est_q_per_stream[f"y{suffix}"] = this_y_q
        est_q_per_stream[f"z{suffix}"] = float(
            -torch.log2(out["likelihoods"][f"z{suffix}"].clamp_min(1e-12)).sum())

    y_oor = max(v for k, v in per_branch.items() if k.startswith("y_oor"))
    z_oor = max(v for k, v in per_branch.items() if k.startswith("z_oor"))

    # Per-stream gap, when the model can tell us where its bytes went.
    stream_gap: dict[str, float] = {}
    # `-inf`, not a small negative number: the excess is **two-sided** (see below),
    # so on a run where every stream came in under its estimate a finite floor would
    # leave `worst_name` empty and the print sites would silently show nothing.
    worst_name, worst_excess, streams_ok = "", float("-inf"), True
    sb = getattr(model, "stream_bytes", None)
    if sb is not None:
        for name, nbytes in sb(packet).items():
            e = est_q_per_stream.get(name)
            if not e:
                continue
            excess = nbytes - e / 8                      # bytes
            pct = 100.0 * (nbytes * 8 - e) / e
            stream_gap[f"gap_{name}_pct"] = pct
            stream_gap[f"excess_{name}_b"] = excess
            # Two arms, because neither alone is the right shape. The floor, measured
            # over 72 stream-readings (both 3k ladders x 3 beta an octave apart x 4
            # validation images): median +5.3 B, and the full spread is **-18.3 to
            # +14.5 B**. Two properties of that spread drive the design:
            #
            #   * It is near-independent of stream size -- the +14.5 B reading is a
            #     26 KB stream and a 972 B stream costs +6.9 B. It is rANS flush plus
            #     the CDF's 16-bit quantisation, which are per-stream costs, not
            #     rates. As a *percentage* the same floor is 0.055% on that 26 KB
            #     stream and 0.71% on the 972 B one, and would be ~4% on a 200 B
            #     stream -- so a percentage-only gate false-alarms on small streams
            #     and a bytes-only gate goes blind on large ones.
            #   * It is **two-sided** for the `y` streams, which are measured against
            #     `est_q`. Quantising sigma onto the 64-entry log grid can round sigma
            #     *up*, making the estimate pessimistic and the real bytes fewer than
            #     predicted. A negative excess is therefore normal and is never a
            #     fault, which is why this arm is `> 16.0` and not `abs(...) > 16.0`.
            #
            # The margin that matters is on the *conjunction*, not on either arm. The
            # +14.5 B reading eats 91% of the byte arm but only 3% of the percentage
            # arm, so it is nowhere near firing; across all 72 readings the closest
            # approach is a 972 B `z_uv` at +6.9 B / +0.71%, which consumes 0.36x of
            # the binding arm. So there is ~2.8x of real headroom, and raising the
            # byte arm would only blind the gate on the small streams it exists to
            # protect. The median-shift bug was 854 excess bytes at +63%: 53x and 31x
            # over the two arms.
            if excess > 16.0 and pct > 2.0:
                streams_ok = False
            # Ranked by **bytes**, not by percentage, and the difference is not
            # cosmetic. Early in training a stream can carry almost no rate at all --
            # a random-init `y` predicts its own near-zero symbols almost perfectly,
            # so its estimate is a fraction of a bit and its 8 real bytes are pure
            # rANS flush. That reads as +526578%, which would win a percentage ranking
            # on every run for the first few hundred steps and bury whatever is
            # actually wrong. Excess bytes is also the quantity a reader can act on:
            # it is what fixing the stream would save.
            if excess > worst_excess:
                worst_name, worst_excess = name, excess

    est = (y_bits + z_bits) / npix
    est_q = (y_bits_q + z_bits) / npix
    act = actual_bits / npix
    mse = float(torch.mean((dec["x_hat"] - x) ** 2)) * MSE_SCALE
    model.train()
    return {
        "est_bpp": est,
        "est_q_bpp": est_q,
        "act_bpp": act,
        "gap_pct": 100.0 * (act - est) / max(est, 1e-9),
        "gap_q_pct": 100.0 * (act - est_q) / max(est_q, 1e-9),
        "sigma_grid_pct": 100.0 * (est_q - est) / max(est, 1e-9),
        "y_exact": y_max == 0.0,
        "z_exact": z_max == 0.0,
        "y_maxerr": y_max,
        "y_oor_pct": y_oor,
        "z_oor_pct": z_oor,
        **per_branch,
        **stream_gap,
        "streams_ok": streams_ok,
        "worst_stream": worst_name,
        "worst_stream_b": worst_excess if worst_name else 0.0,
        "worst_stream_pct": stream_gap.get(f"gap_{worst_name}_pct", 0.0),
        "psnr": 10 * math.log10(MSE_SCALE / max(mse, 1e-9)),
        "table_kib": model.table_bytes() / 1024,
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model, opt, aux_opt, step: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "aux_opt": aux_opt.state_dict(),
        "meta": meta,
    }, tmp)
    tmp.replace(path)          # atomic: Ctrl-C never leaves a truncated .pt


def load_checkpoint(path: Path, model, opt=None, aux_opt=None) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    # strict=False because `_cdf`/`_cdf_len`/`_offset` are entropy-table buffers
    # whose shapes change every `update()`; EntropyModel._load_from_state_dict
    # resizes them, but a checkpoint saved before any update() has none at all.
    model.load_state_dict(blob["model"], strict=False)
    if opt is not None and "opt" in blob:
        opt.load_state_dict(blob["opt"])
    if aux_opt is not None and "aux_opt" in blob:
        aux_opt.load_state_dict(blob["aux_opt"])
    return blob


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def train(args) -> int:
    cfg = load_config(args.tier)
    seed_everything(args.seed if args.seed is not None else cfg.train.seed)
    device = pick_device(args.device, verbose=True)

    model = build_any_model(cfg, args.model).to(device)

    # -- Table II, if a stage was named ------------------------------------
    # Resolved before the loss and the optimiser, because it supplies both: the stage
    # owns beta (stages III/IV of models 1-3 raise it), the loss type (stage I is pure
    # MSE), the step budget, and the freeze list.
    stage = None
    beta_override = False
    if args.stage:
        stage = find_stage(cfg, args.model_id, args.stage,
                           sample_delta_beta=not args.no_sample_delta_beta)
        # The partition invariant, checked on the real model before anything trains.
        # Freezing is the *complement* of the stage's part list, so a parameter in no
        # bucket would be frozen in every stage: it trains nowhere, the loss still
        # falls, and the only symptom is a codec quietly worse than it should be.
        check_partition(model)
        apply_freeze(model, stage.parts)
        # An explicit `--beta` still wins -- it is how the reference software's 0.012
        # for model 1 gets measured against Table II's 0.007 -- but it is announced,
        # because it also silently invalidates the eq. 9 anchor for that ladder.
        if args.beta is None:
            args.beta = stage.beta
        else:
            beta_override = args.beta != stage.beta

    criterion = loss_from_config(cfg, beta=args.beta,
                                 colour_space=args.colour_space,
                                 **(stage.loss_kwargs() if stage else {})).to(device)
    loader, valid = build_loaders(cfg, batch=args.batch, workers=args.workers,
                                 valid_limit=args.valid_images)

    # The optimiser gets only what this stage trains. Both halves of freezing are
    # needed: `apply_freeze` cleared `requires_grad` so no gradients accumulate, and
    # keeping the tensors out of Adam is what stops a momentum buffer from moving a
    # frozen weight anyway.
    aux_ids = {id(p) for p in model.aux_parameters()}
    trainable = (model.main_parameters() if stage is None else
                 [p for p in part_parameters(model, stage.parts)
                  if id(p) not in aux_ids])
    if not trainable:
        raise SystemExit(f"stage {args.stage} trains nothing on {args.model}: "
                         f"parts {stage.parts} are empty on this architecture")
    opt = torch.optim.Adam(trainable, lr=cfg.train.lr)
    # The quantiles live in the `entropy` bucket, so a stage that freezes the entropy
    # network must stop optimising them too. Letting them keep moving under a frozen
    # density would rewiden the CDF table and change the bytes a frozen entropy model
    # produces -- Table II's stage IV freezes `entropy` for every model, so this is not
    # hypothetical. The optimiser is still *built* either way, because `save_checkpoint`
    # serialises its state and a stage-IV checkpoint has to stay loadable by a stage that
    # does train the quantiles; only the backward and the step are gated, so no momentum
    # accumulates while it is off.
    aux_on = stage is None or aux_is_trained(model, stage.parts)
    aux_opt = torch.optim.Adam(model.aux_parameters(), lr=args.aux_lr)

    run = args.name or f"{args.tier}-{args.model}-beta{args.beta:g}"
    if stage is not None:
        run = args.name or (f"{args.tier}-{args.model}-m{stage.model_id}"
                            f"-stage{stage.name}")
    out_dir = CHECKPOINT_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    start = 0
    if args.warm_start:
        ws = Path(args.warm_start)
        if not ws.is_absolute():
            ws = PROJECT_ROOT / ws
        # Weights only, by name, and *not* the optimiser: a Phase 5 checkpoint's Adam
        # state has one fewer tensor per param group than a Phase 6 model does, so
        # `--resume` cannot do this job -- it would raise on the param-group length
        # before it ever got to the weights. Step counter stays at 0 too: this is a
        # new run that starts from someone else's weights, not a continuation.
        blob = torch.load(ws, map_location="cpu", weights_only=False)
        report = model.load_state_dict(blob["model"], strict=False)
        fresh = [k for k in report.missing_keys
                 if "_cdf" not in k and "_offset" not in k]
        print(f"warm-start {ws.name} "
              f"({blob.get('meta', {}).get('model', '?')} @ step "
              f"{blob.get('step', 0):,})")
        print(f"           {len(blob['model']) - len(report.unexpected_keys)} tensors "
              f"loaded, {len(fresh)} initialised fresh"
              + (f" (e.g. {fresh[0]})" if fresh else ""))
        if report.unexpected_keys:
            print(f"           {len(report.unexpected_keys)} tensors in the "
                  f"checkpoint have no home in this model, e.g. "
                  f"{report.unexpected_keys[0]}")
    if args.resume:
        ck = Path(args.resume)
        if not ck.is_absolute():
            ck = out_dir / ck
        if ck.exists():
            blob = load_checkpoint(ck, model, opt, aux_opt)
            start = int(blob.get("step", 0))
            print(f"resumed {ck.name} at step {start:,}")
        elif args.resume != "latest.pt":
            raise FileNotFoundError(ck)

    total = args.iterations or cfg.train.iterations
    if stage is not None and args.iterations is None:
        # The paper's epoch counts become *shares* of a step budget, because the dataset
        # here is not CTTC's 5264 sequences. 64:32:20:12 of `train.iterations` preserves
        # the one thing those counts encode -- stage I is half the run, stage IV a tenth.
        total = stage_steps(cfg, args.model_id, stage.name, total,
                            sample_delta_beta=not args.no_sample_delta_beta)

    print(f"\nrun      {run}")
    print(f"device   {describe_device(device)}")
    print(f"beta     {criterion.beta:g}  (== compressai lambda*255^2 "
          f"{criterion.beta * MSE_SCALE:.0f})")
    if stage is not None:
        print(f"stage    {stage.summary()}")
        if beta_override:
            print(f"         ** --beta {args.beta:g} overrides Table II's "
                  f"{stage.beta:g} for this stage; the eq. 9 anchor below is the "
                  f"config's, so a ladder built from it will be offset")
        print(f"         anchor beta {anchor_beta(cfg, args.model_id):g}  "
              f"(eq. 9 divides by stage I's beta, not this stage's)")
        n_par = sum(p.numel() for p in trainable)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"         {n_par:,} of {n_all:,} scalars trainable "
              f"({100 * n_par / max(n_all, 1):.1f}%)"
              + ("" if aux_on else "; aux optimiser off -- entropy is frozen"))
        if stage.sample_delta_beta:
            lo, hi = (int(v) for v in cfg.rate.beta_train_sample)
            print(f"         dbeta ~ U[{lo}, {hi}] per step (OURS: the paper's single "
                  f"gain vector never sees a non-zero dbeta in training)")
    print(f"steps    {start:,} -> {total:,}   batch {loader.batch_size}  "
          f"crop {cfg.train.crop}")
    print(f"out      {out_dir.relative_to(PROJECT_ROOT)}")
    print()
    print(summarise(model, crop=cfg.train.crop))
    print()

    model.train()
    step = start
    t0 = time.perf_counter()
    window: dict[str, float] = {}
    window_n = 0
    best_val = float("inf")

    stop = False
    while not stop:
        for x in loader:
            if step >= total:
                stop = True
                break

            lr = lr_at(step, total, cfg.train.lr, cfg.train.lr_final)
            for g in opt.param_groups:
                g["lr"] = lr

            x = x.to(device, non_blocking=True)
            # One integer `Delta_beta` per *batch*, not per image: the gain unit is a
            # channel-wise map applied to the whole tensor, and a per-image offset would
            # need the map broadcast per sample, which is neither what the header carries
            # nor what the coder does. Zero unless this stage trains the gain unit.
            d_beta = sample_delta_beta(cfg) if (stage and stage.sample_delta_beta) else 0
            out = model(x, delta_beta=d_beta)     # noise=training -> True here
            r = criterion(out, x)

            if not torch.isfinite(r["loss"]):
                # Stop rather than carry on: once a NaN reaches the weights every
                # subsequent step is noise, and a silently NaN'd run wastes hours.
                print(f"\nstep {step}: loss is {r['loss'].item()} -- stopping. "
                      f"bpp {float(r['bpp']):.4f} mse {float(r['mse']):.4f}")
                save_checkpoint(out_dir / "nan.pt", model, opt, aux_opt, step,
                                {"reason": "nan"})
                return 1

            opt.zero_grad(set_to_none=True)
            r["loss"].backward()
            if cfg.train.grad_clip:
                # Clipped over what this stage trains, not over every parameter. The
                # norm is a global quantity: including frozen tensors whose `.grad` is
                # None is harmless, but including *trained-elsewhere* tensors would make
                # the clip threshold mean something different in every stage.
                torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            opt.step()

            # The aux loss touches only `quantiles`, so its graph is disjoint from
            # the RD graph and the order of the two backward passes is irrelevant.
            aux = model.aux_loss()
            if aux_on:
                aux_opt.zero_grad(set_to_none=True)
                aux.backward()
                aux_opt.step()

            # `.detach()` before `float()`: the tensors still carry graph
            # references here, and torch warns about it on every window. The
            # values are identical either way -- this is only to keep a real
            # training log readable.
            for k in ("loss", "bpp", "mse", "psnr"):
                window[k] = window.get(k, 0.0) + float(r[k].detach())
            # Two-branch only. Averaged over the window like everything else, so
            # the printed chroma share is the window's share and not one step's.
            for k in ("psnr_y", "psnr_u", "psnr_v", "chroma_share"):
                if k in r:
                    window[k] = window.get(k, 0.0) + float(r[k].detach())
            window["aux"] = window.get("aux", 0.0) + float(aux.detach())
            if stage is not None and stage.sample_delta_beta:
                # Averaged like everything else, and worth watching: the mean should sit
                # near the midpoint of `beta_train_sample`, and the window's bpp should
                # visibly track it. A bpp that does not move with dbeta means the gain
                # vector is not reaching the rate -- which trains fine and looks fine.
                window["dbeta"] = window.get("dbeta", 0.0) + d_beta
            window_n += 1
            step += 1

            if step % args.log_every == 0:
                el = time.perf_counter() - t0
                m = {k: v / window_n for k, v in window.items()}
                ips = window_n / max(el, 1e-9)
                eta = (total - step) / max(ips, 1e-9)
                # The chroma columns only appear for a two-branch model, and they
                # are the two numbers that say whether 6:1:1 is doing its job: a
                # chroma share climbing while psnr_u/psnr_v stay flat means the
                # secondary branch is buying rate and not quality.
                extra = ""
                if "chroma_share" in m:
                    extra = (f"  Y/U/V {m['psnr_y']:5.2f}/{m['psnr_u']:5.2f}/"
                             f"{m['psnr_v']:5.2f}  chroma {100 * m['chroma_share']:4.1f}%")
                if "dbeta" in m:
                    extra += f"  dbeta {m['dbeta']:+7.1f}"
                print(f"{step:>7,}/{total:,}  loss {m['loss']:8.4f}  "
                      f"bpp {m['bpp']:6.4f}  psnr {m['psnr']:6.2f}  "
                      f"aux {m['aux']:8.2f}  lr {lr:.2e}  "
                      f"{ips:5.2f} it/s  eta {eta / 3600:5.2f} h{extra}", flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps({"step": step, "lr": lr, "it_s": ips,
                                        **{k: round(v, 6) for k, v in m.items()}}) + "\n")
                window, window_n, t0 = {}, 0, time.perf_counter()

            if args.rtcheck and step % args.rtcheck == 0:
                rt = roundtrip_check(model, valid, device)
                # Threshold on gap_q_pct, not gap_pct: the latter legitimately
                # carries the ~+1.9% cost of the 32-level sigma grid.
                #
                # `streams_ok` is an independent arm, not a refinement of the same
                # test. A single bad stream is diluted by every other stream's bits
                # in the aggregate, so it can sit *inside* the +-0.5% aggregate gate
                # and still be a real fault: the median-shift bug read +1.85% on a
                # two-branch model and would have read well under 0.5% on any model
                # whose bad stream was a smaller share of the total. The aggregate
                # bounds the *rate error*; `streams_ok` bounds the *per-stream
                # disagreement*, and only the second one localises.
                flag = ("ok" if abs(rt["gap_q_pct"]) < 0.5 and rt["y_exact"]
                        and rt["streams_ok"] else "**")
                yhat = "exact" if rt["y_exact"] else f"ERR {rt['y_maxerr']:.3g}"
                # `y_oor_pct` is the worst branch. Print the secondary branch's own
                # numbers too when there is one, so a chroma-only escape problem is
                # attributable rather than just visible.
                oor = f"oor y {rt['y_oor_pct']:.3f}% z {rt['z_oor_pct']:.3f}%"
                if "y_oor_uv_pct" in rt:
                    oor += (f" (uv y {rt['y_oor_uv_pct']:.3f}% "
                            f"z {rt['z_oor_uv_pct']:.3f}%)")
                # Name the worst stream unconditionally, not only when it fails. The
                # number is only interpretable against its own history, and a value
                # that appears for the first time on the run that breaks gives a
                # reader nothing to compare it to. Bytes lead; the percentage follows
                # in parentheses because it is the unstable one -- a stream that is
                # still near zero rate reports a percentage in the thousands while
                # costing 8 bytes of rANS flush.
                worst = (f"  worst {rt['worst_stream']} "
                         f"{rt['worst_stream_b']:+.0f} B "
                         f"({rt['worst_stream_pct']:+.1f}%)"
                         if rt["worst_stream"] else "")
                print(f"        rtcheck  act {rt['act_bpp']:.4f}  "
                      f"vs est {rt['gap_pct']:+6.2f}%  vs est_q {rt['gap_q_pct']:+6.2f}%"
                      f"  (sigma grid {rt['sigma_grid_pct']:+.2f}%)  {oor}{worst}  "
                      f"yhat {yhat}  psnr {rt['psnr']:5.2f}  {flag}", flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps({"step": step, "rtcheck": rt}) + "\n")
                # The rate axis, on the same cadence and the same image. Empty for a
                # fixed-rate model, so this costs nothing outside Phase 8. It is a
                # separate line rather than more columns on the one above because it
                # is a different claim: rtcheck says the coder agrees with the loss at
                # one rate, this says the *ladder* one checkpoint spans is ordered and
                # decodes. Both can pass alone and the codec still be wrong.
                db = delta_beta_check(model, valid, device)
                if db:
                    dflag = "ok" if db["beta_ok"] else "**"
                    rungs = "  ".join(f"{d:+5d} {b:.4f}" for d, b in
                                      zip(db["beta_points"], db["beta_bpps"]))
                    print(f"        dbeta    {rungs}  span {db['beta_span']:4.2f}x"
                          f"  sat {db['beta_sat_high']:.2f}/{db['beta_sat_low']:.2f}%"
                          f"  {dflag}", flush=True)
                    if not db["beta_ok"]:
                        # Say which arm, not just that something failed. The four are
                        # independent faults with different causes: a non-neutral
                        # offset is a gain-unit bug, a cache mismatch is a
                        # `precompress` bug, a rate decrease is a saturated or
                        # sign-flipped vector, and an inexact decode is the coder.
                        bad = [n for n, k in (("neutral", "beta_neutral"),
                                              ("cache", "beta_cache_ok"),
                                              ("monotone", "beta_monotone"),
                                              ("exact", "beta_exact"))
                               if not db[k]]
                        print(f"        ** dbeta fails: {', '.join(bad)}"
                              f"  (worst drop {db['beta_worst_drop']:+.4f} bpp,"
                              f" maxerr {db['beta_maxerr']:.3g})", flush=True)
                    with log_path.open("a") as f:
                        f.write(json.dumps({"step": step, "dbeta": db}) + "\n")

            if step % args.valid_every == 0:
                v = validate(model, valid, criterion, device)
                print(f"        valid    loss {v['loss']:8.4f}  "
                      f"bpp {v['bpp']:6.4f}  psnr {v['psnr']:6.2f}", flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps({"step": step, "valid": v}) + "\n")
                meta = {"tier": args.tier, "model": args.model,
                        "beta": criterion.beta, "valid": v,
                        "valid_set": getattr(valid, "roots", None)}
                if v["loss"] < best_val:
                    best_val = v["loss"]
                    save_checkpoint(out_dir / "best.pt", model, opt, aux_opt, step, meta)
                save_checkpoint(out_dir / "latest.pt", model, opt, aux_opt, step, meta)

    model.update(force=True)
    v = validate(model, valid, criterion, device)
    rt = roundtrip_check(model, valid, device)
    # The final gate walks the *published* ladder (`beta_eval_points`, nine points
    # across the clamp) rather than the three-point cadence version, because this is
    # the number a later `runbench --delta-beta` sweep will be compared against and
    # the ends are where the sigma table saturates. Empty dict for a fixed-rate
    # model, so every pre-Phase-8 run behaves exactly as before.
    db = delta_beta_check(model, valid, device,
                          points=getattr(cfg.rate, "beta_eval_points", None)
                          if hasattr(cfg, "rate") else None)
    save_checkpoint(out_dir / "final.pt", model, opt, aux_opt, step,
                    {"tier": args.tier, "model": args.model, "beta": criterion.beta,
                     "valid": v, "rtcheck": rt,
                     **({"stage": stage.name, "model_id": stage.model_id,
                         "anchor_beta": anchor_beta(cfg, stage.model_id)}
                        if stage is not None else {}),
                     **({"dbeta": db} if db else {}),
                     "valid_set": getattr(valid, "roots", None),
                     "valid_images": [valid.name(i) for i in range(len(valid))]})
    print(f"\nfinal    bpp {v['bpp']:.4f}  psnr {v['psnr']:.2f}")
    print(f"gate     actual {rt['act_bpp']:.4f} bpp")
    print(f"         vs estimate with quantised sigma: {rt['gap_q_pct']:+.2f}%"
          f"   <- the coder/table gate, target +-0.5%")
    print(f"         vs estimate the loss saw:         {rt['gap_pct']:+.2f}%"
          f"   (of which {rt['sigma_grid_pct']:+.2f}% is the 32-level sigma grid)")
    print(f"         out of table range: y {rt['y_oor_pct']:.3f}%  "
          f"z {rt['z_oor_pct']:.3f}%")
    # Every stream, every run. The aggregate above is a rate; this is the table
    # invariant, and it is the one that says *where* a regression is.
    per = sorted(k for k in rt if k.startswith("gap_")
                 and k not in ("gap_pct", "gap_q_pct"))
    if per:
        print("         per stream (bytes vs that stream's own estimate): "
              + "  ".join(f"{k[4:-4]} {rt[k]:+.2f}% ({rt['excess_' + k[4:-4] + '_b']:+.1f} B)"
                          for k in per))
        if not rt["streams_ok"]:
            print(f"         ** {rt['worst_stream']} disagrees with its own table by "
                  f"{rt['worst_stream_b']:+.0f} B ({rt['worst_stream_pct']:+.1f}%) "
                  f"-- the coder is faithful to a table that is not the density the "
                  f"rate loss was trained against")
    print(f"         yhat bit-exact through the coder: {rt['y_exact']}")
    if db:
        # The rate axis of the gate. Four independent claims, printed as four lines
        # rather than one verdict, because they fail for unrelated reasons and a
        # single "dbeta: FAIL" sends a reader to the wrong file.
        print(f"gate     Delta_beta ladder over "
              f"[{db['beta_points'][0]:+d}, {db['beta_points'][-1]:+d}]"
              f"   ({len(db['beta_points'])} points, anchor "
              f"{db['beta_anchor_bpp']:.4f} bpp)")
        for d, b in zip(db["beta_points"], db["beta_bpps"]):
            mark = "  <- anchor" if d == 0 else ""
            print(f"           {d:+5d}  {b:.4f} bpp  "
                  f"({b / max(db['beta_anchor_bpp'], 1e-9):5.2f}x){mark}")
        print(f"         span {db['beta_span']:.2f}x  "
              f"monotone {db['beta_monotone']}"
              + ("" if db["beta_monotone"]
                 else f" (worst drop {db['beta_worst_drop']:+.4f} bpp)"))
        print(f"         Delta_beta = 0 is the identity: offset neutral "
              f"{db['beta_neutral']}, scale unity {db['beta_unit_gain']}"
              f"   <- the second one is False once stage III trains the vector, "
              f"which is correct")
        print(f"         cached encode matches direct encode at every rung: "
              f"{db['beta_cache_ok']}")
        print(f"         yhat bit-exact at every rung: {db['beta_exact']}"
              + ("" if db["beta_exact"] else f"  (maxerr {db['beta_maxerr']:.3g})"))
        print(f"         sigma table saturation at the top rung: "
              f"high {db['beta_sat_high']:.2f}%  low {db['beta_sat_low']:.2f}%"
              f"   <- a few % is the ladder's real ceiling, not a bug")
        print(f"         header {db['beta_header_b']} B carries the rate; one "
              f"checkpoint, {len(db['beta_points'])} rate points")
        if not db["beta_ok"]:
            print("         ** the gain unit does not round-trip -- the ladder "
                  "above is not a rate ladder this checkpoint can actually decode")
    print(f"wrote    {(out_dir / 'final.pt').relative_to(PROJECT_ROOT)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jpegai.train.loop")
    ap.add_argument("--tier", default="tierA")
    ap.add_argument("--model", default="scale", choices=list(KINDS),
                    help="twobranch = Phase 4's primary/secondary YCbCr split; "
                         "twobranch-split = Phase 5's split hyper decoders; "
                         "twobranch-mcm = Phase 6's 4-stage context model "
                         "(-mcm2 / -mcm1 are the stage ablation); twobranch-vr = "
                         "Phase 8's variable-rate codec, split backbone plus a gain "
                         "unit per branch, which is the kind --stage expects. Use "
                         "--warm-start to inherit a Phase 5 ladder's weights")
    ap.add_argument("--beta", type=float, default=None,
                    help="distortion weight; default config.rate.base_model_beta")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu; default auto")
    ap.add_argument("--aux-lr", type=float, default=1e-3)
    ap.add_argument("--colour-space", default="ycbcr", choices=["ycbcr", "rgb"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None, help="checkpoint subdirectory")
    ap.add_argument("--resume", nargs="?", const="latest.pt", default=None)
    ap.add_argument("--warm-start", default=None, metavar="CKPT",
                    help="start from another checkpoint's weights: name-matched, "
                         "optimiser untouched, step counter at 0. This is how a "
                         "Phase 6 mcm run inherits a Phase 5 twobranch-split ladder "
                         "-- every shared parameter lands and MCM begins as the "
                         "identity, so the run opens at Phase 5's rate instead of "
                         "spending its first thousand steps rediscovering it. "
                         "--resume cannot do this: it also loads the optimiser, "
                         "whose param groups no longer match")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--valid-every", type=int, default=2000)
    ap.add_argument("--valid-images", type=int, default=8)
    ap.add_argument("--rtcheck", type=int, default=2000,
                    help="run the estimated-vs-actual gate every N steps; 0 disables")
    # -- Table II's four-stage variable-rate schedule (Phase 8) ------------------
    ap.add_argument("--stage", default=None, choices=list(STAGE_NAMES),
                    help="run one row of the variable-rate paper's Table II. The "
                         "stage supplies beta, the loss type (I is pure MSE, II-IV "
                         "are the mixed loss), the step budget as its share of the "
                         "epoch counts, and the freeze list: I/II train "
                         "encoder+decoder+entropy, III and IV train less. Omit for "
                         "an ordinary single-stage run, which is every phase before "
                         "this one. Chain stages with --warm-start, never --resume: "
                         "the optimiser's param groups change between stages by "
                         "construction")
    ap.add_argument("--model-id", type=int, default=0,
                    help="which of Table II's four rate models to follow, by its "
                         "`id` in config.rate.models -- not its position in the "
                         "list. Model 0 is the low-rate anchor and the only one "
                         "whose stage IV trains the gain unit alone; models 1-3 "
                         "raise beta for stages III/IV and train the decoder "
                         "alongside the gain unit. Only read when --stage is given")
    ap.add_argument("--no-sample-delta-beta", action="store_true",
                    help="hold Delta_beta at 0 through stages III/IV instead of "
                         "sampling config.rate.beta_train_sample. This reproduces "
                         "the reference software's literal behaviour, where the "
                         "gain unit is an 18-vector table and a fixed training beta "
                         "still exercises the interpolation. With JPEG AI's single "
                         "gain vector it means the vector never sees a non-zero "
                         "offset in training and then has to hold across the whole "
                         "clamp at evaluation, so sampling is the default. Use this "
                         "to measure what the sampling buys")
    args = ap.parse_args(argv)
    # Conditional, because `--stage` owns beta: stages III/IV of models 1-3 raise it
    # to `beta_stage34`, and filling in the config default here would silently win
    # over Table II before `train()` ever sees `None`.
    if args.beta is None and args.stage is None:
        args.beta = load_config(args.tier).rate.base_model_beta
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
