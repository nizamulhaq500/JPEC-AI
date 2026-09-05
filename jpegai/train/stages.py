"""Table II's four-stage variable-rate training schedule, as code.

    python -m jpegai.train.stages          # print all four models' schedules

The variable-rate paper's Table II ("VARIABLE RATE TRAINING STRATEGY", arXiv
2503.16288 p. 9) is four rows per model of the form *beta / loss type / epochs /
trained codec parts*. Three of those four columns are settings the training loop
already has. The fourth -- "trained codec parts" -- is the one that needs machinery,
because it is a **freeze list over submodules**, and getting it wrong is invisible:
a stage that accidentally trains the encoder still converges, still passes the
round-trip gate, and simply is not the schedule the standard specifies.

So this module is mostly about making the part list checkable:

* `PARTS` is the vocabulary, and `part_modules()` maps it onto real submodules via
  the model's own `training_parts()`.
* `check_partition()` asserts the buckets are a **partition** of the model's
  parameters. Freezing is implemented as "everything not in this stage's list", so a
  parameter belonging to no bucket would be silently frozen in every stage, and one
  in two buckets would be trained by a stage Table II says must not touch it.
* `schedule()` reads the per-model rows straight out of `config.rate.models`, so the
  paper's numbers live in the config where they can be diffed against the paper, not
  in a dict here.

Where this deviates from the paper, and why
-------------------------------------------
**Delta_beta is sampled during stages III and IV.** The paper does not do this. In the
paper, stages III/IV are trained at a single *raised* `beta_train` (0.03 / 0.2 / 1.0
against stage I's 0.007 / 0.075 / 0.5), and the rate range is covered because the
reference software's gain unit is a **table** of `qp_num = 18` vectors indexed and
interpolated by the sampled beta's position on a fixed grid (`VrqVec`,
`docs/architecture/13-training.md` section 8.4 and `_nrate_n_ft`). JPEG AI then
*simplifies* that: section III-A1 is explicit that "the gain unit in JPEG AI comprises
a single gain vector", because searching and interpolating two vectors "requires a high
computational complexity".

That simplification removes the mechanism by which a fixed training beta ever produced
a non-zero offset. With one vector and one beta, nothing in the objective sees a
non-zero `Delta_beta` at any point in training, yet at evaluation the vector is asked
to hold up across the whole `[-1069, 702]` clamp -- 5.3x down and 3.0x up in sigma.
Sampling `Delta_beta` uniformly from `config.rate.beta_train_sample` in the two stages
that train the gain unit is the cheapest way to keep it honest over that range. It is
marked OURS in the config and is switchable (`--sample-delta-beta`), so the paper's
exact schedule remains runnable.

**Stage epochs become iteration shares.** The loop counts steps, not epochs, because
the dataset here is not CTTC's 5264 sequences. `steps_for()` splits a step budget in
the paper's 64 : 32 : 20 : 12 proportion, which preserves the one thing the epoch
counts actually encode -- that stage I is half the run and stage IV is a tenth of it.

Discrepancies found against the shipped reference configuration (`cfg/train.json`),
recorded because they are the kind of thing that gets silently "corrected" later:

| Quantity | Paper Table II | `cfg/train.json` |
| --- | --- | --- |
| model 1 `beta_train` | 0.007 | **0.012** |
| stage II epochs | 32 | **36** |
| stages | four | **five** (a non-training `Data_Collection` pass) |
| MS-SSIM weight | "Mix", unquantified | 0.5 / 0.5 / 0.4 / 0.3 per model |
| Y:Cb:Cr loss weights | "prioritise luma" | 4:1:1, 4:1:1, 5:1:2, 4:1:2 |

The paper is followed here, since it is the document being implemented. The last two
rows are the interesting ones: our `train.distortion_weights` is a flat 6:1:1 and our
`train.loss.ms_ssim` a flat 0.1, and the reference software makes both *rate
dependent*. That is a Phase 12 experiment, not a silent change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: The freeze vocabulary, in the paper's words. Table II's part lists are written in
#: these terms ("encoder, decoder, entropy network", "decoder, gain unit").
PARTS = ("encoder", "decoder", "entropy", "gain")

#: The reference software's `--frozen_part` vocabulary, and the paper's own long form,
#: mapped onto ours. Accepted everywhere a part name is, so a reader holding either
#: document can pass what it says.
ALIASES = {
    "analysis": "encoder",
    "synthesis": "decoder",
    "gain_unit": "gain",
    "entropy network": "entropy",
    "entropy_network": "entropy",
}

#: Table II's loss column. Stage I is pure MSE; every later stage is the mixed
#: MSE/MS-SSIM loss. The reference software's stage IV is *named*
#: `MSE_VariableRate_12` but passes `--loss_type mix`, so the name is a misnomer and
#: the paper's "Mix" is right.
LOSSES = ("mse", "mix")

#: Stage names, in order. Roman numerals because that is what Table II calls them and
#: what the checkpoint directories are named after.
STAGE_NAMES = ("I", "II", "III", "IV")


def canonical_part(name: str) -> str:
    """A part name from any of the three vocabularies -> ours. Raises on nonsense."""
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in PARTS:
        raise ValueError(f"unknown codec part {name!r}; expected one of "
                         f"{list(PARTS)} (or an alias: {sorted(ALIASES)})")
    return key


def canonical_parts(names) -> tuple[str, ...]:
    """A part list, canonicalised, de-duplicated, and in `PARTS` order.

    Sorted into a canonical order rather than the caller's, so two spellings of the
    same stage compare equal and a printed schedule always reads the same way.
    """
    seen = {canonical_part(n) for n in names}
    return tuple(p for p in PARTS if p in seen)


# ---------------------------------------------------------------------------
# Parts -> modules -> parameters
# ---------------------------------------------------------------------------
def part_modules(model) -> dict[str, list]:
    """`{part: [(label, module)]}`, from the model's own `training_parts()`.

    Asks the model rather than classifying submodule names here. Only the model knows
    whether it has a scale decoder, a context model or a gain unit, and a name-based
    classifier in this file would go quietly wrong the first time a phase adds a
    submodule -- which is exactly the failure the partition check exists to catch, so
    it should not also be the thing doing the classifying.
    """
    fn = getattr(model, "training_parts", None)
    if fn is None:
        raise TypeError(f"{type(model).__name__} has no training_parts(); staged "
                        f"training needs the part -> module map (see "
                        f"TwoBranchCodec.training_parts)")
    parts = fn()
    missing = set(PARTS) - set(parts)
    if missing:
        raise ValueError(f"{type(model).__name__}.training_parts() omits "
                         f"{sorted(missing)}; every part must be present, empty if "
                         f"the model has none")
    return {p: list(parts[p]) for p in PARTS}


def part_parameters(model, parts) -> list:
    """Every parameter under `parts`, de-duplicated, in module order.

    De-duplicated by identity because a shared module -- a tied hyper decoder, say --
    would otherwise hand the same tensor to Adam twice, which raises rather than
    silently double-stepping, but only on some torch versions.
    """
    wanted = canonical_parts(parts)
    mods = part_modules(model)
    out, seen = [], set()
    for p in wanted:
        for _, mod in mods[p]:
            for prm in mod.parameters():
                if id(prm) not in seen:
                    seen.add(id(prm))
                    out.append(prm)
    return out


def check_partition(model) -> dict:
    """Assert `training_parts()` is a partition of `model.parameters()`.

    Returns `{part: n_parameters}` on success and raises on any of the three ways it
    can be wrong: a parameter in two buckets, a parameter in none, or a bucket naming
    a module that is not part of this model at all.

    The "in none" case is the one worth having a test for. Freezing is implemented as
    the complement of the stage's part list, so an unclaimed parameter is frozen in
    **every** stage -- it trains nowhere, the loss still falls, and the only symptom is
    a codec that is quietly worse than it should be.
    """
    mods = part_modules(model)
    owner: dict[int, str] = {}
    clash: list[str] = []
    counts: dict[str, int] = {}
    for part in PARTS:
        n = 0
        for label, mod in mods[part]:
            for name, prm in mod.named_parameters():
                prev = owner.get(id(prm))
                if prev is not None and prev != f"{part}/{label}":
                    clash.append(f"{label}.{name}: {prev} and {part}/{label}")
                owner[id(prm)] = f"{part}/{label}"
                n += 1
        counts[part] = n
    if clash:
        raise ValueError(f"{type(model).__name__}: parameters claimed by two parts: "
                         + "; ".join(sorted(clash)[:8]))

    orphans = [n for n, p in model.named_parameters() if id(p) not in owner]
    if orphans:
        raise ValueError(
            f"{type(model).__name__}: {len(orphans)} parameter(s) belong to no "
            f"training part, so no stage would ever train them: "
            f"{orphans[:8]}{' ...' if len(orphans) > 8 else ''}")

    live = {id(p) for p in model.parameters()}
    stray = sorted({v for k, v in owner.items() if k not in live})
    if stray:
        raise ValueError(f"{type(model).__name__}: training_parts() names modules "
                         f"whose parameters are not this model's: {stray[:8]}")
    return counts


def aux_is_trained(model, parts) -> bool:
    """Does this stage's part list contain the entropy bottleneck's quantiles?

    The quantiles live on the `entropy` bucket, which is where they belong: they decide
    how wide `update()` builds the CDF table, and that is entropy-network business.
    They are nonetheless optimised by a *separate* optimiser against `aux_loss()` (see
    the loop's module docstring), so the stage machinery has to answer this question
    explicitly rather than letting the aux optimiser run unconditionally.

    It must not run unconditionally. A stage that freezes `entropy` has fixed the
    density; letting the quantiles keep moving would then widen or narrow the table
    under a density that is no longer changing, which changes the *bytes* a frozen
    entropy model produces. Stage IV freezes `analysis entropy` in the reference
    software, so this is not hypothetical.
    """
    return "entropy" in canonical_parts(parts)


def apply_freeze(model, parts) -> dict:
    """Set `requires_grad` from a part list. Returns `{part: n_frozen_tensors}`.

    Both halves of freezing are done here and in the loop: `requires_grad = False`
    (this function) *and* keeping the tensors out of the optimiser (the loop, via
    `part_parameters`). The reference software does only the second and adds an
    explicit `torch.set_grad_enabled(False)` around the encoder forward when
    `analysis` is frozen; `requires_grad = False` gets the same saving without a
    special case, since a frozen module stops retaining activations for its own
    weight gradients.

    Doing only one of the two would be a bug either way round. Optimiser-only leaves
    `.grad` accumulating on frozen tensors, which costs memory and makes a later
    `--resume` into an unfrozen stage start from stale gradients. `requires_grad`-only
    leaves Adam holding tensors whose gradient is `None`, which it tolerates silently
    -- and then weight decay or a momentum buffer can still move them.
    """
    keep = set(canonical_parts(parts))
    mods = part_modules(model)
    frozen = {}
    for part in PARTS:
        n = 0
        for _, mod in mods[part]:
            for prm in mod.parameters():
                prm.requires_grad_(part in keep)
                n += 0 if part in keep else 1
        frozen[part] = n
    return frozen


# ---------------------------------------------------------------------------
# Table II
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    """One row of Table II.

    `epochs` is the paper's own number and is used only as a *share* -- see
    `steps_for`. `beta` is the stage's `beta_train`, which for models 1-3 is raised in
    stages III and IV; the eq. (9) anchor is deliberately *not* this value in those
    stages (see `anchor_beta`).
    """

    name: str
    beta: float
    loss: str
    epochs: int
    parts: tuple[str, ...]
    sample_delta_beta: bool = False
    model_id: int = 0

    def __post_init__(self):
        if self.name not in STAGE_NAMES:
            raise ValueError(f"stage name must be one of {list(STAGE_NAMES)}, "
                             f"got {self.name!r}")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {list(LOSSES)}, got {self.loss!r}")
        if not self.parts:
            raise ValueError(f"stage {self.name} trains nothing")

    @property
    def frozen(self) -> tuple[str, ...]:
        """The complement -- what `apply_freeze` will turn off."""
        return tuple(p for p in PARTS if p not in self.parts)

    def loss_kwargs(self) -> dict:
        """Overrides for `loss_from_config`, so the loss matches Table II's column.

        Only stage I needs one: it is pure MSE, so the configured MS-SSIM weight has
        to be forced to zero. Stages II-IV are "Mix" and inherit whatever
        `train.loss.ms_ssim` says, which is the knob the reference software makes
        rate-dependent (0.5 / 0.5 / 0.4 / 0.3) and we currently keep flat.
        """
        return {"ms_ssim_weight": 0.0} if self.loss == "mse" else {}

    def summary(self) -> str:
        f = f"  frozen {'+'.join(self.frozen)}" if self.frozen else ""
        d = "  sampled dbeta" if self.sample_delta_beta else ""
        return (f"stage {self.name:<3} beta {self.beta:<7g} {self.loss:<4} "
                f"{self.epochs:>3} ep  trains {'+'.join(self.parts)}{f}{d}")


def model_entry(config, model_id: int) -> dict:
    """`config.rate.models[...]` by `id`, not by list position.

    By id on purpose: the list order is a YAML detail and the ids are the paper's
    model numbers, which appear in checkpoint names and in every table this project
    reports.
    """
    for m in config.rate.models:
        if int(m["id"]) == int(model_id):
            return m
    have = [int(m["id"]) for m in config.rate.models]
    raise KeyError(f"no model with id {model_id} in config.rate.models; have {have}")


def model_ids(config) -> tuple[int, ...]:
    return tuple(int(m["id"]) for m in config.rate.models)


def anchor_beta(config, model_id: int) -> float:
    """The `beta_train` eq. (9) divides by -- **stage I's**, not stage III/IV's.

    The paper is explicit and the reason is the epoch count: "in the evaluation, the
    betatrain used to create Delta_beta is consistent with the betatrain used in
    training stage I, as this stage has the greatest number of epochs and thus the
    model is most adapted to the betatrain in this stage."

    So for model 1 the anchor is 0.007 even though stages III and IV fine-tune at
    0.03. Using the raised value here would shift every rate point on the ladder by
    `beta_displacement(0.03, 0.007) = +929` -- past the +702 clamp, so the whole ladder
    would collapse onto its top end.
    """
    return float(model_entry(config, model_id)["beta_train"])


def schedule(config, model_id: int = 0, *,
             sample_delta_beta: bool = True) -> tuple[Stage, ...]:
    """Table II's four rows for one model.

    Stages I and II are identical for every model: full network, `beta_train`, MSE then
    Mix. Stages III and IV are where the models differ, and both differences come
    straight from the config:

    * `beta_stage34` -- the raised beta for models 1-3, equal to `beta_train` for
      model 0, which "is designed to cover the low bit rate range" and therefore has
      nothing above it to adapt to.
    * `stage3` / `stage4` -- the part lists. Model 0 trains strictly fewer parts:
      `decoder+entropy` then `gain` alone, against `decoder+entropy+gain` then
      `decoder+gain` for the others.
    """
    m = model_entry(config, model_id)
    ep = config.rate.stage_epochs
    b_early = float(m["beta_train"])
    b_late = float(m["beta_stage34"])
    full = ("encoder", "decoder", "entropy")
    return (
        Stage("I", b_early, "mse", int(ep["I"]), full, False, int(m["id"])),
        Stage("II", b_early, "mix", int(ep["II"]), full, False, int(m["id"])),
        Stage("III", b_late, "mix", int(ep["III"]), canonical_parts(m["stage3"]),
              sample_delta_beta, int(m["id"])),
        Stage("IV", b_late, "mix", int(ep["IV"]), canonical_parts(m["stage4"]),
              sample_delta_beta, int(m["id"])),
    )


def find_stage(config, model_id: int, name: str, **kw) -> Stage:
    """One stage by Roman numeral, case-insensitively."""
    want = str(name).strip().upper()
    for st in schedule(config, model_id, **kw):
        if st.name == want:
            return st
    raise KeyError(f"no stage {name!r}; expected one of {list(STAGE_NAMES)}")


def steps_for(stages, total: int) -> tuple[int, ...]:
    """Split a step budget in the stages' epoch proportions, summing to `total`.

    Largest-remainder apportionment, so the parts sum to `total` **exactly**. Naive
    rounding loses or gains a few steps, and a stage IV that is 12 steps short of its
    share is not a problem -- but a `--stage IV --iterations` that disagrees with what
    the schedule said it would be is, because it makes two runs of "the same" schedule
    non-comparable.

    With the paper's 64/32/20/12 and a 400k budget: 200000 / 100000 / 62500 / 37500.
    """
    if total < 0:
        raise ValueError(f"total steps must be non-negative, got {total}")
    weights = [int(s.epochs) for s in stages]
    if min(weights, default=0) <= 0:
        raise ValueError(f"every stage needs a positive epoch count, got {weights}")
    denom = sum(weights)
    exact = [total * w / denom for w in weights]
    out = [int(math.floor(e)) for e in exact]
    for i in sorted(range(len(out)), key=lambda i: exact[i] - out[i],
                    reverse=True)[:total - sum(out)]:
        out[i] += 1
    return tuple(out)


def stage_steps(config, model_id: int, name: str, total: int, **kw) -> int:
    """The step budget for one stage out of a whole-schedule budget.

    Indexed off the schedule rather than recomputed, so this and `steps_for` cannot
    disagree about the apportionment.
    """
    stages = schedule(config, model_id, **kw)
    per = steps_for(stages, total)
    for st, n in zip(stages, per):
        if st.name == str(name).strip().upper():
            return n
    raise KeyError(f"no stage {name!r}")


# ---------------------------------------------------------------------------
# Delta_beta sampling  (OURS -- see the module docstring)
# ---------------------------------------------------------------------------
def sample_delta_beta(config, generator=None) -> int:
    """One integer `Delta_beta` from `config.rate.beta_train_sample`, inclusive.

    Integer because `Delta_beta` is a header field: a fractional value would train the
    gain vector against offsets no bitstream can request. Uniform on the *integer*
    range rather than on beta, deliberately -- eq. (10) is logarithmic, so uniform in
    `Delta_beta` is uniform in log-rate, which is the axis an RD curve is read on.

    The range is narrower than the `[-1069, 702]` clamp (config says `[-900, 600]`)
    because the clamp's last few hundred are where `Isigma + o` starts running off the
    end of the sigma table: past that the coder's sigma stops growing while the scaled
    residual does not, so the tail of the residual leaves the CDF's support and the
    gradient is measuring escapes rather than rate. `GainUnit.saturation()` is the
    diagnostic for exactly this.
    """
    lo, hi = (int(v) for v in config.rate.beta_train_sample)
    if lo > hi:
        raise ValueError(f"beta_train_sample must be [lo, hi], got [{lo}, {hi}]")
    import torch
    return int(torch.randint(lo, hi + 1, (1,), generator=generator).item())


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse

    from jpegai.config import load_config
    from jpegai.models import build_any_model

    ap = argparse.ArgumentParser(prog="python -m jpegai.train.stages")
    ap.add_argument("--tier", default="tierA")
    ap.add_argument("--model", default="twobranch-vr")
    ap.add_argument("--iterations", type=int, default=400_000)
    args = ap.parse_args(argv)

    cfg = load_config(args.tier)
    print(f"Table II, {args.tier}, {args.iterations:,} steps split "
          f"{'/'.join(str(v) for v in cfg.rate.stage_epochs.values())} by epochs\n")
    for mid in model_ids(cfg):
        sch = schedule(cfg, mid)
        per = steps_for(sch, args.iterations)
        print(f"model {mid}   anchor beta {anchor_beta(cfg, mid):g}   "
              f"(eq. 9 divides by this at every rate point)")
        for st, n in zip(sch, per):
            print(f"    {st.summary()}   {n:>7,} steps")
        print()

    model = build_any_model(cfg, args.model)
    counts = check_partition(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {sum(counts.values())} parameter tensors, "
          f"{total:,} scalars, partitioned")
    mods = part_modules(model)
    for part in PARTS:
        n = sum(p.numel() for _, m in mods[part] for p in m.parameters())
        labels = ", ".join(label for label, _ in mods[part]) or "-"
        print(f"    {part:<8} {counts[part]:>2} tensors  {n:>10,} "
              f"({100 * n / max(total, 1):5.1f}%)  {labels}")

    # Every stage of every model must leave something to train and must agree with
    # `apply_freeze`, which is the operation the loop actually performs.
    for mid in model_ids(cfg):
        for st in schedule(cfg, mid):
            trainable = part_parameters(model, st.parts)
            if not trainable:
                raise SystemExit(f"model {mid} stage {st.name} trains nothing on "
                                 f"{args.model}")
            apply_freeze(model, st.parts)
            on = {id(p) for p in model.parameters() if p.requires_grad}
            if on != {id(p) for p in trainable}:
                raise SystemExit(f"model {mid} stage {st.name}: apply_freeze and "
                                 f"part_parameters disagree")
    apply_freeze(model, PARTS)
    print("\nall stage invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
