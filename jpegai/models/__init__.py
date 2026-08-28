"""Model factory.

One function, `build_any_model`, so that the training loop, the evaluator and the
selftest all resolve a `--model` string the same way. It lives here rather than in
`hyperprior` because `twobranch` imports `hyperprior` -- putting the factory in
`hyperprior` would close the cycle.

The string is also what gets written into a checkpoint's `meta["model"]`, which is
how `jpegai.eval.neural` reconstructs the right architecture from a `.pt` file
months later. Renaming one of these breaks every existing checkpoint, so the names
are treated as part of the on-disk format.
"""

from __future__ import annotations

KINDS = ("scale", "mean-scale", "twobranch", "twobranch-split", "twobranch-fused",
         "twobranch-mcm", "twobranch-mcm2", "twobranch-mcm1")

#: Phase 6's default. `twobranch` and `twobranch-split` stay what they were so
#: Phase 4's and Phase 5's checkpoints keep reconstructing the architecture they were
#: trained with -- these strings are on-disk format, and silently redefining one turns
#: an old `.pt` into wrong weights rather than into a load error.
DEFAULT_KIND = "twobranch-mcm"

#: `kind -> MCM stage count`. The trailing digit is in the *name* rather than a
#: separate flag because `meta["model"]` is the only thing `jpegai.eval.neural` has
#: when it rebuilds an architecture from a checkpoint months later; a stage count kept
#: outside the string would load a 4-stage model from 2-stage weights and the shapes
#: happen to fit, because the ablation deliberately changes only the conditioning.
MCM_STAGES = {"twobranch-mcm": 4, "twobranch-mcm2": 2, "twobranch-mcm1": 1}


def build_any_model(config, kind: str = "scale"):
    """Build `scale`/`mean-scale` (Phase 3), `twobranch` (Phase 4), one of the two
    Phase 5 split-hyper variants, or Phase 6's context-modelled codec.

    * `twobranch-split` -- separate hyper decoder and hyper scale decoder (§VI-E).
    * `twobranch-fused` -- the `--single-hyper-decoder` ablation: one network emits
      both, Balle style. Phase 13 reports the RD delta between the two, which is a
      number the paper argues for on complexity grounds but never publishes.
    * `twobranch-mcm` -- §VI-D's 4-stage checkerboard context model on the luma
      branch, on top of `twobranch-split`. `-mcm2` and `-mcm1` are the plan's stage
      ablation: same networks, less history, fewer sequential passes. `-mcm1` sees no
      history at all and is therefore `twobranch-split` plus a prediction refiner,
      which is what makes it the right zero point to measure MCM's gain against.
    """
    if kind in MCM_STAGES:
        from jpegai.models.twobranch import build_two_branch
        return build_two_branch(config, mean_scale=True, split_hyper=True,
                                mcm=True, mcm_stages=MCM_STAGES[kind])
    if kind in ("twobranch", "twobranch-split", "twobranch-fused"):
        from jpegai.models.twobranch import build_two_branch
        # mean_scale is not optional here: the secondary branch's whole reason for
        # existing is that chroma is predictable from luma, and a scale-only prior
        # cannot express "this chroma latent is probably near this value".
        return build_two_branch(config, mean_scale=True,
                                split_hyper=kind != "twobranch",
                                fused_hyper=kind == "twobranch-fused")
    if kind in ("scale", "mean-scale"):
        from jpegai.models.hyperprior import build_model
        return build_model(config, kind=kind)
    raise ValueError(f"unknown model kind {kind!r}; expected one of {KINDS}")


def is_two_branch(model) -> bool:
    """True for Phase 4's codec. Checked by capability, not by isinstance.

    `gate_branches()` returning more than one entry is exactly the property callers
    care about -- Phase 5 onward will add more multi-branch models and none of them
    should have to be enumerated here.
    """
    return len(model.gate_branches()) > 1
