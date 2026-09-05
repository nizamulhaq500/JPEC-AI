r"""Phase 6: MCM -- Multi-stage Context Modeling, §VI-D and Fig. 2. Luma only.

Phase 5 predicts every latent sample from `ẑ` alone: one network pass produces `p̈`
for the whole grid, and every sample is coded against it independently. That wastes
the latent's own spatial correlation -- neighbouring samples of `y` are strongly
dependent, and a prediction that has already seen half of them is a much better
prediction than one that has seen none.

MCM spends that correlation without giving up parallelism. The `/16` latent grid is
partitioned into four **cosets** by `(i mod 2, j mod 2)`, and the samples are
reconstructed in a fixed order of **stages**; within a stage every sample is
independent, so a stage is a single fully-parallel network pass, and there are four
of them **regardless of image size**.

    stage 0: (0,0)   from p̈ only
    stage 1: (1,1)   from p̈ and the reconstruction of (0,0)
    stage 2: (0,1)   from p̈ and (0,0), (1,1)
    stage 3: (1,0)   from p̈ and (0,0), (1,1), (0,1)

Where the coset order comes from
--------------------------------
`(0,0) -> (1,1) -> (0,1) -> (1,0)` -- **diagonal first** -- is not a guess. It is
`entropy.mcm_group_order` in the config, and docs/06 §5 derives it twice from the
reference software: from `ContextUtils.down_shuffle`, which returns its four parts in
the order `(part1, part4, part2, part3)`, and independently from `context.py`'s
odd-size guards, which drop the last row on stages `{1,3}` and the last column on
stages `{1,2}` and so force `dy=1` on stages 1 and 3 and `dx=1` on stages 1 and 2.
Two unrelated pieces of code, one answer.

Diagonal-first is also the order that makes the 2-stage ablation meaningful: cosets
`(0,0)` and `(1,1)` are one colour of the checkerboard and `(0,1)`/`(1,0)` are the
other, so grouping the list in consecutive pairs turns MCM into the classic
one-shot checkerboard model rather than into an arbitrary half-measure.

Why this does not break eq. (1)/(2)'s invariant
-----------------------------------------------
`SplitHyperBranch.predict` carries the rule that **nothing in the prediction may
depend on `y`**, because the decoder does not have `y`. MCM conditions on
`ŷ`, which is a different thing: the decoder *does* have the reconstruction of every
earlier stage, because it just produced it. The invariant is preserved by structure
here too -- `reconstruct()` is one loop, and the encoder and the decoder run it with
the same arguments in the same order. Neither has a path the other lacks.

The entropy coder never enters the loop
---------------------------------------
This is the part that is easy to get wrong, and getting it right is what makes
§VI-E's decoupling real. MCM refines the **mean**; `Iσ` still comes from the hyper
scale decoder and depends on `ẑ` alone. So the whole residual field `r̂` can be
entropy-decoded in **one** pass, with no network in the loop, and only then does the
4-stage reconstruction turn `r̂` into `ŷ`:

    encoder:  (y, p̈)  --[4 stages]-->  r̂, ŷ      then code r̂ in one stream
    decoder:  decode r̂ in one stream    then  (r̂, p̈) --[4 stages]--> ŷ

which is exactly the two pipelines drawn in docs/02 at lines 141 and 174. A design
that let the context model also predict σ would force the arithmetic decoder to stop
four times and wait for an accelerator, and the "self-contained entropy engine" that
§VI-E exists to build would be gone. The packet layout is therefore **unchanged**
from Phase 5: one `y` stream, one `z` stream, per branch.

The prediction arrives pre-split
--------------------------------
`HyperDecoder` ends in `conv3x3(chs, 4*chs) -> PixelShuffle(2)`, so its pre-shuffle
tensor is `[4*chs, /32]` -- and `/32` is exactly one coset's grid. MCM therefore
takes the hyper decoder **without** the shuffle and reads it as four per-coset
explicit predictions, which is what `HyperToContext9x1b` does in the reference and what
`channels.pred_primary_preshuffle: 384 = 96 * 4` in the config has been recording since
Phase 1. Nothing is upsampled and every context convolution runs at `/32`, at a quarter
of the latent's area.

Which channels belong to which coset is `PixelShuffle`'s question, not ours: coset
`(i, j)` is the strided slice `pred[:, 2i+j :: 4]`. Using the shuffle's own layout is
what makes `join_cosets(split_pred(pred))` equal `pixel_shuffle(pred, 2)` exactly --
i.e. equal to Phase 5's mean field -- so a `twobranch-split` checkpoint really does
warm-start a `twobranch-mcm` run as the identity. See :func:`split_pred`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from jpegai.models.entropy import GaussianConditional, quantize_ste
from jpegai.models.gain import reject_gain
from jpegai.models.hyper import HyperDecoder, SigmaIndex, SplitHyperBranch
from jpegai.models.layers import activation

#: `entropy.mcm_group_order`. Diagonal first; see the module docstring.
GROUP_ORDER: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (0, 1), (1, 0))


def chs2group(chs: int) -> int:
    """The `groups` argument of the MCM context convolutions.

    `contexts/MCM_phases.py` in the reference software asserts `chs % 32 == 0`
    outright and returns `max(1, chs // 32)`, which is why `primary_latent % 32 == 0`
    is a validated constraint in `jpegai.config` rather than a free width. Grouped
    convolutions are what keep a per-stage network cheap enough to run four of them:
    at Tier A's 96 channels this is 3 groups, so the `conv3x3` costs a third of a
    dense one.

    The assert is reproduced rather than softened. A silent `max(1, ...)` fallback on
    a non-multiple would give a *working* model with a different structure from the
    standard's, which is the failure mode this project is least able to detect later.
    """
    if chs % 32:
        raise ValueError(
            f"MCM requires the primary latent to be a multiple of 32, got {chs}. "
            f"This is `chs2group()`'s own assert in contexts/MCM_phases.py; see "
            f"docs/06 §2. Tier A uses 96 and the full model 160, both of which pass."
        )
    return max(1, chs // 32)


def stage_cosets(n_cosets: int, stages: int) -> list[tuple[int, ...]]:
    """Which cosets are reconstructed together, as indices into the coset order.

    `stages == n_cosets` is the standard: one coset per stage, four sequential
    passes. Fewer stages is the plan's ablation, and it is expressed here rather than
    as a separate model so that the *only* thing that changes is how much each
    context network is allowed to see:

        4 -> [(0,), (1,), (2,), (3,)]     each coset sees every earlier one
        2 -> [(0, 1), (2, 3)]             one checkerboard colour, then the other
        1 -> [(0, 1, 2, 3)]               nothing sees anything: Phase 5 + a refiner

    There is always one context network per coset. `stages` changes the *conditioning*
    and the number of sequential passes, not the parameter layout, which is what makes
    the ablation a fair comparison instead of a capacity comparison.
    """
    if stages < 1 or stages > n_cosets or n_cosets % stages:
        raise ValueError(
            f"{stages} stages does not divide {n_cosets} cosets evenly; "
            f"expected one of {[s for s in range(1, n_cosets + 1) if not n_cosets % s]}"
        )
    per = n_cosets // stages
    return [tuple(range(s * per, (s + 1) * per)) for s in range(stages)]


def split_cosets(x: Tensor, order=GROUP_ORDER) -> list[Tensor]:
    """`[C, H, W] -> len(order) x [C, H/2, W/2]`, the down-shuffle of Fig. 2.

    Pure strided indexing, so it is exact, cheap, and its own inverse -- which
    matters more than it sounds: the encoder and the decoder both round-trip through
    this pair, and an implementation that resampled or padded here would put a
    difference between the two that no correctness test on `x_hat` would see.

    Even sizes are required rather than handled. The latent is the image `/16` and the
    image is padded to `geometry.total_downsample = 64`, so the latent is always a
    multiple of 4 and this cannot fire from the codec's own path; the reference
    software's odd-size guards exist because it partitions tiles, which is Phase 12.
    Raising here keeps that a future decision instead of a silent half-coset.
    """
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        raise ValueError(
            f"MCM needs an even latent grid to split into 2x2 cosets, got {h}x{w}. "
            f"The codec pads to a multiple of 64, so a latent this shape did not "
            f"come from `compress`/`forward` -- check the caller, not this function."
        )
    return [x[..., a::2, b::2] for a, b in order]


def join_cosets(parts, order=GROUP_ORDER) -> Tensor:
    """The inverse of :func:`split_cosets`. Interleaves the cosets back onto the grid."""
    if len(parts) != len(order):
        raise ValueError(f"expected {len(order)} cosets, got {len(parts)}")
    b, c, h, w = parts[0].shape
    out = parts[0].new_zeros(b, c, 2 * h, 2 * w)
    for (i, j), p in zip(order, parts):
        out[..., i::2, j::2] = p
    return out


def split_pred(pred: Tensor, order=GROUP_ORDER) -> list[Tensor]:
    """The hyper decoder's pre-shuffle `[4*chs, /32]` tensor, cut into per-coset
    predictions **the way `PixelShuffle` would have cut it**.

    `PixelShuffle(2)` sends input channel `4c + 2i + j` to output channel `c` at
    sub-position `(i, j)`. So coset `(i, j)`'s prediction is the *strided* slice
    `pred[:, 2i+j :: 4]` -- not the contiguous block `pred[:, k*chs : (k+1)*chs]`,
    which is a different permutation of the same numbers.

    This distinction earns a named function because it is invisible to every check
    that would normally catch a wiring mistake. `chunk` returns four tensors of exactly
    the right shape, the codec round-trips bit-exactly with them, and the model trains
    to a perfectly good optimum. What it silently costs is the warm start: `h_s`'s last
    convolution was trained with the shuffle in place, so its slice `2i+j` is the
    prediction for sub-position `(i, j)` *specifically*, and handing that slice to a
    different coset permutes the mean field. A Phase 5 checkpoint would then open its
    Phase 6 run with two of its four predictions swapped rather than as the identity --
    exactly the head start the near-identity init exists to preserve.

    With this slicing `join_cosets(split_pred(pred)) == pixel_shuffle(pred, 2)`
    exactly, i.e. the Phase 5 mean field, which `tests/test_mcm.py` asserts.

    The reference software chunks *contiguously* and is right to: it never uses
    `PixelShuffle`, so its last convolution's channel `k*chs + c` is coset `k`'s
    channel `c` by construction. Both layouts are self-consistent and neither is
    normative -- what is not free is mixing them. docs/06 §5.1.
    """
    n = len(order)
    if n != 4:
        raise ValueError(f"the pre-shuffle layout is defined for a 2x2 tile, got {n}")
    if pred.shape[-3] % n:
        raise ValueError(f"{pred.shape[-3]} channels is not a multiple of {n}")
    return [pred[..., 2 * i + j::n, :, :] for i, j in order]


class ContextStage(nn.Module):
    """One coset's context network: `(explicit prediction, earlier cosets) -> mean`.

    Structure follows `contexts/MCM_phases.py` as recorded in docs/06 §5 -- phase 0 is
    a fusion network over the explicit prediction alone; phases 1/2/3 first collapse
    the concatenation of every previously reconstructed coset with a `conv1x1(k*chs,
    chs)` and then mix it spatially with a grouped `conv3x3`, and fuse that with the
    explicit prediction.

    The `conv1x1` before the `conv3x3` is the load-bearing ordering. `k` cosets
    concatenated is `k*chs` channels, and a `3x3` straight onto that would cost `k`
    times as much as the projection does; collapsing first makes the per-stage cost
    almost independent of which stage it is, which is what lets the paper claim four
    stages at a fixed complexity rather than a cost that grows down the chain.

    Predicting a *correction* to `p̈`, not a mean from scratch
    ---------------------------------------------------------
    `forward` returns `pred + fuse(...)` and `fuse`'s last layer is initialised at
    1% of its default scale, so at initialisation `ctx ≈ pred` to three digits and an
    untrained MCM model codes what Phase 5's model codes. That is deliberate, for
    three reasons and the third is the one that pays:

    * A from-scratch mean starts at zero, so the first thousand steps code `r̂ ≈ y`
      against a σ that was chosen for a residual. That is the same rate explosion
      `HyperScaleDecoder.init_index` exists to avoid, arriving by another route.
    * A Phase 5 `twobranch-split` checkpoint warm-starts a `twobranch-mcm` run
      exactly: `h_a`, `h_s`, `h_scale` and the `z` prior all keep their names and
      shapes, and MCM begins as the identity on top of them.
    * It makes the ablation honest. "MCM gives 4-9% BD-rate over Phase 5" is a claim
      about what the context adds, and it is only that claim if the 1-stage limiting
      case *is* Phase 5 rather than a differently-initialised model that happens to
      be nearby.

    Scaled-down rather than exactly zero: an exact zero on the last layer leaves every
    layer above it with zero gradient on the first step, and `selftest`'s "every
    parameter receives a finite non-zero gradient" check would fail -- correctly, since
    a reader cannot tell that state apart from a genuinely disconnected network.
    """

    def __init__(self, chs: int, n_earlier: int, *,
                 activation_name: str = "relu", init_gain: float = 0.01):
        super().__init__()
        c = int(chs)
        self.n_earlier = int(n_earlier)
        if self.n_earlier:
            self.gather = nn.Sequential(
                nn.Conv2d(self.n_earlier * c, c, 1),
                activation(activation_name, c),
                nn.Conv2d(c, c, 3, padding=1, groups=chs2group(c)),
            )
        else:
            # Stage 0 has nothing to gather. Registered as None rather than as an
            # identity so that `print(model)` shows the asymmetry, which is the whole
            # shape of the model: the first stage is the one MCM cannot help.
            self.gather = None
        self.fuse = nn.Sequential(
            nn.Conv2d(c if self.gather is None else 2 * c, c, 3, padding=1),
            activation(activation_name, c),
            nn.Conv2d(c, c, 3, padding=1),
        )
        with torch.no_grad():
            self.fuse[-1].weight.mul_(float(init_gain))
            self.fuse[-1].bias.zero_()

    def forward(self, pred: Tensor, earlier: list[Tensor]) -> Tensor:
        if self.gather is None:
            h = pred
        else:
            if len(earlier) != self.n_earlier:
                raise ValueError(
                    f"this stage was built to see {self.n_earlier} earlier cosets "
                    f"and was handed {len(earlier)}; the stage schedule and the "
                    f"reconstruction loop have gone out of step"
                )
            h = torch.cat([pred, self.gather(torch.cat(earlier, dim=-3))], dim=-3)
        return pred + self.fuse(h)


class MultiStageContextModel(nn.Module):
    """The 4-stage loop, written **once** and called by both sides of the codec.

    This class exists mainly to make the encoder/decoder symmetry structural rather
    than a rule someone has to remember. Stage `k`'s prediction depends on stages
    `< k`, so an encoder that computed the four contexts in any way other than the
    decoder's own loop would produce residuals the decoder cannot undo -- and the
    symptom is not a crash. It is a reconstruction that drifts a little more with
    every stage, on a codestream that decodes without complaint. So there is one
    :meth:`reconstruct`, it takes either `y` or `r_hat` and never both, and the two
    directions differ by exactly one line inside it.
    """

    def __init__(self, chs: int, *, stages: int = 4, order=GROUP_ORDER,
                 activation_name: str = "relu"):
        super().__init__()
        self.chs = int(chs)
        self.order = tuple(tuple(int(v) for v in g) for g in order)
        if sorted(self.order) != sorted({(i, j) for i in (0, 1) for j in (0, 1)}):
            raise ValueError(
                f"the coset order must cover the 2x2 tile exactly once, got "
                f"{self.order}"
            )
        self.schedule = stage_cosets(len(self.order), int(stages))
        #: How many cosets each coset may condition on: everything in earlier stages.
        self.visible = [0] * len(self.order)
        seen = 0
        for cosets in self.schedule:
            for c in cosets:
                self.visible[c] = seen
            seen += len(cosets)
        # One network per coset, indexed by position in `order` -- so `nets[k]`,
        # `split_pred(pred)[k]` and `order[k]` all mean the same coset, and the hyper
        # decoder's slice for that sub-position is that coset's prediction.
        self.nets = nn.ModuleList([
            ContextStage(chs, self.visible[c], activation_name=activation_name)
            for c in range(len(self.order))
        ])

    @property
    def stages(self) -> int:
        """Sequential network passes per image -- **constant**, not a function of size."""
        return len(self.schedule)

    def reconstruct(self, pred: Tensor, *, y: Tensor | None = None,
                    r_hat: Tensor | None = None, ste: bool = True) -> dict:
        """`{means, y_hat, r_hat}` from the pre-shuffle prediction and one of `y`/`r_hat`.

        `pred` is the hyper decoder's `[4*chs, /32]` output, cut into per-coset
        predictions by :func:`split_pred` -- strided, the way `PixelShuffle` would,
        so that at initialisation the assembled mean field is exactly Phase 5's.

        Pass `y` to run the **encoder** direction: each stage quantises its own
        residual, which is what makes the next stage's input the value the decoder
        will actually have rather than the value the encoder wishes it had. Pass
        `r_hat` to run the **decoder** direction, where the residuals came off the
        bitstream. Exactly one, and it is checked: defaulting to one of them would
        turn "I forgot an argument" into a silent encoder-only reconstruction.

        `ste` follows the rest of the project -- straight-through during training so
        the gradient reaches `g_a`, plain rounding at inference. Both round the same
        way, so switching it cannot move a bit of the bitstream.
        """
        if (y is None) == (r_hat is None):
            raise ValueError("reconstruct() takes exactly one of y= (encoder) or "
                             "r_hat= (decoder)")
        if pred.shape[-3] != self.chs * len(self.order):
            raise ValueError(
                f"expected the pre-shuffle prediction with "
                f"{self.chs * len(self.order)} channels ({len(self.order)} cosets x "
                f"{self.chs}), got {pred.shape[-3]}. A `[chs, /16]` tensor here means "
                f"the hyper decoder was built with shuffle=True."
            )
        pred_parts = split_pred(pred, self.order)
        src = split_cosets(y if r_hat is None else r_hat, self.order)

        n = len(self.order)
        ctx: list[Tensor | None] = [None] * n
        rec: list[Tensor | None] = [None] * n
        res: list[Tensor | None] = [None] * n
        done: list[int] = []
        for cosets in self.schedule:
            # Snapshot the history once per stage, before this stage writes any of its
            # own reconstructions. Every coset in a stage therefore sees the same
            # inputs, which is what "all samples within a stage are independent"
            # means -- and it is why a stage is one parallel pass and not `len(cosets)`
            # dependent ones.
            earlier = [rec[i] for i in done]
            for c in cosets:
                k = self.nets[c](pred_parts[c], earlier)
                if r_hat is None:
                    v = src[c] - k
                    r = quantize_ste(v) if ste else torch.round(v)
                else:
                    r = src[c]
                ctx[c], res[c], rec[c] = k, r, r + k
            done.extend(cosets)
        return {"means": join_cosets(ctx, self.order),
                "y_hat": join_cosets(rec, self.order),
                "r_hat": join_cosets(res, self.order)}

    def extra_repr(self) -> str:
        sched = " -> ".join("+".join(f"{self.order[c]}" for c in st)
                            for st in self.schedule)
        return f"chs={self.chs}, stages={self.stages}, {sched}"


def _no_gain(delta_beta, q_index) -> None:
    """Refuse a quality map on the context-model branch rather than ignore one.

    The three overrides below accept `delta_beta`/`q_index` so that this branch stays
    signature-compatible with `SplitHyperBranch` -- callers should not have to know
    which branch they hold. See `MCMBranch.__init__` for why it cannot be honoured.
    """
    reject_gain(delta_beta, q_index, "MCMBranch",
                "The coset loop quantises internally, so the gain has to be applied "
                "inside it and a spatial map coset-split alongside the latent; see "
                "MCMBranch.__init__. Variable rate runs on `mcm: false` with "
                "`gain: true`.")



class MCMBranch(SplitHyperBranch):
    """Phase 5's branch with the context model in front of the mean. Drop-in.

    Same `forward`/`compress`/`decompress` contract as `SplitHyperBranch` and the
    same `h_a`/`h_s`/`h_scale`/`entropy_bottleneck` parameter names, so a Phase 5
    checkpoint's weights load straight into one of these and the training loop, the
    loss, the rate gate and the benchmark need no changes at all.

    Two things differ, and only these two:

    * `h_s` is built with `shuffle=False`, so its output is the `[4*chs, /32]`
      pre-shuffle tensor whose four slices are the per-coset explicit predictions.
      The parameter shapes are **identical** either way -- `PixelShuffle` has no
      weights -- which is what makes the warm start work.
    * `predict()` no longer returns a `means`, because there isn't one yet: the mean
      is what the 4-stage loop produces, and it needs the residuals. It returns
      `pred` instead, and `means` is present but `None` so that a caller reaching
      for it gets an obvious `None` rather than a stale prediction from `p̈`.
    """

    def __init__(self, latent: int, hyper: int, *, sigma_index: SigmaIndex,
                 stages: int = 4, order=GROUP_ORDER, scale_layers: int = 2,
                 activation_name: str = "relu", precision: int = 16,
                 gain: bool = False, scaler_precision: int = 10):
        if gain:
            raise NotImplementedError(
                "the Phase 8 gain unit is not wired into the context model. It is not "
                "a matter of threading an argument: `MultiStageContextModel."
                "reconstruct` quantises each coset internally, so the gain has to be "
                "applied *inside* that loop (encoder: round(m*(y - ctx)); decoder: "
                "ctx + r_hat/m) and a spatial quality map has to be coset-split "
                "alongside the latent. Doing it by halves would put the encoder and "
                "decoder on different reconstructions, which is exactly the class of "
                "bug this branch's `means=None` invariant exists to prevent. Variable "
                "rate runs on the split-hyper line: `mcm: false` with `gain: true`."
            )
        super().__init__(latent, hyper, sigma_index=sigma_index, fused=False,
                         scale_layers=scale_layers,
                         activation_name=activation_name, precision=precision)
        self.h_s = HyperDecoder(latent, activation_name=activation_name,
                                shuffle=False)
        self.mcm = MultiStageContextModel(latent, stages=stages, order=order,
                                          activation_name=activation_name)

    # -- prediction ---------------------------------------------------------
    def predict(self, z_hat: Tensor, *, quantise: bool = False) -> dict:
        """`{pred, means: None, i_sigma, scales, rows}` from `zhat` alone.

        `means` is `None` on purpose and is not an oversight to tidy away. On this
        branch the mean of a sample is only defined once the samples of every earlier
        stage have been reconstructed, so any code that wants one has to go through
        :meth:`MultiStageContextModel.reconstruct` and therefore has to say whether it
        is the encoder or the decoder. That is the invariant of this whole phase, and
        a `None` here is what enforces it.

        `scales` and `rows` are unchanged from Phase 5 and still depend on `zhat`
        alone -- σ is not context-modelled. See the module docstring: that is what
        keeps the entropy decoder a single self-contained pass.
        """
        pred = self.h_s(z_hat)
        i_sigma = self.h_scale(z_hat)
        if not quantise:
            return {"pred": pred, "means": None, "i_sigma": i_sigma,
                    "scales": self.sigma_index.sigma(i_sigma), "rows": None}
        i_int = self.sigma_index.quantise(i_sigma)
        return {"pred": pred, "means": None, "i_sigma": i_int,
                "scales": self.sigma_index.sigma(i_int.float()),
                "rows": self.sigma_index.table_row(i_int)}

    # -- training -----------------------------------------------------------
    def forward(self, y: Tensor, gc: GaussianConditional, *,
                noise: bool | None = None, ste: bool = True,
                delta_beta: int | float | Tensor = 0,
                q_index: Tensor | None = None) -> dict:
        _no_gain(delta_beta, q_index)
        z = self.h_a(y)
        z_hat, z_lik = self.entropy_bottleneck(z, noise=noise, ste=ste)
        p = self.predict(z_hat)
        mcm = self.mcm.reconstruct(p["pred"], y=y, ste=ste)
        # `gc` is still the thing that produces `y_hat` and the likelihood, given the
        # mean the loop arrived at. Its `hat` is `mean + quantise(y - mean)`, which is
        # the loop's own `y_hat` recomputed -- `tests/test_mcm.py` pins that they
        # agree, so the rate the loss sees and the latent the synthesis transform
        # sees cannot drift apart without a test failing.
        y_hat, y_lik = gc(y, p["scales"], mcm["means"], noise=noise, ste=ste)
        return {"y_hat": y_hat, "y_lik": y_lik, "z_lik": z_lik,
                "z": z, "z_hat": z_hat,
                "scales": p["scales"], "means": mcm["means"],
                "i_sigma": p["i_sigma"],
                # The loop's own outputs, for the gate and for the ablation: `r_hat`
                # is what the coder writes, and its statistics are the only direct
                # evidence that the context model is doing anything.
                "mcm_y_hat": mcm["y_hat"], "r_hat": mcm["r_hat"]}

    # -- real bitstream -----------------------------------------------------
    @torch.no_grad()
    def code_cached(self, pre: dict, gc: GaussianConditional, *,
                    delta_beta: int | float | Tensor = 0,
                    q_index: Tensor | None = None) -> dict:
        """Refused, not inherited.

        `SplitHyperBranch.code_cached` codes `y` against `means` from the scale
        decoder alone, which for this branch is only the *hyper* part of the context.
        Inheriting it would silently drop the context model -- the bitstream would
        decode to a different picture, and the rate would look plausible. Since Fig.
        9's cache exists to serve the Δβ search, and this branch has no Δβ, there is
        nothing to inherit it for.
        """
        raise NotImplementedError(
            "MCMBranch has no cached-encode path: the coset loop is part of the "
            "encode, so there is no 'everything before the gain unit' to cache. Rate "
            "search runs on `mcm: false` with `gain: true`; see MCMBranch.__init__."
        )

    @torch.no_grad()
    def compress(self, y: Tensor, gc: GaussianConditional, *,
                 delta_beta: int | float | Tensor = 0,
                 q_index: Tensor | None = None) -> dict:
        _no_gain(delta_beta, q_index)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        # From the decoded z_hat, never from z -- the decoder has only the former.
        z_hat = self.entropy_bottleneck.decompress(
            z_strings, tuple(z.shape[-2:]), device=z.device)
        p = self.predict(z_hat, quantise=True)
        mcm = self.mcm.reconstruct(p["pred"], y=y, ste=False)
        # One stream, exactly as in Phase 5. `round(y - means)` inside `gc.compress`
        # reproduces the residual the loop already quantised coset by coset, because
        # `means` is that loop's own context assembled back onto the grid.
        return {"y_strings": gc.compress(y, p["scales"], mcm["means"],
                                         indexes=p["rows"]),
                "z_strings": z_strings, "z_shape": tuple(z.shape[-2:])}

    @torch.no_grad()
    def decompress(self, part: dict, gc: GaussianConditional, device, *,
                   delta_beta: int | float | Tensor = 0,
                   q_index: Tensor | None = None) -> dict:
        _no_gain(delta_beta, q_index)
        z_hat = self.entropy_bottleneck.decompress(
            part["z_strings"], tuple(part["z_shape"]), device=device)
        p = self.predict(z_hat, quantise=True)
        # `means=None`: this call returns the *residual field*, and it returns all of
        # it in one pass with no network in the loop. That is §VI-E's decoupling in
        # one line -- the entropy engine never waits for the accelerator.
        r_hat = gc.decompress(part["y_strings"], p["scales"], None,
                              indexes=p["rows"])
        mcm = self.mcm.reconstruct(p["pred"], r_hat=r_hat, ste=False)
        return {"y_hat": mcm["y_hat"], "z_hat": z_hat, "r_hat": r_hat}
