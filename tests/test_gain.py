"""Tests for the variable-rate coding of Phase 8 — the gain unit and quality map.

Four properties carry this phase, and they are the four that would each let a broken
gain unit look healthy.

**Δβ = 0 must be bit-exact.** The gain unit is meant to bolt onto a trained Phase 5/6
checkpoint as an exact no-op, which is the whole reason Table II can train it in 12
epochs with the backbone frozen. If a zero gain vector changed even one byte, then
every rate point including the anchor would be a *different codec* than the one the
BD-rate ladder was measured on, and the anchor's numbers would silently drift.

**Rate must move monotonically with Δβ.** This is the only property a caller actually
depends on: bit-rate matching bisects on Δβ (eq. 14) and bisection on a non-monotone
function converges to nonsense. It is checked on real packet bytes, not on the
likelihood estimate, because the estimate and the coder can disagree.

**The residual is what gets gained, not the latent** (eqs. 7/8). Testing this from the
outside means checking that `y_hat` still tracks `means` as Δβ falls: gaining the
latent would pull the reconstruction toward zero, gaining the residual pulls it toward
the prediction. Those are different pictures, and only one of them is the paper's.

**The quality map is signalled by residuals, never by absolute indices** (eqs. 11/12).
The decoder rebuilds the map from `δq` alone, so an encoder/decoder disagreement about
the `/2` tie-break shows up as a wrong map and thus a wrong σ for every latent element
below and right of the first negative residual. The round trip is tested on maps that
contain negatives for exactly that reason.
"""

from __future__ import annotations

import math

import pytest
import torch

from jpegai.models.gain import (
    DELTA_BETA_MAX, DELTA_BETA_MIN, Q_INDEX_MAX, Q_INDEX_MIN, Q_SCALE_TABLE, GainUnit,
    beta_displacement, beta_ratio, clip_delta_beta, q_scales, spatial_bits,
    spatial_predict, spatial_reconstruct, spatial_residual,
)
from jpegai.models.hyper import SigmaIndex
from jpegai.models.twobranch import TwoBranchCodec

FMT_NAMES = ["444", "422", "420"]


def _codec(fmt="420", *, gain=True, split_hyper=True, mcm=False, **kw):
    """A small split-hyper codec, optionally with the gain unit attached.

    `split_hyper=True` and `mcm=False` are the defaults because the gain unit lives on
    the integer σ index: the split-hyper path is the only one that can carry it, and
    MCM refuses it. Both are overridable so the refusals can be tested too.
    """
    return TwoBranchCodec(luma_latent=32, chroma_latent=16, luma_hyper=32,
                          chroma_hyper=16, analysis_width=(16, 16, 24, 32),
                          synthesis_width=(24, 16, 16, 16),
                          internal_format=fmt, mean_scale=True,
                          split_hyper=split_hyper, mcm=mcm, gain=gain, **kw).eval()


def _unit(channels=8):
    si = SigmaIndex()
    return GainUnit(channels, log_k=si.log_k, step=si.step), si


# ---------------------------------------------------------------------------
# eq. (10): Δβ is ln δβ in units of the σ index
# ---------------------------------------------------------------------------
def test_the_papers_S_sigma_is_the_sigma_grids_own_log_step():
    """The derivation the whole phase rests on.

    The paper defines `S_σ` as "the quantization step for the entropy model" and sets
    `P_β = 2**7`. `SigmaIndex` was built in Phase 5 from the standard's σ range and
    31 intervals, with no knowledge of eq. (10) -- and its log step lands on the
    paper's stated 0.2 to within a fifth of a percent. That is not a coincidence, it
    is the same quantity: eq. (10) is a change of units into the σ index.
    """
    si = SigmaIndex()
    assert si.log_k == pytest.approx((math.log(54.82) - math.log(0.11)) / 31)
    assert si.log_k == pytest.approx(0.2, rel=0.002)
    assert si.step == 2 ** 7


@pytest.mark.parametrize("ratio", [0.2, 0.25, 0.5, 1.0, 2.0, 2.9])
def test_delta_beta_recovers_the_requested_beta_ratio(ratio):
    """Round trip eq. (9) -> eq. (10) -> back, inside the 12-bit field's reach.

    The floor in eq. (10) costs at most one Δβ step, `exp(log_k/step) - 1 = 0.16%`,
    and always downward -- the safe direction, since a rate target is a ceiling. 0.1
    is deliberately absent: it is outside the clamp, and that is a separate test.
    """
    si = SigmaIndex()
    d = beta_displacement(ratio, 1.0, log_k=si.log_k, step=si.step)
    got = beta_ratio(d, log_k=si.log_k, step=si.step)
    assert got <= ratio + 1e-12
    assert got >= ratio * (1 - 0.0016)


def test_the_clamp_is_asymmetric_in_the_direction_the_paper_says():
    """Fig. 6: "the performance decline is more rapid than when providing a lower bit
    rate", so the field reaches further down than up. One model can divide β by 5.3
    but only multiply it by 3.0 -- which is *why* four β_train anchors exist."""
    si = SigmaIndex()
    lo = beta_ratio(DELTA_BETA_MIN, log_k=si.log_k, step=si.step)
    hi = beta_ratio(DELTA_BETA_MAX, log_k=si.log_k, step=si.step)
    assert abs(DELTA_BETA_MIN) > DELTA_BETA_MAX
    assert 1 / lo > hi
    assert (lo, hi) == pytest.approx((0.1876, 3.0008), abs=1e-4)


def test_an_out_of_range_request_clamps_instead_of_overflowing_the_header():
    """The field is 12-bit signed. A 10x rate request must come back as the largest
    representable Δβ, not wrap around into a *rate reduction*."""
    si = SigmaIndex()
    assert beta_displacement(10.0, 1.0, log_k=si.log_k, step=si.step) == DELTA_BETA_MAX
    assert beta_displacement(0.01, 1.0, log_k=si.log_k, step=si.step) == DELTA_BETA_MIN
    assert clip_delta_beta(9999) == DELTA_BETA_MAX
    assert clip_delta_beta(-9999) == DELTA_BETA_MIN
    for d in (DELTA_BETA_MIN, 0, DELTA_BETA_MAX):
        assert clip_delta_beta(d) == d
    assert -2 ** 11 <= DELTA_BETA_MIN and DELTA_BETA_MAX < 2 ** 11


# ---------------------------------------------------------------------------
# the gain unit as an offset on the index
# ---------------------------------------------------------------------------
def test_an_untrained_gain_unit_is_the_exact_identity():
    """Zero vector, Δβ = 0, no map -> `m == 1` and the index is unmoved. This is what
    lets Phase 8 be a fine-tune of a Phase 5 checkpoint rather than a training run."""
    g, _ = _unit()
    o, m = g(0)
    assert torch.equal(o, torch.zeros_like(o))
    assert torch.equal(m, torch.ones_like(m))


def test_scaling_sigma_by_m_is_the_same_as_offsetting_the_index_by_o():
    """The identity that makes the whole mechanism free: because σ is exponential in
    the index, `Iσ + o` *is* `m · σ`. So the gain needs no new arithmetic at all --
    it reuses the σ grid and inherits its clamp.

    Checked strictly *inside* the clamp, and the test asserts that it stayed inside,
    because saturation breaks the identity by design: past the ends of the table σ
    stops moving while `m` does not. That regime is what `saturation()` reports.
    """
    g, si = _unit()
    with torch.no_grad():
        g.vector.normal_(0, 200).clamp_(-300, 300)
    i_sigma = torch.rand(1, 8, 4, 4) * 1200 + 1400
    top = si.levels * si.step - 1
    for d in (-800, -100, 0, 250, 600):
        o, m = g(d)
        shifted = (i_sigma + o).detach()
        assert float(shifted.min()) > 0 and float(shifted.max()) < top
        assert torch.allclose(si.sigma(shifted), m * si.sigma(i_sigma), rtol=1e-5)


def test_the_identity_breaks_only_where_the_table_ends():
    """The complement of the test above, stated so the clamp is documented behaviour
    rather than a surprise: at the top of the table `m·σ` keeps rising and σ' does
    not, which is precisely why `r' = m·r` can escape the coder's range."""
    g, si = _unit()
    top = si.levels * si.step - 1
    i_sigma = torch.full((1, 8, 2, 2), 3900.0)
    o, m = g(DELTA_BETA_MAX)
    pinned = float(si.sigma((i_sigma + o).detach()).max())
    assert pinned == pytest.approx(float(si.sigma(torch.tensor(float(top)))), rel=1e-6)
    assert float((m * si.sigma(i_sigma)).max()) > 2 * pinned


def test_the_offset_is_a_sum_so_delta_beta_and_the_map_compose():
    """A product of exponentials is a sum of exponents, which is why Fig. 4c's joint
    channel-and-spatial map needs no special case: every control adds."""
    g, _ = _unit()
    with torch.no_grad():
        g.vector.normal_(0, 100)
    q = torch.full((1, 1, 4, 4), 3, dtype=torch.long)
    o_both = g.offset(300, q)
    assert torch.allclose(o_both, g.offset(300) + (g.offset(0, q) - g.offset(0)))


@pytest.mark.parametrize("q", list(range(Q_INDEX_MIN, Q_INDEX_MAX + 1)))
def test_every_table_I_index_reproduces_its_published_scale(q):
    """Table I is the normative part of the quality map: the 17 scales are fixed. The
    offset is stored in index units, so this checks the conversion both ways."""
    g, si = _unit(channels=1)
    o = g.offset(0, torch.tensor([[[[q]]]])).detach()
    assert float(g.scale(o)) == pytest.approx(Q_SCALE_TABLE[q - Q_INDEX_MIN], rel=1e-6)
    assert float(q_scales(torch.tensor(q))) == Q_SCALE_TABLE[q - Q_INDEX_MIN]


def test_a_quality_index_outside_table_I_is_refused_not_wrapped():
    """A negative index would wrap to the far end of the table under python
    indexing -- a request for the *lowest* quality would silently get the highest."""
    g, _ = _unit()
    for bad in (Q_INDEX_MIN - 1, Q_INDEX_MAX + 1):
        with pytest.raises(ValueError, match="Table I"):
            g.offset(0, torch.full((1, 1, 2, 2), bad, dtype=torch.long))
    with pytest.raises(TypeError, match="integer"):
        g.offset(0, torch.zeros(1, 1, 2, 2))


def test_a_per_picture_delta_beta_broadcasts_across_the_batch():
    """How the variable-rate feature gets *trained*: sampling Δβ per picture is what
    stops the gain vector from overfitting the one rate the backbone was trained at.
    A shape error here would train every picture at the same rate and look fine."""
    g, _ = _unit()
    d = torch.tensor([-500.0, 0.0, 400.0]).reshape(3, 1, 1, 1)
    o, m = g(d)
    assert o.shape == (3, 8, 1, 1)
    assert float(m[0].mean()) < 1.0 < float(m[2].mean())
    assert torch.allclose(m[1], torch.ones_like(m[1]))


def test_delta_beta_is_not_a_learnable_parameter():
    """Δβ is a per-picture header field chosen at encode time. If it were a parameter
    or a buffer it would be baked into the checkpoint, and every decode would use the
    rate the *last training batch* happened to sample."""
    g, _ = _unit()
    names = dict(g.named_parameters()) | dict(g.named_buffers())
    assert "delta_beta" not in names
    assert [n for n, _ in g.named_parameters()] == ["vector"]


def test_saturation_is_reported_because_the_escape_is_silent():
    """`Iσ + o` clamps at both ends, so saturation never *desyncs* the two ends -- but
    at the top `r' = m·r` keeps growing while σ stops, which produces out-of-range
    escapes. That is the same failure the chroma table-gate bug turned out to be, so
    it gets an explicit measurement rather than a comment."""
    g, si = _unit()
    i_sigma = torch.full((1, 8, 4, 4), 3900.0)
    o, _ = g(DELTA_BETA_MAX)
    stats = g.saturation(i_sigma, o, max_index=si.levels * si.step - 1)
    assert stats["high"] == 1.0 and stats["low"] == 0.0
    o0, _ = g(0)
    assert g.saturation(i_sigma, o0, max_index=si.levels * si.step - 1)["high"] == 0.0


# ---------------------------------------------------------------------------
# eqs. (11) and (12): the spatial map's own coding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["flat", "random", "roi", "negative", "column"])
def test_the_spatial_map_survives_its_residual_coding(kind):
    """`δq -> q` must be exact. The map is coded losslessly, so `spatial_residual`
    may be vectorised over original neighbours, but `spatial_reconstruct` has to walk
    in raster order off its own output -- and those two are easy to write with
    different edge conventions.
    """
    torch.manual_seed(8)
    shape = (1, 1, 6, 9)
    if kind == "flat":
        q = torch.zeros(shape, dtype=torch.long)
    elif kind == "random":
        q = torch.randint(Q_INDEX_MIN, Q_INDEX_MAX + 1, shape)
    elif kind == "roi":
        q = torch.zeros(shape, dtype=torch.long)
        q[..., 2:4, 3:6] = 4
    elif kind == "negative":
        q = torch.full(shape, -7, dtype=torch.long)
        q[..., :, 4:] = 5
    else:
        q = torch.arange(shape[-1]).expand(shape) - 8
        q = q.contiguous()
    dq = spatial_residual(q)
    assert torch.equal(spatial_reconstruct(dq), q)


def test_the_origin_predicts_from_zero_and_the_edges_from_one_neighbour():
    """Eq. (11)'s three special cases, read straight off the paper. Getting the origin
    wrong costs one residual; getting an *edge* wrong corrupts a whole row or column
    and everything the raster scan reaches afterwards."""
    q = torch.tensor([[[[3, 5, 5], [7, 8, 8], [7, 7, 7]]]])
    dq = spatial_residual(q)
    assert int(dq[0, 0, 0, 0]) == 3                     # origin: qp = 0
    assert int(dq[0, 0, 0, 1]) == 5 - 3                 # i == 0: qp = q[i, j-1]
    assert int(dq[0, 0, 1, 0]) == 7 - 3                 # j == 0: qp = q[i-1, j]
    assert int(dq[0, 0, 1, 1]) == 8 - spatial_predict(torch.tensor(7),
                                                      torch.tensor(5))


def test_both_ends_share_one_predictor_function():
    """The paper does not state the `/2` tie-break for negative sums, so ours is a
    choice -- and the only requirement on a choice is that both ends make the same
    one. They do by construction, because there is only one function."""
    left = torch.tensor([-7, -7, -8, 3, -1])
    up = torch.tensor([-8, 8, -7, 4, -2])
    assert spatial_predict(left, up).tolist() == [-8, 0, -8, 3, -2]  # floor, not trunc


def test_a_flat_map_costs_nothing_and_a_random_one_costs_something():
    """The map is charged to the rate. A constant map -- the ordinary case, where the
    whole picture uses one quality -- must cost 0 bits, and must not report `-0.0`:
    a negative byte count in a rate table sends someone hunting for an hour."""
    flat = spatial_bits(spatial_residual(torch.zeros(1, 1, 8, 8, dtype=torch.long)))
    assert flat == 0.0 and math.copysign(1.0, flat) > 0
    torch.manual_seed(0)
    rnd = spatial_residual(torch.randint(Q_INDEX_MIN, Q_INDEX_MAX + 1, (1, 1, 8, 8)))
    assert spatial_bits(rnd) > 100


# ---------------------------------------------------------------------------
# the codec: Δβ = 0 is bit-exact, and rate is monotone in Δβ
# ---------------------------------------------------------------------------
def test_a_gained_codec_at_delta_beta_zero_is_bit_identical_to_an_ungained_one():
    """The property that makes Phase 8 a fine-tune. Same seed builds the same
    backbone; the only difference is the zero-initialised gain vector, and it must
    change *nothing* in the coded data -- not the pixels, not one string.

    The one thing it does change is the picture header: a gained bitstream always
    carries its two Δβ fields, even at the anchor, because a decoder has no other way
    to learn the rate point. That is 3 bytes, and it is charged rather than hidden --
    on a 512x768 Kodak picture at 0.5 bpp it is 0.01% of the packet.
    """
    torch.manual_seed(0)
    plain = _codec(gain=False)
    torch.manual_seed(0)
    gained = _codec(gain=True)
    assert gained.gain and not plain.gain
    for m in (plain, gained):
        m.update(force=True)
    x = torch.rand(1, 3, 64, 64)

    a, b = plain.compress(x), gained.compress(x)
    for part in ("luma", "chroma"):
        assert a[part]["y_strings"] == b[part]["y_strings"]
        assert a[part]["z_strings"] == b[part]["z_strings"]
    assert gained.packet_bytes(b) == plain.packet_bytes(a) + gained.header_bytes(b)
    assert gained.header_bytes(b) == 3
    assert torch.equal(plain.decompress(a)["x_hat"], gained.decompress(b)["x_hat"])
    assert torch.allclose(plain(x)["x_hat"], gained(x)["x_hat"], atol=0)


def test_packet_bytes_fall_with_negative_delta_beta_and_rise_with_positive():
    """The only property bit-rate matching actually needs: eq. (14) fits `log R`
    against Δβ and then *bisects*, which requires monotonicity. Checked on coded
    bytes rather than on the likelihood estimate, because a sign error in the
    residual scaling can leave the estimate monotone while the coder is not.
    """
    torch.manual_seed(0)
    m = _codec()
    m.update(force=True)
    x = torch.rand(1, 3, 64, 64)
    sizes = [m.packet_bytes(m.compress(x, delta_beta=d))
             for d in (-600, -300, 0, 300, 600)]
    assert sizes == sorted(sizes), sizes
    assert sizes[0] < sizes[2] < sizes[-1]


@pytest.mark.parametrize("delta_beta", [-500, 0, 350])
def test_the_gained_latent_survives_the_bitstream_bit_exactly(delta_beta):
    """The round trip that catches an encoder/decoder gain mismatch. `y_hat` from
    `forward` is a float path; `y_hat` from `decompress` comes off the arithmetic
    coder. They must agree exactly, or the reconstruction drifts by an amount that
    looks like ordinary quantisation noise and is not."""
    torch.manual_seed(0)
    m = _codec()
    m.update(force=True)
    x = torch.rand(1, 3, 64, 64)
    packet = m.compress(x, delta_beta=delta_beta)
    out = m.decompress(packet)
    ref = m(x, noise=False, ste=False, delta_beta=delta_beta)
    assert torch.equal(out["y_hat"], ref["y_hat"])
    assert torch.equal(out["y_uv_hat"], ref["y_uv_hat"])


def test_the_decoder_reads_the_rate_point_from_the_packet_not_from_an_argument():
    """A decoder that had to be *told* Δβ would not be a decoder. `decompress` takes
    no `delta_beta`, so the two header fields are the only channel -- and dropping
    them from the packet must change the reconstruction, proving they were used.
    """
    torch.manual_seed(0)
    m = _codec()
    m.update(force=True)
    x = torch.rand(1, 3, 64, 64)
    packet = m.compress(x, delta_beta=-500)
    assert packet["delta_beta"] == (-500, -500)
    good = m.decompress(packet)["x_hat"]
    stripped = {**packet, "delta_beta": (0, 0)}
    assert not torch.allclose(good, m.decompress(stripped)["x_hat"])


def test_the_two_components_can_be_given_different_delta_beta():
    """The paper's "usually set to the same value" is where flexible colour bit
    allocation lives: Y and UV each carry their own field."""
    torch.manual_seed(0)
    m = _codec()
    m.update(force=True)
    x = torch.rand(1, 3, 64, 64)
    packet = m.compress(x, delta_beta={"y": 0, "uv": -600})
    assert packet["delta_beta"] == (0, -600)
    plain = m.compress(x, delta_beta=0)
    y_key = ("luma", "y_strings")
    assert packet[y_key[0]][y_key[1]] == plain[y_key[0]][y_key[1]]     # Y untouched
    assert len(packet["chroma"]["y_strings"][0]) < len(plain["chroma"]["y_strings"][0])


def test_split_delta_beta_refuses_a_dict_that_names_only_one_component():
    """`{"y": 400}` is ambiguous: it could mean "leave UV alone" or "UV too". Both
    readings are defensible, which is exactly why it must not be guessed."""
    m = _codec()
    assert m.split_delta_beta(-100) == (-100, -100)
    assert m.split_delta_beta((3, 4)) == (3, 4)
    assert m.split_delta_beta({"y": 3, "uv": 4}) == (3, 4)
    with pytest.raises(KeyError, match="both"):
        m.split_delta_beta({"y": 400})
    with pytest.raises(KeyError, match="'y'/'uv'"):
        m.split_delta_beta({"luma": 400, "uv": 0})
    with pytest.raises(ValueError, match="sequence"):
        m.split_delta_beta((1, 2, 3))


def test_the_quality_map_reaches_both_branches_on_one_shared_grid():
    """Both latents are /16 of the input, so one map Q serves Y and UV with no
    resampling -- the paper's "the spatial dimensions of the primary and secondary
    components are identical in the latent space". If it only reached luma, chroma
    would be coded at the anchor quality and the ROI would show a colour edge."""
    torch.manual_seed(0)
    m = _codec()
    x = torch.rand(1, 3, 64, 64)
    q = torch.full((1, 1, 4, 4), -6, dtype=torch.long)
    base, mapped = m(x), m(x, q_index=q)
    for key in ("gain", "gain_uv"):
        assert torch.allclose(base[key], torch.ones_like(base[key]))
        assert (mapped[key] < 0.9).all(), key


def test_the_quality_map_is_carried_as_residuals_and_costs_header_bytes():
    """The map is signalled, so it is charged. `header_bytes` covers the two 12-bit
    Δβ fields plus the coded map, and `stream_bytes` must still partition the total
    -- an under-reported rate is the one error the rate gate cannot catch."""
    torch.manual_seed(0)
    m = _codec()
    m.update(force=True)
    x = torch.rand(1, 3, 64, 64)
    q = torch.randint(Q_INDEX_MIN, Q_INDEX_MAX + 1, (1, 1, 4, 4))
    packet = m.compress(x, q_index=q)
    assert torch.equal(spatial_reconstruct(packet["q_residual"]), q)
    assert m.header_bytes(packet) > 3                      # 3 for Δβ, rest is the map
    assert m.header_bytes(m.compress(x)) == 3
    sb = m.stream_bytes(packet)
    assert sb["header"] == m.header_bytes(packet)
    assert sum(sb.values()) == m.packet_bytes(packet)


def test_an_ungained_model_reports_no_header_so_the_old_invariant_holds():
    """Phase 5/6 models must keep `set(stream_bytes) == set(likelihoods)`, which the
    rate gate pairs by name. A zero-byte `header` key would break that pairing for
    every model built before this phase."""
    torch.manual_seed(0)
    m = _codec(gain=False)
    m.update(force=True)
    x = torch.rand(1, 3, 64, 64)
    packet = m.compress(x)
    assert m.header_bytes(packet) == 0
    assert set(m.stream_bytes(packet)) == set(m(x)["likelihoods"])


# ---------------------------------------------------------------------------
# what refuses, and why
# ---------------------------------------------------------------------------
def test_gain_needs_the_split_hyper_path():
    """On the mean-scale path σ is a raw network output, so there is no integer index
    for eq. (10)'s offset to live on."""
    with pytest.raises(ValueError, match="split_hyper"):
        _codec(gain=True, split_hyper=False)


def test_the_context_model_branch_refuses_a_rate_request_rather_than_dropping_it():
    """MCM quantises inside its coset loop, so the gain has to be applied there and
    the map coset-split alongside the latent. Half-implementing it would put the two
    ends on different reconstructions; accepting and ignoring Δβ would produce a
    ladder with two identical points and no error at all."""
    with pytest.raises(NotImplementedError, match="mcm: false"):
        _codec(gain=True, mcm=True)
    torch.manual_seed(0)
    m = _codec(gain=False, mcm=True)
    with pytest.raises(NotImplementedError, match="MCMBranch"):
        m(torch.rand(1, 3, 64, 64), delta_beta=-400)
    assert m(torch.rand(1, 3, 64, 64)) is not None       # Δβ = 0 stays allowed


def test_the_phase_4_branch_also_refuses_instead_of_silently_ignoring():
    """`HyperpriorBranch` accepts the kwargs only for signature compatibility."""
    torch.manual_seed(0)
    plain = _codec(gain=False, split_hyper=False)
    with pytest.raises(NotImplementedError, match="split_hyper"):
        plain(torch.rand(1, 3, 64, 64), delta_beta=200)
    with pytest.raises(NotImplementedError, match="split_hyper"):
        plain(torch.rand(1, 3, 64, 64),
              q_index=torch.zeros(1, 1, 4, 4, dtype=torch.long))
    assert plain(torch.rand(1, 3, 64, 64)) is not None


def test_gain_parameters_selects_only_the_two_vectors():
    """Table II stage IV trains the gain unit alone, backbone frozen. 256 scalars for
    Tier A -- if this returned anything else, "12 epochs" would not be enough."""
    m = _codec()
    params = m.gain_parameters()
    assert len(params) == 2
    assert [tuple(p.shape) for p in params] == [(1, 32, 1, 1), (1, 16, 1, 1)]
    assert sum(p.numel() for p in params) == 48
    assert _codec(gain=False).gain_parameters() == []


def test_the_title_says_when_the_model_can_change_rate():
    assert "+gain" in _codec().summary_title()
    assert "+gain" not in _codec(gain=False).summary_title()


# ---------------------------------------------------------------------------
# the benchmark wrapper: one checkpoint, nine rate points
# ---------------------------------------------------------------------------
# `VariableRateCodec` is how the gain unit becomes an RD curve, and it inherits an
# `__init__` written for a ladder of checkpoints. These tests are all constructor-level
# on purpose: they need no weights, they run in milliseconds, and the first one caught
# the wrapper failing on *every* real sweep with `could not convert string to float:
# 'vr'` -- a base class sorting β labels numerically, given one key spelled `"vr"`.
def _fake_checkpoint(tmp_path):
    """A file to `stat`. The wrapper's constructor never opens it."""
    p = tmp_path / "final.pt"
    p.write_bytes(b"not really a checkpoint")
    return p


def test_the_rate_points_are_the_delta_beta_rungs_not_the_checkpoint_key(tmp_path):
    from jpegai.eval.neural import VariableRateCodec

    vr = VariableRateCodec(_fake_checkpoint(tmp_path), [0, -600, 702, -600])
    assert vr.qualities == [-600, 0, 702]           # sorted, de-duplicated
    assert VariableRateCodec.LABEL not in vr.qualities


def test_the_sweep_refuses_a_point_outside_the_header_field(tmp_path):
    """Clamping would put two identical rate points on the curve, which reads as a codec
    that stopped responding to the rate request rather than as a bad argument."""
    from jpegai.eval.neural import VariableRateCodec

    ck = _fake_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="normative range"):
        VariableRateCodec(ck, [0, DELTA_BETA_MIN - 1])
    with pytest.raises(ValueError, match="normative range"):
        VariableRateCodec(ck, [DELTA_BETA_MAX + 1])
    assert VariableRateCodec(ck, [DELTA_BETA_MIN, DELTA_BETA_MAX]).qualities == \
        [DELTA_BETA_MIN, DELTA_BETA_MAX]


def test_the_cache_name_does_not_depend_on_which_rungs_were_asked_for(tmp_path):
    """The nine rungs share one set of weights, so adding a tenth must not invalidate
    the nine cached measurements that did not change."""
    from jpegai.eval.neural import VariableRateCodec

    ck = _fake_checkpoint(tmp_path)
    nine = VariableRateCodec(ck, [-1069, -800, -600, -400, -200, 0, 200, 450, 702])
    ten = VariableRateCodec(ck, [-1069, -800, -600, -400, -200, 0, 200, 450, 600, 702])
    assert nine.cache_name == ten.cache_name
    assert nine.cache_name.endswith(nine.fingerprint())


def test_the_two_kinds_of_sweep_keep_separate_caches(tmp_path):
    """`0` is a valid Δβ and `0` could be a β label, so a shared cache would cross-read
    a variable-rate rung against a trained ladder's rate point."""
    from jpegai.eval.neural import NeuralCodec, VariableRateCodec

    ck = _fake_checkpoint(tmp_path)
    fixed = NeuralCodec({"0.002": ck})
    swept = VariableRateCodec(ck, [0])
    assert fixed.cache_name != swept.cache_name
