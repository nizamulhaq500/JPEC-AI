"""Tests for the rate-ladder driver's reporting and warm-start plumbing.

The ladder's job is mostly to call the training loop five times, which no unit test
can meaningfully check. What *is* worth testing is the reporting, because it consists
of two warnings that exist to catch mistakes I have already made once:

* **non-monotone rate** — if bpp does not increase with β, a point is undertrained,
  and BD-rate integrated over a non-monotone curve is meaningless rather than merely
  imprecise.
* **mixed validation sets** — `build_loaders` silently falls back to the test set
  when the validation directory is empty, so recovering DIV2K_valid part-way through
  a ladder changes the yardstick between points. That happened during development;
  the bpp column looked fine and was comparing different pictures.

A warning that has quietly stopped firing is worse than no warning, so both are
pinned here.
"""

from __future__ import annotations

import torch

from jpegai.train.runladder import (
    DEFAULT_BETAS, _copy_weights_only, _cross_ladder_seed, _label, _print_summary,
)

VSET = ["data/div2k/DIV2K_valid_HR"]


def _row(beta, bpp, *, psnr=30.0, gap_q=0.1, vset=None, act=None):
    return {
        "beta": beta, "lambda255": beta * 255 ** 2, "step": 50000,
        "valid_bpp": bpp, "valid_psnr": psnr,
        "valid_set": VSET if vset is None else vset,
        "act_bpp": bpp * 1.02 if act is None else act,
        "gap_q_pct": gap_q, "gap_pct": gap_q + 1.9,
        "y_oor_pct": 0.0, "y_exact": True, "path": f"beta{beta:g}/final.pt",
    }


def test_labels_round_trip_through_float():
    """Directory names are parsed back into β by `NeuralCodec.from_directory`."""
    for b in DEFAULT_BETAS + [0.0002, 3.0, 1.0]:
        assert float(_label(b)) == b
    assert "/" not in _label(0.0002) and " " not in _label(0.0002)


def test_default_betas_are_ascending_and_span_the_useful_range():
    assert DEFAULT_BETAS == sorted(DEFAULT_BETAS)          # warm start goes upward
    assert len(DEFAULT_BETAS) >= 4                         # BD-rate needs 4 points
    assert 0.002 in DEFAULT_BETAS                          # JPEG AI's lowest base model


def test_monotone_ladder_reports_the_gate_and_no_warning(capsys):
    _print_summary([_row(b, bpp) for b, bpp in
                    [(0.002, 0.12), (0.012, 0.31), (0.075, 0.78), (0.2, 1.24)]])
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "quantised-sigma estimate" in out
    assert "DIV2K_valid_HR" in out                         # says what it validated on


def test_non_monotone_rate_warns(capsys):
    _print_summary([_row(0.002, 0.12), _row(0.012, 0.55), _row(0.075, 0.41)])
    out = capsys.readouterr().out
    assert "NOT monotone" in out
    assert "undertrained" in out


def test_monotonicity_uses_the_validation_average_not_the_single_image(capsys):
    """`act_bpp` is one image; a warning must not hinge on one picture.

    Here the 8-image validation average is properly ordered while the single-image
    actual bpp is not -- which is normal, two adjacent β can order differently on
    one photograph. Warning on that would cry wolf on a correct ladder.
    """
    rows = [_row(0.002, 0.12, act=0.40), _row(0.012, 0.31, act=0.38)]
    _print_summary(rows)
    assert "NOT monotone" not in capsys.readouterr().out


def test_mixed_validation_sets_warn(capsys):
    _print_summary([_row(0.002, 0.12, vset=["data/kodak"]),
                    _row(0.012, 0.31)])
    out = capsys.readouterr().out
    assert "DIFFERENT image sets" in out
    assert "not comparable" in out
    assert "data/kodak" in out and "DIV2K_valid_HR" in out


def test_failed_coder_gate_warns_and_names_the_point(capsys):
    _print_summary([_row(0.002, 0.12), _row(0.012, 0.31, gap_q=-4.2)])
    out = capsys.readouterr().out
    assert "gate failed" in out
    assert "-4.20%" in out


def test_summary_survives_a_checkpoint_with_no_rtcheck(capsys):
    """A `latest.pt` has `valid` but no `rtcheck`; the table must still print."""
    row = _row(0.002, 0.12)
    for k in ("act_bpp", "gap_q_pct", "y_oor_pct", "y_exact"):
        row[k] = None
    _print_summary([row])
    out = capsys.readouterr().out
    assert "--" in out                                     # placeholders, not a crash
    assert "0.002" in out


def test_warm_start_drops_optimiser_state_and_the_step_counter(tmp_path):
    """Carrying Adam moments or the step count across a β change is subtly wrong.

    The step count would skip LR warmup, and the second-moment estimates were
    accumulated against a different objective. Both degrade the model rather than
    raise, so this is checked structurally.
    """
    src = tmp_path / "final.pt"
    torch.save({"step": 50000,
                "model": {"g_a.0.weight": torch.ones(2, 2)},
                "opt": {"state": {"anything": 1}},
                "aux_opt": {"state": {"anything": 2}},
                "meta": {"beta": 0.002}}, src)

    dst = tmp_path / "warmstart.pt"
    _copy_weights_only(src, dst)
    blob = torch.load(dst, map_location="cpu", weights_only=False)

    assert blob["step"] == 0
    assert "opt" not in blob and "aux_opt" not in blob
    assert torch.equal(blob["model"]["g_a.0.weight"], torch.ones(2, 2))
    assert str(src) in blob["meta"]["warm_start_from"]     # provenance is kept


# --- the cross-ladder seed -------------------------------------------------------
#
# Ladder #3 (`twobranch-mcm`) is seeded from ladder #1 (`twobranch-split`) at matching
# β. The pairing is by directory name -- `beta0.03/final.pt` -- so it is exactly the
# kind of thing that breaks silently when `_label`'s formatting and the lookup drift
# apart. A miss must not be fatal either: a Phase 6 ladder with one extra β should
# train that point from the intra-ladder seed rather than refuse to start.


def test_the_same_beta_of_the_other_ladder_is_found(tmp_path):
    for b in DEFAULT_BETAS:
        p = tmp_path / f"beta{_label(b)}"
        p.mkdir()
        (p / "final.pt").write_bytes(b"")
    for b in DEFAULT_BETAS:
        seed = _cross_ladder_seed(str(tmp_path), _label(b))
        assert seed is not None and seed.parent.name == f"beta{b:g}"


def test_a_missing_beta_falls_back_instead_of_failing(tmp_path, capsys):
    (tmp_path / "beta0.03").mkdir()
    (tmp_path / "beta0.03" / "final.pt").write_bytes(b"")
    assert _cross_ladder_seed(str(tmp_path), "0.5") is None
    out = capsys.readouterr().out
    assert "falling back" in out
    assert "beta0.5" in out                                # says which point missed


def test_no_seed_directory_means_no_cross_ladder_seed():
    assert _cross_ladder_seed(None, "0.03") is None
    assert _cross_ladder_seed("", "0.03") is None


def test_a_directory_with_no_final_pt_is_not_a_seed(tmp_path):
    """A ladder still training has `latest.pt` but no `final.pt`; seeding from a
    half-trained point silently would be worse than saying so."""
    (tmp_path / "beta0.03").mkdir()
    (tmp_path / "beta0.03" / "latest.pt").write_bytes(b"")
    assert _cross_ladder_seed(str(tmp_path), "0.03") is None
