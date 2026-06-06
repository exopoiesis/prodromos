import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.magnetic_endpoint_gate import endpoint_magnetic_gate
from prodromos.magnetic_output_parser import MagneticOutputSummary, parse_output_file


def summary(total, absolute, converged=True):
    return MagneticOutputSummary(
        engine="qe",
        path="x.pwo",
        scf_converged=converged,
        job_done=converged,
        total_magnetization_uB=total,
        absolute_magnetization_uB=absolute,
    )


def test_endpoint_gate_go_when_magnetic_state_matches():
    result = endpoint_magnetic_gate(summary(1.13, 2.43), summary(1.14, 2.10))

    assert result.verdict == "GO"
    assert result.endpoint_split is False
    assert result.delta_total_uB == pytest.approx(0.01)
    assert result.delta_abs_uB == pytest.approx(0.33)


def test_endpoint_gate_no_go_for_converged_sheet_split():
    result = endpoint_magnetic_gate(summary(1.67, 2.38), summary(1.13, 2.10))

    assert result.verdict == "NO-GO_SINGLE_SHEET"
    assert result.endpoint_split is True
    assert result.delta_total_uB == pytest.approx(0.54)
    assert any("shared constrained M" in reason for reason in result.reasons)


def test_endpoint_gate_review_for_incomplete_split_evidence():
    result = endpoint_magnetic_gate(summary(1.67, 2.27, converged=False), summary(1.13, 2.10))

    assert result.verdict == "REVIEW"
    assert result.endpoint_split is True
    assert any("incomplete SCF" in reason for reason in result.reasons)


def test_endpoint_gate_review_when_magnetization_missing():
    result = endpoint_magnetic_gate(summary(None, None), summary(1.13, 2.10))

    assert result.verdict == "REVIEW"
    assert result.endpoint_split is False
    assert result.delta_total_uB is None


def test_per_tm_threshold_avoids_false_nogo_on_slow_drift():
    # Troilite-like: M_total = 0 (AFM), |Δ abs| = 0.52 uB just trips the 0.5 absolute
    # threshold, but over 12 Fe that is 0.043 uB/Fe -> a smooth drift, not a crossing.
    a = summary(0.0, 5.00)
    b = summary(0.0, 5.52)
    # without n_magnetic: absolute channel trips -> NO-GO
    assert endpoint_magnetic_gate(a, b).verdict == "NO-GO_SINGLE_SHEET"
    # with n_magnetic=12: per-TM relative threshold downgrades the drift -> GO
    rel = endpoint_magnetic_gate(a, b, n_magnetic=12)
    assert rel.verdict == "GO"
    assert rel.endpoint_split is False
    assert rel.delta_abs_per_tm_uB == pytest.approx(0.52 / 12)
    assert any("smooth drift" in r for r in rel.reasons)


def test_per_tm_threshold_still_flags_true_crossing():
    # A real crossing: one Fe flips ~2 uB -> |Δ abs| ~ 2.0 over 12 Fe = 0.17/Fe... still
    # below 0.30/TM, BUT the total channel (integer ~2 uB) catches it.
    a = summary(0.0, 5.0)
    b = summary(2.0, 7.0)  # delta_total=2.0 > 0.3 -> total channel splits
    res = endpoint_magnetic_gate(a, b, n_magnetic=12)
    assert res.verdict == "NO-GO_SINGLE_SHEET"


def test_reconcile_band_arbiter_overrides_endpoint_nogo():
    from prodromos.magnetic_band_gate import BandGateResult
    from prodromos.magnetic_endpoint_gate import reconcile_endpoint_and_band

    ep = endpoint_magnetic_gate(summary(0.0, 5.00), summary(0.0, 5.52))  # NO-GO (no n_mag)
    assert ep.verdict == "NO-GO_SINGLE_SHEET"
    band = BandGateResult(verdict="GO", sheet_crossing=False, endpoint_split=False, crossing_edge=-1)
    combined = reconcile_endpoint_and_band(ep, band)
    assert combined["combined_verdict"] == "GO"
    assert combined["arbiter"] == "band"
    assert combined["agree"] is False


def test_reconcile_without_band_keeps_endpoint():
    from prodromos.magnetic_endpoint_gate import reconcile_endpoint_and_band

    ep = endpoint_magnetic_gate(summary(1.13, 2.43), summary(1.14, 2.10))
    combined = reconcile_endpoint_and_band(ep, None)
    assert combined["combined_verdict"] == ep.verdict
    assert combined["arbiter"] == "endpoint"


def test_marc_spin_diagnostic_endpoint_gate():
    root = Path(__file__).parent / "fixtures" / "marc_spin_diagnostic"

    same_sheet = endpoint_magnetic_gate(
        parse_output_file(root / "marc_endA_m113.pwo"),
        parse_output_file(root / "marc_endB_m113.pwo"),
    )
    assert same_sheet.verdict == "GO"

    split_sheet = endpoint_magnetic_gate(
        parse_output_file(root / "marc_endB_m167.pwo"),
        parse_output_file(root / "marc_endB_m113.pwo"),
    )
    assert split_sheet.verdict == "NO-GO_SINGLE_SHEET"
    assert split_sheet.delta_total_uB == pytest.approx(0.54)
