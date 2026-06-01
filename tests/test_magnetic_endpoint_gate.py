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
