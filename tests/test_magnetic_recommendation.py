import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.magnetic_recommendation import (
    EndpointMatrixEntry,
    ProvenanceMismatchError,
    build_recommendation,
    choose_common_endpoint_sheet,
    compute_within_method_delta,
)
from prodromos.magnetic_output_parser import MagneticOutputSummary


def entry(endpoint, label, energy, total=1.13, converged=True):
    return EndpointMatrixEntry(
        endpoint=endpoint,
        target_label=label,
        path=f"{endpoint}_{label}.pwo",
        scf_converged=converged,
        energy_eV=energy,
        total_magnetization_uB=total,
        absolute_magnetization_uB=2.0,
    )


def test_choose_common_endpoint_sheet_when_both_endpoints_prefer_same_m():
    decision = choose_common_endpoint_sheet(
        [
            entry("endA", "m113", -10.0, 1.13),
            entry("endA", "m167", -9.8, 1.67),
            entry("endB", "m113", -11.0, 1.13),
            entry("endB", "m167", -10.7, 1.67),
        ]
    )

    assert decision["common_label"] == "m113"
    assert decision["common_m"] == 1.13


def test_choose_common_endpoint_sheet_rejects_different_endpoint_preferences():
    decision = choose_common_endpoint_sheet(
        [
            entry("endA", "m113", -10.0, 1.13),
            entry("endA", "m167", -9.8, 1.67),
            entry("endB", "m113", -10.7, 1.13),
            entry("endB", "m167", -11.0, 1.67),
        ]
    )

    assert decision["common_label"] is None


def test_build_recommendation_without_endpoint_matrix_requests_matrix():
    rec = build_recommendation()

    assert rec.action == "RUN_ENDPOINT_MATRIX"
    assert rec.required_next_calculations


def test_build_recommendation_from_marc_corpus():
    band_root = Path(__file__).parent / "fixtures" / "marc_tier1_band"
    matrix_root = Path(__file__).parent / "fixtures" / "marc_spin_diagnostic"

    rec = build_recommendation(band_root, matrix_root)

    assert rec.action == "RERUN_SINGLE_SHEET_CONSTRAINED_M"
    assert rec.constrained_magnetization_uB == 1.13
    assert rec.seam_edge == 3


# ---------------------------------------------------------------------------
# N-14: compute_within_method_delta tests
# ---------------------------------------------------------------------------

def _make_summary(
    *,
    energy_eV,
    u_eff=None,
    nspin=2,
    functional="PBE",
    ecut=70.0,
    kpts="automatic 6 6 6  0 0 0",
    path="sample.pwo",
):
    """Helper: build a minimal MagneticOutputSummary with provenance fields."""
    return MagneticOutputSummary(
        engine="qe",
        path=path,
        scf_converged=True,
        job_done=True,
        energy_eV=energy_eV,
        energy_unit="Ry",
        total_magnetization_uB=1.13,
        absolute_magnetization_uB=2.43,
        nspin=nspin,
        u_eff=u_eff,
        functional=functional,
        ecut=ecut,
        kpts=kpts,
    )


def test_compute_within_method_delta_returns_correct_value():
    a = _make_summary(energy_eV=-100.0)
    b = _make_summary(energy_eV=-99.5)
    delta = compute_within_method_delta(a, b)
    assert delta == pytest.approx(0.5)


def test_compute_within_method_delta_negative_delta():
    a = _make_summary(energy_eV=-99.0)
    b = _make_summary(energy_eV=-100.0)
    assert compute_within_method_delta(a, b) == pytest.approx(-1.0)


def test_compute_within_method_delta_raises_on_u_eff_mismatch():
    a = _make_summary(energy_eV=-100.0, u_eff=None)   # U=0
    b = _make_summary(energy_eV=-100.5, u_eff=4.3)    # U=4.3 eV
    with pytest.raises(ProvenanceMismatchError) as exc_info:
        compute_within_method_delta(a, b)
    assert "PROVENANCE_MISMATCH" in str(exc_info.value)
    assert "u_eff" in exc_info.value.mismatches


def test_compute_within_method_delta_raises_on_nspin_mismatch():
    a = _make_summary(energy_eV=-100.0, nspin=1)
    b = _make_summary(energy_eV=-100.5, nspin=2)
    with pytest.raises(ProvenanceMismatchError) as exc_info:
        compute_within_method_delta(a, b)
    assert "nspin" in exc_info.value.mismatches


def test_compute_within_method_delta_raises_on_functional_mismatch():
    a = _make_summary(energy_eV=-100.0, functional="PBE")
    b = _make_summary(energy_eV=-100.5, functional="LDA")
    with pytest.raises(ProvenanceMismatchError) as exc_info:
        compute_within_method_delta(a, b)
    assert "functional" in exc_info.value.mismatches


def test_compute_within_method_delta_raises_on_ecut_mismatch():
    a = _make_summary(energy_eV=-100.0, ecut=60.0)
    b = _make_summary(energy_eV=-100.5, ecut=80.0)
    with pytest.raises(ProvenanceMismatchError) as exc_info:
        compute_within_method_delta(a, b)
    assert "ecut" in exc_info.value.mismatches


def test_compute_within_method_delta_raises_on_kpts_mismatch():
    a = _make_summary(energy_eV=-100.0, kpts="automatic 4 4 4  0 0 0")
    b = _make_summary(energy_eV=-100.5, kpts="automatic 6 6 6  0 0 0")
    with pytest.raises(ProvenanceMismatchError) as exc_info:
        compute_within_method_delta(a, b)
    assert "kpts" in exc_info.value.mismatches


def test_compute_within_method_delta_allows_both_none_provenance():
    # Both None for a dimension = "unknown on both sides" = allowed.
    a = _make_summary(energy_eV=-100.0, functional=None, ecut=None, kpts=None)
    b = _make_summary(energy_eV=-100.5, functional=None, ecut=None, kpts=None)
    # Should not raise
    delta = compute_within_method_delta(a, b)
    assert delta == pytest.approx(-0.5)


def test_compute_within_method_delta_raises_when_one_side_none_provenance():
    # One side has a known functional, the other does not → mismatch.
    a = _make_summary(energy_eV=-100.0, functional="PBE")
    b = _make_summary(energy_eV=-100.5, functional=None)
    with pytest.raises(ProvenanceMismatchError) as exc_info:
        compute_within_method_delta(a, b)
    assert "functional" in exc_info.value.mismatches


def test_compute_within_method_delta_raises_on_missing_energy():
    a = _make_summary(energy_eV=None)
    b = _make_summary(energy_eV=-100.0)
    with pytest.raises(ValueError, match="energy_eV=None"):
        compute_within_method_delta(a, b)


def test_provenance_mismatch_error_carries_multiple_mismatches():
    a = _make_summary(energy_eV=-100.0, u_eff=None, nspin=1)
    b = _make_summary(energy_eV=-100.5, u_eff=4.3, nspin=2)
    with pytest.raises(ProvenanceMismatchError) as exc_info:
        compute_within_method_delta(a, b)
    assert "u_eff" in exc_info.value.mismatches
    assert "nspin" in exc_info.value.mismatches
