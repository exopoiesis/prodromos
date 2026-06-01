import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.magnetic_recommendation import EndpointMatrixEntry, build_recommendation, choose_common_endpoint_sheet


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


def test_build_recommendation_from_marc_corpus_if_available():
    band_root = Path(r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-28_marc_VFe_tier1_v4c\neb_done")
    matrix_root = Path(r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-29\marc_VFe_spin_diagnostic")
    if not band_root.exists() or not matrix_root.exists():
        return

    rec = build_recommendation(band_root, matrix_root)

    assert rec.action == "RERUN_SINGLE_SHEET_CONSTRAINED_M"
    assert rec.constrained_magnetization_uB == 1.13
    assert rec.seam_edge == 3
