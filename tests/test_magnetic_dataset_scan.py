import logging
import sys
from pathlib import Path
import shutil
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.magnetic_dataset_scan import (
    MTOT_SHEET_TOL,
    dedup_by_label,
    find_band_roots,
    rank_per_sheet,
    scan_dataset,
)
from prodromos.magnetic_output_parser import MagneticOutputSummary


def qe_text(total, absolute, energy=-1.0):
    return f"""
    Program PWSCF
    !    total energy              =   {energy:.8f} Ry
         total magnetization       =   {total:.2f} Bohr mag/cell
         absolute magnetization    =   {absolute:.2f} Bohr mag/cell
    convergence has been achieved
    JOB DONE.
    """


def write_band(root, mags):
    for i, (total, absolute) in enumerate(mags, start=1):
        image_dir = root / f"image_{i:02d}"
        image_dir.mkdir(parents=True)
        (image_dir / "espresso.pwo").write_text(qe_text(total, absolute, energy=-float(i)), encoding="utf-8")


def test_dataset_scan_finds_and_classifies_band_roots():
    tmp_root = Path(__file__).resolve().parent / f"_tmp_dataset_scan_{uuid.uuid4().hex}"
    try:
        good = tmp_root / "good_band"
        bad = tmp_root / "bad_band"
        write_band(good, [(1.13, 2.0), (1.14, 2.1), (1.13, 2.0)])
        write_band(bad, [(1.13, 2.0), (1.13, 2.7), (1.13, 2.8)])

        roots = find_band_roots(tmp_root)
        rows = scan_dataset(tmp_root)

        assert roots == [bad, good]
        assert {row.band_root: row.verdict for row in rows} == {
            str(bad): "NO-GO_SINGLE_SHEET",
            str(good): "GO",
        }
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


# ---------------------------------------------------------------------------
# Helpers for unit tests
# ---------------------------------------------------------------------------

def _make_summary(
    *,
    path="sample.pwo",
    scf_converged=True,
    energy_eV=-100.0,
    total_magnetization_uB=1.13,
    absolute_magnetization_uB=2.43,
    nspin=2,
):
    return MagneticOutputSummary(
        engine="qe",
        path=path,
        scf_converged=scf_converged,
        job_done=scf_converged,
        energy_eV=energy_eV,
        energy_unit="Ry",
        total_magnetization_uB=total_magnetization_uB,
        absolute_magnetization_uB=absolute_magnetization_uB,
        nspin=nspin,
    )


# ---------------------------------------------------------------------------
# N-04: dedup_by_label tests
# ---------------------------------------------------------------------------

def test_dedup_by_label_passthrough_unique():
    a = _make_summary(path="job_a.pwo", energy_eV=-100.0)
    b = _make_summary(path="job_b.pwo", energy_eV=-101.0)
    result = dedup_by_label([a, b])
    assert len(result) == 2
    paths = {s.path for s in result}
    assert paths == {"job_a.pwo", "job_b.pwo"}


def test_dedup_by_label_prefers_converged_over_unconverged(tmp_path):
    # Two copies of the same label: one converged, one not.
    converged = _make_summary(path="job_x_worker1.pwo", scf_converged=True, energy_eV=-100.0)
    crashed = _make_summary(path="job_x_worker2.pwo", scf_converged=False, energy_eV=None)

    # Use a label extractor that strips the worker suffix
    get_label = lambda s: Path(s.path).stem.rsplit("_worker", 1)[0]  # noqa: E731
    result = dedup_by_label([converged, crashed], get_label=get_label)

    assert len(result) == 1
    assert result[0].scf_converged is True
    assert result[0].energy_eV == pytest.approx(-100.0)


def test_dedup_by_label_no_converged_returns_one_with_energy():
    crashed1 = _make_summary(path="job_y_w1.pwo", scf_converged=False, energy_eV=None)
    crashed2 = _make_summary(path="job_y_w2.pwo", scf_converged=False, energy_eV=-99.0)
    get_label = lambda s: Path(s.path).stem.rsplit("_w", 1)[0]  # noqa: E731
    result = dedup_by_label([crashed1, crashed2], get_label=get_label)
    assert len(result) == 1
    assert result[0].energy_eV is not None


def test_dedup_by_label_warns_on_energy_disagreement(caplog):
    # Two converged copies with noticeably different energies: should warn.
    get_label = lambda s: "same_job"  # noqa: E731
    a = _make_summary(path="same_a.pwo", scf_converged=True, energy_eV=-100.0)
    b = _make_summary(path="same_b.pwo", scf_converged=True, energy_eV=-100.5)
    with caplog.at_level(logging.WARNING, logger="prodromos.magnetic_dataset_scan"):
        result = dedup_by_label([a, b], get_label=get_label)
    assert len(result) == 1
    assert any("disagreement" in msg.lower() or "disagree" in msg.lower() for msg in caplog.messages)


def test_dedup_by_label_keeps_lowest_energy_among_converged():
    get_label = lambda s: "same_job"  # noqa: E731
    a = _make_summary(path="a.pwo", scf_converged=True, energy_eV=-100.0)
    b = _make_summary(path="b.pwo", scf_converged=True, energy_eV=-100.002)  # within 1 meV tolerance
    # Tight energies: no disagreement warning expected; should keep lower
    result = dedup_by_label([a, b], get_label=get_label)
    assert len(result) == 1
    assert result[0].energy_eV == pytest.approx(-100.002)


# ---------------------------------------------------------------------------
# N-01: rank_per_sheet tests
# ---------------------------------------------------------------------------

def test_rank_per_sheet_single_sheet():
    summaries = [
        _make_summary(path=f"s{i}.pwo", total_magnetization_uB=1.13 + 0.01 * i, energy_eV=-100.0 + i)
        for i in range(4)
    ]
    result = rank_per_sheet(summaries)
    assert result.verdict == "SINGLE_SHEET"
    assert result.cross_sheet_ranking_valid is True
    assert len(result.sheets) == 1
    sheet = next(iter(result.sheets.values()))
    # Sorted by energy ascending
    energies = [s.energy_eV for s in sheet]
    assert energies == sorted(e for e in energies if e is not None)


def test_rank_per_sheet_multi_sheet_refuses_cross_ranking():
    # Two groups: Mtot ~ +1.13 and Mtot ~ -1.13 (different sheets)
    sheet_plus = [
        _make_summary(path=f"p{i}.pwo", total_magnetization_uB=1.13 + 0.05 * i, energy_eV=-100.0 + i)
        for i in range(3)
    ]
    sheet_minus = [
        _make_summary(path=f"m{i}.pwo", total_magnetization_uB=-1.13 - 0.05 * i, energy_eV=-99.0 + i)
        for i in range(3)
    ]
    result = rank_per_sheet(sheet_plus + sheet_minus)
    assert result.verdict == "MULTI_SHEET"
    assert result.cross_sheet_ranking_valid is False
    assert len(result.sheets) >= 2
    # Each sheet is internally sorted by energy
    for sheet_summaries in result.sheets.values():
        if len(sheet_summaries) > 1:
            energies = [s.energy_eV for s in sheet_summaries if s.energy_eV is not None]
            assert energies == sorted(energies)


def test_rank_per_sheet_insufficient_data_when_no_magnetization():
    summaries = [
        MagneticOutputSummary(
            engine="qe",
            path=f"s{i}.pwo",
            scf_converged=True,
            energy_eV=-100.0 + i,
        )
        for i in range(3)
    ]
    result = rank_per_sheet(summaries)
    assert result.verdict == "INSUFFICIENT_DATA"
    assert result.cross_sheet_ranking_valid is False


def test_rank_per_sheet_free_magnetization_scenario():
    # Simulates a free-M screen: Mabs roughly constant (~42 uB) but Mtot
    # varies sign/magnitude across jobs → different sheets.
    free_m_summaries = [
        _make_summary(path="j1.pwo", total_magnetization_uB=+8.0, absolute_magnetization_uB=42.0, energy_eV=-200.0),
        _make_summary(path="j2.pwo", total_magnetization_uB=-8.0, absolute_magnetization_uB=42.1, energy_eV=-200.2),
        _make_summary(path="j3.pwo", total_magnetization_uB=+7.9, absolute_magnetization_uB=41.9, energy_eV=-200.1),
        _make_summary(path="j4.pwo", total_magnetization_uB=-7.8, absolute_magnetization_uB=42.0, energy_eV=-200.3),
    ]
    result = rank_per_sheet(free_m_summaries)
    assert result.verdict == "MULTI_SHEET"
    assert result.cross_sheet_ranking_valid is False
    assert any("cross-sheet" in r.lower() for r in result.reasons)
