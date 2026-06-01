import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.magnetic_output_parser import (
    HA_TO_EV,
    RY_TO_EV,
    detect_engine,
    parse_abacus_output,
    parse_jdftx_output,
    parse_output_file,
    parse_qe_output,
)


def test_qe_parser_takes_final_magnetization_and_local_moments():
    text = """
     Program PWSCF
     !    total energy              =   -10.00000000 Ry
          total magnetization       =     1.00 Bohr mag/cell
          absolute magnetization    =     2.00 Bohr mag/cell
     !    total energy              =   -11.00000000 Ry
          total magnetization       =     1.13 Bohr mag/cell
          absolute magnetization    =     2.43 Bohr mag/cell

     Magnetic moment per site (integrated on atomic sphere of radius R)
     atom    1 (R=0.101)  charge= 13.8328  magn= -0.0089
     atom    2 (R=0.102)  charge= 13.9000  magn=  1.2345

     convergence has been achieved in 21 iterations
     JOB DONE.
    """

    summary = parse_qe_output("sample.pwo", text)

    assert summary.engine == "qe"
    assert summary.scf_converged is True
    assert summary.job_done is True
    assert summary.energy_eV == pytest.approx(-11.0 * RY_TO_EV)
    assert summary.total_magnetization_uB == pytest.approx(1.13)
    assert summary.absolute_magnetization_uB == pytest.approx(2.43)
    assert len(summary.local_moments) == 2
    assert summary.local_moments[1].atom_index == 2
    assert summary.local_moments[1].moment_uB == pytest.approx(1.2345)


def test_qe_parser_flags_unsettled_magnetization_drift():
    # Mtot still climbing at the loose conv_thr end (1.0 -> 1.4 over 3 steps);
    # Mabs essentially flat. Should parse the final value AND flag not-settled.
    text = """
     !    total energy              =   -11.00000000 Ry
          total magnetization       =     1.00 Bohr mag/cell
          absolute magnetization    =     2.40 Bohr mag/cell
          total magnetization       =     1.15 Bohr mag/cell
          absolute magnetization    =     2.41 Bohr mag/cell
          total magnetization       =     1.28 Bohr mag/cell
          absolute magnetization    =     2.42 Bohr mag/cell
          total magnetization       =     1.40 Bohr mag/cell
          absolute magnetization    =     2.43 Bohr mag/cell
     convergence has been achieved in 55 iterations
    """
    summary = parse_qe_output("sample.pwo", text)
    assert summary.total_magnetization_uB == pytest.approx(1.40)
    assert summary.absolute_magnetization_uB == pytest.approx(2.43)
    assert summary.total_magnetization_drift_uB == pytest.approx(0.40)  # |1.40-1.00|
    assert summary.absolute_magnetization_drift_uB == pytest.approx(0.03)
    assert summary.magnetization_settled is False
    assert any("not settled" in w for w in summary.warnings)


def test_qe_parser_reports_settled_magnetization():
    # Both moments flat over the last few steps -> settled, no drift warning.
    text = """
     !    total energy              =   -11.00000000 Ry
          total magnetization       =     1.12 Bohr mag/cell
          absolute magnetization    =     2.42 Bohr mag/cell
          total magnetization       =     1.13 Bohr mag/cell
          absolute magnetization    =     2.43 Bohr mag/cell
          total magnetization       =     1.13 Bohr mag/cell
          absolute magnetization    =     2.43 Bohr mag/cell
     convergence has been achieved in 21 iterations
    """
    summary = parse_qe_output("sample.pwo", text)
    assert summary.magnetization_settled is True
    assert summary.total_magnetization_drift_uB == pytest.approx(0.01)
    assert not any("not settled" in w for w in summary.warnings)


def test_qe_parser_drift_none_with_single_magnetization_print():
    # A single printed value carries no history -> drift None, settled None.
    text = """
     !    total energy              =   -11.00000000 Ry
          total magnetization       =     1.13 Bohr mag/cell
          absolute magnetization    =     2.43 Bohr mag/cell
     convergence has been achieved in 1 iterations
    """
    summary = parse_qe_output("sample.pwo", text)
    assert summary.total_magnetization_uB == pytest.approx(1.13)
    assert summary.total_magnetization_drift_uB is None
    assert summary.magnetization_settled is None


def test_abacus_parser_handles_nonmagnetic_scf_output():
    text = """
    INPUT_PARAMETERS
    nspin = 1
    #SCF IS CONVERGED#
    #TOTAL ENERGY# -139910.20001 eV
    !FINAL_ETOT_IS -139910.2000078939308878 eV
    """

    summary = parse_abacus_output("running_scf.log", text)

    assert summary.engine == "abacus"
    assert summary.scf_converged is True
    assert summary.nspin == 1
    assert summary.energy_eV == pytest.approx(-139910.2000078939308878)
    assert summary.total_magnetization_uB is None
    assert "nspin=1" in summary.warnings[0]


def test_jdftx_parser_takes_last_magnetic_moment_and_energy():
    text = """
    JDFTx 1.7.0
    spintype z-spin
    FillingsUpdate: mu: -0.1 nElectrons: 10.000000 magneticMoment: [ Abs: 5.09526 Tot: +0.00000 ]
    LCAOMinimize: Iter: 12 F: -100.0000000
    FillingsUpdate: mu: -0.2 nElectrons: 10.000000 magneticMoment: [ Abs: 4.50000 Tot: -1.25000 ]
    LCAOMinimize: Iter: 13 F: -101.0000000
    Converged
    """

    summary = parse_jdftx_output("sample.out", text)

    assert summary.engine == "jdftx"
    assert summary.scf_converged is True
    assert summary.energy_eV == pytest.approx(-101.0 * HA_TO_EV)
    assert summary.total_magnetization_uB == pytest.approx(-1.25)
    assert summary.absolute_magnetization_uB == pytest.approx(4.5)


def test_engine_detection_for_known_markers():
    assert detect_engine("x.pwo", "plain") == "qe"
    assert detect_engine("running_scf.log", "#SCF IS CONVERGED#") == "abacus"
    assert detect_engine("x.out", "FillingsUpdate magneticMoment: [ Abs: 1 Tot: 0 ]") == "jdftx"


@pytest.mark.requires_data
def test_marc_spin_diagnostic_qe_corpus_if_available():
    root = Path(r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-29\marc_VFe_spin_diagnostic")
    if not root.exists():
        pytest.skip("local harvested DFT corpus is not available")

    expected = {
        "marc_endA_m113.pwo": (1.13, 2.43),
        "marc_endA_m167.pwo": (1.67, 2.27),
        "marc_endB_m113.pwo": (1.13, 2.10),
        "marc_endB_m167.pwo": (1.67, 2.38),
    }
    for filename, (total_mag, abs_mag) in expected.items():
        summary = parse_output_file(root / filename)
        assert summary.engine == "qe"
        assert summary.total_magnetization_uB == pytest.approx(total_mag)
        assert summary.absolute_magnetization_uB == pytest.approx(abs_mag)
        assert summary.energy_eV is not None

    complete = parse_output_file(root / "marc_endA_m113.pwo")
    assert complete.scf_converged is True
    assert complete.job_done is True

    truncated = parse_output_file(root / "marc_endA_m167.pwo")
    assert truncated.scf_converged is False
    assert truncated.job_done is False
    assert any("last SCF iteration" in warning for warning in truncated.warnings)
