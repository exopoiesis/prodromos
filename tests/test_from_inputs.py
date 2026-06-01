"""Tests for ``prodromos from-inputs`` (QE/ABACUS input -> tm-spec/0.3)."""
from __future__ import annotations

import json

import pytest

from prodromos.from_inputs import convert_to_tmspec, detect_code

pytest.importorskip("tm_spec.validator")
from tm_spec.validator import validate_doc  # noqa: E402

DATE = "2026-06-01"


# --------------------------------------------------------------------------
# fixtures (synthetic minimal inputs)
# --------------------------------------------------------------------------
QE_SCF = """\
&control
  calculation = 'scf'
  prefix = 'fes'
/
&system
  ibrav = 0
  nat = 3
  ntyp = 2
  ecutwfc = 60
  ecutrho = 240
  nspin = 1
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.005
  input_dft = 'PBE'
/
&electrons
  conv_thr = 1.0d-8
/
ATOMIC_SPECIES
Fe 55.845 Fe.upf
S 32.06 S.upf
ATOMIC_POSITIONS angstrom
Fe 0.0 0.0 0.0
S 1.2 1.2 1.2
S 2.4 2.4 2.4
CELL_PARAMETERS angstrom
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
K_POINTS automatic
2 2 2 0 0 0
"""

QE_RELAX = """\
&control
  calculation = 'vc-relax'
  forc_conv_thr = 1.0d-3
/
&system
  ibrav = 0
  nat = 2
  ntyp = 1
  ecutwfc = 50
  nspin = 2
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.01
  starting_magnetization(1) = 0.5
/
&electrons
  conv_thr = 1.0d-7
/
&ions
/
&cell
/
ATOMIC_SPECIES
Fe 55.845 Fe.upf
ATOMIC_POSITIONS angstrom
Fe 0.0 0.0 0.0
Fe 1.4 1.4 1.4
CELL_PARAMETERS angstrom
2.8 0.0 0.0
0.0 2.8 0.0
0.0 0.0 2.8
K_POINTS automatic
8 8 8 0 0 0
"""

ABACUS_INPUT = """\
INPUT_PARAMETERS
calculation     scf
ecutwfc         100
basis_type      pw
nspin           2
smearing_method gauss
smearing_sigma  0.01
dft_functional  pbe
ks_solver       cg
scf_thr         1e-8
"""

ABACUS_STRU = """\
ATOMIC_SPECIES
Fe 55.845 Fe.upf
S 32.06 S.upf

LATTICE_CONSTANT
1.8897259886   # 1 Angstrom in Bohr

LATTICE_VECTORS
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0

ATOMIC_POSITIONS
Direct

Fe
0.0
1
0.0 0.0 0.0 1 1 1

S
0.0
2
0.25 0.25 0.25 1 1 1
0.75 0.75 0.75 1 1 1
"""

ABACUS_KPT = """\
K_POINTS
0
Gamma
4 4 4 0 0 0
"""


def _write_qe(tmp_path, text, name="pw.in"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _write_abacus(tmp_path, *, kpt=True):
    d = tmp_path / "abacus_run"
    d.mkdir()
    (d / "INPUT").write_text(ABACUS_INPUT, encoding="utf-8")
    (d / "STRU").write_text(ABACUS_STRU, encoding="utf-8")
    if kpt:
        (d / "KPT").write_text(ABACUS_KPT, encoding="utf-8")
    return d


# --------------------------------------------------------------------------
# QE
# --------------------------------------------------------------------------
def test_qe_scf_valid_and_singlepoint(tmp_path):
    p = _write_qe(tmp_path, QE_SCF)
    doc = convert_to_tmspec(p, date=DATE)
    schema_errs, rule_issues = validate_doc(doc)
    rule_errors = [m for lvl, m in rule_issues if lvl == "error"]
    assert not schema_errs, schema_errs
    assert not rule_errors, rule_errors

    assert doc["spec"] == "tm-spec/0.3"
    assert doc["kind"] == "SinglePointCalculation"
    assert doc["structure"]["formula"] == "Fe1S2".replace("Fe1", "FeS").replace("SS2", "S2") or True
    # formula: 1 Fe + 2 S -> FeS2 (Hill alphabetical, Fe then S)
    assert doc["structure"]["formula"] == "FeS2"
    lvl = doc["calculation"]["level"]
    assert lvl["xc"] == "PBE"
    assert lvl["basis"] == {"kind": "plane_waves", "cutoff_Ry": 60.0, "rho_cutoff_Ry": 240.0}
    assert lvl["smearing"] == {"kind": "gaussian", "width_Ry": 0.005}
    assert lvl["spin"] == "none"
    assert doc["calculation"]["k_points"]["mesh"] == [2, 2, 2]
    assert doc["calculation"]["code"]["name"] == "QuantumESPRESSO"


def test_qe_relax_kind_and_spin(tmp_path):
    p = _write_qe(tmp_path, QE_RELAX)
    doc = convert_to_tmspec(p, date=DATE)
    schema_errs, rule_issues = validate_doc(doc)
    assert not schema_errs, schema_errs
    assert not [m for lvl, m in rule_issues if lvl == "error"]

    assert doc["kind"] == "RelaxCalculation"
    assert doc["structure"]["formula"] == "Fe2"
    assert doc["calculation"]["level"]["spin"] == "collinear"
    assert doc["calculation"]["level"]["smearing"]["kind"] == "marzari-vanderbilt"
    assert doc["relax_protocol"]["cell_relax"] is True
    assert doc["relax_protocol"]["cell_relax_kind"] == "all"
    # nspin=2 -> magnetic block present
    assert doc["magnetic"]["state"] in {"FM", "AFM-G", "ferri"}
    assert doc["calculation"]["k_points"]["mesh"] == [8, 8, 8]


def test_qe_kind_override(tmp_path):
    p = _write_qe(tmp_path, QE_SCF)
    doc = convert_to_tmspec(p, date=DATE, kind="RelaxCalculation")
    assert doc["kind"] == "RelaxCalculation"


# --------------------------------------------------------------------------
# ABACUS
# --------------------------------------------------------------------------
def test_abacus_dir_valid(tmp_path):
    d = _write_abacus(tmp_path)
    doc = convert_to_tmspec(d, date=DATE)
    schema_errs, rule_issues = validate_doc(doc)
    assert not schema_errs, schema_errs
    assert not [m for lvl, m in rule_issues if lvl == "error"]

    assert doc["kind"] == "SinglePointCalculation"
    # 1 Fe + 2 S
    assert doc["structure"]["formula"] == "FeS2"
    # lattice constant 1 A in Bohr -> vectors should come out ~5 A
    lv = doc["structure"]["lattice_vectors_A"]
    assert abs(lv[0][0] - 5.0) < 1e-3
    lvl = doc["calculation"]["level"]
    assert lvl["xc"] == "PBE"
    assert lvl["basis"]["kind"] == "plane_waves"
    assert lvl["basis"]["cutoff_Ry"] == 100.0
    assert lvl["spin"] == "collinear"
    assert lvl["smearing"]["kind"] == "gaussian"
    assert doc["calculation"]["k_points"]["mesh"] == [4, 4, 4]
    assert doc["calculation"]["code"]["name"] == "ABACUS"


def test_abacus_input_file_resolves_siblings(tmp_path):
    d = _write_abacus(tmp_path)
    doc = convert_to_tmspec(d / "INPUT", code="abacus", date=DATE)
    assert doc["structure"]["formula"] == "FeS2"


def test_abacus_lcao_basis(tmp_path):
    d = tmp_path / "abacus_lcao"
    d.mkdir()
    (d / "INPUT").write_text(
        ABACUS_INPUT.replace("basis_type      pw", "basis_type      lcao"), encoding="utf-8"
    )
    (d / "STRU").write_text(ABACUS_STRU, encoding="utf-8")
    doc = convert_to_tmspec(d, code="abacus", date=DATE)
    assert doc["calculation"]["level"]["basis"]["kind"] == "numeric_AOs"


# --------------------------------------------------------------------------
# auto-detection
# --------------------------------------------------------------------------
def test_detect_qe_by_suffix(tmp_path):
    p = _write_qe(tmp_path, QE_SCF)
    assert detect_code(p) == "qe"


def test_detect_qe_by_content(tmp_path):
    p = _write_qe(tmp_path, QE_SCF, name="job.pwi")
    assert detect_code(p) == "qe"


def test_detect_abacus_dir(tmp_path):
    d = _write_abacus(tmp_path)
    assert detect_code(d) == "abacus"


def test_detect_abacus_input_file(tmp_path):
    d = _write_abacus(tmp_path)
    assert detect_code(d / "INPUT") == "abacus"


def test_auto_dispatch_matches_explicit(tmp_path):
    p = _write_qe(tmp_path, QE_SCF)
    assert convert_to_tmspec(p, code="auto", date=DATE) == convert_to_tmspec(
        p, code="qe", date=DATE
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_from_inputs_json(tmp_path, capsys):
    from prodromos import from_inputs

    p = _write_qe(tmp_path, QE_SCF)
    rc = from_inputs.main([str(p), "--date", DATE, "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["kind"] == "SinglePointCalculation"
    assert doc["structure"]["formula"] == "FeS2"


def test_cli_default_date_placeholder_warns(tmp_path, capsys):
    """Default placeholder date -> id pattern fails -> emitted as a NOTE, rc still 0."""
    from prodromos import from_inputs

    p = _write_qe(tmp_path, QE_SCF)
    rc = from_inputs.main([str(p), "--json"])
    assert rc == 0  # still emits the starter stub
    err = capsys.readouterr().err
    assert "does not yet validate" in err or "YYYY-MM-DD" in err
