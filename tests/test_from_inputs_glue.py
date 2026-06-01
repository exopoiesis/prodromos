"""Onboarding glue: ``prodromos plan <qe.in>`` auto-converts then plans."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("tm_spec.validator")

from prodromos.plan import cli as plan_cli  # noqa: E402

DATE = "2026-06-01"

QE_NEB = """\
&control
  calculation = 'neb'
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

QE_SCF = QE_NEB.replace("calculation = 'neb'", "calculation = 'scf'")


def _write(tmp_path, text, name="pw.in"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_plan_autoconverts_qe_in(tmp_path, capsys):
    p = _write(tmp_path, QE_SCF)
    rc = plan_cli.main([str(p), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "plan"
    # the glue reason must be surfaced
    assert any("auto-converted from QE" in r for r in payload["reasons"])


def test_plan_autoconverts_neb_in(tmp_path, capsys):
    p = _write(tmp_path, QE_NEB, name="my_neb.in")
    rc = plan_cli.main([str(p), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert any("auto-converted from QE" in r for r in payload["reasons"])


def test_plan_autoconvert_preflight_emit(tmp_path, capsys):
    p = _write(tmp_path, QE_SCF)
    rc = plan_cli.main([str(p), "--emit", "preflight", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"]["name"] == "prodromos"


def test_plan_abacus_dir_autoconvert(tmp_path, capsys):
    from tests.test_from_inputs import ABACUS_INPUT, ABACUS_KPT, ABACUS_STRU

    d = tmp_path / "abacus_run"
    d.mkdir()
    (d / "INPUT").write_text(ABACUS_INPUT, encoding="utf-8")
    (d / "STRU").write_text(ABACUS_STRU, encoding="utf-8")
    (d / "KPT").write_text(ABACUS_KPT, encoding="utf-8")
    rc = plan_cli.main([str(d), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert any("auto-converted from ABACUS" in r for r in payload["reasons"])


def test_plan_unparseable_case_asks_for_code(tmp_path, capsys):
    p = tmp_path / "mystery.dat"
    p.write_text("just some text\nnot an input file\n", encoding="utf-8")
    rc = plan_cli.main([str(p), "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "INVALID_CASE"
    assert any("--code" in r for r in payload["reasons"])
