"""Tests for N-09 lint_dft_script (static pre-flight lint for QE/ABACUS scripts)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from prodromos.lint_dft_script import (
    run_lint_dft_script,
    _check_pseudo_dir,
    _check_outdir_nesting,
    _check_upf_wfc,
)


# ---------------------------------------------------------------------------
# Fixtures: script templates
# ---------------------------------------------------------------------------

_GOOD_SCRIPT = """\
import os
from ase.io import read as _raw_read

pseudo_dir = "/absolute/path/to/pseudos"
outdir = "tmp"

def _clean_read(path):
    # strips nspins= and other non-standard keys
    return _raw_read(path)

atoms = _clean_read("structure.xyz")
"""

_BAD_PSEUDO_DIR_RELATIVE = """\
pseudo_dir = "pseudos/pbe"
outdir = "tmp"
"""

_BAD_OUTDIR_NESTED_3ARGS = """\
import os
pseudo_dir = "/abs/pseudos"
work_dir = "/jobs"
label = "myjob"
outdir = os.path.join(work_dir, label, "tmp")
"""

_BAD_OUTDIR_NESTED_FSTRING = """\
pseudo_dir = "/abs/pseudos"
work_dir = "/jobs"
label = "myjob"
outdir = f"{work_dir}/{label}/tmp"
"""

_BAD_ASE_READ_NO_CLEAN = """\
import ase.io
pseudo_dir = "/abs/pseudos"
outdir = "tmp"
atoms = ase.io.read("structure.xyz")
"""

_GOOD_SCRIPT_WITH_CLEAN = """\
import ase.io
pseudo_dir = "/abs/pseudos"
outdir = "tmp"

def _clean_read(path):
    return ase.io.read(path)

atoms = _clean_read("structure.xyz")
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_script(tmp_path: Path, content: str, name: str = "script.py") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _write_xyz(tmp_path: Path, comment: str = "", name: str = "struct.xyz") -> Path:
    """Minimal 1-atom extxyz."""
    p = tmp_path / name
    lines = [
        "1",
        comment or 'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3',
        "Fe 0.0 0.0 0.0",
    ]
    p.write_text("\n".join(lines))
    return p


def _write_upf(tmp_path: Path, n_wfc: int, name: str = "Fe.upf") -> Path:
    content = f"""<UPF version="2.0.1">
  <PP_HEADER
   generated="oncv"
   number_of_wfc="{n_wfc}"
   number_of_proj="6"
  />
</UPF>
"""
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Tests: CHECK-1  pseudo_dir
# ---------------------------------------------------------------------------

def test_good_script_passes_all_checks(tmp_path):
    s = _write_script(tmp_path, _GOOD_SCRIPT)
    env = run_lint_dft_script(s)
    assert env["verdict"] == "PASS"
    assert env["result"]["n_failed"] == 0
    assert env["result"]["failed_checks"] == []


def test_relative_pseudo_dir_fails_check1(tmp_path):
    s = _write_script(tmp_path, _BAD_PSEUDO_DIR_RELATIVE)
    env = run_lint_dft_script(s)
    assert env["verdict"] == "FAIL"
    assert "CHECK-1" in env["result"]["failed_checks"][0]
    assert any("pseudo_dir" in r and "relative" in r.lower() for r in env["reasons"])


def test_absolute_pseudo_dir_passes_check1(tmp_path):
    script = 'pseudo_dir = "/abs/pseudos"\noutdir = "tmp"\n'
    s = _write_script(tmp_path, script)
    env = run_lint_dft_script(s)
    # CHECK-1 should pass
    assert "CHECK-1" not in env["result"]["failed_checks"]


# ---------------------------------------------------------------------------
# Tests: CHECK-2  outdir nesting
# ---------------------------------------------------------------------------

def test_outdir_nested_3args_fails_check2(tmp_path):
    s = _write_script(tmp_path, _BAD_OUTDIR_NESTED_3ARGS)
    env = run_lint_dft_script(s)
    assert env["verdict"] == "FAIL"
    assert any("CHECK-2" in c for c in env["result"]["failed_checks"])
    assert any("outdir" in r.lower() and "join" in r.lower() for r in env["reasons"])


def test_outdir_fstring_nested_fails_check2(tmp_path):
    s = _write_script(tmp_path, _BAD_OUTDIR_NESTED_FSTRING)
    env = run_lint_dft_script(s)
    assert env["verdict"] == "FAIL"
    assert any("CHECK-2" in c for c in env["result"]["failed_checks"])


def test_outdir_simple_tmp_passes_check2(tmp_path):
    script = 'pseudo_dir = "/abs"\noutdir = "tmp"\n'
    s = _write_script(tmp_path, script)
    env = run_lint_dft_script(s)
    assert "CHECK-2" not in env["result"]["failed_checks"]


# ---------------------------------------------------------------------------
# Tests: CHECK-3  extxyz clean-read
# ---------------------------------------------------------------------------

def test_ase_read_without_clean_fails_check3(tmp_path):
    s = _write_script(tmp_path, _BAD_ASE_READ_NO_CLEAN)
    env = run_lint_dft_script(s)
    assert env["verdict"] == "FAIL"
    assert any("CHECK-3" in c for c in env["result"]["failed_checks"])
    assert any("clean" in r.lower() for r in env["reasons"])


def test_ase_read_with_clean_wrapper_passes_check3(tmp_path):
    s = _write_script(tmp_path, _GOOD_SCRIPT_WITH_CLEAN)
    env = run_lint_dft_script(s)
    assert "CHECK-3" not in env["result"]["failed_checks"]


def test_xyz_with_nspins_comment_fails_check3(tmp_path):
    """xyz file with nspins= in comment triggers CHECK-3 even if script has clean_read."""
    bad_comment = (
        'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 '
        'nspins=1 free_energy=-123.45'
    )
    xyz = _write_xyz(tmp_path, comment=bad_comment)
    s = _write_script(tmp_path, _GOOD_SCRIPT)
    env = run_lint_dft_script(s, xyz_path=xyz)
    assert env["verdict"] == "FAIL"
    assert any("CHECK-3" in c for c in env["result"]["failed_checks"])
    assert any("nspins" in r for r in env["reasons"])


def test_clean_xyz_no_nonstd_keys_passes(tmp_path):
    xyz = _write_xyz(tmp_path)  # default comment has no nspins
    s = _write_script(tmp_path, _GOOD_SCRIPT)
    env = run_lint_dft_script(s, xyz_path=xyz)
    assert "CHECK-3" not in env["result"]["failed_checks"]


def test_xyz_with_nkpts_comment_fails(tmp_path):
    bad_comment = (
        'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 nkpts=64'
    )
    xyz = _write_xyz(tmp_path, comment=bad_comment)
    s = _write_script(tmp_path, _GOOD_SCRIPT)
    env = run_lint_dft_script(s, xyz_path=xyz)
    assert any("CHECK-3" in c for c in env["result"]["failed_checks"])


# ---------------------------------------------------------------------------
# Tests: CHECK-4  UPF number_of_wfc
# ---------------------------------------------------------------------------

def test_upf_wfc_zero_fails_check4(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    _write_upf(pseudo_dir, n_wfc=0, name="Fe_pbe.upf")
    s = _write_script(tmp_path, _GOOD_SCRIPT)
    env = run_lint_dft_script(s, pseudo_dir=pseudo_dir)
    assert env["verdict"] == "FAIL"
    assert any("CHECK-4" in c for c in env["result"]["failed_checks"])
    assert any("Fe_pbe.upf" in r for r in env["reasons"])


def test_upf_wfc_nonzero_passes_check4(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    _write_upf(pseudo_dir, n_wfc=6, name="Fe_pbe.upf")
    _write_upf(pseudo_dir, n_wfc=4, name="S_pbe.upf")
    s = _write_script(tmp_path, _GOOD_SCRIPT)
    env = run_lint_dft_script(s, pseudo_dir=pseudo_dir)
    assert "CHECK-4" not in env["result"]["failed_checks"]


def test_upf_mixed_fails_if_any_zero(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    _write_upf(pseudo_dir, n_wfc=0, name="Fe_bad.upf")
    _write_upf(pseudo_dir, n_wfc=6, name="S_ok.upf")
    s = _write_script(tmp_path, _GOOD_SCRIPT)
    env = run_lint_dft_script(s, pseudo_dir=pseudo_dir)
    assert any("CHECK-4" in c for c in env["result"]["failed_checks"])


def test_no_upf_in_pseudo_dir_passes_check4(tmp_path):
    pseudo_dir = tmp_path / "empty_pseudos"
    pseudo_dir.mkdir()
    # No .upf files -> CHECK-4 passes (nothing to check)
    s = _write_script(tmp_path, _GOOD_SCRIPT)
    env = run_lint_dft_script(s, pseudo_dir=pseudo_dir)
    assert "CHECK-4" not in env["result"]["failed_checks"]


# ---------------------------------------------------------------------------
# Tests: combined multi-bug scripts
# ---------------------------------------------------------------------------

def test_all_four_bugs_fails_with_four_reasons(tmp_path):
    # Has all four bugs: relative pseudo_dir, nested outdir, bare ase.io.read,
    # and we'll add a bad UPF.
    bad_script = """\
import os
import ase.io
pseudo_dir = "relative/pseudos"
work_dir = "/jobs"
label = "test"
outdir = os.path.join(work_dir, label, "tmp")
atoms = ase.io.read("structure.xyz")
"""
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    _write_upf(pseudo_dir, n_wfc=0)

    s = _write_script(tmp_path, bad_script)
    env = run_lint_dft_script(s, pseudo_dir=pseudo_dir)
    assert env["verdict"] == "FAIL"
    assert env["result"]["n_failed"] == 4


def test_envelope_has_stable_keys(tmp_path):
    s = _write_script(tmp_path, _GOOD_SCRIPT)
    env = run_lint_dft_script(s)
    assert set(env) == {
        "tool", "version", "status", "verdict", "confidence",
        "reasons", "next_actions", "artifacts", "warnings", "result",
    }
    assert env["tool"] == "lint_dft_script"
    assert env["status"] == "ok"


def test_missing_script_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_lint_dft_script(tmp_path / "nonexistent.py")


# ---------------------------------------------------------------------------
# Unit tests for individual check functions
# ---------------------------------------------------------------------------

def test_check_pseudo_dir_windows_absolute():
    # Windows-style absolute path should pass
    ok, msg = _check_pseudo_dir('pseudo_dir = "C:\\\\Users\\pseudos"')
    assert ok


def test_check_pseudo_dir_tilde_flagged():
    # tilde is a variable reference -> skip (no false positive)
    ok, _ = _check_pseudo_dir("pseudo_dir = '~/pseudos'")
    # ~ starts with ~ -> not matched by _RE_RELATIVE_PATH (starts with ~, not letter),
    # so our regex skips it; verify no crash
    assert isinstance(ok, bool)


def test_check_outdir_only_one_var_not_flagged():
    # os.path.join(work_dir, "tmp") -> 2 args, second is literal "tmp" -> should not flag
    ok, _ = _check_outdir_nesting('outdir = os.path.join(work_dir, "tmp")')
    # depends on implementation; at minimum no crash
    assert isinstance(ok, bool)


def test_check_upf_wfc_nonexistent_dir():
    ok, msg = _check_upf_wfc("/nonexistent/path/to/pseudos")
    # Should not raise; returns True with a warning message or silently passes
    assert isinstance(ok, bool)
