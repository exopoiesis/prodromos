import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.electron_parity_gate import (
    run_electron_parity_gate,
    counts_from_formula_tokens,
    counts_from_structure,
    parse_valence_overrides,
)


def test_pyrite_endpoint_odd_is_nspin2_mandatory():
    # Fe31 S64 H1: 31*16 + 64*6 + 1 = 881 -> ODD (the odd-electron blocker)
    env = run_electron_parity_gate({"Fe": 31, "S": 64, "H": 1})
    assert env["verdict"] == "NSPIN2_MANDATORY"
    assert env["confidence"] == "high"
    assert env["result"]["n_electrons"] == 881
    assert env["result"]["parity"] == "odd"
    assert env["result"]["nspin_required"] == 2
    assert env["result"]["total_magnetization_parity_constraint"] == "odd"
    assert env["result"]["min_abs_total_magnetization_uB"] == 1
    assert any("ODD" in r for r in env["reasons"])


def test_parity_robust_to_fe_valence_choice():
    # Even with Fe=8 valence, 31*8+384+1 = 633 -> still ODD
    env = run_electron_parity_gate({"Fe": 31, "S": 64, "H": 1},
                                   valence_overrides={"Fe": 8})
    assert env["result"]["n_electrons"] == 633
    assert env["verdict"] == "NSPIN2_MANDATORY"


def test_pyrite_pristine_even_with_fe_is_recommended():
    # Fe32 S64: 32*16 + 64*6 = 896 -> EVEN, but Fe present
    env = run_electron_parity_gate({"Fe": 32, "S": 64})
    assert env["verdict"] == "NSPIN2_RECOMMENDED"
    assert env["result"]["parity"] == "even"
    assert env["result"]["open_shell_tm"] == ["Fe"]


def test_closed_shell_no_tm_is_nspin1_ok():
    # H2O: 2*1 + 6 = 8 -> even, no TM
    env = run_electron_parity_gate({"H": 2, "O": 1})
    assert env["verdict"] == "NSPIN1_OK"
    assert env["result"]["nspin_required"] == 1


def test_charge_flips_parity():
    # Fe31 S64 H1 (+1 charge, remove 1 e) -> 880 even -> not mandatory
    env = run_electron_parity_gate({"Fe": 31, "S": 64, "H": 1}, charge=1.0)
    assert env["result"]["n_electrons"] == 880
    assert env["result"]["parity"] == "even"
    assert env["verdict"] == "NSPIN2_RECOMMENDED"  # still has Fe


def test_odd_valence_tm_co_parity():
    # Co=17 (odd valence). One Co + even rest -> odd
    env = run_electron_parity_gate({"Co": 1, "O": 1})  # 17+6=23 odd
    assert env["result"]["parity"] == "odd"
    assert env["verdict"] == "NSPIN2_MANDATORY"


def test_unknown_species_is_review():
    env = run_electron_parity_gate({"Xx": 2})
    assert env["verdict"] == "REVIEW"
    assert env["status"] == "ok"  # NO-GO/REVIEW are scientific verdicts, not failures
    assert "Xx" in env["result"]["unknown_species"]


def test_envelope_has_stable_keys():
    env = run_electron_parity_gate({"Fe": 31, "S": 64, "H": 1})
    assert set(env) == {
        "tool", "version", "status", "verdict", "confidence",
        "reasons", "next_actions", "artifacts", "warnings", "result",
    }
    assert env["tool"] == "electron_parity_gate"


def test_formula_token_parser():
    assert counts_from_formula_tokens(["Fe31", "S64", "H1"]) == {"Fe": 31, "S": 64, "H": 1}


def test_valence_override_parser():
    assert parse_valence_overrides(["Fe=16", "S=6"]) == {"Fe": 16, "S": 6}


def test_real_pyrite_endpoint_vs_pristine():
    base = Path(__file__).parent / "fixtures" / "pyr_VFe"
    endA = run_electron_parity_gate(counts_from_structure(base / "relaxed_endA.xyz"))
    assert endA["verdict"] == "NSPIN2_MANDATORY"   # the trap: H makes it odd
    pristine = run_electron_parity_gate(counts_from_structure(base / "relaxed_pristine.xyz"))
    assert pristine["result"]["parity"] == "even"  # pristine host is even -> nspin=1 looked fine
