"""Tests for the MLIP-confidence gate (§B)."""
from __future__ import annotations

from prodromos.mlip_confidence_gate import run_mlip_confidence_gate


def test_mgv2s4_vanadium_is_dft_required():
    # The documented failure: V near-degenerate 3d -> DFT_REQUIRED.
    env = run_mlip_confidence_gate({"Mg": 1, "V": 2, "S": 4}, migrant="Mg")
    assert env["verdict"] == "DFT_REQUIRED"
    assert "V" in env["result"]["near_degenerate_tm"]


def test_closed_shell_host_trusts_mlip():
    # TiS2 -> Ti4+ d0 closed shell -> nonmagnetic -> TRUST_MLIP.
    env = run_mlip_confidence_gate({"Ti": 1, "S": 2})
    assert env["verdict"] == "TRUST_MLIP"
    assert env["result"]["closed_shell"] is True


def test_no_tm_trusts_mlip():
    env = run_mlip_confidence_gate({"Li": 1, "Al": 1, "O": 2})
    assert env["verdict"] == "TRUST_MLIP"
    assert env["result"]["tm_species"] == []


def test_multivalent_redox_cathode_is_dft_required():
    # LiFePO4-like: Fe multivalent redox in a Li cathode hop -> DFT_REQUIRED.
    env = run_mlip_confidence_gate(
        {"Li": 1, "Fe": 1, "P": 1, "O": 4}, migrant="Li", band_gap_eV=3.0
    )
    assert env["verdict"] == "DFT_REQUIRED"
    assert "Fe" in env["result"]["multivalent_redox_tm"]


def test_clear_insulator_localized_moment_trusts_mlip_low():
    # Large-gap localized-moment host, no migrant context -> TRUST_MLIP (low conf).
    env = run_mlip_confidence_gate({"Fe": 2, "O": 3}, band_gap_eV=2.2)
    assert env["verdict"] == "TRUST_MLIP"
    assert env["confidence"] == "low"


def test_open_shell_no_context_is_review():
    env = run_mlip_confidence_gate({"Fe": 2, "S": 3})
    assert env["verdict"] == "REVIEW"


def test_cli_main_smoke(capsys):
    from prodromos.mlip_confidence_gate import main

    rc = main(["--symbols", "Mg1", "V2", "S4", "--migrant", "Mg"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DFT_REQUIRED" in out
