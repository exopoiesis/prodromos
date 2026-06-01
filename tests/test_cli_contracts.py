"""Tests for MCP-shaped production wrapper contracts."""
import numpy as np
import pytest

from prodromos.cli_contract import response_envelope
from prodromos.master_equation_kinetics import run_kinetic_network


def test_response_envelope_has_stable_keys():
    envelope = response_envelope(tool="x", verdict="GO", result={"a": 1})
    assert set(envelope) == {
        "tool",
        "version",
        "status",
        "verdict",
        "confidence",
        "reasons",
        "next_actions",
        "artifacts",
        "warnings",
        "result",
    }
    assert envelope["tool"] == "x"
    assert envelope["status"] == "ok"
    assert envelope["verdict"] == "GO"


def test_run_kinetic_network_returns_envelope():
    barriers = np.full((2, 2), np.inf)
    barriers[0, 1] = barriers[1, 0] = 0.043
    envelope = run_kinetic_network(barriers, site_labels=["A", "B"], verbose=False)
    assert envelope["tool"] == "analyze_kinetic_network"
    assert envelope["verdict"] == "KINETIC_NETWORK_ANALYZED"
    assert envelope["result"]["n_sites"] == 2
    assert envelope["result"]["arrhenius_E_a_eff_meV"] == pytest.approx(43.0)
