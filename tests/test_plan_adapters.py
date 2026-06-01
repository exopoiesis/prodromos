"""Tests for tm_doc_to_gate_inputs adapters."""
from __future__ import annotations

from prodromos.plan.adapters import (
    NeedsData,
    parse_formula,
    tm_doc_to_gate_inputs,
    to_electron_parity_inputs,
    to_endpoint_provenance_inputs,
)


def test_parse_formula_basic():
    assert parse_formula("Fe31S64H1") == {"Fe": 31, "S": 64, "H": 1}


def test_parse_formula_bare_symbol_counts_one():
    assert parse_formula("FeS") == {"Fe": 1, "S": 1}


def test_parse_formula_symbolic_returns_empty():
    assert parse_formula("(Fe+S x 4)") == {}
    assert parse_formula("") == {}


def _example_doc() -> dict:
    return {
        "structure": {"formula": "Fe32S63H1",
                      "space_group": {"symbol": "Pa-3"}},
        "calculation": {"level": {"spin": "none",
                                  "smearing": {"kind": "gaussian"}}},
        "workflow": {
            "endpoints": {
                "A": {"geometry_origin": "dft_relaxed", "E_eV": -128055.54},
                "B": {"geometry_origin": "dft_relaxed", "E_eV": -128055.54},
            }
        },
    }


def test_electron_parity_extraction():
    out = to_electron_parity_inputs(_example_doc())
    assert isinstance(out, dict)
    assert out["symbol_counts"] == {"Fe": 32, "S": 63, "H": 1}
    assert out["metallic"] is True
    assert out["smearing"] == "gaussian"


def test_electron_parity_needs_data_when_no_formula():
    out = to_electron_parity_inputs({"structure": {}})
    assert isinstance(out, NeedsData)
    assert "structure.formula" in out.missing[0]


def test_endpoint_provenance_extraction_all_dft():
    out = to_endpoint_provenance_inputs(_example_doc())
    assert isinstance(out, dict)
    assert out["provenance"] == "dft_relaxed"


def test_endpoint_provenance_picks_worst_endpoint():
    doc = _example_doc()
    doc["workflow"]["endpoints"]["B"]["geometry_origin"] = "mlip_relaxed"
    out = to_endpoint_provenance_inputs(doc)
    assert isinstance(out, dict)
    assert out["provenance"] == "mlip_relaxed"
    assert out["label"] == "B"


def test_endpoint_provenance_needs_data_when_missing_origin():
    doc = _example_doc()
    del doc["workflow"]["endpoints"]["A"]["geometry_origin"]
    out = to_endpoint_provenance_inputs(doc)
    assert isinstance(out, NeedsData)
    assert any("geometry_origin" in m for m in out.missing)


def test_endpoint_provenance_needs_data_when_no_endpoints():
    out = to_endpoint_provenance_inputs({"structure": {"formula": "Fe2S"}})
    assert isinstance(out, NeedsData)
    assert "workflow.endpoints" in out.missing


def test_dft_output_gates_return_needs_data():
    doc = _example_doc()
    for gate in ("spin-collapse", "magnetic-endpoint", "neb-advisor",
                 "saddle-proximity", "h-barrier-readiness", "symmetry-preflight"):
        out = tm_doc_to_gate_inputs(doc, gate)
        assert isinstance(out, NeedsData), f"{gate} should need data"
        assert out.recommend


def test_unknown_gate_returns_needs_data():
    out = tm_doc_to_gate_inputs(_example_doc(), "no-such-gate")
    assert isinstance(out, NeedsData)
