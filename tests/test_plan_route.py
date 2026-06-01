"""Route-mode walk + emit tests, including tm-spec 0.3 preflight validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from prodromos.plan.emit import to_envelope, to_preflight_block
from prodromos.plan.interpret import walk
from prodromos.plan.policy import POLICY_GRAPH

# tm-spec example lives in the sibling checkout.
_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "tm-spec" / "examples" / "preflight_example.tm.yaml"
)

tm_validator = pytest.importorskip("tm_spec.validator")


def _load_example() -> dict:
    docs = tm_validator.load_doc(_EXAMPLE)
    return docs[0]


def _full_case() -> dict:
    return {
        "structure": {"formula": "Fe32S63H1", "space_group": {"symbol": "Pa-3"}},
        "calculation": {"level": {"spin": "none", "smearing": {"kind": "gaussian"}}},
        "workflow": {
            "endpoints": {
                "A": {"geometry_origin": "dft_relaxed", "E_eV": -1.0},
                "B": {"geometry_origin": "dft_relaxed", "E_eV": -1.0},
            }
        },
    }


def test_route_over_example_reaches_go():
    result = walk(POLICY_GRAPH, _load_example(), mode="route")
    assert result.mode == "route"
    assert result.verdict == "GO"
    assert result.terminal_node == "go_launch_neb"
    # endpoint-provenance + electron-parity actually executed
    gate_ids = [s.sanity_id for s in result.steps]
    assert "G09_geometry_origin" in gate_ids
    assert "G11_electron_parity" in gate_ids


def test_route_envelope_shape():
    result = walk(POLICY_GRAPH, _load_example(), mode="route")
    env = to_envelope(result)
    assert env["tool"] == "plan"
    assert env["verdict"] == "GO"
    assert env["result"]["plan_graph_version"]
    assert env["result"]["gate_trace"]
    assert env["next_actions"]


def test_route_non_dft_endpoint_needs_data():
    doc = _full_case()
    doc["workflow"]["endpoints"]["B"]["geometry_origin"] = "mlip_relaxed"
    result = walk(POLICY_GRAPH, doc, mode="route")
    assert result.verdict == "NEEDS_DATA"
    assert result.terminal_node == "need_relaxed_endpoints"
    assert "relax" in (result.next_action or "").lower()


def test_route_structure_only_needs_endpoints():
    # No workflow.endpoints -> endpoint-provenance adapter returns NeedsData,
    # route stops at NEEDS_DATA and recommends adding endpoints.
    doc = {"structure": {"formula": "Fe2S"}}
    result = walk(POLICY_GRAPH, doc, mode="route")
    assert result.verdict == "NEEDS_DATA"
    assert any("endpoint" in (a or "").lower() for a in result.next_actions + [result.next_action])


def test_route_even_closedshell_goes_straight_to_method_then_go():
    # An even-electron, TM-free system -> NSPIN1_OK skips the nspin choice and
    # goes directly to method_choice -> GO.
    doc = {
        "structure": {"formula": "H2O"},
        "calculation": {"level": {"spin": "none"}},
        "workflow": {"endpoints": {
            "A": {"geometry_origin": "dft_relaxed"},
            "B": {"geometry_origin": "dft_relaxed"},
        }},
    }
    result = walk(POLICY_GRAPH, doc, mode="route")
    assert result.verdict == "GO"
    parity = next(s for s in result.steps if s.sanity_id == "G11_electron_parity")
    assert parity.verdict == "NSPIN1_OK"
    chosen_nodes = [c["node"] for c in result.choices]
    assert "nspin_choice" not in chosen_nodes
    assert "method_choice" in chosen_nodes


def test_emitted_preflight_block_validates_against_tm_spec_0_3():
    """The emitted preflight block must splice into a tm-spec doc and validate.

    Splice the engine-produced block back into the example NEBCalculation (a
    full, valid 0.3 doc) and revalidate -- the realistic in -> out round-trip.
    """
    doc = _load_example()
    result = walk(POLICY_GRAPH, doc, mode="route")
    block = to_preflight_block(result)
    doc["preflight"] = block

    schema_errs, rule_issues = tm_validator.validate_doc(doc)
    errors = [f"{loc}: {msg}" for loc, msg in schema_errs]
    errors += [msg for level, msg in rule_issues if level == "error"]
    assert not errors, f"preflight block failed tm-spec 0.3 validation: {errors}"


def test_preflight_block_has_required_fields():
    result = walk(POLICY_GRAPH, _load_example(), mode="route")
    block = to_preflight_block(result)
    assert block["engine"]["name"] == "prodromos"
    assert block["verdict"] in {"GO", "NO-GO", "INVESTIGATE", "NEEDS_DATA"}
    assert block["confidence"] in {"low", "medium", "high"}
    for g in block["gates"]:
        assert g["id"].startswith("G")
        assert "verdict" in g
    assert "next_action" in block["plan"]


def test_tree_mode_is_stub():
    result = walk(POLICY_GRAPH, _load_example(), mode="tree")
    assert result.mode == "tree"
    assert any("STUB" in w for w in result.warnings)
