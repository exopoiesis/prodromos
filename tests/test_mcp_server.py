"""Unit tests for the thin in-process MCP server.

These call the module-level ``tool_*`` wrappers DIRECTLY -- they never start the
blocking stdio loop. The server object is built at import time, so importing the
module already exercises registration.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

mcp_server = pytest.importorskip(
    "prodromos.mcp_server",
    reason="requires the optional 'mcp' dependency (pip install -e .[mcp])",
)

# tm-spec sibling checkout: prodromos/tests/ -> prodromos/ -> git/ -> tm-spec/
_PREFLIGHT_CASE = (
    Path(__file__).resolve().parents[2] / "tm-spec" / "examples" / "preflight_example.tm.yaml"
)

# The MCP tool surface we expect to expose (snake_case names).
_EXPECTED_TOOLS = {
    "plan",
    "from_inputs",
    "electron_parity",
    "spin_collapse",
    "endpoint_provenance",
    "symmetry_preflight",
    "vfe_preflight",
    "external_reference",
    "lint_dft_script",
    "h_barrier_readiness",
    "neb_advisor",
    "saddle_proximity",
    "multi_endpoint",
    "soap_cluster",
    "master_equation",
    "gp_neb",
    "adaptive_neb",
    "magnetic_parser",
    "magnetic_endpoint",
    "magnetic_band",
    "magnetic_recommend",
}


def test_import_and_server_instance():
    """Module imports and builds a FastMCP instance."""
    from mcp.server.fastmcp import FastMCP

    assert isinstance(mcp_server.server, FastMCP)
    assert callable(mcp_server.main)


def test_module_level_tool_functions_present():
    """Every expected tool has a module-level ``tool_<name>`` wrapper."""
    tool_fns = {name for name in dir(mcp_server) if name.startswith("tool_")}
    expected_fns = {f"tool_{name}" for name in _EXPECTED_TOOLS}
    assert expected_fns <= tool_fns, expected_fns - tool_fns
    # registry maps exactly the expected snake_case names
    assert set(mcp_server._TOOLS) == _EXPECTED_TOOLS


def test_all_tools_registered_with_fastmcp():
    """FastMCP's catalogue exposes exactly the expected tool names."""
    tools = asyncio.run(mcp_server.server.list_tools())
    names = {t.name for t in tools}
    assert names == _EXPECTED_TOOLS, (names ^ _EXPECTED_TOOLS)


def test_h_barrier_readiness_paper_grade():
    """A well-characterised index-1 H-saddle with ZPE -> PAPER_GRADE."""
    env = mcp_server.tool_h_barrier_readiness(
        barrier_eV=0.27,
        has_dft_freq=True,
        n_imag_modes=1,
        imag_mode_H_fraction=0.9,
        dZPE_eV=-0.12,
    )
    assert env["verdict"] == "PAPER_GRADE"


def test_endpoint_provenance_mlip_geometry():
    """MLIP-relaxed endpoint geometry -> NOT_AN_ENDPOINT_MLIP_GEOMETRY."""
    env = mcp_server.tool_endpoint_provenance(
        provenance="mlip_relaxed",
        bond_geometry_ok=True,
    )
    assert env["verdict"] == "NOT_AN_ENDPOINT_MLIP_GEOMETRY"


@pytest.mark.skipif(
    not _PREFLIGHT_CASE.exists(),
    reason="tm-spec examples not available beside the prodromos checkout",
)
def test_plan_on_preflight_example_go():
    """The bundled preflight example case routes to GO."""
    pytest.importorskip(
        "tm_spec",
        reason="plan requires tm-spec (pip install -e ../tm-spec / .[plan])",
    )
    env = mcp_server.tool_plan(str(_PREFLIGHT_CASE))
    assert env["tool"] == "plan"
    assert env["verdict"] == "GO"


def test_electron_parity_odd_count_envelope():
    """Smoke: odd electron count returns a well-formed envelope."""
    env = mcp_server.tool_electron_parity(symbol_counts={"Fe": 31, "S": 64, "H": 1})
    assert env["tool"]
    assert env["verdict"] is not None
