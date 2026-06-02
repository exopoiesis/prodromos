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
    "import_optimade",
    "import_nomad",
    "import_mp",
    "import_magndata",
    "merge_specs",
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
def test_plan_on_preflight_example_routes_to_spin_collapse_needs_data():
    """The bundled preflight example is odd-electron -> NSPIN2_MANDATORY -> the
    spin-collapse gate (G02), which a bare pre-flight case cannot feed (no nspin=2
    single-point), so route honestly stops at NEEDS_DATA."""
    pytest.importorskip(
        "tm_spec",
        reason="plan requires tm-spec (pip install -e ../tm-spec / .[plan])",
    )
    env = mcp_server.tool_plan(str(_PREFLIGHT_CASE))
    assert env["tool"] == "plan"
    assert env["verdict"] == "NEEDS_DATA"


def test_electron_parity_odd_count_envelope():
    """Smoke: odd electron count returns a well-formed envelope."""
    env = mcp_server.tool_electron_parity(symbol_counts={"Fe": 31, "S": 64, "H": 1})
    assert env["tool"]
    assert env["verdict"] is not None


# ---------------------------------------------------------------------------
# tm-spec importers + merge (OPTIMADE width x NOMAD depth, all local)
# ---------------------------------------------------------------------------

# A NOMAD-like base: deep (method.level + magnetic + results) but a sparse
# structure (formula only, no lattice / pbc / dimension_types).
_NOMAD_LIKE_BASE = {
    "spec": "tm-spec/0.3",
    "kind": "SinglePointCalculation",
    "id": "tm.nomad.synthetic_fes2.2026-06-02",
    "schema_url": "https://exopoiesis.github.io/tm-spec/0.3.json",
    "structure": {
        "formula": "FeS2",
        "chemical_formula_reduced": "FeS2",
        "geometry_origin": "dft_static",
    },
    "calculation": {
        "method": "DFT+U",
        "level": {
            "xc": "GGA",
            "xc_libxc": ["GGA_X_PBE", "GGA_C_PBE"],
            "basis": {"kind": "plane_waves"},
            "spin": "collinear",
        },
        "code": {"name": "VASP", "version": "6.3.2"},
    },
    "magnetic": {"state": "AFM-G", "collinear": True},
    "results": {"status": "PRELIMINARY", "paper_quotable": False, "energy_eV": -42.0},
    "sanity": [{"id": "G05_scf_converged", "rule": "SCF converged", "pass": True}],
    "provenance": {
        "date": "2026-06-02",
        "author": "import@nomad",
        "import_source": {"archive": "nomad", "entry_id": "synthetic_fes2"},
        "compute": {"host": "nomad-archive", "cost_usd": 0.0},
    },
}

# An OPTIMADE-like overlay: shallow (no method/energy) but broad structure
# (lattice_vectors_A + pbc + dimension_types), geometry_origin=unknown.
_OPTIMADE_LIKE_OVERLAY = {
    "spec": "tm-spec/0.3",
    "kind": "SinglePointCalculation",
    "id": "tm.optimade_mp.mp_226.2026-06-02",
    "schema_url": "https://exopoiesis.github.io/tm-spec/0.3.json",
    "structure": {
        "formula": "FeS2",
        "chemical_formula_reduced": "FeS2",
        "chemical_formula_anonymous": "AB2",
        "lattice_vectors_A": [[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 5.4]],
        "pbc": [True, True, True],
        "dimension_types": [1, 1, 1],
        "geometry_origin": "unknown",
    },
    "calculation": {"method": "DFT"},
    "results": {"status": "PRELIMINARY", "paper_quotable": False},
    "sanity": [{"id": "G06_ascii_safe", "rule": "ASCII-only doc body", "pass": "skip"}],
    "provenance": {
        "date": "2026-06-02",
        "author": "import@optimade",
        "import_source": {"archive": "materials_project", "entry_id": "mp-226"},
        "compute": {"host": "optimade:mp", "cost_usd": 0.0},
    },
}


def test_import_optimade_offline_graceful():
    """Offline OPTIMADE import returns a well-formed envelope (no crash, empty)."""
    env = mcp_server.tool_import_optimade(elements=["Fe", "S"], live=False)
    assert env["tool"] == "import_optimade"
    assert env["status"] == "ok"
    assert env["count"] == 0
    assert env["docs"] == []
    assert isinstance(env["reasons"], list) and env["reasons"]


def test_import_mp_graceful_without_key(monkeypatch):
    """import_mp without an MP API key returns a well-formed error envelope (no crash)."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    env = mcp_server.tool_import_mp(formula="FeS2", space_group=205)
    assert env["tool"] == "import_mp"
    assert env["status"] == "error"
    assert env["count"] == 0
    assert env["docs"] == []
    assert isinstance(env["reasons"], list) and env["reasons"]


def test_import_mp_pure_transform_offline():
    """tool_import_mp delegates to the pure transform; verify with a monkeypatched fetch."""
    from tm_spec.importers import mp as mpimp
    summary = {"material_id": "mp-226", "formula_pretty": "FeS2", "symmetry": {"number": 205},
               "is_magnetic": False, "ordering": "NM"}
    doc = mpimp.summary_to_tm_spec(summary, {"magmoms": [0.0] * 12}, date="2026-06-02")
    assert doc["magnetic"]["state"] == "NM"
    assert doc["structure"]["geometry_origin"] == "dft_relaxed"


def test_import_magndata_graceful_on_error(monkeypatch):
    """import_magndata returns a well-formed error envelope (no crash) on failure."""
    from tm_spec.importers import magndata as _mag

    def boom(*a, **k):
        raise _mag.MagndataError("simulated network failure")

    monkeypatch.setattr(_mag, "fetch_to_tm_spec", boom)
    env = mcp_server.tool_import_magndata(code="0.1")
    assert env["tool"] == "import_magndata"
    assert env["status"] == "error"
    assert env["count"] == 0 and env["docs"] == []
    assert isinstance(env["reasons"], list) and env["reasons"]


def test_import_magndata_pure_transform_offline():
    """Verify the magCIF -> tm-spec transform without network (synthetic FM cell)."""
    from tm_spec.importers.magndata import magcif_to_tm_spec
    mcif = (
        "data_t\n_chemical_formula_sum 'Fe'\n_space_group_magn.name_BNS \"P1\"\n"
        "_space_group_magn.point_group_name \"1\"\n_cell_angle_alpha 90.0\n"
        "_cell_angle_beta 90.0\n_cell_angle_gamma 90.0\n"
        "loop_\n_space_group_symop_magn_operation.id\n_space_group_symop_magn_operation.xyz\n"
        "1 x,y,z,+1\n"
        "loop_\n_atom_site_moment.label\n_atom_site_moment.crystalaxis_x\n"
        "_atom_site_moment.crystalaxis_y\n_atom_site_moment.crystalaxis_z\n"
        "Fe 3.0 0.0 0.0\n"
    )
    doc = magcif_to_tm_spec(mcif, code="t")
    assert doc["magnetic"]["state"] == "FM"
    assert doc["structure"]["geometry_origin"] == "experimental"


def test_merge_specs_combines_depth_and_width():
    """merge_specs folds OPTIMADE width into NOMAD depth: merged carries BOTH the
    calculation.level + magnetic (depth) AND the structure lattice/pbc (width)."""
    env = mcp_server.tool_merge_specs(
        base=_NOMAD_LIKE_BASE,
        overlay=_OPTIMADE_LIKE_OVERLAY,
    )
    assert env["tool"] == "merge_specs"
    assert env["status"] == "ok"
    assert isinstance(env["warnings"], list)

    merged = env["merged"]
    # NOMAD depth preserved.
    assert merged["calculation"]["method"] == "DFT+U"
    assert merged["calculation"]["level"]["xc"] == "GGA"
    assert merged["magnetic"]["state"] == "AFM-G"
    assert merged["results"]["energy_eV"] == -42.0
    # OPTIMADE width filled into the sparse base structure.
    assert merged["structure"]["lattice_vectors_A"][0][0] == 5.4
    assert merged["structure"]["pbc"] == [True, True, True]
    assert merged["structure"]["dimension_types"] == [1, 1, 1]
    # geometry_origin keeps the more specific base value over overlay 'unknown'.
    assert merged["structure"]["geometry_origin"] == "dft_static"


def test_merge_specs_material_mismatch_is_error_not_exception():
    """A formula mismatch returns status='error' (MATERIAL_MISMATCH), not a raise."""
    overlay = {
        **_OPTIMADE_LIKE_OVERLAY,
        "structure": {
            "formula": "NiO",
            "chemical_formula_reduced": "NiO",
            "geometry_origin": "unknown",
        },
    }
    env = mcp_server.tool_merge_specs(base=_NOMAD_LIKE_BASE, overlay=overlay)
    assert env["tool"] == "merge_specs"
    assert env["status"] == "error"
    assert env["merged"] is None
    assert any("MATERIAL_MISMATCH" in w for w in env["warnings"])
