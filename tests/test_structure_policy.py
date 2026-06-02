"""Structure-mode policy graph (bare SinglePoint/Relax corpus cases).

A SinglePoint/Relax case with no workflow.endpoints (an OPTIMADE/MP/NOMAD import)
must route through the structure-mode triage graph -- electron-parity then
spin-collapse (fed from magnetic.magmoms_uB) -- and emit a magnetic-readiness
verdict, instead of the NEB graph's NEEDS_DATA(workflow.endpoints).
"""
from __future__ import annotations

from prodromos.plan.interpret import walk
from prodromos.plan.policy import (
    POLICY_GRAPH,
    STRUCTURE_POLICY_GRAPH,
    select_policy_graph,
)


def _struct_doc(formula: str, magmoms: dict | None = None, state: str | None = None) -> dict:
    doc: dict = {
        "spec": "tm-spec/0.3",
        "kind": "SinglePointCalculation",
        "structure": {"formula": formula, "geometry_origin": "dft_relaxed"},
        "calculation": {"method": "DFT"},
    }
    mag: dict = {}
    if state is not None:
        mag["state"] = state
    if magmoms is not None:
        mag["magmoms_uB"] = magmoms
    if mag:
        doc["magnetic"] = mag
    return doc


# -- graph selection --------------------------------------------------------

def test_singlepoint_without_endpoints_selects_structure_graph():
    assert select_policy_graph(_struct_doc("FeS2")) is STRUCTURE_POLICY_GRAPH


def test_relax_without_endpoints_selects_structure_graph():
    doc = _struct_doc("FeS2")
    doc["kind"] = "RelaxCalculation"
    assert select_policy_graph(doc) is STRUCTURE_POLICY_GRAPH


def test_singlepoint_with_endpoints_selects_neb_graph():
    doc = _struct_doc("FeS2")
    doc["workflow"] = {"endpoints": {"A": {"geometry_origin": "dft_relaxed"},
                                     "B": {"geometry_origin": "dft_relaxed"}}}
    assert select_policy_graph(doc) is POLICY_GRAPH


def test_neb_kind_selects_neb_graph():
    doc = _struct_doc("FeS2")
    doc["kind"] = "NEBCalculation"
    assert select_policy_graph(doc) is POLICY_GRAPH


# -- route outcomes on the structure graph ----------------------------------

def test_closed_shell_zns_go_nspin1():
    # Zn/S have no open-shell TM -> electron_parity NSPIN1_OK -> terminal directly.
    r = walk(STRUCTURE_POLICY_GRAPH, _struct_doc("ZnS", state="NM"), mode="route")
    assert r.verdict == "GO"
    assert r.terminal_node == "s_nspin1_closed_shell"
    assert len(r.steps) == 1  # only electron_parity ran


def test_diamagnetic_fe_collapses_to_nspin1():
    # pyrite: open-shell Fe -> NSPIN2_RECOMMENDED, but magmoms all ~0 -> collapsed.
    r = walk(STRUCTURE_POLICY_GRAPH, _struct_doc("FeS2", magmoms={str(i): 0.0 for i in range(12)}),
             mode="route")
    assert r.verdict == "GO"
    assert r.terminal_node == "s_nspin1_collapsed"
    assert [s.verdict for s in r.steps] == ["NSPIN2_RECOMMENDED", "NSPIN1_OK"]


def test_magnetic_persists_to_nspin2():
    # vaesite-like AFM: large local moments -> spin_collapse NSPIN2_REQUIRED.
    r = walk(STRUCTURE_POLICY_GRAPH,
             _struct_doc("NiS2", magmoms={"0": 1.3, "1": 1.3, "2": -1.3, "3": -1.3, "4": 0.0}),
             mode="route")
    assert r.verdict == "GO"
    assert r.terminal_node == "s_nspin2_magnetic"
    assert r.steps[-1].verdict == "NSPIN2_REQUIRED"


def test_no_moment_needs_data():
    # open-shell TM, NO magnetic block -> spin_collapse adapter -> NEEDS_DATA.
    r = walk(STRUCTURE_POLICY_GRAPH, _struct_doc("FeS2"), mode="route")
    assert r.verdict == "NEEDS_DATA"
    assert r.terminal_node == "s_spin_collapse"
    assert "magmoms" in r.next_action or "magnetization" in r.next_action


def test_structure_graph_tree_mode_does_not_crash():
    # tree mode is NEB-economics oriented; on a structure doc it must still build
    # (GO terminals -> leaves) without raising.
    r = walk(STRUCTURE_POLICY_GRAPH, _struct_doc("FeS2", magmoms={"0": 1.3, "1": -1.3}),
             mode="tree", budget_usd=100.0)
    assert r.mode == "tree"
    assert r.verdict in {"GO", "NO-GO", "INVESTIGATE"}
