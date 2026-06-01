"""The Fe-S V_Fe+H pre-flight policy graph (declarative Python literals).

Encodes the three QUICK_START scenarios as edges over gate verdicts:
  A. from scratch (structure only)              -> NEEDS_DATA (relax endpoints)
  B. have endA only                             -> NEEDS_DATA (need endB)
  C. have endA + endB [+ band]                  -> run the cheap gates, then GO

Gate ordering is cheapest-first (all $0, but ordered so a hard methodology
blocker -- bad endpoint provenance, wrong spin parity -- fires before the
method choice). Choice nodes (nspin, method) BRANCH only in tree mode; in route
the interpreter picks the myopic-best option. Terminal nodes are the four
pre-flight verdicts: GO / NO-GO / INVESTIGATE / NEEDS_DATA.

This is data-as-code (consilium S1): edge predicates are pure one-field
booleans; all real logic lives inside the gates.
"""
from __future__ import annotations

from prodromos.plan.graph import (
    Edge,
    Node,
    PolicyGraph,
    assert_dag,
    option_is,
    verdict_in,
)


def build_policy() -> PolicyGraph:
    """Construct and statically validate the Fe-S V_Fe+H pre-flight graph."""
    nodes: dict[str, Node] = {}

    def add(node: Node) -> None:
        nodes[node.id] = node

    # -- Gate 1: endpoint provenance (G09) -- cheapest hard blocker -----------
    # A non-dft_relaxed endpoint makes the energy invalid: must relax first.
    add(Node(
        id="endpoint_provenance",
        kind="gate",
        gate="endpoint-provenance",
        inputs=["geometry_origin", "energy_eV", "label"],
        what="are the endpoint geometries DFT-relaxed (energies comparable)?",
        edges=(
            Edge("electron_parity", verdict_in("ENDPOINT_VALID"),
                 label="endpoints valid -> check spin parity"),
            Edge("need_relaxed_endpoints",
                 verdict_in("NOT_AN_ENDPOINT_MLIP_GEOMETRY", "REVIEW"),
                 label="endpoint not DFT-relaxed -> needs relaxation"),
        ),
    ))

    # -- Gate 2: electron parity (G11) ---------------------------------------
    add(Node(
        id="electron_parity",
        kind="gate",
        gate="electron-parity",
        inputs=["symbol_counts", "charge", "metallic", "smearing"],
        what="does electron-count parity force nspin=2?",
        edges=(
            Edge("spin_collapse",
                 verdict_in("NSPIN2_MANDATORY", "NSPIN2_RECOMMENDED"),
                 label="spin polarisation indicated -> spin-collapse check"),
            Edge("method_choice", verdict_in("NSPIN1_OK"),
                 label="closed shell -> nspin=1, pick method"),
            Edge("parity_review", verdict_in("REVIEW"),
                 label="cannot determine parity -> investigate"),
        ),
    ))

    # -- Gate 3: spin-collapse (G02) -- honest chance-node -------------------
    # Parity says nspin=2 is REQUIRED a-priori, but the operational question is
    # whether the seeded local TM moment actually COLLAPSES (so nspin=1 == nspin=2,
    # smooth PES) or PERSISTS (ferrimagnet, nspin=2 mandatory). Some Fe-S systems
    # collapse (mack, pyrite V_Fe+H), others persist (pentlandite ~1.8 uB/TM).
    # In tree mode this branches over SPIN_COLLAPSE_PRIOR (~0.5/0.5); in route it
    # consumes one cheap nspin=2 single-point (NEEDS_DATA on a bare pre-flight case).
    add(Node(
        id="spin_collapse",
        kind="gate",
        gate="spin-collapse",
        inputs=["mabs", "n_tm", "mabs_per_tm"],
        what="does the local TM moment collapse (nspin=1 ok) or persist (nspin=2)?",
        edges=(
            Edge("nspin1_after_collapse", verdict_in("NSPIN1_OK"),
                 label="moment collapsed -> nspin=1 production"),
            Edge("nspin2_after_collapse", verdict_in("NSPIN2_REQUIRED"),
                 label="moment persists -> nspin=2 production"),
        ),
    ))

    # -- Choice pins: each spin-collapse outcome fixes nspin, then picks method.
    # Single-option "choices" so the tree builder records the nspin context on the
    # path (drives leaf labels + calibration key) without an extra branch.
    add(Node(
        id="nspin1_after_collapse",
        kind="choice",
        options=["nspin1"],
        what="moment collapsed: run restricted (nspin=1) production",
        edges=(
            Edge("method_choice", option_is("nspin1"),
                 label="nspin=1 production -> pick NEB method"),
        ),
    ))
    add(Node(
        id="nspin2_after_collapse",
        kind="choice",
        options=["nspin2"],
        what="moment persists: run spin-polarised (nspin=2) production",
        edges=(
            Edge("method_choice", option_is("nspin2"),
                 label="nspin=2 production -> pick NEB method"),
        ),
    ))

    # -- Choice: NEB method {band, dimer, string} ----------------------------
    add(Node(
        id="method_choice",
        kind="choice",
        options=["band", "dimer", "string"],
        what="which NEB/saddle method family to launch",
        edges=(
            Edge("go_launch_neb", option_is("band"),
                 label="band-NEB: well-posed endpoints, launch"),
            Edge("go_launch_neb", option_is("dimer"),
                 label="dimer+chemical-RC: degenerate ridge"),
            Edge("go_launch_neb", option_is("string"),
                 label="string method: asymmetric endpoints"),
        ),
    ))

    # -- Terminals -----------------------------------------------------------
    add(Node(
        id="go_launch_neb",
        kind="terminal",
        terminal="GO",
        what="launch the recommended NEB from the two DFT-relaxed endpoints",
    ))
    add(Node(
        id="need_relaxed_endpoints",
        kind="terminal",
        terminal="NEEDS_DATA",
        what="relax the endpoint(s) under DFT before any barrier comparison",
    ))
    add(Node(
        id="parity_review",
        kind="terminal",
        terminal="INVESTIGATE",
        what="electron count / valence is ambiguous; resolve before committing compute",
    ))

    graph = PolicyGraph(nodes=nodes, root="endpoint_provenance")
    assert_dag(graph)
    return graph


# Module-level singleton (validated at import).
POLICY_GRAPH = build_policy()
