"""GATE_REGISTRY -- single source of truth crosswalking the prodromos gate
subcommands to the canonical tm-spec sanity IDs (docs/gate-registry.md) and a
$0/predictive cost.

Two numbering schemes (prodromos subcommand / N-id, and tm-spec ``Gxx``) WILL
drift unless they share one table (consilium D4). This module is that table.

Each :class:`GateSpec` carries an importable ``run_fn`` callable so the
``plan`` interpreter can EXECUTE a gate in route mode without subprocesses.
A few gates are exposed under non-``run_*`` entry names (the magnetic family);
their callables are still recorded here for completeness even when the first
increment routes them to NEEDS_DATA (they consume parsed DFT outputs that a
pre-flight case document does not yet carry).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from prodromos.electron_parity_gate import run_electron_parity_gate
from prodromos.endpoint_provenance_gate import run_endpoint_provenance_gate
from prodromos.external_reference_gate import run_external_reference_gate
from prodromos.h_barrier_paper_readiness import run_h_barrier_paper_readiness
from prodromos.lint_dft_script import run_lint_dft_script
from prodromos.magnetic_band_gate import analyze_band_images
from prodromos.magnetic_endpoint_gate import endpoint_magnetic_gate
from prodromos.magnetic_output_parser import parse_output_file
from prodromos.magnetic_recommendation import build_recommendation
from prodromos.neb_method_advisor import run_neb_method_advisor
from prodromos.saddle_proximity_gate import run_saddle_proximity_gate
from prodromos.spin_collapse_verdict import run_spin_collapse_verdict
from prodromos.symmetry_preflight_general import run_symmetry_l1


@dataclass(frozen=True)
class GateSpec:
    """One row of the gate crosswalk."""

    subcommand: str          # subcommand as registered in prodromos.__main__.SUBCOMMANDS
    module: str              # dotted module path
    run_fn: Callable         # importable core entry point (NOT a subprocess)
    sanity_id: str           # canonical tm-spec Gxx id (docs/gate-registry.md)
    cost_usd: float          # rough order-of-magnitude predictive cost
    what: str                # one-line description of the check


# Canonical IDs follow tm-spec docs/gate-registry.md (the spec-side source of
# truth for the `id` field). Only gates that have a Gxx AND are consumed by the
# pre-flight planner are listed.
GATE_REGISTRY: dict[str, GateSpec] = {
    "spin-collapse": GateSpec(
        subcommand="spin-collapse",
        module="prodromos.spin_collapse_verdict",
        run_fn=run_spin_collapse_verdict,
        sanity_id="G02_moment_not_collapsed",
        cost_usd=0.0,
        what="local TM moment collapses vs persists -> decides nspin",
    ),
    "symmetry-preflight": GateSpec(
        subcommand="symmetry-preflight",
        module="prodromos.symmetry_preflight_general",
        run_fn=run_symmetry_l1,
        sanity_id="G03_endpoints_distinct",
        cost_usd=0.0,
        what="endpoints A,B are distinct basins (Hungarian L1 same-basin predictor)",
    ),
    "endpoint-provenance": GateSpec(
        subcommand="endpoint-provenance",
        module="prodromos.endpoint_provenance_gate",
        run_fn=run_endpoint_provenance_gate,
        sanity_id="G09_geometry_origin",
        cost_usd=0.0,
        what="endpoint geometry_origin is dft_relaxed (energy validity)",
    ),
    "electron-parity": GateSpec(
        subcommand="electron-parity",
        module="prodromos.electron_parity_gate",
        run_fn=run_electron_parity_gate,
        sanity_id="G11_electron_parity",
        cost_usd=0.0,
        what="electron-count parity vs nspin choice (odd+fixed-occ -> nspin=2)",
    ),
    "magnetic-endpoint": GateSpec(
        subcommand="magnetic-endpoint",
        module="prodromos.magnetic_endpoint_gate",
        run_fn=endpoint_magnetic_gate,
        sanity_id="G12_endpoint_single_sheet",
        cost_usd=0.0,
        what="endpoints lie on one magnetic sheet (GO / NO-GO_SINGLE_SHEET)",
    ),
    "magnetic-band": GateSpec(
        subcommand="magnetic-band",
        module="prodromos.magnetic_band_gate",
        run_fn=analyze_band_images,
        sanity_id="G13_band_single_sheet",
        cost_usd=0.0,
        what="no spin-sheet crossing inside the band (post-pilot)",
    ),
    "magnetic-parser": GateSpec(
        subcommand="magnetic-parser",
        module="prodromos.magnetic_output_parser",
        run_fn=parse_output_file,
        sanity_id="G14_magnetization_settled",
        cost_usd=0.0,
        what="moment converged, not still drifting (drift window)",
    ),
    "magnetic-recommend": GateSpec(
        subcommand="magnetic-recommend",
        module="prodromos.magnetic_recommendation",
        run_fn=build_recommendation,
        sanity_id="G15_provenance_consistent",
        cost_usd=0.0,
        what="compared energies share (U,nspin,functional,ecut,kpts)",
    ),
    "neb-advisor": GateSpec(
        subcommand="neb-advisor",
        module="prodromos.neb_method_advisor",
        run_fn=run_neb_method_advisor,
        sanity_id="G16_method_recommendation",
        cost_usd=0.0,
        what="NEB failure signature -> method family (band/dimer/string)",
    ),
    "saddle-proximity": GateSpec(
        subcommand="saddle-proximity",
        module="prodromos.saddle_proximity_gate",
        run_fn=run_saddle_proximity_gate,
        sanity_id="G17_saddle_on_path",
        cost_usd=0.0,
        what="saddle is the intended transfer, not off-path (DIRECT_TRANSFER_OK)",
    ),
    "lint-dft-script": GateSpec(
        subcommand="lint-dft-script",
        module="prodromos.lint_dft_script",
        run_fn=run_lint_dft_script,
        sanity_id="G18_dft_script_lint",
        cost_usd=0.0,
        what="4 recurring QE/ABACUS deploy bugs (abs pseudo_dir, outdir, clean-read, wfc>0)",
    ),
    "external-reference": GateSpec(
        subcommand="external-reference",
        module="prodromos.external_reference_gate",
        run_fn=run_external_reference_gate,
        sanity_id="G19_external_reference",
        cost_usd=0.0,
        what="public DFT reference exists (NOMAD/OPTIMADE)",
    ),
    "h-barrier-readiness": GateSpec(
        subcommand="h-barrier-readiness",
        module="prodromos.h_barrier_paper_readiness",
        run_fn=run_h_barrier_paper_readiness,
        sanity_id="G20_h_barrier_paper_grade",
        cost_usd=0.0,
        what="barrier has index-1 H-mode + dZPE (PAPER_GRADE vs ELECTRONIC_ONLY)",
    ),
}


def get(subcommand: str) -> GateSpec:
    """Return the :class:`GateSpec` for ``subcommand`` or raise KeyError."""
    return GATE_REGISTRY[subcommand]


def sanity_id_for(subcommand: str) -> str:
    """Return the canonical tm-spec Gxx id for a gate subcommand."""
    return GATE_REGISTRY[subcommand].sanity_id
