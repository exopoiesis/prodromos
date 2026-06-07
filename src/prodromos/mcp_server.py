"""Thin in-process stdio MCP server for prodromos.

Exposes the $0 pre-flight gates, the ``plan`` orchestrator, the
``from-inputs`` onboarding converter, and the tm-spec OPTIMADE/NOMAD importers
plus the local ``merge`` engine (``import_optimade`` / ``import_nomad`` /
``merge_specs``) as MCP tools over stdio (no proxy, no Docker; the importers are
the only tools that touch the network, and degrade softly when offline). Every
gate's core is already a pure ``run_*(...) -> dict``
returning a ``cli_contract.response_envelope``; each MCP tool here is a thin,
typed module-level wrapper (``tool_<name>``) that calls the corresponding core
and returns the envelope (FastMCP serializes the dict to JSON for the LLM).

Design for testability: the wrappers are plain module-level functions
registered via ``server.tool()(tool_<name>)``. Unit tests call ``tool_*``
directly -- they never start the blocking stdio loop. ``main()`` only builds
the server and pumps stdio.
"""
from __future__ import annotations

import functools
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP

from prodromos.cli_contract import response_envelope

server = FastMCP("prodromos")


def _log(msg: str) -> None:
    """Log to STDERR (stdout is the MCP protocol channel -- never write there)."""
    print(f"[prodromos-mcp] {msg}", file=sys.stderr, flush=True)


# ===========================================================================
# 1. plan -- the primary orchestrator
# ===========================================================================
def tool_plan(
    case_path: str,
    mode: str = "route",
    budget_usd: float | None = None,
    top_k: int = 5,
    code: str = "auto",
) -> dict:
    """Pre-flight orchestrator over a tm-spec/0.3 case (or a raw QE/ABACUS
    input, auto-converted).

    Validates the tm-spec case document (gate-0 INVALID_CASE), or, when given a
    bare engine input file/dir (a QE ``.in`` / ABACUS ``INPUT`` / ABACUS run
    directory), auto-converts it on the fly via ``from-inputs`` first; then
    walks the policy graph and emits a response envelope.

    Parameters
    ----------
    case_path : path to a ``.tm.yaml`` / ``.json`` tm-spec case, OR a QE ``.in``
        / ABACUS ``INPUT`` / ABACUS run directory (auto-converted).
    mode : ``"route"`` (execute the $0 gates, recommend ONE next step;
        default) or ``"tree"`` (scored strategy tree, Bellman expectimax + CVaR).
    budget_usd : remaining budget (USD); tree mode uses beta=cost_run/budget for
        CVaR tail control. Omit for pure expected-value scoring.
    top_k : tree mode -- keep only the top-K ranked strategies (default 5).
    code : code for auto-conversion of a bare input (``"qe"`` | ``"abacus"`` |
        ``"auto"``, default auto-detect).

    Returns the ``plan`` response envelope; in ``tree`` mode the envelope's
    ``result`` carries the ranked ``strategies``.
    """
    from prodromos.plan.cli import _load_and_validate
    from prodromos.plan.emit import to_envelope
    from prodromos.plan.interpret import walk
    from prodromos.plan.policy import select_policy_graph

    doc, err, extra_reasons = _load_and_validate(Path(case_path), code=code)
    if err is not None:
        return err

    result = walk(
        select_policy_graph(doc),
        doc,
        mode=mode,
        budget_usd=budget_usd,
        top_k=top_k,
    )
    payload = to_envelope(result)
    if extra_reasons and isinstance(payload.get("reasons"), list):
        payload["reasons"] = extra_reasons + payload["reasons"]
    return payload


# ===========================================================================
# 2. from_inputs -- onboarding converter
# ===========================================================================
def tool_from_inputs(
    path: str,
    code: str = "auto",
    kind: str | None = None,
    date: str | None = None,
) -> dict:
    """Onboard a scientist's own engine input into a tm-spec/0.3 document.

    Convert a Quantum ESPRESSO ``pw.x`` ``.in`` file, an ABACUS ``INPUT`` file,
    or an ABACUS run directory (INPUT + STRU [+ KPT]) into a tm-spec/0.3 starter
    document (dict) that can be fed straight into ``plan``. Genuinely
    human-needed fields are left as ``[TODO_HUMAN]`` placeholders.

    Parameters
    ----------
    path : QE ``.in`` file, ABACUS ``INPUT`` file, or ABACUS run directory.
    code : ``"qe"`` | ``"abacus"`` | ``"auto"`` (default auto-detect).
    kind : optional tm-spec kind override (else derived from ``calculation``).
    date : ISO date string (e.g. ``2026-06-02``) used in ``id`` and
        ``provenance.date``. Defaults to the deterministic placeholder
        ``"YYYY-MM-DD"`` so output is reproducible.

    Returns the tm-spec/0.3 document as a dict.
    """
    from prodromos.from_inputs import SPEC_VERSION, convert_to_tmspec

    doc = convert_to_tmspec(
        path,
        code=code,
        kind=kind,
        date=date if date is not None else "YYYY-MM-DD",
    )
    return response_envelope(
        tool="from_inputs",
        verdict=f"tm-spec/{SPEC_VERSION}",
        confidence="medium",
        reasons=[f"converted {path!r} (code={code}) to a tm-spec/{SPEC_VERSION} starter stub"],
        next_actions=["complete the [TODO_HUMAN] fields, then run the `plan` tool"],
        result=doc,
    )


# ===========================================================================
# 3. standalone gates -- thin wrappers over run_*(...)
# ===========================================================================
def tool_electron_parity(
    symbol_counts: dict[str, int],
    charge: float = 0.0,
    valence_overrides: dict[str, int] | None = None,
    metallic: bool = False,
    smearing: str | None = None,
) -> dict:
    """G11 -- electron-count parity vs nspin choice.

    ``symbol_counts`` e.g. ``{"Fe": 31, "S": 64, "H": 1}``. Odd electron count
    with fixed occupations forces nspin=2; with metallic smearing nspin=1 may be
    acceptable (verify with a collapse test).
    """
    from prodromos.electron_parity_gate import run_electron_parity_gate

    return run_electron_parity_gate(
        symbol_counts,
        charge=charge,
        valence_overrides=valence_overrides,
        metallic=metallic,
        smearing=smearing,
    )


def tool_spin_collapse(
    mabs: float | None = None,
    n_tm: int | None = None,
    mabs_per_tm: float | None = None,
    mtot: float | None = None,
    parity: str | None = None,
    threshold: float = 0.5,
) -> dict:
    """G02 -- does the local TM moment collapse or persist (decides nspin).

    Provide EITHER (``mabs`` absolute magnetization + ``n_tm`` TM-atom count) OR
    a pre-computed ``mabs_per_tm``. ``mtot`` / ``parity`` are optional context.
    """
    from prodromos.spin_collapse_verdict import run_spin_collapse_verdict

    return run_spin_collapse_verdict(
        mabs=mabs,
        n_tm=n_tm,
        mabs_per_tm=mabs_per_tm,
        mtot=mtot,
        parity=parity,
        threshold=threshold,
    )


def tool_endpoint_provenance(
    provenance: str,
    bond_geometry_ok: bool | None = None,
    energy_eV: float | None = None,
    label: str | None = None,
) -> dict:
    """G09 -- endpoint geometry_origin must be ``dft_relaxed`` for energy
    validity.

    Any ``provenance`` other than ``"dft_relaxed"`` (e.g. ``"mlip_relaxed"``)
    yields NOT_AN_ENDPOINT_MLIP_GEOMETRY. ``bond_geometry_ok`` is advisory and
    does NOT change the verdict.
    """
    from prodromos.endpoint_provenance_gate import run_endpoint_provenance_gate

    return run_endpoint_provenance_gate(
        provenance,
        bond_geometry_ok=bond_geometry_ok,
        energy_eV=energy_eV,
        label=label,
    )


def tool_symmetry_preflight(
    pristine_path: str,
    end_a_path: str,
    triple_path: str,
    mineral_name: str = "system",
    known_dft_barrier_meV: float | None = None,
    global_disp_threshold: float = 0.3,
    global_disp_per_atom_A: float = 1.0,
) -> dict:
    """G03 -- endpoints A,B are distinct basins (Hungarian L1 same-basin
    predictor).

    File inputs: ``pristine_path`` (relaxed pristine structure), ``end_a_path``
    (endpoint A structure), ``triple_path`` (the canonical triple JSON).
    """
    from prodromos.symmetry_preflight_general import run_symmetry_l1

    return run_symmetry_l1(
        pristine_path,
        end_a_path,
        triple_path,
        mineral_name=mineral_name,
        known_dft_barrier_meV=known_dft_barrier_meV,
        global_disp_threshold=global_disp_threshold,
        global_disp_per_atom_A=global_disp_per_atom_A,
    )


def tool_vfe_preflight(
    pristine_path: str,
    v_fe_index: int,
    mineral_name: str | None = None,
) -> dict:
    """L0 structural pre-flight for a V_Fe (Fe-vacancy) NEB case.

    ``pristine_path`` is the relaxed pristine structure; ``v_fe_index`` is the
    0-based index of the Fe atom to be removed (the vacancy site).
    """
    from prodromos.vfe_neb_preflight import run_structural_l0

    return run_structural_l0(pristine_path, v_fe_index, mineral_name)


def tool_external_reference(
    elements: list[str],
    reduced_formula: str | None = None,
    space_group: str | None = None,
    timeout: float = 10.0,
    live: bool = True,
) -> dict:
    """G19 -- does a public DFT reference exist (NOMAD / OPTIMADE)?

    ``elements`` e.g. ``["Fe", "S"]``. Set ``live=False`` to skip all network
    calls (offline / test mode -> UNKNOWN).
    """
    from prodromos.external_reference_gate import run_external_reference_gate

    return run_external_reference_gate(
        elements,
        reduced_formula=reduced_formula,
        space_group=space_group,
        timeout=timeout,
        live=live,
    )


def tool_lint_dft_script(
    script_path: str,
    pseudo_dir: str | None = None,
    xyz_path: str | None = None,
) -> dict:
    """G18 -- static lint for the 4 recurring QE/ABACUS deploy bugs.

    Checks: absolute pseudo_dir, outdir not doubled, clean-read xyz, and
    ``number_of_wfc>0`` in the pseudopotentials (when ``pseudo_dir`` given).
    """
    from prodromos.lint_dft_script import run_lint_dft_script

    return run_lint_dft_script(script_path, pseudo_dir=pseudo_dir, xyz_path=xyz_path)


def tool_lint_cp2k_input(inp_path: str, layered: bool = True) -> dict:
    """N-19 -- static pre-flight lint for a CP2K ``.inp`` file.

    Checks (CP2K_LESSONS + РЕШЕНИЕ-109 NEB protocol): D3/VDW_POTENTIAL present
    (FAIL if a layered/vdW system lacks it), OT/+U vs non-Gamma k-points [ТУПИК],
    GAPW for 3d metals, MGRID CUTOFF/REL_CUTOFF, and &BAND sanity
    (NUMBER_OF_REPLICA, K_SPRING, OPTIMIZE_BAND MAX_FORCE). ``layered`` makes the
    missing-D3 check FAIL-level (set False for non-vdW systems).
    """
    from prodromos.lint_cp2k_input import run_lint_cp2k_input

    return run_lint_cp2k_input(inp_path, layered=layered)


def tool_carrier_integrity(
    band: str,
    mobile: str = "H",
    acceptors: str = "S",
    metals: str = "Fe,Ni",
    metal_bond_cut: float = 1.8,
) -> dict:
    """N-18 -- carrier-integrity gate (geometry-only, tool-agnostic).

    Verifies a mobile proton/ion stays bound to its intended ACCEPTOR (e.g. S)
    along an NEB band rather than collapsing onto a METAL (e.g. Fe-hydride) -- the
    foundation-MLIP Fe-H failure mode and a real DFT endpoint risk. ``band`` is a
    multi-image xyz/extxyz/traj or a directory of ``img_*.xyz``. ``acceptors`` and
    ``metals`` are comma-separated element lists.
    """
    from prodromos.carrier_integrity_gate import run_carrier_integrity

    return run_carrier_integrity(
        band,
        mobile=mobile,
        acceptors=[s for s in acceptors.split(",") if s],
        metals=[s for s in metals.split(",") if s],
        metal_bond_cut=metal_bond_cut,
    )


def tool_h_barrier_readiness(
    barrier_eV: float,
    has_dft_freq: bool,
    n_imag_modes: int | None = None,
    imag_mode_H_fraction: float | None = None,
    dZPE_eV: float | None = None,
    h_fraction_threshold: float = 0.5,
) -> dict:
    """G20 -- is an H-transfer barrier paper-grade?

    PAPER_GRADE requires an index-1 saddle (``n_imag_modes==1``) whose imaginary
    mode is H-dominated (``imag_mode_H_fraction >= h_fraction_threshold``) plus a
    ZPE correction (``dZPE_eV``); otherwise ELECTRONIC_ONLY.
    """
    from prodromos.h_barrier_paper_readiness import run_h_barrier_paper_readiness

    return run_h_barrier_paper_readiness(
        barrier_eV=barrier_eV,
        has_dft_freq=has_dft_freq,
        n_imag_modes=n_imag_modes,
        imag_mode_H_fraction=imag_mode_H_fraction,
        dZPE_eV=dZPE_eV,
        h_fraction_threshold=h_fraction_threshold,
    )


def tool_neb_advisor(case: dict | None = None) -> dict:
    """G16 -- map an NEB failure signature to a method family.

    ``case`` is a dict of NEB diagnostics (n_iter, barrier_dense_history_eV,
    node_fmax_history, perp_resid_M100/M400, band_energies_rel_eV,
    migrating_geom, force_localization, parity_verdict, nspin, ...). The first
    gate that fires wins; recommends band / dimer / string.
    """
    from prodromos.neb_method_advisor import run_neb_method_advisor

    return run_neb_method_advisor(case)


def tool_saddle_proximity(
    structure_path: str,
    s_i: int,
    s_k: int,
    h_idx: int | None = None,
    asym_tol: float = 0.3,
    mu_bridge_cutoff: float = 2.8,
) -> dict:
    """G17 -- is the saddle the intended S-H...S transfer, not off-path?

    ``structure_path`` is read with ASE into an Atoms object (a cell is needed
    for the minimum-image convention). ``s_i`` / ``s_k`` are 0-based indices of
    the two S anchors of the intended hop; ``h_idx`` is the transferring H (auto
    if a single H is present).
    """
    from ase.io import read

    from prodromos.saddle_proximity_gate import run_saddle_proximity_gate

    atoms = read(structure_path)
    return run_saddle_proximity_gate(
        atoms,
        s_i=s_i,
        s_k=s_k,
        h_idx=h_idx,
        asym_tol=asym_tol,
        mu_bridge_cutoff=mu_bridge_cutoff,
    )


def tool_multi_endpoint(
    pristine_path: str,
    triple_path: str,
    out_dir: str,
    d_SH: float = 1.35,
    d_FeH: float = 1.60,
    fe_cutoff: float = 3.5,
    ss_cutoff: float = 3.5,
) -> dict:
    """L2 -- enumerate candidate H endpoint sites around a V_Fe pocket.

    Writes candidate endpoint structures to ``out_dir`` and returns a manifest.
    """
    from prodromos.multi_endpoint_enumeration import run_endpoint_enumeration

    return run_endpoint_enumeration(
        pristine_path,
        triple_path,
        out_dir,
        d_SH=d_SH,
        d_FeH=d_FeH,
        fe_cutoff=fe_cutoff,
        ss_cutoff=ss_cutoff,
    )


def tool_soap_cluster(
    relaxed_dir: str,
    summary_json: str,
    threshold: float = 0.5,
) -> dict:
    """Cluster relaxed MLIP minima by SOAP descriptor distance.

    Groups near-identical relaxed structures so only representative minima go to
    DFT single-point screening.
    """
    from prodromos.soap_cluster_minima import run_soap_clustering

    return run_soap_clustering(relaxed_dir, summary_json, threshold=threshold)


def tool_mic_alignment(
    endpoint_a_path: str,
    endpoint_b_path: str,
    write_aligned: str | None = None,
    cross_threshold: float = 0.5,
) -> dict:
    """Pre-flight path-sanity gate: are two NEB endpoints in the same periodic image?

    Reads endpoint A and B (any ASE-readable format) and flags atoms that cross a
    periodic boundary between them (a naive interpolation would route them the long
    way across the cell -> meaningless barrier). ALIGNED vs NEEDS_MIC_ALIGNMENT; on
    the latter the result carries the minimum-image-aligned endpoint-B scaled
    positions (and writes the aligned B to ``write_aligned`` if given).
    """
    from ase.io import read

    from prodromos.mic_alignment_gate import run_mic_alignment

    return run_mic_alignment(
        read(endpoint_a_path),
        read(endpoint_b_path),
        cross_threshold=cross_threshold,
        write_aligned=write_aligned,
    )


def tool_mlip_confidence(
    symbol_counts: dict[str, int],
    charge: float = 0.0,
    band_gap_eV: float | None = None,
    migrant: str | None = None,
    multivalent: bool | None = None,
) -> dict:
    """Predict whether a foundation-MLIP migration barrier is trustworthy (§B).

    Flags hosts whose spin-blind foundation-MLIP barrier should NOT be trusted
    (near-degenerate itinerant 3d like V/Ti/Cr, or multivalent redox TM in a cathode
    context) and routes them to DFT; otherwise TRUST_MLIP. ``symbol_counts`` e.g.
    ``{"Mg": 1, "V": 2, "S": 4}``; pass ``band_gap_eV`` and ``migrant`` for a sharper
    classification.
    """
    from prodromos.mlip_confidence_gate import run_mlip_confidence_gate

    return run_mlip_confidence_gate(
        symbol_counts,
        charge=charge,
        band_gap_eV=band_gap_eV,
        migrant=migrant,
        multivalent=multivalent,
    )


def tool_sublattice_preflight(
    sites: list[dict],
    cell: list[list[float]],
    migrant_a: list[float],
    migrant_b: list[float],
    mode: str = "migrant",
    polaron_index_a: int | None = None,
    polaron_index_b: int | None = None,
    migrant_species: str = "Li",
    redox_elements: list[str] | None = None,
) -> dict:
    """Structure-level (pre-DFT, $0) magnetic-sublattice crossing predictor for an
    ion-migration NEB.

    Predicts GO / NO-GO single-sheet BEFORE any SCF from structure + per-site moment
    signs (MAGNDATA / MP metadata). ``mode="migrant"`` tracks the migrating ion's own
    moment; ``mode="polaron"`` tracks the charge-compensating redox polaron of a
    *nonmagnetic* migrant (Li+/Na+ cathode hop) -- the new failure mode where the
    polaron lands on a different sublattice at A vs B (dM_total ~ 2 uB -> ill-posed).

    ``sites`` is a list of ``{"element", "frac": [x,y,z], "sign": +1|-1|0,
    "moment_uB"?, "label"?}``; if ``sign`` is absent it is derived from
    ``moment_uB``. ``cell`` is the 3x3 lattice (Angstrom). ``migrant_a`` /
    ``migrant_b`` are the migrant fractional coords at the two endpoints. On NO-GO the
    envelope's ``next_actions`` carry the constrained-M / two-species recipe.
    """
    from prodromos.sublattice_preflight import MagSite, run_sublattice_preflight

    mag_sites: list[MagSite] = []
    for s in sites:
        sign = s.get("sign")
        if sign is None:
            m = float(s.get("moment_uB") or 0.0)
            sign = 0 if abs(m) < 0.2 else (1 if m > 0 else -1)
        mag_sites.append(
            MagSite(
                element=s["element"],
                frac=tuple(s["frac"]),
                sign=int(sign),
                moment_uB=s.get("moment_uB"),
                label=s.get("label"),
            )
        )
    return run_sublattice_preflight(
        mag_sites,
        cell,
        tuple(migrant_a),
        tuple(migrant_b),
        mode=mode,
        polaron_index_a=polaron_index_a,
        polaron_index_b=polaron_index_b,
        migrant_species=migrant_species,
        redox_elements=set(redox_elements) if redox_elements else None,
    )


def tool_master_equation(
    barrier_matrix: list[list[float]],
    site_labels: list[str] | None = None,
    site_energies: list[float] | None = None,
    T_K: float = 298.15,
) -> dict:
    """L6 -- master-equation kinetic-network analysis over a barrier matrix.

    ``barrier_matrix[i][j]`` = E_a (eV) for the i->j hop (use a large value /
    inf-like for no edge). Returns equilibrium distribution, slowest relaxation
    and the dominant pathway.
    """
    from prodromos.master_equation_kinetics import run_kinetic_network

    return run_kinetic_network(
        barrier_matrix,
        site_labels=site_labels,
        site_energies=site_energies,
        T_K=T_K,
    )


def tool_gp_neb(
    gp_json: str | None = None,
    band_root: str | None = None,
    allow_magnetic_split: bool = False,
    grid_size: int = 201,
    top_k: int = 3,
) -> dict:
    """GP-surrogate NEB planner -- recommend next sample points from a band.

    Provide EITHER ``gp_json`` (a precomputed GP-input JSON) OR ``band_root`` (a
    band directory to summarise).
    """
    from prodromos.gp_neb_surrogate import run_gp_neb_surrogate

    return run_gp_neb_surrogate(
        gp_json=gp_json,
        band_root=band_root,
        allow_magnetic_split=allow_magnetic_split,
        grid_size=grid_size,
        top_k=top_k,
    )


def tool_adaptive_neb(
    band_json: str | None = None,
    band_root: str | None = None,
    k_min: float = 0.3,
    k_max: float = 3.0,
    fmax: float = 0.05,
    scale_fmax: float = 1.0,
) -> dict:
    """Adaptive NEB algorithm planner -- recommend k-spring / fmax schedule.

    Provide EITHER ``band_json`` OR ``band_root``.
    """
    from prodromos.adaptive_neb_planner import run_adaptive_neb_planner

    return run_adaptive_neb_planner(
        band_json=band_json,
        band_root=band_root,
        k_min=k_min,
        k_max=k_max,
        fmax=fmax,
        scale_fmax=scale_fmax,
    )


# --- magnetic family (consume parsed DFT outputs; accept file paths) -------
def tool_magnetic_parser(path: str, engine: str = "auto") -> dict:
    """G14 -- parse a QE/ABACUS output for magnetization + convergence.

    Returns the magnetic summary (energy, total / absolute magnetization, drift,
    local moments, provenance fields). ``engine`` is ``"auto"`` | ``"qe"`` |
    ``"abacus"``.
    """
    from prodromos.magnetic_output_parser import parse_output_file

    summary = parse_output_file(path, engine=engine)
    return response_envelope(
        tool="magnetic_parser",
        verdict="PARSED",
        confidence="medium",
        result=summary.to_dict(),
    )


def tool_magnetic_endpoint(
    endpoint_a_output: str,
    endpoint_b_output: str,
    delta_total_threshold: float | None = None,
    delta_abs_threshold: float | None = None,
    n_magnetic: int | None = None,
    delta_abs_per_tm_threshold: float | None = None,
) -> dict:
    """G12 -- do NEB endpoints lie on one magnetic sheet?

    ``endpoint_a_output`` / ``endpoint_b_output`` are paths to the two endpoint
    SCF output files; each is parsed for total / absolute magnetization, then
    compared (GO / NO-GO_SINGLE_SHEET). Pass ``n_magnetic`` (number of magnetic/TM
    atoms) to use the PER-TM relative threshold, which stops large-cell slow-drift
    systems from a false NO-GO (troilite).
    """
    from prodromos.magnetic_endpoint_gate import (
        DELTA_ABS_ADJ,
        DELTA_ABS_PER_TM,
        DELTA_TOTAL_ENDPOINT,
        endpoint_magnetic_gate,
    )
    from prodromos.magnetic_output_parser import parse_output_file

    sa = parse_output_file(endpoint_a_output)
    sb = parse_output_file(endpoint_b_output)
    result = endpoint_magnetic_gate(
        sa,
        sb,
        delta_total_threshold=(
            delta_total_threshold if delta_total_threshold is not None else DELTA_TOTAL_ENDPOINT
        ),
        delta_abs_threshold=(
            delta_abs_threshold if delta_abs_threshold is not None else DELTA_ABS_ADJ
        ),
        n_magnetic=n_magnetic,
        delta_abs_per_tm_threshold=(
            delta_abs_per_tm_threshold if delta_abs_per_tm_threshold is not None else DELTA_ABS_PER_TM
        ),
    )
    return response_envelope(
        tool="magnetic_endpoint",
        verdict=result.verdict,
        confidence="medium",
        reasons=list(result.reasons),
        result=result.to_dict(),
    )


def tool_magnetic_verdict(
    endpoint_a_output: str,
    endpoint_b_output: str,
    band_root: str | None = None,
    n_magnetic: int | None = None,
    delta_total_threshold: float | None = None,
    delta_abs_threshold: float | None = None,
    delta_abs_per_tm_threshold: float | None = None,
) -> dict:
    """One combined magnetic verdict: endpoint screen + band arbiter (§C).

    Runs the endpoint gate (pre-launch screen, PER-TM relative threshold when
    ``n_magnetic`` is given) and, if ``band_root`` is supplied, the band gate over
    the full trajectory, then AUTO-RECONCILES them into ONE verdict (the band gate
    is the arbiter; the endpoint NO-GO auto-escalates to the band check). Reports
    both verdicts + the resolution so the user never reconciles them by hand.
    """
    from prodromos.magnetic_endpoint_gate import (
        DELTA_ABS_ADJ,
        DELTA_ABS_PER_TM,
        DELTA_TOTAL_ENDPOINT,
        endpoint_magnetic_gate,
        reconcile_endpoint_and_band,
    )
    from prodromos.magnetic_output_parser import parse_output_file

    ep = endpoint_magnetic_gate(
        parse_output_file(endpoint_a_output),
        parse_output_file(endpoint_b_output),
        delta_total_threshold=(
            delta_total_threshold if delta_total_threshold is not None else DELTA_TOTAL_ENDPOINT
        ),
        delta_abs_threshold=(
            delta_abs_threshold if delta_abs_threshold is not None else DELTA_ABS_ADJ
        ),
        n_magnetic=n_magnetic,
        delta_abs_per_tm_threshold=(
            delta_abs_per_tm_threshold if delta_abs_per_tm_threshold is not None else DELTA_ABS_PER_TM
        ),
    )
    band_result = None
    if band_root:
        from prodromos.magnetic_band_gate import analyze_band_images, load_band

        band_result = analyze_band_images(load_band(band_root))
    combined = reconcile_endpoint_and_band(ep, band_result)
    return response_envelope(
        tool="magnetic_verdict",
        verdict=combined["combined_verdict"],
        confidence="medium",
        reasons=[combined["resolution"], *ep.reasons],
        result={
            "combined": combined,
            "endpoint": ep.to_dict(),
            "band": band_result.to_dict() if band_result is not None else None,
        },
    )


def tool_magnetic_band(band_root: str) -> dict:
    """G13 -- no spin-sheet crossing inside an NEB band (post-pilot).

    ``band_root`` is the band directory holding ``image_XX/`` output files.
    """
    from prodromos.magnetic_band_gate import analyze_band_images, load_band

    result = analyze_band_images(load_band(band_root))
    return response_envelope(
        tool="magnetic_band",
        verdict=result.verdict,
        confidence="medium",
        reasons=list(result.reasons),
        result=result.to_dict(),
    )


def tool_magnetic_recommend(
    band_root: str | None = None,
    endpoint_matrix_dir: str | None = None,
) -> dict:
    """G15 -- recommend how to handle magnetic (dis)continuity of a band.

    Provide ``band_root`` (band directory) and/or ``endpoint_matrix_dir``
    (directory of endpoint magnetic single-points).
    """
    from prodromos.magnetic_recommendation import build_recommendation

    rec = build_recommendation(band_root=band_root, endpoint_matrix_dir=endpoint_matrix_dir)
    return response_envelope(
        tool="magnetic_recommend",
        verdict=rec.action,
        confidence=rec.confidence,
        reasons=list(rec.reasons),
        result=rec.to_dict(),
    )


# ===========================================================================
# 4. tm-spec importers + merge -- OPTIMADE width x NOMAD depth, all local
# ===========================================================================
def tool_magnetic_provenance(
    mp_ordering: str | None = None,
    magndata_ordering: str | None = None,
    material_id: str | None = None,
    formula: str | None = None,
    space_group: int | None = None,
    magndata_code: str | None = None,
    live: bool = False,
) -> dict:
    """Cross-check COMPUTED (MP) vs EXPERIMENTAL (MAGNDATA) magnetic ordering (§C/§C-bis).

    MP often mislabels Fe sulfides/phosphates FM where neutron experiment is AFM;
    seeding an NEB from the wrong ordering puts both endpoints on the wrong sheet.
    This gate compares the two and, on disagreement, WARNs and routes the seed to the
    experimental MAGNDATA block. Pass the orderings directly for a $0 comparison, or
    ``live=True`` with ``material_id``/``formula`` (+ ``magndata_code``) to fetch them
    (MP key from env or ``secrets/mp_api_key.json``).
    """
    from prodromos.magnetic_provenance import run_magnetic_provenance

    return run_magnetic_provenance(
        mp_ordering=mp_ordering,
        magndata_ordering=magndata_ordering,
        material_id=material_id,
        formula=formula,
        space_group=space_group,
        magndata_code=magndata_code,
        live=live,
    )


def tool_import_optimade(
    elements: list[str] | None,
    reduced_formula: str | None = None,
    provider: str = "mp",
    page_limit: int = 20,
    raw_filter: str | None = None,
    live: bool = True,
) -> dict:
    """Query the OPTIMADE federation (MP/NOMAD/OQMD/...) for structures of a
    composition; returns tm-spec/0.3 docs (structure-level, geometry_origin=unknown)
    ready for plan/merge.

    ``elements`` e.g. ``["Fe", "S"]`` builds an ``elements HAS ALL`` filter; pass
    ``reduced_formula`` (e.g. ``"FeS2"``) for an exact reduced-formula query, or a
    verbatim ``raw_filter``. ``provider`` is one of ``mp`` (default) / ``nomad`` /
    ``oqmd`` / ``alexandria``. Set ``live=False`` for offline (returns no docs).

    Each returned doc is a structure-only ``SinglePointCalculation`` with
    ``calculation={"method": "DFT"}`` stub and ``structure.geometry_origin
    ="unknown"`` (OPTIMADE never reports relaxation/XC/energy). Merge one of these
    (width) into a NOMAD import (depth) with ``merge_specs``.

    For NOMAD method/results *depth* by entry_id use ``import_nomad`` (or the CLI
    ``tm-spec import-nomad <entry_id>``).

    Degrades softly: a network error / empty result yields ``status`` +
    ``reasons`` instead of raising.
    """
    from tm_spec.importers.optimade import OptimadeError, import_optimade

    try:
        docs = import_optimade(
            elements=elements,
            reduced_formula=reduced_formula,
            provider=provider,
            page_limit=page_limit,
            raw_filter=raw_filter,
            live=live,
        )
    except OptimadeError as exc:
        return {
            "tool": "import_optimade",
            "status": "error",
            "count": 0,
            "docs": [],
            "reasons": [f"OPTIMADE query failed: {exc}"],
        }
    except Exception as exc:  # network / parse / unexpected -- never raise to the LLM
        return {
            "tool": "import_optimade",
            "status": "error",
            "count": 0,
            "docs": [],
            "reasons": [f"unexpected error querying OPTIMADE: {exc}"],
        }

    reasons: list[str] = []
    if not docs:
        reasons.append(
            "offline mode: no network call made"
            if not live
            else f"no structures returned by provider {provider!r} for the given filter"
        )
    else:
        reasons.append(
            f"imported {len(docs)} OPTIMADE structure doc(s) from provider "
            f"{provider!r} (structure-level, geometry_origin=unknown)"
        )
    return {
        "tool": "import_optimade",
        "status": "ok",
        "count": len(docs),
        "docs": docs,
        "reasons": reasons,
    }


def tool_import_nomad(
    entry_id: str,
    author: str = "import@nomad",
) -> dict:
    """Import a single NOMAD Archive entry (method / XC / magnetic / energy DEPTH)
    into a tm-spec/0.3 doc by ``entry_id``.

    NOMAD archive entries carry the calculation method, XC functional, spin
    treatment and energies that OPTIMADE lacks -- this is the *depth* side that
    pairs with ``import_optimade`` *width* via ``merge_specs``. The kind is
    detected from the NOMAD workflow (SinglePoint / Relax / MD) and
    ``structure.geometry_origin`` is set honestly (never fabricated as
    ``dft_relaxed``).

    ``entry_id`` is a NOMAD entry id (e.g. ``zRzA8h0p1q...``); ``author`` fills
    ``provenance.author``. Anonymous reads suffice for public entries (set the
    ``NOMAD_API_TOKEN`` env var for private ones).

    Degrades softly: a network / HTTP error yields ``status`` + ``reasons``
    instead of raising.
    """
    from tm_spec.importers.nomad import NomadError, fetch_to_tm_spec

    try:
        doc = fetch_to_tm_spec(entry_id, author=author)
    except NomadError as exc:
        return {
            "tool": "import_nomad",
            "status": "error",
            "count": 0,
            "docs": [],
            "reasons": [f"NOMAD import failed: {exc}"],
        }
    except Exception as exc:  # never raise to the LLM
        return {
            "tool": "import_nomad",
            "status": "error",
            "count": 0,
            "docs": [],
            "reasons": [f"unexpected error importing NOMAD entry {entry_id!r}: {exc}"],
        }

    return {
        "tool": "import_nomad",
        "status": "ok",
        "count": 1,
        "docs": [doc],
        "reasons": [
            f"imported NOMAD entry {entry_id!r} as kind={doc.get('kind')!r} "
            f"(geometry_origin={(doc.get('structure') or {}).get('geometry_origin')!r})"
        ],
    }


def tool_import_mp(
    material_id: str | None = None,
    formula: str | None = None,
    space_group: int | None = None,
    author: str = "import@mp",
) -> dict:
    """Import Materials Project COMPUTED magnetism (the 'magnetic depth') into
    tm-spec/0.3 doc(s).

    MP carries the computed magnetic ground state OPTIMADE and NOMAD lack: ordering
    (NM/FM/AFM/FiM), total_magnetization and per-site magmoms. This fills the tm-spec
    ``magnetic`` block (state, collinear, magmoms_uB) -- the third leg paired with
    ``import_optimade`` (structure width) and ``import_nomad`` (method depth) via
    ``merge_specs``.

    Provide ``material_id`` (exact, e.g. ``mp-226``) OR ``formula`` (e.g. ``FeS2``),
    optionally narrowed to one ``space_group`` (most stable polymorph). Geometry is
    MP-relaxed -> geometry_origin=dft_relaxed; AFM subtype is unspecified by MP
    (mapped to AFM-G + surrogate_warning).

    Auth: requires the ``MP_API_KEY`` env var (free MP key). Degrades softly: a
    network / auth / HTTP error yields ``status`` + ``reasons`` instead of raising.
    """
    from tm_spec.importers.mp import MPError, fetch_to_tm_spec

    try:
        docs = fetch_to_tm_spec(
            material_id=material_id,
            formula=formula,
            space_group=space_group,
            author=author,
        )
    except MPError as exc:
        return {
            "tool": "import_mp",
            "status": "error",
            "count": 0,
            "docs": [],
            "reasons": [f"MP import failed: {exc}"],
        }
    except Exception as exc:  # never raise to the LLM
        return {
            "tool": "import_mp",
            "status": "error",
            "count": 0,
            "docs": [],
            "reasons": [f"unexpected error importing MP material: {exc}"],
        }

    return {
        "tool": "import_mp",
        "status": "ok",
        "count": len(docs),
        "docs": docs,
        "reasons": [
            f"imported {len(docs)} MP magnetic-depth doc(s) "
            f"(magnetic.state filled from MP computed ground state)"
        ],
    }


def tool_import_magndata(
    code: str | None = None,
    elements: list[str] | None = None,
    formula: str | None = None,
    max_results: int = 10,
    author: str = "import@magndata",
) -> dict:
    """Import MAGNDATA EXPERIMENTAL magnetic structure(s) into tm-spec/0.3 doc(s).

    MAGNDATA (Bilbao) is the experimental magnetic-structure database (magCIF / BNS
    magnetic space groups) -- the experimental ground-truth anchor that complements
    MP's COMPUTED magnetism (``import_mp``), which can disagree with experiment
    (e.g. MP labels troilite/chalcopyrite FM where neutron diffraction finds AFM).
    Fills the tm-spec ``magnetic`` block with state / collinear / magmoms_uB /
    propagation_vector / bns_group and ``geometry_origin: experimental``. The
    FM/AFM/ferri verdict is derived rigorously from the magnetic symmetry operations
    in the file (net-moment projector), not a hardcoded table.

    Provide EITHER ``code`` (a single MAGNDATA entry code, e.g. ``0.1`` / ``1.0.1`` /
    ``2.10``) OR a SEARCH: ``elements`` (e.g. ``["Fe", "S"]``, element-AND) or
    ``formula`` (e.g. ``"FeS"``, whose element set is searched and whose reduced
    formula filters the matches). Search fetches up to ``max_results`` entries. The
    Bilbao server has a misconfigured TLS cert; this fetches with verification
    disabled (public reference data). Degrades softly on network/parse error.
    """
    from tm_spec.importers.magndata import (
        MagndataError,
        fetch_to_tm_spec,
        search_to_tm_spec,
    )

    try:
        if elements or formula:
            docs = search_to_tm_spec(
                elements=elements, formula=formula, max_results=max_results, author=author
            )
            query = f"elements={elements}" if elements else f"formula={formula!r}"
            if not docs:
                return {"tool": "import_magndata", "status": "ok", "count": 0, "docs": [],
                        "reasons": [f"no MAGNDATA entries matched ({query})"]}
            return {
                "tool": "import_magndata", "status": "ok", "count": len(docs), "docs": docs,
                "reasons": [f"imported {len(docs)} MAGNDATA experimental doc(s) matching {query}; "
                            f"states={[(d.get('magnetic') or {}).get('state') for d in docs]}"],
            }
        if not code:
            return {"tool": "import_magndata", "status": "error", "count": 0, "docs": [],
                    "reasons": ["provide a code, or elements / formula to search"]}
        doc = fetch_to_tm_spec(code, author=author)
    except MagndataError as exc:
        return {
            "tool": "import_magndata",
            "status": "error",
            "count": 0,
            "docs": [],
            "reasons": [f"MAGNDATA import failed: {exc}"],
        }
    except Exception as exc:  # never raise to the LLM
        return {
            "tool": "import_magndata",
            "status": "error",
            "count": 0,
            "docs": [],
            "reasons": [f"unexpected error importing MAGNDATA {code!r}: {exc}"],
        }

    return {
        "tool": "import_magndata",
        "status": "ok",
        "count": 1,
        "docs": [doc],
        "reasons": [
            f"imported MAGNDATA {code!r} (experimental); magnetic.state="
            f"{(doc.get('magnetic') or {}).get('state')!r}, bns_group="
            f"{(doc.get('magnetic') or {}).get('bns_group')!r}"
        ],
    }


def tool_merge_specs(
    base: dict,
    overlay: dict,
    fill_only: bool = True,
    strict_material: bool = True,
) -> dict:
    """Merge two tm-spec docs locally: NOMAD depth (method/magnetic/results) x
    OPTIMADE width (structure). Fill-only; same-material guard.

    ``base`` / ``overlay`` are tm-spec docs as dicts (pass them as JSON). The
    deep ``base`` (typically a NOMAD import) keeps precedence; ``overlay``
    (typically an OPTIMADE import) only fills holes. ``geometry_origin`` keeps the
    more specific value, provenance import_source records are unioned, and sanity
    gates are merged by id.

    With ``strict_material=True`` (default) a formula mismatch returns
    ``status="error"`` (MATERIAL_MISMATCH) rather than raising; set it ``False``
    to downgrade the mismatch to a warning and merge anyway. ``fill_only=False``
    lets overlay scalars win on conflict.

    Returns ``{"tool": "merge_specs", "status", "merged": <doc>, "warnings": [...]}``.
    """
    from tm_spec.merge import MergeError, merge_docs

    try:
        merged, warnings = merge_docs(
            base,
            overlay,
            fill_only=fill_only,
            strict_material=strict_material,
        )
    except MergeError as exc:
        return {
            "tool": "merge_specs",
            "status": "error",
            "merged": None,
            "warnings": [str(exc)],
        }
    except Exception as exc:  # never raise to the LLM
        return {
            "tool": "merge_specs",
            "status": "error",
            "merged": None,
            "warnings": [f"unexpected merge error: {exc}"],
        }

    return {
        "tool": "merge_specs",
        "status": "ok",
        "merged": merged,
        "warnings": list(warnings),
    }


# ===========================================================================
# 5. meta-tools -- batch / bundle (one round-trip; kill client fan-out)
# ===========================================================================
def tool_batch(calls: list[dict]) -> dict:
    """Run MANY prodromos gates in ONE round-trip (sequential, server-side).

    ``calls`` is a list of ``{"tool": <name>, "args": {...}}`` items. Each is
    dispatched to the corresponding gate; per-call errors are captured (never
    raised to the client) and returned aligned by ``index``. This removes the
    need for the client to fan out parallel tool calls -- the failure mode that
    trips the stdio transport.

    Returns an envelope whose ``result.calls`` lists, per call,
    ``{index, tool, status, result|error}``.
    """
    results: list[dict] = []
    for i, call in enumerate(calls or []):
        call = call or {}
        name = call.get("tool")
        args = call.get("args") or {}
        fn = _GATE_TOOLS.get(name)
        if fn is None:
            results.append(
                {"index": i, "tool": name, "status": "error", "error": f"unknown tool {name!r}"}
            )
            continue
        try:
            res = fn(**args)
            results.append({"index": i, "tool": name, "status": "ok", "result": res})
        except Exception as exc:  # never raise to the LLM -- capture per call
            results.append(
                {"index": i, "tool": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            )
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    return response_envelope(
        tool="batch",
        verdict="BATCH_DONE",
        confidence="high",
        reasons=[f"executed {len(results)} call(s) server-side ({n_ok} ok, {n_err} error)"],
        result={"calls": results, "n_ok": n_ok, "n_error": n_err},
    )


def tool_preflight_bundle(
    case_path: str,
    mode: str = "route",
    budget_usd: float | None = None,
    code: str = "auto",
) -> dict:
    """Run the FULL applicable pre-flight gate set for a case in ONE round-trip.

    A pre-flight is a *flow*, not one gate: this loads/validates the tm-spec case
    (or auto-converts a raw QE/ABACUS input via ``from-inputs``), walks the policy
    graph EXECUTING every $0 gate, and returns each gate's verdict plus an overall
    recommendation -- so the client never has to fan out one call per gate.

    Parameters mirror ``plan``. Returns the ``plan`` envelope with an added
    ``result.bundle`` summary (per-gate verdicts + overall verdict + next action).
    """
    env = tool_plan(case_path, mode=mode, budget_usd=budget_usd, code=code)
    res = env.get("result") or {}
    gates = res.get("gates", [])
    env["result"] = {
        **res,
        "bundle": {
            "n_gates": len(gates),
            "gates": gates,
            "overall_verdict": env.get("verdict"),
            "next_action": res.get("next_action"),
        },
    }
    return env


# ===========================================================================
# registration
# ===========================================================================
# (tool_name -> wrapper). Names are snake_case (MCP requires valid identifiers).
# ``_GATE_TOOLS`` is the set ``batch`` may dispatch over (excludes the meta-tools
# to keep batch non-recursive); ``_TOOLS`` is the full registered surface.
_GATE_TOOLS: dict[str, Any] = {
    "plan": tool_plan,
    "from_inputs": tool_from_inputs,
    "electron_parity": tool_electron_parity,
    "spin_collapse": tool_spin_collapse,
    "endpoint_provenance": tool_endpoint_provenance,
    "symmetry_preflight": tool_symmetry_preflight,
    "vfe_preflight": tool_vfe_preflight,
    "external_reference": tool_external_reference,
    "lint_dft_script": tool_lint_dft_script,
    "lint_cp2k_input": tool_lint_cp2k_input,
    "carrier_integrity": tool_carrier_integrity,
    "h_barrier_readiness": tool_h_barrier_readiness,
    "neb_advisor": tool_neb_advisor,
    "saddle_proximity": tool_saddle_proximity,
    "multi_endpoint": tool_multi_endpoint,
    "mic_alignment": tool_mic_alignment,
    "mlip_confidence": tool_mlip_confidence,
    "sublattice_preflight": tool_sublattice_preflight,
    "soap_cluster": tool_soap_cluster,
    "master_equation": tool_master_equation,
    "gp_neb": tool_gp_neb,
    "adaptive_neb": tool_adaptive_neb,
    "magnetic_parser": tool_magnetic_parser,
    "magnetic_endpoint": tool_magnetic_endpoint,
    "magnetic_verdict": tool_magnetic_verdict,
    "magnetic_band": tool_magnetic_band,
    "magnetic_recommend": tool_magnetic_recommend,
    "magnetic_provenance": tool_magnetic_provenance,
    "import_optimade": tool_import_optimade,
    "import_nomad": tool_import_nomad,
    "import_mp": tool_import_mp,
    "import_magndata": tool_import_magndata,
    "merge_specs": tool_merge_specs,
}

_TOOLS: dict[str, Any] = {
    **_GATE_TOOLS,
    "batch": tool_batch,
    "preflight_bundle": tool_preflight_bundle,
}

# Optional server-side per-tool timeout (seconds). 0/unset = no timeout.
try:
    _TOOL_TIMEOUT_S = float(os.environ.get("PRODROMOS_MCP_TOOL_TIMEOUT_S", "0") or 0)
except ValueError:
    _TOOL_TIMEOUT_S = 0.0


def _timeout_envelope(name: str, timeout_s: float) -> dict:
    return response_envelope(
        tool=name,
        verdict="TIMEOUT",
        confidence="low",
        reasons=[f"tool {name!r} exceeded the server-side timeout of {timeout_s:g}s"],
        next_actions=[
            "retry with a smaller input, or call the gate as a library function",
        ],
    )


def _offload(name: str, fn: Any) -> Any:
    """Wrap a sync gate core as an ASYNC tool that runs OFF the event-loop thread.

    In FastMCP 1.27 a sync (``def``) tool is executed INLINE on the asyncio event
    loop (``Tool.run`` -> ``call_fn_with_arg_validation`` with ``is_async=False``),
    which also hosts the stdio reader/writer -- so a running sync tool stalls both
    request intake and response flushing, serializing concurrent calls. Offloading
    each core via ``anyio.to_thread.run_sync`` keeps the loop free. ``functools.wraps``
    preserves the typed signature + docstring so FastMCP still derives the correct
    JSON schema (verified: ``is_async`` becomes True, params survive).
    """

    @functools.wraps(fn)
    async def _async_tool(*args: Any, **kwargs: Any) -> Any:
        call = functools.partial(fn, *args, **kwargs)
        if _TOOL_TIMEOUT_S > 0:
            try:
                with anyio.fail_after(_TOOL_TIMEOUT_S):
                    return await anyio.to_thread.run_sync(call, abandon_on_cancel=True)
            except TimeoutError:
                return _timeout_envelope(name, _TOOL_TIMEOUT_S)
        return await anyio.to_thread.run_sync(call)

    return _async_tool


for _name, _fn in _TOOLS.items():
    server.tool(name=_name)(_offload(_name, _fn))


# ===========================================================================
# singleton guard + clean shutdown (orphan-server defence)
# ===========================================================================
_LOCK_PATH = Path(tempfile.gettempdir()) / "prodromos-mcp.lock"


def _pid_alive(pid: int) -> bool:
    """Best-effort cross-platform check that ``pid`` is a live process."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
                return True
            return False
        os.kill(pid, 0)
        return True
    except (OSError, Exception):  # noqa: BLE001 -- best effort
        return False


def _get_parent_pid() -> int:
    """Return the PID of our parent process (best-effort; 0 on failure)."""
    try:
        if os.name == "nt":
            import ctypes
            import ctypes.wintypes

            TH32CS_SNAPPROCESS = 0x00000002
            our_pid = os.getpid()

            class PROCESSENTRY32(ctypes.Structure):
                _fields_ = [
                    ("dwSize", ctypes.wintypes.DWORD),
                    ("cntUsage", ctypes.wintypes.DWORD),
                    ("th32ProcessID", ctypes.wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", ctypes.wintypes.DWORD),
                    ("cntThreads", ctypes.wintypes.DWORD),
                    ("th32ParentProcessID", ctypes.wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", ctypes.wintypes.DWORD),
                    ("szExeFile", ctypes.c_char * 260),
                ]

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
            if snap == ctypes.wintypes.HANDLE(-1).value:
                return 0
            try:
                entry = PROCESSENTRY32()
                entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
                if not kernel32.Process32First(snap, ctypes.byref(entry)):
                    return 0
                while True:
                    if entry.th32ProcessID == our_pid:
                        return entry.th32ParentProcessID
                    if not kernel32.Process32Next(snap, ctypes.byref(entry)):
                        break
            finally:
                kernel32.CloseHandle(snap)
            return 0
        # Unix: use os.getppid()
        return os.getppid()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return 0


def _start_parent_watch(parent_pid: int, check_interval: float = 5.0) -> None:
    """Daemon thread: exit when our parent process (Claude Code) dies.

    Claude Code spawns prodromos as a stdio-MCP subprocess.  When Claude Code
    crashes or is force-closed the OS does NOT deliver EOF to the child's stdin
    reliably on Windows, so the prodromos process lingers as a zombie.  This
    thread detects that and calls ``os._exit(0)`` so the next session starts
    clean.

    Safe for parallel Claude Code windows: each window has its own parent PID
    and its own prodromos child, so watching the individual parent is harmless.
    """
    import threading
    import time

    def _watch() -> None:
        while True:
            time.sleep(check_interval)
            if not _pid_alive(parent_pid):
                _log(f"parent pid={parent_pid} is gone; exiting to avoid orphan accumulation")
                os._exit(0)

    t = threading.Thread(target=_watch, daemon=True, name="parent-watch")
    t.start()


def _acquire_singleton_lock() -> Path | None:
    """Record our PID in the lockfile; warn if a prior live instance is found.

    A prior live process is almost always an orphan from an ``/mcp`` reconnect
    where Claude Code started a fresh subprocess without sending EOF to the old
    one.  We take the lock so the next startup knows who the current server is.
    The ``_start_parent_watch`` call in ``main()`` is the complementary fix that
    causes the old process to exit when its Claude Code session dies.
    """
    try:
        if _LOCK_PATH.exists():
            try:
                prev = int(_LOCK_PATH.read_text(encoding="utf-8").strip() or "0")
            except (ValueError, OSError):
                prev = 0
            if prev and prev != os.getpid() and _pid_alive(prev):
                _log(
                    f"WARNING: prior instance pid={prev} appears alive; taking the lock "
                    f"(orphan suspect after an /mcp reconnect). this pid={os.getpid()}"
                )
        _LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
        return _LOCK_PATH
    except OSError as exc:  # lockfile is advisory; never block startup on it
        _log(f"could not write lockfile {_LOCK_PATH}: {exc}")
        return None


def _release_singleton_lock(lock: Path | None) -> None:
    if lock is None:
        return
    try:
        if lock.exists() and lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass


def main() -> None:
    """Entry point: run the prodromos MCP server over stdio (blocking).

    ``server.run()`` returns when stdin reaches EOF (the client disconnects), so
    the ``finally`` releases the lock and the process exits cleanly -- a ``/mcp``
    reconnect cannot orphan it. The PID is logged to stderr at start and stop.
    """
    _log(f"starting (pid={os.getpid()}, tools={len(_TOOLS)})")
    parent_pid = _get_parent_pid()
    if parent_pid:
        _start_parent_watch(parent_pid)
        _log(f"parent-watch active (parent pid={parent_pid})")
    lock = _acquire_singleton_lock()
    try:
        server.run()
    finally:
        _log(f"shutting down (pid={os.getpid()})")
        _release_singleton_lock(lock)


if __name__ == "__main__":
    main()
