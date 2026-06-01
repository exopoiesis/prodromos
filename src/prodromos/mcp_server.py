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

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from prodromos.cli_contract import response_envelope

server = FastMCP("prodromos")


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
    from prodromos.plan.policy import POLICY_GRAPH

    doc, err, extra_reasons = _load_and_validate(Path(case_path), code=code)
    if err is not None:
        return err

    result = walk(
        POLICY_GRAPH,
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
) -> dict:
    """G12 -- do NEB endpoints lie on one magnetic sheet?

    ``endpoint_a_output`` / ``endpoint_b_output`` are paths to the two endpoint
    SCF output files; each is parsed for total / absolute magnetization, then
    compared (GO / NO-GO_SINGLE_SHEET).
    """
    from prodromos.magnetic_endpoint_gate import (
        DELTA_ABS_ADJ,
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
    )
    return response_envelope(
        tool="magnetic_endpoint",
        verdict=result.verdict,
        confidence="medium",
        reasons=list(result.reasons),
        result=result.to_dict(),
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
# registration
# ===========================================================================
# (tool_name -> wrapper). Names are snake_case (MCP requires valid identifiers).
_TOOLS: dict[str, Any] = {
    "plan": tool_plan,
    "from_inputs": tool_from_inputs,
    "electron_parity": tool_electron_parity,
    "spin_collapse": tool_spin_collapse,
    "endpoint_provenance": tool_endpoint_provenance,
    "symmetry_preflight": tool_symmetry_preflight,
    "vfe_preflight": tool_vfe_preflight,
    "external_reference": tool_external_reference,
    "lint_dft_script": tool_lint_dft_script,
    "h_barrier_readiness": tool_h_barrier_readiness,
    "neb_advisor": tool_neb_advisor,
    "saddle_proximity": tool_saddle_proximity,
    "multi_endpoint": tool_multi_endpoint,
    "soap_cluster": tool_soap_cluster,
    "master_equation": tool_master_equation,
    "gp_neb": tool_gp_neb,
    "adaptive_neb": tool_adaptive_neb,
    "magnetic_parser": tool_magnetic_parser,
    "magnetic_endpoint": tool_magnetic_endpoint,
    "magnetic_band": tool_magnetic_band,
    "magnetic_recommend": tool_magnetic_recommend,
    "import_optimade": tool_import_optimade,
    "import_nomad": tool_import_nomad,
    "merge_specs": tool_merge_specs,
}

for _name, _fn in _TOOLS.items():
    server.tool(name=_name)(_fn)


def main() -> None:
    """Entry point: run the prodromos MCP server over stdio (blocking)."""
    server.run()


if __name__ == "__main__":
    main()
