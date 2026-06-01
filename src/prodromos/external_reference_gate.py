"""N-07 -- external reference gate: check NOMAD + OPTIMADE for existing DFT data
BEFORE committing expensive compute on a new mineral.

Queries NOMAD (REST API v1) as primary source, falls back to an OPTIMADE provider
(Materials Project) if NOMAD returns nothing or is unreachable.

Verdicts:
    REFERENCE_FOUND     -- n_entries > 0 (attach functional histogram / stoichiometries).
    NO_EXTERNAL_REFERENCE -- n_entries == 0; raise internal validation bar.
    UNKNOWN             -- network error / timeout; status="error".

The ``live=False`` flag suppresses all network calls (for offline/testing use).

NOMAD API (2024-2026):
    POST https://nomad-lab.eu/prod/v1/api/v1/entries/query
    Body: {"query": {"results.material.elements": {"all": [...]}},
           "pagination": {"page_size": 20}, "required": {"include": [...]}}
    Docs: https://nomad-lab.eu/prod/v1/api/v1/extensions/docs

OPTIMADE (2024-2026) - Materials Project provider:
    GET https://optimade.materialsproject.org/v1/structures
    ?filter=elements HAS ALL "Fe","S"&page_limit=20
    Docs: https://www.optimade.org/optimade-python-tools/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import httpx

from prodromos.cli_contract import response_envelope, dump_json

# ---------------------------------------------------------------------------
# NOMAD API constants (POST /entries/query)
# ---------------------------------------------------------------------------
_NOMAD_ENTRIES_URL = "https://nomad-lab.eu/prod/v1/api/v1/entries/query"
_NOMAD_PAGE_SIZE = 50

# ---------------------------------------------------------------------------
# OPTIMADE fallback (Materials Project public endpoint)
# ---------------------------------------------------------------------------
_OPTIMADE_URL = "https://optimade.materialsproject.org/v1/structures"
_OPTIMADE_PAGE_LIMIT = 50


def _build_nomad_query(
    elements: list[str],
    reduced_formula: str | None,
) -> dict:
    """Return the NOMAD entries/query POST body.

    Isolated here so callers can monkeypatch / inspect without touching network.
    """
    # Primary filter: all required elements must be present.
    query: dict[str, Any] = {
        "results.material.elements": {"all": sorted(elements)}
    }
    # Optionally narrow by reduced formula (exact match).
    if reduced_formula:
        query["results.material.chemical_formula_reduced"] = reduced_formula

    return {
        "query": query,
        "pagination": {"page_size": _NOMAD_PAGE_SIZE},
        "required": {
            "include": [
                "entry_id",
                "results.material.chemical_formula_reduced",
                "results.material.elements",
                "results.method.simulation.dft.xc_functional_type",
                "results.properties.structures.structure_original.lattice_parameters",
                "results.properties.magnetic.magnetic_ordering",
            ]
        },
    }


def _parse_nomad_response(data: dict) -> dict:
    """Parse a NOMAD entries/query JSON response.

    Isolated for easy mocking in tests.
    Returns {n_entries, functional_histogram, nearest_stoichiometries, lattice, magnetic}.
    """
    hits = data.get("data", [])
    total = data.get("pagination", {}).get("total", len(hits))

    functional_histogram: dict[str, int] = {}
    stoichiometries: list[str] = []
    lattice_samples: list[dict] = []
    magnetic_samples: list[str] = []

    for entry in hits:
        res = entry.get("results", {})
        mat = res.get("material", {})
        meth = res.get("method", {})
        props = res.get("properties", {})

        # Stoichiometry
        formula = mat.get("chemical_formula_reduced")
        if formula and formula not in stoichiometries:
            stoichiometries.append(formula)

        # XC functional
        xc = (
            meth.get("simulation", {})
            .get("dft", {})
            .get("xc_functional_type", "unknown")
        )
        if xc:
            functional_histogram[xc] = functional_histogram.get(xc, 0) + 1

        # Lattice (first sample only)
        if len(lattice_samples) < 3:
            lp = (
                props.get("structures", {})
                .get("structure_original", {})
                .get("lattice_parameters")
            )
            if lp:
                lattice_samples.append(lp)

        # Magnetic ordering
        mag = props.get("magnetic", {}).get("magnetic_ordering")
        if mag and mag not in magnetic_samples:
            magnetic_samples.append(mag)

    return {
        "n_entries": total,
        "functional_histogram": functional_histogram,
        "nearest_stoichiometries": stoichiometries[:10],
        "lattice": lattice_samples[:3] if lattice_samples else None,
        "magnetic": magnetic_samples if magnetic_samples else None,
        "source": "nomad",
    }


def _build_optimade_filter(
    elements: list[str],
    reduced_formula: str | None,
) -> str:
    """Return an OPTIMADE filter string.

    Isolated for easy mocking in tests.
    """
    if reduced_formula:
        return f'chemical_formula_reduced = "{reduced_formula}"'
    els_quoted = ",".join(f'"{e}"' for e in sorted(elements))
    return f"elements HAS ALL {els_quoted}"


def _parse_optimade_response(data: dict) -> dict:
    """Parse an OPTIMADE /structures JSON response.

    Isolated for easy mocking in tests.
    """
    hits = data.get("data", [])
    meta = data.get("meta", {})
    total = (
        meta.get("data_returned")
        or meta.get("more_data_available")  # sometimes a bool
        or len(hits)
    )
    if isinstance(total, bool):
        total = len(hits)

    stoichiometries: list[str] = []
    functional_histogram: dict[str, int] = {}

    for entry in hits:
        attrs = entry.get("attributes", {})
        formula = attrs.get("chemical_formula_reduced")
        if formula and formula not in stoichiometries:
            stoichiometries.append(formula)

    return {
        "n_entries": int(total) if total else len(hits),
        "functional_histogram": functional_histogram,  # OPTIMADE doesn't expose XC
        "nearest_stoichiometries": stoichiometries[:10],
        "lattice": None,
        "magnetic": None,
        "source": "optimade",
    }


def _query_nomad(
    elements: list[str],
    reduced_formula: str | None,
    timeout: float,
    client,
) -> dict | None:
    """POST to NOMAD.  Returns parsed dict or None on failure."""
    try:
        body = _build_nomad_query(elements, reduced_formula)
        resp = client.post(_NOMAD_ENTRIES_URL, json=body, timeout=timeout)
        resp.raise_for_status()
        return _parse_nomad_response(resp.json())
    except Exception:
        return None


def _query_optimade(
    elements: list[str],
    reduced_formula: str | None,
    timeout: float,
    client,
) -> dict | None:
    """GET OPTIMADE fallback. Returns parsed dict or None on failure."""
    try:
        filt = _build_optimade_filter(elements, reduced_formula)
        params = {"filter": filt, "page_limit": _OPTIMADE_PAGE_LIMIT}
        resp = client.get(_OPTIMADE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        return _parse_optimade_response(resp.json())
    except Exception:
        return None


def run_external_reference_gate(
    elements: list[str],
    reduced_formula: str | None = None,
    space_group: str | None = None,
    timeout: float = 10.0,
    live: bool = True,
) -> dict:
    """N-07 external reference gate (MCP-callable entry point).

    Parameters
    ----------
    elements:
        Chemical element symbols required in the target structure (e.g. ["Fe","S"]).
    reduced_formula:
        Optional reduced formula to narrow the search (e.g. "FeS2").
    space_group:
        Optional space-group label (informational only, echoed in result).
    timeout:
        HTTP timeout in seconds (per request).
    live:
        If False, skip all network calls and return UNKNOWN/offline verdict.
        Useful for tests and offline environments.

    Returns
    -------
    response_envelope dict.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    next_actions: list[str] = []

    # Normalise element list
    elements = sorted(set(str(e).strip().capitalize() for e in elements))
    if not elements:
        raise ValueError("elements must be a non-empty list of chemical symbols")

    # Offline / test mode
    if not live:
        return response_envelope(
            tool="external_reference_gate",
            verdict="UNKNOWN",
            confidence="low",
            status="error",
            reasons=["live=False: network calls suppressed (offline/test mode)"],
            next_actions=["re-run with live=True for a real reference check"],
            result={
                "elements": elements,
                "reduced_formula": reduced_formula,
                "space_group": space_group,
                "exists": None,
                "n_entries": None,
                "functional_histogram": {},
                "nearest_stoichiometries": [],
                "lattice": None,
                "magnetic": None,
                "source": "offline",
            },
        )

    # Live queries
    parsed: dict | None = None
    query_errors: list[str] = []

    with httpx.Client() as client:
        parsed = _query_nomad(elements, reduced_formula, timeout, client)
        if parsed is None:
            query_errors.append("NOMAD query failed (network error or timeout)")
            parsed = _query_optimade(elements, reduced_formula, timeout, client)
            if parsed is None:
                query_errors.append("OPTIMADE fallback also failed")

    if parsed is None:
        # Both sources failed
        for err in query_errors:
            warnings.append(err)
        return response_envelope(
            tool="external_reference_gate",
            verdict="UNKNOWN",
            confidence="low",
            status="error",
            reasons=["All external sources unreachable; cannot assess reference coverage."],
            warnings=warnings,
            next_actions=[
                "check network connectivity",
                "retry with longer --timeout",
                "proceed with heightened internal validation",
            ],
            result={
                "elements": elements,
                "reduced_formula": reduced_formula,
                "space_group": space_group,
                "exists": None,
                "n_entries": None,
                "functional_histogram": {},
                "nearest_stoichiometries": [],
                "lattice": None,
                "magnetic": None,
                "source": "none",
            },
        )

    for err in query_errors:
        warnings.append(err)

    n_entries = parsed["n_entries"]
    exists = n_entries > 0

    result = {
        "elements": elements,
        "reduced_formula": reduced_formula,
        "space_group": space_group,
        "exists": exists,
        "n_entries": n_entries,
        "functional_histogram": parsed["functional_histogram"],
        "nearest_stoichiometries": parsed["nearest_stoichiometries"],
        "lattice": parsed["lattice"],
        "magnetic": parsed["magnetic"],
        "source": parsed["source"],
    }

    if exists:
        verdict = "REFERENCE_FOUND"
        confidence = "high"
        reasons.append(
            f"Found {n_entries} entries for elements={elements}"
            + (f", formula={reduced_formula}" if reduced_formula else "")
            + f" in {parsed['source'].upper()}."
        )
        if parsed["functional_histogram"]:
            hist_str = ", ".join(
                f"{k}:{v}" for k, v in sorted(
                    parsed["functional_histogram"].items(), key=lambda x: -x[1]
                )
            )
            reasons.append(f"XC functional distribution: {hist_str}.")
        if parsed["nearest_stoichiometries"]:
            reasons.append(
                "Nearest stoichiometries: "
                + ", ".join(parsed["nearest_stoichiometries"][:5]) + "."
            )
        if parsed["magnetic"]:
            reasons.append(
                "Reported magnetic orderings: " + ", ".join(parsed["magnetic"]) + "."
            )
        next_actions = [
            "cross-check functional coverage (GGA/GGA+U/hybrid) against planned protocol",
            "verify lattice parameters against experimental reference",
            "review magnetic ordering coverage before choosing nspin",
        ]
    else:
        verdict = "NO_EXTERNAL_REFERENCE"
        confidence = "medium"
        reasons.append(
            f"No entries found for elements={elements}"
            + (f", formula={reduced_formula}" if reduced_formula else "")
            + f" in {parsed['source'].upper()}."
        )
        reasons.append(
            "No external reference means this compound/composition is uncharted territory; "
            "internal validation bars must be higher."
        )
        next_actions = [
            "run a short single-point DFT smoke test to validate structure stability",
            "add an extra convergence tier (e.g. k-point or cutoff scan)",
            "run a spin-collapse check before choosing nspin for production",
            "consider generating a reference structure via CrystalFormer / MatterGen",
        ]

    return response_envelope(
        tool="external_reference_gate",
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        next_actions=next_actions,
        warnings=warnings,
        result=result,
    )


def print_gate(env: dict) -> None:
    r = env.get("result") or {}
    print(f"verdict\t{env['verdict']}\tconfidence\t{env['confidence']}\tstatus\t{env['status']}")
    print(f"elements\t{r.get('elements')}\tformula\t{r.get('reduced_formula')}\tsg\t{r.get('space_group')}")
    print(f"exists\t{r.get('exists')}\tn_entries\t{r.get('n_entries')}\tsource\t{r.get('source')}")
    if r.get("functional_histogram"):
        print(f"functionals\t{r.get('functional_histogram')}")
    if r.get("nearest_stoichiometries"):
        print(f"stoichiometries\t{', '.join(r.get('nearest_stoichiometries', []))}")
    for x in env["reasons"]:
        print(f"reason\t{x}")
    for x in env["next_actions"]:
        print(f"next\t{x}")
    for x in env["warnings"]:
        print(f"warning\t{x}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "N-07 external reference gate: query NOMAD + OPTIMADE for existing DFT "
            "data before committing expensive compute on a new mineral."
        )
    )
    p.add_argument(
        "--elements", nargs="+", required=True,
        help="chemical element symbols, e.g. --elements Fe S"
    )
    p.add_argument(
        "--reduced-formula", default=None,
        help="optional reduced formula to narrow search, e.g. FeS2"
    )
    p.add_argument(
        "--space-group", default=None,
        help="optional space-group label (informational, e.g. Pa-3)"
    )
    p.add_argument(
        "--timeout", type=float, default=10.0,
        help="HTTP timeout per request in seconds (default 10.0)"
    )
    p.add_argument(
        "--offline", action="store_true",
        help="suppress network calls (offline/test mode); returns UNKNOWN"
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    env = run_external_reference_gate(
        elements=args.elements,
        reduced_formula=args.reduced_formula,
        space_group=args.space_group,
        timeout=args.timeout,
        live=not args.offline,
    )
    if args.output:
        dump_json(env, args.output)
    if args.json:
        dump_json(env)
    elif not args.output:
        print_gate(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
