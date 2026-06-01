"""N-02 -- endpoint provenance gate: single-point energy on an MLIP-relaxed geometry
is NOT a valid endpoint energy.

A DFT single-point (SP) energy is only meaningful as an NEB endpoint when the
geometry is a stationary point on the DFT PES -- i.e. when it was DFT-relaxed
(ionic relaxation under DFT forces). If the geometry was produced by an MLIP
relaxation and only a DFT SP was run on it, the global lattice is NOT a DFT
stationary point (grad_DFT V != 0). The resulting energy can be ~20 eV off the
true DFT minimum even when *local* bond lengths (H-S, Fe-S, etc.) look perfectly
physical.

Key insight: local bond-geometry checks (saddle_proximity_gate, bond-length
filters) test NECESSARY but NOT SUFFICIENT conditions. A geometry can pass all
local checks and still be ~20 eV above the DFT minimum because distant atoms are
displaced from their DFT-optimal positions. Only a DFT ionic relaxation certifies
the geometry as a DFT stationary point.

Verdicts:
    ENDPOINT_VALID            -- provenance == 'dft_relaxed'; energy is a valid
                                 DFT endpoint energy.
    NOT_AN_ENDPOINT_MLIP_GEOMETRY -- provenance == 'mlip_relaxed' (or any non-DFT
                                 provenance); energy is not a valid endpoint energy;
                                 DFT ionic relaxation required first.

Any energy-based downstream verdict (barrier ranking, NEB endpoint delta-E) is
DOWNGRADED when provenance != 'dft_relaxed', regardless of local bond geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prodromos.cli_contract import response_envelope, dump_json

TOOL = "endpoint_provenance_gate"

# Recognised provenance tokens
_DFT_RELAXED = "dft_relaxed"
_MLIP_RELAXED = "mlip_relaxed"

# Energy downgrade label inserted into energy-based verdicts
_ENERGY_DOWNGRADE_NOTE = (
    "ENERGY DOWNGRADED: geometry is not DFT-relaxed (grad_DFT V != 0). "
    "SP energy on an MLIP-relaxed geometry can be ~20 eV above the true DFT minimum. "
    "Local bond-length checks are necessary but NOT sufficient to certify an endpoint. "
    "Run DFT ionic relaxation first."
)


def run_endpoint_provenance_gate(
    provenance: str,
    bond_geometry_ok: bool | None = None,
    energy_eV: float | None = None,
    label: str | None = None,
) -> dict:
    """Core gate. Returns a response_envelope dict.

    Parameters
    ----------
    provenance : str
        Geometry provenance string. Canonical values: 'dft_relaxed',
        'mlip_relaxed'. Any value other than 'dft_relaxed' triggers
        NOT_AN_ENDPOINT_MLIP_GEOMETRY.
    bond_geometry_ok : bool or None
        Result of a local bond-geometry check (e.g. saddle_proximity_gate,
        H-S bond-length filter). If True, a note clarifies that local geometry
        passing is necessary but NOT sufficient. Does NOT change the verdict.
    energy_eV : float or None
        Optional SP energy (eV) for echo in result. If provenance is not
        dft_relaxed, the energy is flagged as invalid for barrier ranking.
    label : str or None
        Optional human-readable label for the endpoint (e.g. 'endA', 'endB').
    """
    reasons: list[str] = []
    warnings: list[str] = []
    next_actions: list[str] = []

    prov_norm = str(provenance).strip().lower()
    is_dft_relaxed = prov_norm == _DFT_RELAXED
    energy_valid = is_dft_relaxed

    # --- verdict routing ----
    if is_dft_relaxed:
        verdict = "ENDPOINT_VALID"
        confidence = "high"
        reasons.append(
            f"provenance='{provenance}': geometry was DFT-relaxed (ionic relaxation "
            "under DFT forces). The SP energy is a valid DFT endpoint energy at a "
            "stationary point on the DFT PES."
        )
        if bond_geometry_ok is True:
            reasons.append(
                "bond_geometry_ok=True: local bond lengths are physically reasonable "
                "(necessary condition satisfied)."
            )
        elif bond_geometry_ok is False:
            warnings.append(
                "bond_geometry_ok=False: local bond geometry failed; endpoint may be "
                "at an unusual geometry despite DFT relaxation. Inspect the structure."
            )
        next_actions.append(
            "endpoint energy is valid; proceed to NEB setup or barrier ranking"
        )

    else:
        verdict = "NOT_AN_ENDPOINT_MLIP_GEOMETRY"
        confidence = "high"
        reasons.append(
            f"provenance='{provenance}': geometry was NOT DFT-relaxed. "
            "A DFT single-point on a non-DFT-relaxed geometry is computed at a point "
            "where grad_DFT V != 0. The global lattice is NOT a DFT stationary point, "
            "so the energy can be ~20 eV above the true DFT minimum even when local "
            "bond lengths look physical."
        )
        if bond_geometry_ok is True:
            reasons.append(
                "bond_geometry_ok=True: local bond lengths are physically reasonable, "
                "but this is a NECESSARY condition only -- NOT sufficient. The distant "
                "atoms may still be far from their DFT-optimal positions, making the "
                "SP energy invalid for barrier ranking or NEB endpoint setup."
            )
        elif bond_geometry_ok is False:
            reasons.append(
                "bond_geometry_ok=False: local bond geometry also failed, compounding "
                "the provenance issue."
            )
        warnings.append(_ENERGY_DOWNGRADE_NOTE)
        next_actions.append(
            "run DFT ionic relaxation (BFGS/FIRE) on this geometry to obtain a true "
            "DFT stationary point before using the energy for barrier ranking or NEB"
        )
        next_actions.append(
            "after DFT relaxation, re-run endpoint_provenance_gate with "
            "provenance='dft_relaxed' to certify the endpoint"
        )
        next_actions.append(
            "do NOT use this SP energy in barrier ranking or NEB delta-E comparisons; "
            "even if saddle_proximity_gate / bond-length gate pass, the energy is "
            "not a valid DFT endpoint"
        )

    result = {
        "label": label,
        "provenance": provenance,
        "provenance_normalised": prov_norm,
        "is_dft_relaxed": is_dft_relaxed,
        "energy_eV": energy_eV,
        "energy_valid_for_ranking": energy_valid,
        "bond_geometry_ok": bond_geometry_ok,
        "energy_downgraded": not energy_valid,
    }

    return response_envelope(
        tool=TOOL,
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        next_actions=next_actions,
        warnings=warnings,
        result=result,
    )


def print_gate(env: dict) -> None:
    r = env["result"] or {}
    print(f"verdict\t{env['verdict']}\tconfidence\t{env['confidence']}")
    print(f"provenance\t{r.get('provenance')}\tis_dft_relaxed\t{r.get('is_dft_relaxed')}")
    print(f"energy_eV\t{r.get('energy_eV')}\tenergy_valid_for_ranking\t{r.get('energy_valid_for_ranking')}")
    print(f"bond_geometry_ok\t{r.get('bond_geometry_ok')}\tenergy_downgraded\t{r.get('energy_downgraded')}")
    for x in env["reasons"]:
        print(f"reason\t{x}")
    for x in env["next_actions"]:
        print(f"next\t{x}")
    for x in env["warnings"]:
        print(f"warning\t{x}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "N-02 endpoint provenance gate: verify that a DFT single-point energy "
            "is on a DFT-relaxed geometry (not MLIP-relaxed)."
        )
    )
    p.add_argument(
        "--provenance", required=True,
        help="geometry provenance: 'dft_relaxed' or 'mlip_relaxed' (or other string)",
    )
    p.add_argument(
        "--bond-geometry-ok", type=lambda s: s.lower() in ("true", "1", "yes"),
        default=None,
        help="result of local bond-geometry check: true/false (optional)",
    )
    p.add_argument(
        "--energy-ev", type=float, default=None,
        help="optional DFT SP energy in eV (echoed in result, flagged if invalid)",
    )
    p.add_argument(
        "--label", type=str, default=None,
        help="optional endpoint label (e.g. endA, endB)",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    env = run_endpoint_provenance_gate(
        provenance=args.provenance,
        bond_geometry_ok=args.bond_geometry_ok,
        energy_eV=args.energy_ev,
        label=args.label,
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
