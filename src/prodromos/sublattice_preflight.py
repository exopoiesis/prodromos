"""sublattice_preflight -- structure-level (pre-DFT) magnetic-sublattice crossing
predictor for an ion-migration NEB (roadmap §C-bis, the flagship new gate).

Every existing magnetic sheet gate (``magnetic_endpoint`` / ``magnetic_band``)
consumes *DFT outputs* -- it fires only AFTER the expensive SCF. But which magnetic
sublattice a migrant (or its charge-compensating redox polaron) sits next to is
fixed by **structure + per-site moment signs**, available at $0 from MAGNDATA / MP.
This gate makes the sheet verdict truly pre-DFT.

Two modes:

* ``mode="migrant"`` -- the migrating ion itself is magnetic (e.g. an Fe hop).
  We track the migrant's nearest magnetic-host neighbour's sublattice SIGN at
  endpoint A vs B. A sign flip => the single-sheet NEB would interpolate through a
  spin-sheet crossing => ill-posed barrier.

* ``mode="polaron"`` -- NEW FAILURE MODE. The migrant (Li+/Na+) is *nonmagnetic*
  (naively GO), but its hop is charge-compensated by a **redox polaron** (Fe2+ ->
  Fe3+, d6 -> d5): one host site changes its moment. If that polaron localizes on a
  DIFFERENT magnetic sublattice at A vs B, M_total shifts by ~2 uB and the
  single-sheet NEB interpolates through an unphysical polaron spin-flip. So a
  "nonmagnetic-migrant" cathode hop can be GO *or* NO-GO depending only on where the
  compensating polaron lands -- invisible to a migrant-spin-only analysis. The
  polaron site is either given explicitly (``polaron_index_a/b``) or inferred as the
  migrant's nearest redox-active host neighbour at each endpoint.

Verdicts: GO_SINGLE_SHEET / NO-GO_SINGLE_SHEET / REVIEW.
When NO-GO, the gate emits a ready constrained-M / per-site starting_magnetization
recipe (the routing destination IS a methodology prescription -- the framework's
value made concrete).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from prodromos.cli_contract import dump_json, response_envelope


@dataclass
class MagSite:
    """A magnetic host site with a sublattice sign."""

    element: str
    frac: tuple[float, float, float]
    sign: int  # +1 / -1 (sublattice); 0 = nonmagnetic / undetermined
    moment_uB: float | None = None
    label: str | None = None


def _mic_frac_delta(a: tuple[float, float, float], b: tuple[float, float, float]) -> list[float]:
    """Minimum-image fractional displacement a->b (each component wrapped to [-0.5, 0.5))."""
    return [(b[i] - a[i]) - round(b[i] - a[i]) for i in range(3)]


def _cart_norm(df: list[float], cell: list[list[float]]) -> float:
    """Cartesian length of a fractional displacement under ``cell`` (rows = vectors)."""
    x = [sum(df[k] * cell[k][j] for k in range(3)) for j in range(3)]
    return math.sqrt(sum(c * c for c in x))


def nearest_site(
    migrant_frac: tuple[float, float, float],
    sites: list[MagSite],
    cell: list[list[float]],
    candidates: list[int] | None = None,
) -> tuple[int, float]:
    """Index + cartesian distance (MIC) of the nearest site to ``migrant_frac``.

    ``candidates`` restricts the search to those site indices (e.g. only
    redox-active magnetic sites for polaron inference).
    """
    pool = candidates if candidates is not None else range(len(sites))
    best_i, best_d = -1, math.inf
    for i in pool:
        d = _cart_norm(_mic_frac_delta(migrant_frac, sites[i].frac), cell)
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


def sites_from_magmoms(
    elements: list[str],
    fracs: list[tuple[float, float, float]],
    magmoms_uB: list[float],
    mag_elements: set[str] | None = None,
    zero_tol: float = 0.2,
) -> list[MagSite]:
    """Build magnetic sites from a (tm-spec / MP / MAGNDATA) magmom list.

    Sign = sign(magmom); sites with |magmom| < ``zero_tol`` are sign 0 (treated as
    nonmagnetic). ``mag_elements`` (if given) restricts which species are kept.
    """
    out: list[MagSite] = []
    for el, fr, m in zip(elements, fracs, magmoms_uB):
        if mag_elements is not None and el not in mag_elements:
            continue
        sign = 0 if abs(m) < zero_tol else (1 if m > 0 else -1)
        out.append(MagSite(element=el, frac=tuple(fr), sign=sign, moment_uB=float(m)))
    return out


def assign_signs_by_coordinate(
    sites: list[MagSite],
    axis: int = 1,
    bands: tuple[tuple[float, float, int], ...] = ((0.0, 0.5, 1), (0.5, 1.0, -1)),
) -> list[MagSite]:
    """Assign sublattice signs from a fractional-coordinate band rule.

    The magnetic Wyckoff/coordinate rule for collinear AFM hosts: e.g. LiFePO4
    olivine has Fe on 4c with y ~= 0.25 / 0.75 forming the two AFM sublattices, so
    ``axis=1, bands=((0,0.5,+1),(0.5,1.0,-1))`` separates them. Sites whose axis
    coordinate falls in no band keep sign 0.
    """
    out: list[MagSite] = []
    for s in sites:
        c = s.frac[axis] - math.floor(s.frac[axis])  # wrap to [0,1)
        sign = 0
        for lo, hi, sg in bands:
            if lo <= c < hi:
                sign = sg
                break
        out.append(MagSite(element=s.element, frac=s.frac, sign=sign,
                           moment_uB=s.moment_uB, label=s.label))
    return out


def _recipe(site_a: MagSite, site_b: MagSite, mode: str, migrant_species: str) -> list[str]:
    """The constrained-M / two-species recipe a NO-GO must hand back."""
    who = "migrant" if mode == "migrant" else f"compensating redox polaron (nonmagnetic {migrant_species}+)"
    return [
        f"NO-GO is a single-sheet artefact: the {who} sits on sublattice "
        f"{site_a.sign:+d} at endpoint A but {site_b.sign:+d} at endpoint B "
        f"(M_total would shift ~2 uB across the band).",
        "split the host magnetic species into TWO species (up/down) with opposite "
        "per-site starting_magnetization so the relevant moment keeps its native "
        "sublattice sign along the WHOLE path (build the domain wall explicitly).",
        f"force the {who} site with per-site starting_magnetization "
        f"({site_a.sign:+d} on its A-site, hold it through the band) and verify by "
        "Lowdin d-occupation (d6 vs d5) that the polaron stayed put.",
        "OR run two single-sheet NEBs (A-sublattice-fixed and B-sublattice-fixed) "
        "and report the lower; the naive single-species NEB cannot test this.",
    ]


def run_sublattice_preflight(
    sites: list[MagSite],
    cell: list[list[float]],
    migrant_a: tuple[float, float, float],
    migrant_b: tuple[float, float, float],
    mode: str = "migrant",
    polaron_index_a: int | None = None,
    polaron_index_b: int | None = None,
    migrant_species: str = "Li",
    redox_elements: set[str] | None = None,
) -> dict:
    """Predict GO / NO-GO single-sheet for an ion-migration hop at $0 (pre-DFT).

    Parameters
    ----------
    sites : magnetic host sites (use ``sites_from_magmoms`` or
        ``assign_signs_by_coordinate`` to build them).
    cell : 3x3 lattice (rows = lattice vectors, Angstrom) for MIC distances.
    migrant_a, migrant_b : fractional coords of the migrant at endpoints A, B.
    mode : ``"migrant"`` (migrant is magnetic) or ``"polaron"`` (nonmagnetic
        migrant -> track the charge-compensating redox polaron).
    polaron_index_a/b : explicit host-site indices carrying the polaron at A/B
        (``mode="polaron"`` only); if omitted, inferred as the migrant's nearest
        redox-active magnetic neighbour.
    redox_elements : species eligible to host the polaron (default: any signed site).
    """
    signed = [i for i, s in enumerate(sites) if s.sign != 0]
    if not signed:
        return response_envelope(
            tool="sublattice_preflight",
            verdict="REVIEW",
            confidence="low",
            reasons=["no signed magnetic sublattice sites supplied -- cannot assign sublattices "
                     "(provide magmoms via sites_from_magmoms or a coordinate rule)"],
            result={"mode": mode, "n_signed_sites": 0},
        )

    if mode not in ("migrant", "polaron"):
        return response_envelope(
            tool="sublattice_preflight", verdict="REVIEW", confidence="low",
            reasons=[f"unknown mode {mode!r} (use 'migrant' or 'polaron')"],
            result={"mode": mode},
        )

    # candidate sites for the relevant moment
    if redox_elements is not None:
        cand = [i for i in signed if sites[i].element in redox_elements]
    else:
        cand = signed
    if not cand:
        cand = signed

    if mode == "polaron" and polaron_index_a is not None and polaron_index_b is not None:
        ia, da = polaron_index_a, _cart_norm(_mic_frac_delta(migrant_a, sites[polaron_index_a].frac), cell)
        ib, db = polaron_index_b, _cart_norm(_mic_frac_delta(migrant_b, sites[polaron_index_b].frac), cell)
    else:
        ia, da = nearest_site(migrant_a, sites, cell, candidates=cand)
        ib, db = nearest_site(migrant_b, sites, cell, candidates=cand)

    sa, sb = sites[ia], sites[ib]
    crossing = sa.sign != sb.sign

    detail = {
        "mode": mode,
        "migrant_species": migrant_species,
        "n_signed_sites": len(signed),
        "endpoint_a": {"site_index": ia, "element": sa.element, "sign": sa.sign,
                       "moment_uB": sa.moment_uB, "distance_A": round(da, 3)},
        "endpoint_b": {"site_index": ib, "element": sb.element, "sign": sb.sign,
                       "moment_uB": sb.moment_uB, "distance_A": round(db, 3)},
        "sublattice_sign_flips": crossing,
    }

    if crossing:
        return response_envelope(
            tool="sublattice_preflight",
            verdict="NO-GO_SINGLE_SHEET",
            confidence="medium",
            reasons=[
                f"the relevant {'migrant' if mode == 'migrant' else 'redox-polaron'} moment "
                f"changes magnetic sublattice between endpoints "
                f"(A: site {ia} {sa.element} sign {sa.sign:+d}; "
                f"B: site {ib} {sb.element} sign {sb.sign:+d}) -> a single-sheet NEB would "
                f"interpolate through an unphysical spin-flip (predicted dM_total ~ 2 uB).",
            ],
            next_actions=_recipe(sa, sb, mode, migrant_species),
            result=detail,
        )
    return response_envelope(
        tool="sublattice_preflight",
        verdict="GO_SINGLE_SHEET",
        confidence="medium",
        reasons=[
            f"the relevant moment stays on ONE magnetic sublattice "
            f"(sign {sa.sign:+d}) across both endpoints -> the single-sheet NEB is "
            f"well-posed; DFT only to confirm.",
        ],
        next_actions=["proceed with a standard single-sheet NEB; confirm endpoints with "
                      "magnetic_endpoint (DFT) -- this gate predicts, DFT verifies"],
        result=detail,
    )


def run_from_dict(spec: dict) -> dict:
    """Run the gate from a JSON-style spec.

    Expected keys: ``sites`` (list of ``{element, frac:[x,y,z], sign?|moment_uB?,
    label?}``), ``cell`` (3x3), ``migrant_a`` / ``migrant_b`` ([x,y,z]); optional
    ``mode``, ``polaron_index_a/b``, ``migrant_species``, ``redox_elements``.
    """
    sites: list[MagSite] = []
    for s in spec["sites"]:
        sign = s.get("sign")
        if sign is None:
            m = float(s.get("moment_uB") or 0.0)
            sign = 0 if abs(m) < 0.2 else (1 if m > 0 else -1)
        sites.append(
            MagSite(element=s["element"], frac=tuple(s["frac"]), sign=int(sign),
                    moment_uB=s.get("moment_uB"), label=s.get("label"))
        )
    redox = spec.get("redox_elements")
    return run_sublattice_preflight(
        sites,
        spec["cell"],
        tuple(spec["migrant_a"]),
        tuple(spec["migrant_b"]),
        mode=spec.get("mode", "migrant"),
        polaron_index_a=spec.get("polaron_index_a"),
        polaron_index_b=spec.get("polaron_index_b"),
        migrant_species=spec.get("migrant_species", "Li"),
        redox_elements=set(redox) if redox else None,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Structure-level (pre-DFT) magnetic-sublattice crossing predictor"
    )
    p.add_argument("--input", required=True,
                   help="JSON file: {sites, cell, migrant_a, migrant_b, mode, ...}")
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    spec = json.loads(Path(args.input).read_text(encoding="utf-8"))
    env = run_from_dict(spec)
    if args.output:
        dump_json(env, args.output)
    if args.json:
        dump_json(env)
    elif not args.output:
        print(f"verdict\t{env['verdict']}\tconfidence\t{env['confidence']}")
        for r in env["reasons"]:
            print(f"reason\t{r}")
        for a in env["next_actions"]:
            print(f"next\t{a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
