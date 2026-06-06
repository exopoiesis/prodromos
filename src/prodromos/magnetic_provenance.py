"""magnetic_provenance -- cross-check COMPUTED (Materials Project) vs EXPERIMENTAL
(MAGNDATA) magnetic ordering and route the NEB seed to the trustworthy source
(roadmap §C / §C-bis).

MP's *computed* magnetic ground state frequently disagrees with neutron experiment:
this session found MP labels troilite (mp-2099), LiFePO4 (mp-19017) and alpha-/beta-
NaFeO2 all FM where MAGNDATA / Rousse / McQueen give AFM. Seeding an NEB from a
wrong (FM) ordering puts both endpoints on the wrong magnetic sheet from the start.

This gate compares the two orderings and, on disagreement, WARNs and routes the
seed to the EXPERIMENTAL block (MAGNDATA is ground truth; MP is the surrogate).
The pure :func:`compare_ordering` is the tested core; :func:`run_magnetic_provenance`
adds best-effort live MP fetch (env ``MP_API_KEY`` or ``secrets/mp_api_key.json``).

See [[feedback_mp_use_local_mp_client]]: the MCP ``import_mp`` server cannot see the
MP key (env-only in its process); this gate also reads ``secrets/mp_api_key.json``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prodromos.cli_contract import dump_json, response_envelope

# Normalise the many ordering spellings to: NM / FM / AFM / FiM / None(unknown).
_ORDER_MAP = {
    "nm": "NM", "non-magnetic": "NM", "nonmagnetic": "NM", "dia": "NM", "diamagnetic": "NM",
    "fm": "FM", "ferromagnetic": "FM", "ferromagnet": "FM",
    "afm": "AFM", "antiferromagnetic": "AFM", "antiferromagnet": "AFM",
    "afm-g": "AFM", "afm-a": "AFM", "afm-c": "AFM", "afm-e": "AFM",
    "fim": "FiM", "ferri": "FiM", "ferrimagnetic": "FiM", "ferrimagnet": "FiM",
}


def normalize_ordering(value: str | None) -> str | None:
    if value is None:
        return None
    return _ORDER_MAP.get(str(value).strip().lower(), str(value).strip().upper())


def _is_magnetic(order: str | None) -> bool | None:
    if order is None:
        return None
    return order in ("FM", "AFM", "FiM")


def compare_ordering(mp_ordering: str | None, magndata_ordering: str | None) -> dict:
    """Compare a COMPUTED (MP) and EXPERIMENTAL (MAGNDATA) ordering.

    Returns ``{verdict, mp_ordering, magndata_ordering, agree, seed_source, warning}``.
    Verdicts: AGREE / CONFLICT_BINARY (magnetic-vs-nonmagnetic) / CONFLICT_TYPE
    (FM<->AFM<->FiM) / MAGNDATA_ONLY / MP_ONLY / NO_DATA. The seed source is the
    EXPERIMENTAL block whenever MAGNDATA exists (it is ground truth), else MP.
    """
    mp = normalize_ordering(mp_ordering)
    md = normalize_ordering(magndata_ordering)

    if mp is None and md is None:
        return {"verdict": "NO_DATA", "mp_ordering": None, "magndata_ordering": None,
                "agree": None, "seed_source": None,
                "warning": "no magnetic ordering from MP or MAGNDATA"}
    if md is None:
        return {"verdict": "MP_ONLY", "mp_ordering": mp, "magndata_ordering": None,
                "agree": None, "seed_source": "mp",
                "warning": "no experimental (MAGNDATA) anchor -- MP computed ordering is "
                           "unverified; MP often mislabels Fe sulfides/phosphates FM"}
    if mp is None:
        return {"verdict": "MAGNDATA_ONLY", "mp_ordering": None, "magndata_ordering": md,
                "agree": None, "seed_source": "magndata",
                "warning": None}

    agree = mp == md
    if agree:
        return {"verdict": "AGREE", "mp_ordering": mp, "magndata_ordering": md,
                "agree": True, "seed_source": "magndata", "warning": None}

    binary_conflict = _is_magnetic(mp) != _is_magnetic(md)
    verdict = "CONFLICT_BINARY" if binary_conflict else "CONFLICT_TYPE"
    warning = (
        f"MP computed ordering ({mp}) DISAGREES with MAGNDATA experiment ({md}) "
        f"[{'magnetic-vs-nonmagnetic' if binary_conflict else 'ordering-type'} conflict] "
        f"-> seed the NEB from the EXPERIMENTAL (MAGNDATA) block, NOT the MP magmoms"
    )
    return {"verdict": verdict, "mp_ordering": mp, "magndata_ordering": md,
            "agree": False, "seed_source": "magndata", "warning": warning}


def _read_mp_key() -> str | None:
    """MP key from env, else ``secrets/mp_api_key.json`` walking up from cwd."""
    key = os.environ.get("MP_API_KEY")
    if key:
        return key
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "secrets" / "mp_api_key.json"
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                k = data.get("MP_API_KEY") or data.get("api_key") or data.get("key")
                if k:
                    return str(k)
            except (OSError, ValueError):
                pass
    return None


def _fetch_mp_ordering(material_id, formula, space_group) -> str | None:
    key = _read_mp_key()
    if key and not os.environ.get("MP_API_KEY"):
        os.environ["MP_API_KEY"] = key
    from tm_spec.importers.mp import fetch_to_tm_spec

    docs = fetch_to_tm_spec(material_id=material_id, formula=formula, space_group=space_group)
    if not docs:
        return None
    return (docs[0].get("magnetic") or {}).get("state")


def _fetch_magndata_ordering(code) -> str | None:
    from tm_spec.importers.magndata import fetch_to_tm_spec

    doc = fetch_to_tm_spec(code)
    return (doc.get("magnetic") or {}).get("state")


def run_magnetic_provenance(
    mp_ordering: str | None = None,
    magndata_ordering: str | None = None,
    material_id: str | None = None,
    formula: str | None = None,
    space_group: int | None = None,
    magndata_code: str | None = None,
    live: bool = False,
) -> dict:
    """Gate: compare MP vs MAGNDATA ordering, route the seed, WARN on conflict.

    Supply the orderings directly (``mp_ordering`` / ``magndata_ordering``) for the
    pure $0 comparison, OR set ``live=True`` with ``material_id``/``formula`` (MP) and
    ``magndata_code`` (MAGNDATA) to fetch them. Live fetch degrades softly (a fetch
    error leaves that side ``None`` and is reported in ``reasons``).
    """
    reasons: list[str] = []
    if live:
        if mp_ordering is None and (material_id or formula):
            try:
                mp_ordering = _fetch_mp_ordering(material_id, formula, space_group)
            except Exception as exc:  # soft-degrade -- never raise to the LLM
                reasons.append(f"MP fetch failed: {type(exc).__name__}: {exc}")
        if magndata_ordering is None and magndata_code:
            try:
                magndata_ordering = _fetch_magndata_ordering(magndata_code)
            except Exception as exc:
                reasons.append(f"MAGNDATA fetch failed: {type(exc).__name__}: {exc}")

    cmp = compare_ordering(mp_ordering, magndata_ordering)
    warnings = [cmp["warning"]] if cmp["warning"] else []
    reasons.append(
        f"MP={cmp['mp_ordering']} vs MAGNDATA={cmp['magndata_ordering']} -> "
        f"seed_source={cmp['seed_source']}"
    )
    confidence = "high" if cmp["verdict"] in ("AGREE", "CONFLICT_BINARY") else "medium"
    return response_envelope(
        tool="magnetic_provenance",
        verdict=cmp["verdict"],
        confidence=confidence,
        reasons=reasons,
        warnings=warnings,
        next_actions=(
            ["seed the NEB endpoints from the experimental MAGNDATA magnetic block"]
            if cmp["seed_source"] == "magndata" and not cmp["agree"]
            else []
        ),
        result=cmp,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Magnetic-provenance gate: MP-computed vs MAGNDATA-experimental ordering"
    )
    p.add_argument("--mp-ordering", default=None, help="MP computed ordering (NM/FM/AFM/FiM)")
    p.add_argument("--magndata-ordering", default=None, help="MAGNDATA experimental ordering")
    p.add_argument("--material-id", default=None, help="MP material id (live fetch)")
    p.add_argument("--formula", default=None, help="formula for MP live fetch")
    p.add_argument("--space-group", type=int, default=None)
    p.add_argument("--magndata-code", default=None, help="MAGNDATA entry code (live fetch)")
    p.add_argument("--live", action="store_true", default=False, help="fetch missing orderings online")
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    env = run_magnetic_provenance(
        mp_ordering=args.mp_ordering,
        magndata_ordering=args.magndata_ordering,
        material_id=args.material_id,
        formula=args.formula,
        space_group=args.space_group,
        magndata_code=args.magndata_code,
        live=args.live,
    )
    if args.output:
        dump_json(env, args.output)
    if args.json:
        dump_json(env)
    elif not args.output:
        print(f"verdict\t{env['verdict']}\tseed_source\t{env['result']['seed_source']}")
        for r in env["reasons"]:
            print(f"reason\t{r}")
        for w in env["warnings"]:
            print(f"warning\t{w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
