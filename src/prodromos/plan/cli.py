"""``prodromos plan`` CLI -- stateless pre-flight planner over a tm-spec case.

Usage::

    prodromos plan CASE.tm.yaml [--mode route|tree] [--emit envelope|preflight]
                   [--json] [--output PATH]

The case document is validated through ``tm_spec.validator`` first (gate-0
``INVALID_CASE``); state lives entirely in the case doc (stateless, consilium S6).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prodromos.cli_contract import dump_json, response_envelope
from prodromos.plan.emit import to_envelope, to_preflight_block
from prodromos.plan.interpret import walk
from prodromos.plan.policy import select_policy_graph

TOOL = "plan"

# Suffixes that are already tm-spec documents (no conversion needed).
_TMSPEC_SUFFIXES = {".yaml", ".yml", ".json", ".jsonl", ".ndjson"}


def _looks_like_tmspec(case_path: Path) -> bool:
    """True if the case path is already a tm-spec document we can load directly."""
    name = case_path.name.lower()
    if name.endswith(".tm.yaml") or name.endswith(".tm.yml"):
        return True
    return case_path.suffix.lower() in _TMSPEC_SUFFIXES


def _autoconvert_inputs(case_path: Path, code: str) -> tuple[dict | None, list[str], str]:
    """Auto-convert QE/ABACUS input files to a tm-spec doc.

    Returns (doc_or_None, reasons, detected_code). On failure doc is None and
    reasons carries the human-actionable message (typically: pass --code).
    """
    import datetime as _dt

    from prodromos.from_inputs import convert_to_tmspec, detect_code

    try:
        detected = detect_code(case_path) if code == "auto" else code
    except ValueError as exc:
        return None, [
            f"case '{case_path}' is not a tm-spec document and the code could not "
            f"be auto-detected ({exc}); re-run with --code qe|abacus"
        ], code
    # The in-memory stub is ephemeral (planned, not persisted), so a real date
    # is used here so the schema id-pattern is satisfied and the planner can run.
    # `prodromos from-inputs` itself keeps the deterministic 'YYYY-MM-DD' default.
    today = _dt.date.today().isoformat()
    try:
        doc = convert_to_tmspec(case_path, code=detected, date=today)
    except (FileNotFoundError, ValueError) as exc:
        return None, [f"auto-conversion of '{case_path}' failed: {exc}"], detected
    reason = (
        f"case auto-converted from {detected.upper()} input '{case_path.name}' "
        f"to a tm-spec/0.3 stub (prodromos from-inputs); complete [TODO_HUMAN] "
        "fields before paper-grade use"
    )
    return doc, [reason], detected


def _invalid_case_envelope(reasons: list[str]) -> dict:
    return response_envelope(
        tool=TOOL,
        verdict="INVALID_CASE",
        confidence="low",
        status="error",
        reasons=reasons,
        next_actions=["fix the tm-spec document so it validates against its schema"],
        result={"schema_errors": reasons},
    )


def _load_and_validate(
    case_path: Path, *, code: str = "auto"
) -> tuple[dict | None, dict | None, list[str]]:
    """Return (doc, error_envelope, extra_reasons).

    On success error_envelope is None. ``extra_reasons`` records onboarding
    glue notes (e.g. that the case was auto-converted from a QE/ABACUS input).
    When the case path is NOT a tm-spec document (a .in / INPUT / directory),
    it is auto-converted via ``prodromos from-inputs`` first.
    """
    try:
        from tm_spec.validator import load_doc, validate_doc
    except ModuleNotFoundError:
        return None, _invalid_case_envelope([
            "tm-spec is not installed; install it (pip install -e .[plan]) "
            "to validate and plan over tm-spec cases"
        ]), []
    if not case_path.exists():
        return None, _invalid_case_envelope([f"case file not found: {case_path}"]), []

    extra_reasons: list[str] = []
    if _looks_like_tmspec(case_path) and not case_path.is_dir():
        try:
            docs = load_doc(case_path)
        except Exception as exc:  # noqa: BLE001 -- surface any parse error as INVALID_CASE
            return None, _invalid_case_envelope([f"parse error: {exc}"]), []
        if not docs:
            return None, _invalid_case_envelope(["empty document"]), []
        doc = docs[0]
    else:
        # Onboarding glue: a bare engine input file/dir -> convert on the fly.
        doc, reasons, _detected = _autoconvert_inputs(case_path, code)
        if doc is None:
            return None, _invalid_case_envelope(reasons), []
        extra_reasons = reasons

    schema_errs, rule_issues = validate_doc(doc)
    errors = [f"{loc}: {msg}" for loc, msg in schema_errs]
    errors += [msg for level, msg in rule_issues if level == "error"]
    if errors:
        return None, _invalid_case_envelope(errors), []
    return doc, None, extra_reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prodromos plan",
        description="Stateless pre-flight planner over a tm-spec case document.",
    )
    parser.add_argument(
        "case", type=Path,
        help="path to a .tm.yaml / .json tm-spec case, OR a QE .in / ABACUS "
        "INPUT / ABACUS run directory (auto-converted via from-inputs)",
    )
    parser.add_argument(
        "--code", choices=["qe", "abacus", "auto"], default="auto",
        help="code for auto-conversion when the case is a bare input file "
        "(default: auto-detect)",
    )
    parser.add_argument(
        "--mode",
        choices=["route", "tree"],
        default="route",
        help="route = execute gates and recommend one next step (default); "
        "tree = scored strategy tree (Bellman expectimax + CVaR)",
    )
    parser.add_argument(
        "--emit",
        choices=["envelope", "preflight"],
        default="envelope",
        help="envelope = response_envelope (default); preflight = tm-spec 0.3 preflight block",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="remaining budget (USD); tree mode uses beta=cost_run/budget for CVaR "
        "tail control. Omit for pure expected-value scoring.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="tree mode: keep only the top-K ranked strategies",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--output", help="optional output path (JSON)")
    args = parser.parse_args(argv)

    doc, err, extra_reasons = _load_and_validate(args.case, code=args.code)
    if err is not None:
        if args.output:
            dump_json(err, args.output)
        dump_json(err)
        return 1

    result = walk(
        select_policy_graph(doc),
        doc,
        mode=args.mode,
        budget_usd=args.budget_usd,
        top_k=args.top_k,
    )
    payload = to_envelope(result) if args.emit == "envelope" else to_preflight_block(result)

    # Onboarding glue: record that the case was auto-converted from a raw input.
    if extra_reasons:
        if isinstance(payload.get("reasons"), list):
            payload["reasons"] = extra_reasons + payload["reasons"]
        else:
            payload.setdefault("warnings", [])
            payload["warnings"] = extra_reasons + list(payload.get("warnings") or [])

    if args.output:
        dump_json(payload, args.output)
    # default: always print JSON for `plan` (the envelope/block IS the product)
    dump_json(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
