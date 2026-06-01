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
from prodromos.plan.policy import POLICY_GRAPH

TOOL = "plan"


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


def _load_and_validate(case_path: Path) -> tuple[dict | None, dict | None]:
    """Return (doc, error_envelope). On success error_envelope is None."""
    try:
        from tm_spec.validator import load_doc, validate_doc
    except ModuleNotFoundError:
        return None, _invalid_case_envelope([
            "tm-spec is not installed; install it (pip install -e .[plan]) "
            "to validate and plan over tm-spec cases"
        ])
    if not case_path.exists():
        return None, _invalid_case_envelope([f"case file not found: {case_path}"])
    try:
        docs = load_doc(case_path)
    except Exception as exc:  # noqa: BLE001 -- surface any parse error as INVALID_CASE
        return None, _invalid_case_envelope([f"parse error: {exc}"])
    if not docs:
        return None, _invalid_case_envelope(["empty document"])
    doc = docs[0]
    schema_errs, rule_issues = validate_doc(doc)
    errors = [f"{loc}: {msg}" for loc, msg in schema_errs]
    errors += [msg for level, msg in rule_issues if level == "error"]
    if errors:
        return None, _invalid_case_envelope(errors)
    return doc, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prodromos plan",
        description="Stateless pre-flight planner over a tm-spec case document.",
    )
    parser.add_argument("case", type=Path, help="path to a .tm.yaml / .json tm-spec case")
    parser.add_argument(
        "--mode",
        choices=["route", "tree"],
        default="route",
        help="route = execute gates and recommend one next step (default); "
        "tree = scored strategy tree (STUB, next increment)",
    )
    parser.add_argument(
        "--emit",
        choices=["envelope", "preflight"],
        default="envelope",
        help="envelope = response_envelope (default); preflight = tm-spec 0.3 preflight block",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--output", help="optional output path (JSON)")
    args = parser.parse_args(argv)

    doc, err = _load_and_validate(args.case)
    if err is not None:
        if args.output:
            dump_json(err, args.output)
        dump_json(err)
        return 1

    result = walk(POLICY_GRAPH, doc, mode=args.mode)
    payload = to_envelope(result) if args.emit == "envelope" else to_preflight_block(result)

    if args.output:
        dump_json(payload, args.output)
    # default: always print JSON for `plan` (the envelope/block IS the product)
    dump_json(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
