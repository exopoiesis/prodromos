"""Endpoint magnetic pre-flight gate for DFT NEB inputs.

The gate compares two completed endpoint calculations after they have been
parsed by ``magnetic_output_parser.py``.  It is deliberately conservative:
magnetic sheet disagreement is a NO-GO only when both outputs are complete
enough to trust; incomplete or missing magnetic data becomes REVIEW.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from prodromos.magnetic_output_parser import MagneticOutputSummary, parse_output_file
from prodromos.spin_split_detector import DELTA_ABS_ADJ, DELTA_TOTAL_ENDPOINT


@dataclass
class EndpointGateResult:
    verdict: str
    endpoint_split: bool
    delta_total_uB: float | None
    delta_abs_uB: float | None
    reasons: list[str] = field(default_factory=list)
    endpoint_a: dict | None = None
    endpoint_b: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def endpoint_magnetic_gate(
    endpoint_a: MagneticOutputSummary,
    endpoint_b: MagneticOutputSummary,
    delta_total_threshold: float = DELTA_TOTAL_ENDPOINT,
    delta_abs_threshold: float = DELTA_ABS_ADJ,
) -> EndpointGateResult:
    reasons: list[str] = []
    total_a = endpoint_a.total_magnetization_uB
    total_b = endpoint_b.total_magnetization_uB
    abs_a = endpoint_a.absolute_magnetization_uB
    abs_b = endpoint_b.absolute_magnetization_uB

    missing_mag = total_a is None or total_b is None or abs_a is None or abs_b is None
    if missing_mag:
        reasons.append("missing total/absolute magnetization in at least one endpoint")

    incomplete = endpoint_a.scf_converged is False or endpoint_b.scf_converged is False
    if incomplete:
        reasons.append("at least one endpoint SCF is not converged or output is truncated")

    delta_total = abs(total_a - total_b) if total_a is not None and total_b is not None else None
    delta_abs = abs(abs_a - abs_b) if abs_a is not None and abs_b is not None else None
    endpoint_split = (
        (delta_total is not None and delta_total > delta_total_threshold)
        or (delta_abs is not None and delta_abs > delta_abs_threshold)
    )
    if endpoint_split:
        reasons.append(
            "endpoint magnetic sheets differ: "
            f"delta_total={delta_total:.3g} uB, delta_abs={delta_abs:.3g} uB"
        )

    if missing_mag:
        verdict = "REVIEW"
    elif endpoint_split and not incomplete:
        verdict = "NO-GO_SINGLE_SHEET"
        reasons.append("single-sheet NEB is ill-posed; run endpoint cross-check at shared constrained M")
    elif endpoint_split and incomplete:
        verdict = "REVIEW"
        reasons.append("magnetic split is visible, but incomplete SCF prevents a hard NO-GO")
    elif incomplete:
        verdict = "REVIEW"
    else:
        verdict = "GO"
        reasons.append("endpoint total/absolute magnetization differences are below gate thresholds")

    return EndpointGateResult(
        verdict=verdict,
        endpoint_split=endpoint_split,
        delta_total_uB=delta_total,
        delta_abs_uB=delta_abs,
        reasons=reasons,
        endpoint_a=endpoint_a.to_dict(),
        endpoint_b=endpoint_b.to_dict(),
    )


def print_gate(result: EndpointGateResult) -> None:
    print(f"verdict\t{result.verdict}")
    print(f"endpoint_split\t{'yes' if result.endpoint_split else 'no'}")
    print(f"delta_total_uB\t{_fmt(result.delta_total_uB)}")
    print(f"delta_abs_uB\t{_fmt(result.delta_abs_uB)}")
    for reason in result.reasons:
        print(f"reason\t{reason}")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint_a", type=Path)
    parser.add_argument("endpoint_b", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit normalized JSON")
    parser.add_argument("--delta-total", type=float, default=DELTA_TOTAL_ENDPOINT)
    parser.add_argument("--delta-abs", type=float, default=DELTA_ABS_ADJ)
    args = parser.parse_args(argv)

    result = endpoint_magnetic_gate(
        parse_output_file(args.endpoint_a),
        parse_output_file(args.endpoint_b),
        delta_total_threshold=args.delta_total,
        delta_abs_threshold=args.delta_abs,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print_gate(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
