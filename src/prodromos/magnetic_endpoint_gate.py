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


# Default per-TM relative endpoint threshold (uB per magnetic atom). A genuine
# spin-sheet crossing flips an atom by ~2x its local moment; a smooth large-cell
# drift is a few 0.01 uB/TM. So normalising delta_abs by N_magnetic separates the two.
DELTA_ABS_PER_TM = 0.30


@dataclass
class EndpointGateResult:
    verdict: str
    endpoint_split: bool
    delta_total_uB: float | None
    delta_abs_uB: float | None
    reasons: list[str] = field(default_factory=list)
    endpoint_a: dict | None = None
    endpoint_b: dict | None = None
    delta_abs_per_tm_uB: float | None = None
    n_magnetic: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def endpoint_magnetic_gate(
    endpoint_a: MagneticOutputSummary,
    endpoint_b: MagneticOutputSummary,
    delta_total_threshold: float = DELTA_TOTAL_ENDPOINT,
    delta_abs_threshold: float = DELTA_ABS_ADJ,
    n_magnetic: int | None = None,
    delta_abs_per_tm_threshold: float = DELTA_ABS_PER_TM,
) -> EndpointGateResult:
    """Endpoint magnetic sheet gate.

    When ``n_magnetic`` (the number of magnetic/TM atoms) is given, the absolute
    channel uses a PER-TM relative criterion: ``delta_abs`` only counts as a sheet
    split if it ALSO exceeds ``delta_abs_per_tm_threshold`` uB *per magnetic atom*.
    This stops large-cell slow-drift systems from false-NO-GO (troilite FeS V_S+H:
    delta_abs=0.52 uB just trips the 0.5 absolute threshold, but is 0.043 uB/Fe over
    12 Fe -- a smooth drift, not a crossing; the band gate confirms GO). The total
    channel (integer ~2 uB sheet jumps) keeps the absolute threshold.
    """
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

    delta_abs_per_tm = (
        delta_abs / n_magnetic if (delta_abs is not None and n_magnetic) else None
    )

    total_split = delta_total is not None and delta_total > delta_total_threshold
    abs_split = delta_abs is not None and delta_abs > delta_abs_threshold
    # per-TM downgrade: a slow drift trips the absolute threshold but is small per atom
    if abs_split and delta_abs_per_tm is not None and delta_abs_per_tm <= delta_abs_per_tm_threshold:
        abs_split = False
        reasons.append(
            f"delta_abs={delta_abs:.3g} uB exceeds the absolute threshold but is only "
            f"{delta_abs_per_tm:.3g} uB/TM over {n_magnetic} magnetic atoms "
            f"(<= {delta_abs_per_tm_threshold} uB/TM) -> smooth drift, NOT a sheet crossing "
            f"(defer to the band gate if a trajectory exists)"
        )

    endpoint_split = total_split or abs_split
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
        delta_abs_per_tm_uB=delta_abs_per_tm,
        n_magnetic=n_magnetic,
        reasons=reasons,
        endpoint_a=endpoint_a.to_dict(),
        endpoint_b=endpoint_b.to_dict(),
    )


def reconcile_endpoint_and_band(
    endpoint: EndpointGateResult,
    band: "object | None" = None,
) -> dict:
    """Auto-reconcile the endpoint screen with the full-trajectory band gate.

    The band gate is the ARBITER (it sees every image's moment); the endpoint gate
    is the pre-launch screen for when no trajectory exists yet. When band data
    exists and the two disagree (classic: endpoint NO-GO vs band GO -- troilite),
    the combined verdict follows the band gate, and both verdicts + the resolution
    are reported in ONE envelope. ``band`` is a ``BandGateResult`` (or None).
    """
    ep_v = endpoint.verdict
    if band is None:
        return {
            "combined_verdict": ep_v,
            "arbiter": "endpoint",
            "endpoint_verdict": ep_v,
            "band_verdict": None,
            "resolution": "no band trajectory supplied; endpoint screen stands",
        }
    band_v = getattr(band, "verdict", None)
    agree = ep_v == band_v
    resolution = (
        "endpoint and band agree"
        if agree
        else f"band gate (full trajectory) is the arbiter: {band_v} supersedes endpoint {ep_v}"
    )
    return {
        "combined_verdict": band_v,
        "arbiter": "band",
        "endpoint_verdict": ep_v,
        "band_verdict": band_v,
        "agree": agree,
        "resolution": resolution,
    }


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
    parser.add_argument("--n-magnetic", type=int, default=None,
                        help="number of magnetic/TM atoms -> use the per-TM relative threshold")
    parser.add_argument("--delta-abs-per-tm", type=float, default=DELTA_ABS_PER_TM,
                        help="per-TM relative abs-magnetization threshold (uB/TM)")
    args = parser.parse_args(argv)

    result = endpoint_magnetic_gate(
        parse_output_file(args.endpoint_a),
        parse_output_file(args.endpoint_b),
        delta_total_threshold=args.delta_total,
        delta_abs_threshold=args.delta_abs,
        n_magnetic=args.n_magnetic,
        delta_abs_per_tm_threshold=args.delta_abs_per_tm,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print_gate(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
