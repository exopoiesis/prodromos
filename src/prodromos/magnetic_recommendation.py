"""Magnetic pre-flight recommendation layer.

Combines a band-level magnetic gate with an optional endpoint matrix where the
same endpoints were recomputed at candidate total magnetizations.  The output
is an operational recommendation: reuse, rerun constrained-M, run 4-SCF matrix,
or branch to a two-sheet/MECP treatment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re

from prodromos.magnetic_band_gate import BandGateResult, analyze_band_images, load_band
from prodromos.magnetic_output_parser import MagneticOutputSummary, parse_output_file


@dataclass
class EndpointMatrixEntry:
    endpoint: str
    target_label: str
    path: str
    scf_converged: bool | None
    energy_eV: float | None
    total_magnetization_uB: float | None
    absolute_magnetization_uB: float | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MagneticRecommendation:
    action: str
    confidence: str
    constrained_magnetization_uB: float | None
    seam_edge: int | None
    reasons: list[str] = field(default_factory=list)
    required_next_calculations: list[str] = field(default_factory=list)
    endpoint_matrix: list[EndpointMatrixEntry] = field(default_factory=list)
    band_verdict: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_recommendation(
    band_root: str | Path | None = None,
    endpoint_matrix_dir: str | Path | None = None,
) -> MagneticRecommendation:
    band_result = analyze_band_images(load_band(band_root)) if band_root is not None else None
    matrix = load_endpoint_matrix(endpoint_matrix_dir) if endpoint_matrix_dir is not None else []

    if band_result and band_result.verdict == "GO":
        return MagneticRecommendation(
            action="ACCEPT_MAGNETIC_CONTINUITY",
            confidence="medium",
            constrained_magnetization_uB=None,
            seam_edge=None,
            reasons=["band has no adjacent or endpoint magnetic discontinuity"],
            band_verdict=band_result.verdict,
            endpoint_matrix=matrix,
        )

    if band_result and _band_missing_magnetization(band_result):
        return MagneticRecommendation(
            action="REVIEW_MAGNETIC_INPUT",
            confidence="high",
            constrained_magnetization_uB=None,
            seam_edge=None,
            reasons=["band outputs do not contain final magnetization; cannot certify magnetic continuity"],
            required_next_calculations=["confirm nspin/starting_magnetization settings or rerun magnetic single-points"],
            band_verdict=band_result.verdict,
            endpoint_matrix=matrix,
        )

    if not matrix:
        return MagneticRecommendation(
            action="RUN_ENDPOINT_MATRIX",
            confidence="high" if band_result and band_result.sheet_crossing else "medium",
            constrained_magnetization_uB=None,
            seam_edge=band_result.crossing_edge if band_result and band_result.crossing_edge >= 0 else None,
            reasons=["need endpoint energies on candidate magnetic sheets before choosing constrained-M vs MECP"],
            required_next_calculations=["run both endpoints at both candidate total magnetizations"],
            band_verdict=band_result.verdict if band_result else None,
        )

    matrix_decision = choose_common_endpoint_sheet(matrix)
    reasons = list(matrix_decision["reasons"])
    if band_result and band_result.sheet_crossing:
        reasons.append(
            f"existing band crosses magnetic sheets at image edge {band_result.crossing_edge}<->{band_result.crossing_edge + 1}"
        )

    if matrix_decision["common_label"] is not None:
        constrained_m = matrix_decision["common_m"]
        confidence = "high" if matrix_decision["all_winners_converged"] else "medium"
        if matrix_decision["has_incomplete_competitor"]:
            confidence = "medium"
            reasons.append("one competing endpoint calculation is incomplete; rerun for paper-grade energy ordering")
        return MagneticRecommendation(
            action="RERUN_SINGLE_SHEET_CONSTRAINED_M",
            confidence=confidence,
            constrained_magnetization_uB=constrained_m,
            seam_edge=band_result.crossing_edge if band_result and band_result.crossing_edge >= 0 else None,
            reasons=reasons,
            required_next_calculations=[
                f"run pilot single-points for old seam images at tot_magnetization={constrained_m:.2f}",
                f"if pilot is continuous, rerun NEB with tot_magnetization={constrained_m:.2f} for all images",
                "initialize image magnetization by propagating converged local moments from neighboring images",
            ],
            endpoint_matrix=matrix,
            band_verdict=band_result.verdict if band_result else None,
        )

    return MagneticRecommendation(
        action="BRANCH_TO_MECP_OR_TWO_SEGMENT",
        confidence="medium",
        constrained_magnetization_uB=None,
        seam_edge=band_result.crossing_edge if band_result and band_result.crossing_edge >= 0 else None,
        reasons=reasons,
        required_next_calculations=[
            "split path at the magnetic seam",
            "run separate single-sheet segments or MECP search near the crossing edge",
        ],
        endpoint_matrix=matrix,
        band_verdict=band_result.verdict if band_result else None,
    )


def load_endpoint_matrix(root: str | Path | None) -> list[EndpointMatrixEntry]:
    if root is None:
        return []
    entries: list[EndpointMatrixEntry] = []
    for path in sorted(Path(root).glob("*.pwo")):
        parsed = _parse_endpoint_matrix_name(path)
        if parsed is None:
            continue
        endpoint, target_label = parsed
        summary = parse_output_file(path)
        entries.append(_entry_from_summary(endpoint, target_label, summary))
    return entries


def choose_common_endpoint_sheet(entries: list[EndpointMatrixEntry]) -> dict:
    by_endpoint: dict[str, list[EndpointMatrixEntry]] = {}
    for entry in entries:
        by_endpoint.setdefault(entry.endpoint, []).append(entry)

    winners: dict[str, EndpointMatrixEntry] = {}
    reasons: list[str] = []
    all_winners_converged = True
    has_incomplete_competitor = False
    for endpoint, endpoint_entries in sorted(by_endpoint.items()):
        converged = [entry for entry in endpoint_entries if entry.scf_converged and entry.energy_eV is not None]
        if not converged:
            reasons.append(f"{endpoint}: no converged endpoint-matrix energies")
            continue
        winner = min(converged, key=lambda entry: entry.energy_eV or float("inf"))
        winners[endpoint] = winner
        all_winners_converged = all_winners_converged and bool(winner.scf_converged)
        if any(entry.scf_converged is False for entry in endpoint_entries):
            has_incomplete_competitor = True
        for entry in sorted(endpoint_entries, key=lambda item: item.target_label):
            if entry.energy_eV is not None:
                delta = entry.energy_eV - winner.energy_eV
                status = "converged" if entry.scf_converged else "incomplete"
                reasons.append(f"{endpoint} {entry.target_label}: ΔE={delta:.3f} eV vs local best ({status})")

    labels = {winner.target_label for winner in winners.values()}
    if len(winners) >= 2 and len(labels) == 1:
        label = next(iter(labels))
        moments = [winner.total_magnetization_uB for winner in winners.values() if winner.total_magnetization_uB is not None]
        common_m = sum(moments) / len(moments) if moments else _label_to_magnetization(label)
        reasons.append(f"all endpoints prefer the same magnetic target {label}")
        return {
            "common_label": label,
            "common_m": common_m,
            "reasons": reasons,
            "all_winners_converged": all_winners_converged,
            "has_incomplete_competitor": has_incomplete_competitor,
        }

    if labels:
        reasons.append("endpoints do not share the same lowest-energy magnetic target")
    return {
        "common_label": None,
        "common_m": None,
        "reasons": reasons,
        "all_winners_converged": all_winners_converged,
        "has_incomplete_competitor": has_incomplete_competitor,
    }


def _entry_from_summary(endpoint: str, target_label: str, summary: MagneticOutputSummary) -> EndpointMatrixEntry:
    return EndpointMatrixEntry(
        endpoint=endpoint,
        target_label=target_label,
        path=summary.path,
        scf_converged=summary.scf_converged,
        energy_eV=summary.energy_eV,
        total_magnetization_uB=summary.total_magnetization_uB,
        absolute_magnetization_uB=summary.absolute_magnetization_uB,
        warnings=summary.warnings,
    )


def _parse_endpoint_matrix_name(path: Path) -> tuple[str, str] | None:
    match = re.search(r"(end[A-Za-z0-9]+)_m(\d+)", path.stem)
    if not match:
        return None
    return match.group(1), f"m{match.group(2)}"


def _label_to_magnetization(label: str) -> float | None:
    match = re.fullmatch(r"m(\d+)", label)
    if not match:
        return None
    digits = match.group(1)
    if len(digits) <= 2:
        return float(digits)
    return float(digits) / (10 ** (len(digits) - 1))


# ---------------------------------------------------------------------------
# N-14: within-method energy delta helper
# ---------------------------------------------------------------------------

class ProvenanceMismatchError(ValueError):
    """Raised when two summaries carry incompatible provenance metadata.

    DFT+U total energies are not on the same absolute scale as U=0 energies:
    the double-counting correction shifts all levels by a U-dependent constant.
    Any numeric energy difference between summaries with different (u_eff,
    nspin, functional, ecut, kpts) is therefore meaningless.  This error
    signals that the caller should not use the returned difference.
    """

    def __init__(self, mismatches: dict[str, tuple]) -> None:
        self.mismatches = mismatches
        parts = "; ".join(
            f"{key}: {a!r} vs {b!r}" for key, (a, b) in sorted(mismatches.items())
        )
        super().__init__(f"PROVENANCE_MISMATCH — {parts}")


def _provenance_key(summary: MagneticOutputSummary) -> dict[str, object]:
    return {
        "u_eff": summary.u_eff,
        "nspin": summary.nspin,
        "functional": summary.functional,
        "ecut": summary.ecut,
        "kpts": summary.kpts,
    }


def compute_within_method_delta(
    summary_a: MagneticOutputSummary,
    summary_b: MagneticOutputSummary,
) -> float:
    """Return ``energy_b - energy_a`` (eV) after asserting shared provenance.

    Provenance dimensions compared: ``u_eff``, ``nspin``, ``functional``,
    ``ecut``, ``kpts``.  A dimension is considered "unknown" when **both**
    summaries report ``None`` for it — unknown-vs-unknown is allowed so that
    callers do not need to populate every field.  A ``None`` on one side and
    a concrete value on the other is treated as a mismatch, because it means
    one calculation was run with a known setting while the other was not
    (different run conditions).

    Raises ``ProvenanceMismatchError`` (carries ``mismatches`` dict) if any
    dimension differs.  The caller must not silently fall back to a numeric
    result when this error is raised.

    Raises ``ValueError`` if either summary is missing ``energy_eV``.
    """
    prov_a = _provenance_key(summary_a)
    prov_b = _provenance_key(summary_b)
    mismatches: dict[str, tuple] = {}
    for key in prov_a:
        va, vb = prov_a[key], prov_b[key]
        if va != vb:
            mismatches[key] = (va, vb)
    if mismatches:
        raise ProvenanceMismatchError(mismatches)

    if summary_a.energy_eV is None or summary_b.energy_eV is None:
        raise ValueError(
            "compute_within_method_delta: one or both summaries have energy_eV=None"
        )
    return summary_b.energy_eV - summary_a.energy_eV


def _band_missing_magnetization(result: BandGateResult) -> bool:
    return any(
        image.summary.total_magnetization_uB is None or image.summary.absolute_magnetization_uB is None
        for image in result.images
    )


def print_recommendation(rec: MagneticRecommendation) -> None:
    print(f"action\t{rec.action}")
    print(f"confidence\t{rec.confidence}")
    print(f"band_verdict\t{rec.band_verdict or '-'}")
    print(f"constrained_magnetization_uB\t{_fmt(rec.constrained_magnetization_uB)}")
    print(f"seam_edge\t{rec.seam_edge if rec.seam_edge is not None else '-'}")
    for reason in rec.reasons:
        print(f"reason\t{reason}")
    for calc in rec.required_next_calculations:
        print(f"next\t{calc}")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band-root", type=Path)
    parser.add_argument("--endpoint-matrix-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rec = build_recommendation(args.band_root, args.endpoint_matrix_dir)
    if args.json:
        print(json.dumps(rec.to_dict(), indent=2, sort_keys=True))
    else:
        print_recommendation(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
