"""Scan a harvested DFT dataset tree for magnetic NEB band risks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Sequence

from prodromos.magnetic_band_gate import analyze_band_images, load_band
from prodromos.magnetic_output_parser import MagneticOutputSummary

logger = logging.getLogger(__name__)


@dataclass
class DatasetBandRow:
    band_root: str
    n_images: int
    verdict: str
    sheet_crossing: bool
    endpoint_split: bool
    crossing_edge: int
    max_delta_abs_uB: float | None
    max_delta_total_uB: float | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def find_band_roots(root: str | Path, min_images: int = 2) -> list[Path]:
    """Find directories whose immediate children are NEB ``image_*`` dirs."""
    root = Path(root)
    candidates = [root] if root.is_dir() else []
    if root.is_dir():
        candidates.extend(path for path in root.rglob("*") if path.is_dir())

    band_roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        image_children = [
            child
            for child in candidate.iterdir()
            if child.is_dir() and child.name.lower().startswith("image_")
        ]
        if len(image_children) >= min_images and candidate not in seen:
            band_roots.append(candidate)
            seen.add(candidate)
    return sorted(band_roots, key=lambda path: str(path).lower())


def scan_dataset(root: str | Path, min_images: int = 2, max_bands: int | None = None) -> list[DatasetBandRow]:
    rows: list[DatasetBandRow] = []
    band_roots = find_band_roots(root, min_images=min_images)
    if max_bands is not None:
        band_roots = band_roots[:max_bands]

    for band_root in band_roots:
        try:
            result = analyze_band_images(load_band(band_root))
            rows.append(_row_from_result(band_root, result))
        except Exception as exc:  # pragma: no cover - defensive CLI path
            rows.append(
                DatasetBandRow(
                    band_root=str(band_root),
                    n_images=0,
                    verdict="ERROR",
                    sheet_crossing=False,
                    endpoint_split=False,
                    crossing_edge=-1,
                    max_delta_abs_uB=None,
                    max_delta_total_uB=None,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return rows


def _row_from_result(band_root: Path, result) -> DatasetBandRow:
    return DatasetBandRow(
        band_root=str(band_root),
        n_images=len(result.images),
        verdict=result.verdict,
        sheet_crossing=result.sheet_crossing,
        endpoint_split=result.endpoint_split,
        crossing_edge=result.crossing_edge,
        max_delta_abs_uB=max(result.delta_abs_adj) if result.delta_abs_adj else None,
        max_delta_total_uB=max(result.delta_total_adj) if result.delta_total_adj else None,
        reason=result.reasons[0] if result.reasons else "",
    )


def print_dataset_table(rows: list[DatasetBandRow]) -> None:
    print("verdict\tn_images\tsheet_crossing\tendpoint_split\tmax_dabs\tmax_dtotal\tcrossing_edge\tband_root\treason")
    for row in rows:
        print(
            "\t".join(
                [
                    row.verdict,
                    str(row.n_images),
                    _yes_no(row.sheet_crossing),
                    _yes_no(row.endpoint_split),
                    _fmt(row.max_delta_abs_uB),
                    _fmt(row.max_delta_total_uB),
                    str(row.crossing_edge) if row.crossing_edge >= 0 else "-",
                    row.band_root,
                    row.reason,
                ]
            )
        )


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


# ---------------------------------------------------------------------------
# N-04: dedup by job label
# ---------------------------------------------------------------------------

def dedup_by_label(
    summaries: Sequence[MagneticOutputSummary],
    label_attr: str = "path",
    get_label: "None | (lambda MagneticOutputSummary: str)" = None,
) -> list[MagneticOutputSummary]:
    """Return one summary per job label, preferring the converged copy.

    Parameters
    ----------
    summaries:
        Iterable of parsed summaries (may include partial runs from crashed
        workers — e.g. a 1-line .pwo with scf_converged=False, n_atoms=0).
    get_label:
        Callable that maps a summary to its job label string.  Defaults to
        using ``summary.path`` (the file path), which works when each job
        writes to a uniquely-named file.  Callers that use a flat directory
        where multiple worker copies of the same job exist should pass a
        function that strips the worker-directory component, e.g.::

            get_label=lambda s: Path(s.path).stem.rsplit("_worker", 1)[0]

    Returns
    -------
    list[MagneticOutputSummary]
        One entry per unique label.  When multiple summaries share a label:
        - the converged copy (``scf_converged=True``) is preferred;
        - if multiple converged copies exist, the one with the highest
          absolute energy (least negative, i.e. no data corruption) is kept
          and a warning is logged — callers should inspect the disagreement;
        - if all copies are unconverged, the one with the most complete
          energy (energy_eV is not None) is returned.

    Side-effects
    ------------
    Logs a WARNING for each label where two *converged* copies report
    different energies (|ΔE| > 1 meV) — this indicates worker-directory
    inconsistency that the caller should investigate.
    """
    if get_label is None:
        get_label = lambda s: s.path  # noqa: E731

    by_label: dict[str, list[MagneticOutputSummary]] = {}
    for s in summaries:
        label = get_label(s)
        by_label.setdefault(label, []).append(s)

    result: list[MagneticOutputSummary] = []
    for label, group in by_label.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        converged = [s for s in group if s.scf_converged is True]
        if converged:
            # Warn when two converged copies disagree on energy
            energies = [s.energy_eV for s in converged if s.energy_eV is not None]
            if len(energies) >= 2 and (max(energies) - min(energies)) > 1e-3:
                e_str = ", ".join(f"{e:.6f}" for e in energies)
                logger.warning(
                    "dedup_by_label: duplicate-label energy disagreement for %r: "
                    "converged copies report [%s] eV — keeping lowest-energy copy",
                    label,
                    e_str,
                )
            # Keep the lowest-energy (most negative) converged copy
            best = min(
                converged,
                key=lambda s: s.energy_eV if s.energy_eV is not None else float("inf"),
            )
        else:
            # No converged copy: prefer the one with any energy
            with_energy = [s for s in group if s.energy_eV is not None]
            best = with_energy[0] if with_energy else group[0]

        result.append(best)
    return result


# ---------------------------------------------------------------------------
# N-01: per-sheet grouping (free-magnetization screen / multi-sheet guard)
# ---------------------------------------------------------------------------

#: Tolerance for grouping Mtot values onto the "same" magnetic sheet.
#: Within ±MTOT_SHEET_TOL of each other → same sheet candidate.
MTOT_SHEET_TOL = 0.5  # μB

#: Tolerance for grouping Mabs values.
MABS_SHEET_TOL = 1.0  # μB


@dataclass
class PerSheetRanking:
    """Result of rank_per_sheet.

    Attributes
    ----------
    sheets:
        Dict mapping a sheet label (e.g. ``"sheet_0"``) to the list of
        summaries belonging to that sheet, sorted by ascending energy_eV
        (None energies last).
    cross_sheet_ranking_valid:
        ``True`` only when ALL summaries belong to the same sheet and all
        have a non-None energy_eV.  Any cross-sheet energy comparison is
        meaningless when this is ``False``.
    verdict:
        Human-readable verdict string: ``"SINGLE_SHEET"`` /
        ``"MULTI_SHEET"`` / ``"INSUFFICIENT_DATA"``.
    reasons:
        List of explanation strings.

    Notes
    -----
    Free-magnetization (no ``tot_magnetization`` constraint) SCF runs each
    converge to whatever magnetic sheet the solver finds first.  When
    different runs find *different* sheets (Mtot varies at roughly constant
    Mabs, or Mabs itself splits), ranking their energies cross-sheet is
    invalid: the apparent energy difference includes the inter-sheet gap,
    not the intra-sheet barrier.  This function groups summaries by sheet
    identity and returns per-sheet rankings.  Class-level energy screening
    (is the *set* of calculations as a whole converged?) is still valid
    after grouping; endpoint-level comparison must use summaries from the
    same sheet.
    """

    sheets: dict[str, list[MagneticOutputSummary]]
    cross_sheet_ranking_valid: bool
    verdict: str
    reasons: list[str]


def _sheet_label(
    summary: MagneticOutputSummary,
    sheet_mtot_centers: list[float],
    sheet_mabs_centers: list[float],
) -> str | None:
    """Return the sheet label index (as string) that best matches *summary*.

    Returns None when neither Mtot nor Mabs is available.
    """
    mtot = summary.total_magnetization_uB
    mabs = summary.absolute_magnetization_uB
    if mtot is None and mabs is None:
        return None
    best_idx: int | None = None
    best_dist = float("inf")
    for idx, (mc, ac) in enumerate(zip(sheet_mtot_centers, sheet_mabs_centers)):
        d_mtot = abs(mtot - mc) if mtot is not None and mc is not None else 0.0
        d_mabs = abs(mabs - ac) if mabs is not None and ac is not None else 0.0
        dist = d_mtot + d_mabs
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return f"sheet_{best_idx}" if best_idx is not None else None


def _cluster_values(values: list[float], tol: float) -> list[float]:
    """Greedy 1-D clustering: return list of cluster centers."""
    if not values:
        return []
    centers: list[float] = []
    for v in sorted(values):
        for i, c in enumerate(centers):
            if abs(v - c) <= tol:
                # update center with running mean
                centers[i] = c + (v - c) / 2
                break
        else:
            centers.append(v)
    return centers


def rank_per_sheet(
    summaries: Sequence[MagneticOutputSummary],
    mtot_tol: float = MTOT_SHEET_TOL,
    mabs_tol: float = MABS_SHEET_TOL,
) -> PerSheetRanking:
    """Group summaries by magnetic sheet and return per-sheet energy rankings.

    When a free-M screen produces summaries with varying Mtot (and/or Mabs),
    cross-sheet energy comparison is invalid.  This function:

    1. Clusters Mtot values (and Mabs as secondary axis) to identify sheets.
    2. Assigns each summary to a sheet.
    3. Returns per-sheet lists sorted by ascending energy_eV.
    4. Sets ``cross_sheet_ranking_valid=True`` only for single-sheet results.

    The ``MULTI_SHEET`` verdict is **not** a hard blocker; it signals that
    the caller must work within a single sheet at a time.
    """
    reasons: list[str] = []

    # Collect usable values for clustering
    mtot_vals = [s.total_magnetization_uB for s in summaries if s.total_magnetization_uB is not None]
    mabs_vals = [s.absolute_magnetization_uB for s in summaries if s.absolute_magnetization_uB is not None]

    if not mtot_vals and not mabs_vals:
        return PerSheetRanking(
            sheets={},
            cross_sheet_ranking_valid=False,
            verdict="INSUFFICIENT_DATA",
            reasons=["no total or absolute magnetization available in any summary"],
        )

    mtot_centers = _cluster_values(mtot_vals, mtot_tol) if mtot_vals else [0.0]
    mabs_centers = _cluster_values(mabs_vals, mabs_tol) if mabs_vals else [0.0]

    # Pad shorter list so zip works
    n = max(len(mtot_centers), len(mabs_centers))
    while len(mtot_centers) < n:
        mtot_centers.append(None)  # type: ignore[arg-type]
    while len(mabs_centers) < n:
        mabs_centers.append(None)  # type: ignore[arg-type]

    # Assign summaries to sheets
    sheets: dict[str, list[MagneticOutputSummary]] = {}
    ungrouped: list[MagneticOutputSummary] = []
    for s in summaries:
        label = _sheet_label(s, mtot_centers, mabs_centers)
        if label is None:
            ungrouped.append(s)
        else:
            sheets.setdefault(label, []).append(s)

    if ungrouped:
        sheets.setdefault("sheet_unknown", []).extend(ungrouped)
        reasons.append(
            f"{len(ungrouped)} summary(ies) could not be assigned to a sheet "
            "(missing magnetization data)"
        )

    # Sort each sheet by energy ascending (None energies last)
    for sheet_summaries in sheets.values():
        sheet_summaries.sort(
            key=lambda s: s.energy_eV if s.energy_eV is not None else float("inf")
        )

    n_sheets = sum(1 for k in sheets if k != "sheet_unknown")
    if n_sheets == 0 and "sheet_unknown" in sheets:
        verdict = "INSUFFICIENT_DATA"
        cross_valid = False
    elif n_sheets == 1:
        verdict = "SINGLE_SHEET"
        cross_valid = True
        reasons.append("all summaries fall on a single magnetic sheet; cross-sheet ranking is valid")
    else:
        verdict = "MULTI_SHEET"
        cross_valid = False
        reasons.append(
            f"summaries span {n_sheets} magnetic sheets (Mtot clusters: "
            + ", ".join(f"{c:.2f}" if c is not None else "?" for c in mtot_centers[:n_sheets])
            + " μB); cross-sheet energy ranking refused — use per-sheet groups"
        )

    return PerSheetRanking(
        sheets=sheets,
        cross_sheet_ranking_valid=cross_valid,
        verdict=verdict,
        reasons=reasons,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--min-images", type=int, default=2)
    parser.add_argument("--max-bands", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rows = scan_dataset(args.dataset_root, min_images=args.min_images, max_bands=args.max_bands)
    if args.json:
        print(json.dumps([row.to_dict() for row in rows], indent=2, sort_keys=True))
    else:
        print_dataset_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
