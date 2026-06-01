"""Scan a harvested DFT dataset tree for magnetic NEB band risks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from prodromos.magnetic_band_gate import analyze_band_images, load_band


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
