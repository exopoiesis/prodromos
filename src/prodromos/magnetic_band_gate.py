"""Band-level magnetic pre-flight gate for NEB image outputs.

This extends the endpoint gate to an existing NEB/string band directory with
``image_XX/espresso.pwo``-style outputs.  It detects magnetic sheet jumps
between neighbouring images and reuses the threshold logic from
``spin_split_detector.py``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re

import numpy as np

from prodromos.magnetic_output_parser import MagneticOutputSummary, parse_output_file
from prodromos.spin_split_detector import (
    DELTA_ABS_ADJ,
    DELTA_TOTAL_ENDPOINT,
    FGEOM_LOW,
    BandDiagnostic,
    magnetic_band_diagnostic,
)


@dataclass
class BandImage:
    label: str
    index: int
    path: str
    summary: MagneticOutputSummary

    def to_dict(self) -> dict:
        data = asdict(self)
        data["summary"] = self.summary.to_dict()
        return data


@dataclass
class BandGateResult:
    verdict: str
    sheet_crossing: bool
    endpoint_split: bool
    crossing_edge: int
    delta_total_adj: list[float] = field(default_factory=list)
    delta_abs_adj: list[float] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    recommendation: str | None = None
    images: list[BandImage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "sheet_crossing": self.sheet_crossing,
            "endpoint_split": self.endpoint_split,
            "crossing_edge": self.crossing_edge,
            "delta_total_adj": self.delta_total_adj,
            "delta_abs_adj": self.delta_abs_adj,
            "roles": self.roles,
            "reasons": self.reasons,
            "recommendation": self.recommendation,
            "images": [image.to_dict() for image in self.images],
        }


def discover_band_outputs(root: str | Path) -> list[Path]:
    """Find one output file per immediate ``image_*`` directory."""
    root = Path(root)
    if root.is_file():
        return [root]

    image_dirs = [path for path in root.iterdir() if path.is_dir() and _image_index(path.name) is not None]
    files: list[Path] = []
    for image_dir in sorted(image_dirs, key=lambda path: _image_index(path.name) or -1):
        output = _choose_image_output(image_dir)
        if output is not None:
            files.append(output)
    return files


def load_band(root: str | Path) -> list[BandImage]:
    images: list[BandImage] = []
    for fallback_index, output in enumerate(discover_band_outputs(root), start=1):
        index = _image_index(output.parent.name) or fallback_index
        images.append(
            BandImage(
                label=output.parent.name if output.parent.name else output.stem,
                index=index,
                path=str(output),
                summary=parse_output_file(output),
            )
        )
    return images


def analyze_band_images(
    images: list[BandImage],
    delta_total_threshold: float = DELTA_TOTAL_ENDPOINT,
    delta_abs_threshold: float = DELTA_ABS_ADJ,
    fgeom_low: float = FGEOM_LOW,
) -> BandGateResult:
    reasons: list[str] = []
    if len(images) < 2:
        return BandGateResult(
            verdict="REVIEW",
            sheet_crossing=False,
            endpoint_split=False,
            crossing_edge=-1,
            reasons=["need at least two image outputs to analyze a band"],
            images=images,
        )

    missing_mag = [
        image.label
        for image in images
        if image.summary.total_magnetization_uB is None or image.summary.absolute_magnetization_uB is None
    ]
    incomplete = [image.label for image in images if image.summary.scf_converged is False]
    if missing_mag:
        reasons.append("missing total/absolute magnetization: " + ", ".join(missing_mag))
    if incomplete:
        reasons.append("non-converged or truncated SCF outputs: " + ", ".join(incomplete))

    if missing_mag:
        return BandGateResult(
            verdict="REVIEW",
            sheet_crossing=False,
            endpoint_split=False,
            crossing_edge=-1,
            reasons=reasons,
            images=images,
        )

    mag_total = np.array([image.summary.total_magnetization_uB for image in images], dtype=float)
    mag_abs = np.array([image.summary.absolute_magnetization_uB for image in images], dtype=float)
    energies = _relative_energies(images)
    geom_fmax = np.zeros(len(images), dtype=float)

    diagnostic = magnetic_band_diagnostic(
        mag_total,
        mag_abs,
        geom_fmax,
        energies,
        delta_abs=delta_abs_threshold,
        delta_total=delta_total_threshold,
        fgeom_low=fgeom_low,
    )
    reasons.extend(diagnostic.flags)

    if diagnostic.sheet_crossing or diagnostic.endpoint_split:
        if incomplete:
            verdict = "REVIEW"
            reasons.append("magnetic discontinuity is visible, but incomplete SCF prevents hard NO-GO")
        else:
            verdict = "NO-GO_SINGLE_SHEET"
            reasons.append("do not trust single-sheet NEB barrier across a magnetic discontinuity")
    elif incomplete:
        verdict = "REVIEW"
    else:
        verdict = "GO"
        reasons.append("no adjacent or endpoint magnetic discontinuity detected")

    return _result_from_diagnostic(verdict, diagnostic, reasons, images)


def _result_from_diagnostic(
    verdict: str,
    diagnostic: BandDiagnostic,
    reasons: list[str],
    images: list[BandImage],
) -> BandGateResult:
    return BandGateResult(
        verdict=verdict,
        sheet_crossing=diagnostic.sheet_crossing,
        endpoint_split=diagnostic.endpoint_split,
        crossing_edge=diagnostic.crossing_edge,
        delta_total_adj=diagnostic.d_total_adj,
        delta_abs_adj=diagnostic.d_abs_adj,
        roles=diagnostic.roles,
        reasons=reasons,
        recommendation=diagnostic.recommendation,
        images=images,
    )


def _relative_energies(images: list[BandImage]) -> np.ndarray:
    values = [image.summary.energy_eV for image in images]
    if any(value is None for value in values):
        return np.zeros(len(images), dtype=float)
    arr = np.array(values, dtype=float)
    return arr - float(np.min(arr))


def _image_index(name: str) -> int | None:
    match = re.search(r"image[_-]?(\d+)", name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _choose_image_output(image_dir: Path) -> Path | None:
    preferred = [
        image_dir / "espresso.pwo",
        image_dir / "running_scf.log",
        image_dir / "jdftx.out",
        image_dir / "out",
    ]
    for path in preferred:
        if path.is_file():
            return path
    candidates = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pwo", ".out"}
        ]
    )
    return candidates[0] if candidates else None


def print_band_table(result: BandGateResult) -> None:
    print(f"verdict\t{result.verdict}")
    print(f"sheet_crossing\t{'yes' if result.sheet_crossing else 'no'}")
    print(f"endpoint_split\t{'yes' if result.endpoint_split else 'no'}")
    print(f"crossing_edge\t{result.crossing_edge if result.crossing_edge >= 0 else '-'}")
    for reason in result.reasons:
        print(f"reason\t{reason}")
    if result.recommendation:
        print(f"recommendation\t{result.recommendation}")
    print("")
    print("image\tconv\tErel(eV)\tMtot\tMabs\trole\tpath")
    rel_e = _relative_energies(result.images) if result.images else np.array([])
    for i, image in enumerate(result.images):
        summary = image.summary
        role = result.roles[i] if i < len(result.roles) else "-"
        print(
            "\t".join(
                [
                    image.label,
                    _fmt_bool(summary.scf_converged),
                    _fmt_float(float(rel_e[i])) if i < len(rel_e) else "-",
                    _fmt_float(summary.total_magnetization_uB),
                    _fmt_float(summary.absolute_magnetization_uB),
                    role,
                    image.path,
                ]
            )
        )


def _fmt_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("band_root", type=Path, help="Directory containing image_XX output subdirectories")
    parser.add_argument("--json", action="store_true", help="Emit normalized JSON")
    parser.add_argument("--delta-total", type=float, default=DELTA_TOTAL_ENDPOINT)
    parser.add_argument("--delta-abs", type=float, default=DELTA_ABS_ADJ)
    args = parser.parse_args(argv)

    result = analyze_band_images(
        load_band(args.band_root),
        delta_total_threshold=args.delta_total,
        delta_abs_threshold=args.delta_abs,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print_band_table(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
