"""Normalize magnetic information from DFT output files.

This is the first pre-flight bridge between raw engine output and the
magnetic gates in ``spin_split_detector.py``.  It intentionally keeps a small
schema: final energy, final total/absolute magnetization, convergence flags,
and final local moments when the engine prints them.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Iterable


RY_TO_EV = 13.605693122994
HA_TO_EV = 27.211386245988

# Tolerances for the "magnetization settled" check. A loose energy conv_thr
# (e.g. QE conv_thr=1e-3 Ry = 13.6 meV) can declare SCF convergence while the
# total/absolute magnetization is still drifting. These tolerances mirror the
# magnetic_endpoint_gate thresholds (|dMtot|>0.3, |dMabs|>0.5 uB): a residual
# per-step drift comparable to those thresholds means the parsed moment cannot
# be trusted to gate-level precision. (s159 pent L4: conv_thr=1e-3 screen.)
MTOT_DRIFT_TOL_UB = 0.1
MABS_DRIFT_TOL_UB = 0.5
MAG_DRIFT_WINDOW = 3


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


@dataclass
class LocalMoment:
    atom_index: int
    moment_uB: float | None
    element: str | None = None
    charge: float | None = None
    radius_bohr: float | None = None
    source: str = "final"


@dataclass
class MagneticOutputSummary:
    engine: str
    path: str
    scf_converged: bool | None = None
    job_done: bool | None = None
    energy_eV: float | None = None
    energy_unit: str | None = None
    total_magnetization_uB: float | None = None
    absolute_magnetization_uB: float | None = None
    total_magnetization_drift_uB: float | None = None
    absolute_magnetization_drift_uB: float | None = None
    magnetization_settled: bool | None = None
    local_moments: list[LocalMoment] = field(default_factory=list)
    nspin: int | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def detect_engine(path: str | Path, text: str) -> str:
    path_s = str(path).lower()
    head = text[:10000]
    if "program pwscf" in head.lower() or "quantum espresso" in head.lower():
        return "qe"
    if path_s.endswith(".pwo"):
        return "qe"
    if "jdftx" in head.lower() or "magneticmoment:" in text.lower():
        return "jdftx"
    if "out.abacus" in path_s or "#scf is converged#" in text.lower() or "final_etot_is" in text.lower():
        return "abacus"
    return "unknown"


def _last_float(pattern: str, text: str, flags: int = re.IGNORECASE) -> float | None:
    matches = re.findall(pattern, text, flags)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[0]
    return float(value)


def _all_floats(pattern: str, text: str, flags: int = re.IGNORECASE) -> list[float]:
    out: list[float] = []
    for value in re.findall(pattern, text, flags):
        if isinstance(value, tuple):
            value = value[0]
        out.append(float(value))
    return out


def _tail_drift(values: list[float], window: int = MAG_DRIFT_WINDOW) -> float | None:
    """Magnitude of change across the last ``window`` printed SCF values.

    Returns ``None`` when there is no iteration history to judge (a single
    printed value cannot reveal whether it is still moving). Otherwise the
    absolute difference between the final value and the value ``window`` steps
    earlier (clamped to the available history). This is the cheap, objective
    replacement for eyeballing whether a moment is still drifting at the loose
    energy convergence threshold.
    """
    if len(values) < 2:
        return None
    w = min(window, len(values) - 1)
    return abs(values[-1] - values[-1 - w])


def _first_int(pattern: str, text: str, flags: int = re.IGNORECASE) -> int | None:
    match = re.search(pattern, text, flags)
    return int(match.group(1)) if match else None


def parse_qe_output(path: str | Path, text: str) -> MagneticOutputSummary:
    warnings: list[str] = []
    energy_ry = _last_float(r"!\s+total energy\s+=\s+(" + _FLOAT + r")\s+Ry", text)
    if energy_ry is None:
        energy_ry = _last_float(r"(?:^|\n)\s*total energy\s+=\s+(" + _FLOAT + r")\s+Ry", text)
        if energy_ry is not None:
            warnings.append("QE energy is from the last SCF iteration, not a final ! total energy")
    total_mag_series = _all_floats(r"total magnetization\s*=\s*(" + _FLOAT + r")\s+Bohr mag/cell", text)
    abs_mag_series = _all_floats(r"absolute magnetization\s*=\s*(" + _FLOAT + r")\s+Bohr mag/cell", text)
    total_mag = total_mag_series[-1] if total_mag_series else None
    abs_mag = abs_mag_series[-1] if abs_mag_series else None
    total_mag_drift = _tail_drift(total_mag_series)
    abs_mag_drift = _tail_drift(abs_mag_series)
    nspin = _first_int(r"\bnspin\s*=\s*(\d+)", text)
    if total_mag is None and abs_mag is None:
        warnings.append("QE output has no final magnetization lines")

    settled: bool | None = None
    if total_mag_drift is not None or abs_mag_drift is not None:
        settled = (total_mag_drift is None or total_mag_drift <= MTOT_DRIFT_TOL_UB) and (
            abs_mag_drift is None or abs_mag_drift <= MABS_DRIFT_TOL_UB
        )
        if not settled:
            warnings.append(
                "QE magnetization not settled at SCF end: "
                f"dMtot={total_mag_drift if total_mag_drift is None else round(total_mag_drift, 3)} uB, "
                f"dMabs={abs_mag_drift if abs_mag_drift is None else round(abs_mag_drift, 3)} uB "
                f"over last {MAG_DRIFT_WINDOW} steps (tighten conv_thr before trusting the moment)"
            )

    return MagneticOutputSummary(
        engine="qe",
        path=str(path),
        scf_converged="convergence has been achieved" in text.lower(),
        job_done="job done" in text.lower(),
        energy_eV=energy_ry * RY_TO_EV if energy_ry is not None else None,
        energy_unit="Ry" if energy_ry is not None else None,
        total_magnetization_uB=total_mag,
        absolute_magnetization_uB=abs_mag,
        total_magnetization_drift_uB=total_mag_drift,
        absolute_magnetization_drift_uB=abs_mag_drift,
        magnetization_settled=settled,
        local_moments=_parse_qe_local_moments(text),
        nspin=nspin,
        warnings=warnings,
    )


def _parse_qe_local_moments(text: str) -> list[LocalMoment]:
    marker = "Magnetic moment per site"
    start = text.rfind(marker)
    if start < 0:
        return []

    atom_re = re.compile(
        r"atom\s+(\d+)\s+\(R=([0-9.]+)\)\s+charge=\s*("
        + _FLOAT
        + r")\s+magn=\s*("
        + _FLOAT
        + r")",
        re.IGNORECASE,
    )
    moments: list[LocalMoment] = []
    seen_atom = False
    for line in text[start:].splitlines():
        match = atom_re.search(line)
        if match:
            seen_atom = True
            moments.append(
                LocalMoment(
                    atom_index=int(match.group(1)),
                    radius_bohr=float(match.group(2)),
                    charge=float(match.group(3)),
                    moment_uB=float(match.group(4)),
                    source="qe_atomic_sphere",
                )
            )
        elif seen_atom and line.strip() and not line.lstrip().startswith("atom"):
            break
    return moments


def parse_abacus_output(path: str | Path, text: str) -> MagneticOutputSummary:
    warnings: list[str] = []
    final_energy = _last_float(r"!FINAL_ETOT_IS\s+(" + _FLOAT + r")\s+eV", text)
    if final_energy is None:
        final_energy = _last_float(r"#TOTAL ENERGY#\s+(" + _FLOAT + r")\s+eV", text)

    nspin = _first_int(r"\bnspin\s*=\s*(\d+)", text)
    total_mag = _last_float(
        r"(?:total\s+magnetization|total_mag|magnetization\s+total)\s*[:=]\s*("
        + _FLOAT
        + r")",
        text,
    )
    abs_mag = _last_float(
        r"(?:absolute\s+magnetization|absolute_mag|magnetization\s+absolute)\s*[:=]\s*("
        + _FLOAT
        + r")",
        text,
    )
    if nspin == 1:
        warnings.append("ABACUS run is nspin=1; no spin-polarized final moments expected")
    elif total_mag is None and abs_mag is None:
        warnings.append("ABACUS output has no recognized final magnetization lines")

    return MagneticOutputSummary(
        engine="abacus",
        path=str(path),
        scf_converged="#scf is converged#" in text.lower(),
        job_done=None,
        energy_eV=final_energy,
        energy_unit="eV" if final_energy is not None else None,
        total_magnetization_uB=total_mag,
        absolute_magnetization_uB=abs_mag,
        local_moments=[],
        nspin=nspin,
        warnings=warnings,
    )


def parse_jdftx_output(path: str | Path, text: str) -> MagneticOutputSummary:
    warnings: list[str] = []
    mag_re = re.compile(
        r"magneticMoment:\s*\[\s*Abs:\s*(" + _FLOAT + r")\s+Tot:\s*(" + _FLOAT + r")\s*\]",
        re.IGNORECASE,
    )
    mag_matches = mag_re.findall(text)
    abs_mag = total_mag = None
    if mag_matches:
        abs_mag = float(mag_matches[-1][0])
        total_mag = float(mag_matches[-1][1])
    else:
        warnings.append("JDFTx output has no recognized magneticMoment lines")

    energy_ha = _last_float(r"\bF:\s*(" + _FLOAT + r")", text)
    spin_type = re.search(r"^\s*spintype\s+(\S+)", text, re.IGNORECASE | re.MULTILINE)
    if spin_type and spin_type.group(1).lower() == "no-spin":
        warnings.append("JDFTx spintype is no-spin; no spin-polarized final moments expected")

    return MagneticOutputSummary(
        engine="jdftx",
        path=str(path),
        scf_converged=_jdftx_converged(text),
        job_done=None,
        energy_eV=energy_ha * HA_TO_EV if energy_ha is not None else None,
        energy_unit="Ha" if energy_ha is not None else None,
        total_magnetization_uB=total_mag,
        absolute_magnetization_uB=abs_mag,
        local_moments=[],
        nspin=None,
        warnings=warnings,
    )


def _jdftx_converged(text: str) -> bool | None:
    lower = text.lower()
    if "converged" in lower:
        return True
    if "failed" in lower or "error" in lower:
        return False
    return None


def parse_output_file(path: str | Path, engine: str = "auto") -> MagneticOutputSummary:
    text = read_text(path)
    detected = detect_engine(path, text) if engine == "auto" else engine.lower()
    if detected == "qe":
        return parse_qe_output(path, text)
    if detected == "abacus":
        return parse_abacus_output(path, text)
    if detected == "jdftx":
        return parse_jdftx_output(path, text)
    return MagneticOutputSummary(
        engine="unknown",
        path=str(path),
        warnings=["Could not detect DFT engine"],
    )


def discover_output_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    suffixes = {".pwo", ".out"}
    names = {"running_scf.log", "abacus.out", "espresso.pwo"}
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.name.lower() in names or candidate.suffix.lower() in suffixes:
            files.append(candidate)
    return sorted(files)


def parse_many(paths: Iterable[str | Path], engine: str = "auto") -> list[MagneticOutputSummary]:
    summaries: list[MagneticOutputSummary] = []
    for path in paths:
        summaries.append(parse_output_file(path, engine=engine))
    return summaries


def _format_value(value: float | int | bool | None, precision: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}g}"


def print_table(summaries: list[MagneticOutputSummary]) -> None:
    header = ["engine", "conv", "nspin", "E(eV)", "Mtot", "Mabs", "local", "path"]
    print("\t".join(header))
    for item in summaries:
        print(
            "\t".join(
                [
                    item.engine,
                    _format_value(item.scf_converged),
                    _format_value(item.nspin),
                    _format_value(item.energy_eV),
                    _format_value(item.total_magnetization_uB),
                    _format_value(item.absolute_magnetization_uB),
                    str(len(item.local_moments)),
                    item.path,
                ]
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Output file or directory to scan")
    parser.add_argument("--engine", choices=["auto", "qe", "abacus", "jdftx"], default="auto")
    parser.add_argument("--json", action="store_true", help="Emit normalized JSON")
    parser.add_argument("--max-files", type=int, default=None, help="Limit directory scans")
    args = parser.parse_args(argv)

    files = discover_output_files(args.path)
    if args.max_files is not None:
        files = files[: args.max_files]
    summaries = parse_many(files, engine=args.engine)

    if args.json:
        print(json.dumps([summary.to_dict() for summary in summaries], indent=2, sort_keys=True))
    else:
        print_table(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
