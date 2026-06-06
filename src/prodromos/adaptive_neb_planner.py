"""Magnetic-aware planner for NEB algorithm variants.

This is not a DFT optimizer. It converts an existing band summary into
actionable settings for SPM-style pair springs, adaptive/OM-style reparam,
dyNEB-style active images, and OCINEB/CI image selection.

If magnetic sheet discontinuity is detected, the planner refuses to "fix" it
with geometry-only NEB tricks and recommends constrained-M / MECP branching.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prodromos.cli_contract import dump_json, response_envelope
from prodromos.magnetic_band_gate import analyze_band_images, load_band
from prodromos.spin_split_detector import DELTA_ABS_ADJ, DELTA_TOTAL_ENDPOINT, FGEOM_LOW


def _as_float_array(values, default=None) -> np.ndarray | None:
    if values is None:
        if default is None:
            return None
        values = default
    return np.array(values, dtype=float)


def load_band_json(path: str | Path) -> dict:
    """Load a flexible band JSON schema.

    Supported shapes:
    - {"images": [{"label": ..., "energy_eV": ..., "force_norm_eV_A": ...}]}
    - {"energies_eV": [...], "force_norms_eV_A": [...], ...}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "images" in data:
        images = data["images"]
        labels = [img.get("label", f"image_{i:02d}") for i, img in enumerate(images)]
        return {
            "labels": labels,
            "energies_eV": [img.get("energy_eV") for img in images],
            "force_norms_eV_A": [img.get("force_norm_eV_A", img.get("force_eV_A", 0.0)) for img in images],
            "distances_A": [img.get("distance_A", 1.0) for img in images[:-1]],
            "total_magnetization_uB": [img.get("total_magnetization_uB") for img in images],
            "absolute_magnetization_uB": [img.get("absolute_magnetization_uB") for img in images],
        }
    n = len(data.get("energies_eV", []))
    return {
        "labels": data.get("labels") or [f"image_{i:02d}" for i in range(n)],
        "energies_eV": data.get("energies_eV"),
        "force_norms_eV_A": data.get("force_norms_eV_A") or data.get("forces_eV_A"),
        "distances_A": data.get("distances_A"),
        "total_magnetization_uB": data.get("total_magnetization_uB"),
        "absolute_magnetization_uB": data.get("absolute_magnetization_uB"),
    }


def load_band_root_summary(root: str | Path) -> dict:
    """Load energies/magnetism from image output files under a NEB band root."""
    images = load_band(root)
    return {
        "labels": [img.label for img in images],
        "energies_eV": [img.summary.energy_eV for img in images],
        "force_norms_eV_A": [0.0 for _ in images],
        "distances_A": [1.0 for _ in images[:-1]],
        "total_magnetization_uB": [img.summary.total_magnetization_uB for img in images],
        "absolute_magnetization_uB": [img.summary.absolute_magnetization_uB for img in images],
        "_band_images": images,
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    lo = float(np.nanmin(values[finite]))
    hi = float(np.nanmax(values[finite]))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return np.where(finite, (values - lo) / (hi - lo), 0.0)


def spm_pair_springs(
    energies_eV: np.ndarray,
    distances_A: np.ndarray | None = None,
    k_min: float = 0.3,
    k_max: float = 3.0,
) -> list[dict]:
    """SPM-style local spring pair schedule for adjacent image edges.

    High-energy and overstretched edges get stronger local pair springs.
    """
    n_edges = len(energies_eV) - 1
    if distances_A is None:
        distances_A = np.ones(n_edges)
    edge_energy = np.maximum(energies_eV[:-1], energies_eV[1:])
    energy_score = _normalize(edge_energy)
    spacing_score = _normalize(distances_A)
    score = 0.7 * energy_score + 0.3 * spacing_score
    springs = k_min * (k_max / k_min) ** score
    return [
        {
            "edge": i,
            "images": [i, i + 1],
            "k_spring_eV_A2": float(springs[i]),
            "energy_score": float(energy_score[i]),
            "spacing_score": float(spacing_score[i]),
        }
        for i in range(n_edges)
    ]


def adaptive_reparam_monitor(
    energies_eV: np.ndarray,
    distances_A: np.ndarray | None = None,
    beta_energy: float = 2.0,
) -> list[dict]:
    """OM/adaptive-springs style monitor weights for redistributing images."""
    n_edges = len(energies_eV) - 1
    if distances_A is None:
        distances_A = np.ones(n_edges)
    edge_energy = np.maximum(energies_eV[:-1], energies_eV[1:])
    energy_score = _normalize(edge_energy)
    spacing_score = _normalize(distances_A)
    weights = 1.0 + beta_energy * energy_score + spacing_score
    weights = weights / weights.sum()
    target_cum = np.concatenate([[0.0], np.cumsum(weights)])
    return [
        {
            "edge": i,
            "images": [i, i + 1],
            "monitor_weight": float(weights[i]),
            "target_s_left": float(target_cum[i]),
            "target_s_right": float(target_cum[i + 1]),
        }
        for i in range(n_edges)
    ]


def dyneb_active_images(
    energies_eV: np.ndarray,
    force_norms_eV_A: np.ndarray,
    fmax: float = 0.05,
    scale_fmax: float = 1.0,
) -> list[dict]:
    """dyNEB-style active image schedule.

    Images far from the current highest-energy image get a looser local fmax.
    """
    n = len(energies_eV)
    if n == 0:
        return []
    climber = int(np.nanargmax(energies_eV))
    denom = max(n - 1, 1)
    rows = []
    for i in range(n):
        if i in (0, n - 1):
            active = False
            threshold = 0.0
        else:
            distance_from_climber = abs(i - climber) / denom
            threshold = fmax * (1.0 + scale_fmax * distance_from_climber)
            active = bool(force_norms_eV_A[i] > threshold)
        rows.append(
            {
                "image": i,
                "force_norm_eV_A": float(force_norms_eV_A[i]),
                "local_fmax_eV_A": float(threshold),
                "active": active,
            }
        )
    return rows


def ocineb_climber_plan(energies_eV: np.ndarray, force_norms_eV_A: np.ndarray) -> dict:
    """OCINEB/CI-style climbing image selection."""
    if len(energies_eV) < 3:
        return {
            "climber_image": None,
            "start_climbing": False,
            "reason": "need at least one interior image",
        }
    interior = energies_eV[1:-1]
    climber = int(np.nanargmax(interior) + 1)
    neighbor_margin = float(energies_eV[climber] - max(energies_eV[climber - 1], energies_eV[climber + 1]))
    force = float(force_norms_eV_A[climber])
    start = bool(neighbor_margin >= 0.0 and force < 0.5)
    return {
        "climber_image": climber,
        "start_climbing": start,
        "neighbor_energy_margin_eV": neighbor_margin,
        "climber_force_norm_eV_A": force,
        "reason": "highest interior energy image selected; delay CI if force is still very large",
    }


def _magnetic_gate_from_summary(summary: dict) -> dict:
    mtot = summary.get("total_magnetization_uB")
    mabs = summary.get("absolute_magnetization_uB")
    if mtot is None or mabs is None or any(v is None for v in mtot) or any(v is None for v in mabs):
        return {
            "verdict": "REVIEW",
            "reason": "missing per-image magnetization; geometry-only algorithm choice is uncertified",
        }
    images = summary.get("_band_images")
    if images is not None:
        result = analyze_band_images(images).to_dict()
        return {"verdict": result["verdict"], "result": result}
    energies = _as_float_array(summary["energies_eV"])
    force_norms = _as_float_array(summary.get("force_norms_eV_A"), default=np.zeros(len(energies)))
    from spin_split_detector import magnetic_band_diagnostic

    diag = magnetic_band_diagnostic(
        mtot,
        mabs,
        force_norms,
        energies,
        delta_abs=DELTA_ABS_ADJ,
        delta_total=DELTA_TOTAL_ENDPOINT,
        fgeom_low=FGEOM_LOW,
    )
    verdict = "NO-GO_SINGLE_SHEET" if diag.sheet_crossing or diag.endpoint_split else "GO"
    return {
        "verdict": verdict,
        "result": {
            "roles": diag.roles,
            "sheet_crossing": diag.sheet_crossing,
            "endpoint_split": diag.endpoint_split,
            "crossing_edge": diag.crossing_edge,
            "flags": diag.flags,
            "recommendation": diag.recommendation,
        },
    }


def build_algorithm_plan(
    summary: dict,
    k_min: float = 0.3,
    k_max: float = 3.0,
    fmax: float = 0.05,
    scale_fmax: float = 1.0,
) -> dict:
    energies = _as_float_array(summary["energies_eV"])
    force_norms = _as_float_array(summary.get("force_norms_eV_A"), default=np.zeros(len(energies)))
    distances = _as_float_array(summary.get("distances_A"), default=np.ones(max(len(energies) - 1, 0)))
    labels = summary.get("labels") or [f"image_{i:02d}" for i in range(len(energies))]

    mag = _magnetic_gate_from_summary(summary)
    if mag["verdict"] == "NO-GO_SINGLE_SHEET":
        return {
            "labels": labels,
            "magnetic_gate": mag,
            "primary_recommendation": "DO_NOT_USE_GEOMETRY_ONLY_NEB_VARIANTS",
            "next_actions": [
                "run endpoint matrix / constrained-M pilot",
                "branch to constrained-M NEB or MECP/two-segment workflow",
            ],
            "spm_pair_springs": [],
            "adaptive_monitor": [],
            "dyneb_active_images": [],
            "ocineb_climber": None,
        }

    plan = {
        "labels": labels,
        "magnetic_gate": mag,
        "primary_recommendation": "ADAPTIVE_SPRINGS_PLUS_DYNEB_OCINEB",
        "next_actions": [
            "use SPM/adaptive spring profile to avoid ridge roll-off",
            "relax active dyNEB images first",
            "enable climbing only after highest-energy image force is moderate",
        ],
        "spm_pair_springs": spm_pair_springs(energies, distances, k_min=k_min, k_max=k_max),
        "adaptive_monitor": adaptive_reparam_monitor(energies, distances),
        "dyneb_active_images": dyneb_active_images(energies, force_norms, fmax=fmax, scale_fmax=scale_fmax),
        "ocineb_climber": ocineb_climber_plan(energies, force_norms),
    }
    if mag["verdict"] == "REVIEW":
        plan["next_actions"].insert(0, "complete magnetic outputs before trusting final barrier")
    return plan


def run_adaptive_neb_planner(
    *,
    band_json: str | Path | None = None,
    band_root: str | Path | None = None,
    k_min: float = 0.3,
    k_max: float = 3.0,
    fmax: float = 0.05,
    scale_fmax: float = 1.0,
) -> dict:
    if band_json:
        summary = load_band_json(band_json)
    elif band_root:
        summary = load_band_root_summary(band_root)
    else:
        raise ValueError("one of band_json or band_root is required")
    plan = build_algorithm_plan(summary, k_min=k_min, k_max=k_max, fmax=fmax, scale_fmax=scale_fmax)
    return response_envelope(
        tool="plan_adaptive_neb_algorithms",
        verdict=plan["primary_recommendation"],
        confidence="medium" if plan["magnetic_gate"]["verdict"] != "REVIEW" else "review",
        reasons=[] if plan["magnetic_gate"]["verdict"] == "GO" else [plan["magnetic_gate"].get("reason", "magnetic gate is not GO")],
        next_actions=plan["next_actions"],
        result=plan,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--band-json", help="Band summary JSON")
    src.add_argument("--band-root", help="NEB band root with image_XX outputs")
    parser.add_argument("--k-min", type=float, default=0.3)
    parser.add_argument("--k-max", type=float, default=3.0)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--scale-fmax", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="Print MCP-shaped JSON")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)

    envelope = run_adaptive_neb_planner(
        band_json=args.band_json,
        band_root=args.band_root,
        k_min=args.k_min,
        k_max=args.k_max,
        fmax=args.fmax,
        scale_fmax=args.scale_fmax,
    )
    if args.output:
        dump_json(envelope, args.output)
    if args.json:
        dump_json(envelope)
    else:
        print(f"verdict {envelope['verdict']}")
        for action in envelope["next_actions"]:
            print(f"next    {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
