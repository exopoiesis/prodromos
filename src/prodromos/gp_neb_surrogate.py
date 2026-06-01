"""Magnetic-aware GP-NEB surrogate planner.

Fits a 1D Gaussian-process surrogate along reaction coordinate s in [0, 1]
from sampled image energies and suggests where to evaluate next. This is a
pre-flight / active-learning helper, not a replacement for DFT NEB.

Magnetic rule: if a sheet split is visible, the script refuses to train one
single GP across both sheets unless ``--allow-magnetic-split`` is explicitly
used. In Fe-S systems, GP smoothing across spin sheets can invent a false
barrier or a false well.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from prodromos.adaptive_neb_planner import (
    _as_float_array,
    _magnetic_gate_from_summary,
    load_band_json,
    load_band_root_summary,
)
from prodromos.cli_contract import dump_json, response_envelope


def _normalize(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    lo = float(np.nanmin(values[finite]))
    hi = float(np.nanmax(values[finite]))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return np.where(finite, (values - lo) / (hi - lo), 0.0)


def load_gp_input(path: str | Path | None = None, band_root: str | Path | None = None) -> dict:
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        summary = load_band_json(path)
        n = len(summary["energies_eV"])
        s = raw.get("s") or raw.get("reaction_coordinate") or raw.get("arc_length_fraction")
        summary["s"] = s or np.linspace(0.0, 1.0, n).tolist()
        return summary
    if band_root:
        summary = load_band_root_summary(band_root)
        n = len(summary["energies_eV"])
        summary["s"] = np.linspace(0.0, 1.0, n).tolist()
        return summary
    raise ValueError("one of gp_json or band_root is required")


def fit_gp_energy(s: np.ndarray, energies_eV: np.ndarray) -> tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]:
    finite = np.isfinite(s) & np.isfinite(energies_eV)
    if finite.sum() < 3:
        raise ValueError("GP-NEB needs at least 3 finite energy samples")
    x = s[finite, None]
    y = energies_eV[finite]
    # Centering improves numerical conditioning while preserving barriers.
    y0 = float(y.min())
    y_train = y - y0
    kernel = ConstantKernel(1.0, (1e-4, 1e3)) * RBF(length_scale=0.25, length_scale_bounds=(0.03, 2.0)) + WhiteKernel(
        noise_level=1e-6,
        noise_level_bounds=(1e-10, 1e-2),
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=0)
    gp.fit(x, y_train)
    return gp, x, np.array([y0])


def build_gp_neb_plan(
    summary: dict,
    allow_magnetic_split: bool = False,
    grid_size: int = 201,
    top_k: int = 3,
) -> dict:
    s = _as_float_array(summary.get("s"))
    energies = _as_float_array(summary["energies_eV"])
    labels = summary.get("labels") or [f"image_{i:02d}" for i in range(len(energies))]
    mag = _magnetic_gate_from_summary(summary)

    if mag["verdict"] == "NO-GO_SINGLE_SHEET" and not allow_magnetic_split:
        return {
            "labels": labels,
            "magnetic_gate": mag,
            "primary_recommendation": "DO_NOT_FIT_SINGLE_GP_ACROSS_MAGNETIC_SPLIT",
            "next_actions": [
                "split samples by magnetic sheet or run constrained-M pilot",
                "use MECP/two-segment workflow if endpoints prefer different sheets",
            ],
            "gp": None,
        }

    try:
        gp, _x_train, y0 = fit_gp_energy(s, energies)
    except ValueError as exc:
        return {
            "labels": labels,
            "magnetic_gate": mag,
            "primary_recommendation": "REVIEW_GP_INPUT",
            "next_actions": [str(exc)],
            "gp": None,
        }

    grid = np.linspace(0.0, 1.0, grid_size)
    mu_rel, sigma = gp.predict(grid[:, None], return_std=True)
    mu = mu_rel + y0[0]
    endpoint_ref = max(float(mu[0]), float(mu[-1]))
    barrier = mu - endpoint_ref
    barrier_idx = int(np.nanargmax(barrier))

    acq = 0.75 * _normalize(sigma) + 0.25 * _normalize(mu)
    acq[0] = -np.inf
    acq[-1] = -np.inf
    chosen = []
    used = set()
    for idx in np.argsort(acq)[::-1]:
        if not np.isfinite(acq[idx]):
            continue
        # Keep suggestions separated enough to be useful.
        if any(abs(idx - j) < max(2, grid_size // 40) for j in used):
            continue
        used.add(int(idx))
        chosen.append(
            {
                "s": float(grid[idx]),
                "acquisition": float(acq[idx]),
                "predicted_energy_eV": float(mu[idx]),
                "uncertainty_eV": float(sigma[idx]),
            }
        )
        if len(chosen) >= top_k:
            break

    warnings = []
    if mag["verdict"] == "REVIEW":
        warnings.append("magnetization is missing/incomplete; GP-NEB recommendation is not spin-certified")
    if mag["verdict"] == "NO-GO_SINGLE_SHEET" and allow_magnetic_split:
        warnings.append("forced GP fit across magnetic split; use only for visualization, not final barrier")

    return {
        "labels": labels,
        "magnetic_gate": mag,
        "primary_recommendation": "EVALUATE_GP_SUGGESTED_IMAGES",
        "next_actions": [
            "evaluate suggested s-points with the intended DFT/MLIP level",
            "refit GP after adding samples",
            "do not use GP barrier as final Fe-S magnetic barrier",
        ],
        "warnings": warnings,
        "gp": {
            "kernel": str(gp.kernel_),
            "barrier_s": float(grid[barrier_idx]),
            "predicted_barrier_eV": float(barrier[barrier_idx]),
            "barrier_uncertainty_eV": float(sigma[barrier_idx]),
            "suggested_next_samples": chosen,
        },
    }


def run_gp_neb_surrogate(
    *,
    gp_json: str | Path | None = None,
    band_root: str | Path | None = None,
    allow_magnetic_split: bool = False,
    grid_size: int = 201,
    top_k: int = 3,
) -> dict:
    summary = load_gp_input(gp_json, band_root)
    plan = build_gp_neb_plan(
        summary,
        allow_magnetic_split=allow_magnetic_split,
        grid_size=grid_size,
        top_k=top_k,
    )
    return response_envelope(
        tool="plan_gp_neb_surrogate",
        verdict=plan["primary_recommendation"],
        confidence="medium" if plan["magnetic_gate"]["verdict"] == "GO" else "review",
        reasons=[] if plan["magnetic_gate"]["verdict"] == "GO" else ["magnetic gate is not GO"],
        next_actions=plan["next_actions"],
        warnings=plan.get("warnings", []),
        result=plan,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--gp-json", help="JSON with s/reaction_coordinate and energies_eV")
    src.add_argument("--band-root", help="NEB band root with image_XX outputs")
    parser.add_argument("--allow-magnetic-split", action="store_true")
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print MCP-shaped JSON")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)

    envelope = run_gp_neb_surrogate(
        gp_json=args.gp_json,
        band_root=args.band_root,
        allow_magnetic_split=args.allow_magnetic_split,
        grid_size=args.grid_size,
        top_k=args.top_k,
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
