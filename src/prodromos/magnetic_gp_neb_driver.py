"""Magnetic-aware GP-NEB active-learning driver.

This is the Level-3 version of the GP-NEB idea for this project: a stateful
planner around expensive DFT force/energy evaluations. It does not launch QE,
ABACUS, or jDFTx. Instead it consumes a JSON state of completed samples, applies
the magnetic gate, fits the current GP surrogate when safe, and writes a
``next_sample_manifest`` describing which DFT evaluations should be run next.

Core rule: never fit one GP across a magnetic sheet split unless explicitly
forced for debugging. In Fe-S systems that would smooth over the physics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from prodromos.adaptive_neb_planner import _magnetic_gate_from_summary
from prodromos.cli_contract import dump_json, response_envelope
from prodromos.gp_neb_surrogate import build_gp_neb_plan

TOOL = "run_magnetic_gp_neb_driver"
DEFAULT_SEED_S = [0.25, 0.50, 0.75]


def load_driver_state(path: str | Path) -> dict:
    """Load a magnetic GP-NEB driver state JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _completed_samples(state: dict) -> list[dict]:
    return [
        sample
        for sample in state.get("samples", [])
        if sample.get("status", "complete") == "complete" and sample.get("energy_eV") is not None
    ]


def _pending_or_complete_s_values(state: dict) -> set[float]:
    values = set()
    for sample in state.get("samples", []):
        try:
            values.add(round(float(sample["s"]), 8))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def state_to_band_summary(state: dict) -> dict:
    """Convert completed driver samples into the summary expected by GP tools."""
    samples = sorted(_completed_samples(state), key=lambda row: float(row["s"]))
    labels = [sample.get("label", sample.get("sample_id", f"s_{float(sample['s']):.3f}")) for sample in samples]
    energies = [sample.get("energy_eV") for sample in samples]
    forces = [sample.get("force_norm_eV_A", 0.0) for sample in samples]
    mtot = [sample.get("total_magnetization_uB") for sample in samples]
    mabs = [sample.get("absolute_magnetization_uB") for sample in samples]
    s = [sample.get("s") for sample in samples]
    distances = np.diff(np.asarray(s, dtype=float)).tolist() if len(s) > 1 else []
    return {
        "labels": labels,
        "s": s,
        "energies_eV": energies,
        "force_norms_eV_A": forces,
        "distances_A": distances,
        "total_magnetization_uB": mtot,
        "absolute_magnetization_uB": mabs,
    }


def _finite_sample_count(summary: dict) -> int:
    energies = np.asarray(summary.get("energies_eV", []), dtype=float)
    s = np.asarray(summary.get("s", []), dtype=float)
    if len(energies) != len(s):
        return 0
    return int(np.isfinite(energies).sum())


def _sample_id(s_value: float) -> str:
    return f"gp_s{int(round(1000.0 * s_value)):04d}"


def _magnetic_policy(state: dict) -> dict:
    policy = dict(state.get("magnetic_policy") or {})
    dft = dict(state.get("dft_settings") or {})
    if "constrained_magnetization_uB" not in policy and "tot_magnetization" in dft:
        policy["constrained_magnetization_uB"] = dft.get("tot_magnetization")
    return policy


def interpolate_endpoint_structure(
    *,
    end_a: str | Path,
    end_b: str | Path,
    s_value: float,
    output_path: str | Path,
    mic: bool = True,
) -> Path:
    """Generate a structure at reaction-coordinate fraction ``s_value``.

    The interpolation preserves atom ordering and symbols from endpoint A. If
    periodic cell information is present and ``mic`` is true, endpoint B is first
    wrapped by minimum-image displacements relative to endpoint A. This mirrors
    the IDPP-prewrap lesson: never interpolate a PBC hop through the long cell
    direction when the short image exists.
    """
    from ase.geometry import find_mic
    from ase.io import read, write

    atoms_a = read(str(end_a))
    atoms_b = read(str(end_b))
    if len(atoms_a) != len(atoms_b):
        raise ValueError("endpoint structures must contain the same number of atoms")
    if atoms_a.get_chemical_symbols() != atoms_b.get_chemical_symbols():
        raise ValueError("endpoint structures must have identical atom ordering and symbols")

    s = float(s_value)
    if not 0.0 <= s <= 1.0:
        raise ValueError("s_value must be in [0, 1]")

    delta = atoms_b.get_positions() - atoms_a.get_positions()
    if mic and atoms_a.cell.volume > 1e-12 and any(atoms_a.pbc):
        delta, _lengths = find_mic(delta, atoms_a.cell, atoms_a.pbc)

    atoms = atoms_a.copy()
    atoms.set_positions(atoms_a.get_positions() + s * delta)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, atoms, format="extxyz")
    return out


def attach_generated_structures(
    manifest: list[dict],
    *,
    output_dir: str | Path,
    mic: bool = True,
) -> list[str]:
    """Generate XYZ files for manifest jobs that have endpoint paths."""
    structure_dir = Path(output_dir)
    structure_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for job in manifest:
        generation = job.get("structure_generation") or {}
        end_a = generation.get("end_a")
        end_b = generation.get("end_b")
        if not end_a or not end_b:
            generation["status"] = "skipped_missing_endpoints"
            job["structure_generation"] = generation
            continue
        structure_path = structure_dir / f"{job['sample_id']}.xyz"
        try:
            path = interpolate_endpoint_structure(
                end_a=end_a,
                end_b=end_b,
                s_value=job["s"],
                output_path=structure_path,
                mic=mic,
            )
        except Exception as exc:  # manifest should still be useful for debugging
            generation["status"] = "failed"
            generation["error"] = str(exc)
            job["structure_generation"] = generation
            continue
        generation["status"] = "generated"
        generation["method"] = "linear_mic_endpoint_interpolation" if mic else "linear_endpoint_interpolation"
        job["structure_generation"] = generation
        job["structure_path"] = str(path)
        artifacts.append(str(path))
    return artifacts


def write_job_packets(manifest: list[dict], *, output_dir: str | Path) -> list[str]:
    """Write one engine-neutral job packet directory per manifest job.

    A packet is intentionally not a fake QE/ABACUS/jDFTx input file. It contains
    the generated structure, the full job spec, and a README describing the
    expected external wrapper action. Engine-specific launchers can consume this
    without losing provenance.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for job in manifest:
        job_dir = root / job["sample_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        packet = dict(job)
        structure_path = job.get("structure_path")
        if structure_path and Path(structure_path).exists():
            dest = job_dir / "structure.xyz"
            shutil.copyfile(structure_path, dest)
            packet["packet_structure_path"] = str(dest)
            artifacts.append(str(dest))
        spec_path = job_dir / "job_spec.json"
        spec_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        readme_path = job_dir / "README_RUN.md"
        engine = (job.get("dft_settings") or {}).get("engine", "unknown")
        readme_path.write_text(_job_packet_readme(job, engine), encoding="utf-8")
        artifacts.extend([str(spec_path), str(readme_path)])
    return artifacts


def _job_packet_readme(job: dict, engine: str) -> str:
    magnetic = job.get("magnetic_policy") or {}
    settings = job.get("dft_settings") or {}
    lines = [
        f"# GP-NEB sample {job['sample_id']}",
        "",
        f"- s: {job['s']}",
        f"- purpose: {job.get('purpose')}",
        f"- engine: {engine}",
        f"- magnetic mode: {magnetic.get('mode', 'unspecified')}",
        f"- constrained M: {magnetic.get('constrained_magnetization_uB', 'unspecified')} uB",
        "",
        "## Expected external wrapper steps",
        "",
        "1. Convert `structure.xyz` to the engine-specific input format.",
        "2. Preserve the DFT settings and magnetic policy from `job_spec.json`.",
        "3. Run a singlepoint+force calculation.",
        "4. Parse energy, force norm, total magnetization, and absolute magnetization.",
        "5. Append the completed sample to `gp_driver_state.json` and rerun `magnetic_gp_neb_driver.py`.",
        "",
        "## DFT settings snapshot",
        "",
        "```json",
        json.dumps(settings, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def build_sample_manifest(
    *,
    state: dict,
    suggested_samples: list[dict],
    purpose: str,
    max_new_samples: int,
) -> list[dict]:
    """Build DFT job specs for the next active-learning batch."""
    dft_settings = dict(state.get("dft_settings") or {})
    endpoints = dict(state.get("endpoints") or {})
    magnetic_policy = _magnetic_policy(state)
    existing = _pending_or_complete_s_values(state)
    jobs = []
    for row in suggested_samples:
        s_value = float(row["s"])
        if round(s_value, 8) in existing:
            continue
        sample_id = row.get("sample_id") or _sample_id(s_value)
        jobs.append(
            {
                "sample_id": sample_id,
                "s": s_value,
                "job_type": "DFT_SINGLEPOINT_FORCE",
                "purpose": purpose,
                "structure_generation": {
                    "method": "linear_mic_endpoint_interpolation",
                    "end_a": endpoints.get("end_a"),
                    "end_b": endpoints.get("end_b"),
                    "status": "planned",
                    "note": "use --generate-structures to create this XYZ automatically from endpoints",
                },
                "dft_settings": dft_settings,
                "magnetic_policy": magnetic_policy,
                "selection_metrics": {
                    key: row[key]
                    for key in ("acquisition", "predicted_energy_eV", "uncertainty_eV")
                    if key in row
                },
                "expected_outputs": [
                    "energy_eV",
                    "force_norm_eV_A",
                    "total_magnetization_uB",
                    "absolute_magnetization_uB",
                    "raw_output_path",
                ],
            }
        )
        if len(jobs) >= max_new_samples:
            break
    return jobs


def _seed_suggestions(state: dict, needed: int) -> list[dict]:
    existing = _pending_or_complete_s_values(state)
    candidates = []
    for s_value in DEFAULT_SEED_S:
        if round(float(s_value), 8) not in existing:
            candidates.append({"s": float(s_value), "reason": "seed GP with interior sample"})
        if len(candidates) >= needed:
            break
    return candidates


def _driver_warnings(state: dict, mag: dict) -> list[str]:
    warnings = []
    if mag["verdict"] == "REVIEW":
        warnings.append("magnetization is missing or incomplete; GP suggestions are not spin-certified")
    if not state.get("endpoints"):
        warnings.append("endpoint structure paths are missing; manifest will require external structure generation")
    if not state.get("dft_settings"):
        warnings.append("dft_settings are missing; manifest jobs will not be directly launchable")
    return warnings


def build_driver_plan(
    state: dict,
    *,
    allow_magnetic_split: bool = False,
    grid_size: int = 201,
    max_new_samples: int = 2,
    barrier_uncertainty_target_eV: float = 0.03,
) -> dict:
    """Return the next active-learning action for a magnetic GP-NEB state."""
    summary = state_to_band_summary(state)
    sample_count = _finite_sample_count(summary)
    labels = summary.get("labels", [])

    if sample_count == 0:
        mag = {
            "verdict": "REVIEW",
            "reason": "no completed samples with energy/magnetization yet",
        }
    else:
        mag = _magnetic_gate_from_summary(summary)

    if mag["verdict"] == "NO-GO_SINGLE_SHEET" and not allow_magnetic_split:
        return {
            "primary_recommendation": "HALT_MAGNETIC_SPLIT",
            "confidence": "high",
            "labels": labels,
            "magnetic_gate": mag,
            "sample_count": sample_count,
            "next_sample_manifest": [],
            "gp": None,
            "next_actions": [
                "do not fit one GP across the current magnetic sheet split",
                "run endpoint matrix or constrained-M seam pilot",
                "restart GP driver only after samples are on one magnetic sheet",
            ],
            "warnings": _driver_warnings(state, mag),
        }

    if sample_count < 3:
        needed = min(max_new_samples, 3 - sample_count)
        seeds = _seed_suggestions(state, needed=needed)
        manifest = build_sample_manifest(
            state=state,
            suggested_samples=seeds,
            purpose="seed_gp_neb_surrogate",
            max_new_samples=max_new_samples,
        )
        return {
            "primary_recommendation": "EVALUATE_SEED_SAMPLES",
            "confidence": "medium" if mag["verdict"] == "GO" else "review",
            "labels": labels,
            "magnetic_gate": mag,
            "sample_count": sample_count,
            "next_sample_manifest": manifest,
            "gp": None,
            "next_actions": [
                "evaluate seed DFT singlepoint+force samples",
                "parse energy, force norm, total and absolute magnetization",
                "append completed samples to the driver state and rerun",
            ],
            "warnings": _driver_warnings(state, mag),
        }

    gp_plan = build_gp_neb_plan(
        summary,
        allow_magnetic_split=allow_magnetic_split,
        grid_size=grid_size,
        top_k=max_new_samples,
    )
    if gp_plan["gp"] is None:
        return {
            "primary_recommendation": gp_plan["primary_recommendation"],
            "confidence": "review",
            "labels": labels,
            "magnetic_gate": gp_plan["magnetic_gate"],
            "sample_count": sample_count,
            "next_sample_manifest": [],
            "gp": None,
            "next_actions": gp_plan["next_actions"],
            "warnings": _driver_warnings(state, gp_plan["magnetic_gate"]) + gp_plan.get("warnings", []),
        }

    gp = gp_plan["gp"]
    if gp["barrier_uncertainty_eV"] <= barrier_uncertainty_target_eV:
        recommendation = "READY_FOR_CONFIRMATORY_NEB_OR_DFT_CHECK"
        next_actions = [
            "run confirmatory DFT NEB or targeted saddle singlepoint at predicted barrier",
            "do not publish GP surrogate barrier without DFT confirmation",
        ]
        manifest = []
    else:
        recommendation = "EVALUATE_GP_SUGGESTED_SAMPLES"
        next_actions = [
            "evaluate the suggested DFT singlepoint+force samples",
            "append completed samples to the driver state and rerun",
            "keep magnetic constraints identical across all samples",
        ]
        manifest = build_sample_manifest(
            state=state,
            suggested_samples=gp["suggested_next_samples"],
            purpose="reduce_gp_barrier_uncertainty",
            max_new_samples=max_new_samples,
        )

    return {
        "primary_recommendation": recommendation,
        "confidence": "medium" if gp_plan["magnetic_gate"]["verdict"] == "GO" else "review",
        "labels": labels,
        "magnetic_gate": gp_plan["magnetic_gate"],
        "sample_count": sample_count,
        "next_sample_manifest": manifest,
        "gp": gp,
        "next_actions": next_actions,
        "warnings": _driver_warnings(state, gp_plan["magnetic_gate"]) + gp_plan.get("warnings", []),
    }


def run_magnetic_gp_neb_driver(
    *,
    state_json: str | Path,
    allow_magnetic_split: bool = False,
    grid_size: int = 201,
    max_new_samples: int = 2,
    barrier_uncertainty_target_eV: float = 0.03,
    output_dir: str | Path | None = None,
    generate_structures: bool = False,
    structure_output_dir: str | Path | None = None,
    write_packets: bool = False,
    job_packet_dir: str | Path | None = None,
    mic: bool = True,
) -> dict:
    """Run the driver and optionally write its next-sample manifest."""
    state = load_driver_state(state_json)
    plan = build_driver_plan(
        state,
        allow_magnetic_split=allow_magnetic_split,
        grid_size=grid_size,
        max_new_samples=max_new_samples,
        barrier_uncertainty_target_eV=barrier_uncertainty_target_eV,
    )
    artifacts = []
    if generate_structures:
        if structure_output_dir:
            struct_dir = Path(structure_output_dir)
        elif output_dir:
            struct_dir = Path(output_dir) / "structures"
        else:
            struct_dir = Path("gp_neb_structures")
        artifacts.extend(
            attach_generated_structures(
                plan["next_sample_manifest"],
                output_dir=struct_dir,
                mic=mic,
            )
        )
    if write_packets:
        if job_packet_dir:
            packet_dir = Path(job_packet_dir)
        elif output_dir:
            packet_dir = Path(output_dir) / "job_packets"
        else:
            packet_dir = Path("gp_neb_job_packets")
        artifacts.extend(write_job_packets(plan["next_sample_manifest"], output_dir=packet_dir))
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / "next_sample_manifest.json"
        manifest_path.write_text(
            json.dumps(plan["next_sample_manifest"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        artifacts.append(str(manifest_path))
    return response_envelope(
        tool=TOOL,
        verdict=plan["primary_recommendation"],
        confidence=plan["confidence"],
        reasons=[] if plan["magnetic_gate"]["verdict"] == "GO" else ["magnetic gate is not GO"],
        next_actions=plan["next_actions"],
        warnings=plan["warnings"],
        artifacts=artifacts,
        result=plan,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", required=True, help="Magnetic GP-NEB driver state JSON")
    parser.add_argument("--allow-magnetic-split", action="store_true", help="Debug only; do not use for final Fe-S barriers")
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--max-new-samples", type=int, default=2)
    parser.add_argument("--barrier-uncertainty-target", type=float, default=0.03)
    parser.add_argument("--output-dir", help="Optional directory for next_sample_manifest.json")
    parser.add_argument("--generate-structures", action="store_true", help="Generate XYZ structures for suggested samples")
    parser.add_argument("--structure-output-dir", help="Optional structure directory; defaults to output-dir/structures")
    parser.add_argument("--write-job-packets", action="store_true", help="Write per-sample engine-neutral job packet directories")
    parser.add_argument("--job-packet-dir", help="Optional packet directory; defaults to output-dir/job_packets")
    parser.add_argument("--no-mic", action="store_true", help="Disable minimum-image endpoint interpolation")
    parser.add_argument("--json", action="store_true", help="Print MCP-shaped JSON")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)

    envelope = run_magnetic_gp_neb_driver(
        state_json=args.state_json,
        allow_magnetic_split=args.allow_magnetic_split,
        grid_size=args.grid_size,
        max_new_samples=args.max_new_samples,
        barrier_uncertainty_target_eV=args.barrier_uncertainty_target,
        output_dir=args.output_dir,
        generate_structures=args.generate_structures,
        structure_output_dir=args.structure_output_dir,
        write_packets=args.write_job_packets,
        job_packet_dir=args.job_packet_dir,
        mic=not args.no_mic,
    )
    if args.output:
        dump_json(envelope, args.output)
    if args.json:
        dump_json(envelope)
    else:
        print(f"verdict {envelope['verdict']}")
        for action in envelope["next_actions"]:
            print(f"next    {action}")
        manifest = envelope["result"].get("next_sample_manifest", [])
        if manifest:
            print("samples")
            for job in manifest:
                print(f"  {job['sample_id']} s={job['s']:.3f} purpose={job['purpose']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
