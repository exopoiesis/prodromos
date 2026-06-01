"""Tests for the magnetic-aware GP-NEB active-learning driver."""
import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read, write

from prodromos.magnetic_gp_neb_driver import (
    build_driver_plan,
    interpolate_endpoint_structure,
    run_magnetic_gp_neb_driver,
)


def base_state(samples):
    return {
        "system_id": "toy_fes",
        "endpoints": {
            "end_a": "endA.xyz",
            "end_b": "endB.xyz",
        },
        "dft_settings": {
            "engine": "qe",
            "nspin": 2,
            "hubbard_u_eV": 2.0,
            "tot_magnetization": 1.13,
        },
        "magnetic_policy": {
            "mode": "constrained-M",
            "constrained_magnetization_uB": 1.13,
        },
        "samples": samples,
    }


def completed_sample(s, energy, mtot=1.13, mabs=2.0, force=0.05):
    return {
        "sample_id": f"s{s:.2f}",
        "s": s,
        "status": "complete",
        "energy_eV": energy,
        "force_norm_eV_A": force,
        "total_magnetization_uB": mtot,
        "absolute_magnetization_uB": mabs,
    }


def test_driver_requests_seed_samples_when_gp_has_too_few_points():
    state = base_state(
        [
            completed_sample(0.0, 0.0),
            completed_sample(1.0, 0.01),
        ]
    )
    plan = build_driver_plan(state, max_new_samples=2)
    assert plan["primary_recommendation"] == "EVALUATE_SEED_SAMPLES"
    assert plan["sample_count"] == 2
    assert len(plan["next_sample_manifest"]) == 1
    job = plan["next_sample_manifest"][0]
    assert job["s"] == 0.25
    assert job["magnetic_policy"]["constrained_magnetization_uB"] == 1.13
    assert "energy_eV" in job["expected_outputs"]


def test_driver_halts_on_magnetic_sheet_split():
    state = base_state(
        [
            completed_sample(0.0, 0.0, mabs=2.0),
            completed_sample(0.25, 0.08, mabs=2.0),
            completed_sample(0.50, 0.20, mabs=2.7),
            completed_sample(1.0, 0.01, mabs=2.7),
        ]
    )
    plan = build_driver_plan(state)
    assert plan["magnetic_gate"]["verdict"] == "NO-GO_SINGLE_SHEET"
    assert plan["primary_recommendation"] == "HALT_MAGNETIC_SPLIT"
    assert plan["next_sample_manifest"] == []
    assert plan["gp"] is None


def test_driver_suggests_gp_samples_for_single_sheet_state():
    state = base_state(
        [
            completed_sample(0.0, 0.0, mabs=2.0),
            completed_sample(0.25, 0.08, mabs=2.02),
            completed_sample(0.50, 0.24, mabs=2.05),
            completed_sample(0.75, 0.12, mabs=2.04),
            completed_sample(1.0, 0.01, mabs=2.01),
        ]
    )
    plan = build_driver_plan(
        state,
        grid_size=101,
        max_new_samples=2,
        barrier_uncertainty_target_eV=1e-6,
    )
    assert plan["magnetic_gate"]["verdict"] == "GO"
    assert plan["primary_recommendation"] == "EVALUATE_GP_SUGGESTED_SAMPLES"
    assert plan["gp"]["predicted_barrier_eV"] > 0
    assert len(plan["next_sample_manifest"]) == 2
    assert all(0.0 < job["s"] < 1.0 for job in plan["next_sample_manifest"])


def test_driver_writes_manifest_when_output_dir_requested():
    root = Path("tests") / "_tmp_magnetic_gp_neb_driver"
    root.mkdir(exist_ok=True)
    state_path = root / "state.json"
    out_dir = root / "driver_out"
    manifest_path = out_dir / "next_sample_manifest.json"
    try:
        if manifest_path.exists():
            manifest_path.unlink()
        state = base_state(
            [
                completed_sample(0.0, 0.0),
                completed_sample(1.0, 0.01),
            ]
        )
        state_path.write_text(json.dumps(state), encoding="utf-8")
        envelope = run_magnetic_gp_neb_driver(
            state_json=state_path,
            output_dir=out_dir,
        )
        assert envelope["tool"] == "run_magnetic_gp_neb_driver"
        assert envelope["verdict"] == "EVALUATE_SEED_SAMPLES"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest[0]["job_type"] == "DFT_SINGLEPOINT_FORCE"
    finally:
        if manifest_path.exists():
            manifest_path.unlink()
        if state_path.exists():
            state_path.unlink()
        if out_dir.exists():
            out_dir.rmdir()
        if root.exists():
            root.rmdir()


def test_interpolate_endpoint_structure_uses_minimum_image():
    root = Path("tests") / "_tmp_magnetic_gp_neb_interpolate"
    root.mkdir(exist_ok=True)
    end_a = root / "endA.xyz"
    end_b = root / "endB.xyz"
    out = root / "s0500.xyz"
    try:
        atoms_a = Atoms("H", positions=[[0.5, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
        atoms_b = Atoms("H", positions=[[9.5, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
        write(end_a, atoms_a, format="extxyz")
        write(end_b, atoms_b, format="extxyz")
        interpolate_endpoint_structure(end_a=end_a, end_b=end_b, s_value=0.5, output_path=out)
        mid = read(out)
        assert mid.positions[0, 0] == pytest.approx(0.0)
    finally:
        for path in (out, end_a, end_b):
            if path.exists():
                path.unlink()
        if root.exists():
            root.rmdir()


def test_driver_can_generate_structures_for_manifest_jobs():
    root = Path("tests") / "_tmp_magnetic_gp_neb_structures"
    root.mkdir(exist_ok=True)
    end_a = root / "endA.xyz"
    end_b = root / "endB.xyz"
    state_path = root / "state.json"
    out_dir = root / "driver_out"
    try:
        write(end_a, Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True), format="extxyz")
        write(end_b, Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True), format="extxyz")
        state = base_state(
            [
                completed_sample(0.0, 0.0),
                completed_sample(1.0, 0.01),
            ]
        )
        state["endpoints"] = {"end_a": str(end_a), "end_b": str(end_b)}
        state_path.write_text(json.dumps(state), encoding="utf-8")
        envelope = run_magnetic_gp_neb_driver(
            state_json=state_path,
            output_dir=out_dir,
            generate_structures=True,
        )
        manifest = envelope["result"]["next_sample_manifest"]
        assert manifest[0]["structure_generation"]["status"] == "generated"
        structure_path = Path(manifest[0]["structure_path"])
        assert structure_path.exists()
        atoms = read(structure_path)
        assert atoms.positions[0, 0] == pytest.approx(0.25)
    finally:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if root.exists():
            root.rmdir()


def test_driver_can_write_engine_neutral_job_packets():
    root = Path("tests") / "_tmp_magnetic_gp_neb_packets"
    root.mkdir(exist_ok=True)
    end_a = root / "endA.xyz"
    end_b = root / "endB.xyz"
    state_path = root / "state.json"
    out_dir = root / "driver_out"
    try:
        write(end_a, Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True), format="extxyz")
        write(end_b, Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True), format="extxyz")
        state = base_state(
            [
                completed_sample(0.0, 0.0),
                completed_sample(1.0, 0.01),
            ]
        )
        state["endpoints"] = {"end_a": str(end_a), "end_b": str(end_b)}
        state_path.write_text(json.dumps(state), encoding="utf-8")
        envelope = run_magnetic_gp_neb_driver(
            state_json=state_path,
            output_dir=out_dir,
            generate_structures=True,
            write_packets=True,
        )
        sample_id = envelope["result"]["next_sample_manifest"][0]["sample_id"]
        packet_dir = out_dir / "job_packets" / sample_id
        assert (packet_dir / "job_spec.json").exists()
        assert (packet_dir / "README_RUN.md").exists()
        assert (packet_dir / "structure.xyz").exists()
        spec = json.loads((packet_dir / "job_spec.json").read_text(encoding="utf-8"))
        assert spec["packet_structure_path"].endswith("structure.xyz")
    finally:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if root.exists():
            root.rmdir()
