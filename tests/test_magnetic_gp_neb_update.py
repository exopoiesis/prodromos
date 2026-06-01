"""Tests for updating magnetic GP-NEB driver state from DFT output."""
import json
from pathlib import Path

from prodromos.magnetic_gp_neb_update import completed_sample_from_output, run_magnetic_gp_neb_update, update_driver_state


def qe_text():
    return """
     Program PWSCF
     !    total energy              =   -11.00000000 Ry
          total magnetization       =     1.13 Bohr mag/cell
          absolute magnetization    =     2.43 Bohr mag/cell
     convergence has been achieved in 21 iterations
     JOB DONE.
    """


def test_completed_sample_from_qe_output():
    root = Path("tests") / "_tmp_magnetic_gp_neb_update_sample"
    root.mkdir(exist_ok=True)
    output = root / "espresso.pwo"
    try:
        output.write_text(qe_text(), encoding="utf-8")
        sample = completed_sample_from_output(
            job_spec={"sample_id": "gp_s0250", "s": 0.25, "structure_path": "s0250.xyz"},
            output_file=output,
            force_norm_eV_A=0.07,
        )
        assert sample["sample_id"] == "gp_s0250"
        assert sample["status"] == "complete"
        assert sample["force_norm_eV_A"] == 0.07
        assert sample["total_magnetization_uB"] == 1.13
        assert sample["absolute_magnetization_uB"] == 2.43
        assert sample["structure_path"] == "s0250.xyz"
    finally:
        if output.exists():
            output.unlink()
        if root.exists():
            root.rmdir()


def test_update_driver_state_replaces_same_sample_id():
    state = {
        "samples": [
            {"sample_id": "gp_s0250", "s": 0.25, "energy_eV": 1.0},
            {"sample_id": "gp_s0000", "s": 0.0, "energy_eV": 0.0},
        ]
    }
    updated = update_driver_state(state, {"sample_id": "gp_s0250", "s": 0.25, "energy_eV": 2.0})
    assert len(updated["samples"]) == 2
    assert [row["sample_id"] for row in updated["samples"]] == ["gp_s0000", "gp_s0250"]
    assert updated["samples"][1]["energy_eV"] == 2.0


def test_update_cli_writes_updated_state():
    root = Path("tests") / "_tmp_magnetic_gp_neb_update_cli"
    root.mkdir(exist_ok=True)
    state_path = root / "state.json"
    spec_path = root / "job_spec.json"
    output_path = root / "espresso.pwo"
    updated_path = root / "updated_state.json"
    try:
        state_path.write_text(json.dumps({"samples": []}), encoding="utf-8")
        spec_path.write_text(json.dumps({"sample_id": "gp_s0250", "s": 0.25}), encoding="utf-8")
        output_path.write_text(qe_text(), encoding="utf-8")
        envelope = run_magnetic_gp_neb_update(
            state_json=state_path,
            job_spec_json=spec_path,
            output_file=output_path,
            force_norm_eV_A=0.05,
            updated_state_json=updated_path,
        )
        assert envelope["verdict"] == "SAMPLE_APPENDED"
        assert updated_path.exists()
        updated = json.loads(updated_path.read_text(encoding="utf-8"))
        assert updated["samples"][0]["sample_id"] == "gp_s0250"
        assert updated["samples"][0]["force_norm_eV_A"] == 0.05
    finally:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if root.exists():
            root.rmdir()
