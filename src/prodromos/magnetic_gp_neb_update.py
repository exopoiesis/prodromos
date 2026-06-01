"""Update a magnetic GP-NEB driver state from a completed DFT sample.

This closes the active-learning loop around ``magnetic_gp_neb_driver.py``:

1. driver writes job packets;
2. an external workflow runs QE/ABACUS/jDFTx;
3. this tool parses the raw output and appends/replaces the completed sample in
   ``gp_driver_state.json``;
4. the driver is run again to choose the next sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prodromos.cli_contract import dump_json, response_envelope
from prodromos.magnetic_gp_neb_driver import load_driver_state
from prodromos.magnetic_output_parser import parse_output_file

TOOL = "update_magnetic_gp_neb_state"


def load_job_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def completed_sample_from_output(
    *,
    job_spec: dict,
    output_file: str | Path,
    force_norm_eV_A: float | None = None,
) -> dict:
    summary = parse_output_file(output_file)
    sample = {
        "sample_id": job_spec["sample_id"],
        "s": job_spec["s"],
        "status": "complete" if summary.energy_eV is not None else "review",
        "energy_eV": summary.energy_eV,
        "force_norm_eV_A": force_norm_eV_A,
        "total_magnetization_uB": summary.total_magnetization_uB,
        "absolute_magnetization_uB": summary.absolute_magnetization_uB,
        "engine": summary.engine,
        "raw_output_path": str(output_file),
        "source_job_spec": str(job_spec.get("_path", "")),
        "parser_warnings": summary.warnings,
    }
    if job_spec.get("structure_path"):
        sample["structure_path"] = job_spec["structure_path"]
    if job_spec.get("packet_structure_path"):
        sample["structure_path"] = job_spec["packet_structure_path"]
    return sample


def update_driver_state(state: dict, sample: dict, *, replace: bool = True) -> dict:
    """Append or replace a completed sample in a driver state."""
    out = dict(state)
    samples = list(out.get("samples", []))
    if replace:
        samples = [row for row in samples if row.get("sample_id") != sample.get("sample_id")]
    samples.append(sample)
    samples.sort(key=lambda row: float(row.get("s", 0.0)))
    out["samples"] = samples
    return out


def run_magnetic_gp_neb_update(
    *,
    state_json: str | Path,
    job_spec_json: str | Path,
    output_file: str | Path,
    force_norm_eV_A: float | None = None,
    updated_state_json: str | Path | None = None,
    replace: bool = True,
) -> dict:
    state = load_driver_state(state_json)
    job_spec = load_job_spec(job_spec_json)
    job_spec["_path"] = str(job_spec_json)
    sample = completed_sample_from_output(
        job_spec=job_spec,
        output_file=output_file,
        force_norm_eV_A=force_norm_eV_A,
    )
    updated = update_driver_state(state, sample, replace=replace)
    artifacts = []
    if updated_state_json:
        path = Path(updated_state_json)
        path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        artifacts.append(str(path))
    verdict = "SAMPLE_APPENDED" if sample["status"] == "complete" else "REVIEW_SAMPLE_OUTPUT"
    next_actions = ["rerun magnetic_gp_neb_driver.py with the updated state"]
    if sample["status"] != "complete":
        next_actions.insert(0, "check raw output parsing; energy was not found")
    return response_envelope(
        tool=TOOL,
        verdict=verdict,
        confidence="medium" if sample["status"] == "complete" else "review",
        reasons=sample["parser_warnings"],
        next_actions=next_actions,
        artifacts=artifacts,
        result={
            "sample": sample,
            "n_samples": len(updated["samples"]),
            "updated_state": updated if not updated_state_json else None,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--job-spec", required=True, help="job_spec.json from a GP-NEB job packet")
    parser.add_argument("--output-file", required=True, help="raw QE/ABACUS/jDFTx output")
    parser.add_argument("--force-norm", type=float, help="optional force norm in eV/A")
    parser.add_argument("--updated-state", help="write updated driver state JSON here")
    parser.add_argument("--append-duplicate", action="store_true", help="append without replacing same sample_id")
    parser.add_argument("--json", action="store_true", help="Print MCP-shaped JSON")
    parser.add_argument("--output", help="Optional envelope JSON output path")
    args = parser.parse_args(argv)

    envelope = run_magnetic_gp_neb_update(
        state_json=args.state_json,
        job_spec_json=args.job_spec,
        output_file=args.output_file,
        force_norm_eV_A=args.force_norm,
        updated_state_json=args.updated_state,
        replace=not args.append_duplicate,
    )
    if args.output:
        dump_json(envelope, args.output)
    if args.json:
        dump_json(envelope)
    else:
        print(f"verdict {envelope['verdict']}")
        sample = envelope["result"]["sample"]
        print(f"sample  {sample['sample_id']} s={sample['s']} E={sample['energy_eV']}")
        for action in envelope["next_actions"]:
            print(f"next    {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
