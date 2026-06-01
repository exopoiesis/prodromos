"""N-13 -- saddle proximity gate: verify a located saddle is a CLEAN S_i->S_k
H-transfer, not an off-path / intermediate / mu-bridge structure.

Lifted out of the one-off dimer scripts (infra/gpu_scripts/dimer_pent_136at_qe_VFe.py
report_saddle / proximity-gate) into a reusable, structure-only ($0) tool. Given a
saddle geometry and the two S anchors of the intended hop, it checks:

    * d(H-S_i), d(H-S_k) via the minimum-image convention (MIC).
    * asymmetry  asym = |d(H-S_i) - d(H-S_k)|  -- a clean transfer-state saddle
      sits ~midway between the two anchors (asym < asym_tol).
    * nearest non-H atom to H: for a direct transfer it must be one of the two
      S anchors (anchor_ok).
    * the closest *non-anchor* S (3rd S): if it is within mu_bridge_cutoff the H
      may be forming a mu-S-H-S bridge / off-path intermediate (mu_bridge_warn).

Verdict:
    DIRECT_TRANSFER_OK                    if symmetric_ok AND anchor_ok AND NOT mu_bridge_warn
    OFF_PATH_OR_INTERMEDIATE_INVESTIGATE  otherwise

Reads extxyz with a clean reader that strips non-standard comment keys (e.g.
ABACUS/GPAW 'nspins=1', 'nkpts', 'nbands') which otherwise crash ASE's
SinglePointCalculator. No DFT, no energy needed.
"""
from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import numpy as np

from prodromos.cli_contract import response_envelope, dump_json

DEFAULT_ASYM_TOL = 0.15        # Angstrom
DEFAULT_MU_BRIDGE_CUTOFF = 2.0  # Angstrom


def _clean_read(path: str | Path):
    """Read extxyz stripping non-standard comment keys (e.g. ABACUS/GPAW 'nspins=1',
    'nkpts', 'nbands') that crash ASE's SinglePointCalculator (assert property in
    all_properties -> 'nspins'). Keeps only standard extxyz keys.

    Adapted from infra/gpu_scripts/dimer_pent_136at_qe_VFe.py::_clean_read.
    """
    from ase.io import read
    KEEP = {"Lattice", "Properties", "pbc", "energy", "stress", "free_energy", "charge"}
    lines = Path(path).read_text(errors="replace").splitlines(keepends=True)
    if len(lines) > 1:
        parts = [m.group(0) for m in re.finditer(r'(\w+)=("[^"]*"|\S+)', lines[1])
                 if m.group(1) in KEEP]
        lines[1] = (" ".join(parts) if parts else "") + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as t:
        t.writelines(lines)
        tp = t.name
    try:
        a = read(tp)
    finally:
        Path(tp).unlink()
    return a


def _mic_vec(cell, d):
    """Minimum-image displacement vector under the periodic `cell`."""
    fc = np.linalg.solve(cell.T, d)
    fc -= np.round(fc)
    return cell.T @ fc


def run_saddle_proximity_gate(
    atoms,
    s_i: int,
    s_k: int,
    h_idx: int | None = None,
    asym_tol: float = DEFAULT_ASYM_TOL,
    mu_bridge_cutoff: float = DEFAULT_MU_BRIDGE_CUTOFF,
) -> dict:
    """Core proximity gate. `atoms` is an ASE Atoms (with a cell for MIC).

    `s_i`, `s_k` are 0-based indices of the two S anchors of the intended hop.
    `h_idx` is the transferring H; if None, the single H atom is auto-detected.
    Returns a response_envelope dict. Logic mirrors report_saddle's proximity-gate.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    syms = list(atoms.get_chemical_symbols())
    n = len(atoms)
    cell = np.asarray(atoms.cell.array, dtype=float)

    # Resolve the H atom.
    if h_idx is None:
        h_candidates = [j for j, s in enumerate(syms) if s == "H"]
        if not h_candidates:
            raise ValueError("no H atom found; pass --h-idx explicitly")
        if len(h_candidates) > 1:
            raise ValueError(
                f"{len(h_candidates)} H atoms found ({h_candidates}); pass --h-idx to disambiguate"
            )
        h_idx = h_candidates[0]

    for name, idx in (("--h-idx", h_idx), ("--s-i", s_i), ("--s-k", s_k)):
        if not (0 <= idx < n):
            raise ValueError(f"{name}={idx} out of range [0,{n})")
    if syms[s_i] != "S":
        warnings.append(f"--s-i atom #{s_i} is {syms[s_i]}, not S")
    if syms[s_k] != "S":
        warnings.append(f"--s-k atom #{s_k} is {syms[s_k]}, not S")
    if syms[h_idx] != "H":
        warnings.append(f"--h-idx atom #{h_idx} is {syms[h_idx]}, not H")

    pos = atoms.positions
    d_i = float(np.linalg.norm(_mic_vec(cell, pos[s_i] - pos[h_idx])))
    d_k = float(np.linalg.norm(_mic_vec(cell, pos[s_k] - pos[h_idx])))
    asym = abs(d_i - d_k)

    # nearest non-H atom to H
    ds = sorted((float(np.linalg.norm(_mic_vec(cell, pos[j] - pos[h_idx]))), j)
                for j in range(n) if j != h_idx)
    nn_d, nn_j = ds[0]
    nearest_nonH = f"{syms[nn_j]}{nn_j}"

    # nearest S atoms to H (mu-bridge / off-path detection)
    s_near = sorted((float(np.linalg.norm(_mic_vec(cell, pos[j] - pos[h_idx]))), j)
                    for j in range(n) if syms[j] == "S")
    offpath_S = [(round(d, 3), j) for d, j in s_near if j not in (s_i, s_k)]
    third_S_d = offpath_S[0][0] if offpath_S else 99.0
    mu_bridge_warn = third_S_d < mu_bridge_cutoff

    symmetric_ok = asym <= asym_tol
    anchor_ok = (nn_j in (s_i, s_k)) and (syms[nn_j] == "S")

    if symmetric_ok and anchor_ok and not mu_bridge_warn:
        verdict = "DIRECT_TRANSFER_OK"
        confidence = "high"
        reasons.append(
            f"clean S{s_i}<->S{s_k} transfer saddle: d(H-S{s_i})={d_i:.3f}, "
            f"d(H-S{s_k})={d_k:.3f}, asym={asym:.3f}<={asym_tol} A; nearest non-H = "
            f"{nearest_nonH}@{nn_d:.3f} A is an anchor; no 3rd S within {mu_bridge_cutoff} A."
        )
        next_actions = ["accept the saddle / barrier for this S_i->S_k hop"]
    else:
        verdict = "OFF_PATH_OR_INTERMEDIATE_INVESTIGATE"
        confidence = "high"
        if not symmetric_ok:
            reasons.append(
                f"asymmetric: asym={asym:.3f} > asym_tol={asym_tol} A "
                f"(d(H-S{s_i})={d_i:.3f}, d(H-S{s_k})={d_k:.3f}) -- H not midway between anchors."
            )
        if not anchor_ok:
            reasons.append(
                f"nearest non-H = {nearest_nonH}@{nn_d:.3f} A is NOT an S anchor "
                f"(expected S{s_i} or S{s_k}) -- H has migrated off the intended path."
            )
        if mu_bridge_warn:
            reasons.append(
                f"3rd (non-anchor) S within {mu_bridge_cutoff} A (d={third_S_d:.3f}) -- "
                f"possible mu-S-H-S bridge / intermediate."
            )
        next_actions = [
            "do NOT silently accept the barrier",
            "investigate: re-pick S anchors / re-run dimer/NEB or treat as a distinct intermediate",
        ]

    result = {
        "h_idx": h_idx,
        "s_i": s_i,
        "s_k": s_k,
        "d_HSi": d_i,
        "d_HSk": d_k,
        "asym": asym,
        "asym_tol": asym_tol,
        "nearest_nonH": nearest_nonH,
        "nearest_d": nn_d,
        "third_S_d": third_S_d,
        "mu_bridge_cutoff": mu_bridge_cutoff,
        "mu_bridge_warn": bool(mu_bridge_warn),
        "symmetric_ok": bool(symmetric_ok),
        "anchor_ok": bool(anchor_ok),
        "nearest_S": [[round(d, 3), j] for d, j in s_near[:3]],
    }
    return response_envelope(
        tool="saddle_proximity_gate",
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        next_actions=next_actions,
        warnings=warnings,
        result=result,
    )


def print_gate(env: dict) -> None:
    r = env["result"] or {}
    print(f"verdict\t{env['verdict']}\tconfidence\t{env['confidence']}")
    print(f"d(H-S_i)\t{r.get('d_HSi'):.3f}\td(H-S_k)\t{r.get('d_HSk'):.3f}\tasym\t{r.get('asym'):.3f}")
    print(f"nearest_nonH\t{r.get('nearest_nonH')}@{r.get('nearest_d'):.3f}\t3rd_S\t{r.get('third_S_d')}")
    print(f"symmetric_ok\t{r.get('symmetric_ok')}\tanchor_ok\t{r.get('anchor_ok')}\tmu_bridge_warn\t{r.get('mu_bridge_warn')}")
    for x in env["reasons"]:
        print(f"reason\t{x}")
    for x in env["next_actions"]:
        print(f"next\t{x}")
    for x in env["warnings"]:
        print(f"warning\t{x}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="N-13 saddle proximity gate: verify a clean S_i->S_k H-transfer saddle")
    p.add_argument("--saddle-xyz", type=Path, required=True, help="saddle geometry (extxyz)")
    p.add_argument("--s-i", type=int, required=True, help="0-based index of S anchor i")
    p.add_argument("--s-k", type=int, required=True, help="0-based index of S anchor k")
    p.add_argument("--h-idx", type=int, default=None,
                   help="0-based index of the transferring H (default: auto-detect the lone H)")
    p.add_argument("--asym-tol", type=float, default=DEFAULT_ASYM_TOL,
                   help=f"max |d(H-S_i)-d(H-S_k)| for a symmetric saddle, A (default {DEFAULT_ASYM_TOL})")
    p.add_argument("--mu-bridge-cutoff", type=float, default=DEFAULT_MU_BRIDGE_CUTOFF,
                   help=f"3rd-S distance below which a mu-bridge is flagged, A (default {DEFAULT_MU_BRIDGE_CUTOFF})")
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.saddle_xyz.exists():
        raise SystemExit(f"--saddle-xyz not found: {args.saddle_xyz}")
    atoms = _clean_read(args.saddle_xyz)
    env = run_saddle_proximity_gate(
        atoms, s_i=args.s_i, s_k=args.s_k, h_idx=args.h_idx,
        asym_tol=args.asym_tol, mu_bridge_cutoff=args.mu_bridge_cutoff,
    )
    if args.output:
        dump_json(env, args.output)
    if args.json:
        dump_json(env)
    elif not args.output:
        print_gate(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
