"""
Generalized V_Fe symmetry pre-flight test.

For ANY mineral with (pristine.xyz, endA.xyz, canonical_triple.json):
1. spglib symmetry of pristine
2. Find R such that R(S_i) = S_k AND R(V_Fe) = V_Fe (if V_Fe at high-sym site)
3. Apply R to endA + Hungarian relabel
4. Measure non-H displacement statistics
5. Verdict:
   - max_disp_non_H < 0.1 Å → symmetric MEP exists, NEB should work
   - 0.1-1 Å → marginal, may work with care
   - > 1 Å → asymmetric pocket, standard symmetric NEB inapplicable

Validates the methodology on known cases (mack/greig) + predicts new (pyr).
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import spglib
from ase.io import read
from scipy.optimize import linear_sum_assignment

from prodromos.cli_contract import dump_json, response_envelope


def apply_R(atoms, R, t):
    cell = np.array(atoms.get_cell())
    pos_frac = atoms.get_scaled_positions()
    new_frac = (R @ pos_frac.T).T + t
    new_frac -= np.floor(new_frac)
    return new_frac @ cell


def hungarian_match(pos_target, pos_source, symbols_target, symbols_source, cell):
    """Find optimal element-constrained permutation; return per-atom costs after match."""
    n = len(pos_target)
    LARGE = 1e9
    cost = np.full((n, n), LARGE)
    for i in range(n):
        for j in range(n):
            if symbols_target[i] != symbols_source[j]:
                continue
            delta = pos_source[j] - pos_target[i]
            frac = np.linalg.solve(cell.T, delta)
            frac -= np.round(frac)
            cost[i, j] = np.linalg.norm(frac @ cell)
    row, col = linear_sum_assignment(cost)
    return col, cost[row, col]


def find_qualifying_op(pristine_atoms, S_i, S_k, V_Fe, h_idx, endA_atoms, symprec=0.05):
    """Find R (S_i → S_k AND V_Fe fixed). Returns list of all qualifying ops with H disp."""
    cell = np.array(pristine_atoms.get_cell())
    pos_frac = pristine_atoms.get_scaled_positions()
    pos_endA_frac = endA_atoms.get_scaled_positions()
    spgcell = (cell, pos_frac, pristine_atoms.get_atomic_numbers())
    sym = spglib.get_symmetry(spgcell, symprec=symprec)

    candidates = []
    for R, t in zip(sym["rotations"], sym["translations"]):
        si_new = R @ pos_frac[S_i] + t
        d_si = np.linalg.norm(((si_new - pos_frac[S_k]) - np.round(si_new - pos_frac[S_k])) @ cell)
        if d_si > 0.1:
            continue
        if V_Fe is not None:
            vfe_new = R @ pos_frac[V_Fe] + t
            d_vfe = np.linalg.norm(((vfe_new - pos_frac[V_Fe]) - np.round(vfe_new - pos_frac[V_Fe])) @ cell)
            if d_vfe > 0.1:
                continue
        h_new = R @ pos_endA_frac[h_idx] + t
        h_disp = np.linalg.norm(((h_new - pos_endA_frac[h_idx]) - np.round(h_new - pos_endA_frac[h_idx])) @ cell)
        candidates.append({"R": R, "t": t, "h_disp": float(h_disp)})

    candidates.sort(key=lambda c: -c["h_disp"])
    return candidates


def run_test(mineral_name, pristine_path, endA_path, triple_path,
             known_dft_barrier_meV=None):
    print("\n" + "=" * 78)
    print(f"SYMMETRY PRE-FLIGHT: {mineral_name}")
    print("=" * 78)

    pristine = read(pristine_path)
    endA = read(endA_path)
    with open(triple_path) as f:
        triple = json.load(f)

    print(f"  pristine: {len(pristine)} atoms, {pristine.get_chemical_formula()}")
    print(f"  endA:     {len(endA)} atoms, {endA.get_chemical_formula()}")

    # Detect V_Fe index — in triple or guess
    V_Fe = triple.get("V_Fe_index") or triple.get("V_Fe") or triple.get("vacancy_index")
    S_i = triple.get("S_i_index") or triple.get("S_i")
    S_k = triple.get("S_k_index") or triple.get("S_k")
    hop_dist = triple.get("hop_distance_A") or triple.get("hop_distance") or triple.get("hop")

    print(f"  V_Fe = {V_Fe}, S_i = {S_i}, S_k = {S_k}, hop_dist = {hop_dist} Å")

    # spglib summary of pristine
    cell = np.array(pristine.get_cell())
    pos_frac = pristine.get_scaled_positions()
    spgcell = (cell, pos_frac, pristine.get_atomic_numbers())
    sym_data = spglib.get_symmetry_dataset(spgcell, symprec=0.05)
    print(f"  pristine space group: {sym_data.international} (#{sym_data.number})")
    print(f"  num symmetry ops: {len(sym_data.rotations)}")

    # Find H index in endA
    syms_endA = endA.get_chemical_symbols()
    h_idx = [i for i, s in enumerate(syms_endA) if s == "H"]
    if len(h_idx) != 1:
        print(f"  ✗ Expected 1 H atom, found {len(h_idx)}")
        return {"mineral": mineral_name, "status": "ERROR_NO_H"}
    h_idx = h_idx[0]

    # Find qualifying ops
    if S_i is None or S_k is None or V_Fe is None:
        print(f"  ⚠ Missing triple info (V_Fe={V_Fe}, S_i={S_i}, S_k={S_k}) — partial test only")
        candidates = []
    else:
        candidates = find_qualifying_op(pristine, S_i, S_k, V_Fe, h_idx, endA)
        print(f"  Qualifying symmetry ops (R(S_i)=S_k & R(V_Fe)=V_Fe): {len(candidates)}")
        if candidates:
            print(f"  Best op H displacement: {candidates[0]['h_disp']:.3f} Å")

    # Test 1: pristine R-recovery (baseline)
    if candidates:
        R, t = candidates[0]["R"], candidates[0]["t"]
        pos_pristine_rot = apply_R(pristine, R, t)
        perm_p, costs_p = hungarian_match(
            pristine.get_positions(), pos_pristine_rot,
            pristine.get_chemical_symbols(), pristine.get_chemical_symbols(),
            cell,
        )
        max_p = float(costs_p.max())
        mean_p = float(costs_p.mean())
        print(f"\n  Test 1 [pristine]: max disp {max_p:.4f} Å (mean {mean_p:.4f})")
        print(f"    >0.01 Å atoms: {(costs_p > 0.01).sum()}/{len(pristine)}")

        # Test 2: endA R-recovery (key test)
        pos_endA_rot = apply_R(endA, R, t)
        perm_e, costs_e = hungarian_match(
            endA.get_positions(), pos_endA_rot,
            syms_endA, syms_endA, cell,
        )
        # Exclude H from non-H stats
        non_h_costs = np.delete(costs_e, h_idx)
        max_e = float(non_h_costs.max())
        mean_e = float(non_h_costs.mean())
        n_above_01 = int((non_h_costs > 0.1).sum())
        n_above_1 = int((non_h_costs > 1.0).sum())
        h_cost = float(costs_e[h_idx])

        print(f"\n  Test 2 [endA with V_Fe + H]:")
        print(f"    H displacement: {h_cost:.4f} Å")
        print(f"    Non-H max disp: {max_e:.4f} Å (mean {mean_e:.4f})")
        print(f"    Non-H >0.1 Å: {n_above_01}/{len(syms_endA)-1}")
        print(f"    Non-H >1.0 Å: {n_above_1}/{len(syms_endA)-1}")

        # Verdict
        if max_e < 0.1:
            verdict = "SYMMETRIC ✓"
            pred = "NEB should converge cleanly (symmetric MEP exists)"
        elif max_e < 0.5:
            verdict = "MARGINAL ⚠"
            pred = "NEB may work with care; minor pocket distortion"
        elif max_e < 1.5:
            verdict = "WEAK ASYMMETRY ⚠"
            pred = "NEB feasible but possibly multi-saddle; expect convergence issues"
        else:
            verdict = "STRONG ASYMMETRY ✗"
            pred = "Symmetric MEP framework inapplicable; standard NEB likely fails"

        print(f"\n  VERDICT: {verdict}")
        print(f"  PREDICTION: {pred}")
        if known_dft_barrier_meV is not None:
            print(f"  GROUND TRUTH: barrier = {known_dft_barrier_meV} meV (known)")
        else:
            print(f"  GROUND TRUTH: unknown — this is PREDICTION")

        return {
            "mineral": mineral_name,
            "space_group": sym_data.international,
            "n_atoms": len(endA),
            "h_displacement_A": h_cost,
            "non_h_max_disp_A": max_e,
            "non_h_mean_disp_A": mean_e,
            "non_h_above_0.1A": n_above_01,
            "non_h_above_1A": n_above_1,
            "pristine_max_disp_A": max_p,
            "verdict": verdict,
            "prediction": pred,
            "known_barrier_meV": known_dft_barrier_meV,
        }

    return {"mineral": mineral_name, "status": "NO_QUALIFYING_OP"}


# Cases — fill in paths after searcher returns
CASES = [
    {
        "name": "mack W4 V_Fe (known 43 meV)",
        "pristine": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-03\mack_vfe_w3_aborted_2026-05-03\neb_canonical_mack_72at_qe_VFe\relaxed_pristine.xyz",
        "endA": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-03\mack_vfe_w3_aborted_2026-05-03\neb_canonical_mack_72at_qe_VFe\relaxed_endA.xyz",
        "triple": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-03\mack_vfe_w3_aborted_2026-05-03\neb_canonical_mack_72at_qe_VFe\canonical_triple.json",
        "barrier_meV": 43.0,
    },
    {
        "name": "greig W2 V_Fe (known 1861 meV)",
        "pristine": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-27\w2_greigite_full_neb\greig_neb_full_s150\relaxed_pristine.xyz",
        "endA": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-27\w2_greigite_full_neb\greig_neb_full_s150\relaxed_endA.xyz",
        "triple": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-27\w2_greigite_full_neb\greig_neb_full_s150\canonical_triple.json",
        "barrier_meV": 1861.0,
    },
    {
        "name": "pent W3 V_Fe (PREDICTION — DEFERRED)",
        "pristine": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-10\w3_pent_136at_qe_VFe\prod_dir\relaxed_pristine.xyz",
        "endA": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-10\w3_pent_136at_qe_VFe\prod_dir\relaxed_endA.xyz",
        "triple": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-10\w3_pent_136at_qe_VFe\prod_dir\canonical_triple.json",
        "barrier_meV": None,  # this is the prediction case we already tested
    },
    {
        "name": "pyr V_S2 (known 94.6 meV)",
        "pristine": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-01\pyr_prod_neb_W3\prod_essentials\relaxed_pristine.xyz",
        "endA": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-01\pyr_prod_neb_W3\prod_essentials\relaxed_endA.xyz",
        "triple": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-01\pyr_prod_neb_W3\prod_essentials\canonical_triple.json",
        "barrier_meV": 94.6,
    },
    {
        "name": "pyr V_Fe W2 Tier1 (LIVE TEST — pre-committed prediction 0.05-0.15 Å)",
        "pristine": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-28\pyr_VFe_W2_tier1\neb_canonical_pyr_96at_qe_VFe\relaxed_pristine.xyz",
        "endA": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-28\pyr_VFe_W2_tier1\neb_canonical_pyr_96at_qe_VFe\relaxed_endA.xyz",
        "triple": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-28\pyr_VFe_W2_tier1\neb_canonical_pyr_96at_qe_VFe\canonical_triple.json",
        "barrier_meV": None,  # NEB not yet run; ΔE_endpoints = 0.0000 known
    },
    {
        "name": "marc V_S smoke (NEB status=ok, V_S vacancy)",
        "pristine": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-18_marcasite_smoke_w1\marc_smoke\relaxed_pristine.xyz",
        "endA": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-18_marcasite_smoke_w1\marc_smoke\relaxed_endA.xyz",
        "triple": r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-18_marcasite_smoke_w1\marc_smoke\canonical_triple.json",
        "barrier_meV": None,
    },
]


def run_symmetry_l1(
    pristine_path,
    end_a_path,
    triple_path,
    mineral_name="system",
    known_dft_barrier_meV=None,
    verbose=False,
) -> dict:
    """Run the L1 symmetry gate and return an MCP-shaped envelope."""
    if verbose:
        result = run_test(
            mineral_name,
            pristine_path,
            end_a_path,
            triple_path,
            known_dft_barrier_meV,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_test(
                mineral_name,
                pristine_path,
                end_a_path,
                triple_path,
                known_dft_barrier_meV,
            )

    verdict = result.get("verdict") or result.get("status", "UNKNOWN")
    confidence = "high" if "SYMMETRIC" in verdict or "ASYMMETRY" in verdict else "review"
    reasons = []
    next_actions = []
    if result.get("status"):
        reasons.append(result["status"])
        next_actions.append("fix input files or canonical_triple metadata")
    elif "ASYMMETRY" in verdict:
        next_actions.append("use multi-endpoint/string workflow before ordinary NEB")
    elif "MARGINAL" in verdict:
        next_actions.append("cross-check with MLIP endpoint enumeration and magnetic gates")
    else:
        next_actions.append("continue to magnetic gates before launching NEB")

    return response_envelope(
        tool="run_symmetry_l1",
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        next_actions=next_actions,
        result=result,
    )


def run_validation_set(json_output=False, output=None):
    print("V_Fe Symmetry Pre-Flight Test — Generalized")
    print(f"Tests: {len(CASES)} minerals")

    all_results = []
    for case in CASES:
        if case["triple"] is None:
            print(f"\n⚠ Skipping {case['name']} — triple_json path unknown")
            continue
        for f in [case["pristine"], case["endA"], case["triple"]]:
            if not Path(f).exists():
                print(f"\n✗ Skipping {case['name']} — file missing: {f}")
                break
        else:
            result = run_test(case["name"], case["pristine"], case["endA"],
                              case["triple"], case.get("barrier_meV"))
            all_results.append(result)

    # Summary table
    print("\n" + "=" * 78)
    print("SUMMARY: V_Fe symmetry pre-flight verdicts")
    print("=" * 78)
    print(f"{'Mineral':<40s} {'Non-H max':>12s} {'Verdict':>20s}")
    print("-" * 78)
    for r in all_results:
        if "non_h_max_disp_A" in r:
            print(f"{r['mineral']:<40s} {r['non_h_max_disp_A']:>10.4f} Å {r['verdict']:>20s}")
        else:
            print(f"{r['mineral']:<40s} {'ERROR':>12s} {r.get('status', '?'):>20s}")

    # Save results
    out_path = Path(r"D:\home\ignat\project-third-matter\dft-neb\ph-diagnostic\symmetry_preflight_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    if json_output or output:
        dump_json(
            response_envelope(
                tool="run_symmetry_l1_validation",
                verdict="VALIDATION_SET",
                confidence="reference",
                result=all_results,
            ),
            output,
        )

    return 0


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pristine", help="Path to pristine structure")
    parser.add_argument("--end-a", help="Path to relaxed endpoint A structure")
    parser.add_argument("--triple", help="Path to canonical_triple.json")
    parser.add_argument("--name", default="system", help="Optional display/system name")
    parser.add_argument("--barrier-mev", type=float, help="Optional known/reference barrier")
    parser.add_argument("--json", action="store_true", help="Print MCP-shaped JSON instead of text")
    parser.add_argument("--output", help="Optional path for JSON output")
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Run the local validation set with historical hardcoded paths",
    )
    args = parser.parse_args(argv)

    if args.validation:
        return run_validation_set(json_output=args.json, output=args.output)

    if not args.pristine or not args.end_a or not args.triple:
        parser.error("--pristine, --end-a, and --triple are required unless --validation is used")

    envelope = run_symmetry_l1(
        args.pristine,
        args.end_a,
        args.triple,
        mineral_name=args.name,
        known_dft_barrier_meV=args.barrier_mev,
        verbose=not args.json,
    )
    if args.output:
        dump_json(envelope, args.output)
    if args.json:
        dump_json(envelope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
