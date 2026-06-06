"""
Multi-Endpoint v3 site enumeration — generate 14-17 H candidate positions
for the V_Fe pocket of any Fe-S mineral.

Site classes (chemist taxonomy):
1. **S-H monodentate** (6 sites): H at 1.35 Å along V_Fe → S_j for each S neighbor
2. **μ-S-H-S bridging** (3-4 sites): H at midpoint of close S-S pairs
3. **Fe-hydride terminal** (3-4 sites): H at 1.6 Å along V_Fe → Fe_neighbor for nearest cubane Fe
4. **μ-Fe-H-Fe bridging** (2-3 sites): H at midpoint of close Fe-Fe pairs (cubane edges)
5. **Interstitial trigonal S₃ window** (1-2 sites): H at center of S₃ face

Output: xyz files ready for MACE relax + DFT singlepoint screen.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read, write

from prodromos.cli_contract import dump_json, response_envelope


def mic_vec(p1, p2, cell):
    """MIC vector p1 → p2."""
    delta = p2 - p1
    frac = np.linalg.solve(cell.T, delta)
    frac -= np.round(frac)
    return frac @ cell


def remove_atom_and_add_H(pristine: Atoms, V_idx: int, h_position: np.ndarray) -> Atoms:
    """Create endA-like structure: remove atom at V_idx, append H at h_position."""
    positions = pristine.get_positions()
    symbols = list(pristine.get_chemical_symbols())

    mask = np.ones(len(pristine), dtype=bool)
    mask[V_idx] = False
    new_positions = positions[mask]
    new_symbols = [s for i, s in enumerate(symbols) if mask[i]]

    new_positions = np.vstack([new_positions, h_position[np.newaxis, :]])
    new_symbols.append("H")

    new_atoms = Atoms(
        symbols=new_symbols,
        positions=new_positions,
        cell=pristine.get_cell(),
        pbc=pristine.get_pbc(),
    )
    return new_atoms


def enumerate_sites(pristine_path: str, triple_path: str, out_dir: Path,
                     d_SH: float = 1.35, d_FeH: float = 1.60,
                     fe_cutoff: float = 3.5, ss_cutoff: float = 3.5):
    """Generate trial H positions for V_Fe pocket."""

    pristine = read(pristine_path)
    with open(triple_path) as f:
        triple = json.load(f)

    V_Fe = triple["V_Fe_index"]
    cell = np.array(pristine.get_cell())
    positions = pristine.get_positions()
    symbols = pristine.get_chemical_symbols()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    V_pos = positions[V_Fe]
    pristine_name = Path(pristine_path).name
    print(f"\n=== Site enumeration: {pristine_name} ===")
    print(f"  V_idx = {V_Fe} ({symbols[V_Fe]}-vacancy)")
    print(f"  V position: {V_pos}")
    print(f"  n_atoms: {len(pristine)}, formula: {pristine.get_chemical_formula()}")

    # ============ Class 1: S-H monodentate ============
    s_neighbors = []
    for i, s in enumerate(symbols):
        if s != "S" or i == V_Fe:
            continue
        v = mic_vec(V_pos, positions[i], cell)
        d = np.linalg.norm(v)
        s_neighbors.append((i, d, v))
    s_neighbors.sort(key=lambda x: x[1])
    n_S_coord = 6  # octahedral expected
    s_coord = s_neighbors[:n_S_coord]
    print(f"\n  {n_S_coord} nearest S to V (Class 1 anchors):")
    for s_idx, d, v in s_coord:
        print(f"    S#{s_idx}: d={d:.3f} Å")

    candidates = []
    for i, (s_idx, d_VS, v) in enumerate(s_coord):
        # H at 1.35 Å from S, along (V_Fe → S) line, BEYOND S? No — INWARD from S toward V_Fe.
        # Place H at S_pos - d_SH * (V→S unit vector) = on V-side of S
        v_hat = v / np.linalg.norm(v)
        # H position: 1.35 Å from S toward V centroid
        h_pos = positions[s_idx] - d_SH * v_hat
        candidates.append({
            "id": f"class1_SH_S{s_idx}",
            "class": "S-H monodentate",
            "anchor": f"S#{s_idx}",
            "h_position": h_pos.tolist(),
            "expected_d": {"S": d_SH},
        })

    # ============ Class 2: μ-S-H-S bridging ============
    # H at midpoint of S-S pairs within ss_cutoff
    n_class2 = 0
    s_indices_in_coord = [s[0] for s in s_coord]
    for i in range(len(s_indices_in_coord)):
        for j in range(i + 1, len(s_indices_in_coord)):
            si, sj = s_indices_in_coord[i], s_indices_in_coord[j]
            v_ij = mic_vec(positions[si], positions[sj], cell)
            d_ij = np.linalg.norm(v_ij)
            if d_ij > ss_cutoff:
                continue
            midpoint = positions[si] + 0.5 * v_ij
            # Optional: shift slightly out of plane toward V_Fe (to avoid trivial midpoint)
            candidates.append({
                "id": f"class2_SHS_{si}_{sj}",
                "class": "μ-S-H-S bridging",
                "anchor": f"S#{si}-S#{sj}",
                "h_position": midpoint.tolist(),
                "expected_d": {"S": d_ij / 2.0},
            })
            n_class2 += 1
            if n_class2 >= 4:
                break
        if n_class2 >= 4:
            break

    # ============ Class 3: Fe-hydride terminal ============
    # H at 1.6 Å along V_Fe → Fe_neighbor for nearest cubane Fe
    fe_neighbors = []
    for i, s in enumerate(symbols):
        if s != "Fe" or i == V_Fe:
            continue
        v = mic_vec(V_pos, positions[i], cell)
        d = np.linalg.norm(v)
        fe_neighbors.append((i, d, v))
    fe_neighbors.sort(key=lambda x: x[1])
    print("\n  Nearest 6 Fe to V (Class 3 anchors, cubane test):")
    for fe_idx, d, v in fe_neighbors[:6]:
        marker = " ← CLOSE (cubane edge?)" if d < fe_cutoff else ""
        print(f"    Fe#{fe_idx}: d={d:.3f} Å{marker}")

    # Take 4 nearest Fe for Fe-hydride trials
    fe_class3 = fe_neighbors[:4]
    for fe_idx, d_VFe, v in fe_class3:
        v_hat = v / np.linalg.norm(v)
        # H at 1.6 Å from Fe toward V_Fe (Fe-hydride terminal coordination)
        h_pos = positions[fe_idx] - d_FeH * v_hat
        candidates.append({
            "id": f"class3_FeH_Fe{fe_idx}",
            "class": "Fe-hydride terminal",
            "anchor": f"Fe#{fe_idx}",
            "h_position": h_pos.tolist(),
            "expected_d": {"Fe": d_FeH},
        })

    # ============ Class 4: μ-Fe-H-Fe bridging ============
    # H at midpoint of Fe-Fe pairs within fe_cutoff (cubane edges)
    n_class4 = 0
    fe_indices_4 = [fe[0] for fe in fe_class3]
    for i in range(len(fe_indices_4)):
        for j in range(i + 1, len(fe_indices_4)):
            fi, fj = fe_indices_4[i], fe_indices_4[j]
            v_ij = mic_vec(positions[fi], positions[fj], cell)
            d_ij = np.linalg.norm(v_ij)
            if d_ij > fe_cutoff:
                continue
            midpoint = positions[fi] + 0.5 * v_ij
            candidates.append({
                "id": f"class4_FeHFe_{fi}_{fj}",
                "class": "μ-Fe-H-Fe bridging",
                "anchor": f"Fe#{fi}-Fe#{fj}",
                "h_position": midpoint.tolist(),
                "expected_d": {"Fe": d_ij / 2.0},
            })
            n_class4 += 1
            if n_class4 >= 3:
                break
        if n_class4 >= 3:
            break

    # ============ Class 5: Interstitial trigonal S₃ window ============
    # H at centroid of 3 S forming triangle face
    # Pick 3 S that form most-equilateral triangle among coord
    n_class5 = 0
    from itertools import combinations
    best_triangles = []
    for tri in combinations(range(len(s_indices_in_coord)), 3):
        si, sj, sk = [s_indices_in_coord[t] for t in tri]
        v_ij = mic_vec(positions[si], positions[sj], cell)
        v_ik = mic_vec(positions[si], positions[sk], cell)
        v_jk = mic_vec(positions[sj], positions[sk], cell)
        sides = sorted([np.linalg.norm(v_ij), np.linalg.norm(v_ik), np.linalg.norm(v_jk)])
        # Equilateral score: how close sides are to each other
        equilat = sides[2] / sides[0]  # ratio max/min, closer to 1 = more equilateral
        # Also prefer smaller triangles (closer S₃ — H actually fits)
        if sides[2] > 5.0:  # too large window
            continue
        best_triangles.append((tri, equilat, sides[0], (si, sj, sk)))

    best_triangles.sort(key=lambda x: (x[1], -x[2]))  # equilateral + larger
    for tri_idx, (tri, equilat, side_min, (si, sj, sk)) in enumerate(best_triangles[:2]):
        centroid = (positions[si]
                    + positions[si] + mic_vec(positions[si], positions[sj], cell)
                    + positions[si] + mic_vec(positions[si], positions[sk], cell)) / 3.0
        candidates.append({
            "id": f"class5_S3_{si}_{sj}_{sk}",
            "class": "interstitial S₃ window",
            "anchor": f"S#{si}-S#{sj}-S#{sk}",
            "h_position": centroid.tolist(),
            "expected_d": {"S": side_min / np.sqrt(3)},
        })
        n_class5 += 1

    print(f"\n  Total candidates generated: {len(candidates)}")
    print(f"    Class 1 (S-H monodentate): {sum(1 for c in candidates if 'class1' in c['id'])}")
    print(f"    Class 2 (μ-S-H-S):          {sum(1 for c in candidates if 'class2' in c['id'])}")
    print(f"    Class 3 (Fe-H terminal):    {sum(1 for c in candidates if 'class3' in c['id'])}")
    print(f"    Class 4 (μ-Fe-H-Fe):        {sum(1 for c in candidates if 'class4' in c['id'])}")
    print(f"    Class 5 (S₃ window):        {sum(1 for c in candidates if 'class5' in c['id'])}")

    # ============ Save candidate structures ============
    for cand in candidates:
        h_pos = np.array(cand["h_position"])
        atoms = remove_atom_and_add_H(pristine, V_Fe, h_pos)
        out_xyz = out_dir / f"{cand['id']}.xyz"
        write(out_xyz, atoms, format="extxyz")
        cand["out_xyz"] = str(out_xyz)

    # Save manifest
    manifest = {
        "pristine_input": pristine_path,
        "triple": triple,
        "V_idx": V_Fe,
        "V_element": symbols[V_Fe],
        "n_candidates": len(candidates),
        "candidates": candidates,
        "parameters": {
            "d_SH": d_SH,
            "d_FeH": d_FeH,
            "fe_cutoff_for_pairs": fe_cutoff,
            "ss_cutoff_for_pairs": ss_cutoff,
        },
    }
    manifest_path = out_dir / "enumeration_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Saved {len(candidates)} candidate xyz files + manifest: {manifest_path}")

    return manifest


def run_endpoint_enumeration(
    pristine_path,
    triple_path,
    out_dir,
    d_SH=1.35,
    d_FeH=1.60,
    fe_cutoff=3.5,
    ss_cutoff=3.5,
    verbose=False,
) -> dict:
    """Run L2 endpoint enumeration and return an MCP-shaped envelope."""
    if verbose:
        manifest = enumerate_sites(
            pristine_path,
            triple_path,
            out_dir,
            d_SH=d_SH,
            d_FeH=d_FeH,
            fe_cutoff=fe_cutoff,
            ss_cutoff=ss_cutoff,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = enumerate_sites(
                pristine_path,
                triple_path,
                out_dir,
                d_SH=d_SH,
                d_FeH=d_FeH,
                fe_cutoff=fe_cutoff,
                ss_cutoff=ss_cutoff,
            )

    warnings = []
    if manifest["n_candidates"] == 0:
        warnings.append("no candidate H endpoint sites generated")
    return response_envelope(
        tool="enumerate_endpoint_sites",
        verdict="CANDIDATES_GENERATED" if manifest["n_candidates"] else "REVIEW",
        confidence="medium",
        next_actions=["relax candidates with MACE/CHGNet, then cluster SOAP minima"],
        artifacts=[str(Path(out_dir) / "enumeration_manifest.json")],
        warnings=warnings,
        result=manifest,
    )


def run_historical_examples():
    base = Path(".")

    # Pent V_Fe enumeration
    pent_out = base / "experiments" / "2026-05-28_multi_endpoint_v3" / "pent_VFe"
    enumerate_sites(
        pristine_path=str(base / "results" / "dft_datasets" / "2026-05-10" /
                          "w3_pent_136at_qe_VFe" / "prod_dir" / "relaxed_pristine.xyz"),
        triple_path=str(base / "results" / "dft_datasets" / "2026-05-10" /
                        "w3_pent_136at_qe_VFe" / "prod_dir" / "canonical_triple.json"),
        out_dir=pent_out,
    )

    # Mack V_Fe enumeration (validation case — should give only 1-2 distinct sites)
    mack_out = base / "experiments" / "2026-05-28_multi_endpoint_v3" / "mack_VFe"
    enumerate_sites(
        pristine_path=str(base / "results" / "dft_datasets" / "2026-05-03" /
                          "mack_vfe_w3_aborted_2026-05-03" / "neb_canonical_mack_72at_qe_VFe" /
                          "relaxed_pristine.xyz"),
        triple_path=str(base / "results" / "dft_datasets" / "2026-05-03" /
                        "mack_vfe_w3_aborted_2026-05-03" / "neb_canonical_mack_72at_qe_VFe" /
                        "canonical_triple.json"),
        out_dir=mack_out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pristine", help="Path to pristine structure")
    parser.add_argument("--triple", help="Path to canonical_triple.json")
    parser.add_argument("--out-dir", help="Directory for generated candidate xyz files")
    parser.add_argument("--d-sh", type=float, default=1.35, help="S-H trial distance in Å")
    parser.add_argument("--d-feh", type=float, default=1.60, help="Fe-H trial distance in Å")
    parser.add_argument("--fe-cutoff", type=float, default=3.5, help="Fe-Fe pair cutoff in Å")
    parser.add_argument("--ss-cutoff", type=float, default=3.5, help="S-S pair cutoff in Å")
    parser.add_argument("--json", action="store_true", help="Print MCP-shaped JSON instead of text")
    parser.add_argument("--output", help="Optional path for JSON output")
    parser.add_argument(
        "--historical-examples",
        action="store_true",
        help="Run the original pent/mack examples with historical hardcoded paths",
    )
    args = parser.parse_args(argv)

    if args.historical_examples:
        run_historical_examples()
        return 0

    if not args.pristine or not args.triple or not args.out_dir:
        parser.error("--pristine, --triple, and --out-dir are required unless --historical-examples is used")

    envelope = run_endpoint_enumeration(
        args.pristine,
        args.triple,
        args.out_dir,
        d_SH=args.d_sh,
        d_FeH=args.d_feh,
        fe_cutoff=args.fe_cutoff,
        ss_cutoff=args.ss_cutoff,
        verbose=not args.json,
    )
    if args.output:
        dump_json(envelope, args.output)
    if args.json:
        dump_json(envelope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
