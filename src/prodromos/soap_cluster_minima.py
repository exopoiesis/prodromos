"""
SOAP-based clustering of relaxed H site structures.

After MACE-relax of N candidate H positions → N possibly-degenerate minima.
Cluster by Smooth Overlap of Atomic Positions (SOAP) descriptor distance.

Steps:
1. For each relaxed structure: compute SOAP descriptor centered on H atom
2. Pairwise SOAP distance matrix
3. Hierarchical clustering (Agglomerative) with linkage threshold
4. Output: cluster assignments + representative structure per cluster
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from sklearn.cluster import AgglomerativeClustering

from prodromos.cli_contract import dump_json, response_envelope


def compute_soap_around_H(atoms, r_cut=4.5, n_max=6, l_max=4):
    """SOAP descriptor centered on H atom."""
    species = sorted(set(atoms.get_chemical_symbols()))
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=0.5,
        periodic=True,
        sparse=False,
    )
    syms = atoms.get_chemical_symbols()
    h_idx = [i for i, s in enumerate(syms) if s == "H"]
    if not h_idx:
        raise ValueError("No H atom in structure")
    return soap.create(atoms, centers=[h_idx[0]])[0]


def cluster_structures(relaxed_dir, summary_path, distance_threshold=0.5):
    """Read all relaxed_*.xyz, compute SOAP per H, cluster by SOAP distance."""
    relaxed_dir = Path(relaxed_dir)
    with open(summary_path) as f:
        summary = json.load(f)

    results = [r for r in summary["results"] if "E_final_eV" in r]
    if not results:
        print("No successful relaxations to cluster")
        return None

    print(f"\n=== SOAP Clustering ({len(results)} relaxed structures) ===")
    descriptors = []
    valid_results = []
    for r in results:
        xyz = Path(r["out_xyz"])
        # Try local path if remote path doesn't exist
        if not xyz.exists():
            local_name = xyz.name
            xyz = relaxed_dir / local_name
        if not xyz.exists():
            print(f"  ⚠ Missing: {xyz}")
            continue
        try:
            atoms = read(xyz)
            desc = compute_soap_around_H(atoms)
            descriptors.append(desc)
            valid_results.append(r)
        except Exception as e:
            print(f"  ⚠ {r['candidate_id']}: {e}")

    descriptors = np.array(descriptors)
    if len(valid_results) == 0:
        print("No readable relaxed structures to cluster")
        return []

    print(f"  SOAP shape: {descriptors.shape}")

    # Normalize (unit length per descriptor)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors_norm = descriptors / (norms + 1e-12)

    # Pairwise cosine distance
    similarity = descriptors_norm @ descriptors_norm.T
    distance = 1.0 - similarity
    distance = np.clip(distance, 0, None)  # numerical safety

    print(f"  Pairwise distance: min={distance[distance > 0].min():.4f}, "
          f"max={distance.max():.4f}, mean={distance.mean():.4f}")

    # Agglomerative clustering with cosine distance threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance)

    n_clusters = len(set(labels))
    print(f"\n  → {n_clusters} distinct clusters (threshold {distance_threshold})")

    # Group results by cluster, find representative (lowest E)
    clusters = {}
    for r, label in zip(valid_results, labels):
        clusters.setdefault(int(label), []).append(r)

    print("\n  Cluster summary (sorted by E_rep):")
    cluster_info = []
    for cid, members in clusters.items():
        members.sort(key=lambda x: x["E_final_eV"])
        rep = members[0]
        cluster_info.append({
            "cluster_id": cid,
            "size": len(members),
            "representative": rep,
            "E_representative_eV": rep["E_final_eV"],
            "class": rep["class"],
            "all_members": [m["candidate_id"] for m in members],
        })
    cluster_info.sort(key=lambda x: x["E_representative_eV"])

    for ci in cluster_info:
        print(f"\n  Cluster {ci['cluster_id']} (size {ci['size']}):")
        print(f"    Representative: {ci['representative']['candidate_id']} ({ci['class']})")
        print(f"    E = {ci['E_representative_eV']:.4f} eV")
        if len(ci["all_members"]) > 1:
            print(f"    Members: {', '.join(ci['all_members'])}")

    # Save cluster summary
    out_path = relaxed_dir / "cluster_summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_clusters": n_clusters,
            "distance_threshold": distance_threshold,
            "clusters": cluster_info,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    return cluster_info


def run_soap_clustering(relaxed_dir, summary_json, threshold=0.5, verbose=False) -> dict:
    """Run SOAP clustering and return an MCP-shaped envelope."""
    if verbose:
        clusters = cluster_structures(relaxed_dir, summary_json, threshold)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            clusters = cluster_structures(relaxed_dir, summary_json, threshold)

    clusters = clusters or []
    verdict = "CLUSTERS_FOUND" if clusters else "REVIEW"
    warnings = [] if clusters else ["no clusters found from successful readable relaxations"]
    return response_envelope(
        tool="cluster_mlip_minima",
        verdict=verdict,
        confidence="medium",
        next_actions=["send representative minima to DFT single-point screening"] if clusters else [],
        artifacts=[str(Path(relaxed_dir) / "cluster_summary.json")] if clusters else [],
        warnings=warnings,
        result={
            "relaxed_dir": str(relaxed_dir),
            "summary_json": str(summary_json),
            "distance_threshold": threshold,
            "n_clusters": len(clusters),
            "clusters": clusters,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relaxed-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="SOAP cosine-distance threshold for clustering")
    parser.add_argument("--json", action="store_true", help="Print MCP-shaped JSON instead of text")
    parser.add_argument("--output", help="Optional path for JSON output")
    args = parser.parse_args(argv)

    envelope = run_soap_clustering(
        args.relaxed_dir,
        args.summary_json,
        threshold=args.threshold,
        verbose=not args.json,
    )
    if args.output:
        dump_json(envelope, args.output)
    if args.json:
        dump_json(envelope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
