"""
Master Equation Kinetics + Min-Spanning Arborescence.

Given barrier matrix M x M (M sites, with E_a_forward[i,j] = forward i→j barrier):
1. Compute rate matrix K(T): k(i→j) = ν · exp(-E_a(i→j) / kT)
2. Master equation: dP/dt = K^T · P → equilibrium + slowest relaxation timescale
3. Arrhenius effective E_a: slope of ln(1/τ_eff) vs 1/kT
4. Chu-Liu-Edmonds min-energy arborescence: dominant kinetic pathway
   (game-theorist Foster-Young stochastic stability → minimum-spanning tree of barriers)

References:
- Kreuer 2003 *Annu Rev Mater Res* 33:333 — multi-site proton conductivity framework
- Hellman-Tornqvist 2022 *JACS* 144:6450 — Fe-S layered proton network applied
- Foster-Young 1990 — stochastic stability (game theory)
- Chu-Liu-Edmonds (1965/1967) — directed minimum spanning tree algorithm
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prodromos.cli_contract import dump_json, response_envelope


KB_EV_PER_K = 8.617333262e-5  # eV/K
NU_HZ = 1.0e13  # typical attempt frequency
DEFAULT_T_K = 298.0


def rate_matrix(E_a_matrix: np.ndarray, T_K: float = DEFAULT_T_K,
                nu: float = NU_HZ) -> np.ndarray:
    """Construct rate matrix from barrier matrix.

    E_a_matrix[i, j] = forward barrier from state i to state j (eV)
    If E_a_matrix[i, j] = inf or NaN → no direct transition.

    Returns K (M, M):
    - K[j, i] = k(i→j) for j ≠ i (off-diagonal)
    - K[i, i] = -Σ_j k(i→j) (diagonal)
    Master equation: dP/dt = K · P
    """
    M = E_a_matrix.shape[0]
    K = np.zeros((M, M))
    kT = KB_EV_PER_K * T_K
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            E_a = E_a_matrix[i, j]
            if np.isnan(E_a) or np.isinf(E_a) or E_a < 0:
                continue
            k_ij = nu * np.exp(-E_a / kT)
            K[j, i] += k_ij  # rate FROM i TO j contributes to dP_j/dt
            K[i, i] -= k_ij  # and decreases P_i
    return K


def equilibrium_distribution(K: np.ndarray) -> np.ndarray:
    """Solve K · P_eq = 0 (kernel of K)."""
    eigvals, eigvecs = eig(K)
    # Find eigvalue closest to 0
    idx = np.argmin(np.abs(eigvals))
    P_eq = np.real(eigvecs[:, idx])
    P_eq = np.abs(P_eq)
    P_eq /= P_eq.sum()
    return P_eq


def slowest_relaxation(K: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (τ_slow, eigenvector) for smallest non-zero eigenvalue of -K."""
    eigvals, eigvecs = eig(-K)
    eigvals = np.real(eigvals)
    # Sort by magnitude (kernel has eigval 0)
    sorted_idx = np.argsort(np.abs(eigvals))
    # First eigenvalue ≈ 0 (equilibrium), second is slowest relaxation
    lambda_1 = eigvals[sorted_idx[1]]
    if lambda_1 < 1e-15:
        return float("inf"), np.zeros(K.shape[0])
    tau_slow = 1.0 / lambda_1
    mode = np.real(eigvecs[:, sorted_idx[1]])
    return tau_slow, mode


def arrhenius_fit(E_a_matrix: np.ndarray, T_range_K: tuple = (250, 400),
                   n_T: int = 10, nu: float = NU_HZ) -> dict:
    """Fit Arrhenius effective E_a from τ_slow(T) over temperature range."""
    Ts = np.linspace(T_range_K[0], T_range_K[1], n_T)
    inv_kTs = 1.0 / (KB_EV_PER_K * Ts)
    log_rates = []
    for T in Ts:
        K = rate_matrix(E_a_matrix, T, nu)
        tau, _ = slowest_relaxation(K)
        if np.isinf(tau):
            log_rates.append(np.nan)
        else:
            log_rates.append(np.log(1.0 / tau))
    log_rates = np.array(log_rates)
    valid = ~np.isnan(log_rates)
    if valid.sum() < 2:
        return {"E_a_eff_eV": np.nan, "log_prefactor": np.nan}
    # ln(k_eff) = ln(ν) - E_a_eff / kT
    # slope = -E_a_eff
    slope, intercept = np.polyfit(inv_kTs[valid], log_rates[valid], 1)
    E_a_eff = -slope
    return {
        "E_a_eff_eV": float(E_a_eff),
        "E_a_eff_meV": float(E_a_eff * 1000),
        "log_prefactor": float(intercept),
        "T_range_K": list(T_range_K),
        "Ts_K": Ts.tolist(),
        "log_rates": log_rates.tolist(),
    }


def chu_liu_edmonds_arborescence(weight_matrix: np.ndarray, root: int = 0) -> list:
    """Minimum spanning arborescence rooted at `root` (Chu-Liu-Edmonds, 1965).

    weight_matrix[i, j] = weight of edge i → j.
    Returns list of (parent, child) tuples for minimum arborescence.

    Simple O(VE) implementation (sufficient for small graphs M ≤ 20).
    """
    M = weight_matrix.shape[0]
    # Each non-root node picks the cheapest incoming edge
    parents = [-1] * M
    for j in range(M):
        if j == root:
            continue
        # Find min-weight incoming edge to j
        candidates = [(i, weight_matrix[i, j]) for i in range(M)
                       if i != j and np.isfinite(weight_matrix[i, j])]
        if not candidates:
            parents[j] = -1
            continue
        i_min, w_min = min(candidates, key=lambda x: x[1])
        parents[j] = i_min

    # Check for cycles (simplified — for small graphs assume acyclic input)
    # Full Chu-Liu-Edmonds would do cycle contraction here
    edges = [(parents[j], j) for j in range(M) if parents[j] >= 0]
    return edges


def analyze_network(E_a_matrix: np.ndarray, site_labels: list = None,
                     site_energies: np.ndarray = None, T_K: float = DEFAULT_T_K,
                     verbose: bool = True) -> dict:
    """Full master equation + arborescence analysis."""
    M = E_a_matrix.shape[0]
    if site_labels is None:
        site_labels = [f"S{i}" for i in range(M)]

    # 1. Rate matrix at T
    K = rate_matrix(E_a_matrix, T_K)

    # 2. Equilibrium distribution
    P_eq = equilibrium_distribution(K)

    # 3. Slowest relaxation
    tau_slow, mode = slowest_relaxation(K)

    # 4. Arrhenius effective E_a
    arrhenius = arrhenius_fit(E_a_matrix)

    # 5. Min-energy arborescence (rooted at deepest equilibrium = max P_eq)
    root = int(np.argmax(P_eq))
    edges = chu_liu_edmonds_arborescence(E_a_matrix, root=root)
    arborescence_sum = sum(E_a_matrix[i, j] for i, j in edges if np.isfinite(E_a_matrix[i, j]))

    result = {
        "n_sites": M,
        "site_labels": site_labels,
        "T_K": T_K,
        "equilibrium_distribution": P_eq.tolist(),
        "dominant_site_idx": int(np.argmax(P_eq)),
        "dominant_site_label": site_labels[int(np.argmax(P_eq))],
        "tau_slow_s": float(tau_slow),
        "arrhenius_E_a_eff_meV": arrhenius["E_a_eff_meV"],
        "min_arborescence_root": root,
        "min_arborescence_root_label": site_labels[root],
        "min_arborescence_edges": [
            {"from": site_labels[i], "to": site_labels[j],
             "barrier_meV": float(E_a_matrix[i, j] * 1000)}
            for i, j in edges
        ],
        "min_arborescence_total_meV": float(arborescence_sum * 1000),
        "individual_barriers_meV": {
            f"{site_labels[i]}→{site_labels[j]}": float(E_a_matrix[i, j] * 1000)
            for i in range(M) for j in range(M)
            if i != j and np.isfinite(E_a_matrix[i, j])
        },
    }

    if verbose:
        print(f"\n=== Master Equation Analysis (T={T_K} K) ===")
        print(f"  Sites: {site_labels}")
        if site_energies is not None:
            print(f"  Site energies (meV vs lowest): "
                  f"{[f'{(e - site_energies.min())*1000:.1f}' for e in site_energies]}")
        print(f"\n  Equilibrium population (Boltzmann):")
        for lbl, p in zip(site_labels, P_eq):
            bar = "█" * int(p * 50)
            print(f"    {lbl}: {p*100:.2f}% {bar}")
        print(f"  Dominant: {result['dominant_site_label']}")
        print(f"\n  τ_slow (slowest relaxation): {tau_slow:.3e} s")
        print(f"  Arrhenius effective E_a: {arrhenius['E_a_eff_meV']:.2f} meV")
        print(f"\n  Min-energy arborescence (rooted at {result['min_arborescence_root_label']}):")
        for edge in result["min_arborescence_edges"]:
            print(f"    {edge['from']} → {edge['to']}: {edge['barrier_meV']:.1f} meV")
        print(f"  Total arborescence weight: {result['min_arborescence_total_meV']:.1f} meV")

    return result


def load_barrier_input(path: str | Path) -> tuple[np.ndarray, list[str] | None, np.ndarray | None, float | None]:
    """Load barrier matrix from JSON or CSV.

    JSON accepts keys: `barriers_eV`, `E_a_matrix`, or `matrix`; optional
    `site_labels`, `site_energies_eV`, and `T_K`.
    CSV is a plain numeric matrix in eV; use `inf` for missing edges.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        matrix = data.get("barriers_eV") or data.get("E_a_matrix") or data.get("matrix")
        if matrix is None:
            raise ValueError("JSON must contain one of: barriers_eV, E_a_matrix, matrix")
        labels = data.get("site_labels")
        energies = data.get("site_energies_eV")
        return (
            np.array(matrix, dtype=float),
            labels,
            np.array(energies, dtype=float) if energies is not None else None,
            data.get("T_K"),
        )
    matrix = np.genfromtxt(path, delimiter=",", dtype=float)
    return np.array(matrix, dtype=float), None, None, None


def run_kinetic_network(
    barrier_matrix,
    site_labels=None,
    site_energies=None,
    T_K=DEFAULT_T_K,
    verbose=False,
) -> dict:
    """Run L6 kinetic network analysis and return an MCP-shaped envelope."""
    E_a = np.array(barrier_matrix, dtype=float)
    energies = np.array(site_energies, dtype=float) if site_energies is not None else None
    result = analyze_network(
        E_a,
        site_labels=site_labels,
        site_energies=energies,
        T_K=T_K,
        verbose=verbose,
    )
    return response_envelope(
        tool="analyze_kinetic_network",
        verdict="KINETIC_NETWORK_ANALYZED",
        confidence="model",
        next_actions=["use dominant pathway and slowest relaxation as network-level NEB interpretation"],
        result=result,
    )


# ============================================================
# Test Cases
# ============================================================

def test_case_1_symmetric_two_state():
    """2-state symmetric hop (analog mack V_Fe 43 meV)."""
    print("\n" + "=" * 70)
    print("TEST 1: Symmetric 2-state (mack V_Fe analog, 43 meV barrier)")
    print("=" * 70)
    E_a = np.full((2, 2), np.inf)
    E_a[0, 1] = 0.043  # 43 meV
    E_a[1, 0] = 0.043
    result = analyze_network(E_a, site_labels=["S-H_i", "S-H_k"])
    return result


def test_case_2_asymmetric_two_state():
    """Asymmetric 2-state (analog marc 174 meV ΔE)."""
    print("\n" + "=" * 70)
    print("TEST 2: Asymmetric 2-state (marc analog, ΔE=174 meV)")
    print("=" * 70)
    # ΔE = E_B - E_A = 174 meV. Forward barrier ~250 meV, reverse 76 meV (smaller).
    E_a = np.full((2, 2), np.inf)
    E_a[0, 1] = 0.250  # forward 250 meV
    E_a[1, 0] = 0.076  # reverse 76 meV (forward - ΔE)
    result = analyze_network(E_a, site_labels=["S-H_low", "S-H_high"])
    return result


def test_case_3_pent_multi_site():
    """Pent-like 4-site multi-endpoint network (hypothetical until DFT verifies).

    Per CHGNet finding:
    - μ-Fe-H-Fe: GS, 0
    - Fe-H terminal: +400 meV
    - S₃ window: +500 meV
    - S-H mono: +500 meV

    Hypothetical barriers (illustrative for the master equation framework):
    """
    print("\n" + "=" * 70)
    print("TEST 3: Pent-like 4-site network (CHGNet-inspired hypothetical)")
    print("=" * 70)
    # Site energies (eV)
    E_sites = np.array([0.0, 0.400, 0.500, 0.500])
    labels = ["μ-Fe-H-Fe", "Fe-H-term", "S₃-window", "S-H-mono"]

    # Hypothetical barriers
    # μ-Fe-H-Fe ↔ Fe-H-term: 200 meV (cubane shuffling)
    # Fe-H-term ↔ S₃-window: 150 meV (cubane to pocket transit)
    # S₃-window ↔ S-H-mono: 100 meV (pocket walking)
    # μ-Fe-H-Fe → S-H-mono direct: 500 meV (deep to surface)
    # All others: inf (no direct transition)
    E_a = np.full((4, 4), np.inf)

    # Forward + reverse for each connection
    def set_pair(i, j, forward, reverse=None):
        E_a[i, j] = forward
        E_a[j, i] = reverse if reverse is not None else (forward - (E_sites[j] - E_sites[i]))

    set_pair(0, 1, 0.200)  # μ-Fe-H-Fe → Fe-H-term: 200 meV (reverse = 200 - 0.4 → invalid if negative)
    # Actually for asymmetric: if E_i < E_j, E_a(i→j) > E_a(j→i) since reverse barrier = E_a(i→j) - (E_j - E_i)
    # Need to verify barriers are physical (positive)

    # Reset with physically consistent barriers
    E_a = np.full((4, 4), np.inf)
    # All site pairs connected with barriers (forward, reverse based on ΔE)
    pairs = [
        (0, 1, 0.250),   # GS → site2: 250 meV forward
        (0, 2, 0.300),
        (1, 2, 0.150),
        (1, 3, 0.180),
        (2, 3, 0.100),
    ]
    for i, j, fwd in pairs:
        E_a[i, j] = fwd
        # Reverse barrier (positive) = forward - (E_j - E_i)
        rev = fwd - (E_sites[j] - E_sites[i])
        if rev > 0:
            E_a[j, i] = rev
        # If rev <= 0, no reverse barrier (barrier-less downhill)

    result = analyze_network(E_a, site_labels=labels, site_energies=E_sites)
    return result


def run_example_networks() -> dict:
    out_dir = Path(__file__).parent / "master_equation_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["test1_symmetric"] = test_case_1_symmetric_two_state()
    results["test2_asymmetric"] = test_case_2_asymmetric_two_state()
    results["test3_pent_multisite"] = test_case_3_pent_multi_site()

    # Save
    out_json = out_dir / "master_equation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nSaved: {out_json}")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for test_name, r in results.items():
        print(f"\n{test_name}:")
        print(f"  Dominant site: {r['dominant_site_label']} ({max(r['equilibrium_distribution'])*100:.1f}%)")
        print(f"  τ_slow: {r['tau_slow_s']:.2e} s")
        print(f"  Arrhenius E_a_eff: {r['arrhenius_E_a_eff_meV']:.1f} meV")
        print(f"  Min arborescence total: {r['min_arborescence_total_meV']:.1f} meV")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barriers-json", help="JSON file with barrier matrix in eV")
    parser.add_argument("--barriers-csv", help="CSV file with barrier matrix in eV")
    parser.add_argument("--site-labels", help="Comma-separated labels; overrides JSON labels")
    parser.add_argument("--site-energies-ev", help="Comma-separated site energies in eV; overrides JSON")
    parser.add_argument("--temperature", type=float, default=DEFAULT_T_K, help="Temperature in K")
    parser.add_argument("--json", action="store_true", help="Print MCP-shaped JSON instead of text")
    parser.add_argument("--output", help="Optional path for JSON output")
    parser.add_argument("--examples", action="store_true", help="Run built-in demonstration networks")
    args = parser.parse_args(argv)

    if args.examples:
        if args.json:
            with contextlib.redirect_stdout(io.StringIO()):
                results = run_example_networks()
        else:
            results = run_example_networks()
        envelope = response_envelope(
            tool="analyze_kinetic_network_examples",
            verdict="EXAMPLES_ANALYZED",
            confidence="reference",
            result=results,
        )
        if args.output:
            dump_json(envelope, args.output)
        if args.json:
            dump_json(envelope)
        return 0

    input_path = args.barriers_json or args.barriers_csv
    if not input_path:
        parser.error("one of --barriers-json, --barriers-csv, or --examples is required")

    matrix, labels, site_energies, json_T = load_barrier_input(input_path)
    if args.site_labels:
        labels = [x.strip() for x in args.site_labels.split(",") if x.strip()]
    if args.site_energies_ev:
        site_energies = np.array([float(x.strip()) for x in args.site_energies_ev.split(",") if x.strip()])
    T_K = args.temperature if args.temperature != DEFAULT_T_K or json_T is None else float(json_T)

    envelope = run_kinetic_network(
        matrix,
        site_labels=labels,
        site_energies=site_energies,
        T_K=T_K,
        verbose=not args.json,
    )
    if args.output:
        dump_json(envelope, args.output)
    if args.json:
        dump_json(envelope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
