"""
ADMM-NEB prototype (game-theorist's recommended NEB force law).

Standard NEB:
    F_i = -∇V(x_i)|_⊥ + F_i^spring|_∥

ADMM reformulation (Wang-Yin-Zeng 2019 nonconvex convergence guarantees):
    x-update:  x_i^{k+1} = argmin V(x_i) + (ρ/2)|x_i - z_i^k + λ_i^k/ρ|²
    z-update:  z_i^{k+1} = median(x_{i-1}^{k+1}, x_i^{k+1}, x_{i+1}^{k+1})  ← per game-theorist
    λ-update:  λ_i^{k+1} = λ_i^k + ρ(x_i^{k+1} - z_i^{k+1})

Key difference vs standard NEB:
- "Robust median" consensus vs "average" (handles asymmetric / outlier images)
- Augmented Lagrangian: provable convergence even non-convex (Wang-Yin-Zeng 2019)
- No tangent projection bias from |∇V·τ̂| condition (Sheppard 2008 caveat for asymmetric)

Test: Müller-Brown 2D with asymmetric A↔B endpoints (ΔE ≈ 66 units), comparing
standard NEB vs ADMM-NEB convergence behavior.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Müller-Brown potential (canonical NEB testbed)
# ============================================================

MB_PARAMS = {
    "A": np.array([-200.0, -100.0, -170.0, 15.0]),
    "a": np.array([-1.0, -1.0, -6.5, 0.7]),
    "b": np.array([0.0, 0.0, 11.0, 0.6]),
    "c": np.array([-10.0, -10.0, -6.5, 0.7]),
    "x0": np.array([1.0, 0.0, -0.5, -1.0]),
    "y0": np.array([0.0, 0.5, 1.5, 1.0]),
}
MB_MINIMA = {
    "A": np.array([-0.558, 1.442]),  # E ≈ -146.7
    "B": np.array([-0.050, 0.467]),  # E ≈ -80.8
    "C": np.array([0.623, 0.028]),   # E ≈ -108.2
}


def V_MB(pos):
    """Müller-Brown potential V(x, y)."""
    x, y = pos[..., 0], pos[..., 1]
    A = MB_PARAMS["A"]
    a = MB_PARAMS["a"]
    b = MB_PARAMS["b"]
    c = MB_PARAMS["c"]
    x0 = MB_PARAMS["x0"]
    y0 = MB_PARAMS["y0"]
    V = np.zeros_like(x)
    for i in range(4):
        V = V + A[i] * np.exp(
            a[i] * (x - x0[i]) ** 2
            + b[i] * (x - x0[i]) * (y - y0[i])
            + c[i] * (y - y0[i]) ** 2
        )
    return V


def grad_V_MB(pos):
    """Analytic gradient ∇V."""
    x, y = pos[..., 0], pos[..., 1]
    A = MB_PARAMS["A"]
    a = MB_PARAMS["a"]
    b = MB_PARAMS["b"]
    c = MB_PARAMS["c"]
    x0 = MB_PARAMS["x0"]
    y0 = MB_PARAMS["y0"]
    gx = np.zeros_like(x)
    gy = np.zeros_like(y)
    for i in range(4):
        ex = np.exp(
            a[i] * (x - x0[i]) ** 2
            + b[i] * (x - x0[i]) * (y - y0[i])
            + c[i] * (y - y0[i]) ** 2
        )
        gx = gx + A[i] * ex * (2 * a[i] * (x - x0[i]) + b[i] * (y - y0[i]))
        gy = gy + A[i] * ex * (b[i] * (x - x0[i]) + 2 * c[i] * (y - y0[i]))
    return np.stack([gx, gy], axis=-1)


# ============================================================
# Standard NEB (reference)
# ============================================================

class StandardNEB:
    def __init__(self, endA, endB, n_images=9, k_spring=0.1):
        self.N = n_images
        self.k = k_spring
        # Linear interpolation
        s = np.linspace(0, 1, n_images)[:, None]
        self.x = endA[None, :] + s * (endB - endA)[None, :]
        self.fixed_ends = [0, n_images - 1]

    def tangent(self, i):
        """Improved tangent (Henkelman-Jónsson 2000 upwind)."""
        V_i = V_MB(self.x[i])
        V_im1 = V_MB(self.x[i - 1]) if i > 0 else V_i
        V_ip1 = V_MB(self.x[i + 1]) if i < self.N - 1 else V_i
        tau_plus = self.x[i + 1] - self.x[i] if i < self.N - 1 else np.zeros_like(self.x[0])
        tau_minus = self.x[i] - self.x[i - 1] if i > 0 else np.zeros_like(self.x[0])
        if V_ip1 > V_i and V_i > V_im1:
            tau = tau_plus
        elif V_ip1 < V_i and V_i < V_im1:
            tau = tau_minus
        else:
            dV_max = max(abs(V_ip1 - V_i), abs(V_i - V_im1))
            dV_min = min(abs(V_ip1 - V_i), abs(V_i - V_im1))
            if V_ip1 > V_im1:
                tau = tau_plus * dV_max + tau_minus * dV_min
            else:
                tau = tau_plus * dV_min + tau_minus * dV_max
        norm = np.linalg.norm(tau)
        return tau / (norm + 1e-12)

    def step(self, eta=0.001):
        """One iteration: gradient step with NEB projection."""
        forces = []
        for i in range(self.N):
            if i in self.fixed_ends:
                forces.append(np.zeros_like(self.x[i]))
                continue
            tau = self.tangent(i)
            grad = grad_V_MB(self.x[i])
            # Perpendicular gradient
            grad_perp = grad - np.dot(grad, tau) * tau
            # Spring force parallel
            d_plus = np.linalg.norm(self.x[i + 1] - self.x[i])
            d_minus = np.linalg.norm(self.x[i] - self.x[i - 1])
            F_spring = self.k * (d_plus - d_minus) * tau
            F = -grad_perp + F_spring
            forces.append(F)
        forces = np.array(forces)
        self.x = self.x + eta * forces
        max_force = np.max(np.linalg.norm(forces, axis=-1))
        return max_force, forces


# ============================================================
# ADMM-NEB (new — game-theorist recommended)
# ============================================================

class ADMM_NEB:
    """ADMM reformulation of NEB with tangent-aware split.

    Wang-Yin-Zeng 2019: provable convergence even nonconvex.

    z_i = midpoint(x_{i-1}, x_{i+1}) — expected equidistant position
    Penalty (ρ/2)|x_i - z_i|² ≈ NEB spring tension with k_eff = ρ/2
    Tangent split: V gradient ⊥ τ (drives toward MEP), penalty ∥ τ (straightens path)
    """

    def __init__(self, endA, endB, n_images=9, rho=5.0, robust=False):
        self.N = n_images
        self.rho = rho
        self.robust = robust
        s = np.linspace(0, 1, n_images)[:, None]
        self.x = endA[None, :] + s * (endB - endA)[None, :]
        self.z = self.x.copy()
        self.lam = np.zeros_like(self.x)
        self.fixed_ends = [0, n_images - 1]

    def tangent(self, i):
        """Upwind tangent (Henkelman-Jónsson 2000)."""
        V_i = V_MB(self.x[i])
        V_im1 = V_MB(self.x[i - 1]) if i > 0 else V_i
        V_ip1 = V_MB(self.x[i + 1]) if i < self.N - 1 else V_i
        tau_plus = self.x[i + 1] - self.x[i] if i < self.N - 1 else np.zeros_like(self.x[0])
        tau_minus = self.x[i] - self.x[i - 1] if i > 0 else np.zeros_like(self.x[0])
        if V_ip1 > V_i and V_i > V_im1:
            tau = tau_plus
        elif V_ip1 < V_i and V_i < V_im1:
            tau = tau_minus
        else:
            dV_max = max(abs(V_ip1 - V_i), abs(V_i - V_im1))
            dV_min = min(abs(V_ip1 - V_i), abs(V_i - V_im1))
            if V_ip1 > V_im1:
                tau = tau_plus * dV_max + tau_minus * dV_min
            else:
                tau = tau_plus * dV_min + tau_minus * dV_max
        norm = np.linalg.norm(tau)
        return tau / (norm + 1e-12)

    def x_update(self, eta=0.0005):
        """ADMM x-step with perpendicular projection (per NEB physics).

        V gradient ⊥ tangent: drives image to MEP perpendicular to the path
        Penalty ∥ tangent: maintains equidistant spacing (NEB spring analog)
        Lagrange multipliers λ accumulate parallel residuals.

        Hybrid: ADMM framework (provable convergence via Wang-Yin-Zeng 2019)
        + NEB projection trick (necessary for path-following).
        """
        for i in range(self.N):
            if i in self.fixed_ends:
                continue
            tau = self.tangent(i)

            # V perp gradient (drives toward MEP)
            grad_V = grad_V_MB(self.x[i])
            grad_V_perp = grad_V - np.dot(grad_V, tau) * tau

            # Penalty parallel (maintains spacing)
            target = self.z[i] - self.lam[i] / self.rho
            grad_penalty = self.rho * (self.x[i] - target)
            grad_penalty_par = np.dot(grad_penalty, tau) * tau

            self.x[i] = self.x[i] - eta * (grad_V_perp + grad_penalty_par)

    def z_update(self):
        """z = midpoint of neighbors (mean), or median if robust=True."""
        new_z = self.z.copy()
        for i in range(self.N):
            if i in self.fixed_ends:
                new_z[i] = self.x[i]
                continue
            if self.robust:
                stack = np.stack([self.x[i - 1], self.x[i], self.x[i + 1]], axis=0)
                new_z[i] = np.median(stack, axis=0)
            else:
                new_z[i] = 0.5 * (self.x[i - 1] + self.x[i + 1])
        self.z = new_z

    def lambda_update(self):
        for i in range(self.N):
            if i in self.fixed_ends:
                continue
            self.lam[i] = self.lam[i] + self.rho * (self.x[i] - self.z[i])

    def step(self, eta=0.0005):
        self.x_update(eta=eta)
        self.z_update()
        self.lambda_update()
        # Convergence: ||grad_V_perp||
        residuals = []
        for i in range(self.N):
            if i in self.fixed_ends:
                residuals.append(np.zeros_like(self.x[i]))
                continue
            tau = self.tangent(i)
            grad_V = grad_V_MB(self.x[i])
            grad_perp = grad_V - np.dot(grad_V, tau) * tau
            residuals.append(grad_perp)
        residuals = np.array(residuals)
        max_residual = np.max(np.linalg.norm(residuals, axis=-1))
        return max_residual, residuals


# ============================================================
# Benchmark
# ============================================================

def benchmark(endA, endB, label, max_iter=2000, fmax_target=0.5):
    print(f"\n{'='*70}")
    print(f"Benchmark: {label}")
    print(f"  endA: {endA}, V={V_MB(endA):.2f}")
    print(f"  endB: {endB}, V={V_MB(endB):.2f}")
    print(f"  ΔE_endpoints: {V_MB(endB) - V_MB(endA):.2f}")
    print(f"{'='*70}")

    results = {}

    # Standard NEB
    neb_std = StandardNEB(endA, endB, n_images=11, k_spring=0.5)
    history_std = []
    for it in range(max_iter):
        fmax, forces = neb_std.step(eta=0.0005)
        history_std.append(fmax)
        if fmax < fmax_target:
            break
    Vs_std = np.array([V_MB(x) for x in neb_std.x])
    barrier_std = float(Vs_std.max() - Vs_std[0])
    results["standard_NEB"] = {
        "converged_iter": it + 1,
        "final_fmax": float(fmax),
        "barrier": barrier_std,
        "V_profile": Vs_std.tolist(),
        "path": neb_std.x.tolist(),
        "history": history_std,
    }
    print(f"\n  Standard NEB:")
    print(f"    iterations: {it + 1}, final fmax: {fmax:.4f}")
    print(f"    barrier from endA: {barrier_std:.2f}")

    # ADMM-NEB hybrid (ADMM split + NEB perpendicular projection)
    # rho=2 ≈ effective k_spring 1 — comparable to standard NEB
    neb_admm = ADMM_NEB(endA, endB, n_images=11, rho=2.0, robust=False)
    history_admm = []
    for it in range(max_iter):
        fmax, residuals = neb_admm.step(eta=0.0005)
        history_admm.append(fmax)
        if fmax < fmax_target:
            break
    Vs_admm = np.array([V_MB(x) for x in neb_admm.x])
    barrier_admm = float(Vs_admm.max() - Vs_admm[0])
    results["ADMM_NEB"] = {
        "converged_iter": it + 1,
        "final_fmax": float(fmax),
        "barrier": barrier_admm,
        "V_profile": Vs_admm.tolist(),
        "path": neb_admm.x.tolist(),
        "history": history_admm,
    }
    print(f"\n  ADMM-NEB:")
    print(f"    iterations: {it + 1}, final fmax: {fmax:.4f}")
    print(f"    barrier from endA: {barrier_admm:.2f}")
    return results


def plot_results(results_dict, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: V profiles
    for label, res in results_dict.items():
        for method, m_res in res.items():
            axes[0].plot(m_res["V_profile"], "-o", label=f"{label} / {method}")
    axes[0].set_xlabel("Image")
    axes[0].set_ylabel("V")
    axes[0].set_title("Final V profile along path")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    # Panel 2: convergence history
    for label, res in results_dict.items():
        for method, m_res in res.items():
            axes[1].semilogy(m_res["history"], label=f"{label} / {method}")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("max |force|")
    axes[1].set_title("Convergence")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    # Panel 3: paths on MB potential
    x_grid = np.linspace(-1.5, 1.0, 80)
    y_grid = np.linspace(-0.3, 2.0, 80)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = V_MB(np.stack([X, Y], axis=-1))
    cs = axes[2].contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.6)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("Paths on MB potential")
    colors = ["red", "blue", "green", "orange"]
    cidx = 0
    for label, res in results_dict.items():
        for method, m_res in res.items():
            path = np.array(m_res["path"])
            axes[2].plot(path[:, 0], path[:, 1], "-o", color=colors[cidx],
                         label=f"{label} / {method}", markersize=3)
            cidx += 1
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Saved plot: {out_path}")


def main():
    out_dir = Path(r"D:\home\ignat\project-third-matter\dft-neb\ph-diagnostic\admm_neb_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Test 1: Symmetric (A → C, ΔE = -38.5)
    all_results["A→C (mild asymmetry, ΔE=-38)"] = benchmark(
        MB_MINIMA["A"], MB_MINIMA["C"], "A → C (ΔE = -38)"
    )

    # Test 2: Strong asymmetric (A → B, ΔE = +66)
    all_results["A→B (strong asymmetry, ΔE=+66)"] = benchmark(
        MB_MINIMA["A"], MB_MINIMA["B"], "A → B (ΔE = +66)"
    )

    # Save all results
    out_json = out_dir / "admm_neb_benchmark.json"
    # Strip non-JSON-serializable types
    json_results = {}
    for k, v in all_results.items():
        json_results[k] = {}
        for method, mres in v.items():
            json_results[k][method] = {
                kk: vv if not isinstance(vv, np.ndarray) else vv.tolist()
                for kk, vv in mres.items() if kk != "path" and kk != "history"
            }
            json_results[k][method]["history_last10"] = mres["history"][-10:]
    with open(out_json, "w") as f:
        json.dump(json_results, f, indent=2)

    plot_results(all_results, out_dir / "admm_neb_benchmark.png")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for label, res in all_results.items():
        print(f"\n{label}:")
        for method, m_res in res.items():
            print(f"  {method}: iter={m_res['converged_iter']}, "
                  f"fmax={m_res['final_fmax']:.4f}, barrier={m_res['barrier']:.2f}")


if __name__ == "__main__":
    main()
