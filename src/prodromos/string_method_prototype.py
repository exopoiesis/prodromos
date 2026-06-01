"""
Vanden-Eijnden Simplified String Method prototype (J Chem Phys 126:164103, 2007).

Key difference vs NEB:
- NO spring force
- Spacing maintained by EXPLICIT reparametrization (cubic spline + arclength resample)
- Each iteration:
  1. Gradient descent perpendicular to tangent per image
  2. Reparametrize string by arclength → uniform distribution

Mathematical advantage (per math consilium):
- Decouples right axis: tangent fixed by geometry, normal by force
- Provably converges to MEP (Vanden-Eijnden & Ren 2007 theorem)
- No spring constant tuning
- Robust к asymmetric endpoints

Test: same Müller-Brown asymmetric cases where ADMM-NEB failed:
- A→C (ΔE = -38): mild asymmetry
- A→B (ΔE = +66): strong asymmetry
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Import MB potential from companion prototype
sys.path.insert(0, str(Path(__file__).parent))
from prodromos.admm_neb_prototype import V_MB, grad_V_MB, MB_MINIMA, StandardNEB


class StringMethod:
    """Vanden-Eijnden simplified string method.

    Iteration:
    1. ⊥ gradient descent per image
    2. Cubic spline interpolation
    3. Resample at uniform arclength
    """

    def __init__(self, endA, endB, n_images=11):
        self.N = n_images
        s = np.linspace(0, 1, n_images)[:, None]
        self.x = endA[None, :] + s * (endB - endA)[None, :]
        self.fixed_ends = [0, n_images - 1]

    def tangent(self, i):
        """Finite-difference tangent (3-point central)."""
        if i == 0:
            tau = self.x[1] - self.x[0]
        elif i == self.N - 1:
            tau = self.x[-1] - self.x[-2]
        else:
            tau = self.x[i + 1] - self.x[i - 1]
        return tau / (np.linalg.norm(tau) + 1e-12)

    def gradient_step(self, eta=0.0005):
        """Step 1: each interior image moves ⊥ to local tangent."""
        new_x = self.x.copy()
        for i in range(self.N):
            if i in self.fixed_ends:
                continue
            tau = self.tangent(i)
            grad = grad_V_MB(self.x[i])
            grad_perp = grad - np.dot(grad, tau) * tau
            new_x[i] = self.x[i] - eta * grad_perp
        self.x = new_x

    def reparametrize(self):
        """Step 2: cubic spline interpolation + uniform arclength resample.

        This is the KEY string method step that replaces NEB's spring force.
        """
        # Compute cumulative arclength
        diffs = np.diff(self.x, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        L_total = s[-1]
        if L_total < 1e-9:
            return  # degenerate path
        s_norm = s / L_total  # normalize к [0, 1]

        # Uniform arclength target
        s_uniform = np.linspace(0, 1, self.N)

        # Interpolate per dimension с cubic spline
        new_x = np.zeros_like(self.x)
        for d in range(self.x.shape[1]):
            cs = CubicSpline(s_norm, self.x[:, d])
            new_x[:, d] = cs(s_uniform)

        # Endpoints stay fixed (reparametrization should not move them, but enforce)
        new_x[0] = self.x[0]
        new_x[-1] = self.x[-1]

        self.x = new_x

    def step(self, eta=0.0005):
        self.gradient_step(eta=eta)
        self.reparametrize()

        # Convergence diagnostic: max ⊥ gradient
        forces = []
        for i in range(self.N):
            if i in self.fixed_ends:
                forces.append(np.zeros_like(self.x[i]))
                continue
            tau = self.tangent(i)
            grad = grad_V_MB(self.x[i])
            grad_perp = grad - np.dot(grad, tau) * tau
            forces.append(grad_perp)
        forces = np.array(forces)
        max_force = float(np.max(np.linalg.norm(forces, axis=-1)))
        return max_force, forces


def benchmark(endA, endB, label, max_iter=2000, fmax_target=0.5):
    print(f"\n{'='*70}")
    print(f"Benchmark: {label}")
    print(f"  endA: V={V_MB(endA):.2f}, endB: V={V_MB(endB):.2f}")
    print(f"  ΔE_endpoints: {V_MB(endB) - V_MB(endA):.2f}")
    print(f"{'='*70}")

    results = {}

    # Standard NEB (reference)
    neb_std = StandardNEB(endA, endB, n_images=11, k_spring=0.5)
    history_std = []
    for it in range(max_iter):
        fmax, _ = neb_std.step(eta=0.0005)
        history_std.append(fmax)
        if fmax < fmax_target:
            break
    Vs_std = np.array([V_MB(x) for x in neb_std.x])
    barrier_std = float(Vs_std.max() - Vs_std[0])
    results["standard_NEB"] = {
        "iter": it + 1, "fmax": fmax, "barrier": barrier_std,
        "V_profile": Vs_std.tolist(), "path": neb_std.x.tolist(),
        "history": history_std,
    }
    print(f"\n  Standard NEB: iter={it+1}, fmax={fmax:.4f}, barrier={barrier_std:.2f}")

    # String method
    string = StringMethod(endA, endB, n_images=11)
    history_str = []
    for it in range(max_iter):
        fmax, _ = string.step(eta=0.0005)
        history_str.append(fmax)
        if fmax < fmax_target:
            break
    Vs_str = np.array([V_MB(x) for x in string.x])
    barrier_str = float(Vs_str.max() - Vs_str[0])
    results["string_method"] = {
        "iter": it + 1, "fmax": fmax, "barrier": barrier_str,
        "V_profile": Vs_str.tolist(), "path": string.x.tolist(),
        "history": history_str,
    }
    print(f"\n  String method: iter={it+1}, fmax={fmax:.4f}, barrier={barrier_str:.2f}")
    return results


def plot_results(results_dict, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # V profiles
    for label, res in results_dict.items():
        for method, m_res in res.items():
            axes[0].plot(m_res["V_profile"], "-o", label=f"{label} / {method}", markersize=4)
    axes[0].set_xlabel("Image")
    axes[0].set_ylabel("V")
    axes[0].set_title("Final V profile along path")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    # Convergence
    for label, res in results_dict.items():
        for method, m_res in res.items():
            axes[1].semilogy(m_res["history"], label=f"{label} / {method}")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("max |∇V_⊥|")
    axes[1].set_title("Convergence")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    # Paths on MB
    x_grid = np.linspace(-1.5, 1.0, 80)
    y_grid = np.linspace(-0.3, 2.0, 80)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = V_MB(np.stack([X, Y], axis=-1))
    axes[2].contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.5)
    colors = ["red", "blue", "green", "orange"]
    cidx = 0
    for label, res in results_dict.items():
        for method, m_res in res.items():
            path = np.array(m_res["path"])
            axes[2].plot(path[:, 0], path[:, 1], "-o", color=colors[cidx],
                         label=f"{label} / {method}", markersize=4)
            cidx += 1
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("Paths on MB potential")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Saved plot: {out_path}")


def main():
    out_dir = Path(r"D:\home\ignat\project-third-matter\dft-neb\ph-diagnostic\string_method_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_results["A→C (ΔE=-38)"] = benchmark(MB_MINIMA["A"], MB_MINIMA["C"], "A → C")
    all_results["A→B (ΔE=+66)"] = benchmark(MB_MINIMA["A"], MB_MINIMA["B"], "A → B")

    # Save
    out_json = out_dir / "string_method_benchmark.json"
    json_results = {}
    for k, v in all_results.items():
        json_results[k] = {}
        for method, m in v.items():
            json_results[k][method] = {
                kk: vv if not isinstance(vv, np.ndarray) else vv.tolist()
                for kk, vv in m.items() if kk not in ("path", "history")
            }
            json_results[k][method]["history_last10"] = m["history"][-10:]
    with open(out_json, "w") as f:
        json.dump(json_results, f, indent=2)

    plot_results(all_results, out_dir / "string_method_benchmark.png")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for label, res in all_results.items():
        print(f"\n{label}:")
        for method, m in res.items():
            print(f"  {method:<20s}: iter={m['iter']}, fmax={m['fmax']:.4f}, barrier={m['barrier']:.2f}")


if __name__ == "__main__":
    main()
