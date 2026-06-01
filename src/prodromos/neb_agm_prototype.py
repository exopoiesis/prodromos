"""
NEB-AGM prototype — NEB as Adaptive Game with Memory.

Phase 3 of ROADMAP_NEB_GT_THEORY.md (tasks M3.A + M3.C + M3.D + M3.F),
with the post-s156 corrections:
  * BACKBONE = string method (Vanden-Eijnden 2007), NOT ADMM (retracted via
    tri-consilium — category error, see admm_neb_results/TRI_CONSILIUM_VERDICT).
    Inter-image communication = arclength reparametrization (no spring tuning).
  * Per-image MEMORY (M3.A): each image keeps its own curvature estimate from
    successive (x, g_perp) history -> adaptive per-image step size.
  * Role classifier (M3.C): basin / ridge / climber / stuck.
  * Role-conditional updates (M3.D): the *stuck* + *ridge* roles directly encode
    the lesson from the pyrite/marcasite diagnostic
    (NEB_STALL_DIAGNOSTIC_PLAYBOOK.md): a band image that sits on a sharp ridge
    with weak control "rolls off" sideways -> here ridge images take *careful*
    steps and stuck images get an *escape* kick instead of thrashing.
  * Climbing image = Stackelberg leader on the single highest-V interior image
    (ascend along tangent), as in CI-NEB (FOUNDATIONS Part 6.2).

Spec: GAME_THEORETIC_NEB_FOUNDATIONS.md Part 7.

Benchmark (M3.F): standard NEB vs plain string vs NEB-AGM on Muller-Brown,
including the asymmetric A->B case where standard NEB needs 472 iters and the
plain string needs 33.

This is a TOY (analytic MB) prototype to test decision gates G3.A / G3.B before
any DFT spend. NOT a production optimizer.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from prodromos.admm_neb_prototype import V_MB, grad_V_MB, MB_MINIMA, StandardNEB
from prodromos.string_method_prototype import StringMethod
from prodromos.spin_split_detector import magnetic_band_diagnostic


@dataclass
class SameBasinStatus:
    """Geometric same-basin halt report for role-aware NEB."""
    reason: str
    barrier: float
    endpoint_delta: float
    path_length: float
    basin_like: bool
    curvature_ready: bool
    roles: list[str]


class NEBAGM:
    """String backbone + per-image memory + role-conditional updates.

    Roles:
      basin   : convex perpendicular curvature, low residual -> take a larger step
      ridge   : negative perpendicular curvature -> take a *careful* step
                (prevents the "rolls off the ridge" failure seen in pyrite DFT)
      climber : single highest-V interior image -> ascend along tangent (CI)
      stuck   : perpendicular force has stalled over a window -> escape kick
    """

    ROLES = ("basin", "ridge", "climber", "stuck")

    def __init__(self, endA, endB, n_images=11, base_eta=5e-4,
                 stuck_window=8, stuck_ratio=0.90, climb=True,
                 magmom_provider=None, adaptive_reparam=False, adaptive_beta=2.0,
                 reparam_every=1, same_basin_gate=True,
                 same_basin_burnin=3, same_basin_barrier=1.0,
                 same_basin_endpoint_delta=2.0,
                 same_basin_path_tol=0.20,
                 same_basin_curv_tol=1e-6):
        self.N = n_images
        s = np.linspace(0, 1, n_images)[:, None]
        self.x = endA[None, :] + s * (endB - endA)[None, :]
        self.fixed_ends = [0, n_images - 1]
        self.base_eta = base_eta
        self.climb = climb
        self.stuck_window = stuck_window
        self.stuck_ratio = stuck_ratio
        self.adaptive_reparam = adaptive_reparam   # M3.E
        self.adaptive_beta = adaptive_beta
        self.reparam_every = reparam_every         # inner/outer: reparam every N steps
        self._prev_x = None                        # for path_change (invariant metric)
        self._barrier_hist = []                    # dense-spline barrier history (invariant)
        # M3.K magnetic gate: optional callback (positions array -> (mag_total, mag_abs)).
        # When set, step() runs the spin_split detector and HALTS on a magnetic
        # discontinuity (detect+report, NOT geometric escape — consilium consensus).
        self.magmom_provider = magmom_provider
        self.halt = False
        self.halt_reason = None
        self.spin_status = None      # BandDiagnostic on halt
        self.same_basin_gate = same_basin_gate
        self.same_basin_burnin = same_basin_burnin
        self.same_basin_barrier = same_basin_barrier
        self.same_basin_endpoint_delta = same_basin_endpoint_delta
        self.same_basin_path_tol = same_basin_path_tol
        self.same_basin_curv_tol = same_basin_curv_tol
        self.same_basin_status = None
        self.gperp_norm = np.zeros(n_images)  # per-image perpendicular force (for gate)

        d = self.x.shape[1]
        # per-image MEMORY
        self.prev_x = [None] * n_images          # last position
        self.prev_gperp = [None] * n_images       # last perpendicular gradient
        self.curv = np.full(n_images, np.nan)     # smoothed perpendicular curvature
        self.gperp_hist = [[] for _ in range(n_images)]  # |g_perp| history (for stuck)
        self.roles = ["basin"] * n_images
        self.iter = 0

    # ---- geometry ----
    def tangent(self, i):
        if i == 0:
            tau = self.x[1] - self.x[0]
        elif i == self.N - 1:
            tau = self.x[-1] - self.x[-2]
        else:
            tau = self.x[i + 1] - self.x[i - 1]
        return tau / (np.linalg.norm(tau) + 1e-12)

    def _perp(self, g, tau):
        return g - np.dot(g, tau) * tau

    # ---- per-image memory: curvature estimate from (x, g_perp) history ----
    def _update_curvature(self, i, x_i, gperp_i):
        if self.prev_x[i] is not None:
            dx = x_i - self.prev_x[i]
            dg = gperp_i - self.prev_gperp[i]
            ndx2 = float(dx @ dx)
            if ndx2 > 1e-14:
                c = float(dg @ dx) / ndx2     # directional 2nd derivative (Rayleigh-like)
                if np.isnan(self.curv[i]):
                    self.curv[i] = c
                else:
                    self.curv[i] = 0.5 * self.curv[i] + 0.5 * c   # EMA memory
        self.prev_x[i] = x_i.copy()
        self.prev_gperp[i] = gperp_i.copy()

    # ---- role classifier (M3.C) ----
    def _classify(self, i, Vs, gperp_norm):
        # climber: single highest-V interior image (Stackelberg leader)
        interior = [j for j in range(self.N) if j not in self.fixed_ends]
        i_climb = interior[int(np.argmax([Vs[j] for j in interior]))]
        if self.climb and i == i_climb:
            return "climber"
        # stuck: |g_perp| not decreasing over window
        h = self.gperp_hist[i]
        if len(h) >= self.stuck_window:
            w = h[-self.stuck_window:]
            if min(w) / (max(w) + 1e-12) > self.stuck_ratio and gperp_norm > 1e-3:
                return "stuck"
        # ridge vs basin by perpendicular curvature sign
        c = self.curv[i]
        if not np.isnan(c) and c < 0:
            return "ridge"
        return "basin"

    # ---- role-conditional step magnitude (M3.D) ----
    def _eta_for_role(self, role, i):
        eta = self.base_eta
        c = self.curv[i]
        if role == "basin":
            # curvature-adaptive (memory): flatter -> bigger step, capped
            if not np.isnan(c) and c > 1e-6:
                eta = self.base_eta * float(np.clip(2.0 / (c + 1.0), 0.5, 4.0))
            else:
                eta = self.base_eta * 1.5
        elif role == "ridge":
            eta = self.base_eta * 0.5      # CAREFUL: do not roll off the ridge
        elif role == "climber":
            eta = self.base_eta * 1.0
        elif role == "stuck":
            eta = self.base_eta * 3.0      # escape: bigger move
        return eta

    def gradient_step(self):
        Vs = np.array([V_MB(self.x[i]) for i in range(self.N)])
        new_x = self.x.copy()
        roles = list(self.roles)
        for i in range(self.N):
            if i in self.fixed_ends:
                continue
            tau = self.tangent(i)
            g = grad_V_MB(self.x[i])
            g_par = np.dot(g, tau)
            g_perp = g - g_par * tau
            gpn = float(np.linalg.norm(g_perp))
            self.gperp_hist[i].append(gpn)
            self._update_curvature(i, self.x[i], g_perp)

            role = self._classify(i, Vs, gpn)
            roles[i] = role
            eta = self._eta_for_role(role, i)

            if role == "climber":
                # CI: invert parallel component (ascend) + descend perpendicular
                F = -g_perp + g_par * tau
                new_x[i] = self.x[i] + eta * F
            elif role == "stuck":
                # escape: perpendicular descent + small kick toward lower neighbour
                lo = i - 1 if Vs[i - 1] < Vs[i + 1] else i + 1
                kick = self.x[lo] - self.x[i]
                kick_perp = kick - np.dot(kick, tau) * tau
                kick_perp /= (np.linalg.norm(kick_perp) + 1e-12)
                new_x[i] = self.x[i] - eta * g_perp + 0.3 * eta * gpn * kick_perp
            else:  # basin / ridge
                new_x[i] = self.x[i] - eta * g_perp
        self.x = new_x
        self.roles = roles

    def _monitor(self):
        """M3.E inter-image communication: a monitor function w_i, high where the
        path needs more resolution (near the saddle/ridge). Built from per-image
        perpendicular force + a neighbour-smoothing pass (the 'communication':
        each node's weight is averaged with its neighbours so role/uncertainty
        info propagates along the band)."""
        gp = self.gperp_norm.copy()
        scale = gp.max() if gp.max() > 1e-9 else 1.0
        w = 1.0 + self.adaptive_beta * (gp / scale)   # normalised -> w in [1, 1+beta]
        # neighbour smoothing (3-point) = inter-image communication
        ws = w.copy()
        for i in range(1, self.N - 1):
            ws[i] = 0.25 * w[i - 1] + 0.5 * w[i] + 0.25 * w[i + 1]
        return np.clip(ws, 1.0, 1.0 + self.adaptive_beta)  # bounded -> no node collision

    def reparametrize(self):
        # string backbone: cubic-spline resample. Uniform arclength by default;
        # M3.E adaptive = monitor-weighted node density (denser near saddle).
        from scipy.interpolate import CubicSpline
        diffs = np.diff(self.x, axis=0)
        seg = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 1e-9:
            return
        s_norm = s / s[-1]

        if self.adaptive_reparam:
            # weighted arclength: cumulative ∫ w ds -> equidistribute W (Monitor
            # function string method; nodes cluster where w is large).
            w = self._monitor()
            wmid = 0.5 * (w[:-1] + w[1:])
            dW = wmid * np.diff(s_norm)
            W = np.concatenate([[0.0], np.cumsum(dW)])
            if W[-1] < 1e-12:
                s_targets = np.linspace(0, 1, self.N)
            else:
                W /= W[-1]
                W = np.maximum.accumulate(W + np.arange(self.N) * 1e-9)  # strictly increasing
                W_targets = np.linspace(0, 1, self.N)
                s_targets = np.interp(W_targets, W, s_norm)
                s_targets = np.maximum.accumulate(s_targets)            # monotone guard
                s_targets[0], s_targets[-1] = 0.0, 1.0
        else:
            s_targets = np.linspace(0, 1, self.N)
        if not np.all(np.isfinite(s_targets)):
            s_targets = np.linspace(0, 1, self.N)

        new_x = np.zeros_like(self.x)
        for d in range(self.x.shape[1]):
            new_x[:, d] = CubicSpline(s_norm, self.x[:, d])(s_targets)
        new_x[0] = self.x[0]
        new_x[-1] = self.x[-1]
        self.x = new_x

    def step(self):
        self.iter += 1
        self._prev_x = self.x.copy()      # for invariant path_change metric
        self.gradient_step()
        # inner/outer (consilium): reparam only every N steps, not every step
        # (reparam-every-step injects continuous gauge noise -> node-fmax never settles)
        if self.iter % self.reparam_every == 0:
            clim = [i for i, r in enumerate(self.roles) if r == "climber"]
            x_clim = {i: self.x[i].copy() for i in clim}
            self.reparametrize()
            for i, xc in x_clim.items():
                self.x[i] = xc  # pin climber after reparam

        # convergence diagnostic: per-image perpendicular gradient
        self.gperp_norm = np.zeros(self.N)
        for i in range(self.N):
            if i in self.fixed_ends:
                continue
            tau = self.tangent(i)
            g = grad_V_MB(self.x[i])
            g_perp = g - np.dot(g, tau) * tau
            self.gperp_norm[i] = float(np.linalg.norm(g_perp))
        fmax = float(self.gperp_norm.max())

        # M3.K magnetic gate (guard, runs only if a magmom provider is wired)
        if self.magmom_provider is not None:
            self._magnetic_gate()
        if not self.halt:
            self._same_basin_gate()
        return fmax

    def _magnetic_gate(self):
        """Run spin_split detector on the current band; HALT on a magnetic
        discontinuity. Detect + report, never geometric-escape (consilium)."""
        mag_total, mag_abs = self.magmom_provider(self.x)
        energies = np.array([V_MB(self.x[i]) for i in range(self.N)])
        diag = magnetic_band_diagnostic(mag_total, mag_abs, self.gperp_norm, energies)
        if diag.sheet_crossing or diag.endpoint_split:
            self.halt = True
            self.halt_reason = "spin_split"
            self.spin_status = diag

    def _same_basin_gate(self):
        """Abort a geometrically flat same-basin band.

        Implements the guard proposed in ALTERNATIVES_AND_ROLE_AWARE_NEB.md:
        if the band is flat and all images look basin-like, the result is a
        same-basin artifact candidate, not a meaningful low barrier.
        """
        if (not self.same_basin_gate) or self.iter < self.same_basin_burnin:
            return

        from path_convergence import barrier_on_curve

        Vs = np.array([V_MB(self.x[i]) for i in range(self.N)])
        barrier = float(barrier_on_curve(self.x, V_MB))
        endpoint_delta = float(abs(Vs[-1] - Vs[0]))
        path_length = float(np.linalg.norm(np.diff(self.x, axis=0), axis=1).sum())

        interior = [i for i in range(self.N) if i not in self.fixed_ends]
        interior_roles = [self.roles[i] for i in interior]
        role_basin_like = all(r in ("basin", "climber") for r in interior_roles)

        c = self.curv[interior]
        finite = np.isfinite(c)
        curvature_ready = bool(finite.sum() >= max(1, len(interior) - 1))
        curvature_basin_like = (not curvature_ready) or bool(
            np.nanmin(c[finite]) >= -self.same_basin_curv_tol
        )
        basin_like = bool(role_basin_like and curvature_basin_like)

        flat_profile = (
            barrier <= self.same_basin_barrier
            and endpoint_delta <= self.same_basin_endpoint_delta
        )
        tiny_path = path_length <= self.same_basin_path_tol

        reason = None
        if flat_profile and tiny_path:
            reason = "tiny_path_low_barrier"
        elif flat_profile and basin_like and curvature_ready:
            reason = "flat_convex_band"
        if reason is None:
            return

        self.halt = True
        self.halt_reason = "same_basin"
        self.same_basin_status = SameBasinStatus(
            reason=reason,
            barrier=barrier,
            endpoint_delta=endpoint_delta,
            path_length=path_length,
            basin_like=basin_like,
            curvature_ready=curvature_ready,
            roles=interior_roles,
        )

    def role_counts(self):
        from collections import Counter
        return dict(Counter(self.roles[i] for i in range(self.N) if i not in self.fixed_ends))

    # --- parametrization-invariant convergence (consilium 2026-05-29) ---
    def invariant_residual(self, M=200):
        """sup |∇V_⊥| on a dense uniform resample of the curve (gauge-invariant
        MEP residual). Replaces node-fmax as the PRIMARY convergence metric."""
        from path_convergence import perp_residual_on_curve
        sup, rms = perp_residual_on_curve(self.x, grad_V_MB, M=M)
        return sup, rms

    def path_motion(self, M=200):
        """Mean curve displacement vs previous iteration on common arclength."""
        if self._prev_x is None:
            return np.inf, np.inf
        from path_convergence import path_change
        return path_change(self._prev_x, self.x, M=M)

    def dense_barrier(self, M=200):
        """Gauge-invariant barrier: max V on the dense spline (NOT node-max)."""
        from path_convergence import barrier_on_curve
        return barrier_on_curve(self.x, V_MB, M=M)

    def record_barrier(self, M=200):
        self._barrier_hist.append(self.dense_barrier(M=M))

    def invariant_converged(self, tol_path=2e-3, tol_E=0.05, window=12, M=120):
        """Gauge-invariant convergence (consilium):
          (1) curve stationary: mean path_change < tol_path  (necessary)
          (2) dense-spline barrier stable within tol_E over `window`  (confirm)
        The sup-residual has an interpolation FLOOR at fixed N (so it is reported
        as a quality metric, not an absolute gate — see RESULTS_CONVERGENCE_CONSILIUM)."""
        mean_mot, _ = self.path_motion(M=M)
        if mean_mot >= tol_path:
            return False
        if len(self._barrier_hist) < window:
            return False
        recent = self._barrier_hist[-window:]
        return (max(recent) - min(recent)) < tol_E


def run_method(method, max_iter, fmax_target, criterion="node"):
    """criterion: 'node' (max ⊥-force at nodes < fmax_target, legacy) or
    'invariant' (NEB-AGM only: gauge-invariant path-stationarity + dense-barrier
    stability — consilium 2026-05-29)."""
    from path_convergence import barrier_on_curve
    hist = []
    it = 0
    conv_iter = None
    for it in range(max_iter):
        if isinstance(method, NEBAGM):
            fmax = method.step()
            method.record_barrier()
            if method.halt:        # M3.K magnetic gate fired
                break
            if criterion == "invariant" and it > 3 and method.invariant_converged():
                conv_iter = it + 1
                break
        else:
            fmax, _ = method.step(eta=5e-4)
        hist.append(fmax)
        if criterion == "node" and fmax < fmax_target:
            break
    # gauge-invariant dense-spline barrier (NOT node-max) for fair comparison
    dense_bar = barrier_on_curve(method.x, V_MB)
    Vs = np.array([V_MB(x) for x in method.x])
    return {
        "iter": (conv_iter if conv_iter else it + 1), "fmax": float(fmax),
        "barrier": float(Vs.max() - Vs[0]),          # legacy node-max (gauge-dep)
        "dense_barrier": float(dense_bar),            # gauge-invariant
        "converged_invariant": conv_iter is not None,
        "V_profile": Vs.tolist(), "path": method.x.tolist(), "history": hist,
    }


def benchmark(endA, endB, label, max_iter=2000, fmax_target=0.5, n_images=11):
    print(f"\n{'='*70}\nBenchmark: {label}")
    print(f"  endA V={V_MB(endA):.2f}  endB V={V_MB(endB):.2f}  dE={V_MB(endB)-V_MB(endA):.2f}\n{'='*70}")
    res = {}
    res["standard_NEB"] = run_method(StandardNEB(endA, endB, n_images=n_images, k_spring=0.5), max_iter, fmax_target)
    res["string_method"] = run_method(StringMethod(endA, endB, n_images=n_images), max_iter, fmax_target)
    agm = NEBAGM(endA, endB, n_images=n_images, climb=True,
                 adaptive_reparam=True, reparam_every=3)
    res["NEB_AGM"] = run_method(agm, max_iter, fmax_target, criterion="invariant")
    res["NEB_AGM"]["final_roles"] = agm.role_counts()
    for m, r in res.items():
        extra = f"  roles={r.get('final_roles')}" if "final_roles" in r else ""
        ic = "  inv_conv" if r.get("converged_invariant") else ""
        # report gauge-invariant dense-spline barrier (consilium), node-max in ()
        print(f"  {m:<16s}: iter={r['iter']:>4d}  dense_barrier={r['dense_barrier']:.2f}"
              f"  (node-max {r['barrier']:.2f}){ic}{extra}")
    return res


def plot(results, out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    x = np.linspace(-1.5, 1.0, 80); y = np.linspace(-0.3, 2.0, 80)
    X, Y = np.meshgrid(x, y); Z = V_MB(np.stack([X, Y], -1))
    colors = {"standard_NEB": "red", "string_method": "blue", "NEB_AGM": "green"}
    for label, res in results.items():
        for m, r in res.items():
            ax[0].plot(r["V_profile"], "-o", ms=3, label=f"{label}/{m}")
            ax[1].semilogy(r["history"], color=colors.get(m), label=f"{label}/{m}")
    ax[0].set(title="Final V profile", xlabel="image", ylabel="V"); ax[0].legend(fontsize=6); ax[0].grid(alpha=.3)
    ax[1].set(title="Convergence", xlabel="iter", ylabel="max|grad_perp|"); ax[1].legend(fontsize=6); ax[1].grid(alpha=.3)
    ax[2].contour(X, Y, Z, levels=20, cmap="viridis", alpha=.5)
    lastlabel = list(results.keys())[-1]
    for m, r in results[lastlabel].items():
        p = np.array(r["path"]); ax[2].plot(p[:, 0], p[:, 1], "-o", ms=3, color=colors.get(m), label=m)
    ax[2].set(title=f"Paths ({lastlabel})", xlabel="x", ylabel="y"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    print(f"  saved {out}")


def main():
    out_dir = Path(__file__).parent / "neb_agm_results"
    out_dir.mkdir(exist_ok=True)
    allr = {}
    allr["A->C (dE=-38)"] = benchmark(MB_MINIMA["A"], MB_MINIMA["C"], "A -> C (mild asym)")
    allr["A->B (dE=+66)"] = benchmark(MB_MINIMA["A"], MB_MINIMA["B"], "A -> B (strong asym)")

    js = {}
    for k, v in allr.items():
        js[k] = {}
        for m, r in v.items():
            js[k][m] = {kk: vv for kk, vv in r.items() if kk not in ("path", "history")}
            js[k][m]["history_last10"] = r["history"][-10:]
    json.dump(js, open(out_dir / "neb_agm_benchmark.json", "w"), indent=2)
    plot(allr, out_dir / "neb_agm_benchmark.png")

    print(f"\n{'='*70}\nSUMMARY (iters to fmax<0.5)\n{'='*70}")
    print(f"{'case':<18}{'std NEB':>10}{'string':>10}{'NEB-AGM':>10}")
    for k, v in allr.items():
        print(f"{k:<18}{v['standard_NEB']['iter']:>10}{v['string_method']['iter']:>10}{v['NEB_AGM']['iter']:>10}")


if __name__ == "__main__":
    main()
