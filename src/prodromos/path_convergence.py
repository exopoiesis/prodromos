"""
Parametrization-invariant convergence for moving-node string / NEB-AGM.

Consilium verdict (2026-05-29, math+CS+statmech+game-theorist, UNANIMOUS):
node-fmax (max ⊥-force at NODES) is GAUGE-DEPENDENT — under adaptive
reparametrization the nodes relocate, so it has no invariant limit and never
settles, even on a converged MEP. The MEP is a curve (equivalence class mod
reparametrization); convergence must be measured on parametrization-INVARIANT
quantities of the curve, NOT on node coordinates.

Primary (invariant) criteria implemented here:
  * perp_residual_on_curve  — sup/L2 of |∇V_⊥| on a DENSE UNIFORM resample of the
    spline (the discretised MEP condition ∇V_⊥=0 on the continuous curve).
  * path_change             — L2/Hausdorff between successive curves on a common
    uniform arclength grid (gauge-invariant "did the curve move").
  * barrier                 — CONFIRM only (1st-order stationary at saddle ->
    masks non-convergence; never use alone — all 4 experts).
  * monitor_invariance_gate — game-theorist: a converged MEP barrier must be
    INVARIANT to the monitor function. If the barrier depends on the monitor,
    the path is not on the MEP (monitor is masking off-saddle error).

All metrics reuse node gradients (zero extra eval on DFT, M3.G) except the dense
perp residual which, on an analytic toy, evaluates grad on the resample; on DFT
one interpolates node forces (CS note).
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline


def resample_uniform(x, M=200):
    """Cubic-spline resample of node set x (N,d) to M points uniform in arclength.
    Returns (pts (M,d), tangents (M,d) unit)."""
    x = np.asarray(x, float)
    seg = np.linalg.norm(np.diff(x, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        t = np.zeros_like(x[:1]).repeat(M, 0)
        return np.repeat(x[:1], M, 0), t
    s_norm = s / s[-1]
    su = np.linspace(0, 1, M)
    pts = np.zeros((M, x.shape[1]))
    der = np.zeros((M, x.shape[1]))
    for d in range(x.shape[1]):
        cs = CubicSpline(s_norm, x[:, d])
        pts[:, d] = cs(su)
        der[:, d] = cs(su, 1)
    norms = np.linalg.norm(der, axis=1, keepdims=True)
    tang = der / np.maximum(norms, 1e-12)
    return pts, tang


def perp_residual_on_curve(x, grad_fn, M=200):
    """Invariant MEP residual: |∇V_⊥| on a dense uniform resample of the curve.
    Returns (sup, rms)."""
    pts, tang = resample_uniform(x, M)
    perp = np.empty(M)
    for i in range(M):
        g = grad_fn(pts[i])
        gp = g - np.dot(g, tang[i]) * tang[i]
        perp[i] = np.linalg.norm(gp)
    # interior only (endpoints are fixed minima, tangent ill-defined there)
    interior = perp[1:-1]
    return float(interior.max()), float(np.sqrt(np.mean(interior ** 2)))


def path_change(x_prev, x_curr, M=200):
    """Gauge-invariant curve motion: mean & max ||γ_curr − γ_prev|| on a common
    uniform arclength grid."""
    a, _ = resample_uniform(x_prev, M)
    b, _ = resample_uniform(x_curr, M)
    d = np.linalg.norm(b - a, axis=1)
    return float(d.mean()), float(d.max())


def barrier_on_curve(x, V_fn, M=200):
    pts, _ = resample_uniform(x, M)
    V = np.array([V_fn(p) for p in pts])
    return float(V.max() - max(V[0], V[-1]))


def monitor_invariance_gate(make_runner, monitors, V_fn, tol_barrier=0.5):
    """Game-theorist adversarial gate: run the SAME problem with >=2 different
    monitor functions; a true MEP barrier must agree across them.

    make_runner(monitor) -> relaxed images (np array of nodes).
    Returns dict with per-monitor barrier + invariant flag.
    """
    bars = {}
    for name, mon in monitors.items():
        imgs = make_runner(mon)
        bars[name] = barrier_on_curve(imgs, V_fn)
    vals = list(bars.values())
    spread = max(vals) - min(vals)
    return {"barriers": bars, "spread": spread, "invariant": spread < tol_barrier}
