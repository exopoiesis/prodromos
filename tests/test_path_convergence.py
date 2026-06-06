"""Tests for parametrization-invariant convergence (consilium 2026-05-29)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.admm_neb_prototype import V_MB, MB_MINIMA
from prodromos.neb_agm_prototype import NEBAGM, run_method
from prodromos.path_convergence import (
    resample_uniform, path_change,
    monitor_invariance_gate,
)

A, B = MB_MINIMA["A"], MB_MINIMA["B"]


def test_resample_shape_and_endpoints():
    x = np.array([[0., 0.], [1., 1.], [2., 0.]])
    pts, tang = resample_uniform(x, M=50)
    assert pts.shape == (50, 2)
    assert np.allclose(pts[0], x[0]) and np.allclose(pts[-1], x[-1])
    assert np.allclose(np.linalg.norm(tang, axis=1), 1.0, atol=1e-6)


def test_path_change_zero_for_identical():
    x = np.array([[0., 0.], [1., 0.5], [2., 0.]])
    m, mx = path_change(x, x, M=40)
    assert m < 1e-9 and mx < 1e-9


def test_node_fmax_false_convergence_caught_by_invariant_residual():
    """HEADLINE (consilium): node-fmax can report convergence (≈0) while the
    curve is NOT on the MEP — the invariant residual exposes it."""
    agm = NEBAGM(A, B, n_images=11, climb=False, adaptive_reparam=False)
    run_method(agm, 1500, 0.5)            # converges by node-fmax
    node_fmax = float(agm.gperp_norm.max())
    sup, rms = agm.invariant_residual(M=200)
    assert node_fmax < 0.5, "uniform band should reach node-fmax convergence"
    assert sup > 3.0, "invariant residual must reveal the curve is NOT on the MEP"
    # i.e. node-fmax FALSE-converged; the invariant metric did not.


def test_invariant_residual_ranks_path_quality():
    """Adaptive (nodes clustered at saddle) -> lower invariant residual than uniform."""
    a_uni = NEBAGM(A, B, n_images=11, climb=False, adaptive_reparam=False)
    run_method(a_uni, 1500, 0.5)
    sup_uni, _ = a_uni.invariant_residual()
    a_ada = NEBAGM(A, B, n_images=11, climb=False, adaptive_reparam=True)
    run_method(a_ada, 1500, 0.5)
    sup_ada, _ = a_ada.invariant_residual()
    assert sup_ada < sup_uni, f"adaptive residual {sup_ada:.1f} should beat uniform {sup_uni:.1f}"


def test_dense_spline_barrier_is_monitor_invariant():
    """Game-theorist gate: barrier measured on the dense spline AGREES across
    monitor functions (it is gauge-invariant), unlike the node-max barrier."""
    def make_runner(_climb):
        def run(beta):
            agm = NEBAGM(A, B, n_images=11, climb=True,
                         adaptive_reparam=(beta > 0), adaptive_beta=max(beta, 0.1))
            run_method(agm, 1500, 0.5)
            return agm.x
        return run
    monitors = {"uniform": 0.0, "adaptive2": 2.0, "adaptive4": 4.0}
    g = monitor_invariance_gate(make_runner(True), monitors, V_MB, tol_barrier=0.5)
    assert g["invariant"], f"dense-spline barrier must be monitor-invariant, spread={g['spread']:.2f}"


def test_invariant_criterion_triggers_where_nodefmax_does_not():
    """The gauge-invariant criterion converges on adaptive-reparam A->B (where
    node-fmax never settled in 2000 iters)."""
    agm = NEBAGM(A, B, n_images=11, climb=True, adaptive_reparam=True, reparam_every=3)
    r = run_method(agm, 2000, 0.5, criterion="invariant")
    assert r["converged_invariant"], "invariant criterion must trigger"
    assert r["iter"] < 400, f"should converge fast, got {r['iter']}"


def test_dense_barrier_method_independent():
    """Gauge-invariant (dense-spline) barrier agrees across std-NEB / string /
    NEB-AGM on the same problem — the barrier is a property of the PES, not the
    method/parametrization."""
    from admm_neb_prototype import StandardNEB
    from string_method_prototype import StringMethod
    b_neb = run_method(StandardNEB(A, B, n_images=11, k_spring=0.5), 2000, 0.5)["dense_barrier"]
    b_str = run_method(StringMethod(A, B, n_images=11), 2000, 0.5)["dense_barrier"]
    agm = NEBAGM(A, B, n_images=11, climb=True, adaptive_reparam=True, reparam_every=3)
    b_agm = run_method(agm, 2000, 0.5, criterion="invariant")["dense_barrier"]
    assert max(b_neb, b_str, b_agm) - min(b_neb, b_str, b_agm) < 0.5, \
        f"dense barriers must agree: NEB={b_neb:.2f} str={b_str:.2f} AGM={b_agm:.2f}"


def test_dense_barrier_more_accurate_than_node_max_for_uniform():
    """Dense-spline barrier recovers the true saddle even when nodes don't sit on
    it (node-max underestimates)."""
    agm = NEBAGM(A, B, n_images=11, climb=False, adaptive_reparam=False)
    run_method(agm, 1500, 0.5)
    # compare saddle height: dense spline vs node-max
    pts, _ = resample_uniform(agm.x, 200)
    dense_saddle = max(V_MB(p) for p in pts)
    node_saddle = max(V_MB(x) for x in agm.x)
    assert dense_saddle >= node_saddle - 1e-6, "dense spline must capture saddle at least as high as node-max"
