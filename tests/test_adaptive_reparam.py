"""Tests for M3.E adaptive (monitor-weighted) reparametrization in NEB-AGM."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.admm_neb_prototype import V_MB, MB_MINIMA
from prodromos.neb_agm_prototype import NEBAGM, run_method

A, B = MB_MINIMA["A"], MB_MINIMA["B"]
TRUE_BARRIER = 106.02


def test_monitor_bounded():
    agm = NEBAGM(A, B, n_images=11, adaptive_reparam=True, adaptive_beta=2.0)
    for _ in range(5):
        agm.step()
    w = agm._monitor()
    assert w.min() >= 1.0 - 1e-9
    assert w.max() <= 1.0 + agm.adaptive_beta + 1e-9


def test_adaptive_reparam_stays_finite():
    """Robustness: adaptive reparam + CI must not produce NaN/inf nodes (the
    earlier CubicSpline crash regression)."""
    agm = NEBAGM(A, B, n_images=11, climb=True, adaptive_reparam=True)
    for _ in range(60):
        agm.step()
        assert np.all(np.isfinite(agm.x)), "adaptive reparam produced non-finite nodes"


def test_nodes_strictly_ordered_after_adaptive():
    agm = NEBAGM(A, B, n_images=11, adaptive_reparam=True)
    for _ in range(30):
        agm.step()
    # cumulative arclength must be strictly increasing (no collided nodes)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(agm.x, axis=0), axis=1))]
    assert np.all(np.diff(s) > 1e-6)


def test_adaptive_improves_barrier_without_ci():
    """Headline M3.E: without climbing image, monitor-weighted node density gives
    a more accurate barrier (nodes cluster at the saddle)."""
    a_uni = NEBAGM(A, B, n_images=11, climb=False, adaptive_reparam=False)
    run_method(a_uni, 1500, 0.5)
    bar_uni = max(V_MB(x) for x in a_uni.x) - V_MB(a_uni.x[0])

    a_ada = NEBAGM(A, B, n_images=11, climb=False, adaptive_reparam=True)
    run_method(a_ada, 1500, 0.5)
    bar_ada = max(V_MB(x) for x in a_ada.x) - V_MB(a_ada.x[0])

    assert abs(bar_ada - TRUE_BARRIER) < abs(bar_uni - TRUE_BARRIER), \
        f"adaptive barrier err {abs(bar_ada-TRUE_BARRIER):.2f} should beat uniform {abs(bar_uni-TRUE_BARRIER):.2f}"
