"""Tests for M3.B per-image surrogates (quadratic vs GP)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.admm_neb_prototype import MB_MINIMA
from prodromos.per_image_surrogate import QuadraticSurrogate, GPSurrogate, compare


def test_quadratic_recovers_known_gradient():
    """Fit an exact quadratic -> predict_grad must match analytic g+H(x-x0)."""
    rng = np.random.default_rng(0)
    g_true = np.array([1.5, -2.0])
    H = np.array([[3.0, 0.5], [0.5, 4.0]])
    x0 = np.array([0.2, -0.1])
    X = x0 + 0.3 * rng.standard_normal((30, 2))
    d = X - x0
    y = 1.0 + d @ g_true + 0.5 * np.einsum("ni,ij,nj->n", d, H, d)
    s = QuadraticSurrogate().fit(X, y)
    q = np.array([0.4, 0.0])
    g_pred = s.predict_grad(q)
    g_ana = g_true + H @ (q - x0)
    assert np.linalg.norm(g_pred - g_ana) < 1e-6


def test_gp_gradient_on_linear():
    """GP on a linear field -> gradient ~ the slope."""
    rng = np.random.default_rng(1)
    slope = np.array([2.0, -1.0])
    X = rng.uniform(-1, 1, (40, 2))
    y = X @ slope
    g = GPSurrogate(length_scale=0.5).fit(X, y).predict_grad(np.array([0.0, 0.0]))
    assert np.linalg.norm(g - slope) < 0.5


def test_compare_returns_both_errors():
    A = MB_MINIMA["A"]
    _, out = compare(A, A + np.array([0.05, 0.03]), seeds=range(2), n_points=(8, 16))
    assert set(out.keys()) == {8, 16}
    for n, (qe, ge) in out.items():
        assert qe >= 0 and ge >= 0


def test_gp_beats_quadratic_in_anharmonic_region():
    """Headline M3.B claim: with enough history, GP < quadratic error on a
    steep (anharmonic) MB region."""
    A = MB_MINIMA["A"]
    _, out = compare(A, A + np.array([0.10, 0.06]), seeds=range(3), n_points=(24,))
    qe, ge = out[24]
    assert ge < qe, f"GP ({ge:.2f}) should beat quadratic ({qe:.2f}) at n=24 in anharmonic region"
    assert ge < 15.0, "GP error should be small with 24 points"
