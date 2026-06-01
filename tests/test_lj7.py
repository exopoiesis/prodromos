"""Fast tests for LJ7 stress test (P1-B harder). Avoids the slow minima search."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.lj7_stress_test import (
    lj_energy, lj_grad, fingerprint, random_rotation,
    cartesian_interp_barrier,
)


def test_lj_pair_minimum():
    """Two atoms at r=2^(1/6) -> LJ pair minimum E=-1."""
    r = 2.0 ** (1.0 / 6.0)
    flat = np.array([0, 0, 0, r, 0, 0], float)
    assert abs(lj_energy(flat) - (-1.0)) < 1e-9


def test_lj_grad_matches_numerical():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(21) * 1.2
    g = lj_grad(x)
    eps = 1e-6
    for i in range(0, 21, 7):  # sample a few components
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        num = (lj_energy(xp) - lj_energy(xm)) / (2 * eps)
        assert abs(num - g[i]) < 1e-3


def test_fingerprint_rotation_invariant():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(21)
    R = random_rotation(5)
    xr = (x.reshape(-1, 3) @ R.T).ravel()
    assert np.linalg.norm(fingerprint(x) - fingerprint(xr)) < 1e-9


def test_fingerprint_permutation_invariant():
    rng = np.random.default_rng(3)
    p = rng.standard_normal((7, 3))
    perm = rng.permutation(7)
    a = fingerprint(p.ravel())
    b = fingerprint(p[perm].ravel())
    assert np.linalg.norm(a - b) < 1e-9


def test_cartesian_false_barrier_but_fingerprint_zero():
    """Same structure rotated: Cartesian interp-barrier is huge (FALSE), internal
    fingerprint is ~0 (correct same-basin)."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(21) * 1.3
    R = random_rotation(9)
    xr = (x.reshape(-1, 3) @ R.T).ravel()
    cart = cartesian_interp_barrier(x, xr)
    fp = np.linalg.norm(fingerprint(x) - fingerprint(xr))
    assert fp < 1e-9, "internal fingerprint must see identical structure"
    assert cart > 5.0, "naive Cartesian must show a (false) barrier from misalignment"
