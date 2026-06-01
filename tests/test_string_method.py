"""Tests for string_method_prototype.py (Vanden-Eijnden 2007)."""
import numpy as np
import pytest

from prodromos.string_method_prototype import StringMethod
from prodromos.admm_neb_prototype import V_MB, MB_MINIMA


class TestStringMethodInit:
    def test_endpoints_set(self):
        """Endpoints должны быть at correct positions."""
        endA = MB_MINIMA["A"]
        endB = MB_MINIMA["B"]
        sm = StringMethod(endA, endB, n_images=9)
        np.testing.assert_allclose(sm.x[0], endA)
        np.testing.assert_allclose(sm.x[-1], endB)

    def test_n_images_correct(self):
        sm = StringMethod(MB_MINIMA["A"], MB_MINIMA["B"], n_images=7)
        assert sm.x.shape[0] == 7
        assert sm.N == 7

    def test_initial_path_linear(self):
        """Initial path = linear interpolation."""
        endA = np.array([0.0, 0.0])
        endB = np.array([1.0, 0.0])
        sm = StringMethod(endA, endB, n_images=5)
        # Image 2 (middle) should be at (0.5, 0)
        np.testing.assert_allclose(sm.x[2], [0.5, 0.0])


class TestReparametrize:
    def test_endpoints_preserved(self):
        """Reparametrization не должна сдвигать endpoints."""
        endA = MB_MINIMA["A"]
        endB = MB_MINIMA["B"]
        sm = StringMethod(endA, endB, n_images=9)
        x_before_A = sm.x[0].copy()
        x_before_B = sm.x[-1].copy()
        sm.reparametrize()
        np.testing.assert_allclose(sm.x[0], x_before_A)
        np.testing.assert_allclose(sm.x[-1], x_before_B)

    def test_equidistant_after_reparam(self):
        """After reparametrization, segments should be (nearly) equal length."""
        # Start с non-equidistant path
        endA = np.array([0.0, 0.0])
        endB = np.array([1.0, 0.0])
        sm = StringMethod(endA, endB, n_images=5)
        # Perturb middle image
        sm.x[2] = np.array([0.3, 0.0])  # off-center
        sm.reparametrize()

        diffs = np.diff(sm.x, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        # All segments should be approximately equal length
        assert seg_lengths.std() / seg_lengths.mean() < 0.05


class TestStringMethodConvergence:
    @pytest.mark.slow
    def test_converges_on_mb_symmetric(self):
        """String method should converge на MB A→C (mild asymmetry)."""
        endA = MB_MINIMA["A"]
        endC = MB_MINIMA["C"]
        sm = StringMethod(endA, endC, n_images=11)
        max_iter = 500
        converged = False
        for i in range(max_iter):
            fmax, _ = sm.step(eta=0.0005)
            if fmax < 1.0:
                converged = True
                break
        assert converged, f"String method не сходится за {max_iter} итераций"

    @pytest.mark.slow
    def test_recovers_mb_barrier(self):
        """A→C barrier should be ~106 (known true value)."""
        endA = MB_MINIMA["A"]
        endC = MB_MINIMA["C"]
        sm = StringMethod(endA, endC, n_images=11)
        for _ in range(500):
            fmax, _ = sm.step(eta=0.0005)
            if fmax < 1.0:
                break
        Vs = np.array([V_MB(x) for x in sm.x])
        barrier = Vs.max() - Vs[0]
        # True MB A→C barrier ≈ 106
        assert barrier == pytest.approx(106, abs=10)


class TestTangent:
    def test_tangent_normalized(self):
        """Tangent vector should be unit length."""
        sm = StringMethod(MB_MINIMA["A"], MB_MINIMA["B"], n_images=9)
        tau = sm.tangent(4)
        assert np.linalg.norm(tau) == pytest.approx(1.0)
