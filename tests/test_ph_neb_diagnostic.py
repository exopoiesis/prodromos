"""Tests for ph_neb_diagnostic.py (PH library)."""
import numpy as np
import pytest

from prodromos.ph_neb_diagnostic import (
    persistence_from_grid,
    persistence_from_cloud,
    bottleneck,
    wasserstein,
    same_basin_score,
    summary,
)


class TestPersistenceFromGrid:
    def test_returns_array_with_2_columns(self, simple_2d_grid):
        """Persistence diagram should be N×2 (birth, death pairs)."""
        diag = persistence_from_grid(simple_2d_grid)
        assert diag.ndim == 2
        assert diag.shape[1] == 2

    def test_single_minimum_has_one_dominant_feature(self, simple_2d_grid):
        """Single-minimum potential should have 1 essential class."""
        diag = persistence_from_grid(simple_2d_grid)
        # At least one feature
        assert len(diag) >= 1

    def test_essential_class_capped(self, simple_2d_grid):
        """Infinite-death feature must be replaced by cap_at value."""
        diag = persistence_from_grid(simple_2d_grid, cap_at=10.0)
        # No infinity in result
        assert np.all(np.isfinite(diag))

    def test_double_well_distinguishable(self, simple_2d_grid, double_well_grid):
        """Different topology → different persistence."""
        diag1 = persistence_from_grid(simple_2d_grid, cap_at=10.0)
        diag2 = persistence_from_grid(double_well_grid, cap_at=10.0)
        d = bottleneck(diag1, diag2)
        # Should be non-trivially different
        assert d > 0.01


class TestBottleneck:
    def test_self_distance_zero(self, simple_2d_grid):
        """d(A, A) ≈ 0 (gudhi may return tiny numerical noise like 2e-308)."""
        diag = persistence_from_grid(simple_2d_grid)
        assert bottleneck(diag, diag) == pytest.approx(0.0, abs=1e-10)

    def test_empty_diagrams(self):
        """Bottleneck of two empty diagrams = 0."""
        empty = np.empty((0, 2))
        assert bottleneck(empty, empty) == 0.0

    def test_symmetric(self, simple_2d_grid, double_well_grid):
        """d(A, B) = d(B, A)."""
        d1 = persistence_from_grid(simple_2d_grid)
        d2 = persistence_from_grid(double_well_grid)
        assert bottleneck(d1, d2) == bottleneck(d2, d1)


class TestWasserstein:
    def test_self_distance_zero(self, simple_2d_grid):
        diag = persistence_from_grid(simple_2d_grid)
        assert wasserstein(diag, diag) == 0.0

    def test_empty_diagrams(self):
        empty = np.empty((0, 2))
        assert wasserstein(empty, empty) == 0.0

    def test_distinct_diagrams_nonzero(self, simple_2d_grid, double_well_grid):
        d1 = persistence_from_grid(simple_2d_grid, cap_at=10.0)
        d2 = persistence_from_grid(double_well_grid, cap_at=10.0)
        assert wasserstein(d1, d2) > 0


class TestPersistenceFromCloud:
    def test_returns_array(self, random_cloud_2d):
        points, values = random_cloud_2d
        diag = persistence_from_cloud(points, values, max_edge_length=1.5)
        assert isinstance(diag, np.ndarray)

    def test_persistence_threshold_filters(self, random_cloud_2d):
        """With high persistence threshold, fewer features remain."""
        points, values = random_cloud_2d
        diag_no_filter = persistence_from_cloud(
            points, values, max_edge_length=1.5, persistence_threshold=0.0
        )
        diag_filtered = persistence_from_cloud(
            points, values, max_edge_length=1.5, persistence_threshold=0.5
        )
        assert len(diag_filtered) <= len(diag_no_filter)


class TestSameBasinScore:
    def test_identical_score_one(self, simple_2d_grid):
        """Identical diagrams should give score = 1."""
        diag = persistence_from_grid(simple_2d_grid)
        score = same_basin_score(diag, diag, tau=1.0)
        assert score == pytest.approx(1.0)

    def test_score_in_unit_interval(self, simple_2d_grid, double_well_grid):
        d1 = persistence_from_grid(simple_2d_grid)
        d2 = persistence_from_grid(double_well_grid)
        score = same_basin_score(d1, d2)
        assert 0 <= score <= 1


class TestSummary:
    def test_empty_diagram(self):
        empty = np.empty((0, 2))
        s = summary(empty)
        assert s["n_features"] == 0
        assert s["max_persistence"] == 0.0

    def test_keys(self, simple_2d_grid):
        diag = persistence_from_grid(simple_2d_grid)
        s = summary(diag)
        required = {"n_features", "max_persistence", "total_persistence"}
        assert required.issubset(s.keys())
