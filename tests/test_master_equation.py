"""Tests for master_equation_kinetics.py."""
import numpy as np
import pytest

from prodromos.master_equation_kinetics import (
    KB_EV_PER_K,
    NU_HZ,
    rate_matrix,
    equilibrium_distribution,
    slowest_relaxation,
    arrhenius_fit,
    chu_liu_edmonds_arborescence,
    analyze_network,
)


class TestRateMatrix:
    def test_diagonal_negative(self, symmetric_barrier_matrix):
        """Diagonal of K should be -Σ outgoing rates (negative)."""
        K = rate_matrix(symmetric_barrier_matrix, T_K=298.0)
        assert np.all(np.diag(K) <= 0)

    def test_column_sum_zero(self, symmetric_barrier_matrix):
        """Each column of K must sum к 0 (probability conservation)."""
        K = rate_matrix(symmetric_barrier_matrix)
        col_sums = K.sum(axis=0)
        assert np.allclose(col_sums, 0, atol=1e-12)

    def test_symmetric_barriers_equal_offdiag(self, symmetric_barrier_matrix):
        """Symmetric barriers give equal off-diagonal rates."""
        K = rate_matrix(symmetric_barrier_matrix)
        # k(0→1) and k(1→0) should be equal
        assert K[1, 0] == pytest.approx(K[0, 1])

    def test_asymmetric_barriers_unequal(self, asymmetric_barrier_matrix):
        """Asymmetric barriers give unequal rates."""
        K = rate_matrix(asymmetric_barrier_matrix)
        # Forward (higher barrier) should give smaller k than reverse
        assert K[1, 0] < K[0, 1]  # K[j,i] = k(i→j)

    def test_inf_barrier_zero_rate(self):
        """Infinite barrier gives zero rate."""
        E_a = np.array([[np.inf, np.inf], [np.inf, np.inf]])
        K = rate_matrix(E_a)
        # All zeros
        assert np.allclose(K, 0)

    def test_higher_T_increases_rates(self, symmetric_barrier_matrix):
        """Increasing T should give larger rates."""
        K_300 = rate_matrix(symmetric_barrier_matrix, T_K=300)
        K_400 = rate_matrix(symmetric_barrier_matrix, T_K=400)
        # Off-diagonal rates larger at higher T
        assert K_400[0, 1] > K_300[0, 1]


class TestEquilibriumDistribution:
    def test_normalized(self, symmetric_barrier_matrix):
        K = rate_matrix(symmetric_barrier_matrix)
        P = equilibrium_distribution(K)
        assert P.sum() == pytest.approx(1.0)

    def test_non_negative(self, asymmetric_barrier_matrix):
        K = rate_matrix(asymmetric_barrier_matrix)
        P = equilibrium_distribution(K)
        assert np.all(P >= 0)

    def test_symmetric_equipopulated(self, symmetric_barrier_matrix):
        """Symmetric 2-state → 50/50."""
        K = rate_matrix(symmetric_barrier_matrix)
        P = equilibrium_distribution(K)
        assert P[0] == pytest.approx(P[1], rel=1e-3)

    def test_asymmetric_favors_low_E(self, asymmetric_barrier_matrix):
        """Asymmetric с ΔE > 0 → low-E state dominates."""
        K = rate_matrix(asymmetric_barrier_matrix, T_K=298)
        P = equilibrium_distribution(K)
        # Site 0 is low (forward barrier 250 > reverse 76 → ΔE = +174)
        # Boltzmann: P(0)/P(1) = exp(174/25) ≈ 1000 → P(0) ≈ 0.999
        assert P[0] > 0.99


class TestSlowestRelaxation:
    def test_positive_timescale(self, symmetric_barrier_matrix):
        K = rate_matrix(symmetric_barrier_matrix)
        tau, _ = slowest_relaxation(K)
        assert tau > 0

    def test_higher_barrier_slower(self):
        """Higher barrier → longer relaxation time."""
        E_low = np.full((2, 2), np.inf)
        E_low[0, 1] = E_low[1, 0] = 0.050
        E_high = np.full((2, 2), np.inf)
        E_high[0, 1] = E_high[1, 0] = 0.500

        K_low = rate_matrix(E_low)
        K_high = rate_matrix(E_high)
        tau_low, _ = slowest_relaxation(K_low)
        tau_high, _ = slowest_relaxation(K_high)
        assert tau_high > tau_low


class TestArrheniusFit:
    def test_symmetric_recovers_barrier(self, symmetric_barrier_matrix):
        """For symmetric 43 meV barrier, Arrhenius should give 43 meV."""
        result = arrhenius_fit(symmetric_barrier_matrix)
        assert result["E_a_eff_meV"] == pytest.approx(43.0, abs=2.0)

    def test_asymmetric_returns_reverse(self, asymmetric_barrier_matrix):
        """Asymmetric с reverse 76 meV: slowest mode = reverse (depopulation of high-E)."""
        result = arrhenius_fit(asymmetric_barrier_matrix)
        # Reverse barrier dominates relaxation timescale
        assert result["E_a_eff_meV"] == pytest.approx(76.0, abs=5.0)


class TestChuLiuEdmonds:
    def test_returns_n_minus_1_edges(self):
        """Arborescence on n nodes has n-1 edges."""
        W = np.array([
            [np.inf, 1.0, 2.0],
            [3.0, np.inf, 1.5],
            [2.5, 1.2, np.inf],
        ])
        edges = chu_liu_edmonds_arborescence(W, root=0)
        # Tree on 3 nodes = 2 edges
        assert len(edges) == 2

    def test_root_no_parent(self):
        """Root must have no incoming edge."""
        W = np.array([
            [np.inf, 1.0, 2.0],
            [3.0, np.inf, 1.5],
            [2.5, 1.2, np.inf],
        ])
        edges = chu_liu_edmonds_arborescence(W, root=0)
        for parent, child in edges:
            assert child != 0  # root has no parent

    def test_picks_minimum_weight(self):
        """Each non-root picks the cheapest incoming edge."""
        # Node 1: incoming from 0 (cost 1.0) or from 2 (cost 1.2). Min = 1.0.
        # Node 2: incoming from 0 (cost 2.0) or from 1 (cost 1.5). Min = 1.5.
        W = np.array([
            [np.inf, 1.0, 2.0],
            [3.0, np.inf, 1.5],
            [2.5, 1.2, np.inf],
        ])
        edges = chu_liu_edmonds_arborescence(W, root=0)
        edge_set = set(edges)
        assert (0, 1) in edge_set  # 0→1 cost 1.0
        assert (1, 2) in edge_set  # 1→2 cost 1.5


class TestAnalyzeNetwork:
    def test_returns_required_keys(self, symmetric_barrier_matrix):
        result = analyze_network(symmetric_barrier_matrix, verbose=False)
        required = {
            "n_sites", "site_labels", "T_K", "equilibrium_distribution",
            "tau_slow_s", "arrhenius_E_a_eff_meV", "min_arborescence_edges",
        }
        assert required.issubset(result.keys())

    def test_symmetric_arrhenius_matches(self, symmetric_barrier_matrix):
        """End-to-end: symmetric input → E_a = 43 meV."""
        result = analyze_network(symmetric_barrier_matrix, verbose=False)
        assert result["arrhenius_E_a_eff_meV"] == pytest.approx(43.0, abs=2.0)

    def test_three_state_chain(self, three_state_chain):
        """3-state chain: dominant pathway through arborescence."""
        result = analyze_network(three_state_chain, verbose=False)
        # All states should be reachable
        assert result["n_sites"] == 3
        assert len(result["min_arborescence_edges"]) == 2
