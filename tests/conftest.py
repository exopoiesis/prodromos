"""Shared pytest fixtures."""
import numpy as np
import pytest



@pytest.fixture
def simple_2d_grid():
    """Simple 2D potential V(x,y) = x² + y² (single minimum at origin)."""
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    V = X**2 + Y**2
    return V


@pytest.fixture
def double_well_grid():
    """Double-well potential V(x,y) = (x²-1)² + y² (2 minima at ±1)."""
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    V = (X**2 - 1)**2 + Y**2
    return V


@pytest.fixture
def random_cloud_2d():
    """Random Gaussian cloud in 2D with known V values."""
    rng = np.random.default_rng(42)
    n = 100
    points = rng.normal(0, 1, (n, 2))
    values = np.sum(points**2, axis=1)  # V = x² + y²
    return points, values


@pytest.fixture
def symmetric_barrier_matrix():
    """Symmetric 2-state barrier matrix (43 meV both ways).

    State 0 ↔ State 1, equal forward and reverse.
    """
    E_a = np.full((2, 2), np.inf)
    E_a[0, 1] = 0.043  # 43 meV
    E_a[1, 0] = 0.043
    return E_a


@pytest.fixture
def asymmetric_barrier_matrix():
    """Asymmetric 2-state with ΔE > 0."""
    E_a = np.full((2, 2), np.inf)
    E_a[0, 1] = 0.250
    E_a[1, 0] = 0.076  # rev = forward - ΔE = 250 - 174 = 76 meV
    return E_a


@pytest.fixture
def three_state_chain():
    """3-state linear chain barrier matrix.

    0 --100--> 1 --150--> 2
       <-100--   <-150--
    """
    E_a = np.full((3, 3), np.inf)
    E_a[0, 1] = E_a[1, 0] = 0.100
    E_a[1, 2] = E_a[2, 1] = 0.150
    return E_a
