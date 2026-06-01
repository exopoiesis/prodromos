"""
Persistent Homology diagnostic for NEB same-basin detection.

Task P0-A from ROADMAP_NEB_GT_THEORY.md.
Foundation: GAME_THEORETIC_NEB_FOUNDATIONS.md §2.

Idea: given an ensemble of points (x_i, V_i) sampled around a candidate
basin or image, compute persistence diagram via sublevel filtration of V.
Compare diagrams between ensembles via bottleneck distance — small distance
implies topologically similar local environments, i.e., same basin.

API:
    diag = persistence_from_grid(V_grid_2d)        # sublevel filtration on grid
    diag = persistence_from_cloud(points, values)  # for irregular ensemble (lower-star)
    d = bottleneck(diag1, diag2)                    # topological distance
    s = same_basin_score(diag1, diag2, tau)         # [0,1] convenience score
"""

from __future__ import annotations
import numpy as np
import gudhi as gd


def persistence_from_grid(values_2d: np.ndarray, dim: int = 0,
                           cap_at: float | None = None) -> np.ndarray:
    """
    Compute persistence diagram from sublevel filtration on a regular 2D grid.

    Returns persistence pairs (birth, death) in given homological dimension.
    Essential classes (infinite death) are CAPPED at max(V) by default — this
    captures the depth of the deepest basin, which is the most informative
    topological feature.

    Args:
        values_2d: function values on regular grid
        dim: homological dimension (0 = components, 1 = loops, ...)
        cap_at: optional cap for infinite death; if None, uses max(values_2d)
    """
    cc = gd.CubicalComplex(top_dimensional_cells=values_2d)
    cc.compute_persistence()
    pers = cc.persistence_intervals_in_dimension(dim)
    if len(pers) == 0:
        return np.empty((0, 2))
    cap = float(values_2d.max()) if cap_at is None else float(cap_at)
    pers_cap = pers.copy()
    inf_mask = ~np.isfinite(pers_cap[:, 1])
    pers_cap[inf_mask, 1] = cap
    return pers_cap


def persistence_from_cloud(points: np.ndarray, values: np.ndarray, dim: int = 0,
                            max_edge_length: float = 1.0,
                            cap_at: float | None = None,
                            persistence_threshold: float = 0.0) -> np.ndarray:
    """
    Compute persistence diagram from a point cloud with function values.

    Uses Rips complex with lower-star filtration: each simplex's filtration
    value is the maximum of its vertices' function values.

    Args:
        points: (N, d) coordinates
        values: (N,) function values
        dim: homological dimension
        max_edge_length: Rips parameter (only edges shorter than this)
        cap_at: cap for infinite-death classes; default = max(values)
        persistence_threshold: drop features with (death-birth) < threshold
            (denoising; removes spurious features from random sampling)
    """
    rips = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max(2, dim + 1))

    for v in range(len(points)):
        st.assign_filtration([v], float(values[v]))
    st.make_filtration_non_decreasing()

    st.compute_persistence()
    pers = st.persistence_intervals_in_dimension(dim)
    if len(pers) == 0:
        return np.empty((0, 2))

    cap = float(values.max()) if cap_at is None else float(cap_at)
    pers_cap = pers.copy()
    inf_mask = ~np.isfinite(pers_cap[:, 1])
    pers_cap[inf_mask, 1] = cap

    if persistence_threshold > 0.0:
        keep = (pers_cap[:, 1] - pers_cap[:, 0]) >= persistence_threshold
        pers_cap = pers_cap[keep]

    return pers_cap


def bottleneck(diag1: np.ndarray, diag2: np.ndarray) -> float:
    """Bottleneck distance between two persistence diagrams (L∞-flavored)."""
    if len(diag1) == 0 and len(diag2) == 0:
        return 0.0
    return float(gd.bottleneck_distance(diag1, diag2))


def wasserstein(diag1: np.ndarray, diag2: np.ndarray, order: float = 2.0) -> float:
    """
    Wasserstein-p distance between persistence diagrams.

    More robust to outliers than bottleneck. order=2 is standard for PH.
    Requires `gudhi.wasserstein` module.
    """
    if len(diag1) == 0 and len(diag2) == 0:
        return 0.0
    from gudhi.wasserstein import wasserstein_distance
    return float(wasserstein_distance(diag1, diag2, order=order))


def same_basin_score(diag1: np.ndarray, diag2: np.ndarray, tau: float = 10.0) -> float:
    """
    Heuristic same-basin probability ∈ [0,1].

    tau: characteristic energy scale (units of V).
    Small bottleneck distance → high score → topologically similar → same basin.
    """
    d = bottleneck(diag1, diag2)
    return float(np.exp(-d / tau))


def summary(diag: np.ndarray) -> dict:
    """Summary statistics of a persistence diagram for debug."""
    if len(diag) == 0:
        return {"n_features": 0, "max_persistence": 0.0, "total_persistence": 0.0}
    persistences = diag[:, 1] - diag[:, 0]
    return {
        "n_features": int(len(diag)),
        "max_persistence": float(persistences.max()),
        "total_persistence": float(persistences.sum()),
        "min_birth": float(diag[:, 0].min()),
        "max_death": float(diag[:, 1].max()),
    }
