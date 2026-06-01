"""Unit tests for the spin_split detector + two-sheet toy (M3.K).

Verifies the consilium-designed magnetic-mismatch gate:
  * marc-like crossover -> spin_split role + SHEET_CROSSING + ENDPOINT_SPLIT
  * pyrite-like geometric stuck -> `stuck` role, NO false sheet-crossing
  * clean single sheet -> all ok (negative control)
  * discriminator: low geom force + high spin incoherence => spin_split;
                   high geom force + smooth mag        => stuck
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.spin_split_detector import (
    TwoSheetPES, magnetic_band_diagnostic,
    band_marc_like, band_pyrite_like, band_clean,
    DELTA_ABS_ADJ,
)


# ---- toy PES sanity ----

def test_sheets_cross_in_marc_toy():
    pes = TwoSheetPES()
    xs = pes.seam_x()
    assert 0.0 < xs < 1.0, "marc-like toy must have a sheet crossing in (0,1)"


def test_blind_surrogate_dips_below_true():
    """band-collapse: spin-blind surrogate goes BELOW the min-envelope at seam."""
    pes = TwoSheetPES()
    xs = pes.seam_x()
    assert pes.V_blind(xs) < pes.V_true(xs) - 1e-6


def test_no_crossing_when_HS_far_above():
    pes = TwoSheetPES(dE=2.0, tilt=0.0)
    assert np.isnan(pes.seam_x()), "single-sheet toy must have no crossing"


# ---- detector: the three cases ----

def test_marc_like_flags_spin_split_and_crossing():
    _, _, mt, ma, gf, E = band_marc_like()
    d = magnetic_band_diagnostic(mt, ma, gf, E)
    assert "spin_split" in d.roles
    assert d.sheet_crossing is True
    assert d.endpoint_split is True
    assert "stuck" not in d.roles, "magnetic problem must NOT be labelled geometric stuck"
    assert d.recommendation.startswith(("RUN_DFT", "TWO_SEGMENT"))


def test_pyrite_like_is_geometric_stuck_no_false_crossing():
    _, _, mt, ma, gf, E = band_pyrite_like()
    d = magnetic_band_diagnostic(mt, ma, gf, E)
    assert "stuck" in d.roles
    assert "spin_split" not in d.roles, "single-sheet geom problem must not be spin_split"
    assert d.sheet_crossing is False, "no magnetic crossing in single-sheet case"
    assert d.endpoint_split is False
    assert d.recommendation.startswith("OK_SINGLE_SHEET")


def test_clean_band_no_flags():
    _, _, mt, ma, gf, E = band_clean()
    d = magnetic_band_diagnostic(mt, ma, gf, E)
    assert set(d.roles) == {"ok"}
    assert not d.flags
    assert d.sheet_crossing is False


# ---- discriminator unit logic (game-theorist) ----

def test_discriminator_low_force_high_mismatch_is_spin_split():
    n = 5
    mag_total = np.array([1.13, 1.13, 1.67, 1.67, 1.67])  # jump at edge 1->2
    mag_abs = mag_total.copy()
    geom_fmax = np.full(n, 0.02)                            # all geometrically converged
    energies = np.array([0.0, 0.1, 0.2, 0.1, 0.0])
    d = magnetic_band_diagnostic(mag_total, mag_abs, geom_fmax, energies)
    # images adjacent to the jump (1 and 2) must be spin_split
    assert d.roles[1] == "spin_split" or d.roles[2] == "spin_split"
    assert "stuck" not in d.roles


def test_discriminator_high_force_smooth_mag_is_stuck():
    n = 5
    mag_total = np.full(n, 1.13)                            # smooth, no jump
    mag_abs = mag_total.copy()
    geom_fmax = np.array([0.0, 0.02, 0.7, 0.02, 0.0])       # image 2 stuck
    energies = np.array([0.0, 0.1, 0.25, 0.1, 0.0])
    d = magnetic_band_diagnostic(mag_total, mag_abs, geom_fmax, energies)
    assert d.roles[2] == "stuck"
    assert "spin_split" not in d.roles
    assert d.sheet_crossing is False


def test_endpoint_gate_triggers_on_total_mag_split():
    n = 7
    mag_total = np.linspace(1.13, 1.67, n)                 # gradual but endpoints differ
    mag_abs = mag_total.copy()
    geom_fmax = np.full(n, 0.02)
    energies = np.zeros(n)
    d = magnetic_band_diagnostic(mag_total, mag_abs, geom_fmax, energies)
    assert d.endpoint_split is True   # |1.67-1.13|=0.54 > 0.3


def test_threshold_below_delta_no_crossing():
    """A magnetization wobble below threshold must not trip SHEET_CROSSING."""
    n = 5
    mag_total = np.array([1.13, 1.20, 1.15, 1.18, 1.13])   # max Δadj ~0.07 < 0.5
    mag_abs = mag_total.copy()
    geom_fmax = np.full(n, 0.02)
    energies = np.zeros(n)
    d = magnetic_band_diagnostic(mag_total, mag_abs, geom_fmax, energies)
    assert d.sheet_crossing is False
    assert max(d.d_abs_adj) < DELTA_ABS_ADJ
