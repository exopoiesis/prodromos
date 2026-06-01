"""Tests for the Bearpark-Robb MECP finder (M3.K magnetic core)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.mecp_finder import TwoSheet2D, find_mecp


def test_seam_and_analytic_mecp_consistent():
    pes = TwoSheet2D()
    p, E = pes.mecp_analytic()
    assert abs(p[0] - pes.seam_x()) < 1e-12
    assert abs(p[1]) < 1e-12          # min over y on seam -> y=0
    assert abs(pes._E_A1(p) - pes._E_B1(p)) < 1e-9  # on the seam E_A=E_B


def test_bearpark_robb_finds_analytic_mecp():
    pes = TwoSheet2D()
    p_an, E_an = pes.mecp_analytic()
    res = find_mecp(pes, x0=[pes.xa, 0.5])
    assert res.converged
    assert np.linalg.norm(res.x - p_an) < 1e-3
    assert abs(res.energy - E_an) < 1e-3
    assert abs(res.gap) < 1e-4         # gap closed -> truly on seam


def test_mecp_converges_from_either_sheet_min():
    pes = TwoSheet2D()
    p_an, _ = pes.mecp_analytic()
    for start in ([pes.xa, 0.8], [pes.xb, -0.6]):
        res = find_mecp(pes, x0=start)
        assert res.converged
        assert np.linalg.norm(res.x - p_an) < 1e-3


def test_mecp_offset_shifts_seam():
    """Larger sheet offset c -> seam moves toward sheet-A side (predictable)."""
    s_small = TwoSheet2D(c=0.5).seam_x()
    s_large = TwoSheet2D(c=2.0).seam_x()
    assert s_large > s_small   # bigger c pushes crossing toward +x (A favoured longer)
