"""Tests for P1-B synthetic same-basin stress test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.admm_neb_prototype import MB_MINIMA
from prodromos.synthetic_stress_test import interp_barrier, build_cases, evaluate


def test_same_basin_interp_barrier_near_zero():
    A = MB_MINIMA["A"]
    assert interp_barrier(A, A + 0.1) < 1.0


def test_cross_basin_interp_barrier_large():
    A, B = MB_MINIMA["A"], MB_MINIMA["B"]
    assert interp_barrier(A, B) > 10.0


def test_cases_have_both_classes():
    cases = build_cases()
    assert any(c["same_basin"] for c in cases)
    assert any(not c["same_basin"] for c in cases)


def test_perfect_separation_on_controlled_set():
    rows, cm = evaluate()
    assert cm["FP"] == 0 and cm["FN"] == 0, "L2 interp-barrier must classify controlled cases perfectly"
    same = [r["interp_barrier"] for r in rows if r["truth"]]
    diff = [r["interp_barrier"] for r in rows if not r["truth"]]
    assert max(same) < min(diff), "same-basin and true-MEP barriers must not overlap"


def test_ph_score_directionally_correct():
    rows, _ = evaluate()
    import numpy as np
    sscore = np.mean([r["ph_score"] for r in rows if r["truth"]])
    dscore = np.mean([r["ph_score"] for r in rows if not r["truth"]])
    assert sscore > dscore, "same-basin PH score should exceed true-MEP on average"
