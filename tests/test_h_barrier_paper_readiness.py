"""Tests for N-15 h_barrier_paper_readiness gate."""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from prodromos.h_barrier_paper_readiness import (
    run_h_barrier_paper_readiness,
    DEFAULT_H_FRACTION_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_kwargs(**overrides):
    """A complete paper-grade set of parameters, with optional overrides."""
    base = dict(
        barrier_eV=0.43,
        has_dft_freq=True,
        n_imag_modes=1,
        imag_mode_H_fraction=0.97,
        dZPE_eV=-0.12,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: PAPER_GRADE
# ---------------------------------------------------------------------------

def test_full_data_is_paper_grade():
    env = run_h_barrier_paper_readiness(**_full_kwargs())
    assert env["verdict"] == "PAPER_GRADE"
    assert env["confidence"] == "high"
    r = env["result"]
    assert r["criteria_met"] == 4
    assert r["missing"] == []


def test_paper_grade_barrier_with_zpe_computed():
    env = run_h_barrier_paper_readiness(**_full_kwargs(barrier_eV=0.43, dZPE_eV=-0.12))
    r = env["result"]
    assert r["barrier_with_zpe"] is not None
    assert abs(r["barrier_with_zpe"] - (0.43 - 0.12)) < 1e-9


def test_paper_grade_barrier_with_zpe_arithmetic():
    """Explicit check: 0.250 + (-0.130) = 0.120 eV."""
    env = run_h_barrier_paper_readiness(
        barrier_eV=0.250,
        has_dft_freq=True,
        n_imag_modes=1,
        imag_mode_H_fraction=0.80,
        dZPE_eV=-0.130,
    )
    r = env["result"]
    assert abs(r["barrier_with_zpe"] - 0.120) < 1e-9


def test_paper_grade_zpe_effect_note_present():
    env = run_h_barrier_paper_readiness(**_full_kwargs(barrier_eV=0.43, dZPE_eV=-0.12))
    r = env["result"]
    assert r["zpe_effect_note"] is not None
    assert "lowers" in r["zpe_effect_note"].lower() or "raises" in r["zpe_effect_note"].lower()


def test_paper_grade_at_exact_threshold():
    """H fraction exactly at threshold should be paper grade."""
    env = run_h_barrier_paper_readiness(**_full_kwargs(
        imag_mode_H_fraction=DEFAULT_H_FRACTION_THRESHOLD
    ))
    assert env["verdict"] == "PAPER_GRADE"


def test_paper_grade_envelope_keys():
    env = run_h_barrier_paper_readiness(**_full_kwargs())
    assert set(env) == {
        "tool", "version", "status", "verdict", "confidence",
        "reasons", "next_actions", "artifacts", "warnings", "result",
    }
    assert env["tool"] == "h_barrier_paper_readiness"
    assert env["status"] == "ok"


# ---------------------------------------------------------------------------
# Tests: ELECTRONIC_ONLY — each missing criterion
# ---------------------------------------------------------------------------

def test_no_freq_is_electronic_only():
    env = run_h_barrier_paper_readiness(**_full_kwargs(has_dft_freq=False))
    assert env["verdict"] == "ELECTRONIC_ONLY"
    assert "no DFT frequency" in " ".join(env["result"]["missing"]).lower() \
        or "has_dft_freq" in " ".join(env["result"]["missing"])


def test_two_imag_modes_is_electronic_only():
    env = run_h_barrier_paper_readiness(**_full_kwargs(n_imag_modes=2))
    assert env["verdict"] == "ELECTRONIC_ONLY"
    assert any("2" in m for m in env["result"]["missing"])
    assert any("imagin" in r.lower() or "imag" in r.lower() for r in env["reasons"])


def test_zero_imag_modes_is_electronic_only():
    env = run_h_barrier_paper_readiness(**_full_kwargs(n_imag_modes=0))
    assert env["verdict"] == "ELECTRONIC_ONLY"


def test_low_h_fraction_is_electronic_only():
    env = run_h_barrier_paper_readiness(**_full_kwargs(imag_mode_H_fraction=0.2))
    assert env["verdict"] == "ELECTRONIC_ONLY"
    assert any("0.2" in m or "fraction" in m.lower() for m in env["result"]["missing"])


def test_h_fraction_just_below_threshold_is_electronic_only():
    frac = DEFAULT_H_FRACTION_THRESHOLD - 1e-6
    env = run_h_barrier_paper_readiness(**_full_kwargs(imag_mode_H_fraction=frac))
    assert env["verdict"] == "ELECTRONIC_ONLY"


def test_no_dzpe_is_electronic_only():
    env = run_h_barrier_paper_readiness(**_full_kwargs(dZPE_eV=None))
    assert env["verdict"] == "ELECTRONIC_ONLY"
    assert any("dZPE" in m or "zpe" in m.lower() for m in env["result"]["missing"])


def test_n_imag_modes_none_is_electronic_only():
    env = run_h_barrier_paper_readiness(**_full_kwargs(n_imag_modes=None))
    assert env["verdict"] == "ELECTRONIC_ONLY"
    assert any("n_imag" in m.lower() or "imag" in m.lower() for m in env["result"]["missing"])


def test_imag_h_fraction_none_is_electronic_only():
    env = run_h_barrier_paper_readiness(**_full_kwargs(imag_mode_H_fraction=None))
    assert env["verdict"] == "ELECTRONIC_ONLY"


# ---------------------------------------------------------------------------
# Tests: barrier_with_zpe arithmetic in ELECTRONIC_ONLY (dZPE given but other missing)
# ---------------------------------------------------------------------------

def test_barrier_with_zpe_computed_even_if_electronic_only():
    """If dZPE_eV is given, barrier_with_zpe is always computed regardless of verdict."""
    env = run_h_barrier_paper_readiness(
        barrier_eV=0.30,
        has_dft_freq=False,  # fails check
        n_imag_modes=1,
        imag_mode_H_fraction=0.90,
        dZPE_eV=-0.11,
    )
    assert env["verdict"] == "ELECTRONIC_ONLY"
    r = env["result"]
    assert r["barrier_with_zpe"] is not None
    assert abs(r["barrier_with_zpe"] - (0.30 - 0.11)) < 1e-9


def test_barrier_with_zpe_is_none_when_dzpe_not_given():
    env = run_h_barrier_paper_readiness(
        barrier_eV=0.30,
        has_dft_freq=True,
        n_imag_modes=1,
        imag_mode_H_fraction=0.90,
        dZPE_eV=None,
    )
    assert env["result"]["barrier_with_zpe"] is None


# ---------------------------------------------------------------------------
# Tests: ZPE effect note typical range
# ---------------------------------------------------------------------------

def test_typical_zpe_range_note():
    env = run_h_barrier_paper_readiness(**_full_kwargs(dZPE_eV=-0.12))
    note = env["result"]["zpe_effect_note"]
    assert "lowers" in note.lower()


def test_atypical_large_negative_zpe_warns():
    env = run_h_barrier_paper_readiness(**_full_kwargs(dZPE_eV=-0.30))
    note = env["result"]["zpe_effect_note"]
    assert "larger" in note.lower() or "typical" in note.lower()


def test_positive_zpe_warns():
    env = run_h_barrier_paper_readiness(**_full_kwargs(dZPE_eV=0.05))
    assert any("positive" in w.lower() for w in env["warnings"])


# ---------------------------------------------------------------------------
# Tests: negative barrier warning
# ---------------------------------------------------------------------------

def test_negative_barrier_warns():
    env = run_h_barrier_paper_readiness(**_full_kwargs(barrier_eV=-0.05))
    assert any("negative" in w.lower() for w in env["warnings"])


# ---------------------------------------------------------------------------
# Tests: custom threshold
# ---------------------------------------------------------------------------

def test_custom_high_threshold_fails():
    # fraction=0.60 normally passes (> 0.5), but with threshold=0.80 fails
    env = run_h_barrier_paper_readiness(**_full_kwargs(
        imag_mode_H_fraction=0.60, h_fraction_threshold=0.80
    ))
    assert env["verdict"] == "ELECTRONIC_ONLY"


def test_custom_low_threshold_passes():
    # fraction=0.30 fails default, but with threshold=0.20 passes
    env = run_h_barrier_paper_readiness(**_full_kwargs(
        imag_mode_H_fraction=0.30, h_fraction_threshold=0.20
    ))
    assert env["verdict"] == "PAPER_GRADE"


# ---------------------------------------------------------------------------
# Tests: confidence levels
# ---------------------------------------------------------------------------

def test_paper_grade_confidence_high():
    env = run_h_barrier_paper_readiness(**_full_kwargs())
    assert env["confidence"] == "high"


def test_electronic_only_zero_criteria_confidence_low():
    env = run_h_barrier_paper_readiness(
        barrier_eV=0.30,
        has_dft_freq=False,
        n_imag_modes=None,
        imag_mode_H_fraction=None,
        dZPE_eV=None,
    )
    assert env["verdict"] == "ELECTRONIC_ONLY"
    assert env["confidence"] == "low"


def test_electronic_only_two_criteria_confidence_medium():
    # has_dft_freq=True + n_imag_modes=1 -> 2 criteria met -> medium
    env = run_h_barrier_paper_readiness(
        barrier_eV=0.30,
        has_dft_freq=True,
        n_imag_modes=1,
        imag_mode_H_fraction=None,
        dZPE_eV=None,
    )
    assert env["verdict"] == "ELECTRONIC_ONLY"
    assert env["confidence"] == "medium"


# ---------------------------------------------------------------------------
# Tests: result dict fields
# ---------------------------------------------------------------------------

def test_result_has_all_fields():
    env = run_h_barrier_paper_readiness(**_full_kwargs())
    r = env["result"]
    expected_keys = {
        "barrier_eV", "has_dft_freq", "n_imag_modes", "imag_mode_H_fraction",
        "h_fraction_threshold", "dZPE_eV", "barrier_with_zpe", "zpe_effect_note",
        "criteria_met", "criteria_total", "missing",
    }
    assert expected_keys.issubset(set(r.keys()))
    assert r["criteria_total"] == 4


def test_result_missing_list_empty_when_paper_grade():
    env = run_h_barrier_paper_readiness(**_full_kwargs())
    assert env["result"]["missing"] == []


def test_result_missing_lists_all_four_when_all_absent():
    env = run_h_barrier_paper_readiness(
        barrier_eV=0.30,
        has_dft_freq=False,
        n_imag_modes=None,
        imag_mode_H_fraction=None,
        dZPE_eV=None,
    )
    assert env["result"]["criteria_met"] == 0
    assert len(env["result"]["missing"]) == 4
