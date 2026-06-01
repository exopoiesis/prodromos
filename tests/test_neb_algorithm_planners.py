"""Tests for magnetic-aware NEB algorithm planners."""
import numpy as np
import pytest

from prodromos.adaptive_neb_planner import (
    adaptive_reparam_monitor,
    build_algorithm_plan,
    dyneb_active_images,
    ocineb_climber_plan,
    spm_pair_springs,
)
from prodromos.gp_neb_surrogate import build_gp_neb_plan


def smooth_band_summary():
    return {
        "labels": [f"image_{i:02d}" for i in range(5)],
        "energies_eV": [0.0, 0.08, 0.24, 0.12, 0.01],
        "force_norms_eV_A": [0.0, 0.20, 0.04, 0.08, 0.0],
        "distances_A": [1.0, 1.8, 1.2, 1.0],
        "total_magnetization_uB": [1.13, 1.14, 1.12, 1.13, 1.13],
        "absolute_magnetization_uB": [2.00, 2.05, 2.08, 2.04, 2.02],
        "s": [0.0, 0.25, 0.5, 0.75, 1.0],
    }


def magnetic_split_summary():
    return {
        "labels": [f"image_{i:02d}" for i in range(5)],
        "energies_eV": [0.0, 0.12, 0.18, 0.14, 0.01],
        "force_norms_eV_A": [0.0, 0.03, 0.04, 0.03, 0.0],
        "distances_A": [1.0, 1.0, 1.0, 1.0],
        "total_magnetization_uB": [1.13, 1.13, 1.13, 1.13, 1.13],
        "absolute_magnetization_uB": [2.0, 2.0, 2.7, 2.7, 2.7],
        "s": [0.0, 0.25, 0.5, 0.75, 1.0],
    }


def test_spm_pair_springs_prioritize_high_energy_and_wide_edges():
    springs = spm_pair_springs(
        np.array([0.0, 0.08, 0.24, 0.12, 0.01]),
        np.array([1.0, 1.8, 1.2, 1.0]),
        k_min=0.3,
        k_max=3.0,
    )
    assert len(springs) == 4
    assert all(0.3 <= row["k_spring_eV_A2"] <= 3.0 for row in springs)
    assert springs[1]["k_spring_eV_A2"] > springs[0]["k_spring_eV_A2"]


def test_adaptive_monitor_forms_normalized_target_intervals():
    monitor = adaptive_reparam_monitor(
        np.array([0.0, 0.08, 0.24, 0.12, 0.01]),
        np.array([1.0, 1.8, 1.2, 1.0]),
    )
    assert len(monitor) == 4
    assert sum(row["monitor_weight"] for row in monitor) == pytest.approx(1.0)
    assert monitor[0]["target_s_left"] == pytest.approx(0.0)
    assert monitor[-1]["target_s_right"] == pytest.approx(1.0)


def test_dyneb_and_ocineb_select_active_images_and_climber():
    energies = np.array([0.0, 0.08, 0.24, 0.12, 0.01])
    forces = np.array([0.0, 0.20, 0.04, 0.08, 0.0])
    active = dyneb_active_images(energies, forces, fmax=0.05, scale_fmax=1.0)
    assert active[0]["active"] is False
    assert active[-1]["active"] is False
    assert active[1]["active"] is True
    climber = ocineb_climber_plan(energies, forces)
    assert climber["climber_image"] == 2
    assert climber["start_climbing"] is True


def test_algorithm_plan_refuses_geometry_only_tools_on_magnetic_split():
    plan = build_algorithm_plan(magnetic_split_summary())
    assert plan["magnetic_gate"]["verdict"] == "NO-GO_SINGLE_SHEET"
    assert plan["primary_recommendation"] == "DO_NOT_USE_GEOMETRY_ONLY_NEB_VARIANTS"
    assert plan["spm_pair_springs"] == []
    assert "constrained-M" in " ".join(plan["next_actions"])


def test_algorithm_plan_returns_practical_settings_for_single_sheet_band():
    plan = build_algorithm_plan(smooth_band_summary())
    assert plan["magnetic_gate"]["verdict"] == "GO"
    assert plan["primary_recommendation"] == "ADAPTIVE_SPRINGS_PLUS_DYNEB_OCINEB"
    assert len(plan["spm_pair_springs"]) == 4
    assert len(plan["adaptive_monitor"]) == 4
    assert len(plan["dyneb_active_images"]) == 5
    assert plan["ocineb_climber"]["climber_image"] == 2


def test_gp_neb_plan_suggests_new_samples_for_single_sheet_band():
    plan = build_gp_neb_plan(smooth_band_summary(), grid_size=101, top_k=2)
    assert plan["magnetic_gate"]["verdict"] == "GO"
    assert plan["primary_recommendation"] == "EVALUATE_GP_SUGGESTED_IMAGES"
    assert plan["gp"]["predicted_barrier_eV"] > 0.0
    assert len(plan["gp"]["suggested_next_samples"]) == 2
    assert all(0.0 < row["s"] < 1.0 for row in plan["gp"]["suggested_next_samples"])


def test_gp_neb_plan_refuses_single_gp_across_magnetic_split():
    plan = build_gp_neb_plan(magnetic_split_summary())
    assert plan["magnetic_gate"]["verdict"] == "NO-GO_SINGLE_SHEET"
    assert plan["primary_recommendation"] == "DO_NOT_FIT_SINGLE_GP_ACROSS_MAGNETIC_SPLIT"
    assert plan["gp"] is None
