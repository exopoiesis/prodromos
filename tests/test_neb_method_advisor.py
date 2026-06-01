import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.neb_method_advisor import run_neb_method_advisor


# --- shared synthetic case builders -----------------------------------------
def _flat(value: float, n: int = 20) -> list[float]:
    return [value] * n


def base_case(**overrides) -> dict:
    """A generic, non-degenerate case with enough data to pass gate 0."""
    case = {
        "n_iter": 40,
        "barrier_dense_history_eV": _flat(0.30),
        "node_fmax_history": [0.5] * 20,
        "node_energy_history": [float(i) * 0.001 for i in range(20)],
        "path_change_history": [0.02] * 20,
        "band_energies_rel_eV": [0.0, 0.15, 0.30, 0.16, 0.0],
        "scf_noise_eV": 0.002,
    }
    case.update(overrides)
    return case


# --- REGRESSION 1: pyrite v3.3 ----------------------------------------------
def test_pyrite_v3_3_frozen_energy_log_is_stale():
    # node energies all ~equal (std<1e-9) while path_change>0 -> STALE_ENERGY_LOG
    case = base_case(
        node_energy_history=[1.234567890123] * 20,
        path_change_history=[0.05] * 20,
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "STALE_ENERGY_LOG"
    assert env["verdict"] not in {"CONVERGED_REPORT_BARRIER", "DIMER_SELLA"}
    assert env["result"]["failure_mode"] == "frozen_energy_log_geometry_moving"


def test_pyrite_v3_3_nspin1_odd_is_fix_spin_parity_first():
    # parity gate says NSPIN2 mandatory but run used nspin=1 -> fix parity first
    case = base_case(
        parity_verdict="NSPIN2_MANDATORY",
        nspin=1,
        # also frozen, but parity (gate 1) must win over stale (gate 0b)?  No:
        # gate 0b fires before gate 1. Use a NON-frozen log so parity gate is reached.
        node_energy_history=[0.30 + i * 0.001 for i in range(20)],
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "FIX_SPIN_PARITY_FIRST"
    assert env["verdict"] not in {"CONVERGED_REPORT_BARRIER", "DIMER_SELLA"}


# --- REGRESSION 2: pyrite v5 (overshoot) ------------------------------------
def test_pyrite_v5_overshoot_continue_not_dimer():
    # warm-start + constraint: barrier flat over W, fmax bounces with recovery,
    # perp residual trending down, energy converged.
    fmax_hist = [
        0.9, 0.7, 1.0, 0.6, 1.0, 0.79, 0.50, 0.37, 0.36, 0.27, 0.17,
    ]
    case = base_case(
        barrier_dense_history_eV=_flat(0.42, 18),
        node_fmax_history=fmax_hist,
        node_energy_history=[0.42 + i * 1e-4 for i in range(18)],
        path_change_history=[0.05 - i * 0.002 for i in range(18)],
        perp_resid_M100=0.20,
        perp_resid_M400=0.07,   # trending down but NOT < target/floor -> not gate 5
        band_energies_rel_eV=[0.0, 0.2, 0.42, 0.2, 0.0],
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] in {"CONTINUE_NEB", "CONVERGED_REPORT_BARRIER"}
    assert env["verdict"] != "DIMER_SELLA"


# --- REGRESSION 3: marc (magnetic) ------------------------------------------
def test_marc_magnetic_constrained_m_not_spring_tuning():
    case = base_case(
        magnetic_verdict="NO-GO_SINGLE_SHEET",
        magnetic_action="RERUN_SINGLE_SHEET_CONSTRAINED_M",
        nspin=2,
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "CONSTRAINED_M_NEB"
    assert env["verdict"] not in {
        "STIFFEN_SPRINGS_PLAIN_TO_CI",
        "SWITCH_OPTIMIZER",
        "DIMER_SELLA",
    }


def test_marc_magnetic_branch_to_mecp():
    case = base_case(
        magnetic_verdict="NO-GO_TWO_SHEET",
        magnetic_action="BRANCH_TO_MECP_OR_TWO_SEGMENT",
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "MECP"


def test_marc_magnetic_no_action_defaults_to_mecp():
    case = base_case(magnetic_verdict="NO-GO_SINGLE_SHEET")
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "MECP"


# --- INSUFFICIENT_DATA default ----------------------------------------------
def test_empty_case_is_insufficient_data():
    env = run_neb_method_advisor({})
    assert env["verdict"] == "INSUFFICIENT_DATA"
    assert env["confidence"] == "low"


def test_too_few_iters_is_insufficient_data():
    env = run_neb_method_advisor({"n_iter": 5, "band_energies_rel_eV": [0.0, 0.2, 0.0]})
    assert env["verdict"] == "INSUFFICIENT_DATA"


# --- NEB applicability ------------------------------------------------------
def test_grotthuss_is_neb_inapplicable():
    env = run_neb_method_advisor(base_case(mechanism_hint="grotthuss"))
    assert env["verdict"] == "NEB_INAPPLICABLE"


# --- same-basin -------------------------------------------------------------
def test_same_basin_repick_endpoints():
    case = base_case(
        same_basin={"h_disp": 0.3, "same_nearest_host": True, "dE_eV": 0.001},
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "REPICK_ENDPOINTS"


# --- gauge-invariant convergence --------------------------------------------
def test_converged_when_barrier_stable_and_perp_below_target():
    case = base_case(
        barrier_dense_history_eV=_flat(0.42, 18),
        perp_resid_M100=0.06,
        perp_resid_M400=0.03,   # below target AND drops -> high confidence converged
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "CONVERGED_REPORT_BARRIER"
    assert env["result"]["barrier_status"]["paper_gradeable"] is True


# --- intermediate well ------------------------------------------------------
def test_real_intermediate_is_multi_endpoint():
    case = base_case(
        band_energies_rel_eV=[0.0, 0.2, -0.4, 0.2, 0.0],
        migrating_geom={"nearest_nonH_elem": "S", "nearest_nonH_dist": 1.40},
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "MULTI_ENDPOINT"
    assert env["confidence"] == "high"


def test_artifact_well_is_multi_endpoint_with_dft_note():
    case = base_case(
        band_energies_rel_eV=[0.0, 0.2, -3.0, 0.2, 0.0],
        migrating_geom={"nearest_nonH_elem": "Fe"},
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "MULTI_ENDPOINT"
    assert any("DFT" in s for s in env["result"]["verification_followup"])


# --- roll-off triple AND ----------------------------------------------------
def test_rolloff_all_three_true_is_fixedline():
    case = base_case(
        migrating_geom={"perp_disp_ratio": 0.7, "nearest_anchor_switch": True},
        force_localization={"f_frac_react": 0.95, "f_frac_metal": 0.02},
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "FIXEDLINE_CONSTRAINT"
    assert env["result"]["cheapest_disambiguating_test"].startswith("$0")
    assert any("dimer" in s.lower() for s in env["result"]["do_not_do"])


def test_rolloff_only_one_true_is_not_rolloff():
    # perp_disp_ratio high but anchor switch False and force not localized
    case = base_case(
        migrating_geom={"perp_disp_ratio": 0.7, "nearest_anchor_switch": False},
        force_localization={"f_frac_react": 0.2, "f_frac_metal": 0.30},
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] != "FIXEDLINE_CONSTRAINT"


# --- flat-top freq branch ---------------------------------------------------
def test_flat_top_soft_imaginary_index1_is_converged_not_dimer():
    # stall: flat barrier, M-invariant perp residual at floor; freq n_imag=1 soft
    case = base_case(
        barrier_dense_history_eV=_flat(0.42, 18),
        node_fmax_history=[0.06] * 18,
        perp_resid_M100=0.12,
        perp_resid_M400=0.12,   # M-invariant, above target -> stall, not gate 5
        freq={"n_imaginary": 1, "omega_imag_cm1": 400.0, "bond_type": "S-H"},
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "CONVERGED_REPORT_BARRIER"
    assert env["verdict"] != "DIMER_SELLA"
    assert any("ZPE" in s for s in env["result"]["verification_followup"])


def test_two_imaginary_modes_is_repick_endpoints():
    case = base_case(
        barrier_dense_history_eV=_flat(0.42, 18),
        node_fmax_history=[0.06] * 18,
        perp_resid_M100=0.12,
        perp_resid_M400=0.12,
        freq={"n_imaginary": 2, "omega_imag_cm1": 500.0, "bond_type": "S-H"},
    )
    env = run_neb_method_advisor(case)
    assert env["verdict"] == "REPICK_ENDPOINTS"


# --- method ladder / diverge ------------------------------------------------
def test_diverge_escalates_to_next_untried_method():
    case = base_case(
        barrier_dense_history_eV=[0.30 + i * 0.05 for i in range(18)],
        perp_resid_M100=0.10,
        perp_resid_M400=0.40,  # monotonic up
        path_change_history=[0.01 + i * 0.005 for i in range(18)],
        methods_already_tried=["FIRE"],
    )
    env = run_neb_method_advisor(case)
    # FIRE already tried -> next rung is NEBOptimizer_ode/LBFGS -> SWITCH_OPTIMIZER
    assert env["verdict"] in {
        "SWITCH_OPTIMIZER",
        "STIFFEN_SPRINGS_PLAIN_TO_CI",
        "FIXEDLINE_CONSTRAINT",
        "STRING_METHOD",
    }


# --- envelope stability -----------------------------------------------------
def test_envelope_has_stable_keys():
    env = run_neb_method_advisor(base_case())
    assert set(env) == {
        "tool", "version", "status", "verdict", "confidence",
        "reasons", "next_actions", "artifacts", "warnings", "result",
    }
    assert env["tool"] == "neb_method_advisor"


def test_result_has_required_fields():
    env = run_neb_method_advisor(base_case())
    result = env["result"]
    for key in (
        "verdict", "confidence", "failure_mode", "excluded", "gate_trace",
        "signature_metrics", "barrier_status", "expected_barrier_range_meV",
        "out_of_range", "cheapest_disambiguating_test", "method_ladder",
        "do_not_do", "verification_followup",
    ):
        assert key in result, f"missing result key: {key}"
    assert isinstance(result["gate_trace"], list)
    assert isinstance(result["method_ladder"], list)


def test_expected_range_flag_out_of_range():
    case = base_case(
        barrier_dense_history_eV=_flat(0.80),  # 800 meV
        expected_barrier_range_meV=[100, 400],
    )
    env = run_neb_method_advisor(case)
    assert env["result"]["expected_barrier_range_meV"] == [100, 400]
    assert env["result"]["out_of_range"] is True
