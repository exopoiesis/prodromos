"""Tests for N-02 endpoint_provenance_gate and related additions to
electron_parity_gate (N-08) and neb_method_advisor (N-12)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.endpoint_provenance_gate import run_endpoint_provenance_gate
from prodromos.electron_parity_gate import run_electron_parity_gate
from prodromos.neb_method_advisor import run_neb_method_advisor


# ===========================================================================
# N-02: endpoint_provenance_gate
# ===========================================================================

class TestEndpointProvenanceGateDFTRelaxed:
    """dft_relaxed -> ENDPOINT_VALID in all cases."""

    def test_dft_relaxed_is_endpoint_valid(self):
        env = run_endpoint_provenance_gate(geometry_origin="dft_relaxed")
        assert env["verdict"] == "ENDPOINT_VALID"
        assert env["confidence"] == "high"

    def test_dft_relaxed_energy_valid_for_ranking(self):
        env = run_endpoint_provenance_gate(geometry_origin="dft_relaxed", energy_eV=-12345.6)
        assert env["result"]["energy_valid_for_ranking"] is True
        assert env["result"]["energy_downgraded"] is False
        assert env["result"]["energy_eV"] == pytest.approx(-12345.6)

    def test_dft_relaxed_bond_geometry_ok_true(self):
        env = run_endpoint_provenance_gate(geometry_origin="dft_relaxed", bond_geometry_ok=True)
        assert env["verdict"] == "ENDPOINT_VALID"
        # bond_geometry_ok=True mentioned in reasons
        assert any("necessary condition" in r.lower() for r in env["reasons"])

    def test_dft_relaxed_bond_geometry_ok_false_gives_warning(self):
        env = run_endpoint_provenance_gate(geometry_origin="dft_relaxed", bond_geometry_ok=False)
        assert env["verdict"] == "ENDPOINT_VALID"
        # there should be a warning about unusual geometry
        assert any("unusual" in w.lower() or "bond" in w.lower() for w in env["warnings"])

    def test_dft_relaxed_label_echoed(self):
        env = run_endpoint_provenance_gate(geometry_origin="dft_relaxed", label="endA")
        assert env["result"]["label"] == "endA"


class TestEndpointProvenanceGateGeometryOriginAlias:
    """The renamed `geometry_origin` param and the deprecated `provenance` alias."""

    def test_geometry_origin_in_result(self):
        env = run_endpoint_provenance_gate(geometry_origin="dft_relaxed")
        assert env["result"]["geometry_origin"] == "dft_relaxed"
        # deprecated alias key still mirrors it
        assert env["result"]["provenance"] == "dft_relaxed"

    def test_deprecated_provenance_kwarg_still_works(self):
        env = run_endpoint_provenance_gate(provenance="dft_relaxed")
        assert env["verdict"] == "ENDPOINT_VALID"
        assert env["result"]["geometry_origin"] == "dft_relaxed"

    def test_deprecated_provenance_kwarg_mlip(self):
        env = run_endpoint_provenance_gate(provenance="mlip_relaxed")
        assert env["verdict"] == "NOT_AN_ENDPOINT_MLIP_GEOMETRY"

    def test_deprecated_provenance_emits_deprecation_note(self):
        env = run_endpoint_provenance_gate(provenance="dft_relaxed")
        assert any("deprecat" in r.lower() for r in env["reasons"])

    def test_geometry_origin_no_deprecation_note(self):
        env = run_endpoint_provenance_gate(geometry_origin="dft_relaxed")
        assert not any("deprecat" in r.lower() for r in env["reasons"])

    def test_geometry_origin_wins_over_provenance(self):
        env = run_endpoint_provenance_gate(
            geometry_origin="dft_relaxed", provenance="mlip_relaxed"
        )
        assert env["verdict"] == "ENDPOINT_VALID"
        assert env["result"]["geometry_origin"] == "dft_relaxed"

    def test_missing_both_raises(self):
        with pytest.raises(ValueError):
            run_endpoint_provenance_gate()

    def test_cli_geometry_origin_flag(self):
        from prodromos.endpoint_provenance_gate import main
        rc = main(["--geometry-origin", "dft_relaxed", "--json"])
        assert rc == 0

    def test_cli_deprecated_provenance_flag_still_works(self, capsys):
        from prodromos.endpoint_provenance_gate import main
        rc = main(["--provenance", "dft_relaxed", "--json"])
        assert rc == 0
        captured = capsys.readouterr()
        # deprecation goes to stderr
        assert "deprecated" in captured.err.lower()


class TestEndpointProvenanceGateMLIPRelaxed:
    """mlip_relaxed -> NOT_AN_ENDPOINT_MLIP_GEOMETRY, energy downgraded."""

    def test_mlip_relaxed_is_not_an_endpoint(self):
        env = run_endpoint_provenance_gate(geometry_origin="mlip_relaxed")
        assert env["verdict"] == "NOT_AN_ENDPOINT_MLIP_GEOMETRY"
        assert env["confidence"] == "high"

    def test_mlip_relaxed_energy_not_valid(self):
        env = run_endpoint_provenance_gate(geometry_origin="mlip_relaxed", energy_eV=-12000.0)
        assert env["result"]["energy_valid_for_ranking"] is False
        assert env["result"]["energy_downgraded"] is True

    def test_mlip_relaxed_bond_geometry_ok_true_does_not_change_verdict(self):
        """Core contract: bond_geometry_ok=True does NOT upgrade verdict to VALID."""
        env = run_endpoint_provenance_gate(
            provenance="mlip_relaxed", bond_geometry_ok=True
        )
        assert env["verdict"] == "NOT_AN_ENDPOINT_MLIP_GEOMETRY"
        assert env["result"]["energy_valid_for_ranking"] is False
        # reasons must explain why local check is not sufficient
        combined = " ".join(env["reasons"]).lower()
        assert "not sufficient" in combined or "necessary" in combined

    def test_mlip_relaxed_downgrade_warning_present(self):
        env = run_endpoint_provenance_gate(geometry_origin="mlip_relaxed")
        assert any("downgrad" in w.lower() or "~20 ev" in w.lower() for w in env["warnings"])

    def test_mlip_relaxed_next_actions_require_dft_relaxation(self):
        env = run_endpoint_provenance_gate(geometry_origin="mlip_relaxed")
        combined = " ".join(env["next_actions"]).lower()
        assert "dft ionic relaxation" in combined or "dft relaxation" in combined

    def test_mlip_relaxed_bond_false_compounds_issue(self):
        env = run_endpoint_provenance_gate(
            provenance="mlip_relaxed", bond_geometry_ok=False
        )
        assert env["verdict"] == "NOT_AN_ENDPOINT_MLIP_GEOMETRY"
        assert any("compound" in r.lower() or "also failed" in r.lower() for r in env["reasons"])


class TestEndpointProvenanceGateUnknownProvenance:
    """Any non-dft_relaxed provenance -> NOT_AN_ENDPOINT_MLIP_GEOMETRY."""

    def test_unknown_provenance_is_not_valid(self):
        env = run_endpoint_provenance_gate(geometry_origin="unknown")
        assert env["verdict"] == "NOT_AN_ENDPOINT_MLIP_GEOMETRY"
        assert env["result"]["energy_valid_for_ranking"] is False

    def test_empty_string_provenance_is_not_valid(self):
        env = run_endpoint_provenance_gate(geometry_origin="")
        assert env["verdict"] == "NOT_AN_ENDPOINT_MLIP_GEOMETRY"

    def test_ase_relaxed_provenance_is_not_valid(self):
        # ASE/LAMMPS relaxation without DFT forces also not valid
        env = run_endpoint_provenance_gate(geometry_origin="ase_bfgs_mlip")
        assert env["verdict"] == "NOT_AN_ENDPOINT_MLIP_GEOMETRY"


class TestEndpointProvenanceGateEnvelope:
    """Stable response_envelope contract."""

    def test_envelope_has_stable_keys(self):
        env = run_endpoint_provenance_gate(geometry_origin="dft_relaxed")
        assert set(env) == {
            "tool", "version", "status", "verdict", "confidence",
            "reasons", "next_actions", "artifacts", "warnings", "result",
        }
        assert env["tool"] == "endpoint_provenance_gate"

    def test_result_has_required_fields(self):
        env = run_endpoint_provenance_gate(geometry_origin="mlip_relaxed")
        r = env["result"]
        for key in (
            "label", "geometry_origin", "provenance", "provenance_normalised",
            "is_dft_relaxed", "energy_eV", "energy_valid_for_ranking",
            "bond_geometry_ok", "energy_downgraded",
        ):
            assert key in r, f"missing result key: {key}"

    def test_is_dft_relaxed_flag(self):
        assert run_endpoint_provenance_gate(geometry_origin="dft_relaxed")["result"]["is_dft_relaxed"] is True
        assert run_endpoint_provenance_gate(geometry_origin="mlip_relaxed")["result"]["is_dft_relaxed"] is False


# ===========================================================================
# N-08: electron_parity_gate -- collapse-test spec for metallic/smearing
# ===========================================================================

class TestElectronParityGateN08:
    """N-08: NSPIN2_MANDATORY + metallic=True/smearing -> collapse-test spec."""

    def test_odd_metallic_true_emits_collapse_test_spec(self):
        # Fe31 S64 H1 is ODD (N_e=881). With metallic=True -> collapse-test spec.
        env = run_electron_parity_gate({"Fe": 31, "S": 64, "H": 1}, metallic=True)
        assert env["verdict"] == "NSPIN2_MANDATORY"
        combined = " ".join(env["next_actions"])
        # Must contain the specific collapse-test spec
        assert "COLLAPSE TEST" in combined
        assert "nspin=2 U=0 single-point" in combined
        assert "seeded starting_magnetization" in combined
        assert "magnetization_settled" in combined

    def test_odd_smearing_gaussian_emits_collapse_test_spec(self):
        env = run_electron_parity_gate(
            {"Fe": 31, "S": 64, "H": 1}, smearing="gaussian"
        )
        assert env["verdict"] == "NSPIN2_MANDATORY"
        combined = " ".join(env["next_actions"])
        assert "COLLAPSE TEST" in combined

    def test_odd_smearing_cold_emits_collapse_test_spec(self):
        env = run_electron_parity_gate(
            {"Fe": 31, "S": 64, "H": 1}, smearing="cold"
        )
        combined = " ".join(env["next_actions"])
        assert "COLLAPSE TEST" in combined

    def test_odd_no_metallic_no_spec(self):
        """Without metallic or smearing: generic spec only, no COLLAPSE TEST."""
        env = run_electron_parity_gate({"Fe": 31, "S": 64, "H": 1})
        combined = " ".join(env["next_actions"])
        assert "COLLAPSE TEST" not in combined

    def test_even_metallic_no_spec(self):
        """Even parity + metallic: no collapse-test spec (only applies to odd)."""
        env = run_electron_parity_gate({"Fe": 32, "S": 64}, metallic=True)
        assert env["verdict"] != "NSPIN2_MANDATORY"
        combined = " ".join(env["next_actions"])
        assert "COLLAPSE TEST" not in combined

    def test_metallic_context_echoed_in_result(self):
        env = run_electron_parity_gate({"Fe": 31, "S": 64, "H": 1}, metallic=True)
        assert env["result"]["metallic_smearing_context"] is True

    def test_no_metallic_echoed_false(self):
        env = run_electron_parity_gate({"Fe": 31, "S": 64, "H": 1})
        assert env["result"]["metallic_smearing_context"] is False

    def test_collapse_spec_references_spin_collapse_verdict(self):
        """Spec must cross-reference spin_collapse_verdict (the magnetization_settled field)."""
        env = run_electron_parity_gate(
            {"Fe": 31, "S": 64, "H": 1}, smearing="mv"
        )
        combined = " ".join(env["next_actions"])
        # Cross-reference to spin_collapse_verdict tool
        assert "spin_collapse_verdict" in combined
        # Must mention the interpretation step
        assert "mabs_per_tm" in combined or "magnetization_settled" in combined

    def test_existing_tests_still_pass_with_new_signature(self):
        """Regression: old call without metallic/smearing still works."""
        env = run_electron_parity_gate({"Fe": 31, "S": 64, "H": 1})
        assert env["verdict"] == "NSPIN2_MANDATORY"
        assert env["result"]["n_electrons"] == 881
        assert env["result"]["parity"] == "odd"


# ===========================================================================
# N-12: neb_method_advisor -- dimer_chemical_rc branch
# ===========================================================================

def _base_case(**overrides) -> dict:
    """Generic non-degenerate case with enough data to pass gate 0."""
    case = {
        "n_iter": 40,
        "barrier_dense_history_eV": [0.30] * 20,
        "node_fmax_history": [0.5] * 20,
        "node_energy_history": [float(i) * 0.001 for i in range(20)],
        "path_change_history": [0.02] * 20,
        "band_energies_rel_eV": [0.0, 0.15, 0.30, 0.16, 0.0],
        "scf_noise_eV": 0.002,
    }
    case.update(overrides)
    return case


class TestNEBMethodAdvisorN12:
    """N-12: failure_signature_hint in dimer-RC set -> DIMER_CHEMICAL_RC."""

    def test_rolloff_signature_gives_dimer_chemical_rc(self):
        case = _base_case(failure_signature_hint="roll-off")
        env = run_neb_method_advisor(case)
        assert env["verdict"] == "DIMER_CHEMICAL_RC"
        assert env["confidence"] == "high"

    def test_frozen_energy_signature_gives_dimer_chemical_rc(self):
        case = _base_case(failure_signature_hint="frozen-energy")
        env = run_neb_method_advisor(case)
        assert env["verdict"] == "DIMER_CHEMICAL_RC"

    def test_same_basin_signature_gives_dimer_chemical_rc(self):
        # Note: this is the failure_signature_hint path, NOT the same_basin dict path (gate 4)
        # They are distinct inputs. The hint path fires at gate 4b.
        case = _base_case(failure_signature_hint="same-basin")
        env = run_neb_method_advisor(case)
        assert env["verdict"] == "DIMER_CHEMICAL_RC"

    def test_multi_site_pocket_signature_gives_dimer_chemical_rc(self):
        case = _base_case(failure_signature_hint="multi-site pocket")
        env = run_neb_method_advisor(case)
        assert env["verdict"] == "DIMER_CHEMICAL_RC"

    def test_dimer_rc_result_contains_recipe(self):
        case = _base_case(failure_signature_hint="roll-off")
        env = run_neb_method_advisor(case)
        result = env["result"]
        combined = " ".join(result.get("reasons", []))
        # Recipe: unit H-vector along S_i->S_k (MIC)
        assert "S_i" in combined or "S_k" in combined or "MIC" in combined
        assert "dimer" in combined.lower() or "Sella" in combined

    def test_dimer_rc_do_not_do_no_more_band_tuning(self):
        case = _base_case(failure_signature_hint="roll-off")
        env = run_neb_method_advisor(case)
        combined = " ".join(env["result"].get("do_not_do", [])).lower()
        assert "band-neb" in combined or "k_spring" in combined or "optimizer" in combined

    def test_dimer_rc_verification_followup(self):
        case = _base_case(failure_signature_hint="frozen-energy")
        env = run_neb_method_advisor(case)
        combined = " ".join(env["result"].get("verification_followup", [])).lower()
        # Must reference saddle proximity gate or index-1 confirmation
        assert "index-1" in combined or "saddle" in combined or "proximity" in combined

    def test_unknown_failure_signature_does_not_fire_dimer_rc(self):
        case = _base_case(failure_signature_hint="overshoot")
        env = run_neb_method_advisor(case)
        assert env["verdict"] != "DIMER_CHEMICAL_RC"

    def test_no_failure_signature_does_not_fire_dimer_rc(self):
        case = _base_case()
        env = run_neb_method_advisor(case)
        assert env["verdict"] != "DIMER_CHEMICAL_RC"


class TestNEBMethodAdvisorN02Advisory:
    """N-02 advisory in neb_method_advisor: mlip provenance -> warning."""

    def test_mlip_provenance_triggers_warning(self):
        case = _base_case(endpoint_provenance="mlip_relaxed")
        env = run_neb_method_advisor(case)
        combined = " ".join(env["warnings"]).lower()
        assert "provenance" in combined or "mlip" in combined or "dft-relaxed" in combined

    def test_dft_provenance_no_extra_warning(self):
        case = _base_case(endpoint_provenance="dft_relaxed")
        env = run_neb_method_advisor(case)
        # Only the standard gauge-dependent warning; no provenance warning
        prov_warnings = [w for w in env["warnings"] if "provenance" in w.lower()]
        assert len(prov_warnings) == 0

    def test_no_provenance_key_no_extra_warning(self):
        case = _base_case()
        env = run_neb_method_advisor(case)
        prov_warnings = [w for w in env["warnings"] if "provenance" in w.lower()]
        assert len(prov_warnings) == 0
