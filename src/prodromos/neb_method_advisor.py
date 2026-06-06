"""NEB Method Advisor -- a top-level ROUTER for NEB failure modes.

This tool ingests an NEB calculation's signature (band + optimizer history +
geometry + optional freq/magnetic signals) as an already-computed ``case`` dict
and recommends the right METHOD FAMILY (continue-NEB / stiffen / switch-optimizer
/ FixedLine / string / dimer-Sella / MECP / constrained-M / re-pick /
multi-endpoint / fix-parity / report-converged), encoding the
cross-mineral failure-mode lessons.

Core principle (see NEB_METHOD_ADVISOR_DESIGN.md): the advisor is a ROUTER, NOT a
re-solver. It does NOT re-decide what an upstream gate already decided. It runs a
hard, ordered set of gates; the first gate that fires returns the verdict and
later gates are NOT evaluated.

It does NOT call heavy DFT/MLIP. It consumes a ``case`` dict of already-computed
signals, so it is fully unit-testable. All ``case`` keys are optional; missing
signals lower confidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from statistics import pstdev
from typing import Any

from prodromos.cli_contract import dump_json, response_envelope

TOOL = "neb_method_advisor"

# --- thresholds (from the design doc) ---------------------------------------
W_MIN = 15                 # minimum n_iter and history window
EPS_E_FLOOR_EV = 0.01      # barrier-drift floor (~96-atom SCF noise)
SCF_NOISE_DEFAULT = 0.002  # eV, used when scf_noise not provided
PERP_RESID_TARGET = 0.05   # eV/A perpendicular residual target
STALE_ENERGY_STD = 1e-9    # std below this with motion -> frozen energy log
SAME_BASIN_H_DISP = 1.0    # Angstrom
INTERMEDIATE_DEPTH_EV = 0.1
INTERMEDIATE_ARTIFACT_EV = -2.0
ROLLOFF_PERP_DISP_RATIO = 0.5
ROLLOFF_F_FRAC_REACT = 0.9
ROLLOFF_F_FRAC_METAL = 0.05
FLAT_TOP_FRACTION = 0.4    # |omega_imag| < 0.4 * omega_stretch -> flat-top

# bond-type stretch frequencies (cm^-1) for the relative flat-top threshold
OMEGA_STRETCH_CM1 = {
    "S-H": 2500.0,
    "Fe-H": 1800.0,
    "O-H": 3000.0,
}
OMEGA_STRETCH_DEFAULT = 2500.0

# method ladder, cheapest untried first
METHOD_LADDER = [
    "FIRE",
    "NEBOptimizer_ode",
    "LBFGS",
    "k_spring_up_plain_to_CI",
    "FIXEDLINE_CONSTRAINT",
    "STRING_METHOD",
    "DIMER_SELLA",
]


@dataclass
class AdvisorResult:
    """Structured router result; serialized into response_envelope.result."""

    verdict: str
    confidence: str
    failure_mode: str | None = None
    excluded: list[str] = field(default_factory=list)
    gate_trace: list[str] = field(default_factory=list)
    signature_metrics: dict = field(default_factory=dict)
    barrier_status: dict = field(default_factory=dict)
    expected_barrier_range_meV: list | None = None
    out_of_range: bool | None = None
    cheapest_disambiguating_test: str | None = None
    method_ladder: list[dict] = field(default_factory=list)
    do_not_do: list[str] = field(default_factory=list)
    verification_followup: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "failure_mode": self.failure_mode,
            "excluded": self.excluded,
            "gate_trace": self.gate_trace,
            "signature_metrics": self.signature_metrics,
            "barrier_status": self.barrier_status,
            "expected_barrier_range_meV": self.expected_barrier_range_meV,
            "out_of_range": self.out_of_range,
            "cheapest_disambiguating_test": self.cheapest_disambiguating_test,
            "method_ladder": self.method_ladder,
            "do_not_do": self.do_not_do,
            "verification_followup": self.verification_followup,
            "reasons": self.reasons,
            "next_actions": self.next_actions,
        }


# --- helpers ----------------------------------------------------------------
def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _floats(value: Any) -> list[float]:
    out: list[float] = []
    for item in _as_list(value):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _slope(values: list[float]) -> float:
    """Least-squares slope over an evenly-spaced index axis."""
    n = len(values)
    if n < 2:
        return 0.0
    xbar = (n - 1) / 2.0
    ybar = sum(values) / n
    num = sum((i - xbar) * (values[i] - ybar) for i in range(n))
    den = sum((i - xbar) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def _count_local_minima(values: list[float]) -> int:
    if len(values) < 3:
        return 0
    count = 0
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            count += 1
    return count


def _is_monotonic_up(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    return all(values[i + 1] >= values[i] for i in range(len(values) - 1))


def _drift(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values) - min(values)


def _omega_stretch(bond_type: str | None) -> float:
    if not bond_type:
        return OMEGA_STRETCH_DEFAULT
    return OMEGA_STRETCH_CM1.get(bond_type, OMEGA_STRETCH_DEFAULT)


def _ranked_ladder(methods_already_tried: list[str]) -> list[dict]:
    tried = {str(m) for m in methods_already_tried}
    rows: list[dict] = []
    rank = 0
    for method in METHOD_LADDER:
        already = method in tried
        rows.append(
            {
                "method": method,
                "already_tried": already,
                "rank": rank if not already else None,
                "applicability_gate": _ladder_gate(method),
            }
        )
        if not already:
            rank += 1
    return rows


def _ladder_gate(method: str) -> str:
    gates = {
        "FIRE": "robust default optimizer; always applicable",
        "NEBOptimizer_ode": "ASE NEBOptimizer ODE/LBFGS; smoother on flat regions",
        "LBFGS": "quasi-Newton; needs reasonable Hessian behaviour",
        "k_spring_up_plain_to_CI": "stiffen springs then plain->CI; use if band rolls off ridge",
        "FIXEDLINE_CONSTRAINT": "constrain migrating atom to chord; degenerate-saddle / wrong-RC",
        "STRING_METHOD": "growing/freezing string; asymmetric or bad initial guess",
        "DIMER_SELLA": "Sella eigenvector-following; ONLY near a confirmed saddle",
    }
    return gates.get(method, "applicability not specified")


def _next_untried(method_ladder: list[dict]) -> str | None:
    for row in method_ladder:
        if not row["already_tried"]:
            return row["method"]
    return None


# --- core router ------------------------------------------------------------
def run_neb_method_advisor(case: dict | None = None) -> dict:
    """Route an NEB signature ``case`` dict to a method-family verdict.

    Returns a response_envelope (tool="neb_method_advisor"). The first gate that
    fires wins; later gates are not evaluated.
    """
    case = dict(case or {})
    res = AdvisorResult(verdict="INSUFFICIENT_DATA", confidence="low")

    # gather signals
    n_iter = case.get("n_iter")
    barrier_hist = _floats(case.get("barrier_dense_history_eV"))
    node_fmax_hist = _floats(case.get("node_fmax_history"))
    node_energy_hist = _floats(case.get("node_energy_history"))
    path_change_hist = _floats(case.get("path_change_history"))
    perp_m100 = case.get("perp_resid_M100")
    perp_m400 = case.get("perp_resid_M400")
    scf_noise = case.get("scf_noise_eV")
    band = _floats(case.get("band_energies_rel_eV"))
    mig = dict(case.get("migrating_geom") or {})
    floc = dict(case.get("force_localization") or {})
    parity_verdict = case.get("parity_verdict")
    nspin = case.get("nspin")
    magnetic_verdict = case.get("magnetic_verdict")
    magnetic_action = case.get("magnetic_action")
    same_basin = dict(case.get("same_basin") or {})
    freq = dict(case.get("freq") or {})
    methods_already_tried = _as_list(case.get("methods_already_tried"))
    mechanism_hint = case.get("mechanism_hint")
    expected_range = case.get("expected_barrier_range_meV")
    failure_signature_hint = case.get("failure_signature_hint")
    endpoint_provenance = case.get("endpoint_provenance")  # N-02 advisory

    scf_noise_eff = float(scf_noise) if scf_noise is not None else SCF_NOISE_DEFAULT
    eps_e = max(5.0 * scf_noise_eff, EPS_E_FLOOR_EV)

    # windowed views
    barrier_w = barrier_hist[-W_MIN:] if barrier_hist else []
    fmax_w = node_fmax_hist[-W_MIN:] if node_fmax_hist else []
    perp_target = PERP_RESID_TARGET

    barrier_drift = _drift(barrier_w)
    barrier_slope = _slope(barrier_w) if barrier_w else None
    fmax_local_minima = _count_local_minima(node_fmax_hist) if node_fmax_hist else 0
    path_change_mean = (
        sum(path_change_hist) / len(path_change_hist) if path_change_hist else None
    )
    node_energy_std = pstdev(node_energy_hist) if len(node_energy_hist) > 1 else None
    perp_disp_ratio = mig.get("perp_disp_ratio")
    anchor_switch = bool(mig.get("nearest_anchor_switch"))
    f_frac_react = floc.get("f_frac_react")
    f_frac_metal = floc.get("f_frac_metal")
    barrier_eV = max(barrier_w) if barrier_w else (max(band) if band else None)

    # roll-off triple-AND (all four required: a,b,c1,c2)
    rolloff_a = perp_disp_ratio is not None and float(perp_disp_ratio) > ROLLOFF_PERP_DISP_RATIO
    rolloff_b = anchor_switch
    rolloff_c1 = f_frac_react is not None and float(f_frac_react) > ROLLOFF_F_FRAC_REACT
    rolloff_c2 = f_frac_metal is not None and float(f_frac_metal) < ROLLOFF_F_FRAC_METAL
    rolloff_all = rolloff_a and rolloff_b and rolloff_c1 and rolloff_c2

    # barrier convergence (gauge-invariant)
    barrier_converged = barrier_drift is not None and barrier_drift < eps_e
    floor_drops = (
        perp_m100 is not None
        and perp_m400 is not None
        and float(perp_m400) < float(perp_m100)
    )
    perp_below_target = perp_m400 is not None and float(perp_m400) < perp_target
    perp_invariant = (
        perp_m100 is not None
        and perp_m400 is not None
        and abs(float(perp_m400) - float(perp_m100)) < 0.2 * perp_target
    )

    res.signature_metrics = {
        "n_iter": n_iter,
        "W": W_MIN,
        "eps_E_eV": eps_e,
        "scf_noise_eV": scf_noise_eff,
        "barrier_drift_W_eV": barrier_drift,
        "barrier_slope_W": barrier_slope,
        "barrier_eV": barrier_eV,
        "node_fmax_floor": min(fmax_w) if fmax_w else None,
        "node_fmax_floor_gauge_dependent": True,
        "node_fmax_local_minima": fmax_local_minima,
        "node_energy_std": node_energy_std,
        "path_change_mean": path_change_mean,
        "perp_resid_M100": perp_m100,
        "perp_resid_M400": perp_m400,
        "perp_resid_target": perp_target,
        "perp_disp_ratio": perp_disp_ratio,
        "nearest_anchor_switch": anchor_switch,
        "force_localization": {"f_frac_react": f_frac_react, "f_frac_metal": f_frac_metal},
        "rolloff_triple_and": rolloff_all,
        "barrier_converged": barrier_converged,
    }
    res.method_ladder = _ranked_ladder(methods_already_tried)

    # expected-range triangulation flag
    if expected_range and barrier_eV is not None and len(expected_range) == 2:
        lo, hi = float(expected_range[0]), float(expected_range[1])
        res.expected_barrier_range_meV = [lo, hi]
        res.out_of_range = not (lo <= barrier_eV * 1000.0 <= hi)
    elif expected_range:
        res.expected_barrier_range_meV = list(expected_range)
        res.out_of_range = None

    def finish(verdict: str) -> dict:
        res.verdict = verdict
        _fill_barrier_status(res, barrier_converged, perp_below_target or floor_drops, barrier_eV, eps_e)
        extra_warnings: list[str] = ["node-fmax is gauge-dependent; convergence judged on invariant metrics"]
        # N-02 advisory: endpoint provenance check
        if endpoint_provenance is not None and str(endpoint_provenance).lower() != "dft_relaxed":
            extra_warnings.append(
                "ENDPOINT PROVENANCE WARNING: endpoint_provenance is not 'dft_relaxed' "
                f"(got '{endpoint_provenance}'). Single-point energies on a non-DFT-relaxed "
                "geometry (grad V != 0) are NOT valid endpoint energies -- the global lattice "
                "may be ~20 eV off even if local bond lengths look physical. "
                "Run DFT ionic relaxation on the endpoint BEFORE using its energy for barrier "
                "ranking or NEB setup. Use endpoint_provenance_gate.py for a full verdict."
            )
        return response_envelope(
            tool=TOOL,
            verdict=res.verdict,
            confidence=res.confidence,
            reasons=res.reasons,
            next_actions=res.next_actions,
            warnings=extra_warnings,
            result=res.to_dict(),
        )

    # --- Gate 0: data sufficiency -------------------------------------------
    res.gate_trace.append("0_data_sufficiency")
    has_history = bool(barrier_hist or node_energy_hist or band)
    if n_iter is None or float(n_iter) < W_MIN or not has_history:
        res.failure_mode = "insufficient_data"
        res.confidence = "low"
        res.reasons.append(
            f"need n_iter>={W_MIN} and a band/history; have n_iter={n_iter}, history={has_history}"
        )
        res.cheapest_disambiguating_test = "run >=W iterations and dump dense-spline barrier + node energies"
        res.gate_trace[-1] += ":INSUFFICIENT_DATA"
        return finish("INSUFFICIENT_DATA")
    res.gate_trace[-1] += ":pass"

    # --- Gate 0b: STALE_ENERGY_LOG ------------------------------------------
    res.gate_trace.append("0b_stale_energy_log")
    motion = path_change_hist and max(path_change_hist) > 0
    if node_energy_std is not None and node_energy_std < STALE_ENERGY_STD and motion:
        res.failure_mode = "frozen_energy_log_geometry_moving"
        res.confidence = "high"
        res.reasons.append(
            "node energies are frozen (std<1e-9) while path_change>0: stale/frozen energy log"
        )
        res.cheapest_disambiguating_test = (
            "verify writer flushes energies each iter; re-run a single SCF single-point to confirm parser"
        )
        res.do_not_do = [
            "do NOT interpret a flat barrier as convergence",
            "do NOT switch to dimer/flat-top reasoning on a frozen log",
        ]
        res.excluded = [
            "CONVERGED_REPORT_BARRIER (ruled out by 0b: energy log frozen, not truly converged)",
            "DIMER_SELLA (ruled out by 0b)",
        ]
        res.gate_trace[-1] += ":STALE_ENERGY_LOG"
        return finish("STALE_ENERGY_LOG")
    res.gate_trace[-1] += ":pass"

    # --- Gate 1: electron parity --------------------------------------------
    res.gate_trace.append("1_electron_parity")
    if parity_verdict == "NSPIN2_MANDATORY" and nspin is not None and int(nspin) == 1:
        res.failure_mode = "wrong_spin_parity"
        res.confidence = "high"
        res.reasons.append(
            "odd electron count requires nspin=2 (parity gate NSPIN2_MANDATORY) but run used nspin=1"
        )
        res.cheapest_disambiguating_test = "rerun a single nspin=2 single-point on one image to confirm magnetization"
        res.do_not_do = [
            "do NOT tune springs/optimizer before fixing spin parity",
            "do NOT trust any barrier from the nspin=1 run",
        ]
        res.excluded = ["all geometry/optimizer modes (ruled out by 1: spin parity must be fixed first)"]
        res.gate_trace[-1] += ":FIX_SPIN_PARITY_FIRST"
        return finish("FIX_SPIN_PARITY_FIRST")
    res.gate_trace[-1] += ":pass"

    # --- Gate 2: NEB applicability ------------------------------------------
    res.gate_trace.append("2_neb_applicability")
    if mechanism_hint in {"grotthuss", "solvation", "liquid_proton"}:
        res.failure_mode = "neb_inapplicable_collective_mechanism"
        res.confidence = "high"
        res.reasons.append(
            f"mechanism_hint={mechanism_hint}: collective/solvent-coupled motion, single MEP ill-defined"
        )
        res.cheapest_disambiguating_test = "run short AIMD / metadynamics free-energy estimate instead of NEB"
        res.do_not_do = ["do NOT force a single-image-chain NEB on a Grotthuss/solvation mechanism"]
        res.verification_followup = ["AIMD or OPES/metadynamics free-energy barrier (RESHENIE-070)"]
        res.gate_trace[-1] += ":NEB_INAPPLICABLE"
        return finish("NEB_INAPPLICABLE")
    res.gate_trace[-1] += ":pass"

    # --- Gate 3: magnetic delegation ----------------------------------------
    res.gate_trace.append("3_magnetic")
    if isinstance(magnetic_verdict, str) and magnetic_verdict.startswith("NO-GO"):
        res.failure_mode = "magnetic_sheet_crossing"
        res.confidence = "high"
        action = magnetic_action or ""
        res.reasons.append(
            f"magnetic gate verdict={magnetic_verdict}; delegating to magnetic recommendation action={action or 'none'}"
        )
        res.do_not_do = [
            "do NOT tune springs/optimizer to 'fix' a magnetic sheet crossing",
            "do NOT run single-sheet NEB across two magnetic sheets (ill-posed)",
        ]
        res.excluded = ["spring tuning / optimizer switch (ruled out by 3: magnetic, not geometric)"]
        if action == "RERUN_SINGLE_SHEET_CONSTRAINED_M":
            res.cheapest_disambiguating_test = "constrained-M endpoint pilot single-points at the shared total magnetization"
            res.verification_followup = ["rerun full NEB at fixed tot_magnetization once pilot is continuous"]
            res.gate_trace[-1] += ":CONSTRAINED_M_NEB"
            return finish("CONSTRAINED_M_NEB")
        if action.startswith("BRANCH_TO_MECP"):
            res.cheapest_disambiguating_test = "MECP search near the magnetic seam image"
            res.gate_trace[-1] += ":MECP"
            return finish("MECP")
        # no explicit action -> default to MECP per spec
        res.cheapest_disambiguating_test = "MECP search near the magnetic seam image"
        res.gate_trace[-1] += ":MECP"
        return finish("MECP")
    res.gate_trace[-1] += ":pass"

    # --- Gate 4: same-basin (Test A) ----------------------------------------
    res.gate_trace.append("4_same_basin")
    sb_h = same_basin.get("h_disp")
    sb_same_host = bool(same_basin.get("same_nearest_host"))
    sb_de = same_basin.get("dE_eV")
    if (
        sb_h is not None
        and float(sb_h) < SAME_BASIN_H_DISP
        and sb_same_host
        and sb_de is not None
        and abs(float(sb_de)) < 5.0 * scf_noise_eff
    ):
        res.failure_mode = "endpoints_in_same_basin"
        res.confidence = "high"
        res.reasons.append(
            f"endpoints look like the same state: h_disp={float(sb_h):.3g} A < {SAME_BASIN_H_DISP} A, "
            f"same nearest host, dE={float(sb_de):.3g} eV < 5*SCF_noise"
        )
        res.cheapest_disambiguating_test = "re-enumerate distinct V_Fe endpoints (different nearest host atom)"
        res.do_not_do = ["do NOT report a near-zero barrier from same-basin endpoints as a real barrier"]
        res.excluded = ["CONVERGED_REPORT_BARRIER (ruled out by 4: endpoints are the same basin)"]
        res.gate_trace[-1] += ":REPICK_ENDPOINTS"
        return finish("REPICK_ENDPOINTS")
    res.gate_trace[-1] += ":pass"

    # --- Gate 4b: dimer + chemical-RC seed (N-12) ---------------------------
    # Band-NEB on Fe-S V_Fe+H pockets repeatedly rolls off the ridge or freezes
    # when the transition state manifold is degenerate (cubic symmetry, multi-site
    # pocket). For signatures in this set, the correct escalation is DIMER with a
    # chemical reaction coordinate seed -- NOT another band-tune.
    _DIMER_RC_SIGNATURES = {"roll-off", "frozen-energy", "same-basin", "multi-site pocket"}
    res.gate_trace.append("4b_dimer_chemical_rc")
    if (
        failure_signature_hint is not None
        and str(failure_signature_hint).lower() in _DIMER_RC_SIGNATURES
    ):
        res.failure_mode = "dimer_chemical_rc_indicated"
        res.confidence = "high"
        sig = str(failure_signature_hint).lower()
        res.reasons.append(
            f"failure_signature_hint='{sig}' is in the dimer-chemical-RC set "
            f"{sorted(_DIMER_RC_SIGNATURES)}: band-NEB has exhausted its optimizer "
            f"space on this Fe-S V_Fe+H pocket (ridge degeneracy / symmetry-induced "
            f"transition manifold). Further band-tuning will NOT converge."
        )
        res.reasons.append(
            "Recipe: (1) form the unit H-transfer vector v = (S_k - S_i) / |S_k - S_i| "
            "using minimum-image convention (MIC); (2) seed the dimer displacement along v "
            "starting from the constrained-NEB midpoint or the highest-energy band image; "
            "(3) run Sella/dimer with this chemical-RC seed to locate the true index-1 saddle."
        )
        res.cheapest_disambiguating_test = (
            "seed Sella/dimer from unit H-vector S_i->S_k (MIC) at the NEB midpoint / "
            "highest band image -- $0 MLIP dimer pilot to confirm index-1 before DFT"
        )
        res.do_not_do = [
            "do NOT run another band-NEB with different k_spring or optimizer (already exhausted)",
            "do NOT blame convergence on smearing/spin without verifying the RC first",
        ]
        res.verification_followup = [
            "confirm index-1 via Hessian/freq after dimer convergence",
            "check saddle_proximity_gate: d(H-S_i) vs d(H-S_k) asymmetry < 0.15 A",
        ]
        res.gate_trace[-1] += ":DIMER_CHEMICAL_RC"
        return finish("DIMER_CHEMICAL_RC")
    res.gate_trace[-1] += ":pass"

    # --- Gate 5: gauge-invariant convergence --------------------------------
    res.gate_trace.append("5_gauge_invariant_convergence")
    perp_ok = floor_drops or perp_below_target
    if barrier_converged and perp_ok:
        res.failure_mode = None
        res.confidence = "high" if perp_below_target and floor_drops else "medium"
        why = []
        if barrier_converged:
            why.append(f"barrier drift over W ({barrier_drift:.4g} eV) < eps_E ({eps_e:.4g} eV)")
        if floor_drops:
            why.append("perp residual floor drops M100->M400 (interpolation artifact, not real residual)")
        if perp_below_target:
            why.append(f"perp residual at M400 ({float(perp_m400):.3g}) < target ({perp_target})")
        res.reasons.extend(why)
        res.verification_followup = ["report barrier on the dense spline (not node-max); cite invariant residual"]
        res.do_not_do = ["do NOT switch optimizers or run a dimer on a converged path"]
        res.excluded = ["DIMER_SELLA / method ladder (ruled out by 5: path is converged)"]
        res.gate_trace[-1] += ":CONVERGED_REPORT_BARRIER"
        return finish("CONVERGED_REPORT_BARRIER")
    res.gate_trace[-1] += ":pass"

    # --- Gate 6: intermediate well in band ----------------------------------
    res.gate_trace.append("6_intermediate_well")
    if len(band) >= 3:
        end_floor = min(band[0], band[-1])
        interior = band[1:-1]
        i_min_local = min(range(len(interior)), key=lambda k: interior[k])
        e_min = interior[i_min_local]
        # a real intermediate well: interior min is below the endpoint floor by
        # more than INTERMEDIATE_DEPTH_EV. "spanning" is read as: the well region
        # (images at or below the endpoint floor) covers >=1 interior image, with
        # at least 2 interior images present so a genuine basin (not a single
        # spline glitch) can form.
        below_floor = sum(1 for v in interior if v < end_floor)
        is_well = (
            e_min < end_floor - INTERMEDIATE_DEPTH_EV
            and len(interior) >= 2
            and below_floor >= 1
        )
        if is_well:
            nearest = mig.get("nearest_nonH_elem")
            nearest_dist = mig.get("nearest_nonH_dist")
            artifact = (
                e_min < INTERMEDIATE_ARTIFACT_EV
                or nearest == "Fe"
                or (nearest_dist is not None and float(nearest_dist) < 1.1)
            )
            real_geom = (
                nearest == "S"
                and (nearest_dist is None or 1.30 <= float(nearest_dist) <= 1.50)
                and e_min >= -1.5
            )
            res.failure_mode = "intermediate_minimum"
            if artifact:
                res.confidence = "medium"
                res.reasons.append(
                    f"interior minimum {e_min:.3g} eV looks like an ARTIFACT "
                    f"(depth<{INTERMEDIATE_ARTIFACT_EV} eV or nearest=Fe / d<1.1 A)"
                )
                res.cheapest_disambiguating_test = "DFT single-point at the well image; check parity/spin if DFT, OOD if MLIP"
                res.do_not_do = ["do NOT trust an MLIP-only deep well as a real intermediate"]
                res.verification_followup = ["DFT-only re-evaluation of the well; recheck electron parity/spin"]
                res.gate_trace[-1] += ":MULTI_ENDPOINT_artifact"
                return finish("MULTI_ENDPOINT")
            if real_geom:
                res.confidence = "high"
                res.reasons.append(
                    f"interior minimum {e_min:.3g} eV below endpoints with H-S geometry: real intermediate"
                )
                res.cheapest_disambiguating_test = "relax the intermediate as a new endpoint; split into two NEB segments"
                res.verification_followup = ["two-segment NEB through the relaxed intermediate"]
                res.gate_trace[-1] += ":MULTI_ENDPOINT"
                return finish("MULTI_ENDPOINT")
            # well present but geometry unverified -> still multi-endpoint, lower confidence
            res.confidence = "medium"
            res.reasons.append(f"interior minimum {e_min:.3g} eV below endpoints; geometry unverified")
            res.cheapest_disambiguating_test = "relax the interior minimum and verify nearest non-H atom"
            res.gate_trace[-1] += ":MULTI_ENDPOINT_unverified"
            return finish("MULTI_ENDPOINT")
    res.gate_trace[-1] += ":pass"

    # --- Gate 7: roll-off (triple AND) --------------------------------------
    res.gate_trace.append("7_rolloff_triple_and")
    if rolloff_all:
        res.failure_mode = "degenerate_saddle_rolloff"
        res.confidence = "high"
        res.reasons.append(
            "roll-off triple-AND satisfied: perp_disp_ratio>0.5 AND nearest_anchor_switch AND "
            "f_frac_react>0.9 AND f_frac_metal<0.05 (band slides off the ridge toward a degenerate saddle)"
        )
        res.cheapest_disambiguating_test = (
            "$0 MLIP attractor-probe: relax the worst image with MLIP to see which basin it falls into "
            "BEFORE committing to FixedLine"
        )
        res.do_not_do = [
            "do NOT blame electronics/parity (force is on H+anchors, not metal)",
            "do NOT run a dimer (band is not near a single saddle)",
        ]
        res.verification_followup = ["FixedLine-constrained NEB along the endA-endB chord for the migrating atom"]
        res.gate_trace[-1] += ":FIXEDLINE_CONSTRAINT"
        return finish("FIXEDLINE_CONSTRAINT")
    res.gate_trace[-1] += ":pass"

    # --- Gate 8: trend (overshoot / diverge / stall) ------------------------
    res.gate_trace.append("8_trend")
    slope_flat = barrier_slope is not None and abs(barrier_slope) < eps_e
    perp_trending_down = (
        perp_m100 is not None and perp_m400 is not None and float(perp_m400) < float(perp_m100)
    ) or (perp_m400 is not None and float(perp_m400) < 1.5 * perp_target)
    overshoot = slope_flat and perp_trending_down and fmax_local_minima >= 2
    perp_up = (
        perp_m100 is not None and perp_m400 is not None and float(perp_m400) > float(perp_m100)
    )
    path_not_decreasing = (
        len(path_change_hist) >= 2 and _is_monotonic_up(path_change_hist[-W_MIN:])
    ) or (not path_change_hist)
    diverge = perp_up and path_not_decreasing
    stall = slope_flat and perp_invariant and not overshoot

    if overshoot:
        res.failure_mode = "transient_overshoot"
        res.confidence = "medium"
        res.reasons.append(
            "overshoot signature: flat barrier slope, perp residual trending down, "
            f">=2 local fmax minima ({fmax_local_minima}) with recovery -- keep iterating"
        )
        res.cheapest_disambiguating_test = "continue NEB and re-evaluate invariant residual after another W iters"
        res.do_not_do = [
            "do NOT switch optimizers on a transient fmax bounce",
            "do NOT run a dimer (path is still settling, energy already converged)",
        ]
        res.excluded = ["DIMER_SELLA / SWITCH_OPTIMIZER (ruled out by 8: overshoot, not stall/diverge)"]
        res.gate_trace[-1] += ":CONTINUE_NEB"
        return finish("CONTINUE_NEB")

    if diverge:
        res.failure_mode = "diverging_path"
        res.confidence = "medium"
        next_method = _next_untried(res.method_ladder)
        res.reasons.append(
            "diverge signature: perp residual monotonically up and path_change not decreasing; "
            f"escalate to next untried method: {next_method}"
        )
        res.cheapest_disambiguating_test = f"switch to {next_method} (cheapest untried rung of the ladder)"
        res.do_not_do = ["do NOT keep iterating the same diverging optimizer"]
        verdict = _ladder_to_verdict(next_method)
        res.gate_trace[-1] += f":{verdict}"
        return finish(verdict)

    if stall:
        res.gate_trace[-1] += ":stall->flat_top"
        # fall through to gate 9 (flat-top branch)
    else:
        res.gate_trace[-1] += ":no_clear_trend"

    # --- Gate 9: flat-top (+freq) -------------------------------------------
    res.gate_trace.append("9_flat_top")
    n_imag = freq.get("n_imaginary")
    omega_imag = freq.get("omega_imag_cm1")
    bond_type = freq.get("bond_type")
    if freq and n_imag is not None:
        n_imag = int(n_imag)
        if n_imag > 1:
            res.failure_mode = "ridge_multiple_imaginary"
            res.confidence = "high"
            res.reasons.append(
                f"n_imaginary={n_imag} (>1): the saddle is a ridge, the reaction coordinate is wrong"
            )
            res.cheapest_disambiguating_test = "re-pick endpoints / reaction coordinate; re-enumerate the path"
            res.do_not_do = ["do NOT report a barrier from a >index-1 saddle"]
            res.excluded = ["CONVERGED_REPORT_BARRIER (ruled out by 9: ridge, n_imag>1)"]
            res.gate_trace[-1] += ":REPICK_ENDPOINTS"
            return finish("REPICK_ENDPOINTS")
        if n_imag == 1:
            omega_stretch = _omega_stretch(bond_type)
            soft = omega_imag is not None and abs(float(omega_imag)) < FLAT_TOP_FRACTION * omega_stretch
            if soft:
                res.failure_mode = "flat_top_index1_confirmed"
                res.confidence = "high"
                res.reasons.append(
                    f"index-1 saddle with soft imaginary (|{float(omega_imag):.0f}| < "
                    f"{FLAT_TOP_FRACTION}*{omega_stretch:.0f} cm-1 for {bond_type or 'default'}): "
                    "flat-top TS CONFIRMED -- barrier is VALID"
                )
                res.cheapest_disambiguating_test = "DFT frequency at the saddle + ZPE correction"
                res.do_not_do = [
                    "do NOT run a dimer: index-1 is already confirmed, the barrier is valid",
                ]
                res.verification_followup = ["DFT freq (confirm single imaginary) + ZPE-corrected barrier"]
                res.excluded = ["DIMER_SELLA (ruled out by 9: index-1 flat-top confirmed, barrier valid)"]
                res.gate_trace[-1] += ":CONVERGED_REPORT_BARRIER"
                return finish("CONVERGED_REPORT_BARRIER")
            # single imaginary, not soft -> a clean index-1 TS as well
            res.failure_mode = "index1_saddle_confirmed"
            res.confidence = "high"
            res.reasons.append(
                f"single imaginary mode ({float(omega_imag) if omega_imag is not None else 'n/a'} cm-1): "
                "index-1 saddle confirmed, barrier valid"
            )
            res.cheapest_disambiguating_test = "DFT frequency at the saddle + ZPE correction"
            res.verification_followup = ["DFT freq + ZPE-corrected barrier"]
            res.do_not_do = ["do NOT run a dimer: index-1 already confirmed"]
            res.gate_trace[-1] += ":CONVERGED_REPORT_BARRIER"
            return finish("CONVERGED_REPORT_BARRIER")
    # no freq available: dimer ONLY if proximity gate passes
    if stall:
        proximity_ok = barrier_converged and (path_change_mean is not None and path_change_mean < perp_target)
        if proximity_ok:
            res.failure_mode = "stall_near_saddle_no_freq"
            res.confidence = "medium"
            res.reasons.append(
                "stall near a stable saddle (barrier stable, path_change small): "
                "dimer/Sella proximity gate satisfied"
            )
            res.cheapest_disambiguating_test = "seed Sella/dimer from the tangent-curvature soft mode at the climb image"
            res.do_not_do = ["do NOT run a bare dimer far from a saddle (it diverges to the wrong saddle)"]
            res.verification_followup = ["confirm index-1 via DFT/MLIP Hessian after Sella/dimer"]
            res.gate_trace[-1] += ":DIMER_SELLA"
            return finish("DIMER_SELLA")
        res.failure_mode = "stall_no_freq_proximity_unknown"
        res.confidence = "medium"
        res.reasons.append(
            "stall at floor but no frequency and dimer proximity gate not satisfied: "
            "verify the saddle with a frequency calculation before any dimer"
        )
        res.cheapest_disambiguating_test = "compute a (cheap MLIP) Hessian / frequency at the climb image first"
        res.do_not_do = ["do NOT launch a dimer without a confirmed near-saddle proximity gate"]
        res.verification_followup = ["MLIP or DFT frequency at the climb image, then re-route"]
        res.gate_trace[-1] += ":freq_verification_first"
        return finish("INSUFFICIENT_DATA")
    res.gate_trace[-1] += ":pass"

    # --- default fall-through -----------------------------------------------
    res.failure_mode = "no_gate_fired"
    res.confidence = "low"
    res.reasons.append("no gate fired on the available signals; gather more invariant metrics")
    res.cheapest_disambiguating_test = "dump perp residual at M=100 and M=400 + dense-spline barrier history"
    return finish("INSUFFICIENT_DATA")


def _ladder_to_verdict(method: str | None) -> str:
    mapping = {
        "FIRE": "SWITCH_OPTIMIZER",
        "NEBOptimizer_ode": "SWITCH_OPTIMIZER",
        "LBFGS": "SWITCH_OPTIMIZER",
        "k_spring_up_plain_to_CI": "STIFFEN_SPRINGS_PLAIN_TO_CI",
        "FIXEDLINE_CONSTRAINT": "FIXEDLINE_CONSTRAINT",
        "STRING_METHOD": "STRING_METHOD",
        "DIMER_SELLA": "DIMER_SELLA",
    }
    if method is None:
        return "STRING_METHOD"
    return mapping.get(method, "SWITCH_OPTIMIZER")


def _fill_barrier_status(
    res: AdvisorResult,
    energy_converged: bool,
    fmax_converged: bool,
    barrier_eV: float | None,
    eps_e: float,
) -> None:
    paper_gradeable = bool(energy_converged and fmax_converged and res.verdict == "CONVERGED_REPORT_BARRIER")
    if paper_gradeable:
        quality = "paper_grade"
    elif res.verdict in {"FIXEDLINE_CONSTRAINT", "CONSTRAINED_M_NEB"}:
        quality = "constrained_upper_bound"
    else:
        quality = "estimate"
    reason = (
        "energy + invariant residual converged"
        if paper_gradeable
        else f"verdict={res.verdict}; energy_converged={energy_converged}, fmax_converged={fmax_converged}"
    )
    res.barrier_status = {
        "energy_converged": bool(energy_converged),
        "fmax_converged": bool(fmax_converged),
        "barrier_eV": barrier_eV,
        "quality": quality,
        "paper_gradeable": paper_gradeable,
        "reason": reason,
    }


# --- human-readable output --------------------------------------------------
def print_advice(envelope: dict) -> None:
    result = envelope.get("result") or {}
    print(f"verdict\t{envelope.get('verdict')}")
    print(f"confidence\t{envelope.get('confidence')}")
    print(f"failure_mode\t{result.get('failure_mode') or '-'}")
    bs = result.get("barrier_status") or {}
    print(f"barrier_eV\t{_fmt(bs.get('barrier_eV'))}")
    print(f"quality\t{bs.get('quality', '-')}")
    print(f"paper_gradeable\t{bs.get('paper_gradeable')}")
    if result.get("expected_barrier_range_meV") is not None:
        print(f"expected_range_meV\t{result['expected_barrier_range_meV']}\tout_of_range={result.get('out_of_range')}")
    print(f"cheapest_disambiguating_test\t{result.get('cheapest_disambiguating_test') or '-'}")
    for gate in result.get("gate_trace", []):
        print(f"gate\t{gate}")
    for reason in envelope.get("reasons", []):
        print(f"reason\t{reason}")
    for action in result.get("do_not_do", []):
        print(f"do_not_do\t{action}")
    for step in result.get("verification_followup", []):
        print(f"verify\t{step}")
    print("method_ladder:")
    for row in result.get("method_ladder", []):
        mark = "tried" if row["already_tried"] else f"rank {row['rank']}"
        print(f"  {row['method']}\t{mark}\t{row['applicability_gate']}")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, help="Path to a JSON case dict of NEB signals")
    parser.add_argument("--json", action="store_true", help="Print MCP-shaped JSON envelope")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)

    case: dict = {}
    if args.case_json:
        case = json.loads(Path(args.case_json).read_text(encoding="utf-8"))

    envelope = run_neb_method_advisor(case)

    if args.output:
        dump_json(envelope, args.output)
    if args.json:
        dump_json(envelope)
    else:
        print_advice(envelope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
