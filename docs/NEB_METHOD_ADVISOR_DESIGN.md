# NEB Method Advisor — Design Spec (chemist+physicist OPUS consilium)

**Purpose.** A top-level ROUTER that ingests an NEB calculation's signature (band +
optimizer history + geometry + optional freq/magnetic) and recommends the right
METHOD FAMILY (continue-NEB / stiffen / switch-optimizer / FixedLine / string /
dimer-Sella / MECP / constrained-M / re-pick / multi-endpoint / fix-parity /
report-converged), encoding the cross-mineral failure-mode lessons so
the framework advises itself instead of relying on human recall.

Tool: the `neb_method_advisor` module (CLI + `--json`, `response_envelope`).

## Core principle (both reviewers)
The advisor is a ROUTER, NOT a re-solver. It does NOT re-decide what an upstream
gate already decided. Hard gate ordering, each delegating to the existing tool:

```
0. data sufficiency      -> INSUFFICIENT_DATA if n_iter<W, no coords, etc.
0b. STALE_ENERGY_LOG     -> std(node_E across iters)<1e-9 while path_change>0
1. M0 electron parity    -> electron_parity_gate; odd+nspin1 -> FIX_SPIN_PARITY_FIRST
2. NEB applicability      -> Grotthuss/solvation hint -> NEB_INAPPLICABLE (AIMD)
3. magnetic              -> magnetic_recommendation; sheet split -> MECP / CONSTRAINED_M
4. same-basin (Test A)   -> h_disp<1.0 & same nearest-host & dE<5*noise -> REPICK_ENDPOINTS
5. gauge-invariant conv  -> path_convergence.invariant_converged -> CONVERGED_REPORT_BARRIER
6. profile (intermediate)-> well below endpoints? real/artifact/detour
7. roll-off (triple AND) -> degenerate vs wrong-RC -> FixedLine / $0-probe-first
8. trend (overshoot/stall/diverge) -> method ladder (cheapest untried)
9. flat-top (+freq)      -> index-1 confirmed -> barrier VALID; else dimer-near-saddle
```
If any earlier gate fires, later ones are NOT evaluated. Default verdict =
`INSUFFICIENT_DATA`. Every recommendation carries confidence + excluded alternatives.

## Signatures & thresholds (gauge-invariant; physicist)
All convergence judged on INVARIANT metrics from the `path_convergence` module
(`barrier_on_curve` dense-spline, `perp_residual_on_curve`, `path_change`), NOT
node-fmax. node-fmax = weak corroborating signal only, with gauge-dependence warning.

- **barrier_converged**: `max(barrier_dense[-W:]) - min(barrier_dense[-W:]) < eps_E`,
  `W>=15`, `eps_E = max(5*SCF_noise, 0.01 eV)` (SCF_noise scales with N_atoms;
  conv_thr 1e-7 Ry ~ 1e-5 eV total -> barrier floor ~0.01 eV for ~96 atoms).
- **interpolation-floor vs real residual (M-sweep)**: run `perp_residual_on_curve` at
  M=100 and M=400. floor drops with M -> interpolation artifact (CONVERGED). floor
  M-invariant & >target -> real residual (flat-top OR undersaddle -> freq check).
- **roll-off (triple AND, all required)**:
  (a) `perp_disp_ratio = d_perp(H_worst from endA-endB chord) / |r_H(endB)-r_H(endA)| > 0.5`;
  (b) `nearest_anchor_switch`: worst-image nearest non-H not in {endA_anchor, endB_anchor};
  (c) `force_localization`: F_frac(H+anchors) > 0.9 AND F_frac(metal) < 0.05
      (else force on metal -> electronic/parity, NOT roll-off).
- **overshoot vs stall vs diverge** (trend over W>=15, energy-first):
  - overshoot: slope(barrier_dense,W)~0 AND median(perp_resid_inv,W) trending down OR <1.5*target
    AND >=2 local fmax minima with recovery -> CONTINUE_NEB (NO method switch).
  - stall: slope(barrier_dense,W)~0 AND perp_resid_inv flat-at-floor over W AND M-invariant.
  - diverge: perp_resid_inv monotonic UP over W AND path_change NOT decreasing.
- **flat-top TS (chemist, RELATIVE threshold)**: requires a saddle Hessian (freq).
  `is_flat_top = |omega_imag| < 0.4 * omega_stretch(bond_type)`:
  S-H ~2500 -> <~600i; Fe-H ~1800 -> <~400i; O-H ~3000 -> <~700i.
  NOT absolute 300. Heavy-atom bond-breaking (S-S/Fe-S): soft imaginary is NORMAL,
  flat-top concept N/A. `n_imaginary==1` required for TS; >1 -> ridge -> re-pick RC.
  **soft imaginary + index-1 confirmed => barrier VALID, dimer NOT needed**
  (recommend report + ZPE; dimer only if index-1 NOT yet confirmed AND near-saddle).
- **intermediate well (chemist)**: interior `E_rel[i] < min(E_end) - 0.1 eV` (>3kT@298),
  spanning >=2 images, with `d(H-S) in [1.30,1.50]`, nearest non-H = S, depth > -1.5 eV
  -> REAL intermediate -> MULTI_ENDPOINT. depth < -2 eV OR nearest=Fe OR d(H-S)<1.1
  -> ARTIFACT (MLIP -> DFT-only; DFT -> check parity/spin). Above endpoints -> detour, NOT intermediate.
- **MLIP path-broken (chemist)**: judge on MID-BAND vs endpoints (endpoints can be sane);
  MACE & CHGNet break at DIFFERENT sites/geometries -> OOD artifact; same deep well same
  geometry -> possibly real intermediate (M-3).

## Method ladder (physicist; cheapest untried first; needs `methods_already_tried`)
`FIRE -> NEBOptimizer(ode)/LBFGS -> k_spring up + plain->CI -> FixedLine ->
string/growing-string -> dimer/Sella`. Each with applicability gate.
**Dimer proximity gate (mandatory)**: barrier_dense stable over W AND path_change<tol
AND seed mode available (soft imaginary from MLIP Hessian or tangent curvature).
Far-from-saddle -> dimer diverges to wrong saddle. Sella preferred over bare dimer
(eigenvector-following, robust on flat-top). Asymmetric/bad-guess -> growing/freezing
string before dimer. Spin-crossing -> MECP (not dimer).

## Output (response_envelope.result)
- `verdict`: one of CONTINUE_NEB / STIFFEN_SPRINGS_PLAIN_TO_CI / SWITCH_OPTIMIZER /
  FIXEDLINE_CONSTRAINT / STRING_METHOD / DIMER_SELLA / MECP / CONSTRAINED_M_NEB /
  REPICK_ENDPOINTS / MULTI_ENDPOINT / FIX_SPIN_PARITY_FIRST / NEB_INAPPLICABLE /
  CONVERGED_REPORT_BARRIER / STALE_ENERGY_LOG / INSUFFICIENT_DATA.
- `confidence` low/medium/high (high requires multi-source corroboration).
- `failure_mode` + `excluded` (list of ruled-out modes with the gate that ruled them out).
- `gate_trace` (ordered gates consulted + each verdict).
- `signature_metrics` (ALL numbers incl perp_resid at M=100&400, barrier_drift_W,
  path_change_mean, perp_disp_ratio, nearest_anchor_switch, force_localization, n_iter, W,
  node_fmax_floor[gauge-dep flag]).
- `barrier_status`: {energy_converged, fmax_converged, barrier_eV, quality
  (paper_grade|constrained_upper_bound|estimate), paper_gradeable, reason}.
- `expected_barrier_range`: cross-mineral triangulation -> validation flag if out-of-range.
- `cheapest_disambiguating_test`: next $0/cheap step (e.g. $0 CHGNet attractor-probe for
  roll-off; $0.5 nspin=2 single-point for frozen-energy; spin-free MLIP for suspected magnetic).
- `method_ladder` (ranked, with applicability_gate each), `do_not_do`, `verification_followup`.

## Regression tests (acceptance — must pass)
1. pyrite (frozen 12-digit, nspin=1 odd-e) -> FIX_SPIN_PARITY_FIRST or STALE_ENERGY_LOG,
   NOT flat-top/dimer.
2. pyrite (warm-start+constraint, fmax 1.0->0.5->0.37 overshoot->0.17, energy converged)
   -> CONTINUE_NEB or CONVERGED_REPORT_BARRIER, NOT DIMER.
3. marc (magnetic sheet crossing) -> MECP / CONSTRAINED_M_NEB via magnetic delegation,
   NOT spring tuning.
Plus unit tests per gate + INSUFFICIENT_DATA default + stale-log detector + roll-off triple-AND.

## Delegation map (reuse, do not duplicate)
electron_parity_gate.run_electron_parity_gate | magnetic_recommendation.build_recommendation
| neb_agm_prototype._classify + same-basin gate | path_convergence.* (+ M-sweep wrapper)
| adaptive_neb_planner.run_adaptive_neb_planner (param-tuning branch).

Sources: chemist+physicist OPUS consilium; NEB_STALL_DIAGNOSTIC_PLAYBOOK.md;
3 regression cases; the `path_convergence` module (gauge-invariance).
