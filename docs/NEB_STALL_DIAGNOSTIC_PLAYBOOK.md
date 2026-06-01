# NEB-Stall Diagnostic Playbook

**What:** a reproducible procedure for diagnosing "DFT NEB is stuck — why and how to fix it." Separates three distinct root causes (optimizer/spring · wrong endpoints/multi-site · magnetic ambiguity) using cheap ($0) steps BEFORE repeating an expensive DFT run.

**When to apply:** NEB is not converging — fmax is stuck at a floor (well above target) and/or **trending upward**, the energy of the top image is frozen, BFGS/FIRE cycles for hundreds of steps without energy descent.

**Validated on:** pyrite V_Fe (root cause = spring/optimizer) and marcasite V_Fe (root cause = wrong endpoints + magnetic mismatch). Two opposing diagnoses produced by a single procedure.

---

## Procedure (4 steps, cheap to expensive)

### Step 1 — Analyze `neb.traj`: where does the force live? (free, local)

Module: the `neb_stuck_analysis` gate. What it computes:
- per-image E_rel + fmax (which image is stuck),
- **per-atom force breakdown on the worst image** (which atoms carry the residual force),
- nearest neighbors of the migrating atom.

Interpreting the force distribution:

| Force concentrated on... | Likely root cause |
|---|---|
| Migrating atom (H) + anchor (S) → metal < 1% | **path geometry / spring** (NOT electronics) |
| Metal / spread across the lattice | Possibly **electronics** (nspin/U/smearing) |

> Lesson: for pyrite the force was H 52% + S 48%, **Fe 0.8%** → hypothesis about nspin/V_Fe hole was immediately ruled out. Measure, don't guess.

### Step 2 — Compare the initial IDPP path with the final band

Module: the `neb_path_geom` gate (reads the first `n_movable` frames of traj = IDPP, the last frames = final).
- **IDPP clean, final = mess WITHOUT energy descent** → band "rolled off the ridge" = optimizer/spring issue. Proceed to Step 4 → expect clean MLIP.
- **IDPP already bad** (migrating atom disturbing neighbors) → path/endpoint problem from the outset.

### Step 3 — Endpoints: symmetry + magnetic state

- **Geometry:** non-H displacement endA→endB (MIC, without relabeling or Hungarian matching, see the `symmetry_preflight_general` gate). < 0.15 Å → nearly mirror-image pair; large displacement → different sites.
- **Magnetism (if nspin=2):** `grep "total magnetization\|absolute magnetization\|^!" sp_end{A,B}/espresso.pwo`. **Different total/abs mag at endpoints = magnetic-state mismatch (condition C2)** → NEB interpolates a spin flip → will not converge. Check starting_magnetization and presence of +U.

### Step 4 — $0 MLIP validation: MACE + CHGNet NEB on the SAME DFT endpoints

Module: the `neb_vfe_validate_mlip` runner (env `MLIP_MODEL=mace|chgnet`, `OUT_TAG`, `MINERAL_NAME`). Protocol: IDPP(mic) → plain NEB (k=1.0) → CI-NEB, FIRE.
For a new mineral — extract endpoints (the `extract_endpoints` helper as template) and adapt `OUT_TAG`.

Why MLIP specifically: cheap ($0, ~1–3 min on local GPU), and **MACE/CHGNet do not model spin** → separates path geometry from magnetism. Cross-checking two MLIPs catches OOD artifacts of either model (e.g. pent MACE 30 eV artifact).

---

## Step 4 interpretation table

| MLIP result | Diagnosis | What to do |
|---|---|---|
| Both converge cleanly, barrier is physical, ΔE_endpoints matches DFT, endpoints = minima | **Path OK, root cause = spring/optimizer** | k_spring ×5–10 (0.3→1.5–3.0), plain→CI, `NEBOptimizer(ode)`/LBFGS instead of FIRE. (= pyrite) |
| Both give band-collapse (intermediates below endpoints), barrier_fwd=0 | **Endpoints NOT true minima / multi-site** | L2 multi-endpoint enumeration → find real minima → reselect the pair. (= marcasite) |
| Both reproduce endpoint asymmetry ΔE (spin-free!) | **Asymmetry is geometric**, not magnetic | Different sites — this is real, not a spin artifact |
| ΔE_endpoints(MLIP)≈0, but DFT≫0 + different mag (Step 3) | **Asymmetry is magnetic** (C2) | Use identical starting_mag on all images; add +U; (spin-aware MLIP — research direction) |
| MACE and CHGNet STRONGLY disagree (×10+) | **OOD artifact of one MLIP** | Trust the agreement, not the magnitude; run L4 DFT single-point |

---

## Two worked case studies (reference examples)

### Pyrite V_Fe — root cause: spring/optimizer ✅ resolved
- Step 1: force H 52% + S 48%, Fe 0.8% → not electronics.
- Step 2: IDPP ideal (symmetric hop, 0.628 eV), final = mess without E descent → ridge-rolling.
- Step 4: MACE **182 meV**, CHGNet **223 meV** — both converge in 29–68 steps, in agreement, within the predicted 150–400 meV range.
- **Fix:** k 0.3→1.5–3.0 + plain→CI. Expected DFT barrier ~200–300 meV.

### Marcasite V_Fe — root cause: wrong endpoints + magnetic mismatch
- Step 3: endA/endB show different mag (1.67/2.56 vs 1.13/1.91 μB), non-H displacement 0.13 Å (lattices identical), but ΔE=174 meV.
- Step 4: MACE −198 / CHGNet −103 meV (both spin-free reproduce the asymmetry → it is GEOMETRIC); both show band-collapse (image7 −601/−364 below endB) → endpoints are not true minima.
- **Fix:** L2 multi-endpoint enumeration → reselect pair + magnetic consistency (+U=2). NOT the pyrite recipe.

---

## Module registry

| Module | Purpose |
|--------|----------|
| the `neb_stuck_analysis` gate | Step 1 — per-image E/fmax + per-atom force breakdown from neb.traj |
| the `neb_path_geom` gate | Step 2 — IDPP vs final band, geometry of migrating atom |
| the `extract_endpoints` helper | extract endA/endB from QE `.pwi`/`.pwo` + compare geometry (template for a new mineral) |
| the `neb_vfe_validate_mlip` runner | Step 4 — MACE/CHGNet NEB on DFT endpoints (parameterized by `OUT_TAG`/`MINERAL_NAME`/`MLIP_MODEL`) |

**Related:** the universal "NEB band rolls off ridge" lesson, `EVIDENCE_FRAMEWORK_V2_COMPLETE.md` (L0–L6), the same-basin endpoint lesson (different failure mode — same-basin).

---

*Created from the diagnostic work on pyrite + marcasite V_Fe NEB.*
