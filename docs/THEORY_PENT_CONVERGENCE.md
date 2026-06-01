# Sufficient Conditions for NEB Convergence to TRUE MEP

## Formal checklist applied to pentlandite V_Fe pre-flight

**Task:** P0-D (ROADMAP NEB-GT theory §1.4)
**Type:** formal restatement of existing theorems + pent-specific application
**Style:** theorem-style statements, sparse prose, math-first
**Cross-refs:** does not duplicate `GAME_THEORETIC_NEB_FOUNDATIONS.md` (game-theoretic angle treated separately).

---

## §1. Variational setup

### §1.1. NEB as constrained finite-dimensional minimization

Let V: ℝ^{3N_atoms} → ℝ be the DFT PES of a system of N_atoms nuclei. Let x_A, x_B ∈ ℝ^{3N_atoms} be two endpoint minima (∇V(x_A) = ∇V(x_B) = 0, Hessian positive-definite at both).

Discrete NEB band — an ordered chain of N_img images x = (x_1, ..., x_N_img), x_1 ≡ x_A, x_N_img ≡ x_B (or closed-end relaxation). Standard NEB objective:

```
Φ(x_1, ..., x_N_img) = Σᵢ V(x_i) + (k_spring/2) Σᵢ ||x_{i+1} − x_i||²
```

NEB force on image i — projection of −∇Φ onto the direction perpendicular to the band tangent:

```
F_i^NEB = −∇V(x_i)|_⊥ + F_i^spring|_∥
        = −[∇V(x_i) − (∇V(x_i) · τ̂_i) τ̂_i] + k_spring (||x_{i+1} − x_i|| − ||x_i − x_{i−1}||) τ̂_i
```

where τ̂_i is the unit tangent (Henkelman-Jónsson 2000 upwind formula).

[FACT] Stationary states of the NEB force (F_i^NEB = 0 ∀ i) are points where ∇V(x_i) lies along τ̂_i (band on a gradient flow line) and spring tensions are balanced. This is the **discrete approximation of the MEP**: the continuous curve γ(s) along which ∇V(γ(s)) ∥ γ'(s).

[FACT] Climbing-image NEB (CI-NEB, Henkelman-Uberuaga-Jónsson 2000): for index i* with locally maximal V, the inverted force F_i*^climb = −∇V(x_i*) + 2(∇V(x_i*) · τ̂) τ̂ drives the image toward the exact saddle.

### §1.2. Mountain Pass Theorem — existence of saddle

**Theorem (Ambrosetti-Rabinowitz 1973, J Funct Anal 14:349-381):** Let X be a Banach space, Φ ∈ C¹(X, ℝ) satisfying the Palais-Smale condition (PS). Suppose there exist e ∈ X, r > 0 such that:

(MP1) ||e|| > r;
(MP2) max(Φ(0), Φ(e)) < inf_{||x||=r} Φ(x) ≡ b.

Then c = inf_{γ ∈ Γ} max_{t ∈ [0,1]} Φ(γ(t)) ≥ b is a critical value of Φ. Here Γ is the set of continuous paths from 0 to e.

[FACT] Applied to NEB: x_A and x_B are two local minima of V. The MP theorem guarantees the existence of a critical point x_TS (saddle) with V(x_TS) ≥ max(V(x_A), V(x_B)), provided (PS) holds and the potential is coercive. This is an **independent existence proof of the MEP saddle**, separate from the potential-game framing (§6.1 in `GAME_THEORETIC_NEB_FOUNDATIONS.md`).

[CLAIM, Jabri 2003 monograph "The Mountain Pass Theorem", Cambridge] Generalizations to the Mountain Pass of Second Order (uniqueness) require additional conditions: Φ ∈ C², saddle non-degenerate (Morse index 1), no symmetry-related degeneracies. These conditions become practical test items below (§2.3–2.4).

### §1.3. Connection with potential games (independent existence)

**Theorem (Monderer-Shapley 1996, Games Econ Behav 14:124-143):** The NEB objective Φ is a potential function for a game where players = images, strategies = positions, payoff u_i = −Φ projected onto player i. Pure Nash equilibrium ⇔ stationary point of Φ. Existence is guaranteed if the strategy space is compact and Φ is continuous.

[FACT] This provides a **second independent existence proof** of a stationary NEB band. Details — `GAME_THEORETIC_NEB_FOUNDATIONS.md` §6.1.

**Critical:** existence of a saddle ≠ NEB converges to the TRUE saddle. NEB may converge to a **wrong critical point** (alternate saddle, degenerate minimum manifold, artifact). This motivates §2.

---

## §2. Sufficient conditions for convergence to TRUE MEP

Seven conditions, ranked by severity of violation in our pent setup.

### §2.1. C1 — Coercivity / boundedness

**Formal statement:**
V: ℝ^{3N} → ℝ is coercive: V(x) → +∞ as ||x|| → ∞, OR the domain is compact (periodic boundary conditions). Equivalent: sublevel sets {x: V(x) ≤ c} are compact ∀ c < ∞.

**Source:** Boyd & Vandenberghe 2004 "Convex Optimization" §3.1.3 (Cambridge). Standard in variational analysis.

**Practical test for pent:**
- [FACT] DFT with PBC trivially gives a compact configuration space modulo discrete translations and permutations. C1 satisfied trivially.
- Subtle violation: if the supercell is too small → vacancy interacts through PBC image → effective potential periodicity rather than localized basin. Pent V_Fe 136-atom 3×3×3 supercell: V_Fe–V_Fe periodic distance > 11 Å → interaction negligible (per supercell calibration).

**Failure mode:** None expected for pent. Marginal only if N_atoms < 50 (V_Fe defect leakage).

### §2.2. C2 — Lipschitz continuity of gradient (smoothness)

**Formal statement:**
There exists L > 0 such that ||∇V(x) − ∇V(y)|| ≤ L ||x − y|| for all x, y in a neighborhood of the path. Equivalent: V ∈ C^{1,1}_loc, Hessian operator bounded in operator norm.

**Source:** Nocedal & Wright 2006 "Numerical Optimization" 2nd ed Ch 3 §3.2 + Ch 4 §4.1 (Springer). Standard assumption for first-order optimizer convergence theorems (gradient descent, BFGS line-search, FIRE).

**Practical test for pent:**
- [QUESTION] The DFT PES is C^∞ smooth for a fixed Born-Oppenheimer surface IF the electronic SCF has converged. Cusps arise only at nuclear coincidences (||x_i − x_j|| → 0), which are not reached in physical configurations.
- Subtle violations:
  1. **Spin-state transitions:** discontinuous Hessian at level crossings of magnetic configurations (Wang 2019 for greigite; NM collapse observed in marcasite). Pent V_Fe: AFM/FM/NM manifolds may join non-smoothly if nspin=2 is used with perturbed starting magnetizations.
  2. **Spurious SCF convergence to an excited state:** one image converges to a different electronic state → V(x) discontinuous as a function of x. Mitigation: tight SCF tolerance (conv_thr ≤ 1e-9), force monotonic magnetization annealing.
- Practical test: compute the numerical Hessian via finite differences in the neighborhood of an endpoint (radius ~0.1 Å). If the eigenvalue spectrum changes discontinuously under a small perturbation → C2 violation suspected.

**Failure mode:** BFGS line search rejects steps, fmax oscillates without descending, ||∇V|| jumps between iterations. Manifests as the iteration counter increasing without energy descent (observed in the pent BFGS forensic iter 14 signature).

### §2.3. C3 — Endpoint isolation (anti-same-basin)

**Formal statement:**
Let H_A = ∇²V(x_A), λ_min^A = smallest positive eigenvalue. Basin radius (Morse) R_basin^A ~ ||∇V(x)|| / λ_min^A where ∇V is approximately linear. **Condition:**

```
||x_A − x_B|| > 2 max(R_basin^A, R_basin^B)
```

i.e. endpoints separated by more than the sum of their basin radii.

**Source:** Folklore in optimization (Nocedal-Wright Ch 3.4, "region of attraction" definition); formalized in Liu-Chen-Ortner 2022 (arXiv:2204.07467) for discrete MEPs as stationary states of NEB optimizer.

**Practical test for pent (THIS IS OUR MAIN CHECK):**
- Compute endpoint Hessian eigenvalues at x_A, x_B (BFGS-derived approximation from endpoint relax is sufficient — no full ∇² evaluation needed).
- Element-aware displacement check (P0-B Layer 1): max(|x_A^{atom} − x_B^{atom}|) for each element.
  - Threshold for pent V_Fe + H migration: H displacement > 0.5 Å, OR ≥2 heavy atoms displaced > 0.5 Å → HIGH confidence different basins.
  - Threshold violated → P0-A PH layer required.
- Cheap MLIP NEB pre-screening (P0-C): one-shot MACE-MP-0 NEB. Compute path-integrated curvature, MACE force RMS at endpoints. If both endpoints MACE-relax to structures with d_MACE(x_A^MACE, x_B^MACE) < 0.3 Å OR identical PH signature → ALERT.

**Failure mode (our empirical signature, multiple past incidents):**
- E_endA ≈ E_endB to within SCF noise (< 5 meV)
- Tangent τ̂_i degenerates (numerically zero or random direction)
- NEB forces never decrease below floor ~0.3 eV/Å
- BFGS / FIRE iterations oscillate, fmax plateau
- **Real cost:** $30–100 per occurrence × 3 incidents = $200–300 burned.

C3 enforcement is a **mandatory pre-flight gate** in the 8-step RC-check protocol (see the project decision log).

### §2.4. C4 — Initial path in convergence basin of TRUE MEP

**Formal statement:**
Let M* = TRUE MEP (manifold of points satisfying ∇V|_⊥ = 0 along the correct connecting curve). The initial NEB path x^{(0)} = (x_1^{(0)}, ..., x_N^{(0)}) must lie within neighborhood B(M*, ε) where the iterative NEB optimizer is contracting — i.e. ||x^{(k)} − M*|| decreasing monotonically.

**Source:**
- Liu, Chen, Ortner 2022 arXiv:2204.07467 ("Convergence of the Discrete Minimum Energy Path"): under coercivity + smoothness + isolation of MEP, NEB discrete path converges to continuous MEP with optimal rate O(1/N_img²) as N_img → ∞. **BUT the theorem assumes the initial guess is close to the limit.**
- Smidstrup-Pedersen-Stokbro-Jónsson 2014 J Chem Phys 140:214106 (IDPP method): characterizes when linear interpolation fails (steep gradient walls, multiple basins between endpoints).

**Practical test for pent:**
- **Cheap MLIP NEB pre-screening (P0-C):** run MACE-MP-0 NEB on 9 images starting from a linear OR IDPP initial guess. Inspect:
  1. Saddle position location (image with max V) — does it correspond to the middle of the Fe-S coordination hop?
  2. Path topology (number of inflection points, monotonicity of V along path) — single saddle, not multi-step?
  3. Energy variance along path < 2 max(barrier estimate) — no extreme excursions through unphysical configurations.
- **Persistent Homology Layer 2 (deferred):** if P0-C is ambiguous, sample the local PES at 3–5 internal images, compute PH signatures, bottleneck distances between neighbors — should grow monotonically as |i − j| increases (single-basin transition each step).

**Failure mode:**
- IDPP linear interpolation crosses an Fe nucleus → SCF on early iterations spikes to V > V(x_TS_real), F_perp tilts catastrophically.
- Multi-basin path: NEB converges to the **first-saddle Nash equilibrium**, missing the global path. Same-basin signature if the initial path lies entirely within one basin (a marcasite R3 V_S+H case confirmed retrospectively).
- **Our 3-incident scoreboard:**
  - mackinawite: same-basin V_S+H pair S_42↔S_35 (RC choice failure)
  - pentlandite: BFGS iter 14 forensic, geometry within basin throughout
  - marcasite: spglib confirmed all 64 S in single Pnnm 4g orbit

### §2.5. C5 — Spring stiffness regime

**Formal statement:**
For finite k_spring, the NEB band lies on a **biased manifold**: image distribution is uniform only in the limit k_spring → ∞, AND the saddle approximation degrades if k_spring → 0 (band collapses to the endpoint pair).

Henkelman-Jónsson 2000 J Chem Phys 113:9978-9985 gives the empirical optimal range:

```
k_spring ∈ [0.5, 1.0] eV/Å²  (for typical molecular & condensed-matter NEB)
```

**Source:**
- Henkelman-Jónsson 2000 — foundational NEB paper. §III practical recommendations.
- Sheppard-Terrell-Henkelman 2008 J Chem Phys 128:134106 — comprehensive optimizer comparison; consistent k_spring sensitivity.
- Maragakis-Andreev-Brumer 2002 J Chem Phys 117:4651 — adaptive spring lengths, demonstrates that k_spring underestimation causes path collapse.

**Practical test for pent:**
- Use k_spring = 0.1 eV/Å² per Ruttinger-Sharma-Clancy 2022 JCTC 18:2993 protocol (n=226 reactions, validated), OR k_spring = 0.5 eV/Å² per QE precedent. Both within the "safe" range.
- Sensitivity probe: in the MLIP pre-screen, run two k values (0.1 and 0.5), check that the saddle estimate is consistent (ΔE_saddle < 50 meV). If divergent → C5 violation suspected.

**Failure mode:**
- k_spring too low → all images collapse toward an endpoint (band shrinks), saddle missed.
- k_spring too high → band too rigid to follow PES curvature, saddle position biased toward the straight-interpolation midpoint.
- Manifests as: spring forces dominate F_perp throughout iteration; image spacing varies wildly.

### §2.6. C6 — Optimizer step-size / Lipschitz-compatible learning rate

**Formal statement:**
For BFGS line search: each accepted step satisfies the Wolfe conditions, ensuring sufficient descent: Φ(x^{k+1}) ≤ Φ(x^k) − c · ||p^k||² for some c > 0 (where p^k = step direction). For FIRE: time-step Δt < 2/√(L) where L is the Lipschitz constant.

**Source:** Nocedal-Wright 2006 Ch 3 (BFGS line search, Wolfe conditions) + Bitzek-Koskinen-Gähler-Moseler-Gumbsch 2006 PRL 97:170201 (FIRE).

**Practical test for pent:**
- Monitor the BFGS step rejection rate in early iterations. Rejection rate > 30% over the first 20 steps → step-size too large relative to L (try maxstep 0.1 → 0.05 Å).
- For FIRE: if dt_max never reaches steady-state OR FIRE keeps shrinking dt every few steps → L estimate too small (system stiffer than expected; switch to BFGS).
- **For pent specifically:** the pent BFGS forensic iter 14 had max_step=0.10 Å + line_search=Wolfe. Recommendation: keep these settings and monitor over the first 30 iterations (mandatory expert QA review pre-deploy).

**Failure mode:**
- Step rejected → backtrack → tiny step → no progress → fmax plateau at ~0.2–0.5 eV/Å (looks like a convergence issue, actually an optimizer issue).
- Or: huge step accepted → energy spike → next iteration reverts → oscillation.

### §2.7. C7 — CI image identification and saddle isolation

**Formal statement:**
The climbing image i* must be located at the TRUE saddle, not on a shoulder. **Necessary condition:** ∇²V(x_{i*})|_{τ̂} < 0 (negative curvature along the reaction direction) AND ∇²V(x_{i*})|_⊥ ≥ 0 (positive curvature in all other directions). I.e. Morse index 1.

**Source:** Henkelman-Uberuaga-Jónsson 2000 J Chem Phys 113:9901 (CI-NEB original). Verification through phonon analysis or finite-difference Hessian at the converged image.

**Practical test for pent:**
- Post-NEB single saddle Hessian via ASE `vibrations.Vibrations` or finite differences (~6N_atoms × 2 SCF — expensive but mandatory for a paper-grade saddle).
- Expected: exactly one imaginary mode (negative eigenvalue), with mode displacement vector ∥ τ̂ at the saddle.
- If 2+ imaginary modes → shoulder, not true TS. If 0 imaginary modes → false-positive CI (image actually a local minimum on the band).
- **Cheap proxy during NEB:** monitor `(∇V · τ̂) / ||∇V||` at the climbing image. Should approach 1.0 (force vector aligned with band tangent) as iterations proceed. If it oscillates between ±1.0 → CI bouncing between shoulders.

**Failure mode:**
- Multiple saddles between endpoints; CI converges to the lower-barrier one but reports its energy as "the" E_a. Mitigation: check intermediate images for additional V peaks (band non-monotonic between endpoint and CI).
- Saddle has Morse index 2+: CI converges to a **ridge point** rather than a true TS. Requires Hessian analysis to detect.

---

## §3. Mapping conditions to Phase 0 pre-flight tests

| Condition | Phase 0 tool | Status | Coverage gap |
|---|---|---|---|
| C1 (coercivity) | None needed (DFT+PBC trivial) | ✓ Built-in | — |
| C2 (Lipschitz gradient) | Magnetic configuration scan (Tier 1 AFM+U mack / marc) | Partial — only validated post-hoc for mack, pent untested | Need pent AFM+U Tier 1 pre-NEB (deferred) |
| C3 (endpoint isolation) | **P0-B Layer 1 structural** (the `fes_retrospective` gate) + P0-A PH for borderline | ✓ Validated 3/3 retrospective | n=3 sample only; need pent-specific endpoints application |
| C4 (initial path basin) | **P0-C MLIP pre-screening** (planned, not built) | TODO | MAJOR GAP — most likely failure mode, untested for pent |
| C5 (spring stiffness) | k_spring sensitivity probe at MLIP level (cheap) | Not done for pent | Add to P0-C scope |
| C6 (optimizer step-size) | Monitor first-30-iter BFGS rejection rate | Real-time only, no a-priori test | Workaround: launch + watch closely + autorenice if rejection > 30% |
| C7 (CI saddle isolation) | Post-NEB Hessian analysis (mandatory paper-grade) | Not implemented in script | Add finite-diff Hessian step to converged-NEB harvest pipeline |

**Open gaps blocking pent launch:**
1. **C4 / P0-C MLIP pre-screening pipeline** — main risk mitigation, not yet built. ETA 2–3 days work.
2. **C2 / pent AFM+U Tier 1** — chemistry/electronic-structure validation. ETA 2–3 days A100 ($30–50).
3. **C5 sensitivity probe** — cheap, ~1 hr work, add to P0-C scope.
4. **C7 finite-diff Hessian** — post-NEB step, not a launch blocker, but a blocker for a paper-quotable saddle. Cost ~$30–50 additional A100 time.

**Conditions covered by existing tools (no new work):** C1, C3 (Layer 1), C6 (watch-only).

---

## §4. Pent-specific application

### §4.1. System characterization (recap)

[FACT] Pentlandite (Fe₉S₈), space group Fm-3̄m, cubic. Fe occupies Wyckoff 4b (octahedral) + 32f (tetrahedral cubane Fe₄S₄ cluster). Pent V_Fe NEB target: vacancy in 4b site + H migration between adjacent S anchors.

[FACT] Cell size for pent V_Fe 3×3×3 supercell: 136 atoms (after V_Fe removal). E_endA = pending production NEB; endpoint smoke → endA DONE, fmax=0.026.

[FACT] Symmetry: Fm-3̄m cubic → endpoint pair (S_i ↔ S_j) related by point-group operation if both are in the same Wyckoff orbit. Greigite Fd-3̄m precedent: cubic symmetry → mirror endpoints (E_rxn=0.000 eV) → NEB converged in 40 FIRE iterations. Strong analogy.

### §4.2. Condition-by-condition risk assessment

**C1 (coercivity):** ✓ trivially safe.

**C2 (Lipschitz / smooth):**
- [QUESTION] Pent has both Fe octahedral (4b) AND tetrahedral cubane (32f) sites. A V_Fe defect localized at 4b leaves Fe-Fe interactions in the nearby cubane intact, but the spin state of vacancy-neighbor Fe?
- Cubane Fe₄S₄ has rich magnetic phenomenology (Lu-Peng 2019). With V_Fe nearby, the cubane electronic structure can flip between configurations during NEB → C2 marginal.
- **Mitigation:** Tier 1 AFM+U pent test before NEB launch (analog of mack Tier 1). The AFM+U recipe applies (plain mixing + david + mixing_fixed_ns=15 + tot_mag=0).

**C3 (endpoint isolation):**
- [FACT] pent endA done, fmax=0.026, geometry chemistry-sane.
- TODO: extract endB candidate via canonical_triple picker for pent 4b Wyckoff orbit, apply structural diagnostic Layer 1, verify H displacement endA→endB > 0.5 Å.
- Greigite analogy: cubic-symmetric octahedral V_Fe converged cleanly (H displacement 2.628 Å, hop_dist 4.853 Å). Pent expected similar scale (cubic, octahedral 4b).

**C4 (initial path basin):**
- [HYPOTHESIS] HIGH RISK. Largest gap in our pre-flight. Pent has multiple Wyckoff orbits (4b vs 32f) → multiple candidate pathways. Linear interpolation may cross an unphysical intermediate (e.g. H near the cubane Fe₄S₄ cluster) → SCF spike → BFGS diverge.
- **Mitigation MUST:** P0-C MLIP pre-screening before launch. Expected ETA 2–3 days. Cost ~$0 (local GPU + MACE-MP-0 medium).

**C5 (spring stiffness):**
- Default k_spring = 0.1 eV/Å² (Ruttinger 2022) or 0.5 eV/Å² (QE precedent). Greigite used 0.1 → converged in 40 FIRE iterations. Recommend k_spring = 0.1 for pent consistency.

**C6 (optimizer):**
- Use FIRE following greigite precedent (FIRE-based converged where BFGS thrashed in marc / pent same-basin attempts). Switch to FIRE as primary, BFGS as fallback (emerging pattern).

**C7 (CI saddle isolation):**
- After convergence, run finite-diff Hessian at climbing image. Expected 1 imaginary mode along the S-S axis (H migration direction). Cost +$30–50 A100 time.

### §4.3. Most likely violation for pent V_Fe

Ranked by probability and impact:

1. **C4 (initial path)** — HIGH probability, HIGH impact. Cubane Fe₄S₄ clusters create rich PES topology; linear interpolation likely crosses an unphysical region. P0-C MLIP screening is the only mitigation.
2. **C2 (Lipschitz / smooth)** — MEDIUM probability, MEDIUM impact. Cubane magnetic ambiguity. Tier 1 AFM+U test required.
3. **C3 (endpoint isolation)** — LOW probability per greigite analogy. Cubic Fm-3̄m → symmetric endpoints expected.
4. C5/C6/C7 — LOW probability with correct script setup + post-NEB Hessian.

### §4.4. Recommended pre-flight sequence for pent

```
[Day 1-2]    Build P0-C pipeline (MLIP NEB pre-screening) on local GPU
[Day 3]      Run P0-C on pent V_Fe (3 endpoint candidates, MACE-MP-0)
             → GO/NO-GO/INVESTIGATE
[Day 4-5]    Tier 1 AFM+U single-point on pent V_Fe endA + endB (A100 ~$30-50)
             → C2 magnetic verification
[Day 6]      Apply Layer 1 structural diagnostic to pent endpoints (cheap)
             → C3 verification
[Day 7]      QA gate: chemist+physicist+mathematician OPUS expert review of full pre-flight + NEB script
             → mandatory gate
[Day 8+]     If ALL PASS → launch pent V_Fe NEB on A100 (1-2 weeks, $400-800)
             Real-time monitoring: BFGS/FIRE rejection rate, image energy descent
[After]      Post-NEB finite-diff Hessian saddle Morse-index check ($30-50)
```

**Total pre-flight cost:** ~$60–100 + 1 week of intellectual work.
**Saved on a bad launch:** $400–800.
**ROI:** 6–10× if P0-C catches even one hidden bug (which it likely will — C4 risk).

---

## §5. Open questions / honest gaps

### §5.1. What cannot be proven a priori

1. **Uniqueness of MEP:** for non-convex V with many saddles, the MEP may not be unique. The Mountain Pass theorem gives existence, not uniqueness. The NEB-AGM framework (`GAME_THEORETIC_NEB_FOUNDATIONS.md` §6.1, equilibrium selection) explicitly addresses this, but it remains an open theoretical direction, not a solved problem.

2. **Global convergence in the non-convex setting:** BFGS / FIRE are local optimizers. **No theorem guarantees** convergence to a global minimum of Φ in finite iterations under realistic conditions (non-convex V). The conditions ensure that a stationary point (local minimum) is reached, but not that it is the optimal MEP. Empirical mitigation only (multiple initial guesses, MLIP screening).

3. **Rate of convergence for discrete MEP error:** Liu-Chen-Ortner 2022 proves O(1/N_img²) optimal rate as N_img → ∞ under assumptions. For finite N_img = 9 (our setting), the constant in O(...) is unknown — it could be 0.001 or 100. No principled way to estimate the gap |E_a^discrete − E_a^continuous| from a 9-image NEB alone.

4. **Magnetic configuration optimum:** AFM+U for transition-metal sulfides — large literature but open theoretical questions (Hubbard U value choice, double-counting scheme — Liechtenstein vs Dudarev). Marcasite demonstrated that even with conservative settings, NM collapse can occur unexpectedly. **No a-priori proof** that our chosen Hubbard parameters yield the CORRECT V(x), rather than just a self-consistent V(x).

### §5.2. Where theory is weak but practice works

- **Endpoint Hessian as basin radius estimate.** Theoretically valid only in the quadratic approximation neighborhood. Outside this regime, "basin" is fuzzy. In practice (Layer 1 diagnostic): element-aware displacement thresholds 0.3–0.5 Å — empirically effective, no underlying theorem.

- **k_spring = 0.1 eV/Å² universal value.** No theoretical derivation, only empirical (Ruttinger 2022 on 226 organics, QE FeS extension). Could be system-dependent, particularly for high-Z or strongly magnetic systems.

- **FIRE vs BFGS choice.** No theory predicts WHEN one beats the other for NEB. Practice: FIRE for soft-mode-rich Fe-S systems, BFGS for sharper PES. Empirical heuristic, no theoretical foundation.

- **PH bottleneck distance threshold τ for same-basin detection.** P0-A Müller-Brown test gave 3+ orders of magnitude margin → τ very robust on the toy problem. P0-B on FeS: not yet directly tested at the PH layer (Layer 1 structural was sufficient there). **Open: τ for high-dimensional Fe-S** is unknown.

### §5.3. What "one and a half years of work" would prove

These are candidate research directions for Phase 1+ papers:

1. **Same-basin detector theorem:** formally prove that PH bottleneck distance d_W(PH(x_A), PH(x_B)) < τ_crit implies endpoints are in the same basin (with controlled false-positive rate). Currently empirical only.

2. **Sufficient conditions for NEB convergence to global minimum of Φ.** Currently we have only local conditions. Global conditions would require either (a) convexity (V is NOT convex), (b) special structure (e.g. Polyak-Łojasiewicz inequality, recently popular in ML), or (c) randomization (stochastic NEB).

3. **Cross-mineral transfer convergence theorem:** if NEB for mineral M_1 converged, when can we use the M_1 path geometry as initial guess for M_2? Currently an empirical heuristic (greigite cubic → pent cubic prognosis). A formal theorem would require a manifold-mapping framework.

4. **Magnetic-configuration-aware NEB convergence:** existing theorems assume a single PES. Real Fe-S has multiple electronic-magnetic configurations. Joint optimization over both atomic positions AND spin state → multi-fidelity NEB. Open.

5. **Discrete-to-continuous rate constants for finite N_img:** improvement over Liu-Chen-Ortner 2022 (asymptotic rate) to **non-asymptotic bounds** for practical N_img = 9. Would directly tell us "9 images sufficient for 50 meV accuracy" or "need 15".

---

## §6. References

### Primary sources (cited in conditions §2)

1. **Ambrosetti & Rabinowitz 1973** — "Dual variational methods in critical point theory and applications", J Funct Anal 14:349-381. DOI: [10.1016/0022-1236(73)90051-7](https://doi.org/10.1016/0022-1236(73)90051-7). Mountain Pass Theorem foundational.

2. **Boyd & Vandenberghe 2004** — "Convex Optimization", Cambridge University Press. Chapter 3 (coercivity, smoothness). ISBN 978-0521833783.

3. **Nocedal & Wright 2006** — "Numerical Optimization", 2nd ed Springer. Chapters 3-4 (line search, BFGS, Wolfe conditions). ISBN 978-0387303031.

4. **Henkelman & Jónsson 2000** — "Improved tangent estimate in the nudged elastic band method for finding minimum energy paths and saddle points", J Chem Phys 113:9978-9985. DOI: [10.1063/1.1323224](https://doi.org/10.1063/1.1323224).

5. **Henkelman, Uberuaga, Jónsson 2000** — "A climbing image nudged elastic band method for finding saddle points and minimum energy paths", J Chem Phys 113:9901-9904. DOI: [10.1063/1.1329672](https://doi.org/10.1063/1.1329672).

6. **Sheppard, Terrell, Henkelman 2008** — "Optimization methods for finding minimum energy paths", J Chem Phys 128:134106. DOI: [10.1063/1.2841941](https://doi.org/10.1063/1.2841941). Open access: <https://theory.cm.utexas.edu/henkelman/pubs/sheppard08_134106.pdf>.

7. **Maragakis, Andreev, Brumer, Reichman, Kaxiras 2002** — "Adaptive nudged elastic band approach for transition state calculation", J Chem Phys 117:4651-4658. DOI: [10.1063/1.1495401](https://doi.org/10.1063/1.1495401).

8. **Liu, Chen, Ortner 2022/2025** — "Convergence of the Discrete Minimum Energy Path", arXiv:[2204.07467](https://arxiv.org/abs/2204.07467). Optimal O(1/N_img²) rate proven for NEB stationary states.

9. **Monderer & Shapley 1996** — "Potential Games", Games Econ Behav 14:124-143. DOI: [10.1006/game.1996.0044](https://doi.org/10.1006/game.1996.0044).

10. **Smidstrup, Pedersen, Stokbro, Jónsson 2014** — "Improved initial guess for minimum energy path calculations" (IDPP method), J Chem Phys 140:214106. DOI: [10.1063/1.4878664](https://doi.org/10.1063/1.4878664). arXiv:[1406.1512](https://arxiv.org/abs/1406.1512).

11. **Ruttinger, Sharma, Clancy 2022** — "Protocol for Directing Nudged Elastic Band Calculations to the Minimum Energy Pathway: Nurturing Errant Calculations Back to Convergence", J Chem Theory Comput 18:2993-3005. DOI: [10.1021/acs.jctc.1c00926](https://doi.org/10.1021/acs.jctc.1c00926).

12. **Bitzek, Koskinen, Gähler, Moseler, Gumbsch 2006** — "Structural Relaxation Made Simple" (FIRE optimizer), Phys Rev Lett 97:170201. DOI: [10.1103/PhysRevLett.97.170201](https://doi.org/10.1103/PhysRevLett.97.170201).

### Secondary / monographs

13. **Jabri 2003** — "The Mountain Pass Theorem: Variants, Generalizations and Some Applications", Cambridge Univ Press. ISBN 978-0521827218. Comprehensive treatment of MP-theorem extensions.

14. **Toselli & Widlund 2004** — "Domain Decomposition Methods: Algorithms and Theory", Springer. ISBN 978-3540206965. Reference for C4-related parallelization theory (relevant to NEB-AGM, see GAME_THEORETIC_NEB_FOUNDATIONS.md §1.3).

### Cross-references within the project

- `GAME_THEORETIC_NEB_FOUNDATIONS.md` — game-theoretic framing (§6.1 potential game = independent existence proof, §6.3 mean-field game = continuous-image limit). Companion to this document.
- The cross-mineral V_Fe barrier pattern notes — empirical scaling (mack 43 meV / greig 1861 meV / marc TBD) for §4.2 analogies.
- The MACK/PENT NEB protocol notes — 3-incident scoreboard same-basin signatures.
- The 8-step operational pre-flight checklist (this document = formal foundation behind it).

---

## §7. Statement of scope

[FACT] This document is a **formal restatement of existing theorems** + **applied to the pent setup**. No new theorems are invented.

Decision gate G0 for pent V_Fe NEB launch requires:
- P0-A ✓ DONE (PH prototype)
- P0-B ✓ DONE Layer 1 (retrospective)
- P0-C TODO (MLIP pre-screening pipeline)
- **P0-D ✓ DONE (this document)**

**Conditions verified as non-blocking for launch:** C1, C3 (pending pent-specific endpoint extraction), C5, C6.

**Conditions requiring work before launch:** C2 (Tier 1 AFM+U pent), **C4 (P0-C pipeline)**.

**Conditions verified only post-NEB:** C7 (Hessian saddle Morse check).

**Bottom line:** this checklist + P0-C pipeline + Tier 1 AFM+U — if all three pass for pent, launch with confidence; expected $400–800 spend is justified. If C4 fails (most likely violation) → no launch, investigate alternative endpoint pairs OR use OPES.

---

**Document status:** v1.0 first draft.
**Revision:** chemist + physicist + mathematician OPUS expert consilium pending before G0 decision.
**Word count:** ~3800 words (~8 pages A4).
