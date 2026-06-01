# Sufficient Conditions for NEB Convergence to TRUE MEP

## Formal checklist applied к pentlandite V_Fe pre-flight

**Task:** P0-D (ROADMAP NEB-GT theory §1.4)
**Тип:** formal restatement existing theorems + pent-specific application
**Стиль:** theorem-style statements, sparse prose, math-first
**Cross-refs:** не дублирует `GAME_THEORETIC_NEB_FOUNDATIONS.md` (game-theoretic angle separate).

---

## §1. Variational setup

### §1.1. NEB как constrained finite-dimensional minimization

Пусть V: ℝ^{3N_atoms} → ℝ — DFT PES системы N_atoms ядер. Пусть x_A, x_B ∈ ℝ^{3N_atoms} — два endpoint minima (∇V(x_A) = ∇V(x_B) = 0, Hessian положительно определена в обоих).

Discrete NEB band — упорядоченная цепь N_img images x = (x_1, ..., x_N_img), x_1 ≡ x_A, x_N_img ≡ x_B (либо closed-end relaxation). Standard NEB-objective:

```
Φ(x_1, ..., x_N_img) = Σᵢ V(x_i) + (k_spring/2) Σᵢ ||x_{i+1} − x_i||²
```

NEB-force на image i — projection of −∇Φ онто перпендикулярную касательной к band:

```
F_i^NEB = −∇V(x_i)|_⊥ + F_i^spring|_∥
        = −[∇V(x_i) − (∇V(x_i) · τ̂_i) τ̂_i] + k_spring (||x_{i+1} − x_i|| − ||x_i − x_{i−1}||) τ̂_i
```

где τ̂_i — единичная касательная (Henkelman-Jónsson 2000 upwind formula).

[ФАКТ] Stationary states of NEB-force (F_i^NEB = 0 ∀ i) — это точки where ∇V(x_i) лежит вдоль τ̂_i (band на gradient flow line) и spring tensions выровнены. Это **discrete approximation MEP**: continuous curve γ(s) along which ∇V(γ(s)) ∥ γ'(s).

[ФАКТ] Climbing-image NEB (CI-NEB, Henkelman-Uberuaga-Jónsson 2000): для индекса i* with locally maximal V, inverted force F_i*^climb = −∇V(x_i*) + 2(∇V(x_i*) · τ̂) τ̂ — image carbarkaется к точному saddle.

### §1.2. Mountain Pass Theorem — existence of saddle

**Theorem (Ambrosetti-Rabinowitz 1973, J Funct Anal 14:349-381):** Let X be a Banach space, Φ ∈ C¹(X, ℝ) satisfying Palais-Smale condition (PS). Suppose there exist e ∈ X, r > 0 such that:

(MP1) ||e|| > r;
(MP2) max(Φ(0), Φ(e)) < inf_{||x||=r} Φ(x) ≡ b.

Тогда c = inf_{γ ∈ Γ} max_{t ∈ [0,1]} Φ(γ(t)) ≥ b is a critical value Φ. Здесь Γ — set of continuous paths from 0 to e.

[ФАКТ] Применительно к NEB: x_A and x_B — два local minima V. MP-theorem гарантирует существование критической точки x_TS (saddle) с V(x_TS) ≥ max(V(x_A), V(x_B)) при условии что (PS) выполнено и потенциал coercive. Это **independent existence proof of MEP saddle**, отдельный от potential-game framing (§6.1 в `GAME_THEORETIC_NEB_FOUNDATIONS.md`).

[CLAIM, Jabri 2003 monograph "The Mountain Pass Theorem", Cambridge] Generalizations к Mountain Pass of Second Order (uniqueness) требуют additional conditions: Φ ∈ C², saddle non-degenerate (Morse index 1), no symmetry-related degeneracies. Эти conditions становятся practical-test items ниже (§2.3-2.4).

### §1.3. Connection с potential games (independent existence)

**Theorem (Monderer-Shapley 1996, Games Econ Behav 14:124-143):** NEB-objective Φ is а potential function для game where players = images, strategies = positions, payoff u_i = −Φ projected onto player i. Pure Nash equilibrium ⇔ stationary point Φ. Existence гарантирована если strategy space compact + Φ continuous.

[ФАКТ] Это даёт **second independent existence proof** stationary NEB band. Подробности — `GAME_THEORETIC_NEB_FOUNDATIONS.md` §6.1.

**Critical:** существование saddle ≠ NEB сходится к TRUE saddle. NEB может сойтись к **wrong critical point** (alternate saddle, degenerate minimum manifold, artifact). Это motivation для §2.

---

## §2. Sufficient conditions for convergence к TRUE MEP

Семь conditions, ranked по severity violation в нашем pent setup.

### §2.1. C1 — Coercivity / boundedness

**Formal statement:**
V: ℝ^{3N} → ℝ is coercive: V(x) → +∞ as ||x|| → ∞, OR domain is compact (periodic boundary conditions). Equivalent: sublevel sets {x: V(x) ≤ c} are compact ∀ c < ∞.

**Source:** Boyd & Vandenberghe 2004 "Convex Optimization" §3.1.3 (Cambridge). Standard в variational analysis.

**Practical test для pent:**
- [ФАКТ] DFT с PBC trivially gives compact configuration space modulo discrete translations + permutations. C1 satisfied trivially.
- Subtle violation: если supercell слишком маленькая → vacancy interacts через PBC image → effective potential periodicity, не локализованный basin. Pent V_Fe 136-atom 3×3×3 supercell: V_Fe-V_Fe periodic distance > 11 Å → interaction negligible (per supercell calibration).

**Failure mode:** None expected для pent. Marginal только if N_atoms < 50 (V_Fe defect leakage).

### §2.2. C2 — Lipschitz continuity of gradient (smoothness)

**Formal statement:**
Exists L > 0 such that ||∇V(x) − ∇V(y)|| ≤ L ||x − y|| for all x, y in a neighborhood of the path. Equivalent: V ∈ C^{1,1}_loc, Hessian operator bounded по operator norm.

**Source:** Nocedal & Wright 2006 "Numerical Optimization" 2nd ed Ch 3 §3.2 + Ch 4 §4.1 (Springer). Standard assumption for first-order optimizer convergence theorems (gradient descent, BFGS line-search, FIRE).

**Practical test для pent:**
- [ВОПРОС] DFT PES is C^∞ smooth для fixed Born-Oppenheimer surface IF electronic SCF converged. Cusps возникают только at nuclear coincidences (||x_i − x_j|| → 0), которые в physical configurations НЕ достигаются.
- Subtle violations:
  1. **Spin-state transitions:** discontinuous Hessian при level crossing magnetic configurations (Wang 2019 для greigite; observed marcasite NM collapse). Pent V_Fe: AFM/FM/NM manifolds могут сшиваться non-smoothly если nspin=2 со starting magnetization perturbations.
  2. **Spurious SCF convergence к excited state:** на один image сходится к other electronic state → V(x) discontinuous как function of x. Mitigation: tight SCF tol (conv_thr ≤ 1e-9), force monotonic magnetization annealing.
- Practical test: вычислить numerical Hessian via finite differences в окрестности endpoint (radius ~0.1 Å). Если eigenvalue spectrum changes discontinuously при small perturbation → C2 violation suspected.

**Failure mode:** BFGS line search rejects steps, fmax oscillates без descending, ||∇V|| jumps между iterations. Manifests as iteration counter растёт без energy descent (observed pent BFGS forensic iter 14 signature).

### §2.3. C3 — Endpoint isolation (anti-same-basin)

**Formal statement:**
Let H_A = ∇²V(x_A), λ_min^A = smallest positive eigenvalue. Basin radius (Morse) R_basin^A ~ ||∇V(x)|| / λ_min^A где ∇V — приближённо линеен. **Condition:**

```
||x_A − x_B|| > 2 max(R_basin^A, R_basin^B)
```

i.e. endpoints separated by more than sum of their basin radii.

**Source:** Folklore in optimization (Nocedal-Wright Ch 3.4, "region of attraction" definition); formalized в Liu-Chen-Ortner 2022 (arXiv:2204.07467) for discrete MEPs as stationary states of NEB optimizer.

**Practical test для pent (THIS IS OUR MAIN CHECK):**
- Compute endpoint Hessian eigenvalues at x_A, x_B (BFGS-derived approximation от endpoint relax sufficient — не full ∇² evaluation needed).
- Element-aware displacement check (P0-B Layer 1): max(|x_A^{atom} − x_B^{atom}|) для каждого element.
  - Threshold для pent V_Fe + H migration: H displacement > 0.5 Å, OR ≥2 heavy atoms displaced > 0.5 Å → HIGH confidence different basins.
  - Threshold violated → P0-A PH layer required.
- Cheap MLIP NEB pre-screening (P0-C): one-shot MACE-MP-0 NEB. Compute path-integrated curvature, MACE force RMS at endpoints. If both endpoints MACE-relax к structures with d_MACE(x_A^MACE, x_B^MACE) < 0.3 Å OR identical PH signature → ALERT.

**Failure mode (наш empirical signature, multiple past incidents):**
- E_endA ≈ E_endB to within SCF noise (< 5 meV)
- Tangent τ̂_i degenerates (numerically zero or random direction)
- NEB-forces never decrease below floor ~0.3 eV/Å
- BFGS / FIRE iterations oscillate, fmax plateau
- **Real cost in our project:** $30-100 per occurrence × 3 incidents = $200-300 burned.

C3 enforcement is **mandatory pre-flight gate** в the 8-step RC-check protocol (see the project decision log).

### §2.4. C4 — Initial path в convergence basin of TRUE MEP

**Formal statement:**
Let M* = TRUE MEP (manifold of points satisfying ∇V|_⊥ = 0 along correct connecting curve). Initial NEB path x^{(0)} = (x_1^{(0)}, ..., x_N^{(0)}) must lie within neighborhood B(M*, ε) where the iterative NEB optimizer is contracting — i.e. ||x^{(k)} − M*|| decreasing monotonically.

**Source:**
- Liu, Chen, Ortner 2022 arXiv:2204.07467 ("Convergence of the Discrete Minimum Energy Path"): under coercivity + smoothness + isolation of MEP, NEB discrete path converges к continuous MEP with optimal rate O(1/N_img²) as N_img → ∞. **BUT theorem assumes initial guess близок к limit.**
- Smidstrup-Pedersen-Stokbro-Jónsson 2014 J Chem Phys 140:214106 (IDPP method): characterizes when linear interpolation fails (steep gradient walls, multiple basins between endpoints).

**Practical test для pent:**
- **Cheap MLIP NEB pre-screening (P0-C):** прогнать MACE-MP-0 NEB на 9 images на linear OR IDPP initial guess. Inspect:
  1. Saddle position location (image with max V) — corresponds к Fe-S coordination middle of hop?
  2. Path topology (number of inflection points, monotonicity of V along path) — single saddle, not multi-step?
  3. Energy variance along path < 2 max(barrier estimate) — no extreme excursions через unphysical configurations.
- **Persistent Homology Layer 2 (deferred):** если P0-C ambiguous, sample local PES at 3-5 internal images, compute PH signatures, bottleneck distances between neighbors — should grow monotonically as |i − j| increases (single-basin transition each step).

**Failure mode:**
- IDPP linear interpolation crosses Fe nucleus → SCF on early iterations spikes to V > V(x_TS_real), F_perp tilts catastrophically.
- Multi-basin path: NEB сходится к **first-saddle Nash equilibrium**, missing global path. Same-basin signature если initial path entirely lies в one basin (a marcasite R3 V_S+H case confirmed retrospectively).
- **Our 3-incident scoreboard:**
  - mackinawite: same-basin V_S+H pair S_42↔S_35 (RC choice failure)
  - pentlandite: BFGS iter 14 forensic, geometry within basin throughout
  - marcasite: spglib confirmed all 64 S в single Pnnm 4g orbit

### §2.5. C5 — Spring stiffness regime

**Formal statement:**
For finite k_spring, NEB band lies on **biased manifold**: image distribution is uniform only in the limit k_spring → ∞, AND saddle approximation degrades если k_spring → 0 (band collapses к endpoint pair).

Henkelman-Jónsson 2000 J Chem Phys 113:9978-9985 gives empirical optimal range:

```
k_spring ∈ [0.5, 1.0] eV/Å²  (for typical molecular & condensed-matter NEB)
```

**Source:**
- Henkelman-Jónsson 2000 — foundational NEB paper. §III practical recommendations.
- Sheppard-Terrell-Henkelman 2008 J Chem Phys 128:134106 — comprehensive optimizer comparison; consistent k_spring sensitivity.
- Maragakis-Andreev-Brumer 2002 J Chem Phys 117:4651 — adaptive spring lengths, demonstrates k_spring underestimation causes path collapse.

**Practical test для pent:**
- Use k_spring = 0.1 eV/Å² per Ruttinger-Sharma-Clancy 2022 JCTC 18:2993 protocol (n=226 reactions, validated), OR k_spring = 0.5 eV/Å² per our QE precedent. Both within "safe" range.
- Sensitivity probe: на MLIP pre-screen, run two k values (0.1 and 0.5), check that saddle estimate consistent (ΔE_saddle < 50 meV). If divergent → C5 violation suspected.

**Failure mode:**
- k_spring too low → all images collapse toward endpoint (band shrinks), saddle missed.
- k_spring too high → band rigid, can't follow PES curvature, saddle position biased к straight interpolation midpoint.
- Manifests as: spring forces dominate F_perp throughout iteration; image spacing varies wildly.

### §2.6. C6 — Optimizer step-size / Lipschitz-compatible learning rate

**Formal statement:**
For BFGS line search: each accepted step satisfies Wolfe conditions, ensuring sufficient descent: Φ(x^{k+1}) ≤ Φ(x^k) − c · ||p^k||² для some c > 0 (where p^k = step direction). For FIRE: time-step Δt < 2/√(L) where L is Lipschitz constant.

**Source:** Nocedal-Wright 2006 Ch 3 (BFGS line search, Wolfe conditions) + Bitzek-Koskinen-Gähler-Moseler-Gumbsch 2006 PRL 97:170201 (FIRE).

**Practical test для pent:**
- Monitor BFGS step rejection rate в early iterations. Rejection rate > 30% над first 20 steps → step-size too large relative to L (try maxstep 0.1 → 0.05 Å).
- For FIRE: if dt_max never reached steady-state OR FIRE keeps shrinking dt every few steps → L estimate too small (system stiffer than expected; switch to BFGS).
- **For pent specifically:** observed pent BFGS forensic iter 14 had max_step=0.10 Å + line_search=Wolfe. Recommend keep, monitor over first 30 iter (chemist+physicist OPUS review pre-deploy mandatory).

**Failure mode:**
- Step rejected → backtrack → tiny step → no progress → fmax plateau at ~0.2-0.5 eV/Å (looks like convergence issue, actually optimizer issue).
- Or: huge step accepted → energy spike → next iteration reverts → oscillation.

### §2.7. C7 — CI image identification и saddle isolation

**Formal statement:**
Climbing image i* must be located on the TRUE saddle, not on a shoulder. **Necessary condition:** ∇²V(x_{i*})|_{τ̂} < 0 (negative curvature along reaction direction) AND ∇²V(x_{i*})|_⊥ ≥ 0 (positive curvature in all other directions). I.e. Morse index 1.

**Source:** Henkelman-Uberuaga-Jónsson 2000 J Chem Phys 113:9901 (CI-NEB original). Verification through phonon analysis or finite-difference Hessian at converged image.

**Practical test для pent:**
- Post-NEB single saddle Hessian via ASE `vibrations.Vibrations` или finite-difference (~6N_atoms × 2 SCF — expensive but mandatory для paper-grade saddle).
- Expected: exactly one imaginary mode (negative eigenvalue), with mode displacement vector ∥ τ̂ at saddle.
- If 2+ imaginary modes → shoulder, not true TS. If 0 imaginary modes → false-positive CI (image actually local minimum on band).
- **Cheap proxy during NEB:** monitor `(∇V · τ̂) / ||∇V||` at climbing image. Should approach 1.0 (force vector aligned with band tangent) as iterations proceed. If oscillates between ±1.0 → CI bouncing between shoulders.

**Failure mode:**
- Multiple saddles between endpoints, CI converges к lower-barrier one but reports its energy as "the" E_a. Mitigation: check intermediate images for additional V peaks (band non-monotonic between endpoint and CI).
- Saddle has Morse index 2+: CI converges к **ridge point** не true TS. Requires Hessian analysis to detect.

---

## §3. Mapping conditions к Phase 0 pre-flight tests

| Condition | Phase 0 tool | Status | Coverage gap |
|---|---|---|---|
| C1 (coercivity) | None needed (DFT+PBC trivial) | ✓ Built-in | — |
| C2 (Lipschitz gradient) | Magnetic configuration scan (Tier 1 AFM+U mack / marc) | Partial — only validated post-hoc for mack, pent untested | Need pent AFM+U Tier 1 pre-NEB (deferred) |
| C3 (endpoint isolation) | **P0-B Layer 1 structural** (the `fes_retrospective` gate) + P0-A PH for borderline | ✓ Validated 3/3 retrospective | n=3 sample only; need pent-specific endpoints application |
| C4 (initial path basin) | **P0-C MLIP pre-screening** (planned, not built) | TODO | MAJOR GAP — most likely failure mode untested for pent |
| C5 (spring stiffness) | k_spring sensitivity probe @ MLIP level (cheap) | Not done for pent | Add к P0-C scope |
| C6 (optimizer step-size) | Monitor first-30-iter BFGS rejection rate | Real-time only, no a-priori test | Workaround: launch + watch closely + autorenice if rejection > 30% |
| C7 (CI saddle isolation) | Post-NEB Hessian analysis (mandatory paper-grade) | Not implemented в скрипте | Add finite-diff Hessian step to converged-NEB harvest pipeline |

**Open gaps blocking pent launch:**
1. **C4 / P0-C MLIP pre-screening pipeline** — main risk mitigation, not yet built. ETA 2-3 days work.
2. **C2 / pent AFM+U Tier 1** — chemistry/electronic-structure validation. ETA 2-3 days A100 ($30-50).
3. **C5 sensitivity probe** — cheap, ~1 hr work, add to P0-C scope.
4. **C7 finite-diff Hessian** — post-NEB step, не блокер для launch, но блокер для paper-quotable saddle. Cost ~$30-50 additional A100 time.

**Conditions covered by existing tools (no new work):** C1, C3 (Layer 1), C6 (watch-only).

---

## §4. Pent-specific application

### §4.1. System characterization (recap)

[ФАКТ] Pentlandite (Fe₉S₈), space group Fm-3̄m, cubic. Fe occupies Wyckoff 4b (octahedral) + 32f (tetrahedral cubane Fe₄S₄ cluster). Pent V_Fe NEB target: vacancy in 4b site + H migration between adjacent S anchors.

[ФАКТ] Cell size for pent V_Fe 3×3×3 supercell: 136 atoms (после V_Fe removal). E_endA = pending production NEB; endpoint smoke → endA DONE, fmax=0.026.

[ФАКТ] Symmetry: Fm-3̄m cubic → endpoint pair (S_i ↔ S_j) related by point-group operation if both в same Wyckoff orbit. Greigite Fd-3̄m precedent: cubic symmetry → mirror endpoints (E_rxn=0.000 eV) → NEB converged 40 FIRE iters. Strong analogy.

### §4.2. Condition-by-condition risk assessment

**C1 (coercivity):** ✓ trivially safe.

**C2 (Lipschitz / smooth):**
- [ВОПРОС] Pent has both Fe octahedral (4b) AND tetrahedral cubane (32f) sites. V_Fe defect localized к 4b leaves Fe-Fe interactions in nearby cubane intact, but spin state of vacancy-neighbor Fe?
- Cubane Fe₄S₄ has rich magnetic phenomenology (Lu-Peng 2019). With V_Fe nearby, cubane electronic structure can flip between configurations during NEB → C2 marginal.
- **Mitigation:** Tier 1 AFM+U pent test before NEB launch (analog mack Tier 1). The AFM+U recipe applies (plain mixing + david + mixing_fixed_ns=15 + tot_mag=0).

**C3 (endpoint isolation):**
- [ФАКТ] pent endA done, fmax=0.026, geometry chemistry-sane.
- TODO: extract endB candidate via canonical_triple picker for pent 4b Wyckoff orbit, apply structural diagnostic Layer 1, verify H displacement endA→endB > 0.5 Å.
- Greigite analogy: cubic-symmetric octahedral V_Fe converged cleanly (H displacement 2.628 Å, hop_dist 4.853 Å). Pent expected similar scale (cubic, octahedral 4b).

**C4 (initial path basin):**
- [ГИПОТЕЗА] HIGH RISK. Largest gap in our pre-flight. Pent has multiple Wyckoff orbits (4b vs 32f) → multiple candidate pathways. Linear interpolation may cross unphysical intermediate (e.g. H near cubane Fe₄S₄ cluster) → SCF spike → BFGS diverge.
- **Mitigation MUST:** P0-C MLIP pre-screening before launch. Expected ETA 2-3 days. Cost ~$0 (local GPU + MACE-MP-0 medium).

**C5 (spring stiffness):**
- Default k_spring = 0.1 eV/Å² (Ruttinger 2022) or 0.5 eV/Å² (our QE precedent). Greigite used 0.1 → converged at 40 FIRE iters. Recommend k_spring = 0.1 for pent consistency.

**C6 (optimizer):**
- Use FIRE following greigite precedent (FIRE-based converged where BFGS thrashed in marc / pent same-basin attempts). Switch к FIRE primary, BFGS fallback (emerging pattern).

**C7 (CI saddle isolation):**
- After convergence, run finite-diff Hessian at climbing image. Expected 1 imaginary mode along S-S axis (H migration direction). Cost +$30-50 A100 time.

### §4.3. Most likely violation для pent V_Fe

Ranked by probability and impact:

1. **C4 (initial path)** — HIGH probability, HIGH impact. Cubane Fe₄S₄ clusters создают rich PES topology, linear interpolation likely crosses unphysical region. P0-C MLIP screening is the only mitigation.
2. **C2 (Lipschitz / smooth)** — MEDIUM probability, MEDIUM impact. Cubane magnetic ambiguity. Tier 1 AFM+U test required.
3. **C3 (endpoint isolation)** — LOW probability per greigite analogy. Cubic Fm-3̄m → symmetric endpoints expected.
4. C5/C6/C7 — LOW probability при правильном script setup + post-NEB Hessian.

### §4.4. Recommended pre-flight sequence для pent

```
[Day 1-2]    Build P0-C pipeline (MLIP NEB pre-screening) on local GPU
[Day 3]      Run P0-C on pent V_Fe (3 endpoint candidates, MACE-MP-0)
             → GO/NO-GO/INVESTIGATE
[Day 4-5]    Tier 1 AFM+U single-point на pent V_Fe endA + endB (A100 ~$30-50)
             → C2 magnetic verification
[Day 6]      Apply Layer 1 structural diagnostic к pent endpoints (cheap)
             → C3 verification
[Day 7]      QA gate: chemist+physicist OPUS review of full pre-flight + NEB script
             → mandatory gate
[Day 8+]     If ALL PASS → launch pent V_Fe NEB на A100 (1-2 weeks, $400-800)
             Real-time monitoring: BFGS/FIRE rejection rate, image energy descent
[After]      Post-NEB finite-diff Hessian saddle Morse-index check ($30-50)
```

**Total pre-flight cost:** ~$60-100 + 1 week intellectual work.
**Saved on bad launch:** $400-800.
**ROI:** 6-10× если P0-C ловит хотя бы одну скрытую ошибку (which it likely will — C4 risk).

---

## §5. Open questions / honest gaps

### §5.1. Что мы не можем доказать a priori

1. **Uniqueness of MEP:** для non-convex V с многими saddles, MEP может быть не единственный. Mountain Pass даёт existence, не uniqueness. NEB-AGM framework (`GAME_THEORETIC_NEB_FOUNDATIONS.md` §6.1, equilibrium selection) explicit address this but it's an open theoretical direction, not solved.

2. **Global convergence в non-convex setting:** BFGS / FIRE are local optimizers. **No theorem гарантирует** convergence к global minimum Φ in finite iterations under realistic conditions (non-convex V). Conditions ensure stationary point (local minimum) reached, но not что it's the optimal MEP. Empirical mitigation only (multiple initial guesses, MLIP screening).

3. **Rate of convergence для discrete MEP error:** Liu-Chen-Ortner 2022 proves O(1/N_img²) optimal rate as N_img → ∞ при assumptions. Для finite N_img = 9 (our setting), constant in O(...) unknown — could be 0.001 or 100. No principled way to estimate gap |E_a^discrete − E_a^continuous| из 9-image NEB alone.

4. **Magnetic configuration optimum:** AFM+U for transition-metal sulfides — large literature но open theoretical questions (Hubbard U value choice, double-counting scheme — Liechtenstein vs Dudarev). Marcasite demonstrated even с conservative settings, NM collapse can occur unexpectedly. **No a-priori proof** что our chosen Hubbard parameters give CORRECT V(x), не just self-consistent V(x).

### §5.2. Где теория слабая, но practice работает

- **Endpoint Hessian as basin radius estimate.** Theoretically only valid в quadratic approximation neighborhood. Outside this regime, "basin" is fuzzy. Practice (Layer 1 diagnostic): element-aware displacement thresholds 0.3-0.5 Å — empirically work, no underlying theorem.

- **k_spring = 0.1 eV/Å² universal value.** No theoretical derivation, only empirical (Ruttinger 2022 на 226 organics, our QE FeS extension). Could be system-dependent, particularly для high-Z или strongly magnetic systems.

- **FIRE vs BFGS choice.** No theory predicts WHEN one beats the other для NEB. Practice: FIRE for soft-mode-rich Fe-S systems, BFGS for sharper PES. Empirical heuristic, no foundation.

- **PH bottleneck distance threshold τ для same-basin detection.** P0-A Müller-Brown test gave 3+ orders of magnitude margin → τ very robust on toy. P0-B on FeS: not yet directly tested at PH layer (Layer 1 structural sufficient there). **Open: τ for high-dimensional Fe-S** unknown.

### §5.3. Что бы доказали "за полтора года работы"

These are candidate research directions для Phase 1+ papers:

1. **Same-basin detector theorem:** prove формально that PH bottleneck distance d_W(PH(x_A), PH(x_B)) < τ_crit implies endpoints are в одном basin (with controlled false-positive rate). Currently empirical only.

2. **Sufficient conditions для NEB convergence к global minimum Φ.** Currently we have local conditions. Global conditions would require either (a) convexity (V is NOT convex), (b) special structure (e.g. Polyak-Łojasiewicz inequality, recently popular в ML), or (c) randomization (stochastic NEB).

3. **Cross-mineral transfer convergence theorem:** if NEB for mineral M_1 converged, when can we use M_1 path geometry as initial guess for M_2? Currently empirical heuristic (greigite cubic → pent cubic prognosis). Formal theorem would require manifold-mapping framework.

4. **Magnetic-configuration-aware NEB convergence:** existing theorems assume single PES. Real Fe-S has multiple electronic-magnetic configurations. Joint optimization over both atomic positions AND spin state → multi-fidelity NEB. Open.

5. **Discrete-to-continuous rate constants для finite N_img:** improvement над Liu-Chen-Ortner 2022 (asymptotic rate) к **non-asymptotic bounds** для practical N_img = 9. Would directly tell us "9 images sufficient для 50 meV accuracy" или "need 15".

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

14. **Toselli & Widlund 2004** — "Domain Decomposition Methods: Algorithms and Theory", Springer. ISBN 978-3540206965. Reference для C4-related parallelization theory (relevant to NEB-AGM, see GAME_THEORETIC_NEB_FOUNDATIONS.md §1.3).

### Cross-refs внутри проекта

- `GAME_THEORETIC_NEB_FOUNDATIONS.md` — game-theoretic framing (§6.1 potential game = independent existence proof, §6.3 mean-field game = continuous-image limit). Companion to this document.
- the cross-mineral V_Fe barrier pattern notes — empirical scaling (mack 43 meV / greig 1861 meV / marc TBD) for §4.2 analogies.
- the MACK/PENT NEB protocol notes — 3-incident scoreboard same-basin signatures.
- the 8-step operational pre-flight checklist (this document = formal foundation behind it).

---

## §7. Statement of scope

[ФАКТ] Этот документ — **formal restatement existing theorems** + **applied к pent setup**. Не выдумка нового theorem.

Decision gate G0 для pent V_Fe NEB launch requires:
- P0-A ✓ DONE (PH prototype)
- P0-B ✓ DONE Layer 1 (retrospective)
- P0-C TODO (MLIP pre-screening pipeline)
- **P0-D ✓ DONE (this document)**

**Conditions verified to НЕТ блокеры для launch:** C1, C3 (pending pent-specific endpoint extraction), C5, C6.

**Conditions требующие work до launch:** C2 (Tier 1 AFM+U pent), **C4 (P0-C pipeline)**.

**Conditions verified только post-NEB:** C7 (Hessian saddle Morse check).

**Bottom line:** этот checklist + P0-C pipeline + Tier 1 AFM+U → если все три pass для pent, launch с confidence; expected $400-800 spend justified. Если C4 fails (most likely violation) → no launch, investigate alternative endpoint pairs OR use OPES.

---

**Document status:** v1.0 first draft.
**Ревизия:** chemist + physicist + mathematician OPUS consilium pending перед G0 decision.
**Word count:** ~3800 words (~8 страниц A4).
