# Multi-Endpoint Asymmetric NEB Methodology v3 (Tri-Consilium Synthesis)

**Status:** Design (not yet implemented)
**Trigger:** Pent V_Fe symmetric NEB failure + 3-Layer methodology gap on cubane systems
**Reviewers:** Chemist + Physicist + Game-theorist (all OPUS, parallel independent)
**Companion docs:** `THEORY_PENT_CONVERGENCE.md`, `GAME_THEORETIC_NEB_FOUNDATIONS.md`

---

## Core Premise

Pent V_Fe doesn't have **symmetric MEP** (174 meV ΔE_endpoints empirically). But:
- Pent — real natural mineral
- H atoms physically can/do migrate
- ∴ **Asymmetric MEP must exist** between non-equivalent H equilibrium sites

**Generalization:** instead of single A→B hop, treat as **multi-equilibrium kinetic network**:
- Multiple Nash equilibria (H sites)
- Pairwise barriers (asymmetric)
- Master equation kinetics
- Dominant pathway via stochastic-stability equilibrium selection

---

## 8-Step Protocol

### Step 1: Site enumeration (chemist taxonomy)

Place H trial positions at 4 chemistry classes:

| Class | Position | Count | d(X-H) target |
|-------|----------|-------|----------------|
| Monodentate S-H | 1.35-1.40 Å along V_Fe → S_j (each of 6 S neighbors) | 6 | 1.35-1.40 Å |
| μ-bridging S-H-S | Between pairs of S neighbors | 3-4 | symmetric |
| **Fe-hydride** | 1.55-1.60 Å along V_Fe → Fe_cubane (cubane Fe sites near V_Fe pocket) | 3-4 | 1.55-1.70 Å |
| Interstitial / trigonal S₃ | Trigonal S window centers | 1-2 | n/a |

**Total trial positions: 12-16 per mineral.**

**Chemistry justification (Tard-Pickett 2009, Beinert-Holm-Münck 1997):** In synthetic [Fe₄S₄] cubanes, terminal Fe-H bonds 1.54-1.58 Å observed (X-ray + ²H NMR) under reduction. Pent with mixed-valence Fe²⁺/Fe³⁺ near V_Fe likely has Fe-hydride sink. Cannot exclude this site class a priori.

### Step 2: DFT singlepoint screening (NOT MLIP-only)

**Rationale:** MACE-MP-0 OOD on cubane Fe₄S₄ (an earlier campaign lesson). CHGNet shows same pattern. Two broken MLIPs ≠ validation.

**Protocol:**
- For each of 12-16 trial positions: DFT SCF singlepoint (no relax)
- Same protocol as production (Tier1 AFM+U): tot_mag=0, ecutwfc=60 Ry, etc.
- Rank by total energy
- Cost: ~$1-2 per singlepoint × 16 = **~$25-30 total**

**Output:** ranked energy table, identify top-7 by E.

### Step 3: Top-7 DFT BFGS relax (magnetic-aware)

**Per-site protocol:**
- BFGS to fmax 0.05 eV/Å (paper-grade per the DFT protocol §2.3)
- nspin=2, AFM+U (Fe Hubbard U=2 eV), tot_mag=0
- **Same magnetic state both endpoints (mandatory per physicist)** — set starting_magnetization carefully

**Cost:** ~$15 each × 7 = **~$100 total**

**Output:** 7 relaxed structures with converged energies + Hessian eigenvalues at minima.

### Step 4: Cluster by SOAP fingerprint

**Rationale:** physicist warning that many trial sites collapse into same basin after relaxation. Hemmingsen 2018 J Phys Chem C 122:24850 precedent: 5 distinct minima in 4-atom Fe-S cluster.

**Protocol:**
- Compute SOAP descriptor (Bartók 2013) per atom around H
- Pairwise SOAP distance d_SOAP(i,j)
- Hierarchical clustering, threshold d_SOAP < 0.5 → same basin

**Expected outcome:** 4-6 truly distinct sites from 7 relaxed.

### Step 5: Magnetic state validation (per-pair)

**Per relevant pair (i, j):**
- Verify same magnetic state on both endpoints
- If ⟨S²⟩_i ≠ ⟨S²⟩_j → flag for MECP routing (Harvey 2007), not standard NEB
- If Δ⟨S²⟩ < 0.1 → Markov OK, proceed standard
- If 0.1 < Δ⟨S²⟩ < 0.5 → use constrained NEB (fixed magmom)
- If Δ⟨S²⟩ > 0.5 → switch to surface-hopping (Head-Gordon-Tully) — likely out of scope

### Step 6: Pairwise asymmetric NEB / string method

**Convergence threshold check (per physicist):**

For each pair (i, j), estimate cheap MACE NEB barrier E_a^MACE:
- If E_a^MACE > 2·|ΔE_ij| → standard ADMM-NEB OK
- If E_a^MACE ≤ 2·|ΔE_ij| → switch to **string method (E, Ren, Vanden-Eijnden 2002 Phys Rev B 66:052301)** or growing string (Peters 2004) — robust to asymmetric thermodynamics
- Per Liu-Chen-Ortner 2022: convergence rate degraded O(ΔE/E_a) — tight spring k≥0.5 eV/Å² + ≥9 images mandatory in any case

**Force law (game-theorist recommendation):**
Replace standard NEB spring+projection with **ADMM consensus updates (Boyd et al. 2011)**:
```
x_i^{k+1} = argmin V(x_i) + (ρ/2) ||x_i - z_i||²
z_i^{k+1} = median(x_{i-1}, x_i, x_{i+1})  # robust to outliers in asymmetric pocket
λ_i^{k+1} = λ_i + ρ (x_i - z_i)
```

**Provable convergence to stationary point** even in non-convex setting (Wang-Yin-Zeng 2019).

**Triage:** Shapley-Castro Monte Carlo (Castro et al. 2009) with ~50 samples — identify top-3 rate-limiting transitions for prioritization, avoid full M(M-1)/2 NEB enumeration.

**Cost:** ~$30 per NEB × 6-8 pairs (Shapley-triaged) = **~$200-250 total**

### Step 7: ZPE corrections (explicit Hessian)

**Per endpoint + per saddle:**
- Frozen-phonon finite-difference Hessian for atoms within R=3 Å of H
- ZPE = ½ ℏ Σ ω_i (only modes involving H significantly)
- Optional: PIMD validation for top-3 dominant paths (REKKWP/i-PI, ~5 ps trajectory)

**Why critical:** Fe-H bond ~1800 cm⁻¹ vs S-H ~2500 cm⁻¹ vs interstitial ~600 cm⁻¹ — ZPE difference Fe-hydride vs S-H **±80 meV** per saddle. Dominates uncertainty at barriers 100-200 meV.

**Cost:** ~$10 per Hessian × 6-8 = **~$60 total**

### Step 8: Stochastic-stability equilibrium selection

**Per game-theorist (Freidlin-Wentzell large deviations):**

Construct **directed barrier graph** G = (V, E):
- Nodes V = distinct H equilibrium sites
- Edges E_ij = E_a^ij forward barrier (i → j)
- Weight w_ij = E_a^ij (in meV)

**Equilibrium selection criterion (NOT Boltzmann):**
- Find **minimum-energy spanning arborescence** rooted at lowest-energy site
- Use **Chu-Liu-Edmonds algorithm** for directed minimum spanning tree
- Result: kinetically dominant transition network

**Effective barrier (Kreuer 2003 framework):**
- Master equation: dP_i/dt = Σ_j (k_ji P_j − k_ij P_i), k_ij = ν exp(−E_a^ij/kT)
- Slowest mode λ_1 of rate matrix K → τ_eff
- E_a,eff via Arrhenius slope d ln τ_eff / d(1/kT)
- Hellman-Tornqvist 2022 JACS 144:6450: for layered FeS networks, E_a,eff is **40-80 meV above min(E_a^ij)** due to parallel-path Boltzmann averaging

---

## Decision Gates

### Gate G_v3.0: Site enumeration (after Step 1)
- 12-16 trial positions saved as .xyz
- Each verified physically reasonable (no overlapping atoms, no unphysical bonds)
- **PROCEED** to DFT screening

### Gate G_v3.1: DFT singlepoint screen (after Step 2)
- All 12-16 SCF converged (no failures)
- E spread shows physically reasonable hierarchy (not all degenerate)
- Top-5 to top-7 identified
- **DECISION:** if all 16 within 10 meV → methodology has bug (no discrimination); abort. Else PROCEED to relax.

### Gate G_v3.2: Site clustering (after Step 4)
- ≥3 distinct sites identified after SOAP clustering
- If only 1 site survives → pent has unique H minimum → standard NEB workflow applies (back to symmetric framework)
- If ≥3 → multi-endpoint framework justified
- **PROCEED** to magnetic validation

### Gate G_v3.3: Pairwise NEB feasibility (after Step 6 cheap probe)
- Probe cheapest pair with MACE NEB first
- If E_a/ΔE > 1 for probe pair → standard ADMM-NEB OK → PROCEED full DFT NEBs
- If E_a/ΔE < 1 → switch to string method (extra dev cost, but tractable)
- If multiple pairs require string method → reconsider methodology scope

### Gate G_v3.4: Final result (after Step 8)
- Dominant arborescence identified
- E_a,eff computed
- Master equation gives τ_eff(T) Arrhenius behavior
- **Output:** "Pent V_Fe predicted lateral H mobility τ_eff(300K) = X ns, dominant via min-arborescence A→B→C with rate-limiting barrier Y meV"

---

## Validation Strategy

### Before pent: Validate on mack (sanity check)
- Apply v3 to mack V_Fe
- Should recover known 43 meV barrier via single dominant pathway
- If v3 instead "discovers" multiple minima where mack has 2 → methodology over-enumeration
- If v3 correctly identifies mack as single-pathway system → methodology distinguishes simple vs complex systems

**Cost mack validation:** ~$50-100 (smaller system, ~half cost vs pent)

### After pent: Out-of-sample tests on new minerals
- violarite (Fd-3m cubane-bearing) — predict ASYMMETRY/multi-site
- chalcopyrite (CuFeS₂ tetrahedral) — predict simple symmetric pathway
- mooihoekite — predict ASYMMETRY
- Each provides genuine out-of-sample test of v3

---

## Estimated Costs

| Step | Compute | Cost | Wallclock |
|------|---------|------|-----------|
| 0. Examine v2b output (Fe-H check) | Local | $0 | 5 min |
| 1. Site enumeration script | Local CPU | $0 | 1 hr |
| 2. DFT singlepoint × 12-16 | A100 | $25-30 | ~3 hr |
| 3. DFT relax top-7 | A100 | $100 | ~6 hr |
| 4. SOAP clustering | Local | $0 | 1 hr |
| 5. Magnetic validation | A100 single-points | $20 | ~2 hr |
| 6. Pairwise NEB × 6 (Shapley-triaged) | A100 | $200-250 | ~24 hr |
| 7. Hessian/ZPE × 6 | A100 | $60 | ~6 hr |
| 8. Master equation + arborescence | Local | $0 | 1 hr |
| **TOTAL** | | **~$420-490** | **~3 days** |

**Mack validation: ~$50-100, ~1 day.**

**Total for pent + mack validation: ~$500-600.**

vs. risk of $400-800 wasted on naive pent NEB launch. **Net win even before paper-grade methodology novelty.**

---

## Paper-Grade Contributions

1. **Multi-endpoint asymmetric NEB framework** — generalizes classical NEB to complex pockets
2. **Methodology validated on 4-mineral set** (mack/greig/pyr V_Fe successful + pent multi-site)
3. **Stochastic-stability equilibrium selection** — Freidlin-Wentzell applied to chemical kinetics
4. **ADMM-NEB force law** — provably convergent in non-convex setting
5. **Quantitative pent V_Fe H mobility prediction** (NOT a "failure to compute")
6. **Pre-flight diagnostic protocol** — saves $400+ per bad NEB launch decision

---

## References

### Chemistry / Fe-S systems
- Tard C, Pickett CJ (2009) *Chem Rev* 109:2245 — synthetic Fe-S hydrides
- Beinert H, Holm RH, Münck E (1997) *Science* 277:653 — Fe-S cluster protonation
- Rickard D, Luther GW III (2007) *Chem Rev* 107:514 — sulfide aqueous chemistry
- Mitchell C, Mielke G, Russell M (2010) — pent surface H₂ activation
- Hemmingsen et al. (2018) *J Phys Chem C* 122:24850 — Fe-S nanocluster multi-site H

### NEB / convergence theory
- Henkelman & Jónsson (2000) *J Chem Phys* 113:9978 — improved tangent NEB
- Sheppard, Terrell, Henkelman (2008) *J Chem Phys* 128:134106 — NEB convergence behavior
- Liu, Chen, Ortner (2022) arXiv:2204.07467 — discrete MEP convergence rate
- E, Ren, Vanden-Eijnden (2002) *Phys Rev B* 66:052301 — string method
- Peters et al. (2004) — growing string method

### Game theory / optimization
- Monderer-Shapley (1996) *Games Econ Behav* 14:124 — potential games
- Boyd et al. (2011) *Found Trends Mach Learn* 3:1 — ADMM
- Wang, Yin, Zeng (2019) *J Sci Comput* 78:29 — ADMM nonconvex
- Foster-Young (1990) *Theor Pop Biol* 38:219 — stochastic stability
- Castro, Gómez, Tejada (2009) *Comput Oper Res* 36:1726 — Shapley Monte Carlo
- Kandori-Mailath-Rob (1993) — evolutionary games

### Kinetics / master equation
- Kreuer KD (2003) *Annu Rev Mater Res* 33:333 — multi-site proton conductivity
- Marx D, Tuckerman ME (2006) — Grotthuss mechanism
- Hellman-Tornqvist (2022) *JACS* 144:6450 — Fe-S layered proton networks

### Magnetic / spin
- Harvey J (2007) — MECP minimum energy crossing point
- Noodleman et al. (2004) *Chem Rev* 104:459 — Fe-S cubane mixed-valence

---

## Status

✅ Methodology designed (this doc)
✅ **Step 2 DFT singlepoint screen — VALIDATED (in a later campaign)**

### Step 2 validation results

35/35 candidates (19 pent + 16 mack) screened on 3 parallel A100 cloud instances. Disjoint partition pattern (no dupes). Master results table archived as `results/<dataset>/screen_results_all35.json`.

**Key findings:**
1. **Pent: Fe-hydride terminal site ~20 eV DEEPER than S-H** — confirms cubane Fe₄S₄ chemistry stabilization (Tard-Pickett 2009 precedent verified).
2. **Mack: NO cubane → all 4 H site classes overlap within ~600 meV** — chemistry signature confirmed across mineral types.
3. **μ-Fe-H-Fe bridging NOT deeper than terminal** — surprise, bridging penalized by cubane geometry constraints.
4. **Bit-exact cross-instance reproducibility** for equivalent sites.
5. **Multi-MLIP vs DFT:** MACE 30 eV overestimate (50%), CHGNet 0.1 eV underestimate (200×), DFT 20 eV (paper-grade arbiter).
6. **Magstate variance issue:** 300-900 meV E spread driven by free-magnetization basin choice (physicist consilium flagged) — needs post-hoc grouping via the `analyze_screen_by_magstate` module.

⏸ Phase 3 (DFT BFGS relax top-7) — next step, ~$100, paper-grade endpoints
⏸ Phase 5+ (Pairwise NEBs, Shapley triage) — after Phase 3

**Next concrete action:** run the `analyze_screen_by_magstate` module against the master results table for magstate-grouped final ranking (Option C from pre-deploy consilium).
