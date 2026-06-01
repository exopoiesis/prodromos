# Game-Theoretic Foundations of NEB:
## Deep analogies from optimization theory, distributed systems, and multi-agent games

**Context:** Third Matter project, same-basin trap problem in FeS V_Fe NEB
**Cross-ref:** `ALTERNATIVES_AND_ROLE_AWARE_NEB.md` (landscape + lit-scan)
**Status:** conceptual framework + concrete new formulation NEB-AGM. Research direction.

---

## Part 0. Context and motivation

The core idea: **each NEB image is a cell with a local PES model that accumulates knowledge**. Standard NEB treats all 9 images as homogeneous degrees of freedom of a single optimization variable. We want heterogeneity: each image = independent agent with a local surrogate.

This document explores deep analogies in:
- Variational analysis (ADMM, Domain Decomposition)
- Topology (Discrete Morse, Persistent Homology)
- Probability (EnKF, Replica Exchange, Wasserstein flows)
- Optimal control (DDP, MPC)
- ML (Mixture of Experts, GNN, Continual Learning)
- **Game theory** (Potential Games, Stackelberg, Mean-Field Games, Shapley, No-Regret)

And presents a concrete new formulation: **NEB-AGM (NEB as Adaptive Game with Memory)** — a hybrid combining the most productive analogies.

**Central thesis:** the same-basin trap is an **equilibrium selection problem**, not a technical bug. This reframing opens the arsenal of game theory for attacking the problem.

---

## Part 1. Variational viewpoint: ADMM + Domain Decomposition ⭐

### 1.1. ADMM (Alternating Direction Method of Multipliers)

**Boyd et al. 2011** — standard method for distributed convex optimization. Decomposes:
```
minimize    Σᵢ fᵢ(xᵢ)
subject to  A x = b   (coupling constraint)
```
via **augmented Lagrangian** with dual variables λ:
```
L_ρ(x, λ) = Σᵢ fᵢ(xᵢ) + λᵀ(Ax − b) + (ρ/2)‖Ax − b‖²
```
and iterative update:
```
xᵢ^{k+1} = argmin_x  fᵢ(x) + (ρ/2)‖A_i x + ... − b + λᵏ/ρ‖²     [local]
λ^{k+1}   = λᵏ + ρ(Ax^{k+1} − b)                                  [dual update]
```

### 1.2. NEB reformulated as ADMM

**Standard NEB force equation:**
```
F_i = −∇V(x_i)|_⊥ + k_spring (Δs_right − Δs_left) τ̂_i
```

**Equivalent ADMM form:**
- **Local objectives** f_i(x_i) = V(x_i)  (can be replaced by surrogate V̂_i!)
- **Coupling constraint** |x_{i+1} − x_i| = h_target  (equidistant)
- **Dual variables** λ_i  = **spring tension** between i and i+1
- **Augmented Lagrangian:**
  ```
  L_ρ = Σᵢ V(xᵢ) + Σᵢ λᵢ (|xᵢ₊₁−xᵢ| − h) + (ρ/2) Σᵢ (|xᵢ₊₁−xᵢ| − h)²
  ```

**What this gives:**
1. **Per-image local model:** swap V(x_i) → V̂_i(x_i) (local GP/quadratic) is trivial, ADMM is unchanged
2. **Provable convergence:** for convex local f theorems exist (Boyd 2011, He & Yuan 2012). For non-convex (our case) — recent extensions (Wang et al. 2019, "ADMM for nonconvex problems")
3. **Parallelism:** x_i updates on even/odd images are independent → 2× speedup
4. **Natural handling of stuck images:** if local minimization does not converge → λ accumulates → ADMM "forces" movement via dual penalty

### 1.3. Domain Decomposition Methods (DDM)

The numerical PDE community has used **Schwarz alternating, FETI, BDDC** methods for 40 years. Each subdomain solves the PDE locally; interface conditions smooth the result.

**Direct analogy:**
- Subdomain Ωᵢ ↔ neighborhood of image i on PES
- Local solver ↔ local minimization V̂_i
- Interface condition ↔ spring constraint
- **Atlas of a manifold** = collection of overlapping charts = images. This is not a metaphor; it formally coincides with manifold theory.

**Mature theory:** Toselli & Widlund "Domain Decomposition Methods" (2004), 500 pages of rigorous math. Ready to import.

### 1.4. Multigrid view

**Multigrid methods** use a hierarchy of resolutions. In NEB:
- Coarse path: 5 images (fast convergence on large-scale topology)
- Fine path: 9 images (accurate saddle localization)
- Cycling: V-cycle / W-cycle between levels

**Not published in NEB.** Standard practice is a fixed N=9 images.

---

## Part 2. Topological viewpoint: Discrete Morse + Persistent Homology

### 2.1. Discrete Morse Theory (Forman 2002)

Combinatorial flows on simplicial complex K. Discrete Morse function f: K → ℝ with critical simplices (no pairing with neighbors). Analog of classical Morse theory, but discrete.

**Application to NEB:**
- Each image lives on a simplex (vertex/edge/face/...)
- Local f_i on each simplex is built from local model V̂_i
- Combinatorial gradient flow = discrete analog of NEB updates
- **Advantage:** explicit combinatorial structure → algorithmic guarantees

### 2.2. Persistent Homology (PH) for NEB diagnostics

**Edelsbrunner et al. 2002.** Tracking topological features (connected components, loops, voids) as a parameter changes.

**Concrete diagnostic tool for same-basin trap:**

1. For each image i compute the **persistence diagram** of local PES V̂_i (via sublevel filtration)
2. The persistence diagram contains a **barcode** — the lifetime of each topological feature
3. **Basin signature:** 0-dim persistence is long-lived (deep minimum)
4. **Saddle signature:** 1-dim persistence appears (handle)
5. **Cross-image comparison:** Wasserstein/bottleneck distance between barcodes

**Diagnosing the same-basin trap:**
```
if bottleneck_distance(PH(V̂_1), PH(V̂_5)) < threshold:
    → images 1 and 5 have topologically identical local environments
    → they are in the same basin
    → ALERT: same-basin trap detected
```

**This is a formal automated detector of our hand-crafted "structural sanity gate".**

**Lit-status:** no NEB paper with PH has been found (lit-scan part 6.5). PH applications in materials exist (Hiraoka et al. 2016, "Hierarchical structures in amorphous solids"), but not for transition paths.

### 2.3. Connection with the Mountain Pass Theorem

The Mountain Pass Theorem guarantees the existence of a saddle. Persistent homology **constructively builds** the homology class invariant responsible for the saddle. This may yield an **algorithmic Mountain Pass Theorem** — explicit construction instead of a variational existence proof.

---

## Part 3. Probabilistic viewpoint: EnKF + Replica Exchange + Wasserstein flows

### 3.1. Ensemble Kalman Filter (EnKF)

**Evensen 1994.** Data assimilation: each ensemble member has a state estimate, updated via ensemble covariance.

**EnKF for NEB:**
- Each image i = ensemble member with state x_i and local covariance Σ_i
- Local model V̂_i updated EnKF-style on each DFT evaluation
- **Cross-image consensus** via ensemble covariance structure

EnKF is used in meteorology with **thousands** of local models in real time. The technology is ready; only notation adaptation is needed.

### 3.2. Replica Exchange / Parallel Tempering

**Sugita & Okamoto 1999.** Replicas at different temperatures exchange configurations via Metropolis.

**Connection to NEB:** in standard NEB images **do not exchange** — only through springs. If each image is a "temperature" replica (with its own T_i), they can swap configurations. This gives:
- **Stuck escape:** a frozen image exchanges with a hot one and receives a kick
- **Heterogeneous exploration:** different images at different temperatures simultaneously

**Combined approach: Replica Exchange NEB (REN-NEB)** — mentioned in several papers of the 2010s (e.g., Kim et al. 2014), but not mainstream.

### 3.3. Wasserstein gradient flows

**Jordan-Kinderlehrer-Otto (JKO) scheme 1998.** Gradient flow on probability distributions in Wasserstein-2 metric.

**Deep connection to MEP:**
- Each image parameterizes a distribution p_i (not a point!)
- Path = flow in Wasserstein space W₂
- MEP = geodesic in this metric space

**This is where optimal transport meets MEP**, and it is typically overlooked in the NEB literature. The connection via the **Brenier-Benamou formula** (Brenier 1991, Benamou 2000) gives a fluid-dynamic interpretation — path as a flow of "probability mass" from A to B with minimal kinetic energy.

**Practical implication:** instead of point images one can use **distribution images** (parameterized GMMs or normalizing flows). This integrates with TPT (committor function) and Boltzmann generators into a unified framework.

---

## Part 4. Optimal Control viewpoint: DDP + MPC ⭐

### 4.1. Differential Dynamic Programming (DDP)

**Jacobson & Mayne 1970.** Trajectory optimization via **local quadratic approximations** of the value function at each stage, stitched via Bellman recursion.

**Direct import into NEB:**

**Forward pass (information accumulation):**
```
For each image i (forward):
  V̂_i(x) = V(x_i⁰) + g_iᵀ(x − x_i⁰) + ½(x − x_i⁰)ᵀ H_i (x − x_i⁰)
  # g_i — gradient at x_i⁰ (cheap from current DFT)
  # H_i — local Hessian estimate (BFGS-style update per image)
```

**Backward pass (consensus update):**
```
For each image i (backward from N to 1):
  K_i = compute_gain_matrix(H_i, H_{i+1}, spring_coupling)
  # Bellman: value-to-go = local cost + propagated future cost
  
  next_step_direction_i = −K_i · g_i  # incorporates downstream info
```

**Per-image quadratic model V̂_i** — this is literally what DDP does at each stage. **Each image holds its own Hessian**, updated BFGS-style from local DFT evaluations. In standard NEB today — a single global BFGS on the concatenated state vector.

**Mature in robotics:** Tassa, Mansard, Todorov (iLQR/DDP), used in MPC for humanoid control with >100 DOF in real time.

**Lit-status:** **not published in NEB context** (verified).

### 4.2. Model Predictive Control (MPC) on chain graph

MPC: rolling horizon optimization with a local model.
- Image i has an MPC controller with horizon H
- Looks ahead H steps along the path
- Local model V̂_i updated every iteration
- **Distributed MPC** (Camponogara et al. 2002) — each MPC consults its neighbors

This provides a **predictive component** — each image "anticipates" what its neighbors will do.

### 4.3. Connection to Pontryagin Maximum Principle

PMP: optimal trajectory γ*(s) satisfies Hamiltonian equations:
```
γ̇ = ∂H/∂p
ṗ = −∂H/∂γ
```
where p is the costate (Lagrange multiplier).

**In NEB:** spring tension λ_i ↔ discrete costate p_i. The ADMM formulation (Part 1) is a finite-difference analog of PMP.

---

## Part 5. ML viewpoint: MoE + GNN + Continual Learning

### 5.1. Mixture of Experts (MoE) — exactly Lance/MaPE

**Shazeer et al. 2017** (sparse MoE) → Lance 2026.

**For NEB:**
- Each image = expert for a region of PES
- Gating function g(x) = role assignment (MaPE analog)
- Sparse activation: only relevant experts are active at evaluation point x
- Training: experts specialize automatically (load balancing loss)

**Implementation:**
- 9 lightweight neural networks (one per image)
- 1 gating network — selects which expert is active
- Gating trained end-to-end via differentiable assignment

### 5.2. Graph Neural Networks (GNN)

Images = nodes, springs = edges. Message passing:
```
m_{i→j} = MLP_message(x_i, x_j, edge_features)
x_j^{k+1} = MLP_update(x_j^k, Σ_i m_{i→j})
```

**Replacement for fixed springs:** spring constants k_spring replaced by learned attention weights between images. Each image learns "with whom coordination is stronger".

**Recent precedent:** Equivariant GNNs (Satorras et al. 2021, Schütt et al. 2021, NequIP/Allegro/MACE) are already used for **PES** in MLIP. Extension to **path-level** GNN, where nodes = images of a path, is natural.

### 5.3. Continual Learning

Each image moves → its local model must adapt without forgetting accumulated knowledge.

**Applicable methods:**
- **Elastic Weight Consolidation (Kirkpatrick et al. 2017):** Fisher information-weighted regularization
- **Experience Replay:** keep buffer of past (x, ∇V) pairs, retrain periodically
- **Progressive Networks:** add capacity instead of overwriting

This solves the problem of "an image moves 10 Å during optimization, making its initial-position local GP useless at the final position".

---

## Part 6. Game Theory viewpoint ⭐⭐⭐ (most productive)

### 6.1. NEB as a Potential Game

**Monderer & Shapley 1996.** Potential game: a game where Nash equilibria coincide with local minima of a scalar potential function.

**Formal NEB-as-game:**
- Players: N = {1, ..., 9} (images)
- Strategy of player i: position x_i ∈ ℝᵈ
- Payoff of player i:
  ```
  u_i(x_1, ..., x_N) = −V(x_i) − (k/2)[(x_i − x_{i−1})² + (x_{i+1} − x_i)²]
  ```
- Potential function:
  ```
  Φ(x_1, ..., x_N) = Σᵢ V(x_i) + (k/2) Σᵢ (x_{i+1} − x_i)²
  ```

**Potential game property:** ∂u_i/∂x_i = −∂Φ/∂x_i for all i. **Verification:** ∂u_i/∂x_i = −∇V(x_i) − k[(x_i−x_{i−1}) − (x_{i+1}−x_i)]. ∂Φ/∂x_i = ∇V(x_i) + k[(x_i−x_{i−1}) − (x_{i+1}−x_i)]. ✓

→ **NEB is formally a potential game.**

**What this gives:**

1. **Existence:** the Monderer-Shapley theorem guarantees the existence of a pure Nash equilibrium → independent proof of the existence of a NEB solution (parallel to the Mountain Pass Theorem)

2. **Convergence:** **best-response dynamics** in potential games converge monotonically to a Nash equilibrium (Φ decreases). This explains why naive gradient descent on images works.

3. **Multiplicity = MEP multiplicity:**
   Potential games can have multiple Nash equilibria. **Each equilibrium = candidate path.**
   - "Good" equilibrium = true MEP (correct saddle)
   - "Bad" equilibrium = same-basin trap, alternate saddle, etc.
   
   **Problem reframing:** the same-basin trap is not a technical failure; it is convergence to the **wrong Nash equilibrium**. This is an **equilibrium selection problem** with a rich theoretical literature (Harsanyi-Selten 1988 "A General Theory of Equilibrium Selection", risk dominance, evolutionary stability).

4. **Mechanism design:** we design the payoff (force law) so that selfish play converges to the globally optimal equilibrium. **Current NEB is a poorly designed mechanism**: the payoff structure admits collapse to bad equilibria.

   **Better mechanism design ideas:**
   - Asymmetric springs: different k for basin vs transition regions
   - Time-varying payoff: scheduled k(t) with annealing
   - Side payments: extra reward for images escaping local minima
   - Punishment terms: penalty for collapsing into a neighbor basin

### 6.2. CI-NEB as a Stackelberg Game

**Stackelberg 1934.** Commitment game: the leader moves first, followers respond optimally.

**CI-NEB is structurally a Stackelberg game:**
- Climbing image (leader) commits to the climbing direction
- All other images (followers) react, maintaining path topology

**Stackelberg equilibrium** is generally **better than Nash** for the leader. This formally explains why CI-NEB outperforms plain NEB — the leader-follower structure breaks degenerate Nash equilibria.

**Extensions:**
- **Multi-leader Stackelberg:** several leader images at different stages
- **Dynamic Stackelberg:** leadership transfers between images as the path evolves
- **Stackelberg-Nash hybrid:** leaders play Stackelberg among themselves, followers play Nash

**This yields principally new algorithms**, with no analog in the NEB literature.

### 6.3. Mean-Field Games (MFG)

**Lasry & Lions 2006-2007, Huang-Caines-Malhamé.** Continuum limit of a multi-agent game. Each agent responds to the average field of the others via a **self-consistent HJB + Fokker-Planck pair**:

```
−∂V/∂t + H(x, ∇V) = F[m]        (HJB: each agent's value function)
∂m/∂t − div(m ∇_p H) = 0          (FP: distribution evolves)
m(0) = m_0, m(T) = m_T            (boundary distributions)
```

**For NEB as N→∞:**
- Path γ(s) → flow of agents
- Distribution m(s, x) = density of images at arclength s
- Self-consistent equations couple the local agent model (HJB) with the global field (FP)

**Deep connection to optimal transport:**
- MFG in Lagrangian form = Benamou-Brenier formula
- Benamou-Brenier connects to Wasserstein gradient flow (Part 3.3)
- Wasserstein flow connects to TPT via committor

**This is a triangle** Game-Theory ↔ OT ↔ TPT **that no one has closed in the NEB literature**.

**Practical implication:** gives a **continuous-image NEB** where N→∞ becomes a principle rather than an approximation.

### 6.4. Cooperative Game Theory: Shapley Value

**Shapley 1953.** Fair allocation of payoff in a cooperative game:
```
φ_i = Σ_{S ⊆ N\{i}} (|S|! (n−|S|−1)!)/n! · [v(S∪{i}) − v(S)]
```
where v(S) is the worth of coalition S.

**Application to NEB:**
- v(S) = "how much barrier reduction the coalition S of images provides"
- φ_i = marginal contribution of image i, averaged over all permutations
- **Computation:** Monte Carlo over random permutations (Castro et al. 2009)

**Uses:**

1. **Adaptive DFT budget allocation:** images with high Shapley value → more DFT compute. Active learning is natural.

2. **Image pruning:** φ_i ≈ 0 → image is redundant. Can be removed from path. Variable-N NEB.

3. **Stuck detection:** φ_i drops sharply between iterations → image is stuck, contributing nothing. ALERT.

4. **Convergence insurance:** Aumann-Shapley value gives a **continuous extension** — pricing the contribution of each image in the continuous limit.

**Lit-status:** **no NEB paper with Shapley** (verified lit-scan). This is an open direction.

### 6.5. No-Regret Learning + Correlated Equilibrium

**Online learning** framing: each image is an agent running a no-regret algorithm.

**Algorithms:**
- **Online Gradient Descent (OGD):** Zinkevich 2003
- **Follow The Regularized Leader (FTRL):** Hazan 2016 textbook
- **Multiplicative Weights / Hedge:** Freund-Schapire 1997
- **EXP3 / EXP3.P:** for bandit feedback (when gradient is unknown)

**Theorem (Foster-Vohra 1997, Hart-Mas-Colell 2000):** if all agents use a no-regret algorithm, joint play converges to a **correlated equilibrium** (Aumann 1974). This is weakly weaker than Nash, but achievable without full information and without best-response computation.

**For federated NEB:** each image does not know the global PES, only local information via DFT calls. No-regret learning guarantees convergence to a correlated equilibrium without explicit coordination. **This is a formal federated framework.**

**Practical algorithm:**
```
For each image i, each iteration:
  Observe local gradient g_i (cheap, from V̂_i)
  Update local model V̂_i (continual learning)
  Take FTRL/OGD step: x_i^{k+1} = x_i^k − η·g_i + regularization
  Exchange spring tension λ with neighbors (communication)
```

**What is new:** rigorous convergence guarantees **under minimal per-image information**. Global convexity, smoothness, etc. are not required.

### 6.6. Evolutionary Game Theory (EGT)

**Maynard Smith 1973.** Strategies evolve via replicator dynamics:
```
ẋ_i = x_i (u_i(x) − ū(x))
```

**For NEB:** **distinct path candidates** compete. Selection pressure favors paths with a lower barrier. This is a population-based search (analogous to matrix PSO from the lit-scan), but with a **rigorous evolutionary theorem** (Folk theorem for replicator dynamics).

**ESS (Evolutionarily Stable Strategy):** a path that cannot be invaded by mutations. This is **stronger than Nash** — it guarantees robust convergence.

---

## Part 7. Synthesis: proposed NEB-AGM formulation

**NEB-AGM = NEB as Adaptive Game with Memory.**

A hybrid combining the most productive analogies:

### 7.1. Architecture

```
For each image i ∈ {1, ..., N}:
  • Local quadratic model V̂_i(x) (DDP-style)
    - Initialized: V(x_i⁰) + 0·(x − x_i⁰) + λ·I  (trivial Hessian)
    - Updated: BFGS-per-image on (x_i, ∇V(x_i)) history
    
  • Persistence diagram PH_i (topological signature of local PES)
    - Computed periodically via sublevel filtration of V̂_i
    
  • Role label r_i ∈ {basin, ridge, saddle, climber, stuck, transition}
    - Assigned by classifier C(eigenvals(H_i), |g_i|, displacement_var, PH_i)
    
  • Shapley value φ_i (importance score)
    - Estimated via Monte Carlo permutations every K iterations
    
  • No-regret state (FTRL accumulator) ω_i
    - For role-conditioned update step
```

### 7.2. Game structure

- **Players:** N images
- **Strategies:** positions x_i
- **Payoff (role-conditioned):**
  ```
  u_i(x; r_i, λ) = −V̂_i(x_i) − Σ_{j∈neighbors} λ_{ij}(|x_i − x_j| − h_target)
                   + role_bonus(r_i, x_i)
  ```
- **Mechanism design:** role_bonus designed to break degenerate Nash equilibria. Examples:
  - basin role: small bonus for moving toward path interior
  - stuck role: large bonus for any move away from current basin
  - climber role: bonus for moving along τ̂ (ascending)

### 7.3. Update rule (per iteration)

```
1. Observation phase:
   For each image i:
     If φ_i high OR confidence_low(V̂_i):
       Evaluate true V(x_i), ∇V(x_i) via DFT
       Update V̂_i (continual learning)
     Else:
       Use surrogate V̂_i

2. Role assignment phase:
   For each i: r_i ← C(eigenvals(H_i), |g_i|, PH_i, neighbor_dist)

3. Game step (ADMM-style):
   For each i (parallel via even/odd splitting):
     x_i^{k+1} = argmin_x [V̂_i(x) − u_i(x; r_i, λ^k) + (ρ/2)||consensus residual||²]
   
4. Dual update:
   For each spring (i, i+1):
     λ_{i,i+1}^{k+1} = λ_{i,i+1}^k + ρ(|x_{i+1}^{k+1} − x_i^{k+1}| − h_target)

5. Diagnostic phase (every K iterations):
   - Compute Shapley {φ_i}
   - Compute pairwise PH bottleneck distances → same-basin detection
   - Mechanism design adjustment: if same-basin detected → modify role bonuses

6. Convergence check:
   - max|F_i| < fmax  (standard)
   - AND  Shapley distribution stable
   - AND  no role transitions in last K iterations
   - AND  PH-based diagnostics PASS (no same-basin signature)
```

### 7.4. Theoretical guarantees (what can be proven)

| Property | Source | Conditions |
|---|---|---|
| Existence of Nash equilibrium | Potential game (§6.1) | Φ coercive |
| Convergence to Nash | Best-response in potential game (§6.1) | Lipschitz gradients |
| Provable ADMM step convergence | Boyd 2011 / Wang 2019 | Either convex local f or specific non-convex conditions |
| No-regret bound | Hazan 2016 | Convex local f̂ |
| Shapley axiom satisfaction | Shapley 1953 | Always |

**Honest caveat:** not all hold simultaneously for non-convex V. The open theoretical problem is **consistency of all guarantees** in our setting. This is part of the novelty for a methodology paper.

### 7.5. Empirical predictions

1. **Same-basin trap detection** via PH bottleneck distance → automated termination + restart suggestion. Should catch our known mackinawite / pentlandite / marcasite same-basin incidents retrospectively.

2. **Compute efficiency:** per-image surrogates V̂_i + Shapley-guided DFT allocation should give **2-5× reduction in DFT calls** (similar to ML-NEB, but per-image localization is tighter).

3. **Robust convergence on multi-saddle systems:** mechanism design + role differentiation should reduce sticking at degenerate Nash equilibria.

---

## Part 8. Comparative table of all angles

| Angle | Unique contribution | Theory maturity | Implementation difficulty | Relation to same-basin problem |
|---|---|---|---|---|
| **ADMM / DDM** | Provable convergence + parallelism + per-image surrogate trivially | Mature | Medium | Indirect (via λ dual) |
| **Persistent Homology** | Automated same-basin detection | Mature | Medium-High (PH libs) | **Direct diagnostic** |
| **EnKF** | Bayesian per-image model with consensus | Mature | Low (libraries exist) | Indirect |
| **Replica Exchange** | Stuck escape via temperature swaps | Mature | Low | **Direct treatment** |
| **Wasserstein flows** | Distribution-valued images, OT connection | Maturing | High | Indirect |
| **DDP** | Per-image quadratic V̂_i with rigorous backward pass | Mature in robotics | Medium | Indirect |
| **MoE / Lance** | Role-aware via gating function | Mature | Low (MoE libs) | **Direct (MaPE analog)** |
| **GNN** | Learned attention vs fixed springs | Mature | Low | Indirect |
| **Potential Game** | Existence + multiplicity + **equilibrium selection framing** | Mature | Conceptual | **Reframes the problem** |
| **Stackelberg / CI-NEB** | Leader-follower mechanism, explains CI-NEB | Mature | Low | Indirect |
| **Mean-Field Game** | Continuous-image limit, OT-TPT connection | Mature in math | High | Indirect |
| **Shapley value** | Importance scoring per image | Mature | Medium (MC estimation) | **Diagnostic** |
| **No-Regret learning** | Federated convergence guarantee with minimal info | Mature | Medium | Indirect |
| **Evolutionary game theory** | ESS robust to mutations | Mature | Low | Indirect |

**Top 3 by "direct relevance to our same-basin pain":**
1. Persistent Homology (automated detector)
2. Potential Game framing (reframes problem as equilibrium selection)
3. MoE / role-aware (MaPE-style heterogeneity)

**Top 3 by "publishable novelty":**
1. NEB-AGM hybrid (this document)
2. PH-based same-basin detector as a standalone contribution
3. Potential Game perspective paper as a standalone contribution

---

## Part 9. Concrete experiments (validation roadmap)

### 9.1. Toy potential validation
- **Müller-Brown 2D:** classic NEB benchmark
- Implement NEB-AGM
- Compare: standard NEB vs CI-NEB vs string vs NEB-AGM
- Metric: convergence rate, robustness to bad initial path

### 9.2. Persistent Homology same-basin detector
- Standalone tool: take any converged NEB output, run PH diagnostic
- Run on **actual mack/pent/marc data** (already harvested)
- Verify: PH bottleneck < threshold for same-basin cases (retrospective validation)
- Publishable: short methods note "Persistent Homology as a Diagnostic for NEB Same-Basin Artifacts"

### 9.3. Potential Game perspective paper
- No new algorithm, pure reframing
- Show that:
  - NEB is formally a potential game
  - Same-basin trap = Nash equilibrium selection failure
  - Suggest mechanism design fixes
- Target: J Chem Phys "Perspective" article

### 9.4. Full NEB-AGM on Fe-S V_Fe benchmark
- Implementation on the existing pipeline (ASE + QE)
- Benchmark on mack/pent/marc/greig/pyr
- Honest comparison vs standard CI-NEB
- If wins → algorithmic paper J Chem Theory Comp

---

## Part 10. Publication strategy

**Three-paper potential:**

### Paper A: Diagnostic (low-risk, fast)
- **Title:** "Persistent Homology as a Diagnostic for Same-Basin Artifacts in Nudged Elastic Band Calculations"
- **Format:** Methods note / short paper
- **Time:** 2-3 months
- **Risk:** Low (clean methodological contribution)
- **Target:** J Chem Theory Comp or J Chem Phys Communications

### Paper B: Perspective (medium-risk, fast)
- **Title:** "Game-Theoretic Foundations of Nudged Elastic Band Methods"
- **Format:** Perspective / Position paper
- **Time:** 4-6 months
- **Risk:** Medium (reviewers may ask for a new algorithm)
- **Target:** Annu Rev Phys Chem or J Phys Chem Lett Perspective

### Paper C: Algorithm (high-risk, slow)
- **Title:** "Adaptive Game-Theoretic Nudged Elastic Band with Per-Image Memory (NEB-AGM)"
- **Format:** Full methodology paper + benchmarks
- **Time:** 12-18 months
- **Risk:** High (must beat existing methods convincingly)
- **Target:** J Chem Theory Comp or Comm Phys

**Sequence:** A → B → C. Each subsequent paper builds on the credibility of the previous one. Total = 18-27 months for the 3-paper sequence.

---

## Part 11. Honest self-critique

### 11.1. Where the framing is solid

- Potential game framing: rigorous, verifiable, just needs writing
- ADMM reformulation: mathematically clean, parallels existing literature
- PH diagnostic: directly applicable, low risk

### 11.2. Where it may be over-sold

- **Mean-Field Game connection:** elegant but practical implementation for DFT is hard. Could be hype.
- **NEB-AGM as a monolithic algorithm:** combines many components; debugging will be a nightmare. Modular implementation is safer.
- **"Equilibrium selection" framing:** Harsanyi-Selten theory was developed for finite normal-form games; application to continuous strategy spaces is non-trivial.

### 11.3. Where prior art may be fatal

- **Spring Pair Method (Cao 2024, lit-scan §6.5):** already a 2-agent formulation with explicit communication. If someone generalizes to N agents, our game-theoretic perspective becomes derivative.
- **String method:** "chain of replicas" formalism already exists. Reviewer-1: "this is just the string method with game-theory labels".
- **Replica Exchange NEB:** mentioned in the 2010s. If actively developed — a direct competitor.

### 11.4. Conservative recommendation

**Do not attempt all 3 papers simultaneously.** Start with Paper A (PH diagnostic) — this is a **standalone polishable result** that validates the approach. If the PH detector works on the existing dataset → confidence boost for Papers B and C.

Papers B and C — **only after the main Third Matter paper is submitted**. Not a distraction.

---

## Part 12. Open questions

1. **Convergence theory:** what conditions on V guarantee convergence of NEB-AGM? Open theoretical problem.

2. **Shapley computation cost:** MC estimation requires O(N²) evaluations per round. Acceptable for N=9? Approximation needed?

3. **Role transition stability:** what prevents images from oscillating between roles every iteration? Regularization/hysteresis needed.

4. **PH sensitivity:** what threshold for bottleneck distance? Should depend on the system. Does a universal threshold exist?

5. **Connection to TPT:** committor function via the MFG-OT-TPT triangle — formal proof? Could give rate constants directly.

6. **Mechanism design dictionary:** what specific role_bonus terms provably break degenerate Nash equilibria?

7. **Empirical question:** does this outperform CI-NEB on FeS V_Fe? **Without an empirical win, the theory is not publishable.**

---

## Part 13. Reading list (prioritized)

### Tier 1 (read first, 2-3 weeks)
- Boyd et al. 2011, "Distributed Optimization via ADMM" (Foundations and Trends in ML)
- Monderer & Shapley 1996, "Potential Games" (Games Econ Behav)
- Lasry & Lions 2007, "Mean field games" (Japanese J Math)
- Edelsbrunner & Harer "Computational Topology: An Introduction" (Ch 1-3 for PH basics)

### Tier 2 (foundational, 1-2 months)
- Toselli & Widlund "Domain Decomposition Methods" (selected chapters)
- Hazan "Introduction to Online Convex Optimization" (textbook)
- Harsanyi & Selten 1988 "A General Theory of Equilibrium Selection" (sections on risk dominance)
- Tassa, Mansard, Todorov 2014 "Control-Limited DDP" (IROS)

### Tier 3 (specialized)
- Wang et al. 2019 "ADMM for nonconvex problems" (Math Prog)
- Castro et al. 2009 "Polynomial calculation of Shapley value" (Comput Oper Res)
- Hiraoka et al. 2016 "Hierarchical structures in amorphous solids" (PNAS) — PH in materials
- Carmona & Delarue "Probabilistic Theory of Mean Field Games" (Vol I-II) — for MFG depth

---

## Part 14. Cross-references

**Within the project:**
- `ALTERNATIVES_AND_ROLE_AWARE_NEB.md` (companion document, landscape + lit-scan)
- Empirical same-basin diagnosis notes (MACK/PENT NEB protocol)
- OPES pilot tracker (orthogonal alternative — OPES)
- Cross-mineral V_Fe barrier pattern notes (pattern across Fe-S minerals)
- Hand-crafted same-basin endpoint detector (to be formalized via PH)

**External anchors:**
- ByteDance Lance / MaPE 2026 (inspiration)
- Boyd ADMM (mathematical foundation)
- Monderer-Shapley potential games (game-theoretic foundation)
- Edelsbrunner-Harer PH (topological diagnostic)

---

## Part 15. Status

- **Type:** research direction proposal + framework + concrete formulation
- **Confidence level:**
  - Potential game framing: [FACT] HIGH (just verify and write)
  - PH diagnostic: [FACT] HIGH (testable on existing data)
  - NEB-AGM full algorithm: [HYPOTHESIS] MEDIUM (needs prototype)
  - MFG-OT-TPT triangle: [HYPOTHESIS] SPECULATIVE (deep math, hard to prove)
- **When to return:** after the Third Matter main paper is submitted
- **Who should review:** mathematician (game theory rigor), computer scientist (algorithm complexity), physicist (NEB practical), stat-mech theorist (TPT/MFG connection)
- **Nearest action (after current priorities):** PH diagnostic prototype on existing mack/pent/marc data — **standalone, low-risk, publishable independently**.

**Do not implement now.** Parking lot for post-Third-Matter research program.
