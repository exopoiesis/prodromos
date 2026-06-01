# NEB Alternatives and the Role-Aware NEB Concept with Local Memory

**Context:** Third Matter project, same-basin trap problem in FeS V_Fe NEB (mack/pent/marc)
**Status:** seed document for future research / methodology paper

---

## Part 0. Problem Statement

> "We have a mountain range with many peaks, and we need to find the optimal route through the valleys (for a caravan) connecting 9 points, with the constraint that there are elastic springs between the points that prevent them from spreading too far apart or collapsing too close together. It is fairly well established that such an optimal route exists — perhaps not running through the absolute lowest valleys, perhaps slightly higher on the slopes, but it does exist. The task is to solve this mathematically."

This is the classical **Minimum Energy Path (MEP)** problem on a potential energy surface.

**Correspondence:**
- Mountain range = potential energy surface V(x) (multi-dimensional)
- 9 points = 9 images in NEB chain
- Elastic springs = elastic spring forces along the path tangent
- "Must not expand/contract" = equidistant constraint (holonomic)
- "Route not through the lowest valleys, but higher" = MEP passes through a saddle point, not the absolute minimum
- "Such a route exists" = **Mountain Pass Theorem** (Ambrosetti-Rabinowitz)

This formulation is **mathematically precise**: NEB is reformulated from first principles, which independently leads to the Mountain Pass Theorem.

---

## Part 1. Five Fundamentally Different Mathematical Views of MEP

NEB is only one approach. There are at least 5 non-equivalent mathematical formulations.

### 1.1. Variational View (what NEB essentially is)

Minimization of the action functional:
```
S[γ] = ∫₀¹ L(γ(s), γ̇(s)) ds
```
with boundary conditions γ(0) = A (reactant), γ(1) = B (product).

**NEB** is a specific discretized gradient descent with a holonomic constraint, implemented via springs. Equivalent formulation with Lagrange multiplier:
```
F_i = -∇V(x_i)|_⊥ + k(|x_{i+1} - x_i| - |x_i - x_{i-1}|) τ̂_i
```
where τ̂_i is the tangent estimate (Henkelman improved tangent), k is the spring constant.

**Alternatives within the variational formulation:**

- **String method** (E & Vanden-Eijnden 2002). Instead of springs — *reparametrization* to equidistant spacing at each iteration. Often more stable on stiff potentials, especially for same-basin traps. Compute cost comparable to NEB, but spring oscillation pathologies are eliminated.
- **Simplified string method** (E, Ren, Vanden-Eijnden 2007). Even simpler: only steepest descent + spline reparametrization.
- **Growing String Method** (Peters et al. 2004). The path *grows* from both endpoints toward each other. Does not require a good initial guess (unlike our IDPP). Useful when A and B differ greatly.
- **Geometric MEP / MaxFlux** (Olender & Elber 1996). Path as a geodesic in Riemannian metric `g_ij = δ_ij · exp(V/kT)`. Purely geometric, no springs. Numerically problematic when T→0 (metric becomes singular).
- **Doubly Nudged Elastic Band (DNEB)** (Trygubenko & Wales 2004). Adds a second projection to combat corner-cutting.

### 1.2. Topological View — Mountain Pass Theorem

**Ambrosetti-Rabinowitz (1973):** between two minima on a coercive potential, a critical point of a specific Morse index (saddle) **is guaranteed to exist**.

Formally: let V: ℝⁿ → ℝ have two local minima A and B. Then

```
c = inf_{γ ∈ Γ} max_{s ∈ [0,1]} V(γ(s))
```

where Γ is the set of paths from A to B, is a critical value of V. It is attained at a saddle point.

This is an **existence theorem** ("such a route exists"). Methods that search for the saddle without knowledge of the path derive from this theorem:

- **Dimer method** (Henkelman & Jónsson 1999). Two nearby structures → eigenvector estimate → ascent along minimum curvature direction. Does not require endpoint B!
- **Gentlest Ascent Dynamics (GAD)** (E & Zhou 2011). Continuous dynamics toward the saddle via eigenvector following.
- **Eigenvector following / Lanczos saddle search**.
- **Activation-Relaxation Technique (ART)** (Mousseau et al.).

After finding the saddle, the path is recovered by **steepest descent in both directions** (Intrinsic Reaction Coordinate, IRC). This is the inverse direction of NEB.

### 1.3. Probabilistic View — Transition Path Theory (TPT)

**E & Vanden-Eijnden 2006, Metzner et al. 2009.** Instead of "one optimal path" — an *ensemble* of reactive trajectories.

The key object is the **committor function** `q(x)` = probability that a trajectory starting at x reaches B before A:
```
L q = 0 in Ω \ (A ∪ B)
q|_A = 0, q|_B = 1
```
where L is the generator of the stochastic dynamics (e.g., overdamped Langevin: L = -∇V·∇ + kT Δ).

MEP is a special case of TPT at T → 0: instanton, exponential concentration on the most probable path (Freidlin-Wentzell).

**Methods:**
- **Transition Path Sampling (TPS)** (Bolhuis & Chandler 1998). MCMC over trajectories, not configurations.
- **Transition Interface Sampling (TIS)** (van Erp et al.). Layer-by-layer.
- **Forward Flux Sampling (FFS)** (Allen, Frenkel et al.). For rare events.
- **Milestoning** (Faradjian & Elber 2004).

**Advantage:** provides *kinetics* (rate constant), not just a barrier. This is ultimately what we need for the paper.

**Disadvantage for DFT:** requires hundreds/thousands of trajectories. Impossible at $1/hr GPU per MD step.

### 1.4. Optimal Control / Freidlin-Wentzell

Path = solution to the Freidlin-Wentzell action minimization problem for stochastic dynamics:
```
dx = -∇V(x) dt + √(2kT) dW
```

Action functional:
```
I[γ] = (1/4kT) ∫₀^T |γ̇ + ∇V(γ)|² dt
```

Minimizing I[γ] subject to γ(0) = A, γ(T) = B gives the **instanton path**. At T → 0 it collapses to the MEP.

This reformulation via the **Pontryagin maximum principle** is an optimal control problem:
- "Control" u(t) = γ̇ + ∇V(γ)
- Objective function: ∫ |u|² dt
- Constraint: dynamics, boundary conditions

The connection to control theory opens up a suite of methods: HJB equations, model predictive control, reinforcement learning for path-finding.

**Practical methods:**
- **Minimum Action Method (MAM)** (E, Ren, Vanden-Eijnden 2004).
- **Adaptive MAM** (Zhou et al. 2008).
- **gMAM** (geometric MAM) (Heymann & Vanden-Eijnden 2008) — for overdamped Langevin without T.

### 1.5. ML / Generative Approaches (hot 2024-2026)

- **Boltzmann Generators** (Noé et al. 2019). Normalizing flows for direct sampling from p(x) ∝ exp(-V/kT). Yield paths as samples.
- **Diffusion models for transition paths** (several preprints 2024-2025). Score matching on conditional distributions p(γ | A, B).
- **Implicit Transition Path Sampling** with score matching.
- **GFlowNets for path sampling** (Bengio et al.).
- **NeuralMD / Equivariant transformers** for direct parameterization of γ_θ(s).
- **OPES** (Invernizzi-Parrinello 2020-2024). On-the-fly Probability Enhanced Sampling. Fundamentally different approach: collective variable bias, not path-based.

**Connection with MLIPs:** MACE, NequIP, Allegro, CHGNet — *accumulate knowledge of the PES*, but do not use it for path routing.

---

## Part 2. Why Other Researchers Do Not Consider Alternatives

The honest answer is a mix of technical reasons (~40%) and sociology of science (~60%).

### 2.1. Technical Reasons

**1. Cost-method mismatch with DFT.**
- TPS/FFS require *hundreds* of MD trajectories — impossible at DFT $1/hr GPU
- Diffusion models / Boltzmann generators require a *training set* — materials systems lack one (unlike proteins/water)
- String method ≈ same cost as NEB, but requires an interpolation step where accuracy is lost
- MAM/gMAM require numerical optimization in high dimension

**2. Convergence without ground truth.**
- Methods papers compare on toy potentials (Müller-Brown, LJ Lennard-Jones) where the solution is known
- For real FeS nobody knows the true MEP → no benchmark → each group reports its own result on its own system, no comparison possible
- Reviewers cannot distinguish "method failed" from "method correct, system weird"

**3. Numerical pathologies.**
- Geometric MEP (Olender-Elber) at T→0: metric `exp(V/kT)` becomes singular, numerically unstable
- String method with poor interpolation → corner cutting in curved valleys
- Dimer method: if eigenvalue spectrum is poor (degenerate), does not converge

### 2.2. Sociological Reasons

**4. Software lock-in.**
- VASP, QE, ASE — NEB integrated, plug-and-play
- String method: Vanden-Eijnden's code in Matlab, not integrated into DFT superstructure
- OpenPathSampling, Pyretis (TPS) — separate ecosystem, steep learning curve
- "Who would write String for ABACUS on A100?" — nobody, because there is no grant for this

**5. Community fragmentation.**

| Method | Community | Cited journals |
|---|---|---|
| NEB | materials/catalysis | Phys Rev B, J Chem Phys, J Am Chem Soc |
| TPS/TPT | biophysics/proteins | PNAS, J Chem Theory Comp, J Phys Chem B |
| MAM/optimal control | applied math | SIAM, Comm Pure Appl Math, J Comp Phys |
| ML/Diffusion | ML community | NeurIPS, ICML, ICLR |

These communities barely overlap at conferences. An FeS chemist has never heard of Freidlin-Wentzell. A diffusion-model researcher does not know what fmax convergence in NEB means.

**6. Pedagogical inertia.**
- Sholl-Steckel "Density Functional Theory" — only NEB
- Jensen "Computational Chemistry" — mentions Dimer in passing
- First-year PhD students learn NEB → after 10 years write grants for NEB → become reviewers for NEB papers
- The cycle reproduces. Nobody teaches the string method in DFT courses.

**7. Survivorship bias in the literature.**
- Published NEB results are the ones that converged. Those that did not converge are attributed to "system complexity", not to the method
- Nobody writes a paper "NEB did not work on our system" — career suicide
- The **same-basin trap** we observe (mack/pent/marc) is a known problem in a narrow circle (Henkelman group discusses it), but does not appear in textbooks

**8. Confirmation bias at the field level.**
- "Standard methodology" passes peer review faster
- Reviewer: "Why didn't you use NEB? It's standard" — requires 2-3 pages of justification in SI
- Using NEB = 0 friction; using String/GAD = explain to 3 reviewers

**9. Selection bias on systems.**
- Easy barriers (single saddle, clear endpoints) → NEB works → publish → reinforce NEB
- Hard systems (floppy, multi-saddle, same-basin) → researchers *switch to easier systems*, not to a different method
- This explains why FeS is so underexplored — it is squarely in the "hard" category

**10. Citation/career economics.**
- A methods paper on String method accumulates ~500 citations over 20 years
- An applications paper "NEB on a new catalyst" — 200 citations in 3 years
- PhD/postdoc chooses applications → NEB

### 2.3. What Is Changing Now (2024-2026)

- **Finite-T string method** is growing in free-energy applications (Parrinello, Vanden-Eijnden collaborations)
- **ML committors** (Roux 2024, Vanden-Eijnden 2025) — TPT is becoming accessible
- **OPES** (Invernizzi) is actively displacing metadynamics → potentially NEB for kinetics as well
- **Diffusion models for paths** — several preprints 2025 on arXiv, but zero in materials journals so far
- **Equivariant neural networks** for path parameterization — emerging

### 2.4. Implication for Our Paper

The same-basin trap in FeS is **literally a signal that we are at the frontier**, not a methodological failure. Most published Fe-S NEB papers are either surface calculations (easier), or vacancies in simple metals (easier), or short hops (easier). **Bulk Fe-S with competing saddles is underexplored precisely because NEB is finicky on it**, and the alternatives are inconvenient in DFT pipelines.

This works in our favor: **"we benchmarked NEB and identified a failure mode in the Fe-S V_Fe class"** — this is a valid scientific contribution, not a shortcoming.

---

## Part 3. Idea: Role-Aware NEB with Local Memory (Inspired by Lance/MaPE)

### 3.1. Lance/MaPE — What It Is and Why It Works

ByteDance Lance (2026, https://lance-project.github.io/):
- Dual-stream MoE architecture for image/video understanding + generation
- 6 billion shared, 3 billion active parameters
- **Key innovation: MaPE (Mixed-role aware Position Encoding)**

Standard positional encoding in a transformer tells a token "where you are" (spatial/temporal coordinates). MaPE adds "**why you are here**" — a role label:
- "I am for understanding" (input image for analysis)
- "I am a condition" (text prompt, reference image)
- "I am being generated" (noisy token at denoising stage)

Without a role label, the transformer confuses heterogeneous visual tokens within the same sequence → degradation on mixed tasks. With MaPE the separation is clean.

### 3.2. Intuition

> "Perhaps something could accelerate the calculation... an approach where at each step each cell accumulates knowledge and can move toward the optimal point, without failures"

This contains **two independent components**:
1. **Memory/surrogate per cell:** each cell (image) accumulates knowledge
2. **Role-aware updates:** cells have different roles and are processed differently

### 3.3. Component 1: Local Memory per Image

**Already exists** in the NEB literature under the name **ML-NEB / GP-NEB**:
- **Koistinen et al. 2017** (J Chem Phys 147, 152720) — Gaussian Process surrogate globally for the entire path
- **Garrido Torres et al. 2019** (Phys Rev Lett 122, 156001) — active learning NEB with GP
- **MACE-NEB / NequIP-NEB** — what we are currently doing

However, in existing approaches the surrogate is **global** (one GP/MLIP for the entire path).

The intuition **"each image — its own knowledge-accumulating cell"** is **less explored**:
- Image in basin → accumulates curvature of the basin floor (local Hessian estimate)
- Image on ridge → accumulates eigenvector saddle direction
- Image in transition → accumulates tangent direction history

**Advantage of the local approach:**
- 9 cheap local models instead of 1 expensive global one
- Sparse data per image (only local neighborhood) → fast GP convergence
- Statistical treatment: a stuck image has statistics in one basin → detectable → can be forcibly ejected from that basin

**This is a genuine research gap**, especially for the same-basin trap (our problem).

### 3.4. Component 2: Role-Aware Updates (MaPE Analog)

**Also partially exists**, but in extremely primitive form:
- **Climbing Image NEB** (Henkelman et al. 2000) = 1 image has the role "climber", the rest are identical
- **Multi-climbing NEB** = several climbers (for multi-saddle paths)
- **Adaptive image NEB** (Maras et al. 2016) = roles assigned by curvature

But the MaPE analog goes further. In standard NEB currently:
- All 7 middle images receive the **same force law**: F_i = -∇V|_⊥ + k(Δs_right - Δs_left)τ̂
- Climbing image (if CI-NEB) has a **different force law**: F_CI = -∇V + 2(∇V·τ̂)τ̂
- This **binary scheme** is a loss of information

**What a MaPE analog could provide:**

| Role | Force law | When |
|---|---|---|
| `basin-slider` | F = -∇V (pure gradient) | image deep in basin, |∇V| small, eigenvalues>0 |
| `ridge-walker` | F = -∇V\|_⊥ + soft spring | standard NEB behavior |
| `saddle-approacher` | F = -∇V + α(∇V·τ̂)τ̂, α∈[0,2] | near path maximum, large \|∇V·τ̂\| |
| `climber` (CI) | F = -∇V + 2(∇V·τ̂)τ̂ | one image at the peak |
| `stuck-in-wrong-basin` | F = -∇V + β·escape_direction | detected via local displacement statistics |
| `transition-walker` | F = -∇V\|_⊥ + stiff spring + reparametrize | passing through a narrow bottleneck between basins |

**Labels are updated** every N iterations based on local statistics:
- Curvature spectrum (eigenvalues of local Hessian estimate)
- Displacement history (variance of last k steps)
- Neighbor distance (if distance to neighbor is much smaller than average → potentially stuck)
- Role confidence (Bayesian-like posterior)

### 3.5. Fundamental Differences from Lance — What Does NOT Transfer

To avoid overstretching the analogy, the boundary is as follows:

| Lance/MaPE | NEB |
|---|---|
| Transformer attention — **global** connection between all tokens | NEB springs — **local** (nearest neighbors only) |
| Roles needed to avoid **interference** in attention | NEB has no interference — forces are local |
| Learns from **millions** of examples | NEB solves **1** task, ~50-200 iter |
| Tokens — discrete (vocabulary) | Images — continuous (coordinate space) |
| Generation vs understanding — **semantic difference** | Basin vs saddle — **geometric difference** |

**Key conceptual distinction:** MaPE solves the problem of *semantic confusion* in a mixed sequence. NEB does *not* have a confusion problem — it has the problem of *treating geometrically heterogeneous images identically*. These are different diseases. But **the solution is structurally similar** — role labels + role-conditional processing.

### 3.6. Combining the Components: Proposed Algorithm

**Adaptive Role-Aware NEB with Local Memory (ARNN-LM):**

```
Initialization:
  - 9 images (IDPP or linear interpolation)
  - Each image i: GP_i = GaussianProcess(kernel=RBF, prior=zero)
  - Each image i: role_i = "transition-walker" (initial guess)
  - History: trajectory_i = [], gradient_history_i = []

For iteration = 1, 2, ..., max_iter:

  # Step 1: DFT calls (with active learning)
  For each image i:
    If confidence(GP_i) > threshold AND iteration > burn_in:
      grad_i = GP_i.predict_gradient(x_i)
      [cheap shortcut]
    Else:
      grad_i = DFT(x_i)
      GP_i.update(x_i, V(x_i), grad_i)
      [accumulates local knowledge]
    trajectory_i.append(x_i)
    gradient_history_i.append(grad_i)

  # Step 2: Role assignment
  For each image i:
    eigenvals_i = GP_i.estimate_local_hessian_spectrum(x_i)
    displacement_var_i = var(trajectory_i[-k:])
    neighbor_dist_i = (|x_{i+1}-x_i| + |x_i-x_{i-1}|)/2

    role_i = classify_role(
      eigenvals_i,
      |grad_i|,
      grad_i · τ̂_i,
      displacement_var_i,
      neighbor_dist_i
    )
    # Returns one of: basin-slider, ridge-walker, saddle-approacher,
    #                 climber, stuck, transition-walker

  # Step 3: Role-conditional force computation
  For each image i:
    F_i = compute_force(grad_i, τ̂_i, role_i, spring_constants[role_i])

  # Step 4: Optimizer step (FIRE or BFGS)
  x_new = optimizer.step(x, F)

  # Step 5: Check convergence
  If max|F_i| < fmax AND no role changes for 5 iter:
    converged
```

### 3.7. Is This Substantially Different from Existing Work?

- **From ML-NEB:** that uses 1 global surrogate (1 MLIP/GP for the whole path), 1 force law. Here — 9 local + 6 force laws.
- **From adaptive NEB (Maras):** that assigns roles via a simple one-shot curvature snapshot, no memory, no surrogate.
- **From CI-NEB:** that uses binary classification: climber vs non-climber.
- **From string method:** no roles at all, only reparametrization.

**Yes, this is publishable novelty.**

### 3.8. Connection to Our Pain — Same-Basin Trap

Our diagnosis:
- Endpoints A and B are relaxed with BFGS → both land in the same local minimum (same basin)
- All 9 images are interpolated between two "neighboring" points in one basin
- NEB "converges" to a flat path with dE < SCF noise
- The apparent barrier is an artifact of size 0±20 meV

**How role-aware NEB addresses this:**
1. After burn-in, each image has a local GP with an estimate of the curvature spectrum
2. If **all 9 images** have eigenvalues > 0 (locally convex) → the ENTIRE path is in a basin
3. Roles "basin-slider" are assigned → force = -∇V → collapse to minimum
4. **Trigger:** GP-estimated barrier < threshold AND all-basin diagnostic → ABORT with diagnosis "same-basin endpoints"
5. Before re-launch, a **structural perturbation** of the endpoints is proposed (break degeneracy) — exactly what we do manually via the "post-BFGS structural sanity gate"

**This is an automated version of our hard-won lesson.**

### 3.9. Realistic Assessment of Prospects

- **Completely crazy?** No, the individual components exist in the literature
- **Already done?** Local surrogates per-image + dynamic role assignment + escape force — **no publication found**. May exist in 2025-2026 preprints, lit-search needed
- **Will it solve the same-basin trap?** Possibly. Stuck-detection via local statistics + escape force = direct therapy for our problem
- **DFT applicability?** The compute overhead of local GP is negligible (~ms) against 1 DFT call (~hours). Should work
- **How much effort?** ~3-6 months for a prototype on our pipeline. Not our current priority, but **record as an idea for a future paper**
- **Where to publish?** J Chem Phys (if methodology focus) or Phys Rev Materials (if application focus on Fe-S benchmarks)

### 3.10. Extension to the General Task (Not NEB-Specific)

The more general idea — "cells with memory moving toward the optimum without failure" — is actually a **well-known paradigm** in optimization:
- **Distributed/multi-agent optimization** (consensus algorithms)
- **Particle Swarm Optimization** with memory (Kennedy & Eberhart 1995)
- **Agent-based MEP** (several obscure papers from the 2010s)
- **Federated learning** in optimization context

NEB can be reinterpreted as **federated learning over path-points**, where springs = communication protocol, locally accumulated knowledge = local model parameters, role labels = client class.

**This is an interesting conceptual frame, but the mainstream optimization community has not gone there** for the same sociological reasons described in Part 2 (community fragmentation: optimization theorists and NEB practitioners do not overlap).

---

## Part 4. Action Items

### 4.1. Immediate (next sessions)
- [ ] lit-search on 2024-2026 papers on "role-aware NEB", "adaptive image NEB", "GP per image NEB", "local surrogate NEB"
- [ ] Check our mack/pent/marc results for "all-eigenvalues-positive" signature — if confirmed, this is retrospective validation of the stuck-detector idea

### 4.2. Medium-term (if the idea survives lit-review)
- [ ] Prototype on Müller-Brown toy potential (3 images, 2D, 1 hour of work) — proof of concept
- [ ] Extension to LJ7 cluster transition (classical NEB benchmark) — sanity check
- [ ] FeS V_Fe pilot — try on our provenly-broken cases

### 4.3. Long-term (methodology paper)
- [ ] Strict comparison: NEB vs CI-NEB vs string vs ARNN-LM on a set of canonical hard problems
- [ ] Benchmark set: include FeS V_Fe (mack/pent/marc) — this is our unique contribution
- [ ] Theory: convergence properties of dynamic role assignment, proof that role-changes do not destabilize the fixed point

---

## Part 5. Connection to Related Project Files

**Closely related:**
- the MACK/PENT NEB protocol notes — same-basin trap diagnosis
- the OPES pilot tracker — alternative approach via OPES
- the pyrite V_Fe experiment plan — next test cubic vs ortho
- the cross-mineral V_Fe barrier pattern notes — pattern across 3 minerals
- the MLIP benchmark protocol — where MLIPs are already used
- the 8-step RC-check pre-flight notes
- the same-basin endpoint diagnosis rule
- the NEB-reuse triple-picker technical-trap note

**Conceptually related:**
- Mountain Pass Theorem ↔ "existence of MEP" intuition
- TPT/committor ↔ what we ultimately need for kinetics (rate)
- OPES ↔ orthogonal alternative to NEB via CV bias

---

## Part 6. Literature Anchors (for future lit-search)

**Classic NEB:**
- Henkelman & Jónsson 2000, "Improved tangent estimate in the nudged elastic band method", J Chem Phys 113, 9978
- Henkelman, Uberuaga, Jónsson 2000, "A climbing image NEB method", J Chem Phys 113, 9901

**String method:**
- E, Ren, Vanden-Eijnden 2002, "String method for the study of rare events", Phys Rev B 66, 052301
- E, Ren, Vanden-Eijnden 2007, "Simplified and improved string method", J Chem Phys 126, 164103

**Saddle search without endpoint:**
- Henkelman & Jónsson 1999, "A dimer method for finding saddle points on high dimensional potential surfaces", J Chem Phys 111, 7010
- E & Zhou 2011, "Gentlest ascent dynamics", Nonlinearity 24, 1831

**TPT / TPS:**
- Bolhuis, Chandler, Dellago, Geissler 2002, "Transition Path Sampling: Throwing Ropes Over Rough Mountain Passes", Annu Rev Phys Chem 53, 291
- E, Vanden-Eijnden 2006, "Towards a theory of transition paths", J Stat Phys 123, 503

**ML-NEB:**
- Koistinen et al. 2017, "Nudged elastic band calculations accelerated with Gaussian process regression", J Chem Phys 147, 152720
- Garrido Torres et al. 2019, "Low-scaling algorithm for nudged elastic band calculations using a surrogate machine learning model", Phys Rev Lett 122, 156001

**Mountain Pass Theorem:**
- Ambrosetti & Rabinowitz 1973, "Dual variational methods in critical point theory and applications", J Funct Anal 14, 349

**OPES (our current orthogonal approach):**
- Invernizzi & Parrinello 2020, "Rethinking metadynamics: from bias potentials to probability distributions", J Phys Chem Lett 11, 2731

**Lance/MaPE (inspiration):**
- ByteDance Intelligent Creation Lab, "Lance: Unified Image and Video Understanding/Generation/Editing", 2026, https://lance-project.github.io/

---

## Part 6.5. Lit-Scan: NEB as Federated/Multi-Agent Optimization

**Type:** surface-level scan (~25 min of work, ~12 WebSearch queries). NOT a full review.
**Focus:** 2024-2026 papers at the intersection of NEB/MEP methodology and distributed/federated/multi-agent ML.

### 6.5.A. No Direct Match Found

Following targeted searches on combinations:
- "NEB federated", "elastic band distributed optimization"
- "minimum energy path multi-agent", "string method distributed consensus"
- "client server molecular optimization federated"

**No paper explicitly reinterpreting NEB as a federated learning / consensus optimization framework exists in the open literature.**

The conceptual bridge "images = clients, springs = communication protocol, climbing image = special role" — **has not been published in this form**. This is an empty cell in the literature as of 2026-05.

### 6.5.B. Adjacent Work (Conceptually Close, Risk of Prior Art)

Sorted by relevance to the framing "per-client local task + coordinated path":

1. **⚠️ Spring Pair Method (SPM), Cao et al. 2024** — arxiv 2407.04373
   - "Single pair of spring-coupled particles" evolves on PES, using MEP as compass
   - **2-agent saddle search with explicit spring communication** — conceptually the closest to federated framing
   - Not generalized to N images / distributed framework
   - **The most dangerous prior art for the algorithmic part of our idea**

2. **Matrix Particle Swarm Optimization for transition paths in solids**, Qian et al. (npj Comp Mat) — nature.com/articles/s41524-020-0286-9
   - PSO-swarm (multi-agent!) searches for MEP in pressure-induced structural transformations
   - Each particle = candidate path
   - **Multi-agent explicitly**, but population-based metaheuristic, not federated with local models

3. **Efficient NEB via Neural Network Bayesian Algorithm Execution** — arxiv 2512.14993 (2025-12)
   - Global surrogate NN + Bayesian active learning
   - "1-2 orders of magnitude reduction in force evaluations"
   - Surrogate is **global**, not per-image. Centralized active learning, not federated

4. **⚠️ Enhanced CI-NEB with Hessian Eigenmode Alignment** — arxiv 2601.12630 (2026)
   - Hybrid CI-NEB + min-mode following
   - **Role-aware**: climbing image receives special dynamics with Hessian eigenmode
   - **The closest to "role-aware NEB" framing** found in the search
   - Binary role (climber vs others), but the direction is correct

5. **⚠️ Modified NEB with adaptive spring lengths (Onsager-Machlup discretization)** — arxiv 2106.06275
   - Replaces harmonic spring with discretized OM-action → **springs become adaptive along the path**
   - Directly relevant to "springs = adaptive communication channel" framing
   - A continuous formalism, not federated

6. **Action-Minimization Meets Generative Modeling (Onsager-Machlup)** — arxiv 2504.18506 (2025)
7. **GAP: Guided Diffusion for A Priori Transition State Sampling** — openreview 2025
8. **Flow matching for reaction pathway generation** — arxiv 2507.10530 (2025-07)
9. **Generative Model for Reaction Path (Hayashi et al., JCTC 2025)** — arxiv 2401.10721
10. **Diffusion methods for generating transition paths (Triplett & Lu)** — arxiv 2309.10276
11. **Deep RL for Most Probable Transition Pathway** — arxiv 2304.12994 + 2404.05905 (2024)

### 6.5.C. Active Researchers / Groups at the Intersection

| Name | Institution | Direction | Recent work |
|---|---|---|---|
| **Hannes Jónsson** | U Iceland | Original NEB author, GP-accelerated saddle searches | GPR-accelerated min-mode following, arxiv 2505.12519 (2025-05) |
| **Graeme Henkelman** | UT Austin | Climbing-image NEB, ML-acceleration | No direct federated work |
| **Jianfeng Lu** | Duke | Applied math + TPT, score-based generative | arxiv 2309.10276 for rare transitions |
| **Pratyush Tiwary** | UMD | TPS + deep learning CV | 2025 perspective "ML and Statistical Mechanics" |
| **Frank Noé** | FU Berlin / Microsoft | Boltzmann generators, neural samplers | Cross-pollinating, not NEB per se |
| **Cecilia Clementi** | FU Berlin | Coarse-graining + ML for path sampling | Similar profile to Noé |
| **Weinan E / E. Vanden-Eijnden** | Princeton / NYU | String method theorists, TPT | Formalism already close ("chain of replicas"), but not federated |

### 6.5.D. Honest Gap Assessment

**Is this publishable novelty?** Most likely **YES, as a methodological reframing** — with caveats:

**What makes the idea publishable:**
- No direct match found (verified)
- Spring Pair Method (2024) and matrix PSO are the closest neighbors, but N=2 or population-based, not federated with heterogeneity/role-awareness
- Existing NEB literature treats images as **homogeneous degrees of freedom of a single optimization variable**, not as independent agents with local objectives

**What reduces novelty:**
- String method already implies "chain of replicas" — formalism is close
- Adaptive springs already explored (Onsager-Machlup version, arxiv 2106.06275)
- Climbing image is already "role-aware" in a broad sense. Hessian-eigenmode CI-NEB (arxiv 2601.12630) makes this even more explicit
- Per-image GP surrogates have been discussed (though dominant practice is global GP with per-image acquisition)

**Best-guess publication strategy:**
- **Position / perspective paper:** "NEB through the lens of federated/consensus optimization" with concrete takeaways (heterogeneous per-image surrogates with consensus update on springs → better for multi-basin paths). Low risk of reviewer pushback.
- **Algorithmic paper:** requires concrete benefit (e.g., our FeS same-basin trap as demonstration), otherwise risk of "this is just string method with labels". SPM 2024 is the most dangerous prior art.
- **Hybrid:** position section + concrete algorithm + FeS V_Fe benchmark — optimal for J Chem Theory Comp or Phys Rev Materials.

### 6.5.E. Most Likely Missed (Queries for Deeper Search)

If we decide to develop the idea, these additional searches are needed:

1. `"image-parallel" NEB GP local Bayesian "uncertainty quantification" 2024 2025`
   — search specifically for per-image UQ in acceleration papers. May be hidden in GPAW/ORCA/VASP implementations
2. `"replica" "local model" "consensus" molecular path optimization 2023 2024 2025`
   — intersection with replica exchange community, where "local Hamiltonian, global path" appears
3. `"distributed" "saddle search" OR "transition state" molecular 2024 2025 reinforcement learning`
   — RL framing may overlap with multi-agent in Eldar/Bertsekas style
4. **Forward-citation search** from original NEB (Mills/Jónsson 1995) with filter `federated OR distributed OR consensus`
   — if anyone did this, they should cite the foundational paper

### 6.5.F. Action Items after Lit-Scan

- [ ] **Read first** (top-3 prior art):
  - arxiv 2407.04373 — Spring Pair Method (Cao 2024)
  - arxiv 2601.12630 — Hessian-eigenmode CI-NEB (2026)
  - arxiv 2106.06275 — Onsager-Machlup adaptive springs
- [ ] If these 3 do not close the idea → launch deep lit-search with queries from 6.5.E
- [ ] Decide: position paper vs algorithmic paper vs hybrid

### 6.5.G. References

- Spring Pair Method (Cao 2024): https://arxiv.org/pdf/2407.04373
- Matrix PSO transition paths: https://www.nature.com/articles/s41524-020-0286-9
- Efficient NEB via NN BAE (2025): https://arxiv.org/pdf/2512.14993
- Action-Minimization × Generative Modeling (2025): https://arxiv.org/pdf/2504.18506
- Flow matching reaction pathway (2025): https://arxiv.org/abs/2507.10530
- Generative Model Reaction Path (Hayashi 2025): https://arxiv.org/abs/2401.10721
- Modified NEB OM adaptive springs: https://arxiv.org/pdf/2106.06275
- Hessian-eigenmode CI-NEB (2026): https://arxiv.org/pdf/2601.12630
- Diffusion transition paths (Triplett & Lu): https://arxiv.org/pdf/2309.10276
- GPR min-mode following (Jónsson 2025): https://arxiv.org/pdf/2505.12519
- Deep RL transition pathway (2024): https://arxiv.org/pdf/2404.05905
- Low-scaling NEB surrogate ML (Koistinen 2018 foundational): https://arxiv.org/abs/1811.08022

---

## Part 7. Document Status

- **Type:** brainstorm + research direction proposal
- **Confidence level:** idea is viable at the conceptual level, requires lit-review to confirm novelty
- **When to revisit:** after lit-search; if nothing similar is found → prototype on toy potential
- **Who should review:** physicist (NEB methodology), mathematician (variational/optimal control angle), computer scientist (GP/surrogate design), statistical mechanics theorist (TPT connection)

**Do not implement now.** Parallel track alongside the main paper. Possibly the next paper after Third Matter.
