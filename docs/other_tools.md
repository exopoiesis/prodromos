# Related work: NEB / saddle-search algorithms

An important clarification first: SPM, OM/adaptive springs, dyNEB, OCINEB, and GP-NEB are
**not** "tools" at the level of QE / ABACUS / MACE / CHGNet / jDFTx. They are mostly
*algorithmic variants* of NEB / saddle search / path-optimization accelerators. They sit
**on top of** an energy/force calculator:

```
QE / ABACUS / jDFTx / MACE / CHGNet
        give E(R), F(R)
              ↓
NEB / string / dyNEB / GP-NEB / OCINEB / our NEB-AGM
        move images / the path
              ↓
pre-flight verdict: GO / NO-GO / INVESTIGATE
```

For our purpose they matter not as "switch to different software," but as a **map of known
mathematical solutions** — so we know what already exists, what ideas can be borrowed, and
where our problem is genuinely different.

## The point

The right goal right now is sharper than "beat everyone":
**do not launch an expensive DFT NEB until it has been checked, mathematically /
topologically, that the input is meaningful.**

In other words, we are not building "the fastest NEB" — we are building a **pre-flight
certificate / fail-fast framework.**

---

## 1. SPM — Spring Pair Method

**Source:** Cui & Jiang, 2024, *"A spring pair method of finding saddle points using the
minimum energy path as a compass"* — https://arxiv.org/abs/2407.04373

**What it is.** A method for finding a saddle point through two spring-coupled particles,
rather than a full NEB chain of 7–15 images.

```
Ordinary NEB:   A -- img1 -- img2 -- ... -- imgN -- B
SPM:            particle 1 -- spring -- particle 2
```

The pair moves across the PES so as to approach the MEP, orient itself along the MEP
tangent, and use that direction to climb to the saddle.

**Why it was invented.** Ordinary saddle-search methods often require a Hessian/eigenvector
or can walk to an irrelevant saddle. SPM tries to obtain the saddle direction from the
*local geometry of the MEP*, without a Hessian.

**Inputs.** A PES oracle (energy and gradient/forces) — DFT, an MLIP, or an analytic toy
PES. Does not strictly require a full endpoint-pair NEB.

**Useful as pre-flight?** Yes, but in a limited way.
- *Useful:* a cheap MLIP probe ("is there any nearby saddle/pass in this pocket at all?");
  an alternative to dimer/MMF for local saddle search after an endpoint screen; a good
  hard-toy baseline for our role-aware logic.
- *Does not solve:* same-basin endpoint collapse; magnetic sheets; the question "are A and B
  truly different basins?"; the multi-endpoint network.

**Verdict.** SPM does not replace our framework, but it is worth keeping as a local
saddle-probe after L1/L2 — especially on MACE/CHGNet when an endpoint pair looks suspicious.

---

## 2. OM / adaptive springs — Onsager–Machlup adaptive-spring NEB

**Source:** Mandelli & Parrinello, 2021, *"A modified nudged elastic band algorithm with
adaptive spring lengths"* — https://arxiv.org/abs/2106.06275

**What it is.** Standard NEB has a harmonic spring term, so images end up roughly evenly
distributed along the path. OM/adaptive springs replaces that term with a discretized
Onsager–Machlup action, so images automatically **concentrate near the saddle / important
region** of the path.

```
Ordinary NEB:        images uniform along path length, but the saddle may fall between images
OM/adaptive springs: more resolution where the path matters most
```

**Inputs.** Endpoints A/B; energy/forces; an ordinary NEB-like workflow.

**Useful as pre-flight?** Yes, as a check of *path resolution* — but not as an
endpoint-validator.
- *Useful:* if a cheap MLIP NEB gives a different barrier under different image
  distributions, that is a warning; if a saddle appears only with adaptive springs, the
  ordinary NEB was poorly parameterized; the idea is close to our adaptive-reparam in NEB-AGM.
- *Does not solve:* if A and B are in the same basin, OM still elegantly optimizes a
  meaningless path; if the MLIP drops into a 20 eV well, OM does not rescue it; if there is a
  spin-sheet split, OM does not see magnetism.

**Verdict.** For our purposes OM/adaptive springs is not a primary method but a sanity check:
"does the conclusion depend on the image distribution?" If it does, DFT must not be launched.

---

## 3. dyNEB — dynamic NEB

**Source:** Lindgren, Kastlunger, Peterson, 2019, *"Scaled and Dynamic Optimizations of
Nudged Elastic Bands"* — https://arxiv.org/abs/1906.10257 ·
ASE docs: https://ase-lib.org/ase/mep.html

**What it is.** dyNEB saves force calls by recognizing that not all images are equally
important at every iteration: an image that has nearly converged need not be optimized as
aggressively, while an image near the saddle deserves more attention. ASE ships `DyNEB`,
which makes it closer to a genuinely usable tool than SPM.

**Inputs.** Ordinary NEB images; an ASE-compatible calculator; usable in principle with
MACE/CHGNet/QE through the ASE layer.

**Useful as pre-flight?** More as *cost reduction* once the input has already passed the gate.
- *Useful:* faster MLIP pre-flight NEB; cheaper DFT NEB once the endpoints/path are trusted;
  a good baseline for "not all images are equal," close to the role-aware idea.
- *Does not solve:* does not prove the endpoints are different; does not catch a topological
  same-basin artifact on its own; does not cure a wrong reaction coordinate.

**Verdict.** dyNEB is an operational optimization, not an epistemic gate. Use it — but only
after our L0–L3 checks.

---

## 4. OCINEB / Hessian eigenmode alignment

**Source:** Goswami et al., 2026, *"Enhanced Climbing Image Nudged Elastic Band method with
Hessian Eigenmode Alignment"* — https://arxiv.org/abs/2601.12630

**What it is.** A hybrid:

```
CI-NEB finds the approximate saddle region
        ↓
minimum-mode following / Hessian eigenmode alignment
        ↓
the saddle point is refined
```

So it is less a "pre-flight" than a *post-NEB saddle refinement*. The idea: the climbing
image points well at the relevant transition region but can converge slowly or get stuck —
so MMF/dimer-like refinement is switched on, using the minimum-curvature direction.

**Inputs.** An existing NEB/CI-NEB path; the ability to estimate a min-mode / Hessian-like
direction (expensive with DFT, cheaper with an MLIP).

**Useful as pre-flight?** Indirectly.
- *Useful:* after a cheap MLIP path, check "does the climber really lead to a normal
  saddle?"; if OCINEB/MMF walks to a *different* saddle, the original path is unstable; can
  serve as a validator for the top-1/top-2 transition candidates.
- *Does not solve:* endpoint same-basin collapse before the NEB; magnetic discontinuities;
  multi-site endpoint ambiguity.

**Verdict.** OCINEB is strong prior art against any claim "we were first to do a role-aware
climber." For our goal it simply says: climber refinement is already known, so we should
focus not on the climber but on **pre-DFT diagnosis of the input topology.**

---

## 5. GP-NEB — Gaussian-process-accelerated NEB

**Sources:**
- Koistinen et al., 2017, *"Nudged elastic band calculations accelerated with Gaussian
  process regression"* — https://arxiv.org/abs/1706.04606
- Garrido Torres et al., 2019, *"Low-Scaling Algorithm for NEB Using a Surrogate ML Model"*
  — https://pubmed.ncbi.nlm.nih.gov/31050513/
- eOn documents a native GP-accelerated NEB: https://eondocs.org/user_guide/neb.html

**What it is.** Instead of computing DFT forces on every image at every iteration, a
surrogate PES is built with a Gaussian process.

```
a few expensive DFT E/F
        ↓
GP surrogate PES
        ↓
many cheap NEB steps on the GP
        ↓
new DFT calls only where uncertainty is high
```

**Inputs.** Coordinate descriptors; E/F samples; an uncertainty model; an active-learning loop.

**Useful as pre-flight?** Yes, but carefully.
- *Useful:* it can estimate the uncertainty of a path; it enables a cheap "is this path
  learnable / smooth?" screen; our local per-image surrogate is a small local instance of
  this idea.
- *Dangerous:* Fe–S cubanes, spin states, Fe–H hydrides, U/magnetism are all
  out-of-distribution risks; a GP can elegantly interpolate a physically wrong surface; for
  our minerals "low surrogate energy" does not equal "DFT-valid path."

**Verdict.** GP-NEB is good as an uncertainty-aware pre-flight layer, but not as an oracle.
For us the right recipe is MACE + CHGNet + structural/PH/magnetic gates + DFT single-points,
with the GP as an *additional local memory / uncertainty estimate* — not as the source of truth.

---

# How this connects to our goal

> Check the input before DFT, so we don't run empty calculations; decide as much as possible
> mathematically/topologically rather than by brute force.

Then these methods should be used not as "competitors" but as **categories of signals.**

## What we actually need

Before a DFT NEB we must answer:

1. **Endpoint integrity** — are A and B different basins, or a same-basin artifact?
2. **Path topology** — is there a meaningful mountain pass between them, not a
   multi-site/multi-saddle mess?
3. **Path stability** — does the cheap path avoid sliding into an unphysical well?
4. **Magnetic-sheet consistency** — are the endpoints/images on one spin sheet?
5. **Calculator agreement** — do MACE and CHGNet at least agree qualitatively?
6. **Parameter robustness** — does the verdict survive changes in `k_spring`, image count,
   reparameterization, optimizer?

That is where the prior art helps.

## How it fits into our framework

```
L0  pristine / symmetry / chemistry gates
L3  topological/geometric path diagnostics:
    - persistent homology
    - internal-coordinate same-basin score
    - dense-spline barrier
    - plain string
    - dyNEB / standard NEB
    - adaptive reparam / OM-like spacing
    - optional SPM local saddle probe
L5  only then DFT NEB
```

**Key principle:** if different cheap mathematical viewpoints give different answers, that is
**not** "we should run DFT and see." It is **INVESTIGATE / NO-GO.**

---

# Practical takeaway

The strongest path right now:

1. Do not try to "beat the NEB literature."
2. Build a pre-flight, theorem-like certificate:
   - endpoints distinct;
   - path not same-basin;
   - no spin-sheet discontinuity;
   - MLIP cross-check not OOD;
   - barrier/path qualitatively stable under method perturbations.
3. Use SPM / dyNEB / OM / OCINEB / GP-NEB as **diagnostic perturbations**, not as a
   replacement for QE / ABACUS / jDFTx.

The `same_basin_gate` added to NEB-AGM follows exactly this philosophy: better to stop with
the diagnosis "the input is meaningless" than to pay for a beautiful zero DFT barrier.

---

## Focus on magnetism

A particular sore point — and the main reason to push this problem algorithmically — is
magnetism. MACE/CHGNet handle it poorly; the others have various +U and umbrella schemes but
have not yet proven themselves on our Fe–S minerals either. So the focus for pre-flight
hypotheses is magnetism.

Pre-flight should be centered precisely on magnetic ambiguity: what can be checked before a
DFT NEB, what cannot be trusted in an MLIP, and where a minimal DFT calculation is needed so
that we don't build a path between different spin sheets. (Details in
[`MAGNETIC_FIRST_PREFLIGHT_PLAN.md`](MAGNETIC_FIRST_PREFLIGHT_PLAN.md).)
