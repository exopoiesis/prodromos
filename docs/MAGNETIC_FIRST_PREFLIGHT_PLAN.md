# Magnetic-First Pre-Flight Plan for Fe-S NEB

**Scope:** Fe-S V_Fe / H migration workflows before expensive DFT NEB  
**Decision:** magnetic hypotheses are checked before optimizer/parameter hypotheses.

## Core Reframing

For Fe-S minerals the object is not a single smooth PES:

```text
V(R)
```

but a family of magnetic/electronic sheets:

```text
V_M(R),  M in {magnetic states / AFM patterns / local Fe moments}
```

A standard NEB is mathematically well-posed only after we know which case we are in:

1. **Single-sheet case:** both endpoints and all images can be kept on the same magnetic sheet.
2. **Metastable endpoint artifact:** one endpoint converged to the wrong magnetic state; re-relax endpoints on the common ground-state sheet.
3. **Real spin crossover:** endpoints genuinely live on different sheets; the barrier is a seam/MECP problem, not an ordinary saddle on one smooth PES.

Spin-blind MLIPs (MACE/CHGNet) cannot decide this. They may still be useful for geometry, endpoint enumeration, and obvious path-collapse detection, but not for barrier values near a spin seam.

## Magnetic-First Hypotheses

### H-M0 — Is This Mineral Spin-Sensitive?

**Claim:** nspin=2 Fe-S systems with itinerant or mixed-valence Fe are high-risk unless proven otherwise.

**Signals:**

- `nspin=2` required in production.
- Fe vacancies, cubane Fe4S4 motifs, S-S dimers, mixed Fe coordination, low-symmetry pockets.
- Prior evidence of endpoint total/absolute magnetization shifts.

**Verdict:**

- If low-risk closed-shell/diamagnetic and stable moments: normal geometry-first pipeline may proceed.
- If high-risk: enable all magnetic gates before any DFT NEB.

### H-M1 — Endpoint Magnetic Sheet Consistency

**Claim:** endpoints are not NEB-compatible if relaxed endpoint A and endpoint B settle on different magnetic sheets at nearly the same geometry.

**Cheap test:**

Parse endpoint `.pwo`:

```text
total magnetization
absolute magnetization
local Fe moments if available
```

Gate:

```text
|Δtotal_mag| > 0.3 μB  OR  |Δabs_mag| > 0.5 μB
```

with small geometry/RMSD difference -> **MAGNETIC NO-GO**.

**Action:**

Do not run DFT NEB. Run H-M2.

### H-M2 — Metastable Endpoint vs Real Spin Crossover

**Claim:** a magnetic mismatch is either a convergence artifact or a real crossing. These require different workflows.

**Decisive low-cost DFT test:**

Run both endpoints at both candidate total magnetizations:

```text
endA @ M_A
endA @ M_B
endB @ M_A
endB @ M_B
```

No ionic relaxation first; SCF/singlepoint or short constrained relax if needed.

**Interpretation:**

- Both endpoints prefer the same M -> previous mismatch was a metastable endpoint artifact.
- Each endpoint prefers a different M by more than thermal/noise scale -> real spin crossover.
- Differences within ~kT to 3kT -> ambiguous; report magnetic error bar and avoid single-number barrier.

**Action:**

- Same preferred M -> re-relax endpoints with constrained/common magnetic state, then single-sheet NEB.
- Different preferred M -> two-sheet / MECP workflow.

### H-M3 — Band Magnetic Continuity

**Claim:** even if endpoints pass, intermediate images can jump sheets during SCF.

**Test on any cheap DFT path probe:**

For each image:

```text
M_total(i), M_abs(i), local Fe moments, E(i), geom force/residual
```

Band gate:

```text
adjacent |Δabs_mag(i,i+1)| > 0.5 μB -> SHEET_CROSSING
low geom force + high spin mismatch -> spin_split, not geometric stuck
high geom force + smooth spin -> geometric stuck
```

**Action:**

- `spin_split`: halt/report; do not apply geometric escape.
- `stuck`: geometry/optimizer fix is allowed.
- `mixed`: resolve magnetism first, then geometry.

### H-M4 — MLIP Geometry/Spin Separation

**Claim:** MACE/CHGNet can test geometry hypotheses but cannot validate magnetic barriers.

**Allowed uses:**

- endpoint enumeration;
- H placement and pocket discovery;
- geometry-only band collapse warning;
- MACE-vs-CHGNet disagreement as OOD signal;
- approximate path for DFT image initialization.

**Forbidden uses:**

- reporting spin-seam barrier;
- deciding whether two magnetic sheets are physically connected;
- declaring a DFT magnetic asymmetry false because spin-blind MLIP disagrees.

**Positive signal:**

If spin-blind MLIPs and DFT agree on endpoint asymmetry direction, asymmetry is likely geometric, not purely magnetic.

### H-M5 — Single-Sheet Sufficiency

**Claim:** if a common magnetic sheet is selected, constrained-M NEB is the mathematically controlled approximation.

**Pre-flight requirements:**

- endpoints stable on common `tot_magnetization`;
- image initialization uses spin-IDPP / neighbor propagation of starting moments;
- no large adjacent moment jumps in a pilot band;
- U/k-point/smearing settings held fixed across endpoints and images.

**Action:**

Proceed to DFT NEB only after these pass.

### H-M6 — Real Spin Crossover / MECP

**Claim:** if endpoints are genuinely on different magnetic sheets, the relevant object is a crossing seam / MECP, not a smooth NEB saddle.

**Workflow:**

1. Find or bracket crossing edge.
2. Run two single-sheet path segments up to the seam.
3. Use MECP finder / Bearpark-Robb style projected gradient for seam point.
4. Report:
   - geometric barrier on each sheet;
   - spin-crossing contribution;
   - min-envelope barrier;
   - optional Boltzmann free-energy envelope if sheets are thermally competitive.

**Action:**

Do not publish one ordinary NEB barrier as if the path lived on one PES.

## Revised Pre-Flight Order

```text
M0 magnetic risk triage
M1 endpoint magnetic gate
M2 4-SCF endpoint cross-check if M1 fails or marginal
L0/L1 structural/symmetry gates
L2 MLIP geometry enumeration only
M3 magnetic continuity on pilot DFT images
L3 PH/path topology checks
M5 single-sheet constrained-M readiness OR M6 MECP branch
DFT NEB only after magnetic branch is resolved
```

## Why +U / Umbrella / Better Sampling Is Not Enough

`+U` changes localization and relative sheet energies, but it does not guarantee that every image converges to the same sheet.

Umbrella/OPES-style sampling improves exploration along a chosen CV, but it does not by itself define a single magnetic sheet or prevent SCF from changing electronic state along the path.

Therefore the pre-flight certificate must check **sheet identity**, not just sampling coverage or optimizer convergence.

## Implementation Status

Already present:

- the `spin_split_detector` module: two-sheet toy + `magnetic_band_diagnostic(...)`.
- the `mecp_finder` module: toy MECP finder.
- `NEBAGM(..., magmom_provider=...)`: magnetic gate wiring.
- tests for `spin_split`, `mixed`, `stuck`, endpoint split, and clean single-sheet negative controls.
- the `magnetic_output_parser` module: normalized QE/ABACUS/jDFTx parser for energy, SCF status, total/absolute magnetization, and QE local moments.
- the `magnetic_endpoint_gate` module: endpoint `GO / REVIEW / NO-GO` gate using the current marcasite-calibrated thresholds.
- the `magnetic_band_gate` module: `image_XX/espresso.pwo` band scanner for adjacent magnetic sheet jumps.
- the `magnetic_dataset_scan` module: dataset-level triage over harvested dataset band roots.

Current corpus findings:

- A marcasite V_Fe tier-1 band (`marc_VFe_tier1`): **calculation completed, but `NO-GO_SINGLE_SHEET` for ordinary NEB interpretation**. Adjacent sheet crossing at `image_04 -> image_05`, `Δabs = 0.55 μB`, endpoint split present, and energy sawtooth co-located with the magnetic jump. This does not mean "QE failed"; it means the computed band crosses magnetic sheets and should not be treated as one smooth single-sheet MEP/barrier without the H-M2/H-M6 follow-up.
- A greigite full NEB band: **GO** for magnetic continuity. No adjacent or endpoint magnetic discontinuity detected.
- Pyrite and older spinless QE bands: **REVIEW**, not magnetic `GO`, because the outputs do not contain final magnetic summaries. If `nspin=1` was intentional, the magnetic gate is not applicable; if this was a Fe-S magnetic run, the input needs correction before DFT NEB.

Next engineering targets:

1. Endpoint 4-SCF manifest generator for H-M2.
2. Magnetic pre-flight report combining H-M1/H-M2/H-M3 with structural gates.
3. Optional spin-IDPP initializer: propagate converged local Fe moments image-to-image.
4. ABACUS magnetic-output expansion once an `nspin=2` ABACUS corpus sample is available.
