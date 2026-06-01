# Prodromos — Roadmap

Open product items, carried from the running improvement log used while applying the
framework to real campaigns. Severity: **P0** = silent wrong verdict · **P1** = missing a
gate a campaign needed · **P2** = ergonomics / robustness.

The full annotated history of each item (observed gap → fix → status, with `.pwo`
evidence) lives in the `dft-neb/FRAMEWORK_IMPROVEMENT_NOTES.md` back-office log.

## Highest leverage

- **MCP server.** All gates are CLI/JSON-ready (shared `cli_contract` envelope) but the
  server itself is unimplemented. This is the single highest-leverage next step — it lets
  agents call each gate independently and consume structured verdicts.

## Open gates (P1)

- **N-02 — "single-point on MLIP geometry is not an endpoint" gate.** SP energy on a
  non-DFT-relaxed geometry can be ~20 eV off even when *local* bond lengths look physical.
  Add a `provenance` field (mlip_relaxed vs dft_relaxed) that downgrades any energy-based
  verdict; advisory in `neb_method_advisor`.
- **N-05 — L0(pristine) vs L1(endpoint) disagreement guard.** `symmetry_preflight_general`
  should WARN when non-H displacement is implausibly global (large fraction of atoms moved
  >1 Å) — that may signal a non-stationary MLIP geometry, not true symmetry breaking.
- **N-07 — external-reference gate (NOMAD / OPTIMADE).** Before spending DFT on a new
  mineral, query whether a public DFT reference exists; emit `NO_EXTERNAL_REFERENCE` to
  raise the internal-validation bar, or attach reference numbers. Prototyped as standalone
  scripts; not yet a framework CLI tool.
- **N-09 — `lint_dft_script` pre-flight.** Static check of a QE/ABACUS script + inputs for
  the 4 recurring deploy bugs (absolute `pseudo_dir`; non-nested `outdir`; clean-read xyz;
  `number_of_wfc>0` on every pseudo). All $0/local; would prevent A100 fast-fails.
- **N-12 — `dimer_chemical_rc` branch in `neb_method_advisor`.** Failure signature ∈
  {roll-off, frozen-energy, same-basin, multi-site pocket} → recommend a dimer (min-mode)
  with a chemical-RC seed (H along S_i→S_k, MIC), not another band tune.
- **N-14 — (U, nspin, functional) provenance guard.** Tag every parsed energy with
  `(U_eff, nspin, functional, ecut, kpts)`; any barrier/asymmetry computation asserts both
  legs share provenance, else refuses. (DFT+U totals are not on the U=0 scale.)
- **N-15 — H-transfer paper-readiness gate.** A quoted H-migration barrier is not
  paper-grade until DFT freq confirms exactly one imaginary mode (H-dominated along the
  transfer axis) and ΔZPE‡ is reported. `h_barrier_paper_readiness`:
  `ELECTRONIC_ONLY` vs `PAPER_GRADE`.

## Ergonomics / robustness (P2)

- **N-01 (finish) — auto-wire the magnetic gate into the L4 screen aggregator** so a
  free-magnetization screen refuses to emit a cross-sheet energy ranking (per-sheet groups
  instead).
- **N-04 — harvest dedup by job label** across worker dirs; prefer the converged copy,
  warn on duplicate-label disagreement (avoid picking up MPI-failed empty `.pwo`).
- **N-06 — Hungarian `--log-assignment`** to dump the correspondence (S-anchors → S-anchors)
  for human/agent confirmation on asymmetric relaxed endpoints.
- **N-08 — M0 emits the concrete collapse-test spec** when `NSPIN2_MANDATORY` + smearing
  hint, cross-linked to the `magnetization_settled` field.
- ABACUS magnetic parser needs an nspin=2 corpus sample to harden final-moment (and drift)
  extraction — currently QE-only for drift.

## Out of scope (kept as external manual wrappers, by design)

- MLIP relax on gomer, DFT manifest/launch, NEB rerun, harvest. These are execution
  infrastructure of the Third Matter project, not part of the diagnostic tool.

## Done (shipped this far)

- N-03 magnetization drift/settled signal · N-11 `spin_collapse_verdict` (the criterion
  that ends the nspin=1↔2 flip-flop) · N-13 `saddle_proximity_gate` extracted to a reusable
  tool. Suite at 206 tests.
