# Prodromos — Roadmap

Each item below is distilled from a running improvement log kept while applying the
framework to real Fe–S DFT NEB campaigns (observed gap → proposed fix → status).
Severity: **P0** = silent wrong verdict · **P1** = missing a gate a campaign needed ·
**P2** = ergonomics / robustness.

## Now shipped: orchestrator, converters, MCP

The framework is no longer a set of standalone gates — it is an orchestrated tool:

- **`plan` orchestrator** — route mode (myopic value-of-information, EXECUTEs $0 gates)
  and tree mode (Bellman expectimax + CVaR + Beta-Binomial calibration over a PolicyGraph);
  case = a tm-spec/0.3 document; emits a tm-spec `preflight` block. Hardened by a
  game-theory + CS + information-theory consilium.
- **Converters** — `from-inputs` (QE/ABACUS input → tm-spec/0.3) and tm-spec's
  `import-nomad` (→ 0.3); `prodromos plan my.in` auto-converts a raw input. Zero manual
  spec authoring to onboard.
- **MCP server** — thin in-process stdio (`prodromos-mcp`); `plan` + `from_inputs` + one
  tool per gate; no proxy/network/Docker.

## Next
- **`plan` tree calibration from real campaign outcomes** (`update_from_outcomes`): the
  Beta-Binomial table currently seeds from public methodological hit-rates only.
- **HTTP/SSE transport + Docker** — only if a shared remote host is needed.
- **Paper / Zenodo packaging** — CITATION.cff, Zenodo DOI, examples, CI, badges
  (see the MagNEB_Preflight manuscript readiness gap-list).

## Shipped gates

All gate items are implemented, each with `cli_contract` envelope output, a `run_*`
entry point, a `prodromos` subcommand, and unit tests (suite: 448 tests).

| Item | Gate / change | Subcommand |
|------|---------------|------------|
| N-01 | free-magnetization screen → per-sheet grouping; refuses cross-sheet energy ranking | (in `magnetic-recommend` / dataset scan) |
| N-02 | endpoint-provenance gate: SP on an MLIP geometry is not a valid endpoint energy (local bond geometry necessary but not sufficient) | `endpoint-provenance` |
| N-03 | magnetization drift / `magnetization_settled` signal in the parser | `magnetic-parser` |
| N-04 | harvest dedup by job label; prefer the converged copy; warn on disagreement | (in dataset scan) |
| N-05 | global-displacement WARNING when non-H displacement is implausibly global | `symmetry-preflight` |
| N-06 | `--log-assignment` Hungarian-correspondence dump + optional self-consistency check | `symmetry-preflight` |
| N-07 | external-reference gate (NOMAD + OPTIMADE fallback); `NO_EXTERNAL_REFERENCE` raises the bar | `external-reference` |
| N-08 | M0 emits the concrete nspin=2 collapse-test spec under odd+smearing | `electron-parity` |
| N-09 | `lint_dft_script` static pre-flight for the 4 recurring QE/ABACUS deploy bugs | `lint-dft-script` |
| N-11 | `spin_collapse_verdict` — the criterion that ends the nspin=1↔2 flip-flop | `spin-collapse` |
| N-12 | `dimer_chemical_rc` branch in the method advisor (roll-off/frozen/same-basin/multi-site → dimer + chemical-RC seed) | `neb-advisor` |
| N-13 | `saddle_proximity_gate` extracted as a reusable tool | `saddle-proximity` |
| N-14 | `(U, nspin, functional, ecut, kpts)` provenance guard; within-method-delta refuses cross-provenance comparison | `magnetic-parser` / `magnetic-recommend` |
| N-15 | H-transfer paper-readiness gate (index-1 imaginary mode + ΔZPE‡) | `h-barrier-readiness` |

## Backlog (carried)

- **ABACUS magnetic parser**: needs an nspin=2 corpus sample to harden final-moment and
  drift extraction (currently QE-only for provenance/drift).
- **External execution wrappers** (MLIP relax, DFT manifest/launch, NEB rerun, harvest)
  remain deliberately out of scope — they are execution infrastructure, not diagnostics.
