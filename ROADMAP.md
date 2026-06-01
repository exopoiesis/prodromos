# Prodromos — Roadmap

Each item below is distilled from a running improvement log kept while applying the
framework to real Fe–S DFT NEB campaigns (observed gap → proposed fix → status).
Severity: **P0** = silent wrong verdict · **P1** = missing a gate a campaign needed ·
**P2** = ergonomics / robustness.

## Next: MCP server

All gates are now implemented and uniformly expose the shared `cli_contract` JSON
envelope (`verdict / confidence / reasons / next_actions / artifacts / warnings /
result`) through a `run_*` entry point plus a `prodromos <subcommand>` CLI. The
remaining highest-leverage step is the **MCP server** that surfaces each gate as an
independent tool so agents can call them and consume structured verdicts.

Every gate ships a pure `run_*(...) -> dict` function with no argparse/stdout side
effects, so the MCP layer is a thin adapter over those functions.

## Shipped

All previously open items are implemented, each with `cli_contract` envelope output,
a `run_*` entry point, a `prodromos` subcommand, and unit tests (suite: 349 tests).

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
