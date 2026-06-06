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
- **Corpus importers + merge** — tm-spec's `import-optimade` (structure *width*),
  `import-nomad` (method *depth*), `import-mp` (computed magnetic *depth*: ordering +
  per-site magmoms), folded locally with `merge_specs`. Exposed as MCP tools so a whole
  validation corpus can be pulled and planned through the server.
- **Two policy graphs, selected by case kind** — the NEB V_Fe+H pre-flight graph
  (`endpoint_provenance → … → GO`) AND a **structure-mode triage graph** for bare
  SinglePoint/Relax corpus cases (`electron_parity → spin_collapse → nspin recommendation`),
  so a structure-only import resolves to a magnetic-readiness verdict instead of
  `NEEDS_DATA(workflow.endpoints)`. `select_policy_graph(doc)` picks by kind/endpoints.
- **MCP server** — thin in-process stdio (`prodromos-mcp`); `plan` + `from_inputs` + one
  tool per gate + the three importers + `merge_specs`; no proxy/network/Docker. **Concurrency-safe
  (2026-06-06):** every tool runs OFF the event loop via `anyio.to_thread.run_sync` (sync tools no
  longer block stdio intake/flush), with a PID-singleton guard + clean EOF shutdown, two meta-tools
  (`batch`, `preflight_bundle`) to avoid client fan-out, and an opt-in per-tool timeout. Verified by
  a deterministic in-memory transport harness.

## Next
- **`plan` tree calibration from real campaign outcomes** (`update_from_outcomes`): the
  Beta-Binomial table currently seeds from public methodological hit-rates only — the
  structure-mode graph now makes the NOMAD/OPTIMADE/MP corpus a calibration source.
- **HTTP/SSE transport + Docker** — only if a shared remote host is needed.
- **Paper / Zenodo packaging** — CITATION.cff, Zenodo DOI, examples, CI, badges
  (see the MagNEB_Preflight manuscript readiness gap-list).

## Shipped gates

All gate items are implemented, each with `cli_contract` envelope output, a `run_*`
entry point, a `prodromos` subcommand, and unit tests (full suite green).

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
| N-16 | `import-mp` magnetic-depth importer (MP computed ordering + per-site magmoms → tm-spec `magnetic` block); MCP `import_mp` | tm-spec `import-mp` |
| N-17 | structure-mode policy graph for bare SinglePoint/Relax corpus cases (parity → spin-collapse → nspin verdict); `select_policy_graph` by kind; `spin_collapse` adapter now reads `magnetic.magmoms_uB` | `plan` (structure mode) |
| N-18 | `external-reference` NOMAD 422 self-heal: strips a non-doc-quantity `include` field named in the 422 body and retries (kills the recurring "flap"-masquerade-of-422) | `external-reference` |
| N-19 | `import-magndata` experimental magnetic anchor (MAGNDATA magCIF → tm-spec `magnetic` block); FM/AFM/ferri derived rigorously from the file's magnetic symmetry ops (axial-vector projector), not a hardcoded table; MCP `import_magndata` | tm-spec `import-magndata` |
| N-20 | smarter M0: formal-oxidation d-count (`infer_closed_shell`) rejects closed-shell d⁰/d¹⁰ TM cells; vacancy-odd vs open-shell-TM-odd discriminator under metallic smearing | `electron-parity` |
| N-21 | `mlip_confidence` gate — flags hosts where a foundation-MLIP barrier is untrustworthy (near-degenerate itinerant 3d / multivalent redox cathode) → route to DFT | MCP `mlip_confidence` |
| N-22 | `sublattice_preflight` — structure-level (pre-DFT, $0) magnetic-sublattice-crossing predictor; migrant + redox-polaron modes; emits two-species / constrained-M recipe | MCP `sublattice_preflight` |
| N-23 | `magnetic_verdict` — per-TM *relative* endpoint ΔM threshold (no false NO-GO on slow drift) + auto-reconcile endpoint screen with the band-gate arbiter into one verdict | MCP `magnetic_verdict` / `magnetic_endpoint(n_magnetic=...)` |
| N-24 | `magnetic_provenance` — MP-computed vs MAGNDATA-experimental ordering cross-check → route seed to experiment, WARN on conflict | MCP `magnetic_provenance` |
| N-25 | band-directory discovery recognises `endA`/`endB`/`neb_imgNN` (not just `image_NN`) and orders endpoint→interior→endpoint (correct adjacency) | `magnetic-band` |
| meta | `batch` (many gates, one round-trip) + `preflight_bundle` (all $0 gates for a case) — collapse MCP client fan-out | MCP `batch` / `preflight_bundle` |

## Backlog (carried)

- **ABACUS magnetic parser**: needs an nspin=2 corpus sample to harden final-moment and
  drift extraction (currently QE-only for provenance/drift).
- **External execution wrappers** (MLIP relax, DFT manifest/launch, NEB rerun, harvest)
  remain deliberately out of scope — they are execution infrastructure, not diagnostics.
