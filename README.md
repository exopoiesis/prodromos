# Prodromos

> πρόδρομος — *"the one who runs ahead."*

**Prodromos** is a **$0 pre-flight / post-flight diagnostic gate** for expensive
DFT NEB and saddle-point searches on Fe–S minerals. It runs *ahead* of the costly
A100 calculation ($200–800 per NEB) and reports whether the path is worth taking —
catching the failure modes that otherwise burn GPU-days on a meaningless barrier.

It is **not** a replacement for Quantum ESPRESSO / ABACUS / jDFTx. It is a thin,
cheap diagnostic layer *around* them.

## What it catches before you spend DFT

- **same-basin endpoints** — endpoints A and B relax into the *same* basin, so the
  "barrier" is fictitious (the historical incidents mack / pent / marc that motivated
  the whole project);
- **spin-sheet discontinuities** — endpoints sit on different magnetic sheets, making a
  single-sheet NEB ill-posed;
- **MLIP out-of-distribution artifacts** — foundation MLIP (MACE/CHGNet) geometries that
  look locally physical but carry tens of eV of unrelaxed-lattice error;
- **multi-endpoint H landscapes** — several competing H sites that a symmetric NEB misses;
- **optimizer / spring failures** — roll-off, frozen-energy degenerate paths;
- **odd-electron / nspin pitfalls** — parity flips and moment-collapse-vs-persist
  (the criterion that ends the nspin=1↔2 flip-flop), with a formal-oxidation d-count
  that rejects closed-shell d⁰/d¹⁰ hosts a-priori;
- **redox-polaron sublattice crossings (pre-DFT)** — a *nonmagnetic* migrant (Li⁺/Na⁺)
  whose charge-compensating polaron lands on a different magnetic sublattice at the two
  endpoints → an ill-posed single-sheet NEB, predicted from structure + moment signs at $0;
- **periodic-boundary endpoint mis-alignment** — endpoints not in the same minimum-image
  cell, so a naive NEB interpolation routes atoms the long way across the cell (caught and
  auto-aligned before the run);
- **untrustworthy foundation-MLIP barriers** — near-degenerate itinerant 3d / multivalent
  redox hosts where a spin-blind MLIP barrier should be routed to DFT;
- **computed-vs-experimental magnetic mislabels** — MP labels many Fe sulfides/phosphates
  FM where neutron experiment is AFM; cross-checked against MAGNDATA before seeding an NEB;
- **kinetic-network consequences** — once barriers exist, master-equation kinetics.

## The Evidence Framework (cheapest test first, stop at first hard diagnosis)

| Layer | What | Cost |
|-------|------|------|
| L0 | pristine-crystal analysis (spglib, cubane/dimer/anisotropy) | $0 |
| **L1** | Hungarian symmetry test on the relaxed endpoint (strongest predictor) | $0 |
| L2 | MACE multi-endpoint screen + SOAP clustering | $0 (gomer) |
| **L3** | CHGNet cross-check | $0 (gomer) |
| L4 | DFT single-point verification (AFM+U) | $10–30 |
| L5 | NEB / string method | $200+ |
| L6 | Master-equation kinetics | $0 |

Plus a **magnetic-first** track of gates (parity → spin-collapse → endpoint/band sheet
gates → constrained-M / MECP routing) for nspin=2 minerals.

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows
# source .venv/bin/activate && pip install -e ".[dev]" # POSIX
```

## CLI

```bash
prodromos --help                  # list all subcommands
prodromos electron-parity --help  # any subcommand forwards to its own argparse
```

Subcommands: `plan` (orchestrator), `from-inputs` (QE/ABACUS → tm-spec),
`electron-parity`, `spin-collapse`, `saddle-proximity`, `endpoint-provenance`,
`symmetry-preflight`, `vfe-preflight`, `magnetic-parser`, `magnetic-endpoint`,
`magnetic-band`, `magnetic-recommend`, `magnetic-provenance`, `mlip-confidence`,
`mic-alignment`, `sublattice-preflight`, `multi-endpoint`, `soap-cluster`,
`adaptive-neb`, `neb-advisor`, `gp-neb`, `master-equation`, `external-reference`,
`lint-dft-script`, `h-barrier-readiness`.

Every production-facing gate emits a stable JSON envelope (`tool`, `verdict`,
`confidence`, `reasons`, `next_actions`, `artifacts`, `warnings`, `result`) via
`prodromos.cli_contract` — CLI- and MCP-ready. Add `--json` for machine output,
`--output result.json` to also write it to disk.

## MCP server

A thin in-process stdio MCP server exposes the same gates to an LLM agent
(Claude Desktop / Claude Code). It runs in-process — no proxy, no network, no
Docker — and every tool returns the same JSON envelope as the CLI. The server is
**concurrency-safe**: every tool runs off the asyncio event loop (via
`anyio.to_thread.run_sync`), so a sync gate never blocks stdio intake/flush or
serializes parallel calls; it also has a PID-singleton guard, clean EOF shutdown,
and an opt-in per-tool timeout (`PRODROMOS_MCP_TOOL_TIMEOUT_S`).

```bash
.venv/Scripts/python.exe -m pip install -e ".[mcp]"   # adds mcp>=1.0
prodromos-mcp                                          # stdio loop (blocking)
```

Register it with an MCP client (e.g. Claude Desktop `claude_desktop_config.json`
or Claude Code):

```json
{
  "mcpServers": {
    "prodromos": { "command": "prodromos-mcp" }
  }
}
```

Main tools: **`plan`** (pre-flight orchestrator over a tm-spec/0.3 case or a raw
QE/ABACUS input, auto-converted), **`from_inputs`** (onboard a QE/ABACUS input
into a tm-spec/0.3 doc), plus one tool per gate: `electron_parity`,
`spin_collapse`, `endpoint_provenance`, `symmetry_preflight`, `vfe_preflight`,
`external_reference`, `lint_dft_script`, `h_barrier_readiness`, `neb_advisor`,
`saddle_proximity`, `multi_endpoint`, `mlip_confidence`, `mic_alignment`,
`sublattice_preflight`, `soap_cluster`, `master_equation`, `gp_neb`, `adaptive_neb`,
`magnetic_parser`, `magnetic_endpoint`, `magnetic_verdict`, `magnetic_band`,
`magnetic_recommend`, `magnetic_provenance`. Corpus importers: `import_optimade`,
`import_nomad`, `import_mp`, `import_magndata` (by code, or search by
`elements`/`formula`), `merge_specs`. Plus two meta-tools that run many gates in one
round-trip (no client fan-out): `batch` and `preflight_bundle` (31 gate tools + 2 meta-tools).

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q   # 529 passed (self-contained; bundled DFT fixtures)
```

Toy-PES validation (Müller-Brown, LJ7, double-well) plus structural/magnetic gate
unit tests. Prototypes (NEB-AGM, string method, MECP, GP-NEB, master equation) are
validated on analytic potentials.

## Quick start

See [`QUICK_START.md`](QUICK_START.md) for the end-to-end pipeline (inputs → verdict)
and how to interpret a harvested NEB result.

## Documentation

The methodology, theory, and playbooks behind the gates are in [`docs/`](docs/):
the Evidence Framework (L0→L6), the 7 convergence conditions (C1–C7), the
game-theoretic foundations, the magnetic-first logic, the NEB-stall playbook, and the
positioning against alternative MEP formulations. Start at [`docs/README.md`](docs/README.md).

## Project

Part of [Third Matter](https://exopoiesis.space). Author: Igor Morozov
(igor@exopoiesis.space). Internal research tool — see `LICENSE`.
