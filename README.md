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
  (the criterion that ends the nspin=1↔2 flip-flop);
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

Subcommands: `electron-parity`, `spin-collapse`, `saddle-proximity`,
`symmetry-preflight`, `vfe-preflight`, `magnetic-parser`, `magnetic-endpoint`,
`magnetic-band`, `magnetic-recommend`, `multi-endpoint`, `soap-cluster`,
`adaptive-neb`, `neb-advisor`, `gp-neb`, `master-equation`.

Every production-facing gate emits a stable JSON envelope (`tool`, `verdict`,
`confidence`, `reasons`, `next_actions`, `artifacts`, `warnings`, `result`) via
`prodromos.cli_contract` — CLI- and MCP-ready. Add `--json` for machine output,
`--output result.json` to also write it to disk.

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q   # 206 passed
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
