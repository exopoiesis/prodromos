# QUICK_START — Prodromos pre-flight pipeline

**Goal:** decide whether an expensive DFT NEB is worth running, what to fix before
submission, and how to read an already-harvested NEB result. Each step is cheaper
than the next — **stop as soon as a hard diagnosis is found.**

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows
.venv/Scripts/python.exe -m pytest -q                  # expect: 206 passed
```

All gates run either as `prodromos <subcommand> ...` (installed console script) or
`python -m prodromos.<module> ...`. Add `--json` for machine output and
`--output result.json` to persist the envelope.

## Pipeline (cheapest → most expensive)

### Before you start: is this charted territory? ($0)

```bash
prodromos external-reference --elements Fe S --reduced-formula FeS2   # NOMAD + OPTIMADE
```
`REFERENCE_FOUND` (attach functional/lattice/magnetic coverage) vs `NO_EXTERNAL_REFERENCE`
(uncharted → raise the internal-validation bar). `UNKNOWN` if offline.

### 0. Odd-electron / nspin parity (always, $0)

```bash
prodromos electron-parity <structure> ...
```
`NSPIN2_MANDATORY` (odd electrons, fixed-occupation) vs `NSPIN1_OK`. For a metallic
defect with smearing, follow up with one cheap nspin=2 SP and the collapse verdict:

```bash
prodromos spin-collapse ...    # NSPIN1_OK (moment collapses) vs NSPIN2_REQUIRED
```

### 1. Structural symmetry pre-flight (L0/L1, $0)

```bash
prodromos symmetry-preflight <relaxed_endpoint> ...   # SYMMETRIC / MARGINAL / ASYMMETRIC
prodromos vfe-preflight ...                            # V_Fe pre-flight checklist gate
prodromos endpoint-provenance --provenance mlip_relaxed ...   # is this a valid endpoint?
```
L1 (Hungarian test on the relaxed endpoint) is the strongest single predictor of
same-basin trouble — but trust it only on a DFT-relaxed endpoint. `endpoint-provenance`
flags energies taken on an MLIP geometry (`NOT_AN_ENDPOINT_MLIP_GEOMETRY`): local bond
lengths can look physical while the single-point energy is tens of eV off.

### 2–3. MLIP multi-endpoint screen (L2/L3, $0 on gomer)

```bash
prodromos multi-endpoint ...     # enumerate candidate H endpoint sites
prodromos soap-cluster ...       # SOAP-cluster relaxed candidates → distinct minima
```
Run MACE *and* CHGNet; agreement on asymmetry ⇒ geometric, not magnetic.

### Magnetic gates (nspin=2 minerals — marc / pent / greig)

```bash
prodromos magnetic-parser <dir>                 # parse QE/ABACUS/jDFTx → table (M_tot, M_abs, drift)
prodromos magnetic-endpoint <endA.pwo> <endB.pwo>  # GO / REVIEW / NO-GO_SINGLE_SHEET
prodromos magnetic-band <neb_done_dir>          # sheet-crossing / sawtooth detection across a band
prodromos magnetic-recommend ...                # constrained-M re-run vs MECP / two-segment routing
```
**Rule:** spin-blind MLIP is for *geometry only*, never for a barrier at a spin seam.

### 5. NEB method selection + acceleration ($200+ run)

```bash
prodromos lint-dft-script --script run_neb.py --pseudo-dir <dir> --xyz-path endA.xyz   # catch 4 deploy bugs ($0)
prodromos neb-advisor ...     # failure-signature → method (band tune / dimer + chemical-RC seed / string)
prodromos adaptive-neb ...    # spring schedule, dyNEB active images, climber choice
prodromos gp-neb ...          # GP-NEB next-sample suggestion (single-sheet GO only)
```
`lint-dft-script` statically checks a QE/ABACUS script + inputs for the recurring deploy
bugs (relative `pseudo_dir`, nested `outdir`, unclean extxyz read, `number_of_wfc=0`
pseudos) before any A100 time is spent. The advisor's `DIMER_CHEMICAL_RC` branch fires on
roll-off / frozen-energy / same-basin / multi-site signatures.

### Post-flight: saddle QA + kinetics

```bash
prodromos saddle-proximity <saddle_xyz> ...   # DIRECT_TRANSFER_OK vs OFF_PATH_OR_INTERMEDIATE
prodromos h-barrier-readiness --barrier-ev 0.27 --has-dft-freq --n-imag-modes 1 --imag-mode-h-fraction 0.9 --dzpe-ev -0.12
prodromos master-equation ...                 # Freidlin-Wentzell network kinetics from a barrier matrix
```
`h-barrier-readiness` gates a quoted barrier as `PAPER_GRADE` only with a confirmed
single H-dominated imaginary mode and a reported ΔZPE‡; otherwise `ELECTRONIC_ONLY`.

## JSON / MCP contract

Every gate returns the same envelope shape (`prodromos.cli_contract.response_envelope`):
`tool, version, status, verdict, confidence, reasons, next_actions, artifacts,
warnings, result`. This is what a future MCP server (see `ROADMAP.md`) exposes per gate.

## Deep methodology

The full Evidence Framework v2 derivation, the 7 sufficient convergence conditions
(C1–C7), the game-theoretic foundations, and the magnetic-first methodology are
documented in [`docs/`](docs/). This document is the operational quick-start for the
shipping tool.
