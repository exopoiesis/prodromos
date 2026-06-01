# Prodromos — Decisions

## D-001 (2026-06-01) — Name: Prodromos

The framework that grew in the Third Matter `dft-neb/` working folder was promoted to a
standalone tool and named **Prodromos** (πρόδρομος, "the one who runs ahead" / forerunner).
The name captures the function literally: a $0 pre-flight that runs ahead of the expensive
DFT NEB and reports whether the path is worth taking. Distribution name `tm-prodromos`,
import package `prodromos`, console script `prodromos`. Aesthetic consistent with sibling
Third Matter tools (`aletheia`, `exopoiesis`).

Alternatives considered: *Skopos* (σκοπός, scout/aim), *Diabasis* (διάβασις, the mountain
pass itself). Prodromos chosen for the most literal "pre-flight" reading.

## D-002 (2026-06-01) — Own repo under `git/`, packaged like `arxiv-radar-mcp`

Created as its own repository `git/prodromos/` (the Third Matter repo container), mirroring
the proven layout of `arxiv-radar-mcp`: `src/<pkg>/` package, `tests/`, `pyproject.toml`
(hatchling), `.gitignore`, `LICENSE`, console-script entry point. Remote target
`github.com/exopoiesis/prodromos`.

The previous `ph-diagnostic/` flat-script suite (sibling-import scripts run via
`python foo.py`) was migrated into a real package: 29 core modules → `src/prodromos/`,
26 test files → `tests/`, all intra-suite imports rewritten to absolute `prodromos.*`,
the `sys.path` hack dropped from `conftest.py`. A thin `__main__.py` dispatches 15
production subcommands to the per-module argparse via `runpy`. **206 tests pass** in the
new package (independently verified, editable install from `src/prodromos`).

`gudhi` and `pot` added to dependencies — `ph_neb_diagnostic` (persistent-homology path
diagnostic) requires them.

## D-003 (2026-06-01) — Code ships here, knowledge stays in `dft-neb/`

Per the Third Matter rule "knowledge → `knowledge/`/back-office, not duplicated": the
shipping *tool* (code, tests, CLI, README, QUICK_START, ROADMAP, DECISIONS) lives in this
repo; the *knowledge* (Evidence Framework v2 derivation, the 7 convergence conditions, the
game-theoretic foundations, magnetic-NEB consilia, per-mineral campaign records, session
checkpoints, deploy plans, the running improvement log) stays in `dft-neb/` as the
back-office knowledge base and tracker. Methodology docs are **not** duplicated into the
repo; `docs/` cross-references them. This keeps a single source of truth.
