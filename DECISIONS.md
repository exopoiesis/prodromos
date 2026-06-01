# Prodromos — Decisions

## D-001 (2026-06-01) — Name: Prodromos

The framework that grew in an internal Third Matter working folder was promoted to a
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

The previous flat-script suite (sibling-import scripts run via `python foo.py`) was
migrated into a real package: 29 core modules → `src/prodromos/`,
26 test files → `tests/`, all intra-suite imports rewritten to absolute `prodromos.*`,
the `sys.path` hack dropped from `conftest.py`. A thin `__main__.py` dispatches 15
production subcommands to the per-module argparse via `runpy`. **206 tests pass** in the
new package (independently verified, editable install from `src/prodromos`).

`gudhi` and `pot` added to dependencies — `ph_neb_diagnostic` (persistent-homology path
diagnostic) requires them.

## D-003 (2026-06-01) — Public repo is self-contained; only internal records stay back-office

This repository is **public** (for collaborators), so anything useful to others must live
here, self-contained. The shareable methodology, theory, and playbooks were sanitized
(stripped of internal session ids, infra names, instance ids, internal file/decision refs,
private email) and moved into [`docs/`](docs/) as their **canonical** home — 9 documents
covering the Evidence Framework (L0→L6), the 7 convergence conditions (C1–C7), the
game-theoretic foundations, magnetic-first logic, the NEB-stall playbook, the method-advisor
design, and positioning. New shareable material is added **here**, not in private notes.

What stays internal (not published): per-mineral campaign result records, DFT-deploy plans,
dated consilia, session checkpoints, infra/cost specifics, and raw genesis idea-dumps. The
project's private working folder keeps those and now points *into* this repo for anything
shareable.
