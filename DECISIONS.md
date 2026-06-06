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

## D-004 (2026-06-01) — All framework improvement items implemented; MCP-ready

The full open backlog (N-01…N-15) is implemented: 4 new gates (`external-reference`,
`lint-dft-script`, `h-barrier-readiness`, `endpoint-provenance`) and enhancements to the
parity, method-advisor, magnetic, and symmetry gates. Every gate now exposes a pure
`run_*(...) -> dict` returning the shared `cli_contract` envelope, plus a `prodromos`
subcommand (19 total). This is the "ready for MCP" state: the planned MCP server is a thin
adapter over the `run_*` functions, no further gate work required.

Test suite is fully self-contained for CI: previously data-gated tests now run against
scrubbed marc/pyrite DFT fixtures committed under `tests/fixtures/` (paths/usernames
stripped). **353 passed, 0 skipped.**

Public-hygiene pass: internal session ids, absolute machine paths, and references to
unpublished back-office docs were stripped from source comments/docstrings and demo
runners; all docs and code comments are English. Line endings normalized to LF via
`.gitattributes`.

## D-005 (2026-06-06) — Framework backlog cleared: MCP transport fix + 4 new gates

The post-Paper-1 improvement backlog (the private `PRODROMOS_FRAMEWORK_ROADMAP.md`) was worked
down to the compute-bound items. Shipped:

- **§F MCP transport hang (PRODUCTION BLOCKER).** Root cause confirmed by a deterministic
  in-memory harness (`tests/test_mcp_transport.py`): FastMCP 1.27 runs a sync tool INLINE on the
  asyncio loop while the lowlevel server dispatches every request on that same loop, so one sync
  tool blocks intake + flush and serializes all concurrent calls. Fix: every tool registered via
  `_offload(name, fn)` — a `functools.wraps`-ed async wrapper that runs the sync core on
  `anyio.to_thread.run_sync` (typed schema + docstring preserved). Plus a PID-lockfile singleton
  guard + clean EOF shutdown, two meta-tools (`batch`, `preflight_bundle`) that collapse client
  fan-out into one round-trip, and an opt-in per-tool timeout (`PRODROMOS_MCP_TOOL_TIMEOUT_S`).
  The harness proves inline serializes (~N×sleep) and offload runs concurrent (~sleep).
- **N-20 `electron_parity` smarter M0** — formal-oxidation d-count (`infer_closed_shell`) rejects
  d⁰/d¹⁰ closed shells; vacancy-odd vs TM-odd discriminator under metallic smearing.
- **N-21 `mlip_confidence`** — flags hosts where a foundation-MLIP barrier is untrustworthy
  (near-degenerate itinerant 3d / multivalent redox cathode) → routes to DFT.
- **N-22 `sublattice_preflight`** — structure-level (pre-DFT, $0) magnetic-sublattice-crossing
  predictor; `mode="migrant"` and `mode="polaron"` (the redox-polaron failure mode for nonmagnetic
  Li⁺/Na⁺ cathode hops); emits the two-species / constrained-M recipe on NO-GO.
- **N-23 `magnetic_verdict`** — per-TM *relative* endpoint ΔM threshold (kills false NO-GO on
  large-cell slow drift, e.g. troilite) + auto-reconcile of the endpoint screen with the
  full-trajectory band gate (band gate is the arbiter) into one combined verdict.
- **N-24 `magnetic_provenance`** — cross-checks MP-computed vs MAGNDATA-experimental ordering →
  routes the NEB seed to the experimental block, WARNs on conflict (MP mislabels Fe sulfides/
  phosphates FM). MP key from env or `secrets/mp_api_key.json`.
- **Bug** — band-directory discovery now recognises `endA`/`endB`/`neb_imgNN` (not just `image_NN`)
  and orders endpoint-A → interior → endpoint-B (was dropping endpoints / mis-ordering adjacency).

MCP surface: 29 gate tools + 2 meta-tools. **Full suite green (516 tests).** Remaining backlog is
compute-bound [VAL] (external NEB-band corpus, prospective 5-paper DFT sweep) or the deferred [SW]
decision-engine note — no further pure-software gate work outstanding.
