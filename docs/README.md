# Prodromos — docs

Methodology, theory, and playbooks behind the gates. This is the canonical home for
the shareable documentation — read here, and **add new shareable material here**, not
in any private working notes.

## Methodology

- [`EVIDENCE_FRAMEWORK_V2_COMPLETE.md`](EVIDENCE_FRAMEWORK_V2_COMPLETE.md) — the end-to-end
  L0→L6 pipeline: cheapest test first, stop at the first hard diagnosis.
- [`MULTI_ENDPOINT_METHODOLOGY_V3.md`](MULTI_ENDPOINT_METHODOLOGY_V3.md) — 8-step protocol for
  asymmetric multi-endpoint systems (when a symmetric NEB does not apply).
- [`MAGNETIC_FIRST_PREFLIGHT_PLAN.md`](MAGNETIC_FIRST_PREFLIGHT_PLAN.md) — magnetic-first
  logic for nspin=2 minerals: parity → spin-collapse → endpoint/band sheet gates → routing.

## Theory

- [`GAME_THEORETIC_NEB_FOUNDATIONS.md`](GAME_THEORETIC_NEB_FOUNDATIONS.md) — the
  game-theoretic framing: same-basin trap as an equilibrium-selection problem, NEB-AGM,
  links to ADMM / persistent homology / Freidlin-Wentzell / mean-field games.
- [`THEORY_PENT_CONVERGENCE.md`](THEORY_PENT_CONVERGENCE.md) — the 7 sufficient conditions
  (C1–C7) for NEB convergence to the true MEP, worked through a real V_Fe case.

## Playbooks & design

- [`NEB_STALL_DIAGNOSTIC_PLAYBOOK.md`](NEB_STALL_DIAGNOSTIC_PLAYBOOK.md) — "my NEB is stuck":
  a $0 procedure to tell spring/optimizer vs endpoint vs same-basin vs magnetic-sheet.
- [`NEB_METHOD_ADVISOR_DESIGN.md`](NEB_METHOD_ADVISOR_DESIGN.md) — design of the method
  advisor (failure-signature → method: band tune / dimer+chemical-RC / string).

## Positioning

- [`ALTERNATIVES_AND_ROLE_AWARE_NEB.md`](ALTERNATIVES_AND_ROLE_AWARE_NEB.md) — landscape of
  inequivalent MEP formulations and the "role-aware NEB" idea.
- [`other_tools.md`](other_tools.md) — related tools and where Prodromos sits among them.

---

For how to *run* the gates, see [`../QUICK_START.md`](../QUICK_START.md). Open product
items are in [`../ROADMAP.md`](../ROADMAP.md).
