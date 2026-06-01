"""Prior outcome distributions for gate nodes + leaf economics (tree mode).

In tree (SIMULATE) mode the planner does NOT execute gates (that would burn the
very $0 diagnostics route mode exists to run, consilium D1). Instead each gate
node branches over a PRIOR distribution of its possible verdicts, and each GO
leaf carries a monetised economics tuple.

All numbers here are GENERIC, declarative defaults -- methodological, not
campaign-specific. They are deliberately conservative and centralised so a
caller can override them (e.g. from a tm-spec ``preflight.economics`` block or a
config) without touching the engine. No private per-mineral outcome is encoded.
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# gate verdict priors  (gate subcommand -> {verdict: prior probability})
# --------------------------------------------------------------------------
# These are weak, generic priors over the verdicts the pre-flight gates emit.
# They sum to 1.0 per gate. They encode only methodology ("an unseen case is
# more likely to have valid DFT-relaxed endpoints than not"), never a specific
# campaign's realised outcome.
GATE_VERDICT_PRIORS: dict[str, dict[str, float]] = {
    "endpoint-provenance": {
        "ENDPOINT_VALID": 0.75,
        "NOT_AN_ENDPOINT_MLIP_GEOMETRY": 0.20,
        "REVIEW": 0.05,
    },
    "electron-parity": {
        "NSPIN1_OK": 0.55,
        "NSPIN2_MANDATORY": 0.25,
        "NSPIN2_RECOMMENDED": 0.15,
        "REVIEW": 0.05,
    },
    # spin-collapse is wired as an honest chance-node AFTER a NSPIN2_* parity
    # verdict (policy.py G02). Its verdict prior is the SPIN_COLLAPSE_PRIOR below,
    # expressed in the spin_collapse_verdict gate's verdict vocabulary so the tree
    # builder can branch on it directly.
    "spin-collapse": {
        "NSPIN1_OK": 0.5,         # the seeded local moment collapses (nspin=1 ok)
        "NSPIN2_REQUIRED": 0.5,   # a localized moment persists (nspin=2 required)
    },
}

# Choice-node options that are TRUE chance forks vs deterministic decisions.
# nspin after a NSPIN2_* verdict is decided (not random); but the *production*
# spin still carries a moment-collapse chance the design flags as a real
# chance-node (some systems collapse to nspin=1, others stay ferrimagnetic).
# Represented as a prior on whether the nspin=1 shortcut is admissible after a
# spin-polarised verdict. The GATE_VERDICT_PRIORS["spin-collapse"] entry above is
# this same prior re-expressed in the gate's verdict vocabulary (NSPIN1_OK /
# NSPIN2_REQUIRED) for the tree builder.
SPIN_COLLAPSE_PRIOR = {"collapses_nspin1_ok": 0.5, "persists_nspin2": 0.5}


# --------------------------------------------------------------------------
# leaf economics
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LeafEconomics:
    """Monetised inputs for a GO leaf's utility (consilium S8.2 / H5).

    All in USD on one utility scale. These are free parameters the caller SHOULD
    set per campaign; defaults are generic placeholders. ``v_paper`` monetises a
    paper-grade barrier; ``v_fail`` is the (usually negative) value of a failed
    run's deliverable; ``cost_run`` / ``cost_redo`` are the run and retry costs.
    """

    v_paper: float = 1000.0
    v_estimate: float = 400.0     # an electronic-only / upper-bound barrier still helps
    v_fail: float = 0.0           # deadweight; a failed run yields no science
    cost_run: float = 200.0
    cost_redo: float = 200.0


# Per-method economics: cheaper / safer methods cost less and (generically) have
# a slightly different paper-grade reach. Methodological defaults only.
METHOD_ECONOMICS: dict[str, LeafEconomics] = {
    # band CI-NEB: standard, full cost, paper-grade reachable
    "band": LeafEconomics(v_paper=1000.0, cost_run=200.0, cost_redo=200.0),
    # dimer / chemical-RC: pricier (sensitive), used on degenerate ridges
    "dimer": LeafEconomics(v_paper=1000.0, cost_run=300.0, cost_redo=300.0),
    # string: asymmetric endpoints, mid cost
    "string": LeafEconomics(v_paper=1000.0, cost_run=250.0, cost_redo=250.0),
}

DEFAULT_ECONOMICS = LeafEconomics()


def economics_for(method: str | None) -> LeafEconomics:
    return METHOD_ECONOMICS.get(method or "", DEFAULT_ECONOMICS)
