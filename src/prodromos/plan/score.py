"""Bellman expectimax + CVaR scoring of a plan tree (consilium C1-C4).

The strategy tree is built by :mod:`prodromos.plan.interpret` in SIMULATE mode.
This module scores it by *value-of-the-tree* backward induction -- NOT an
additive path sum (the naive ``P*value - sum(cost)`` mis-orders strategies,
consilium C2).

Node value (game-theory consilium S8.2):
  * CHANCE node (a gate's prior outcome distribution, or any nature move):
        value = sum_o  P(o) * value(child_o)            -- expectation E
  * DECISION node (a real choice: nspin, method, endpoint):
        value = max(0, max_options value(child))        -- max, with 0 = STOP
    STOP is the reference action (U=0); it wins EMERGENTLY through ``max(0, .)``
    when every run has negative utility. There is deliberately NO separate
    "saved cost" STOP formula (consilium C3) -- that would over-value STOP.
  * LEAF (a concrete expensive run):
        U = p_success * V_paper(quality) - E[cost]
    where E[cost] accounts for a retry on failure:
        E[cost] = cost_run + p_fail * cost_redo
    and ``p_success`` is the calibrated LOWER credible bound (consilium C5), NOT
    a product of per-gate confidences.

Tail risk (consilium C4): each leaf's cost is modelled as a 2-point distribution
(success -> cost_run; failure -> cost_run + cost_redo). CVaR_alpha is the
expected value over the worst alpha tail of the *utility* distribution. The
robustness weight ``beta = cost_run / budget_remaining`` blends EV and CVaR; with
no budget (beta = 0) scoring is pure EV. At high beta, branches that tie on EV
but differ on tail are ranked by their tail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

NodeKind = Literal["DECISION", "CHANCE", "TERMINAL"]


@dataclass
class PlanNode:
    """A node of the scored strategy tree (built in interpret.SIMULATE)."""

    kind: NodeKind
    label: str = ""
    # DECISION/CHANCE: children. For CHANCE the prob is the gate-outcome prior;
    # for DECISION the prob is ignored (the planner picks the max).
    children: list[tuple[float, "PlanNode"]] = field(default_factory=list)

    # TERMINAL (leaf):
    is_stop: bool = False                 # STOP / NO-GO leaf -> reference U = 0
    p_success: float = 0.0                # calibrated lower bound (C5)
    v_paper: float = 0.0                  # monetised paper value of a success
    v_fail: float = 0.0                   # value of a failed run's deliverable (usually <= 0)
    cost_run: float = 0.0                 # cost actually incurred to run
    cost_redo: float = 0.0                # extra cost to redo after a failure
    paper_grade_reachable: bool = False
    method: str = ""
    proposed_workflow_ref: str | None = None
    # provenance of the path that produced this leaf (for reporting)
    path: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# leaf utility distribution + CVaR
# --------------------------------------------------------------------------
def leaf_utility_points(node: PlanNode) -> list[tuple[float, float]]:
    """Return the 2-point (probability, utility) distribution of a run leaf.

    success: utility = V_paper - cost_run         with prob p_success
    failure: utility = v_fail  - (cost_run+cost_redo) with prob (1 - p_success)

    A STOP leaf is the degenerate distribution {(1.0, 0.0)} -- the reference.
    """
    if node.is_stop:
        return [(1.0, 0.0)]
    p = max(0.0, min(1.0, node.p_success))
    u_succ = node.v_paper - node.cost_run
    u_fail = node.v_fail - (node.cost_run + node.cost_redo)
    return [(p, u_succ), (1.0 - p, u_fail)]


def expected_cost(node: PlanNode) -> float:
    """E[cost] of a run leaf, accounting for a retry on failure (consilium C2 leaf)."""
    if node.is_stop:
        return 0.0
    p_fail = 1.0 - max(0.0, min(1.0, node.p_success))
    return node.cost_run + p_fail * node.cost_redo


def expected_utility(node: PlanNode) -> float:
    """E[U] of a run leaf = p_success*V_paper - E[cost] (+ failure deliverable)."""
    pts = leaf_utility_points(node)
    return sum(p * u for p, u in pts)


def cvar(points: list[tuple[float, float]], alpha: float) -> float:
    """CVaR_alpha of a discrete utility distribution: mean over the worst alpha tail.

    ``points`` = [(prob, utility), ...]. CVaR here is the expectation of utility
    conditional on being in the lowest-alpha probability mass (a coherent tail
    measure; Rockafellar-Uryasev). For alpha >= 1 it is just the mean; for the
    degenerate STOP point it is 0.
    """
    if not points:
        return 0.0
    if alpha >= 1.0:
        return sum(p * u for p, u in points)
    # sort ascending by utility (worst first) and accumulate prob mass to alpha
    ordered = sorted(points, key=lambda pu: pu[1])
    acc = 0.0
    weighted = 0.0
    for prob, util in ordered:
        take = min(prob, alpha - acc)
        if take <= 0.0:
            break
        weighted += take * util
        acc += take
        if acc >= alpha:
            break
    if acc <= 0.0:
        return ordered[0][1]
    return weighted / acc


def leaf_value(node: PlanNode, beta: float, alpha: float) -> float:
    """Robust value of a leaf: (1-beta)*E[U] + beta*CVaR_alpha[U] (consilium C4).

    With beta = 0 this is pure expected utility; with beta -> 1 it is the tail.
    STOP leaves return 0 (reference point) regardless of beta/alpha.
    """
    if node.is_stop:
        return 0.0
    pts = leaf_utility_points(node)
    ev = sum(p * u for p, u in pts)
    if beta <= 0.0:
        return ev
    return (1.0 - beta) * ev + beta * cvar(pts, alpha)


# --------------------------------------------------------------------------
# backward induction (the ~30-line core, consilium S8.2)
# --------------------------------------------------------------------------
def value(node: PlanNode, beta: float = 0.0, alpha: float = 0.2) -> float:
    """Bellman expectimax value of ``node``.

    TERMINAL -> leaf_value (STOP = 0).
    CHANCE   -> E over children = sum P(o) * value(child).
    DECISION -> max(0, max_options value(child)); 0 = STOP reference, so STOP
                is emergent, not a bespoke term (consilium C3).
    """
    if node.kind == "TERMINAL":
        return leaf_value(node, beta, alpha)
    if node.kind == "CHANCE":
        return sum(p * value(ch, beta, alpha) for p, ch in node.children)
    if node.kind == "DECISION":
        if not node.children:
            return 0.0
        best = max(value(ch, beta, alpha) for _, ch in node.children)
        return max(0.0, best)  # STOP (U=0) is always an implicit option
    raise ValueError(f"unknown node kind {node.kind!r}")


def beta_from_budget(cost_run: float, budget_remaining: float | None) -> float:
    """beta = clip(cost_run / budget_remaining, 0, 1); None budget -> 0 (pure EV)."""
    if budget_remaining is None or budget_remaining <= 0.0:
        return 0.0
    return max(0.0, min(1.0, cost_run / budget_remaining))


# --------------------------------------------------------------------------
# leaf collection + ranking (strategies)
# --------------------------------------------------------------------------
@dataclass
class ScoredStrategy:
    """A scored leaf strategy, ready for the preflight ``strategies[]`` block."""

    label: str
    method: str
    expected_cost_usd: float
    p_success: float
    cvar_usd: float
    utility: float
    paper_grade_reachable: bool
    proposed_workflow_ref: str | None = None
    is_stop: bool = False
    path: tuple[str, ...] = ()


def _cvar_cost(node: PlanNode, alpha: float) -> float:
    """CVaR of the *cost* distribution in USD (tail cost), for the report column.

    cost points: success -> cost_run; failure -> cost_run + cost_redo.
    Reported as a positive USD figure (the worst-tail expected spend).
    """
    if node.is_stop:
        return 0.0
    p = max(0.0, min(1.0, node.p_success))
    cost_pts = [(p, node.cost_run), (1.0 - p, node.cost_run + node.cost_redo)]
    # worst tail of COST = the largest costs; CVaR over -cost then negate.
    neg = cvar([(pr, -c) for pr, c in cost_pts], alpha)
    return -neg


def collect_leaves(root: PlanNode) -> list[PlanNode]:
    """Depth-first enumerate all TERMINAL leaves under ``root``."""
    out: list[PlanNode] = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.kind == "TERMINAL":
            out.append(n)
        else:
            stack.extend(ch for _, ch in n.children)
    return out


def stochastically_dominated(a: PlanNode, b: PlanNode) -> bool:
    """True if leaf ``a`` is first-order stochastically dominated by ``b``.

    SD-1 (Hadar-Russell): b dominates a iff CDF_b(u) <= CDF_a(u) for all u, with
    strict somewhere. We prune ONLY on strict SD-1 of the UTILITY distribution
    (consilium C / game-theory S5.2) -- never on a (cost, confidence) Pareto
    projection, which is unsafe.
    """
    if a.is_stop or b.is_stop:
        return False
    pa = sorted(leaf_utility_points(a), key=lambda pu: pu[1])
    pb = sorted(leaf_utility_points(b), key=lambda pu: pu[1])
    # build the union of utility breakpoints; compare CDFs there.
    levels = sorted({u for _, u in pa} | {u for _, u in pb})

    def cdf(points: list[tuple[float, float]], x: float) -> float:
        return sum(p for p, u in points if u <= x)

    strict = False
    for x in levels:
        ca, cb = cdf(pa, x), cdf(pb, x)
        if cb > ca + 1e-12:           # b has MORE mass at/below x -> b worse here
            return False
        if cb < ca - 1e-12:
            strict = True
    return strict


def prune_dominated(leaves: list[PlanNode]) -> list[PlanNode]:
    """Drop leaves strictly SD-1 dominated by another (safe pruning, C/S5.2)."""
    survivors: list[PlanNode] = []
    for a in leaves:
        if any(stochastically_dominated(a, b) for b in leaves if b is not a):
            continue
        survivors.append(a)
    return survivors


def rank_strategies(
    root: PlanNode,
    *,
    budget_remaining: float | None = None,
    alpha: float = 0.2,
    beam: int = 8,
    top_k: int | None = None,
    include_stop: bool = True,
) -> list[ScoredStrategy]:
    """Score, prune, and rank the leaf strategies of a plan tree by robust U.

    Ranking value per leaf uses ``beta = cost_run/budget_remaining`` (per-leaf,
    so an expensive leaf near the budget is judged more on its tail). Pruning is
    SD-1 only. ``beam`` caps the kept set; ``top_k`` further truncates the output.
    """
    leaves = collect_leaves(root)
    run_leaves = [n for n in leaves if not n.is_stop]
    survivors = prune_dominated(run_leaves) if run_leaves else []

    scored: list[ScoredStrategy] = []
    for n in survivors:
        b = beta_from_budget(n.cost_run, budget_remaining)
        u = leaf_value(n, b, alpha)
        scored.append(ScoredStrategy(
            label=n.label,
            method=n.method,
            expected_cost_usd=round(expected_cost(n), 4),
            p_success=round(n.p_success, 4),
            cvar_usd=round(_cvar_cost(n, alpha), 4),
            utility=round(u, 4),
            paper_grade_reachable=n.paper_grade_reachable,
            proposed_workflow_ref=n.proposed_workflow_ref,
            is_stop=False,
            path=n.path,
        ))

    # STOP is the reference strategy (U = 0); surface it so a negative-U field
    # makes "do not run" visible and rankable alongside the runs.
    if include_stop:
        scored.append(ScoredStrategy(
            label="STOP",
            method="no-go (do not commit the expensive run)",
            expected_cost_usd=0.0,
            p_success=1.0,
            cvar_usd=0.0,
            utility=0.0,
            paper_grade_reachable=False,
            is_stop=True,
        ))

    scored.sort(key=lambda s: s.utility, reverse=True)
    # de-duplicate by label, keeping the best-utility instance (the same
    # (method, nspin) leaf can be reached via distinct chance branches, e.g.
    # NSPIN2_MANDATORY vs NSPIN2_RECOMMENDED both routing to the nspin choice).
    seen_labels: set[str] = set()
    deduped: list[ScoredStrategy] = []
    for s in scored:
        if s.label in seen_labels:
            continue
        seen_labels.add(s.label)
        deduped.append(s)
    deduped = deduped[:beam]
    if top_k is not None:
        deduped = deduped[:top_k]
    return deduped
