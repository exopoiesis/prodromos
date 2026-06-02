"""Graph interpreters: route (EXECUTE) and tree (SIMULATE stub).

route is NOT a greedy walk of a pre-built tree (consilium D1, the P0 killer):
on a gate node it actually runs the gate's ``run_fn`` on inputs adapted from the
case and follows the edge for the FACTUAL verdict; on a choice node it picks the
myopic-best option. tree (next increment) instead branches over prior outcome
distributions and scores leaves by Bellman expectimax.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from prodromos.plan import PLAN_GRAPH_VERSION
from prodromos.plan.adapters import NeedsData, parse_formula, tm_doc_to_gate_inputs
from prodromos.plan.calibrate import Calibrator, default_calibrator, make_key
from prodromos.plan.graph import Node, PolicyGraph
from prodromos.plan.priors import (
    GATE_VERDICT_PRIORS,
    economics_for,
)
from prodromos.plan.registry import GATE_REGISTRY
from prodromos.plan.score import PlanNode, rank_strategies

MAX_STEPS = 64  # hard cap; the graph is a DAG so this is a belt-and-braces guard.


@dataclass
class GateStep:
    """One executed gate node in a route trace."""

    node_id: str
    gate: str
    sanity_id: str
    verdict: str
    confidence: str | None = None
    reasons: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    needs_data: NeedsData | None = None


@dataclass
class WalkResult:
    """Outcome of a route walk over the policy graph."""

    mode: str
    verdict: str
    confidence: str
    plan_graph_version: str
    steps: list[GateStep] = field(default_factory=list)
    choices: list[dict] = field(default_factory=list)
    terminal_node: str | None = None
    next_action: str | None = None
    next_action_cost_usd: float = 0.0
    reasons: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # tree (SIMULATE) mode only: ranked leaf strategies (see score.ScoredStrategy).
    strategies: list = field(default_factory=list)


# --------------------------------------------------------------------------
# myopic choice (first increment honest placeholder)
# --------------------------------------------------------------------------
def _myopic_best_option(node: Node, ctx: dict) -> str:
    """Pick a choice option in route mode.

    HONEST PLACEHOLDER: choose the "cheapest gate that could change the
    recommendation" -- the existing ``cheapest_disambiguating_test`` notion in
    neb_method_advisor -- approximated here as the first declared option (the
    options lists are authored cheapest-first / safest-default-first in
    policy.py). We do NOT fabricate an EVPI from entropy.

    # TODO: upgrade to calibrated greedy-EVPI (decision-loss VoI, $) -- NOT
    #       entropy / not a confidence product (consilium C1).
    """
    upstream = ctx.get("upstream_verdict")
    if node.options is None:
        raise ValueError(f"choice node {node.id!r} has no options")
    # Honest, deterministic default: prefer the safe option implied by upstream.
    if node.id == "nspin_choice":
        # electron-parity already gave the factual verdict upstream; respect it.
        if upstream == "NSPIN2_MANDATORY":
            return "nspin2"
        if upstream == "NSPIN2_RECOMMENDED":
            return "nspin2"
        return "nspin1"
    # method_choice (and any future choice): cheapest/safest-first option.
    return node.options[0]


# --------------------------------------------------------------------------
# route executor
# --------------------------------------------------------------------------
def _project_inputs(node: Node, gate_inputs: dict) -> dict:
    """Project the adapter output onto the node's declared inputs (memo key)."""
    if not node.inputs:
        return dict(gate_inputs)
    return {k: gate_inputs.get(k) for k in node.inputs if k in gate_inputs}


def _memo_key(gate: str, gate_inputs: dict) -> str:
    return gate + ":" + json.dumps(gate_inputs, sort_keys=True, default=str)


def _run_gate(node: Node, doc: dict, memo: dict[str, dict]) -> tuple[dict | None, NeedsData | None]:
    """Adapt + execute one gate node. Returns (envelope, needs_data)."""
    spec = GATE_REGISTRY[node.gate]
    adapted = tm_doc_to_gate_inputs(doc, node.gate)
    if isinstance(adapted, NeedsData):
        return None, adapted
    key = _memo_key(node.gate, adapted)
    if key in memo:
        return memo[key], None
    envelope = spec.run_fn(**adapted)
    memo[key] = envelope
    return envelope, None


def _route(graph: PolicyGraph, doc: dict) -> WalkResult:
    res = WalkResult(
        mode="route",
        verdict="INVESTIGATE",
        confidence="low",
        plan_graph_version=PLAN_GRAPH_VERSION,
    )
    memo: dict[str, dict] = {}
    seen_trace: set[str] = set()
    upstream_verdict: str | None = None

    node = graph.node(graph.root)
    for _ in range(MAX_STEPS):
        # loop_guard: a (node, upstream) pair revisited means a non-progressing
        # walk on a bad case patch -- bail with NEEDS_DATA rather than spin.
        trace_token = f"{node.id}|{upstream_verdict}"
        if trace_token in seen_trace:
            res.verdict = "NEEDS_DATA"
            res.confidence = "low"
            res.warnings.append(
                f"loop_guard: revisited {node.id!r} without progress; "
                "the case is missing data needed to advance the plan"
            )
            res.terminal_node = node.id
            res.next_action = "supply the missing case fields and re-run plan"
            return res
        seen_trace.add(trace_token)

        if node.kind == "terminal":
            res.verdict = node.terminal
            res.terminal_node = node.id
            res.next_action = node.what
            # surface cost of the recommended next action from the last gate's
            # registry cost (all $0 in the pre-flight increment).
            res.confidence = res.confidence if res.steps else "medium"
            if node.terminal == "GO":
                res.confidence = "high" if res.steps else "medium"
                res.reasons.append("all evaluated pre-flight gates passed; path is well-posed")
            return res

        if node.kind == "gate":
            envelope, needs = _run_gate(node, doc, memo)
            spec = GATE_REGISTRY[node.gate]
            if needs is not None:
                step = GateStep(
                    node_id=node.id,
                    gate=node.gate,
                    sanity_id=spec.sanity_id,
                    verdict="NEEDS_DATA",
                    confidence="low",
                    reasons=[f"missing: {', '.join(needs.missing)}"],
                    next_actions=[needs.recommend] if needs.recommend else [],
                    needs_data=needs,
                )
                res.steps.append(step)
                res.verdict = "NEEDS_DATA"
                res.confidence = "low"
                res.terminal_node = node.id
                res.next_action = needs.recommend or f"obtain inputs for {node.gate}"
                res.next_action_cost_usd = spec.cost_usd
                res.reasons.extend(step.reasons)
                res.next_actions.extend(step.next_actions)
                return res
            verdict = envelope.get("verdict")
            step = GateStep(
                node_id=node.id,
                gate=node.gate,
                sanity_id=spec.sanity_id,
                verdict=verdict,
                confidence=envelope.get("confidence"),
                reasons=list(envelope.get("reasons") or []),
                next_actions=list(envelope.get("next_actions") or []),
            )
            res.steps.append(step)
            res.confidence = step.confidence or res.confidence
            upstream_verdict = verdict
            edge = _select_edge(node, {"verdict": verdict, "result": envelope.get("result")})
            if edge is None:
                res.verdict = "INVESTIGATE"
                res.terminal_node = node.id
                res.warnings.append(
                    f"no edge matched verdict {verdict!r} at gate node {node.id!r}"
                )
                res.next_action = (
                    f"verdict {verdict!r} from {node.gate} has no routing edge; "
                    "extend the policy graph"
                )
                return res
            node = graph.node(edge.dst)
            continue

        if node.kind == "choice":
            option = _myopic_best_option(node, {"upstream_verdict": upstream_verdict})
            res.choices.append({
                "node": node.id,
                "chosen": option,
                "options": list(node.options or []),
                "rule": "myopic placeholder (cheapest-disambiguating); TODO greedy-EVPI",
            })
            edge = _select_edge(node, {"option": option})
            if edge is None:
                res.verdict = "INVESTIGATE"
                res.terminal_node = node.id
                res.warnings.append(
                    f"no edge matched option {option!r} at choice node {node.id!r}"
                )
                return res
            node = graph.node(edge.dst)
            continue

        raise ValueError(f"unknown node kind {node.kind!r} at {node.id!r}")

    res.verdict = "NEEDS_DATA"
    res.warnings.append(f"exceeded MAX_STEPS={MAX_STEPS}; aborting walk")
    return res


def _select_edge(node: Node, ctx: dict):
    """First-match edge selection (mirrors the router's first-gate-wins)."""
    for edge in node.edges:
        if edge.predicate(ctx):
            return edge
    return None


# --------------------------------------------------------------------------
# tree simulator (SIMULATE -- branch over priors, score by backward induction)
# --------------------------------------------------------------------------
DEPTH_CAP = 6  # consilium CS S2: depth 4-6, hard cap as belt-and-braces


def _chemistry_signature(doc: dict) -> str:
    """Coarse chemistry signature for calibration backoff (NOT a mineral name).

    Built from the element set + space-group symbol, so it pools by chemistry
    family ("V_Fe chemistry universal") rather than by mineral identity. Returns
    ``"*"`` (the global cell) when nothing usable is present.
    """
    structure = doc.get("structure") or {}
    formula = structure.get("formula")
    counts = parse_formula(formula) if isinstance(formula, str) else {}
    elems = "-".join(sorted(counts)) if counts else "*"
    sg = structure.get("space_group") or {}
    sg_sym = sg.get("symbol") if isinstance(sg, dict) else None
    return f"{elems}|{sg_sym or '*'}"


def _verdict_dist(node: Node) -> dict[str, float]:
    """Prior distribution over a gate node's verdicts (tree mode).

    Falls back to a single deterministic branch per declared edge when no prior
    is registered for the gate (we do NOT execute the gate in tree mode).
    """
    if node.gate in GATE_VERDICT_PRIORS:
        return GATE_VERDICT_PRIORS[node.gate]
    # No registered prior: assume each declared edge's first verdict is equally
    # likely. We can only see verdicts the edges route on, so approximate by
    # spreading mass uniformly across edges (each carries one routing verdict).
    n = max(1, len(node.edges))
    return {f"_edge{i}": 1.0 / n for i in range(n)}


@dataclass
class _TreeCtx:
    """Path-accumulated context for leaf economics + calibration key."""

    method: str | None = None
    nspin: int | None = None
    verdicts: tuple[str, ...] = ()


def _build_tree(
    graph: PolicyGraph,
    node: Node,
    ctx: _TreeCtx,
    calib: Calibrator,
    chem_sig: str,
    depth: int,
) -> PlanNode:
    """Recursively build the scored PlanNode tree from the policy graph.

    gate node  -> CHANCE node branching over the verdict prior;
    choice node-> DECISION node branching over all options;
    terminal   -> leaf (GO = run leaf with calibrated p_success + economics;
                  others = STOP leaf, reference U = 0).
    """
    if depth > DEPTH_CAP:
        # safety: treat as STOP rather than recurse unbounded (DAG => unreachable
        # in practice, this is belt-and-braces).
        return PlanNode(kind="TERMINAL", label=node.id, is_stop=True, path=ctx.verdicts)

    if node.kind == "terminal":
        if node.terminal == "GO":
            econ = economics_for(ctx.method)
            key = make_key(chem_sig, ctx.method, ctx.nspin, ctx.verdicts)
            p = calib.p_success_lower(key)
            return PlanNode(
                kind="TERMINAL",
                label=_leaf_label(ctx),
                is_stop=False,
                p_success=p,
                v_paper=econ.v_paper,
                v_estimate=econ.v_estimate,
                v_fail=econ.v_fail,
                cost_run=econ.cost_run,
                cost_redo=econ.cost_redo,
                est_frac=econ.est_frac,
                paper_grade_reachable=True,
                method=_method_label(ctx),
                path=ctx.verdicts,
            )
        # NO-GO / NEEDS_DATA / INVESTIGATE -> STOP reference leaf
        return PlanNode(
            kind="TERMINAL",
            label=f"STOP:{node.terminal}",
            is_stop=True,
            method=node.what or node.terminal or "",
            path=ctx.verdicts,
        )

    if node.kind == "gate":
        children: list[tuple[float, PlanNode]] = []
        dist = _verdict_dist(node)
        for verdict, prob in dist.items():
            edge = _select_edge(node, {"verdict": verdict})
            if edge is None:
                # verdict with no routing edge -> dead-end STOP, keep mass honest
                children.append((prob, PlanNode(
                    kind="TERMINAL", label=f"STOP:{node.id}:{verdict}",
                    is_stop=True, path=ctx.verdicts,
                )))
                continue
            child_ctx = _TreeCtx(
                method=ctx.method,
                nspin=ctx.nspin,
                verdicts=(*ctx.verdicts, verdict) if not verdict.startswith("_edge") else ctx.verdicts,
            )
            child = _build_tree(graph, graph.node(edge.dst), child_ctx, calib, chem_sig, depth + 1)
            children.append((prob, child))
        return PlanNode(kind="CHANCE", label=node.id, children=children)

    if node.kind == "choice":
        children = []
        for option in (node.options or []):
            edge = _select_edge(node, {"option": option})
            if edge is None:
                continue
            child_ctx = _TreeCtx(
                method=_apply_option_method(ctx.method, option),
                nspin=_apply_option_nspin(ctx.nspin, option),
                verdicts=ctx.verdicts,
            )
            child = _build_tree(graph, graph.node(edge.dst), child_ctx, calib, chem_sig, depth + 1)
            children.append((1.0, child))  # prob ignored for DECISION nodes
        return PlanNode(kind="DECISION", label=node.id, children=children)

    raise ValueError(f"unknown node kind {node.kind!r} at {node.id!r}")


def _apply_option_method(current: str | None, option: str) -> str | None:
    if option in ("band", "dimer", "string"):
        return option
    return current


def _apply_option_nspin(current: int | None, option: str) -> int | None:
    if option == "nspin1":
        return 1
    if option == "nspin2":
        return 2
    return current


def _method_label(ctx: _TreeCtx) -> str:
    method = ctx.method or "band"
    spin = f"nspin={ctx.nspin}" if ctx.nspin else "nspin=auto"
    return f"{method}-NEB, {spin}"


def _leaf_label(ctx: _TreeCtx) -> str:
    method = ctx.method or "band"
    spin = f"_nspin{ctx.nspin}" if ctx.nspin else ""
    return f"{method}_neb{spin}"


def _tree(
    graph: PolicyGraph,
    doc: dict,
    *,
    budget_usd: float | None = None,
    top_k: int | None = None,
    beam: int = 8,
    alpha: float = 0.2,
    calib: Calibrator | None = None,
) -> WalkResult:
    """SIMULATE mode -- scored strategy tree (Bellman expectimax + CVaR).

    Branches gate nodes over prior verdict distributions and choice nodes over
    all options, scores leaves by calibrated p_success (lower bound) and CVaR
    tail control, prunes by strict stochastic dominance, and returns the ranked
    strategies (consilium C1-C5).
    """
    res = WalkResult(
        mode="tree",
        verdict="INVESTIGATE",
        confidence="medium",
        plan_graph_version=PLAN_GRAPH_VERSION,
    )
    calibrator = calib or default_calibrator()
    chem_sig = _chemistry_signature(doc)
    root = _build_tree(
        graph, graph.node(graph.root), _TreeCtx(), calibrator, chem_sig, depth=0
    )
    strategies = rank_strategies(
        root,
        budget_remaining=budget_usd,
        alpha=alpha,
        beam=beam,
        top_k=top_k,
    )
    res.strategies = strategies

    # overall verdict: best non-STOP strategy beats STOP (U>0) -> GO; else NO-GO.
    best = strategies[0] if strategies else None
    if best is None:
        res.verdict = "INVESTIGATE"
        res.next_action = "no strategies enumerated; check the policy graph"
        return res
    if best.is_stop or best.utility <= 0.0:
        res.verdict = "NO-GO"
        res.confidence = "high" if best.is_stop else "medium"
        res.next_action = (
            "do not commit the expensive run: every enumerated strategy has "
            "non-positive expected utility (STOP is the optimal action)"
        )
        res.reasons.append(
            "max(0, .) backward induction: best run utility <= 0 (real-option STOP)"
        )
    else:
        res.verdict = "GO"
        res.confidence = "high" if best.p_success >= 0.7 else "medium"
        res.next_action = (
            f"launch strategy {best.label!r} ({best.method}); "
            f"E[cost]=${best.expected_cost_usd}, U={best.utility}"
        )
        res.next_action_cost_usd = best.expected_cost_usd
        res.reasons.append(
            f"top strategy by robust utility (beta from budget, CVaR_{alpha} tail): "
            f"{best.label}"
        )
    res.next_actions = [s.label for s in strategies]
    return res


# --------------------------------------------------------------------------
# public entry
# --------------------------------------------------------------------------
def walk(
    graph: PolicyGraph,
    doc: dict,
    mode: str = "route",
    *,
    budget_usd: float | None = None,
    top_k: int | None = None,
    calib: Calibrator | None = None,
) -> WalkResult:
    """Walk ``graph`` over the tm-spec ``doc`` in route (EXECUTE) or tree (SIMULATE)."""
    if mode == "route":
        return _route(graph, doc)
    if mode == "tree":
        return _tree(graph, doc, budget_usd=budget_usd, top_k=top_k, calib=calib)
    raise ValueError(f"unknown mode {mode!r} (expected 'route' or 'tree')")
